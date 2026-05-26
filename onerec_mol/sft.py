from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .executor import get_action_constraints
from .grpo import EditGrammarConstrainedLogitsProcessor
from .dataset import strip_atom_mapping
from .token_policy import suggest_fg_ids_for_task
from .tokenizer import encode_mol_quantized
from .vocab import register_domain_tokens


def _patch_accelerate_deepspeed_import() -> None:
    """
    Some environments have a broken DeepSpeed install that is discovered by Accelerate
    even for plain single-GPU Trainer runs. If importing DeepSpeed fails, replace the
    unwrap helper with a minimal no-DeepSpeed fallback.
    """
    try:
        import deepspeed  # noqa: F401

        return
    except Exception:
        pass

    try:
        import accelerate.accelerator as accel_mod
        import accelerate.utils.other as other_mod
    except Exception:
        return

    def _fallback_extract_model_from_parallel(model, keep_fp32_wrapper: bool = True, keep_torch_compile: bool = True):
        current = model
        seen: set[int] = set()
        while hasattr(current, "module") and id(current) not in seen:
            seen.add(id(current))
            current = current.module
        return current

    other_mod.extract_model_from_parallel = _fallback_extract_model_from_parallel
    accel_mod.extract_model_from_parallel = _fallback_extract_model_from_parallel


def _patch_peft_torchao_import() -> None:
    try:
        import peft.import_utils as import_utils
        import peft.tuners.lora.torchao as lora_torchao
    except Exception:
        return

    def _torchao_unavailable() -> bool:
        return False

    import_utils.is_torchao_available = _torchao_unavailable
    lora_torchao.is_torchao_available = _torchao_unavailable


def _encode_example(tokenizer, prompt: str, completion: str, max_length: int) -> dict[str, list[int]]:
    prompt_text = str(prompt).strip()
    completion_text = str(completion).strip()
    full_text = f"{prompt_text}\n{completion_text}{tokenizer.eos_token or ''}"

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]

    if len(input_ids) > int(max_length):
        overflow = len(input_ids) - int(max_length)
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _encode_example_with_soft_prefix(
    tokenizer,
    prompt: str,
    completion: str,
    max_length: int,
    tokenizer_ckpt: str,
) -> dict[str, Any]:
    item = _encode_example(tokenizer, prompt, completion, max_length)
    start_smiles_tagged = _extract_start_smiles_tagged_from_prompt(prompt)
    try:
        plain = strip_atom_mapping(start_smiles_tagged)
        item["mol_quantized"] = encode_mol_quantized(plain, tokenizer_ckpt)
    except Exception:  # noqa: BLE001
        item["mol_quantized"] = [0.0 for _ in range(256)]
    return item


class _SoftMolPrefixModel(nn.Module):
    def __init__(self, base_model: nn.Module, input_dim: int, prefix_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = getattr(base_model, "config", None)
        self.prefix_len = int(prefix_len)
        embed_dim = int(base_model.get_input_embeddings().embedding_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), embed_dim * self.prefix_len),
            nn.Dropout(float(dropout)),
        )
        self.embed_dim = embed_dim

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            return self.base_model.gradient_checkpointing_enable(*args, **kwargs)
        return None

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def save_pretrained(self, save_directory: str, *args, **kwargs) -> None:
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(str(out_dir), *args, **kwargs)
        torch.save(
            {
                "state_dict": self.projector.state_dict(),
                "input_dim": int(self.projector[0].normalized_shape[0]),
                "prefix_len": int(self.prefix_len),
                "embed_dim": int(self.embed_dim),
            },
            out_dir / "soft_mol_prefix_projector.pt",
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, mol_quantized=None, **kwargs):
        if mol_quantized is None:
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        token_embeds = self.base_model.get_input_embeddings()(input_ids)
        prefix = self.projector(mol_quantized.to(device=token_embeds.device, dtype=torch.float32))
        prefix = prefix.to(dtype=token_embeds.dtype)
        prefix = prefix.view(token_embeds.shape[0], self.prefix_len, self.embed_dim)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        if attention_mask is not None:
            prefix_mask = torch.ones(
                (attention_mask.shape[0], self.prefix_len),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        if labels is not None:
            prefix_labels = torch.full(
                (labels.shape[0], self.prefix_len),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels = torch.cat([prefix_labels, labels], dim=1)
        return self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, mol_quantized=None, **kwargs):
        if mol_quantized is None:
            return self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        token_embeds = self.base_model.get_input_embeddings()(input_ids)
        prefix = self.projector(mol_quantized.to(device=token_embeds.device, dtype=torch.float32))
        prefix = prefix.to(dtype=token_embeds.dtype).view(token_embeds.shape[0], self.prefix_len, self.embed_dim)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        if attention_mask is not None:
            prefix_mask = torch.ones(
                (attention_mask.shape[0], self.prefix_len),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return self.base_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )


class _SoftPrefixCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        mol_quantized = [list(f.pop("mol_quantized")) for f in features]
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of > 1:
            rem = max_len % self.pad_to_multiple_of
            if rem:
                max_len += self.pad_to_multiple_of - rem
        pad_id = int(self.tokenizer.pad_token_id)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            n_pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(list(f["input_ids"]) + [pad_id] * n_pad)
            batch["attention_mask"].append(list(f["attention_mask"]) + [0] * n_pad)
            batch["labels"].append(list(f["labels"]) + [-100] * n_pad)
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "mol_quantized": torch.tensor(mol_quantized, dtype=torch.float32),
        }


def _infer_target_modules(model) -> list[str]:
    suffixes: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffixes.add(name.split(".")[-1])
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"]
    target = [x for x in preferred if x in suffixes]
    return target if target else sorted(suffixes)


_START_SMILES_RE = re.compile(r"Start molecule\s*\(atom-mapped SMILES\):\s*\n([^\n]+)")
_TASK_ID_RE = re.compile(r"Task ID:\s*([0-9]+)")


def _extract_start_smiles_tagged_from_prompt(prompt: str) -> str:
    text = str(prompt or "")
    m = _START_SMILES_RE.search(text)
    if m:
        return str(m.group(1)).strip()
    return ""


def _extract_task_id_from_prompt(prompt: str) -> int | None:
    text = str(prompt or "")
    m = _TASK_ID_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:  # noqa: BLE001
        return None


def train_sft(
    model_name: str,
    train_jsonl: str,
    val_jsonl: str,
    output_dir: str,
    config: dict[str, Any],
) -> str:
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    cfg = {
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_train_embeddings": False,
        "learning_rate": 2e-4,
        "num_train_epochs": 2,
        "max_steps": -1,
        "max_length": 1024,
        "batch_size": 16,
        "micro_batch_size": 2,
        "weight_decay": 0.0,
        "logging_steps": 10,
        "eval_steps": 50,
        "save_steps": 50,
        "save_total_limit": 2,
        "seed": 42,
        "codebook_size": 256,
        "num_codebooks": 8,
        "mol_token_format": "shared",
        "max_atom_map": 256,
        "max_fg_id": 64,
        "register_domain_vocab": True,
        "masked_ce": True,
        "gradient_checkpointing": True,
        "use_soft_mol_prefix": False,
        "soft_mol_prefix_tokenizer_ckpt": "",
        "soft_mol_prefix_len": 8,
        "soft_mol_prefix_input_dim": 256,
        "soft_mol_prefix_dropout": 0.0,
        "save_safetensors": False,
        "resume_adapter_path": "",
    }
    cfg.update(config or {})

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if bool(cfg.get("register_domain_vocab", True)):
        token_reg = register_domain_tokens(
            tokenizer,
            codebook_size=int(cfg["codebook_size"]),
            num_codebooks=int(cfg.get("num_codebooks", 8)),
            mol_token_format=str(cfg.get("mol_token_format", "shared")),
            max_atom_map=int(cfg["max_atom_map"]),
            max_fg_id=int(cfg["max_fg_id"]),
        )
    else:
        token_reg = {"num_requested": 0, "num_added": 0, "num_existing": 0}
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token_id is not None else "[PAD]"
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    if int(token_reg["num_added"]) > 0:
        model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if bool(cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()

    target_modules: list[str] = []
    finetune_mode = "full"
    if bool(cfg.get("use_lora", True)):
        _patch_peft_torchao_import()
        from peft import LoraConfig, PeftModel, get_peft_model

        resume_adapter_path = str(cfg.get("resume_adapter_path", "")).strip()
        if resume_adapter_path:
            model = PeftModel.from_pretrained(model, resume_adapter_path, is_trainable=True)
            peft_cfg = getattr(model, "peft_config", {})
            active_adapter = getattr(model, "active_adapter", None)
            active_cfg = peft_cfg.get(active_adapter) if isinstance(peft_cfg, dict) else None
            raw_targets = getattr(active_cfg, "target_modules", None)
            target_modules = sorted(str(x) for x in raw_targets) if raw_targets else []
        else:
            target_modules = _infer_target_modules(model)
            modules_to_save: list[str] = []
            if bool(cfg.get("lora_train_embeddings", False)):
                modules_to_save = ["embed_tokens", "lm_head"]
            peft_config = LoraConfig(
                r=int(cfg["lora_r"]),
                lora_alpha=int(cfg["lora_alpha"]),
                lora_dropout=float(cfg["lora_dropout"]),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
                modules_to_save=modules_to_save or None,
            )
            model = get_peft_model(model, peft_config)
        finetune_mode = "lora"

    use_soft_mol_prefix = bool(cfg.get("use_soft_mol_prefix", False))
    if use_soft_mol_prefix:
        if not str(cfg.get("soft_mol_prefix_tokenizer_ckpt", "")).strip():
            raise ValueError("soft_mol_prefix_tokenizer_ckpt is required when use_soft_mol_prefix=true")
        model = _SoftMolPrefixModel(
            base_model=model,
            input_dim=int(cfg["soft_mol_prefix_input_dim"]),
            prefix_len=int(cfg["soft_mol_prefix_len"]),
            dropout=float(cfg["soft_mol_prefix_dropout"]),
        )
        finetune_mode = f"{finetune_mode}+soft_mol_prefix"

    raw = load_dataset("json", data_files={"train": str(train_jsonl), "eval": str(val_jsonl)})
    if use_soft_mol_prefix:
        encode_fn = lambda ex: _encode_example_with_soft_prefix(  # noqa: E731
            tokenizer,
            ex["prompt"],
            ex["completion"],
            int(cfg["max_length"]),
            str(cfg["soft_mol_prefix_tokenizer_ckpt"]),
        )
    else:
        encode_fn = lambda ex: _encode_example(tokenizer, ex["prompt"], ex["completion"], int(cfg["max_length"]))  # noqa: E731
    proc = raw.map(
        encode_fn,
        remove_columns=raw["train"].column_names,
    )

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=int(cfg["micro_batch_size"]),
        per_device_eval_batch_size=int(cfg["micro_batch_size"]),
        gradient_accumulation_steps=max(1, int(cfg["batch_size"]) // max(1, int(cfg["micro_batch_size"]))),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        max_steps=int(cfg["max_steps"]),
        bf16=True,
        logging_steps=int(cfg["logging_steps"]),
        eval_steps=int(cfg["eval_steps"]),
        save_steps=int(cfg["save_steps"]),
        save_total_limit=int(cfg["save_total_limit"]),
        eval_strategy="steps",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=int(cfg["seed"]),
        save_safetensors=bool(cfg.get("save_safetensors", False)),
    )

    _patch_accelerate_deepspeed_import()

    class _MaskedCETrainer(Trainer):
        def __init__(self, *args, tokenizer_for_mask=None, use_masked_ce=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.tokenizer_for_mask = tokenizer_for_mask
            self.use_masked_ce = bool(use_masked_ce)
            self._fsm_cache: dict[str, EditGrammarConstrainedLogitsProcessor] = {}
            self._constraints_cache: dict[str, dict[str, Any]] = {}

        @staticmethod
        def _policy_with_teacher_token(policy: dict[str, Any] | None, target_id: int) -> dict[str, Any] | None:
            if not policy:
                return policy
            kind = str(policy.get("kind", ""))
            ids = sorted(set(int(x) for x in policy.get("ids", [])))
            tgt = int(target_id)
            if kind == "allow":
                ids.append(tgt)
                return {"kind": "allow", "ids": sorted(set(ids))}
            if kind == "block":
                return {"kind": "block", "ids": [int(x) for x in ids if int(x) != tgt]}
            return policy

        @staticmethod
        def _apply_policy(logit_vec: torch.Tensor, policy: dict[str, Any] | None) -> torch.Tensor:
            if not policy:
                return logit_vec
            kind = str(policy.get("kind", ""))
            ids = sorted(set(int(x) for x in policy.get("ids", [])))
            if kind == "allow":
                if not ids:
                    return logit_vec
                masked = torch.full_like(logit_vec, float("-inf"))
                allow_ids = torch.tensor(ids, device=logit_vec.device, dtype=torch.long)
                masked[allow_ids] = logit_vec[allow_ids]
                return masked
            if kind == "block":
                if not ids:
                    return logit_vec
                out = logit_vec.clone()
                block_ids = torch.tensor(ids, device=logit_vec.device, dtype=torch.long)
                out[block_ids] = float("-inf")
                return out
            return logit_vec

        def _fsm_for_prompt(self, prompt_text: str) -> EditGrammarConstrainedLogitsProcessor | None:
            if self.tokenizer_for_mask is None:
                return None
            start_smiles = _extract_start_smiles_tagged_from_prompt(prompt_text)
            if not start_smiles:
                return None
            task_id = _extract_task_id_from_prompt(prompt_text)
            cache_key = f"{start_smiles}||{int(task_id) if task_id is not None else -1}"
            cached = self._fsm_cache.get(cache_key)
            if cached is not None:
                return cached
            constraints = self._constraints_cache.get(cache_key)
            if constraints is None:
                constraints = get_action_constraints(start_smiles)
                preferred_fg_ids = suggest_fg_ids_for_task(task_id)
                if preferred_fg_ids:
                    constraints = dict(constraints)
                    constraints["preferred_fg_ids"] = [int(x) for x in preferred_fg_ids]
                self._constraints_cache[cache_key] = constraints
            fsm = EditGrammarConstrainedLogitsProcessor(
                tokenizer=self.tokenizer_for_mask,
                prompt_lengths=[0],
                action_constraints=[constraints],
            )
            self._fsm_cache[cache_key] = fsm
            return fsm

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            if (not self.use_masked_ce) or labels is None or self.tokenizer_for_mask is None:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)

            forward_inputs = dict(inputs)
            forward_inputs.pop("labels", None)
            outputs = model(**forward_inputs)
            logits = outputs.logits

            if logits.shape[:2] != labels.shape[:2]:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)

            masked_logits = logits.float().clone()
            input_ids = inputs["input_ids"]
            bsz, seq_len = int(labels.shape[0]), int(labels.shape[1])

            for b in range(bsz):
                label_row = labels[b]
                non_ignored = torch.nonzero(label_row.ne(-100), as_tuple=False)
                if non_ignored.numel() == 0:
                    continue
                completion_start = int(non_ignored[0].item())
                if completion_start <= 0:
                    continue
                prompt_ids = input_ids[b, :completion_start].tolist()
                prompt_text = self.tokenizer_for_mask.decode(prompt_ids, skip_special_tokens=False)
                fsm = self._fsm_for_prompt(prompt_text)
                if fsm is None:
                    continue

                generated_ids: list[int] = []
                for pos in range(completion_start, seq_len):
                    tgt = int(label_row[pos].item())
                    if tgt == -100:
                        continue
                    logit_pos = pos - 1
                    if logit_pos < 0:
                        generated_ids.append(int(input_ids[b, pos].item()))
                        continue
                    policy = fsm.next_token_policy(generated_ids, row_idx=0)
                    policy = self._policy_with_teacher_token(policy, tgt)
                    masked_logits[b, logit_pos, :] = self._apply_policy(masked_logits[b, logit_pos, :], policy)
                    generated_ids.append(int(input_ids[b, pos].item()))

            shift_logits = masked_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if return_outputs:
                return loss, outputs
            return loss

    trainer = _MaskedCETrainer(
        model=model,
        args=train_args,
        train_dataset=proc["train"],
        eval_dataset=proc["eval"],
        data_collator=(
            _SoftPrefixCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
            if use_soft_mol_prefix
            else DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")
        ),
        tokenizer_for_mask=tokenizer,
        use_masked_ce=bool(cfg.get("masked_ce", False)) and not use_soft_mol_prefix,
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "sft_meta.json").write_text(
        json.dumps(
            {
                "model_name": str(model_name),
                "train_jsonl": str(train_jsonl),
                "val_jsonl": str(val_jsonl),
                "finetune_mode": str(finetune_mode),
                "target_modules": target_modules,
                "token_registration": token_reg,
                **cfg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(out_dir)
