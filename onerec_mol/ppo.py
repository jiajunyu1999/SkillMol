from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .executor import execute_edit_seq
from .grpo import _completion_logprob_and_count_tensor
from .grpo import _infer_target_modules
from .grpo import run_rollout
from .reward import compute_reward
from .vocab import register_domain_tokens


class SequenceValueHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.value_head = nn.Linear(int(hidden_size), 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden_states.to(torch.float32)).squeeze(-1)


def _prepare_prompt_completion_inputs(tokenizer, prompt: str, completion: str, device: torch.device) -> tuple[torch.Tensor, int]:
    prompt_ids = tokenizer(str(prompt), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    full_text = f"{str(prompt).rstrip()}\n{str(completion).strip()}"
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    return full_ids.to(device), int(prompt_ids.shape[0])


def _sequence_value(policy_model, value_head: nn.Module, tokenizer, prompt: str, completion: str) -> torch.Tensor:
    full_ids, prompt_len = _prepare_prompt_completion_inputs(tokenizer, prompt, completion, policy_model.device)
    if int(full_ids.numel()) <= int(prompt_len):
        return torch.zeros((), device=policy_model.device, dtype=torch.float32)
    attention_mask = torch.ones_like(full_ids).unsqueeze(0)
    output = policy_model(
        input_ids=full_ids.unsqueeze(0),
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden = output.hidden_states[-1][0]
    last_completion_idx = int(full_ids.shape[0] - 1)
    return value_head(last_hidden[last_completion_idx]).to(torch.float32)


def _run_mini_eval_if_needed(step: int, out_dir: Path, cfg: dict[str, Any], model_name: str, policy, tokenizer) -> None:
    eval_every = int(cfg.get("eval_every_steps", 0) or 0)
    if eval_every <= 0 or int(step) % eval_every != 0:
        return
    eval_dir = out_dir / f"eval_rows{int(cfg.get('eval_max_rows', 20))}_step{int(step):04d}"
    if (eval_dir / "meta.json").exists():
        return
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_adapter_dir = out_dir / f"eval_adapter_step{int(step):04d}"
    if not (eval_adapter_dir / "adapter_config.json").exists():
        eval_adapter_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(str(eval_adapter_dir))
        tokenizer.save_pretrained(str(eval_adapter_dir))
    cmd = [
        sys.executable,
        "-u",
        "05_infer_rerank/eval_chatdrug_14tasks_unified.py",
        "--model_path",
        str(model_name),
        "--adapter_dir",
        str(eval_adapter_dir),
        "--tokenizer_ckpt",
        str(cfg.get("eval_tokenizer_ckpt", "outputs/run_20260412_workflow1/tokenizer/tokenizer.pt")),
        "--test_csv",
        str(cfg.get("eval_test_csv", "data/test_chatdrug.csv")),
        "--output_dir",
        str(eval_dir),
        "--n_samples",
        str(int(cfg.get("eval_n_samples", 1))),
        "--max_rows",
        str(int(cfg.get("eval_max_rows", 20))),
        "--search_width",
        str(int(cfg.get("eval_search_width", 2))),
        "--search_depth",
        str(int(cfg.get("eval_search_depth", 2))),
        "--extra_prompt_path",
        str(cfg.get("eval_extra_prompt_path", "skill.md")),
        "--no_mol_tokens",
        "--device",
        str(cfg.get("eval_device", "cuda:1")),
    ]
    env = os.environ.copy()
    if str(cfg.get("eval_device", "cuda:1")).strip().lower() == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    log_path = eval_dir / "eval.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        subprocess.run(cmd, cwd=str(Path.cwd()), env=env, stdout=log_f, stderr=subprocess.STDOUT, check=False)
    meta_path = eval_dir / "meta.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        overall = dict(meta.get("overall", {}))
    except Exception:
        return
    curve_path = out_dir / "mini_eval_curve.csv"
    exists = curve_path.exists()
    with curve_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "valid", "loose", "strict", "avg_reward", "eval_dir"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "step": int(step),
                "valid": float(overall.get("valid_ratio", 0.0)),
                "loose": float(overall.get("loose_hit_ratio", 0.0)),
                "strict": float(overall.get("strict_hit_ratio", 0.0)),
                "avg_reward": float(overall.get("avg_reward", 0.0)),
                "eval_dir": str(eval_dir),
            }
        )


def train_ppo(
    model_name: str,
    sft_ckpt_path: str,
    rl_jsonl: str,
    output_dir: str,
    config: dict[str, Any],
) -> str:
    from .grpo import _load_rl_records, _patch_peft_torchao_import

    cfg = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "ref_device": "cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"),
        "seed": 42,
        "steps": 100,
        "batch_size": 2,
        "ppo_epochs": 2,
        "learning_rate": 5e-6,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.001,
        "kl_coef": 0.02,
        "rollout_group_size": 4,
        "grpo_group_size": 4,
        "max_new_tokens": 96,
        "rollout_search_depth": 2,
        "rollout_search_width": 3,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 0,
        "constraint_guidance": True,
        "constrained_decoding": True,
        "reward_mode": "process_valid_hit",
        "invalid_reward": -1.0,
        "process_valid_base": 0.0,
        "process_hit_weight": 0.0,
        "process_loose_reward": 1.0,
        "process_strict_reward": 2.0,
        "process_margin_weight": 1.0,
        "process_margin_cap": 1.0,
        "process_strict_bonus": 0.0,
        "strict_valid_base": 0.5,
        "strict_loose_base": 1.5,
        "strict_hit_base": 4.0,
        "strict_success_weight": 0.5,
        "strict_margin_reward_weight": 0.5,
        "save_every": 20,
        "grad_clip": 1.0,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "codebook_size": 256,
        "max_atom_map": 256,
        "max_fg_id": 64,
        "register_domain_vocab": True,
        "gradient_checkpointing": True,
        "eval_every_steps": 0,
        "eval_max_rows": 20,
        "eval_n_samples": 1,
        "eval_search_width": 2,
        "eval_search_depth": 2,
        "eval_device": "cpu",
        "eval_extra_prompt_path": "skill.md",
        "eval_tokenizer_ckpt": "outputs/run_20260412_workflow1/tokenizer/tokenizer.pt",
        "eval_test_csv": "data/test_chatdrug.csv",
    }
    cfg.update(config or {})

    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    torch.cuda.manual_seed_all(int(cfg["seed"]))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(str(sft_ckpt_path), trust_remote_code=True)
    if bool(cfg.get("register_domain_vocab", True)):
        register_domain_tokens(
            tokenizer,
            codebook_size=int(cfg["codebook_size"]),
            max_atom_map=int(cfg["max_atom_map"]),
            max_fg_id=int(cfg["max_fg_id"]),
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token_id is not None else "[PAD]"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(str(model_name), torch_dtype=dtype)
    if int(base.get_input_embeddings().num_embeddings) != int(len(tokenizer)):
        base.resize_token_embeddings(len(tokenizer))
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tokenizer.pad_token_id
    _patch_peft_torchao_import()
    policy = PeftModel.from_pretrained(base, str(sft_ckpt_path), is_trainable=True)

    ref_base = AutoModelForCausalLM.from_pretrained(str(model_name), torch_dtype=dtype)
    if int(ref_base.get_input_embeddings().num_embeddings) != int(len(tokenizer)):
        ref_base.resize_token_embeddings(len(tokenizer))
    if getattr(ref_base.config, "pad_token_id", None) is None:
        ref_base.config.pad_token_id = tokenizer.pad_token_id
    _patch_peft_torchao_import()
    ref_policy = PeftModel.from_pretrained(ref_base, str(sft_ckpt_path), is_trainable=False)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    hidden_size = int(getattr(policy.config, "hidden_size"))
    value_head = SequenceValueHead(hidden_size)

    tokenizer.save_pretrained(str(out_dir))

    device = torch.device(str(cfg["device"]))
    ref_device = torch.device(str(cfg.get("ref_device", cfg["device"])))
    policy.to(device)
    ref_policy.to(ref_device)
    value_head.to(device)
    policy.train()
    policy.config.use_cache = False
    if bool(cfg.get("gradient_checkpointing", False)):
        try:
            policy.gradient_checkpointing_enable()
            if hasattr(policy, "enable_input_require_grads"):
                policy.enable_input_require_grads()
        except Exception:
            pass
    ref_policy.config.use_cache = False

    trainable = [p for p in policy.parameters() if p.requires_grad] + [p for p in value_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]))

    records = _load_rl_records(str(rl_jsonl))
    gen_cfg = {
        "tokenizer": tokenizer,
        "group_size": int(cfg.get("rollout_group_size", cfg.get("grpo_group_size", 4))),
        "do_sample": True,
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "rollout_search_depth": int(cfg.get("rollout_search_depth", 2)),
        "rollout_search_width": int(cfg.get("rollout_search_width", 3)),
        "temperature": float(cfg["temperature"]),
        "top_p": float(cfg["top_p"]),
        "top_k": int(cfg["top_k"]),
        "constraint_guidance": bool(cfg.get("constraint_guidance", True)),
        "constrained_decoding": bool(cfg.get("constrained_decoding", True)),
        "invalid_reward": float(cfg.get("invalid_reward", -1.0)),
        "reward_mode": str(cfg.get("reward_mode", "process_valid_hit")),
        "process_valid_base": float(cfg.get("process_valid_base", 0.0)),
        "process_hit_weight": float(cfg.get("process_hit_weight", 0.0)),
        "process_strict_bonus": float(cfg.get("process_strict_bonus", 0.0)),
        "process_loose_reward": float(cfg.get("process_loose_reward", 1.0)),
        "process_strict_reward": float(cfg.get("process_strict_reward", 2.0)),
        "process_margin_weight": float(cfg.get("process_margin_weight", 1.0)),
        "process_margin_cap": float(cfg.get("process_margin_cap", 1.0)),
        "strict_valid_base": float(cfg.get("strict_valid_base", 0.5)),
        "strict_loose_base": float(cfg.get("strict_loose_base", 1.5)),
        "strict_hit_base": float(cfg.get("strict_hit_base", 4.0)),
        "strict_success_weight": float(cfg.get("strict_success_weight", 0.5)),
        "strict_margin_reward_weight": float(cfg.get("strict_margin_reward_weight", 0.5)),
        "use_recoverability_shaping": False,
    }

    for step in range(1, int(cfg["steps"]) + 1):
        batch = random.choices(records, k=min(int(cfg["batch_size"]), len(records)))
        samples: list[dict[str, Any]] = []

        with torch.no_grad():
            for rollout_id, rec in enumerate(batch):
                rollout = run_rollout(rec, (policy, tokenizer), compute_reward, execute_edit_seq, gen_cfg)
                for cand in rollout:
                    prompt = str(rec["prompt"])
                    completion = str(cand.get("completion", ""))
                    old_sum = float(cand.get("logprob", 0.0))
                    old_tok = max(1, int(cand.get("token_count", 0)))
                    old_avg = float(old_sum / float(old_tok))
                    ref_sum_t, ref_tok = _completion_logprob_and_count_tensor(ref_policy, tokenizer, prompt, completion)
                    ref_avg = float((ref_sum_t / float(max(1, int(ref_tok)))).item()) if int(ref_tok) > 0 else 0.0
                    env_reward = float(cand.get("reward_result", {}).get("reward", 0.0))
                    total_reward = float(env_reward - float(cfg["kl_coef"]) * (old_avg - ref_avg))
                    value_pred = float(_sequence_value(policy, value_head, tokenizer, prompt, completion).item())
                    samples.append(
                        {
                            "prompt": prompt,
                            "completion": completion,
                            "reward": total_reward,
                            "env_reward": env_reward,
                            "old_logprob_avg": old_avg,
                            "old_value": value_pred,
                            "task_id": int(cand.get("task_id", -1)),
                            "execute_result": dict(cand.get("execute_result", {})),
                            "reward_result": dict(cand.get("reward_result", {})),
                            "rollout_id": int(rollout_id),
                        }
                    )

        if not samples:
            continue

        rewards_t = torch.tensor([float(x["reward"]) for x in samples], device=device, dtype=torch.float32)
        values_t = torch.tensor([float(x["old_value"]) for x in samples], device=device, dtype=torch.float32)
        advantages_t = rewards_t - values_t
        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / advantages_t.std(unbiased=False).clamp_min(1e-6)
        for idx, item in enumerate(samples):
            item["advantage"] = float(advantages_t[idx].item())

        epoch_losses: list[float] = []
        for _ in range(max(1, int(cfg.get("ppo_epochs", 1)))):
            optimizer.zero_grad(set_to_none=True)
            losses: list[torch.Tensor] = []
            for item in samples:
                prompt = str(item["prompt"])
                completion = str(item["completion"])
                new_sum_t, new_tok = _completion_logprob_and_count_tensor(policy, tokenizer, prompt, completion)
                new_avg = new_sum_t / float(max(1, int(new_tok)))
                old_avg_t = torch.tensor(float(item["old_logprob_avg"]), device=device, dtype=torch.float32)
                advantage = torch.tensor(float(item["advantage"]), device=device, dtype=torch.float32)
                ratio = torch.exp(new_avg - old_avg_t)
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - float(cfg["clip_eps"]), 1.0 + float(cfg["clip_eps"])) * advantage
                policy_loss = -torch.min(unclipped, clipped)

                value_pred = _sequence_value(policy, value_head, tokenizer, prompt, completion)
                reward_target = torch.tensor(float(item["reward"]), device=device, dtype=torch.float32)
                old_value_t = torch.tensor(float(item["old_value"]), device=device, dtype=torch.float32)
                value_clipped = old_value_t + torch.clamp(
                    value_pred - old_value_t,
                    -float(cfg["clip_eps"]),
                    float(cfg["clip_eps"]),
                )
                value_loss = 0.5 * torch.max(
                    torch.square(value_pred - reward_target),
                    torch.square(value_clipped - reward_target),
                )
                # Reuse the current policy logprob estimate as an entropy proxy to avoid an extra forward pass.
                entropy_proxy = (-new_avg).to(torch.float32)
                loss = policy_loss + float(cfg["value_coef"]) * value_loss - float(cfg["entropy_coef"]) * entropy_proxy
                losses.append(loss)
            if not losses:
                continue
            total_loss = torch.stack(losses).mean()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(cfg["grad_clip"]))
            optimizer.step()
            epoch_losses.append(float(total_loss.item()))

        loss_value = float(sum(epoch_losses) / max(len(epoch_losses), 1)) if epoch_losses else 0.0
        avg_reward = float(sum(float(x["reward"]) for x in samples) / max(len(samples), 1))
        avg_env_reward = float(sum(float(x["env_reward"]) for x in samples) / max(len(samples), 1))
        valid_ratio = float(
            sum(1.0 for x in samples if bool(x["execute_result"].get("is_valid_mol", False))) / max(len(samples), 1)
        )
        strict_ratio = float(
            sum(1.0 for x in samples if bool(x["reward_result"].get("metrics", {}).get("strict_hit", False))) / max(len(samples), 1)
        )
        loose_ratio = float(
            sum(1.0 for x in samples if bool(x["reward_result"].get("metrics", {}).get("loose_hit", False))) / max(len(samples), 1)
        )
        format_ratio = float(
            sum(1.0 for x in samples if bool(x["execute_result"].get("is_valid_syntax", False))) / max(len(samples), 1)
        )
        kl_mean = float(
            sum(float(x["old_logprob_avg"]) for x in samples) / max(len(samples), 1)
            - sum(
                float(
                    (
                        _completion_logprob_and_count_tensor(ref_policy, tokenizer, str(x["prompt"]), str(x["completion"]))[0]
                        / float(max(1, _completion_logprob_and_count_tensor(ref_policy, tokenizer, str(x["prompt"]), str(x["completion"]))[1]))
                    ).item()
                )
                for x in samples
            )
            / max(len(samples), 1)
        )
        log_row = {
            "step": int(step),
            "loss": float(loss_value),
            "avg_reward": float(avg_reward),
            "avg_env_reward": float(avg_env_reward),
            "valid_mol_ratio": float(valid_ratio),
            "strict_hit_ratio": float(strict_ratio),
            "loose_hit_ratio": float(loose_ratio),
            "format_valid_ratio": float(format_ratio),
            "approx_kl_to_ref": float(kl_mean),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
        print(json.dumps(log_row, ensure_ascii=False))

        if int(cfg.get("save_every", 0)) > 0 and step % int(cfg["save_every"]) == 0:
            ckpt = out_dir / f"checkpoint-{step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            torch.save(value_head.state_dict(), ckpt / "value_head.pt")
        _run_mini_eval_if_needed(step, out_dir, cfg, model_name, policy, tokenizer)

    policy.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    torch.save(value_head.state_dict(), out_dir / "value_head.pt")
    (out_dir / "ppo_meta.json").write_text(
        json.dumps(
            {
                "model_name": str(model_name),
                "sft_ckpt_path": str(sft_ckpt_path),
                "rl_jsonl": str(rl_jsonl),
                **cfg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(out_dir)
