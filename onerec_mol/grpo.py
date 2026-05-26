from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessor
from transformers import LogitsProcessorList

from .executor import execute_edit_seq
from .executor import format_action_constraint_guidance
from .executor import get_action_constraints
from .reward import compute_reward
from .token_policy import suggest_fg_ids_for_task
from .vocab import load_fg_id_to_smiles
from .vocab import register_domain_tokens


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


def _infer_target_modules(model) -> list[str]:
    suffixes: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffixes.add(name.split(".")[-1])
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"]
    target = [x for x in preferred if x in suffixes]
    return target if target else sorted(suffixes)


def _resolve_model_and_tokenizer(model, gen_config: dict[str, Any]) -> tuple[Any, Any]:
    if isinstance(model, dict):
        m = model.get("model")
        tok = model.get("tokenizer")
        if m is None or tok is None:
            raise ValueError("Model dict must contain `model` and `tokenizer`.")
        return m, tok
    if isinstance(model, tuple) and len(model) == 2:
        return model[0], model[1]
    tok = (gen_config or {}).get("tokenizer")
    if tok is None:
        raise ValueError("Tokenizer is required via gen_config['tokenizer'] or model dict/tuple.")
    return model, tok


def _completion_logprob_from_output(model, output_ids: torch.Tensor, prompt_len: int) -> tuple[float, int]:
    with torch.no_grad():
        logits = model(input_ids=output_ids.unsqueeze(0), attention_mask=torch.ones_like(output_ids).unsqueeze(0)).logits[0]
    if int(output_ids.numel()) <= int(prompt_len):
        return 0.0, 0
    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    target_ids = output_ids[1:]
    start = max(int(prompt_len) - 1, 0)
    token_log_probs = log_probs[start:, :].gather(1, target_ids[start:].unsqueeze(1)).squeeze(1)
    return float(token_log_probs.sum().item()), int(token_log_probs.numel())


def _completion_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    prompt_ids = tokenizer(str(prompt), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    full_text = f"{str(prompt).rstrip()}\n{str(completion).strip()}"
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
    prompt_len = int(prompt_ids.shape[0])
    score, _ = _completion_logprob_from_output(model, full_ids, prompt_len=prompt_len)
    return float(score)


def _completion_logprob_and_count_tensor(model, tokenizer, prompt: str, completion: str) -> tuple[torch.Tensor, int]:
    prompt_ids = tokenizer(str(prompt), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    full_text = f"{str(prompt).rstrip()}\n{str(completion).strip()}"
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
    prompt_len = int(prompt_ids.shape[0])
    logits = model(input_ids=full_ids.unsqueeze(0), attention_mask=torch.ones_like(full_ids).unsqueeze(0)).logits[0]
    if int(full_ids.numel()) <= int(prompt_len):
        return torch.zeros((), device=model.device, dtype=torch.float32), 0
    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    target_ids = full_ids[1:]
    start = max(int(prompt_len) - 1, 0)
    token_log_probs = log_probs[start:, :].gather(1, target_ids[start:].unsqueeze(1)).squeeze(1)
    if token_log_probs.numel() == 0:
        return torch.zeros((), device=model.device, dtype=torch.float32), 0
    return token_log_probs.sum(), int(token_log_probs.numel())


def _completion_logprob_tensor(model, tokenizer, prompt: str, completion: str) -> torch.Tensor:
    score, _ = _completion_logprob_and_count_tensor(model, tokenizer, prompt, completion)
    return score


def _valid_generated_mask(
    gen_ids: torch.Tensor,
    *,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> torch.Tensor:
    mask = torch.ones_like(gen_ids, dtype=torch.bool)
    first_eos = None
    if eos_token_id is not None:
        eos_pos = (gen_ids == int(eos_token_id)).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            first_eos = int(eos_pos[0].item())
            if first_eos + 1 < int(mask.numel()):
                mask[first_eos + 1 :] = False
    if pad_token_id is not None:
        pad_mask = gen_ids.ne(int(pad_token_id))
        if first_eos is not None and int(pad_token_id) == int(eos_token_id):
            pad_mask[first_eos] = True
        mask &= pad_mask
    return mask


def _rows_from_generate_output(hf_model, tokenizer, prompt_len: int, gen_output) -> list[dict[str, Any]]:
    sequences = gen_output.sequences
    rows: list[dict[str, Any]] = []

    transition_scores = None
    if getattr(gen_output, "scores", None):
        try:
            transition_scores = hf_model.compute_transition_scores(
                sequences,
                gen_output.scores,
                normalize_logits=True,
            )
        except Exception:  # noqa: BLE001
            transition_scores = None

    for i in range(sequences.shape[0]):
        full_ids = sequences[i]
        gen_ids = full_ids[prompt_len:]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if transition_scores is None:
            logprob, token_count = _completion_logprob_from_output(hf_model, full_ids, prompt_len=prompt_len)
        else:
            row_scores = transition_scores[i]
            valid_mask = _valid_generated_mask(
                gen_ids[: row_scores.shape[0]],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if valid_mask.numel() == 0:
                logprob = 0.0
                token_count = 0
            else:
                picked = row_scores[: valid_mask.shape[0]][valid_mask]
                logprob = float(picked.sum().item()) if picked.numel() > 0 else 0.0
                token_count = int(picked.numel())

        rows.append(
            {
                "completion": completion,
                "logprob": float(logprob),
                "token_count": int(token_count),
            }
        )
    return rows


class EditGrammarConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        *,
        tokenizer,
        prompt_lengths: list[int],
        action_constraints: list[dict[str, Any]],
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_lengths = [int(x) for x in prompt_lengths]
        self.vocab = dict(tokenizer.get_vocab())
        self.unk_token_id = tokenizer.unk_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

        self.edit_set_id = self._token_id_required("<EDIT_SET>")
        self.end_edit_set_id = self._token_id_required("</EDIT_SET>")
        self.edit_id = self._token_id_required("<EDIT>")
        self.end_edit_id = self._token_id_required("</EDIT>")
        self.op_add_id = self._token_id_required("<OP_ADD>")
        self.op_remove_id = self._token_id_required("<OP_REMOVE>")
        self.op_replace_id = self._token_id_required("<OP_REPLACE>")
        self.anchor_id = self._token_id_required("<ANCHOR>")
        self.fgid_id = self._token_id_required("<FGID>")
        self.fgsmi_id = self._token_id_required("<FGSMI>")
        self.rmatom_id = self._token_id_required("<RMATOM>")
        def _looks_like_control_token(token_text: str) -> bool:
            t = str(token_text)
            t = t.lstrip(" \t\r\nĠ▁")
            return t.startswith("<") and t.endswith(">")

        self.domain_control_token_ids = sorted(
            {
                int(tok_id)
                for tok, tok_id in tokenizer.get_vocab().items()
                if _looks_like_control_token(str(tok))
            }
        )
        self.whitespace_only_token_ids = sorted(
            {
                int(tok_id)
                for tok, tok_id in tokenizer.get_vocab().items()
                if str(tok).lstrip(" \t\r\nĠ▁").strip() == ""
            }
        )

        self.fg_id_tokens: list[int] = []
        for fgid in sorted(load_fg_id_to_smiles()):
            tok_id = self._token_id_optional(f"<FGID_{int(fgid)}>")
            if tok_id is not None:
                self.fg_id_tokens.append(int(tok_id))
        self.fg_id_tokens = sorted(set(self.fg_id_tokens))
        self.fg_id_token_set = set(int(x) for x in self.fg_id_tokens)

        self.numeric_token_ids_by_value: dict[int, list[int]] = {}
        for tok, tok_id in self.vocab.items():
            text = str(tok)
            text = text.lstrip(" \t\r\nĠ▁")
            if not re.fullmatch(r"[0-9]+", text):
                continue
            val = int(text)
            self.numeric_token_ids_by_value.setdefault(val, []).append(int(tok_id))
        for key in list(self.numeric_token_ids_by_value):
            self.numeric_token_ids_by_value[key] = sorted(set(self.numeric_token_ids_by_value[key]))

        fg_id_to_smiles = {int(k): str(v) for k, v in load_fg_id_to_smiles().items()}
        self.fg_id_value_by_token_id: dict[int, int] = {}
        for fgid in sorted(fg_id_to_smiles):
            for tok_id in self._fgid_value_token_ids(int(fgid)):
                self.fg_id_value_by_token_id[int(tok_id)] = int(fgid)

        self.fg_smiles_token_ids_by_fg_id: dict[int, list[int]] = {}
        tokenizer_call = getattr(self.tokenizer, "__call__", None)
        if callable(tokenizer_call):
            for fgid, fg_smiles in fg_id_to_smiles.items():
                try:
                    token_ids = tokenizer_call(str(fg_smiles), add_special_tokens=False).get("input_ids", [])
                    ids = [int(x) for x in token_ids]
                except Exception:  # noqa: BLE001
                    ids = []
                if ids:
                    self.fg_smiles_token_ids_by_fg_id[int(fgid)] = ids

        self.valid_anchor_tokens_by_row: list[list[int]] = []
        self.removable_anchor_tokens_by_row: list[list[int]] = []
        self.removable_token_map_by_row: list[dict[int, list[int]]] = []
        self.preferred_fg_tokens_by_row: list[list[int]] = []
        self.first_op_whitelist_by_row: list[list[int]] = []
        self.max_edits_by_row: list[int | None] = []
        for raw in action_constraints:
            constraints = dict(raw or {})
            anchor_maps = [int(x) for x in constraints.get("valid_anchor_maps", [])]
            preferred_fg_ids = [int(x) for x in constraints.get("preferred_fg_ids", [])]
            removable_by_anchor = {
                int(anchor): [int(x) for x in targets]
                for anchor, targets in dict(constraints.get("removable_by_anchor", {})).items()
            }
            valid_anchor_token_set: set[int] = set()
            removable_token_map: dict[int, list[int]] = {}
            for anchor, targets in removable_by_anchor.items():
                anchor_token_ids = self._anchor_value_token_ids(int(anchor))
                target_token_ids: list[int] = []
                for target in targets:
                    target_token_ids.extend(self._anchor_value_token_ids(int(target)))
                unique_target_ids = sorted(set(int(x) for x in target_token_ids))
                if not unique_target_ids:
                    continue
                for anchor_tok in anchor_token_ids:
                    removable_token_map[int(anchor_tok)] = unique_target_ids
                valid_anchor_token_set.update(int(x) for x in anchor_token_ids)
            for anchor in anchor_maps:
                valid_anchor_token_set.update(self._anchor_value_token_ids(int(anchor)))
            self.valid_anchor_tokens_by_row.append(sorted(valid_anchor_token_set))
            self.removable_anchor_tokens_by_row.append(sorted(set(int(x) for x in removable_token_map)))
            self.removable_token_map_by_row.append(removable_token_map)
            preferred_fg_token_set: set[int] = set()
            for fgid in preferred_fg_ids:
                preferred_fg_token_set.update(int(x) for x in self._fgid_value_token_ids(int(fgid)))
            self.preferred_fg_tokens_by_row.append(sorted(preferred_fg_token_set))
            raw_ops = constraints.get("first_op_whitelist", [])
            first_ops: set[int] = set()
            if isinstance(raw_ops, str):
                raw_ops = [x.strip() for x in raw_ops.split(",") if x.strip()]
            if isinstance(raw_ops, (list, tuple, set)):
                for op in raw_ops:
                    tok = self._normalize_op_token_id(op)
                    if tok is not None:
                        first_ops.add(int(tok))
            self.first_op_whitelist_by_row.append(sorted(first_ops))
            raw_max_edits = constraints.get("max_edits")
            max_edits: int | None = None
            if raw_max_edits is not None:
                try:
                    val = int(raw_max_edits)
                    if val > 0:
                        max_edits = int(val)
                except Exception:  # noqa: BLE001
                    max_edits = None
            self.max_edits_by_row.append(max_edits)

        terminal = []
        if self.eos_token_id is not None:
            terminal.append(int(self.eos_token_id))
        if self.pad_token_id is not None:
            terminal.append(int(self.pad_token_id))
        if not terminal and self.end_edit_set_id is not None:
            terminal.append(int(self.end_edit_set_id))
        self.terminal_token_ids = sorted(set(terminal))

    def _token_id_optional(self, token: str) -> int | None:
        raw = self.vocab.get(str(token))
        if raw is not None:
            return int(raw)
        tok_id = self.tokenizer.convert_tokens_to_ids(str(token))
        if tok_id is None:
            return None
        tok_id = int(tok_id)
        if self.unk_token_id is not None and int(tok_id) == int(self.unk_token_id):
            return None
        return tok_id

    def _token_id_required(self, token: str) -> int:
        tok_id = self._token_id_optional(str(token))
        if tok_id is None:
            raise ValueError(f"Required DSL token missing in tokenizer vocab: {token}")
        return int(tok_id)

    def _anchor_value_token_ids(self, value: int) -> list[int]:
        val = int(value)
        out: set[int] = set()
        amap_tok = self._token_id_optional(f"<AMAP_{val}>")
        if amap_tok is not None:
            out.add(int(amap_tok))
        out.update(int(x) for x in self.numeric_token_ids_by_value.get(val, []))
        return sorted(out)

    def _fgid_value_token_ids(self, value: int) -> list[int]:
        val = int(value)
        out: set[int] = set()
        fgid_tok = self._token_id_optional(f"<FGID_{val}>")
        if fgid_tok is not None:
            out.add(int(fgid_tok))
        out.update(int(x) for x in self.numeric_token_ids_by_value.get(val, []))
        return sorted(out)

    def _content_policy(self, allowed_control_ids: list[int], *, allow_fg_id_tokens: bool) -> dict[str, Any]:
        blocked = set(self.domain_control_token_ids)
        blocked.update(int(x) for x in self.whitespace_only_token_ids)
        blocked.difference_update(int(x) for x in allowed_control_ids)
        if allow_fg_id_tokens:
            blocked.difference_update(int(x) for x in self.fg_id_tokens)
        if self.eos_token_id is not None:
            blocked.add(int(self.eos_token_id))
        if self.pad_token_id is not None:
            blocked.add(int(self.pad_token_id))
        return {"kind": "block", "ids": sorted(blocked)}

    def _terminal_policy(self) -> dict[str, Any]:
        return {"kind": "allow", "ids": list(self.terminal_token_ids)}

    def _selected_fgid_from_generated(self, generated_ids: list[int]) -> int | None:
        ids = [int(x) for x in generated_ids]
        for i, tok_id in enumerate(ids):
            if int(tok_id) != int(self.fgid_id):
                continue
            if i + 1 >= len(ids):
                return None
            return self._fgid_value_from_token_id(int(ids[i + 1]))
        return None

    def _fgid_value_from_token_id(self, token_id: int) -> int | None:
        out = self.fg_id_value_by_token_id.get(int(token_id))
        if out is None:
            return None
        return int(out)

    def _normalize_op_token_id(self, raw_op: Any) -> int | None:
        if raw_op is None:
            return None
        if isinstance(raw_op, int):
            if int(raw_op) in {int(self.op_add_id), int(self.op_remove_id), int(self.op_replace_id)}:
                return int(raw_op)
            return None
        text = str(raw_op).strip().upper()
        if not text:
            return None
        if text.startswith("<") and text.endswith(">"):
            text = text[1:-1]
        if text.startswith("OP_"):
            text = text[3:]
        if text == "ADD":
            return int(self.op_add_id)
        if text == "REMOVE":
            return int(self.op_remove_id)
        if text == "REPLACE":
            return int(self.op_replace_id)
        return None

    def _fgsmi_exact_sequence_policy(self, generated_ids: list[int], *, followup_token_id: int) -> dict[str, Any]:
        ids = [int(x) for x in generated_ids]
        fgsmi_positions = [i for i, tok in enumerate(ids) if int(tok) == int(self.fgsmi_id)]
        if not fgsmi_positions:
            return self._terminal_policy()
        fgsmi_idx = int(fgsmi_positions[-1])
        fgid = self._selected_fgid_from_generated(ids[:fgsmi_idx])
        if fgid is None:
            return self._terminal_policy()
        target = list(self.fg_smiles_token_ids_by_fg_id.get(int(fgid), []))
        if not target:
            return {"kind": "allow", "ids": [int(followup_token_id)]}

        emitted = ids[fgsmi_idx + 1 :]
        shared = min(len(emitted), len(target))
        if emitted[:shared] != target[:shared]:
            return self._terminal_policy()
        if len(emitted) < len(target):
            return {"kind": "allow", "ids": [int(target[len(emitted)])]}
        if len(emitted) == len(target):
            return {"kind": "allow", "ids": [int(followup_token_id)]}
        return self._terminal_policy()

    def _allowed_tokens_for_partial_edit(
        self,
        *,
        edit_tokens: list[int],
        valid_anchor_tokens: list[int],
        removable_anchor_tokens: list[int],
        removable_token_map: dict[int, list[int]],
        allowed_fg_tokens: list[int],
        current_edit_index: int,
        first_op_whitelist_token_ids: list[int],
        max_edit_count: int | None,
    ) -> dict[str, Any]:
        if max_edit_count is not None and int(current_edit_index) > int(max_edit_count):
            return self._terminal_policy()
        if not edit_tokens or int(edit_tokens[0]) != int(self.edit_id):
            return self._terminal_policy()
        if len(edit_tokens) == 1:
            allowed_ops: list[int] = []
            if valid_anchor_tokens:
                allowed_ops.append(int(self.op_add_id))
            if removable_anchor_tokens:
                allowed_ops.extend([int(self.op_remove_id), int(self.op_replace_id)])
            if int(current_edit_index) == 1 and first_op_whitelist_token_ids:
                allowed_ops = [x for x in allowed_ops if int(x) in set(int(y) for y in first_op_whitelist_token_ids)]
            if not allowed_ops:
                return self._terminal_policy()
            return {"kind": "allow", "ids": sorted(set(allowed_ops))}

        op_id = int(edit_tokens[1]) if len(edit_tokens) >= 2 else -1
        if len(edit_tokens) == 2:
            return {"kind": "allow", "ids": [self.anchor_id]}
        if len(edit_tokens) == 3:
            if op_id == self.op_add_id:
                if not valid_anchor_tokens:
                    return self._terminal_policy()
                return {"kind": "allow", "ids": valid_anchor_tokens}
            if op_id in {self.op_remove_id, self.op_replace_id}:
                if not removable_anchor_tokens:
                    return self._terminal_policy()
                return {"kind": "allow", "ids": removable_anchor_tokens}
            return self._terminal_policy()

        anchor_token_id = int(edit_tokens[3]) if len(edit_tokens) >= 4 else -1
        removable_targets = list(removable_token_map.get(anchor_token_id, []))

        if op_id == self.op_add_id:
            if len(edit_tokens) == 4:
                return {"kind": "allow", "ids": [self.fgid_id]}
            if edit_tokens and int(edit_tokens[-1]) == self.fgid_id:
                return {"kind": "allow", "ids": list(allowed_fg_tokens)}
            if (
                len(edit_tokens) >= 2
                and int(edit_tokens[-2]) == int(self.fgid_id)
                and self._fgid_value_from_token_id(int(edit_tokens[-1])) is not None
            ):
                return {"kind": "allow", "ids": [self.fgsmi_id, self.end_edit_id]}
            if self.fgsmi_id in edit_tokens[4:]:
                return self._fgsmi_exact_sequence_policy(edit_tokens, followup_token_id=self.end_edit_id)
            return self._terminal_policy()

        if op_id == self.op_remove_id:
            if len(edit_tokens) == 4:
                if not removable_targets:
                    return self._terminal_policy()
                if len(removable_targets) == 1:
                    return {"kind": "allow", "ids": [self.rmatom_id, self.end_edit_id]}
                return {"kind": "allow", "ids": [self.rmatom_id]}
            if len(edit_tokens) == 5:
                if not removable_targets:
                    return self._terminal_policy()
                return {"kind": "allow", "ids": removable_targets}
            if len(edit_tokens) == 6:
                return {"kind": "allow", "ids": [self.end_edit_id]}
            return self._terminal_policy()

        if op_id == self.op_replace_id:
            if len(edit_tokens) == 4:
                return {"kind": "allow", "ids": [self.fgid_id]}
            if edit_tokens and int(edit_tokens[-1]) == self.fgid_id:
                return {"kind": "allow", "ids": list(allowed_fg_tokens)}
            if (
                len(edit_tokens) >= 2
                and int(edit_tokens[-2]) == int(self.fgid_id)
                and self._fgid_value_from_token_id(int(edit_tokens[-1])) is not None
            ):
                return {"kind": "allow", "ids": [self.fgsmi_id]}
            if (
                len(edit_tokens) >= 2
                and int(edit_tokens[-2]) == self.rmatom_id
                and int(edit_tokens[-1]) in set(removable_targets)
            ):
                return {"kind": "allow", "ids": [self.end_edit_id]}
            if edit_tokens and int(edit_tokens[-1]) == self.rmatom_id:
                if not removable_targets:
                    return self._terminal_policy()
                return {"kind": "allow", "ids": removable_targets}
            if self.fgsmi_id in edit_tokens[4:]:
                return self._fgsmi_exact_sequence_policy(edit_tokens, followup_token_id=self.rmatom_id)
            return self._terminal_policy()

        return self._terminal_policy()

    def _allowed_tokens_for_generated(self, row_idx: int, generated_ids: list[int]) -> dict[str, Any] | None:
        prompt_idx = int(row_idx % max(len(self.prompt_lengths), 1))
        valid_anchor_tokens = list(self.valid_anchor_tokens_by_row[prompt_idx])
        removable_anchor_tokens = list(self.removable_anchor_tokens_by_row[prompt_idx])
        removable_token_map = dict(self.removable_token_map_by_row[prompt_idx])
        preferred_fg_tokens = list(self.preferred_fg_tokens_by_row[prompt_idx])
        first_op_whitelist_token_ids = list(self.first_op_whitelist_by_row[prompt_idx])
        max_edit_count = self.max_edits_by_row[prompt_idx]
        allowed_fg_tokens = (
            sorted(set(int(x) for x in preferred_fg_tokens))
            if preferred_fg_tokens
            else list(self.fg_id_tokens)
        )

        if not generated_ids:
            return {"kind": "allow", "ids": [self.edit_set_id]}
        if int(generated_ids[0]) != int(self.edit_set_id):
            return self._terminal_policy()
        body = [int(x) for x in generated_ids[1:]]
        if not body:
            return {"kind": "allow", "ids": [self.edit_id]}
        if int(body[-1]) == int(self.end_edit_set_id):
            return self._terminal_policy()

        open_edit: list[int] = []
        closed_edit_count = 0
        for i, tok in enumerate(body):
            tok = int(tok)
            if not open_edit:
                if tok == int(self.end_edit_set_id):
                    if i == len(body) - 1 and int(closed_edit_count) > 0:
                        return self._terminal_policy()
                    return self._terminal_policy()
                if tok != int(self.edit_id):
                    return self._terminal_policy()
                open_edit = [int(self.edit_id)]
                continue
            open_edit.append(int(tok))
            if tok == int(self.end_edit_id):
                open_edit = []
                closed_edit_count += 1

        if not open_edit:
            if int(closed_edit_count) <= 0:
                return {"kind": "allow", "ids": [self.edit_id]}
            if max_edit_count is not None and int(closed_edit_count) >= int(max_edit_count):
                return {"kind": "allow", "ids": [self.end_edit_set_id]}
            return {"kind": "allow", "ids": [self.edit_id, self.end_edit_set_id]}

        current_edit_index = int(closed_edit_count) + 1
        return self._allowed_tokens_for_partial_edit(
            edit_tokens=open_edit,
            valid_anchor_tokens=valid_anchor_tokens,
            removable_anchor_tokens=removable_anchor_tokens,
            removable_token_map=removable_token_map,
            allowed_fg_tokens=allowed_fg_tokens,
            current_edit_index=current_edit_index,
            first_op_whitelist_token_ids=first_op_whitelist_token_ids,
            max_edit_count=max_edit_count,
        )

    def next_token_policy(self, generated_ids: list[int], row_idx: int = 0) -> dict[str, Any] | None:
        return self._allowed_tokens_for_generated(int(row_idx), [int(x) for x in generated_ids])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        constrained = scores.clone()
        for row_idx in range(input_ids.shape[0]):
            prompt_len = int(self.prompt_lengths[int(row_idx % max(len(self.prompt_lengths), 1))])
            generated_ids = [int(x) for x in input_ids[row_idx, prompt_len:].tolist()]
            policy = self._allowed_tokens_for_generated(row_idx, generated_ids)
            if not policy:
                continue
            kind = str(policy.get("kind", ""))
            ids = sorted(set(int(x) for x in policy.get("ids", [])))
            if kind == "allow":
                if not ids:
                    ids = list(self.terminal_token_ids)
                if not ids:
                    continue
                mask = torch.full_like(constrained[row_idx], float("-inf"))
                allowed_tensor = torch.tensor(ids, device=constrained.device, dtype=torch.long)
                mask[allowed_tensor] = constrained[row_idx, allowed_tensor]
                constrained[row_idx] = mask
            elif kind == "block":
                if not ids:
                    continue
                blocked_tensor = torch.tensor(ids, device=constrained.device, dtype=torch.long)
                constrained[row_idx, blocked_tensor] = float("-inf")
        return constrained


def _build_logits_processors(
    *,
    tokenizer,
    prompt_lengths: list[int],
    gen_config: dict[str, Any],
) -> LogitsProcessorList:
    processors = LogitsProcessorList()
    if not bool((gen_config or {}).get("constrained_decoding", True)):
        return processors
    raw_constraints = list((gen_config or {}).get("action_constraints", []))
    if raw_constraints and len(raw_constraints) != len(prompt_lengths):
        if len(raw_constraints) == 1:
            constraints = [dict(raw_constraints[0]) for _ in prompt_lengths]
        else:
            repeat = max(1, int(len(prompt_lengths) / max(len(raw_constraints), 1)))
            constraints = []
            for item in raw_constraints:
                constraints.extend([dict(item)] * repeat)
            constraints = constraints[: len(prompt_lengths)]
    else:
        constraints = raw_constraints
    if not constraints:
        return processors
    processors.append(
        EditGrammarConstrainedLogitsProcessor(
            tokenizer=tokenizer,
            prompt_lengths=prompt_lengths,
            action_constraints=constraints,
        )
    )
    return processors


def _augment_prompt_with_constraints(prompt: str, meta: dict[str, Any], gen_config: dict[str, Any]) -> str:
    if not bool((gen_config or {}).get("constraint_guidance", True)):
        return str(prompt)
    start_smiles_tagged = str((meta or {}).get("start_smiles_tagged", "")).strip()
    if not start_smiles_tagged:
        return str(prompt)
    suffix = format_action_constraint_guidance(start_smiles_tagged)
    if not suffix.strip():
        return str(prompt)
    base = str(prompt).rstrip()
    return f"{base}\n\n{suffix}"


def sample_completions(model, prompt: str, group_size: int, gen_config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = dict(gen_config or {})
    sampler = cfg.get("sampler")
    if callable(sampler):
        raw = sampler(prompt=prompt, group_size=int(group_size), gen_config=cfg)
        out = []
        for item in raw:
            out.append(
                {
                    "completion": str(item.get("completion", "")),
                    "logprob": float(item.get("logprob", 0.0)),
                }
            )
        return out

    hf_model, tokenizer = _resolve_model_and_tokenizer(model, cfg)
    do_sample = bool(cfg.get("do_sample", True))
    max_new_tokens = int(cfg.get("max_new_tokens", 128))
    temperature = float(cfg.get("temperature", 0.8))
    top_p = float(cfg.get("top_p", 0.95))
    top_k = int(cfg.get("top_k", 0))

    encoded = tokenizer(str(prompt), return_tensors="pt").to(hf_model.device)
    prompt_len = int(encoded["input_ids"].shape[1])
    logits_processor = _build_logits_processors(
        tokenizer=tokenizer,
        prompt_lengths=[prompt_len] * max(1, int(group_size)),
        gen_config=cfg,
    )

    gen_kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": int(group_size),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "logits_processor": logits_processor,
        # Avoid expensive auto-compile warmup in constrained decoding loops.
        "disable_compile": True,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        if top_k > 0:
            gen_kwargs["top_k"] = top_k

    with torch.no_grad():
        gen_output = hf_model.generate(**encoded, **gen_kwargs, return_dict_in_generate=True, output_scores=True)

    return _rows_from_generate_output(hf_model, tokenizer, prompt_len=prompt_len, gen_output=gen_output)


def sample_completions_batch(
    model,
    prompts: list[str],
    group_size: int,
    gen_config: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    cfg = dict(gen_config or {})
    prompt_list = [str(x) for x in prompts]
    if not prompt_list:
        return []

    sampler = cfg.get("sampler")
    if callable(sampler):
        return [sample_completions(model, prompt=p, group_size=group_size, gen_config=cfg) for p in prompt_list]

    hf_model, tokenizer = _resolve_model_and_tokenizer(model, cfg)
    do_sample = bool(cfg.get("do_sample", True))
    max_new_tokens = int(cfg.get("max_new_tokens", 128))
    temperature = float(cfg.get("temperature", 0.8))
    top_p = float(cfg.get("top_p", 0.95))
    top_k = int(cfg.get("top_k", 0))

    encoded = tokenizer(prompt_list, return_tensors="pt", padding=True).to(hf_model.device)
    input_width = int(encoded["input_ids"].shape[1])
    repeated_prompt_lengths: list[int] = []
    for _ in prompt_list:
        repeated_prompt_lengths.extend([int(input_width)] * max(1, int(group_size)))
    logits_processor = _build_logits_processors(
        tokenizer=tokenizer,
        prompt_lengths=repeated_prompt_lengths,
        gen_config=cfg,
    )

    gen_kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": int(group_size),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "logits_processor": logits_processor,
        # Avoid expensive auto-compile warmup in constrained decoding loops.
        "disable_compile": True,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        if top_k > 0:
            gen_kwargs["top_k"] = top_k

    with torch.no_grad():
        gen_output = hf_model.generate(**encoded, **gen_kwargs, return_dict_in_generate=True, output_scores=True)

    sequences = gen_output.sequences
    transition_scores = None
    if getattr(gen_output, "scores", None):
        try:
            transition_scores = hf_model.compute_transition_scores(
                sequences,
                gen_output.scores,
                normalize_logits=True,
            )
        except Exception:  # noqa: BLE001
            transition_scores = None

    rows_by_prompt: list[list[dict[str, Any]]] = [[] for _ in prompt_list]
    group = max(1, int(group_size))
    for i in range(sequences.shape[0]):
        prompt_idx = min(int(i // group), len(prompt_list) - 1)
        full_ids = sequences[i]
        gen_ids = full_ids[input_width:]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if transition_scores is None:
            # In constrained decoding rerank/eval, reward is computed from executed edits.
            # Keep logprob as a neutral placeholder when transition scores are unavailable.
            logprob = 0.0
            token_count = 0
        else:
            row_scores = transition_scores[i]
            valid_mask = _valid_generated_mask(
                gen_ids[: row_scores.shape[0]],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if valid_mask.numel() == 0:
                logprob = 0.0
                token_count = 0
            else:
                picked = row_scores[: valid_mask.shape[0]][valid_mask]
                logprob = float(picked.sum().item()) if picked.numel() > 0 else 0.0
                token_count = int(picked.numel())
        rows_by_prompt[prompt_idx].append(
            {
                "completion": completion,
                "logprob": float(logprob),
                "token_count": int(token_count),
            }
        )
    return rows_by_prompt


def run_rollout(record: dict[str, Any], model, reward_fn, exec_fn, gen_config: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = str(record.get("prompt", ""))
    meta = dict(record.get("meta", {}))
    for key in [
        "invalid_reward",
        "valid_bonus",
        "strict_bonus",
        "task_strict_bonus_map",
        "loose_bonus",
        "edit_penalty_coef",
        "property_scale",
        "strict_margin_scale",
        "almost_strict_bonus",
        "almost_strict_floor",
        "bottleneck_scale",
        "edit_success_magnitude_cap",
        "reward_mode",
        "process_valid_base",
        "process_hit_weight",
        "process_strict_bonus",
        "process_loose_reward",
        "process_strict_reward",
        "process_margin_weight",
        "process_margin_cap",
        "strict_valid_base",
        "strict_loose_base",
        "strict_hit_base",
        "strict_success_weight",
        "strict_margin_reward_weight",
        "lex_c_weight",
        "lex_b_weight",
        "lex_a_weight",
        "lex_complete_base",
        "lex_complete_a_weight",
        "lex_complete_b_weight",
        "lex_gap_weight",
        "lex_strict_eps",
    ]:
        if key in gen_config:
            meta[key] = gen_config[key]
    task_id = int(meta.get("task_id", -1))
    task_strict_bonus_map = (gen_config or {}).get("task_strict_bonus_map", {})
    if isinstance(task_strict_bonus_map, dict):
        bonus_val = task_strict_bonus_map.get(str(task_id), task_strict_bonus_map.get(int(task_id)))
        if bonus_val is not None:
            try:
                meta["strict_bonus"] = float(bonus_val)
            except Exception:  # noqa: BLE001
                pass
    group_size = int((gen_config or {}).get("group_size", 2))
    rollout_depth = max(1, int((gen_config or {}).get("rollout_search_depth", 3)))
    rollout_width = max(1, int((gen_config or {}).get("rollout_search_width", 3)))
    cfg = dict(gen_config or {})
    cfg["constrained_decoding"] = bool(cfg.get("constrained_decoding", True))
    use_recoverability_shaping = bool(cfg.get("use_recoverability_shaping", True))
    recoverability_beta = float(cfg.get("recoverability_beta", 1.0))
    original_start_tagged = str(meta.get("start_smiles_tagged", ""))
    use_task_fg_constraints = bool(cfg.get("use_task_fg_constraints", False))
    raw_task_fg_ids = cfg.get("task_fg_constraint_task_ids", [])
    task_fg_ids: set[int] = set()
    if isinstance(raw_task_fg_ids, str):
        task_fg_ids = {
            int(x.strip())
            for x in str(raw_task_fg_ids).split(",")
            if str(x).strip()
        }
    elif isinstance(raw_task_fg_ids, (list, tuple, set)):
        for x in raw_task_fg_ids:
            try:
                task_fg_ids.add(int(x))
            except Exception:  # noqa: BLE001
                continue
    if task_id in task_fg_ids:
        use_task_fg_constraints = True
    preferred_fg_ids: list[int] = []
    if use_task_fg_constraints:
        preferred_fg_ids = [int(x) for x in suggest_fg_ids_for_task(meta.get("task_id"))]

    first_op_whitelist = None
    max_edits_override = None
    if bool(cfg.get("use_task_op_constraints", False)):
        first_op_map = cfg.get("task_first_op_whitelist", {})
        if isinstance(first_op_map, dict):
            first_op_whitelist = first_op_map.get(str(task_id), first_op_map.get(int(task_id)))
        max_edits_map = cfg.get("task_max_edits_map", {})
        if isinstance(max_edits_map, dict):
            max_edits_override = max_edits_map.get(str(task_id), max_edits_map.get(int(task_id)))

    def _cfg_for_state(start_tagged_now: str) -> tuple[str, dict[str, Any]]:
        action_constraint = get_action_constraints(str(start_tagged_now))
        if preferred_fg_ids:
            action_constraint = dict(action_constraint)
            action_constraint["preferred_fg_ids"] = list(preferred_fg_ids)
        if first_op_whitelist is not None:
            action_constraint = dict(action_constraint)
            action_constraint["first_op_whitelist"] = first_op_whitelist
        if max_edits_override is not None:
            action_constraint = dict(action_constraint)
            action_constraint["max_edits"] = max_edits_override
        step_cfg = dict(cfg)
        step_cfg["action_constraints"] = [action_constraint]
        base_prompt = prompt
        if original_start_tagged and str(start_tagged_now) and original_start_tagged in base_prompt:
            base_prompt = base_prompt.replace(original_start_tagged, str(start_tagged_now), 1)
        step_meta = dict(meta)
        step_meta["start_smiles_tagged"] = str(start_tagged_now)
        step_prompt = _augment_prompt_with_constraints(base_prompt, step_meta, step_cfg)
        return step_prompt, step_cfg

    rewards: list[float] = []
    out: list[dict[str, Any]] = []
    for _ in range(max(1, group_size)):
        cur_start = str(meta.get("start_smiles_tagged", ""))
        path_completions: list[str] = []
        recoverability_path: list[float] = []
        prev_recoverability = 0.0
        recoverability_shaping = 0.0
        acc_logprob = 0.0
        acc_tokens = 0
        last_execute_result = {}
        last_reward_result = {"reward": float(cfg.get("invalid_reward", -3.0)), "metrics": {}}
        for _dep in range(max(1, rollout_depth)):
            step_prompt, step_cfg = _cfg_for_state(cur_start)
            sampled = sample_completions(model, prompt=step_prompt, group_size=max(1, rollout_width), gen_config=step_cfg)
            if not sampled:
                break
            step_candidates: list[dict[str, Any]] = []
            for item in sampled:
                completion = str(item.get("completion", ""))
                execute_result = exec_fn(str(cur_start), completion)
                reward_result = reward_fn(meta, execute_result)
                step_candidates.append(
                    {
                        "completion": completion,
                        "logprob": float(item.get("logprob", 0.0)),
                        "token_count": int(item.get("token_count", 0)),
                        "execute_result": execute_result,
                        "reward_result": reward_result,
                    }
                )
            if not step_candidates:
                break
            step_recoverability = max(
                (
                    1.0
                    if bool(dict(x.get("reward_result", {}).get("metrics", {})).get("strict_hit", False))
                    else 0.0
                )
                for x in step_candidates
            )
            recoverability_path.append(float(step_recoverability))
            if use_recoverability_shaping:
                recoverability_shaping += float(recoverability_beta * (step_recoverability - prev_recoverability))
            prev_recoverability = float(step_recoverability)
            best = max(step_candidates, key=_branch_rank_key)
            path_completions.append(str(best.get("completion", "")))
            acc_logprob += float(best.get("logprob", 0.0))
            acc_tokens += int(best.get("token_count", 0))
            last_execute_result = dict(best.get("execute_result", {}))
            last_reward_result = dict(best.get("reward_result", {}))
            next_tagged = str(last_execute_result.get("edited_smiles_tagged", "") or "")
            if not next_tagged.strip():
                break
            cur_start = next_tagged
        reward = float(last_reward_result.get("reward", float(cfg.get("invalid_reward", -3.0))))
        if use_recoverability_shaping:
            reward += float(recoverability_shaping)
        rewards.append(reward)
        out.append(
            {
                "completion": " || ".join(path_completions),
                "logprob": float(acc_logprob),
                "token_count": int(acc_tokens),
                "execute_result": dict(last_execute_result),
                "reward_result": dict(last_reward_result),
                "advantage": 0.0,
                "task_id": int(meta.get("task_id", -1)),
                "recoverability_score": float(
                    sum(recoverability_path) / max(1, len(recoverability_path))
                ),
                "recoverability_shaping": float(recoverability_shaping),
            }
        )
    if out:
        best_idx = max(range(len(out)), key=lambda i: _branch_rank_key(out[i]))
        for i in range(len(out)):
            out[i]["is_top1_target"] = 1 if i == best_idx else 0
    mean_reward = float(sum(rewards) / max(len(rewards), 1))
    for idx, cand in enumerate(out):
        cand["advantage"] = float(rewards[idx] - mean_reward)
    return out


def _load_rl_records(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if not str(obj.get("prompt", "")).strip():
                continue
            meta = dict(obj.get("meta", {}))
            if not str(meta.get("start_smiles_tagged", "")).strip():
                continue
            rows.append({"prompt": str(obj["prompt"]), "meta": meta})
    if not rows:
        raise ValueError(f"No usable RL rows in {path}")
    return rows


def _strict_quality_from_metrics(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    valid = 1.0 if bool(metrics.get("is_valid_mol", False)) else 0.0
    strict = 1.0 if bool(metrics.get("strict_hit", False)) else 0.0
    loose = 1.0 if bool(metrics.get("loose_hit", False)) else 0.0
    min_gain = float(metrics.get("min_normalized_gain", 0.0))
    bounded_gain = max(0.0, min(1.5, min_gain))
    # Priority: valid molecule > strict hit > loose hit > near-threshold progress.
    quality = 3.0 * valid + 2.0 * strict + 0.6 * loose + bounded_gain
    return float(quality), float(strict), float(loose), float(min_gain)


def _branch_rank_key(cand: dict[str, Any]) -> tuple[float, float, float, float]:
    valid = 1.0 if bool(cand.get("execute_result", {}).get("is_valid_mol", False)) else 0.0
    metrics = dict(cand.get("reward_result", {}).get("metrics", {}))
    strict = 1.0 if bool(metrics.get("strict_hit", False)) else 0.0
    loose = 1.0 if bool(metrics.get("loose_hit", False)) else 0.0
    min_gain = float(metrics.get("min_normalized_gain", 0.0))
    reward = float(cand.get("reward_result", {}).get("reward", -1e9))
    return float(valid), float(strict), float(loose), float(min_gain), float(reward)


def train_grpo(
    model_name: str,
    sft_ckpt_path: str,
    rl_jsonl: str,
    output_dir: str,
    config: dict[str, Any],
) -> str:
    cfg = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "steps": 100,
        "batch_size": 1,
        "ppo_epochs": 2,
        "learning_rate": 1e-5,
        "clip_eps": 0.2,
        "entropy_coef": 0.003,
        "grpo_group_size": 2,
        "grpo_adv_clip": 4.0,
        "task_priority_sampling": True,
        "task_priority_alpha": 1.5,
        "task_priority_floor": 0.2,
        "task_sampling_boost": {},
        "task_hardness_init": {},
        "task_hardness_momentum": 0.9,
        "task_hardness_min": 0.05,
        "task_hardness_max": 2.0,
        "max_new_tokens": 96,
        "rollout_search_depth": 3,
        "rollout_search_width": 3,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 0,
        "constraint_guidance": True,
        "constrained_decoding": True,
        "use_task_fg_constraints": False,
        "task_fg_constraint_task_ids": [],
        "use_task_op_constraints": False,
        "task_first_op_whitelist": {},
        "task_max_edits_map": {},
        "strict_margin_scale": 1.25,
        "task_strict_bonus_map": {},
        "almost_strict_bonus": 0.5,
        "almost_strict_floor": 0.7,
        "bottleneck_scale": 1.0,
        "edit_success_magnitude_cap": 2.0,
        "reward_mode": "bottleneck_lex",
        "lex_c_weight": 0.55,
        "lex_b_weight": 0.35,
        "lex_a_weight": 0.10,
        "lex_complete_base": 1.0,
        "lex_complete_a_weight": 0.10,
        "lex_complete_b_weight": 0.15,
        "lex_gap_weight": 0.30,
        "lex_strict_eps": 1e-6,
        "deficit_worst_weight": 0.55,
        "deficit_mean_weight": 0.30,
        "deficit_top2_weight": 0.15,
        "deficit_focus_weight": 0.25,
        "adv_normalize": False,
        "use_bottleneck_adv_gate": False,
        "adv_gate_tau_default": 0.60,
        "adv_gate_tau_map": {},
        "adv_gate_temp": 0.10,
        "use_strict_priority_adv": True,
        "strict_priority_adv_blend": 0.90,
        "strict_positive_adv_floor": 0.50,
        "non_hit_positive_damp": 0.20,
        "use_token_avg_ratio": True,
        "skip_zero_token_candidates": True,
        "use_strict_weighted_ppo": True,
        "use_frontier_weighted_ppo": False,
        "frontier_weight_temp": 1.0,
        "frontier_invalid_weight": 0.10,
        "frontier_include_invalid": True,
        "frontier_weight_min": 0.05,
        "frontier_weight_max": 3.0,
        "use_tail_cvar_ppo": False,
        "tail_cvar_q": 0.30,
        "tail_cvar_alpha": 0.50,
        "tail_cvar_gamma": 2.0,
        "tail_cvar_eps": 1e-3,
        "tail_cvar_w_max": 3.0,
        "tail_cvar_closure_lambda": 0.30,
        "strict_gate_when_available": True,
        "strict_gate_non_strict_weight": 0.05,
        "strict_gate_strict_boost": 1.50,
        "strict_pairwise_coef": 0.80,
        "strict_pairwise_margin": 0.05,
        "strict_pairwise_all_negatives": True,
        "sibling_rank_coef": 0.0,
        "sibling_rank_margin": 0.05,
        "sibling_rank_all_pairs": False,
        "sibling_rank_valid_only": True,
        "use_recoverability_shaping": True,
        "recoverability_beta": 1.0,
        "use_recoverability_weight": True,
        "recoverability_weight_kappa": 1.5,
        "recoverability_weight_cap": 3.0,
        "top1_imitation_coef": 1.00,
        "strict_target_for_ppo": 0.80,
        "strict_scarcity_boost_scale": 4.0,
        "elite_replay_coef": 0.15,
        "elite_replay_per_step": 2,
        "elite_pool_per_task": 8,
        "near_replay_coef": 0.05,
        "near_replay_per_step": 2,
        "near_pool_per_task": 8,
        "near_pool_min_gain": 0.65,
        "near_pool_min_gain_map": {},
        "near_pool_multi_only": False,
        "near_pool_min_cover_ratio": 0.0,
        "near_pool_min_strict_count": 0,
        "near_pool_strict_eps": 1e-6,
        "save_every": 20,
        "grad_clip": 1.0,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "codebook_size": 256,
        "max_atom_map": 256,
        "max_fg_id": 64,
        "register_domain_vocab": True,
        "gradient_checkpointing": False,
    }
    cfg.update(config or {})

    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    torch.cuda.manual_seed_all(int(cfg["seed"]))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    tokenizer_load_path = str(sft_ckpt_path).strip() or str(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=True)
    if bool(cfg.get("register_domain_vocab", True)):
        register_domain_tokens(
            tokenizer,
            codebook_size=int(cfg["codebook_size"]),
            num_codebooks=int(cfg["num_codebooks"]),
            mol_token_format=str(cfg["mol_token_format"]),
            max_atom_map=int(cfg["max_atom_map"]),
            max_fg_id=int(cfg["max_fg_id"]),
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token_id is not None else "[PAD]"

    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    if int(base.get_input_embeddings().num_embeddings) != int(len(tokenizer)):
        base.resize_token_embeddings(len(tokenizer))
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tokenizer.pad_token_id

    if str(sft_ckpt_path).strip():
        _patch_peft_torchao_import()
        model = PeftModel.from_pretrained(base, str(sft_ckpt_path), is_trainable=True)
        init_mode = "resume_sft_adapter"
    else:
        target_modules = _infer_target_modules(base)
        peft_config = LoraConfig(
            r=int(cfg["lora_r"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(base, peft_config)
        init_mode = "fresh_lora"

    device = torch.device(str(cfg["device"]))
    model.to(device)
    model.train()
    model.config.use_cache = False
    if bool(cfg.get("gradient_checkpointing", False)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]))

    records = _load_rl_records(str(rl_jsonl))
    gen_cfg = {
        "tokenizer": tokenizer,
        "group_size": int(cfg["grpo_group_size"]),
        "do_sample": True,
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "rollout_search_depth": int(cfg.get("rollout_search_depth", 3)),
        "rollout_search_width": int(cfg.get("rollout_search_width", 3)),
        "temperature": float(cfg["temperature"]),
        "top_p": float(cfg["top_p"]),
        "top_k": int(cfg["top_k"]),
        "constraint_guidance": bool(cfg.get("constraint_guidance", True)),
        "constrained_decoding": bool(cfg.get("constrained_decoding", True)),
        "use_task_fg_constraints": bool(cfg.get("use_task_fg_constraints", False)),
        "task_fg_constraint_task_ids": list(cfg.get("task_fg_constraint_task_ids", [])),
        "use_task_op_constraints": bool(cfg.get("use_task_op_constraints", False)),
        "task_first_op_whitelist": dict(cfg.get("task_first_op_whitelist", {})),
        "task_max_edits_map": dict(cfg.get("task_max_edits_map", {})),
        "invalid_reward": float(cfg.get("invalid_reward", -3.0)),
        "valid_bonus": float(cfg.get("valid_bonus", 1.0)),
        "strict_bonus": float(cfg.get("strict_bonus", 5.0)),
        "task_strict_bonus_map": dict(cfg.get("task_strict_bonus_map", {})),
        "loose_bonus": float(cfg.get("loose_bonus", 2.0)),
        "edit_penalty_coef": float(cfg.get("edit_penalty_coef", 0.1)),
        "property_scale": float(cfg.get("property_scale", 1.0)),
        "strict_margin_scale": float(cfg.get("strict_margin_scale", 1.25)),
        "almost_strict_bonus": float(cfg.get("almost_strict_bonus", 0.5)),
        "almost_strict_floor": float(cfg.get("almost_strict_floor", 0.7)),
        "bottleneck_scale": float(cfg.get("bottleneck_scale", 1.0)),
        "edit_success_magnitude_cap": float(cfg.get("edit_success_magnitude_cap", 2.0)),
        "reward_mode": str(cfg.get("reward_mode", "bottleneck_lex")),
        "lex_c_weight": float(cfg.get("lex_c_weight", 0.55)),
        "lex_b_weight": float(cfg.get("lex_b_weight", 0.35)),
        "lex_a_weight": float(cfg.get("lex_a_weight", 0.10)),
        "lex_complete_base": float(cfg.get("lex_complete_base", 1.0)),
        "lex_complete_a_weight": float(cfg.get("lex_complete_a_weight", 0.10)),
        "lex_complete_b_weight": float(cfg.get("lex_complete_b_weight", 0.15)),
        "lex_gap_weight": float(cfg.get("lex_gap_weight", 0.30)),
        "lex_strict_eps": float(cfg.get("lex_strict_eps", 1e-6)),
        "deficit_worst_weight": float(cfg.get("deficit_worst_weight", 0.55)),
        "deficit_mean_weight": float(cfg.get("deficit_mean_weight", 0.30)),
        "deficit_top2_weight": float(cfg.get("deficit_top2_weight", 0.15)),
        "deficit_focus_weight": float(cfg.get("deficit_focus_weight", 0.25)),
        "use_recoverability_shaping": bool(cfg.get("use_recoverability_shaping", True)),
        "recoverability_beta": float(cfg.get("recoverability_beta", 1.0)),
    }

    raw_hardness_init = cfg.get("task_hardness_init", {})
    hardness_init_map: dict[int, float] = {}
    if isinstance(raw_hardness_init, dict):
        for k, v in raw_hardness_init.items():
            try:
                hardness_init_map[int(k)] = float(v)
            except Exception:  # noqa: BLE001
                continue
    raw_sampling_boost = cfg.get("task_sampling_boost", {})
    sampling_boost_map: dict[int, float] = {}
    if isinstance(raw_sampling_boost, dict):
        for k, v in raw_sampling_boost.items():
            try:
                sampling_boost_map[int(k)] = float(v)
            except Exception:  # noqa: BLE001
                continue
    raw_adv_gate_tau_map = cfg.get("adv_gate_tau_map", {})
    adv_gate_tau_map: dict[int, float] = {}
    if isinstance(raw_adv_gate_tau_map, dict):
        for k, v in raw_adv_gate_tau_map.items():
            try:
                adv_gate_tau_map[int(k)] = float(v)
            except Exception:  # noqa: BLE001
                continue
    raw_near_pool_min_gain_map = cfg.get("near_pool_min_gain_map", {})
    near_pool_min_gain_map: dict[int, float] = {}
    if isinstance(raw_near_pool_min_gain_map, dict):
        for k, v in raw_near_pool_min_gain_map.items():
            try:
                near_pool_min_gain_map[int(k)] = float(v)
            except Exception:  # noqa: BLE001
                continue

    task_hardness: dict[int, float] = {}
    for rec in records:
        tid = int(rec.get("meta", {}).get("task_id", -1))
        init_val = float(hardness_init_map.get(int(tid), 1.0))
        task_hardness[int(tid)] = float(task_hardness.get(int(tid), init_val))
    elite_pool: dict[int, dict[str, dict[str, Any]]] = {}
    near_pool: dict[int, dict[str, dict[str, Any]]] = {}

    for step in range(1, int(cfg["steps"]) + 1):
        batch_size = min(int(cfg["batch_size"]), len(records))
        if bool(cfg.get("task_priority_sampling", True)):
            alpha = float(cfg.get("task_priority_alpha", 1.5))
            floor = float(cfg.get("task_priority_floor", 0.2))
            rec_weights: list[float] = []
            for rec in records:
                tid = int(rec.get("meta", {}).get("task_id", -1))
                hardness = float(task_hardness.get(int(tid), 1.0))
                boost = float(sampling_boost_map.get(int(tid), 1.0))
                rec_weights.append(float(max(1e-6, (floor + hardness**alpha) * max(1e-6, boost))))
            picked = random.choices(records, weights=rec_weights, k=int(batch_size))
            batch = list(picked)
        else:
            batch = random.sample(records, k=int(batch_size))

        all_rollouts: list[dict[str, Any]] = []
        prompt_refs: list[str] = []
        for rollout_id, rec in enumerate(batch):
            rollout = run_rollout(rec, (model, tokenizer), compute_reward, execute_edit_seq, gen_cfg)
            rewards = torch.tensor([float(x["reward_result"]["reward"]) for x in rollout], dtype=torch.float32)
            if rewards.numel() > 1:
                adv = rewards - rewards.mean()
                if bool(cfg.get("adv_normalize", False)):
                    adv = adv / rewards.std(unbiased=False).clamp_min(1e-6)
                if bool(cfg.get("use_bottleneck_adv_gate", False)):
                    tau_default = float(cfg.get("adv_gate_tau_default", 0.60))
                    temp = max(1e-6, float(cfg.get("adv_gate_temp", 0.10)))
                    min_gain_vals: list[float] = []
                    tau_vals: list[float] = []
                    strict_vals: list[float] = []
                    for cand in rollout:
                        metrics = dict(cand.get("reward_result", {}).get("metrics", {}))
                        min_gain_vals.append(float(metrics.get("min_normalized_gain", 0.0)))
                        strict_vals.append(1.0 if bool(metrics.get("strict_hit", False)) else 0.0)
                        tid = int(cand.get("task_id", -1))
                        tau_vals.append(float(adv_gate_tau_map.get(int(tid), tau_default)))
                    min_gain_t = torch.tensor(min_gain_vals, dtype=torch.float32)
                    tau_t = torch.tensor(tau_vals, dtype=torch.float32)
                    strict_t = torch.tensor(strict_vals, dtype=torch.float32)
                    q = torch.where(
                        strict_t > 0.5,
                        torch.ones_like(min_gain_t),
                        torch.sigmoid((min_gain_t - tau_t) / temp),
                    )
                    adv = torch.where(adv > 0.0, adv * q, adv)
                if bool(cfg.get("use_strict_priority_adv", True)):
                    qualities: list[float] = []
                    strict_vals: list[float] = []
                    loose_vals: list[float] = []
                    for cand in rollout:
                        metrics = dict(cand.get("reward_result", {}).get("metrics", {}))
                        q, strict_f, loose_f, _ = _strict_quality_from_metrics(metrics)
                        qualities.append(float(q))
                        strict_vals.append(float(strict_f))
                        loose_vals.append(float(loose_f))
                    q_t = torch.tensor(qualities, dtype=torch.float32)
                    q_center = q_t - q_t.mean()
                    q_std = q_t.std(unbiased=False)
                    if float(q_std.item()) > 1e-6:
                        q_rank = q_center / q_std.clamp_min(1e-6)
                    else:
                        q_rank = torch.zeros_like(q_center)
                    blend = float(cfg.get("strict_priority_adv_blend", 0.65))
                    blend = max(0.0, min(1.0, blend))
                    adv = (1.0 - blend) * adv + blend * q_rank
                    strict_t = torch.tensor(strict_vals, dtype=torch.float32)
                    loose_t = torch.tensor(loose_vals, dtype=torch.float32)
                    strict_floor = float(cfg.get("strict_positive_adv_floor", 0.15))
                    if strict_floor > 0.0:
                        adv = torch.where(
                            strict_t > 0.5,
                            torch.maximum(adv, torch.full_like(adv, strict_floor)),
                            adv,
                        )
                    non_hit_damp = float(cfg.get("non_hit_positive_damp", 0.50))
                    non_hit_damp = max(0.0, min(1.0, non_hit_damp))
                    adv = torch.where(
                        (strict_t < 0.5) & (loose_t < 0.5) & (adv > 0.0),
                        adv * non_hit_damp,
                        adv,
                    )
                adv = torch.clamp(adv, -float(cfg["grpo_adv_clip"]), float(cfg["grpo_adv_clip"]))
            else:
                adv = torch.zeros_like(rewards)
            for idx, cand in enumerate(rollout):
                cand["advantage"] = float(adv[idx].item())
                cand["rollout_id"] = int(rollout_id)
                cand["prompt_ref"] = str(rec["prompt"])
                all_rollouts.append(cand)
                prompt_refs.append(str(rec["prompt"]))
                if bool(cand.get("reward_result", {}).get("metrics", {}).get("strict_hit", False)):
                    tid = int(cand.get("task_id", -1))
                    completion = str(cand.get("completion", ""))
                    reward_val = float(cand.get("reward_result", {}).get("reward", 0.0))
                    token_count = int(cand.get("token_count", 0))
                    task_pool = elite_pool.setdefault(int(tid), {})
                    key = f"{rec['prompt']}\n<ELITE>\n{completion}"
                    old = task_pool.get(key)
                    if old is None or reward_val > float(old.get("reward", -1e9)):
                        task_pool[key] = {
                            "prompt": str(rec["prompt"]),
                            "completion": completion,
                            "reward": reward_val,
                            "token_count": token_count,
                        }
                    keep_n = max(1, int(cfg.get("elite_pool_per_task", 8)))
                    if len(task_pool) > keep_n:
                        top = sorted(task_pool.items(), key=lambda kv: float(kv[1].get("reward", -1e9)), reverse=True)[
                            :keep_n
                        ]
                        elite_pool[int(tid)] = {k: v for k, v in top}
                else:
                    metrics = dict(cand.get("reward_result", {}).get("metrics", {}))
                    loose_hit = bool(metrics.get("loose_hit", False))
                    min_gain = float(metrics.get("min_normalized_gain", 0.0))
                    tid = int(cand.get("task_id", -1))
                    near_floor_default = float(cfg.get("near_pool_min_gain", 0.65))
                    near_floor = float(near_pool_min_gain_map.get(int(tid), near_floor_default))
                    near_strict_eps = float(cfg.get("near_pool_strict_eps", 1e-6))
                    gains = [float(x) for x in metrics.get("normalized_gains", [])]
                    strict_cnt = int(sum(1 for g in gains if g > (1.0 + near_strict_eps)))
                    cover_ratio = float(strict_cnt / max(len(gains), 1))
                    is_multi_obj = bool(len(gains) > 1)
                    min_cover_ratio = float(cfg.get("near_pool_min_cover_ratio", 0.0))
                    min_strict_cnt = int(cfg.get("near_pool_min_strict_count", 0))
                    multi_only = bool(cfg.get("near_pool_multi_only", False))
                    meets_near = bool(
                        loose_hit
                        and min_gain >= near_floor
                        and (cover_ratio >= min_cover_ratio)
                        and (strict_cnt >= min_strict_cnt)
                        and ((not multi_only) or is_multi_obj)
                    )
                    if meets_near:
                        completion = str(cand.get("completion", ""))
                        reward_val = float(cand.get("reward_result", {}).get("reward", 0.0))
                        token_count = int(cand.get("token_count", 0))
                        task_pool = near_pool.setdefault(int(tid), {})
                        key = f"{rec['prompt']}\n<NEAR>\n{completion}"
                        old = task_pool.get(key)
                        new_obj = {
                            "prompt": str(rec["prompt"]),
                            "completion": completion,
                            "reward": reward_val,
                            "token_count": token_count,
                            "near_gain": float(min_gain),
                            "near_cover_ratio": float(cover_ratio),
                            "near_strict_count": int(strict_cnt),
                            "near_multi_obj": bool(is_multi_obj),
                        }
                        if old is None:
                            task_pool[key] = new_obj
                        else:
                            old_gain = float(old.get("near_gain", 0.0))
                            old_cover = float(old.get("near_cover_ratio", 0.0))
                            if float(min_gain) > old_gain or (
                                abs(float(min_gain) - old_gain) <= 1e-8
                                and cover_ratio > old_cover
                            ) or (
                                abs(float(min_gain) - old_gain) <= 1e-8
                                and abs(float(cover_ratio) - old_cover) <= 1e-8
                                and reward_val > float(old.get("reward", -1e9))
                            ):
                                task_pool[key] = new_obj
                        keep_n = max(1, int(cfg.get("near_pool_per_task", 8)))
                        if len(task_pool) > keep_n:
                            top = sorted(
                                task_pool.items(),
                                key=lambda kv: (
                                    float(kv[1].get("near_gain", -1e9)),
                                    float(kv[1].get("near_cover_ratio", -1e9)),
                                    float(kv[1].get("reward", -1e9)),
                                ),
                                reverse=True,
                            )[:keep_n]
                            near_pool[int(tid)] = {k: v for k, v in top}

        step_task_total: dict[int, int] = {}
        step_task_strict: dict[int, int] = {}
        for cand in all_rollouts:
            tid = int(cand.get("task_id", -1))
            step_task_total[int(tid)] = int(step_task_total.get(int(tid), 0)) + 1
            if bool(cand.get("reward_result", {}).get("metrics", {}).get("strict_hit", False)):
                step_task_strict[int(tid)] = int(step_task_strict.get(int(tid), 0)) + 1
        hardness_momentum = float(cfg.get("task_hardness_momentum", 0.9))
        hardness_momentum = max(0.0, min(0.999, hardness_momentum))
        hardness_min = float(cfg.get("task_hardness_min", 0.05))
        hardness_max = float(cfg.get("task_hardness_max", 2.0))
        if hardness_max < hardness_min:
            hardness_max = hardness_min
        for tid, total in step_task_total.items():
            strict_rate = float(step_task_strict.get(int(tid), 0)) / max(int(total), 1)
            old = float(task_hardness.get(int(tid), 1.0))
            new = hardness_momentum * old + (1.0 - hardness_momentum) * (1.0 - strict_rate)
            task_hardness[int(tid)] = float(max(hardness_min, min(hardness_max, new)))

        batch_strict_rate = float(
            sum(1.0 for x in all_rollouts if bool(x.get("reward_result", {}).get("metrics", {}).get("strict_hit", False)))
            / max(len(all_rollouts), 1)
        )
        strict_target = float(cfg.get("strict_target_for_ppo", 0.80))
        scarcity = max(0.0, strict_target - batch_strict_rate)
        scarcity_scale = max(0.0, float(cfg.get("strict_scarcity_boost_scale", 4.0)))
        strict_scarcity_boost = float(1.0 + scarcity_scale * scarcity)

        use_tail_cvar_ppo = bool(cfg.get("use_tail_cvar_ppo", False))
        tail_task_weight: dict[int, float] = {}
        tail_threshold_by_task: dict[int, float] = {}
        if use_tail_cvar_ppo and step_task_total:
            tail_gamma = max(0.0, float(cfg.get("tail_cvar_gamma", 2.0)))
            tail_eps = max(1e-8, float(cfg.get("tail_cvar_eps", 1e-3)))
            tail_w_max = max(1.0, float(cfg.get("tail_cvar_w_max", 3.0)))
            raw_by_task: dict[int, float] = {}
            for tid, total in step_task_total.items():
                strict_rate = float(step_task_strict.get(int(tid), 0)) / max(int(total), 1)
                raw_by_task[int(tid)] = float((max(0.0, 1.0 - strict_rate) + tail_eps) ** tail_gamma)
            mean_raw = float(sum(raw_by_task.values()) / max(len(raw_by_task), 1))
            for tid, raw in raw_by_task.items():
                normed = float(raw / max(mean_raw, 1e-8))
                tail_task_weight[int(tid)] = float(max(1.0, min(tail_w_max, normed)))

            tail_lambda = max(0.0, float(cfg.get("tail_cvar_closure_lambda", 0.30)))
            tail_q = float(cfg.get("tail_cvar_q", 0.30))
            tail_q = max(0.0, min(1.0, tail_q))
            utils_by_task: dict[int, list[float]] = {}
            for cand in all_rollouts:
                tid = int(cand.get("task_id", -1))
                metrics = dict(cand.get("reward_result", {}).get("metrics", {}))
                strict_f = 1.0 if bool(metrics.get("strict_hit", False)) else 0.0
                min_gain = float(metrics.get("min_normalized_gain", 0.0))
                closure = max(-1.0, min(0.0, min_gain - 1.0))
                utility = float(strict_f + tail_lambda * closure)
                utils_by_task.setdefault(int(tid), []).append(utility)
            for tid, vals in utils_by_task.items():
                vals_sorted = sorted(float(x) for x in vals)
                if not vals_sorted:
                    continue
                idx = int(round((len(vals_sorted) - 1) * tail_q))
                idx = max(0, min(len(vals_sorted) - 1, idx))
                tail_threshold_by_task[int(tid)] = float(vals_sorted[idx])

        ppo_epochs = max(1, int(cfg.get("ppo_epochs", 1)))
        epoch_losses: list[float] = []
        for _ in range(ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            group_scores: dict[int, list[tuple[torch.Tensor, int]]] = {}
            group_frontier_scores: dict[int, list[tuple[torch.Tensor, float, bool]]] = {}
            group_has_strict: dict[int, bool] = {}
            frontier_weight_by_idx: dict[int, float] = {}
            if bool(cfg.get("strict_gate_when_available", True)):
                for cand in all_rollouts:
                    gid = int(cand.get("rollout_id", -1))
                    metrics = dict(cand.get("reward_result", {}).get("metrics", {}))
                    is_strict = bool(metrics.get("strict_hit", False))
                    group_has_strict[gid] = bool(group_has_strict.get(gid, False) or is_strict)
            if bool(cfg.get("use_frontier_weighted_ppo", False)):
                fw_temp = max(1e-6, float(cfg.get("frontier_weight_temp", 1.0)))
                fw_invalid = max(0.0, float(cfg.get("frontier_invalid_weight", 0.10)))
                fw_include_invalid = bool(cfg.get("frontier_include_invalid", True))
                by_group_idx: dict[int, list[tuple[int, dict[str, Any]]]] = {}
                for cand_idx, cand in enumerate(all_rollouts):
                    gid = int(cand.get("rollout_id", -1))
                    by_group_idx.setdefault(gid, []).append((int(cand_idx), cand))
                for members in by_group_idx.values():
                    scored: list[tuple[int, bool, float, float, float]] = []
                    for cand_idx, cand in members:
                        metrics = dict(cand.get("reward_result", {}).get("metrics", {}))
                        scored.append(
                            (
                                int(cand_idx),
                                bool(cand.get("execute_result", {}).get("is_valid_mol", False)),
                                float(cand.get("reward_result", {}).get("reward", -1e9)),
                                1.0 if bool(metrics.get("strict_hit", False)) else 0.0,
                                1.0 if bool(metrics.get("loose_hit", False)) else 0.0,
                            )
                        )
                    ranked_valid = [x for x in scored if bool(x[1])]
                    ranked_valid.sort(key=lambda x: (float(x[3]), float(x[4]), float(x[2])), reverse=True)
                    raw_weights: dict[int, float] = {}
                    for rank, item in enumerate(ranked_valid):
                        cand_idx = int(item[0])
                        raw_weights[cand_idx] = float(math.exp(-float(rank) / fw_temp))
                    for cand_idx, is_valid, _reward, _strict, _loose in scored:
                        if bool(is_valid):
                            continue
                        raw_weights[int(cand_idx)] = float(fw_invalid if fw_include_invalid else 0.0)
                    denom = float(sum(max(0.0, w) for w in raw_weights.values()))
                    if denom <= 0.0:
                        for cand_idx, _is_valid, _reward, _strict, _loose in scored:
                            frontier_weight_by_idx[int(cand_idx)] = 1.0
                    else:
                        scale = float(len(scored)) / denom
                        for cand_idx, _is_valid, _reward, _strict, _loose in scored:
                            frontier_weight_by_idx[int(cand_idx)] = float(
                                max(0.0, raw_weights.get(int(cand_idx), 0.0)) * scale
                            )
            for cand_idx, (prompt, cand) in enumerate(zip(prompt_refs, all_rollouts)):
                old_logprob = float(cand["logprob"])
                old_token_count = int(cand.get("token_count", 0))
                if bool(cfg.get("skip_zero_token_candidates", True)) and old_token_count <= 0:
                    continue
                new_logprob, new_token_count = _completion_logprob_and_count_tensor(model, tokenizer, prompt, cand["completion"])
                use_token_avg_ratio = bool(cfg.get("use_token_avg_ratio", True))
                if use_token_avg_ratio:
                    old_denom = max(1, old_token_count)
                    new_denom = max(1, int(new_token_count))
                    old_ref = torch.tensor(old_logprob / float(old_denom), device=model.device, dtype=torch.float32)
                    new_ref = new_logprob / float(new_denom)
                else:
                    old_ref = torch.tensor(old_logprob, device=model.device, dtype=torch.float32)
                    new_ref = new_logprob
                ratio = torch.exp(new_ref - old_ref)
                adv = torch.tensor(float(cand["advantage"]), device=model.device, dtype=torch.float32)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - float(cfg["clip_eps"]), 1.0 + float(cfg["clip_eps"])) * adv
                policy_loss = -torch.min(unclipped, clipped)
                entropy_proxy = -new_ref
                metrics = dict(cand.get("reward_result", {}).get("metrics", {}))
                _, strict_f, loose_f, min_gain = _strict_quality_from_metrics(metrics)
                weight = 1.0
                if bool(cfg.get("use_strict_weighted_ppo", True)):
                    gain_term = max(0.0, min(1.0, float(min_gain)))
                    weight = float(1.0 + 1.20 * strict_f + 0.35 * loose_f + 0.45 * gain_term)
                if bool(cfg.get("strict_gate_when_available", True)):
                    gid = int(cand.get("rollout_id", -1))
                    if bool(group_has_strict.get(gid, False)):
                        if strict_f > 0.5:
                            weight *= float(max(0.0, cfg.get("strict_gate_strict_boost", 1.50)))
                        else:
                            weight *= float(max(0.0, cfg.get("strict_gate_non_strict_weight", 0.05)))
                if bool(cfg.get("use_recoverability_weight", True)):
                    rec_score = float(cand.get("recoverability_score", 0.0))
                    rec_kappa = float(cfg.get("recoverability_weight_kappa", 1.5))
                    rec_cap = float(cfg.get("recoverability_weight_cap", 3.0))
                    rec_mul = math.exp(max(0.0, min(1.0, rec_score)) * max(0.0, rec_kappa))
                    weight *= float(min(max(0.05, rec_mul), max(0.05, rec_cap)))
                if bool(cfg.get("use_frontier_weighted_ppo", False)):
                    fw = float(frontier_weight_by_idx.get(int(cand_idx), 1.0))
                    fw_min = max(0.0, float(cfg.get("frontier_weight_min", 0.05)))
                    fw_max = max(fw_min, float(cfg.get("frontier_weight_max", 3.0)))
                    weight *= float(min(max(fw, fw_min), fw_max))
                if use_tail_cvar_ppo:
                    tid = int(cand.get("task_id", -1))
                    metrics = dict(cand.get("reward_result", {}).get("metrics", {}))
                    strict_f = 1.0 if bool(metrics.get("strict_hit", False)) else 0.0
                    min_gain = float(metrics.get("min_normalized_gain", 0.0))
                    closure = max(-1.0, min(0.0, min_gain - 1.0))
                    tail_lambda = max(0.0, float(cfg.get("tail_cvar_closure_lambda", 0.30)))
                    utility = float(strict_f + tail_lambda * closure)
                    thr = float(tail_threshold_by_task.get(int(tid), utility))
                    is_tail = 1.0 if utility <= thr + 1e-8 else 0.0
                    task_w = float(tail_task_weight.get(int(tid), 1.0))
                    tail_alpha = float(cfg.get("tail_cvar_alpha", 0.50))
                    tail_alpha = max(0.0, min(1.0, tail_alpha))
                    tail_mul = float(tail_alpha + (1.0 - tail_alpha) * is_tail * task_w)
                    weight *= float(max(0.05, tail_mul))
                losses.append(policy_loss * float(weight) - float(cfg["entropy_coef"]) * entropy_proxy)
                gid = int(cand.get("rollout_id", -1))
                target_flag = 1 if int(cand.get("is_top1_target", 0)) == 1 else 0
                group_scores.setdefault(gid, []).append((new_ref, target_flag))
                group_frontier_scores.setdefault(gid, []).append(
                    (
                        new_ref,
                        float(cand.get("reward_result", {}).get("reward", -1e9)),
                        bool(cand.get("execute_result", {}).get("is_valid_mol", False)),
                    )
                )
            if losses:
                policy_objective = torch.stack(losses).mean()
            else:
                policy_objective = torch.zeros((), device=model.device, dtype=torch.float32)

            top1_losses: list[torch.Tensor] = []
            for entries in group_scores.values():
                if not entries:
                    continue
                target_idx = -1
                scores: list[torch.Tensor] = []
                for i, (score_t, is_target) in enumerate(entries):
                    scores.append(score_t)
                    if int(is_target) == 1:
                        target_idx = int(i)
                if target_idx < 0:
                    continue
                score_vec = torch.stack(scores)
                logp_vec = torch.log_softmax(score_vec, dim=0)
                top1_losses.append(-logp_vec[target_idx])
            top1_imitation_loss = (
                torch.stack(top1_losses).mean()
                if top1_losses
                else torch.zeros((), device=model.device, dtype=torch.float32)
            )

            strict_pairwise_coef = float(cfg.get("strict_pairwise_coef", 0.20))
            strict_pairwise_margin = float(cfg.get("strict_pairwise_margin", 0.05))
            strict_pairwise_all_negatives = bool(cfg.get("strict_pairwise_all_negatives", True))
            pairwise_losses: list[torch.Tensor] = []
            if strict_pairwise_coef > 0.0:
                by_group: dict[int, list[dict[str, Any]]] = {}
                for cand in all_rollouts:
                    gid = int(cand.get("rollout_id", -1))
                    by_group.setdefault(gid, []).append(cand)
                for cands in by_group.values():
                    positives: list[dict[str, Any]] = []
                    negatives: list[dict[str, Any]] = []
                    for cand in cands:
                        metrics = dict(cand.get("reward_result", {}).get("metrics", {}))
                        if bool(metrics.get("strict_hit", False)):
                            positives.append(cand)
                        else:
                            negatives.append(cand)
                    if not positives or not negatives:
                        continue
                    margin = torch.tensor(strict_pairwise_margin, device=model.device, dtype=torch.float32)
                    if strict_pairwise_all_negatives:
                        pos_refs: list[torch.Tensor] = []
                        neg_refs: list[torch.Tensor] = []
                        for pos in positives:
                            pos_prompt = str(pos.get("prompt_ref", ""))
                            pos_lp, pos_tok = _completion_logprob_and_count_tensor(
                                model,
                                tokenizer,
                                pos_prompt,
                                str(pos.get("completion", "")),
                            )
                            pos_refs.append(pos_lp / float(max(1, int(pos_tok))))
                        for neg in negatives:
                            neg_prompt = str(neg.get("prompt_ref", ""))
                            neg_lp, neg_tok = _completion_logprob_and_count_tensor(
                                model,
                                tokenizer,
                                neg_prompt,
                                str(neg.get("completion", "")),
                            )
                            neg_refs.append(neg_lp / float(max(1, int(neg_tok))))
                        for pos_ref in pos_refs:
                            for neg_ref in neg_refs:
                                pairwise_losses.append(torch.relu(margin - (pos_ref - neg_ref)))
                    else:
                        pos = max(
                            positives,
                            key=lambda x: float(x.get("reward_result", {}).get("reward", -1e9)),
                        )
                        neg = max(
                            negatives,
                            key=lambda x: (
                                float(
                                    x.get("reward_result", {}).get("metrics", {}).get("min_normalized_gain", 0.0)
                                ),
                                float(x.get("reward_result", {}).get("reward", -1e9)),
                            ),
                        )
                        pos_prompt = str(pos.get("prompt_ref", ""))
                        neg_prompt = str(neg.get("prompt_ref", ""))
                        pos_lp, pos_tok = _completion_logprob_and_count_tensor(
                            model,
                            tokenizer,
                            pos_prompt,
                            str(pos.get("completion", "")),
                        )
                        neg_lp, neg_tok = _completion_logprob_and_count_tensor(
                            model,
                            tokenizer,
                            neg_prompt,
                            str(neg.get("completion", "")),
                        )
                        pos_ref = pos_lp / float(max(1, int(pos_tok)))
                        neg_ref = neg_lp / float(max(1, int(neg_tok)))
                        pairwise_losses.append(torch.relu(margin - (pos_ref - neg_ref)))
            pairwise_loss = (
                torch.stack(pairwise_losses).mean()
                if pairwise_losses
                else torch.zeros((), device=model.device, dtype=torch.float32)
            )

            sibling_rank_coef = float(cfg.get("sibling_rank_coef", 0.0))
            sibling_rank_margin = float(cfg.get("sibling_rank_margin", 0.05))
            sibling_rank_all_pairs = bool(cfg.get("sibling_rank_all_pairs", False))
            sibling_rank_valid_only = bool(cfg.get("sibling_rank_valid_only", True))
            sibling_rank_losses: list[torch.Tensor] = []
            if sibling_rank_coef > 0.0:
                for entries in group_frontier_scores.values():
                    ranked: list[tuple[torch.Tensor, float]] = []
                    for score_t, reward_v, is_valid in entries:
                        if sibling_rank_valid_only and not bool(is_valid):
                            continue
                        ranked.append((score_t, float(reward_v)))
                    if len(ranked) < 2:
                        continue
                    ranked.sort(key=lambda x: float(x[1]), reverse=True)
                    if sibling_rank_all_pairs:
                        pair_indices = [
                            (int(i), int(j))
                            for i in range(len(ranked))
                            for j in range(i + 1, len(ranked))
                        ]
                    else:
                        pair_indices = [(int(i), int(i + 1)) for i in range(len(ranked) - 1)]
                    for hi_i, lo_i in pair_indices:
                        hi_score, hi_reward = ranked[hi_i]
                        lo_score, lo_reward = ranked[lo_i]
                        reward_gap = max(0.0, float(hi_reward) - float(lo_reward))
                        if reward_gap <= 1e-8:
                            continue
                        margin_val = float(sibling_rank_margin) * float(min(1.0, reward_gap))
                        margin = torch.tensor(margin_val, device=model.device, dtype=torch.float32)
                        sibling_rank_losses.append(torch.relu(margin - (hi_score - lo_score)))
            sibling_rank_loss = (
                torch.stack(sibling_rank_losses).mean()
                if sibling_rank_losses
                else torch.zeros((), device=model.device, dtype=torch.float32)
            )

            base_elite_coef = float(cfg.get("elite_replay_coef", 0.0))
            elite_coef = float(base_elite_coef * strict_scarcity_boost)
            imitation_losses: list[torch.Tensor] = []
            if elite_coef > 0.0 and elite_pool:
                batch_task_ids = {int(rec.get("meta", {}).get("task_id", -1)) for rec in batch}
                elite_candidates: list[dict[str, Any]] = []
                for tid in batch_task_ids:
                    elite_candidates.extend(list(elite_pool.get(int(tid), {}).values()))
                if not elite_candidates:
                    for pool in elite_pool.values():
                        elite_candidates.extend(list(pool.values()))
                base_pick_n = max(0, int(cfg.get("elite_replay_per_step", 2)))
                pick_n = min(max(0, int(round(base_pick_n * strict_scarcity_boost))), len(elite_candidates))
                if pick_n > 0:
                    if len(elite_candidates) > pick_n:
                        picked_elites = random.sample(elite_candidates, k=pick_n)
                    else:
                        picked_elites = elite_candidates
                    for elite in picked_elites:
                        lp, lp_tok = _completion_logprob_and_count_tensor(
                            model,
                            tokenizer,
                            str(elite.get("prompt", "")),
                            str(elite.get("completion", "")),
                        )
                        tok = max(1, int(elite.get("token_count", 0)), int(lp_tok))
                        imitation_losses.append((-lp) / float(tok))
            imitation_loss = (
                torch.stack(imitation_losses).mean()
                if imitation_losses
                else torch.zeros((), device=model.device, dtype=torch.float32)
            )
            base_near_coef = float(cfg.get("near_replay_coef", 0.0))
            near_coef = float(base_near_coef * (1.0 + 0.5 * (strict_scarcity_boost - 1.0)))
            near_losses: list[torch.Tensor] = []
            if near_coef > 0.0 and near_pool:
                batch_task_ids = {int(rec.get("meta", {}).get("task_id", -1)) for rec in batch}
                near_candidates: list[dict[str, Any]] = []
                for tid in batch_task_ids:
                    near_candidates.extend(list(near_pool.get(int(tid), {}).values()))
                if not near_candidates:
                    for pool in near_pool.values():
                        near_candidates.extend(list(pool.values()))
                base_near_pick_n = max(0, int(cfg.get("near_replay_per_step", 2)))
                pick_n = min(max(0, int(round(base_near_pick_n * strict_scarcity_boost))), len(near_candidates))
                if pick_n > 0:
                    sorted_near = sorted(
                        near_candidates,
                        key=lambda x: (
                            float(x.get("near_gain", 0.0)),
                            float(x.get("near_cover_ratio", 0.0)),
                            float(x.get("reward", -1e9)),
                        ),
                        reverse=True,
                    )
                    picked_near = sorted_near[:pick_n]
                    for near in picked_near:
                        lp, lp_tok = _completion_logprob_and_count_tensor(
                            model,
                            tokenizer,
                            str(near.get("prompt", "")),
                            str(near.get("completion", "")),
                        )
                        tok = max(1, int(near.get("token_count", 0)), int(lp_tok))
                        gain = float(near.get("near_gain", 0.0))
                        cover = float(near.get("near_cover_ratio", 0.0))
                        weight = max(0.05, min(1.0, 0.6 * gain + 0.4 * cover))
                        near_losses.append(((-lp) / float(tok)) * float(weight))
            near_loss = (
                torch.stack(near_losses).mean()
                if near_losses
                else torch.zeros((), device=model.device, dtype=torch.float32)
            )
            top1_coef = float(cfg.get("top1_imitation_coef", 1.0))
            loss = (
                policy_objective
                + strict_pairwise_coef * pairwise_loss
                + sibling_rank_coef * sibling_rank_loss
                + top1_coef * strict_scarcity_boost * top1_imitation_loss
                + elite_coef * imitation_loss
                + near_coef * near_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(cfg["grad_clip"]))
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        loss_value = float(sum(epoch_losses) / max(len(epoch_losses), 1))

        reward_mean = float(sum(x["reward_result"]["reward"] for x in all_rollouts) / max(len(all_rollouts), 1))
        valid_ratio = float(
            sum(1.0 for x in all_rollouts if bool(x["execute_result"].get("is_valid_mol", False)))
            / max(len(all_rollouts), 1)
        )
        strict_ratio = float(
            sum(1.0 for x in all_rollouts if bool(x["reward_result"].get("metrics", {}).get("strict_hit", False)))
            / max(len(all_rollouts), 1)
        )
        loose_ratio = float(
            sum(1.0 for x in all_rollouts if bool(x["reward_result"].get("metrics", {}).get("loose_hit", False)))
            / max(len(all_rollouts), 1)
        )
        format_ratio = float(
            sum(1.0 for x in all_rollouts if bool(x["execute_result"].get("is_valid_syntax", False)))
            / max(len(all_rollouts), 1)
        )
        recoverability_mean = float(
            sum(float(x.get("recoverability_score", 0.0)) for x in all_rollouts) / max(len(all_rollouts), 1)
        )
        recoverability_shaping_mean = float(
            sum(float(x.get("recoverability_shaping", 0.0)) for x in all_rollouts) / max(len(all_rollouts), 1)
        )
        log_row = {
            "step": int(step),
            "loss": float(loss_value),
            "avg_reward": reward_mean,
            "valid_mol_ratio": valid_ratio,
            "strict_hit_ratio": strict_ratio,
            "loose_hit_ratio": loose_ratio,
            "format_valid_ratio": format_ratio,
            "recoverability_mean": recoverability_mean,
            "recoverability_shaping_mean": recoverability_shaping_mean,
            "elite_pool_size": int(sum(len(v) for v in elite_pool.values())),
            "near_pool_size": int(sum(len(v) for v in near_pool.values())),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
        print(json.dumps(log_row, ensure_ascii=False))

        if int(cfg["save_every"]) > 0 and step % int(cfg["save_every"]) == 0:
            ckpt = out_dir / f"checkpoint-{step}"
            model.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "grpo_meta.json").write_text(
        json.dumps(
            {
                "model_name": str(model_name),
                "sft_ckpt_path": str(sft_ckpt_path),
                "rl_jsonl": str(rl_jsonl),
                "init_mode": str(init_mode),
                **cfg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(out_dir)
