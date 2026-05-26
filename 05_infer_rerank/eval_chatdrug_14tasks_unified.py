from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onerec_mol.chem_utils import ensure_atom_maps, mol_from_smiles, mol_to_tagged_smiles, smiles_without_atom_maps
from onerec_mol.constants import TASK_TEXT
from onerec_mol.dataset import build_rl_record
from onerec_mol.inference import infer_with_rerank, infer_with_tree_search
from onerec_mol.sft import _SoftMolPrefixModel, _patch_peft_torchao_import
from onerec_mol.tokenizer import encode_mol, encode_mol_batch, encode_mol_quantized
from onerec_mol.vocab import register_domain_tokens


DEFAULT_TASK_IDS_14 = [101, 102, 103, 104, 105, 106, 107, 108, 201, 202, 203, 204, 205, 206]


def _set_seed(seed: int) -> None:
    if int(seed) < 0:
        return
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:  # noqa: BLE001
        pass
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        from transformers import set_seed

        set_seed(int(seed))
    except Exception:  # noqa: BLE001
        pass


def _load_extra_prompt(path: str) -> str:
    if not str(path).strip():
        return ""
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return ""
    return (
        "Additional task instructions:\n"
        f"{raw}\n\n"
        "Use these instructions when choosing molecular edits.\n\n"
    )


def _apply_extra_prompt(record: dict[str, Any], extra_prompt: str) -> dict[str, Any]:
    if not extra_prompt:
        return record
    out = dict(record)
    out["prompt"] = f"{extra_prompt}{record.get('prompt', '')}"
    meta = dict(out.get("meta", {}))
    meta["extra_prompt_applied"] = True
    out["meta"] = meta
    return out


def _parse_task_ids(raw: str) -> list[int]:
    if not str(raw).strip():
        return list(DEFAULT_TASK_IDS_14)
    out = []
    for x in str(raw).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _load_task_search_overrides(raw: str) -> dict[str, dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--task_search_overrides_json must decode to a JSON object.")
    out: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError("Each task search override value must be a JSON object.")
        for task_key in str(key).replace(",", " ").split():
            if task_key.strip():
                out[str(int(task_key))] = dict(value)
    return out


def _coerce_override_value(current: Any, value: Any) -> Any:
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _apply_task_search_override(
    *,
    task_id: int,
    gen_cfg: dict[str, Any],
    overrides: dict[str, dict[str, Any]],
    default_search_width: int,
    default_search_depth: int,
    default_n_samples: int,
) -> tuple[dict[str, Any], int, int, int]:
    override = overrides.get(str(int(task_id)), {})
    if not override:
        return gen_cfg, int(default_search_width), int(default_search_depth), int(default_n_samples)
    next_cfg = dict(gen_cfg)
    local_search_width = int(default_search_width)
    local_search_depth = int(default_search_depth)
    local_n_samples = int(default_n_samples)
    for key, value in override.items():
        key = str(key)
        if key == "search_width":
            local_search_width = int(value)
        elif key == "search_depth":
            local_search_depth = max(1, int(value))
        elif key == "n_samples":
            local_n_samples = int(value)
        else:
            current = next_cfg.get(key)
            next_cfg[key] = _coerce_override_value(current, value) if current is not None else value
    if "search_width" in override and "max_search_candidates" not in override:
        next_cfg["max_search_candidates"] = int(gen_cfg.get("max_search_candidates", 30) or 30)
    return next_cfg, local_search_width, local_search_depth, local_n_samples


def _ensure_tagged(smiles: str) -> str:
    mol = ensure_atom_maps(mol_from_smiles(str(smiles)))
    return str(mol_to_tagged_smiles(mol))


def _safe_properties(smiles: str) -> dict[str, float] | None:
    from onerec_mol.reward import compute_properties

    try:
        return compute_properties(smiles)
    except Exception:  # noqa: BLE001
        return None


def _evaluate_one_task(task_id: int, before_props: dict[str, float] | None, after_props: dict[str, float] | None) -> tuple[int, int]:
    if before_props is None or after_props is None:
        return 0, 0
    from onerec_mol.reward import evaluate_task_hit

    hit = evaluate_task_hit(int(task_id), before_props, after_props)
    return int(bool(hit["loose_hit"])), int(bool(hit["strict_hit"]))


def _update_summary_stats(summary: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
    tid = str(int(row["task_id"]))
    if tid not in summary:
        return
    stats = summary[tid]
    stats["count"] += 1
    stats["parse_valid"] += int(row.get("is_parse_valid", 0) or 0)
    stats["valid"] += int(row.get("is_valid_mol", 0) or 0)
    stats["loose"] += int(row.get("loose_hit", 0) or 0)
    stats["strict"] += int(row.get("strict_hit", 0) or 0)
    stats["reward_sum"] += float(row.get("best_reward", -3.0) or 0.0)


def _summary_rows_from_stats(summary: dict[str, dict[str, Any]], task_ids: list[int]) -> list[dict[str, Any]]:
    summary_rows = []
    for tid in task_ids:
        stats = summary[str(tid)]
        denom = max(int(stats["count"]), 1)
        summary_rows.append(
            {
                "task_id": int(tid),
                "count": int(stats["count"]),
                "parse_valid_ratio": float(stats["parse_valid"]) / denom,
                "valid_ratio": float(stats["valid"]) / denom,
                "loose_hit_ratio": float(stats["loose"]) / denom,
                "strict_hit_ratio": float(stats["strict"]) / denom,
                "avg_reward": float(stats["reward_sum"]) / denom,
            }
        )
    return summary_rows


def _overall_from_summary_rows(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary_df = pd.DataFrame(summary_rows)
    return {
        "count": int(sum(int(x["count"]) for x in summary_rows)),
        "parse_valid_ratio": float(summary_df["parse_valid_ratio"].mean()) if len(summary_df) else 0.0,
        "valid_ratio": float(summary_df["valid_ratio"].mean()) if len(summary_df) else 0.0,
        "loose_hit_ratio": float(summary_df["loose_hit_ratio"].mean()) if len(summary_df) else 0.0,
        "strict_hit_ratio": float(summary_df["strict_hit_ratio"].mean()) if len(summary_df) else 0.0,
        "avg_reward": float(summary_df["avg_reward"].mean()) if len(summary_df) else 0.0,
    }


def _write_progress_outputs(
    *,
    out_dir: Path,
    rows_out: list[dict[str, Any]],
    summary: dict[str, dict[str, Any]],
    task_ids: list[int],
    meta: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    raw_csv = out_dir / "predictions_14tasks.csv"
    if rows_out:
        with raw_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
    summary_rows = _summary_rows_from_stats(summary, task_ids)
    summary_df = pd.DataFrame(summary_rows).sort_values("task_id")
    summary_csv = out_dir / "summary_14tasks.csv"
    summary_df.to_csv(summary_csv, index=False)
    overall = _overall_from_summary_rows(summary_rows)
    meta_out = dict(meta)
    meta_out["overall"] = overall
    meta_out["completed_pairs"] = int(len(rows_out))
    (out_dir / "meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw_csv, summary_csv, overall


def _load_model_and_tokenizer(
    *,
    model_path: str,
    adapter_dir: str,
    device: str,
    register_vocab: bool,
    codebook_size: int,
    num_codebooks: int,
    mol_token_format: str,
    max_atom_map: int,
    max_fg_id: int,
    use_soft_mol_prefix: bool,
    soft_mol_prefix_len: int,
    soft_mol_prefix_input_dim: int,
):
    adapter = str(adapter_dir).strip()
    tok_path = adapter if adapter else str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if bool(register_vocab):
        register_domain_tokens(
            tokenizer,
            codebook_size=int(codebook_size),
            num_codebooks=int(num_codebooks),
            mol_token_format=str(mol_token_format),
            max_atom_map=int(max_atom_map),
            max_fg_id=int(max_fg_id),
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token_id is not None else "[PAD]"
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if adapter:
        from peft import PeftModel

        _patch_peft_torchao_import()
        base = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype)
        if int(base.get_input_embeddings().num_embeddings) != int(len(tokenizer)):
            base.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base, adapter)
        mode = "peft_adapter"
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype)
        if int(model.get_input_embeddings().num_embeddings) != int(len(tokenizer)):
            model.resize_token_embeddings(len(tokenizer))
        mode = "full_model"

    model.to(torch.device(device))
    if bool(use_soft_mol_prefix):
        projector_path = Path(adapter) / "soft_mol_prefix_projector.pt"
        if not projector_path.exists():
            raise FileNotFoundError(f"Missing soft prefix projector: {projector_path}")
        wrapped = _SoftMolPrefixModel(
            base_model=model,
            input_dim=int(soft_mol_prefix_input_dim),
            prefix_len=int(soft_mol_prefix_len),
        )
        payload = torch.load(str(projector_path), map_location="cpu")
        wrapped.projector.load_state_dict(payload["state_dict"])
        wrapped.to(torch.device(device))
        model = wrapped
        mode = f"{mode}+soft_mol_prefix"
    model.eval()
    return model, tokenizer, mode


def _make_soft_prefix_sampler(model, tokenizer, tokenizer_ckpt: str):
    def _sampler(prompt: str, group_size: int, gen_config: dict[str, Any]) -> list[dict[str, Any]]:
        from onerec_mol.chem_utils import mol_from_smiles, smiles_without_atom_maps
        from onerec_mol.grpo import _build_logits_processors, _rows_from_generate_output
        from onerec_mol.sft import _extract_start_smiles_tagged_from_prompt

        start_tagged = _extract_start_smiles_tagged_from_prompt(str(prompt))
        try:
            plain = str(smiles_without_atom_maps(mol_from_smiles(start_tagged)))
            mol_vec = encode_mol_quantized(plain, tokenizer_ckpt)
        except Exception:  # noqa: BLE001
            mol_vec = [0.0 for _ in range(256)]

        encoded = tokenizer(str(prompt), return_tensors="pt").to(model.base_model.device)
        prompt_len = int(encoded["input_ids"].shape[1])
        gen_kwargs = {
            "do_sample": bool(gen_config.get("do_sample", True)),
            "max_new_tokens": int(gen_config.get("max_new_tokens", 96)),
            "num_return_sequences": int(group_size),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "logits_processor": _build_logits_processors(
                tokenizer=tokenizer,
                prompt_lengths=[prompt_len] * max(1, int(group_size)),
                gen_config=gen_config,
            ),
            "disable_compile": True,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = float(gen_config.get("temperature", 0.8))
            gen_kwargs["top_p"] = float(gen_config.get("top_p", 0.95))
            top_k = int(gen_config.get("top_k", 0))
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
        mol_quantized = torch.tensor([mol_vec], dtype=torch.float32, device=model.base_model.device)
        with torch.no_grad():
            gen_output = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                mol_quantized=mol_quantized,
                **gen_kwargs,
            )
        return _rows_from_generate_output(model.base_model, tokenizer, prompt_len=prompt_len, gen_output=gen_output)

    return _sampler


def main() -> None:
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="Base/full model path.")
    ap.add_argument("--adapter_dir", type=str, default="", help="Optional LoRA/PEFT adapter directory.")
    ap.add_argument("--tokenizer_ckpt", type=str, required=True, help="GNN-RQ tokenizer checkpoint path.")
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--task_ids", type=str, default="")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--row_offset", type=int, default=0, help="Skip this many input rows before applying --max_rows.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=-1, help="Set >=0 for reproducible sampling.")
    ap.add_argument(
        "--seed_per_pair",
        action="store_true",
        help="Reseed before each row/task pair so per-task search routing does not perturb later tasks.",
    )
    ap.add_argument("--write_every_rows", type=int, default=1, help="Write partial CSV/summary every N input rows.")
    ap.add_argument("--resume_from_existing", action="store_true", help="Skip row/task pairs already in output predictions.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--search_width", type=int, default=0, help="Per-step branch width for multi-step editing.")
    ap.add_argument("--search_depth", type=int, default=1, help="Number of editing steps to chain.")
    ap.add_argument(
        "--search_ranker",
        type=str,
        default="reward",
        choices=[
            "reward",
            "heuristic",
            "non_rdkit",
            "logprob_prior",
            "task_prior",
            "task_prior_struct",
            "directive_router",
            "router",
            "task_prior_valid",
            "policy_prior",
            "consensus",
            "consensus_prior",
            "proxy_property",
            "property_proxy",
            "proxy_property_atomctx",
            "property_proxy_atomctx",
            "proxy_logprob",
            "logprob_proxy",
            "search_score",
            "path_score",
            "tree_score",
            "search_score_sum",
            "path_sum",
            "portfolio_scorer",
            "portfolio",
            "learned_scorer",
            "learned",
            "logprob",
            "normalized_logprob",
            "hybrid",
            "hybrid_atomctx",
            "hybrid_default",
            "conservative_hybrid",
            "ensemble",
            "hybrid_learned",
            "self_rerank",
            "llm_rerank",
        ],
        help="Candidate selector for multi-step search. Non-reward rankers do not use RDKit reward for selection.",
    )
    ap.add_argument("--search_beam_width", type=int, default=1)
    ap.add_argument("--max_search_candidates", type=int, default=30)
    ap.add_argument(
        "--tree_frontier_ranker",
        type=str,
        default="",
        help="Optional ranker for expanding/sorting tree frontier. Defaults to --search_ranker.",
    )
    ap.add_argument(
        "--tree_final_ranker",
        type=str,
        default="",
        help="Optional final selector over all tree candidates. Defaults to --search_ranker.",
    )
    ap.add_argument(
        "--tree_score_mode",
        type=str,
        default="sum",
        choices=["sum", "mean", "avg", "average", "last", "step"],
        help="How to aggregate per-step no-RDKit scores in tree search.",
    )
    ap.add_argument(
        "--tree_depth_penalty",
        type=float,
        default=0.0,
        help="Penalty applied per additional edit step when selecting/sorting tree nodes.",
    )
    ap.add_argument(
        "--selection_validity_mode",
        type=str,
        default="auto",
        choices=["auto", "valid_first", "parse_first", "none"],
        help="Candidate pool for selection. auto uses parse_first for non-reward rankers and valid_first for reward.",
    )
    ap.add_argument("--use_task_fg_constraints", action="store_true")
    ap.add_argument("--plan_guided_prompt", action="store_true")
    ap.add_argument("--heuristic_logprob_weight", type=float, default=0.45)
    ap.add_argument("--heuristic_diversity_weight", type=float, default=0.20)
    ap.add_argument("--heuristic_prior_weight", type=float, default=0.30)
    ap.add_argument("--heuristic_repetition_weight", type=float, default=0.15)
    ap.add_argument("--task_prior_logprob_weight", type=float, default=0.25)
    ap.add_argument("--task_prior_keyword_weight", type=float, default=0.30)
    ap.add_argument("--task_prior_fg_weight", type=float, default=0.35)
    ap.add_argument("--task_prior_op_weight", type=float, default=0.20)
    ap.add_argument("--task_prior_valid_weight", type=float, default=0.0)
    ap.add_argument("--task_prior_repetition_weight", type=float, default=0.15)
    ap.add_argument("--consensus_weight", type=float, default=0.45)
    ap.add_argument("--consensus_prior_weight", type=float, default=0.40)
    ap.add_argument("--consensus_logprob_weight", type=float, default=0.15)
    ap.add_argument("--proxy_property_weight", type=float, default=0.55)
    ap.add_argument("--proxy_fg_weight", type=float, default=0.25)
    ap.add_argument("--proxy_logprob_weight", type=float, default=0.20)
    ap.add_argument("--proxy_logprob_ranker_logprob_weight", type=float, default=0.60)
    ap.add_argument("--proxy_logprob_ranker_proxy_weight", type=float, default=0.40)
    ap.add_argument("--proxy_logprob_ranker_fg_weight", type=float, default=0.0)
    ap.add_argument("--proxy_logprob_ranker_action_penalty", type=float, default=0.25)
    ap.add_argument("--proxy_logprob_ranker_missing_remove_penalty", type=float, default=0.20)
    ap.add_argument("--proxy_logprob_ranker_duplicate_anchor_penalty", type=float, default=0.20)
    ap.add_argument("--proxy_logprob_ranker_unknown_anchor_penalty", type=float, default=0.15)
    ap.add_argument("--proxy_logprob_ranker_anchor_rm_penalty", type=float, default=0.10)
    ap.add_argument("--hybrid_logprob_weight", type=float, default=0.55)
    ap.add_argument("--hybrid_task_prior_weight", type=float, default=0.20)
    ap.add_argument("--hybrid_proxy_weight", type=float, default=0.10)
    ap.add_argument("--hybrid_consensus_weight", type=float, default=0.10)
    ap.add_argument("--hybrid_action_penalty", type=float, default=0.25)
    ap.add_argument("--hybrid_duplicate_anchor_penalty", type=float, default=0.35)
    ap.add_argument("--hybrid_duplicate_fg_penalty", type=float, default=0.10)
    ap.add_argument("--hybrid_missing_fgsmi_penalty", type=float, default=0.10)
    ap.add_argument("--hybrid_missing_remove_atom_penalty", type=float, default=0.30)
    ap.add_argument("--hybrid_unknown_atom_penalty", type=float, default=0.35)
    ap.add_argument("--hybrid_anchor_rm_overlap_penalty", type=float, default=0.25)
    ap.add_argument("--ensemble_hybrid_weight", type=float, default=0.75)
    ap.add_argument("--ensemble_learned_weight", type=float, default=0.15)
    ap.add_argument("--ensemble_logprob_weight", type=float, default=0.10)
    ap.add_argument("--task_first_op_whitelist_json", type=str, default="")
    ap.add_argument("--task_max_edits_json", type=str, default="")
    ap.add_argument("--candidate_scorer_path", type=str, default="")
    ap.add_argument(
        "--candidate_scorer_paths_json",
        type=str,
        default="",
        help="Optional JSON map from task id or task-id groups to learned scorer paths.",
    )
    ap.add_argument(
        "--task_search_overrides_json",
        type=str,
        default="",
        help="Optional JSON map from task id or task-id groups to per-task generation/search overrides.",
    )
    ap.add_argument("--self_rerank_max_choices", type=int, default=8)
    ap.add_argument(
        "--self_rerank_prefilter_ranker",
        type=str,
        default="",
        help="Optional no-RDKit ranker used to choose candidates shown to self_rerank.",
    )
    ap.add_argument("--dump_candidates_jsonl", type=str, default="")
    ap.add_argument("--dump_max_candidates", type=int, default=30)
    ap.add_argument("--codebook_size", type=int, default=256)
    ap.add_argument("--num_codebooks", type=int, default=8)
    ap.add_argument("--mol_token_format", type=str, default="shared", choices=["shared", "positional"])
    ap.add_argument("--max_atom_map", type=int, default=256)
    ap.add_argument("--max_fg_id", type=int, default=64)
    ap.add_argument("--no_mol_tokens", action="store_true")
    ap.add_argument("--no_atom_map_tokens", action="store_true")
    ap.add_argument("--no_register_domain_vocab", action="store_true")
    ap.add_argument("--extra_prompt_path", type=str, default="")
    ap.add_argument("--use_soft_mol_prefix", action="store_true")
    ap.add_argument("--soft_mol_prefix_len", type=int, default=8)
    ap.add_argument("--soft_mol_prefix_input_dim", type=int, default=256)
    args = ap.parse_args()
    _set_seed(int(args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_ids = _parse_task_ids(args.task_ids)
    test_df = pd.read_csv(args.test_csv)
    if int(args.row_offset) > 0:
        test_df = test_df.iloc[int(args.row_offset) :].copy()
    if int(args.max_rows) > 0:
        test_df = test_df.head(int(args.max_rows)).copy()
    if "mol" not in test_df.columns:
        raise KeyError("test_csv must contain `mol` column.")
    base_rows = test_df.to_dict(orient="records")

    model, tokenizer, mode = _load_model_and_tokenizer(
        model_path=str(args.model_path),
        adapter_dir=str(args.adapter_dir),
        device=str(args.device),
        register_vocab=not bool(args.no_register_domain_vocab),
        codebook_size=int(args.codebook_size),
        num_codebooks=int(args.num_codebooks),
        mol_token_format=str(args.mol_token_format),
        max_atom_map=int(args.max_atom_map),
        max_fg_id=int(args.max_fg_id),
        use_soft_mol_prefix=bool(args.use_soft_mol_prefix),
        soft_mol_prefix_len=int(args.soft_mol_prefix_len),
        soft_mol_prefix_input_dim=int(args.soft_mol_prefix_input_dim),
    )
    soft_sampler = (
        _make_soft_prefix_sampler(model, tokenizer, str(args.tokenizer_ckpt))
        if bool(args.use_soft_mol_prefix)
        else None
    )

    rows_out: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    summary = {
        str(t): {"count": 0, "parse_valid": 0, "valid": 0, "loose": 0, "strict": 0, "reward_sum": 0.0}
        for t in task_ids
    }

    include_mol_tokens = not bool(args.no_mol_tokens)
    include_atom_map_tokens = not bool(args.no_atom_map_tokens)
    extra_prompt = _load_extra_prompt(str(args.extra_prompt_path))
    search_width = int(args.search_width) if int(args.search_width) > 0 else int(args.n_samples)
    search_depth = max(1, int(args.search_depth))
    task_search_overrides = _load_task_search_overrides(str(args.task_search_overrides_json))
    batch_rq_tokens = (
        encode_mol_batch([str(row["mol"]) for row in base_rows], args.tokenizer_ckpt) if include_mol_tokens else []
    )
    base_meta = {
        "mode": mode,
        "model_path": str(args.model_path),
        "adapter_dir": str(args.adapter_dir),
        "tokenizer_ckpt": str(args.tokenizer_ckpt),
        "test_csv": str(args.test_csv),
        "n_samples": int(args.n_samples),
        "row_offset": int(args.row_offset),
        "task_ids": task_ids,
        "rows": len(test_df),
        "include_mol_tokens": bool(include_mol_tokens),
        "include_atom_map_tokens": bool(include_atom_map_tokens),
        "mol_token_format": str(args.mol_token_format),
        "search_width": int(search_width),
        "search_depth": int(search_depth),
        "search_ranker": str(args.search_ranker),
        "tree_frontier_ranker": str(args.tree_frontier_ranker),
        "tree_final_ranker": str(args.tree_final_ranker),
        "search_beam_width": int(args.search_beam_width),
        "max_search_candidates": int(args.max_search_candidates),
        "tree_score_mode": str(args.tree_score_mode),
        "tree_depth_penalty": float(args.tree_depth_penalty),
        "selection_validity_mode": str(args.selection_validity_mode),
        "use_task_fg_constraints": bool(args.use_task_fg_constraints),
        "plan_guided_prompt": bool(args.plan_guided_prompt),
        "use_soft_mol_prefix": bool(args.use_soft_mol_prefix),
        "extra_prompt_path": str(args.extra_prompt_path),
        "extra_prompt_applied": bool(extra_prompt),
        "seed": int(args.seed),
        "seed_per_pair": bool(args.seed_per_pair),
        "candidate_scorer_path": str(args.candidate_scorer_path),
        "candidate_scorer_paths_json": str(args.candidate_scorer_paths_json),
        "task_search_overrides_json": str(args.task_search_overrides_json),
        "self_rerank_prefilter_ranker": str(args.self_rerank_prefilter_ranker),
    }
    completed_pairs: set[tuple[int, int]] = set()
    raw_csv = out_dir / "predictions_14tasks.csv"
    if bool(args.resume_from_existing) and raw_csv.exists():
        prev_df = pd.read_csv(raw_csv)
        for prev_row in prev_df.to_dict(orient="records"):
            pair = (int(prev_row["row_index"]), int(prev_row["task_id"]))
            if pair in completed_pairs:
                continue
            completed_pairs.add(pair)
            row_dict = dict(prev_row)
            rows_out.append(row_dict)
            _update_summary_stats(summary, row_dict)
        if rows_out:
            print(json.dumps({"resumed_pairs": int(len(rows_out)), "from": str(raw_csv)}, ensure_ascii=False))

    row_offset = int(args.row_offset)
    for local_idx, row in enumerate(base_rows):
        idx = int(row_offset + local_idx)
        start_smiles = str(row["mol"])
        try:
            start_tagged = _ensure_tagged(start_smiles)
            plain = start_smiles
            rq_tokens = list(batch_rq_tokens[local_idx]) if include_mol_tokens else []
        except Exception:  # noqa: BLE001
            for tid in task_ids:
                rows_out.append(
                    {
                        "row_index": idx,
                        "task_id": int(tid),
                        "start_smiles": start_smiles,
                        "start_smiles_tagged": "",
                        "best_completion": "",
                        "best_edited_smiles": "",
                        "best_reward": -3.0,
                        "is_parse_valid": 0,
                        "is_valid_mol": 0,
                        "loose_hit": 0,
                        "strict_hit": 0,
                    }
                )
                summary[str(tid)]["count"] += 1
                summary[str(tid)]["reward_sum"] += -3.0
            continue

        before_props = _safe_properties(plain)
        for tid in task_ids:
            if (int(idx), int(tid)) in completed_pairs:
                continue
            if bool(args.seed_per_pair) and int(args.seed) >= 0:
                _set_seed(int(args.seed) + 1009 * int(idx) + int(tid))
            sample = {
                "sample_id": f"test_{idx}_task_{tid}",
                "task_id": int(tid),
                "optimization_target": TASK_TEXT.get(int(tid), f"optimize task {tid}"),
                "start_smiles_tagged": start_tagged,
            }
            rl_record = build_rl_record(
                sample,
                rq_tokens,
                include_mol_tokens=include_mol_tokens,
                include_atom_map_tokens=include_atom_map_tokens,
                mol_token_format=str(args.mol_token_format),
            )
            rl_record = _apply_extra_prompt(rl_record, extra_prompt)
            if bool(args.plan_guided_prompt):
                from onerec_mol.inference import _task_plan_text

                plan_text = _task_plan_text(int(tid), TASK_TEXT.get(int(tid), f"optimize task {tid}"))
                if plan_text.strip():
                    rl_record = dict(rl_record)
                    rl_record["prompt"] = f"{plan_text}\n{rl_record['prompt']}"
            meta = rl_record["meta"]

            gen_cfg = {
                "tokenizer": tokenizer,
                "do_sample": True,
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": 0,
                "constrained_decoding": True,
                "constraint_guidance": True,
                "constraint_sample_multiplier": 2,
                "constraint_max_rounds": 2,
                "search_ranker": str(args.search_ranker),
                "tree_frontier_ranker": str(args.tree_frontier_ranker),
                "tree_final_ranker": str(args.tree_final_ranker),
                "search_beam_width": int(args.search_beam_width),
                "max_search_candidates": int(args.max_search_candidates),
                "tree_score_mode": str(args.tree_score_mode),
                "tree_depth_penalty": float(args.tree_depth_penalty),
                "selection_validity_mode": (
                    ("parse_first" if str(args.search_ranker) != "reward" else "valid_first")
                    if str(args.selection_validity_mode) == "auto"
                    else str(args.selection_validity_mode)
                ),
                "use_task_fg_constraints": bool(args.use_task_fg_constraints),
                "heuristic_logprob_weight": float(args.heuristic_logprob_weight),
                "heuristic_diversity_weight": float(args.heuristic_diversity_weight),
                "heuristic_prior_weight": float(args.heuristic_prior_weight),
                "heuristic_repetition_weight": float(args.heuristic_repetition_weight),
                "task_prior_logprob_weight": float(args.task_prior_logprob_weight),
                "task_prior_keyword_weight": float(args.task_prior_keyword_weight),
                "task_prior_fg_weight": float(args.task_prior_fg_weight),
                "task_prior_op_weight": float(args.task_prior_op_weight),
                "task_prior_valid_weight": float(args.task_prior_valid_weight),
                "task_prior_repetition_weight": float(args.task_prior_repetition_weight),
                "consensus_weight": float(args.consensus_weight),
                "consensus_prior_weight": float(args.consensus_prior_weight),
                "consensus_logprob_weight": float(args.consensus_logprob_weight),
                "proxy_property_weight": float(args.proxy_property_weight),
                "proxy_fg_weight": float(args.proxy_fg_weight),
                "proxy_logprob_weight": float(args.proxy_logprob_weight),
                "proxy_logprob_ranker_logprob_weight": float(args.proxy_logprob_ranker_logprob_weight),
                "proxy_logprob_ranker_proxy_weight": float(args.proxy_logprob_ranker_proxy_weight),
                "proxy_logprob_ranker_fg_weight": float(args.proxy_logprob_ranker_fg_weight),
                "proxy_logprob_ranker_action_penalty": float(args.proxy_logprob_ranker_action_penalty),
                "proxy_logprob_ranker_missing_remove_penalty": float(args.proxy_logprob_ranker_missing_remove_penalty),
                "proxy_logprob_ranker_duplicate_anchor_penalty": float(args.proxy_logprob_ranker_duplicate_anchor_penalty),
                "proxy_logprob_ranker_unknown_anchor_penalty": float(args.proxy_logprob_ranker_unknown_anchor_penalty),
                "proxy_logprob_ranker_anchor_rm_penalty": float(args.proxy_logprob_ranker_anchor_rm_penalty),
                "hybrid_logprob_weight": float(args.hybrid_logprob_weight),
                "hybrid_task_prior_weight": float(args.hybrid_task_prior_weight),
                "hybrid_proxy_weight": float(args.hybrid_proxy_weight),
                "hybrid_consensus_weight": float(args.hybrid_consensus_weight),
                "hybrid_action_penalty": float(args.hybrid_action_penalty),
                "hybrid_duplicate_anchor_penalty": float(args.hybrid_duplicate_anchor_penalty),
                "hybrid_duplicate_fg_penalty": float(args.hybrid_duplicate_fg_penalty),
                "hybrid_missing_fgsmi_penalty": float(args.hybrid_missing_fgsmi_penalty),
                "hybrid_missing_remove_atom_penalty": float(args.hybrid_missing_remove_atom_penalty),
                "hybrid_unknown_atom_penalty": float(args.hybrid_unknown_atom_penalty),
                "hybrid_anchor_rm_overlap_penalty": float(args.hybrid_anchor_rm_overlap_penalty),
                "ensemble_hybrid_weight": float(args.ensemble_hybrid_weight),
                "ensemble_learned_weight": float(args.ensemble_learned_weight),
                "ensemble_logprob_weight": float(args.ensemble_logprob_weight),
                "task_first_op_whitelist": json.loads(args.task_first_op_whitelist_json)
                if str(args.task_first_op_whitelist_json).strip()
                else {},
                "task_max_edits_map": json.loads(args.task_max_edits_json)
                if str(args.task_max_edits_json).strip()
                else {},
                "candidate_scorer_path": str(args.candidate_scorer_path),
                "candidate_scorer_paths_json": str(args.candidate_scorer_paths_json),
                "self_rerank_max_choices": int(args.self_rerank_max_choices),
                "self_rerank_prefilter_ranker": str(args.self_rerank_prefilter_ranker),
            }
            if soft_sampler is not None:
                gen_cfg["sampler"] = soft_sampler
            gen_cfg, local_search_width, local_search_depth, local_n_samples = _apply_task_search_override(
                task_id=int(tid),
                gen_cfg=gen_cfg,
                overrides=task_search_overrides,
                default_search_width=int(search_width),
                default_search_depth=int(search_depth),
                default_n_samples=int(args.n_samples),
            )

            if int(local_search_depth) <= 1:
                result = infer_with_rerank(
                    model={"model": model, "tokenizer": tokenizer},
                    prompt=str(rl_record["prompt"]),
                    meta=meta,
                    n_samples=int(local_n_samples),
                    gen_config=gen_cfg,
                )
            else:
                prompt_cache: dict[str, str] = {}

                def _build_prompt_for_start(start_tagged_now: str) -> str:
                    cached = prompt_cache.get(str(start_tagged_now))
                    if cached is not None:
                        return cached
                    task_sample = {
                        "sample_id": str(meta.get("sample_id", "")),
                        "task_id": int(meta.get("task_id", 0)),
                        "optimization_target": str(meta.get("optimization_target", "")),
                        "start_smiles_tagged": str(start_tagged_now),
                    }
                    step_tokens: list[int] = []
                    if include_mol_tokens:
                        try:
                            plain_now = str(smiles_without_atom_maps(mol_from_smiles(str(start_tagged_now))))
                            step_tokens = encode_mol(plain_now, args.tokenizer_ckpt)
                        except Exception:
                            step_tokens = []
                    rec = build_rl_record(
                        task_sample,
                        step_tokens,
                        include_mol_tokens=include_mol_tokens,
                        include_atom_map_tokens=include_atom_map_tokens,
                        mol_token_format=str(args.mol_token_format),
                    )
                    rec = _apply_extra_prompt(rec, extra_prompt)
                    if bool(args.plan_guided_prompt):
                        from onerec_mol.inference import _task_plan_text

                        plan_text = _task_plan_text(int(meta.get("task_id", 0)), str(meta.get("optimization_target", "")))
                        if plan_text.strip():
                            rec = dict(rec)
                            rec["prompt"] = f"{plan_text}\n{rec['prompt']}"
                    prompt = str(rec["prompt"])
                    prompt_cache[str(start_tagged_now)] = prompt
                    return prompt

                result = infer_with_tree_search(
                    model={"model": model, "tokenizer": tokenizer},
                    meta=meta,
                    build_prompt_fn=_build_prompt_for_start,
                    width=int(local_search_width),
                    depth=int(local_search_depth),
                    gen_config=gen_cfg,
                )

            best_smiles = str(result.get("best_edited_smiles", "") or "")
            best_reward = float(result.get("best_reward", -3.0))
            best_candidates = list(result.get("all_candidates", []))
            best_execute_result = result.get("best_execute_result", {})
            if not isinstance(best_execute_result, dict):
                best_execute_result = {}
            best_parse_valid = int(bool(best_execute_result.get("is_valid_syntax", False)))
            best_valid_mol = int(bool(best_execute_result.get("is_valid_mol", False)))

            after_props = _safe_properties(best_smiles) if best_smiles else None
            loose_hit, strict_hit = _evaluate_one_task(int(tid), before_props, after_props)

            if str(args.dump_candidates_jsonl).strip():
                for cand_rank, cand in enumerate(best_candidates[: max(0, int(args.dump_max_candidates))], start=1):
                    if not isinstance(cand, dict):
                        continue
                    cand_exec = cand.get("execute_result", {})
                    if not isinstance(cand_exec, dict):
                        cand_exec = {}
                    cand_smiles = str(cand_exec.get("edited_smiles", "") or "")
                    cand_props = _safe_properties(cand_smiles) if cand_smiles else None
                    cand_loose, cand_strict = _evaluate_one_task(int(tid), before_props, cand_props)
                    candidate_rows.append(
                        {
                            "row_index": int(idx),
                            "task_id": int(tid),
                            "candidate_rank": int(cand_rank),
                            "depth": int(cand.get("depth", 1) or 1),
                            "search_score": float(cand.get("search_score", 0.0) or 0.0),
                            "heuristic_score": float(cand.get("heuristic_score", 0.0) or 0.0),
                            "logprob": float(cand.get("logprob", 0.0) or 0.0),
                            "token_count": int(cand.get("token_count", 0) or 0),
                            "completion": str(cand.get("completion", "")),
                            "edited_smiles": cand_smiles,
                            "is_parse_valid": int(bool(cand_exec.get("is_valid_syntax", False))),
                            "is_valid_mol": int(bool(cand_exec.get("is_valid_mol", False))),
                            "loose_hit": int(cand_loose),
                            "strict_hit": int(cand_strict),
                        }
                    )

            row_out = {
                "row_index": idx,
                "task_id": int(tid),
                "start_smiles": plain,
                "start_smiles_tagged": start_tagged,
                "best_completion": str(result.get("best_completion", "")),
                "best_depth": int(result.get("best_depth", 1 if int(local_search_depth) <= 1 else 0)),
                "best_path": " || ".join([str(x) for x in result.get("best_path", [])]),
                "num_candidates": int(len(best_candidates)),
                "best_edited_smiles": best_smiles,
                "best_reward": best_reward,
                "is_parse_valid": int(best_parse_valid),
                "is_valid_mol": int(best_valid_mol),
                "loose_hit": int(loose_hit),
                "strict_hit": int(strict_hit),
            }
            rows_out.append(row_out)
            completed_pairs.add((int(idx), int(tid)))
            _update_summary_stats(summary, row_out)
        if (local_idx + 1) % max(1, int(args.write_every_rows)) == 0 or (local_idx + 1) == len(base_rows):
            _raw_csv, _summary_csv, _overall = _write_progress_outputs(
                out_dir=out_dir,
                rows_out=rows_out,
                summary=summary,
                task_ids=task_ids,
                meta=base_meta,
            )
            print(
                json.dumps(
                    {
                        "progress_rows": int(local_idx + 1),
                        "total_rows": int(len(test_df)),
                        "completed_pairs": int(len(rows_out)),
                        "overall": _overall,
                    },
                    ensure_ascii=False,
                )
            )

    if str(args.dump_candidates_jsonl).strip():
        dump_path = Path(args.dump_candidates_jsonl)
        if not dump_path.is_absolute():
            dump_path = out_dir / dump_path
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as f:
            for item in candidate_rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    raw_csv, summary_csv, overall = _write_progress_outputs(
        out_dir=out_dir,
        rows_out=rows_out,
        summary=summary,
        task_ids=task_ids,
        meta=base_meta,
    )
    print(json.dumps({"raw_csv": str(raw_csv), "summary_csv": str(summary_csv), "overall": overall}, ensure_ascii=False))


if __name__ == "__main__":
    main()
