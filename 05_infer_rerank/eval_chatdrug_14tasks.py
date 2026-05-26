from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from onerec_mol.chem_utils import ensure_atom_maps, mol_from_smiles, mol_to_tagged_smiles
from onerec_mol.constants import TASK_TEXT
from onerec_mol.dataset import build_rl_record
from onerec_mol.inference import infer_with_rerank
from onerec_mol.tokenizer import encode_mol_batch
from onerec_mol.vocab import register_domain_tokens


DEFAULT_TASK_IDS_14 = [101, 102, 103, 104, 105, 106, 107, 108, 201, 202, 203, 204, 205, 206]


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


def main() -> None:
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--tokenizer_ckpt", type=str, required=True, help="GNN-RQ tokenizer checkpoint path.")
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--task_ids", type=str, default="")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_ids = _parse_task_ids(args.task_ids)
    test_df = pd.read_csv(args.test_csv)
    if int(args.max_rows) > 0:
        test_df = test_df.head(int(args.max_rows)).copy()
    if "mol" not in test_df.columns:
        raise KeyError("test_csv must contain `mol` column.")
    base_rows = test_df.to_dict(orient="records")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    register_domain_tokens(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token_id is not None else "[PAD]"
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    if int(base.get_input_embeddings().num_embeddings) != int(len(tokenizer)):
        base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.to(torch.device(args.device))
    model.eval()

    rows_out: list[dict[str, Any]] = []
    summary = {
        str(t): {"count": 0, "valid": 0, "loose": 0, "strict": 0, "reward_sum": 0.0}
        for t in task_ids
    }
    batch_rq_tokens = encode_mol_batch([str(row["mol"]) for row in base_rows], args.tokenizer_ckpt)

    for idx, row in enumerate(base_rows):
        start_smiles = str(row["mol"])
        try:
            start_tagged = _ensure_tagged(start_smiles)
            plain = start_smiles
            rq_tokens = list(batch_rq_tokens[idx])
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
            sample = {
                "sample_id": f"test_{idx}_task_{tid}",
                "task_id": int(tid),
                "optimization_target": TASK_TEXT.get(int(tid), f"optimize task {tid}"),
                "start_smiles_tagged": start_tagged,
            }
            rl_record = build_rl_record(sample, rq_tokens)
            meta = rl_record["meta"]
            result = infer_with_rerank(
                model={"model": model, "tokenizer": tokenizer},
                prompt=str(rl_record["prompt"]),
                meta=meta,
                n_samples=int(args.n_samples),
                gen_config={
                    "tokenizer": tokenizer,
                    "do_sample": True,
                    "max_new_tokens": 96,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "top_k": 0,
                    "constrained_decoding": True,
                    "constraint_guidance": True,
                    "constraint_sample_multiplier": 2,
                    "constraint_max_rounds": 2,
                },
            )

            best_smiles = str(result.get("best_edited_smiles", "") or "")
            after_props = _safe_properties(best_smiles) if best_smiles else None
            loose_hit, strict_hit = _evaluate_one_task(int(tid), before_props, after_props)
            is_valid_mol = int(after_props is not None)
            best_reward = float(result.get("best_reward", -3.0))

            rows_out.append(
                {
                    "row_index": idx,
                    "task_id": int(tid),
                    "start_smiles": plain,
                    "start_smiles_tagged": start_tagged,
                    "best_completion": str(result.get("best_completion", "")),
                    "best_edited_smiles": best_smiles,
                    "best_reward": best_reward,
                    "is_valid_mol": is_valid_mol,
                    "loose_hit": int(loose_hit),
                    "strict_hit": int(strict_hit),
                }
            )
            stats = summary[str(tid)]
            stats["count"] += 1
            stats["valid"] += int(is_valid_mol)
            stats["loose"] += int(loose_hit)
            stats["strict"] += int(strict_hit)
            stats["reward_sum"] += float(best_reward)
        if (idx + 1) % 10 == 0:
            print(json.dumps({"progress_rows": int(idx + 1), "total_rows": int(len(test_df))}, ensure_ascii=False))

    raw_csv = out_dir / "predictions_14tasks.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else [])
        writer.writeheader()
        writer.writerows(rows_out)

    summary_rows = []
    for tid in task_ids:
        stats = summary[str(tid)]
        denom = max(int(stats["count"]), 1)
        summary_rows.append(
            {
                "task_id": int(tid),
                "count": int(stats["count"]),
                "valid_ratio": float(stats["valid"]) / denom,
                "loose_hit_ratio": float(stats["loose"]) / denom,
                "strict_hit_ratio": float(stats["strict"]) / denom,
                "avg_reward": float(stats["reward_sum"]) / denom,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("task_id")
    summary_csv = out_dir / "summary_14tasks.csv"
    summary_df.to_csv(summary_csv, index=False)

    overall = {
        "count": int(sum(x["count"] for x in summary_rows)),
        "valid_ratio": float(summary_df["valid_ratio"].mean()) if len(summary_df) else 0.0,
        "loose_hit_ratio": float(summary_df["loose_hit_ratio"].mean()) if len(summary_df) else 0.0,
        "strict_hit_ratio": float(summary_df["strict_hit_ratio"].mean()) if len(summary_df) else 0.0,
        "avg_reward": float(summary_df["avg_reward"].mean()) if len(summary_df) else 0.0,
    }
    meta = {
        "base_model": str(args.base_model),
        "adapter_dir": str(args.adapter_dir),
        "tokenizer_ckpt": str(args.tokenizer_ckpt),
        "test_csv": str(args.test_csv),
        "n_samples": int(args.n_samples),
        "task_ids": task_ids,
        "rows": len(test_df),
        "overall": overall,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"raw_csv": str(raw_csv), "summary_csv": str(summary_csv), "overall": overall}, ensure_ascii=False))


if __name__ == "__main__":
    main()
