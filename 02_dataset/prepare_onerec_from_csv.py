from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from onerec_mol.constants import TASK_TEXT
from onerec_mol.dataset import build_rl_record, build_sft_record, dump_jsonl, strip_atom_mapping


DEFAULT_TASK_IDS_14 = [101, 102, 103, 104, 105, 106, 107, 108, 201, 202, 203, 204, 205, 206]


def _serialize_edits(edits: list[dict[str, Any]], *, clean_action_vocab: bool = False) -> str:
    chunks = ["<EDIT_SET>"]
    for edit in edits:
        op = str(edit.get("op", "")).strip().lower()
        if op not in {"add", "remove", "replace"}:
            continue
        anchor_raw = int(edit["anchor_atom_map"])
        anchor = f"<AMAP_{anchor_raw}>" if clean_action_vocab else str(anchor_raw)
        chunks.extend(["<EDIT>", f"<OP_{op.upper()}>", "<ANCHOR>", anchor])
        if op in {"add", "replace"} and edit.get("fg_id") is not None:
            fg_raw = int(edit["fg_id"])
            fg_tok = f"<FGID_{fg_raw}>" if clean_action_vocab else str(fg_raw)
            chunks.extend(["<FGID>", fg_tok])
        if (not clean_action_vocab) and op in {"add", "replace"} and edit.get("fg_smiles"):
            chunks.extend(["<FGSMI>", str(edit["fg_smiles"])])
        if op in {"remove", "replace"} and edit.get("removed_atom_map") is not None:
            rm_raw = int(edit["removed_atom_map"])
            rm_tok = f"<AMAP_{rm_raw}>" if clean_action_vocab else str(rm_raw)
            chunks.extend(["<RMATOM>", rm_tok])
        chunks.append("</EDIT>")
    chunks.append("</EDIT_SET>")
    return " ".join(chunks)


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


def _stratified_cap(df: pd.DataFrame, task_ids: list[int], per_task_cap: int, seed: int) -> pd.DataFrame:
    rng = random.Random(int(seed))
    out_parts = []
    for tid in task_ids:
        sub = df[df["task_id"] == int(tid)]
        if len(sub) <= int(per_task_cap) or int(per_task_cap) <= 0:
            out_parts.append(sub)
            continue
        idx = list(sub.index)
        rng.shuffle(idx)
        out_parts.append(sub.loc[idx[: int(per_task_cap)]])
    out = pd.concat(out_parts, axis=0).sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--task_ids", type=str, default="")
    ap.add_argument("--per_task_cap", type=int, default=2000)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer_smiles_cap", type=int, default=30000)
    ap.add_argument("--clean_action_vocab", type=int, default=1, help="1: use clean action tokens (<AMAP_i>, <FGID_i>)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = _parse_task_ids(args.task_ids)

    df = pd.read_csv(args.train_csv)
    required_cols = {"task_id", "start_smiles_tagged", "edits_json"}
    missing = sorted(c for c in required_cols if c not in df.columns)
    if missing:
        raise KeyError(f"Missing required columns in train csv: {missing}")

    df = df[df["task_id"].isin(task_ids)].copy()
    df = df[df["edits_json"].notna()].copy()
    df = df[df["start_smiles_tagged"].notna()].copy()
    df = _stratified_cap(df, task_ids, int(args.per_task_cap), int(args.seed))

    raw_records = []
    for i, row in enumerate(df.to_dict(orient="records")):
        try:
            edits = json.loads(str(row["edits_json"]))
            if not isinstance(edits, list) or not edits:
                continue
            completion = _serialize_edits(edits, clean_action_vocab=bool(int(args.clean_action_vocab)))
            task_id = int(row["task_id"])
            raw_records.append(
                {
                    "sample_id": f"task{task_id}_{i:06d}",
                    "task_id": task_id,
                    "optimization_target": TASK_TEXT.get(task_id, f"optimize task {task_id}"),
                    "start_smiles_tagged": str(row["start_smiles_tagged"]),
                    "gold_edit_seq": completion,
                }
            )
        except Exception:  # noqa: BLE001
            continue

    rng = random.Random(int(args.seed))
    rng.shuffle(raw_records)
    val_n = max(1, int(round(float(args.val_ratio) * len(raw_records)))) if raw_records else 0
    val_raw = raw_records[:val_n]
    train_raw = raw_records[val_n:]

    tokenizer_smiles = []
    for rec in train_raw:
        try:
            tokenizer_smiles.append(strip_atom_mapping(str(rec["start_smiles_tagged"])))
        except Exception:  # noqa: BLE001
            continue
    tokenizer_smiles = sorted(set(tokenizer_smiles))
    if int(args.tokenizer_smiles_cap) > 0 and len(tokenizer_smiles) > int(args.tokenizer_smiles_cap):
        rng.shuffle(tokenizer_smiles)
        tokenizer_smiles = tokenizer_smiles[: int(args.tokenizer_smiles_cap)]

    raw_train_path = out_dir / "raw_train.jsonl"
    raw_val_path = out_dir / "raw_val.jsonl"
    dump_jsonl(train_raw, str(raw_train_path))
    dump_jsonl(val_raw, str(raw_val_path))

    tok_smiles_txt = out_dir / "tokenizer_smiles.txt"
    tok_smiles_txt.write_text("\n".join(tokenizer_smiles), encoding="utf-8")

    # Placeholders to be filled after tokenizer training.
    (out_dir / "next_step_instructions.txt").write_text(
        (
            "1) Train tokenizer with tokenizer_smiles.txt\n"
            "2) Run 02_dataset/build_records.py to build sft_train/sft_val/rl_train from raw jsonl\n"
        ),
        encoding="utf-8",
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "train_csv": str(args.train_csv),
                "task_ids": task_ids,
                "per_task_cap": int(args.per_task_cap),
                "raw_train_rows": len(train_raw),
                "raw_val_rows": len(val_raw),
                "tokenizer_smiles_rows": len(tokenizer_smiles),
                "clean_action_vocab": bool(int(args.clean_action_vocab)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "raw_train_jsonl": str(raw_train_path),
                "raw_val_jsonl": str(raw_val_path),
                "tokenizer_smiles_txt": str(tok_smiles_txt),
                "raw_train_rows": len(train_raw),
                "raw_val_rows": len(val_raw),
                "tokenizer_smiles_rows": len(tokenizer_smiles),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
