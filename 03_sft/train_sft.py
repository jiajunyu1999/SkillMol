from __future__ import annotations

import argparse
import json
from pathlib import Path

from onerec_mol.sft import train_sft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="outputs/sft")
    ap.add_argument("--config_json", type=str, default="")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8")) if str(args.config_json).strip() else {}
    out = train_sft(
        model_name=args.model_name,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        config=cfg,
    )
    print(out)


if __name__ == "__main__":
    main()

