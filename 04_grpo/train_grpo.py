from __future__ import annotations

import argparse
import json
from pathlib import Path

from onerec_mol.grpo import train_grpo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--sft_ckpt", type=str, required=True)
    ap.add_argument("--rl_jsonl", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="outputs/grpo")
    ap.add_argument("--config_json", type=str, default="")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8")) if str(args.config_json).strip() else {}
    out = train_grpo(
        model_name=args.model_name,
        sft_ckpt_path=args.sft_ckpt,
        rl_jsonl=args.rl_jsonl,
        output_dir=args.output_dir,
        config=cfg,
    )
    print(out)


if __name__ == "__main__":
    main()

