from __future__ import annotations

import argparse
import json
from pathlib import Path

from onerec_mol.tokenizer import train_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles_txt", type=str, required=True, help="One SMILES per line.")
    ap.add_argument("--output_dir", type=str, default="outputs/tokenizer")
    ap.add_argument("--config_json", type=str, default="")
    args = ap.parse_args()

    smiles = [x.strip() for x in Path(args.smiles_txt).read_text(encoding="utf-8").splitlines() if x.strip()]
    cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8")) if str(args.config_json).strip() else {}
    cfg["output_dir"] = str(args.output_dir)
    ckpt = train_tokenizer(smiles, cfg)
    print(ckpt)


if __name__ == "__main__":
    main()

