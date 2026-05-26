from __future__ import annotations

import argparse
import json
from pathlib import Path

from onerec_mol.dataset import build_rl_record, build_sft_record, dump_jsonl, strip_atom_mapping
from onerec_mol.tokenizer import encode_mol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--tokenizer_ckpt", type=str, required=True)
    ap.add_argument("--sft_out", type=str, default="outputs/data/sft.jsonl")
    ap.add_argument("--rl_out", type=str, default="outputs/data/rl.jsonl")
    ap.add_argument("--include_mol_tokens", type=int, default=1)
    ap.add_argument("--include_atom_map_tokens", type=int, default=1)
    ap.add_argument("--mol_token_format", type=str, default="shared", choices=["shared", "positional"])
    args = ap.parse_args()

    include_mol_tokens = bool(int(args.include_mol_tokens))
    include_atom_map_tokens = bool(int(args.include_atom_map_tokens))
    sft_records = []
    rl_records = []
    with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            if include_mol_tokens:
                plain = strip_atom_mapping(str(sample.get("start_smiles_tagged", "")))
                rq_tokens = encode_mol(plain, args.tokenizer_ckpt)
            else:
                rq_tokens = []
            sft_records.append(
                build_sft_record(
                    sample,
                    rq_tokens,
                    include_mol_tokens=include_mol_tokens,
                    include_atom_map_tokens=include_atom_map_tokens,
                    mol_token_format=str(args.mol_token_format),
                )
            )
            rl_records.append(
                build_rl_record(
                    sample,
                    rq_tokens,
                    include_mol_tokens=include_mol_tokens,
                    include_atom_map_tokens=include_atom_map_tokens,
                    mol_token_format=str(args.mol_token_format),
                )
            )

    dump_jsonl(sft_records, args.sft_out)
    dump_jsonl(rl_records, args.rl_out)
    print(json.dumps({"sft_rows": len(sft_records), "rl_rows": len(rl_records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
