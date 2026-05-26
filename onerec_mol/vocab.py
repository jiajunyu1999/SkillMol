from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ACTION_TOKENS: list[str] = [
    "<EDIT_SET>",
    "</EDIT_SET>",
    "<EDIT>",
    "</EDIT>",
    "<OP_ADD>",
    "<OP_REMOVE>",
    "<OP_REPLACE>",
    "<ANCHOR>",
    "<FGID>",
    "<FGSMI>",
    "<RMATOM>",
]


def build_mol_tokens(codebook_size: int, *, num_codebooks: int = 8, mol_token_format: str = "shared") -> list[str]:
    if str(mol_token_format).strip().lower() in {"positional", "codebook", "codebook_pos"}:
        return [
            f"<MOLTOK_{cb}_{i}>"
            for cb in range(max(0, int(num_codebooks)))
            for i in range(max(0, int(codebook_size)))
        ]
    return [f"<MOLTOK_{i}>" for i in range(max(0, int(codebook_size)))]


def build_atom_map_tokens(max_atom_map: int) -> list[str]:
    upper = max(0, int(max_atom_map))
    return [f"<AMAP_{i}>" for i in range(upper + 1)]


def build_fg_id_tokens(max_fg_id: int) -> list[str]:
    upper = max(0, int(max_fg_id))
    return [f"<FGID_{i}>" for i in range(upper + 1)]


def build_domain_tokens(
    *,
    codebook_size: int = 256,
    num_codebooks: int = 8,
    mol_token_format: str = "shared",
    max_atom_map: int = 256,
    max_fg_id: int = 64,
) -> list[str]:
    out = []
    out.extend(ACTION_TOKENS)
    out.extend(
        build_mol_tokens(
            codebook_size=codebook_size,
            num_codebooks=num_codebooks,
            mol_token_format=mol_token_format,
        )
    )
    out.extend(build_atom_map_tokens(max_atom_map=max_atom_map))
    out.extend(build_fg_id_tokens(max_fg_id=max_fg_id))
    dedup = []
    seen: set[str] = set()
    for tok in out:
        if tok in seen:
            continue
        seen.add(tok)
        dedup.append(tok)
    return dedup


def register_domain_tokens(
    tokenizer,
    *,
    codebook_size: int = 256,
    num_codebooks: int = 8,
    mol_token_format: str = "shared",
    max_atom_map: int = 256,
    max_fg_id: int = 64,
) -> dict[str, int]:
    tokens = build_domain_tokens(
        codebook_size=int(codebook_size),
        num_codebooks=int(num_codebooks),
        mol_token_format=str(mol_token_format),
        max_atom_map=int(max_atom_map),
        max_fg_id=int(max_fg_id),
    )
    vocab = tokenizer.get_vocab()
    to_add = [t for t in tokens if t not in vocab]
    added = int(tokenizer.add_tokens(to_add, special_tokens=False))
    return {
        "num_requested": int(len(tokens)),
        "num_added": int(added),
        "num_existing": int(len(tokens) - len(to_add)),
    }


@lru_cache(maxsize=1)
def load_fg_id_to_smiles() -> dict[int, str]:
    root = Path(__file__).resolve().parents[1]
    fg_path = root / "data" / "fg_list_small.json"
    if not fg_path.exists():
        return {}
    raw = json.loads(fg_path.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    if not isinstance(raw, dict):
        return out
    for smiles, idx in raw.items():
        if not isinstance(smiles, str):
            continue
        if not isinstance(idx, int):
            continue
        out[int(idx)] = str(smiles)
    return out
