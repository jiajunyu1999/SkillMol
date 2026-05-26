from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .constants import TASK_TEXT


def strip_atom_mapping(smiles_tagged: str) -> str:
    from .chem_utils import mol_from_smiles, smiles_without_atom_maps

    mol = mol_from_smiles(str(smiles_tagged))
    return str(smiles_without_atom_maps(mol))


def format_discrete_tokens(rq_tokens: list[int], *, mol_token_format: str = "shared") -> str:
    if str(mol_token_format).strip().lower() in {"positional", "codebook", "codebook_pos"}:
        return " ".join(f"<MOLTOK_{i}_{int(t)}>" for i, t in enumerate(rq_tokens))
    return " ".join(f"<MOLTOK_{int(t)}>" for t in rq_tokens)


def extract_atom_map_tokens(start_smiles_tagged: str) -> list[str]:
    maps = sorted({int(x) for x in re.findall(r":([0-9]+)\]", str(start_smiles_tagged))})
    return [f"<AMAP_{i}>" for i in maps]


def _build_prompt(
    *,
    task_id: int,
    optimization_target: str,
    start_smiles_tagged: str,
    rq_tokens: list[int],
    include_mol_tokens: bool = True,
    include_atom_map_tokens: bool = True,
    mol_token_format: str = "shared",
) -> str:
    target = str(optimization_target).strip() or TASK_TEXT.get(int(task_id), f"optimize task {task_id}")
    parts = [
        "You are a molecular editing assistant.\n\n"
        f"Task ID: {int(task_id)}\n"
        f"Optimization target: {target}\n\n"
        "Start molecule (atom-mapped SMILES):\n"
        f"{start_smiles_tagged}\n\n"
    ]
    if bool(include_mol_tokens):
        tok_text = format_discrete_tokens(rq_tokens, mol_token_format=str(mol_token_format))
        parts.append("Discrete molecule tokens:\n")
        parts.append(f"{tok_text}\n\n")
    if bool(include_atom_map_tokens):
        amap_tokens = extract_atom_map_tokens(start_smiles_tagged)
        amap_text = " ".join(amap_tokens) if amap_tokens else "(none)"
        parts.append("Source atom-map tokens (use only these for <ANCHOR> and <RMATOM>):\n")
        parts.append(f"{amap_text}\n\n")
    parts.append("Output only the edit sequence in the required format.")
    return "".join(parts)


def build_sft_record(
    sample: dict[str, Any],
    rq_tokens: list[int],
    *,
    include_mol_tokens: bool = True,
    include_atom_map_tokens: bool = True,
    mol_token_format: str = "shared",
) -> dict[str, Any]:
    task_id = int(sample.get("task_id", 0))
    prompt = _build_prompt(
        task_id=task_id,
        optimization_target=str(sample.get("optimization_target", "")),
        start_smiles_tagged=str(sample.get("start_smiles_tagged", "")),
        rq_tokens=list(rq_tokens),
        include_mol_tokens=bool(include_mol_tokens),
        include_atom_map_tokens=bool(include_atom_map_tokens),
        mol_token_format=str(mol_token_format),
    )
    completion = str(sample.get("gold_edit_seq", "")).strip()
    return {
        "prompt": prompt,
        "completion": completion,
        "meta": {
            "sample_id": str(sample.get("sample_id", "")),
            "task_id": task_id,
        },
    }


def build_rl_record(
    sample: dict[str, Any],
    rq_tokens: list[int],
    *,
    include_mol_tokens: bool = True,
    include_atom_map_tokens: bool = True,
    mol_token_format: str = "shared",
) -> dict[str, Any]:
    task_id = int(sample.get("task_id", 0))
    prompt = _build_prompt(
        task_id=task_id,
        optimization_target=str(sample.get("optimization_target", "")),
        start_smiles_tagged=str(sample.get("start_smiles_tagged", "")),
        rq_tokens=list(rq_tokens),
        include_mol_tokens=bool(include_mol_tokens),
        include_atom_map_tokens=bool(include_atom_map_tokens),
        mol_token_format=str(mol_token_format),
    )
    return {
        "prompt": prompt,
        "meta": {
            "sample_id": str(sample.get("sample_id", "")),
            "task_id": task_id,
            "optimization_target": str(sample.get("optimization_target", "")),
            "start_smiles_tagged": str(sample.get("start_smiles_tagged", "")),
        },
    }


def dump_jsonl(records: list[dict[str, Any]], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
