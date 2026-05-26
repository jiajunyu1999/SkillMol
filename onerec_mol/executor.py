from __future__ import annotations

from functools import lru_cache
import re
from typing import Any

from .apply import apply_edits
from .chem_utils import atom_idx_by_map, ensure_atom_maps, mol_from_smiles, pick_anchor_atom_maps, smiles_without_atom_maps, terminal_neighbors
from .edits import _scaffold_atom_indices
from .vocab import load_fg_id_to_smiles


def _append_unique_candidate(candidates: list[str], seen: set[str], text: str) -> None:
    candidate = str(text).strip()
    if not candidate or candidate in seen:
        return
    candidates.append(candidate)
    seen.add(candidate)


def _collect_repair_candidates(text: str) -> tuple[str, str | None, list[str]]:
    start_tag = "<EDIT_SET>"
    end_tag = "</EDIT_SET>"
    edit_tag = "<EDIT>"
    end_edit_tag = "</EDIT>"

    base = str(text).strip()
    candidates: list[str] = []
    seen: set[str] = set()

    start_idx = base.find(start_tag)
    boundary_clipped: str | None = None
    if start_idx >= 0:
        end_idx = base.find(end_tag, start_idx + len(start_tag))
        if end_idx > start_idx:
            boundary_clipped = base[start_idx : end_idx + len(end_tag)].strip()
            _append_unique_candidate(candidates, seen, boundary_clipped)

    end_idx = base.find(end_tag)
    if end_idx >= 0:
        first_edit_idx = base.find(edit_tag)
        if first_edit_idx >= 0 and first_edit_idx < end_idx:
            repaired = f"{start_tag} {base[first_edit_idx : end_idx + len(end_tag)].strip()}"
            _append_unique_candidate(candidates, seen, repaired)

    if start_idx >= 0:
        last_end_edit_idx = base.rfind(end_edit_tag)
        if last_end_edit_idx > start_idx:
            repaired = base[start_idx : last_end_edit_idx + len(end_edit_tag)].strip()
            repaired = f"{repaired} {end_tag}"
            _append_unique_candidate(candidates, seen, repaired)

    if edit_tag in base and end_edit_tag in base:
        first_edit_idx = base.find(edit_tag)
        last_end_edit_idx = base.rfind(end_edit_tag)
        if first_edit_idx >= 0 and last_end_edit_idx > first_edit_idx:
            block = base[first_edit_idx : last_end_edit_idx + len(end_edit_tag)].strip()
            _append_unique_candidate(candidates, seen, block)
            _append_unique_candidate(candidates, seen, f"{start_tag} {block} {end_tag}")

    _append_unique_candidate(candidates, seen, base)
    return base, boundary_clipped, candidates


def repair_edit_seq(edit_seq: str) -> tuple[str, bool]:
    base, boundary_clipped, candidates = _collect_repair_candidates(edit_seq)
    if not base:
        return "", False

    # Accept boundary clipping unconditionally: removing outer noise is semantics-preserving.
    if boundary_clipped is not None and boundary_clipped != base:
        return boundary_clipped, True

    # For structural repairs such as adding missing wrapper tags, only keep candidates
    # that become parse-valid after repair. This avoids over-aggressive fixes.
    for candidate in candidates:
        if candidate == base:
            continue
        parsed = parse_edit_seq(candidate)
        if bool(parsed["is_valid_syntax"]):
            return candidate, True

    return base, False


def _parse_number_or_tagged_id(block: str, field: str, prefix: str) -> int | None:
    m = re.search(rf"{re.escape(field)}\s*(?:<{re.escape(prefix)}_([0-9]+)>|([0-9]+))", block)
    if m is None:
        return None
    val = m.group(1) or m.group(2)
    if val is None:
        return None
    return int(val)


def parse_edit_seq(edit_seq: str) -> dict[str, Any]:
    text = str(edit_seq).strip()
    if not text:
        return {"is_valid_syntax": False, "actions": [], "error": "parse_error: empty_edit_seq"}
    if "<EDIT_SET>" not in text or "</EDIT_SET>" not in text:
        return {"is_valid_syntax": False, "actions": [], "error": "parse_error: missing_edit_set_tags"}

    blocks = re.findall(r"<EDIT>(.*?)</EDIT>", text, flags=re.DOTALL)
    if not blocks:
        return {"is_valid_syntax": False, "actions": [], "error": "parse_error: no_edit_block"}

    actions: list[dict[str, Any]] = []
    for idx, block in enumerate(blocks, start=1):
        op = None
        if "<OP_ADD>" in block:
            op = "ADD"
        elif "<OP_REMOVE>" in block:
            op = "REMOVE"
        elif "<OP_REPLACE>" in block:
            op = "REPLACE"
        if op is None:
            return {"is_valid_syntax": False, "actions": [], "error": f"parse_error: missing_op_at_edit_{idx}"}

        anchor = _parse_number_or_tagged_id(block, "<ANCHOR>", "AMAP")
        if anchor is None:
            return {"is_valid_syntax": False, "actions": [], "error": f"parse_error: missing_anchor_at_edit_{idx}"}

        action: dict[str, Any] = {
            "op": op,
            "anchor": int(anchor),
        }

        fgid = _parse_number_or_tagged_id(block, "<FGID>", "FGID")
        if fgid is not None:
            action["fg_id"] = int(fgid)

        fgsmiles_match = re.search(r"<FGSMI>\s*(.*?)(?=(<RMATOM>|<FGID>|</EDIT>|$))", block, flags=re.DOTALL)
        if fgsmiles_match is not None:
            fg_smi = fgsmiles_match.group(1).strip()
            if fg_smi:
                action["fg_smi"] = fg_smi

        rmatom = _parse_number_or_tagged_id(block, "<RMATOM>", "AMAP")
        if rmatom is not None:
            action["removed_atom_map"] = int(rmatom)

        if op in {"ADD", "REPLACE"} and not action.get("fg_smi") and action.get("fg_id") is not None:
            fg_vocab = load_fg_id_to_smiles()
            fg_smi = fg_vocab.get(int(action["fg_id"]))
            if fg_smi:
                action["fg_smi"] = str(fg_smi)

        if op in {"ADD", "REPLACE"} and not action.get("fg_smi"):
            return {
                "is_valid_syntax": False,
                "actions": [],
                "error": f"parse_error: missing_fg_smi_at_edit_{idx}",
            }
        actions.append(action)

    return {"is_valid_syntax": True, "actions": actions, "error": None}


def serialize_edit_actions(actions: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for action in actions:
        op = str(action.get("op", "")).strip().upper()
        anchor = int(action.get("anchor"))
        parts = ["<EDIT>", f"<OP_{op}>", "<ANCHOR>", str(anchor)]
        fg_id = action.get("fg_id")
        if fg_id is not None:
            parts.extend(["<FGID>", str(int(fg_id))])
        fg_smi = str(action.get("fg_smi", "") or "").strip()
        if fg_smi:
            parts.extend(["<FGSMI>", fg_smi])
        removed_atom_map = action.get("removed_atom_map")
        if removed_atom_map is not None:
            parts.extend(["<RMATOM>", str(int(removed_atom_map))])
        parts.append("</EDIT>")
        blocks.append(" ".join(parts))
    return f"<EDIT_SET> {' '.join(blocks)} </EDIT_SET>".strip()


@lru_cache(maxsize=4096)
def get_action_constraints(start_smiles_tagged: str) -> dict[str, Any]:
    text = str(start_smiles_tagged).strip()
    if not text:
        return {"valid_anchor_maps": [], "removable_by_anchor": {}}

    mol = ensure_atom_maps(mol_from_smiles(text))
    scaffold_idx = _scaffold_atom_indices(mol)
    valid_anchor_maps = sorted(int(x) for x in pick_anchor_atom_maps(mol))
    removable_by_anchor: dict[int, list[int]] = {}

    for anchor_map in valid_anchor_maps:
        anchor_idx = atom_idx_by_map(mol, int(anchor_map))
        terminal_maps = [int(mol.GetAtomWithIdx(i).GetAtomMapNum()) for i in terminal_neighbors(mol, anchor_idx)]
        branch_maps: list[int] = []
        if anchor_idx in scaffold_idx:
            for nbr in mol.GetAtomWithIdx(anchor_idx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx in scaffold_idx:
                    continue
                branch_maps.append(int(nbr.GetAtomMapNum()))
        candidates = sorted(set(terminal_maps + branch_maps))
        if candidates:
            removable_by_anchor[int(anchor_map)] = candidates

    return {
        "valid_anchor_maps": valid_anchor_maps,
        "removable_by_anchor": removable_by_anchor,
    }


def format_action_constraint_guidance(
    start_smiles_tagged: str,
    *,
    max_anchors: int = 24,
    max_targets_per_anchor: int = 8,
) -> str:
    constraints = get_action_constraints(start_smiles_tagged)
    removable_by_anchor = dict(constraints.get("removable_by_anchor", {}))
    if not removable_by_anchor:
        return (
            "Action constraints:\n"
            "Do not use <OP_REMOVE> or <OP_REPLACE> for this molecule unless the edit is clearly valid.\n"
        )

    lines = [
        "Action constraints:",
        "For <OP_REMOVE> or <OP_REPLACE>, use only these <ANCHOR>/<RMATOM> pairs.",
    ]
    shown = 0
    for anchor in sorted(removable_by_anchor):
        if shown >= int(max_anchors):
            break
        targets = removable_by_anchor[anchor][: int(max_targets_per_anchor)]
        target_text = " ".join(f"<AMAP_{int(x)}>" for x in targets)
        lines.append(f"<ANCHOR> <AMAP_{int(anchor)}> -> <RMATOM> {target_text}")
        shown += 1
    lines.append("If an anchor is not listed above, do not use it with <OP_REMOVE> or <OP_REPLACE>.")
    return "\n".join(lines) + "\n"


def enforce_action_constraints(
    start_smiles_tagged: str,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    constraints = get_action_constraints(start_smiles_tagged)
    valid_anchor_maps = set(int(x) for x in constraints.get("valid_anchor_maps", []))
    removable_by_anchor = {
        int(anchor): [int(x) for x in targets]
        for anchor, targets in dict(constraints.get("removable_by_anchor", {})).items()
    }

    changed = False
    normalized_actions: list[dict[str, Any]] = []
    for idx, action in enumerate(actions, start=1):
        normalized = dict(action)
        op = str(normalized.get("op", "")).strip().upper()
        anchor = int(normalized.get("anchor"))
        removed_atom_map = normalized.get("removed_atom_map")
        if removed_atom_map is not None:
            removed_atom_map = int(removed_atom_map)
        if anchor not in valid_anchor_maps:
            if op in {"REMOVE", "REPLACE"} and removed_atom_map is not None:
                owner_anchors = [
                    int(candidate_anchor)
                    for candidate_anchor, targets in removable_by_anchor.items()
                    if removed_atom_map in targets
                ]
                if len(owner_anchors) == 1:
                    normalized["anchor"] = int(owner_anchors[0])
                    anchor = int(owner_anchors[0])
                    changed = True
                else:
                    return {
                        "is_valid": False,
                        "error": f"constraint_violation: invalid_anchor_at_edit_{idx}",
                        "actions": list(actions),
                        "changed": False,
                    }
            else:
                return {
                    "is_valid": False,
                    "error": f"constraint_violation: invalid_anchor_at_edit_{idx}",
                    "actions": list(actions),
                    "changed": False,
                }
        if anchor not in valid_anchor_maps:
            return {
                "is_valid": False,
                "error": f"constraint_violation: invalid_anchor_at_edit_{idx}",
                "actions": list(actions),
                "changed": False,
            }

        if op in {"REMOVE", "REPLACE"}:
            candidates = list(removable_by_anchor.get(anchor, []))
            if not candidates:
                return {
                    "is_valid": False,
                    "error": f"constraint_violation: no_removable_target_for_anchor_at_edit_{idx}",
                    "actions": list(actions),
                    "changed": False,
                }
            if removed_atom_map is None:
                if len(candidates) == 1:
                    normalized["removed_atom_map"] = int(candidates[0])
                    changed = True
            else:
                if removed_atom_map not in candidates:
                    if len(candidates) == 1:
                        normalized["removed_atom_map"] = int(candidates[0])
                        changed = True
                    else:
                        return {
                            "is_valid": False,
                            "error": f"constraint_violation: invalid_removed_atom_at_edit_{idx}",
                            "actions": list(actions),
                            "changed": False,
                        }
        normalized_actions.append(normalized)

    return {
        "is_valid": True,
        "error": None,
        "actions": normalized_actions,
        "changed": bool(changed),
    }


def _classify_apply_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "atom map" in msg or "not found" in msg:
        return "invalid_anchor"
    if "dummy atom" in msg or "attachment" in msg:
        return "attachment_failed"
    if "sanitize" in msg or "kekul" in msg:
        return "sanitize_failed"
    return "apply_failed"


def apply_edit_actions(start_smiles_tagged: str, actions: list[dict[str, Any]]) -> dict[str, Any]:
    if not str(start_smiles_tagged).strip():
        return {
            "is_valid_syntax": True,
            "is_valid_mol": False,
            "edited_smiles": None,
            "edited_smiles_tagged": None,
            "actions": list(actions),
            "error": "invalid_start_smiles",
        }

    normalized_actions: list[dict[str, Any]] = []
    for a in actions:
        normalized_actions.append(
            {
                "op": str(a.get("op", "")).strip().lower(),
                "anchor_atom_map": int(a.get("anchor")),
                "fg_id": a.get("fg_id"),
                "fg_smiles": a.get("fg_smi"),
                "removed_atom_map": a.get("removed_atom_map"),
            }
        )

    try:
        edited_tagged = apply_edits(str(start_smiles_tagged), normalized_actions)
        edited_plain = smiles_without_atom_maps(mol_from_smiles(edited_tagged))
        return {
            "is_valid_syntax": True,
            "is_valid_mol": True,
            "edited_smiles": str(edited_plain),
            "edited_smiles_tagged": str(edited_tagged),
            "actions": list(actions),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "is_valid_syntax": True,
            "is_valid_mol": False,
            "edited_smiles": None,
            "edited_smiles_tagged": None,
            "actions": list(actions),
            "error": _classify_apply_error(exc),
        }


def execute_edit_seq(start_smiles_tagged: str, edit_seq: str) -> dict[str, Any]:
    normalized_edit_seq, was_repaired = repair_edit_seq(edit_seq)
    parsed = parse_edit_seq(normalized_edit_seq)
    if not bool(parsed["is_valid_syntax"]):
        return {
            "is_valid_syntax": False,
            "is_valid_mol": False,
            "edited_smiles": None,
            "edited_smiles_tagged": None,
            "actions": [],
            "error": parsed["error"],
            "normalized_edit_seq": normalized_edit_seq,
            "was_repaired": bool(was_repaired),
        }
    constraint_result = enforce_action_constraints(start_smiles_tagged, parsed["actions"])
    if not bool(constraint_result["is_valid"]):
        return {
            "is_valid_syntax": True,
            "is_valid_mol": False,
            "edited_smiles": None,
            "edited_smiles_tagged": None,
            "actions": list(parsed["actions"]),
            "error": str(constraint_result["error"]),
            "normalized_edit_seq": normalized_edit_seq,
            "was_repaired": bool(was_repaired),
            "constraint_repaired": False,
        }
    constrained_actions = list(constraint_result["actions"])
    constraint_repaired = bool(constraint_result["changed"])
    if constraint_repaired:
        normalized_edit_seq = serialize_edit_actions(constrained_actions)
    executed = apply_edit_actions(start_smiles_tagged, constrained_actions)
    return {
        "is_valid_syntax": bool(parsed["is_valid_syntax"]),
        "is_valid_mol": bool(executed["is_valid_mol"]),
        "edited_smiles": executed["edited_smiles"],
        "edited_smiles_tagged": executed["edited_smiles_tagged"],
        "actions": constrained_actions,
        "error": executed["error"],
        "normalized_edit_seq": normalized_edit_seq,
        "was_repaired": bool(was_repaired),
        "constraint_repaired": bool(constraint_repaired),
    }
