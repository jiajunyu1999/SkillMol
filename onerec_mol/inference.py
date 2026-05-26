from __future__ import annotations

import json
import re
from typing import Any

import torch

from .executor import execute_edit_seq
from .executor import format_action_constraint_guidance
from .executor import get_action_constraints
from .grpo import sample_completions
from .grpo import sample_completions_batch
from .constants import TASK_DIRECTIVES
from .reward import compute_properties
from .reward import compute_reward
from .token_policy import suggest_fg_ids_for_task


_ACTION_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*")
_FGID_RE = re.compile(r"(?:<FGID>\s*(?:<FGID_([0-9]+)>|([0-9]+))|<FGID_([0-9]+)>)")
_AMAP_RE = re.compile(r"(?:<AMAP_([0-9]+)>|:([0-9]+)\])")
_LEARNED_SCORER_CACHE: dict[str, Any] = {}
_TASK_FEATURE_IDS = [101, 102, 103, 104, 105, 106, 107, 108, 201, 202, 203, 204, 205, 206]

_FG_PROXY: dict[int, dict[str, float]] = {
    0: {"logp": -0.6, "tpsa": 1.2, "hba": 1.0, "hbd": 1.0, "qed": 0.2},
    1: {"logp": -0.3, "tpsa": 1.0, "hba": 1.0, "hbd": 0.0, "qed": 0.1},
    2: {"logp": -0.5, "tpsa": 1.1, "hba": 1.0, "hbd": 1.0, "qed": 0.2},
    3: {"logp": 1.2, "tpsa": 0.0, "hba": 0.0, "hbd": 0.0, "qed": -0.1},
    4: {"logp": 0.9, "tpsa": 0.0, "hba": 0.0, "hbd": 0.0, "qed": 0.0},
    5: {"logp": 1.0, "tpsa": 0.0, "hba": 0.0, "hbd": 0.0, "qed": -0.1},
    6: {"logp": -0.1, "tpsa": 1.0, "hba": 2.0, "hbd": 0.0, "qed": -0.1},
    7: {"logp": 0.3, "tpsa": 0.0, "hba": 0.0, "hbd": 0.0, "qed": 0.0},
    8: {"logp": -0.7, "tpsa": 0.8, "hba": 1.0, "hbd": 2.0, "qed": 0.1},
    9: {"logp": -0.2, "tpsa": 0.6, "hba": 1.0, "hbd": 0.0, "qed": 0.1},
    10: {"logp": -0.5, "tpsa": 1.4, "hba": 2.0, "hbd": 1.0, "qed": 0.1},
    11: {"logp": 0.2, "tpsa": 0.4, "hba": 1.0, "hbd": 0.0, "qed": 0.1},
    12: {"logp": 0.3, "tpsa": 0.4, "hba": 1.0, "hbd": 0.0, "qed": 0.1},
    13: {"logp": -0.1, "tpsa": 0.7, "hba": 2.0, "hbd": 1.0, "qed": 0.1},
    14: {"logp": -0.2, "tpsa": 0.8, "hba": 2.0, "hbd": 0.0, "qed": 0.2},
    15: {"logp": -0.5, "tpsa": 1.5, "hba": 2.0, "hbd": 2.0, "qed": 0.1},
    16: {"logp": -0.1, "tpsa": 0.5, "hba": 1.0, "hbd": 1.0, "qed": 0.1},
    17: {"logp": -0.4, "tpsa": 1.4, "hba": 2.0, "hbd": 2.0, "qed": 0.1},
    18: {"logp": 0.0, "tpsa": 1.0, "hba": 2.0, "hbd": 0.0, "qed": -0.1},
    19: {"logp": -0.2, "tpsa": 1.3, "hba": 2.0, "hbd": 1.0, "qed": -0.1},
    20: {"logp": 1.0, "tpsa": 0.0, "hba": 0.0, "hbd": 0.0, "qed": -0.1},
    21: {"logp": 0.8, "tpsa": 0.0, "hba": 0.0, "hbd": 0.0, "qed": 0.0},
    22: {"logp": 0.0, "tpsa": 0.8, "hba": 1.0, "hbd": 2.0, "qed": 0.0},
    23: {"logp": 0.7, "tpsa": 0.3, "hba": 1.0, "hbd": 0.0, "qed": -0.1},
    24: {"logp": 0.4, "tpsa": 0.5, "hba": 1.0, "hbd": 0.0, "qed": 0.0},
    25: {"logp": 0.6, "tpsa": 0.2, "hba": 1.0, "hbd": 0.0, "qed": 0.0},
    26: {"logp": 0.2, "tpsa": 0.7, "hba": 2.0, "hbd": 0.0, "qed": 0.0},
}


_PROPERTY_PRIOR_TERMS: dict[tuple[str, int], dict[str, list[str]]] = {
    ("logp", -1): {
        "positive": [
            "polar",
            "hydroxyl",
            "amine",
            "amide",
            "carbonyl",
            "heteroatom",
            "oxygen",
            "nitrogen",
            "remove alkyl",
            "remove hydrophobic",
        ],
        "negative": ["alkyl", "hydrophobic", "phenyl", "aromatic", "halogen", "tert", "butyl"],
    },
    ("logp", 1): {
        "positive": ["alkyl", "hydrophobic", "phenyl", "aromatic", "halogen", "methyl", "ethyl"],
        "negative": ["hydroxyl", "amine", "amide", "polar", "oxygen", "nitrogen"],
    },
    ("qed", 1): {
        "positive": ["simplify", "remove", "reduce", "methyl", "amide", "heteroatom"],
        "negative": ["large", "bulky", "fused", "complex", "macrocycle"],
    },
    ("qed", -1): {
        "positive": ["bulky", "large", "complex", "aromatic", "ring"],
        "negative": ["simplify", "remove", "reduce"],
    },
    ("tpsa", -1): {
        "positive": ["remove oxygen", "remove nitrogen", "remove hydroxyl", "remove amine", "alkyl", "hydrophobic"],
        "negative": ["hydroxyl", "amine", "amide", "carbonyl", "oxygen", "nitrogen", "polar"],
    },
    ("tpsa", 1): {
        "positive": ["hydroxyl", "amine", "amide", "carbonyl", "oxygen", "nitrogen", "polar", "heteroatom"],
        "negative": ["alkyl", "hydrophobic", "remove oxygen", "remove nitrogen"],
    },
    ("hba", 1): {
        "positive": ["oxygen", "nitrogen", "carbonyl", "ether", "amide", "acceptor", "heteroatom"],
        "negative": ["remove oxygen", "remove nitrogen", "alkyl"],
    },
    ("hbd", 1): {
        "positive": ["hydroxyl", "amine", "nh", "donor", "amide"],
        "negative": ["remove hydroxyl", "remove amine", "ether", "alkyl"],
    },
}


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-8:
        return [0.5 for _ in values]
    return [(float(x) - lo) / (hi - lo) for x in values]


def _action_terms(text: str) -> set[str]:
    return {x.lower() for x in _ACTION_WORD_RE.findall(str(text))}


def _edit_type(text: str) -> str:
    lowered = str(text).lower()
    for key in ["replace", "remove", "delete", "add", "attach", "insert", "modify", "substitute"]:
        if key in lowered:
            return key
    return "other"


def _task_keyword_prior(text: str, task_id: int | str | None) -> float:
    lowered = str(text).lower()
    score = 0.0
    for prop, direction in TASK_DIRECTIVES.get(int(task_id or -1), []):
        terms = _PROPERTY_PRIOR_TERMS.get((str(prop), int(direction)), {})
        for phrase in terms.get("positive", []):
            if str(phrase).lower() in lowered:
                score += 1.0
        for phrase in terms.get("negative", []):
            if str(phrase).lower() in lowered:
                score -= 1.0
    return score


def _heuristic_candidate_scores(
    candidates: list[dict[str, Any]],
    *,
    previous_actions: list[str],
    task_id: int | str | None,
    gen_config: dict[str, Any],
) -> list[float]:
    if not candidates:
        return []

    previous_terms: set[str] = set()
    previous_types: list[str] = []
    for action in previous_actions:
        previous_terms |= _action_terms(action)
        previous_types.append(_edit_type(action))

    raw_logprob: list[float] = []
    raw_prior: list[float] = []
    diversity: list[float] = []
    repetition: list[float] = []

    for cand in candidates:
        text = str(cand.get("completion", "") or cand.get("raw_completion", ""))
        token_count = int(cand.get("token_count", 0) or 0)
        if token_count <= 0:
            token_count = max(1, len(text.split()))
        raw_logprob.append(float(cand.get("logprob", 0.0)) / max(1, token_count))
        raw_prior.append(_task_keyword_prior(text, task_id))

        terms = _action_terms(text)
        if not terms or not previous_terms:
            diversity.append(1.0)
        else:
            overlap = len(terms & previous_terms) / max(1, len(terms))
            diversity.append(1.0 - float(overlap))

        etype = _edit_type(text)
        repetition.append(float(previous_types.count(etype)))

    logprob = _minmax(raw_logprob)
    prior = _minmax(raw_prior)
    rep = _minmax(repetition)
    w_logprob = float(gen_config.get("heuristic_logprob_weight", 0.45))
    w_div = float(gen_config.get("heuristic_diversity_weight", 0.20))
    w_prior = float(gen_config.get("heuristic_prior_weight", 0.30))
    w_rep = float(gen_config.get("heuristic_repetition_weight", 0.15))
    return [
        w_logprob * logprob[i] + w_div * diversity[i] + w_prior * prior[i] - w_rep * rep[i]
        for i in range(len(candidates))
    ]


def _extract_fg_ids(text: str) -> list[int]:
    out: list[int] = []
    for match in _FGID_RE.finditer(str(text)):
        val = match.group(1) or match.group(2) or match.group(3)
        if val is None:
            continue
        out.append(int(val))
    return out


def _extract_atom_maps(text: str) -> set[int]:
    out: set[int] = set()
    for match in _AMAP_RE.finditer(str(text)):
        val = match.group(1) or match.group(2)
        if val is not None:
            out.add(int(val))
    return out


def _candidate_op_counts(candidate: dict[str, Any]) -> dict[str, int]:
    counts = {"ADD": 0, "REMOVE": 0, "REPLACE": 0}
    actions = candidate.get("execute_result", {}).get("actions", [])
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                continue
            op = str(action.get("op", "")).strip().upper()
            if op in counts:
                counts[op] += 1
    text = str(candidate.get("completion", "") or candidate.get("raw_completion", ""))
    for op in list(counts):
        counts[op] = max(int(counts[op]), int(text.count(f"<OP_{op}>")))
    return counts


def _proxy_delta(candidate: dict[str, Any]) -> dict[str, float]:
    out = {"logp": 0.0, "tpsa": 0.0, "hba": 0.0, "hbd": 0.0, "qed": 0.0}
    actions = candidate.get("execute_result", {}).get("actions", [])
    if not isinstance(actions, list):
        actions = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        op = str(action.get("op", "")).strip().upper()
        mult = 1.0 if op == "ADD" else (0.65 if op == "REPLACE" else -0.7)
        fgid = action.get("fg_id")
        if fgid is not None and int(fgid) in _FG_PROXY:
            proxy = _FG_PROXY[int(fgid)]
        else:
            fg_smi = str(action.get("fg_smi", "") or "")
            proxy = {
                "logp": 0.3 * fg_smi.count("c") + 0.1 * fg_smi.count("C") - 0.3 * (fg_smi.count("N") + fg_smi.count("O")),
                "tpsa": 0.5 * (fg_smi.count("N") + fg_smi.count("O") + fg_smi.count("S")),
                "hba": float(fg_smi.count("N") + fg_smi.count("O")),
                "hbd": 1.0 if ("N" in fg_smi or "O" in fg_smi) else 0.0,
                "qed": 0.0,
            }
        for key in out:
            out[key] += mult * float(proxy.get(key, 0.0))
    return out


def _atom_context_proxy(source_smiles_tagged: str, atom_map: int | None) -> dict[str, float]:
    if atom_map is None:
        return {"logp": 0.0, "tpsa": 0.0, "hba": 0.0, "hbd": 0.0, "qed": 0.0}
    text = str(source_smiles_tagged or "")
    pattern = re.compile(r"(\[[^\]]*:" + re.escape(str(int(atom_map))) + r"\])")
    match = pattern.search(text)
    if match is None:
        return {"logp": 0.0, "tpsa": 0.0, "hba": 0.0, "hbd": 0.0, "qed": 0.0}
    atom = match.group(1)
    window = text[max(0, match.start() - 12) : min(len(text), match.end() + 12)]
    upper = atom.upper()
    win_upper = window.upper()
    is_hetero = any(x in upper for x in ["N", "O", "S", "P"])
    is_oxygen = "O" in upper
    is_nitrogen = "N" in upper
    is_halogen = any(x in upper for x in ["F", "CL", "BR", "I"])
    is_carbon = "C" in upper and not is_hetero
    aromatic = "c" in atom or "n" in atom or "o" in atom or "s" in atom
    carbonyl = "=O" in win_upper or "=[O" in win_upper
    donor = ("H" in upper and (is_oxygen or is_nitrogen)) or "[OH" in upper or "[NH" in upper
    return {
        "logp": (0.35 if is_carbon else 0.0) + (0.25 if aromatic else 0.0) + (0.45 if is_halogen else 0.0) - (0.45 if is_hetero else 0.0),
        "tpsa": (1.0 if is_oxygen else 0.0) + (0.8 if is_nitrogen else 0.0) + (0.7 if "S" in upper else 0.0) + (0.6 if carbonyl else 0.0),
        "hba": float(is_oxygen or is_nitrogen or "S" in upper),
        "hbd": float(donor),
        "qed": -0.05 if is_halogen else (0.05 if is_hetero else 0.0),
    }


def _proxy_delta_with_atom_context(candidate: dict[str, Any]) -> dict[str, float]:
    out = _proxy_delta(candidate)
    actions = candidate.get("execute_result", {}).get("actions", [])
    if not isinstance(actions, list):
        return out
    source = str(candidate.get("source_smiles_tagged", "") or candidate.get("start_smiles_tagged", "") or "")
    for action in actions:
        if not isinstance(action, dict):
            continue
        op = str(action.get("op", "")).strip().upper()
        if op not in {"REMOVE", "REPLACE"}:
            continue
        removed_map = action.get("removed_atom_map")
        if removed_map is None:
            continue
        removed_proxy = _atom_context_proxy(source, int(removed_map))
        for key in out:
            out[key] -= float(removed_proxy.get(key, 0.0))
    return out


def _proxy_property_candidate_scores(
    candidates: list[dict[str, Any]],
    *,
    task_id: int | str | None,
    gen_config: dict[str, Any],
) -> list[float]:
    if not candidates:
        return []
    directives = TASK_DIRECTIVES.get(int(task_id or -1), [])
    raw_proxy: list[float] = []
    raw_fg: list[float] = []
    preferred_fg = set(int(x) for x in suggest_fg_ids_for_task(task_id))
    use_atom_context = str(gen_config.get("proxy_delta_fn", "") or "").strip().lower() in {
        "atom_context",
        "atomctx",
    }
    for cand in candidates:
        delta = _proxy_delta_with_atom_context(cand) if use_atom_context else _proxy_delta(cand)
        score = 0.0
        for prop, direction in directives:
            key = str(prop).lower()
            score += float(direction) * float(delta.get(key, 0.0))
        raw_proxy.append(score)

        fg_ids = []
        actions = cand.get("execute_result", {}).get("actions", [])
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict) and action.get("fg_id") is not None:
                    fg_ids.append(int(action["fg_id"]))
        fg_ids.extend(_extract_fg_ids(str(cand.get("completion", ""))))
        fg_ids = sorted(set(int(x) for x in fg_ids))
        if fg_ids and preferred_fg:
            raw_fg.append(sum(1.0 for x in fg_ids if x in preferred_fg) / max(1, len(fg_ids)))
        elif preferred_fg:
            raw_fg.append(-0.25)
        else:
            raw_fg.append(0.0)

    proxy = _minmax(raw_proxy)
    fg = _minmax(raw_fg)
    logprob = _logprob_candidate_scores(candidates)
    w_proxy = float(gen_config.get("proxy_property_weight", 0.55))
    w_fg = float(gen_config.get("proxy_fg_weight", 0.25))
    w_logprob = float(gen_config.get("proxy_logprob_weight", 0.20))
    return [w_proxy * proxy[i] + w_fg * fg[i] + w_logprob * logprob[i] for i in range(len(candidates))]


def _proxy_property_atomctx_candidate_scores(
    candidates: list[dict[str, Any]],
    *,
    task_id: int | str | None,
    gen_config: dict[str, Any],
) -> list[float]:
    cfg = {**dict(gen_config or {}), "proxy_delta_fn": "atom_context"}
    return _proxy_property_candidate_scores(candidates, task_id=task_id, gen_config=cfg)


def _candidate_structure_penalties(candidate: dict[str, Any]) -> dict[str, float]:
    actions = candidate.get("execute_result", {}).get("actions", [])
    if not isinstance(actions, list):
        actions = []
    source_atom_maps = _extract_atom_maps(str(candidate.get("source_smiles_tagged", "") or ""))
    counts = _candidate_op_counts(candidate)
    num_actions = int(sum(counts.values()))
    anchors: list[int] = []
    rm_atoms: list[int] = []
    fg_ids: list[int] = []
    missing_fgsmi = 0
    missing_remove_atom = 0
    unknown_anchor = 0
    unknown_remove_atom = 0
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("anchor") is not None:
            anchors.append(int(action["anchor"]))
            if source_atom_maps and int(action["anchor"]) not in source_atom_maps:
                unknown_anchor += 1
        if action.get("removed_atom_map") is not None:
            rm_atoms.append(int(action["removed_atom_map"]))
            if source_atom_maps and int(action["removed_atom_map"]) not in source_atom_maps:
                unknown_remove_atom += 1
        if action.get("fg_id") is not None:
            fg_ids.append(int(action["fg_id"]))
        op = str(action.get("op", "")).strip().upper()
        if op in {"ADD", "REPLACE"} and not str(action.get("fg_smi", "") or "").strip():
            missing_fgsmi += 1
        if op in {"REMOVE", "REPLACE"} and action.get("removed_atom_map") is None:
            missing_remove_atom += 1
    if not actions:
        text = str(candidate.get("completion", "") or candidate.get("raw_completion", ""))
        fg_ids = _extract_fg_ids(text)
    duplicate_anchors = max(0, len(anchors) - len(set(anchors)))
    duplicate_rm_atoms = max(0, len(rm_atoms) - len(set(rm_atoms)))
    duplicate_fg_ids = max(0, len(fg_ids) - len(set(fg_ids)))
    anchor_rm_overlap = sum(1 for anchor, rm_atom in zip(anchors, rm_atoms) if int(anchor) == int(rm_atom))
    return {
        "too_many_actions": float(max(0, num_actions - 3)),
        "duplicate_anchors": float(duplicate_anchors),
        "duplicate_rm_atoms": float(duplicate_rm_atoms),
        "duplicate_fg_ids": float(duplicate_fg_ids),
        "missing_fgsmi": float(missing_fgsmi),
        "missing_remove_atom": float(missing_remove_atom),
        "unknown_anchor": float(unknown_anchor),
        "unknown_remove_atom": float(unknown_remove_atom),
        "anchor_rm_overlap": float(anchor_rm_overlap),
    }


def _hybrid_candidate_scores(
    candidates: list[dict[str, Any]],
    *,
    previous_actions: list[str],
    task_id: int | str | None,
    gen_config: dict[str, Any],
) -> list[float]:
    if not candidates:
        return []
    logprob = _logprob_candidate_scores(candidates)
    task_prior = _minmax(
        _task_prior_candidate_scores(
            candidates,
            previous_actions=previous_actions,
            task_id=task_id,
            gen_config={**dict(gen_config or {}), "task_prior_valid_weight": 0.0},
        )
    )
    proxy = _minmax(_proxy_property_candidate_scores(candidates, task_id=task_id, gen_config=gen_config))
    consensus = _minmax(
        _consensus_candidate_scores(
            candidates,
            previous_actions=previous_actions,
            task_id=task_id,
            gen_config={**dict(gen_config or {}), "consensus_prior_weight": 0.0, "consensus_logprob_weight": 0.0},
        )
    )
    w_logprob = float(gen_config.get("hybrid_logprob_weight", 0.55))
    w_task = float(gen_config.get("hybrid_task_prior_weight", 0.20))
    w_proxy = float(gen_config.get("hybrid_proxy_weight", 0.10))
    w_consensus = float(gen_config.get("hybrid_consensus_weight", 0.10))
    action_penalty = float(gen_config.get("hybrid_action_penalty", 0.25))
    dup_anchor_penalty = float(gen_config.get("hybrid_duplicate_anchor_penalty", 0.35))
    dup_fg_penalty = float(gen_config.get("hybrid_duplicate_fg_penalty", 0.10))
    missing_fgsmi_penalty = float(gen_config.get("hybrid_missing_fgsmi_penalty", 0.10))
    missing_remove_atom_penalty = float(gen_config.get("hybrid_missing_remove_atom_penalty", 0.30))
    unknown_atom_penalty = float(gen_config.get("hybrid_unknown_atom_penalty", 0.35))
    anchor_rm_penalty = float(gen_config.get("hybrid_anchor_rm_overlap_penalty", 0.25))
    scores: list[float] = []
    for i, cand in enumerate(candidates):
        penalties = _candidate_structure_penalties(cand)
        penalty = (
            action_penalty * penalties["too_many_actions"]
            + dup_anchor_penalty * penalties["duplicate_anchors"]
            + dup_anchor_penalty * penalties["duplicate_rm_atoms"]
            + dup_fg_penalty * penalties["duplicate_fg_ids"]
            + missing_fgsmi_penalty * penalties["missing_fgsmi"]
            + missing_remove_atom_penalty * penalties["missing_remove_atom"]
            + unknown_atom_penalty * penalties["unknown_anchor"]
            + unknown_atom_penalty * penalties["unknown_remove_atom"]
            + anchor_rm_penalty * penalties["anchor_rm_overlap"]
        )
        scores.append(
            w_logprob * logprob[i]
            + w_task * task_prior[i]
            + w_proxy * proxy[i]
            + w_consensus * consensus[i]
            - float(penalty)
        )
    return scores


def _proxy_logprob_candidate_scores(
    candidates: list[dict[str, Any]],
    *,
    task_id: int | str | None,
    gen_config: dict[str, Any],
) -> list[float]:
    if not candidates:
        return []
    logprob = _logprob_candidate_scores(candidates)
    raw_proxy: list[float] = []
    task = int(task_id or -1)
    for cand in candidates:
        delta = _proxy_delta(cand)
        proxy_score = 0.0
        for prop, direction in TASK_DIRECTIVES.get(task, []):
            proxy_score += float(direction) * float(delta.get(str(prop).lower(), 0.0))
        raw_proxy.append(proxy_score)
    proxy = _minmax(raw_proxy)
    preferred = set(int(x) for x in suggest_fg_ids_for_task(task_id))
    raw_fg: list[float] = []
    for cand in candidates:
        fg_ids: list[int] = []
        actions = cand.get("execute_result", {}).get("actions", [])
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict) and action.get("fg_id") is not None:
                    fg_ids.append(int(action["fg_id"]))
        fg_ids.extend(_extract_fg_ids(str(cand.get("completion", "") or cand.get("raw_completion", ""))))
        fg_ids = sorted(set(int(x) for x in fg_ids))
        if fg_ids and preferred:
            raw_fg.append(sum(1.0 for x in fg_ids if x in preferred) / max(1, len(fg_ids)))
        elif preferred:
            raw_fg.append(-0.25)
        else:
            raw_fg.append(0.0)
    fg = _minmax(raw_fg)
    w_logprob = float(gen_config.get("proxy_logprob_ranker_logprob_weight", 0.60))
    w_proxy = float(gen_config.get("proxy_logprob_ranker_proxy_weight", 0.40))
    w_fg = float(gen_config.get("proxy_logprob_ranker_fg_weight", 0.0))
    action_penalty = float(gen_config.get("proxy_logprob_ranker_action_penalty", 0.25))
    missing_remove_penalty = float(gen_config.get("proxy_logprob_ranker_missing_remove_penalty", 0.20))
    duplicate_anchor_penalty = float(gen_config.get("proxy_logprob_ranker_duplicate_anchor_penalty", 0.20))
    unknown_anchor_penalty = float(gen_config.get("proxy_logprob_ranker_unknown_anchor_penalty", 0.15))
    anchor_rm_penalty = float(gen_config.get("proxy_logprob_ranker_anchor_rm_penalty", 0.10))
    scores: list[float] = []
    for i, cand in enumerate(candidates):
        penalties = _candidate_structure_penalties(cand)
        penalty = (
            action_penalty * penalties["too_many_actions"]
            + missing_remove_penalty * penalties["missing_remove_atom"]
            + duplicate_anchor_penalty * penalties["duplicate_anchors"]
            + unknown_anchor_penalty * penalties["unknown_anchor"]
            + anchor_rm_penalty * penalties["anchor_rm_overlap"]
        )
        scores.append(w_logprob * logprob[i] + w_proxy * proxy[i] + w_fg * fg[i] - penalty)
    return scores


def _task_prior_candidate_scores(
    candidates: list[dict[str, Any]],
    *,
    previous_actions: list[str],
    task_id: int | str | None,
    gen_config: dict[str, Any],
) -> list[float]:
    if not candidates:
        return []

    preferred_fg = set(int(x) for x in suggest_fg_ids_for_task(task_id))
    directives = TASK_DIRECTIVES.get(int(task_id or -1), [])
    wants_logp_down = any(str(prop).lower() == "logp" and int(direction) < 0 for prop, direction in directives)
    wants_logp_up = any(str(prop).lower() == "logp" and int(direction) > 0 for prop, direction in directives)
    wants_tpsa_down = any(str(prop).lower() == "tpsa" and int(direction) < 0 for prop, direction in directives)
    wants_polar_up = any(
        (str(prop).lower() in {"tpsa", "hba", "hbd"} and int(direction) > 0)
        for prop, direction in directives
    )
    wants_qed_up = any(str(prop).lower() == "qed" and int(direction) > 0 for prop, direction in directives)
    wants_qed_down = any(str(prop).lower() == "qed" and int(direction) < 0 for prop, direction in directives)

    raw_logprob: list[float] = []
    raw_keyword: list[float] = []
    raw_fg: list[float] = []
    raw_op: list[float] = []
    raw_valid: list[float] = []
    raw_depth: list[float] = []
    raw_repeat: list[float] = []

    previous_terms: set[str] = set()
    for action in previous_actions:
        previous_terms |= _action_terms(action)

    for cand in candidates:
        text = str(cand.get("completion", "") or cand.get("raw_completion", ""))
        token_count = int(cand.get("token_count", 0) or 0)
        if token_count <= 0:
            token_count = max(1, len(text.split()))
        raw_logprob.append(float(cand.get("logprob", 0.0)) / max(1, token_count))
        raw_keyword.append(_task_keyword_prior(text, task_id))

        fg_ids = []
        actions = cand.get("execute_result", {}).get("actions", [])
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict) and action.get("fg_id") is not None:
                    fg_ids.append(int(action["fg_id"]))
        fg_ids.extend(_extract_fg_ids(text))
        fg_ids = sorted(set(int(x) for x in fg_ids))
        if fg_ids and preferred_fg:
            raw_fg.append(sum(1.0 for x in fg_ids if x in preferred_fg) / max(1, len(fg_ids)))
        elif preferred_fg:
            raw_fg.append(-0.25)
        else:
            raw_fg.append(0.0)

        counts = _candidate_op_counts(cand)
        op_score = 0.0
        if wants_logp_down or wants_tpsa_down or wants_qed_up:
            op_score += 0.35 * counts["REMOVE"] + 0.20 * counts["REPLACE"] - 0.20 * max(0, counts["ADD"] - 1)
        if wants_logp_up or wants_polar_up or wants_qed_down:
            op_score += 0.30 * counts["ADD"] + 0.18 * counts["REPLACE"]
        if wants_polar_up:
            op_score -= 0.20 * counts["REMOVE"]
        if wants_logp_up:
            op_score -= 0.18 * counts["REMOVE"]
        raw_op.append(float(op_score))

        raw_valid.append(
            1.0
            if bool(cand.get("execute_result", {}).get("is_valid_mol", False))
            else (0.25 if bool(cand.get("execute_result", {}).get("is_valid_syntax", False)) else 0.0)
        )
        depth = int(cand.get("depth", 1) or 1)
        raw_depth.append(-0.08 * max(0, depth - 1))

        terms = _action_terms(text)
        if terms and previous_terms:
            raw_repeat.append(len(terms & previous_terms) / max(1, len(terms)))
        else:
            raw_repeat.append(0.0)

    logprob = _minmax(raw_logprob)
    keyword = _minmax(raw_keyword)
    fg = _minmax(raw_fg)
    op = _minmax(raw_op)
    valid = raw_valid
    repeat = _minmax(raw_repeat)

    w_logprob = float(gen_config.get("task_prior_logprob_weight", 0.25))
    w_keyword = float(gen_config.get("task_prior_keyword_weight", 0.30))
    w_fg = float(gen_config.get("task_prior_fg_weight", 0.35))
    w_op = float(gen_config.get("task_prior_op_weight", 0.20))
    w_valid = float(gen_config.get("task_prior_valid_weight", 0.25))
    w_repeat = float(gen_config.get("task_prior_repetition_weight", 0.15))
    return [
        w_logprob * logprob[i]
        + w_keyword * keyword[i]
        + w_fg * fg[i]
        + w_op * op[i]
        + w_valid * valid[i]
        + raw_depth[i]
        - w_repeat * repeat[i]
        for i in range(len(candidates))
    ]


def _task_prior_struct_candidate_scores(
    candidates: list[dict[str, Any]],
    *,
    previous_actions: list[str],
    task_id: int | str | None,
    gen_config: dict[str, Any],
) -> list[float]:
    scores = _task_prior_candidate_scores(
        candidates,
        previous_actions=previous_actions,
        task_id=task_id,
        gen_config=gen_config,
    )
    if not candidates:
        return scores
    action_penalty = float(gen_config.get("task_prior_struct_action_penalty", 0.18))
    dup_anchor_penalty = float(gen_config.get("task_prior_struct_duplicate_anchor_penalty", 0.35))
    dup_fg_penalty = float(gen_config.get("task_prior_struct_duplicate_fg_penalty", 0.10))
    missing_fgsmi_penalty = float(gen_config.get("task_prior_struct_missing_fgsmi_penalty", 0.12))
    missing_remove_atom_penalty = float(gen_config.get("task_prior_struct_missing_remove_atom_penalty", 0.35))
    unknown_atom_penalty = float(gen_config.get("task_prior_struct_unknown_atom_penalty", 0.35))
    anchor_rm_penalty = float(gen_config.get("task_prior_struct_anchor_rm_overlap_penalty", 0.30))
    out: list[float] = []
    for score, cand in zip(scores, candidates):
        penalties = _candidate_structure_penalties(cand)
        penalty = (
            action_penalty * penalties["too_many_actions"]
            + dup_anchor_penalty * penalties["duplicate_anchors"]
            + dup_anchor_penalty * penalties["duplicate_rm_atoms"]
            + dup_fg_penalty * penalties["duplicate_fg_ids"]
            + missing_fgsmi_penalty * penalties["missing_fgsmi"]
            + missing_remove_atom_penalty * penalties["missing_remove_atom"]
            + unknown_atom_penalty * penalties["unknown_anchor"]
            + unknown_atom_penalty * penalties["unknown_remove_atom"]
            + anchor_rm_penalty * penalties["anchor_rm_overlap"]
        )
        out.append(float(score) - float(penalty))
    return out


def _directive_router_candidate_scores(
    candidates: list[dict[str, Any]],
    *,
    previous_actions: list[str],
    task_id: int | str | None,
    gen_config: dict[str, Any],
) -> list[float]:
    if not candidates:
        return []
    task_scores = _minmax(
        _task_prior_candidate_scores(
            candidates,
            previous_actions=previous_actions,
            task_id=task_id,
            gen_config=gen_config,
        )
    )
    hybrid_scores = _minmax(
        _hybrid_candidate_scores(
            candidates,
            previous_actions=previous_actions,
            task_id=task_id,
            gen_config=gen_config,
        )
    )
    directives = TASK_DIRECTIVES.get(int(task_id or -1), [])
    single_decrease = (
        len(directives) == 1
        and str(directives[0][0]).lower() in {"logp", "tpsa"}
        and int(directives[0][1]) < 0
    )
    has_constructive_increase = any(
        str(prop).lower() in {"logp", "tpsa", "hba", "hbd"} and int(direction) > 0
        for prop, direction in directives
    )
    if single_decrease:
        w_hybrid = float(gen_config.get("directive_router_single_decrease_hybrid_weight", 0.70))
    elif has_constructive_increase or len(directives) > 1:
        w_hybrid = float(gen_config.get("directive_router_constructive_hybrid_weight", 0.25))
    else:
        w_hybrid = float(gen_config.get("directive_router_default_hybrid_weight", 0.45))
    w_hybrid = min(max(float(w_hybrid), 0.0), 1.0)
    return [
        (1.0 - w_hybrid) * float(task_scores[i]) + w_hybrid * float(hybrid_scores[i])
        for i in range(len(candidates))
    ]


def _candidate_signature(candidate: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    actions = candidate.get("execute_result", {}).get("actions", [])
    ops: list[str] = []
    fg_ids: list[int] = []
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                continue
            op = str(action.get("op", "")).strip().upper()
            if op:
                ops.append(op)
            if action.get("fg_id") is not None:
                fg_ids.append(int(action["fg_id"]))
    text = str(candidate.get("completion", "") or candidate.get("raw_completion", ""))
    if not ops:
        counts = _candidate_op_counts(candidate)
        ops = [op for op in ["ADD", "REMOVE", "REPLACE"] for _ in range(max(0, int(counts[op])))]
    if not fg_ids:
        fg_ids = _extract_fg_ids(text)
    op_sig = "+".join(ops) if ops else _edit_type(text).upper()
    return op_sig, tuple(sorted(set(int(x) for x in fg_ids)))


def _consensus_candidate_scores(
    candidates: list[dict[str, Any]],
    *,
    previous_actions: list[str],
    task_id: int | str | None,
    gen_config: dict[str, Any],
) -> list[float]:
    if not candidates:
        return []

    prior_scores = _task_prior_candidate_scores(
        candidates,
        previous_actions=previous_actions,
        task_id=task_id,
        gen_config={**dict(gen_config or {}), "task_prior_valid_weight": 0.0},
    )

    signatures = [_candidate_signature(cand) for cand in candidates]
    sig_counts: dict[tuple[str, tuple[int, ...]], int] = {}
    op_counts: dict[str, int] = {}
    fg_counts: dict[int, int] = {}
    for op_sig, fg_sig in signatures:
        sig_counts[(op_sig, fg_sig)] = int(sig_counts.get((op_sig, fg_sig), 0)) + 1
        op_counts[op_sig] = int(op_counts.get(op_sig, 0)) + 1
        for fgid in fg_sig:
            fg_counts[int(fgid)] = int(fg_counts.get(int(fgid), 0)) + 1

    raw_consensus: list[float] = []
    for op_sig, fg_sig in signatures:
        score = float(sig_counts.get((op_sig, fg_sig), 0))
        score += 0.5 * float(op_counts.get(op_sig, 0))
        if fg_sig:
            score += sum(float(fg_counts.get(int(fgid), 0)) for fgid in fg_sig) / max(1, len(fg_sig))
        raw_consensus.append(score)

    consensus = _minmax(raw_consensus)
    prior = _minmax(prior_scores)
    w_consensus = float(gen_config.get("consensus_weight", 0.45))
    w_prior = float(gen_config.get("consensus_prior_weight", 0.40))
    w_logprob = float(gen_config.get("consensus_logprob_weight", 0.15))
    logprob = _logprob_candidate_scores(candidates)
    return [
        w_consensus * consensus[i] + w_prior * prior[i] + w_logprob * logprob[i]
        for i in range(len(candidates))
    ]


def _candidate_feature_rows(
    candidates: list[dict[str, Any]],
    *,
    task_id: int | str | None,
) -> list[list[float]]:
    if not candidates:
        return []
    sigs = [_candidate_signature(cand) for cand in candidates]
    sig_count: dict[tuple[str, tuple[int, ...]], int] = {}
    op_count: dict[str, int] = {}
    fg_count: dict[int, int] = {}
    for sig in sigs:
        sig_count[sig] = int(sig_count.get(sig, 0)) + 1
        op_count[sig[0]] = int(op_count.get(sig[0], 0)) + 1
        for fgid in sig[1]:
            fg_count[int(fgid)] = int(fg_count.get(int(fgid), 0)) + 1

    task = int(task_id or -1)
    preferred = set(int(x) for x in suggest_fg_ids_for_task(task))
    logprob_scores = _logprob_candidate_scores(candidates)
    out: list[list[float]] = []
    denom_rank = max(1, len(candidates) - 1)
    for idx, (cand, norm_logprob, sig) in enumerate(zip(candidates, logprob_scores, sigs)):
        text = str(cand.get("completion", "") or cand.get("raw_completion", ""))
        actions = cand.get("execute_result", {}).get("actions", [])
        fg_ids: list[int] = []
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict) and action.get("fg_id") is not None:
                    fg_ids.append(int(action["fg_id"]))
        fg_ids.extend(_extract_fg_ids(text))
        fg_ids = sorted(set(int(x) for x in fg_ids))
        preferred_frac = (
            sum(1.0 for x in fg_ids if int(x) in preferred) / max(1, len(fg_ids))
            if fg_ids and preferred
            else 0.0
        )
        counts = _candidate_op_counts(cand)
        delta = _proxy_delta(cand)
        proxy_task_delta = 0.0
        for prop, direction in TASK_DIRECTIVES.get(task, []):
            proxy_task_delta += float(direction) * float(delta.get(str(prop).lower(), 0.0))
        penalties = _candidate_structure_penalties(cand)
        out.append(
            [
                float(norm_logprob),
                float(cand.get("token_count", 0) or 0),
                float(cand.get("depth", 1) or 1),
                float(sum(counts.values())),
                float(counts.get("ADD", 0)),
                float(counts.get("REMOVE", 0)),
                float(counts.get("REPLACE", 0)),
                float(len(fg_ids)),
                float(preferred_frac),
                float(_task_keyword_prior(text, task)),
                float(proxy_task_delta),
                float(delta.get("logp", 0.0)),
                float(delta.get("tpsa", 0.0)),
                float(delta.get("hba", 0.0)),
                float(delta.get("hbd", 0.0)),
                float(delta.get("qed", 0.0)),
                float(sig_count.get(sig, 0)),
                float(op_count.get(sig[0], 0)),
                float(max([fg_count.get(int(x), 0) for x in sig[1]], default=0)),
                float(task) / 206.0,
                *[1.0 if task == int(tid) else 0.0 for tid in _TASK_FEATURE_IDS],
                float(idx) / float(denom_rank),
                1.0 / float(idx + 1),
                float(bool(cand.get("execute_result", {}).get("is_valid_syntax", False))),
                float(penalties["too_many_actions"]),
                float(penalties["duplicate_anchors"]),
                float(penalties["duplicate_rm_atoms"]),
                float(penalties["duplicate_fg_ids"]),
                float(penalties["missing_fgsmi"]),
                float(penalties["missing_remove_atom"]),
                float(penalties["unknown_anchor"]),
                float(penalties["unknown_remove_atom"]),
                float(penalties["anchor_rm_overlap"]),
                float(cand.get("search_score", 0.0) or 0.0),
                float(cand.get("heuristic_score", 0.0) or 0.0),
                float(cand.get("search_score_sum", 0.0) or 0.0),
                float(cand.get("final_search_score", 0.0) or 0.0),
            ]
        )
    return out


def _learned_candidate_scores(
    candidates: list[dict[str, Any]],
    *,
    task_id: int | str | None,
    gen_config: dict[str, Any],
) -> list[float]:
    model_path = _candidate_scorer_path_for_task(task_id, gen_config)
    if not model_path:
        return _logprob_candidate_scores(candidates)
    if model_path not in _LEARNED_SCORER_CACHE:
        import joblib

        _LEARNED_SCORER_CACHE[model_path] = joblib.load(model_path)
    payload = _LEARNED_SCORER_CACHE[model_path]
    clf = payload.get("model") if isinstance(payload, dict) else payload
    feats = _candidate_feature_rows(candidates, task_id=task_id)
    if not feats:
        return []
    if isinstance(payload, dict) and payload.get("feature_indices"):
        indices = [int(x) for x in payload.get("feature_indices", [])]
        feats = [[float(row[i]) for i in indices if 0 <= int(i) < len(row)] for row in feats]
    n_model_features = int(getattr(clf, "n_features_in_", len(feats[0])))
    if n_model_features != len(feats[0]):
        fixed_feats = []
        for row in feats:
            if n_model_features <= len(row):
                fixed_feats.append(row[:n_model_features])
            else:
                fixed_feats.append(list(row) + [0.0 for _ in range(n_model_features - len(row))])
        feats = fixed_feats
    scorer_type = str(payload.get("scorer_type", "") if isinstance(payload, dict) else "").strip().lower()
    if scorer_type in {"pairwise", "pairwise_ranker"}:
        if len(feats) == 1:
            return [1.0]
        wins = [0.0 for _ in feats]
        for i, feat_i in enumerate(feats):
            pair_rows: list[list[float]] = []
            opponents: list[int] = []
            for j, feat_j in enumerate(feats):
                if i == j:
                    continue
                pair_rows.append([float(a) - float(b) for a, b in zip(feat_i, feat_j)])
                opponents.append(j)
            if not pair_rows:
                continue
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(pair_rows)
                pair_scores = [float(x) for x in probs[:, 1].tolist()]
            elif hasattr(clf, "decision_function"):
                pair_scores = _minmax([float(x) for x in clf.decision_function(pair_rows)])
            else:
                pair_scores = _minmax([float(x) for x in clf.predict(pair_rows)])
            for _j, score in zip(opponents, pair_scores):
                wins[i] += float(score)
        return _minmax(wins)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(feats)
        return [float(x) for x in probs[:, 1].tolist()]
    if hasattr(clf, "decision_function"):
        raw = [float(x) for x in clf.decision_function(feats)]
        return _minmax(raw)
    raw = [float(x) for x in clf.predict(feats)]
    return _minmax(raw)


_PORTFOLIO_RULES = ["search_score", "search_score_sum", "heuristic_score", "mean_path_score"]


def _portfolio_rule_score(candidate: dict[str, Any], rule: str) -> float:
    if rule == "search_score_sum":
        return float(candidate.get("search_score_sum", candidate.get("search_score", 0.0)) or 0.0)
    if rule == "heuristic_score":
        return float(candidate.get("heuristic_score", candidate.get("search_score", 0.0)) or 0.0)
    if rule == "mean_path_score":
        depth = max(1.0, float(candidate.get("depth", 1) or 1))
        return float(candidate.get("search_score_sum", candidate.get("search_score", 0.0)) or 0.0) / depth
    return float(candidate.get("search_score", 0.0) or 0.0)


def _portfolio_rule_features(candidates: list[dict[str, Any]], cand_idx: int, rule: str) -> list[float]:
    features: list[float] = []
    for name in _PORTFOLIO_RULES:
        scores = [_portfolio_rule_score(cand, name) for cand in candidates]
        value = float(scores[cand_idx]) if cand_idx < len(scores) else 0.0
        ordered = sorted([float(x) for x in scores], reverse=True)
        top = ordered[0] if ordered else 0.0
        second = ordered[1] if len(ordered) > 1 else top
        rank = 1 + sum(1 for x in scores if float(x) > value)
        features.extend(
            [
                value,
                top - value,
                value - second if abs(value - top) < 1e-8 else 0.0,
                float(rank) / max(1.0, float(len(scores))),
                1.0 if name == rule else 0.0,
            ]
        )
    return features


def _portfolio_candidate_scores(
    candidates: list[dict[str, Any]],
    *,
    task_id: int | str | None,
    gen_config: dict[str, Any],
) -> list[float]:
    if not candidates:
        return []
    model_path = _candidate_scorer_path_for_task(task_id, gen_config)
    if not model_path:
        return [float(cand.get("search_score", 0.0) or 0.0) for cand in candidates]
    if model_path not in _LEARNED_SCORER_CACHE:
        import joblib

        _LEARNED_SCORER_CACHE[model_path] = joblib.load(model_path)
    payload = _LEARNED_SCORER_CACHE[model_path]
    clf = payload.get("model") if isinstance(payload, dict) else payload
    base_feats = _candidate_feature_rows(candidates, task_id=task_id)
    scores = [-1.0e9 for _ in candidates]
    rows: list[list[float]] = []
    row_indices: list[int] = []
    seen: set[tuple[int, str]] = set()
    for rule in _PORTFOLIO_RULES:
        best_idx = max(range(len(candidates)), key=lambda i: _portfolio_rule_score(candidates[i], rule))
        key = (int(best_idx), str(rule))
        if key in seen or best_idx >= len(base_feats):
            continue
        seen.add(key)
        rows.append(list(base_feats[best_idx]) + _portfolio_rule_features(candidates, best_idx, rule))
        row_indices.append(best_idx)
    if not rows:
        return [float(cand.get("search_score", 0.0) or 0.0) for cand in candidates]
    n_model_features = int(getattr(clf, "n_features_in_", len(rows[0])))
    fixed_rows = []
    for row in rows:
        if n_model_features <= len(row):
            fixed_rows.append(row[:n_model_features])
        else:
            fixed_rows.append(list(row) + [0.0 for _ in range(n_model_features - len(row))])
    scorer_type = str(payload.get("scorer_type", "") if isinstance(payload, dict) else "").strip().lower()
    if scorer_type == "portfolio_pairwise" and len(fixed_rows) > 1:
        raw = []
        for i, feat_i in enumerate(fixed_rows):
            pair_rows = []
            for j, feat_j in enumerate(fixed_rows):
                if i == j:
                    continue
                pair_rows.append([float(a) - float(b) for a, b in zip(feat_i, feat_j)])
            raw.append(float(sum(float(x) for x in clf.predict_proba(pair_rows)[:, 1].tolist())))
    elif scorer_type == "portfolio_pairwise":
        raw = [1.0]
    else:
        raw = [float(x) for x in clf.predict(fixed_rows)]
    for idx, score in zip(row_indices, raw):
        scores[int(idx)] = float(score)
    return scores


def _candidate_scorer_path_for_task(task_id: int | str | None, gen_config: dict[str, Any]) -> str:
    paths = gen_config.get("candidate_scorer_paths")
    if paths is None:
        paths_json = str(gen_config.get("candidate_scorer_paths_json", "") or "").strip()
        if paths_json:
            try:
                paths = json.loads(paths_json)
            except json.JSONDecodeError:
                paths = None

    if isinstance(paths, dict):
        task_keys: list[str] = []
        try:
            task_keys.append(str(int(task_id)))
        except Exception:  # noqa: BLE001
            pass
        task_keys.append(str(task_id))
        for key in task_keys:
            value = paths.get(key)
            if value:
                return str(value).strip()

        task_int: int | None
        try:
            task_int = int(task_id)
        except Exception:  # noqa: BLE001
            task_int = None
        if task_int is not None:
            for key, value in paths.items():
                if not value:
                    continue
                if key in {"default", "*"}:
                    continue
                try:
                    key_tasks = [int(x) for x in str(key).replace(",", " ").split()]
                except Exception:  # noqa: BLE001
                    key_tasks = []
                if task_int in key_tasks:
                    return str(value).strip()

        for key in ("default", "*"):
            value = paths.get(key)
            if value:
                return str(value).strip()

    return str(gen_config.get("candidate_scorer_path", "")).strip()


def _ensemble_candidate_scores(
    candidates: list[dict[str, Any]],
    *,
    previous_actions: list[str],
    task_id: int | str | None,
    gen_config: dict[str, Any],
) -> list[float]:
    if not candidates:
        return []
    hybrid = _minmax(
        _hybrid_candidate_scores(
            candidates,
            previous_actions=previous_actions,
            task_id=task_id,
            gen_config=gen_config,
        )
    )
    learned = _minmax(_learned_candidate_scores(candidates, task_id=task_id, gen_config=gen_config))
    logprob = _logprob_candidate_scores(candidates)
    w_hybrid = float(gen_config.get("ensemble_hybrid_weight", 0.75))
    w_learned = float(gen_config.get("ensemble_learned_weight", 0.15))
    w_logprob = float(gen_config.get("ensemble_logprob_weight", 0.10))
    return [
        w_hybrid * hybrid[i] + w_learned * learned[i] + w_logprob * logprob[i]
        for i in range(len(candidates))
    ]


def _logprob_candidate_scores(candidates: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for cand in candidates:
        text = str(cand.get("completion", "") or cand.get("raw_completion", ""))
        token_count = int(cand.get("token_count", 0) or 0)
        if token_count <= 0:
            token_count = max(1, len(text.split()))
        values.append(float(cand.get("logprob", 0.0)) / max(1, token_count))
    return _minmax(values)


def _is_valid_candidate(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("execute_result", {}).get("is_valid_mol", False))


def _valid_first_pool(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [x for x in candidates if _is_valid_candidate(x)]
    return valid if valid else list(candidates)


def _candidate_pool(candidates: list[dict[str, Any]], *, validity_mode: str) -> list[dict[str, Any]]:
    mode = str(validity_mode or "valid_first").strip().lower()
    if mode == "valid_first":
        return _valid_first_pool(candidates)
    if mode == "parse_first":
        parsed = [x for x in candidates if bool(x.get("execute_result", {}).get("is_valid_syntax", False))]
        return parsed if parsed else list(candidates)
    return list(candidates)


def _is_non_rdkit_ranker(search_ranker: str) -> bool:
    return str(search_ranker).strip().lower() in {
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
    }


def _candidate_scores_for_ranker(
    search_ranker: str,
    *,
    model,
    candidates: list[dict[str, Any]],
    previous_actions: list[str],
    task_id: int | str | None,
    task_text: str,
    gen_config: dict[str, Any],
) -> list[float] | None:
    ranker = str(search_ranker).strip().lower()
    if ranker in {"heuristic", "non_rdkit", "logprob_prior"}:
        return _heuristic_candidate_scores(
            candidates,
            previous_actions=previous_actions,
            task_id=task_id,
            gen_config=gen_config,
        )
    if ranker in {"task_prior", "task_prior_valid", "policy_prior"}:
        return _task_prior_candidate_scores(
            candidates,
            previous_actions=previous_actions,
            task_id=task_id,
            gen_config=gen_config,
        )
    if ranker in {"task_prior_struct", "policy_prior_struct"}:
        return _task_prior_struct_candidate_scores(
            candidates,
            previous_actions=previous_actions,
            task_id=task_id,
            gen_config=gen_config,
        )
    if ranker in {"directive_router", "router"}:
        return _directive_router_candidate_scores(
            candidates,
            previous_actions=previous_actions,
            task_id=task_id,
            gen_config=gen_config,
        )
    if ranker in {"consensus", "consensus_prior"}:
        return _consensus_candidate_scores(
            candidates,
            previous_actions=previous_actions,
            task_id=task_id,
            gen_config=gen_config,
        )
    if ranker in {"proxy_property", "property_proxy"}:
        return _proxy_property_candidate_scores(
            candidates,
            task_id=task_id,
            gen_config=gen_config,
        )
    if ranker in {"proxy_property_atomctx", "property_proxy_atomctx"}:
        return _proxy_property_atomctx_candidate_scores(
            candidates,
            task_id=task_id,
            gen_config=gen_config,
        )
    if ranker in {"proxy_logprob", "logprob_proxy"}:
        return _proxy_logprob_candidate_scores(
            candidates,
            task_id=task_id,
            gen_config=gen_config,
        )
    if ranker in {"search_score", "path_score", "tree_score"}:
        return [float(cand.get("search_score", 0.0) or 0.0) for cand in candidates]
    if ranker in {"search_score_sum", "path_sum"}:
        return [float(cand.get("search_score_sum", cand.get("search_score", 0.0)) or 0.0) for cand in candidates]
    if ranker in {"learned_scorer", "learned"}:
        return _learned_candidate_scores(
            candidates,
            task_id=task_id,
            gen_config=gen_config,
        )
    if ranker in {"portfolio_scorer", "portfolio"}:
        return _portfolio_candidate_scores(
            candidates,
            task_id=task_id,
            gen_config=gen_config,
        )
    if ranker == "hybrid_default":
        return _hybrid_candidate_scores(
            candidates,
            previous_actions=previous_actions,
            task_id=task_id,
            gen_config={},
        )
    if ranker in {"hybrid", "conservative_hybrid"}:
        return _hybrid_candidate_scores(
            candidates,
            previous_actions=previous_actions,
            task_id=task_id,
            gen_config=gen_config,
        )
    if ranker == "hybrid_atomctx":
        return _hybrid_candidate_scores(
            candidates,
            previous_actions=previous_actions,
            task_id=task_id,
            gen_config={**dict(gen_config or {}), "proxy_delta_fn": "atom_context"},
        )
    if ranker in {"ensemble", "hybrid_learned"}:
        return _ensemble_candidate_scores(
            candidates,
            previous_actions=previous_actions,
            task_id=task_id,
            gen_config=gen_config,
        )
    if ranker in {"logprob", "normalized_logprob"}:
        return _logprob_candidate_scores(candidates)
    if ranker in {"self_rerank", "llm_rerank"}:
        if isinstance(model, dict):
            hf_model, tokenizer = model["model"], model["tokenizer"]
        elif isinstance(model, tuple) and len(model) == 2:
            hf_model, tokenizer = model
        else:
            raise ValueError("self_rerank requires model dict/tuple with tokenizer.")
        rerank_cfg = dict(gen_config or {})
        rerank_cfg["self_rerank_task_id"] = task_id
        return _self_rerank_candidate_scores(
            hf_model,
            tokenizer,
            task_text=str(task_text),
            previous_actions=previous_actions,
            candidates=candidates,
            gen_config=rerank_cfg,
        )
    return None


def _self_rerank_candidate_scores(
    model,
    tokenizer,
    *,
    task_text: str,
    previous_actions: list[str],
    candidates: list[dict[str, Any]],
    gen_config: dict[str, Any],
) -> list[float]:
    if not candidates:
        return []
    max_choices = max(1, int(gen_config.get("self_rerank_max_choices", 8)))
    cand_subset = candidates[:max_choices]
    subset_indices = list(range(len(cand_subset)))
    prefilter = str(gen_config.get("self_rerank_prefilter_ranker", "") or "").strip().lower()
    if prefilter and prefilter not in {"self_rerank", "llm_rerank"}:
        pre_scores = _candidate_scores_for_ranker(
            prefilter,
            model=(model, tokenizer),
            candidates=candidates,
            previous_actions=previous_actions,
            task_id=gen_config.get("self_rerank_task_id"),
            task_text=task_text,
            gen_config={k: v for k, v in dict(gen_config or {}).items() if str(k) != "self_rerank_prefilter_ranker"},
        )
        if pre_scores is not None:
            ranked = sorted(range(len(candidates)), key=lambda i: float(pre_scores[i]), reverse=True)
            subset_indices = ranked[:max_choices]
            cand_subset = [candidates[i] for i in subset_indices]
    lines = [
        "You are choosing one molecular edit action for a molecule optimization task.",
        "Use medicinal-chemistry reasoning and the provided task skills. Do not compute molecular properties.",
        f"Task: {task_text}",
    ]
    if previous_actions:
        lines.append("Previous accepted actions:")
        for action in previous_actions[-3:]:
            lines.append(str(action))
    lines.append("Candidate actions:")
    for i, cand in enumerate(cand_subset, start=1):
        lines.append(f"{i}. {str(cand.get('completion', '')).strip()}")
    lines.append("Return only the number of the best candidate action.")
    prompt = "\n".join(lines)
    hf_model, tok = model, tokenizer
    encoded = tok(prompt, return_tensors="pt").to(hf_model.device)
    with torch.no_grad():
        out = hf_model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=4,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            disable_compile=True,
        )
    answer = tok.decode(out[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
    match = re.search(r"\d+", str(answer))
    chosen = int(match.group(0)) if match else 1
    chosen = min(max(1, chosen), len(cand_subset)) - 1
    scores = [0.0 for _ in candidates]
    scores[int(subset_indices[chosen])] = 1.0
    return scores


def _task_plan_text(task_id: int | str | None, task_text: str) -> str:
    directives = TASK_DIRECTIVES.get(int(task_id or -1), [])
    clauses: list[str] = []
    for prop, direction in directives:
        name = str(prop).upper()
        if str(prop).lower() == "logp":
            clauses.append("prefer polarizing edits and avoid hydrophobic growth" if int(direction) < 0 else "prefer hydrophobic growth and avoid adding polar donors")
        elif str(prop).lower() == "tpsa":
            clauses.append("increase polar surface with heteroatoms" if int(direction) > 0 else "reduce polar groups and avoid new heteroatoms")
        elif str(prop).lower() == "hba":
            clauses.append("add or expose acceptor atoms such as O/N")
        elif str(prop).lower() == "hbd":
            clauses.append("add or expose donor groups such as OH/NH")
        elif str(prop).lower() == "qed":
            clauses.append("balance size, polarity, and lipophilicity" if int(direction) > 0 else "push the molecule away from drug-like balance")
        else:
            clauses.append(f"{'increase' if int(direction) > 0 else 'decrease'} {name}")
    if not clauses:
        return ""
    return (
        "Decision plan for this task:\n"
        f"- Objective: {task_text}.\n"
        + "".join(f"- {x}.\n" for x in clauses)
        + "- Prefer one chemically plausible local edit per step and avoid undoing previous edits.\n"
    )


def _prepare_reward_meta(meta: dict[str, Any]) -> dict[str, Any]:
    reward_meta = dict(meta)
    start_smiles_tagged = str(reward_meta.get("start_smiles_tagged", "")).strip()
    if not start_smiles_tagged or "_before_props" in reward_meta:
        return reward_meta
    try:
        from .chem_utils import mol_from_smiles, smiles_without_atom_maps

        start_plain = str(smiles_without_atom_maps(mol_from_smiles(start_smiles_tagged)))
        reward_meta["_before_props"] = compute_properties(start_plain)
    except Exception:  # noqa: BLE001
        pass
    return reward_meta


def _augment_prompt_with_constraints(prompt: str, meta: dict[str, Any], gen_config: dict[str, Any]) -> str:
    if not bool((gen_config or {}).get("constraint_guidance", True)):
        return str(prompt)
    start_smiles_tagged = str((meta or {}).get("start_smiles_tagged", "")).strip()
    if not start_smiles_tagged:
        return str(prompt)
    suffix = format_action_constraint_guidance(start_smiles_tagged)
    if not suffix.strip():
        return str(prompt)
    base = str(prompt).rstrip()
    return f"{base}\n\n{suffix}"


def _build_action_constraint(
    start_smiles_tagged: str,
    task_id: int | str | None,
    *,
    use_task_fg_constraints: bool = False,
) -> dict[str, Any]:
    constraint = dict(get_action_constraints(str(start_smiles_tagged)))
    if bool(use_task_fg_constraints):
        preferred_fg_ids = suggest_fg_ids_for_task(task_id)
        if preferred_fg_ids:
            constraint["preferred_fg_ids"] = [int(x) for x in preferred_fg_ids]
    return constraint


def _apply_generation_constraint_overrides(
    constraint: dict[str, Any],
    task_id: int | str | None,
    gen_config: dict[str, Any],
) -> dict[str, Any]:
    out = dict(constraint)
    tid_values: list[Any] = []
    try:
        tid_values.append(int(task_id))
    except Exception:  # noqa: BLE001
        pass
    tid_values.append(str(task_id))

    first_op_map = gen_config.get("task_first_op_whitelist", {})
    if isinstance(first_op_map, dict):
        for key in tid_values:
            if key in first_op_map:
                out["first_op_whitelist"] = first_op_map[key]
                break

    max_edits_map = gen_config.get("task_max_edits_map", {})
    if isinstance(max_edits_map, dict):
        for key in tid_values:
            if key in max_edits_map:
                out["max_edits"] = max_edits_map[key]
                break
    return out


def infer_with_rerank(
    model,
    prompt: str,
    meta: dict[str, Any],
    n_samples: int,
    gen_config: dict[str, Any],
) -> dict[str, Any]:
    cfg = dict(gen_config or {})
    cfg["constrained_decoding"] = bool(cfg.get("constrained_decoding", True))
    use_task_fg_constraints = bool(cfg.get("use_task_fg_constraints", False))
    search_ranker = str(cfg.get("search_ranker", "reward")).strip().lower()
    selection_validity_mode = str(
        cfg.get("selection_validity_mode", "parse_first" if _is_non_rdkit_ranker(search_ranker) else "valid_first")
    ).strip().lower()
    reward_meta = _prepare_reward_meta(meta)
    prompt_text = _augment_prompt_with_constraints(str(prompt), reward_meta, cfg)
    cfg["action_constraints"] = [
        _apply_generation_constraint_overrides(
            _build_action_constraint(
                str(reward_meta.get("start_smiles_tagged", "")),
                reward_meta.get("task_id"),
                use_task_fg_constraints=use_task_fg_constraints,
            ),
            reward_meta.get("task_id"),
            cfg,
        )
    ]
    max_rounds = max(1, int(cfg.get("constraint_max_rounds", 1)))
    sample_multiplier = max(1, int(cfg.get("constraint_sample_multiplier", 1)))
    max_search_candidates = int(cfg.get("max_search_candidates", 0) or 0)

    candidates: list[dict[str, Any]] = []
    seen_completions: set[str] = set()
    for _ in range(max_rounds):
        if max_search_candidates > 0 and len(candidates) >= max_search_candidates:
            break
        group_size = max(1, int(n_samples)) * sample_multiplier
        if max_search_candidates > 0:
            group_size = max(1, min(group_size, max_search_candidates - len(candidates)))
        sampled = sample_completions(
            model,
            prompt=prompt_text,
            group_size=group_size,
            gen_config=cfg,
        )

        for item in sampled:
            if max_search_candidates > 0 and len(candidates) >= max_search_candidates:
                break
            raw_completion = str(item.get("completion", ""))
            execute_result = execute_edit_seq(str(reward_meta.get("start_smiles_tagged", "")), raw_completion)
            completion = str(execute_result.get("normalized_edit_seq", "") or raw_completion)
            dedup_key = completion.strip() or raw_completion.strip()
            if dedup_key in seen_completions:
                continue
            seen_completions.add(dedup_key)
            if _is_non_rdkit_ranker(search_ranker):
                reward_result = {}
                reward = 0.0
            else:
                reward_result = compute_reward(reward_meta, execute_result)
                reward = float(reward_result.get("reward", -3.0))
            candidates.append(
                {
                    "completion": completion,
                    "raw_completion": raw_completion,
                    "source_smiles_tagged": str(reward_meta.get("start_smiles_tagged", "")),
                    "logprob": float(item.get("logprob", 0.0)),
                    "token_count": int(item.get("token_count", 0) or 0),
                    "depth": 1,
                    "execute_result": execute_result,
                    "reward_result": reward_result,
                    "reward": float(reward),
                }
            )

        valid_count = sum(int(bool(x.get("execute_result", {}).get("is_valid_mol", False))) for x in candidates)
        if valid_count >= max(1, int(n_samples)):
            break

    if not candidates:
        return {
            "best_completion": "",
            "best_reward": -3.0,
            "best_edited_smiles": None,
            "all_candidates": [],
        }

    pool = _candidate_pool(candidates, validity_mode=selection_validity_mode)
    ranker_scores = _candidate_scores_for_ranker(
        search_ranker,
        model=model,
        candidates=pool,
        previous_actions=[],
        task_id=reward_meta.get("task_id"),
        task_text=str(reward_meta.get("optimization_target", "")),
        gen_config=cfg,
    )
    if ranker_scores is not None:
        for cand, score in zip(pool, ranker_scores):
            cand["search_score"] = float(score)
        best = max(pool, key=lambda x: float(x.get("search_score", 0.0)))
    elif search_ranker in {"self_rerank", "llm_rerank"}:
        scores = []
        for cand, score in zip(pool, scores):
            cand["search_score"] = float(score)
        best = max(pool, key=lambda x: float(x.get("search_score", 0.0)))
    else:
        best = max(_candidate_pool(candidates, validity_mode=selection_validity_mode), key=lambda x: float(x["reward"]))
    if _is_non_rdkit_ranker(search_ranker):
        reward_result = compute_reward(reward_meta, best.get("execute_result", {}))
        best["reward_result"] = reward_result
        best["reward"] = float(reward_result.get("reward", -3.0))
    return {
        "best_completion": str(best["completion"]),
        "best_reward": float(best["reward"]),
        "best_edited_smiles": best["execute_result"].get("edited_smiles"),
        "best_execute_result": best.get("execute_result", {}),
        "all_candidates": candidates,
    }


def infer_with_tree_search(
    model,
    meta: dict[str, Any],
    build_prompt_fn,
    width: int,
    depth: int,
    gen_config: dict[str, Any],
) -> dict[str, Any]:
    cfg = dict(gen_config or {})
    cfg["constrained_decoding"] = bool(cfg.get("constrained_decoding", True))
    use_task_fg_constraints = bool(cfg.get("use_task_fg_constraints", False))
    search_ranker = str(cfg.get("search_ranker", "reward")).strip().lower()
    tree_frontier_ranker = str(cfg.get("tree_frontier_ranker", "") or search_ranker).strip().lower()
    tree_final_ranker = str(cfg.get("tree_final_ranker", "") or search_ranker).strip().lower()
    search_beam_width = max(1, int(cfg.get("search_beam_width", 1)))
    max_search_candidates = int(cfg.get("max_search_candidates", 0) or 0)
    tree_score_mode = str(cfg.get("tree_score_mode", "sum")).strip().lower()
    tree_depth_penalty = float(cfg.get("tree_depth_penalty", 0.0) or 0.0)
    selection_validity_mode = str(
        cfg.get("selection_validity_mode", "parse_first" if _is_non_rdkit_ranker(search_ranker) else "valid_first")
    ).strip().lower()
    reward_meta = _prepare_reward_meta(meta)
    beam_width = max(1, int(width))
    max_depth = max(1, int(depth))
    start_tagged = str(reward_meta.get("start_smiles_tagged", ""))
    sample_multiplier = max(1, int(cfg.get("constraint_sample_multiplier", 1)))

    root = {
        "start_smiles_tagged": start_tagged,
        "path_completions": [],
        "step_rewards": [],
        "heuristic_scores": [],
        "search_score": 0.0,
        "depth": 0,
    }
    frontier = [root]
    all_nodes: list[dict[str, Any]] = []

    def _path_search_score(path_scores: list[float]) -> float:
        if not path_scores:
            return 0.0
        if tree_score_mode in {"mean", "avg", "average"}:
            score = float(sum(path_scores)) / float(len(path_scores))
        elif tree_score_mode in {"last", "step"}:
            score = float(path_scores[-1])
        else:
            score = float(sum(path_scores))
        return score - tree_depth_penalty * float(max(0, len(path_scores) - 1))

    for dep in range(1, max_depth + 1):
        if max_search_candidates > 0 and len(all_nodes) >= max_search_candidates:
            break
        expanded: list[dict[str, Any]] = []
        group_size = max(1, beam_width)
        if max_search_candidates > 0:
            remaining = max_search_candidates - len(all_nodes)
            if remaining <= 0:
                break
            group_size = max(1, min(group_size, (remaining + max(len(frontier), 1) - 1) // max(len(frontier), 1)))
        prompts = [
            _augment_prompt_with_constraints(
                str(build_prompt_fn(str(node["start_smiles_tagged"]))),
                {"start_smiles_tagged": str(node["start_smiles_tagged"])},
                cfg,
            )
            for node in frontier
        ]
        step_cfg = dict(cfg)
        step_cfg["action_constraints"] = [
            _apply_generation_constraint_overrides(
                _build_action_constraint(
                    str(node["start_smiles_tagged"]),
                    reward_meta.get("task_id"),
                    use_task_fg_constraints=use_task_fg_constraints,
                ),
                reward_meta.get("task_id"),
                cfg,
            )
            for node in frontier
        ]
        sampled_groups = sample_completions_batch(
            model,
            prompts=prompts,
            # Keep per-step sampling bounded by width, then continue with valid-first branches.
            group_size=group_size,
            gen_config=step_cfg,
        )
        for node, sampled in zip(frontier, sampled_groups):
            if max_search_candidates > 0 and len(all_nodes) >= max_search_candidates:
                break
            node_start = str(node["start_smiles_tagged"])
            seen_child_completions: set[str] = set()
            node_children: list[dict[str, Any]] = []
            for item in sampled:
                if max_search_candidates > 0 and len(all_nodes) + len(node_children) >= max_search_candidates:
                    break
                raw_completion = str(item.get("completion", ""))
                execute_result = execute_edit_seq(node_start, raw_completion)
                completion = str(execute_result.get("normalized_edit_seq", "") or raw_completion)
                dedup_key = completion.strip() or raw_completion.strip()
                if dedup_key in seen_child_completions:
                    continue
                seen_child_completions.add(dedup_key)
                if _is_non_rdkit_ranker(search_ranker):
                    reward_result = {}
                    reward = 0.0
                else:
                    reward_result = compute_reward(reward_meta, execute_result)
                    reward = float(reward_result.get("reward", -3.0))
                next_tagged = execute_result.get("edited_smiles_tagged")
                child = {
                    "start_smiles_tagged": str(next_tagged) if next_tagged else "",
                    "source_smiles_tagged": str(node_start),
                    "completion": completion,
                    "raw_completion": raw_completion,
                    "parent_path_completions": list(node["path_completions"]),
                    "path_completions": list(node["path_completions"]) + [completion],
                    "step_rewards": list(node["step_rewards"]) + [reward],
                    "heuristic_scores": list(node.get("heuristic_scores", [])),
                    "depth": int(dep),
                    "logprob": float(item.get("logprob", 0.0)),
                    "token_count": int(item.get("token_count", 0) or 0),
                    "search_score": float(node.get("search_score", 0.0)),
                    "execute_result": execute_result,
                    "reward_result": reward_result,
                    "reward": reward,
                }
                node_children.append(child)
            step_scores = _candidate_scores_for_ranker(
                tree_frontier_ranker,
                model=model,
                candidates=node_children,
                previous_actions=list(node.get("path_completions", [])),
                task_id=reward_meta.get("task_id"),
                task_text=str(reward_meta.get("optimization_target", "")),
                gen_config=cfg,
            )
            if step_scores is None:
                step_scores = [float(x.get("reward", -3.0)) for x in node_children]
            for child, step_score in zip(node_children, step_scores):
                path_scores = list(child.get("heuristic_scores", [])) + [float(step_score)]
                child["heuristic_score"] = float(step_score)
                child["heuristic_scores"] = path_scores
                child["search_score_sum"] = float(sum(path_scores))
                child["search_score"] = _path_search_score(path_scores)
                expanded.append(child)
                all_nodes.append(child)
        if not expanded:
            break
        pooled_expanded = _candidate_pool(expanded, validity_mode=selection_validity_mode)
        valid_for_next = [x for x in pooled_expanded if bool(x.get("start_smiles_tagged", ""))]
        if not valid_for_next:
            break
        if _is_non_rdkit_ranker(tree_frontier_ranker):
            valid_for_next = sorted(valid_for_next, key=lambda x: float(x.get("search_score", 0.0)), reverse=True)
        else:
            valid_for_next = sorted(valid_for_next, key=lambda x: float(x["reward"]), reverse=True)
        frontier = valid_for_next[:search_beam_width]

    if not all_nodes:
        return {
            "best_completion": "",
            "best_reward": -3.0,
            "best_edited_smiles": None,
            "best_path": [],
            "best_depth": 0,
            "all_candidates": [],
        }

    best_pool = _candidate_pool(all_nodes, validity_mode=selection_validity_mode)
    final_scores = _candidate_scores_for_ranker(
        tree_final_ranker,
        model=model,
        candidates=best_pool,
        previous_actions=[],
        task_id=reward_meta.get("task_id"),
        task_text=str(reward_meta.get("optimization_target", "")),
        gen_config=cfg,
    )
    if final_scores is not None:
        for cand, score in zip(best_pool, final_scores):
            cand["final_search_score"] = float(score)
        best = max(best_pool, key=lambda x: float(x.get("final_search_score", 0.0)))
        best["search_score"] = float(best.get("final_search_score", 0.0))
    elif _is_non_rdkit_ranker(tree_final_ranker):
        best = max(best_pool, key=lambda x: float(x.get("search_score", 0.0)))
    else:
        best = max(best_pool, key=lambda x: float(x["reward"]))
    if _is_non_rdkit_ranker(tree_final_ranker):
        reward_result = compute_reward(reward_meta, best.get("execute_result", {}))
        best["reward_result"] = reward_result
        best["reward"] = float(reward_result.get("reward", -3.0))
    return {
        "best_completion": str(best.get("completion", "")),
        "best_reward": float(best.get("reward", -3.0)),
        "best_search_score": float(best.get("search_score", sum(best.get("heuristic_scores", [])))),
        "best_edited_smiles": best.get("execute_result", {}).get("edited_smiles"),
        "best_execute_result": best.get("execute_result", {}),
        "best_path": list(best.get("path_completions", [])),
        "best_depth": int(best.get("depth", 0)),
        "all_candidates": all_nodes,
    }
