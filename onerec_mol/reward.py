from __future__ import annotations

from functools import lru_cache
from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Lipinski, QED, rdMolDescriptors

from .constants import PROPERTY_THRESHOLDS, TASK_DIRECTIVES


@lru_cache(maxsize=8192)
def _compute_properties_cached(smiles: str) -> tuple[float, float, float, float, float]:
    from .chem_utils import mol_from_smiles

    mol = mol_from_smiles(str(smiles))
    return (
        float(Crippen.MolLogP(mol)),
        float(QED.qed(mol)),
        float(rdMolDescriptors.CalcTPSA(mol)),
        float(Lipinski.NumHAcceptors(mol)),
        float(Lipinski.NumHDonors(mol)),
    )


@lru_cache(maxsize=8192)
def _start_props_from_tagged(start_smiles_tagged: str) -> tuple[str, tuple[float, float, float, float, float]]:
    from .chem_utils import mol_from_smiles, smiles_without_atom_maps

    start_plain = str(smiles_without_atom_maps(mol_from_smiles(str(start_smiles_tagged))))
    return start_plain, _compute_properties_cached(start_plain)


def _props_dict(values: tuple[float, float, float, float, float]) -> dict[str, float]:
    return {
        "logp": float(values[0]),
        "qed": float(values[1]),
        "tpsa": float(values[2]),
        "hba": float(values[3]),
        "hbd": float(values[4]),
    }


def compute_properties(smiles: str) -> dict[str, float]:
    try:
        values = _compute_properties_cached(str(smiles))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid SMILES: {smiles}") from exc
    return _props_dict(values)


def _task_directives_from_text(optimization_target: str) -> list[tuple[str, int]]:
    text = str(optimization_target).lower()
    directives: list[tuple[str, int]] = []
    if "qed" in text:
        directives.append(("qed", 1 if "increase" in text else -1))
    if "logp" in text:
        directives.append(("logp", 1 if "increase" in text else -1))
    if "tpsa" in text:
        directives.append(("tpsa", 1 if "increase" in text else -1))
    if "acceptor" in text or "hba" in text:
        directives.append(("hba", 1 if "increase" in text else -1))
    if "donor" in text or "hbd" in text:
        directives.append(("hbd", 1 if "increase" in text else -1))
    return directives


def _evaluate_directives(
    directives: list[tuple[str, int]],
    before_props: dict[str, float],
    after_props: dict[str, float],
) -> dict[str, Any]:
    strict_all = True
    loose_all = True
    normalized_gains: list[float] = []
    positive_gains: list[float] = []
    success_flags: list[float] = []
    deltas: dict[str, float] = {}
    for prop, direction in directives:
        before = float(before_props.get(prop, 0.0))
        after = float(after_props.get(prop, 0.0))
        delta = float(after - before)
        signed = float(delta if int(direction) > 0 else -delta)
        deltas[prop] = delta
        thr = float(PROPERTY_THRESHOLDS.get(prop, 0.0))
        loose_ok = signed > 0.0
        strict_ok = signed > thr
        if not loose_ok:
            loose_all = False
        if not strict_ok:
            strict_all = False
        denom = max(thr, 1e-6)
        norm_gain = float(signed / denom)
        normalized_gains.append(norm_gain)
        positive_gains.append(float(max(0.0, norm_gain)))
        success_flags.append(1.0 if loose_ok else 0.0)
    property_improvement = float(sum(normalized_gains) / max(len(normalized_gains), 1))
    success_rate = float(sum(success_flags) / max(len(success_flags), 1))
    positive_gain_mean = float(sum(positive_gains) / max(len(positive_gains), 1))
    min_gain = float(min(normalized_gains)) if normalized_gains else 0.0
    strict_margin = float(min_gain - 1.0)
    return {
        "strict_hit": bool(strict_all),
        "loose_hit": bool(loose_all),
        "property_improvement": property_improvement,
        "success_rate": success_rate,
        "positive_gain_mean": positive_gain_mean,
        "deltas": deltas,
        "normalized_gains": normalized_gains,
        "min_normalized_gain": min_gain,
        "strict_margin": strict_margin,
    }


def evaluate_task_hit(task_id: int, before_props: dict[str, float], after_props: dict[str, float]) -> dict[str, Any]:
    directives = TASK_DIRECTIVES.get(int(task_id), [])
    if not directives:
        return {
            "strict_hit": False,
            "loose_hit": False,
            "property_improvement": 0.0,
            "success_rate": 0.0,
            "positive_gain_mean": 0.0,
            "deltas": {},
            "normalized_gains": [],
            "min_normalized_gain": 0.0,
            "strict_margin": -1.0,
        }
    return _evaluate_directives(list(directives), before_props, after_props)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def compute_reward(sample_meta: dict[str, Any], execute_result: dict[str, Any]) -> dict[str, Any]:
    invalid_reward = float(sample_meta.get("invalid_reward", -1.0))
    valid_bonus = float(sample_meta.get("valid_bonus", 1.0))
    strict_bonus = float(sample_meta.get("strict_bonus", 5.0))
    loose_bonus = float(sample_meta.get("loose_bonus", 2.0))
    edit_penalty_coef = float(sample_meta.get("edit_penalty_coef", 0.1))
    property_scale = float(sample_meta.get("property_scale", 1.0))
    strict_margin_scale = float(sample_meta.get("strict_margin_scale", 1.25))
    almost_strict_bonus = float(sample_meta.get("almost_strict_bonus", 0.5))
    almost_strict_floor = float(sample_meta.get("almost_strict_floor", 0.7))
    bottleneck_scale = float(sample_meta.get("bottleneck_scale", 1.0))
    reward_mode = str(sample_meta.get("reward_mode", "bottleneck_lex")).strip().lower()
    lex_c_weight = float(sample_meta.get("lex_c_weight", 0.55))
    lex_b_weight = float(sample_meta.get("lex_b_weight", 0.35))
    lex_a_weight = float(sample_meta.get("lex_a_weight", 0.10))
    lex_complete_base = float(sample_meta.get("lex_complete_base", 1.0))
    lex_complete_a_weight = float(sample_meta.get("lex_complete_a_weight", 0.10))
    lex_complete_b_weight = float(sample_meta.get("lex_complete_b_weight", 0.15))
    lex_gap_weight = float(sample_meta.get("lex_gap_weight", 0.30))
    lex_strict_eps = float(sample_meta.get("lex_strict_eps", 1e-6))
    deficit_worst_weight = float(sample_meta.get("deficit_worst_weight", 0.55))
    deficit_mean_weight = float(sample_meta.get("deficit_mean_weight", 0.30))
    deficit_top2_weight = float(sample_meta.get("deficit_top2_weight", 0.15))
    deficit_focus_weight = float(sample_meta.get("deficit_focus_weight", 0.25))
    edit_success_magnitude_cap = float(sample_meta.get("edit_success_magnitude_cap", 2.0))
    process_valid_base = float(sample_meta.get("process_valid_base", 0.0))
    process_hit_weight = float(sample_meta.get("process_hit_weight", 0.0))
    process_strict_bonus = float(sample_meta.get("process_strict_bonus", 0.0))
    process_loose_reward = float(sample_meta.get("process_loose_reward", 1.0))
    process_strict_reward = float(sample_meta.get("process_strict_reward", 2.0))
    process_margin_weight = float(sample_meta.get("process_margin_weight", 1.0))
    process_margin_cap = float(sample_meta.get("process_margin_cap", 1.0))
    strict_valid_base = float(sample_meta.get("strict_valid_base", 0.5))
    strict_loose_base = float(sample_meta.get("strict_loose_base", 1.5))
    strict_hit_base = float(sample_meta.get("strict_hit_base", 4.0))
    strict_success_weight = float(sample_meta.get("strict_success_weight", 0.5))
    strict_margin_reward_weight = float(sample_meta.get("strict_margin_reward_weight", 0.5))

    if not bool(execute_result.get("is_valid_mol", False)):
        return {
            "reward": invalid_reward,
            "components": {
                "valid_bonus": 0.0,
                "strict_hit_bonus": 0.0,
                "loose_hit_bonus": 0.0,
                "property_delta": 0.0,
                "strict_margin": 0.0,
                "almost_strict_bonus": 0.0,
                "bottleneck_gain": 0.0,
                "edit_success_product": 0.0,
                "edit_penalty": 0.0,
                "invalid_penalty": invalid_reward,
            },
            "metrics": {
                "strict_hit": False,
                "loose_hit": False,
                "success_rate": 0.0,
                "success_magnitude": 0.0,
                "min_normalized_gain": 0.0,
                "normalized_gains": [],
                "strict_margin_raw": -1.0,
                "qed_before": None,
                "qed_after": None,
            },
        }

    start_tagged = str(sample_meta.get("start_smiles_tagged", ""))
    edited_smiles = str(execute_result.get("edited_smiles", ""))
    if not start_tagged or not edited_smiles:
        return {
            "reward": invalid_reward,
            "components": {
                "valid_bonus": 0.0,
                "strict_hit_bonus": 0.0,
                "loose_hit_bonus": 0.0,
                "property_delta": 0.0,
                "strict_margin": 0.0,
                "almost_strict_bonus": 0.0,
                "bottleneck_gain": 0.0,
                "edit_success_product": 0.0,
                "edit_penalty": 0.0,
                "invalid_penalty": invalid_reward,
            },
            "metrics": {
                "strict_hit": False,
                "loose_hit": False,
                "success_rate": 0.0,
                "success_magnitude": 0.0,
                "min_normalized_gain": 0.0,
                "normalized_gains": [],
                "strict_margin_raw": -1.0,
            },
        }

    try:
        cached_before = sample_meta.get("_before_props")
        if isinstance(cached_before, dict):
            before_props = {str(k): float(v) for k, v in cached_before.items()}
        else:
            _, before_vals = _start_props_from_tagged(start_tagged)
            before_props = _props_dict(before_vals)
        after_props = compute_properties(edited_smiles)
    except Exception:
        return {
            "reward": invalid_reward,
            "components": {
                "valid_bonus": 0.0,
                "strict_hit_bonus": 0.0,
                "loose_hit_bonus": 0.0,
                "property_delta": 0.0,
                "strict_margin": 0.0,
                "almost_strict_bonus": 0.0,
                "bottleneck_gain": 0.0,
                "edit_success_product": 0.0,
                "edit_penalty": 0.0,
                "invalid_penalty": invalid_reward,
            },
            "metrics": {
                "strict_hit": False,
                "loose_hit": False,
                "success_rate": 0.0,
                "success_magnitude": 0.0,
                "min_normalized_gain": 0.0,
                "normalized_gains": [],
                "strict_margin_raw": -1.0,
            },
        }

    task_id = int(sample_meta.get("task_id", 0))
    hit = evaluate_task_hit(task_id, before_props, after_props)
    if not TASK_DIRECTIVES.get(task_id):
        parsed_directives = _task_directives_from_text(sample_meta.get("optimization_target", ""))
        if parsed_directives:
            hit = _evaluate_directives(list(parsed_directives), before_props, after_props)

    strict_hit = bool(hit["strict_hit"])
    loose_hit = bool(hit["loose_hit"])
    success_rate = _clip(float(hit.get("success_rate", 0.0)), 0.0, 1.0)
    success_magnitude = _clip(float(hit.get("positive_gain_mean", 0.0)), 0.0, edit_success_magnitude_cap)
    edit_success_product = float(success_rate * success_magnitude)
    property_term = _clip(float(hit["property_improvement"]) * property_scale, -2.0, 2.0)
    strict_margin = float(hit.get("strict_margin", -1.0))
    strict_margin_term = float(strict_margin_scale) * _clip(strict_margin, -0.75, 0.5)
    floor = _clip(almost_strict_floor, 0.0, 0.99)
    min_gain = float(hit.get("min_normalized_gain", 0.0))
    bottleneck_term = float(bottleneck_scale) * _clip(min_gain, -0.25, 1.25)
    almost_term = 0.0
    if (not strict_hit) and loose_hit and min_gain >= floor:
        progress = _clip((min_gain - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0)
        almost_term = float(almost_strict_bonus) * float(progress)
    num_edits = len(list(execute_result.get("actions", [])))
    edit_penalty = -edit_penalty_coef * float(num_edits)
    strict_hit_bonus = 0.0
    loose_hit_bonus = 0.0
    dense_term = 0.0
    deficit_max = 0.0
    deficit_mean = 0.0
    deficit_top2 = 0.0

    use_lex = reward_mode in {"bottleneck_lex", "bottleneck", "lex"}
    use_deficit_lex = reward_mode in {"bottleneck_deficit_lex", "deficit_lex", "bottleneck_deficit"}
    use_edit_product = reward_mode in {"edit_product", "edit_success_product"}
    use_process_valid_hit = reward_mode in {"process_valid_hit", "process_hit", "valid_then_hit"}
    use_strict_first = reward_mode in {"strict_valid_hit", "strict_first", "valid_loose_strict"}
    gains = [float(x) for x in hit.get("normalized_gains", [])]
    if use_strict_first:
        reward = float(strict_valid_base)
        if loose_hit:
            loose_hit_bonus = float(max(0.0, strict_loose_base - strict_valid_base))
            reward = float(strict_loose_base + strict_success_weight * success_rate)
        if strict_hit:
            strict_hit_bonus = float(max(0.0, strict_hit_base - max(strict_loose_base, strict_valid_base)))
            reward = float(
                strict_hit_base
                + strict_success_weight * success_rate
                + strict_margin_reward_weight * _clip(min_gain - 1.0, 0.0, 1.0)
            )
    elif use_process_valid_hit:
        if strict_hit:
            margin_norm = _clip((min_gain - 1.0) / max(process_margin_cap, 1e-6), 0.0, 1.0)
            strict_hit_bonus = float(process_strict_reward)
            strict_margin_term = float(process_margin_weight * margin_norm)
            reward = float(process_strict_reward + strict_margin_term)
            if process_strict_bonus != 0.0:
                reward += float(process_strict_bonus)
                strict_hit_bonus += float(process_strict_bonus)
        elif loose_hit:
            loose_hit_bonus = float(process_loose_reward)
            reward = float(process_loose_reward)
        else:
            reward = float(process_valid_base)
    elif use_edit_product:
        reward = float(edit_success_product)
    elif (use_lex or use_deficit_lex) and gains:
        z = [_clip(g - 1.0, -1.0, 1.0) for g in gains]
        c = float(sum(1.0 for g in gains if g > (1.0 + lex_strict_eps)) / max(len(gains), 1))
        b = float(min(z))
        a = float(sum(z) / max(len(z), 1))
        strict_gap = _clip(1.0 - float(min_gain), 0.0, 2.0)
        if use_deficit_lex:
            deficits = [_clip(1.0 - g, 0.0, 2.0) for g in gains]
            deficit_max = float(max(deficits)) if deficits else 0.0
            deficit_mean = float(sum(deficits) / max(len(deficits), 1))
            top2 = sorted(deficits, reverse=True)[: max(1, min(2, len(deficits)))]
            deficit_top2 = float(sum(top2) / max(len(top2), 1))
            near_threshold = _clip(float(min_gain), 0.0, 1.0)
            if strict_hit:
                dense_term = (
                    float(lex_complete_base)
                    + float(lex_complete_b_weight) * b
                    + float(lex_complete_a_weight) * a
                )
            else:
                dense_term = (
                    float(lex_c_weight) * c
                    + float(lex_a_weight) * a
                    + float(deficit_focus_weight) * near_threshold
                    - float(deficit_worst_weight) * deficit_max
                    - float(deficit_mean_weight) * deficit_mean
                    - float(deficit_top2_weight) * deficit_top2
                    - float(lex_gap_weight) * strict_gap
                )
        else:
            if strict_hit:
                dense_term = (
                    float(lex_complete_base)
                    + float(lex_complete_b_weight) * b
                    + float(lex_complete_a_weight) * a
                )
            else:
                dense_term = (
                    float(lex_c_weight) * c
                    + float(lex_b_weight) * b
                    + float(lex_a_weight) * a
                    - float(lex_gap_weight) * strict_gap
                )
        reward = dense_term + almost_term + edit_penalty
        if strict_hit:
            strict_hit_bonus = strict_bonus
            reward += strict_hit_bonus
        elif loose_hit and loose_bonus != 0.0:
            loose_hit_bonus = loose_bonus
            reward += loose_hit_bonus
    else:
        reward = valid_bonus + property_term + strict_margin_term + bottleneck_term + almost_term + edit_penalty
        if strict_hit:
            strict_hit_bonus = strict_bonus
            reward += strict_hit_bonus
        elif loose_hit:
            loose_hit_bonus = loose_bonus
            reward += loose_hit_bonus

    return {
        "reward": float(reward),
        "components": {
            "valid_bonus": float(valid_bonus),
            "strict_hit_bonus": float(strict_hit_bonus),
            "loose_hit_bonus": float(loose_hit_bonus),
            "property_delta": float(property_term),
            "strict_margin": float(strict_margin_term),
            "bottleneck_gain": float(bottleneck_term),
            "edit_success_product": float(edit_success_product),
            "process_valid_base": float(process_valid_base),
            "process_hit_value": float(process_hit_weight * success_rate),
            "almost_strict_bonus": float(almost_term),
            "dense_lex": float(dense_term),
            "deficit_max": float(deficit_max),
            "deficit_mean": float(deficit_mean),
            "deficit_top2": float(deficit_top2),
            "edit_penalty": float(edit_penalty),
            "invalid_penalty": 0.0,
        },
        "metrics": {
            "strict_hit": bool(strict_hit),
            "loose_hit": bool(loose_hit),
            "success_rate": float(success_rate),
            "success_magnitude": float(success_magnitude),
            "min_normalized_gain": float(min_gain),
            "normalized_gains": [float(x) for x in gains],
            "strict_margin_raw": float(strict_margin),
            "qed_before": float(before_props.get("qed", 0.0)),
            "qed_after": float(after_props.get("qed", 0.0)),
            "logp_before": float(before_props.get("logp", 0.0)),
            "logp_after": float(after_props.get("logp", 0.0)),
            "tpsa_before": float(before_props.get("tpsa", 0.0)),
            "tpsa_after": float(after_props.get("tpsa", 0.0)),
            "hba_before": float(before_props.get("hba", 0.0)),
            "hba_after": float(after_props.get("hba", 0.0)),
            "hbd_before": float(before_props.get("hbd", 0.0)),
            "hbd_after": float(after_props.get("hbd", 0.0)),
        },
    }
