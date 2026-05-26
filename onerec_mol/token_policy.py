from __future__ import annotations

from typing import Iterable

from .constants import TASK_DIRECTIVES

# Heuristic FG buckets for task-aware token constraints.
# IDs follow data/fg_list_small.json.
_FG_HYDROPHOBIC: set[int] = {3, 4, 5, 7, 20, 21, 23, 25}
_FG_POLAR: set[int] = {0, 1, 2, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 26}
_FG_HBA_UP: set[int] = {0, 1, 2, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 25, 26}
_FG_HBD_UP: set[int] = {0, 2, 8, 10, 15, 17, 19, 22}
_FG_QED_UP: set[int] = {0, 2, 8, 11, 12, 13, 14, 16, 17, 22, 23, 24, 25, 26}


def _score_fg_candidates(directives: Iterable[tuple[str, int]]) -> list[int]:
    score: dict[int, int] = {}

    def _add(ids: set[int]) -> None:
        for fid in ids:
            score[int(fid)] = int(score.get(int(fid), 0)) + 1

    for prop, direction in directives:
        p = str(prop).strip().lower()
        d = int(direction)
        if p == "logp":
            _add(_FG_HYDROPHOBIC if d > 0 else _FG_POLAR)
        elif p == "tpsa":
            _add(_FG_POLAR if d > 0 else _FG_HYDROPHOBIC)
        elif p == "hba" and d > 0:
            _add(_FG_HBA_UP)
        elif p == "hbd" and d > 0:
            _add(_FG_HBD_UP)
        elif p == "qed" and d > 0:
            _add(_FG_QED_UP)

    if not score:
        return []
    best = max(int(v) for v in score.values())
    keep_floor = max(1, int(best) - 1)
    # Keep near-best candidates for exploration while biasing toward task-relevant FG IDs.
    ranked = sorted(((-int(v), int(fid)) for fid, v in score.items() if int(v) >= int(keep_floor)))
    return [int(fid) for _, fid in ranked[:12]]


def suggest_fg_ids_for_task(task_id: int | str | None) -> list[int]:
    if task_id is None:
        return []
    try:
        tid = int(task_id)
    except Exception:  # noqa: BLE001
        return []
    directives = list(TASK_DIRECTIVES.get(int(tid), []))
    if not directives:
        return []
    return _score_fg_candidates(directives)
