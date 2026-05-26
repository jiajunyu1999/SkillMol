from __future__ import annotations

TASK_TEXT: dict[int, str] = {
    101: "decrease logP",
    102: "increase logP",
    103: "increase QED",
    104: "decrease QED",
    105: "decrease TPSA",
    106: "increase TPSA",
    107: "increase hydrogen bond acceptors",
    108: "increase hydrogen bond donors",
    201: "decrease logP and increase hydrogen bond acceptors",
    202: "increase logP and increase hydrogen bond acceptors",
    203: "decrease logP and increase hydrogen bond donors",
    204: "increase logP and increase hydrogen bond donors",
    205: "decrease logP and decrease TPSA",
    206: "decrease logP and increase TPSA",
    301: "increase QED and increase DRD2",
}

# (property_key, direction) direction: +1 increase, -1 decrease
TASK_DIRECTIVES: dict[int, list[tuple[str, int]]] = {
    101: [("logp", -1)],
    102: [("logp", 1)],
    103: [("qed", 1)],
    104: [("qed", -1)],
    105: [("tpsa", -1)],
    106: [("tpsa", 1)],
    107: [("hba", 1)],
    108: [("hbd", 1)],
    201: [("logp", -1), ("hba", 1)],
    202: [("logp", 1), ("hba", 1)],
    203: [("logp", -1), ("hbd", 1)],
    204: [("logp", 1), ("hbd", 1)],
    205: [("logp", -1), ("tpsa", -1)],
    206: [("logp", -1), ("tpsa", 1)],
    301: [("qed", 1), ("drd2", 1)],
}

PROPERTY_THRESHOLDS: dict[str, float] = {
    "logp": 0.5,
    "qed": 0.1,
    "tpsa": 10.0,
    "hba": 1.0,
    "hbd": 1.0,
    "drd2": 0.05,
}

