from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import nn


ATOM_VOCAB = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "H", "Si", "Se", "other"]
BOND_VOCAB = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]
HYBRIDIZATION_VOCAB = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]


@dataclass
class TokenizerConfig:
    num_codebooks: int = 8
    codebook_size: int = 256
    gnn_hidden_dim: int = 256
    gnn_layers: int = 4
    embedding_dim: int = 256
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    commitment_beta: float = 0.25
    seed: int = 42
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs/tokenizer"

    @staticmethod
    def from_dict(config: dict[str, Any] | None) -> "TokenizerConfig":
        if config is None:
            return TokenizerConfig()
        base = TokenizerConfig()
        for key, val in (config or {}).items():
            if hasattr(base, key):
                setattr(base, key, val)
        return base


def _one_hot(value: Any, vocab: list[Any]) -> list[float]:
    out = [0.0 for _ in vocab]
    idx = vocab.index(value) if value in vocab else len(vocab) - 1
    out[idx] = 1.0
    return out


def _atom_features(atom: Chem.Atom) -> list[float]:
    symbol = atom.GetSymbol()
    sym_key = symbol if symbol in ATOM_VOCAB[:-1] else "other"
    degree = min(int(atom.GetDegree()), 5)
    formal_charge = int(atom.GetFormalCharge())
    charge_bucket = formal_charge if formal_charge in {-2, -1, 0, 1, 2} else 0
    hybrid = atom.GetHybridization()
    hfeat = _one_hot(hybrid if hybrid in HYBRIDIZATION_VOCAB else "other", HYBRIDIZATION_VOCAB + ["other"])

    return (
        _one_hot(sym_key, ATOM_VOCAB)
        + _one_hot(degree, [0, 1, 2, 3, 4, 5])
        + _one_hot(charge_bucket, [-2, -1, 0, 1, 2])
        + hfeat
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def _bond_features(bond: Chem.Bond) -> list[float]:
    btype = bond.GetBondType()
    type_feat = _one_hot(btype if btype in BOND_VOCAB else BOND_VOCAB[0], BOND_VOCAB)
    return type_feat + [float(bond.GetIsConjugated()), float(bond.IsInRing())]


def smiles_to_graph(smiles: str) -> dict[str, torch.Tensor]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if mol.GetNumAtoms() == 0:
        raise ValueError(f"Empty molecule: {smiles}")

    node_feats = [_atom_features(atom) for atom in mol.GetAtoms()]
    edge_index: list[list[int]] = [[], []]
    edge_feats: list[list[float]] = []
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        feat = _bond_features(bond)
        edge_index[0].extend([i, j])
        edge_index[1].extend([j, i])
        edge_feats.extend([feat, feat])

    if not edge_feats:
        edge_index = [[0], [0]]
        edge_feats = [[0.0 for _ in range(len(BOND_VOCAB) + 2)]]

    return {
        "node_features": torch.tensor(node_feats, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_features": torch.tensor(edge_feats, dtype=torch.float32),
    }


class GraphConvLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.nei_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, h[src])
        deg = torch.zeros(h.size(0), 1, device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0)
        out = F.relu(self.self_proj(h) + self.nei_proj(agg))
        return self.norm(out)


class MoleculeGNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, embedding_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GraphConvLayer(hidden_dim) for _ in range(int(num_layers))])
        self.out_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        x = graph["node_features"]
        edge_index = graph["edge_index"]
        h = F.relu(self.in_proj(x))
        for layer in self.layers:
            h = layer(h, edge_index)
        pooled = h.mean(dim=0)
        return self.out_proj(pooled)


class ResidualQuantizer(nn.Module):
    def __init__(self, num_codebooks: int, codebook_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.codebooks = nn.Parameter(
            torch.randn(self.num_codebooks, self.codebook_size, self.embedding_dim) * 0.02
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        quantized = torch.zeros_like(z)
        all_ids: list[torch.Tensor] = []
        for idx in range(self.num_codebooks):
            cb = self.codebooks[idx]  # [C, D]
            dist = torch.cdist(residual, cb, p=2) ** 2
            token_ids = torch.argmin(dist, dim=1)
            q = cb[token_ids]
            all_ids.append(token_ids)
            quantized = quantized + q
            residual = residual - q
        tokens = torch.stack(all_ids, dim=1)
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        return quantized, tokens, codebook_loss, commitment_loss

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        residual = z
        all_ids: list[torch.Tensor] = []
        for idx in range(self.num_codebooks):
            cb = self.codebooks[idx]
            dist = torch.cdist(residual, cb, p=2) ** 2
            token_ids = torch.argmin(dist, dim=1)
            q = cb[token_ids]
            all_ids.append(token_ids)
            residual = residual - q
        return torch.stack(all_ids, dim=1)


class GNNRQTokenizer(nn.Module):
    def __init__(self, in_dim: int, cfg: TokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = MoleculeGNNEncoder(
            in_dim=int(in_dim),
            hidden_dim=int(cfg.gnn_hidden_dim),
            num_layers=int(cfg.gnn_layers),
            embedding_dim=int(cfg.embedding_dim),
        )
        self.quantizer = ResidualQuantizer(
            num_codebooks=int(cfg.num_codebooks),
            codebook_size=int(cfg.codebook_size),
            embedding_dim=int(cfg.embedding_dim),
        )

    def forward(self, graphs: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.stack([self.encoder(g) for g in graphs], dim=0)
        quantized, token_ids, codebook_loss, commitment_loss = self.quantizer(z)
        loss = codebook_loss + float(self.cfg.commitment_beta) * commitment_loss
        return loss, token_ids, quantized

    @torch.no_grad()
    def encode_graphs(self, graphs: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        z = torch.stack([self.encoder(g) for g in graphs], dim=0)
        return self.quantizer.encode(z)


def _valid_graphs(smiles_list: list[str]) -> list[dict[str, torch.Tensor]]:
    graphs = []
    for smi in smiles_list:
        try:
            graphs.append(smiles_to_graph(smi))
        except Exception:  # noqa: BLE001
            continue
    if not graphs:
        raise ValueError("No valid molecules found for tokenizer training.")
    return graphs


def _to_device_graph(graph: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in graph.items()}


def train_tokenizer(train_smiles_list: list[str], config: dict[str, Any]) -> str:
    cfg = TokenizerConfig.from_dict(config)
    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    graphs = _valid_graphs([str(x) for x in train_smiles_list if str(x).strip()])
    in_dim = int(graphs[0]["node_features"].shape[1])
    model = GNNRQTokenizer(in_dim=in_dim, cfg=cfg)
    device = torch.device(str(cfg.device))
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    idxs = list(range(len(graphs)))
    for _ in range(int(cfg.num_epochs)):
        random.shuffle(idxs)
        for st in range(0, len(idxs), int(cfg.batch_size)):
            batch_ids = idxs[st : st + int(cfg.batch_size)]
            batch_graphs = [_to_device_graph(graphs[i], device) for i in batch_ids]
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model(batch_graphs)
            loss.backward()
            optimizer.step()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "tokenizer.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "in_dim": in_dim,
        },
        ckpt_path,
    )
    (out_dir / "tokenizer_meta.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return str(ckpt_path)


@lru_cache(maxsize=8)
def _load_tokenizer(ckpt_path: str) -> tuple[GNNRQTokenizer, torch.device]:
    payload = torch.load(str(ckpt_path), map_location="cpu")
    cfg = TokenizerConfig.from_dict(payload["config"])
    model = GNNRQTokenizer(in_dim=int(payload["in_dim"]), cfg=cfg)
    model.load_state_dict(payload["state_dict"])
    device = torch.device(str(cfg.device))
    model.to(device)
    model.eval()
    return model, device


def encode_mol(smiles: str, tokenizer_ckpt_path: str) -> list[int]:
    model, device = _load_tokenizer(str(tokenizer_ckpt_path))
    graph = _to_device_graph(smiles_to_graph(str(smiles)), device)
    with torch.no_grad():
        token_ids = model.encode_graphs([graph])[0]
    return [int(x) for x in token_ids.detach().cpu().tolist()]


def encode_mol_quantized(smiles: str, tokenizer_ckpt_path: str) -> list[float]:
    model, device = _load_tokenizer(str(tokenizer_ckpt_path))
    graph = _to_device_graph(smiles_to_graph(str(smiles)), device)
    with torch.no_grad():
        z = model.encoder(graph).unsqueeze(0)
        quantized, _, _, _ = model.quantizer(z)
    return [float(x) for x in quantized[0].detach().cpu().tolist()]


def encode_mol_batch(smiles_list: list[str], tokenizer_ckpt_path: str) -> list[list[int]]:
    model, device = _load_tokenizer(str(tokenizer_ckpt_path))
    out: list[list[int]] = []
    valid_graphs: list[dict[str, torch.Tensor]] = []
    valid_idx: list[int] = []
    for idx, smi in enumerate(smiles_list):
        try:
            g = _to_device_graph(smiles_to_graph(str(smi)), device)
            valid_graphs.append(g)
            valid_idx.append(idx)
        except Exception:  # noqa: BLE001
            out.append([])
    mapped: dict[int, list[int]] = {}
    if valid_graphs:
        with torch.no_grad():
            tokens = model.encode_graphs(valid_graphs).detach().cpu().tolist()
        for idx, tok in zip(valid_idx, tokens):
            mapped[int(idx)] = [int(t) for t in tok]
    full: list[list[int]] = []
    for i in range(len(smiles_list)):
        full.append(mapped.get(i, []))
    return full


def encode_mol_quantized_batch(smiles_list: list[str], tokenizer_ckpt_path: str) -> list[list[float]]:
    model, device = _load_tokenizer(str(tokenizer_ckpt_path))
    mapped: dict[int, list[float]] = {}
    valid_graphs: list[dict[str, torch.Tensor]] = []
    valid_idx: list[int] = []
    for idx, smi in enumerate(smiles_list):
        try:
            valid_graphs.append(_to_device_graph(smiles_to_graph(str(smi)), device))
            valid_idx.append(idx)
        except Exception:  # noqa: BLE001
            continue
    if valid_graphs:
        with torch.no_grad():
            z = torch.stack([model.encoder(g) for g in valid_graphs], dim=0)
            quantized, _, _, _ = model.quantizer(z)
        for idx, vec in zip(valid_idx, quantized.detach().cpu().tolist()):
            mapped[int(idx)] = [float(x) for x in vec]
    dim = int(model.cfg.embedding_dim)
    return [mapped.get(i, [0.0 for _ in range(dim)]) for i in range(len(smiles_list))]
