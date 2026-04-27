from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_triples(triple_path: Path) -> tuple[np.ndarray, dict, dict]:
    df = pd.read_csv(triple_path)
    entities = sorted(set(df["head"]).union(set(df["tail"])))
    relations = sorted(set(df["relation"]))
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}

    triples = np.stack(
        [
            df["head"].map(entity2id).to_numpy(),
            df["relation"].map(relation2id).to_numpy(),
            df["tail"].map(entity2id).to_numpy(),
        ],
        axis=1,
    ).astype(np.int64)
    return triples, entity2id, relation2id


def build_norm_adj(num_nodes: int, edge_index: np.ndarray) -> torch.Tensor:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dst in edge_index:
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0
    np.fill_diagonal(adj, 1.0)

    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=(deg > 0))
    deg_inv_sqrt[deg_inv_sqrt == np.inf] = 0
    d = np.diag(deg_inv_sqrt)
    norm = d @ adj @ d
    return torch.tensor(norm, dtype=torch.float32)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

