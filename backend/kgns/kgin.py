from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KGINLite(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_entities: int,
        num_relations: int,
        emb_dim: int = 64,
        num_intents: int = 4,
        alpha: float = 0.6,
        beta: float = 0.3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.dropout = nn.Dropout(float(dropout))

        self.user_embedding = nn.Embedding(self.num_users, emb_dim)
        self.item_embedding = nn.Embedding(self.num_items, emb_dim)
        self.entity_embedding = nn.Embedding(self.num_entities, emb_dim)
        self.relation_embedding = nn.Embedding(self.num_relations, emb_dim)
        self.intent_embedding = nn.Embedding(int(num_intents), emb_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.xavier_uniform_(self.intent_embedding.weight)

    def forward(
        self,
        edge_item_idx: torch.Tensor,
        edge_rel_idx: torch.Tensor,
        edge_ent_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_base = self.user_embedding.weight
        item_base = self.item_embedding.weight
        entity_all = self.entity_embedding.weight
        relation_all = self.relation_embedding.weight
        intent_all = self.intent_embedding.weight

        # KG relation-aware aggregation: item <- (relation, entity)
        msg = entity_all[edge_ent_idx] + relation_all[edge_rel_idx]
        agg = torch.zeros_like(item_base)
        agg = agg.index_add(0, edge_item_idx, msg)

        cnt = torch.zeros(self.num_items, 1, dtype=item_base.dtype, device=item_base.device)
        ones = torch.ones(edge_item_idx.shape[0], 1, dtype=item_base.dtype, device=item_base.device)
        cnt = cnt.index_add(0, edge_item_idx, ones)
        kg_item = agg / torch.clamp(cnt, min=1.0)

        # Intent routing: item -> latent intents
        intent_logits = torch.matmul(item_base, intent_all.t())
        intent_att = F.softmax(intent_logits, dim=1)
        intent_item = torch.matmul(intent_att, intent_all)

        item_out = item_base + self.alpha * kg_item + self.beta * intent_item
        if self.training:
            item_out = self.dropout(item_out)
        return user_base, item_out

    def intent_orthogonality_loss(self) -> torch.Tensor:
        intents = F.normalize(self.intent_embedding.weight, dim=1)
        sim = torch.matmul(intents, intents.t())
        eye = torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
        return torch.mean((sim - eye) ** 2)


@dataclass
class TrainResult:
    user_emb: np.ndarray
    item_emb: np.ndarray
    avg_loss: float


def sample_bpr_triplets(
    user_pos_items: dict[int, list[int]],
    user_pos_sets: dict[int, set[int]],
    num_items: int,
    num_samples: int,
    rng: random.Random,
) -> np.ndarray:
    users = list(user_pos_items.keys())
    if not users:
        return np.zeros((0, 3), dtype=np.int64)

    triplets = []
    for _ in range(int(num_samples)):
        u = users[rng.randrange(len(users))]
        pos_list = user_pos_items[u]
        pos = pos_list[rng.randrange(len(pos_list))]
        pos_set = user_pos_sets[u]

        neg = rng.randrange(num_items)
        guard = 0
        while neg in pos_set and guard < 100:
            neg = rng.randrange(num_items)
            guard += 1
        if neg in pos_set:
            continue
        triplets.append((u, pos, neg))

    if not triplets:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(triplets, dtype=np.int64)


def train_kgin_bpr(
    interactions: np.ndarray,
    kg_edges: np.ndarray,
    num_users: int,
    num_items: int,
    num_entities: int,
    num_relations: int,
    emb_dim: int = 64,
    num_intents: int = 4,
    alpha: float = 0.6,
    beta: float = 0.3,
    dropout: float = 0.1,
    epochs: int = 30,
    lr: float = 1e-3,
    reg: float = 1e-4,
    intent_reg: float = 1e-3,
    batch_size: int = 4096,
    samples_per_epoch: int = 200000,
    seed: int = 42,
    device: str = "cpu",
) -> TrainResult:
    if interactions.size == 0:
        raise ValueError("interactions is empty")
    if kg_edges.size == 0:
        raise ValueError("kg_edges is empty")

    device_t = torch.device(device)
    rng = random.Random(int(seed))

    model = KGINLite(
        num_users=num_users,
        num_items=num_items,
        num_entities=num_entities,
        num_relations=num_relations,
        emb_dim=emb_dim,
        num_intents=num_intents,
        alpha=alpha,
        beta=beta,
        dropout=dropout,
    ).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    edge_item_idx = torch.tensor(kg_edges[:, 0], dtype=torch.long, device=device_t)
    edge_rel_idx = torch.tensor(kg_edges[:, 1], dtype=torch.long, device=device_t)
    edge_ent_idx = torch.tensor(kg_edges[:, 2], dtype=torch.long, device=device_t)

    user_pos_items: dict[int, list[int]] = {}
    for u, i in interactions:
        user_pos_items.setdefault(int(u), []).append(int(i))
    user_pos_sets = {u: set(v) for u, v in user_pos_items.items()}

    total_loss = 0.0
    total_steps = 0

    for _ in range(int(epochs)):
        triplets = sample_bpr_triplets(
            user_pos_items=user_pos_items,
            user_pos_sets=user_pos_sets,
            num_items=num_items,
            num_samples=samples_per_epoch,
            rng=rng,
        )
        if triplets.shape[0] == 0:
            continue

        perm = np.random.permutation(len(triplets))
        triplets = triplets[perm]

        model.train()
        for start in range(0, len(triplets), int(batch_size)):
            batch = triplets[start : start + int(batch_size)]
            if batch.shape[0] == 0:
                continue

            user_out, item_out = model(edge_item_idx, edge_rel_idx, edge_ent_idx)
            u_idx = torch.tensor(batch[:, 0], dtype=torch.long, device=device_t)
            p_idx = torch.tensor(batch[:, 1], dtype=torch.long, device=device_t)
            n_idx = torch.tensor(batch[:, 2], dtype=torch.long, device=device_t)

            u = user_out[u_idx]
            p = item_out[p_idx]
            n = item_out[n_idx]

            pos_scores = torch.sum(u * p, dim=1)
            neg_scores = torch.sum(u * n, dim=1)
            bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

            l2 = (u.norm(2).pow(2) + p.norm(2).pow(2) + n.norm(2).pow(2)) / float(batch.shape[0])
            orth = model.intent_orthogonality_loss()
            loss = bpr_loss + float(reg) * l2 + float(intent_reg) * orth

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            total_steps += 1

    model.eval()
    with torch.no_grad():
        user_out, item_out = model(edge_item_idx, edge_rel_idx, edge_ent_idx)

    avg_loss = total_loss / max(1, total_steps)
    return TrainResult(
        user_emb=user_out.detach().cpu().numpy(),
        item_emb=item_out.detach().cpu().numpy(),
        avg_loss=float(avg_loss),
    )
