# Variational Graph Autoencoder using R-GCN

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from rgcn_model import RGCNScratch, DistMultDecoder

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def sample_vgae_negatives(
    pos_edges: torch.Tensor,
    num_nodes: int,
    neg_ratio: int = 3,
    known_edge_ids: torch.Tensor | None = None,
    num_relations: int | None = None,
    max_attempts: int = 5,
) -> torch.Tensor:
    """Sample negatives by corrupting either endpoint."""
    neg_edges = pos_edges.repeat_interleave(neg_ratio, dim=0).clone()
    rows = torch.arange(len(neg_edges), device=pos_edges.device)
    attempts = max_attempts if known_edge_ids is not None else 1

    for _ in range(attempts):
        if len(rows) == 0:
            break
        corrupt_src = torch.rand(len(rows), device=pos_edges.device) < 0.5
        random_nodes = torch.randint(0, num_nodes, (len(rows),), device=pos_edges.device)
        neg_edges[rows[corrupt_src], 0] = random_nodes[corrupt_src]
        neg_edges[rows[~corrupt_src], 1] = random_nodes[~corrupt_src]
        if known_edge_ids is None:
            break
        rows = rows[
            edge_id_isin(neg_edges[rows], known_edge_ids, num_nodes, num_relations)
        ]

    return neg_edges


def add_reciprocal_edges(edges: torch.Tensor, num_relations: int) -> torch.Tensor:
    reciprocal = edges.clone()
    reciprocal[:, 0] = edges[:, 1]
    reciprocal[:, 1] = edges[:, 0]
    reciprocal[:, 2] = edges[:, 2] + num_relations
    return torch.cat([edges, reciprocal], dim=0)


def edge_ids(
    edges: torch.Tensor, num_nodes: int, num_relations: int | None = None
) -> torch.Tensor:
    if num_relations is None:
        num_relations = int(edges[:, 2].max().item()) + 1
    return (edges[:, 0] * num_relations + edges[:, 2]) * num_nodes + edges[:, 1]


def build_known_edge_ids(
    edge_sets, num_nodes: int, num_relations: int, sort: bool = True
) -> torch.Tensor:
    ids = torch.cat([edge_ids(edges, num_nodes, num_relations) for edges in edge_sets])
    ids = torch.unique(ids)
    return torch.sort(ids).values if sort else ids


def edge_id_isin(
    edges: torch.Tensor,
    known_edge_ids: torch.Tensor,
    num_nodes: int,
    num_relations: int | None = None,
) -> torch.Tensor:
    if len(known_edge_ids) == 0:
        return torch.zeros(len(edges), dtype=torch.bool, device=edges.device)
    ids = edge_ids(edges, num_nodes, num_relations)
    idx = torch.searchsorted(known_edge_ids, ids)
    valid = idx < len(known_edge_ids)
    idx = idx.clamp(max=max(len(known_edge_ids) - 1, 0))
    return valid & (known_edge_ids[idx] == ids)


def sample_vgae_tail_negatives(
    src: torch.Tensor,
    rel: torch.Tensor,
    num_nodes: int,
    num_negatives: int,
    known_edge_ids: torch.Tensor | None = None,
    num_relations: int | None = None,
    max_attempts: int = 5,
) -> torch.Tensor:
    neg_dst = torch.randint(0, num_nodes, (len(src), num_negatives), device=src.device)
    if known_edge_ids is None:
        return neg_dst

    flat_src = src[:, None].expand(-1, num_negatives).reshape(-1)
    flat_rel = rel[:, None].expand(-1, num_negatives).reshape(-1)
    flat_dst = neg_dst.reshape(-1)
    rows = torch.arange(len(flat_dst), device=src.device)

    for _ in range(max_attempts):
        if len(rows) == 0:
            break
        edges = torch.stack([flat_src[rows], flat_dst[rows], flat_rel[rows]], dim=1)
        rows = rows[edge_id_isin(edges, known_edge_ids, num_nodes, num_relations)]
        if len(rows) == 0:
            break
        flat_dst[rows] = torch.randint(0, num_nodes, (len(rows),), device=src.device)

    return flat_dst.view(len(src), num_negatives)


def kl_divergence(
    mu: torch.Tensor, log_var: torch.Tensor, node_ids: torch.Tensor | None = None
) -> torch.Tensor:
    kl_per_node = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    if node_ids is not None:
        kl_per_node = kl_per_node[torch.unique(node_ids)]
    return kl_per_node.mean()


def kl_beta_for_epoch(
    epoch: int,
    max_beta: float,
    warmup_epochs: int | None = None,
    pretrain_epochs: int = 0,
) -> float:
    if max_beta <= 0 or epoch <= pretrain_epochs:
        return 0.0
    if warmup_epochs is None:
        warmup_epochs = 10
    return max_beta * min(1.0, (epoch - pretrain_epochs) / max(1, warmup_epochs))


class VGAE(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_relations, latent_dim=None, feat_dim=None):
        super().__init__()
        latent_dim = latent_dim or hidden_dim
        self.uses_node_features = feat_dim is not None
        self.encoder = RGCNScratch(
            num_nodes,
            num_relations,
            hidden_dim=hidden_dim,
            feat_dim=feat_dim,
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_var_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = DistMultDecoder(num_relations, latent_dim)

    def forward(self, edge_index, edge_type, node_features=None, num_nodes=None):
        if self.uses_node_features and node_features is None:
            raise ValueError("node_features must be provided when VGAE is initialized with feat_dim")
        if not self.uses_node_features and node_features is not None:
            raise ValueError("Pass feat_dim to VGAE if you want to use node_features")

        h = self.encoder(edge_index, edge_type, node_features, num_nodes)
        mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        z = (
            mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
            if self.training
            else mu
        )
        return z, mu, log_var

    def fit(
        self,
        graph,
        epochs,
        lr=1e-3,
        batch_size=4096,
        neg_ratio=3,
        kl_beta=1e-4,
        kl_warmup_epochs=10,
        kl_pretrain_epochs=5,
        weight_decay=1e-5,
        use_adamw=True,
        filtered_negatives=False,
        reciprocal_edges=False,
        return_history=False,
    ):
        self.to(DEVICE)
        super().train(True)
        optimizer_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
        optimizer = optimizer_cls(self.parameters(), lr=lr, weight_decay=weight_decay)
        total_loss = 0.0
        history = {
            "loss": [],
            "bce_loss": [],
            "kl_loss": [],
            "kl_beta": [],
            "pos_logit": [],
            "neg_logit": [],
            "mu_abs": [],
        }
        node_features = graph.node_features.to(DEVICE) if graph.node_features is not None else None
        base_num_relations = graph.num_relations
        effective_num_relations = base_num_relations * 2 if reciprocal_edges else base_num_relations
        if self.decoder.relation_emb.num_embeddings < effective_num_relations:
            raise ValueError(
                "Initialize VGAE with num_relations=graph.num_relations * 2 "
                "when reciprocal_edges=True"
            )
        train_edges = graph.train_edges.to(DEVICE)
        train_edges = (
            add_reciprocal_edges(train_edges, base_num_relations)
            if reciprocal_edges
            else train_edges
        )
        known_edge_ids = None
        if filtered_negatives:
            known_edge_sets = [
                split.to(DEVICE)
                for split in (graph.train_edges, graph.val_edges, graph.test_edges)
            ]
            if reciprocal_edges:
                known_edge_sets = [
                    add_reciprocal_edges(edges, base_num_relations)
                    for edges in known_edge_sets
                ]
            known_edge_ids = build_known_edge_ids(
                known_edge_sets, graph.num_nodes, effective_num_relations
            )
        edge_index = train_edges[:, :2].t().contiguous()
        edge_type = train_edges[:, 2]

        for epoch in tqdm(range(1, epochs + 1)):
            shuffled_edges = train_edges[torch.randperm(len(train_edges), device=DEVICE)]
            epoch_loss = 0.0
            epoch_bce = 0.0
            epoch_kl = 0.0
            epoch_pos_logit = 0.0
            epoch_neg_logit = 0.0
            epoch_mu_abs = 0.0
            num_batches = 0
            effective_kl_beta = kl_beta_for_epoch(
                epoch,
                kl_beta,
                warmup_epochs=kl_warmup_epochs,
                pretrain_epochs=kl_pretrain_epochs,
            )

            for start in range(0, len(shuffled_edges), batch_size):
                pos_edges = shuffled_edges[start : start + batch_size]
                neg_edges = sample_vgae_negatives(
                    pos_edges,
                    graph.num_nodes,
                    neg_ratio,
                    known_edge_ids=known_edge_ids,
                    num_relations=effective_num_relations,
                )
                all_edges = torch.cat([pos_edges, neg_edges], dim=0)
                labels = torch.cat(
                    [
                        torch.ones(len(pos_edges), device=DEVICE),
                        torch.zeros(len(neg_edges), device=DEVICE),
                    ]
                )

                z, mu, log_var = self(
                    edge_index,
                    edge_type,
                    node_features=node_features,
                    num_nodes=graph.num_nodes,
                )

                logits = self.decoder(z, all_edges[:, 0], all_edges[:, 1], all_edges[:, 2])
                bce_loss = F.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=labels.new_tensor(neg_ratio)
                )
                active_nodes = all_edges[:, :2].reshape(-1)
                kl_loss = kl_divergence(mu, log_var, active_nodes)
                loss = bce_loss + effective_kl_beta * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_bce += bce_loss.item()
                epoch_kl += kl_loss.item()
                epoch_pos_logit += logits[: len(pos_edges)].mean().item()
                epoch_neg_logit += logits[len(pos_edges) :].mean().item()
                epoch_mu_abs += mu[torch.unique(active_nodes)].abs().mean().item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            total_loss += avg_loss
            history["loss"].append(avg_loss)
            history["bce_loss"].append(epoch_bce / num_batches)
            history["kl_loss"].append(epoch_kl / num_batches)
            history["kl_beta"].append(effective_kl_beta)
            history["pos_logit"].append(epoch_pos_logit / num_batches)
            history["neg_logit"].append(epoch_neg_logit / num_batches)
            history["mu_abs"].append(epoch_mu_abs / num_batches)

        return history if return_history else total_loss / epochs
