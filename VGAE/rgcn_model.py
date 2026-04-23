"""
rgcn_model.py
=============
Dataset-agnostic R-GCN implementation.

Contains everything needed to build and train a Relational Graph Convolutional
Network for link prediction — with no dependency on any specific dataset.

Usage
-----
    from rgcn_model import (
        RelationalGraph,
        RGCNLinkPredictor,       # from-scratch encoder + DistMult decoder
        RGCNLinkPredictorPyG,    # PyG encoder + DistMult decoder
        train,
        evaluate,
        plot_history,
    )

Reference
---------
Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional
Networks", ESWC 2018. https://arxiv.org/abs/1703.06103
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ---------------------------------------------------------------------------
# 1. Shared Data Structure
# ---------------------------------------------------------------------------


@dataclass
class RelationalGraph:
    """
    A dataset-agnostic container for a relational graph.

    All edge tensors have shape (E, 3): [src, dst, rel_type]
    Node features are optional; if absent the model uses a learned embedding.
    """

    name: str  # human-readable name e.g. "tkgl-smallpedia"
    num_nodes: int  # total number of nodes
    num_relations: int  # total number of relation types

    # Edge splits — each is a LongTensor of shape (E_split, 3): [src, dst, rel]
    train_edges: torch.Tensor
    val_edges: torch.Tensor
    test_edges: torch.Tensor

    # Optional: pre-computed node feature matrix (num_nodes, feat_dim)
    # If None the model will use a learned embedding table instead
    node_features: Optional[torch.Tensor] = None

    # Optional: human-readable label maps for interpretability
    node_labels: dict = field(default_factory=dict)  # int → string
    rel_labels: dict = field(default_factory=dict)  # int → string

    def summary(self):
        print(f"RelationalGraph: {self.name}")
        print(f"  Nodes         : {self.num_nodes:,}")
        print(f"  Relation types: {self.num_relations:,}")
        print(f"  Train edges   : {len(self.train_edges):,}")
        print(f"  Val edges     : {len(self.val_edges):,}")
        print(f"  Test edges    : {len(self.test_edges):,}")
        print(
            f"  Node features : {'yes' if self.node_features is not None else 'no (will use learned embeddings)'}"
        )


# ---------------------------------------------------------------------------
# 2. R-GCN From Scratch
# ---------------------------------------------------------------------------


class RGCNLayerScratch(nn.Module):
    """
    Single R-GCN layer with basis decomposition, implemented from scratch.

    Maps:  (num_nodes, in_dim)  →  (num_nodes, out_dim)

    Parameters
    ----------
    in_dim       : input feature dimension
    out_dim      : output feature dimension
    num_relations: number of distinct edge types  (|R| in the paper)
    num_bases    : number of basis matrices B  (paper recommends 10–100)
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int, num_bases: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

        # Basis decomposition parameters (Eq. 3)
        # V_b: B shared basis matrices, each (in_dim, out_dim)
        self.bases = nn.Parameter(torch.Tensor(num_bases, in_dim, out_dim))
        # a_rb: per-relation coefficients over the bases (num_relations, num_bases)
        self.coefficients = nn.Parameter(torch.Tensor(num_relations, num_bases))

        # Self-loop weight W_0 (Eq. 2)
        self.self_loop = nn.Parameter(torch.Tensor(in_dim, out_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        # Glorot (Xavier) uniform initialisation — same as the paper
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.self_loop)
        nn.init.xavier_uniform_(self.coefficients)

    def forward(
        self,
        x: torch.Tensor,  # (num_nodes, in_dim)
        edge_index: torch.Tensor,  # (2, E)  — row 0 = src, row 1 = dst
        edge_type: torch.Tensor,  # (E,)    — relation index per edge
    ) -> torch.Tensor:  # (num_nodes, out_dim)

        num_nodes = x.size(0)

        # Step 1: compute W_r for each relation via basis decomposition
        # W_r = sum_b a_rb * V_b  →  shape: (num_relations, in_dim, out_dim)
        W = torch.einsum("rb,bio->rio", self.coefficients, self.bases)

        # Step 2: aggregate neighbour messages for each relation
        src, dst = edge_index[0], edge_index[1]

        # Degree normalisation: compute inverse degree once
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(len(dst), device=x.device))
        deg_inv = 1.0 / deg.clamp(min=1.0)

        agg = torch.zeros(num_nodes, self.out_dim, device=x.device)

        # Memory-efficient aggregation: loop over relations
        for r in range(self.num_relations):
            edge_mask = edge_type == r
            if not edge_mask.any():
                continue

            r_src = src[edge_mask]
            r_dst = dst[edge_mask]

            # Transform source node features with the relation-specific weight
            msg = x[r_src] @ W[r]
            msg = msg * deg_inv[r_dst].unsqueeze(1)
            agg.scatter_add_(0, r_dst.unsqueeze(1).expand(-1, self.out_dim), msg)

        # Step 3: add self-loop (W_0 * h_i) and activate
        out = agg + x @ self.self_loop
        return out


class RGCNScratch(nn.Module):
    """
    Multi-layer R-GCN encoder (from scratch).

    Produces node embeddings from a relational graph.
    If the graph has no node features, a learned embedding table is used
    as the input layer (common practice for knowledge graphs).

    Parameters
    ----------
    num_nodes    : number of nodes
    num_relations: number of relation types
    hidden_dim   : dimension of hidden (and output) representations
    num_layers   : number of R-GCN layers (paper uses 2)
    num_bases    : number of basis matrices for decomposition
    feat_dim     : input feature dimension; if None uses a learned embedding
    dropout      : dropout rate applied between layers
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_bases: int = 30,
        feat_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout

        if feat_dim is None:
            self.embedding = nn.Embedding(num_nodes, hidden_dim)
            in_dim = hidden_dim
        else:
            self.embedding = None
            self.input_proj = nn.Linear(feat_dim, hidden_dim)
            in_dim = hidden_dim

        self.layers = nn.ModuleList(
            [
                RGCNLayerScratch(in_dim, hidden_dim, num_relations, num_bases)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:

        if self.embedding is not None:
            N = num_nodes or (edge_index.max().item() + 1)
            x = self.embedding(torch.arange(N, device=edge_index.device))
        else:
            x = F.relu(self.input_proj(node_features))

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


# ---------------------------------------------------------------------------
# 3. DistMult Decoder + Full Model Wrappers
# ---------------------------------------------------------------------------


class DistMultDecoder(nn.Module):
    """
    DistMult link prediction decoder (Section 4 of the paper).

    Scores a triple (s, r, o) as:  h_s . diag(R_r) . h_o
    """

    def __init__(self, num_relations: int, hidden_dim: int):
        super().__init__()
        self.relation_emb = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(
        self,
        node_emb: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        rel: torch.Tensor,
    ) -> torch.Tensor:

        h_s = node_emb[src]
        h_o = node_emb[dst]
        r = self.relation_emb(rel)
        scores = (h_s * r * h_o).sum(dim=-1)
        return scores


class RGCNLinkPredictor(nn.Module):
    """
    Full model: R-GCN encoder (from scratch) + DistMult decoder.
    This is the complete architecture from Section 4 of the paper.
    """

    def __init__(
        self,
        num_nodes,
        num_relations,
        hidden_dim=64,
        num_layers=2,
        num_bases=30,
        feat_dim=None,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = RGCNScratch(
            num_nodes,
            num_relations,
            hidden_dim,
            num_layers,
            num_bases,
            feat_dim,
            dropout,
        )
        self.decoder = DistMultDecoder(num_relations, hidden_dim)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        rel: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:

        node_emb = self.encoder(edge_index, edge_type, node_features, num_nodes)
        scores = self.decoder(node_emb, src, dst, rel)
        return scores


# ---------------------------------------------------------------------------
# 4. PyG Model Variants (optional — requires torch_geometric)
# ---------------------------------------------------------------------------

try:
    from torch_geometric.nn import RGCNConv

    PYG_AVAILABLE = True

    class RGCNPyG(nn.Module):
        """
        R-GCN encoder built with PyTorch Geometric's RGCNConv.

        Functionally identical to RGCNScratch.
        PyG's RGCNConv accepts num_bases directly and handles
        the basis decomposition internally.
        """

        def __init__(
            self,
            num_nodes,
            num_relations,
            hidden_dim=64,
            num_layers=2,
            num_bases=30,
            feat_dim=None,
            dropout=0.1,
        ):
            super().__init__()
            self.dropout = dropout

            if feat_dim is None:
                self.embedding = nn.Embedding(num_nodes, hidden_dim)
                in_dim = hidden_dim
            else:
                self.embedding = None
                self.input_proj = nn.Linear(feat_dim, hidden_dim)
                in_dim = hidden_dim

            self.layers = nn.ModuleList(
                [
                    RGCNConv(
                        in_dim,
                        hidden_dim,
                        num_relations=num_relations,
                        num_bases=num_bases,
                    )
                    for _ in range(num_layers)
                ]
            )

        def forward(self, edge_index, edge_type, node_features=None, num_nodes=None):
            if self.embedding is not None:
                N = num_nodes or (edge_index.max().item() + 1)
                x = self.embedding(torch.arange(N, device=edge_index.device))
            else:
                x = F.relu(self.input_proj(node_features))

            for i, layer in enumerate(self.layers):
                x = layer(x, edge_index, edge_type)
                if i < len(self.layers) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            return x

    class RGCNLinkPredictorPyG(nn.Module):
        """Full model: PyG R-GCN encoder + DistMult decoder."""

        def __init__(
            self,
            num_nodes,
            num_relations,
            hidden_dim=64,
            num_layers=2,
            num_bases=30,
            feat_dim=None,
            dropout=0.1,
        ):
            super().__init__()
            self.encoder = RGCNPyG(
                num_nodes,
                num_relations,
                hidden_dim,
                num_layers,
                num_bases,
                feat_dim,
                dropout,
            )
            self.decoder = DistMultDecoder(num_relations, hidden_dim)

        def forward(
            self,
            edge_index,
            edge_type,
            src,
            dst,
            rel,
            node_features=None,
            num_nodes=None,
        ):
            node_emb = self.encoder(edge_index, edge_type, node_features, num_nodes)
            return self.decoder(node_emb, src, dst, rel)

except ImportError:
    PYG_AVAILABLE = False
    RGCNPyG = None
    RGCNLinkPredictorPyG = None


# ---------------------------------------------------------------------------
# 5. Training & Evaluation Utilities
# ---------------------------------------------------------------------------


def sample_negatives(
    pos_edges: torch.Tensor, num_nodes: int, neg_ratio: int = 1
) -> torch.Tensor:
    """
    For each positive edge (s, r, o), sample neg_ratio negative edges
    by randomly corrupting the destination node.

    Returns a tensor of shape (E * neg_ratio, 3).
    """
    src = pos_edges[:, 0].repeat(neg_ratio)
    rel = pos_edges[:, 2].repeat(neg_ratio)
    dst = torch.randint(0, num_nodes, (len(src),), device=pos_edges.device)
    return torch.stack([src, dst, rel], dim=1)


def train_epoch(
    model: nn.Module,
    graph: RelationalGraph,
    optimizer: torch.optim.Optimizer,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    batch_size: int = 4096,
    neg_ratio: int = 1,
) -> float:
    model.train()
    train_edges = graph.train_edges.to(DEVICE)
    total_loss = 0.0
    num_batches = 0

    perm = torch.randperm(len(train_edges))
    train_edges = train_edges[perm]

    for start in range(0, len(train_edges), batch_size):
        pos = train_edges[start : start + batch_size]
        neg = sample_negatives(pos, graph.num_nodes, neg_ratio)

        all_edges = torch.cat([pos, neg], dim=0)
        labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).to(DEVICE)

        scores = model(
            edge_index,
            edge_type,
            all_edges[:, 0],
            all_edges[:, 1],
            all_edges[:, 2],
            node_features=graph.node_features,
            num_nodes=graph.num_nodes,
        )

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    graph: RelationalGraph,
    split_edges: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_negatives: int = 50,
) -> dict:
    """
    Evaluate using Mean Reciprocal Rank (MRR) and Hits@K.
    For each positive triple we rank it against num_negatives corrupted triples.
    """
    model.eval()
    split_edges = split_edges.to(DEVICE)

    node_emb = model.encoder(
        edge_index,
        edge_type,
        node_features=graph.node_features,
        num_nodes=graph.num_nodes,
    )

    ranks = []
    for pos in split_edges:
        s, o, r = pos[0], pos[1], pos[2]

        pos_score = model.decoder(
            node_emb, s.unsqueeze(0), o.unsqueeze(0), r.unsqueeze(0)
        ).item()

        neg_dst = torch.randint(0, graph.num_nodes, (num_negatives,), device=DEVICE)
        neg_scores = model.decoder(
            node_emb, s.expand(num_negatives), neg_dst, r.expand(num_negatives)
        )

        rank = (neg_scores >= pos_score).sum().item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@3": float(np.mean(ranks <= 3)),
        "hits@10": float(np.mean(ranks <= 10)),
    }


def train(
    model: nn.Module,
    graph: RelationalGraph,
    num_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 4096,
    eval_every: int = 5,
    neg_ratio: int = 1,
) -> dict:
    """Full training loop, shared across both datasets and both model variants."""

    model = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)

    if graph.node_features is not None:
        graph.node_features = graph.node_features.to(DEVICE)

    train_e = graph.train_edges.to(DEVICE)
    edge_index = train_e[:, :2].t().contiguous()
    edge_type = train_e[:, 2]

    history = {"train_loss": [], "val_mrr": [], "val_hits10": []}

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(
            model, graph, optimizer, edge_index, edge_type, batch_size, neg_ratio
        )
        history["train_loss"].append(loss)

        if epoch % eval_every == 0:
            metrics = evaluate(model, graph, graph.val_edges, edge_index, edge_type)
            history["val_mrr"].append(metrics["mrr"])
            history["val_hits10"].append(metrics["hits@10"])
            print(
                f"Epoch {epoch:3d} | Loss {loss:.4f} | "
                f"Val MRR {metrics['mrr']:.4f} | "
                f"Hits@10 {metrics['hits@10']:.4f}"
            )
        else:
            print(f"Epoch {epoch:3d} | Loss {loss:.4f}")

    return history


def plot_history(history: dict, title: str):
    """Plot training loss and validation metrics from a history dict."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13)
    ax1.plot(history["train_loss"])
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax2.plot(history["val_mrr"], label="MRR")
    ax2.plot(history["val_hits10"], label="Hits@10")
    ax2.set_title("Validation Metrics")
    ax2.set_xlabel("Eval step")
    ax2.legend()
    plt.tight_layout()
    plt.show()
