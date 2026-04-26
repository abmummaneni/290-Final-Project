"""
rgcn_model.py
=============
Dataset-agnostic R-GCN implementation.

Contains everything needed to build and train a Relational Graph Convolutional
Network for link prediction — with no dependency on any specific dataset.

Usage
-----
    from model.rgcn_model import (
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
matplotlib.rcParams['figure.dpi'] = 120

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
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
    name: str                          # human-readable name e.g. "tkgl-smallpedia"
    num_nodes: int                     # total number of nodes
    num_relations: int                 # total number of relation types

    # Edge splits — each is a LongTensor of shape (E_split, 3): [src, dst, rel]
    train_edges: torch.Tensor
    val_edges:   torch.Tensor
    test_edges:  torch.Tensor

    # Optional: pre-computed node feature matrix (num_nodes, feat_dim)
    # If None the model will use a learned embedding table instead
    node_features: Optional[torch.Tensor] = None

    # Optional: human-readable label maps for interpretability
    node_labels: dict = field(default_factory=dict)  # int → string
    rel_labels:  dict = field(default_factory=dict)  # int → string

    # Optional: node classification labels and masks
    # class_labels: LongTensor of shape (num_nodes,) with class indices (-1 = unlabeled)
    # train_mask / val_mask / test_mask: BoolTensor of shape (num_nodes,)
    class_labels: Optional[torch.Tensor] = None
    num_classes: int = 0
    class_names: dict = field(default_factory=dict)   # int → string
    train_mask: Optional[torch.Tensor] = None
    val_mask:   Optional[torch.Tensor] = None
    test_mask:  Optional[torch.Tensor] = None

    # Optional: per-edge boolean mask marking "crime" edges (kills, assaults,
    # theft, etc.) that should be hidden at evaluation time in the detective
    # scenario.  One BoolTensor per split, shape (E_split,).
    train_crime_mask: Optional[torch.Tensor] = None
    val_crime_mask:   Optional[torch.Tensor] = None
    test_crime_mask:  Optional[torch.Tensor] = None

    def summary(self):
        print(f"RelationalGraph: {self.name}")
        print(f"  Nodes         : {self.num_nodes:,}")
        print(f"  Relation types: {self.num_relations:,}")
        print(f"  Train edges   : {len(self.train_edges):,}")
        print(f"  Val edges     : {len(self.val_edges):,}")
        print(f"  Test edges    : {len(self.test_edges):,}")
        print(f"  Node features : {'yes' if self.node_features is not None else 'no (will use learned embeddings)'}")
        if self.class_labels is not None:
            labeled = (self.class_labels >= 0).sum().item()
            print(f"  Class labels  : {labeled:,} labeled nodes, {self.num_classes} classes")
            if self.train_mask is not None:
                print(f"  Train/Val/Test: {self.train_mask.sum().item()}/{self.val_mask.sum().item()}/{self.test_mask.sum().item()} labeled nodes")
        if self.train_crime_mask is not None:
            n_crime = self.train_crime_mask.sum().item() + self.val_crime_mask.sum().item() + self.test_crime_mask.sum().item()
            print(f"  Crime edges   : {n_crime:,} (masked at eval time)")


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
    def __init__(self, in_dim: int, out_dim: int,
                 num_relations: int, num_bases: int):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_relations = num_relations
        self.num_bases     = num_bases

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

    def forward(self,
                x: torch.Tensor,          # (num_nodes, in_dim)
                edge_index: torch.Tensor, # (2, E)  — row 0 = src, row 1 = dst
                edge_type: torch.Tensor   # (E,)    — relation index per edge
               ) -> torch.Tensor:         # (num_nodes, out_dim)

        num_nodes = x.size(0)

        # Step 1: compute W_r for each relation via basis decomposition
        # W_r = sum_b a_rb * V_b  →  shape: (num_relations, in_dim, out_dim)
        W = torch.einsum('rb,bio->rio', self.coefficients, self.bases)

        # Step 2: aggregate neighbour messages for each relation
        src, dst = edge_index[0], edge_index[1]

        agg = torch.zeros(num_nodes, self.out_dim, device=x.device)

        # Memory-efficient aggregation: loop over relations
        for r in range(self.num_relations):
            edge_mask = (edge_type == r)
            if not edge_mask.any():
                continue

            r_src = src[edge_mask]
            r_dst = dst[edge_mask]

            # Per-relation degree normalisation: c_{i,r} = |N_i^r|  (Eq. 2)
            deg_r = torch.zeros(num_nodes, device=x.device)
            deg_r.scatter_add_(0, r_dst, torch.ones(len(r_dst), device=x.device))
            deg_r_inv = 1.0 / deg_r.clamp(min=1.0)

            # Transform source node features with the relation-specific weight
            msg = x[r_src] @ W[r]
            msg = msg * deg_r_inv[r_dst].unsqueeze(1)
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
    def __init__(self,
                 num_nodes: int,
                 num_relations: int,
                 hidden_dim: int   = 64,
                 num_layers: int   = 2,
                 num_bases: int    = 30,
                 feat_dim: Optional[int] = None,
                 dropout: float    = 0.1):
        super().__init__()
        self.dropout = dropout

        if feat_dim is None:
            self.embedding = nn.Embedding(num_nodes, hidden_dim)
            in_dim = hidden_dim
        else:
            self.embedding = None
            self.input_proj = nn.Linear(feat_dim, hidden_dim)
            in_dim = hidden_dim

        self.layers = nn.ModuleList([
            RGCNLayerScratch(in_dim, hidden_dim, num_relations, num_bases)
            for _ in range(num_layers)
        ])

    def forward(self,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor,
                node_features: Optional[torch.Tensor] = None,
                num_nodes: Optional[int] = None
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

    def forward(self,
                node_emb: torch.Tensor,
                src: torch.Tensor,
                dst: torch.Tensor,
                rel: torch.Tensor
               ) -> torch.Tensor:

        h_s = node_emb[src]
        h_o = node_emb[dst]
        r   = self.relation_emb(rel)
        scores = (h_s * r * h_o).sum(dim=-1)
        return scores


class NodeClassifier(nn.Module):
    """
    Entity classification decoder (Section 3 of the paper).

    Applies softmax(W @ h_i) to produce class probabilities.
    """
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        return self.linear(node_emb)  # raw logits; softmax applied in loss


class RGCNLinkPredictor(nn.Module):
    """
    Full model: R-GCN encoder (from scratch) + DistMult decoder.
    This is the complete architecture from Section 4 of the paper.
    """
    def __init__(self, num_nodes, num_relations, hidden_dim=64,
                 num_layers=2, num_bases=30, feat_dim=None, dropout=0.1):
        super().__init__()
        self.encoder = RGCNScratch(
            num_nodes, num_relations, hidden_dim,
            num_layers, num_bases, feat_dim, dropout
        )
        self.decoder = DistMultDecoder(num_relations, hidden_dim)

    def forward(self,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor,
                src: torch.Tensor,
                dst: torch.Tensor,
                rel: torch.Tensor,
                node_features: Optional[torch.Tensor] = None,
                num_nodes: Optional[int] = None
               ) -> torch.Tensor:

        node_emb = self.encoder(edge_index, edge_type, node_features, num_nodes)
        scores   = self.decoder(node_emb, src, dst, rel)
        return scores


class RGCNMultiTask(nn.Module):
    """
    R-GCN encoder + DistMult link decoder + node classification head.

    Supports joint training for link prediction and villain prediction.
    """
    def __init__(self, num_nodes, num_relations, num_classes, hidden_dim=64,
                 num_layers=2, num_bases=30, feat_dim=None, dropout=0.1):
        super().__init__()
        self.encoder    = RGCNScratch(
            num_nodes, num_relations, hidden_dim,
            num_layers, num_bases, feat_dim, dropout
        )
        self.link_decoder = DistMultDecoder(num_relations, hidden_dim)
        self.node_classifier = NodeClassifier(hidden_dim, num_classes)

    def encode(self, edge_index, edge_type, node_features=None, num_nodes=None):
        return self.encoder(edge_index, edge_type, node_features, num_nodes)

    def forward(self, edge_index, edge_type, src, dst, rel,
                node_features=None, num_nodes=None):
        node_emb = self.encode(edge_index, edge_type, node_features, num_nodes)
        link_scores = self.link_decoder(node_emb, src, dst, rel)
        class_logits = self.node_classifier(node_emb)
        return link_scores, class_logits


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
        def __init__(self, num_nodes, num_relations, hidden_dim=64,
                     num_layers=2, num_bases=30, feat_dim=None, dropout=0.1):
            super().__init__()
            self.dropout = dropout

            if feat_dim is None:
                self.embedding = nn.Embedding(num_nodes, hidden_dim)
                in_dim = hidden_dim
            else:
                self.embedding = None
                self.input_proj = nn.Linear(feat_dim, hidden_dim)
                in_dim = hidden_dim

            self.layers = nn.ModuleList([
                RGCNConv(in_dim, hidden_dim,
                         num_relations=num_relations,
                         num_bases=num_bases)
                for _ in range(num_layers)
            ])

        def forward(self, edge_index, edge_type,
                    node_features=None, num_nodes=None):
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
        def __init__(self, num_nodes, num_relations, hidden_dim=64,
                     num_layers=2, num_bases=30, feat_dim=None, dropout=0.1):
            super().__init__()
            self.encoder = RGCNPyG(
                num_nodes, num_relations, hidden_dim,
                num_layers, num_bases, feat_dim, dropout
            )
            self.decoder = DistMultDecoder(num_relations, hidden_dim)

        def forward(self, edge_index, edge_type, src, dst, rel,
                    node_features=None, num_nodes=None):
            node_emb = self.encoder(edge_index, edge_type, node_features, num_nodes)
            return self.decoder(node_emb, src, dst, rel)

except ImportError:
    PYG_AVAILABLE = False
    RGCNPyG = None
    RGCNLinkPredictorPyG = None


# ---------------------------------------------------------------------------
# 5. Training & Evaluation Utilities
# ---------------------------------------------------------------------------

def sample_negatives(pos_edges: torch.Tensor,
                     num_nodes: int,
                     neg_ratio: int = 1) -> torch.Tensor:
    """
    For each positive edge (s, r, o), sample neg_ratio negative edges
    by randomly corrupting either the subject or object (50/50).

    Returns a tensor of shape (E * neg_ratio, 3).
    """
    n = len(pos_edges) * neg_ratio
    src = pos_edges[:, 0].repeat(neg_ratio)
    dst = pos_edges[:, 1].repeat(neg_ratio)
    rel = pos_edges[:, 2].repeat(neg_ratio)

    # For each negative sample, randomly corrupt subject or object
    corrupt_src = torch.rand(n, device=pos_edges.device) < 0.5
    random_nodes = torch.randint(0, num_nodes, (n,), device=pos_edges.device)
    src = torch.where(corrupt_src, random_nodes, src)
    dst = torch.where(~corrupt_src, random_nodes, dst)

    return torch.stack([src, dst, rel], dim=1)


def train_epoch(model: nn.Module,
                graph: RelationalGraph,
                optimizer: torch.optim.Optimizer,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor,
                batch_size: int = 4096,
                neg_ratio: int = 1
               ) -> float:
    model.train()
    train_edges = graph.train_edges.to(DEVICE)
    total_loss  = 0.0
    num_batches = 0

    perm = torch.randperm(len(train_edges))
    train_edges = train_edges[perm]

    for start in range(0, len(train_edges), batch_size):
        pos = train_edges[start : start + batch_size]
        neg = sample_negatives(pos, graph.num_nodes, neg_ratio)

        all_edges = torch.cat([pos, neg], dim=0)
        labels = torch.cat([
            torch.ones(len(pos)),
            torch.zeros(len(neg))
        ]).to(DEVICE)

        scores = model(
            edge_index, edge_type,
            all_edges[:, 0], all_edges[:, 1], all_edges[:, 2],
            node_features=graph.node_features,
            num_nodes=graph.num_nodes
        )

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model: nn.Module,
             graph: RelationalGraph,
             split_edges: torch.Tensor,
             edge_index: torch.Tensor,
             edge_type: torch.Tensor,
             num_negatives: int = 50
            ) -> dict:
    """
    Evaluate using Mean Reciprocal Rank (MRR) and Hits@K.
    For each positive triple we rank it against num_negatives corrupted triples.
    """
    model.eval()
    split_edges = split_edges.to(DEVICE)

    node_emb = model.encoder(
        edge_index, edge_type,
        node_features=graph.node_features,
        num_nodes=graph.num_nodes
    )

    # Support both RGCNLinkPredictor (model.decoder) and RGCNMultiTask (model.link_decoder)
    decoder = getattr(model, 'link_decoder', None) or model.decoder

    ranks = []
    for pos in split_edges:
        s, o, r = pos[0], pos[1], pos[2]

        pos_score = decoder(
            node_emb,
            s.unsqueeze(0), o.unsqueeze(0), r.unsqueeze(0)
        ).item()

        neg_dst = torch.randint(0, graph.num_nodes, (num_negatives,), device=DEVICE)
        neg_scores = decoder(
            node_emb,
            s.expand(num_negatives), neg_dst, r.expand(num_negatives)
        )

        rank = (neg_scores >= pos_score).sum().item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        'mrr':     float(np.mean(1.0 / ranks)),
        'hits@1':  float(np.mean(ranks <= 1)),
        'hits@3':  float(np.mean(ranks <= 3)),
        'hits@10': float(np.mean(ranks <= 10)),
    }


def train(model: nn.Module,
          graph: RelationalGraph,
          num_epochs: int   = 50,
          lr: float         = 1e-3,
          batch_size: int   = 4096,
          eval_every: int   = 5,
          neg_ratio: int    = 1
         ) -> dict:
    """Full training loop, shared across both datasets and both model variants."""

    model = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)

    if graph.node_features is not None:
        graph.node_features = graph.node_features.to(DEVICE)

    train_e    = graph.train_edges.to(DEVICE)
    edge_index = train_e[:, :2].t().contiguous()
    edge_type  = train_e[:, 2]

    history = {'train_loss': [], 'val_mrr': [], 'val_hits10': []}

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, graph, optimizer, edge_index, edge_type,
                           batch_size, neg_ratio)
        history['train_loss'].append(loss)

        if epoch % eval_every == 0:
            metrics = evaluate(model, graph, graph.val_edges, edge_index, edge_type)
            history['val_mrr'].append(metrics['mrr'])
            history['val_hits10'].append(metrics['hits@10'])
            print(f"Epoch {epoch:3d} | Loss {loss:.4f} | "
                  f"Val MRR {metrics['mrr']:.4f} | "
                  f"Hits@10 {metrics['hits@10']:.4f}")
        else:
            print(f"Epoch {epoch:3d} | Loss {loss:.4f}")

    return history


def plot_history(history: dict, title: str):
    """Plot training loss and validation metrics from a history dict."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13)
    ax1.plot(history['train_loss'])
    ax1.set_title('Training Loss'); ax1.set_xlabel('Epoch')
    ax2.plot(history['val_mrr'],    label='MRR')
    ax2.plot(history['val_hits10'], label='Hits@10')
    ax2.set_title('Validation Metrics'); ax2.set_xlabel('Eval step')
    ax2.legend()
    plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# 6. Edge Masking for Detective Evaluation
# ---------------------------------------------------------------------------

def build_detective_graph(graph: RelationalGraph) -> tuple:
    """
    Build edge_index and edge_type for the detective scenario:
    uses ALL training edges for message passing, but removes crime edges
    so the model cannot see who committed the crime.

    Returns
    -------
    (edge_index, edge_type) with crime edges removed
    """
    train_e = graph.train_edges.to(DEVICE)
    if graph.train_crime_mask is not None:
        keep = ~graph.train_crime_mask.to(DEVICE)
        train_e = train_e[keep]
    edge_index = train_e[:, :2].t().contiguous()
    edge_type  = train_e[:, 2]
    return edge_index, edge_type


# ---------------------------------------------------------------------------
# 6. Multi-task Training (Link Prediction + Node Classification)
# ---------------------------------------------------------------------------

def train_multitask(
    model: RGCNMultiTask,
    graph: RelationalGraph,
    num_epochs: int       = 50,
    lr: float             = 1e-3,
    batch_size: int       = 4096,
    eval_every: int       = 5,
    neg_ratio: int        = 1,
    cls_weight: float     = 1.0,
    link_weight: float    = 1.0,
    class_balance: bool   = True,
) -> dict:
    """
    Joint training loop: link prediction (DistMult) + node classification.

    Parameters
    ----------
    cls_weight    : weight for classification loss in the combined objective
    link_weight   : weight for link prediction loss in the combined objective
    class_balance : if True, weight the classification loss by inverse class
                    frequency so minority classes (Villain, Witness) get
                    proportionally more gradient signal
    """
    import time

    model = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)

    if graph.node_features is not None:
        graph.node_features = graph.node_features.to(DEVICE)

    class_labels = graph.class_labels.to(DEVICE)
    train_mask   = graph.train_mask.to(DEVICE)
    val_mask     = graph.val_mask.to(DEVICE)

    # Compute per-class weights from training label distribution
    if class_balance:
        train_labels = graph.class_labels[graph.train_mask]
        counts = torch.bincount(train_labels, minlength=graph.num_classes).float()
        inv_freq = 1.0 / counts.clamp(min=1.0)
        class_weights = (inv_freq / inv_freq.sum()) * graph.num_classes
        class_weights = class_weights.to(DEVICE)
        print(f"Class weights: {', '.join(f'{graph.class_names[i]}={class_weights[i]:.2f}' for i in range(graph.num_classes))}")
    else:
        class_weights = None

    train_e    = graph.train_edges.to(DEVICE)
    edge_index = train_e[:, :2].t().contiguous()
    edge_type  = train_e[:, 2]

    history = {
        'train_loss': [], 'link_loss': [], 'cls_loss': [],
        'val_mrr': [], 'val_hits10': [],
        'val_cls_acc': [], 'train_cls_acc': [],
    }

    epoch_times = []
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t_epoch = time.time()
        model.train()
        total_link_loss = 0.0
        num_batches = 0

        # Shuffle training edges
        perm = torch.randperm(len(train_e))
        shuffled_edges = train_e[perm]

        for start in range(0, len(shuffled_edges), batch_size):
            pos = shuffled_edges[start : start + batch_size]
            neg = sample_negatives(pos, graph.num_nodes, neg_ratio)

            all_edges = torch.cat([pos, neg], dim=0)
            labels = torch.cat([
                torch.ones(len(pos)),
                torch.zeros(len(neg))
            ]).to(DEVICE)

            link_scores, class_logits = model(
                edge_index, edge_type,
                all_edges[:, 0], all_edges[:, 1], all_edges[:, 2],
                node_features=graph.node_features,
                num_nodes=graph.num_nodes,
            )

            # Link prediction loss
            loss_link = F.binary_cross_entropy_with_logits(link_scores, labels)

            # Node classification loss (only on train-split characters)
            loss_cls = F.cross_entropy(
                class_logits[train_mask],
                class_labels[train_mask],
                weight=class_weights,
            )

            loss = link_weight * loss_link + cls_weight * loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_link_loss += loss_link.item()
            num_batches += 1

        avg_link = total_link_loss / num_batches
        history['link_loss'].append(avg_link)
        history['cls_loss'].append(loss_cls.item())
        history['train_loss'].append(avg_link + loss_cls.item())

        elapsed = time.time() - t_epoch
        epoch_times.append(elapsed)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        remaining = avg_epoch * (num_epochs - epoch)

        # Training classification accuracy
        model.eval()
        with torch.no_grad():
            _, logits = model(
                edge_index, edge_type,
                train_e[0:1, 0], train_e[0:1, 1], train_e[0:1, 2],
                node_features=graph.node_features,
                num_nodes=graph.num_nodes,
            )
            train_acc = (logits[train_mask].argmax(1) == class_labels[train_mask]).float().mean().item()
            history['train_cls_acc'].append(train_acc)

        if epoch % eval_every == 0:
            metrics = evaluate(model, graph, graph.val_edges, edge_index, edge_type)
            val_acc = evaluate_classification(model, graph, val_mask, edge_index, edge_type)
            history['val_mrr'].append(metrics['mrr'])
            history['val_hits10'].append(metrics['hits@10'])
            history['val_cls_acc'].append(val_acc)
            print(f"Epoch {epoch:3d} | Link {avg_link:.4f} | Cls {loss_cls.item():.4f} | "
                  f"Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f} | "
                  f"MRR {metrics['mrr']:.4f} | H@10 {metrics['hits@10']:.4f} | "
                  f"{elapsed:.1f}s | ETA {remaining:.0f}s")
        else:
            print(f"Epoch {epoch:3d} | Link {avg_link:.4f} | Cls {loss_cls.item():.4f} | "
                  f"Train Acc {train_acc:.3f} | {elapsed:.1f}s | ETA {remaining:.0f}s")

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    return history


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    graph: RelationalGraph,
    mask: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
) -> float:
    """Compute classification accuracy on masked nodes."""
    model.eval()
    mask = mask.to(DEVICE)
    class_labels = graph.class_labels.to(DEVICE)

    node_emb = model.encoder(
        edge_index, edge_type,
        node_features=graph.node_features,
        num_nodes=graph.num_nodes,
    )
    logits = model.node_classifier(node_emb)
    preds  = logits[mask].argmax(dim=1)
    acc    = (preds == class_labels[mask]).float().mean().item()
    return acc


def plot_multitask_history(history: dict, title: str):
    """Plot training curves for multi-task model."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(title, fontsize=13)

    axes[0].plot(history['link_loss'], label='Link')
    axes[0].plot(history['cls_loss'], label='Classification')
    axes[0].set_title('Training Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['train_cls_acc'], label='Train')
    if history['val_cls_acc']:
        eval_x = [i * (len(history['train_cls_acc']) // max(1, len(history['val_cls_acc'])))
                   for i in range(1, len(history['val_cls_acc']) + 1)]
        axes[1].plot(eval_x, history['val_cls_acc'], label='Val')
    axes[1].set_title('Classification Accuracy'); axes[1].set_xlabel('Epoch')
    axes[1].legend()

    if history['val_mrr']:
        axes[2].plot(history['val_mrr'], label='MRR')
        axes[2].plot(history['val_hits10'], label='Hits@10')
    axes[2].set_title('Link Prediction (Val)'); axes[2].set_xlabel('Eval step')
    axes[2].legend()

    plt.tight_layout(); plt.show()
