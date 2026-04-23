# Variational Graph Autoencoder using R-GCN


import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .rgcn_model import RGCNScratch, DistMultDecoder, sample_negatives
except ImportError:
    from rgcn_model import RGCNScratch, DistMultDecoder, sample_negatives

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


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

    def fit(self, graph, epochs, lr=0.01, neg_ratio=1, kl_beta=1.0):
        self.to(DEVICE)
        super().train(True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        total_loss = 0.0
        node_features = graph.node_features.to(DEVICE) if graph.node_features is not None else None
        train_edges = graph.train_edges.to(DEVICE)

        for _ in tqdm(range(epochs)):
            train_edges = train_edges[torch.randperm(len(train_edges), device=DEVICE)]
            neg_edges = sample_negatives(train_edges, graph.num_nodes, neg_ratio)
            all_edges = torch.cat([train_edges, neg_edges], dim=0)
            labels = torch.cat(
                [
                    torch.ones(len(train_edges), device=DEVICE),
                    torch.zeros(len(neg_edges), device=DEVICE),
                ]
            )

            edge_index = train_edges[:, :2].t().contiguous()
            edge_type = train_edges[:, 2]
            z, mu, log_var = self(
                edge_index,
                edge_type,
                node_features=node_features,
                num_nodes=graph.num_nodes,
            )

            logits = self.decoder(z, all_edges[:, 0], all_edges[:, 1], all_edges[:, 2])
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
            kl_loss = -0.5 * torch.sum(
                1 + log_var - mu.pow(2) - log_var.exp(),
                dim=1,
            ).mean()
            loss = bce_loss + kl_beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / epochs
