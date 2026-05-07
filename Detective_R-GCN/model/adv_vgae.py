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




# JUST FOR linbk prediciton
class VGAE(nn.Module):
        def __init__(self, num_nodes, hidden_dim, num_relations, prior_latent_dist = None,latent_dim = None, feat_dim = None):
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
                self.log_sigma_head = nn.Linear(hidden_dim, latent_dim)
                self.decoder = DistMultDecoder(num_relations, latent_dim)
                self.prior_latent_dist = prior_latent_dist
                self.discriminator = nn.Sequential(
                        nn.Linear(latent_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                )





        def forward(self, edge_index, edge_type, node_features=None, num_nodes=None):
                h = self.encoder(edge_index, edge_type, node_features, num_nodes)
               
                mu =self.mu_head(h)
                log_sig = self.log_sigma_head(h)
                z = (
                mu + torch.exp(log_sig) * torch.randn_like(log_sig)
                if self.training
                else mu
                )
                return z, mu, log_sig
        

        def fit(
                self,
                graph,
                epochs,
                batch_size,
                neg_ratio = 3,
                lr=1e-3, 
                weight_decay=1e-5

        ):
                
                self.to(DEVICE)
                super().train(True)
                optimizer = torch.optim.AdamW(
                list(self.encoder.parameters())
                + list(self.mu_head.parameters())
                + list(self.log_sigma_head.parameters())
                + list(self.decoder.parameters()),
                lr=lr,
                weight_decay=weight_decay,
                )

                disc_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                )

                history = {
                "loss": [],
                "bce_loss": [],
                "disc_loss": [],
                "adv_loss": [],
                }
                
                node_features = graph.node_features.to(DEVICE) if graph.node_features is not None else None
                
                #effective R = R

                train_edges = graph.train_edges.to(DEVICE) #[s, d, r]
                if batch_size == -1:
                        batch_size = len(train_edges)
                edge_index = train_edges[:, :2].t().contiguous()
                edge_type = train_edges[:, 2]

                effective_num_relations = graph.num_relations

                for epoch in tqdm(range(0, epochs)):
                        epoch_loss = 0.0
                        num_batches = 0
                        shuffled_edges = train_edges[torch.randperm(len(train_edges), device=DEVICE)]

                        for batch in range(0, len(shuffled_edges), batch_size):
                                                pos_edges = shuffled_edges[batch : batch + batch_size]
                                                neg_edges = sample_vgae_negatives(
                                                        pos_edges,
                                                        graph.num_nodes,
                                                        neg_ratio,
                                                        known_edge_ids=None,
                                                        num_relations=effective_num_relations,
                                                )
                                                all_edges = torch.cat([pos_edges, neg_edges], dim=0)
                                                labels = torch.cat(
                                                [
                                                        torch.ones(len(pos_edges), device=DEVICE),
                                                        torch.zeros(len(neg_edges), device=DEVICE),
                                                ]
                                                )

                                                z, mu, log_sig = self(
                                                        edge_index,
                                                        edge_type,
                                                        node_features=node_features, #none
                                                        num_nodes=graph.num_nodes,
                                                )
                                                disc_loss = 0
                                                if self.prior_latent_dist is not None:
                                                ##### Adv Reg
                                                        prior_logits = self.discriminator(self.prior_latent_dist[:(len(z.detach()))].to(z.device)).squeeze(-1)
                                                        data_logits = self.discriminator(z.detach()).squeeze(-1)

                                                        prior_labels = torch.ones_like(prior_logits)
                                                        data_labels = torch.zeros_like(data_logits)
                                                        disc_loss = (F.binary_cross_entropy_with_logits(prior_logits, prior_labels) + F.binary_cross_entropy_with_logits(data_logits, data_labels) )

                                                else:
                                                        prior_logits = self.discriminator(torch.randn_like(z.detach())).squeeze(-1)
                                                        data_logits = self.discriminator(z.detach()).squeeze(-1)

                                                        prior_labels = torch.ones_like(prior_logits)
                                                        data_labels = torch.zeros_like(data_logits)
                                                        disc_loss = (F.binary_cross_entropy_with_logits(prior_logits, prior_labels) + F.binary_cross_entropy_with_logits(data_logits, data_labels) )
                                                disc_optimizer.zero_grad()
                                                disc_loss.backward()
                                                disc_optimizer.step()




                                                z, mu, log_sig = self(
                                                        edge_index,
                                                        edge_type,
                                                        node_features=node_features, #none
                                                        num_nodes=graph.num_nodes,
                                                )

                                                adv_logits = self.discriminator(z).squeeze(-1)
                                                adv_loss = F.binary_cross_entropy_with_logits(
                                                adv_logits,
                                                torch.ones_like(adv_logits)
                                                )


                                                #####

                                                logits = self.decoder(z, all_edges[:, 0], all_edges[:, 1], all_edges[:, 2])
                                                bce_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=labels.new_tensor(neg_ratio))
                                                adv_weight=.05
                                                loss = bce_loss + adv_loss * adv_weight

                                                
                                                optimizer.zero_grad()
                                                loss.backward()
                                                optimizer.step()
                                                history["loss"].append(loss.item())
                                                history["bce_loss"].append(bce_loss.item())
                                                history["disc_loss"].append(disc_loss.item())
                                                history["adv_loss"].append(adv_loss.item())

                return history