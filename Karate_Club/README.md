# Phase 1 — Karate Club (KC)

## Dataset Overview

**Zachary's Karate Club** (1977) is a social network of 34 members of a university karate club,
observed over 2 years. A conflict between the instructor (node 0) and the club president (node 33)
caused the club to split into two factions. The ground-truth community labels are known, making
this an ideal testbed for unsupervised embedding methods.

| Property | Value |
|----------|-------|
| Nodes | 34 (club members) |
| Edges | 78 (undirected social interactions) |
| Labels | 2 (faction after the split) |
| Edge weights | Available (count of shared social contexts) |

## Data Representation

- **Adjacency matrix** $A \in \mathbb{R}^{34 \times 34}$, symmetric (undirected graph)
- **Degree matrix** $D = \text{diag}(d_1, \ldots, d_{34})$ where $d_i = \sum_j A_{ij}$
- **Graph Laplacian** $L = D - A$
- **Normalized Laplacian** $\mathcal{L} = I - D^{-1/2} A D^{-1/2}$

PyTorch Geometric stores edges in COO format (`edge_index` shape `[2, 156]` — 78 edges x 2 directions).

## Notebooks

| Notebook | Description |
|----------|-------------|
| `00_data_exploration.ipynb` | Load data, inspect structure, visualize graph and basic stats |
| `01_laplacian_eigenmaps.ipynb` | Classical spectral embedding baseline |
| `02_autoencoder.ipynb` | Graph autoencoder, reconstruction, comparison to spectral baseline |

## Research Questions

1. **Reconstruction quality** — How well does an AE reconstruct the adjacency structure?
2. **Community recovery** — Do learned embeddings separate the two factions unsupervised?
3. **Laplacian eigenmap baseline** — How do spectral embeddings compare to learned ones?
4. **Weighted vs unweighted** — Does using edge weights change reconstruction or embedding geometry?
5. **Link prediction** — Can the decoder predict held-out masked edges?
6. **Latent dimension sensitivity** — How does performance degrade as latent dim decreases?
