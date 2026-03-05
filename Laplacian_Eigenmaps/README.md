# Laplacian Eigenmaps Experiments

This folder contains two notebooks:

- `laplacian_eigenmaps_graph_tests.ipynb`
- `laplacian_eigenmaps_ogb.ipynb`

Both are focused on testing graph encodings produced by Laplacian Eigenmaps, but at different scales.

## 1) `laplacian_eigenmaps_graph_tests.ipynb`

Purpose: sanity-check Laplacian Eigenmaps on small synthetic graphs where geometry is interpretable.

### What it tests

- Implements normalized Laplacian Eigenmaps.
- Runs embeddings on:
  - path graph
  - cycle graph
  - star graph
  - two-community stochastic block model (SBM)
- Visualizes 2D embeddings for each graph.
- Compares SBM spring layout vs SBM Laplacian Eigenmaps layout.
- Computes within-community vs between-community embedding distances for SBM.

### What the tests mean

- Path/cycle/star checks validate that the encoder preserves expected structural patterns:
  - path should map to a smooth 1D-like progression
  - cycle should reflect circular symmetry
  - star should isolate hub/leaf role differences
- SBM checks whether community structure separates in spectral space.
- The SBM separation ratio (`between / average within`) is a simple quantitative signal:
  - larger than 1 suggests communities are more separated than internally spread.

Use this notebook to verify correctness and intuition before moving to real benchmark data.

## 2) `laplacian_eigenmaps_ogb.ipynb`

Purpose: evaluate Laplacian Eigenmaps features on real downstream graph prediction tasks from OGB.

### What it tests

- Loads two OGB molecular graph datasets: `ogbg-molbace` (1513 molecules) and `ogbg-molbbbp` (2039 molecules).
- Each sample is one molecule as a graph with keys `edge_index`, `edge_feat`, `node_feat`, `num_nodes`, plus one graph label `y` (binary: `0` or `1`).
- `node_feat` has 9 atom features per node and `edge_feat` has 3 bond features per edge.
- Uses OGB scaffold splits (`train`, `valid`, `test`), which separate molecules by core scaffold for more realistic generalization.
- Classification task:
  - `ogbg-molbace`: predict BACE activity (active vs inactive).
  - `ogbg-molbbbp`: predict blood-brain-barrier permeability (yes vs no).
- For each graph, builds the normalized Laplacian, computes the smallest non-trivial eigenpairs, and forms a fixed-length graph vector.
- With `n_components=8`, each graph becomes a 24-dim vector: 8 eigenvalues + 8 mean absolute eigenvector coordinates + 8 coordinate standard deviations.
- Trains logistic regression on train split features and evaluates on validation/test with ROC-AUC (primary) and accuracy (secondary).
- Reports runtime per dataset and plots validation vs test ROC-AUC.

### What the tests mean

- This is a baseline pipeline: unsupervised spectral graph encoding + simple linear classifier.
- ROC-AUC is the primary metric for these binary tasks:
  - higher ROC-AUC means better ranking of positive vs negative graphs.
- Accuracy is threshold-dependent and secondary.
- Performance indicates how much predictive information is captured by structure-only spectral features.

## Running the notebooks

From the repository root:

```bash
conda env create -f environment.yml
conda activate 290-final-project
```

Open and run interactively:

```bash
jupyter notebook Laplacian_Eigenmaps
```

Or execute both notebooks from the command line:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
jupyter nbconvert --to notebook --execute --inplace \
  Laplacian_Eigenmaps/laplacian_eigenmaps_graph_tests.ipynb

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
jupyter nbconvert --to notebook --execute --inplace \
  Laplacian_Eigenmaps/laplacian_eigenmaps_ogb.ipynb
```

Notes:

- OGB datasets download to `Laplacian_Eigenmaps/ogb_data/` on first run.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is set for current OGB/PyTorch cache compatibility.
