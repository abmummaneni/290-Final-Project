# Mystery Corpus R-GCN

Relational Graph Convolutional Network trained on a corpus of mystery fiction narratives for link prediction across character, location, occupation, and organization nodes.

---

## Repository Structure

```
.
├── json_dir/                          # One JSON file per mystery story (~577 files)
├── load_mystery_graphs_updated.py     # Corpus loader — JSON → RelationalGraph
├── rgcn_model.py                      # Dataset-agnostic R-GCN model and training loop
├── detective_data.ipynb               # Training and evaluation notebook
└── checkpoints/                       # Auto-created on first run
    ├── detective_graph.pkl            # Cached graph (skip JSON re-parsing)
    ├── rgcn_scratch_detective.pt      # Trained from-scratch model weights
    └── rgcn_pyg_detective.pt          # Trained PyG model weights (optional)
```

---

## Setup

```bash
pip install torch torch-geometric
pip install "numpy==1.26.4"
```

Python 3.10+ is recommended. No GPU is required — training completes in under an hour on a modern CPU, or a few minutes on any CUDA GPU.

---

## Quick Start

1. Make sure `json_dir/` is in the project root (or update `JSON_DIR` in the notebook).
2. Open `detective_data.ipynb` and run all cells in order.
3. On the first run the loader parses all JSON files and caches the result to `checkpoints/detective_graph.pkl`. Subsequent runs load from the cache and skip parsing.
4. The trained model is saved to `checkpoints/rgcn_scratch_detective.pt`. Re-running the notebook will load the checkpoint and skip training.

---

## Data Format — `json_dir/`

Each file represents one mystery story as a typed, attributed graph with four node categories and a free-text edge list.

### Node types and their features

| Node type | Features (dim) |
|---|---|
| `characters` | gender, social_status, narrative_introduction_timing, has_alibi, present_at_crime_scene, has_motive, is_concealing_information, has_hidden_relationship, motive_type, narrative_prominence — **10 dims** |
| `occupations` | authority_level, access_level, capability_level — 3 dims, zero-padded to 10 |
| `locations` | accessibility, isolability, evidentiary_value — 3 dims, zero-padded to 10 |
| `organizations` | institutional_power, secrecy_level, financial_scale — 3 dims, zero-padded to 10 |

All feature values are floats. Missing values default to `0.0`. The full feature matrix has shape `(num_nodes, 10)` and is passed directly to the R-GCN encoder, replacing the learned embedding table used for datasets without pre-computed features.

### Edge format

```json
{
  "source": "char_0",
  "target": "loc_0",
  "relation": "resides at",
  "directed": true
}
```

`relation` is a free-text string. The loader normalises these into 8 coarse relation types (see below).

---

## Loader — `load_mystery_graphs_updated.py`

### What it does

Reads every `.json` file in a given directory and merges them into a single `RelationalGraph` object ready to pass to `train()` in `rgcn_model.py`.

**Story-level split:** entire stories are held out for validation and test, so the model is never evaluated on narratives it saw during training. Default: 80% train / 10% val / 10% test.

**Two-stage relation collapse:**

- **Stage 1** (`normalize_relation`) — ~60 regex patterns collapse hundreds of free-text relation strings into ~35 intermediate canonical forms (e.g. `"married to"`, `"kills"`, `"resides at"`).
- **Stage 2** (`coarsen_relation`) — a lookup table maps every intermediate form to one of 8 coarse relation types suited for link prediction.

### Coarse relation schema (v3)

| ID | Name | Covers | ~% of corpus |
|---|---|---|---|
| 0 | `kills` | lethal violence, death locations | 4.9% |
| 1 | `harms` | non-lethal violence, coercion, conflict | 3.9% |
| 2 | `investigates` | detection, legal process, testimony, tip-offs | 11.9% |
| 3 | `deceives` | active deception, concealment, false identity, criminal alliance | 7.2% |
| 4 | `personal_bond` | family, romance, friendship, emotional ties | 19.3% |
| 5 | `professional` | employment, mentorship, medical care | 24.2% |
| 6 | `spatial` | residence, physical location, proximity | 18.3% |
| 7 | `social` | residual narrative contact, institutional affiliation | 10.3% |

The schema was arrived at empirically over three iterations. The key design constraint is that no type should fall below ~4% (which starves the basis decomposition and DistMult decoder of training signal) or exceed ~25% (which causes one relation to dominate the loss). The current range is 3.9%–24.2%.

### Usage

```python
from load_mystery_graphs_updated import load_mystery_graphs

graph = load_mystery_graphs(
    json_dir="json_dir",      # path to folder of .json files
    val_fraction=0.1,         # fraction of stories held out for validation
    test_fraction=0.1,        # fraction of stories held out for test
    seed=42,
)
graph.summary()
```

### Observed corpus statistics

```
Stories        : 577  (train: 462, val: 57, test: 57)
Nodes          : 14,020
Relation types : 8
Train edges    : 12,866
Val edges      :  1,583
Test edges     :  1,498
Node feat dim  : 10
```

---

## Model — `rgcn_model.py`

Dataset-agnostic R-GCN implementation. Contains the encoder, decoder, training loop, and evaluation utilities. This file does not need to be modified for the mystery corpus.

**Encoder:** Multi-layer R-GCN with basis decomposition (Schlichtkrull et al., ESWC 2018). Two variants are available — a from-scratch PyTorch implementation and a PyG (`torch_geometric`) implementation.

**Decoder:** DistMult — scores a triple `(subject, relation, object)` as `h_s · diag(W_r) · h_o`.

**Evaluation:** Mean Reciprocal Rank (MRR) and Hits@K, computed by ranking each positive test triple against 50 randomly corrupted negatives.

### Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `hidden_dim` | 64 | Node embedding dimension |
| `num_layers` | 2 | R-GCN depth |
| `num_bases` | 30 | Basis matrices for decomposition |
| `dropout` | 0.1 | Applied between layers |
| `lr` | 1e-3 | Adam learning rate |
| `num_epochs` | 50 | Training epochs |
| `batch_size` | 4096 | Edges per batch |
| `neg_ratio` | 3 | Negative samples per positive edge |

`neg_ratio=3` (rather than the default 1) is used here because `kills` and `harms` together make up only ~9% of edges. Increasing the ratio ensures the model sees enough negative examples for rare relation types to learn a meaningful signal.

---

## Training Notebook — `detective_data.ipynb`

Runs the full pipeline end-to-end. Sections mirror `wikidata.ipynb` for consistency.

| Section | Contents |
|---|---|
| 0 | Imports |
| 1 | Load corpus → `RelationalGraph` (cached after first run) |
| 2 | Train or load from-scratch R-GCN |
| 3 | Test set evaluation (MRR, Hits@1/3/10) |
| 4 | Train or load PyG R-GCN (`TRAIN_PYG = False` by default) |
| 5 | Scratch vs PyG comparison plot |
| 6 | Link prediction inference (`predict_object`, `predict_relation`) |
| 7 | Example predictions using characters from the corpus |
| 8 | Graph exploration utilities (`entity_stats`, relation distribution table) |
| 9 | Manual checkpoint management (force save / reload / graph rebuild) |

### Inference functions

**`predict_object(subject, relation, top_k=10)`** — given a node name and relation type, returns the top-k most likely target nodes.

```python
predict_object("Blythe Connor", "kills")
predict_object("House", "spatial")
```

**`predict_relation(subject, obj, top_k=5)`** — given two node names, returns the most likely coarse relation types between them.

```python
predict_relation("Blythe Connor", "Sam")
```

Node names must match exactly as stored in `det_graph.node_labels` (character names, location names, occupation names, organization names as they appear in the JSON files). Use the partial-match output to find the correct spelling if a lookup fails.

### Updating the JSON directory

If your JSON files are not in `json_dir/` at the project root, update this line in Section 0 of the notebook:

```python
JSON_DIR = os.path.join(PROJECT_DIR, "json_dir")
```

After changing the loader or the JSON files, delete `checkpoints/detective_graph.pkl` to force a rebuild, or use the commented cell at the bottom of Section 9.

---

## Reference

Schlichtkrull, M., Kipf, T. N., Bloem, P., Van Den Berg, R., Titov, I., & Welling, M. (2018). *Modeling Relational Data with Graph Convolutional Networks.* ESWC 2018. https://arxiv.org/abs/1703.06103
