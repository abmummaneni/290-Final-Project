# R-GCN: Relational Graph Convolutional Network

Implementation of [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (Schlichtkrull et al., 2017).

## Setup

```bash
pip install py-tgb torch torch_geometric networkx matplotlib pandas numpy requests
```

## Running the notebook

Open `rgcn_cuda.ipynb` in Jupyter or Google Colab and run all cells top to bottom.

**First run** downloads the dataset and trains the model (~30–60 min on GPU). Checkpoints are saved automatically — subsequent runs load the trained weights and skip training entirely.

---

## Checkpoint directory

All saved files go to a single folder controlled by `CHECKPOINT_DIR` (set at the top of cell 3). Change this one variable to match your setup:

**Google Colab + Drive (recommended for sharing with teammates):**
```python
from google.colab import drive
drive.mount("/content/drive")
import os
os.environ["RGCN_CHECKPOINT_DIR"] = "/content/drive/MyDrive/rgcn_checkpoints"
```
Run this *before* cell 3, or edit cell 3 directly.

**Local machine:** the default `./checkpoints/` works with no changes.

---

## Sharing checkpoints with teammates

After one person trains the model, share the `CHECKPOINT_DIR` folder. It contains:

| File | Contents |
|---|---|
| `rgcn_scratch_wikidata.pt` | Trained from-scratch model weights |
| `rgcn_pyg_wikidata.pt` | Trained PyG model weights |
| `wiki_graph.pkl` | Processed graph (edges, splits, label maps) |
| `wikidata_labels_cache.json` | English labels for all Wikidata Q/P codes |

Teammates who have this folder set as their `CHECKPOINT_DIR` will skip training and label fetching completely.

**If you already have `wikidata_labels_cache.json`**, copy it into `CHECKPOINT_DIR` before running the notebook:
```python
import shutil
shutil.copy("wikidata_labels_cache.json", os.path.join(CHECKPOINT_DIR, "wikidata_labels_cache.json"))
```

---

## TGB data directory

The TGB dataset location is found automatically. If auto-detection fails, set it manually before the data loading cell:
```python
import os
os.environ["TGB_DATA_DIR"] = "/path/to/your/tgb/datasets/tkgl_smallpedia"
```

---

## Datasets

- **tkgl-smallpedia** — downloaded automatically via `py-tgb` on first run
- **Mystery movies** — placeholder in Section 6, to be implemented
