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

## Project folders

Cell 3 creates two local folders next to the notebook:

- `./datasets/` for the local `tkgl-smallpedia` download and `wikidata_labels_cache.json`
- `./checkpoints/` for model weights and `wiki_graph.pkl`

**Google Colab + Drive (recommended for sharing):**
```python
from google.colab import drive
drive.mount("/content/drive")
PROJECT_DIR = "/content/drive/MyDrive/rgcn"
```
Then set `PROJECT_DIR` to that value in cell 3 before running the rest of the notebook.

**Local machine:** the defaults work with no changes.

---

## Sharing checkpoints with teammates

After one person trains the model, share the `checkpoints/` folder. It contains:

| File | Contents |
|---|---|
| `rgcn_scratch_wikidata.pt` | Trained from-scratch model weights |
| `rgcn_pyg_wikidata.pt` | Trained PyG model weights |
| `wiki_graph.pkl` | Processed graph (edges, splits, label maps) |
Teammates who copy this folder into their project will skip training completely.

**If you already have `wikidata_labels_cache.json`**, copy it into `DATA_DIR` before running the notebook:
```python
import shutil
shutil.copy("wikidata_labels_cache.json", os.path.join(DATA_DIR, "wikidata_labels_cache.json"))
```

---

## TGB data directory

No environment variables are needed. The notebook now downloads `tkgl-smallpedia` into `./datasets/tkgl_smallpedia/` if it is missing, and then points `py-tgb` at that local folder.

## Wikidata labels

The first label-fetch run is long because it resolves tens of thousands of Wikidata IDs. The notebook now saves `wikidata_labels_cache.json` incrementally while fetching, so if Wikidata rate-limits the requests or the session stops, rerunning the cell resumes from the saved cache instead of starting over.

---

## Datasets

- **tkgl-smallpedia** — downloaded automatically via `py-tgb` on first run
- **Mystery movies** — placeholder in Section 6, to be implemented
