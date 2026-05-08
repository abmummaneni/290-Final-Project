# Detective R-GCN

**MATH 290 Final Project — Adam Abramowitz, Max DeSantis, Isaac Dreeben, Abhi Mummaneni**

We train a Relational Graph Convolutional Network (R-GCN) to play detective: given a murder mystery in which a victim has been found, the model identifies which character is the villain. Each story is represented as a heterogeneous knowledge graph — characters, locations, organizations, and occupations connected by typed relations (works with, lives at, deceives, investigates, etc.) — and the R-GCN performs node classification (Villain / Victim / Witness / Uninvolved) over that graph.

The reference paper is Schlichtkrull et al., 2017 — *Modeling Relational Data with Graph Convolutional Networks* ([arXiv:1703.06103](https://arxiv.org/abs/1703.06103)). A per-equation implementation checklist is in [RGCN.md](RGCN.md).

---

## The Task

A victim has been found. The model has access to the full cast of characters, their features (alibi, motive, concealment, hidden relationships, etc.), and every relationship in the graph **except** the `kills` edges, which are masked at evaluation time so they can't reveal the answer. The model must predict the villain from circumstantial evidence alone.

- **Input:** heterogeneous knowledge graph + character features (8 observable traits)
- **Output:** Villain / Uninvolved label for each character (the victim is given)
- **Headline metrics:** villain precision and villain recall

---

## The Dataset

576 murder mystery plots — novels, films, TV episodes, podcasts, short stories — scraped from Wikipedia and converted into knowledge graphs by a two-pass LLM extraction pipeline (`mixtral:8x7b` via Ollama, run locally).

| Statistic | Value |
|---|---|
| Stories | 576 |
| Characters | ~5,900 |
| Total nodes | ~14,000 |
| Total edges | ~32,000 |
| Relation types (with inverses) | 16 |

Eight base relations — `kills`, `harms`, `investigates`, `deceives`, `personal_bond`, `professional`, `spatial`, `social` — each with a learned inverse, giving the directional message passing the paper calls for.

---

## The Model

A 2-layer R-GCN encoder + multi-task heads, trained for 50 epochs with Adam:

- **R-GCN encoder** — typed message passing with separate weight matrices per relation, basis decomposition, per-relation neighbor normalization `c_{i,r} = |N_i^r|`, no dropout (matches the paper).
- **Node classifier** — softmax over 4 character classes; cross-entropy loss on labeled nodes.
- **DistMult link-prediction decoder** — trained jointly so the encoder learns embeddings that both classify characters and predict edges.

Crime-revealing edges (`kills`, `killed by`, etc.) are masked at evaluation time. We verified that this masking does not change predictions — the model was already reasoning from circumstantial evidence, not the answer.

---

## Results

5-seed cross-validation, narrative-prominence features excluded so the model can't cheat off story structure:

| Metric | R-GCN (mean ± std) | LogReg Baseline (mean ± std) |
|---|---|---|
| **Villain Precision** | **0.825 ± 0.033** | 0.684 ± 0.060 |
| **Villain Recall** | 0.708 ± 0.019 | **0.752 ± 0.046** |
| **Villain F1** | **0.762 ± 0.025** | 0.715 ± 0.045 |
| **Overall Accuracy** | **0.904 ± 0.007** | 0.869 ± 0.019 |

The R-GCN leads on F1, precision (+14 points), and overall accuracy. The LogReg has slightly higher recall but produces roughly twice as many false accusations.

### Spectral ablation

| Approach | Test Accuracy | Villain F1 |
|---|---|---|
| Spectral only (Laplacian eigenmaps, no features) | 25.5% | 0.113 |
| Features-only LogReg (no graph) | 58.5% | 0.705 |
| Features + Spectral (concatenated) | 58.8% | 0.696 |
| **R-GCN (typed relational message passing)** | **90.4%** | **0.762** |

Graph topology alone is essentially useless. The R-GCN's 32-point lead over LogReg comes specifically from typed message passing over the 16 relation types, not from generic graph structure.

---

## Held-Out Inference Case Studies

Three real-world cases were excluded from training entirely and analyzed inference-only — the model was asked to rank all characters by predicted villain probability on stories it has never seen.

### Case 1 — *Zodiac* (2007 film)

The Zodiac killings are unsolved in real life. **Arthur Leigh Allen is the prime suspect named by the film** but was never charged. The model ranked Arthur Leigh Allen as the **#1 most likely villain (P = 1.0000)**, with the unidentified-villain placeholder at #2 and all other characters far below.

### Case 2 — *In the Dark: Season 3* (podcast)

Investigates the wrongful prosecution of Curtis Flowers (six trials, eventual exoneration) by DA Doug Evans. The model independently surfaced three results from one held-out case:

1. **Identified the wrongful-prosecution antagonists** — Doug Evans #1 and his investigator John Johnson #3, the two characters the podcast names.
2. **Surfaced three concealed alternate suspects** — Hemphill (#2), Presley (#4), Gamble (#5) — labeled "Suspect" in the source data, which is **not a class the model trains on**. It flagged them purely from features and graph context.
3. **Exonerated the wrongly accused** — Curtis Flowers ranked **dead last (P = 0.0000)**, even though the data file initially had him mislabeled as Villain. The model overrode the wrong label using the evidence.

### Case 3 — Zodiac case, factual record (2023–2025)

The same Zodiac case as Case 1, but with a synopsis built exclusively from official law-enforcement sources, contemporaneous reporting, and 2023–2025 forensic updates (Donna Lass DNA identification, Allen's DNA exclusion). The Fincher film is explicitly excluded.

| Snapshot | #1 ranked suspect | Allen rank |
|---|---|---|
| **Case 1 (2007 fiction)** | **Arthur Leigh Allen** | **#1** (P = 1.0000) |
| **Case 3 (factual, 2023–2025 update)** | **Lawrence Kane** (Vallejo PD's officially-developed suspect) | **#38, last** (P = 0.0004) |

Both are defensible: the film centers Allen because in 2007 he was the only police-named suspect; the factual record centers Kane because of the 2023 Donna Lass DNA identification connecting Kane to a confirmed homicide victim. **The model is reading the structural evidence in whichever synopsis it is given, not memorizing pop-culture suspect rankings.**

Together these three cases support the paper's central claim: the R-GCN learns transferable patterns of evidence-based reasoning over knowledge graphs, rather than memorizing labels or surface features.

---

## Repository Layout

```
Detective_R-GCN/
├── README.md                       ← this file
├── RGCN.md                         ← Schlichtkrull et al. checklist
├── requirements.txt
│
├── data/                           ← all data
│   ├── candidates.xlsx             ← master candidate list
│   ├── manifest.json
│   ├── synopses/                   ← Wikipedia-scraped synopses (755 .txt)
│   ├── synopses_detailed/          ← curated detailed synopses for re-extraction
│   ├── graphs/                     ← 576 heterogeneous graph JSONs (training data)
│   └── graphs_simple/              ← character-only weighted graphs (spectral baseline)
│
├── pipeline/                       ← data-building code
│   ├── scraper/                    ← Wikipedia scraping
│   ├── extraction/                 ← two-pass Ollama graph extraction
│   ├── extraction_simple/          ← character-only graph builder
│   └── reextract.py                ← re-run extraction on detailed synopses
│
├── model/
│   ├── rgcn_model.py               ← encoder, DistMult decoder, classifier, multi-task wrapper
│   └── load_mystery_graphs.py      ← graph JSONs → single RelationalGraph
│
├── analysis/
│   ├── failure_analysis.py         ← per-story diagnostic
│   ├── inference_analysis.py       ← held-out inference (Zodiac, In the Dark)
│   ├── spectral_baseline.py        ← Laplacian eigenmaps comparison
│   ├── results/                    ← saved analysis outputs
│   └── notebooks/
│       ├── detective_training.ipynb       ← main training notebook
│       └── graph_visualization.ipynb
│
├── docs/                           ← graph schema, prompt templates
├── checkpoints/                    ← cached graph + trained weights (auto-created)
└── logs/
```

---

## Reproducing the Results

```bash
# from project root
source venv/bin/activate
pip install -r Detective_R-GCN/requirements.txt
pip install torch torch-geometric numpy matplotlib

cd Detective_R-GCN

# train (notebook): parses all 576 graphs, builds the model, trains 50 epochs
jupyter notebook analysis/notebooks/detective_training.ipynb

# headline analyses (each writes to analysis/results/)
python analysis/inference_analysis.py                    # Cases 1 and 2
python analysis/inference_analysis.py --targets ZODIAC_FACTUAL   # Case 3
python analysis/spectral_baseline.py                     # spectral ablation
python analysis/failure_analysis.py                      # per-story failure modes
```

Training takes ~2 minutes on an M1 Max for 50 epochs. The first run caches the parsed graph to `checkpoints/detective_graph.pkl`; subsequent runs load from cache.

Entry IDs follow `{medium}_{index}`: `NOV` = Novel, `SHO` = Short Story, `FLM` = Film, `TVE` = TV Episode, `POD` = Podcast.
