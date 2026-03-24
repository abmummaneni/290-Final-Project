# Detective Project — Comprehensive Handoff & Reference

**Last updated:** 2026-03-23
**Purpose:** Read this file at the start of any new chat to fully resume work on the Detective subproject. This is the single source of truth for project status, architecture, and next steps.

---

## Project Overview

This is a multi-phase pipeline that builds a **heterogeneous knowledge graph dataset of murder mysteries** for use with a graph autoencoder (αLoNGAE). The pipeline scrapes synopses, extracts structured graph data via LLM, and ultimately trains a model to predict villains, motivations, and hidden relationships.

**Team:** Adam Abramowitz, Max DeSantis, Isaac Dreeben, Abhi Mummaneni (MATH 290 Final Project)

**Conceptual framing:** The project builds a **detective model**. The graph encodes motive (character features), opportunity (location/access features), and capability (occupation features) — exactly the dimensions a detective reasons about.

**Working directory:** `/Users/abhi/ml/290Final/Detective/`
**Candidate list:** `murder_mystery_candidates_v2.xlsx` (755 entries across novels, films, TV episodes, short stories, and podcasts)
**Graph schema:** `murder_mystery_graph_schema.md` (full node/edge/feature definitions)

---

## Directory Structure

```
290-Final-Project/
├── Karate_Club/                          ← Graph autoencoder examples (reference for Phase 4)
│   ├── 00_data_exploration.ipynb
│   ├── 01_laplacian_eigenmaps.ipynb
│   └── 02_autoencoder.ipynb
├── requirements.txt                      ← Project-level Python deps
├── venv/                                 ← Virtual environment
│
└── Detective/                            ← Main pipeline
    ├── PROJECT.md                        ← THIS FILE
    ├── graph_visualization.ipynb         ← Notebook for dataset summaries + graph plots
    ├── murder_mystery_graph_schema.md    ← Full graph schema definition
    ├── murder_mystery_candidates_v2.xlsx ← Master candidate list (755 entries)
    ├── requirements.txt                  ← Python deps for scraper/extraction
    ├── .env                              ← Optional local env overrides
    │
    ├── scraper/                          ← Phase 1: Synopsis scraping code
    │   ├── __init__.py
    │   ├── main.py                       ← Entry point: python -m scraper.main --all
    │   ├── loader.py                     ← Loads xlsx, assigns entry IDs
    │   ├── wikipedia.py                  ← Wikipedia scraping logic
    │   ├── cleaner.py                    ← Text cleaning
    │   └── validator.py                  ← Quality scoring (word count, characters, events, resolution)
    │
    └── extraction/                       ← Phase 2: Graph extraction code + checked-in outputs
        ├── __init__.py
        ├── main.py                       ← Entry point: python -m extraction.main --all
        ├── extractor.py                  ← Two-pass Ollama extraction + edge normalization
        ├── prompt.py                     ← Pass 1 (nodes) and Pass 2 (edges) prompts
        ├── normalize_labels.py           ← Label normalization script (run after extraction)
        └── data/
            ├── extraction_status.json    ← Tracks extraction status per entry
            └── graphs/                   ← One .json per entry (576 graph files)
```

**Checked-in snapshot note:** This repo currently includes the extracted graph corpus under `extraction/data/graphs/`, but does not include the scraper runtime artifacts (`data/manifest.json`, `data/cleaned/`, `logs/`) that existed during the original pipeline runs.

---

## Current Status (2026-03-23)

### Phase 1: Synopsis Scraping — COMPLETE IN PRIOR RUNS

| Status | Count | Description |
|---|---|---|
| SUCCESS | 576 | Good synopsis, extracted, graph available |
| PARTIAL | 37 | Synopsis found but < 150 words (not extracted) |
| DUPLICATE | 96 | Duplicate titles removed |
| EXCLUDED | 46 | Not single-case oriented or extraction failed |

The scraper/extraction code is still present, but this checkout does not include the generated `data/manifest.json` or cleaned synopsis text files from those earlier runs.

### Phase 2: Graph Extraction — COMPLETE AND CHECKED IN

- **576 graph JSON files are present** in `extraction/data/graphs/`
- Labels normalized to 5 valid classes (Villain, Victim, Witness, Uninvolved, UNK)
- Using **mixtral:8x7b** via local Ollama (two-pass approach)
- Visualization notebook available at `graph_visualization.ipynb`

#### Dataset Statistics

| Metric | Mean | Median | Min | Max |
|---|---|---|---|---|
| Characters per graph | 10.2 | 9 | 3 | 45 |
| Edges per graph | 27.7 | 24 | 8 | 254 |
| Locations per graph | 5.1 | 5 | 0 | 17 |
| Occupations per graph | 6.3 | 6 | 0 | 30 |
| Organizations per graph | 2.7 | 3 | 0 | 9 |

#### Medium Breakdown

| Medium | Count |
|---|---|
| Novel | 340 |
| TV Episode | 126 |
| Film | 62 |
| Podcast | 28 |
| Short Story | 20 |

#### Label Distribution (5,907 characters total)

| Label | Count | % |
|---|---|---|
| Uninvolved | 2,930 | 49.6% |
| Villain | 1,331 | 22.5% |
| Victim | 1,043 | 17.7% |
| Witness | 539 | 9.1% |
| UNK | 64 | 1.1% |

#### Common Edge Types

Top relation types in the checked-in graph corpus:
- `employed as` (1,650)
- `related to` (1,130)
- `present at` (1,078)
- `resides at` (1,047)
- `affiliated with` (729)
- `romantically involved with` (559)
- `kills` (552)
- `investigates` (529)

#### Known Minor Issues (acceptable for training)
- **NOV_026** — has villain but no victim labeled
- **TVE_081** (Murder on the Orient Express) — no single villain (all passengers are collectively guilty)
- **2 entries** with no villain but have victim (NOV_026 edge case, TVE_081 by design)
- **Status file mismatch** — `extraction/data/extraction_status.json` lists 578 `SUCCESS` entries, but only 576 graph files are present; stale IDs are `SHO_018` and `SHO_019`

### Phase 3: Validation — COMPLETE

- Label normalization applied via `extraction/normalize_labels.py`
- Compound labels resolved by priority: Villain > Victim > Witness > Uninvolved > UNK
- Non-standard labels (Suspect, Detective, Investigator, etc.) mapped to valid classes
- Problematic extractions (3 entries) excluded after re-extraction attempts
- All 576 remaining graphs validated for structural integrity

### Phase 4: Model Training — NOT STARTED (NEXT STEP)

- Convert graph JSONs into PyTorch Geometric format for αLoNGAE
- Train for villain prediction, motivation prediction, hidden relationship detection
- Run ablation study variants defined in the graph schema
- Reconcile `extraction_status.json` with the checked-in graph corpus before automating around it

---

## What Needs to Happen Next

### Step 0: Reconcile checked-in metadata (recommended)
- Remove or explain the two stale `SUCCESS` entries in `extraction/data/extraction_status.json` (`SHO_018`, `SHO_019`)
- Decide whether scraper artifacts (`data/manifest.json`, cleaned synopsis texts, logs) should remain local-only or be regenerated if Phase 1 is rerun from this checkout

### Step 1: Convert graphs to PyTorch Geometric format (team)
Write a data loader that reads the 576 graph JSONs from `extraction/data/graphs/` and converts them to PyTorch Geometric `HeteroData` objects:
- Build node feature matrices for each node type (character, occupation, location, organization)
- Build adjacency tensors per relation type
- Extract ground truth labels for villain prediction (character label field)
- Handle UNK labels by masking them during training

### Step 2: Train αLoNGAE (team)
- Implement the αLoNGAE architecture (reference: `Karate_Club/` notebooks)
- Primary task: 4-class villain prediction (Villain, Victim, Witness, Uninvolved)
- Secondary task: motivation prediction (motive_type feature)
- Tertiary task: hidden relationship detection (link prediction)
- Joint loss with per-instance masking (see graph schema §4.4)

### Step 3: Run ablation study (team)
10 model variants defined in the graph schema (§5):
Baseline, AE-NoGraph, LoNGAE-NoFeatures, LoNGAE-Static, LoNGAE-Full, LoNGAE-Narrative, SingleTask, MultiTask, Undirected, SimpleGraph

---

## Environment Setup

### Virtual environment
```bash
cd /Users/abhi/ml/290Final
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r Detective/requirements.txt
```

### Ollama setup
- Running locally at `http://localhost:11434`
- Default model: `mixtral:8x7b` (set via OLLAMA_MODEL env var or .env)
- Also available: `llama3:8b`, `llama3:70b` (too slow)
- **Do NOT use Claude API for extraction** — use Ollama only (cost reasons)

---

## Key Operations

### Running the scraper (Phase 1 — complete, for reference)
```bash
cd /Users/abhi/ml/290Final/Detective

# Full run (skips already-completed)
python -m scraper.main --all

# Single entry
python -m scraper.main --id NOV_001
```

### Running graph extraction (Phase 2 — complete, for reference)
```bash
cd /Users/abhi/ml/290Final/Detective

# Single entry
python -m extraction.main --id NOV_005

# All remaining (skips already-extracted)
caffeinate -i python -m extraction.main --all
```

### Normalizing labels (run after any new extraction)
```bash
cd /Users/abhi/ml/290Final/Detective

# Dry run (preview changes)
python -m extraction.normalize_labels --dry-run

# Apply changes
python -m extraction.normalize_labels
```

### Re-extracting specific entries
To re-extract an entry (e.g. after updating its synopsis):
1. Delete the graph: `rm extraction/data/graphs/{ID}.json`
2. Remove from extraction status:
```python
import json
with open('extraction/data/extraction_status.json') as f:
    s = json.load(f)
del s['ENTRY_ID']
with open('extraction/data/extraction_status.json', 'w') as f:
    json.dump(s, f, indent=2)
```
3. Re-run: `python -m extraction.main --id ENTRY_ID`
4. Re-normalize: `python -m extraction.normalize_labels`

---

## Entry ID Convention

```
{medium_code}_{zero_padded_index}

NOV = Novel, SHO = Short Story, FLM = Film, TVE = TV Episode, POD = Podcast

Examples: NOV_001, FLM_042, TVE_007, SHO_003, POD_012
```

IDs are assigned by order in the xlsx after sorting by (Medium, Year, Title). Stable across runs.

---

## Two-Pass Extraction Approach

The extractor sends two separate prompts to the LLM per entry:

1. **Pass 1 (nodes):** Extract characters, occupations, locations, organizations with features
2. **Pass 2 (edges):** Given the extracted nodes + synopsis, extract all relationships

This dramatically improves edge count (especially character-character relationships) vs single-pass. Edge normalization maps non-schema relation types to valid schema types.

---

## Visualization

Use `graph_visualization.ipynb` to inspect the checked-in dataset:
- dataset-level numeric and categorical summaries
- label counts and edge-type frequency tables
- histogram/bar-chart views of graph size
- network visualization for any selected graph ID in `extraction/data/graphs/`

---

## Graph JSON Structure (per entry)

Each file in `extraction/data/graphs/{ID}.json`:
```json
{
  "characters": [{"id": "char_0", "name": "...", "label": "Villain|Victim|Witness|Uninvolved|UNK", "features": {...}}],
  "occupations": [{"id": "occ_0", "name": "...", "features": {...}}],
  "locations": [{"id": "loc_0", "name": "...", "features": {...}}],
  "organizations": [{"id": "org_0", "name": "...", "features": {...}}],
  "edges": [{"source": "char_0", "target": "char_1", "relation": "married to", "directed": true}],
  "metadata": {"entry_id": "...", "title": "...", "author": "...", "year": ..., "medium": "..."}
}
```

---

## Graph Schema Summary

Full schema is in `murder_mystery_graph_schema.md`. Key points:

### Node Types
| Type | Feature Vector | Key Features |
|---|---|---|
| Character | 10 features | gender, social status, introduction timing, alibi, scene presence, motive, concealment, hidden relationships, motive type, narrative prominence |
| Occupation | 3 features | authority, access, capability |
| Location | 3 features | accessibility, isolability, evidentiary value |
| Organization | 3 features | institutional power, secrecy, financial scale |

### Edge Types
- **Character → Occupation:** `employed as`, `formerly employed as`
- **Character → Location:** `resides at`, `present at`, `owns`
- **Character → Organization:** `affiliated with`, `leads`, `employed by`
- **Character → Character:** Open vocabulary (married to, suspects, blackmails, etc.)
- **Location → Location:** `near to`
- **Organization → Location:** `located at`

### Inference Tasks
1. **Villain prediction** (primary) — 4-class node classification: Villain, Victim, Witness, Uninvolved
2. **Motivation prediction** (secondary) — predict motive type: jealousy, money, revenge, love
3. **Hidden relationship detection** (tertiary) — link prediction on unobserved character-character edges

### Ablation Variants
Baseline, AE-NoGraph, LoNGAE-NoFeatures, LoNGAE-Static, LoNGAE-Full, LoNGAE-Narrative, SingleTask, MultiTask, Undirected, SimpleGraph

---

## Quality Scoring

`scraper/validator.py` scores each synopsis in [0, 1]:

| Signal | Weight | Notes |
|---|---|---|
| Word count (target: 300+) | 0.35 | Sigmoid-scaled; 300w = 0.5, 600w = 0.9 |
| Named character count (3+) | 0.30 | Proxy for cast graph extractability |
| Event verb density | 0.20 | kill/die/murder etc. — proxy for event edge density |
| Resolution/reveal sentence | 0.15 | Binary: does synopsis reveal the villain? |

Entries with quality_score < 0.4 are flagged `needs_review = true`.

---

## manifest.json Schema

`manifest.json` is not checked into this repo snapshot, but when Phase 1 is rerun it is expected to have the following structure:

```json
{
  "NOV_001": {
    "entry_id": "NOV_001",
    "title": "And Then There Were None",
    "author": "Agatha Christie",
    "year": 1939,
    "medium": "Novel",
    "subgenre": "Classic Whodunit",
    "villain_reveal": "Explicit",
    "synopsis_quality_flag": "H",
    "scrape_status": "SUCCESS",
    "wikipedia_url": "https://en.wikipedia.org/wiki/...",
    "section_found": "Plot",
    "word_count_raw": 842,
    "word_count_cleaned": 731,
    "quality_score": 0.91,
    "needs_review": false,
    "notes": ""
  }
}
```

Possible `scrape_status` values: `SUCCESS`, `PARTIAL`, `NEEDS_MANUAL`, `FAILED`, `DUPLICATE`, `EXCLUDED`

---

## Important Preferences & Constraints

- **No Claude API for extraction** — use Ollama only (cost reasons)
- **Use `caffeinate -i`** for long-running extractions (prevents Mac sleep)
- If Phase 1 is rerun locally, updated synopses should go into `data/cleaned/{ID}.txt`
- `data/raw/` was removed in the original workflow; `data/cleaned/` is the intended single source of truth
- 96 duplicate entries were marked `DUPLICATE` in the original manifest — do not re-extract these
- 46 entries were marked `EXCLUDED` — not single-case oriented or extraction failed
- Extraction was developed against a local Ollama setup on Apple Silicon

---

## Authors

MATH 290 Final Project — Adam Abramowitz, Max DeSantis, Isaac Dreeben, Abhi Mummaneni
