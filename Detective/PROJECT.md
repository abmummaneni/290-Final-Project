# Detective Project — Comprehensive Handoff & Reference

**Last updated:** 2026-03-23
**Purpose:** Read this file at the start of any new chat to fully resume work on the Detective subproject. This is the single source of truth for project status, architecture, and next steps.

---

## Project Overview

This is a multi-phase pipeline that builds a **heterogeneous knowledge graph dataset of murder mysteries** for use with a graph autoencoder (αLoNGAE). The pipeline scrapes synopses, extracts structured graph data via LLM, and ultimately trains a model to predict villains, motivations, and hidden relationships.

**Team:** Adam Abramowitz, Max DeSantis, Isaac Dreeben, Abhi Mummaneni (MATH 290 Final Project)

**Conceptual framing:** The project builds a **detective model**. The graph encodes motive (character features), opportunity (location/access features), and capability (occupation features) — exactly the dimensions a detective reasons about.

**Working directory:** `/Users/maxdesantis/dev/290-Final-Project/Detective/`
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
    ├── murder_mystery_graph_schema.md    ← Full graph schema definition
    ├── murder_mystery_candidates_v2.xlsx ← Master candidate list (755 entries)
    ├── requirements.txt                  ← Python deps for scraper/extraction
    ├── .env                              ← Contains ANTHROPIC_API_KEY (not used)
    │
    ├── scraper/                          ← Phase 1: Synopsis scraping (COMPLETE)
    │   ├── __init__.py
    │   ├── main.py                       ← Entry point: python -m scraper.main --all
    │   ├── loader.py                     ← Loads xlsx, assigns entry IDs
    │   ├── wikipedia.py                  ← Wikipedia scraping logic
    │   ├── cleaner.py                    ← Text cleaning
    │   └── validator.py                  ← Quality scoring (word count, characters, events, resolution)
    │
    ├── extraction/                       ← Phase 2: Graph extraction (COMPLETE)
    │   ├── __init__.py
    │   ├── main.py                       ← Entry point: python -m extraction.main --all
    │   ├── extractor.py                  ← Two-pass Ollama extraction + edge normalization
    │   ├── prompt.py                     ← Pass 1 (nodes) and Pass 2 (edges) prompts
    │   ├── normalize_labels.py           ← Label normalization script (run after extraction)
    │   └── data/
    │       ├── extraction_status.json    ← Tracks extraction status per entry
    │       └── graphs/                   ← One .json per entry (576 graph files)
    │
    ├── data/
    │   ├── cleaned/                      ← One .txt per entry (cleaned synopsis text)
    │   └── manifest.json                 ← Master status tracker for all 755 entries
    │
    └── logs/
        ├── scrape.log
        └── extraction.log
```

---

## Current Status (2026-03-23)

### Phase 1: Synopsis Scraping — COMPLETE

| Status | Count | Description |
|---|---|---|
| SUCCESS | 576 | Good synopsis, extracted, graph available |
| PARTIAL | 37 | Synopsis found but < 150 words (not extracted) |
| DUPLICATE | 96 | Duplicate titles removed |
| EXCLUDED | 46 | Not single-case oriented or extraction failed |

### Phase 2: Graph Extraction — COMPLETE

- **576 graphs extracted** with zero failures
- Labels normalized to 5 valid classes (Villain, Victim, Witness, Uninvolved, UNK)
- Using **mixtral:8x7b** via local Ollama (two-pass approach)
- Average: ~3 min per entry

#### Dataset Statistics

| Metric | Mean | Median | Min | Max |
|---|---|---|---|---|
| Characters per graph | 10.2 | 9 | 1 | 45 |
| Edges per graph | 27.5 | 24 | 1 | 254 |
| Locations per graph | 5.1 | 5 | 0 | 17 |
| Occupations per graph | 6.3 | 6 | 0 | 30 |
| Organizations per graph | 2.7 | 3 | 0 | 9 |

#### Label Distribution (5,926 characters total)

| Label | Count | % |
|---|---|---|
| Uninvolved | 2,944 | 49.7% |
| Villain | 1,331 | 22.5% |
| Victim | 1,044 | 17.6% |
| Witness | 539 | 9.1% |
| UNK | 68 | 1.1% |

#### Known Minor Issues (acceptable for training)
- **NOV_026** — has villain but no victim labeled
- **TVE_081** (Murder on the Orient Express) — no single villain (all passengers are collectively guilty)
- **2 entries** with no villain but have victim (NOV_026 edge case, TVE_081 by design)

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

---

## What Needs to Happen Next

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
cd /Users/maxdesantis/dev/290-Final-Project
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
cd /Users/maxdesantis/dev/290-Final-Project/Detective

# Full run (skips already-completed)
python -m scraper.main --all

# Single entry
python -m scraper.main --id NOV_001
```

### Running graph extraction (Phase 2 — complete, for reference)
```bash
cd /Users/maxdesantis/dev/290-Final-Project/Detective

# Single entry
python -m extraction.main --id NOV_005

# All remaining (skips already-extracted)
caffeinate -i python -m extraction.main --all
```

### Normalizing labels (run after any new extraction)
```bash
cd /Users/maxdesantis/dev/290-Final-Project/Detective

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
- User puts updated synopses directly into `data/cleaned/{ID}.txt`
- `data/raw/` has been deleted — `data/cleaned/` is the single source of truth
- 96 duplicate entries marked DUPLICATE in manifest — do not re-extract these
- 46 entries marked EXCLUDED — not single-case oriented or extraction failed
- Max is running on M1 Max with 64GB RAM

---

## Authors

MATH 290 Final Project — Adam Abramowitz, Max DeSantis, Isaac Dreeben, Abhi Mummaneni
