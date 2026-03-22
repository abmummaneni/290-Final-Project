# Detective Project — Comprehensive Handoff & Reference

**Last updated:** 2026-03-22
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
    ├── FIX_SYN.txt                       ← Entries with low quality scores (< 0.7)
    ├── FIX_MANUAL.txt                    ← Entries with no synopsis (274 remaining)
    │
    ├── scraper/                          ← Phase 1: Synopsis scraping
    │   ├── __init__.py
    │   ├── main.py                       ← Entry point: python -m scraper.main --all
    │   ├── loader.py                     ← Loads xlsx, assigns entry IDs
    │   ├── wikipedia.py                  ← Wikipedia scraping logic
    │   ├── cleaner.py                    ← Text cleaning
    │   └── validator.py                  ← Quality scoring (word count, characters, events, resolution)
    │
    ├── extraction/                       ← Phase 2: Graph extraction from synopses
    │   ├── __init__.py
    │   ├── main.py                       ← Entry point: python -m extraction.main --all
    │   ├── extractor.py                  ← Two-pass Ollama extraction + edge normalization
    │   ├── prompt.py                     ← Pass 1 (nodes) and Pass 2 (edges) prompts
    │   └── data/
    │       ├── extraction_status.json    ← Tracks extraction status per entry
    │       └── graphs/                   ← One .json per entry (graph data)
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

## Current Status (2026-03-22)

### Phase 1: Synopsis Scraping — MOSTLY COMPLETE

| Status | Count | Description |
|---|---|---|
| SUCCESS | 348 | Good synopsis, deduplicated, eligible for extraction |
| PARTIAL | 37 | Synopsis found but < 150 words |
| NEEDS_MANUAL | 274 | No synopsis found — need manual input |
| DUPLICATE | 96 | Duplicate titles removed (kept best quality version) |

#### Synopsis Quality (348 SUCCESS entries)

| Quality Range | Count |
|---|---|
| >= 0.7 (good) | 330 |
| 0.4 - 0.7 (needs improvement) | 18 |

### Phase 2: Graph Extraction — 339 of 348 DONE

- **339 graphs extracted** so far (the original SUCCESS batch)
- 9 newly scored entries (from latest manual synopsis additions) still need extraction
- Using **mixtral:8x7b** via local Ollama (two-pass approach)
- Zero failures on last full run
- Average: ~3.5 min per entry, ~33 edges per graph

### Phase 3: Validation — NOT STARTED

- Validate extracted JSON using a second LLM pass
- Resolve conflicts and assemble final graph dataset

### Phase 4: Model Training — NOT STARTED

- Convert graph JSONs into PyTorch Geometric format for αLoNGAE
- Train for villain prediction, motivation prediction, hidden relationship detection
- Run ablation study variants defined in the graph schema

---

## What Needs to Happen Next

### Step 1: Fill missing synopses (all team members)
Work through `FIX_MANUAL.txt` (274 entries) and `FIX_SYN.txt` (18 low-quality entries). Place new/improved synopsis text directly in `data/cleaned/{ID}.txt`.

### Step 2: Re-score and re-extract (Max)
After synopses are updated, run the re-scoring snippet (see below), then clear extraction status for updated entries and run extraction.

### Step 3: Train the model (team)
Convert the graph JSONs to adjacency tensors + feature matrices. Train αLoNGAE. The `Karate_Club/` notebooks in the project root have working examples of graph autoencoders to build from.

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

### Running the scraper (Phase 1)
```bash
cd /Users/maxdesantis/dev/290-Final-Project/Detective

# Full run (skips already-completed)
python -m scraper.main --all

# Single entry
python -m scraper.main --id NOV_001

# Only NEEDS_MANUAL entries
python -m scraper.main --status NEEDS_MANUAL

# Dry run
python -m scraper.main --dry-run
```

### Running graph extraction (Phase 2)
```bash
cd /Users/maxdesantis/dev/290-Final-Project/Detective

# Single entry
python -m extraction.main --id NOV_005

# Batch of N entries
python -m extraction.main --limit 50

# All remaining (skips already-extracted)
python -m extraction.main --all

# Use caffeinate to prevent Mac sleep during long runs
caffeinate -i python -m extraction.main --all
```

### Re-scoring updated synopses
When synopsis .txt files in `data/cleaned/` have been updated, re-score them and update the manifest:

```python
import json
from scraper.validator import score_synopsis

with open('data/manifest.json') as f:
    m = json.load(f)

for eid, info in sorted(m.items()):
    if info['scrape_status'] in ('SUCCESS', 'PARTIAL', 'NEEDS_MANUAL'):
        try:
            with open(f'data/cleaned/{eid}.txt') as f2:
                text = f2.read().strip()
            wc = len(text.split())
            if wc > info.get('word_count_cleaned', 0):
                score = score_synopsis(text)
                m[eid]['word_count_cleaned'] = wc
                m[eid]['quality_score'] = score
                m[eid]['scrape_status'] = 'SUCCESS' if wc >= 150 else 'PARTIAL'
                m[eid]['needs_review'] = score < 0.4
        except:
            pass

with open('data/manifest.json', 'w') as f:
    json.dump(m, f, indent=2, ensure_ascii=False)
```

Then clear updated entries from `extraction/data/extraction_status.json` and re-run extraction.

### Regenerating FIX_SYN.txt and FIX_MANUAL.txt
These are worklists. Regenerate after re-scoring:
- `FIX_SYN.txt` — SUCCESS entries with quality_score < 0.7
- `FIX_MANUAL.txt` — NEEDS_MANUAL entries (no synopsis at all)

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

## Wikipedia Scraping Strategy

### Primary: wikipedia-api library
Search by title, fetch "Plot" / "Synopsis" / "Plot summary" etc. sections.

### Section title variants (tried in order)
Plot, Synopsis, Plot summary, Story, Storyline, Summary, Overview, Plot overview, Episode summary, Season summary

### Fallback: requests + BeautifulSoup
Direct HTML scraping of Wikipedia if the API returns empty.

### TV/Podcast entries
Many don't have their own pages — try the show's main page episode list. If nothing found, mark NEEDS_MANUAL.

### Rate limiting
0.5s delay between requests. `tenacity` for retries (3 attempts, exponential backoff).

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

Possible `scrape_status` values: `SUCCESS`, `PARTIAL`, `NEEDS_MANUAL`, `FAILED`, `DUPLICATE`

---

## Important Preferences & Constraints

- **No Claude API for extraction** — use Ollama only (cost reasons)
- **Use `caffeinate -i`** for long-running extractions (prevents Mac sleep)
- User puts updated synopses directly into `data/cleaned/{ID}.txt`
- `data/raw/` has been deleted — `data/cleaned/` is the single source of truth
- 96 duplicate entries marked DUPLICATE in manifest — do not re-extract these
- Max is running on M1 Max with 64GB RAM

---

## Authors

MATH 290 Final Project — Adam Abramowitz, Max DeSantis, Isaac Dreeben, Abhi Mummaneni
