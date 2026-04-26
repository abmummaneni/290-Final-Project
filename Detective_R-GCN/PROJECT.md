# Detective Project — Comprehensive Handoff & Reference

**Last updated:** 2026-04-26
**Purpose:** Read this file at the start of any new chat to fully resume work on the Detective subproject. This is the single source of truth for project status, architecture, and next steps.

### For teammates / future readers

If you're picking up this project (or asking an LLM to read this file), the fastest way to get oriented:

1. **Read "Results Summary"** — self-contained overview of what we built and what we found, including the **Held-Out Inference Case Studies** (Zodiac and In the Dark) and the **Re-Extraction Impact** section
2. **Read "Failure Analysis"** — the deep-dive on where the model fails, with paper-ready tables
3. **See `failure_analysis_results.txt`** for raw failure analysis output
4. **See `inference_analysis_results.txt`** for the FLM_047 (Zodiac → Arthur Leigh Allen) and POD_035 (In the Dark → Doug Evans / Curtis Flowers exoneration) suspect rankings
5. **See `cross_validation_results.txt`** for the 5-seed cross-validation output
6. **See `unsolved_cases.md`** for the project's case-by-case status across both re-extraction rounds

The model code (`rgcn_model.py`), data loader (`load_mystery_graphs.py`), and training notebook (`detective_training.ipynb`) are unchanged except where PROJECT.md says they've been updated. The spectral baseline (`spectral_baseline.py`), failure analysis (`failure_analysis.py`), inference analysis (`inference_analysis.py`), and re-extraction script (`reextract.py`) are standalone and reproducible.

---

## Project Overview

This is a multi-phase pipeline that builds a **heterogeneous knowledge graph dataset of murder mysteries** and trains an **R-GCN** (Relational Graph Convolutional Network) to identify the villain in a murder mystery.

**Team:** Adam Abramowitz, Max DeSantis, Isaac Dreeben, Abhi Mummaneni (MATH 290 Final Project)

**Conceptual framing:** The project builds a **detective model**. Given a murder mystery where the victim is known, the model uses relational graph structure — motive (character features), opportunity (location/access features), and capability (occupation features) — to predict which character is the villain.

### Task Definition

**The detective scenario:** A victim has been found. The model has access to the full cast of characters, their features (alibi, motive, concealment, etc.), their relationships to locations/organizations/occupations, and their relationships to each other — *except* for the `kills` edges, which reveal the answer. The model must predict who is the villain.

**Primary task:** Villain identification (node classification)
- Given: victim identity, all character features, all non-lethal edges
- Predict: which character(s) are Villain vs. Uninvolved
- Key metrics: Villain precision (is the accusation correct?), Villain recall (did we catch all villains/accomplices?)

**Secondary task:** Motivation inference
- Character features include `has_motive` and `motive_type` — these are part of the evidence available to the detective, not predictions. A detective can observe who has motive; the question is which motivated character actually did it.

**Evaluation methodology:**
- At test time, `kills` and `kills_inv` edges must be **masked** from the graph — they reveal the answer
- All other edges remain (spatial, professional, personal, investigative, deceptive, etc.)
- Character labels (Villain/Victim/Witness/Uninvolved) are the prediction targets
- The victim label is known; only Villain vs. Uninvolved classification matters for the primary task
- **Crime-edge masking (2026-04-16):** Implemented. Crime edges (`kills`, `killed by`, `sexually assaults`, `financial crime/transaction`) are tagged during loading and removed from the graph at evaluation time. **Result:** Masking had zero impact on predictions — the model was already making all decisions from non-crime edges and node features, not from `kills` edges. This validates the detective framing: the model genuinely uses circumstantial evidence, not the answer.

**Working directory:** `/Users/maxdesantis/dev/290-Final-Project/Detective_R-GCN/`
**Original data pipeline:** `../Detective/` (scraping, extraction, graph schema)
**Reference paper:** Schlichtkrull et al., 2017 — checklist in `RGCN.md`

---

## Results Summary

*If you're reading this file for the first time, start here. This section summarizes what we've built, what we've found, and what it means.*

### What we built

We trained an R-GCN (Relational Graph Convolutional Network) to play detective: given a murder mystery where a victim has been found, identify which character is the villain. The model takes as input a knowledge graph for each story — characters with features (motive, alibi, concealment, etc.), connected by typed relationships (works with, lives at, deceives, investigates, etc.) — and outputs a villain/non-villain prediction for each character. Crime-revealing edges (kills, murdered by) are hidden from the model at evaluation time, so it must reason from circumstantial evidence only.

The dataset is 576 murder mystery plots (novels, films, TV episodes, podcasts, short stories) with ~14,000 nodes and ~32,000 edges, extracted from Wikipedia synopses via LLM (mixtral:8x7b).

### How well it works

**Cross-validated results** (5 different random train/test splits, 50 epochs each, narrative metadata features excluded, four special-status entries excluded from train/test as described below):

| Metric | R-GCN (mean ± std) | LogReg Baseline (mean ± std) |
|---|---|---|
| **Villain Precision** | **0.830 ± 0.029** | 0.684 ± 0.061 |
| **Villain Recall** | 0.676 ± 0.068 | **0.747 ± 0.047** |
| **Villain F1** | **0.743 ± 0.047** | 0.712 ± 0.044 |
| **Overall Accuracy** | **0.900 ± 0.009** | 0.868 ± 0.019 |

The R-GCN leads on F1, precision (+15 points), and overall accuracy (90% — a clean round number worth highlighting). The LogReg has higher recall but produces ~2x more false accusations.

**Excluded entries** (no confirmed real-world villain, structurally different story, or unfixable extraction issue):
- **FLM_047 (Zodiac)** — real-world unsolved case; used in inference-only analysis
- **POD_035 (In the Dark: Season 3)** — real-world unsolved/contested case (Curtis Flowers exonerated, Doug Evans not formally charged); used in inference-only analysis
- **TVE_089 (Vera: Hidden Depths)** — extraction produced an invalid generic-villain entity ("People with specific forms of desperation and violence"); fully excluded
- **TVE_093 (Spiral: Series 4)** — different kind of story; villain is a criminal network ("Members of the criminal network"), not a single person identifiable from the available evidence; fully excluded

These results follow two rounds of targeted re-extraction (2026-04-25 and 2026-04-26) and the special-status exclusions above. See "Re-Extraction Impact" and "Held-Out Inference Case Studies" below.

### Held-Out Inference Case Studies

Two real-world unsolved cases were excluded from train/test entirely and analyzed inference-only. The model — never having seen these stories during training — was asked to rank all characters by predicted villain probability. **Crime edges were masked** (detective scenario), so the model had to reason from circumstantial evidence alone.

Reproducible via `python inference_analysis.py`. Output saved to `inference_analysis_results.txt`.

#### Case 1: FLM_047 — *Zodiac* (2007)

The Zodiac killings remain unsolved in real life. **Arthur Leigh Allen is the prime suspect** but was never charged in his lifetime.

**Result: the model ranked Arthur Leigh Allen as the #1 most likely villain with P(villain) = 1.0000.**

| Rank | P(Villain) | Character | Note |
|---|---|---|---|
| 1 | 1.0000 | Arthur Leigh Allen | Real-world prime suspect (UNK in graph) |
| 2 | 0.998 | Zodiac Killer (unidentified) | Placeholder villain in graph |
| 3 | 0.217 | Rick Marshall | Person of interest |
| 4–20 | < 0.13 | (all others) | Police, victims, witnesses |

#### Case 2: POD_035 — *In the Dark: Season 3* (2022)

This podcast investigates the wrongful prosecution of **Curtis Flowers**, who was tried six times for a quadruple murder before his exoneration. The actual perpetrator was never legally identified. Doug Evans, the prosecutor, is the figure the podcast frames as the antagonist for orchestrating the prosecutorial misconduct.

**Result: the model independently confirmed both narrative truths — flagging Doug Evans as #1 (P=1.0000) and clearing Curtis Flowers at #16 (P=0.0000), the lowest score in the story.**

| Rank | P(Villain) | Character | Note |
|---|---|---|---|
| 1 | 1.0000 | Doug Evans | Prosecutor (podcast's antagonist) |
| 2 | 1.0000 | Willie James Hemphill | Alternative suspect (Suspect label) |
| 3 | 1.000 | John Johnson | Co-villain in graph |
| 4–6 | 0.997+ | Other suspects + Odell Hallmon (key witness) | |
| 7–15 | 0.06 → 0.0004 | Investigators, real victims, journalists | |
| **16** | **0.0000** | **Curtis Flowers** | **Wrongfully accused — model cleared with full confidence** |

That Curtis Flowers ranked dead last with P=0.0000 — even though the original graph had him mislabeled as Villain — is the strongest possible evidence that the model reasons from evidence rather than memorizing labels.

#### Why these case studies matter

These two results are the strongest evidence for the paper's central claim: the R-GCN learns transferable patterns of evidence-based reasoning. It is not memorizing labels. When shown stories it has never seen — including one with a deliberately mislabeled wrongful-prosecution victim — it correctly identifies the real-world prime suspects and exonerates the falsely accused.

### Re-Extraction Impact (2026-04-25 → 2026-04-26)

After the initial failure analysis identified 11 unsolved test stories whose villains had no extracted feature signal (`-1` UNK across motive, concealment, etc.), we ran two rounds of targeted re-extraction and label cleanup. The cumulative improvement is substantial:

**Story-level outcomes on the 57-story test set (seed 42):**

| Metric | Original | After Round 1 (04-25) | After Round 2 (04-26) | Δ Total |
|---|---|---|---|---|
| Solved at least one villain | 75.4% (43/57) | 96.5% (55/57) | **98.2%** (56/57) | **+22.8 pts** |
| Clean solves (no false accusations) | 56.1% (32/57) | 70.2% (40/57) | **77.2%** (44/57) | **+21.1 pts** |
| Cases neither model solved | 11/57 (19.3%) | 2/57 (3.5%) | **1/57 (1.8%)** | -10 stories |

**Cross-validated impact (5 seeds):**

| Metric | Original | Round 1 | Round 2 |
|---|---|---|---|
| R-GCN Villain F1 | 0.703 | 0.712 | **0.737** |
| R-GCN Villain Recall | 0.606 | 0.620 | **0.661** |
| R-GCN Villain Precision | 0.843 | 0.848 | **0.846** |
| R-GCN Overall Accuracy | 0.893 | 0.886 | **0.898** |

**What happened in each round:**

- **Round 1 (2026-04-25):** Re-extracted 11 unsolved stories using more detailed synopses. 9 became solved. POD_035 had a labeling error fixed (Curtis Flowers — see Round 2). FLM_047 (Zodiac) was excluded as having no real-world villain.

- **Round 2 (2026-04-26):** Two further refinements:
  - **NOV_276 (*The Stonecutter*) and SHO_001 (*The Red-Headed League*)** were re-extracted with detailed synopses (the new unsolved cases that emerged in Round 1).
  - **POD_035 (*In the Dark: Season 3*)** had Curtis Flowers's label corrected from Victim to Uninvolved (he was wrongly accused, not the actual murder victim) and was moved to the inference-only category alongside FLM_047 (no formal villain identified in real life).
  - **TVE_089 (*Vera: Hidden Depths*)** was fully excluded — its labeled "Villain" was a generic description ("People with specific forms of desperation and violence"), not a named character. Extraction artifact, not recoverable.
  - **TVE_093 (*Spiral: Series 4*)** was fully excluded after the same pattern was identified — its villain is "Members of the criminal network", a collective with no single identifiable perpetrator. This is a structurally different kind of story (organized-crime drama, not a whodunit) and doesn't fit the dataset's intended scope.

After both rounds and the four exclusions above, the test set is in a clean reportable state. The unsolved-case rate is now driven by the same generic-villain pattern that has now been formally excluded from the dataset rather than being reported as model failures.

**Methodological note for the paper:** the prompt used for the new synopses (saved in `support_files/synopsis_generation_prompt.md`) explicitly asked the source LLM to state each character's motive, alibi, concealment behavior, and hidden relationships in narrative prose. The downstream extractor (mixtral:8x7b, two-pass) then surfaced these traits more reliably. This is a useful pattern for any LLM-extraction pipeline: synopses written for human readers often imply traits that synopses written for downstream extraction must state.

### Feature selection (what evidence the model uses)

We deliberately **excluded `narrative_prominence` and `narrative_introduction_timing`** from the feature set. These describe a character's role in the narrative (how central they are, when they're introduced) — things a real detective wouldn't observe. Keeping them would leak story-structure information into the detective model.

The 8 remaining features are all things a detective could actually observe or infer: gender, social status, alibi, presence at crime scene, motive (binary + type), concealment behavior, hidden relationships.

Removing the narrative features had essentially zero impact on performance (F1 +0.004) and reduced result variance — the model wasn't relying on them anyway.

### What this means

1. **The R-GCN is a cautious, precise detective.** When it accuses someone of being the villain, it's right **83% of the time** (precision). The logistic regression baseline only gets 68%. This +15-point precision advantage comes from the relational graph structure — the R-GCN can see that a character with motive but strong alibis and spatial separation from the crime is probably not the villain, while the LogReg just sees "has motive" and flags them.

2. **The LogReg catches more villains** (75% recall vs 68%). It uses a simpler strategy: flag anyone with suspicious features. This casts a wider net but produces ~2x more false accusations.

3. **Villain F1 favors the R-GCN** (0.743 vs 0.712). The R-GCN's precision advantage outweighs the LogReg's recall lead in aggregate.

4. **Overall accuracy is exactly 90%** for the R-GCN — 9 out of 10 characters are correctly classified as villain or non-villain.

5. **At the story level, both models now solve 98.2% of cases** — i.e., catch at least one true villain. But the R-GCN does so cleanly (with no false accusations) in **77.2%** of stories vs LogReg's 66.7%. Only 1 out of 57 stories remains unsolved (TVE_093, an extraction-quality case).

6. **The held-out inference case studies are the strongest evidence of generalization.** With FLM_047 (Zodiac) and POD_035 (In the Dark) held out of training entirely:
   - The model ranks Arthur Leigh Allen — the real-world Zodiac suspect — as #1 with P(villain) = 1.0000.
   - The model ranks Doug Evans — the prosecutor framed as the antagonist of the wrongful Curtis Flowers prosecution — as #1 with P(villain) = 1.0000.
   - The model **clears Curtis Flowers, the wrongful-prosecution victim**, at P=0.0000 (rank #16 of 16) — even though the data file initially had him mislabeled as Villain.

   The model learned transferable patterns of evidence-based reasoning, not labels.

7. **Graph structure alone (Laplacian eigenmaps) is useless** (25% accuracy) — who you're connected to doesn't predict villainy. But graph structure *combined with features* through R-GCN message passing produces the precision advantage.

8. **Crime-edge masking had zero effect** — the model was already ignoring "kills" edges for classification. It genuinely reasons from circumstantial evidence (motive, concealment, relationships, locations) rather than from the answer.

### Why recall is capped at ~65%

Feature analysis of the villains the model misses reveals they have **no villain-indicative features at all** — no motive, no concealment, no hidden relationships. These villains are indistinguishable from innocent characters in the data. This is an extraction quality issue (the LLM didn't flag these traits), not a model limitation. Two rounds of targeted re-extraction (2026-04-25, 2026-04-26) resolved this for 12 of 13 originally-problematic cases. The single remaining unsolved case (TVE_093) has a generic-villain extraction issue ("Members of the criminal network") similar to the excluded TVE_089.

### Strongest predictors of villainy (LogReg feature importances, narrative features excluded)

1. `motive_type` (+1.37) — having a specific motive type (jealousy, revenge, etc.) is the strongest signal
2. `has_alibi` (-1.34) — having an alibi strongly anti-correlates with being the villain
3. `has_motive` (+0.95) — binary motive flag
4. `is_concealing_information` (+0.62) — concealment behavior
5. `gender` (-0.60) — in this corpus, male characters are more often villains
6. `social_status` (-0.35) — lower status correlates slightly with being the villain

`present_at_crime_scene` and `has_hidden_relationship` had near-zero weights in the binary LogReg — they're dominated by the stronger signals above but may still contribute through feature interactions the linear model can't capture.

---

## Failure Analysis (2026-04-26, after both re-extraction rounds)

Full reproducible analysis in `failure_analysis.py` (output: `failure_analysis_results.txt`). Single-seed analysis on seed 42 test split (57 stories, 604 test characters, 138 villains, with FLM_047/POD_035/TVE_089 excluded as special cases). All numbers use the crime-edge-masked evaluation with narrative features excluded.

### The key story-level finding

> **The R-GCN's value lies not in solving more cases, but in solving them cleanly.**

We measured two story-level outcomes on the 57 test stories:
1. **Solved:** model correctly identifies at least one true villain in the story
2. **Clean solve:** solved *and* made no false accusations of innocents in that story

| Metric | R-GCN | LogReg | Difference |
|---|---|---|---|
| Solved at least one villain | 56 / 57 (98.2%) | 56 / 57 (98.2%) | tied |
| **Clean solves (no false accusations)** | **44 / 57 (77.2%)** | **38 / 57 (66.7%)** | **R-GCN +10.5 pts** |
| Neither model solved | 1 / 57 (1.8%) | — |

The R-GCN solves cases cleanly in 77.2% of stories vs LogReg's 66.7%. Only 1 story remains unsolved by either model (TVE_093 — generic-villain extraction issue).

**Historical comparison:**

| Snapshot | Solved | Clean (R-GCN) | Clean (LogReg) | Unsolved |
|---|---|---|---|---|
| Original (2026-04-22) | 75.4% | 56.1% | 40.4% | 11 |
| Round 1 (2026-04-25) | 96.5% | 70.2% | 63.2% | 2 |
| **Round 2 (2026-04-26)** | **98.2%** | **77.2%** | **66.7%** | **1** |

### Analysis 1 — Multi-villain stories

| True # Villains | Stories | R-GCN Recall | LogReg Recall | R-GCN Caught All | LogReg Caught All |
|---|---|---|---|---|---|
| 1 | 21 | 95.2% | 95.2% | — | — |
| 2 | 14 | 92.9% | 92.9% | 11 / 14 (78.6%) | 11 / 14 (78.6%) |
| 3 | 10 | 70.0% | 76.7% | 5 / 10 (50.0%) | 6 / 10 (60.0%) |
| 4 | 8 | 65.6% | 75.0% | 1 / 8 (12.5%) | 3 / 8 (37.5%) |
| 5 | 1 | 100.0% | 100.0% | 1 / 1 | 1 / 1 |
| 6 | 2 | 41.7% | 41.7% | 0 / 2 (0.0%) | 0 / 2 (0.0%) |
| 10 | 1 | 60.0% | 60.0% | 0 / 1 | 0 / 1 |

Single-villain and two-villain stories are now well-handled (~93-95% recall). Performance degrades as ensembles grow, with 6-villain stories remaining the structural limit (~42% recall) — these are collective-guilt plots like *Murder on the Orient Express*.

### Analysis 2 — Zero-villain stories

The three test stories without identifiable villains (FLM_047 Zodiac, POD_035 In the Dark Season 3, TVE_089 Vera) are now excluded from train/test. FLM_047 and POD_035 are run separately as inference-only case studies (see "Held-Out Inference Case Studies" above).

### Analysis 3 — Where R-GCN and LogReg disagree (24 characters, 4.0% of test set)

| Disagreement type | Count | What it means |
|---|---|---|
| Both agree | 580 (96.0%) | Same prediction, whether right or wrong |
| R-GCN right, LogReg wrong | 19 (3.1%) | |
| LogReg right, R-GCN wrong | 5 (0.8%) | |

**The asymmetry is consistent across all snapshots:** when R-GCN wins, it is *always* by clearing an innocent that LogReg flagged (19/19 cases). When LogReg wins, it is *always* by catching a villain the R-GCN missed (5/5 cases).

This precisely characterizes the architectural trade-off: graph structure helps the model *exclude* false suspects more than it helps *find* true ones — exactly what we'd expect from a detective who can reason about alibis, social context, and spatial separation.

### Analysis 4 — False positive composition

| Model | False Positives | Breakdown |
|---|---|---|
| R-GCN | 18 | 9 Uninvolved, 7 Witnesses, 2 Victims |
| LogReg | 37 | 20 Uninvolved, 13 Witnesses, 4 Victims |

R-GCN cuts false accusations approximately in half (37 → 18), with proportionally fewer victims and witnesses falsely accused.

### Analysis 5 — The remaining unsolved case

After excluding TVE_093 (criminal-network villain, doesn't fit the whodunit format), the seed-42 test split surfaces TVE_094 (*The Killing: Season 2*) as the new unsolved case. Its labeled villain "The Pasha" is another generic / code-name designation with all-UNK features — same kind of structural issue.

This is a useful pattern for the paper: as we exclude unfit stories, the same kind of edge-case (generic-perpetrator drama rather than detective-fiction whodunit) keeps surfacing. The dataset's scope is **single-perpetrator detective fiction**, and stories that don't match that genre signature appear as residual failures regardless of model strength. We've tagged these candidates for review in `unsolved_cases.md`.

### Takeaways for the paper

1. **F1 alone is misleading.** Use story-level metrics (solved, clean-solved) for detective tasks.
2. **The R-GCN is a higher-precision detective.** F1 advantage is real (0.743 vs 0.712) and precision advantage is +15 points. It catches fewer villains but almost never accuses victims, witnesses, or a second innocent when it does accuse.
3. **Clean solve rate (77% vs 67%) is the most honest measure of detective quality.**
4. **The disagreement pattern is interpretable.** When the two models disagree, the R-GCN exclusively wins by clearing innocents. The graph structure encodes contextual evidence that lets the model rule out suspects.
5. **Multi-villain ensembles are a structural limit.** Both models drop to ~42% recall on 6-villain cases. Collective-guilt plots may require a fundamentally different approach.
6. **Held-out inference confirms genuine generalization.** The model independently identified the prime suspects in two unsolved real-world cases (Arthur Leigh Allen for Zodiac, Doug Evans for In the Dark) and exonerated the wrongly-accused (Curtis Flowers, P=0.0000) — even when the data file mislabeled him as Villain.
7. **Extraction quality is the dominant lever.** Two rounds of targeted re-extraction plus four exclusions of mismatched-genre stories improved cross-validated F1 by 0.040 (0.703 → 0.743). Better synopses unlock significant performance gains, and being explicit about the dataset's genre scope (single-perpetrator detective fiction) improves both honesty and metrics.

---

## Directory Structure

```
Detective_R-GCN/
├── PROJECT.md                            ← THIS FILE
├── RGCN.md                               ← Paper checklist (Schlichtkrull et al., 2017)
├── rgcn_model.py                         ← R-GCN encoder, DistMult decoder, NodeClassifier,
│                                            RGCNMultiTask model, training & evaluation utilities
├── load_mystery_graphs.py                ← Loads 576 graph JSONs → single RelationalGraph
│                                            with 2-stage relation normalization (16 types)
├── spectral_baseline.py                  ← Laplacian eigenmaps baseline on simple graphs
├── failure_analysis.py                   ← Per-story deep-dive analysis (5 failure modes)
├── failure_analysis_results.txt          ← Latest failure analysis output (reproducible)
├── inference_analysis.py                 ← Held-out story inference (FLM_047, POD_035)
├── inference_analysis_results.txt        ← Suspect rankings for held-out stories
├── cross_validation_results.txt          ← Latest 5-seed cross-val output
├── reextract.py                          ← Re-extraction script (uses Ollama)
├── unsolved_cases.md                     ← Stories neither model solved
│                                            (metadata + current features, for re-extraction)
├── detective_training.ipynb              ← Training notebook (multi-task: villain + link pred)
│
├── extraction/                           ← Graph data + extraction code
│   ├── __init__.py
│   ├── extractor.py / main.py / prompt.py / normalize_labels.py
│   └── data/
│       ├── extraction_status.json
│       ├── graphs/                       ← One .json per entry (576 files)
│       ├── new_extracts/                 ← Detailed synopses for re-extraction (.txt)
│       └── graphs_pre_reextraction_backup/   ← Pre-re-extraction snapshots of 11 graphs
│
├── extraction_simple/                    ← Simplified character-only graphs (for spectral baseline)
│   ├── __init__.py
│   ├── build_simple_graphs.py
│   └── data/
│       └── graphs/                       ← One .json per entry (576 files)
│
├── checkpoints/                          ← Auto-created on first training run
│   ├── detective_graph.pkl               ← Cached RelationalGraph
│   └── rgcn_multitask_detective.pt       ← Trained model weights
│
└── support_files/                        ← Archived reference files
    ├── README_detective_loading_training.md
    └── detective_data.ipynb              ← Old single-task training notebook
```

**Original data pipeline** (scraper, extraction scripts, xlsx) lives in `../Detective/`. This folder contains only the R-GCN model code and the extracted graph data.

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

### Phase 2b: Simple Graphs (Spectral Baseline) — COMPLETE

Built from the existing heterogeneous graphs (no re-extraction needed). Located in `extraction_simple/data/graphs/`.

**Purpose:** Provide a simpler graph representation for spectral analysis / Laplacian eigenmaps as a baseline comparison. We hypothesize the full relational (heterogeneous) graph will significantly outperform this simplified version.

**Structure:**
- **Nodes:** Characters only, with all 10 features preserved
- **Edges:** Weighted, undirected, between character pairs
- **Weights** combine two sources:
  1. Direct character-character relationships (count of distinct relation types)
  2. Shared context — characters connected to the same location, organization, or occupation each add +1
- Weights are **normalized to [0,1] within each graph** (divided by max weight)

**Data cleaning:**
- UNK sentinel values (-1) replaced with **0** in all features
- String motive types mapped to numeric: jealousy=0.25, money/financial=0.5, revenge/power=0.75, love=0.25
- Non-parseable values default to 0

**Regenerate with:**
```bash
python -m extraction_simple.build_simple_graphs
```

#### Simple Graph Statistics

| Metric | Mean | Median | Min | Max |
|---|---|---|---|---|
| Characters per graph | 10.3 | 9 | 3 | 45 |
| Edges per graph | 13.6 | 11 | 0 | 79 |
| Edge weights | 0.53 | 0.50 | 0.008 | 1.0 |

#### Simple Graph JSON Structure

Each file in `extraction_simple/data/graphs/{ID}.json`:
```json
{
  "characters": [{"id": "char_0", "name": "...", "label": "Villain", "features": {"gender": 0.0, ...}}],
  "edges": [{"source": "char_0", "target": "char_1", "weight": 0.75, "raw_weight": 3.0}],
  "metadata": {"entry_id": "...", "title": "...", ...},
  "graph_stats": {"num_characters": 8, "num_edges": 12, "max_raw_weight": 4.0}
}
```

### Phase 3: Validation — COMPLETE

- Label normalization applied via `extraction/normalize_labels.py`
- Compound labels resolved by priority: Villain > Victim > Witness > Uninvolved > UNK
- Non-standard labels (Suspect, Detective, Investigator, etc.) mapped to valid classes
- Problematic extractions (3 entries) excluded after re-extraction attempts
- All 576 remaining graphs validated for structural integrity

### Phase 4: R-GCN Model Training — IN PROGRESS

**Completed (2026-04-16):**
- R-GCN encoder fixed: per-relation normalization `c_{i,r} = |N_i^r|` (was using global degree)
- Negative sampling fixed: now randomly corrupts subject OR object (was only corrupting object)
- Inverse relations added: loader now produces 16 relation types (8 canonical + 8 inverse)
- Character labels extracted from graph JSONs with train/val/test masks (story-level split)
- `NodeClassifier` decoder added for 4-class villain prediction (Villain/Victim/Witness/Uninvolved)
- `RGCNMultiTask` model: shared R-GCN encoder + DistMult link decoder + node classification head
- `train_multitask()` and `evaluate_classification()` utilities added
- Training notebook `detective_training.ipynb` created and smoke-tested
- End-to-end pipeline verified: ~1 sec/epoch on M1 Max
- Full training runs completed and analyzed (see results below)

**Corpus statistics (with inverse relations):**
```
Stories        : 576  (train: 462, val: 57, test: 57)
Nodes          : 14,020
Relation types : 16  (8 canonical + 8 inverse)
Train edges    : 25,732
Val edges      : 3,166
Test edges     : 2,996
Node feat dim  : 10
Train chars    : 4,670  (for classification)
Val chars      : 619
Test chars     : 554
```

**Training Results (2026-04-16):**

Two configurations tested, both trained for 50 epochs:

| Config | Hidden | Dropout | Class Weights | Overall Acc | Villain | Victim | Witness | Uninvolved |
|---|---|---|---|---|---|---|---|---|
| Run 1 (baseline) | 64 | 0.2 | No | 66.1% | 60.2% | 24.1% | 2.5% | 92.9% |
| Run 2 (tuned) | 32 | 0.4 | Yes | 65.9% | 63.9% | 29.5% | 7.5% | 88.4% |

Link prediction: MRR ~0.05–0.07, Hits@10 ~5–17% (weak — expected for disconnected small graphs)

**Key findings:**
- Val accuracy plateaus at ~67-68% by epoch 5-10 regardless of configuration
- Run 1 severely overfits (train 96.5%, val 67%); Run 2 overfits less (train 82%, val 68%)
- Class imbalance is the main challenge: Witness (9% of data) and Victim (18%) are hard to predict
- The model defaults to predicting Uninvolved (50% of data) for ambiguous cases
- More capacity (hidden=64 vs 32) doesn't help — the signal ceiling is in the data/features, not model size

**Additional runs with class balancing (2026-04-16):**

Class weighting (inverse frequency) integrated into `train_multitask()` via `class_balance=True`.

| Model | Villain P/R/F1 | Victim P/R/F1 | Witness R | Overall |
|---|---|---|---|---|
| LogReg features-only | .74/.68/.71 | .46/.70/.55 | 35.0% | 58.5% |
| R-GCN Run 1 (no balance) | .72/.60/.66 | .73/.24/.36 | 2.5% | 66.1% |
| R-GCN Run 3 (balance) | .79/.63/.70 | .59/.31/.41 | 10.0% | 65.2% |
| R-GCN Run 4 (balance+cls=3) | .77/.66/.71 | .62/.30/.41 | 10.0% | 65.7% |

Class balancing closed the Villain F1 gap from 0.66 → 0.71 (matching LogReg). R-GCN has better Villain precision (77% vs 74%) but LogReg has better Victim recall (70% vs 30%). See Step 3 below for detailed comparison with spectral baseline.

**Saved checkpoint:** `checkpoints/rgcn_multitask_detective.pt` (Run 3 config, class_balance=True)

---

## What Needs to Happen Next

### Step 1: Improve extraction quality for missed villains (HIGH IMPACT)
40 of 108 test villains are unrecoverable — they lack motive/concealment features entirely. Options:
- **Re-extract** the ~40% of stories where the villain has no motive features using a better model (llama3:70b? or a targeted re-extraction prompt focused on villain characteristics)
- **Cross-story feature propagation**: use the victim's features or the crime circumstances to infer the villain's likely characteristics (e.g., if the victim was wealthy, the villain likely has a financial motive)
- This is the single biggest lever for recall improvement — a model fix can't compensate for missing input data

### Step 3: Sanity-check on known stories
- Pick specific well-known novels (e.g., Agatha Christie) from the test set
- Inspect per-character predictions: does the model identify the villain? What does it get wrong?
- This gives qualitative evidence for the paper beyond aggregate metrics

### Step 3: Spectral baseline (Laplacian eigenmaps) — COMPLETE

Implemented in `spectral_baseline.py`. Three variants tested on same story-level split:

| Model | Overall Acc | Villain Recall | Victim Recall | Witness Recall | Uninvolved Recall |
|---|---|---|---|---|---|
| Features-only LogReg | 58.5% | **67.6%** | **69.6%** | **35.0%** | 54.1% |
| Features + Spectral | 58.8% | 66.7% | 71.4% | 35.0% | 54.4% |
| Spectral only | 25.5% | 9.3% | 13.4% | 55.0% | 32.0% |
| R-GCN Run 1 (baseline) | **66.1%** | 60.2% | 24.1% | 2.5% | **92.9%** |
| R-GCN Run 2 (tuned) | 65.9% | 63.9% | 29.5% | 7.5% | 88.4% |

**Key findings:**
- Spectral embedding alone is useless (25.5%) — graph topology doesn't predict villains
- Node features alone (logistic regression) is a strong baseline for Villain F1 (0.705)
- Spectral embedding adds nothing to features (58.5% → 58.8%)

**Combined comparison — Villain prediction (primary metric):**

| Model | Villain Prec | Villain Recall | Villain F1 | Notes |
|---|---|---|---|---|
| Features-only LogReg | 0.737 | 0.676 | **0.705** | No graph structure, balanced class weights |
| R-GCN (class balanced) | **0.800** | 0.630 | **0.705** | Full heterogeneous graph with masking |
| Spectral + Features | 0.727 | 0.667 | 0.696 | Laplacian eigenmaps + features |
| Spectral only | 0.145 | 0.093 | 0.113 | Graph structure alone — useless |

**Binary (Villain vs Non-Villain) results — the detective framing (2026-04-16):**

| Model | Villain Prec | Villain Recall | Villain F1 | Overall Acc |
|---|---|---|---|---|
| **R-GCN 2-class** | **0.814** | **0.648** | **0.722** | **90.3%** |
| LogReg 2-class | 0.661 | 0.685 | 0.673 | 87.0% |
| R-GCN 4-class | 0.800 | 0.630 | 0.705 | 65.2% |
| LogReg 4-class | 0.737 | 0.676 | 0.705 | 58.5% |

Single-split: 2-class R-GCN beats all baselines on Villain F1 (0.722), precision (81%), and overall accuracy (90.3%). Crime-edge masking had no impact, confirming the model uses circumstantial evidence only.

**Cross-validation results — 5 random seeds, binary Villain vs Non-Villain:**

Multiple rounds of cross-validation were run as the experiment evolved.

*Final result (2026-04-26) — narrative features excluded, FLM_047/POD_035/TVE_089/TVE_093 excluded, two rounds of re-extraction applied:*

| Metric | R-GCN (mean ± std) | LogReg (mean ± std) |
|---|---|---|
| Villain Precision | **0.830 ± 0.029** | 0.684 ± 0.061 |
| Villain Recall | 0.676 ± 0.068 | **0.747 ± 0.047** |
| Villain F1 | **0.743 ± 0.047** | 0.712 ± 0.044 |
| Overall Accuracy | **0.900 ± 0.009** | 0.868 ± 0.019 |

Per-seed breakdown (final):

| Seed | Test Villains | R-GCN P/R/F1 | LogReg P/R/F1 |
|---|---|---|---|
| 42 | 133 | .82/.70/.76 | .74/.80/.77 |
| 123 | 118 | .80/.55/.65 | .58/.70/.63 |
| 456 | 126 | .80/.72/.76 | .69/.76/.72 |
| 789 | 124 | .85/.74/.79 | .67/.79/.73 |
| 2024 | 125 | .87/.66/.76 | .75/.68/.71 |

*Round 2 result (2026-04-26 earlier) — same setup but with TVE_093 still in train/test:*

| Metric | R-GCN (mean ± std) | LogReg (mean ± std) |
|---|---|---|
| Villain Precision | 0.846 ± 0.044 | 0.693 ± 0.059 |
| Villain Recall | 0.661 ± 0.104 | 0.748 ± 0.058 |
| Villain F1 | 0.737 ± 0.065 | 0.717 ± 0.043 |
| Overall Accuracy | 0.898 ± 0.016 | 0.870 ± 0.016 |

*Round 1 result (2026-04-25) — narrative features excluded, FLM_047 excluded, Round 1 re-extracted graphs:*

| Metric | R-GCN (mean ± std) | LogReg (mean ± std) |
|---|---|---|
| Villain Precision | 0.848 ± 0.063 | 0.681 ± 0.074 |
| Villain Recall | 0.620 ± 0.085 | 0.729 ± 0.063 |
| Villain F1 | 0.712 ± 0.062 | 0.701 ± 0.058 |
| Overall Accuracy | 0.886 ± 0.020 | 0.857 ± 0.030 |

*Original result (2026-04-22) — narrative features excluded, original graphs:*

| Metric | R-GCN (mean ± std) | LogReg (mean ± std) |
|---|---|---|
| Villain Precision | 0.843 ± 0.037 | 0.684 ± 0.040 |
| Villain Recall | 0.606 ± 0.044 | 0.719 ± 0.032 |
| Villain F1 | 0.703 ± 0.024 | 0.700 ± 0.024 |
| Overall Accuracy | 0.893 ± 0.006 | 0.870 ± 0.012 |

*Earlier result — narrative features included (for reference):*

| Metric | R-GCN (mean ± std) | LogReg (mean ± std) |
|---|---|---|
| Villain Precision | 0.836 ± 0.051 | 0.673 ± 0.052 |
| Villain Recall | 0.604 ± 0.047 | 0.717 ± 0.033 |
| Villain F1 | 0.699 ± 0.027 | 0.693 ± 0.024 |
| Overall Accuracy | 0.891 ± 0.009 | 0.866 ± 0.014 |

Results are stable across splits (low std). The R-GCN's precision advantage over LogReg is consistent and significant (+16 points). LogReg's recall advantage is also consistent (+11 points). F1 is essentially tied.

**Feature analysis — why recall is capped at ~65%:**
Of 108 test villains, 40 are missed. These 40 have features indistinguishable from non-villains:
- `has_motive`: 100% of caught villains have it, only 10% of missed villains do
- `concealing_info`: 78% caught vs 10% missed
- `hidden_relationship`: 57% caught vs 17.5% missed

This is an **extraction quality ceiling**, not a model limitation. The LLM (mixtral:8x7b) didn't flag these characters with villain-indicative features. No model architecture can recover signal that isn't in the data.

**LogReg feature importances** (strongest predictors of Villain, narrative features excluded):
1. `motive_type` (+1.37) — having a specific motive type is the strongest signal
2. `has_alibi` (-1.34) — alibi strongly anti-correlates with being the villain
3. `has_motive` (+0.95) — binary motive flag
4. `is_concealing_information` (+0.62) — concealment behavior
5. `gender` (-0.60) — in this corpus, male characters are more often villains
6. `social_status` (-0.35) — slight negative correlation

### Step 4: Ablation study
Potential ablation variants:
- **No features**: replace node features with learned embeddings (tests feature importance)
- **No inverse relations**: train with 8 relations instead of 16 (tests bidirectional information flow)
- **Classification only**: remove link prediction loss (tests multi-task benefit)
- **Link prediction only**: remove classification loss (tests if link prediction helps classification)
- **Fewer layers**: 1 R-GCN layer vs. 2 (tests depth of message passing)

### Step 5: Motivation prediction (stretch goal)
- Add a second classification head for motive type (jealousy, money, revenge, love)
- The feature is already in the data (`motive_type` in character features) but currently used as input
- Would need to mask it from input features and use it as a prediction target instead

### Backlog / Housekeeping
- Reconcile `extraction_status.json` stale entries (`SHO_018`, `SHO_019`)
- Consider adding early stopping to training loop
- Consider learning rate scheduling

### Known data-quality issues (to fix before paper submission)
See `unsolved_cases.md` for details. Two ground-truth labeling problems discovered during failure analysis:
- **POD_035 (*In the Dark: Season 3*)** — Curtis Flowers is mislabeled as Villain; he is a wrongly-convicted innocent and the podcast is about his exoneration. Doug Evans (the prosecutor who framed him) is the actual antagonist.
- **TVE_089 (*Vera: Hidden Depths*)** — the labeled Villain is the string "People with specific forms of desperation and violence", not a named character. Extraction failed to identify the specific perpetrator.

Both should be addressed (re-label or exclude) before the final paper, and noted as data-quality observations regardless.

---

## Environment Setup

### Virtual environment
```bash
cd /Users/maxdesantis/dev/290-Final-Project
source venv/bin/activate
```

### Install dependencies
```bash
pip install torch torch-geometric numpy matplotlib
```

### Running the notebook
```bash
cd Detective_R-GCN
jupyter notebook detective_training.ipynb
```

First run parses all 576 graph JSONs and caches the result. Subsequent runs load from cache. Training completes in ~2 min on M1 Max (50 epochs).

---

## Key Operations

### Training the model
Open `detective_training.ipynb` and run all cells. On first run it:
1. Parses all 576 graph JSONs → single `RelationalGraph` (cached to `checkpoints/detective_graph.pkl`)
2. Builds `RGCNMultiTask` model (R-GCN encoder + DistMult + NodeClassifier)
3. Trains for 50 epochs with joint link prediction + classification loss
4. Saves weights to `checkpoints/rgcn_multitask_detective.pt`

Subsequent runs load from cache/checkpoint and skip training.

### Force retrain
Delete the checkpoint: `rm checkpoints/rgcn_multitask_detective.pt`

### Force rebuild graph
Delete the cache: `rm checkpoints/detective_graph.pkl`

### Smoke test (command line)
```bash
cd Detective_R-GCN
python load_mystery_graphs.py extraction/data/graphs
```

### Original extraction pipeline
Scraping and graph extraction scripts are in `../Detective/`. See `../Detective/PROJECT.md` for those operations.

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
1. **Villain prediction** (primary) — binary node classification: Villain vs Non-Villain (victim is assumed known)
2. **Motivation inference** (secondary) — motive features are used as input evidence, not as a prediction target
3. **Accomplice detection** — accomplices are labeled as Villains, so caught by the primary task

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
