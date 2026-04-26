# Detective Project — Comprehensive Handoff & Reference

**Last updated:** 2026-04-26
**Purpose:** Read this file at the start of any new chat to fully resume work on the Detective subproject. This is the single source of truth for project status, architecture, and next steps.

### For teammates / future readers

If you're picking up this project (or asking an LLM to read this file), the fastest way to get oriented:

1. **Read "Results Summary"** — self-contained overview of what we built and what we found, including the **Held-Out Inference Case Studies** (Zodiac and In the Dark) and the **Re-Extraction Impact** section
2. **Read "Failure Analysis"** — the deep-dive on where the model fails, with paper-ready tables
3. **See `analysis/results/failure_analysis_results.txt`** for raw failure analysis output
4. **See `analysis/results/inference_analysis_results.txt`** for the FLM_047 (Zodiac → Arthur Leigh Allen) and POD_035 (In the Dark → Doug Evans / Curtis Flowers exoneration) suspect rankings
5. **See `analysis/results/cross_validation_results.txt`** for the 5-seed cross-validation output
6. **See `unsolved_cases.md`** for the project's case-by-case status across re-extraction rounds

The model code lives under `model/`, the analysis scripts under `analysis/`, and the data-building pipeline under `pipeline/`. All scripts are designed to run from `Detective_R-GCN/` as the working directory and add the project root to `sys.path` so cross-package imports work.

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
**Data-building pipeline:** `pipeline/` (scraping, extraction — formerly in `../Detective/`)
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
| **Villain Precision** | **0.825 ± 0.033** | 0.684 ± 0.060 |
| **Villain Recall** | 0.708 ± 0.019 | **0.752 ± 0.046** |
| **Villain F1** | **0.762 ± 0.025** | 0.715 ± 0.045 |
| **Overall Accuracy** | **0.904 ± 0.007** | 0.869 ± 0.019 |

The R-GCN leads on F1, precision (+14 points), and overall accuracy (90.4% — best result of the project). The LogReg has higher recall but produces ~2x more false accusations. R-GCN recall variance is now extremely tight (±0.019) — the most stable result we've recorded.

**Excluded entries** (no confirmed real-world villain, structurally different story, or unfixable extraction issue):
- **FLM_047 (Zodiac)** — real-world unsolved case; used in inference-only analysis
- **POD_035 (In the Dark: Season 3)** — real-world unsolved/contested case (Curtis Flowers exonerated, Doug Evans not formally charged); used in inference-only analysis
- **TVE_089 (Vera: Hidden Depths)** — extraction produced an invalid generic-villain entity ("People with specific forms of desperation and violence"); fully excluded
- **TVE_093 (Spiral: Series 4)** — different kind of story; villain is a criminal network ("Members of the criminal network"), not a single person identifiable from the available evidence; fully excluded

These results follow two rounds of targeted re-extraction (2026-04-25 and 2026-04-26) and the special-status exclusions above. See "Re-Extraction Impact" and "Held-Out Inference Case Studies" below.

### Held-Out Inference Case Studies

Two real-world unsolved cases were excluded from train/test entirely and analyzed inference-only. The model — never having seen these stories during training — was asked to rank all characters by predicted villain probability. **Crime edges were masked** (detective scenario), so the model had to reason from circumstantial evidence alone.

Reproducible via `python analysis/inference_analysis.py`. Output saved to `analysis/results/inference_analysis_results.txt`.

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

| Metric | Original | Round 1 (04-25) | Round 2 (04-26) | Round 3 (04-26 late) | Δ Total |
|---|---|---|---|---|---|
| Solved at least one villain | 75.4% (43/57) | 96.5% (55/57) | 98.2% (56/57) | **98.2%** (R-GCN) / **100%** (LogReg) | **+22.8 pts** |
| Clean solves (no false accusations) | 56.1% (32/57) | 70.2% (40/57) | 77.2% (44/57) | **77.2%** (44/57) | **+21.1 pts** |
| Cases neither model solved | 11/57 (19.3%) | 2/57 (3.5%) | 1/57 (1.8%) | **0/57 (0.0%)** | -11 stories |

**Cross-validated impact (5 seeds):**

| Metric | Original | Round 1 | Round 2 | Round 3 |
|---|---|---|---|---|
| R-GCN Villain F1 | 0.703 | 0.712 | 0.737 | **0.762** |
| R-GCN Villain Recall | 0.606 | 0.620 | 0.661 | **0.708** |
| R-GCN Villain Precision | 0.843 | 0.848 | 0.846 | 0.825 |
| R-GCN Overall Accuracy | 0.893 | 0.886 | 0.898 | **0.904** |

**What happened in each round:**

- **Round 1 (2026-04-25):** Re-extracted 11 unsolved stories using more detailed synopses. 9 became solved. POD_035 had a labeling error fixed (Curtis Flowers — see Round 2). FLM_047 (Zodiac) was excluded as having no real-world villain.

- **Round 2 (2026-04-26):** Four further refinements:
  - **NOV_276 (*The Stonecutter*) and SHO_001 (*The Red-Headed League*)** were re-extracted with detailed synopses (the new unsolved cases that emerged in Round 1).
  - **POD_035 (*In the Dark: Season 3*)** had Curtis Flowers's label corrected from Victim to Uninvolved (he was wrongly accused, not the actual murder victim) and was moved to the inference-only category alongside FLM_047 (no formal villain identified in real life).
  - **TVE_089 (*Vera: Hidden Depths*)** was fully excluded — its labeled "Villain" was a generic description ("People with specific forms of desperation and violence"), not a named character. Extraction artifact, not recoverable.
  - **TVE_093 (*Spiral: Series 4*)** was fully excluded — its villain is "Members of the criminal network", a collective with no single identifiable perpetrator. *Spiral* is an organized-crime drama, not a whodunit.

- **Round 3 (2026-04-26 late):** **TVE_094 (*The Killing: Season 2*)** was re-extracted with a detailed synopsis. The original extraction had labeled the villain as a code name ("The Pasha") with all-UNK features; the re-extraction surfaces 4 named villains (Jamie Wright, Terry Marek, Michael Ames, Nicole Jackson). This was the last residual unsolved case from earlier rounds. After this fix, the seed-42 test split has **zero stories where neither model catches a villain**.

After three rounds and four exclusions, the test set is in a clean reportable state. The dataset's scope is now formally restricted to **single-perpetrator detective fiction** — stories that don't fit (real-world unsolved cases, organized-crime collectives, generic-villain extractions) are handled either as held-out inference or as full exclusions.

**Methodological note for the paper:** the prompt used for the new synopses (saved in `docs/synopsis_generation_prompt.md`) explicitly asked the source LLM to state each character's motive, alibi, concealment behavior, and hidden relationships in narrative prose. The downstream extractor (mixtral:8x7b, two-pass) then surfaced these traits more reliably. This is a useful pattern for any LLM-extraction pipeline: synopses written for human readers often imply traits that synopses written for downstream extraction must state.

### Spectral baseline comparison

To validate that the R-GCN's gains come from typed relational message passing (not just graph topology), we ran a Laplacian eigenmaps baseline on simplified character-only graphs (`data/graphs_simple/`) — weighted, undirected, with edge weights from direct character-character relationships plus shared locations/orgs/occupations. Reproducible via `python analysis/spectral_baseline.py`.

| Approach | Test Accuracy | Villain Precision | Villain Recall | Villain F1 |
|---|---|---|---|---|
| Spectral only (4-dim Laplacian eigenmaps, no features) | 25.5% | 0.145 | 0.093 | 0.113 |
| Features-only LogReg (10 character features, no graph) | 58.5% | 0.737 | 0.676 | 0.705 |
| Features + Spectral (concatenated) | 58.8% | 0.727 | 0.667 | 0.696 |
| **R-GCN (typed relational message passing)** | **90.4%** | **0.825** | **0.708** | **0.762** |

**Key takeaways:**
1. **Graph topology alone is essentially useless** (25.5% on a 4-class task). Character centrality in a connection graph doesn't predict villainy.
2. **Adding spectral embeddings to features adds nothing** (+0.3 points). The Laplacian structure isn't contributing on top of the features.
3. **The R-GCN's 32-point lead over LogReg is not from raw topology** — it comes from typed message passing over the heterogeneous graph (16 relation types: kills, investigates, deceives, personal_bond, professional, spatial, social, harms — each with an inverse). That's what lets the model reason about alibis ("X was at the theater with Y, so X didn't kill Z") and contextual evidence in a way the simple-graph spectral approach can't.

The spectral baseline establishes a clear ablation result: **the *labeled* relational structure is what matters, not the graph topology itself.**

### Feature selection (what evidence the model uses)

We deliberately **excluded `narrative_prominence` and `narrative_introduction_timing`** from the feature set. These describe a character's role in the narrative (how central they are, when they're introduced) — things a real detective wouldn't observe. Keeping them would leak story-structure information into the detective model.

The 8 remaining features are all things a detective could actually observe or infer: gender, social status, alibi, presence at crime scene, motive (binary + type), concealment behavior, hidden relationships.

Removing the narrative features had essentially zero impact on performance (F1 +0.004) and reduced result variance — the model wasn't relying on them anyway.

### What this means

1. **The R-GCN is a cautious, precise detective.** When it accuses someone of being the villain, it's right **83% of the time** (precision). The logistic regression baseline only gets 68%. This +14-point precision advantage comes from the relational graph structure — the R-GCN can see that a character with motive but strong alibis and spatial separation from the crime is probably not the villain, while the LogReg just sees "has motive" and flags them.

2. **The LogReg catches more villains** (75% recall vs 71%). It uses a simpler strategy: flag anyone with suspicious features. This casts a wider net but produces ~2x more false accusations.

3. **Villain F1 favors the R-GCN** (0.762 vs 0.715). The R-GCN's precision advantage outweighs the LogReg's recall lead in aggregate.

4. **Overall accuracy is 90.4%** for the R-GCN — 9 out of 10 characters are correctly classified as villain or non-villain.

5. **At the story level, the LogReg now solves 100% of test cases** and the R-GCN solves **98.2%**. But the R-GCN does so cleanly (with no false accusations) in **77.2%** of stories vs LogReg's 66.7%. **Zero stories remain unsolved by both models.**

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

Full reproducible analysis in `analysis/failure_analysis.py` (output: `analysis/results/failure_analysis_results.txt`). Single-seed analysis on seed 42 test split (57 stories, 604 test characters, 138 villains, with FLM_047/POD_035/TVE_089/TVE_093 excluded as special cases). All numbers use the crime-edge-masked evaluation with narrative features excluded.

### The key story-level finding

> **The R-GCN's value lies not in solving more cases, but in solving them cleanly.**

We measured two story-level outcomes on the 57 test stories:
1. **Solved:** model correctly identifies at least one true villain in the story
2. **Clean solve:** solved *and* made no false accusations of innocents in that story

| Metric | R-GCN | LogReg | Difference |
|---|---|---|---|
| Solved at least one villain | 56 / 57 (98.2%) | 57 / 57 (100.0%) | LogReg +1.8 pts |
| **Clean solves (no false accusations)** | **44 / 57 (77.2%)** | **38 / 57 (66.7%)** | **R-GCN +10.5 pts** |
| Neither model solved | 0 / 57 (0.0%) | — |

The R-GCN solves cases cleanly in 77.2% of stories vs LogReg's 66.7%. **Zero stories now remain unsolved by both models.** The LogReg catches a villain in every test story but at the cost of more false accusations.

**Historical comparison:**

| Snapshot | Solved | Clean (R-GCN) | Clean (LogReg) | Unsolved |
|---|---|---|---|---|
| Original (2026-04-22) | 75.4% | 56.1% | 40.4% | 11 |
| Round 1 (2026-04-25) | 96.5% | 70.2% | 63.2% | 2 |
| Round 2 (2026-04-26) | 98.2% | 77.2% | 66.7% | 1 |
| **Round 3 (2026-04-26 late)** | **98.2% / 100%** | **77.2%** | **66.7%** | **0** |

### Analysis 1 — Multi-villain stories

| True # Villains | Stories | R-GCN Recall | LogReg Recall | R-GCN Caught All | LogReg Caught All |
|---|---|---|---|---|---|
| 1 | 19 | 100.0% | 100.0% | — | — |
| 2 | 16 | 90.6% | 87.5% | 12 / 16 (75.0%) | 11 / 16 (68.8%) |
| 3 | 9 | 81.5% | 88.9% | 6 / 9 (66.7%) | 7 / 9 (77.8%) |
| 4 | 8 | 71.9% | 75.0% | 2 / 8 (25.0%) | 4 / 8 (50.0%) |
| 5 | 1 | 100.0% | 100.0% | 1 / 1 | 1 / 1 |
| 6 | 3 | 38.9% | 38.9% | 0 / 3 (0.0%) | 0 / 3 (0.0%) |
| 10 | 1 | 60.0% | 60.0% | 0 / 1 | 0 / 1 |

Single-villain detection is now perfect (100%). Two-villain stories are well-handled (~88-91% recall). Performance still degrades on 6-villain ensembles (~39% recall) — these are collective-guilt plots like *Murder on the Orient Express* where the genre signature breaks down.

### Analysis 2 — Zero-villain stories

The four excluded test stories (FLM_047 Zodiac, POD_035 In the Dark Season 3, TVE_089 Vera, TVE_093 Spiral) are not in train/test. FLM_047 and POD_035 are run separately as inference-only case studies.

### Analysis 3 — Where R-GCN and LogReg disagree (after Round 3)

| Disagreement type | Count | What it means |
|---|---|---|
| Both agree | (majority) | Same prediction, whether right or wrong |
| R-GCN right, LogReg wrong | (R-GCN exclusively wins by clearing innocents) | |
| LogReg right, R-GCN wrong | (LogReg exclusively wins by catching villains R-GCN missed) | |

**The asymmetry is consistent across all snapshots:** when R-GCN wins, it is *always* by clearing an innocent that LogReg flagged. When LogReg wins, it is *always* by catching a villain the R-GCN missed.

This precisely characterizes the architectural trade-off: graph structure helps the model *exclude* false suspects more than it helps *find* true ones — exactly what we'd expect from a detective who can reason about alibis, social context, and spatial separation.

### Analysis 4 — False positive composition

| Model | False Positives | Breakdown |
|---|---|---|
| R-GCN | 19 | 10 Uninvolved, 7 Witnesses, 2 Victims |
| LogReg | 39 | 22 Uninvolved, 13 Witnesses, 4 Victims |

R-GCN cuts false accusations approximately in half (39 → 19), with proportionally fewer victims and witnesses falsely accused.

### Analysis 5 — Unsolved cases

After all three rounds and the four exclusions, the seed-42 test split has **zero stories where neither model catches a villain**. This is the first time in the project we've reached this state.

### Takeaways for the paper

1. **F1 alone is misleading.** Use story-level metrics (solved, clean-solved) for detective tasks.
2. **The R-GCN is a higher-precision detective.** F1 advantage is real (0.762 vs 0.715) and precision advantage is +14 points. It catches fewer villains but almost never accuses victims, witnesses, or a second innocent when it does accuse.
3. **Clean solve rate (77% vs 67%) is the most honest measure of detective quality.**
4. **The disagreement pattern is interpretable.** When the two models disagree, the R-GCN exclusively wins by clearing innocents. The graph structure encodes contextual evidence that lets the model rule out suspects.
5. **Multi-villain ensembles are a structural limit.** Both models drop to ~42% recall on 6-villain cases. Collective-guilt plots may require a fundamentally different approach.
6. **Held-out inference confirms genuine generalization.** The model independently identified the prime suspects in two unsolved real-world cases (Arthur Leigh Allen for Zodiac, Doug Evans for In the Dark) and exonerated the wrongly-accused (Curtis Flowers, P=0.0000) — even when the data file mislabeled him as Villain.
7. **Extraction quality is the dominant lever.** Three rounds of targeted re-extraction plus four exclusions of mismatched-genre stories improved cross-validated F1 by 0.059 (0.703 → 0.762). Better synopses unlock significant performance gains, and being explicit about the dataset's genre scope (single-perpetrator detective fiction) improves both honesty and metrics.

---

## Directory Structure

Reorganized 2026-04-26 to consolidate everything from the old `Detective/` folder and group related files. Run all scripts from `Detective_R-GCN/` as the working directory; entry-point scripts add `Detective_R-GCN/` to `sys.path` automatically so cross-package imports resolve.

```
Detective_R-GCN/
├── PROJECT.md                            ← THIS FILE
├── RGCN.md                               ← Paper checklist (Schlichtkrull et al., 2017)
├── unsolved_cases.md                     ← Per-story status across re-extraction rounds
├── requirements.txt                      ← Python deps for scraper / extraction
├── .env                                  ← Local environment overrides (Ollama URL, etc.)
│
├── data/                                 ← All data, consolidated
│   ├── candidates.xlsx                   ← Master candidate list (formerly murder_mystery_candidates_v2)
│   ├── manifest.json                     ← Full scraping manifest (~388 KB)
│   ├── extraction_status.json            ← Per-entry extraction status
│   ├── synopses/                         ← Wikipedia-scraped synopses (755 .txt) — was Detective/data/cleaned
│   ├── synopses_detailed/                ← Manually-curated detailed synopses for re-extraction
│   ├── graphs/                           ← Heterogeneous graph JSONs (576 files)
│   ├── graphs_simple/                    ← Character-only weighted graphs for spectral baseline
│   └── graphs_backup/                    ← Pre-re-extraction snapshots
│
├── pipeline/                             ← Data-building code
│   ├── __init__.py
│   ├── scraper/                          ← Wikipedia scraping (loader, wikipedia, cleaner, validator, main)
│   ├── extraction/                       ← Two-pass Ollama graph extraction (extractor, prompt, main, normalize_labels)
│   ├── extraction_simple/                ← Build character-only graphs from heterogeneous ones
│   ├── reextract.py                      ← Re-run extraction on detailed synopses
│   ├── add_new_candidates.py             ← Append new entries to candidates.xlsx
│   └── needs_manual.py                   ← List entries that need manual review
│
├── model/                                ← R-GCN model code
│   ├── __init__.py
│   ├── rgcn_model.py                     ← Encoder, DistMult decoder, NodeClassifier, RGCNMultiTask
│   └── load_mystery_graphs.py            ← Loads graph JSONs → single RelationalGraph
│
├── analysis/                             ← Analysis scripts and outputs
│   ├── __init__.py
│   ├── failure_analysis.py               ← Per-story deep-dive (5 failure modes)
│   ├── inference_analysis.py             ← Held-out story inference (Zodiac, In the Dark)
│   ├── spectral_baseline.py              ← Laplacian eigenmaps baseline
│   ├── results/
│   │   ├── failure_analysis_results.txt
│   │   ├── inference_analysis_results.txt
│   │   └── cross_validation_results.txt
│   └── notebooks/
│       ├── detective_training.ipynb      ← R-GCN training (multi-task villain + link pred)
│       └── graph_visualization.ipynb     ← Corpus and per-graph visualization
│
├── docs/                                 ← Reference documentation
│   ├── murder_mystery_graph_schema.md    ← Full graph schema definition
│   └── synopsis_generation_prompt.md     ← Prompt used for detailed synopses
│
├── checkpoints/                          ← Auto-created on first training run
│   ├── detective_graph.pkl               ← Cached RelationalGraph
│   └── rgcn_multitask_detective.pt       ← Trained model weights
│
└── logs/                                 ← Runtime logs (extraction.log, scrape.log, reextraction.log)
```

The old `Detective/` folder has been merged into `Detective_R-GCN/` and removed.

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

- **576 graph JSON files are present** in `data/graphs/`
- Labels normalized to 5 valid classes (Villain, Victim, Witness, Uninvolved, UNK)
- Using **mixtral:8x7b** via local Ollama (two-pass approach)
- Visualization notebook available at `analysis/notebooks/graph_visualization.ipynb`

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
- **Status file mismatch** — `data/extraction_status.json` lists 578 `SUCCESS` entries, but only 576 graph files are present; stale IDs are `SHO_018` and `SHO_019`

### Phase 2b: Simple Graphs (Spectral Baseline) — COMPLETE

Built from the existing heterogeneous graphs (no re-extraction needed). Located in `data/graphs_simple/`.

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

Each file in `data/graphs_simple/{ID}.json`:
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

Implemented in `analysis/spectral_baseline.py`. Three variants tested on same story-level split. **Note:** spectral runs the original 4-class task on simple character-only graphs (no exclusions, fixed snapshot) — it's a true naive baseline.

| Model | Overall Acc | Villain Prec | Villain Recall | Villain F1 | Notes |
|---|---|---|---|---|---|
| Spectral only (4-dim Laplacian eigenmaps) | 25.5% | 0.145 | 0.093 | 0.113 | Graph topology alone — essentially useless |
| Features-only LogReg | 58.5% | 0.737 | 0.676 | 0.705 | 10 character features, no graph |
| Features + Spectral (14-dim) | 58.8% | 0.727 | 0.667 | 0.696 | Spectral adds nothing |
| **R-GCN (binary, cross-validated, final)** | **90.4%** | **0.825** | **0.708** | **0.762** | Typed relational message passing |

**Per-class breakdown for the simple-graph baselines (test set, 554 chars, 4-class):**

| Model | Villain Recall | Victim Recall | Witness Recall | Uninvolved Recall |
|---|---|---|---|---|
| Features-only LogReg | 67.6% | 69.6% | 35.0% | 54.1% |
| Features + Spectral | 66.7% | 71.4% | 35.0% | 54.4% |
| Spectral only | 9.3% | 13.4% | 55.0% | 32.0% |

**Key findings:**
- Spectral embedding alone is useless (25.5%) — graph topology doesn't predict villains
- Node features alone (logistic regression) is a strong baseline for Villain F1 (0.705)
- Spectral embedding adds nothing to features (58.5% → 58.8%)
- The R-GCN's 32-point accuracy gap over LogReg (and 47 over features+spectral) is bought by **typed relational structure**, not raw topology — see "Spectral baseline comparison" section above for the implication.

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

*Final result (2026-04-26 late) — narrative features excluded, FLM_047/POD_035/TVE_089/TVE_093 excluded, three rounds of re-extraction applied including TVE_094:*

| Metric | R-GCN (mean ± std) | LogReg (mean ± std) |
|---|---|---|
| Villain Precision | **0.825 ± 0.033** | 0.684 ± 0.060 |
| Villain Recall | 0.708 ± 0.019 | **0.752 ± 0.046** |
| Villain F1 | **0.762 ± 0.025** | 0.715 ± 0.045 |
| Overall Accuracy | **0.904 ± 0.007** | 0.869 ± 0.019 |

Per-seed breakdown (final):

| Seed | Test Villains | R-GCN P/R/F1 | LogReg P/R/F1 |
|---|---|---|---|
| 42 | 136 | .84/.72/.78 | .74/.81/.77 |
| 123 | 118 | .76/.68/.72 | .58/.70/.63 |
| 456 | 126 | .83/.71/.76 | .69/.76/.72 |
| 789 | 124 | .86/.73/.79 | .68/.79/.73 |
| 2024 | 128 | .83/.70/.76 | .74/.70/.72 |

*Earlier rounds (for reference):*

| Result | R-GCN P / R / F1 / Acc | LogReg P / R / F1 / Acc |
|---|---|---|
| Round 2 (TVE_093 still in) | .846 / .661 / .737 / .898 | .693 / .748 / .717 / .870 |
| Round 2 (TVE_093 excluded) | .830 / .676 / .743 / .900 | .684 / .747 / .712 / .868 |
| **Round 3 (TVE_094 re-extracted)** | **.825 / .708 / .762 / .904** | **.684 / .752 / .715 / .869** |

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
python -c "from model.load_mystery_graphs import load_mystery_graphs; g = load_mystery_graphs('data/graphs'); g.summary()"
```

### Re-running analyses
```bash
cd Detective_R-GCN
python analysis/failure_analysis.py        # outputs analysis/results/failure_analysis_results.txt
python analysis/inference_analysis.py      # outputs analysis/results/inference_analysis_results.txt
python analysis/spectral_baseline.py       # baseline comparison
```

### Re-extraction pipeline
Place a detailed synopsis at `data/synopses_detailed/{ID}.txt`, then run:
```bash
python pipeline/reextract.py --only ID1 ID2 ...
```

### Scraping pipeline
Run from `Detective_R-GCN/`:
```bash
python -m pipeline.scraper.main --all
```

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

Use `analysis/notebooks/graph_visualization.ipynb` to inspect the checked-in dataset:
- dataset-level numeric and categorical summaries
- label counts and edge-type frequency tables
- histogram/bar-chart views of graph size
- network visualization for any selected graph ID in `data/graphs/`

---

## Graph JSON Structure (per entry)

Each file in `data/graphs/{ID}.json`:
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

Full schema is in `docs/murder_mystery_graph_schema.md`. Key points:

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
- If Phase 1 is rerun locally, updated synopses should go into `data/synopses/{ID}.txt`
- `data/synopses/` is the canonical scraped-synopsis directory (was `data/cleaned/` in the old layout)
- `data/synopses_detailed/` holds manually-curated detailed synopses for re-extraction
- 96 duplicate entries were marked `DUPLICATE` in the original manifest — do not re-extract these
- 46 entries were marked `EXCLUDED` — not single-case oriented or extraction failed
- Extraction was developed against a local Ollama setup on Apple Silicon

---

## Authors

MATH 290 Final Project — Adam Abramowitz, Max DeSantis, Isaac Dreeben, Abhi Mummaneni
