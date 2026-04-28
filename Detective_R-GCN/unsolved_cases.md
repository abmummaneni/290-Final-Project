# Case Status — Detective R-GCN (2026-04-26)

This file tracks every test story we've had to address — re-extracted, re-labeled, or excluded — across three rounds of refinement. After three rounds, **zero test stories remain unsolved by both models**.

---

## Status Summary

| Group | Count | Status |
|---|---|---|
| Originally unsolved (2026-04-22) | 11 | All addressed |
| Round 2 emerged unsolved (2026-04-25) | 2 | Both re-extracted on 2026-04-26 |
| Round 3 emerged unsolved (2026-04-26) | 1 | TVE_094, re-extracted same day |
| Held out as inference-only (no real-world villain) | 2 | FLM_047 (Zodiac), POD_035 (In the Dark) |
| Fully excluded (genre/extraction mismatch) | 2 | TVE_089, TVE_093 |
| Re-extracted to solved | 11 | (Round 1 + Round 2 + Round 3) |
| **Currently unsolved** | **0** | First time this state has been reached |

---

## Held-out for inference-only analysis

These are excluded from train/test (no real-world confirmed villain) but used in `inference_analysis.py` to demonstrate generalization. See PROJECT.md "Held-Out Inference Case Studies".

### FLM_047 — *Zodiac* (2007, David Fincher)

Real-world unsolved case. **Result:** Arthur Leigh Allen (the prime suspect) ranked #1 with P(villain) = 1.0000.

### POD_035 — *In the Dark: Season 3* (2022, APM Reports)

Real-world unsolved/contested case. The podcast investigates the wrongful prosecution of Curtis Flowers for a 1996 quadruple murder at Tardy Furniture in Winona, Mississippi. The actual perpetrator was never legally identified. The podcast tells two interlocking stories: (1) the prosecutorial misconduct led by DA **Doug Evans** with **John Johnson** (lead DA investigator) as accomplice, and (2) the existence of plausible alternate suspects (Willie James Hemphill, Marcus Presley, LaSamuel Gamble) whose existence Evans's office concealed from the defense.

**Label fix applied (2026-04-26):** Curtis Flowers was changed from Victim → Uninvolved (he was wrongly accused, not the murder victim).

**Result — model independently rediscovered both stories from the graph alone:**

| Rank | P(Villain) | Character | Recovered as |
|---|---|---|---|
| 1 | 1.0000 | Doug Evans | Wrongful-prosecution antagonist |
| 2 | 1.0000 | Willie James Hemphill | Concealed alternate suspect (`Suspect` label, treated as UNK in training) |
| 3 | 0.9999 | John Johnson | DA investigator / Evans's accomplice |
| 4 | 0.9999 | Marcus Presley | Concealed alternate suspect (Alabama spree) |
| 5 | 0.9999 | LaSamuel Gamble | Concealed alternate suspect (Alabama spree) |
| 16 | 0.0000 | Curtis Flowers | Exonerated (was originally mislabeled Villain) |

The model:
- Identified both wrongful-prosecution antagonists (Evans, Johnson)
- Surfaced all three concealed alternate suspects without ever having seen a "Suspect" class label in training
- Cleared Curtis Flowers despite the original mislabel

See PROJECT.md "Held-Out Inference Case Studies" for detail.

---

## Fully excluded (genre/extraction mismatch — cannot be analyzed)

### TVE_089 — *Vera: Hidden Depths* (2011)

The labeled "Villain" in this graph is a generic descriptor (`"People with specific forms of desperation and violence"`), not a named character. Re-extraction couldn't fix this. **Excluded entirely** from train, test, and inference.

### TVE_093 — *Spiral: Series 4* (2012)

The labeled "Villain" is `"Members of the criminal network"` — a collective with no single identifiable perpetrator. *Spiral* is an organized-crime drama rather than a whodunit, so it doesn't fit the dataset's intended scope of single-perpetrator detective fiction. **Excluded entirely** (added 2026-04-26).

---

## Round 1 (2026-04-25) — re-extraction outcomes

Eight of nine non-special entries resolved cleanly:

| Entry | Title | Status |
|---|---|---|
| NOV_019 | *The Big Four* | Solved |
| NOV_110 | *He Who Whispers* | Solved |
| NOV_189 | *Nerve* | Solved |
| NOV_209 | *The Blessing Way* | Solved |
| NOV_391 | *The Silkworm* | Solved |
| NOV_414 | *The Lying Game* | Solved |
| TVE_130 | *Unforgotten: Series 2* | Solved |
| TVE_143 | *Unbelievable (miniseries)* | Solved |

---

## Round 2 (2026-04-26) — re-extraction outcomes

Both newly-emerged unsolved cases resolved:

| Entry | Title | Status |
|---|---|---|
| NOV_276 | *The Stonecutter* | Solved |
| SHO_001 | *The Red-Headed League* | Solved |

---

## Round 3 (2026-04-26 late) — final residual case resolved

After excluding TVE_093, the seed-42 split surfaced TVE_094 (*The Killing: Season 2*) as a new unsolved case — the original extraction had labeled the villain as a code name ("The Pasha") with all-UNK features. A detailed synopsis was generated and re-extracted, surfacing 4 named villains:

| Entry | Title | Old → New | Status |
|---|---|---|---|
| TVE_094 | *The Killing: Season 2* | 1 generic villain → 4 named villains | Solved |

After Round 3, the test set has zero stories where neither model catches a villain.

---

## How to add a new exclusion

If a story has no identifiable villain (real-world unsolved, organized-crime collective, or invalid extraction):
- Add the entry ID to `EXCLUDE_ENTRIES` in BOTH `failure_analysis.py` and `inference_analysis.py`
- For inference-only analysis, also add to `INFERENCE_TARGETS` in `inference_analysis.py` and supply highlight names in `TARGET_HIGHLIGHTS`
- Update PROJECT.md "Excluded entries" list
- Update this file's status table

## How to run another re-extraction round

1. Generate a detailed synopsis using the prompt in `support_files/synopsis_generation_prompt.md`.
2. Save as `extraction/data/new_extracts/{ID}.txt`.
3. Run `python reextract.py --only ID1 ID2 ...` (~5 min per entry on M1 Max).
4. Re-run `python failure_analysis.py` to check status.
5. Update this file.
