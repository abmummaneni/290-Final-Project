# Case Status — Detective R-GCN (2026-04-26)

This file tracks the status of every test story the model couldn't initially handle. After two rounds of targeted re-extraction, label cleanup, and four exclusions, the corpus is in a clean state for reporting.

---

## Status Summary

| Group | Count | Status |
|---|---|---|
| Originally unsolved (2026-04-22) | 11 | All addressed |
| Round 2 emerged unsolved (2026-04-25) | 2 | Both re-extracted on 2026-04-26 |
| Held out as inference-only (no real-world villain) | 2 | FLM_047 (Zodiac), POD_035 (In the Dark) |
| Fully excluded (genre/extraction mismatch) | 2 | TVE_089, TVE_093 |
| Re-extracted to solved | 10 | (the rest of Round 1 + Round 2) |
| Currently unsolved | 1 | TVE_094 — same generic-villain pattern (candidate for next exclusion) |

---

## Held-out for inference-only analysis

These are excluded from train/test (no real-world confirmed villain) but used in `inference_analysis.py` to demonstrate generalization. See PROJECT.md "Held-Out Inference Case Studies".

### FLM_047 — *Zodiac* (2007, David Fincher)

Real-world unsolved case. **Result:** Arthur Leigh Allen (the prime suspect) ranked #1 with P(villain) = 1.0000.

### POD_035 — *In the Dark: Season 3* (2022, APM Reports)

Real-world unsolved/contested case. The podcast investigates the wrongful prosecution of Curtis Flowers; the actual perpetrator was never identified, though the podcast frames Doug Evans (the prosecutor) as the antagonist for orchestrating prosecutorial misconduct.

**Label fix applied (2026-04-26):** Curtis Flowers was changed from Victim → Uninvolved (he was wrongly accused, not the murder victim).

**Result:**
- Doug Evans ranked #1 with P(villain) = 1.0000 (model agrees with podcast's antagonist framing)
- Curtis Flowers ranked #16 (last) with P(villain) = 0.0000 (model exonerates him)

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

## Flagged for review (next exclusion candidate)

After excluding TVE_093, the seed-42 test split surfaces another story with the same generic-villain pattern:

### TVE_094 — *The Killing: Season 2*

- **Author:** Various
- **Year:** 2012
- **Medium:** TV Episode
- **Total characters:** 9
- **Labeled villains:** 1

- Villain: `The Pasha` (features all UNK)

This appears to follow the same code-name / generic-perpetrator pattern as TVE_089 and TVE_093. Consider excluding (treat like TVE_093) or generating a more detailed synopsis that names a specific perpetrator.

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
