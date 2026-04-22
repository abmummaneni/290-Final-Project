# Unsolved Cases — Candidates for Re-Extraction

These 11 test stories were cases where **neither the R-GCN nor the LogReg baseline caught any villain**. Based on feature analysis, most likely have villains whose key features (motive, concealment, hidden relationship) were not flagged during LLM extraction.

**Suggested workflow:** For each case, find a more detailed synopsis (longer Wikipedia plot section, detailed review/analysis, or fandom wiki) and re-run extraction. Focus on synopses that explicitly describe character motives, concealment behavior, and hidden relationships.

**Source of unsolved list:** `failure_analysis.py` run on seed 42, 50 epochs, crime-edge masking, narrative features excluded.

---

## Known data-quality issues to address before or alongside re-extraction

These two entries have ground-truth problems. Fixing the labels (or flagging them as excluded) should come before any re-extraction effort, since they inflate the "unsolved" count for reasons unrelated to model or extraction quality.

### POD_035 — *In the Dark: Season 3* — MISLABELED
Curtis Flowers is labeled as a **Villain** in the current graph, but in real life (and in the podcast's framing) he was a **wrongly-convicted innocent** — the podcast is about his exoneration. Doug Evans, the prosecutor who framed him, is the actual antagonist. This is a ground-truth labeling error, not a model failure.

**Action items:**
- Re-label Curtis Flowers as Uninvolved (or Victim of wrongful prosecution)
- Verify Doug Evans's label is appropriate
- Consider whether podcasts about real-life exonerations fit the dataset's intended scope

### TVE_089 — *Vera: Hidden Depths* — INVALID VILLAIN ENTITY
The labeled Villain in this graph is not a character — it's the string `"People with specific forms of desperation and violence"`, which is a generic description rather than a named perpetrator. Extraction failed to identify a specific character as the villain.

**Action items:**
- Re-extract this entry with a better synopsis that names the perpetrator
- If still ambiguous, consider excluding from the test set

These should be noted as **data quality observations** in the final paper regardless of whether re-extraction is pursued.

---

## FLM_047 — *Zodiac*

- **Author:** David Fincher
- **Year:** 2007
- **Medium:** Film
- **Total characters in graph:** 17
- **Labeled villains:** 1
- **Labeled victims:** 5

### Current villain features (what the model had to work with)

| Villain | has_motive | motive_type | concealing_info | hidden_rel | has_alibi | present_at_scene |
|---|---|---|---|---|---|---|
| Arthur Leigh Allen | -1 | -1 | -1 | -1 | -1 | -1 |

### Victim(s): Darlene Ferrin, Mike Mageau, Bryan Hartnell, Cecelia Shepard, Paul Stine

**Wikipedia search:** https://en.wikipedia.org/w/index.php?search=Zodiac

---

## NOV_019 — *The Big Four*

- **Author:** Agatha Christie
- **Year:** 1927
- **Medium:** Novel
- **Total characters in graph:** 13
- **Labeled villains:** 4
- **Labeled victims:** 3

### Current villain features (what the model had to work with)

| Villain | has_motive | motive_type | concealing_info | hidden_rel | has_alibi | present_at_scene |
|---|---|---|---|---|---|---|
| Li Chang Yen | -1 | -1 | -1 | -1 | -1 | -1 |
| Abe Ryland | -1 | -1 | -1 | -1 | -1 | -1 |
| Claude Darrell | -1 | -1 | -1 | -1 | -1 | -1 |
| Doctor Quentin | -1 | -1 | -1 | -1 | -1 | -1 |

### Victim(s): Mayerling, Jonathan Whalley, Gilmour Wilson

**Wikipedia search:** https://en.wikipedia.org/w/index.php?search=The+Big+Four

---

## NOV_110 — *He Who Whispers*

- **Author:** John Dickson Carr
- **Year:** 1946
- **Medium:** Novel
- **Total characters in graph:** 6
- **Labeled villains:** 1
- **Labeled victims:** 2

### Current villain features (what the model had to work with)

| Villain | has_motive | motive_type | concealing_info | hidden_rel | has_alibi | present_at_scene |
|---|---|---|---|---|---|---|
| Fay Seton | -1 | -1 | -1 | -1 | -1 | 1 |

### Victim(s): Miles Hammond, Howard Brooke

**Wikipedia search:** https://en.wikipedia.org/w/index.php?search=He+Who+Whispers

---

## NOV_189 — *Nerve*

- **Author:** Dick Francis
- **Year:** 1964
- **Medium:** Novel
- **Total characters in graph:** 7
- **Labeled villains:** 1
- **Labeled victims:** 3

### Current villain features (what the model had to work with)

| Villain | has_motive | motive_type | concealing_info | hidden_rel | has_alibi | present_at_scene |
|---|---|---|---|---|---|---|
| Corin Kellar | -1 | -1 | -1 | -1 | -1 | -1 |

### Victim(s): Art Mathews, Grant Oldfield, Pip Pankhurst

**Wikipedia search:** https://en.wikipedia.org/w/index.php?search=Nerve

---

## NOV_209 — *The Blessing Way*

- **Author:** Tony Hillerman
- **Year:** 1970
- **Medium:** Novel
- **Total characters in graph:** 10
- **Labeled villains:** 2
- **Labeled victims:** 3

### Current villain features (what the model had to work with)

| Villain | has_motive | motive_type | concealing_info | hidden_rel | has_alibi | present_at_scene |
|---|---|---|---|---|---|---|
| Dr. Hall | -1 | -1 | -1 | -1 | -1 | -1 |
| George / Eddie | -1 | -1 | -1 | -1 | -1 | -1 |

### Victim(s): Luis Horseman, J. R. Canfield / John, Ellen Leon

**Wikipedia search:** https://en.wikipedia.org/w/index.php?search=The+Blessing+Way

---

## NOV_391 — *The Silkworm*

- **Author:** Robert Galbraith
- **Year:** 2014
- **Medium:** Novel
- **Total characters in graph:** 13
- **Labeled villains:** 2
- **Labeled victims:** 2

### Current villain features (what the model had to work with)

| Villain | has_motive | motive_type | concealing_info | hidden_rel | has_alibi | present_at_scene |
|---|---|---|---|---|---|---|
| Elizabeth Tassel | -1 | -1 | -1 | -1 | -1 | -1 |
| Michael Fancourt | -1 | -1 | -1 | -1 | -1 | -1 |

### Victim(s): Leonora Quine, Owen Quine

**Wikipedia search:** https://en.wikipedia.org/w/index.php?search=The+Silkworm

---

## NOV_414 — *The Lying Game*

- **Author:** Ruth Ware
- **Year:** 2017
- **Medium:** Novel
- **Total characters in graph:** 6
- **Labeled villains:** 1
- **Labeled victims:** 1

### Current villain features (what the model had to work with)

| Villain | has_motive | motive_type | concealing_info | hidden_rel | has_alibi | present_at_scene |
|---|---|---|---|---|---|---|
| Ambrose Atagon | -1 | -1 | -1 | 1 | -1 | -1 |

### Victim(s): Kate Atagon

**Wikipedia search:** https://en.wikipedia.org/w/index.php?search=The+Lying+Game

---

## POD_035 — *In the Dark: Season 3*

- **Author:** APM Reports
- **Year:** 2022
- **Medium:** Podcast
- **Total characters in graph:** 18
- **Labeled villains:** 2
- **Labeled victims:** 5

### Current villain features (what the model had to work with)

| Villain | has_motive | motive_type | concealing_info | hidden_rel | has_alibi | present_at_scene |
|---|---|---|---|---|---|---|
| Curtis Flowers | -1 | -1 | -1 | -1 | -1 | -1 |
| Doug Evans | -1 | -1 | -1 | -1 | -1 | -1 |

### Victim(s): Bertha Tardy, Carmen Rigby, Robert Golden, Derrick Stewart, Jacob Wetterling

**Wikipedia search:** https://en.wikipedia.org/w/index.php?search=In+the+Dark:+Season+3

---

## TVE_089 — *Vera: Hidden Depths*

- **Author:** Peter Hoar
- **Year:** 2011
- **Medium:** TV Episode
- **Total characters in graph:** 7
- **Labeled villains:** 1
- **Labeled victims:** 2

### Current villain features (what the model had to work with)

| Villain | has_motive | motive_type | concealing_info | hidden_rel | has_alibi | present_at_scene |
|---|---|---|---|---|---|---|
| People with specific forms of desperation and violence | -1 | -1 | -1 | -1 | -1 | -1 |

### Victim(s): Woman in river/body of water, Second victim

**Wikipedia search:** https://en.wikipedia.org/w/index.php?search=Vera:+Hidden+Depths

---

## TVE_130 — *Unforgotten: Series 2*

- **Author:** Andy Wilson
- **Year:** 2017
- **Medium:** TV Episode
- **Total characters in graph:** 8
- **Labeled villains:** 1
- **Labeled victims:** 1

### Current villain features (what the model had to work with)

| Villain | has_motive | motive_type | concealing_info | hidden_rel | has_alibi | present_at_scene |
|---|---|---|---|---|---|---|
| Ray Dawson | -1 | -1 | -1 | -1 | -1 | -1 |

### Victim(s): David Walker

**Wikipedia search:** https://en.wikipedia.org/w/index.php?search=Unforgotten:+Series+2

---

## TVE_143 — *Unbelievable (miniseries)*

- **Author:** Susannah Grant
- **Year:** 2019
- **Medium:** TV Episode
- **Total characters in graph:** 9
- **Labeled villains:** 1
- **Labeled victims:** 1

### Current villain features (what the model had to work with)

| Villain | has_motive | motive_type | concealing_info | hidden_rel | has_alibi | present_at_scene |
|---|---|---|---|---|---|---|
| Marc O'Leary | -1 | -1 | -1 | -1 | -1 | 1 |

### Victim(s): Marie Adler

**Wikipedia search:** https://en.wikipedia.org/w/index.php?search=Unbelievable+(miniseries)

---
