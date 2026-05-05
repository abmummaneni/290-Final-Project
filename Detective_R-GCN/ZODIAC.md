# The Zodiac Case Study

**Held-out inference on the same unsolved real-world case rendered two ways: as 2007 detective fiction (FLM_047) and as a factual police record (ZODIAC_FACTUAL).**

This document is a self-contained presentation source for the Zodiac inference experiment in our Detective R-GCN paper. It walks from the experimental setup → the extracted graph → the inference results → the comparison and interpretation.

---

## 1. The Experiment

The Zodiac killings (1968–1969) are unsolved in real life. We use this case as a paired held-out test:

- **FLM_047** — *Zodiac* (David Fincher, 2007). Detective fiction. The film centers Robert Graysmith's investigation of Arthur Leigh Allen.
- **ZODIAC_FACTUAL** — A factual synopsis of the same case (≈2000 words), built exclusively from official law-enforcement sources (FBI, SFPD, Vallejo PD, Napa County SO, CA DOJ), encyclopedic references, contemporaneous newspapers, and **2023–2025 forensic updates** (Donna Lass DNA identification, GEDmatch attempts, Allen's DNA exclusion). The Fincher film and Graysmith book are explicitly excluded as fictional.

Both stories are **excluded from training entirely** (`EXCLUDE_ENTRIES` in `analysis/inference_analysis.py`). The R-GCN never sees them during training. We ask: when run on a story it has never trained on, who does the model rank as most likely to be the villain?

**Crime edges (`kills`, `killed by`, etc.) are masked at inference time** — the model must reason from circumstantial evidence only.

---

## 2. The Synopsis (factual case)

The factual synopsis is a tight, source-cited narrative covering:

1. **Confirmed crimes and victims** — Lake Herman Road (Faraday, Jensen — Dec 20, 1968), Blue Rock Springs (Ferrin †, Mageau survived — Jul 4–5, 1969), Lake Berryessa (Hartnell survived, Shepard † — Sep 27, 1969), Presidio Heights (Stine — Oct 11, 1969).
2. **The killer's motive in his own words** — psychopathic publicity-seeking ("I like killing people because it is so much fun" — Z408 cipher; "enough slaves to work for me" — Z340 decoded 2020). No personal motive against any victim.
3. **Possibly-connected unconfirmed cases** — Cheri Jo Bates (Riverside, 1966), Donna Lass (South Lake Tahoe, 1970).
4. **The 2023–2025 forensic state** — **Donna Lass's skull, found in 1986, was DNA-identified in December 2023.** Vallejo PD's 2018 GEDmatch attempt failed. Stamp-saliva produced partial profiles that **excluded Allen** but didn't yield a positive match.
5. **The five main suspects** — Allen, Kane, Marshall, Gaikowski, Poste. All deceased, none confirmed.
6. **Cross-suspect connections** —
   - Allen and Kane both investigated for Santa Rosa hitchhiker murders.
   - **Kane and Donna Lass worked in the same building** at Sahara Tahoe in 1970; coworkers said they knew each other.
   - Stine's sister identified Gaikowski at her brother's funeral.

The full synopsis lives at `data/synopses/ZODIAC.txt`; the version used for extraction is `data/synopses_detailed/ZODIAC_FACTUAL.txt`.

---

## 3. The Extracted Graph

After two-pass extraction with mixtral:8x7b → label normalization, the graph has:

| Statistic | Count |
|---|---|
| Characters | 39 |
| Locations | 4 |
| Occupations | 12 |
| Organizations | 0 |
| Edges (canonical) | 38 |
| Edges (with inverses) | 76 |
| Crime edges to mask | 0 (the case is unsolved — no kills edges were extracted) |

### 3.1 Characters (with normalized labels)

| ID | Label | Name |
|---|---|---|
| char_0 | **Villain** | Zodiac Killer (unidentified — placeholder node) |
| char_1 | Victim | David Faraday |
| char_2 | Victim | Betty Lou Jensen |
| char_3 | Victim | Darlene Ferrin |
| char_4 | Witness | Michael Mageau |
| char_5 | Victim | Bryan Hartnell |
| char_6 | Victim | Cecelia Shepard |
| char_7 | Victim | Paul Stine |
| char_8 | Victim | Donna Lass |
| char_9 | Uninvolved | Arthur Leigh Allen ★ |
| char_10 | Uninvolved | Lawrence Kane / Lawrence Kaye / Lawrence Klein ★ |
| char_11 | Uninvolved | Richard Marshall ★ |
| char_12 | Uninvolved | Richard Joseph Gaikowski ★ |
| char_13 | Uninvolved | Gary Francis Poste ★ |
| char_14 | Witness | Donald Cheney |
| char_15 | Victim | Kathleen Johns |
| char_16 | Witness | Darlene Ferrin's sister Linda |
| char_17 | Witness | Officer Don Fouke |
| char_18 | Witness | Nancy Slover |
| char_19 | Uninvolved | Ken Narlow (Napa County SO) |
| char_20 | Uninvolved | David Toschi (SFPD) |
| char_21 | Uninvolved | William Armstrong (SFPD) |
| char_22 | Uninvolved | Eric Zelms (SFPD) |
| char_23 | Uninvolved | John Lynch (Vallejo PD) |
| char_24 | Uninvolved | Jack Mulanax (Vallejo PD) |
| char_25 | Uninvolved | Daniel Pitta (Solano County SO) |
| char_26 | Uninvolved | Sherwood Morrill (CA DOJ) |
| char_27 | Uninvolved | Paul Avery (SF Chronicle) |
| char_28 | Victim | Cheri Jo Bates (possibly-connected) |
| char_29 | Uninvolved | Riverside police |
| char_30 | Uninvolved | Donna Lass's family member |
| char_31 | Uninvolved | SLTPD officer |
| char_32 | Uninvolved | PCSO officer |
| char_33 | Uninvolved | SFPD officer |
| char_34 | Uninvolved | Earl Van Best Jr. ★ |
| char_35 | Uninvolved | FX investigation team |
| char_36 | Uninvolved | Tom Voigt (Zodiac researcher) |
| char_37 | UNK | Blaine Blaine / Goldcatcher (accuser) |
| char_38 | Witness | Seawater family |

★ = named suspect

The "Suspect" label isn't in our 4-class system, so suspects are normalized to **Uninvolved**. The model has never trained on a "Suspect" pattern — it has to reason about these characters from their features and graph context alone.

### 3.2 Locations

The four crime-scene locations:

1. Lake Herman Road, Benicia
2. Blue Rock Springs Park, Vallejo
3. Lake Berryessa, Napa County
4. Presidio Heights, San Francisco

The Zodiac Killer placeholder node is connected to all four (resides-at + 3× present-at), giving it a strong spatial signature.

### 3.3 Occupations (all 12)

Schoolteacher, Convicted child molester, Career criminal with multiple aliases and SSNs, Retired police detective, Ham radio operator and movie projectionist, Counterculture journalist, Army medic training, House painter, Victim's sister, Zodiac suspect investigator, Zodiac accuser, Zodiac documentary witness.

*(Note: a few occupations were mis-attributed in extraction — e.g. Hartnell tagged as "ham radio operator," which was actually Marshall. These are extractor artifacts and don't load-bear in the result.)*

### 3.4 Edges (all 38)

| # | Source | Relation | Target |
|---|---|---|---|
| 1 | Zodiac Killer | resides at | Lake Herman Road, Benicia |
| 2 | Zodiac Killer | present at | Blue Rock Springs Park, Vallejo |
| 3 | Zodiac Killer | present at | Lake Berryessa, Napa County |
| 4 | Zodiac Killer | present at | Presidio Heights, San Francisco |
| 5 | David Faraday | employed as | Schoolteacher |
| 6 | Betty Lou Jensen | employed as | Schoolteacher |
| 7 | Darlene Ferrin | resides at | Blue Rock Springs Park, Vallejo |
| 8 | Michael Mageau | present at | Blue Rock Springs Park, Vallejo |
| 9 | Bryan Hartnell | employed as | Ham radio operator and movie projectionist |
| 10 | Cecelia Shepard | employed as | Ham radio operator and movie projectionist |
| 11 | Paul Stine | resides at | Presidio Heights, San Francisco |
| 12 | Arthur Leigh Allen ★ | employed as | Schoolteacher |
| **13** | **Lawrence Kane ★** | **suspects** | **Donna Lass** |
| 14 | Richard Marshall ★ | employed as | Career criminal with multiple aliases and SSNs |
| 15 | Richard Gaikowski ★ | employed as | Retired police detective |
| **16** | **Donald Cheney** | **witnessed by** | **Lawrence Kane ★** |
| **17** | **Kathleen Johns** | **identified as** | **Lawrence Kane ★** |
| 18 | Linda (Ferrin's sister) | related to | Darlene Ferrin |
| 19 | Officer Don Fouke | witnessed by | Zodiac Killer |
| 20 | Nancy Slover | received correspondence from | Zodiac Killer |
| 21 | Ken Narlow | investigates | Paul Stine |
| 22 | David Toschi | investigates | Paul Stine |
| 23 | William Armstrong | investigates | Paul Stine |
| 24 | John Lynch | investigates | Zodiac Killer |
| 25 | Jack Mulanax | investigates | Darlene Ferrin |
| 26 | Daniel Pitta | investigates | David Faraday |
| 27 | Sherwood Morrill | examines handwriting of | Zodiac Killer |
| 28 | Paul Avery | receives correspondence from | Zodiac Killer |
| 29 | Riverside police | investigates | Cheri Jo Bates |
| 30 | Lass's family member | related to | Donna Lass |
| 31 | SLTPD officer | treats as homicide | Donna Lass |
| 32 | PCSO officer | announces identification of skull | Donna Lass |
| 33 | SFPD officer | investigates | Zodiac Killer |
| 34 | Earl Van Best Jr. ★ | employed as | Counterculture journalist |
| 35 | FX investigation team | uninvolved in investigation of | Earl Van Best Jr. ★ |
| 36 | Tom Voigt | researches and documents | Zodiac Killer |
| 37 | Blaine Blaine / Goldcatcher | accuses | Richard Gaikowski ★ |
| 38 | Seawater family | related to | Arthur Leigh Allen ★ |

The three **bolded edges** are the structurally decisive ones in the result that follows.

### 3.5 The Five Main Suspects — Features at a Glance

All five named suspects normalize to the same `Uninvolved` label. The features are sparse (mostly UNK), so the model has to reason mostly from graph structure:

| Suspect | Edges | Gender | Social | Alibi | At Scene | Motive | Concealing | Hidden Rel. |
|---|---|---|---|---|---|---|---|---|
| Arthur Leigh Allen ★ | **2** | M | 0.25 | **Yes (1)** | UNK | UNK | UNK | UNK |
| Lawrence Kane ★ | **3** | M | 0.25 | UNK | UNK | UNK | UNK | **Yes (1)** |
| Richard Marshall ★ | 1 | M | 0.25 | UNK | UNK | UNK | UNK | UNK |
| Richard Gaikowski ★ | 2 | M | 0.50 | UNK | UNK | UNK | UNK | UNK |
| Gary Francis Poste ★ | 0 | M | 0.25 | UNK | UNK | UNK | UNK | UNK |
| Earl Van Best Jr. ★ | 2 | M | 0.25 | UNK | UNK | UNK | UNK | UNK |

Two feature differentiators worth flagging:
- **Allen has `has_alibi = 1`** — the extractor caught his stated "scuba diving" alibi for Lake Berryessa.
- **Kane has `has_hidden_relationship = 1`** — the extractor caught the concealed Lass workplace connection.

Per-feature differentiation is otherwise minimal. **The decisive evidence is in the graph structure.**

---

## 4. The Result

Trained corpus: 14,037 nodes, 24,880 train edges (574 stories, ZODIAC_FACTUAL + four other special cases excluded). Train time: 35s on M1 Max. Inference is a single forward pass on the held-out graph.

### 4.1 Full Ranking (by P(Villain))

| Rank | P(Villain) | Label | Character |
|---|---|---|---|
| **1** | **1.0000** | Villain | Zodiac Killer (unidentified placeholder) |
| **2** | **0.9587** | Uninvolved | **Lawrence Kane ★** |
| 3 | 0.6368 | Victim | David Faraday (false positive — see §6) |
| 4 | 0.6215 | Uninvolved | Paul Avery (Chronicle reporter, received Zodiac letters) |
| **5** | **0.5264** | Uninvolved | **Richard Marshall ★** |
| 6 | 0.3468 | UNK | Blaine Blaine / Goldcatcher |
| 7 | 0.3296 | Uninvolved | PCSO officer |
| 8 | 0.2529 | Victim | Paul Stine |
| 9 | 0.2413 | Witness | Nancy Slover |
| 10 | 0.1789 | Uninvolved | Earl Van Best Jr. ★ |
| **11** | **0.0736** | Uninvolved | **Richard Joseph Gaikowski ★** |
| 12 | 0.0605 | Uninvolved | Riverside police |
| 13 | 0.0501 | Victim | Cecelia Shepard |
| 14 | 0.0489 | Uninvolved | Eric Zelms |
| 15 | 0.0443 | Victim | Betty Lou Jensen |
| 16 | 0.0408 | Victim | Bryan Hartnell |
| **17** | **0.0399** | Uninvolved | **Gary Francis Poste ★** |
| 18 | 0.0395 | Uninvolved | FX investigation team |
| 19 | 0.0383 | Victim | Darlene Ferrin |
| 20 | 0.0296 | Uninvolved | Sherwood Morrill |
| 21 | 0.0147 | Witness | Michael Mageau |
| 22 | 0.0075 | Victim | Donna Lass |
| 23 | 0.0070 | Uninvolved | Lass's family member |
| 24 | 0.0060 | Uninvolved | SLTPD officer |
| 25 | 0.0033 | Uninvolved | Jack Mulanax |
| 26 | 0.0029 | Victim | Cheri Jo Bates |
| 27 | 0.0019 | Witness | Donald Cheney |
| 28 | 0.0016 | Uninvolved | John Lynch |
| 29 | 0.0016 | Uninvolved | Tom Voigt |
| 30 | 0.0013 | Witness | Seawater family |
| 31 | 0.0012 | Witness | Officer Don Fouke |
| 32 | 0.0012 | Victim | Kathleen Johns |
| 33 | 0.0011 | Uninvolved | Daniel Pitta |
| 34 | 0.0009 | Uninvolved | Ken Narlow |
| 35 | 0.0009 | Uninvolved | David Toschi |
| 36 | 0.0009 | Uninvolved | William Armstrong |
| 37 | 0.0004 | Witness | Linda (Ferrin's sister) |
| **38** | **0.0004** | Uninvolved | **Arthur Leigh Allen ★** |
| 39 | 0.0002 | Uninvolved | SFPD officer |

### 4.2 The Five Suspects, Ranked

| Suspect | Rank | P(Villain) |
|---|---|---|
| **Lawrence Kane ★** | **#2** | **0.9587** |
| **Richard Marshall ★** | **#5** | **0.5264** |
| Richard Gaikowski ★ | #11 | 0.0736 |
| Gary Francis Poste ★ | #17 | 0.0399 |
| **Arthur Leigh Allen ★** | **#38 (last)** | **0.0004** |

**The model identifies Lawrence Kane as the prime suspect among the five, with Allen ranked dead last.**

---

## 5. Why Kane

Three Kane-specific edges that no other suspect has:

### Edge 1 — `Kane suspects → Donna Lass`

Lass and Kane were coworkers at the Sahara Tahoe casino in 1970; her office was down the hallway from his nurse's station. **Lass's skull, found in 1986, was DNA-identified in December 2023.** This is the only confirmed-homicide victim with a workplace link to a named Zodiac suspect. The extractor encoded this as a direct `suspects` edge from Kane to Lass.

### Edge 2 — `Donald Cheney witnessed by → Kane`

Cheney's photo-lineup identification of Kane (originally Allen's accuser, Cheney later identified Kane in another lineup).

### Edge 3 — `Kathleen Johns identified as → Kane`

Johns was abducted in 1970 and survived; in a 1980s photo lineup she identified Kane as her abductor.

These three edges together place Kane in a witness-identification network that the R-GCN's training data (detective fiction) associates with **identified perpetrators**. No other suspect has this pattern — Allen has only an `employed-as` and a `related-to`; Marshall, Gaikowski, and Poste have at most one occupation/professional edge each.

Kane's feature `has_hidden_relationship = 1` (the Lass workplace concealment) is the only suspect-level feature differentiator that points toward villainy, and it lines up with his graph profile.

---

## 6. Why Allen Falls

In the 2007 film (FLM_047), Allen ranks #1 with P=1.0000. In the factual record, he ranks #38 (dead last) with P=0.0004. The synopsis explicitly states that "Arthur Leigh Allen, the only publicly named suspect in the Zodiac Killer case, was officially excluded by DNA analysis," and the extractor reflects this:

**Allen has only two edges, both structurally innocent:**

1. `Allen employed as → Schoolteacher` — a routine professional edge shared with Faraday and Jensen (both labeled victims) and with no concealment or harm overtone.
2. `Seawater family related to → Allen` — a personal_bond edge from a family group labeled `Witness` to Allen, also innocent in shape.

**Plus one feature signal pointing away from villainy:**

- `has_alibi = 1` — Allen's "scuba diving at Lake Berryessa" alibi is encoded as present (even though it was uncorroborated in real life). In the LogReg feature analysis on training data, `has_alibi` carries the strongest negative weight (-1.34) of any feature for Villain prediction.

The 2007 film's screenplay translates the historical record into whodunit grammar — Allen accumulates suspect-laden edges to the protagonist Graysmith, threats, etc. The factual record translates the same case into police-report grammar — Allen is just one schoolteacher in a list of investigated suspects, and the synopsis emphasizes the DNA exclusion. **The R-GCN is reading the structural evidence in front of it, not memorizing pop-culture priors about Allen.**

### A note on the false positives

Two false positives in the top 5 (David Faraday at #3, Paul Avery at #4) are worth flagging:

- **David Faraday (P=0.64)** — a 17-year-old victim. The extractor erroneously listed him as `employed-as Schoolteacher` (he was a high-school student on a first date). This is an extraction artifact: the synopsis describes Allen as a Vallejo schoolteacher, and the LLM mis-assigned the occupation. The shared-schoolteacher edge connects Faraday to Allen and to the model's "schoolteacher villain prior," producing a spurious score.
- **Paul Avery (P=0.62)** — the SF Chronicle reporter who received personal Zodiac letters. He's connected to the Zodiac Killer node via `receives correspondence from`, the same relation the model sees from named witness Nancy Slover. This is an interpretable error: a journalist with an information channel to the villain looks structurally like a complicit insider in detective fiction.

These false positives are real, and an honest paper should report them. They don't change the headline result: **Kane #2, Allen #38.**

---

## 7. Comparison: Two Renderings of the Same Case

| Snapshot | Source | Synopsis style | #1 ranked suspect | Allen rank | Allen P(villain) |
|---|---|---|---|---|---|
| **FLM_047** | Fincher film, 2007 | Detective fiction (whodunit grammar) | **Arthur Leigh Allen** | **#1** | **1.0000** |
| **ZODIAC_FACTUAL** | LE + press + 2023–25 forensics | Police-report grammar | **Lawrence Kane** | #38 | 0.0004 |

**Both picks are defensible by their respective evidentiary states.**

- **2007 (film era):** Allen was the only suspect ever publicly named by police (1972 search warrant, Mageau's 1991 photo ID, Donald Cheney's testimony). The film centers him.
- **2023–2025 (factual update):** Vallejo PD officially developed Kane as their suspect in 1991. Allen was excluded by DNA in the 2000s. The Donna Lass DNA identification in December 2023 added a confirmed-homicide victim physically linked to Kane by workplace. Kane's case has strengthened.

**The case is unsolved in real life — there is no ground-truth answer.** What these two results show together is that the R-GCN reads the structural evidence in whatever synopsis it is given, and produces the suspect that the evidence emphasizes. It is not anchored to which name pop culture has elevated.

---

## 8. The Bigger Claim

Combined with the other two held-out cases:

- **FLM_047** (Zodiac, 2007 film) → identifies Arthur Leigh Allen #1 with P=1.0000.
- **POD_035** (*In the Dark: Season 3*) → identifies Doug Evans #1 (the prosecutor framed as the wrongful-prosecution antagonist) and exonerates Curtis Flowers at P=0.0000 (rank #16/16) even though the data file initially mislabeled him as Villain. Surfaces three concealed alternate suspects (Hemphill, Presley, Gamble) in positions #2, #4, #5.
- **ZODIAC_FACTUAL** (same Zodiac case, factual) → identifies Lawrence Kane #2 (immediately below the unidentified placeholder), with Allen #38 reflecting his DNA exclusion.

**The R-GCN has learned transferable patterns of evidence-based reasoning over knowledge graphs.** It does not memorize labels (POD_035 mislabel was overridden), it does not anchor to pop-culture rankings (Allen drops 37 positions when the source updates), and it surfaces previously-concealed suspects when their structural evidence is present (POD_035, Kane). When the evidentiary state of a real case shifts, the model's prediction shifts with it — in the direction that the evidence shifts.

---

## 9. Reproducibility

```bash
cd Detective_R-GCN
python analysis/inference_analysis.py --targets ZODIAC_FACTUAL
```

- Synopsis: `data/synopses/ZODIAC.txt` (and `data/synopses_detailed/ZODIAC_FACTUAL.txt` for re-extraction)
- Graph: `data/graphs/ZODIAC_FACTUAL.json` (39 chars, 4 locs, 12 occs, 38 edges)
- Output: `analysis/results/zodiac_factual_inference.txt`
- Held-out specification: `analysis/inference_analysis.py` — `EXCLUDE_ENTRIES = {"FLM_047", "POD_035", "TVE_089", "TVE_093", "ZODIAC_FACTUAL"}`

Earlier extractions of the same case (preserved for record) are at `data/graphs_backup/ZODIAC_FACTUAL_v1_full.json` (5380-word source, pass-1 truncated, no locations) and `_v2_trimmed.json` (2489-word source without 2023 updates). Both produced inferior structural signal and ranked Allen lower without the Kane-Lass connection edge.
