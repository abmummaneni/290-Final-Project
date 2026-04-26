# Synopsis Generation Prompt

This is the prompt used to generate the more detailed synopses in `extraction/data/new_extracts/`. It targets the specific features the downstream extractor (mixtral:8x7b two-pass) needs to flag during its node extraction step.

**When to use:** When a story is on the unsolved list and feature analysis shows its villains have all-UNK features, the original synopsis likely didn't make villain traits explicit. Pass this prompt to a high-quality LLM (Claude, GPT, Gemini) along with the case ID, title, author, year, and medium.

**Output:** A 600-1000 word narrative synopsis. Save to `extraction/data/new_extracts/{ID}.txt`, then run `python reextract.py --only {ID}`.

---

```
You are helping build a knowledge graph dataset for a murder mystery
classification model. I need a detailed plot synopsis of [TITLE] by
[AUTHOR] ([YEAR], [MEDIUM]) that surfaces specific information about each
character. The synopsis will be processed by a separate LLM extractor to
build a graph, so character traits must be stated explicitly rather than
implied.

Length target: 600-1000 words.

Required content for EACH named character (especially anyone involved in
the crime — perpetrator, victim, suspects, witnesses):

1. **Role in the crime** — explicitly state whether they are the villain,
   victim, witness, or uninvolved. If there are accomplices, name them and
   describe their role.

2. **Motive** — for every character with any plausible motive, state it
   directly. Use one of these motive types when applicable: financial /
   money, love, jealousy, revenge, power, glory, entitlement, manipulation,
   protecting family, psychopathic tendencies. Say "X has a motive of [type]
   because [reason]" or "Y has no apparent motive."

3. **Alibi** — for each suspect, state whether they have an alibi for the
   time of the crime. Be explicit: "Z has an alibi: she was at the theater
   with W" or "A has no alibi for the time of the murder."

4. **Presence at crime scene** — state who was physically present at the
   crime scene at the time of the crime, and who was elsewhere.

5. **Concealment behavior** — note any character who is hiding information,
   lying, withholding evidence, or behaving evasively. Say "X is concealing
   the fact that..." rather than implying it.

6. **Hidden relationships** — name any secret relationships between
   characters: secret affairs, undisclosed family ties, prior acquaintance
   pretending to be strangers, secret partnerships, etc. State them
   directly: "X and Y were secretly married" or "Z is actually W's
   illegitimate son."

7. **Social/professional details** — gender, occupation, social status
   (high/middle/low), and key relationships (married to, employed by,
   friend of, etc.).

8. **The resolution** — explicitly state how the crime was solved and who
   was revealed as the perpetrator(s). Do not be coy or use phrases like
   "the truth is eventually revealed" — name the villain and explain how
   they did it.

Write in third person, narrative prose. Avoid bullet points within the
synopsis itself — embed all of the above as natural sentences in the plot
description. The downstream extractor benefits from explicit, sentence-level
statements like "Mrs. Jenkins killed her husband for the inheritance and
faked an alibi by claiming she was visiting her sister."

Include spoilers — this dataset is for solved-mystery analysis, so the
ending must be revealed.
```

---

## Tips when using this prompt

- **Verify the LLM's facts** — for older novels especially, LLMs sometimes confuse plots between similar mysteries. Cross-check the named villain against Wikipedia.
- **For multi-villain stories** — explicitly tell the LLM "this story has multiple villains" so it doesn't collapse them into one perpetrator.
- **For ambiguous cases (e.g. real-life unsolved crimes)** — consider excluding from train/test instead. See FLM_047 (Zodiac) for an example handled via `exclude_entries={"FLM_047"}`.

## Validated outcome (2026-04-25)

This prompt was used to generate synopses for the original 11 unsolved cases. After re-extraction:
- 9 of 11 cases became solved
- 1 was excluded (FLM_047 — no real-world villain)
- 1 had pre-existing data quality issues that should be revisited (TVE_089)

Cross-validated R-GCN F1 improved from 0.703 → 0.712, and story-level "solved" rate improved from 75% → 96.5%. See `PROJECT.md` Re-Extraction Impact section for full details.
