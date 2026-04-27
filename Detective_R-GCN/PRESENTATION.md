# Detective R-GCN — 15-Minute Presentation Blueprint

**Team:** Adam Abramowitz, Max DeSantis, Isaac Dreeben, Abhi Mummaneni
**Course:** MATH 290 Final Project
**Topic:** Relational Graph Convolutional Networks for Villain Identification in Murder Mysteries

---

## Instructions for Slide Builder

This document contains everything needed to build a ~14-slide presentation for a 15-minute slot. **Target 12.5 minutes of presentation content + 2.5 minutes of Q&A.** Each section below maps to one slide (or a small group). For each slide we provide:

- **Title** — the slide heading
- **Key points** — bullet points to include (keep slides sparse; these are speaker notes + content)
- **Visuals** — suggested diagrams, tables, or figures
- **Speaker notes** — what the presenter should say (~1 min per slide, some heavier)
- **Equations** — exact LaTeX where needed

The tone should be: technically precise but accessible to a graduate math audience that knows linear algebra and ML basics but may not know GNNs. The detective metaphor is the hook — lean into it.

**Pacing guide (~12.5 min total):**
- Slides 1–3 (Title, Motivation, Detective Scenario): ~2.5 min
- Slides 4–6 (Dataset, Example Graph, Graph Construction): ~3 min
- Slides 7–9 (Encoder Math, Decoders, Training Nuances): ~3 min
- Slides 10–12 (Results, Disagreements, Case Studies): ~3 min
- Slides 13–14 (Takeaways, Questions): ~1 min
- Then open floor for Q&A with appendix slides on standby

---

## Slide 1: Title Slide

**Title:** Playing Detective with Relational Graph Convolutional Networks

**Subtitle:** Villain Identification in Murder Mystery Knowledge Graphs

**Content:**
- Team names: Adam Abramowitz, Max DeSantis, Isaac Dreeben, Abhi Mummaneni
- Course: MATH 290 Final Project
- Date: Spring 2026

**Visual:** Optional — a stylized knowledge graph snippet or a magnifying glass over a graph.

---

## Slide 2: Problem & Motivation

**Title:** Why Graphs? Why Mysteries?

**Key points:**
- Murder mysteries are a natural testbed for relational reasoning — the detective must weigh *who* is connected to *whom*, *how*, and *where*
- Traditional ML (logistic regression on features) can flag suspicious characters, but it can't reason about alibis, spatial separation, or social context
- A character with motive but a verified alibi at a distant location is probably innocent — that reasoning requires *relational structure*
- Knowledge graphs encode exactly this: typed, directed relationships between entities (characters, locations, occupations, organizations)
- **Our question:** Can a graph neural network trained on relational evidence identify the villain in a murder mystery — without ever seeing the "kills" edge?

**Speaker notes:**
The key insight is that detective reasoning is inherently relational. You don't just look at a suspect's features in isolation — you look at who they know, where they were, what they do for a living, and how all of that connects to the crime. That's exactly what a knowledge graph captures. We wanted to see if an R-GCN — a GNN designed for multi-relational data — could learn to do this kind of reasoning from data.

---

## Slide 3: The Detective Scenario

**Title:** Task Definition: The Detective Scenario

**Key points:**
- **Setup:** A victim has been found. The model has access to:
  - The full cast of characters and their features (motive, alibi, concealment, social status, etc.)
  - All relationships: who works where, who lives where, who knows whom, who deceives whom
  - **But NOT the crime edges** — `kills` and `killed_by` are hidden (they reveal the answer)
- **Task:** Binary classification — for each character, predict Villain vs. Non-Villain
- **Evaluation:** Crime-revealing edges are masked at test time. The model must reason from circumstantial evidence only.
- **Key validation:** We confirmed crime-edge masking had **zero impact** on predictions — the model was already ignoring "kills" edges and genuinely reasoning from circumstantial evidence.

**Visual:** A simple diagram showing a small murder mystery graph with characters, locations, occupations connected by labeled edges. Show the `kills` edge as dashed/hidden with a ❌ or strikethrough. Annotate that the model sees everything else.

**Speaker notes:**
This is the core framing. Think of the model as a detective arriving at a crime scene. It can see the social network, who has motive, who has an alibi, where everyone lives and works. But it can't see who actually committed the crime. We hide those edges. The question is whether the relational structure — the typed connections between entities — gives the model enough to make an accusation. And when we tested this, masking the crime edges had literally zero effect on predictions. The model was already making all its decisions from the circumstantial evidence.

---

## Slide 4: Dataset Overview

**Title:** 576 Murder Mysteries as Knowledge Graphs

**Key points:**
- **Sources:** Novels (340), TV episodes (126), films (62), podcasts (28), short stories (20)
- **Extraction pipeline:** Wikipedia synopses → LLM extraction (Mixtral 8x7b, two-pass) → structured knowledge graphs
- **Scale:** 14,020 nodes, ~32,000 edges, 16 relation types (8 canonical + 8 inverse)
- **Node types:** Characters (with 8 observable features), Locations (3 features), Occupations (3 features), Organizations (3 features)
- **Character features the detective can observe:** gender, social status, has alibi, present at crime scene, has motive, motive type, is concealing information, has hidden relationship
- **Deliberately excluded features:** narrative prominence and introduction timing (a real detective wouldn't know these — they leak story structure)
- **Label distribution:** 49.6% Uninvolved, 22.5% Villain, 17.7% Victim, 9.1% Witness

**Visual:** A compact stats table + a pie/bar chart of the label distribution. Optionally show the medium breakdown.

| Metric | Value |
|---|---|
| Stories | 576 |
| Nodes | 14,020 |
| Edges | ~32,000 |
| Relation types | 16 (8 + 8 inverse) |
| Avg characters/story | 10.2 |
| Avg edges/story | 27.7 |

**Speaker notes:**
We built a dataset of 576 murder mystery plots scraped from Wikipedia and extracted into structured knowledge graphs using an LLM. Each story becomes a heterogeneous graph with characters, locations, occupations, and organizations as nodes, connected by typed directed edges. We deliberately excluded narrative features like "how central is this character to the story" — a real detective wouldn't know that. The 8 features we kept are all things a detective could actually observe or infer.

---

## Slide 5: Worked Example — "The Red-Headed League"

**Title:** From Synopsis to Knowledge Graph: *The Red-Headed League* (Doyle, 1891)

**Key points:**
- **Synopsis (abridged):** Jabez Wilson, a London pawnbroker, consults Sherlock Holmes after a bizarre organization called the Red-Headed League — which paid him to copy the encyclopedia — suddenly dissolved. Holmes deduces that Wilson's assistant, Vincent Spaulding (real name John Clay), invented the League to keep Wilson away from his shop while Clay dug a tunnel from the shop's cellar into the vault of the City and Suburban Bank next door. Holmes, Watson, Inspector Jones, and bank chairman Mr. Merryweather lie in wait in the vault and catch Clay and his accomplice Duncan Ross red-handed.

- **Extracted graph: 7 characters, 5 locations, 8 occupations, 3 organizations, 18 edges**

- **Characters (with labels and key features):**

| Character | Label | Has Motive | Concealing | Hidden Rel. |
|---|---|---|---|---|
| Sherlock Holmes | Uninvolved | — | — | — |
| Dr. Watson | Uninvolved | — | — | — |
| Mr. Jabez Wilson | Uninvolved | — | — | — |
| **Vincent Spaulding / John Clay** | **Villain** | **yes** | **yes** | **yes** |
| **Duncan Ross / Archie** | **Villain** | **yes** | **yes** | **yes** |
| Inspector Peter Jones | Uninvolved | — | — | — |
| Mr. Merryweather | Uninvolved | — | — | — |

- **Sample edges (showing diversity of relation types):**

| Source | Relation | Target | Type |
|---|---|---|---|
| Holmes | `employed as` | Consulting Detective | char → occ |
| Wilson | `affiliated with` | Red-Headed League | char → org |
| Duncan Ross | `leads` | Red-Headed League | char → org |
| Spaulding/Clay | `present at` | Saxe-Coburg Square | char → loc |
| Spaulding/Clay | `near to` | City & Suburban Bank vault | char → loc |
| Holmes | `investigates` | Spaulding/Clay | char → char |
| Inspector Jones | `accuses` | Spaulding/Clay | char → char |
| Watson | `friends with` | Holmes | char → char |

- **What the model sees at test time:** All of the above — **except** any crime edges. The model must infer that Spaulding/Clay is the villain from the circumstantial evidence: he has motive, is concealing information, has a hidden relationship, is present near the bank vault, and is the target of investigation.

**Visual:** Draw the full graph for this story. Use color-coded nodes: blue = Character, green = Location, orange = Occupation, purple = Organization. Directed labeled edges between them. Optionally put a dashed box around the villain nodes to show what the model is trying to predict. This is the single most important visual in the deck — spend time making it clear and attractive.

**Speaker notes:**
Let's walk through a concrete example. In "The Red-Headed League," Jabez Wilson comes to Holmes because a strange organization dissolved. Holmes deduces that Wilson's assistant — Vincent Spaulding, really the criminal John Clay — created the League as a distraction while he tunneled into the neighboring bank vault.

Here's what this looks like as a knowledge graph. Seven characters connected to locations like Baker Street and the bank vault, occupations like Consulting Detective and Criminal, and organizations like the Red-Headed League and Scotland Yard. The edges capture the relational structure: Holmes investigates Spaulding, Wilson is affiliated with the League, Ross leads the League, Spaulding is present near the bank vault.

Now notice the two villains: Spaulding/Clay and Duncan Ross. They're the only characters with motive, concealment, and hidden relationships flagged. They're connected to the Red-Headed League — a sham organization with maximum secrecy. They're physically near the crime scene. That's exactly the pattern the R-GCN learns to detect across 576 stories.

---

## Slide 6: Relational Graph Construction

**Title:** Building the Knowledge Graph

**Key points:**
- **Heterogeneous, directed multigraph:** 4 node types × multiple edge types
- **Node types and their features:**
  - **Character:** 8 features (motive, alibi, concealment, etc.)
  - **Location:** accessibility, isolability, evidentiary value
  - **Occupation:** authority, access, capability levels
  - **Organization:** institutional power, secrecy, financial scale
- **Edge types (canonical):** `employed as`, `resides at`, `present at`, `affiliated with`, `investigates`, `deceives`, `personal_bond`, `kills` (+ others)
- **Inverse edges:** Every edge type gets an automatic inverse (e.g., `investigates` → `investigated_by`). This is critical — information must flow in both directions through the graph.
- **Why inverses matter:** Without them, if A investigates B, only B receives a message from A during message passing. With inverses, A also learns that it is investigating B. Bidirectional information flow lets the model reason about relationships from both sides.

**Visual:** Use the Red-Headed League graph from Slide 5 — highlight one edge (e.g., Holmes `investigates` Spaulding) and show the automatic inverse (`investigated_by`) with a dashed arrow going the other direction. Color-code by node type.

**Speaker notes:**
Each story becomes a heterogeneous directed multigraph. Characters connect to locations (lives at, present at), occupations (employed as), organizations (affiliated with), and to each other (deceives, investigates, personal bond). A critical design choice: we add inverse edges for every relationship. If detective A investigates suspect B, we want information flowing both ways during message passing. Without inverses, only B would receive a message about A — A wouldn't know it's investigating B. This bidirectionality is essential for the R-GCN to reason about relationships from both perspectives.

---

## Slide 7: R-GCN Encoder — The Math

**Title:** R-GCN: Relational Message Passing

**Key points:**
- Standard GCN aggregates neighbor features with a single shared weight matrix
- **R-GCN uses a separate weight matrix per relation type** — this is what lets it distinguish between "X lives with Y" and "X deceives Y"
- **Propagation rule** (Schlichtkrull et al., 2018):

$$h_i^{(l+1)} = \sigma\!\left(W_0^{(l)} h_i^{(l)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)}\right)$$

- **Three components:**
  1. **Self-loop:** $W_0^{(l)} h_i^{(l)}$ — the node retains its own information
  2. **Relational aggregation:** For each relation type $r$, aggregate transformed neighbor features with relation-specific weights $W_r^{(l)}$
  3. **Per-relation normalization:** $c_{i,r} = |\mathcal{N}_i^r|$ — normalize by the number of neighbors under each relation, not global degree

- **Basis decomposition** (parameter sharing to avoid explosion):

$$W_r^{(l)} = \sum_{b=1}^{B} a_{rb}^{(l)} V_b^{(l)}$$

  - $B$ shared basis matrices $V_b$ (learned), relation-specific scalar coefficients $a_{rb}$ (learned)
  - Reduces parameters from $|\mathcal{R}| \times d \times d$ to $B \times d \times d + |\mathcal{R}| \times B$
  - We use $B = 30$ bases across 16 relation types

- **Architecture:** 2 R-GCN layers, hidden dim = 32, ReLU activation between layers, no activation on final layer
- Input features projected to hidden dim via a linear layer before the first R-GCN layer

**Visual:** A diagram showing the message-passing process: a central node receiving differently-colored messages from neighbors along different relation types (e.g., blue for "works with", red for "deceives", green for "lives at"). Show the self-loop as a separate arrow. Optionally show the basis decomposition as a visual: $B$ shared basis matrices being linearly combined with per-relation coefficients.

**Speaker notes:**
This is the core of the R-GCN. The key difference from a standard GCN is that each relation type gets its own weight matrix. When the model aggregates information from a node's neighbors, it treats "X works with Y" completely differently from "X deceives Y." That's exactly what we need — in a murder mystery, being someone's coworker means something very different from being their blackmailer.

The self-loop term lets the node keep its own features. The per-relation normalization prevents high-degree relation types from dominating. And basis decomposition keeps the parameter count manageable — instead of 16 separate weight matrices, we learn 30 shared basis matrices and combine them with per-relation coefficients. Each relation's weight matrix is a learned linear combination of these bases.

---

## Slide 8: Decoders — DistMult + Node Classifier

**Title:** Two Decoders: Link Prediction + Villain Classification

**Key points:**
- The R-GCN encoder produces node embeddings $h_i^{(L)}$ — shared representations used by two decoder heads

- **Decoder 1 — DistMult (Link Prediction):**

$$f(v_i, r, v_j) = \sum_k w_{rk} \cdot h_{ik}^{(L)} \cdot h_{jk}^{(L)}$$

  - Scores whether a triple $(v_i, r, v_j)$ is true
  - Learns one relation embedding $w_r$ per relation type
  - Element-wise product of source embedding, relation embedding, and target embedding
  - Trained with sigmoid cross-entropy: push real triples high, corrupted triples low

- **Decoder 2 — Node Classifier (Villain Prediction):**

$$\hat{y}_i = \text{softmax}(W_{\text{cls}} \cdot h_i^{(L)})$$

  - Binary classification: Villain vs. Non-Villain for each character node
  - Cross-entropy loss over labeled nodes only (unlabeled nodes still participate in message passing)

- **Multi-task training:** Joint loss = link prediction loss + classification loss
  - The link prediction objective acts as a regularizer — it forces the encoder to learn embeddings that capture relational structure, not just class-discriminative features

**Visual:** A diagram showing the shared encoder feeding into two decoder branches. Left branch: DistMult scoring triples. Right branch: softmax classifier over character nodes.

**Speaker notes:**
We use the R-GCN encoder to produce node embeddings, then feed those into two decoders simultaneously. DistMult scores whether relational triples are true — it's a link prediction objective that forces the embeddings to encode relational structure. The node classifier is a simple linear layer plus softmax that predicts villain vs non-villain for each character. Training both jointly means the encoder has to learn embeddings that are good for *both* tasks — capturing relational patterns while also being discriminative for villain classification. The link prediction loss acts as a regularizer.

---

## Slide 9: Training Nuances

**Title:** Training Details: Negative Sampling & Design Choices

**Key points:**

- **Negative sampling (triplet corruption):**
  - For each real triple $(s, r, o)$, generate corrupted triples by randomly replacing either the subject or the object (never both, never the relation)
  - The model learns to score real triples high and corrupted triples low
  - Loss:

$$\mathcal{L}_{\text{link}} = -\sum_{(s,r,o) \in \mathcal{E}} \left[\log \sigma(f(s,r,o)) + \frac{1}{N}\sum_{n=1}^{N} \log(1 - \sigma(f(s',r,o')))\right]$$

- **Bidirectional edges (inverse relations):**
  - Every canonical relation gets an automatic inverse (16 total = 8 + 8)
  - Without inverses, message passing is asymmetric — information only flows along edge direction
  - With inverses, the model can reason from both sides of every relationship

- **Crime-edge masking at evaluation:**
  - At test time, `kills`, `killed_by`, `sexually_assaults`, and financial crime edges are removed
  - Validated that masking had zero impact — model was already reasoning from circumstantial evidence

- **Feature selection:**
  - Excluded narrative prominence and introduction timing (story-structure leakage)
  - Removing them had ~zero impact on F1 and reduced variance — model wasn't using them

- **Class balancing:** Inverse-frequency weighting on the classification loss to handle 50% Uninvolved vs 22% Villain imbalance

**Speaker notes:**
A few important training details. First, negative sampling: for every real edge in the graph, we create fake edges by corrupting either the source or target entity. The model learns to distinguish real from fake. This is how the link prediction objective works — it's contrastive learning on graph triples.

Second, bidirectional edges. We explicitly add inverse relations. If A investigates B, we also add "B is investigated by A." This ensures information flows both ways during message passing.

Third, at test time we mask all crime-revealing edges. But here's the validation: when we tested this, masking had literally zero effect. The model had already learned to make all its predictions from circumstantial evidence — motive, alibi, concealment, social connections. It wasn't using the "kills" edges at all.

---

## Slide 10: Results — Metrics

**Title:** Results: R-GCN vs. Baselines

**Key points:**
- **Cross-validated results** (5 random train/test splits, 50 epochs each):

| Metric | R-GCN | LogReg Baseline |
|---|---|---|
| **Villain Precision** | **0.825 ± 0.033** | 0.684 ± 0.060 |
| **Villain Recall** | 0.708 ± 0.019 | **0.752 ± 0.046** |
| **Villain F1** | **0.762 ± 0.025** | 0.715 ± 0.045 |
| **Overall Accuracy** | **0.904 ± 0.007** | 0.869 ± 0.019 |

- **Interpretation:**
  - R-GCN is a **precise detective**: when it accuses someone, it's right 83% of the time (vs 68% for LogReg)
  - LogReg catches slightly more villains (75% vs 71%) but produces **2× more false accusations**
  - R-GCN F1 leads by 5 points; overall accuracy is 90.4%

- **Story-level metrics** (seed 42, 57 test stories):
  - R-GCN solves 98.2% of cases; LogReg solves 100%
  - R-GCN makes **clean solves** (no false accusations) in **77.2%** of stories vs LogReg's 66.7%
  - **Zero stories** remain unsolved by both models

- **Spectral baseline comparison** (validates that *typed* relations matter, not just topology):

| Approach | Villain F1 | Overall Acc |
|---|---|---|
| Laplacian eigenmaps only | 0.113 | 25.5% |
| Features-only LogReg | 0.705 | 58.5% |
| Features + Spectral | 0.696 | 58.8% |
| **R-GCN** | **0.762** | **90.4%** |

- **Key ablation insight:** Graph topology alone is useless (25.5%). Adding spectral embeddings to features adds nothing. The R-GCN's 32-point accuracy lead comes from **typed relational message passing** — not raw graph structure.

**Visual:** Two side-by-side tables (cross-val results and spectral comparison). Optionally a bar chart comparing F1 scores across approaches. Highlight the precision gap.

**Speaker notes:**
Here are the numbers. The R-GCN leads on precision by 14 points — when it accuses someone, it's right 83% of the time. The logistic regression catches slightly more villains but makes twice as many false accusations. At the story level, the R-GCN solves cases cleanly — no false accusations — in 77% of stories compared to 67% for the baseline.

The spectral baseline is the key ablation. Laplacian eigenmaps on the graph topology alone are essentially useless — knowing who's connected to whom doesn't predict villainy. But when you add typed relational message passing — letting the model distinguish between "works with" and "deceives" — you get a 32-point accuracy jump. It's the *labeled* relational structure that matters, not the topology itself.

---

## Slide 11: Results — Disagreement Pattern

**Title:** Where the Models Disagree

**Key points:**
- When R-GCN and LogReg disagree, there's a clean asymmetry:
  - **R-GCN wins by clearing innocents** — it can rule out suspects using relational context (alibis, spatial separation, social structure)
  - **LogReg wins by catching villains** — it casts a wider net by flagging anyone with suspicious features
- This is exactly the architectural trade-off: **graph structure helps the model *exclude* false suspects more than it helps *find* true ones**
- A character with motive but strong alibis and spatial separation from the crime is probably innocent — the R-GCN can see this; the LogReg just sees "has motive" and flags them

- **False positive breakdown:**

| Model | False Positives | Uninvolved accused | Witnesses accused | Victims accused |
|---|---|---|---|---|
| R-GCN | 19 | 10 | 7 | 2 |
| LogReg | 39 | 22 | 13 | 4 |

- R-GCN cuts false accusations in half (39 → 19)

**Speaker notes:**
This slide tells the real story of what the graph structure buys you. When the two models disagree, the R-GCN always wins by clearing innocent characters — it can see that someone has an alibi, lives far from the crime scene, or has a professional relationship that explains their connection to the victim. The LogReg just sees feature values in isolation. The graph structure is better at ruling people out than finding the guilty — which is actually how real detective work operates.

---

## Slide 12: Case Studies — Generalization to Unsolved Cases

**Title:** Held-Out Inference: Real-World Unsolved Cases

**Key points:**
- Two real-world unsolved cases held out of training entirely — the model had never seen these stories
- Crime edges masked (full detective scenario)

- **Case 1: *Zodiac* (2007 film)**
  - The Zodiac killings remain unsolved in real life. Arthur Leigh Allen is the prime suspect.
  - **Result: Model ranked Arthur Leigh Allen #1 with P(villain) = 1.0000**
  - The unidentified "Zodiac Killer" placeholder ranked #2 at 0.998

| Rank | P(Villain) | Character |
|---|---|---|
| 1 | 1.0000 | **Arthur Leigh Allen** (real-world prime suspect) |
| 2 | 0.998 | Zodiac Killer (unidentified placeholder) |
| 3 | 0.217 | Rick Marshall (person of interest) |
| 4–20 | < 0.13 | Police, victims, witnesses |

- **Case 2: *In the Dark: Season 3* (podcast)**
  - Curtis Flowers was wrongfully prosecuted 6 times before exoneration. Doug Evans (prosecutor) is the podcast's antagonist.
  - **Result: Doug Evans ranked #1 (P = 1.0000). Curtis Flowers ranked LAST (#16, P = 0.0000).**
  - The model independently exonerated the wrongfully accused — even though the data file initially had Curtis Flowers mislabeled as Villain

| Rank | P(Villain) | Character |
|---|---|---|
| 1 | 1.0000 | **Doug Evans** (prosecutor / podcast antagonist) |
| 2 | 1.0000 | Willie James Hemphill (alternative suspect) |
| **16** | **0.0000** | **Curtis Flowers** (wrongfully accused — model cleared him) |

- **Why this matters:** The model learned transferable patterns of evidence-based reasoning. It's not memorizing labels. When shown stories it has never seen — including one where the wrongfully accused person was mislabeled as Villain in the data — it correctly identifies the real suspects and exonerates the innocent.

**Visual:** Two side-by-side ranking tables (Zodiac and In the Dark). Highlight the key characters in bold or color.

**Speaker notes:**
These are the results I'm most proud of. We held out two real-world unsolved cases — the model never saw them during training. For the Zodiac case, the model ranked Arthur Leigh Allen — the real-world prime suspect — as the number one most likely villain with maximum confidence. For the In the Dark podcast about Curtis Flowers's wrongful prosecution, the model ranked the prosecutor Doug Evans as the number one suspect and ranked Curtis Flowers — the wrongfully accused man — dead last with a probability of zero. And here's the kicker: the original data file had Curtis Flowers *mislabeled* as Villain. The model didn't memorize that label. It looked at the relational evidence and independently concluded he was innocent. That's the strongest evidence we have that this model is doing genuine evidence-based reasoning.

---

## Slide 13: Takeaways

**Title:** What We Learned

**Key points:**
1. **Typed relational structure is what matters** — graph topology alone is useless; it's the *labeled* relationships (investigates, deceives, personal bond) that drive performance
2. **The R-GCN is a cautious, precise detective** — higher precision, fewer false accusations, but slightly lower recall than a simple feature-based approach
3. **Graph structure helps exclude suspects more than find them** — the architectural advantage is in ruling out innocents via relational context (alibis, spatial separation)
4. **The model generalizes to unseen stories** — held-out inference on real-world unsolved cases confirms transferable evidence-based reasoning
5. **Data quality is the dominant lever** — extraction quality (whether the LLM flags villain-indicative features) matters more than model architecture. Three rounds of targeted re-extraction improved F1 by 6 points.
6. **Multi-task training helps** — the link prediction objective regularizes the encoder, forcing it to learn relational structure beyond what's needed for classification alone

**Speaker notes:**
Three big takeaways. First, it's not graph topology that helps — it's the typed relationships. The spectral baseline proved that. Second, the R-GCN's advantage is specifically in ruling out false suspects. When you can reason about alibis and spatial relationships, you make fewer false accusations. Third, the model genuinely generalizes — the held-out case studies prove it's learning patterns, not memorizing answers. And the biggest practical lesson: better input data matters more than better models. Our biggest performance gains came from improving the LLM extraction pipeline, not from tuning the R-GCN.

---

## Slide 14: Questions

**Title:** Questions?

**Content:**
- Team: Adam Abramowitz, Max DeSantis, Isaac Dreeben, Abhi Mummaneni
- Reference: Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks" (ESWC 2018)
- Dataset: 576 murder mystery knowledge graphs
- Code and data available in the project repository
- *Have appendix slides ready for deep-dive questions on features, multi-villain performance, re-extraction, schema, and hyperparameters*

---

## Appendix — Extra Slides (if time / Q&A needs them)

### A1: Feature Importance (LogReg coefficients)

| Feature | Coefficient | Direction |
|---|---|---|
| `motive_type` | +1.37 | Specific motive type is strongest villain signal |
| `has_alibi` | -1.34 | Alibi strongly anti-correlates with villain |
| `has_motive` | +0.95 | Binary motive flag |
| `is_concealing_information` | +0.62 | Concealment behavior |
| `gender` | -0.60 | Male characters are more often villains in this corpus |
| `social_status` | -0.35 | Lower status slightly correlates with villain |

### A2: Multi-Villain Performance

| # Villains | Stories | R-GCN Recall | LogReg Recall |
|---|---|---|---|
| 1 | 19 | 100% | 100% |
| 2 | 16 | 90.6% | 87.5% |
| 3 | 9 | 81.5% | 88.9% |
| 4 | 8 | 71.9% | 75.0% |
| 6 | 3 | 38.9% | 38.9% |

Both models struggle with 6-villain ensemble plots (e.g., *Murder on the Orient Express*). This is a structural limit of the detective-fiction framing.

### A3: Re-Extraction Impact

| Round | R-GCN F1 | R-GCN Recall | Stories Unsolved |
|---|---|---|---|
| Original | 0.703 | 0.606 | 11 |
| Round 1 | 0.712 | 0.620 | 2 |
| Round 2 | 0.737 | 0.661 | 1 |
| Round 3 (final) | **0.762** | **0.708** | **0** |

### A4: Graph Schema Detail

**Node types:** Character (8 features), Location (3), Occupation (3), Organization (3)

**Common edge types:** employed as, resides at, present at, affiliated with, investigates, deceives, personal bond, kills (+ inverses of each)

**Feature encoding:** All ordinal features normalized [0,1] within each graph. UNK values encoded as -1. Binary features: yes=1, no=0, UNK=-1.

### A5: Architecture Hyperparameters

| Parameter | Value |
|---|---|
| R-GCN layers | 2 |
| Hidden dimension | 32 |
| Basis matrices (B) | 30 |
| Dropout | 0.4 |
| Optimizer | Adam |
| Epochs | 50 |
| Class balancing | Inverse frequency weighting |
| Normalization | Per-relation: $c_{i,r} = |\mathcal{N}_i^r|$ |
