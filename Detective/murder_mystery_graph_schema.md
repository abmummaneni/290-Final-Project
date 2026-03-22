# Murder Mystery Knowledge Graph: Unified Schema

## Overview

This document defines the schema for a heterogeneous, directed multigraph representation of murder mystery books and films. The graph is designed for use with a graph autoencoder architecture (following the $\alpha$LoNGAE framework) to support multi-task inference: villain prediction, motivation prediction, and hidden relationship detection.

**Core design principles:**
- All ordinal features are **relative within a single graph instance** (time-agnostic)
- The graph is a **directed multigraph** — edges are typed, directed, and multiple edges between the same pair of nodes are permitted
- Node features follow an **open UNK policy** — features not extractable from a synopsis are marked UNK and treated as inference targets
- Ground truth labels are **sparse** — only explicitly revealed information is used for evaluation

---

## 1. Node Types

The graph contains four node types: `Character`, `Occupation`, `Location`, and `Organization`.

---

### 1.1 Character Nodes

Characters are the primary nodes of interest. Their labels are the classification targets.

#### Node Label (Classification Target)

| Label | Description |
|---|---|
| `Villain` | The perpetrator of the crime |
| `Victim` | The target of the crime |
| `Witness` | A character with relevant knowledge |
| `Uninvolved` | A character with no direct role |

> These are 4-class classification targets, not features. A character may not receive a label if their role is ambiguous in the synopsis.

#### Node Feature Vector $\mathbf{x}_{\text{char}} \in [0,1]^9$

Features are categorized by availability at inference time.

**Static features** — always extractable:

| Index | Feature | Encoding |
|---|---|---|
| 0 | Gender | Categorical: `male=0`, `female=1`, `other=0.5`, `UNK=-1` |
| 1 | Social status | Ordinal $[0,1]$: relative within graph |
| 2 | Narrative introduction timing | Ordinal $[0,1]$: early=0, late=1 |

**Revealed features** — extractable from full synopsis, may be UNK at inference:

| Index | Feature | Encoding |
|---|---|---|
| 3 | Has alibi | Binary: `yes=1`, `no=0`, `UNK=-1` |
| 4 | Present at crime scene | Binary: `yes=1`, `no=0`, `UNK=-1` |
| 5 | Has motive | Binary: `yes=1`, `no=0`, `UNK=-1` |

**Inferred features** — may be UNK even in training data:

| Index | Feature | Encoding |
|---|---|---|
| 6 | Is concealing information | Binary: `yes=1`, `no=0`, `UNK=-1` |
| 7 | Has hidden relationship | Binary: `yes=1`, `no=0`, `UNK=-1` |
| 8 | Motive type | Categorical: `jealousy`, `money`, `revenge`, `love`, `UNK=-1` |

**Narrative feature** (held out for ablation study):

| Index | Feature | Encoding |
|---|---|---|
| 9 | Narrative prominence | Ordinal $[0,1]$: relative page/screen time within graph |

> Narrative prominence is excluded from the primary model but retained for ablation comparison experiments.

---

### 1.2 Occupation Nodes

Occupation nodes are shared, reusable entities across books. A character connects to an occupation via an `employed as` edge.

#### Node Feature Vector $\mathbf{x}_{\text{occ}} \in [0,1]^3$

| Index | Feature | Encoding |
|---|---|---|
| 0 | Authority level | Ordinal $[0,1]$: relative within graph |
| 1 | Access level | Ordinal $[0,1]$: degree of access to people, rooms, information |
| 2 | Capability level | Ordinal $[0,1]$: proxy for education and ability to execute complex crime |

> Example ordinal anchors for capability: `unskilled < skilled trade < educated professional < expert/specialist`

---

### 1.3 Location Nodes

Location nodes represent physical places relevant to the narrative.

#### Node Feature Vector $\mathbf{x}_{\text{loc}} \in [0,1]^3$

| Index | Feature | Encoding |
|---|---|---|
| 0 | Accessibility | Ordinal $[0,1]$: public=0, private=1, relative within graph |
| 1 | Isolability | Ordinal $[0,1]$: likelihood crime occurs unwitnessed |
| 2 | Evidentiary value | Ordinal $[0,1]$: likelihood physical evidence is found here |

> Proximity between locations is captured as a weighted edge in the graph structure, not as a node feature.

---

### 1.4 Organization Nodes

Organization nodes represent institutions, firms, families, or groups relevant to the narrative.

#### Node Feature Vector $\mathbf{x}_{\text{org}} \in [0,1]^3$

| Index | Feature | Encoding |
|---|---|---|
| 0 | Institutional power | Ordinal $[0,1]$: relative influence within the story's social context |
| 1 | Secrecy level | Ordinal $[0,1]$: opacity of internal workings to outsiders |
| 2 | Financial scale | Ordinal $[0,1]$: relative wealth concentration |

> All ordinal encodings are relative within the graph instance. A church group in an isolated village may score as high on institutional power as a law firm in an urban setting.

---

## 2. Edge Types

The graph is a **directed multigraph**. Multiple typed edges between the same pair of nodes are permitted. All edges carry a relation type label; some carry additional weight or feature information.

### 2.1 Character → Occupation Edges

| Relation | Direction | Notes |
|---|---|---|
| `employed as` | directed | Primary occupation assignment |
| `formerly employed as` | directed | Historical occupational signal |

### 2.2 Character → Location Edges

| Relation | Direction | Notes |
|---|---|---|
| `resides at` | directed | Primary residence |
| `present at` | directed | Known to be at location during relevant period |
| `owns` | directed | Property ownership |

### 2.3 Character → Organization Edges

| Relation | Direction | Notes |
|---|---|---|
| `affiliated with` | directed | General membership or association |
| `leads` | directed | Position of authority within organization |
| `employed by` | directed | Formal employment relationship |

### 2.4 Character → Character Edges

Open vocabulary — relation types are extracted from synopses and normalized post-hoc if needed. All edges are directed.

**Example relation types:**

| Relation | Direction | Notes |
|---|---|---|
| `married to` | symmetric (two directed edges) | |
| `related to` | symmetric or directed | Family relationships |
| `friends with` | symmetric | |
| `employs` | directed | |
| `blackmails` | directed | |
| `suspects` | directed | |
| `witnessed by` | directed | |
| `in conflict with` | directed or symmetric | |
| `business partner of` | symmetric | |

> Multiple edges between the same character pair are permitted. For example, a character may be both `married to` and `business partner of` another character.

### 2.5 Location → Location Edges

| Relation | Direction | Notes |
|---|---|---|
| `near to` | symmetric (two directed edges) | Weighted by relative proximity within story setting |

### 2.6 Organization → Location Edges

| Relation | Direction | Notes |
|---|---|---|
| `located at` | directed | Physical presence of organization |

---

## 3. Graph-Level Properties

Each graph instance (one book or film) carries the following metadata, stored outside the feature matrices:

| Property | Description |
|---|---|
| `title` | Title of the work |
| `author` | Author or director |
| `year` | Year of publication or release |
| `medium` | Book or film |
| `synopsis_source` | URL or identifier of synopsis used for extraction |

> These are identifiers and provenance metadata only — they are not fed into the model.

---

## 4. Inference Targets and Evaluation Framework

### 4.1 Primary Task: Villain Prediction

4-class node classification over character nodes: `{Villain, Victim, Witness, Uninvolved}`.

Ground truth available for all books where the villain is explicitly revealed. Expected to be the most abundant label.

### 4.2 Secondary Task: Motivation Prediction

Prediction of motive type feature (index 8 of $\mathbf{x}_{\text{char}}$): `{jealousy, money, revenge, love}`.

Ground truth used **only when explicitly stated** in synopsis. Expected to be less abundant than villain labels.

### 4.3 Tertiary Task: Hidden Relationship Detection

Link prediction on character-character edges not present in the input graph but revealed in the synopsis resolution.

Ground truth used **only when explicitly stated**. Expected to be least abundant.

### 4.4 Joint Loss Function

$$\mathcal{L}_{\text{total}} = \text{MASK}_{\text{villain}} \cdot \mathcal{L}_{\text{villain}} + \text{MASK}_{\text{motivation}} \cdot \mathcal{L}_{\text{motivation}} + \text{MASK}_{\text{relationship}} \cdot \mathcal{L}_{\text{relationship}}$$

Where each mask is applied independently per instance and evaluation metrics are computed only over masked (observed) ground truth.

---

## 5. Ablation Study Design

The following model variants are defined for comparative evaluation:

| Variant | Description |
|---|---|
| **Baseline** | Graph Laplacian embeddings + downstream classifier |
| **AE-NoGraph** | Standard autoencoder on node features only (no graph structure) |
| **LoNGAE-NoFeatures** | $\alpha$LoNGAE on graph topology only, no $\mathbf{X}$ |
| **LoNGAE-Static** | $\alpha$LoNGAE with static character features only |
| **LoNGAE-Full** | $\alpha$LoNGAE with all features including inferred/UNK |
| **LoNGAE-Narrative** | $\alpha$LoNGAE-Full + narrative prominence feature |
| **SingleTask** | Villain prediction only, no auxiliary tasks |
| **MultiTask** | Joint villain + motivation + hidden relationship |
| **Undirected** | LoNGAE-Full with directed edges collapsed to undirected |
| **SimpleGraph** | LoNGAE-Full with multigraph collapsed to simple graph |

---

## 6. Adjacency Representation

Following the $\alpha$LoNGAE framework, the adjacency matrix generalizes to a **relation-specific adjacency tensor**:

$$\mathbf{A}_r \in \{1, 0, \text{UNK}\}^{N \times N}, \quad r \in \mathcal{R}$$

Where $\mathcal{R}$ is the set of all relation types, $1$ denotes a known present edge of type $r$, $0$ denotes a known absent edge, and $\text{UNK}$ denotes an unobserved or missing edge — the target of link prediction.

The augmented input per node $i$ is:

$$\bar{\mathbf{a}}_i = [\mathbf{a}_i \| \mathbf{x}_i] \in \mathbb{R}^{N + F}$$

Where $\mathbf{a}_i$ is the adjacency vector across all relation types and $\mathbf{x}_i$ is the node feature vector.
