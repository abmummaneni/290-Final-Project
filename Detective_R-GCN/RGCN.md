# R-GCN Implementation Reference
**Paper:** *Modeling Relational Data with Graph Convolutional Networks*  
Schlichtkrull et al., 2017 — [arXiv:1703.06103](https://arxiv.org/abs/1703.06103)

Use this file as a reference when reviewing R-GCN implementations. Paste your code and ask Claude to check it against the checklist sections below.

---

## How to Use This File

Paste your code into the chat alongside a question like:
- *"Check my R-GCN layer against the propagation rule"*
- *"Is my basis decomposition correct?"*
- *"Does my DistMult scorer match the paper?"*
- *"Check my negative sampling loop"*

Claude will compare your implementation against the specifications below.

---

## 1. Graph Representation

The paper models knowledge graphs as directed labeled multigraphs:

$$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})$$

**Checklist:**
- [ ] Edges are **directed** and **labeled** — each edge is a triple `(subject, relation, object)`
- [ ] Both canonical and **inverse relations** are included in `R` during encoding (e.g., `born_in` and `born_in_inv` are both present) — the paper explicitly states this
- [ ] Inverse relations are added programmatically by reversing each edge and assigning a new relation type id
- [ ] Node features at layer 0: if no features available, use a **unique one-hot vector** per node (`h_i^(0) = e_i`)
- [ ] For block decomposition: the one-hot input is mapped to a dense representation via a single linear transformation before the first R-GCN layer

---

## 2. R-GCN Propagation Rule

The core update equation (Eq. 2 in the paper):

$$h_i^{(l+1)} = \sigma\!\left(W_0^{(l)} h_i^{(l)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)}\right)$$

**Checklist:**
- [ ] **Self-connection term** `W_0 @ h_i` is present and uses a **separate** weight matrix from the relation matrices
- [ ] Neighbor aggregation is **partitioned by relation type** — neighbors under different relations are aggregated separately before summing
- [ ] **Normalization constant** `c_{i,r}` is applied per relation — typically `|N_i^r|` (number of neighbors of node i under relation r), not global degree
- [ ] Normalization divides each neighbor's contribution **before** accumulation, not after
- [ ] **Activation function** (ReLU) is applied **after** the full sum (self + all relations), not per-relation
- [ ] For the **final layer** (output layer): no activation function — raw logits go directly to the classifier or decoder
- [ ] Layer input `h_i^(l)` has shape `[num_nodes, d_l]`; layer output `h_i^(l+1)` has shape `[num_nodes, d_{l+1}]`

---

## 3. Basis Decomposition (Eq. 3 in paper)

$$W_r^{(l)} = \sum_{b=1}^{B} a_{rb}^{(l)} V_b^{(l)}$$

**Checklist:**
- [ ] `B` basis matrices `V_b` of shape `[d_{l+1}, d_l]` are defined — **shared across all relations**
- [ ] Per-relation coefficients `a_rb` of shape `[num_relations, B]` are learned — **relation-specific**
- [ ] `W_r` is constructed as a linear combination: `W_r = sum_b(a_rb * V_b)`
- [ ] Only `a_rb` depends on `r`; `V_b` does not
- [ ] `B` is a hyperparameter with `B < num_relations` (paper uses B ∈ {10, 20, 30, 40})
- [ ] If `B = 0` (no decomposition): each `W_r` is a fully independent matrix of shape `[d_{l+1}, d_l]`
- [ ] Basis decomposition is applied **per layer** — each layer has its own `V_b^(l)` and `a_rb^(l)`

---

## 4. Block-Diagonal Decomposition (Eq. 4 in paper)

$$W_r^{(l)} = \bigoplus_{b=1}^{B} Q_{br}^{(l)}, \quad Q_{br}^{(l)} \in \mathbb{R}^{(d_{l+1}/B) \times (d_l/B)}$$

**Checklist:**
- [ ] Each `W_r` is block-diagonal with `B` blocks of size `[d_{l+1}/B, d_l/B]`
- [ ] `d_l` and `d_{l+1}` must be divisible by `B`
- [ ] Off-diagonal blocks are **zero** — no cross-talk between subspaces
- [ ] Each relation still has its **own** block-diagonal matrix (unlike basis decomposition, no cross-relation sharing)
- [ ] In practice: implement as `B` separate small dense matrices per relation and apply them to non-overlapping slices of the input, then concatenate

---

## 5. Entity Classification Decoder

$$\hat{y}_i = \text{softmax}(W h_i^{(L)})$$

**Checklist:**
- [ ] Classification matrix `W` of shape `[num_classes, d_L]` is applied to the **final layer** embeddings
- [ ] **Softmax** is applied to produce a probability distribution over classes
- [ ] Loss is **cross-entropy** over labeled nodes only: `L = -sum_{i in labeled} log(y_hat[i, y_i])`
- [ ] Unlabeled nodes still **participate in message passing** — they are only excluded from the loss computation
- [ ] Paper uses **2 R-GCN layers** for entity classification
- [ ] Paper uses **16 hidden units** (10 for AM dataset)
- [ ] **L2 regularization** on first-layer weights: `lambda in {0, 5e-4}` depending on dataset
- [ ] **No dropout** used in the paper

---

## 6. Link Prediction: DistMult Decoder

$$f(v_i, r, v_j) = h_i^{(L)\top} \operatorname{diag}(w_r) h_j^{(L)} = \sum_k w_{rk} \cdot h_{ik}^{(L)} \cdot h_{jk}^{(L)}$$

**Checklist:**
- [ ] One relation embedding vector `w_r` of shape `[d_L]` per relation type — **learned jointly** with the encoder
- [ ] Score is a **coordinate-wise weighted dot product** — element-wise multiply `h_i * h_j`, then dot with `w_r`
- [ ] Score is **symmetric**: `f(v_i, r, v_j) == f(v_j, r, v_i)` — verify this holds in your implementation
- [ ] Raw score is passed through **sigmoid** before the loss: `sigma(f(v_i, r, v_j))`
- [ ] The decoder is language-model style: it produces a **scalar score**, not a probability distribution

---

## 7. Negative Sampling and Training Loss

$$\mathcal{L} = -\sum_{(v_i, r, v_j) \in \mathcal{E}} \left(\log \sigma(f(v_i, r, v_j)) + \frac{1}{N}\sum_{n=1}^{N} \log(1 - \sigma(f(v_i', r, v_j')))\right)$$

**Checklist:**
- [ ] For each positive triple, generate `N` negative triples by **randomly replacing subject OR object** with a random entity
- [ ] Relation type `r` is **kept fixed** during corruption — only the entity is replaced
- [ ] Negative samples are **not filtered** for accidental true triples in the paper's setup
- [ ] Negative loss is averaged over `N` samples (divided by `N`)
- [ ] Sigmoid `sigma` is applied to the raw DistMult score before computing log
- [ ] Positive triples: maximize `log sigma(f)` → push score high
- [ ] Negative triples: maximize `log(1 - sigma(f))` → push score low
- [ ] Both encoder and decoder parameters are updated jointly via this loss

---

## 8. Architecture and Hyperparameters

### Entity Classification (from paper Table 6)
| Dataset | L2 penalty | # Basis functions B | # Hidden units |
|---------|-----------|-------------------|----------------|
| AIFB    | 0         | 0 (none)          | 16             |
| MUTAG   | 5e-4      | 30                | 16             |
| BGS     | 5e-4      | 40                | 16             |
| AM      | 5e-4      | 40                | 10             |

- Number of layers: **2**
- Epochs: **50**
- Normalization: `c_{i,r} = |N_i^r|`
- Dropout: **none**
- Optimizer: **Adam**

### Link Prediction
- Number of layers: **2** (with fully-connected layers interspersed)
- Optimizer: **Adam**
- Normalization: `c_{i,r} = |N_i^r|`

**Checklist:**
- [ ] Number of R-GCN layers matches your intended depth
- [ ] Normalization constant is computed **per node per relation**, not globally
- [ ] Adam optimizer is used (not SGD)
- [ ] For entity classification: L2 penalty is applied to **first layer weights only**

---

## 9. Common Implementation Mistakes

These are the most frequent ways an R-GCN implementation can silently deviate from the paper:

| Mistake | What the paper actually does |
|---------|------------------------------|
| Sharing `W_0` with relation matrices | `W_0` is a **separate** matrix for self-connections |
| Normalizing by global degree | Normalize by **per-relation** neighbor count `|N_i^r|` |
| Excluding unlabeled nodes from message passing | All nodes participate in propagation; only labeled nodes contribute to loss |
| Omitting inverse relations | Inverse relations must be explicitly added to `R` |
| Applying activation after final layer | No activation on the output layer |
| Using a single `W` for all relations | Each relation `r` has its own `W_r^(l)` (possibly via decomposition) |
| Replacing both subject and object in one negative sample | Replace **either** subject **or** object, not both simultaneously |
| Applying basis decomposition across layers | Each layer has **independent** basis matrices `V_b^(l)` |

---

## 10. Equations Quick Reference

| Symbol | Meaning |
|--------|---------|
| `h_i^(l)` | Hidden state of node i at layer l, shape `[d_l]` |
| `W_r^(l)` | Relation-specific weight matrix at layer l, shape `[d_{l+1}, d_l]` |
| `W_0^(l)` | Self-connection weight matrix at layer l, shape `[d_{l+1}, d_l]` |
| `N_i^r` | Set of neighbors of node i under relation r |
| `c_{i,r}` | Normalization constant = `|N_i^r|` |
| `V_b^(l)` | b-th basis matrix at layer l (basis decomp), shape `[d_{l+1}, d_l]` |
| `a_rb^(l)` | Scalar coefficient for relation r, basis b, layer l |
| `Q_br^(l)` | b-th block for relation r at layer l (block decomp), shape `[d_{l+1}/B, d_l/B]` |
| `w_r` | DistMult relation embedding, shape `[d_L]` |
| `f(v_i, r, v_j)` | DistMult score for triple (i, r, j) |
| `B` | Number of bases (basis decomp) or blocks (block decomp) |
| `L` | Number of R-GCN layers |
