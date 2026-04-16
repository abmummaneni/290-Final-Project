"""
spectral_baseline.py
====================
Laplacian eigenmaps baseline for villain prediction on simplified
character-only mystery graphs.

Approach:
    1. For each story graph, build weighted adjacency matrix
    2. Compute normalized Laplacian eigenmaps (d smallest non-trivial eigenvectors)
    3. Concatenate spectral embedding with node features
    4. Train a logistic regression classifier across all stories
    5. Same story-level train/val/test split as the R-GCN for fair comparison

Usage:
    python spectral_baseline.py [--graph_dir PATH] [--embed_dim 4]
"""

import json
import os
import random
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------------------------------------------------------------------
# Feature extraction (same mapping as load_mystery_graphs.py)
# ---------------------------------------------------------------------------

CHARACTER_FEATURES = [
    "gender", "social_status", "narrative_introduction_timing",
    "has_alibi", "present_at_crime_scene", "has_motive",
    "is_concealing_information", "has_hidden_relationship",
    "motive_type", "narrative_prominence",
]

MOTIVE_TYPE_MAP = {
    "financial": 0.33, "money": 0.33, "love": 0.33,
    "jealousy": 0.50, "protecting family": 0.50, "manipulation": 0.50,
    "power": 0.66, "glory": 0.66, "entitlement": 0.66,
    "revenge": 1.00, "psychopathic tendencies": 1.00,
    "other": 0.50, "unknown": 0.00, "unk": 0.00,
}

CLASS_MAP = {"villain": 0, "victim": 1, "witness": 2, "uninvolved": 3}
CLASS_NAMES = {0: "Villain", 1: "Victim", 2: "Witness", 3: "Uninvolved"}


def safe_float(k, v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return MOTIVE_TYPE_MAP.get(v.strip().lower(), 0.0)
    return 0.0


def extract_features(char):
    feats = char.get("features", {})
    return [safe_float(k, feats.get(k, 0.0)) for k in CHARACTER_FEATURES]


# ---------------------------------------------------------------------------
# Laplacian eigenmaps
# ---------------------------------------------------------------------------

def laplacian_eigenmaps(adj_matrix, embed_dim=4):
    """
    Compute normalized Laplacian eigenmaps.

    Parameters
    ----------
    adj_matrix : np.ndarray of shape (n, n), weighted adjacency matrix
    embed_dim  : number of eigenvectors to keep (smallest non-trivial)

    Returns
    -------
    embedding : np.ndarray of shape (n, embed_dim)
    """
    n = adj_matrix.shape[0]
    if n <= 1:
        return np.zeros((n, embed_dim))

    # Degree matrix
    deg = adj_matrix.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))

    # Normalized Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}
    L_norm = np.eye(n) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    # Eigendecomposition (symmetric → real eigenvalues)
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

    # Skip the first eigenvector (trivial, all-constant for connected components)
    # Take the next embed_dim smallest
    actual_dim = min(embed_dim, n - 1)
    if actual_dim <= 0:
        return np.zeros((n, embed_dim))

    embedding = eigenvectors[:, 1:1 + actual_dim]

    # Zero-pad if graph is too small
    if actual_dim < embed_dim:
        padding = np.zeros((n, embed_dim - actual_dim))
        embedding = np.hstack([embedding, padding])

    return embedding


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_and_embed(graph_dir, embed_dim=4, seed=42,
                   val_fraction=0.1, test_fraction=0.1):
    """
    Load all simple graph JSONs, compute spectral embeddings,
    and split by story.

    Returns
    -------
    dict with keys: X_train, y_train, X_val, y_val, X_test, y_test
         and 'feature_names' for interpretability
    """
    json_files = sorted([f for f in os.listdir(graph_dir) if f.endswith(".json")])
    if not json_files:
        raise FileNotFoundError(f"No .json files in {graph_dir}")

    # Story-level split (same seed and logic as load_mystery_graphs.py)
    rng = random.Random(seed)
    shuffled = json_files[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = max(1, int(test_fraction * n_total))
    n_val = max(1, int(val_fraction * n_total))
    n_train = n_total - n_val - n_test

    train_files = set(shuffled[:n_train])
    val_files = set(shuffled[n_train:n_train + n_val])
    test_files = set(shuffled[n_train + n_val:])

    print(f"Story split — train: {len(train_files)}, "
          f"val: {len(val_files)}, test: {len(test_files)}")

    # Accumulators
    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}

    for fname in json_files:
        with open(os.path.join(graph_dir, fname)) as f:
            data = json.load(f)

        chars = data.get("characters", [])
        edges = data.get("edges", [])
        n = len(chars)
        if n < 2:
            continue

        # Determine split
        if fname in train_files:
            split = "train"
        elif fname in val_files:
            split = "val"
        else:
            split = "test"

        # Build adjacency matrix
        id_to_idx = {c["id"]: i for i, c in enumerate(chars)}
        adj = np.zeros((n, n))
        for e in edges:
            i = id_to_idx.get(e["source"])
            j = id_to_idx.get(e["target"])
            if i is not None and j is not None:
                w = float(e.get("weight", 1.0))
                adj[i, j] = w
                adj[j, i] = w  # undirected

        # Spectral embedding
        spec_emb = laplacian_eigenmaps(adj, embed_dim)

        # Node features + spectral embedding → combined feature vector
        for idx, char in enumerate(chars):
            label_str = char.get("label", "UNK").strip().lower()
            cls = CLASS_MAP.get(label_str, -1)
            if cls < 0:
                continue  # skip UNK

            node_feats = extract_features(char)
            combined = node_feats + list(spec_emb[idx])

            splits[split][0].append(combined)
            splits[split][1].append(cls)

    result = {}
    for split_name in ["train", "val", "test"]:
        X, y = splits[split_name]
        result[f"X_{split_name}"] = np.array(X) if X else np.zeros((0, 10 + embed_dim))
        result[f"y_{split_name}"] = np.array(y) if y else np.zeros(0, dtype=int)

    feature_names = CHARACTER_FEATURES + [f"spectral_{i}" for i in range(embed_dim)]
    result["feature_names"] = feature_names

    print(f"Train: {len(result['y_train'])} chars, "
          f"Val: {len(result['y_val'])} chars, "
          f"Test: {len(result['y_test'])} chars")

    return result


def run_baseline(graph_dir, embed_dim=4, seed=42):
    """Run the full spectral baseline and print results."""

    data = load_and_embed(graph_dir, embed_dim=embed_dim, seed=seed)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Also run features-only baseline for comparison
    n_feats = len(CHARACTER_FEATURES)
    configs = [
        ("Features only (10-dim)", X_train[:, :n_feats], X_val[:, :n_feats], X_test[:, :n_feats]),
        ("Spectral only (4-dim)", X_train[:, n_feats:], X_val[:, n_feats:], X_test[:, n_feats:]),
        (f"Features + Spectral ({n_feats + embed_dim}-dim)", X_train, X_val, X_test),
    ]

    for name, Xtr, Xva, Xte in configs:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        clf.fit(Xtr, y_train)

        for split_name, Xs, ys in [("Train", Xtr, y_train),
                                    ("Val", Xva, y_val),
                                    ("Test", Xte, y_test)]:
            preds = clf.predict(Xs)
            acc = accuracy_score(ys, preds)
            print(f"\n  {split_name} accuracy: {acc:.4f}")

            if split_name == "Test":
                print(f"\n  Per-class breakdown:")
                print(f"  {'Class':<15} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
                print(f"  {'-'*47}")
                report = classification_report(
                    ys, preds, target_names=[CLASS_NAMES[i] for i in range(4)],
                    output_dict=True, zero_division=0
                )
                for cls_id in range(4):
                    name_str = CLASS_NAMES[cls_id]
                    r = report[name_str]
                    print(f"  {name_str:<15} {r['precision']:>8.3f} {r['recall']:>8.3f} "
                          f"{r['f1-score']:>8.3f} {int(r['support']):>8}")

                print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
                cm = confusion_matrix(ys, preds)
                names = [CLASS_NAMES[i] for i in range(4)]
                print(f"  {'':>15}" + "".join(f"{n:>12}" for n in names))
                for i in range(4):
                    row = f"  {names[i]:>15}"
                    for j in range(4):
                        row += f"{cm[i, j]:>12}"
                    print(row)

    return data


if __name__ == "__main__":
    import sys
    graph_dir = sys.argv[1] if len(sys.argv) > 1 else "extraction_simple/data/graphs"
    embed_dim = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    run_baseline(graph_dir, embed_dim=embed_dim)
