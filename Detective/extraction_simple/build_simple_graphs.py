"""
Build simplified character-only weighted graphs from the full heterogeneous graphs.

Each simple graph has:
- Nodes: characters only, with all 10 features (UNK/-1 replaced with 0)
- Edges: weighted, undirected, between character pairs
- Weights: count of (direct relations + shared locations + shared orgs + shared occs),
           normalized to [0,1] within each graph

Input:  extraction/data/graphs/{ID}.json  (576 heterogeneous graphs)
Output: extraction_simple/data/graphs/{ID}.json  (576 simple graphs)

Run:
    cd /Users/maxdesantis/dev/290-Final-Project/Detective
    python -m extraction_simple.build_simple_graphs
"""

import json
import os
from collections import defaultdict

HETERO_DIR = os.path.join(os.path.dirname(__file__), "..", "extraction", "data", "graphs")
SIMPLE_DIR = os.path.join(os.path.dirname(__file__), "data", "graphs")

# Character feature keys in canonical order
CHAR_FEATURES = [
    "gender",
    "social_status",
    "narrative_introduction_timing",
    "has_alibi",
    "present_at_crime_scene",
    "has_motive",
    "is_concealing_information",
    "has_hidden_relationship",
    "motive_type",
    "narrative_prominence",
]

# Map motive_type strings/values to numeric [0,1]
# Schema categories: jealousy=0.25, money=0.5, revenge=0.75, love=1.0, UNK=0
MOTIVE_MAP = {
    "jealousy": 0.25,
    "money": 0.5,
    "financial": 0.5,
    "revenge": 0.75,
    "love": 0.25,
    "power": 0.75,
}


def sanitize_value(key: str, val) -> float:
    """Convert a feature value to a clean float, replacing UNK/-1 with 0."""
    if key == "motive_type":
        if isinstance(val, str):
            return MOTIVE_MAP.get(val.lower(), 0.0)
        if isinstance(val, (int, float)):
            return 0.0 if val == -1 or val == -1.0 else float(val)
        return 0.0
    if isinstance(val, (list, dict)):
        return 0.0
    if isinstance(val, str):
        return 0.0
    if val == -1 or val == -1.0:
        return 0.0
    return float(val)


def build_simple_graph(hetero: dict) -> dict:
    """Convert a heterogeneous graph to a simple character-only weighted graph."""
    characters = hetero.get("characters", [])
    edges = hetero.get("edges", [])
    occupations = hetero.get("occupations", [])
    locations = hetero.get("locations", [])
    organizations = hetero.get("organizations", [])

    # Identify character IDs
    char_ids = {c["id"] for c in characters}

    # Build simple character nodes with sanitized features
    simple_chars = []
    for c in characters:
        features = {}
        for key in CHAR_FEATURES:
            val = c.get("features", {}).get(key, 0)
            features[key] = sanitize_value(key, val)
        simple_chars.append({
            "id": c["id"],
            "name": c["name"],
            "label": c.get("label", "UNK"),
            "features": features,
        })

    # --- Compute pairwise weights ---
    # Use frozenset of (char_a, char_b) as key for undirected edges
    pair_weights = defaultdict(float)

    # 1. Direct character-character edges: +1 per distinct relation
    for e in edges:
        src, tgt = e["source"], e["target"]
        if src in char_ids and tgt in char_ids and src != tgt:
            pair = frozenset([src, tgt])
            pair_weights[pair] += 1.0

    # 2. Shared context via non-character nodes
    # Build reverse index: non-char node -> set of characters connected to it
    shared_context = defaultdict(set)
    for e in edges:
        src, tgt = e["source"], e["target"]
        # Character -> non-character
        if src in char_ids and tgt not in char_ids:
            shared_context[tgt].add(src)
        # Non-character -> character (shouldn't happen often but handle it)
        if tgt in char_ids and src not in char_ids:
            shared_context[src].add(tgt)

    # For each non-character node, all connected characters share context
    for node_id, connected_chars in shared_context.items():
        char_list = sorted(connected_chars)
        for i in range(len(char_list)):
            for j in range(i + 1, len(char_list)):
                pair = frozenset([char_list[i], char_list[j]])
                pair_weights[pair] += 1.0

    # 3. Normalize weights to [0, 1] within this graph
    max_weight = max(pair_weights.values()) if pair_weights else 1.0
    if max_weight == 0:
        max_weight = 1.0

    simple_edges = []
    for pair, weight in sorted(pair_weights.items(), key=lambda x: sorted(x[0])):
        a, b = sorted(pair)
        simple_edges.append({
            "source": a,
            "target": b,
            "weight": round(weight / max_weight, 4),
            "raw_weight": weight,
        })

    return {
        "characters": simple_chars,
        "edges": simple_edges,
        "metadata": hetero.get("metadata", {}),
        "graph_stats": {
            "num_characters": len(simple_chars),
            "num_edges": len(simple_edges),
            "max_raw_weight": max_weight,
        },
    }


def main():
    os.makedirs(SIMPLE_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(HETERO_DIR) if f.endswith(".json"))
    total_chars = 0
    total_edges = 0
    skipped = 0

    for fname in files:
        with open(os.path.join(HETERO_DIR, fname)) as f:
            hetero = json.load(f)

        simple = build_simple_graph(hetero)

        # Skip graphs with fewer than 2 characters (can't form edges)
        if simple["graph_stats"]["num_characters"] < 2:
            skipped += 1
            continue

        with open(os.path.join(SIMPLE_DIR, fname), "w") as f:
            json.dump(simple, f, indent=2, ensure_ascii=False)

        total_chars += simple["graph_stats"]["num_characters"]
        total_edges += simple["graph_stats"]["num_edges"]

    written = len(files) - skipped
    print(f"Built {written} simple graphs ({skipped} skipped for <2 characters)")
    print(f"  Avg characters/graph: {total_chars / max(written, 1):.1f}")
    print(f"  Avg edges/graph: {total_edges / max(written, 1):.1f}")


if __name__ == "__main__":
    main()
