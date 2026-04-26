"""
Zodiac (FLM_047) Suspect Prediction Analysis

The Zodiac case is unique in our corpus: there is no confirmed villain in
real life, only suspects (the most prominent being Arthur Leigh Allen).
We exclude FLM_047 from training and test data, then run our trained model
on it as inference-only and rank the characters by their predicted villain
probability.

The hypothesis: if our R-GCN learned generalizable patterns of villain
identification, it should rank Arthur Leigh Allen (the prime suspect) high
on the list — without ever having been trained on the answer.

Usage:
    python zodiac_analysis.py [--seed 42]
"""

import argparse
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout

import torch
import torch.nn.functional as F
from torch.optim import Adam

# Make project root importable regardless of cwd
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.rgcn_model import (
    DEVICE, RGCNScratch, NodeClassifier, DistMultDecoder,
    sample_negatives,
)
from model.load_mystery_graphs import load_mystery_graphs


EXCLUDE_FEATURES = {"narrative_prominence", "narrative_introduction_timing"}
EXCLUDE_ENTRIES = {"FLM_047", "POD_035", "TVE_089", "TVE_093"}
INFERENCE_TARGETS = ["FLM_047", "POD_035"]  # TVE_089 and TVE_093 fully excluded


class RGCNBinaryDetective(torch.nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim=64,
                 num_layers=2, num_bases=8, feat_dim=10, dropout=0.2):
        super().__init__()
        self.encoder = RGCNScratch(
            num_nodes, num_relations, hidden_dim,
            num_layers, num_bases, feat_dim, dropout
        )
        self.link_decoder = DistMultDecoder(num_relations, hidden_dim)
        self.node_classifier = NodeClassifier(hidden_dim, 2)

    def encode(self, edge_index, edge_type, node_features=None, num_nodes=None):
        return self.encoder(edge_index, edge_type, node_features, num_nodes)

    def forward(self, edge_index, edge_type, src, dst, rel,
                node_features=None, num_nodes=None):
        emb = self.encode(edge_index, edge_type, node_features, num_nodes)
        return self.link_decoder(emb, src, dst, rel), self.node_classifier(emb)


def train_corpus_model(graph, binary_labels, epochs=50, seed=42):
    """Train the binary R-GCN on the corpus (with FLM_047 excluded)."""
    torch.manual_seed(seed)

    counts = torch.bincount(binary_labels[graph.train_mask], minlength=2).float()
    inv_freq = 1.0 / counts.clamp(min=1.0)
    class_weights = ((inv_freq / inv_freq.sum()) * 2).to(DEVICE)

    model = RGCNBinaryDetective(
        num_nodes=graph.num_nodes, num_relations=graph.num_relations,
        hidden_dim=64, num_layers=2, num_bases=8,
        feat_dim=graph.node_features.shape[1], dropout=0.2,
    ).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3)

    graph.node_features = graph.node_features.to(DEVICE)
    labels_d = binary_labels.to(DEVICE)
    train_mask = graph.train_mask.to(DEVICE)

    train_e = graph.train_edges.to(DEVICE)
    edge_index = train_e[:, :2].t().contiguous()
    edge_type = train_e[:, 2]

    print(f"Training on {graph.num_nodes} nodes, {len(train_e)} train edges...")
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_e))
        shuffled = train_e[perm]
        for start in range(0, len(shuffled), 4096):
            pos = shuffled[start:start+4096]
            neg = sample_negatives(pos, graph.num_nodes, 3)
            all_e = torch.cat([pos, neg])
            labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).to(DEVICE)
            link_scores, class_logits = model(
                edge_index, edge_type, all_e[:, 0], all_e[:, 1], all_e[:, 2],
                node_features=graph.node_features, num_nodes=graph.num_nodes,
            )
            loss = F.binary_cross_entropy_with_logits(link_scores, labels) + \
                   F.cross_entropy(class_logits[train_mask], labels_d[train_mask],
                                   weight=class_weights)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    print(f"Trained in {time.time()-t0:.1f}s")
    return model


def predict_held_out(model, json_dir, target_entry, highlight_names=None):
    """
    Inference on a held-out graph: load it, run the encoder, and rank
    characters by predicted villain probability.

    The encoder is feature-based (input_proj on node features) and the
    classifier is per-node, so the trained weights apply to any new graph
    with the same feature dimension and relation set.

    Crime edges (kills, killed by, sexually assaults, financial
    crime/transaction) are masked at inference time to enforce the
    detective scenario — the model must reason from circumstantial
    evidence only.

    Parameters
    ----------
    highlight_names : list of substrings to highlight in output (e.g.
                      ["arthur leigh allen"] for Zodiac, ["doug evans"]
                      for In the Dark).
    """
    highlight_names = [s.lower() for s in (highlight_names or [])]

    # Load the JSON directly and build the inputs manually
    with open(os.path.join(json_dir, f"{target_entry}.json")) as f:
        data = json.load(f)

    chars = data.get("characters", [])
    occs = data.get("occupations", [])
    locs = data.get("locations", [])
    orgs = data.get("organizations", [])

    # Build local node ID → index mapping
    from model.load_mystery_graphs import (
        extract_node_features, normalize_relation, coarsen_relation,
        NUM_BASE_RELATIONS, CRIME_INTERMEDIATES,
    )

    local_to_idx = {}
    features = []
    char_info = []  # (idx, name, label)
    idx = 0
    for node_type, nodes in [("characters", chars), ("occupations", occs),
                              ("locations", locs), ("organizations", orgs)]:
        for n in nodes:
            local_to_idx[n["id"]] = idx
            features.append(extract_node_features(n, node_type, EXCLUDE_FEATURES))
            if node_type == "characters":
                char_info.append((idx, n.get("name", "?"), n.get("label", "UNK")))
            idx += 1

    # Build edges (canonical + inverse), MASKING crime edges (detective scenario)
    edges = []
    n_crime_masked = 0
    for e in data.get("edges", []):
        if e["source"] not in local_to_idx or e["target"] not in local_to_idx:
            continue
        intermediate = normalize_relation(e["relation"])
        if intermediate in CRIME_INTERMEDIATES:
            n_crime_masked += 1
            continue  # mask this edge — it reveals the answer
        coarse = coarsen_relation(intermediate)
        s = local_to_idx[e["source"]]
        t = local_to_idx[e["target"]]
        edges.append((s, t, coarse))
        edges.append((t, s, coarse + NUM_BASE_RELATIONS))

    if not edges:
        print("No edges for this story; cannot run inference.")
        return None

    edge_t = torch.tensor(edges, dtype=torch.long).to(DEVICE)
    edge_index = edge_t[:, :2].t().contiguous()
    edge_type = edge_t[:, 2]
    feat_t = torch.tensor(features, dtype=torch.float).to(DEVICE)
    num_nodes = len(features)

    # Run the encoder + classifier
    model.eval()
    with torch.no_grad():
        node_emb = model.encoder(edge_index, edge_type,
                                  node_features=feat_t, num_nodes=num_nodes)
        logits = model.node_classifier(node_emb)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu()  # P(villain)

    # Rank characters
    ranked = sorted(char_info, key=lambda x: -probs[x[0]].item())

    title = data["metadata"].get("title", "?")
    year = data["metadata"].get("year", "?")
    print(f"\n{'='*78}")
    print(f"{target_entry} — {title} — Suspect Ranking")
    print(f"{'='*78}\n")
    print(f"Story: {title} ({year})")
    print(f"Total characters: {len(chars)}")
    print(f"Total edges (with inverse): {len(edges)}")
    print(f"Crime edges masked: {n_crime_masked}\n")

    print(f"{'Rank':<5} {'P(Villain)':<11} {'Label':<13} {'Name'}")
    print("-" * 78)
    for rank, (idx, name, label) in enumerate(ranked, 1):
        p = probs[idx].item()
        marker = "  ★" if any(h in name.lower() for h in highlight_names) else ""
        print(f"{rank:<5} {p:<11.4f} {label:<13} {name}{marker}")

    return ranked, probs


# Back-compat alias for older calls
predict_zodiac = predict_held_out


# Real-world suspect highlights for each target story (case-insensitive substring match)
TARGET_HIGHLIGHTS = {
    "FLM_047": ["arthur leigh allen"],          # prime suspect, never charged
    "POD_035": ["doug evans"],                   # prosecutor; podcast frames as antagonist
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graph_dir",
                        default=os.path.join(_PROJECT_ROOT, "data", "graphs"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--out",
                        default=os.path.join(_PROJECT_ROOT, "analysis", "results",
                                              "inference_analysis_results.txt"))
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Override target entry IDs (default: INFERENCE_TARGETS)")
    args = parser.parse_args()

    targets = args.targets or INFERENCE_TARGETS

    print(f"Loading corpus (excluding {sorted(EXCLUDE_ENTRIES)})...")
    with redirect_stdout(io.StringIO()):
        graph = load_mystery_graphs(
            args.graph_dir, seed=args.seed,
            exclude_features=EXCLUDE_FEATURES,
            exclude_entries=EXCLUDE_ENTRIES,
        )
    print(f"Corpus: {graph.num_nodes} nodes, {len(graph.train_edges)} train edges")

    binary_labels = (graph.class_labels == 0).long()
    binary_labels[graph.class_labels < 0] = -1

    model = train_corpus_model(graph, binary_labels, epochs=args.epochs, seed=args.seed)

    # Capture both stdout and write to file
    import sys

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    with open(args.out, "w") as f:
        f.write(f"Held-Out Inference Analysis (real-world / no-confirmed-villain cases)\n")
        f.write(f"=====================================================================\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Targets: {targets}\n")
        f.write(f"Excluded entries (train+test): {sorted(EXCLUDE_ENTRIES)}\n")
        f.write(f"Excluded features: {sorted(EXCLUDE_FEATURES)}\n")
        f.write(f"Run: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        sys.stdout = Tee(sys.__stdout__, f)
        try:
            for target in targets:
                highlights = TARGET_HIGHLIGHTS.get(target, [])
                predict_held_out(model, args.graph_dir, target_entry=target,
                                 highlight_names=highlights)
        finally:
            sys.stdout = sys.__stdout__

    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
