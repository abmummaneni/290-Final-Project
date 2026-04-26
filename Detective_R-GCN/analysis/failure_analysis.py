"""
failure_analysis.py
===================
Deep-dive analysis of R-GCN and LogReg failure modes on the detective task.

Analyses:
    1. Multi-villain stories — stories with >1 villain, per-story catch rate
    2. Zero-villain stories — stories labeled with no villain, over-prediction rate
    3. R-GCN vs LogReg disagreements — cases where one wins, other loses
    4. False positive analysis — characters wrongly accused of being villain
    5. Story-level error analysis — fraction of stories where at least one
       villain was caught (the "did we solve the case?" metric)

Usage:
    python failure_analysis.py [--seed 42] [--out failure_analysis_results.txt]

The output file is structured for inclusion in the project paper.
"""

import argparse
import json
import os
import random
import sys
import time
import io
from collections import Counter, defaultdict
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.linear_model import LogisticRegression

# Make project root importable regardless of cwd
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.rgcn_model import (
    DEVICE, RGCNScratch, NodeClassifier, DistMultDecoder,
    sample_negatives,
)
from model.load_mystery_graphs import load_mystery_graphs, CHARACTER_FEATURES


EXCLUDE_FEATURES = {"narrative_prominence", "narrative_introduction_timing"}
EXCLUDE_ENTRIES = {
    "FLM_047",   # Zodiac — no real-world identified villain (inference-only)
    "POD_035",   # In the Dark Season 3 — Curtis Flowers wrongful prosecution; no confirmed villain (inference-only)
    "TVE_089",   # Vera: Hidden Depths — invalid villain entity in extraction (fully excluded)
    "TVE_093",   # Spiral: Series 4 — different kind of story; villain is a criminal network, not a person (fully excluded)
}


# ---------------------------------------------------------------------------
# 1. Model definition (match the cross-val binary detective model)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Story membership reconstruction
# ---------------------------------------------------------------------------

def build_story_membership(json_dir: str, seed: int,
                            exclude_entries: set = None):
    """
    Reproduce the loader's node iteration order to map each global node index
    back to its source story.

    Returns
    -------
    node_story        : list of length num_nodes, each entry is the filename
    story_of_test     : dict story_id → list of (global_node_idx, char_label)
                        for character nodes in that story
    all_stories       : dict story_id → metadata from the JSON
    """
    exclude_entries = exclude_entries or set()
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
    json_files = [f for f in json_files
                  if f.replace(".json", "") not in exclude_entries]

    node_story = []
    story_of_test = defaultdict(list)
    all_stories = {}

    node_offset = 0
    for fname in json_files:
        with open(os.path.join(json_dir, fname)) as f:
            data = json.load(f)

        story_id = fname.replace(".json", "")
        all_stories[story_id] = {
            "title": data.get("metadata", {}).get("title", story_id),
            "medium": data.get("metadata", {}).get("medium", "?"),
            "year": data.get("metadata", {}).get("year", "?"),
            "characters": data.get("characters", []),
        }

        for node_type in ["characters", "occupations", "locations", "organizations"]:
            for node in data.get(node_type, []):
                node_story.append(story_id)
                if node_type == "characters":
                    story_of_test[story_id].append({
                        "global_idx": node_offset,
                        "name": node.get("name", "?"),
                        "label": node.get("label", "UNK").strip().lower(),
                        "features": node.get("features", {}),
                    })
                node_offset += 1

    return node_story, story_of_test, all_stories


# ---------------------------------------------------------------------------
# 3. Train both models
# ---------------------------------------------------------------------------

def train_rgcn(graph, binary_labels, epochs=50, seed=42):
    """Train the binary R-GCN detective model."""
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

    # Get predictions with crime-edge masking
    keep = ~graph.train_crime_mask.to(DEVICE)
    masked = train_e[keep]
    mei = masked[:, :2].t().contiguous()
    met = masked[:, 2]

    model.eval()
    with torch.no_grad():
        emb = model.encode(mei, met, node_features=graph.node_features,
                           num_nodes=graph.num_nodes)
        logits = model.node_classifier(emb)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu()
        preds = logits.argmax(1).cpu()
    return preds, probs


def train_logreg(graph, binary_labels, seed=42):
    """Train a balanced LogReg baseline."""
    train_idx = torch.where(graph.train_mask)[0]
    X_tr = graph.node_features.cpu()[train_idx].numpy()
    y_tr = binary_labels[graph.train_mask].numpy()

    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=seed)
    clf.fit(X_tr, y_tr)

    X_all = graph.node_features.cpu().numpy()
    preds = torch.tensor(clf.predict(X_all))
    probs = torch.tensor(clf.predict_proba(X_all)[:, 1])
    return preds, probs, clf


# ---------------------------------------------------------------------------
# 4. Build test-set story records
# ---------------------------------------------------------------------------

def build_test_story_records(graph, binary_labels,
                              rgcn_preds, logreg_preds,
                              story_of_test, node_story):
    """
    Build a per-story record for every test story. Each record holds the
    ground-truth labels and both models' predictions for every character.

    Returns
    -------
    List of dicts, one per test story.
    """
    test_mask = graph.test_mask.numpy().astype(bool)
    test_story_ids = sorted({node_story[i] for i in range(len(node_story))
                             if test_mask[i]})

    records = []
    for sid in test_story_ids:
        chars = story_of_test[sid]
        # Filter to characters that are in the test mask and have a valid label
        story_record = {
            "story_id": sid,
            "characters": [],
        }
        for c in chars:
            gi = c["global_idx"]
            if not test_mask[gi]:
                continue
            if binary_labels[gi].item() < 0:
                continue  # UNK, skip
            story_record["characters"].append({
                "name": c["name"],
                "global_idx": gi,
                "true_label": c["label"],      # raw string
                "is_villain": binary_labels[gi].item() == 1,
                "rgcn_pred": rgcn_preds[gi].item(),       # 1 = villain
                "logreg_pred": logreg_preds[gi].item(),
            })
        if story_record["characters"]:
            records.append(story_record)

    return records


# ---------------------------------------------------------------------------
# 5. Analyses
# ---------------------------------------------------------------------------

def analyze_multi_villain(records, out):
    """Analysis 1: how well does each model handle multi-villain stories?"""
    out.write("=" * 78 + "\n")
    out.write("ANALYSIS 1: MULTI-VILLAIN STORIES\n")
    out.write("=" * 78 + "\n\n")

    buckets = defaultdict(list)
    for r in records:
        n_villains = sum(c["is_villain"] for c in r["characters"])
        buckets[n_villains].append(r)

    out.write(f"Distribution of true villain count per test story:\n")
    for k in sorted(buckets.keys()):
        out.write(f"  {k} villains: {len(buckets[k]):>3} stories\n")
    out.write("\n")

    for n_v in sorted(k for k in buckets if k >= 1):
        stories = buckets[n_v]
        if not stories:
            continue
        # For each story, how many villains did each model catch?
        rgcn_caught = []
        logreg_caught = []
        for r in stories:
            v_chars = [c for c in r["characters"] if c["is_villain"]]
            rgcn_caught.append(sum(c["rgcn_pred"] for c in v_chars))
            logreg_caught.append(sum(c["logreg_pred"] for c in v_chars))

        def pct(vals, total):
            return sum(vals) / max(total, 1)

        total_villains = n_v * len(stories)
        out.write(f"Stories with {n_v} villain(s) — {len(stories)} stories, "
                  f"{total_villains} villains total:\n")
        out.write(f"  R-GCN villain recall:   {pct(rgcn_caught, total_villains):.3f} "
                  f"({sum(rgcn_caught)}/{total_villains})\n")
        out.write(f"  LogReg villain recall:  {pct(logreg_caught, total_villains):.3f} "
                  f"({sum(logreg_caught)}/{total_villains})\n")

        # For multi-villain stories, how often does the model catch ALL villains?
        if n_v >= 2:
            rgcn_all = sum(1 for c in rgcn_caught if c == n_v)
            logreg_all = sum(1 for c in logreg_caught if c == n_v)
            out.write(f"  R-GCN caught ALL villains in story: {rgcn_all}/{len(stories)} "
                      f"({rgcn_all/len(stories):.1%})\n")
            out.write(f"  LogReg caught ALL villains in story: {logreg_all}/{len(stories)} "
                      f"({logreg_all/len(stories):.1%})\n")
        out.write("\n")

    return buckets


def analyze_zero_villain(records, out):
    """Analysis 2: stories with no villain — are we over-predicting?"""
    out.write("=" * 78 + "\n")
    out.write("ANALYSIS 2: ZERO-VILLAIN STORIES\n")
    out.write("=" * 78 + "\n\n")

    zero_stories = [r for r in records
                    if not any(c["is_villain"] for c in r["characters"])]

    if not zero_stories:
        out.write("No test stories had zero villains.\n\n")
        return zero_stories

    out.write(f"Found {len(zero_stories)} test story(ies) with no labeled villain:\n\n")
    for r in zero_stories:
        rgcn_flags = sum(c["rgcn_pred"] for c in r["characters"])
        logreg_flags = sum(c["logreg_pred"] for c in r["characters"])
        total = len(r["characters"])
        out.write(f"  {r['story_id']}: {total} characters\n")
        out.write(f"    R-GCN falsely flagged {rgcn_flags}/{total} as villain\n")
        out.write(f"    LogReg falsely flagged {logreg_flags}/{total} as villain\n")
        if rgcn_flags > 0:
            names = [c["name"] for c in r["characters"] if c["rgcn_pred"]]
            out.write(f"    R-GCN flagged: {names}\n")
        out.write("\n")

    return zero_stories


def analyze_disagreements(records, out):
    """Analysis 3: where R-GCN and LogReg disagree."""
    out.write("=" * 78 + "\n")
    out.write("ANALYSIS 3: R-GCN vs LogReg DISAGREEMENTS\n")
    out.write("=" * 78 + "\n\n")

    # Collapse to character level
    rgcn_right_lr_wrong = []   # R-GCN correct, LogReg wrong
    lr_right_rgcn_wrong = []   # LogReg correct, R-GCN wrong

    for r in records:
        for c in r["characters"]:
            truth = 1 if c["is_villain"] else 0
            if c["rgcn_pred"] == truth and c["logreg_pred"] != truth:
                rgcn_right_lr_wrong.append((r["story_id"], c))
            elif c["logreg_pred"] == truth and c["rgcn_pred"] != truth:
                lr_right_rgcn_wrong.append((r["story_id"], c))

    total_chars = sum(len(r["characters"]) for r in records)
    both_agree = total_chars - len(rgcn_right_lr_wrong) - len(lr_right_rgcn_wrong)

    out.write(f"Total test characters: {total_chars}\n")
    out.write(f"Both models agree: {both_agree} ({both_agree/total_chars:.1%})\n")
    out.write(f"R-GCN right, LogReg wrong: {len(rgcn_right_lr_wrong)} "
              f"({len(rgcn_right_lr_wrong)/total_chars:.1%})\n")
    out.write(f"LogReg right, R-GCN wrong: {len(lr_right_rgcn_wrong)} "
              f"({len(lr_right_rgcn_wrong)/total_chars:.1%})\n\n")

    # Break down by what kind of error
    def decompose(pairs, label):
        villain_caught_only_by = sum(1 for _, c in pairs if c["is_villain"])
        innocent_cleared_only_by = sum(1 for _, c in pairs if not c["is_villain"])
        out.write(f"{label}:\n")
        out.write(f"  Villains caught only by this model: {villain_caught_only_by}\n")
        out.write(f"  Innocents cleared only by this model: {innocent_cleared_only_by}\n")
        out.write("\n")

    decompose(rgcn_right_lr_wrong, "R-GCN wins")
    decompose(lr_right_rgcn_wrong, "LogReg wins")

    # Show a few examples of each
    def sample_cases(pairs, n=5):
        shown = pairs[:n]
        for sid, c in shown:
            verdict = "VILLAIN" if c["is_villain"] else "innocent"
            feats = c["features"] if isinstance(c, dict) and "features" in c else {}
            out.write(f"    {sid} — {c['name']} (truly {verdict})\n")

    out.write("Sample cases where R-GCN wins (first 5):\n")
    sample_cases(rgcn_right_lr_wrong, 5)
    out.write("\nSample cases where LogReg wins (first 5):\n")
    sample_cases(lr_right_rgcn_wrong, 5)
    out.write("\n")

    return rgcn_right_lr_wrong, lr_right_rgcn_wrong


def analyze_false_positives(records, out):
    """Analysis 4: innocent characters falsely accused."""
    out.write("=" * 78 + "\n")
    out.write("ANALYSIS 4: FALSE POSITIVES (INNOCENTS FALSELY ACCUSED)\n")
    out.write("=" * 78 + "\n\n")

    for model_key, model_name in [("rgcn_pred", "R-GCN"), ("logreg_pred", "LogReg")]:
        false_positives = []
        for r in records:
            for c in r["characters"]:
                if not c["is_villain"] and c[model_key] == 1:
                    false_positives.append((r["story_id"], c))

        out.write(f"{model_name}: {len(false_positives)} false positives\n")

        # What classes are these innocents actually?
        by_label = Counter(c["true_label"] for _, c in false_positives)
        out.write(f"  True labels of falsely accused:\n")
        for label, cnt in by_label.most_common():
            out.write(f"    {label:<15} {cnt:>4}\n")
        out.write("\n")

    return None


def analyze_story_level(records, out):
    """Analysis 5: did we solve the case? Per-story success."""
    out.write("=" * 78 + "\n")
    out.write("ANALYSIS 5: STORY-LEVEL ERROR ANALYSIS (\"DID WE SOLVE THE CASE?\")\n")
    out.write("=" * 78 + "\n\n")

    out.write("For each test story, we ask: did the model correctly identify\n")
    out.write("AT LEAST ONE true villain? (Analogous to a detective finding the\n")
    out.write("murderer, even if they miss accomplices.)\n\n")

    rgcn_solved = 0
    logreg_solved = 0
    rgcn_solved_cleanly = 0    # caught a villain AND made no false accusations
    logreg_solved_cleanly = 0

    both_solved = 0
    rgcn_only = []
    logreg_only = []
    neither = []

    stories_with_villain = [r for r in records
                            if any(c["is_villain"] for c in r["characters"])]
    n_stories = len(stories_with_villain)

    for r in stories_with_villain:
        v_chars = [c for c in r["characters"] if c["is_villain"]]
        innocents = [c for c in r["characters"] if not c["is_villain"]]

        rgcn_hit = any(c["rgcn_pred"] for c in v_chars)
        logreg_hit = any(c["logreg_pred"] for c in v_chars)
        rgcn_fp = any(c["rgcn_pred"] for c in innocents)
        logreg_fp = any(c["logreg_pred"] for c in innocents)

        if rgcn_hit: rgcn_solved += 1
        if logreg_hit: logreg_solved += 1
        if rgcn_hit and not rgcn_fp: rgcn_solved_cleanly += 1
        if logreg_hit and not logreg_fp: logreg_solved_cleanly += 1

        if rgcn_hit and logreg_hit:
            both_solved += 1
        elif rgcn_hit and not logreg_hit:
            rgcn_only.append(r["story_id"])
        elif logreg_hit and not rgcn_hit:
            logreg_only.append(r["story_id"])
        else:
            neither.append(r["story_id"])

    out.write(f"Test stories with at least 1 villain: {n_stories}\n\n")
    out.write(f"                              {'R-GCN':>12} {'LogReg':>12}\n")
    out.write(f"{'Caught at least one villain:':<32} "
              f"{rgcn_solved:>5} ({rgcn_solved/n_stories:.1%})  "
              f"{logreg_solved:>5} ({logreg_solved/n_stories:.1%})\n")
    out.write(f"{'  ...with NO false accusations:':<32} "
              f"{rgcn_solved_cleanly:>5} ({rgcn_solved_cleanly/n_stories:.1%})  "
              f"{logreg_solved_cleanly:>5} ({logreg_solved_cleanly/n_stories:.1%})\n\n")

    out.write(f"Agreement on solving:\n")
    out.write(f"  Both caught a villain:   {both_solved:>3} ({both_solved/n_stories:.1%})\n")
    out.write(f"  R-GCN only:              {len(rgcn_only):>3} ({len(rgcn_only)/n_stories:.1%})\n")
    out.write(f"  LogReg only:             {len(logreg_only):>3} ({len(logreg_only)/n_stories:.1%})\n")
    out.write(f"  Neither (case unsolved): {len(neither):>3} ({len(neither)/n_stories:.1%})\n\n")

    # Show the "neither" stories — these are the hardest cases
    out.write(f"Unsolved stories (neither model caught any villain):\n")
    for sid in neither:
        out.write(f"  {sid}\n")
    out.write("\n")

    return {
        "rgcn_solved": rgcn_solved,
        "logreg_solved": logreg_solved,
        "n_stories": n_stories,
        "both_solved": both_solved,
        "rgcn_only": rgcn_only,
        "logreg_only": logreg_only,
        "neither": neither,
    }


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graph_dir",
                        default=os.path.join(_PROJECT_ROOT, "data", "graphs"))
    parser.add_argument("--out",
                        default=os.path.join(_PROJECT_ROOT, "analysis", "results",
                                              "failure_analysis_results.txt"))
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    # Open output file
    outfile = open(args.out, "w")

    def dual_write(s):
        print(s, end="")
        outfile.write(s)

    class DualWriter:
        def write(self, s):
            dual_write(s)

    out = DualWriter()

    out.write(f"Detective R-GCN Failure Analysis\n")
    out.write(f"=================================\n")
    out.write(f"Seed: {args.seed}\n")
    out.write(f"Excluded features: {sorted(EXCLUDE_FEATURES)}\n")
    out.write(f"Excluded entries: {sorted(EXCLUDE_ENTRIES)}\n")
    out.write(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Load graph
    out.write("Loading graph...\n")
    with redirect_stdout(io.StringIO()):
        graph = load_mystery_graphs(args.graph_dir, seed=args.seed,
                                     exclude_features=EXCLUDE_FEATURES,
                                     exclude_entries=EXCLUDE_ENTRIES)

    # Binary labels
    binary_labels = (graph.class_labels == 0).long()
    binary_labels[graph.class_labels < 0] = -1

    out.write(f"Stories — train/val/test split using seed {args.seed}\n")
    out.write(f"Test characters: {graph.test_mask.sum().item()}\n")
    out.write(f"Test villains: {(binary_labels[graph.test_mask] == 1).sum().item()}\n\n")

    # Train models
    out.write("Training R-GCN...\n")
    t = time.time()
    rgcn_preds, rgcn_probs = train_rgcn(graph, binary_labels, epochs=args.epochs, seed=args.seed)
    out.write(f"  Trained in {time.time()-t:.1f}s\n")

    out.write("Training LogReg...\n")
    logreg_preds, logreg_probs, _ = train_logreg(graph, binary_labels, seed=args.seed)
    out.write("  Done.\n\n")

    # Reconstruct story membership
    out.write("Reconstructing story membership from JSON files...\n")
    node_story, story_of_test, all_stories = build_story_membership(
        args.graph_dir, args.seed, exclude_entries=EXCLUDE_ENTRIES)
    out.write(f"  {len(all_stories)} total stories, {len(node_story)} nodes mapped.\n\n")

    # Build per-story records
    records = build_test_story_records(
        graph, binary_labels, rgcn_preds, logreg_preds,
        story_of_test, node_story)
    out.write(f"Test story records built: {len(records)} stories "
              f"with valid test characters.\n\n")

    # Run analyses
    analyze_multi_villain(records, out)
    analyze_zero_villain(records, out)
    analyze_disagreements(records, out)
    analyze_false_positives(records, out)
    analyze_story_level(records, out)

    outfile.close()
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
