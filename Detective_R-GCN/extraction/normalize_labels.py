"""
Normalize character labels across all extracted graphs.

Valid labels: Villain, Victim, Witness, Uninvolved, UNK

Rules:
1. Compound labels (e.g. "Villain|Victim") → resolved by priority:
   Villain > Victim > Witness > Uninvolved > UNK
2. Non-standard labels mapped to valid classes (see LABEL_MAP)
3. Labels are case-normalized

Run:
    python -m extraction.normalize_labels [--dry-run]
"""

import json
import os
import argparse
from collections import Counter

GRAPH_DIR = os.path.join(os.path.dirname(__file__), "data", "graphs")

VALID_LABELS = {"Villain", "Victim", "Witness", "Uninvolved", "UNK"}

# Priority order for resolving compound labels
PRIORITY = ["Villain", "Victim", "Witness", "Uninvolved", "UNK"]

# Direct mappings for non-standard single labels
LABEL_MAP = {
    "suspect": "Uninvolved",
    "suspects": "Uninvolved",
    "investigator": "Uninvolved",
    "detective": "Uninvolved",
    "private eye": "Uninvolved",
    "enforcer": "Uninvolved",
    "client": "Uninvolved",
    "superhero": "Uninvolved",
    "family": "Uninvolved",
    "host/producer": "Uninvolved",
    "holmes in disguise": "Uninvolved",
    "protagonist": "Uninvolved",
    "witnesses": "Witness",
    "witness?": "Witness",
    "victim?": "Victim",
    "victims": "Victim",
    "lynnwood police": "Uninvolved",
    "golden police": "Uninvolved",
    "westminster police": "Uninvolved",
    "authority": "Uninvolved",
}


def normalize_label(raw_label: str) -> str:
    """Normalize a single label string to a valid label."""
    raw = raw_label.strip()

    # Already valid
    if raw in VALID_LABELS:
        return raw

    # Case-insensitive direct match
    if raw.lower() in {v.lower() for v in VALID_LABELS}:
        for v in VALID_LABELS:
            if raw.lower() == v.lower():
                return v

    # Direct map for known non-standard labels
    if raw.lower() in LABEL_MAP:
        return LABEL_MAP[raw.lower()]

    # Compound label: split on | and resolve by priority
    parts = [p.strip() for p in raw.split("|")]

    # Clean each part: strip -1 artifacts, apply direct map
    cleaned = []
    for p in parts:
        if p in ("-1", ""):
            continue
        # Check direct map
        if p.lower() in LABEL_MAP:
            cleaned.append(LABEL_MAP[p.lower()])
        elif p in VALID_LABELS:
            cleaned.append(p)
        else:
            # Case-insensitive match against valid labels
            matched = False
            for v in VALID_LABELS:
                if p.lower() == v.lower():
                    cleaned.append(v)
                    matched = True
                    break
            if not matched:
                # Check if it's a known non-standard label
                if p.lower() in LABEL_MAP:
                    cleaned.append(LABEL_MAP[p.lower()])
                else:
                    cleaned.append("UNK")

    if not cleaned:
        return "UNK"

    # Resolve by priority
    for priority_label in PRIORITY:
        if priority_label in cleaned:
            return priority_label

    return "UNK"


def main():
    parser = argparse.ArgumentParser(description="Normalize character labels")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(GRAPH_DIR) if f.endswith(".json"))
    changes = Counter()
    total_chars = 0
    changed_files = 0

    for fname in files:
        path = os.path.join(GRAPH_DIR, fname)
        with open(path) as f:
            g = json.load(f)

        file_changed = False
        for c in g.get("characters", []):
            total_chars += 1
            old = c.get("label", "UNK")
            new = normalize_label(old)
            if old != new:
                changes[f"{old} -> {new}"] += 1
                c["label"] = new
                file_changed = True

        if file_changed:
            changed_files += 1
            if not args.dry_run:
                with open(path, "w") as f:
                    json.dump(g, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"Scanned {len(files)} graphs, {total_chars} characters")
    print(f"Changed {sum(changes.values())} labels across {changed_files} files")
    if args.dry_run:
        print("(DRY RUN — no files modified)")
    print()

    if changes:
        print("Changes:")
        for change, count in changes.most_common():
            print(f"  {count:>4}x  {change}")

    # Post-normalization label distribution
    label_dist = Counter()
    for fname in files:
        path = os.path.join(GRAPH_DIR, fname)
        with open(path) as f:
            g = json.load(f)
        for c in g.get("characters", []):
            label_dist[c.get("label", "UNK")] += 1

    print(f"\nLabel distribution after normalization:")
    for label, count in label_dist.most_common():
        pct = 100 * count / sum(label_dist.values())
        print(f"  {label:<15s} {count:>5}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
