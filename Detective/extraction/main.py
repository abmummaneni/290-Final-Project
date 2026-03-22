"""
Phase 2 main: extract structured graph data from cleaned synopses.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from .extractor import get_client, extract_graph, normalize_edges, validate_graph, save_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/extraction.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

MANIFEST_PATH = "data/manifest.json"
CLEANED_DIR = "data/cleaned"
GRAPHS_DIR = "extraction/data/graphs"
EXTRACTION_STATUS_PATH = "extraction/data/extraction_status.json"


def load_extraction_status():
    if os.path.exists(EXTRACTION_STATUS_PATH):
        with open(EXTRACTION_STATUS_PATH) as f:
            return json.load(f)
    return {}


def save_extraction_status(status):
    os.makedirs(os.path.dirname(EXTRACTION_STATUS_PATH), exist_ok=True)
    with open(EXTRACTION_STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


def get_eligible_entries(manifest, min_quality=0.4):
    """Return entries eligible for extraction (SUCCESS with quality >= threshold)."""
    eligible = {}
    for eid, info in manifest.items():
        if info["scrape_status"] == "SUCCESS" and info.get("quality_score", 0) >= min_quality:
            eligible[eid] = info
    return eligible


def run_extraction(entry_ids=None, limit=None, min_quality=0.4):
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    eligible = get_eligible_entries(manifest, min_quality)
    extraction_status = load_extraction_status()

    # Filter to requested IDs or all eligible
    if entry_ids:
        targets = {eid: eligible[eid] for eid in entry_ids if eid in eligible}
    else:
        # Skip already-extracted entries
        targets = {eid: info for eid, info in eligible.items()
                   if eid not in extraction_status or extraction_status[eid]["status"] != "SUCCESS"}

    if limit:
        targets = dict(list(targets.items())[:limit])

    if not targets:
        logger.info("No entries to extract.")
        return

    logger.info(f"Extracting {len(targets)} entries (of {len(eligible)} eligible)")

    client = get_client()
    results = {"SUCCESS": 0, "FAILED": 0, "VALIDATION_WARNINGS": 0}

    for eid, info in tqdm(targets.items(), desc="Extracting"):
        cleaned_path = os.path.join(CLEANED_DIR, f"{eid}.txt")
        if not os.path.exists(cleaned_path):
            logger.warning(f"No cleaned file for {eid}, skipping")
            continue

        with open(cleaned_path) as f:
            synopsis = f.read().strip()

        if not synopsis:
            logger.warning(f"Empty synopsis for {eid}, skipping")
            continue

        try:
            graph = extract_graph(
                client, eid,
                title=info["title"],
                author=info["author"],
                year=info["year"],
                medium=info["medium"],
                synopsis=synopsis,
            )

            graph = normalize_edges(graph)
            validation_errors = validate_graph(graph)
            if validation_errors:
                logger.warning(f"{eid} validation warnings: {validation_errors}")
                results["VALIDATION_WARNINGS"] += 1

            # Add metadata to graph
            graph["metadata"] = {
                "entry_id": eid,
                "title": info["title"],
                "author": info["author"],
                "year": info["year"],
                "medium": info["medium"],
            }

            path = save_graph(graph, eid, GRAPHS_DIR)
            extraction_status[eid] = {
                "status": "SUCCESS",
                "validation_errors": validation_errors,
                "character_count": len(graph.get("characters", [])),
                "edge_count": len(graph.get("edges", [])),
                "location_count": len(graph.get("locations", [])),
                "organization_count": len(graph.get("organizations", [])),
            }
            results["SUCCESS"] += 1
            logger.info(
                f"{eid}: {len(graph['characters'])} chars, "
                f"{len(graph['edges'])} edges, "
                f"{len(graph.get('locations', []))} locs"
            )

        except Exception as e:
            logger.error(f"{eid} extraction failed: {e}")
            extraction_status[eid] = {"status": "FAILED", "error": str(e)}
            results["FAILED"] += 1

        # Save status incrementally
        save_extraction_status(extraction_status)

        # Small delay to avoid overwhelming Ollama
        time.sleep(0.5)

    logger.info(
        f"Done. SUCCESS={results['SUCCESS']}, "
        f"FAILED={results['FAILED']}, "
        f"VALIDATION_WARNINGS={results['VALIDATION_WARNINGS']}"
    )
    print(f"\nSummary:")
    for k, v in results.items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract graph data from synopses")
    parser.add_argument("--id", nargs="+", help="Extract specific entry IDs")
    parser.add_argument("--limit", type=int, help="Limit number of entries to extract")
    parser.add_argument("--all", action="store_true", help="Extract all eligible entries")
    parser.add_argument("--min-quality", type=float, default=0.4, help="Minimum quality score")
    args = parser.parse_args()

    if args.id:
        run_extraction(entry_ids=args.id)
    elif args.all:
        run_extraction(min_quality=args.min_quality)
    elif args.limit:
        run_extraction(limit=args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
