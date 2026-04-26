"""
Re-extract graphs for the 11 unsolved cases using new, more detailed synopses.

Reads synopses from extraction/data/new_extracts/{ID}.txt, extracts graphs via
Ollama (mixtral:8x7b), and writes them to extraction/data/graphs/{ID}.json,
overwriting the previous version.

Pulls metadata (title, author, year, medium) from the existing JSON files.
"""

import json
import logging
import os
import sys
import time
import traceback

from extraction.extractor import (
    get_client,
    extract_graph,
    normalize_edges,
    validate_graph,
    save_graph,
)

NEW_EXTRACTS_DIR = "extraction/data/new_extracts"
GRAPHS_DIR = "extraction/data/graphs"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "reextraction.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Entry IDs to skip (already extracted)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only extract these entry IDs")
    args = parser.parse_args()

    txt_files = sorted([f for f in os.listdir(NEW_EXTRACTS_DIR) if f.endswith(".txt")])
    if not txt_files:
        logger.error(f"No .txt files in {NEW_EXTRACTS_DIR}")
        return

    if args.only:
        txt_files = [f for f in txt_files if f.replace(".txt", "") in args.only]
    if args.skip:
        txt_files = [f for f in txt_files if f.replace(".txt", "") not in args.skip]

    logger.info(f"Re-extracting {len(txt_files)} entries: "
                f"{[f.replace('.txt','') for f in txt_files]}")

    client = get_client()
    results = {"SUCCESS": 0, "FAILED": 0, "VALIDATION_WARNINGS": 0}
    failures = []

    for idx, fname in enumerate(txt_files, 1):
        eid = fname.replace(".txt", "")
        synopsis_path = os.path.join(NEW_EXTRACTS_DIR, fname)
        old_graph_path = os.path.join(GRAPHS_DIR, f"{eid}.json")

        # Pull metadata from existing JSON
        if not os.path.exists(old_graph_path):
            logger.error(f"[{idx}/{len(txt_files)}] {eid}: no existing JSON for metadata, skipping")
            failures.append(eid)
            continue

        with open(old_graph_path) as f:
            old_data = json.load(f)
        meta = old_data.get("metadata", {})
        title = meta.get("title", "?")
        author = meta.get("author", "?")
        year = meta.get("year", "?")
        medium = meta.get("medium", "?")

        with open(synopsis_path) as f:
            synopsis = f.read().strip()

        logger.info(f"[{idx}/{len(txt_files)}] {eid}: {title} ({len(synopsis.split())} words)")

        t0 = time.time()
        try:
            graph = extract_graph(
                client, eid,
                title=title, author=author, year=year, medium=medium,
                synopsis=synopsis,
            )

            graph = normalize_edges(graph)
            validation_errors = validate_graph(graph)
            if validation_errors:
                logger.warning(f"{eid} validation warnings: {validation_errors}")
                results["VALIDATION_WARNINGS"] += 1

            graph["metadata"] = {
                "entry_id": eid,
                "title": title,
                "author": author,
                "year": year,
                "medium": medium,
            }

            save_graph(graph, eid, GRAPHS_DIR)
            results["SUCCESS"] += 1
            logger.info(
                f"{eid} saved — {len(graph['characters'])} chars, "
                f"{len(graph['edges'])} edges, "
                f"{len(graph.get('locations', []))} locs, "
                f"{time.time()-t0:.1f}s"
            )

        except Exception as e:
            logger.error(f"{eid} FAILED: {e}")
            logger.error(traceback.format_exc())
            failures.append(eid)
            results["FAILED"] += 1

    logger.info(f"\nFinal summary:")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")
    if failures:
        logger.info(f"  Failed entries: {failures}")


if __name__ == "__main__":
    main()
