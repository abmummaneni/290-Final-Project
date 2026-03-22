"""Entry point for the murder mystery synopsis scraper pipeline."""

import argparse
import json
import logging
import os
import sys

from tqdm import tqdm

# Add project root to path so we can import scraper modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.loader import load_candidates
from scraper.wikipedia import scrape_synopsis
from scraper.cleaner import clean_synopsis
from scraper.validator import score_synopsis

# Paths relative to the Detective project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX_PATH = os.path.join(PROJECT_ROOT, "murder_mystery_candidates_v2.xlsx")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_PATH = os.path.join(LOG_DIR, "scrape.log")

QUALITY_THRESHOLD = 0.4
PARTIAL_WORD_THRESHOLD = 150


def setup_logging():
    """Configure logging to file and console."""
    os.makedirs(LOG_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(),
        ],
    )


def load_manifest() -> dict:
    """Load existing manifest or return empty dict."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict):
    """Save manifest to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def process_entry(entry: dict, manifest: dict) -> dict:
    """Scrape, clean, validate, and store a single entry.

    Returns the updated manifest entry.
    """
    entry_id = entry["entry_id"]
    title = entry["title"]
    author = entry["author"]
    medium = entry["medium"]
    year = int(entry["year"]) if entry["year"] else 0

    logger = logging.getLogger(__name__)
    logger.info("Processing %s: %s", entry_id, title)

    # Scrape
    try:
        result = scrape_synopsis(title, author, medium, year)
    except Exception:
        logger.error("Fatal scrape error for %s", entry_id, exc_info=True)
        result = {"text": "", "wikipedia_url": "", "section_found": "", "status": "FAILED"}

    raw_text = result["text"]
    cleaned_text = clean_synopsis(raw_text) if raw_text else ""

    # Determine status
    word_count_raw = len(raw_text.split()) if raw_text else 0
    word_count_cleaned = len(cleaned_text.split()) if cleaned_text else 0

    if result["status"] == "SUCCESS" and word_count_cleaned < PARTIAL_WORD_THRESHOLD:
        result["status"] = "PARTIAL"

    # Score quality
    quality_score = score_synopsis(cleaned_text) if cleaned_text else 0.0

    # Save raw text
    os.makedirs(RAW_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_DIR, f"{entry_id}.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    # Save cleaned text
    os.makedirs(CLEANED_DIR, exist_ok=True)
    cleaned_path = os.path.join(CLEANED_DIR, f"{entry_id}.txt")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    # Build manifest entry
    manifest_entry = {
        "entry_id": entry_id,
        "title": title,
        "author": author,
        "year": year,
        "medium": medium,
        "subgenre": entry.get("subgenre", ""),
        "villain_reveal": entry.get("villain_reveal", ""),
        "synopsis_quality_flag": entry.get("synopsis_quality_flag", ""),
        "scrape_status": result["status"],
        "wikipedia_url": result["wikipedia_url"],
        "section_found": result["section_found"],
        "word_count_raw": word_count_raw,
        "word_count_cleaned": word_count_cleaned,
        "quality_score": quality_score,
        "needs_review": quality_score < QUALITY_THRESHOLD,
        "notes": "",
    }

    return manifest_entry


def main():
    parser = argparse.ArgumentParser(description="Murder Mystery Synopsis Scraper")
    parser.add_argument("--all", action="store_true", help="Scrape all entries")
    parser.add_argument("--id", type=str, help="Scrape a single entry by ID (e.g. NOV_001)")
    parser.add_argument("--status", type=str, help="Re-scrape entries with given status")
    parser.add_argument("--dry-run", action="store_true", help="Load xlsx and print entries without scraping")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Load candidates
    logger.info("Loading candidate list from %s", XLSX_PATH)
    df = load_candidates(XLSX_PATH)
    logger.info("Loaded %d entries", len(df))

    if args.dry_run:
        print(f"\n{'ID':<10} {'Medium':<15} {'Year':<6} {'Title'}")
        print("-" * 80)
        for _, row in df.iterrows():
            print(f"{row['entry_id']:<10} {row['medium']:<15} {row['year']:<6} {row['title']}")
        print(f"\nTotal: {len(df)} entries")
        return

    # Load existing manifest
    manifest = load_manifest()

    # Determine which entries to process
    if args.id:
        entries = df[df["entry_id"] == args.id]
        if entries.empty:
            logger.error("Entry ID not found: %s", args.id)
            sys.exit(1)
    elif args.status:
        # Re-scrape entries with a specific status
        ids_to_rescrape = [
            eid for eid, info in manifest.items()
            if info.get("scrape_status") == args.status
        ]
        entries = df[df["entry_id"].isin(ids_to_rescrape)]
        logger.info("Found %d entries with status '%s'", len(entries), args.status)
    elif args.all:
        entries = df
    else:
        parser.print_help()
        sys.exit(1)

    # Process entries
    success = partial = needs_manual = failed = skipped = 0

    for _, row in tqdm(entries.iterrows(), total=len(entries), desc="Scraping"):
        entry_id = row["entry_id"]

        # Skip already-completed entries (unless re-scraping by status)
        if not args.status and entry_id in manifest and manifest[entry_id].get("scrape_status") == "SUCCESS":
            skipped += 1
            continue

        entry = row.to_dict()
        manifest_entry = process_entry(entry, manifest)
        manifest[entry_id] = manifest_entry

        # Save manifest after each entry (incremental save)
        save_manifest(manifest)

        status = manifest_entry["scrape_status"]
        if status == "SUCCESS":
            success += 1
        elif status == "PARTIAL":
            partial += 1
        elif status == "NEEDS_MANUAL":
            needs_manual += 1
        elif status == "FAILED":
            failed += 1

    # Summary
    logger.info(
        "Done. SUCCESS=%d, PARTIAL=%d, NEEDS_MANUAL=%d, FAILED=%d, SKIPPED=%d",
        success, partial, needs_manual, failed, skipped,
    )
    print(f"\nSummary:")
    print(f"  SUCCESS:      {success}")
    print(f"  PARTIAL:      {partial}")
    print(f"  NEEDS_MANUAL: {needs_manual}")
    print(f"  FAILED:       {failed}")
    print(f"  SKIPPED:      {skipped}")


if __name__ == "__main__":
    main()
