"""
Phase 2 extractor: sends synopses to a local Ollama model
for structured graph extraction using a two-pass approach.

Pass 1: Extract nodes (characters, occupations, locations, organizations)
Pass 2: Extract edges/relationships between the nodes

Configure via .env:
  OLLAMA_MODEL  — model name (default: mixtral:8x7b)
  OLLAMA_URL    — server URL (default: http://localhost:11434)
"""

import json
import logging
import os
import re
import time

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from .prompt import SYSTEM_PROMPT, PASS1_PROMPT, PASS2_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mixtral:8x7b")


def get_client():
    """No client object needed for Ollama — returns None."""
    return None


def _clean_json_text(text):
    """Extract and repair JSON from model response."""
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            if part.strip().startswith("{") or part.strip().startswith("json"):
                text = part.strip()
                if text.lower().startswith("json"):
                    text = text[4:].strip()
                break

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text


def _repair_truncated_json(text):
    """Attempt to close truncated JSON by balancing brackets."""
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")

    if open_braces > 0 or open_brackets > 0:
        text = re.sub(r',\s*"[^"]*$', '', text)
        text = re.sub(r',\s*\{[^}]*$', '', text)
        text = re.sub(r',\s*$', '', text)
        text += "]" * open_brackets + "}" * open_braces

    return text


def _parse_json_response(text, entry_id, pass_name):
    """Parse JSON from Ollama response with repair fallback."""
    cleaned = _clean_json_text(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = _repair_truncated_json(cleaned)
        try:
            result = json.loads(repaired)
            logger.info(f"{entry_id}: {pass_name} JSON repaired successfully")
            return result
        except json.JSONDecodeError:
            debug_dir = "extraction/data/debug"
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"{entry_id}_{pass_name}_raw.txt")
            with open(debug_path, "w") as f:
                f.write(f"model: {OLLAMA_MODEL}\n\n{text}")
            logger.error(f"{entry_id}: {pass_name} JSON parse failed, raw saved to {debug_path}")
            raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def _call_ollama(prompt):
    """Make a single Ollama API call."""
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "num_predict": 8192,
                "temperature": 0.1,
            },
            "format": "json",
        },
        timeout=600,
    )
    response.raise_for_status()
    result = response.json()
    return result["message"]["content"].strip()


def _build_nodes_summary(nodes):
    """Build a readable summary of extracted nodes for pass 2."""
    lines = []

    lines.append("Characters:")
    for c in nodes.get("characters", []):
        label = c.get("label", "UNK")
        lines.append(f"  {c['id']}: {c['name']} (label: {label})")

    lines.append("\nOccupations:")
    for o in nodes.get("occupations", []):
        lines.append(f"  {o['id']}: {o['name']}")

    lines.append("\nLocations:")
    for loc in nodes.get("locations", []):
        lines.append(f"  {loc['id']}: {loc['name']}")

    lines.append("\nOrganizations:")
    for org in nodes.get("organizations", []):
        lines.append(f"  {org['id']}: {org['name']}")

    return "\n".join(lines)


def extract_graph(client, entry_id, title, author, year, medium, synopsis):
    """Two-pass extraction: nodes first, then edges."""

    # ── Pass 1: Extract nodes ──
    logger.info(f"Extracting {entry_id}: {title} — Pass 1: nodes (model: {OLLAMA_MODEL})")

    pass1_prompt = PASS1_PROMPT.format(
        title=title, author=author, year=year, medium=medium, synopsis=synopsis,
    )
    pass1_text = _call_ollama(pass1_prompt)
    nodes = _parse_json_response(pass1_text, entry_id, "pass1")

    char_count = len(nodes.get("characters", []))
    logger.info(f"{entry_id}: Pass 1 done — {char_count} characters, "
                f"{len(nodes.get('locations', []))} locations, "
                f"{len(nodes.get('occupations', []))} occupations")

    # ── Pass 2: Extract edges ──
    logger.info(f"{entry_id}: Pass 2: edges")

    nodes_summary = _build_nodes_summary(nodes)
    pass2_prompt = PASS2_PROMPT.format(
        title=title, synopsis=synopsis, nodes_summary=nodes_summary,
    )
    pass2_text = _call_ollama(pass2_prompt)
    edges_result = _parse_json_response(pass2_text, entry_id, "pass2")

    # ── Combine ──
    graph = {
        "characters": nodes.get("characters", []),
        "occupations": nodes.get("occupations", []),
        "locations": nodes.get("locations", []),
        "organizations": nodes.get("organizations", []),
        "edges": edges_result.get("edges", []),
    }

    return graph


# Normalize non-schema relation types to the closest valid type
RELATION_NORMALIZATIONS = {
    # Family
    "brother of": "related to",
    "sister of": "related to",
    "daughter of": "related to",
    "son of": "related to",
    "father of": "related to",
    "mother of": "related to",
    "parent of": "related to",
    "child of": "related to",
    "uncle of": "related to",
    "aunt of": "related to",
    "cousin of": "related to",
    "nephew of": "related to",
    "niece of": "related to",
    "grandfather of": "related to",
    "grandmother of": "related to",
    "grandchild of": "related to",
    "ancestor of": "related to",
    "descendant of": "related to",
    "sibling of": "related to",
    "sister-in-law of": "related to",
    "brother-in-law of": "related to",
    "stepfather of": "related to",
    "stepmother of": "related to",
    # Social
    "meets with": "works with",
    "knows": "friends with",
    "acquaintance of": "friends with",
    "neighbor of": "lives with",
    "abuses": "in conflict with",
    "threatens": "in conflict with",
    "argues with": "in conflict with",
    "fights with": "in conflict with",
    "dislikes": "in conflict with",
    "enemies with": "in conflict with",
    "jealous of": "in conflict with",
    "distrusts": "suspects",
    "questions": "investigates",
    "interrogates": "investigates",
    "follows": "investigates",
    "spies on": "investigates",
    "watches": "investigates",
    "tracks": "investigates",
    "lies to": "deceives",
    "manipulates": "deceives",
    "tricks": "deceives",
    "betrays": "deceives",
    "frames": "deceives",
    "helps": "protects",
    "saves": "protects",
    "defends": "protects",
    "guards": "protects",
    "assists": "works with",
    "collaborates with": "works with",
    "partners with": "business partner of",
    "loves": "romantically involved with",
    "engaged to": "romantically involved with",
    "dating": "romantically involved with",
    "affair with": "romantically involved with",
    "lover of": "romantically involved with",
    "ex-wife of": "formerly married to",
    "ex-husband of": "formerly married to",
    "murders": "kills",
    "poisons": "kills",
    "shoots": "kills",
    "stabs": "kills",
    # Occupation/org
    "works for": "employed by",
    "member of": "affiliated with",
    "belongs to": "affiliated with",
    "founded": "leads",
    "runs": "leads",
    "manages": "leads",
    "lives at": "resides at",
    "visits": "present at",
    "goes to": "present at",
    "found at": "present at",
    "works at": "present at",
}


def normalize_edges(graph):
    """Normalize non-schema relation types and remove invalid edges."""
    valid_ids = set()
    for node_type in ["characters", "occupations", "locations", "organizations"]:
        for node in graph.get(node_type, []):
            valid_ids.add(node["id"])

    cleaned_edges = []
    for edge in graph.get("edges", []):
        # Skip edges missing required fields
        if "source" not in edge or "target" not in edge or "relation" not in edge:
            continue

        # Skip edges with invalid node references
        if edge["source"] not in valid_ids or edge["target"] not in valid_ids:
            continue

        # Normalize relation type
        relation = edge["relation"].lower().strip()
        if relation in RELATION_NORMALIZATIONS:
            edge["relation"] = RELATION_NORMALIZATIONS[relation]
        else:
            edge["relation"] = relation

        cleaned_edges.append(edge)

    graph["edges"] = cleaned_edges
    return graph


def validate_graph(graph):
    """Basic structural validation of extracted graph."""
    errors = []

    if "characters" not in graph or not graph["characters"]:
        errors.append("No characters extracted")

    if "edges" not in graph or not graph["edges"]:
        errors.append("No edges extracted")

    valid_ids = set()
    for node_type in ["characters", "occupations", "locations", "organizations"]:
        for node in graph.get(node_type, []):
            valid_ids.add(node["id"])

    for edge in graph.get("edges", []):
        if edge["source"] not in valid_ids:
            errors.append(f"Edge source '{edge['source']}' not found in nodes")
        if edge["target"] not in valid_ids:
            errors.append(f"Edge target '{edge['target']}' not found in nodes")

    labels = [c.get("label", "UNK") for c in graph.get("characters", [])]
    if all(l == "UNK" for l in labels):
        errors.append("No characters have labels assigned")

    return errors


def save_graph(graph, entry_id, output_dir="extraction/data/graphs"):
    """Save extracted graph JSON to file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{entry_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    return path
