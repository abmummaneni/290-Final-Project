"""
Extraction prompt templates for converting murder mystery synopses
into structured graph data conforming to the graph schema.

Two-pass approach:
  Pass 1: Extract all nodes (characters, occupations, locations, organizations)
  Pass 2: Given the nodes, extract all edges/relationships
"""

SYSTEM_PROMPT = """\
You are a structured data extraction system for murder mystery narratives.
You extract precise JSON from synopses. Return ONLY valid JSON, no commentary."""

# ── Pass 1: Node extraction ──

PASS1_PROMPT = """\
Extract all characters, occupations, locations, and organizations from this murder mystery synopsis.

**Title:** {title}
**Author:** {author}
**Year:** {year}
**Medium:** {medium}

**Synopsis:**
{synopsis}

---

Return a JSON object with ONLY the nodes (no edges yet). Use this exact structure:

{{
  "characters": [
    {{
      "id": "char_0",
      "name": "Character Name",
      "label": "Villain|Victim|Witness|Uninvolved|UNK",
      "features": {{
        "gender": 0,
        "social_status": 0.5,
        "narrative_introduction_timing": 0.0,
        "has_alibi": -1,
        "present_at_crime_scene": -1,
        "has_motive": -1,
        "is_concealing_information": -1,
        "has_hidden_relationship": -1,
        "motive_type": -1,
        "narrative_prominence": 0.5
      }}
    }}
  ],
  "occupations": [
    {{
      "id": "occ_0",
      "name": "Occupation Name",
      "features": {{
        "authority_level": 0.5,
        "access_level": 0.5,
        "capability_level": 0.5
      }}
    }}
  ],
  "locations": [
    {{
      "id": "loc_0",
      "name": "Location Name",
      "features": {{
        "accessibility": 0.5,
        "isolability": 0.5,
        "evidentiary_value": 0.5
      }}
    }}
  ],
  "organizations": [
    {{
      "id": "org_0",
      "name": "Organization Name",
      "features": {{
        "institutional_power": 0.5,
        "secrecy_level": 0.5,
        "financial_scale": 0.5
      }}
    }}
  ]
}}

**Feature encoding guide:**

Character features:
- gender: male=0, female=1, other=0.5, unknown=-1
- social_status: 0=lowest in story, 1=highest in story (relative)
- narrative_introduction_timing: 0=introduced first, 1=introduced last (relative)
- has_alibi: yes=1, no=0, unknown=-1
- present_at_crime_scene: yes=1, no=0, unknown=-1
- has_motive: yes=1, no=0, unknown=-1
- is_concealing_information: yes=1, no=0, unknown=-1
- has_hidden_relationship: yes=1, no=0, unknown=-1
- motive_type: jealousy=0.0, money=0.33, revenge=0.66, love=1.0, unknown=-1
- narrative_prominence: 0=least prominent, 1=most prominent (relative)

Character labels: Villain, Victim, Witness, Uninvolved, UNK

Extract ALL named characters, even minor ones. Use -1 for unknown features.
Return ONLY valid JSON."""

# ── Pass 2: Edge extraction ──

PASS2_PROMPT = """\
Now extract ALL relationships (edges) between the nodes listed below, based on the synopsis.

**Title:** {title}

**Synopsis:**
{synopsis}

**Extracted nodes:**
{nodes_summary}

---

For EVERY pair of characters who interact or have a relationship, create an edge.
Think carefully about each character pair — consider family ties, romantic relationships,
professional relationships, conflicts, suspicions, and any other connections.

Return a JSON object with ONLY the edges array:

{{
  "edges": [
    {{
      "source": "char_0",
      "target": "char_1",
      "relation": "relationship type",
      "directed": true
    }}
  ]
}}

**Edge relation types to use:**

Character → Character:
  married to, related to, friends with, employs, blackmails, suspects,
  witnessed by, in conflict with, business partner of, romantically involved with,
  mentor of, rivals with, works with, lives with, discovered by,
  kills, investigates, deceives, protects, accuses, hires

Character → Occupation: employed as, formerly employed as
Character → Location: resides at, present at, owns
Character → Organization: affiliated with, leads, employed by
Location → Location: near to
Organization → Location: located at

IMPORTANT:
- For symmetric relations (married to, friends with, related to, etc.), create TWO directed edges (one in each direction)
- Include edges between characters AND other node types (occupations, locations, organizations)
- Be thorough — extract EVERY relationship you can find in the synopsis
- Use the exact node IDs from the list above (char_0, occ_0, loc_0, org_0, etc.)

Return ONLY valid JSON."""


# Keep the single-pass prompt for backward compatibility
EXTRACTION_PROMPT = PASS1_PROMPT
