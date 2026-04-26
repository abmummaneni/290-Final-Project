"""Synopsis quality scoring for cleaned text."""

import math
import re


# Event/crime verbs that signal narrative density
EVENT_VERBS = re.compile(
    r"\b(kill|killed|kills|murder|murdered|murders|die|died|dies|death|dead|"
    r"shoot|shot|stab|stabbed|poison|poisoned|strangle|strangled|"
    r"attack|attacked|assault|assaulted|threaten|threatened|"
    r"arrest|arrested|confess|confessed|reveal|revealed|discover|discovered|"
    r"investigate|investigated|suspect|suspected|accuse|accused|"
    r"betray|betrayed|blackmail|blackmailed|escape|escaped)\b",
    re.IGNORECASE,
)

# Resolution/reveal indicators
RESOLUTION_PATTERNS = re.compile(
    r"\b(turns out|revealed to be|unmasked|confesses|is the (killer|murderer|villain)|"
    r"guilty|perpetrator|committed the|was responsible|the culprit|"
    r"it is discovered that|finally reveals)\b",
    re.IGNORECASE,
)


def _sigmoid_score(value: float, midpoint: float, steepness: float = 0.01) -> float:
    """Compute a sigmoid-scaled score in [0, 1]."""
    return 1.0 / (1.0 + math.exp(-steepness * (value - midpoint)))


def _word_count_score(word_count: int) -> float:
    """Score based on word count. Target: 300+ words.

    300 words → ~0.5, 600 words → ~0.9
    """
    return _sigmoid_score(word_count, midpoint=300, steepness=0.01)


def _character_count_score(text: str) -> float:
    """Score based on named character mentions.

    Heuristic: count capitalized multi-word sequences that look like names.
    Target: 3+ distinct names.
    """
    # Match capitalized words that look like names (2+ chars, not sentence starters)
    # Split into sentences first, then look for capitalized words mid-sentence
    words = text.split()
    name_candidates = set()
    for i, word in enumerate(words):
        # Skip first word of sentences
        if i > 0 and not words[i - 1].endswith((".", "!", "?")):
            clean = re.sub(r"[^a-zA-Z]", "", word)
            if clean and clean[0].isupper() and len(clean) >= 2:
                name_candidates.add(clean)

    count = len(name_candidates)
    return _sigmoid_score(count, midpoint=3, steepness=0.8)


def _event_density_score(text: str) -> float:
    """Score based on density of crime/event verbs."""
    words = text.split()
    if not words:
        return 0.0
    matches = len(EVENT_VERBS.findall(text))
    density = matches / len(words) * 100  # events per 100 words
    return min(1.0, density / 2.0)  # 2 events per 100 words → 1.0


def _resolution_score(text: str) -> float:
    """Binary score: does the synopsis contain a resolution/reveal?"""
    return 1.0 if RESOLUTION_PATTERNS.search(text) else 0.0


def score_synopsis(text: str) -> float:
    """Compute a quality score in [0, 1] for a cleaned synopsis.

    Weights:
        Word count:        0.35
        Character count:   0.30
        Event density:     0.20
        Resolution:        0.15
    """
    word_count = len(text.split())

    wc_score = _word_count_score(word_count)
    cc_score = _character_count_score(text)
    ed_score = _event_density_score(text)
    rs_score = _resolution_score(text)

    total = (
        0.35 * wc_score
        + 0.30 * cc_score
        + 0.20 * ed_score
        + 0.15 * rs_score
    )

    return round(total, 4)
