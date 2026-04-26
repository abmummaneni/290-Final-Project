"""Text cleaning and normalization for scraped synopses."""

import re


def clean_synopsis(raw_text: str) -> str:
    """Clean raw scraped text into normalized synopsis text.

    Steps:
    1. Remove wiki markup remnants (references, citations, etc.)
    2. Remove spoiler tags and templates
    3. Normalize whitespace
    4. Remove extraneous metadata lines
    """
    text = raw_text

    # Remove reference markers like [1], [2], [citation needed]
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[note \d+\]", "", text, flags=re.IGNORECASE)

    # Remove wiki template remnants {{...}}
    text = re.sub(r"\{\{[^}]*\}\}", "", text)

    # Remove HTML tags that might have leaked through
    text = re.sub(r"<[^>]+>", "", text)

    # Remove "edit" section markers
    text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)

    # Normalize unicode quotes and dashes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00a0", " ")  # non-breaking space

    # Collapse multiple newlines into paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove empty lines at start and end
    text = text.strip()

    return text
