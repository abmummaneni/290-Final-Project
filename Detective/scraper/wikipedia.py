"""Wikipedia scraping logic with API library + BeautifulSoup fallback."""

import logging
import re
import time

import requests
import wikipediaapi
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Section titles to try, in priority order
PLOT_SECTION_NAMES = [
    "Plot",
    "Synopsis",
    "Plot summary",
    "Story",
    "Storyline",
    "Summary",
    "Overview",
    "Plot overview",
    "Episode summary",
    "Season summary",
]

# Polite delay between requests (seconds)
REQUEST_DELAY = 0.5


def _get_wiki_client() -> wikipediaapi.Wikipedia:
    """Create a Wikipedia API client with a polite user-agent."""
    return wikipediaapi.Wikipedia(
        user_agent="MurderMysteryGraphProject/1.0 (academic research)",
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )


def _extract_section_text(page, section_names: list[str]) -> tuple[str, str]:
    """Try to extract text from known plot section names.

    Returns (text, section_name_found) or ("", "") if none found.
    """
    for name in section_names:
        section = page.section_by_title(name)
        if section and section.text.strip():
            return section.text.strip(), name

    # Try subsections one level deep
    for top_section in page.sections:
        for name in section_names:
            sub = top_section.section_by_title(name)
            if sub and sub.text.strip():
                return sub.text.strip(), f"{top_section.title} > {name}"

    return "", ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_page_api(title: str) -> wikipediaapi.WikipediaPage:
    """Fetch a Wikipedia page via the API library with retries."""
    wiki = _get_wiki_client()
    page = wiki.page(title)
    return page


def _build_search_queries(title: str, author: str, medium: str, year: int) -> list[str]:
    """Build a ranked list of Wikipedia search queries for an entry."""
    queries = [title]

    if medium == "Novel" or medium == "Short Story":
        queries.append(f"{title} (novel)")
        queries.append(f"{title} (short story)")
        queries.append(f"{title} ({author} novel)")
    elif medium == "Film":
        queries.append(f"{title} (film)")
        queries.append(f"{title} ({year} film)")
    elif medium == "TV Episode":
        queries.append(f"{title} (TV series)")
    elif medium == "Podcast":
        queries.append(f"{title} (podcast)")

    return queries


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_html_fallback(title_slug: str) -> str:
    """Fallback: fetch raw HTML from Wikipedia and extract plot section."""
    url = f"https://en.wikipedia.org/wiki/{title_slug}"
    resp = requests.get(url, headers={"User-Agent": "MurderMysteryGraphProject/1.0"}, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # Find the plot/synopsis heading
    for section_name in PLOT_SECTION_NAMES:
        heading = soup.find("span", {"id": section_name})
        if not heading:
            heading = soup.find("span", {"id": section_name.replace(" ", "_")})
        if heading:
            # Collect all <p> tags until the next heading
            paragraphs = []
            element = heading.parent.find_next_sibling()
            while element:
                if element.name and re.match(r"h[1-4]", element.name):
                    break
                if element.name == "p":
                    paragraphs.append(element.get_text())
                element = element.find_next_sibling()
            if paragraphs:
                return "\n\n".join(paragraphs)

    return ""


def scrape_synopsis(title: str, author: str, medium: str, year: int) -> dict:
    """Scrape a synopsis from Wikipedia for the given entry.

    Returns a dict with keys:
        - text: the raw synopsis text
        - wikipedia_url: the URL of the page used
        - section_found: the section title where text was found
        - status: SUCCESS, PARTIAL, NEEDS_MANUAL, or FAILED
    """
    result = {
        "text": "",
        "wikipedia_url": "",
        "section_found": "",
        "status": "NEEDS_MANUAL",
    }

    queries = _build_search_queries(title, author, medium, year)

    for query in queries:
        try:
            time.sleep(REQUEST_DELAY)
            page = _fetch_page_api(query)

            if not page.exists():
                logger.debug("Page not found for query: %s", query)
                continue

            text, section = _extract_section_text(page, PLOT_SECTION_NAMES)
            if text:
                result["text"] = text
                result["wikipedia_url"] = page.fullurl
                result["section_found"] = section
                result["status"] = "SUCCESS"
                logger.info("Found synopsis for '%s' via query '%s' (section: %s)",
                            title, query, section)
                return result

            # Page exists but no plot section — try HTML fallback
            title_slug = page.title.replace(" ", "_")
            time.sleep(REQUEST_DELAY)
            fallback_text = _fetch_html_fallback(title_slug)
            if fallback_text:
                result["text"] = fallback_text
                result["wikipedia_url"] = page.fullurl
                result["section_found"] = "HTML fallback"
                result["status"] = "SUCCESS"
                logger.info("Found synopsis for '%s' via HTML fallback", title)
                return result

        except Exception:
            logger.warning("Error scraping '%s' with query '%s'", title, query, exc_info=True)
            continue

    logger.warning("No synopsis found for '%s'", title)
    return result
