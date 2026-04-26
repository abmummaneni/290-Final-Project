"""Load and parse the murder mystery candidate list from xlsx."""

import os
import pandas as pd


# Column names in the xlsx (Sheet: "Candidate List")
_COLUMNS = [
    "Title",
    "Author / Director",
    "Year",
    "Medium",
    "Subgenre",
    "Villain Reveal",
    "Synopsis Quality",
    "Series Name",
    "Series Entry #",
    "Notes",
]

# Medium → ID prefix mapping
MEDIUM_CODES = {
    "Novel": "NOV",
    "Short Story": "SHO",
    "Film": "FLM",
    "TV Episode": "TVE",
    "Podcast": "POD",
}


def _assign_entry_id(row_index: int, medium: str) -> str:
    """Assign a stable entry ID based on medium code and row index."""
    code = MEDIUM_CODES.get(medium, "UNK")
    return f"{code}_{row_index + 1:03d}"


def load_candidates(xlsx_path: str) -> pd.DataFrame:
    """Load candidate list from xlsx and assign stable entry IDs.

    The xlsx is sorted by (Medium, Year, Title) and each entry gets an ID
    like NOV_001, FLM_042, etc. based on its position within its medium group.

    Returns a DataFrame with an 'entry_id' column and all original columns.
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Candidate list not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path, sheet_name="Candidate List")

    # Sort by Medium, Year, Title for stable ID assignment
    df = df.sort_values(["Medium", "Year", "Title"]).reset_index(drop=True)

    # Assign entry IDs per medium group
    df["entry_id"] = ""
    for medium in df["Medium"].unique():
        mask = df["Medium"] == medium
        indices = df.loc[mask].index
        for i, idx in enumerate(indices):
            df.at[idx, "entry_id"] = _assign_entry_id(i, medium)

    # Rename for easier access
    df = df.rename(columns={
        "Author / Director": "author",
        "Title": "title",
        "Year": "year",
        "Medium": "medium",
        "Subgenre": "subgenre",
        "Villain Reveal": "villain_reveal",
        "Synopsis Quality": "synopsis_quality_flag",
        "Series Name": "series_name",
        "Series Entry #": "series_entry",
        "Notes": "notes",
    })

    return df


def get_entry(df: pd.DataFrame, entry_id: str) -> dict:
    """Get a single entry by its entry_id as a dict."""
    row = df[df["entry_id"] == entry_id]
    if row.empty:
        raise KeyError(f"Entry not found: {entry_id}")
    return row.iloc[0].to_dict()


def get_entries_by_medium(df: pd.DataFrame, medium: str) -> pd.DataFrame:
    """Filter entries by medium type."""
    return df[df["medium"] == medium]
