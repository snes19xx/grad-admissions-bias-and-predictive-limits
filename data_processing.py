"""
data_processing.py
Cleans raw data files for database queries and statistical analyses.
"""

import re

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------------------------------------
def _normalize_school(name: str) -> str | None:
    """
    Map a raw GradCafe school name to the institution of interest.

    GradCafe allows free-text entry so the same school appears under many
    spelling variants. This function resolves them to one of the two target
    names, or returns None for any entry that does not match either target
    (e.g. 'West Chester University of Pennsylvania').
    """
    s = str(name).lower().strip()

    # UofT variants: must contain 'toronto'
    if "toronto" in s:
        return "University of Toronto"

    # UPenn variants: must contain 'pennsylvania' but must NOT be a different
    # Pennsylvania school (West Chester, Clarion, Indiana, Lock Haven, etc.)
    # The following list is from Wikipedia:
    # [https://en.wikipedia.org/wiki/List_of_colleges_and_universities_in_Pennsylvania]
    pennsylvania_exclusions = [
        "west chester", "clarion", "indiana university",
        "lock haven", "millersville", "shippensburg",
        "bloomsburg", "kutztown", "east stroudsburg",
        "slippery rock", "cheyney",
    ]
    if "pennsylvania" in s or "upenn" in s or "penn " in s:
        if any(excl in s for excl in pennsylvania_exclusions):
            return None
        return "University of Pennsylvania"

    return None
# -------------------------------------------------------------------------------------------------------

def clean_admissions_data(csv_file: str) -> pd.DataFrame:
    """
    Load a raw GradCafe CSV and clean it for database queries and modelling.

    Steps:
    - Use (_normalize_school) function to nromalize school names
    - Extract binary Status_Binary from the decision text (i.e. accepted|rejected).
    - Parse numeric GPA; null out values above 4.0.
    - Extract GRE_Total and create a binary GRE_Reported column.
    - Drop rows missing both Status_Binary and GPA.
    """
    df = pd.read_csv(csv_file)

    keep = ["School", "Program", "Season", "Date_Added", "Decision_Detail", "GPA", "GRE"]
    df_clean = df[keep].copy()
    df_clean = df_clean.rename(columns={"Date_Added": "Date", "Decision_Detail": "Decision"})

    df_clean["School"] = df_clean["School"].apply(_normalize_school)
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=["School"])

    first_word = df_clean["Decision"].astype(str).str.extract(
        r"^\s*(accept|accepted|reject|rejected)", flags=re.IGNORECASE
    )[0]

    df_clean["Status_Binary"] = np.select(
        [
            first_word.str.contains(r"accept", case=False, na=False),
            first_word.str.contains(r"reject", case=False, na=False),
        ],
        [1, 0],
        default=np.nan,
    )

    df_clean["GPA"] = pd.to_numeric(
        df_clean["GPA"].astype(str).str.extract(r"([234]\.\d+)")[0],
        errors="coerce",
    )
    # Values above 4.0 are on a non-standard scale (4.33/10 or 100 scales)
    df_clean.loc[df_clean["GPA"] > 4.0, "GPA"] = np.nan

    df_clean["GRE_Total"] = pd.to_numeric(
        df_clean["GRE"].astype(str).str.extract(r"gre.*?(\d{3})", flags=re.IGNORECASE)[0],
        errors="coerce",
    )
    df_clean["GRE_Reported"] = df_clean["GRE_Total"].notna().astype(int)

    df_clean = df_clean.dropna(subset=["Status_Binary", "GPA"])
    return df_clean

# ------------------------------------------- EOF ----------------------------------------------------------