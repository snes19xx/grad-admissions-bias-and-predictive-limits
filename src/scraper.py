"""
scraper.py
Handles web scraping operations for admissions data.
Works with any institution available at thegradcafe.com
----
NOTE:This script takes a while to complete and
You MUST install cloudscraper to test/run this script
"""

import random
import re
import time

import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup


def scrape_all_masters(university_name: str, max_pages: int = None) -> pd.DataFrame:
    """
    Scrape Master's degree admission records for a given university from TheGradCafe.
    Uses cloudscraper to bypass Cloudflare protection, parses the HTML table,
    and applies filtering to keep only records where the school name
    matches the input keywords and the degree is tagged as Masters. Results are
    saved to a CSV.
    ----------
    Parameters
    ----------
    university_name : str
        The university to query and filter against.
    max_pages : int or None
        Maximum pages to scrape. Defaults to None (all pages).
    --------
    Returns
    --------
    pd.DataFrame
        Records with columns: School, Program, Season, Status, GPA, GRE, Comment.
    """
    base_url = "https://www.thegradcafe.com/survey/"
    scraper = cloudscraper.create_scraper()
    all_results = []
    page = 1

    ignore_words = {"university", "of", "the", "at", "college", "institute", "state"}
    search_keywords = [
        w.lower() for w in university_name.split() if w.lower() not in ignore_words
    ]
    if not search_keywords:
        search_keywords = [university_name.lower()]

    while True:
        if max_pages and page > max_pages:
            break

        params = {
            "q":              university_name,
            "sort":           "newest",
            "institution":    "",
            "program":        "",
            "degree":         "Masters",
            "season":         "",
            "decision":       "",
            "decision_start": "",
            "decision_end":   "",
            "added_start":    "",
            "added_end":      "",
            "page":           page,
        }

        print(f"Scraping page {page} for {university_name}...")
        response = scraper.get(base_url, params=params)

        if response.status_code != 200:
            print(f"Failed on page {page}. Status: {response.status_code}")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        table_body = soup.find("tbody")

        if not table_body:
            print("No table found. Reached the end or blocked.")
            break

        rows = table_body.find_all("tr", recursive=False)
        if not rows:
            print("No rows found.")
            break

        current_record = {}
        is_valid = False
        valid_on_page = 0

        for row in rows:
            tds = row.find_all("td", recursive=False)

            if len(tds) >= 4:
                if current_record and is_valid:
                    all_results.append(current_record)
                    valid_on_page += 1

                school = tds[0].get_text(strip=True)
                program_text = tds[1].get_text(separator="|", strip=True)
                program = program_text.split("|")[0] if "|" in program_text else program_text

                degree_span = tds[1].find_all("span", class_="tw-text-gray-500")
                degree_type = (
                    degree_span[-1].get_text(strip=True).lower() if degree_span else ""
                )

                school_lower = school.lower()
                is_school_match = any(kw in school_lower for kw in search_keywords)
                is_valid = is_school_match and "master" in degree_type

                if is_valid:
                    current_record = {
                        "School":          school,
                        "Program":         program,
                        "Date_Added":      tds[2].get_text(strip=True),
                        "Decision_Detail": tds[3].get_text(strip=True),
                        "Season":          None,
                        "Status":          None,
                        "GPA":             None,
                        "GRE":             None,
                        "Comment":         None,
                    }

            elif len(tds) == 1 and tds[0].get("colspan") == "3":
                if is_valid and current_record:
                    tag_divs = tds[0].find_all(
                        "div", class_=lambda x: x and "tw-inline-flex" in x
                    )
                    for tag in [d.get_text(strip=True) for d in tag_divs]:
                        tl = tag.lower()
                        if re.match(r"^[fs]\d{2}$", tl) or "fall" in tl or "spring" in tl:
                            current_record["Season"] = tag
                        elif any(s in tl for s in ["international", "american", "domestic", "canadian"]):
                            current_record["Status"] = tag
                        elif "gpa" in tl:
                            current_record["GPA"] = tag.replace("GPA", "").strip()
                        elif "gre" in tl:
                            current_record["GRE"] = (
                                current_record["GRE"] + f" | {tag}"
                                if current_record["GRE"]
                                else tag
                            )

            elif len(tds) == 1 and tds[0].get("colspan") == "100%":
                if is_valid and current_record:
                    p_tag = tds[0].find("p", class_="tw-text-gray-500")
                    if p_tag:
                        current_record["Comment"] = p_tag.get_text(strip=True)

        if current_record and is_valid:
            all_results.append(current_record)
            valid_on_page += 1

        if valid_on_page == 0:
            print("Zero valid records on this page. Stopping.")
            break

        page += 1
        time.sleep(random.uniform(2, 5))

    df = pd.DataFrame(all_results)
    filename = f"{university_name.lower().replace(' ', '_')}_all_masters_decisions.csv"
    if not df.empty:
        df.to_csv(filename, index=False)
        print(f"Saved {len(df):,} records to '{filename}'")
    else:
        print("No valid records found.")
    return df
# ------------------------------------------------- EOF ------------------------------------------------------
