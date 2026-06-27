"""
database.py
Manages SQLite connections, table creation, and queries.
"""

import re
import sqlite3

import pandas as pd


# -------------------------------------------------------------------------------------------------------
def create_database(
    uoft_df: pd.DataFrame,
    upenn_df: pd.DataFrame,
    db_path: str = "admissions.db",
) -> sqlite3.Connection:
    """
    Build and fill a normalized SQLite database from two cleaned DataFrames.

    Creates two tables: institutions and applications. Every row in applications
    references its parent institution via institution_id, the foreign key linking
    the two tables.
    ----------
    Parameters
    ----------
    uoft_df : pd.DataFrame
    upenn_df : pd.DataFrame
    db_path : str
    -------
    Returns
    -------
    sqlite3.Connection
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS applications")
    cur.execute("DROP TABLE IF EXISTS institutions")

    cur.execute("""
        CREATE TABLE institutions (
            id      INTEGER PRIMARY KEY,
            name    TEXT NOT NULL,
            city    TEXT,
            country TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE applications (
            id             INTEGER PRIMARY KEY,
            institution_id INTEGER NOT NULL,
            program        TEXT,
            season         TEXT,
            date           TEXT,
            decision       TEXT,
            status_binary  INTEGER,
            gpa            REAL,
            gre_total      REAL,
            gre_reported   INTEGER,
            FOREIGN KEY (institution_id) REFERENCES institutions(id)
        )
    """)

    cur.execute(
        "INSERT INTO institutions (name, city, country) VALUES (?, ?, ?)",
        ("University of Toronto", "Toronto", "Canada"),
    )
    uoft_id = cur.lastrowid

    cur.execute(
        "INSERT INTO institutions (name, city, country) VALUES (?, ?, ?)",
        ("University of Pennsylvania", "Philadelphia", "USA"),
    )
    upenn_id = cur.lastrowid

    def _insert_apps(df: pd.DataFrame, inst_id: int) -> None:
        for row in df.itertuples(index=False):
            cur.execute(
                """
                INSERT INTO applications
                    (institution_id, program, season, date, decision,
                     status_binary, gpa, gre_total, gre_reported)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inst_id,
                    row.Program  if pd.notna(row.Program)  else None,
                    row.Season   if pd.notna(row.Season)   else None,
                    row.Date     if pd.notna(row.Date)     else None,
                    row.Decision if pd.notna(row.Decision) else None,
                    int(row.Status_Binary),
                    float(row.GPA),
                    float(row.GRE_Total) if pd.notna(row.GRE_Total) else None,
                    int(row.GRE_Reported),
                ),
            )

    _insert_apps(uoft_df, uoft_id)
    _insert_apps(upenn_df, upenn_id)
    conn.commit()
    print(f"Database created at '{db_path}'.")
    return conn
# -------------------------------------------------------------------------------------------------------
def query_join(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    JOIN: unified view of all applications with institution data.
    """
    sql = """
        SELECT
            i.name          AS institution,
            a.program,
            a.season,
            a.status_binary AS accepted,
            a.gpa,
            a.gre_total,
            a.gre_reported
        FROM  applications AS a
        JOIN  institutions AS i ON a.institution_id = i.id
        ORDER BY i.name, a.gpa DESC
    """
    return pd.read_sql_query(sql, conn)
# -------------------------------------------------------------------------------------------------------
def query_groupby_acceptance_rate(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    GROUP BY: overall acceptance rate per institution using SUM and COUNT.
    """
    sql = """
        SELECT
            i.name                                            AS institution,
            COUNT(*)                                          AS total_applications,
            SUM(a.status_binary)                              AS accepted,
            100.0 * SUM(a.status_binary) / COUNT(*)          AS acceptance_rate_pct
        FROM  applications AS a
        JOIN  institutions AS i ON a.institution_id = i.id
        GROUP BY i.id
        ORDER BY acceptance_rate_pct DESC
    """
    return pd.read_sql_query(sql, conn)
# -------------------------------------------------------------------------------------------------------
def query_aggregate_gpa(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Aggregate: AVG, MIN, MAX GPA by institution and admission outcome.
    """
    sql = """
        SELECT
            i.name          AS institution,
            a.status_binary AS accepted,
            AVG(a.gpa)      AS avg_gpa,
            MIN(a.gpa)      AS min_gpa,
            MAX(a.gpa)      AS max_gpa,
            COUNT(*)        AS n
        FROM  applications AS a
        JOIN  institutions AS i ON a.institution_id = i.id
        GROUP BY i.id, a.status_binary
        ORDER BY i.name, a.status_binary DESC
    """
    return pd.read_sql_query(sql, conn)
# -------------------------------------------------------------------------------------------------------
def query_by_institution(conn: sqlite3.Connection, institution_name: str) -> pd.DataFrame:
    """
    Retrieve applications for a specific institution using LIKE.
    """
    sql = """
        SELECT
            i.name          AS institution,
            a.program,
            a.season,
            a.gpa,
            a.gre_total,
            a.status_binary AS accepted
        FROM  applications AS a
        JOIN  institutions AS i ON a.institution_id = i.id
        WHERE LOWER(i.name) LIKE LOWER(?)
        ORDER BY a.gpa DESC
    """
    return pd.read_sql_query(sql, conn, params=(f"%{institution_name}%",))
# -------------------------------------------------------------------------------------------------------
def query_acceptance_by_gpa_range(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Acceptance rate within GPA range per institution.

    Bins: <3.0, 3.0-3.25, 3.25-3.5, 3.5-3.75, 3.75-4.0.
    Directly addresses Q1: at what GPA range does admission probability
    meaningfully change?
    """
    sql = """
        SELECT
            i.name          AS institution,
            a.gpa,
            a.status_binary AS accepted
        FROM  applications AS a
        JOIN  institutions AS i ON a.institution_id = i.id
        WHERE a.gpa IS NOT NULL AND a.status_binary IS NOT NULL
    """
    df = pd.read_sql_query(sql, conn)

    bins   = [0.0, 3.0, 3.25, 3.5, 3.75, 4.01]
    labels = ["< 3.0", "3.0–3.25", "3.25–3.5", "3.5–3.75", "3.75–4.0"]
    df["gpa_range"] = pd.cut(df["gpa"], bins=bins, labels=labels, right=False)

    result = (
        df.groupby(["institution", "gpa_range"], observed=True)
        .agg(n=("accepted", "count"), accepted=("accepted", "sum"))
        .reset_index()
    )
    result["acceptance_rate_pct"] = (
        100.0 * result["accepted"] / result["n"]
    ).round(1)
    return result
# -------------------------------------------------------------------------------------------------------
def query_gre_acceptance(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    GROUP BY: acceptance rate by GRE reporting status per institution.

    Addresses Q3: does submitting a GRE score correlate with a higher
    admission rate in programs where it is optional?
    """
    sql = """
        SELECT
            i.name                                            AS institution,
            a.gre_reported,
            COUNT(*)                                          AS total,
            SUM(a.status_binary)                              AS accepted,
            100.0 * SUM(a.status_binary) / COUNT(*)          AS acceptance_rate_pct
        FROM  applications AS a
        JOIN  institutions AS i ON a.institution_id = i.id
        GROUP BY i.id, a.gre_reported
        ORDER BY i.name, a.gre_reported
    """
    return pd.read_sql_query(sql, conn)
# -------------------------------------------------------------------------------------------------------
def query_top_programs(conn: sqlite3.Connection, min_n: int = 15) -> pd.DataFrame:
    """
    Most and least competitive programs by acceptance rate across both schools,
    requiring at least min_n applicants for statistical stability.
    """
    sql = """
        SELECT
            i.name                                            AS institution,
            a.program,
            COUNT(*)                                          AS total,
            SUM(a.status_binary)                              AS accepted,
            100.0 * SUM(a.status_binary) / COUNT(*)          AS acceptance_rate_pct
        FROM  applications AS a
        JOIN  institutions AS i ON a.institution_id = i.id
        WHERE a.program IS NOT NULL
        GROUP BY i.id, a.program
        HAVING COUNT(*) >= ?
        ORDER BY acceptance_rate_pct ASC
    """
    return pd.read_sql_query(sql, conn, params=(min_n,))
# -------------------------------------------------------------------------------------------------------
def query_temporal_raw(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    All rows with a non-null season and decision for yearly trends.
    Year parsing is done with _parse_year function below.
    """
    sql = """
        SELECT
            i.name          AS institution,
            a.season,
            a.status_binary AS accepted
        FROM  applications AS a
        JOIN  institutions AS i ON a.institution_id = i.id
        WHERE a.season IS NOT NULL AND a.status_binary IS NOT NULL
    """
    return pd.read_sql_query(sql, conn)
# -------------------------------------------------------------------------------------------------------
def _parse_year(s: str):
    """
    Extract a 4-digit year from strings like 'Fall 2026' or 'F16'."""
    m = re.search(r"(\d{4})", str(s))
    if m:
        return int(m.group(1))
    m = re.search(r"[FSfs](\d{2})\b", str(s))
    if m:
        return 2000 + int(m.group(1))
    return None
## --------------------------------------------- EOF ------------------------------------------------------
