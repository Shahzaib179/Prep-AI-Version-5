"""Tables + CSV import for the Merit and Path Finder agents (reuses db.connect)."""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import pandas as pd

from db import connect

SCHEMAS: dict[str, list[str]] = {
    "merit_history": ["exam", "institution", "program", "category", "year", "closing_merit", "source_url"],
    "pf_programs": ["institution", "program", "degree", "field", "city", "duration_years", "min_inter_pct", "min_matric_pct",
                    "accepted_entry_tests", "required_subjects", "accreditation_body", "accreditation_status",
                    "tuition_per_year_pkr", "source_url", "last_verified"],
    "pf_scholarships": ["name", "provider", "level", "field", "min_pct", "max_family_income_pkr", "covers", "deadline",
                        "eligibility_text", "source_url", "last_verified"],
    "pf_careers": ["career", "field", "required_degrees", "typical_path", "licensing_body", "source_url"],
}
REQUIRED = {"merit_history": ["institution", "program", "year", "closing_merit", "source_url"],
            "pf_programs": ["institution", "program", "source_url"], "pf_scholarships": ["name", "source_url"],
            "pf_careers": ["career", "source_url"]}


def init_agent_tables() -> None:
    with connect() as con:
        for t, cols in SCHEMAS.items():
            con.execute(f"CREATE TABLE IF NOT EXISTS {t} (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                        + ", ".join(f"{c} TEXT" for c in cols) + ", created_at TEXT)")


def template_csv(table: str) -> bytes:
    return (",".join(SCHEMAS[table]) + "\n").encode()


def import_csv(table: str, raw: bytes, replace: bool = False) -> tuple[int, list[str]]:
    """Import rows. Rows missing a required field (incl. source_url) are rejected, so every record is traceable."""
    df = pd.read_csv(io.BytesIO(raw), dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED[table] if c not in df.columns]
    if missing:
        return 0, [f"Missing required column(s): {', '.join(missing)}"]
    problems, good = [], []
    for i, row in df.iterrows():
        rec = {c: str(row.get(c, "")).strip() for c in SCHEMAS[table]}
        bad = [c for c in REQUIRED[table] if not rec[c]]
        if table == "merit_history" and not bad:
            try:
                float(rec["closing_merit"]); int(rec["year"])
            except ValueError:
                bad = ["year/closing_merit must be numeric"]
        (problems.append(f"Row {i + 2}: missing/invalid {', '.join(bad)}") if bad else good.append(rec))
    now = datetime.utcnow().isoformat()
    with connect() as con:
        if replace and good:
            con.execute(f"DELETE FROM {table}")
        for rec in good:
            con.execute(f"INSERT INTO {table}({','.join(SCHEMAS[table])},created_at) VALUES({','.join('?' * len(SCHEMAS[table]))},?)",
                        [rec[c] for c in SCHEMAS[table]] + [now])
    return len(good), problems


def fetch(table: str, where: str = "", params: tuple = ()) -> list[dict[str, Any]]:
    with connect() as con:
        return [dict(r) for r in con.execute(f"SELECT * FROM {table} {where}", params).fetchall()]


def count(table: str) -> int:
    with connect() as con:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
