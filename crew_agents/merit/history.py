"""Previous-year closing-merit store + admission-chance classification.

Closing merits come from three places, in this priority order:
  1. CSV uploaded by the student (their own copy of an official merit list)
  2. Rows saved by the Merit Crew after it found AND verified them in a fetched page
  3. Nothing -> the app says "no data" and never invents a number
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from db import connect

REQUIRED_COLS = ("exam", "year", "institution", "program", "closing_merit")


def init_tables() -> None:
    with connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS closing_merits (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, exam TEXT, year INTEGER,
            institution TEXT, program TEXT, category TEXT DEFAULT 'open merit', closing_merit REAL,
            source_url TEXT DEFAULT '', quote TEXT DEFAULT '', origin TEXT DEFAULT 'upload',
            verified INTEGER DEFAULT 0, created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_cm_exam ON closing_merits(student_id, exam, year);
        CREATE TABLE IF NOT EXISTS merit_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, exam TEXT, aggregate REAL,
            inputs_json TEXT, result_json TEXT, created_at TEXT
        );
        """)


def add_rows(student_id: str, rows: list[dict[str, Any]], origin: str = "upload") -> int:
    n = 0
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    with connect() as con:
        for r in rows:
            try:
                cm = float(str(r["closing_merit"]).replace("%", "").strip())
                year = int(float(r["year"]))
            except (KeyError, ValueError):
                continue
            if not (0 < cm <= 100):
                continue
            con.execute(
                "INSERT INTO closing_merits(student_id,exam,year,institution,program,category,closing_merit,"
                "source_url,quote,origin,verified,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (student_id, str(r.get("exam", "")).strip(), year, str(r.get("institution", "")).strip(),
                 str(r.get("program", "")).strip(), str(r.get("category") or "open merit").strip(), cm,
                 str(r.get("source_url", "")), str(r.get("quote", "")), origin, int(bool(r.get("verified"))), now))
            n += 1
    return n


def parse_csv(data: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    cols = [c.strip().lower() for c in (reader.fieldnames or [])]
    missing = [c for c in REQUIRED_COLS if c not in cols]
    if missing:
        return [], [f"Missing required column(s): {', '.join(missing)}. Required: {', '.join(REQUIRED_COLS)} "
                    "(optional: category, source_url)."]
    rows = [{k.strip().lower(): v for k, v in row.items()} for row in reader]
    return rows, []


def get_rows(student_id: str, exam: str | None = None) -> list[dict[str, Any]]:
    q, args = "SELECT * FROM closing_merits WHERE student_id=?", [student_id]
    if exam:
        q += " AND lower(exam)=lower(?)"; args.append(exam)
    q += " ORDER BY year DESC, closing_merit DESC"
    with connect() as con:
        return [dict(r) for r in con.execute(q, args).fetchall()]


def clear_rows(student_id: str, origin: str | None = None) -> None:
    with connect() as con:
        if origin:
            con.execute("DELETE FROM closing_merits WHERE student_id=? AND origin=?", (student_id, origin))
        else:
            con.execute("DELETE FROM closing_merits WHERE student_id=?", (student_id,))


@dataclass
class Recommendation:
    institution: str
    program: str
    category: str
    year: int
    closing_merit: float
    margin: float            # your aggregate - closing merit (positive = above last year's cut-off)
    band: str                # Safe | Target | Reach | Unlikely
    verified: bool
    source_url: str
    trend_note: str = ""


def band_for(margin: float, safe: float = 2.0, target: float = -1.0, reach: float = -4.0) -> str:
    """Thresholds are in aggregate percentage points and are shown to the user (not hidden)."""
    if margin >= safe:
        return "Safe"
    if margin >= target:
        return "Target"
    if margin >= reach:
        return "Reach"
    return "Unlikely"


def recommend(aggregate: float, rows: list[dict[str, Any]], safe: float = 2.0, target: float = -1.0,
              reach: float = -4.0) -> list[Recommendation]:
    """Use the most recent year per (institution, program, category); add a trend note if 2+ years exist."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["institution"].lower(), r["program"].lower(), (r.get("category") or "").lower()), []).append(r)
    out: list[Recommendation] = []
    for _, g in groups.items():
        g.sort(key=lambda x: x["year"], reverse=True)
        latest = g[0]
        margin = aggregate - latest["closing_merit"]
        note = ""
        if len(g) > 1:
            delta = g[0]["closing_merit"] - g[1]["closing_merit"]
            direction = "rose" if delta > 0 else "fell" if delta < 0 else "was unchanged"
            note = f"Closing merit {direction} {abs(delta):.2f} pts from {g[1]['year']} to {g[0]['year']}."
            if delta > 0:
                note += " If the trend continues, expect a higher cut-off."
        out.append(Recommendation(latest["institution"], latest["program"], latest.get("category") or "open merit",
                                  int(latest["year"]), float(latest["closing_merit"]), round(margin, 2),
                                  band_for(margin, safe, target, reach), bool(latest.get("verified")) or latest.get("origin") == "upload",
                                  latest.get("source_url", ""), note))
    order = {"Safe": 0, "Target": 1, "Reach": 2, "Unlikely": 3}
    out.sort(key=lambda r: (order[r.band], -r.closing_merit))
    return out
