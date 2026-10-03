"""Path Finder Agent: grounded career/program/scholarship guidance.
Facts (programs, eligibility, accreditation, fees, scholarships) are shown straight from verified DB records.
The LLM only explains fit and drafts the roadmap, may cite only supplied record IDs, and must say
"Not available in verified data" for anything missing. Unknown citations are stripped."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from career_db import fetch
from groq_service import generate_json
from web_search import search_web

TRUSTED_WEB = (".edu.pk", ".gov.pk", "hec.gov.pk", "pmdc.pk", "pec.org.pk", "pharmacycouncil.pk", "pnc.org.pk", ".edu", ".gov")
NA = "Not available in verified data"


@dataclass
class StudentProfile:
    stream: str = ""                 # e.g. "Pre-Medical"
    inter_pct: float | None = None
    matric_pct: float | None = None
    interests: str = ""
    goal: str = ""
    city: str = ""
    max_tuition_pkr: float | None = None
    family_income_pkr: float | None = None
    entry_tests_taken: list[str] = field(default_factory=list)


def _num(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def check_eligibility(p: StudentProfile, prog: dict[str, Any]) -> dict[str, Any]:
    """Returns status Eligible / Not eligible / Needs confirmation with plain-language reasons."""
    fails, unknown = [], []
    for key, mine, label in (("min_inter_pct", p.inter_pct, "Intermediate %"), ("min_matric_pct", p.matric_pct, "Matric %")):
        need = _num(prog.get(key))
        if need is None:
            unknown.append(f"{label} requirement: {NA}")
        elif mine is None:
            unknown.append(f"Your {label} was not provided")
        elif mine < need:
            fails.append(f"{label} {mine:g} is below the required {need:g}")
    tests = [t.strip().lower() for t in str(prog.get("accepted_entry_tests", "")).replace(";", ",").split(",") if t.strip()]
    if tests and not any(t.lower() in tests for t in p.entry_tests_taken):
        unknown.append(f"Entry test needed: {', '.join(tests)}")
    subj = str(prog.get("required_subjects", "")).strip()
    if subj:
        unknown.append(f"Check required subjects: {subj}")
    status = "Not eligible" if fails else ("Needs confirmation" if unknown else "Eligible")
    return {"status": status, "fails": fails, "notes": unknown}


def eligible_scholarships(p: StudentProfile, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for s in rows:
        mp, mi = _num(s.get("min_pct")), _num(s.get("max_family_income_pkr"))
        if mp is not None and p.inter_pct is not None and p.inter_pct < mp:
            continue
        flags = []
        if mi is not None:
            flags.append("Income limit applies" if p.family_income_pkr is None else "")
            if p.family_income_pkr is not None and p.family_income_pkr > mi:
                continue
        out.append({**s, "flags": [f for f in flags if f]})
    return out


def _interest_score(row: dict[str, Any], text: str, cols: tuple[str, ...]) -> int:
    words = {w for w in text.lower().replace(",", " ").split() if len(w) > 3}
    hay = " ".join(str(row.get(c, "")) for c in cols).lower()
    return sum(1 for w in words if w in hay)


class PathFinderAgent:
    name = "Path Finder Agent"

    def run(self, p: StudentProfile, use_web: bool = False) -> dict[str, Any]:
        query = f"{p.stream} {p.interests} {p.goal}"
        programs = fetch("pf_programs")
        for r in programs:
            r["eligibility"] = check_eligibility(p, r)
            r["_score"] = _interest_score(r, query, ("program", "field", "degree"))
        if p.city:
            programs = [r for r in programs if not r.get("city") or p.city.lower() in r["city"].lower() or r["_score"] > 1] or programs
        if p.max_tuition_pkr is not None:
            programs = [r for r in programs if _num(r.get("tuition_per_year_pkr")) is None or _num(r["tuition_per_year_pkr"]) <= p.max_tuition_pkr]
        rank = {"Eligible": 0, "Needs confirmation": 1, "Not eligible": 2}
        programs = sorted(programs, key=lambda r: (rank[r["eligibility"]["status"]], -r["_score"]))[:12]
        scholarships = eligible_scholarships(p, fetch("pf_scholarships"))
        scholarships = sorted(scholarships, key=lambda s: -_interest_score(s, query, ("name", "field", "eligibility_text")))[:8]
        careers = sorted(fetch("pf_careers"), key=lambda c: -_interest_score(c, query, ("career", "field", "required_degrees")))[:6]
        web = []
        if use_web and (len(programs) < 3 or not careers):
            web = [r for r in search_web(f"{p.goal or p.interests} {p.stream} Pakistan admission requirements", 8)
                   if any(h in r["url"].lower() for h in TRUSTED_WEB)][:4]

        valid = {f"P{r['id']}" for r in programs} | {f"S{r['id']}" for r in scholarships} | {f"C{r['id']}" for r in careers}
        valid |= {f"W{i + 1}" for i in range(len(web))}
        ctx = "\n".join(
            [f"[P{r['id']}] {r['institution']} | {r['program']} | field={r['field']} | eligibility={r['eligibility']['status']} {r['eligibility']['fails']}"
             f" | accreditation={r.get('accreditation_body') or NA}:{r.get('accreditation_status') or NA}" for r in programs]
            + [f"[S{r['id']}] {r['name']} | {r['provider']} | covers={r.get('covers') or NA} | deadline={r.get('deadline') or NA}" for r in scholarships]
            + [f"[C{r['id']}] {r['career']} | field={r['field']} | degrees={r.get('required_degrees') or NA} | path={r.get('typical_path') or NA}" for r in careers]
            + [f"[W{i + 1}] (unverified web) {r['title']}: {r['snippet'][:300]} ({r['url']})" for i, r in enumerate(web)])
        data: dict[str, Any] = {}
        if ctx.strip():
            data = generate_json(f"""You are Path Finder, an education and career guidance assistant for Pakistani students.
STRICT RULES: use ONLY the records below. Never invent universities, fees, deadlines, merit, accreditation or scholarships.
If something is missing say exactly "{NA}". Cite record IDs (like P3, S2, C1, W1) for every claim. Web records are unverified; say so.
STUDENT: stream={p.stream}; inter%={p.inter_pct}; matric%={p.matric_pct}; interests={p.interests}; goal={p.goal}; city={p.city}
RECORDS:
{ctx}
Return JSON: {{"summary": str, "career_paths": [{{"career": str, "why": str, "source_ids": [str]}}],
"program_fit": [{{"id": str, "reason": str}}],
"roadmap": [{{"phase": str, "timeframe": str, "actions": [str], "source_ids": [str]}}], "gaps": [str]}}
Roadmap phases: Now, Before applications, Entry tests, Admission, During degree, Career start.""")
        for sec in ("career_paths", "roadmap"):
            for item in data.get(sec, []) or []:
                item["source_ids"] = [i for i in item.get("source_ids", []) if i in valid]
        data["program_fit"] = [x for x in data.get("program_fit", []) or [] if x.get("id") in valid]
        gaps = data.get("gaps", []) or []
        if not ctx.strip():
            gaps = ["No verified records matched. Import programs, scholarships and careers in the Data tab (or enable trusted web search)."]
        return {"ai": data, "programs": programs, "scholarships": scholarships, "careers": careers, "web": web, "gaps": gaps,
                "disclaimer": "Always confirm eligibility, fees, deadlines and accreditation on the official university/HEC/PMDC/PEC website before applying."}
