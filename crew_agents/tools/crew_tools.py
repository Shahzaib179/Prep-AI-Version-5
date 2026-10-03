"""CrewAI tool wrappers (thin) around the pure functions in web_tools / registries.

Tool output is always capped (~3.5k chars) because Groq free-tier requests have small token budgets.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_agents.tools import web_tools
from crew_agents.pathfinder import eligibility as elig

KNOWLEDGE = Path(__file__).resolve().parents[2] / "data" / "knowledge"

# ---------------------------------------------------------------- generic web
class _SearchIn(BaseModel):
    query: str = Field(..., description="Short, specific web search query (3-10 words).")


class FreeWebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = ("Search the web (DuckDuckGo, free). Returns titles, URLs, snippets, ranked with official / "
                        "institutional sources first. Snippets are NOT proof - open the page with read_webpage.")
    args_schema: Type[BaseModel] = _SearchIn

    def _run(self, query: str) -> str:
        return web_tools.search_report(query)[:3500]


class _ReadIn(BaseModel):
    url: str = Field(..., description="Full http(s) URL taken from a previous search result.")
    keywords: str = Field("", description="Comma-separated words to find on the page, e.g. 'closing merit, MBBS, 2024'.")


class ReadWebpageTool(BaseTool):
    name: str = "read_webpage"
    description: str = ("Fetch a web page or PDF and return verbatim excerpts around the keywords. Always pass keywords "
                        "so the excerpt is small. Quote ONLY text returned by this tool.")
    args_schema: Type[BaseModel] = _ReadIn

    def _run(self, url: str, keywords: str = "") -> str:
        return web_tools.read_report(url, keywords)[:3800]


class _SiteIn(BaseModel):
    query: str = Field(..., description="What to find, e.g. 'recognized universities list'.")
    body: str = Field(..., description="Regulator/body key: HEC, PMDC, PEC, PNMC, PCP, PVMC, PCATP, PBC, NUST, NUMS, UET, FAST, or ANY_PK_GOV.")


SITE_MAP = {
    "HEC": "hec.gov.pk", "PMDC": "pmdc.pk", "PEC": "pec.org.pk", "PNMC": "pnmc.gov.pk", "PCP": "pcp.org.pk",
    "PVMC": "pvmc.gov.pk", "PCATP": "pcatp.org.pk", "PBC": "pakistanbarcouncil.org", "NUST": "nust.edu.pk",
    "NUMS": "numspak.edu.pk", "UET": "uet.edu.pk", "FAST": "nu.edu.pk", "ANY_PK_GOV": "gov.pk",
}


class OfficialSourceSearchTool(BaseTool):
    name: str = "official_source_search"
    description: str = ("Search ONLY one official regulator/institution website (site: filter). Use for accreditation, "
                        "recognition, admission policy and scholarship pages. body must be one of: " + ", ".join(SITE_MAP))
    args_schema: Type[BaseModel] = _SiteIn

    def _run(self, query: str, body: str) -> str:
        dom = SITE_MAP.get(body.upper().strip())
        if not dom:
            return f"Unknown body '{body}'. Choose from: {', '.join(SITE_MAP)}"
        return web_tools.search_report(f"site:{dom} {query}")[:3500]


# ---------------------------------------------------------------- merit tools
class _FormulaIn(BaseModel):
    exam: str = Field(..., description="Exam key or name: MDCAT, NUMS, ECAT, NUST, FAST, NTS, CUSTOM.")


class FormulaRegistryTool(BaseTool):
    name: str = "formula_registry"
    description: str = ("Return the app's stored merit formula profile (weights, test total, status, variants, official page hint) "
                        "for an exam. Use it as the hypothesis to verify against the official prospectus.")
    args_schema: Type[BaseModel] = _FormulaIn

    def _run(self, exam: str) -> str:
        from crew_agents.merit.formulas import PROFILES
        e = exam.lower()
        for p in PROFILES.values():
            if e in p.key or e in p.exam.lower():
                return json.dumps({"key": p.key, "exam": p.exam, "institution": p.institution, "weights_percent": p.weights,
                                   "test_total": p.test_total, "status": p.status, "official_hint": p.official_hint,
                                   "variants": [{"label": v.label, "weights": v.weights} for v in p.variants]})
        return "No profile found. Available: " + ", ".join(PROFILES)


# ---------------------------------------------------------------- pathfinder tools
class _CareerIn(BaseModel):
    keyword: str = Field(..., description="Career/occupation keyword in English, e.g. 'data analyst'.")


class CareerDataTool(BaseTool):
    name: str = "career_data_esco"
    description: str = ("Look up occupations in ESCO (European Commission's free occupation classification, no API key). "
                        "Returns occupation titles and short descriptions. Use for career definitions; it does not "
                        "contain Pakistan salaries or job-market data.")
    args_schema: Type[BaseModel] = _CareerIn

    def _run(self, keyword: str) -> str:
        try:
            r = requests.get("https://ec.europa.eu/esco/api/search",
                             params={"language": "en", "type": "occupation", "text": keyword, "limit": 4},
                             headers={"User-Agent": web_tools.UA}, timeout=15)
            r.raise_for_status()
            items = r.json().get("_embedded", {}).get("results", [])
        except Exception as exc:  # noqa: BLE001
            return f"ESCO lookup failed ({exc}). Use web_search instead; do not invent occupation data."
        if not items:
            return "No ESCO occupations matched."
        out = []
        for it in items:
            out.append(f"- {it.get('title')} | uri: {it.get('uri')} | {(it.get('description') or {}).get('en', {}).get('literal', '') if isinstance(it.get('description'), dict) else ''}"[:600])
        return "SOURCE: ESCO (ec.europa.eu/esco)\n" + "\n".join(out)


class _RegIn(BaseModel):
    field: str = Field(..., description="Field of study/profession, e.g. 'medicine', 'engineering', 'pharmacy', 'law', 'computer science'.")


class AccreditationRegistryTool(BaseTool):
    name: str = "accreditation_registry"
    description: str = ("Return which Pakistani regulator(s) accredit/recognise a field, what to check, and the official site hint. "
                        "This only says WHERE to verify - you must still confirm a specific institution/program on the official site.")
    args_schema: Type[BaseModel] = _RegIn

    def _run(self, field: str) -> str:
        data = json.loads((KNOWLEDGE / "regulators.json").read_text(encoding="utf-8"))
        f = field.lower()
        hits = [d for d in data["fields"] if any(k in f or f in k for k in d["keywords"])]
        return json.dumps({"matches": hits or [], "always_check": data["always_check"], "note": data["note"]})


class _EligIn(BaseModel):
    student_json: str = Field(..., description="JSON of the student's facts (hssc_pct, ssc_pct, stream, subjects, age, nationality...).")
    requirements_json: str = Field(..., description="JSON of the program's requirements that you extracted from an official page: "
                                   "{min_hssc_pct, min_ssc_pct, accepted_streams[], required_subjects[], min_age, max_age, nationality[], entry_test}.")


class EligibilityCheckTool(BaseTool):
    name: str = "eligibility_check"
    description: str = ("Deterministic eligibility check. Compares student facts with requirements and returns per-criterion "
                        "PASS / FAIL / UNKNOWN. Never guess a missing requirement - leave it out and it is reported UNKNOWN.")
    args_schema: Type[BaseModel] = _EligIn

    def _run(self, student_json: str, requirements_json: str) -> str:
        try:
            s, r = json.loads(student_json), json.loads(requirements_json)
        except json.JSONDecodeError as exc:
            return f"Invalid JSON: {exc}"
        return json.dumps(elig.check(s, r))


class _ScholIn(BaseModel):
    name_or_topic: str = Field(..., description="Scholarship name or topic, e.g. 'HEC need based scholarship' or 'Chevening Pakistan'.")


class ScholarshipRegistryTool(BaseTool):
    name: str = "scholarship_registry"
    description: str = ("Starter list of well-known scholarship programmes with their OFFICIAL portal domain (no amounts or "
                        "deadlines stored - those change every cycle). Use it to know where to look, then read the official page.")
    args_schema: Type[BaseModel] = _ScholIn

    def _run(self, name_or_topic: str) -> str:
        data = json.loads((KNOWLEDGE / "scholarships_seed.json").read_text(encoding="utf-8"))
        q = name_or_topic.lower()
        hits = [d for d in data["programmes"] if q in json.dumps(d).lower() or any(w in json.dumps(d).lower() for w in q.split() if len(w) > 3)]
        return json.dumps({"programmes": hits[:8] or data["programmes"][:8], "note": data["note"]})
