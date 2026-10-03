"""Path Finder - five CrewAI agents run as separate, resumable stages.

  1 Career Explorer           careers that fit the student's interests/skills (ESCO + web)
  2 Program & University Scout  real programs with official links
  3 Eligibility & Accreditation Verifier  deterministic eligibility + recognition evidence (HEC/PMDC/PEC...)
  4 Scholarship Scout          funding options with official pages
  5 Roadmap Planner            personalised timeline built ONLY from the verified stages

Grounding contract: every factual claim carries {url, quote}; Python re-checks the quote against the
fetched page. Claims that fail are shown as "Unverified" and never used for the eligibility/accreditation verdict.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from crew_agents.evidence import TIER_LABEL, trust_tier
from crew_agents.pathfinder import eligibility
from crew_agents.tools import web_tools
from crew_agents.util import clip, extract_json

STAGES = ["careers", "programs", "verification", "scholarships", "roadmap"]


@dataclass
class StudentProfile:
    name: str = ""
    education: str = "Intermediate (FSc Pre-Medical)"
    stream: str = ""
    subjects: list[str] = field(default_factory=list)
    ssc_pct: float | None = None
    hssc_pct: float | None = None
    tests_taken: list[str] = field(default_factory=list)
    age: int | None = None
    nationality: str = "Pakistani"
    domicile: str = ""
    interests: str = ""
    strengths: str = ""
    goals: str = ""
    budget_pkr_per_year: int | None = None
    willing_to_relocate: bool = True
    study_abroad: bool = False
    need_scholarship: bool = False

    def facts(self) -> dict[str, Any]:
        return {"ssc_pct": self.ssc_pct, "hssc_pct": self.hssc_pct, "stream": self.stream, "subjects": self.subjects,
                "age": self.age, "nationality": self.nationality, "tests_taken": self.tests_taken}

    def brief(self) -> str:
        d = {k: v for k, v in self.__dict__.items() if v not in (None, "", [], False) and k != "name"}
        return json.dumps(d, ensure_ascii=False)


def _tools(names: list[str]):
    from crew_agents.tools import crew_tools as t
    reg = {"search": t.FreeWebSearchTool, "read": t.ReadWebpageTool, "official": t.OfficialSourceSearchTool,
           "career": t.CareerDataTool, "accred": t.AccreditationRegistryTool, "elig": t.EligibilityCheckTool,
           "schol": t.ScholarshipRegistryTool}
    return [reg[n]() for n in names]


def _check_items(items: list[dict], url_key: str = "source_url", quote_key: str = "quote") -> list[dict]:
    """Tag each item with evidence status by re-checking the quote against the cached page text."""
    from crew_agents.evidence import _norm
    out = []
    for it in items or []:
        url, quote = it.get(url_key, ""), it.get(quote_key, "")
        page = web_tools.PAGE_CACHE.get(url, "")
        it["evidence"] = "verified" if (quote and page and _norm(quote) in _norm(page)) else "unverified"
        it["source_tier"] = TIER_LABEL[trust_tier(url)] if url else "none"
        out.append(it)
    return out


def run_pathfinder_stage(stage: str, profile: StudentProfile, state: dict[str, Any], model: str | None = None,
                         on_wait: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run one stage; returns the stage result dict. `state` holds earlier stage results."""
    from crew_agents.runner import run_stage
    brief = profile.brief()

    if stage == "careers":
        raw = run_stage(
            role="Career Explorer", goal="Suggest 4-5 realistic career paths that fit this student.",
            backstory="You are an evidence-based career counsellor for Pakistani students. You explain what the job is, "
                      "the typical education route, and why it fits - and you do not quote salaries unless a source states them.",
            description=(f"Student profile: {brief}\nUse career_data_esco for occupation definitions and web_search/read_webpage for "
                         "Pakistan-specific routes (e.g. required degrees, licensing). Return JSON: "
                         '{"careers": [{"title": "...", "why_fit": "link to the student\'s interests/strengths", "typical_route": "degree -> licence -> ...", '
                         '"fields_of_study": ["..."], "source_url": "...", "quote": "verbatim text from the page"}]}'),
            expected_output="A single JSON object.", tools=_tools(["career", "search", "read"]), model=model, max_iter=6, on_wait=on_wait)
        data = extract_json(raw) or {}
        return {"careers": _check_items(data.get("careers", []))}

    if stage == "programs":
        careers = [c.get("title") for c in state.get("careers", {}).get("careers", [])][:5]
        raw = run_stage(
            role="Program & University Scout", goal="Find real degree programs and institutions matching the careers and student constraints.",
            backstory="You find programs on university and regulator websites. You never list a program you could not find on a page.",
            description=(f"Student profile: {brief}\nTarget careers: {careers}\nFind 4-6 programs (Pakistan"
                         f"{' and abroad' if profile.study_abroad else ''}). Use official_source_search (HEC, UET, NUST, FAST...) or web_search, "
                         "then read_webpage. Respect budget/relocation constraints when stated. Return JSON: "
                         '{"programs": [{"institution": "...", "program": "...", "career_link": "...", "city": "...", "admission_test": "..." or null, '
                         '"requirements": {"min_hssc_pct": n|null, "min_ssc_pct": n|null, "accepted_streams": [], "required_subjects": [], "entry_test": "..."|null}, '
                         '"source_url": "...", "quote": "verbatim text containing the requirement or program name"}]}'),
            expected_output="A single JSON object.", tools=_tools(["official", "search", "read"]), model=model, max_iter=7, on_wait=on_wait)
        data = extract_json(raw) or {}
        return {"programs": _check_items(data.get("programs", []))}

    if stage == "verification":
        results = []
        for prog in state.get("programs", {}).get("programs", [])[:6]:
            # deterministic eligibility from the requirements the scout extracted (only if its evidence verified)
            req = prog.get("requirements") or {}
            req = {k: v for k, v in req.items() if v not in (None, "", [])}
            if prog.get("evidence") != "verified":
                elig = {"overall": "NOT CHECKED - requirement source unverified", "criteria": [], "disclaimer": ""}
            else:
                elig = eligibility.check(profile.facts(), req)
            raw = run_stage(
                role="Accreditation Verifier", goal="Verify recognition/accreditation of one institution and program from official sources.",
                backstory="You check HEC recognition and professional-council accreditation (PMDC, PEC, NCEAC, NBEAC, PNMC...). "
                          "You report 'NOT FOUND' rather than assuming a degree is recognised.",
                description=(f"Institution: {prog.get('institution')}\nProgram: {prog.get('program')}\n"
                             "1) call accreditation_registry for the field to learn which regulator applies; "
                             "2) use official_source_search on that regulator (and HEC) and read_webpage to confirm the institution/program appears. "
                             "Return JSON: "
                             '{"hec_recognised": "yes|no|not_found", "council": "...", "council_status": "accredited|not_found|unclear", '
                             '"source_url": "...", "quote": "verbatim text naming the institution/program"}'),
                expected_output="A single JSON object.", tools=_tools(["accred", "official", "read"]), model=model, max_iter=5, max_tokens=1500, on_wait=on_wait)
            acc = extract_json(raw) or {}
            acc = _check_items([acc])[0] if acc else {"evidence": "unverified", "hec_recognised": "not_found"}
            if acc.get("evidence") != "verified":     # claim not backed by a fetched page -> downgrade
                acc["hec_recognised"] = "not_found" if acc.get("hec_recognised") == "yes" else acc.get("hec_recognised", "not_found")
                acc["council_status"] = "not_found" if acc.get("council_status") == "accredited" else acc.get("council_status", "not_found")
                acc["downgraded"] = "Claim could not be matched to a fetched official page, so it is shown as not verified."
            results.append({"institution": prog.get("institution"), "program": prog.get("program"),
                            "eligibility": elig, "accreditation": acc})
        return {"verification": results}

    if stage == "scholarships":
        progs = [f"{p.get('institution')} - {p.get('program')}" for p in state.get("programs", {}).get("programs", [])][:4]
        raw = run_stage(
            role="Scholarship Scout", goal="Find legitimate scholarships and financial aid the student may qualify for.",
            backstory="You only list scholarships found on official provider pages and you never invent amounts or deadlines.",
            description=(f"Student profile: {brief}\nShortlisted programs: {progs}\nUse scholarship_registry to know where to look, then "
                         "official_source_search / web_search + read_webpage to open the CURRENT official page. Return JSON: "
                         '{"scholarships": [{"name": "...", "provider": "...", "level": "...", "eligibility_summary": "...", "deadline": "..." or null, '
                         '"covers": "..." or null, "source_url": "...", "quote": "verbatim text from the official page"}]}. '
                         "Deadline/covers must be null unless quoted from the page. Max 6."),
            expected_output="A single JSON object.", tools=_tools(["schol", "official", "search", "read"]), model=model, max_iter=7, on_wait=on_wait)
        data = extract_json(raw) or {}
        return {"scholarships": _check_items(data.get("scholarships", []))}

    if stage == "roadmap":
        ok_programs = [v for v in state.get("verification", {}).get("verification", [])]
        compact = {
            "careers": [c.get("title") for c in state.get("careers", {}).get("careers", []) if c.get("evidence") == "verified"],
            "programs": [{"institution": v["institution"], "program": v["program"], "eligibility": v["eligibility"]["overall"],
                          "hec": v["accreditation"].get("hec_recognised"), "council": v["accreditation"].get("council_status")} for v in ok_programs],
            "scholarships": [{"name": s.get("name"), "deadline": s.get("deadline")} for s in state.get("scholarships", {}).get("scholarships", [])
                             if s.get("evidence") == "verified"],
        }
        raw = run_stage(
            role="Roadmap Planner", goal="Create a personalised step-by-step roadmap from verified information only.",
            backstory="You turn verified findings into a practical timeline. You label every item that still needs checking.",
            description=(f"Student profile: {brief}\nVerified findings: {json.dumps(compact)}\n"
                         "Build a roadmap with phases: Now (0-1 month), Next 3 months, 6-12 months, Long term. Each step: action, why, "
                         "'verify_at' (which official site to confirm). Include entry-test prep and backup options. Items using programs whose eligibility is "
                         "not ELIGIBLE or whose accreditation is not_found must be marked 'needs verification'. Do NOT add deadlines, fees or merit numbers "
                         'not present in the findings. Return JSON: {"summary": "...", "phases": [{"phase": "Now", "steps": [{"action": "...", "why": "...", '
                         '"verify_at": "...", "status": "ready|needs verification"}]}], "backup_plan": "..."}'),
            expected_output="A single JSON object.", tools=[], model=model, max_iter=2, max_tokens=1800, on_wait=on_wait)
        return {"roadmap": extract_json(raw) or {"summary": clip(raw, 1200), "phases": []}}

    raise ValueError(f"Unknown stage {stage}")
