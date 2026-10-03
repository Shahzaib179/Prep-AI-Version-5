"""Merit Aggregate & Admission Advisor - three CrewAI agents + a deterministic calculator.

  Agent 1  Formula Verifier     - confirms the weights on the official prospectus (CrewAI + free web tools)
  Calculator (pure Python)      - the aggregate itself; the LLM never does arithmetic
  Agent 2  Merit History Scout  - finds previous-year closing merits; every row must pass evidence verification
  Agent 3  Admission Advisor    - explains the Python-computed Safe/Target/Reach table, no new numbers allowed
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Callable

from crew_agents.evidence import trust_tier, verify_evidence, TIER_LABEL
from crew_agents.merit import formulas as F
from crew_agents.merit import history
from crew_agents.tools import web_tools
from crew_agents.util import clip, extract_json


@dataclass
class MeritReport:
    profile: F.FormulaProfile
    primary: F.MeritResult
    variants: list[tuple[str, F.MeritResult]]
    formula_check: dict[str, Any] = field(default_factory=dict)
    recommendations: list[history.Recommendation] = field(default_factory=list)
    rejected_rows: list[dict[str, Any]] = field(default_factory=list)
    advice: str = ""
    log: list[str] = field(default_factory=list)


def _tools():
    from crew_agents.tools.crew_tools import (FreeWebSearchTool, ReadWebpageTool, FormulaRegistryTool,
                                              OfficialSourceSearchTool)
    return FreeWebSearchTool(), ReadWebpageTool(), FormulaRegistryTool(), OfficialSourceSearchTool()


# ------------------------------------------------------------------ Agent 1
def verify_formula(profile: F.FormulaProfile, year: int, model: str | None, on_wait=None) -> dict[str, Any]:
    from crew_agents.runner import run_stage
    search, read, registry, official = _tools()
    raw = run_stage(
        role="Admission Formula Verifier",
        goal=f"Confirm the merit-aggregate weights used by {profile.institution} for {profile.exam} admissions ({year}).",
        backstory="You verify admission formulas against official prospectuses and admission guides. Secondary websites often "
                  "disagree, so you trust only the regulator / institution and clearly flag conflicts.",
        description=(f"Stored hypothesis: {json.dumps({'weights_percent': profile.weights, 'test_total': profile.test_total})}\n"
                     f"Official page hint: {profile.official_hint}\n"
                     f"Find the CURRENT official weights for {profile.exam} ({profile.institution}), year {year}. Search with "
                     "official_source_search first, then open the best page with read_webpage using keywords like "
                     "'weightage, merit, aggregate, formula'. Return JSON: "
                     '{"found": true|false, "weights_percent": {"ssc": n, "hssc": n, "test": n}, "test_total": n|null, '
                     '"source_url": "...", "quote": "verbatim text containing the numbers", "conflicts": "short note or empty"}'),
        expected_output="A single JSON object as specified.", tools=[search, read, registry, official],
        model=model, max_iter=6, on_wait=on_wait)
    data = extract_json(raw) or {}
    check: dict[str, Any] = {"found": bool(data.get("found")), "source_url": data.get("source_url", ""),
                             "quote": data.get("quote", ""), "conflicts": data.get("conflicts", ""),
                             "verified": False, "matches_profile": None, "proposed_weights": None}
    page = web_tools.PAGE_CACHE.get(check["source_url"], "")
    w = data.get("weights_percent") or {}
    try:
        nums = [float(w[k]) for k in (F.SSC, F.HSSC, F.TEST)]
    except (KeyError, TypeError, ValueError):
        nums = []
    # Verified = quote really on the fetched page AND every non-zero weight number appears in the quote.
    if nums and page and round(sum(nums)) == 100 and \
            all(verify_evidence(n, check["quote"], page) for n in nums if n):
        check["verified"] = True
        check["proposed_weights"] = dict(zip((F.SSC, F.HSSC, F.TEST), nums))
        check["matches_profile"] = all(abs(a - b) < 0.01 for a, b in zip(nums, (profile.weights[F.SSC], profile.weights[F.HSSC], profile.weights[F.TEST])))
        check["tier"] = TIER_LABEL[trust_tier(check["source_url"])]
    return check


# ------------------------------------------------------------------ Agent 2
def research_history(profile: F.FormulaProfile, institution: str, program: str, years: list[int], model: str | None,
                     on_wait=None) -> tuple[list[dict], list[dict]]:
    from crew_agents.runner import run_stage
    search, read, _, official = _tools()
    raw = run_stage(
        role="Admission Merit History Scout",
        goal=f"Find previous-year closing merit (last admitted aggregate) for {program} at {institution} ({profile.exam}).",
        backstory="You collect closing-merit data from official merit lists and prospectuses. You report only numbers you can quote.",
        description=(f"Find closing/last-admitted merit for {program} at {institution}, years {years}, open merit seats unless the page says otherwise. "
                     "Use web_search / official_source_search, then read_webpage with keywords like 'closing merit, last merit, merit list, "
                     f"{program}'. Return JSON: "
                     '{"rows": [{"year": 2024, "institution": "...", "program": "...", "category": "open merit", "closing_merit": 93.5, '
                     '"source_url": "...", "quote": "verbatim text containing the number"}]}. '
                     "Return at most 6 rows. If nothing verifiable is found return {\"rows\": []}."),
        expected_output="A single JSON object as specified.", tools=[search, read, official],
        model=model, max_iter=7, on_wait=on_wait)
    data = extract_json(raw) or {}
    good, bad = [], []
    for r in (data.get("rows") or []):
        page = web_tools.PAGE_CACHE.get(r.get("source_url", ""), "")
        ok = verify_evidence(r.get("closing_merit"), r.get("quote", ""), page)
        r["verified"] = ok
        r["tier"] = trust_tier(r.get("source_url", ""))
        (good if ok else bad).append(r)
    return good, bad


# ------------------------------------------------------------------ Agent 3
def write_advice(report: MeritReport, student_goal: str, model: str | None, on_wait=None) -> str:
    from crew_agents.runner import run_stage
    table = [{"institution": r.institution, "program": r.program, "year": r.year, "closing_merit": r.closing_merit,
              "margin": r.margin, "band": r.band, "verified": r.verified, "trend": r.trend_note} for r in report.recommendations[:12]]
    return run_stage(
        role="Admission Advisor",
        goal="Explain the student's admission position honestly using ONLY the computed data supplied.",
        backstory="You are a careful admissions counsellor in Pakistan. You never add numbers that are not in the data, "
                  "and you remind students that last year's cut-off does not guarantee this year's result.",
        description=(f"Student goal: {student_goal or 'not stated'}\nExam: {report.profile.exam}\n"
                     f"Aggregate (primary formula): {report.primary.aggregate:.2f}%\n"
                     f"Range across formula variants: {F.sensitivity_spread(report.variants)}\n"
                     f"Eligibility issues: {report.primary.eligibility_issues or 'none flagged'}\n"
                     f"Formula verification: {json.dumps({k: report.formula_check.get(k) for k in ('found','verified','matches_profile','conflicts')})}\n"
                     f"Recommendation table (Safe/Target/Reach computed by code): {json.dumps(table)}\n\n"
                     "Write 150-250 words: where the student stands, which options are Safe/Target/Reach and why, "
                     "what uncertainty remains (formula status, unverified rows, year-to-year change), and 2-3 concrete next steps "
                     "(e.g. check official prospectus, backup options, improve test score if a retake exists). Do not invent figures."),
        expected_output="Plain-text advice, no JSON.", tools=[], model=model, max_iter=2, max_tokens=900, on_wait=on_wait)


# ------------------------------------------------------------------ orchestrator
def run_merit(*, profile_key: str, data: F.MeritInput, student_id: str, institution: str = "", program: str = "",
              student_goal: str = "", year: int | None = None, verify_online: bool = True, history_online: bool = True, use_verified_weights: bool = False,
              custom_weights: dict[str, float] | None = None, model: str | None = None, use_ai_advice: bool = True,
              progress: Callable[[str], None] | None = None) -> MeritReport:
    say = progress or (lambda m: None)
    year = year or date.today().year
    profile = F.PROFILES[profile_key]
    weights = custom_weights
    report = MeritReport(profile, None, [], log=[])  # type: ignore[arg-type]
    web_tools.reset_cache()
    history.init_tables()

    # 1) formula verification (CrewAI)
    if verify_online:
        say("Agent 1 - Formula Verifier: checking the official prospectus...")
        try:
            report.formula_check = verify_formula(profile, year, model, on_wait=say)
        except Exception as exc:  # noqa: BLE001
            report.formula_check = {"found": False, "error": str(exc)}
            report.log.append(f"Formula verifier failed: {exc}")
        fc = report.formula_check
        if use_verified_weights and fc.get("verified") and fc.get("proposed_weights"):
            weights = fc["proposed_weights"]
            report.log.append("Using weights verified on the official source.")

    # 2) deterministic calculation
    report.primary = F.calculate(profile, data, weights)
    report.variants = F.calculate_all_variants(profile, data)
    if weights:
        report.variants[0] = ("Weights used (custom/verified)", report.primary)

    # 3) closing-merit history
    rows = history.get_rows(student_id, profile.exam)
    if history_online and institution.strip():
        say("Agent 2 - Merit History Scout: searching previous-year closing merits...")
        try:
            default_prog = "MBBS" if profile.exam in ("MDCAT", "NUMS") else "BS Engineering/Computing"
            good, bad = research_history(profile, institution, program.strip() or default_prog,
                                         [year - 1, year - 2, year - 3], model, on_wait=say)
            if good:
                history.add_rows(student_id, [{**g, "exam": profile.exam, "verified": True} for g in good], origin="agent")
            report.rejected_rows = bad
            rows = history.get_rows(student_id, profile.exam)
        except Exception as exc:  # noqa: BLE001
            report.log.append(f"Merit history scout failed: {exc}")
    if institution.strip():
        rows = [r for r in rows if institution.strip().lower() in r["institution"].lower()
                or r["institution"].lower() in institution.strip().lower()] or rows
    report.recommendations = history.recommend(report.primary.aggregate, rows)

    # 4) narrative (CrewAI) - optional, calculation above already stands on its own
    if use_ai_advice:
        say("Agent 3 - Admission Advisor: writing guidance...")
        try:
            report.advice = write_advice(report, student_goal, model, on_wait=say)
        except Exception as exc:  # noqa: BLE001
            report.log.append(f"Advisor failed: {exc}")
    return report
