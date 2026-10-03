"""Deterministic merit-aggregate engine.

The LLM never does arithmetic here. Every formula is DATA (a FormulaProfile) and every
calculation is plain Python, so results are reproducible and unit-tested.

IMPORTANT: Pakistani admission formulas change between years / programs and secondary
websites frequently disagree (e.g. ECAT is quoted as 17/50/33, 25/45/30 and 10/40/50 by
different sites). Each profile therefore carries:
  * status   - "verified-official" | "provisional" (needs checking against the current prospectus)
  * sources  - where the weights came from
  * variants - alternative weight sets that are reported alongside the main result
The Merit Crew's Formula Verifier agent tries to confirm a profile against the official
prospectus at run time; users can also override the weights in the UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# Component keys used everywhere
SSC, HSSC, TEST = "ssc", "hssc", "test"


@dataclass(frozen=True)
class Variant:
    label: str
    weights: dict[str, float]          # component -> weight in percent (must sum to 100)
    note: str = ""


@dataclass(frozen=True)
class FormulaProfile:
    key: str
    exam: str
    institution: str
    weights: dict[str, float]
    test_total: float                  # maximum marks of the entry test
    status: str = "provisional"
    effective: str = ""
    sources: tuple[str, ...] = ()
    official_hint: str = ""            # official page the verifier agent should check first
    min_hssc_pct: float | None = None  # eligibility floors (checked, never silently applied)
    min_ssc_pct: float | None = None
    min_test_pct: float | None = None
    variants: tuple[Variant, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        total = round(sum(self.weights.values()), 6)
        if total != 100:
            raise ValueError(f"{self.key}: weights must sum to 100, got {total}")


PROFILES: dict[str, FormulaProfile] = {}


def _add(p: FormulaProfile) -> None:
    PROFILES[p.key] = p


_add(FormulaProfile(
    key="mdcat_pmdc", exam="MDCAT", institution="PMDC standard (public & private medical/dental colleges)",
    weights={SSC: 10, HSSC: 40, TEST: 50}, test_total=180, status="provisional", effective="MDCAT 2025/2026",
    sources=("https://www.pmdc.pk (admission regulations / MDCAT notification)",),
    official_hint="pmdc.pk MDCAT admission policy",
    min_test_pct=55.0,
    notes="Passing marks: 55% MBBS, 50% BDS (BDS: pass min_test_pct manually). Several secondary sites quote this "
          "uniformly; confirm the provincial prospectus (UHS, DUHS, KMU...) for your seat category.",
))
_add(FormulaProfile(
    key="nums_mbbs", exam="NUMS", institution="NUMS-affiliated medical colleges (AMC, CMH...)",
    weights={SSC: 10, HSSC: 40, TEST: 50}, test_total=200, status="provisional", effective="2025/2026",
    sources=("https://numspak.edu.pk (admission prospectus)",), official_hint="numspak.edu.pk admission prospectus",
    min_hssc_pct=60.0, min_test_pct=55.0,
    variants=(Variant("NUMS 50% FSc + 50% test (no Matric)", {SSC: 0, HSSC: 50, TEST: 50},
                      "Quoted by some 2026 secondary sources; verify in the official prospectus."),),
    notes="Secondary sources DISAGREE on whether Matric counts for NUMS. Both results are shown.",
))
_add(FormulaProfile(
    key="ecat_uet_lahore", exam="ECAT", institution="UET Lahore (FSc Pre-Engineering stream)",
    weights={SSC: 17, HSSC: 50, TEST: 33}, test_total=400, status="provisional", effective="Fall 2026 (per CampusAxis citing the UET guide)",
    sources=("https://admission.uet.edu.pk/uploads/downloads/admission-guide.pdf",),
    official_hint="admission.uet.edu.pk admission guide",
    variants=(
        Variant("Legacy 25/45/30", {SSC: 25, HSSC: 45, TEST: 30}, "Older UET formula still quoted by some sites."),
    ),
    notes="Some programs (e.g. Architecture) have their own weights, and one source mentions a Spring-2026 split that includes interview marks (not modelled here). Verify program-specific rules.",
))
_add(FormulaProfile(
    key="nust_net", exam="NUST NET", institution="NUST (all undergraduate faculties, NET route)",
    weights={SSC: 10, HSSC: 15, TEST: 75}, test_total=200, status="provisional", effective="2025-26 / 2026",
    sources=("https://nust.edu.pk/admissions/undergraduates/",), official_hint="nust.edu.pk merit generation criteria",
    min_hssc_pct=60.0, min_ssc_pct=60.0,
    variants=(Variant("A-Level result awaited: 25% O-Level + 75% NET", {SSC: 25, HSSC: 0, TEST: 75}, "Applies while A-Level result is pending."),),
    notes="Best NET attempt is used. Entry test total is 200 for NET.",
))
_add(FormulaProfile(
    key="fast_nu", exam="FAST-NUCES", institution="FAST-NUCES (admission test route)",
    weights={SSC: 10, HSSC: 40, TEST: 50}, test_total=100, status="provisional", effective="unverified",
    sources=(), official_hint="nu.edu.pk admissions merit criteria",
    notes="PLACEHOLDER weights - verify before use. Use the weight override in the UI with the official numbers.",
))
_add(FormulaProfile(
    key="nts_nat_generic", exam="NTS NAT", institution="Generic NTS-based university (set weights from the university's prospectus)",
    weights={SSC: 10, HSSC: 40, TEST: 50}, test_total=100, status="provisional", effective="varies",
    sources=(), official_hint="the university's own admission prospectus",
    notes="NTS only conducts the test (NAT-IE/IM/ICS...). Merit weights are set by each university, so this profile is a "
          "starting point only. Override the weights using the official prospectus.",
))
_add(FormulaProfile(
    key="custom", exam="Custom", institution="Any institution (you supply weights)",
    weights={SSC: 10, HSSC: 40, TEST: 50}, test_total=100, status="user-defined", effective="-",
    notes="Use for GIKI, PIEAS, COMSATS, IBA, LUMS... after reading their official criteria.",
))


@dataclass
class MeritInput:
    ssc_obtained: float
    ssc_total: float
    hssc_obtained: float
    hssc_total: float
    test_obtained: float
    test_total: float | None = None    # defaults to the profile's test_total


@dataclass
class MeritResult:
    profile_key: str
    weights: dict[str, float]
    percentages: dict[str, float]
    contributions: dict[str, float]
    aggregate: float
    warnings: list[str] = field(default_factory=list)
    eligibility_issues: list[str] = field(default_factory=list)


class MeritError(ValueError):
    pass


def _pct(obtained: float, total: float, label: str) -> float:
    if total is None or total <= 0:
        raise MeritError(f"{label}: total marks must be greater than 0.")
    if obtained < 0:
        raise MeritError(f"{label}: obtained marks cannot be negative.")
    if obtained > total:
        raise MeritError(f"{label}: obtained marks ({obtained}) exceed total marks ({total}).")
    return obtained / total * 100.0


def calculate(profile: FormulaProfile, data: MeritInput, weights: dict[str, float] | None = None) -> MeritResult:
    w = dict(weights or profile.weights)
    if round(sum(w.values()), 6) != 100:
        raise MeritError(f"Weights must sum to 100 (got {sum(w.values())}).")
    test_total = data.test_total or profile.test_total
    pcts = {
        SSC: _pct(data.ssc_obtained, data.ssc_total, "Matric/SSC"),
        HSSC: _pct(data.hssc_obtained, data.hssc_total, "FSc/HSSC"),
        TEST: _pct(data.test_obtained, test_total, f"{profile.exam} test"),
    }
    contrib = {k: pcts[k] * w.get(k, 0.0) / 100.0 for k in pcts}
    res = MeritResult(profile.key, w, pcts, contrib, round(sum(contrib.values()), 4))

    if profile.min_ssc_pct is not None and pcts[SSC] < profile.min_ssc_pct:
        res.eligibility_issues.append(f"Matric {pcts[SSC]:.2f}% is below the {profile.min_ssc_pct:.0f}% minimum.")
    if profile.min_hssc_pct is not None and pcts[HSSC] < profile.min_hssc_pct:
        res.eligibility_issues.append(f"FSc {pcts[HSSC]:.2f}% is below the {profile.min_hssc_pct:.0f}% minimum.")
    if profile.min_test_pct is not None and pcts[TEST] < profile.min_test_pct:
        res.eligibility_issues.append(f"Test score {pcts[TEST]:.2f}% is below the {profile.min_test_pct:.0f}% minimum.")
    if profile.status != "verified-official":
        res.warnings.append(f"Formula status: {profile.status}. Confirm weights against the current official prospectus.")
    return res


def calculate_all_variants(profile: FormulaProfile, data: MeritInput) -> list[tuple[str, MeritResult]]:
    out = [("Primary formula", calculate(profile, data))]
    for v in profile.variants:
        out.append((v.label, calculate(profile, data, v.weights)))
    return out


def sensitivity_spread(results: Iterable[tuple[str, MeritResult]]) -> tuple[float, float]:
    vals = [r.aggregate for _, r in results]
    return (min(vals), max(vals))
