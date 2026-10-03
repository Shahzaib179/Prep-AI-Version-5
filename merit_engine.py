"""Merit Aggregate Agent: deterministic calculation + history-based recommendations.
Pure Python (no Streamlit) so it is easy to test. All numbers come from the formula
or from imported official merit data; nothing is invented."""
from __future__ import annotations

from typing import Any

# weights are percentages. verified=False -> UI warns the student to confirm with the prospectus.
FORMULAS: dict[str, dict[str, Any]] = {
    "MDCAT (PMDC standard: 10/40/50)": {"w": (10, 40, 50), "test_total": 180, "verified": True,
        "note": "Standard MBBS/BDS formula reported consistently by multiple sources. Some bodies (e.g. NUMS, KMU, federal colleges) publish variants."},
    "NUMS": {"w": (0, 50, 50), "test_total": 200, "verified": False,
        "note": "Sources disagree on NUMS weights. Confirm in the current NUMS prospectus."},
    "ECAT (UET)": {"w": (10, 40, 50), "test_total": 400, "verified": False,
        "note": "Typical UET weighting; confirm in the current UET prospectus."},
    "NUST NET": {"w": (10, 15, 75), "test_total": 200, "verified": False,
        "note": "Typical NUST weighting; confirm in the current NUST prospectus."},
    "NTS NAT / University entry test": {"w": (10, 40, 50), "test_total": 100, "verified": False,
        "note": "NTS only conducts the test. Each university sets its own weights."},
    "Custom": {"w": (10, 40, 50), "test_total": 100, "verified": False, "note": "Enter the weights from your target university."},
}


def calculate_aggregate(matric: float, matric_total: float, inter: float, inter_total: float,
                        test: float, test_total: float, weights: tuple[float, float, float]) -> dict[str, Any]:
    errors: list[str] = []
    for label, got, tot in (("Matric", matric, matric_total), ("Intermediate", inter, inter_total), ("Entry test", test, test_total)):
        if tot <= 0:
            errors.append(f"{label} total marks must be above 0.")
        elif got < 0 or got > tot:
            errors.append(f"{label} marks must be between 0 and {tot:g}.")
    if abs(sum(weights) - 100) > 0.01:
        errors.append(f"Weights must add up to 100 (currently {sum(weights):g}).")
    if errors:
        return {"ok": False, "errors": errors}
    parts = {k: round(g / t * w, 4) for k, g, t, w in
             (("matric", matric, matric_total, weights[0]), ("inter", inter, inter_total, weights[1]), ("test", test, test_total, weights[2]))}
    return {"ok": True, "aggregate": round(sum(parts.values()), 3), "parts": parts,
            "percentages": {"matric": round(matric / matric_total * 100, 2), "inter": round(inter / inter_total * 100, 2),
                            "test": round(test / test_total * 100, 2)}}


def recommend(aggregate: float, history: list[dict[str, Any]], years: int = 3, include_unlikely: bool = False) -> list[dict[str, Any]]:
    """history rows: institution, program, category, year, closing_merit, source_url.
    Compares the student's aggregate with the last `years` closing merits of each (institution, program, category)."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in history:
        try:
            r = {**r, "year": int(r["year"]), "closing_merit": float(r["closing_merit"])}
        except (KeyError, TypeError, ValueError):
            continue
        groups.setdefault((r["institution"], r["program"], r.get("category") or "Open merit"), []).append(r)
    out = []
    for (inst, prog, cat), rows in groups.items():
        rows = sorted(rows, key=lambda x: x["year"], reverse=True)[:years]
        latest, highest = rows[0], max(x["closing_merit"] for x in rows)
        margin = aggregate - latest["closing_merit"]
        if aggregate >= highest + 1:
            band = "Safe"
        elif margin >= 0:
            band = "Target"
        elif margin >= -2:
            band = "Reach"
        else:
            band = "Unlikely"
        if band == "Unlikely" and not include_unlikely:
            continue
        trend = None
        if len(rows) >= 2:
            diff = rows[0]["closing_merit"] - rows[-1]["closing_merit"]
            trend = "rising" if diff > 0.5 else "falling" if diff < -0.5 else "stable"
        out.append({"institution": inst, "program": prog, "category": cat, "band": band, "your_aggregate": aggregate,
                    "latest_year": latest["year"], "latest_closing": latest["closing_merit"], "margin": round(margin, 2),
                    "years_used": [x["year"] for x in rows], "trend": trend, "source_url": latest.get("source_url", "")})
    order = {"Safe": 0, "Target": 1, "Reach": 2, "Unlikely": 3}
    return sorted(out, key=lambda x: (order[x["band"]], -x["latest_closing"]))
