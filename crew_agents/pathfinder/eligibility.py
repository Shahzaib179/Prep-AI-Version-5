"""Deterministic eligibility checker. Missing information -> UNKNOWN, never a guess."""
from __future__ import annotations

from typing import Any

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"


def _num(v: Any) -> float | None:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _norm_list(v: Any) -> list[str]:
    if not v:
        return []
    if isinstance(v, str):
        v = [v]
    return [str(x).strip().lower() for x in v if str(x).strip()]


def check(student: dict[str, Any], req: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def add(criterion: str, required: Any, have: Any, status: str) -> None:
        rows.append({"criterion": criterion, "required": required, "student": have, "status": status})

    for key, label in (("min_hssc_pct", "Intermediate (HSSC/FSc/A-Level) %"), ("min_ssc_pct", "Matric (SSC/O-Level) %")):
        need = _num(req.get(key))
        if need is None:
            continue
        have = _num(student.get(key.replace("min_", "")))
        add(label, f">= {need:g}", have, UNKNOWN if have is None else PASS if have >= need else FAIL)

    streams = _norm_list(req.get("accepted_streams"))
    if streams:
        have = str(student.get("stream", "")).strip().lower()
        add("Intermediate stream", streams, have or None, UNKNOWN if not have else PASS if any(s in have or have in s for s in streams) else FAIL)

    subs = _norm_list(req.get("required_subjects"))
    if subs:
        mine = _norm_list(student.get("subjects"))
        if not mine:
            add("Required subjects", subs, None, UNKNOWN)
        else:
            missing = [s for s in subs if not any(s in m or m in s for m in mine)]
            add("Required subjects", subs, mine, PASS if not missing else FAIL)

    age = _num(student.get("age"))
    for key, label, cmp in (("min_age", "Minimum age", lambda a, n: a >= n), ("max_age", "Maximum age", lambda a, n: a <= n)):
        need = _num(req.get(key))
        if need is not None:
            add(label, need, age, UNKNOWN if age is None else PASS if cmp(age, need) else FAIL)

    nat = _norm_list(req.get("nationality"))
    if nat:
        have = str(student.get("nationality", "")).strip().lower()
        add("Nationality", nat, have or None, UNKNOWN if not have else PASS if any(n in have or have in n for n in nat) else FAIL)

    if req.get("entry_test"):
        taken = _norm_list(student.get("tests_taken"))
        et = str(req["entry_test"]).lower()
        add("Entry test", req["entry_test"], taken or None, UNKNOWN if not taken else PASS if any(et in t or t in et for t in taken) else FAIL)

    statuses = {r["status"] for r in rows}
    if not rows:
        overall = "NO REQUIREMENTS PROVIDED"
    elif FAIL in statuses:
        overall = "NOT ELIGIBLE"
    elif UNKNOWN in statuses:
        overall = "INCOMPLETE - some facts missing"
    else:
        overall = "ELIGIBLE (on the criteria checked)"
    return {"overall": overall, "criteria": rows,
            "disclaimer": "Only the criteria listed were checked. Seat quotas, domicile, medical fitness and "
                          "deadlines must be confirmed in the official prospectus."}
