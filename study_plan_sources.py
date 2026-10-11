"""Study plans built from CHAPTERS of the Prep AI database or of the student's own documents.

Flow: chapter names -> hybrid search (meaning + keywords) for each chapter in the FAISS database / the student's document index
      -> the best passages become the ONLY material the AI may plan from -> JSON plan -> validated and fitted to the daily time budget.
If the AI is unavailable or returns something unusable, a deterministic plan is built from the same chapters, so the student always gets a plan.
Pure logic: the search function and the AI call are injected (tests need neither a model nor the internet).
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Callable

MAX_CHAPTERS = 8
PER_CHAPTER = 5            # passages read per chapter
CONTEXT_CHARS = 9000       # prompt budget for all passages together
MIN_SCORE = 0.22           # below this a chapter is reported as "not found in the material" instead of planned from noise
MIN_DAYS, MAX_DAYS = 1, 14
MAX_SESSIONS_PER_DAY = 6
ACTIVITIES = ("Read and make notes", "Practice MCQs", "Revise and self-test")

Search = Callable[[str, int], list[dict[str, Any]]]
Ai = Callable[[str], Any]


def parse_chapters(text: str, limit: int = MAX_CHAPTERS) -> list[str]:
    """'Genetics; Cell cycle\nBiomolecules' -> ['Genetics', 'Cell cycle', 'Biomolecules'] (de-duplicated, trimmed)."""
    seen, out = set(), []
    for part in re.split(r"[\n;,]+", text or ""):
        name = re.sub(r"\s+", " ", part).strip(" -•*\t.")[:80]
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out[:limit]


def plan_days(requested: int, exam_date: str | None, today: date | None = None) -> int:
    """Plan length: what the student asked for, shortened if the exam is sooner (never below 1 day, never above 14)."""
    n = max(MIN_DAYS, min(MAX_DAYS, int(requested)))
    try:
        left = (date.fromisoformat(exam_date or "") - (today or date.today())).days
    except ValueError:
        return n
    return max(MIN_DAYS, min(n, left)) if left >= 1 else n


def _label(p: dict[str, Any]) -> str:
    page = p.get("page")
    return f"{p.get('filename') or p.get('title') or 'Material'}" + (f" p.{page}" if page else "")


def gather(chapters: list[str], search: Search, per_chapter: int = PER_CHAPTER, min_score: float = MIN_SCORE,
           char_budget: int = CONTEXT_CHARS) -> dict[str, Any]:
    """Search the material once per chapter. Returns the passages, a prompt-ready context, and the chapters that were not found."""
    blocks, missing = [], []
    share = max(400, char_budget // max(1, len(chapters)))
    for ch in chapters:
        hits = [h for h in (search(ch, per_chapter) or []) if h.get("text")]
        if not hits or max(float(h.get("hybrid_score", 1.0)) for h in hits) < min_score:
            missing.append(ch)
            continue
        text, used, passages = [], 0, []
        for h in hits:
            body = " ".join(h["text"].split())
            if used + len(body) > share and passages:
                break
            body = body[:max(200, share - used)]
            passages.append({"label": _label(h), "text": body})
            text.append(f"[{_label(h)}] {body}")
            used += len(body)
        blocks.append({"chapter": ch, "passages": passages, "context": "\n".join(text)})
    context = "\n\n".join(f"### CHAPTER: {b['chapter']}\n{b['context']}" for b in blocks)
    labels = sorted({p["label"] for b in blocks for p in b["passages"]})
    return {"blocks": blocks, "missing": missing, "context": context, "labels": labels}


def build_prompt(material: dict[str, Any], days: int, hours: float, exam_name: str, exam_date: str, subject: str, goals: str, weak_text: str = "") -> str:
    chapters = ", ".join(b["chapter"] for b in material["blocks"])
    return f"""Create a practical {days}-day study plan for a student, built ONLY from the study material below.
Exam: {exam_name}   Exam date: {exam_date}   Subject: {subject}
Hours per day: {hours}   Chapters to cover: {chapters}
Student goals: {goals or 'none given'}
Weak topics from the student's quiz history (give them extra revision if they match a chapter): {weak_text or 'none yet'}

Rules:
- Split each chapter into concrete sub-topics that really appear in the material, and plan those (not generic advice).
- Cover every chapter. Early days: learn and make notes. Middle days: MCQ practice. Last day(s): revision and a self-test.
- Each day's session minutes add up to about {int(hours * 60)}.
- For every session add "source": the label in square brackets of the passage it comes from (copy it exactly), or "" if none.
Return JSON: {{"days": [{{"day": "Day 1", "sessions": [{{"subject": "{subject}", "topic": "Chapter - sub-topic", "minutes": 45, "activity": "Read and make notes", "source": "label"}}]}}]}}

MATERIAL:
{material['context']}"""


def normalize_plan(data: Any, days: int, hours: float, labels: set[str] | list[str], subject: str) -> list[dict[str, Any]]:
    """Accept what the AI returned, keep only usable sessions, fit each day to the time budget, drop unknown source labels."""
    raw = data if isinstance(data, list) else (data or {}).get("days", []) if isinstance(data, dict) else []
    budget = max(10, int(round(hours * 60)))
    known = set(labels)
    plan = []
    for n, d in enumerate(raw[:days], start=1):
        if not isinstance(d, dict):
            continue
        sessions = []
        for s in (d.get("sessions") or [])[:MAX_SESSIONS_PER_DAY]:
            if not isinstance(s, dict) or not str(s.get("topic", "")).strip():
                continue
            try:
                minutes = max(5, int(float(s.get("minutes", 0))))
            except (TypeError, ValueError):
                minutes = 20
            src = str(s.get("source", "") or "").strip()
            sessions.append({"subject": str(s.get("subject") or subject)[:60], "topic": str(s["topic"]).strip()[:140], "minutes": minutes,
                             "activity": str(s.get("activity") or ACTIVITIES[0]).strip()[:140], "source": src if src in known else ""})
        if not sessions:
            continue
        total = sum(s["minutes"] for s in sessions)
        for s in sessions:                                   # scale to the student's daily time, in 5-minute steps
            s["minutes"] = max(5, int(round(s["minutes"] * budget / total / 5)) * 5)
        plan.append({"day": str(d.get("day") or f"Day {n}")[:30], "sessions": sessions})
    return plan


def fallback_plan(chapters: list[str], days: int, hours: float, subject: str, labels_by_chapter: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """No-AI plan: learn every chapter, then practise, then revise; the last day is a mixed self-test."""
    labels_by_chapter = labels_by_chapter or {}
    budget = max(10, int(round(hours * 60)))
    learn_days = max(1, round(days * 0.5)) if days > 1 else 1
    plan = []
    for n in range(1, days + 1):
        if n <= learn_days:
            picks = [chapters[i] for i in range(len(chapters)) if i % learn_days == n - 1] or [chapters[(n - 1) % len(chapters)]]
            acts = [ACTIVITIES[0]] * len(picks)
        elif n < days or days == 1:
            picks = [chapters[(n - learn_days - 1 + k) % len(chapters)] for k in range(min(2, len(chapters)))]
            acts = [ACTIVITIES[1]] * len(picks)
        else:
            picks, acts = list(chapters), [ACTIVITIES[2]] * len(chapters)
        picks, acts = picks[:MAX_SESSIONS_PER_DAY], acts[:MAX_SESSIONS_PER_DAY]
        each = max(5, int(round(budget / len(picks) / 5)) * 5)
        plan.append({"day": f"Day {n}", "sessions": [{"subject": subject, "topic": c, "minutes": each, "activity": a, "source": labels_by_chapter.get(c, "")} for c, a in zip(picks, acts)]})
    return plan


def make_plan(chapters: list[str], material: dict[str, Any], days: int, hours: float, exam_name: str, exam_date: str, subject: str,
              goals: str, weak_text: str, ai: Ai) -> tuple[list[dict[str, Any]], bool]:
    """Returns (plan, used_ai). Chapters not found in the material are excluded; if nothing was found the caller gets a ValueError."""
    if not material["blocks"]:
        raise ValueError("None of those chapters were found in the material. Check the spelling or try the chapter's usual name.")
    found = [b["chapter"] for b in material["blocks"]]
    first_label = {b["chapter"]: b["passages"][0]["label"] for b in material["blocks"]}
    try:
        plan = normalize_plan(ai(build_prompt(material, days, hours, exam_name, exam_date, subject, goals, weak_text)), days, hours, material["labels"], subject)
    except Exception:  # noqa: BLE001 - AI down, rate limit, invalid JSON
        plan = []
    if len(plan) >= max(1, days // 2 + days % 2):           # at least half of the days came back usable
        return plan, True
    return fallback_plan(found, days, hours, subject, first_label), False
