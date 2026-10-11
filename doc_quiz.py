"""Quizzes from a tutor's own documents (PDF, DOCX, TXT, MD).

Steps: validate and read the files -> split into chunks -> plan several generation rounds so the quiz covers the whole document
(or the chapter asked for) instead of the first few pages -> generate round by round without repeats.
Pure logic: embedding/search and the AI call are injected, so this is tested without models or internet.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import progress
from documents import chunk_pages, extract_document, file_hash

MAX_FILES = 5
MAX_FILE_MB = 15
MAX_CHUNKS = 700           # keeps embedding time and memory reasonable on a free host (about 400 pages of text)
PER_ROUND = 10             # questions requested per AI call
CONTEXT_CHARS = 11000


def fingerprint(files: list[tuple[str, bytes]]) -> str:
    return file_hash("|".join(f"{n}:{file_hash(d)}" for n, d in sorted(files, key=lambda x: x[0])).encode())


def prepare(files: list[tuple[str, bytes]]) -> dict[str, Any]:
    """Read and chunk the uploaded files. Raises ValueError with a message a tutor can act on."""
    if not files:
        raise ValueError("Upload at least one document.")
    if len(files) > MAX_FILES:
        raise ValueError(f"Upload at most {MAX_FILES} documents at a time.")
    pages: list[dict[str, Any]] = []
    info: list[dict[str, Any]] = []
    warnings: list[str] = []
    for name, data in files:
        name = os.path.basename(name)
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            raise ValueError(f"{name} is larger than {MAX_FILE_MB} MB. Split it into smaller parts.")
        try:
            extracted = extract_document(data, name, name)
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - corrupt or password-protected files
            raise ValueError(f"{name} could not be read ({type(exc).__name__}). Is it damaged or password-protected?") from None
        chars = sum(len(p["text"]) for p in extracted)
        if chars < 200:
            raise ValueError(f"{name} contains almost no readable text. Scanned documents (images of pages) are not supported; upload a text-based PDF or DOCX.")
        if name.lower().endswith(".pdf") and len(extracted) < 3 and chars < 1500:
            warnings.append(f"{name}: very little text was found; the quiz may be limited.")
        pages.extend(extracted)
        info.append({"filename": name, "pages": len(extracted), "characters": chars})
    chunks = chunk_pages(pages)
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(f"These documents are too long ({len(chunks)} sections; the limit is {MAX_CHUNKS}). Upload one chapter or a shorter file.")
    return {"chunks": chunks, "info": info, "warnings": warnings}


def _format(chunks: list[dict[str, Any]], limit: int = CONTEXT_CHARS) -> str:
    out, used = [], 0
    for c in chunks:
        block = f"[{c.get('filename', 'Document')} | page {c.get('page') or 'N/A'}]\n{c['text']}"
        if used + len(block) > limit and out:
            break
        out.append(block)
        used += len(block)
    return "\n\n".join(out)


def plan_rounds(chunks: list[dict[str, Any]], count: int, topic: str = "", search: Callable[[str, int], list[dict[str, Any]]] | None = None,
                per_round: int = PER_ROUND) -> list[dict[str, Any]]:
    """Split ``count`` questions into rounds; each round gets DIFFERENT passages so the quiz spreads over the material."""
    if not chunks or count <= 0:
        return []
    rounds_n = max(1, -(-count // per_round))
    base, extra = divmod(count, rounds_n)
    sizes = [base + (1 if i < extra else 0) for i in range(rounds_n)]
    if topic.strip() and search:
        hits = search(topic.strip(), min(len(chunks), max(6 * rounds_n, 8))) or chunks
        pools = [hits[i::rounds_n] or hits for i in range(rounds_n)]
    else:
        ordered = sorted(chunks, key=lambda c: c.get("chunk_id", 0))
        size = -(-len(ordered) // rounds_n)
        pools = [ordered[i * size:(i + 1) * size] or ordered for i in range(rounds_n)]
    plan = []
    for n, pool in zip(sizes, pools):
        # keep the whole slice within the prompt budget by sampling it evenly
        step = max(1, len(pool) * 900 // CONTEXT_CHARS)
        picked = pool[::step] if len(pool) * 900 > CONTEXT_CHARS else pool
        plan.append({"n": n, "context": _format(picked), "chunks": picked})
    return plan


def generate(plan: list[dict[str, Any]], make_round: Callable[[int, str], list[dict[str, Any]]], count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the rounds. ``make_round(n, context)`` returns validated MCQs. Returns (questions, source chunks used). A failed round is skipped."""
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    errors = 0
    for rnd in plan:
        if len(questions) >= count:
            break
        try:
            batch = make_round(min(rnd["n"] + 2, PER_ROUND + 2), rnd["context"]) or []   # ask slightly more to survive duplicates/invalid items
        except Exception:  # noqa: BLE001 - one bad round must not lose the others
            errors += 1
            continue
        added = 0
        for q in batch:
            h = progress.qhash(q.get("question", ""))
            if h in seen or len(questions) >= count or added >= rnd["n"] + 1:
                continue
            seen.add(h)
            questions.append(q)
            added += 1
        if added:
            sources.extend(rnd["chunks"][:3])
    if not questions and errors:
        raise RuntimeError("The AI could not generate questions right now. Please try again in a minute.")
    return questions[:count], sources
