from __future__ import annotations

import json
import re
import time
from typing import Any, Callable


def extract_json(text: str) -> Any:
    """Parse JSON from an LLM answer (handles ```json fences and surrounding prose). Returns None on failure."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = t.find(open_c), t.rfind(close_c)
        if s >= 0 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except json.JSONDecodeError:
                continue
    return None


def is_rate_limit(exc: Exception) -> bool:
    m = str(exc).lower()
    return "429" in m or "rate limit" in m or "rate_limit" in m or "too many requests" in m or "tokens per minute" in m


def with_backoff(fn: Callable[[], Any], tries: int = 3, wait: float = 25.0,
                 on_wait: Callable[[str], None] | None = None) -> Any:
    """Groq free tiers have tight tokens-per-minute limits; wait and retry instead of failing the whole run."""
    last: Exception | None = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not is_rate_limit(exc) or i == tries - 1:
                raise
            msg = f"Groq rate limit reached - waiting {int(wait * (i + 1))}s before retry {i + 2}/{tries}..."
            if on_wait:
                on_wait(msg)
            time.sleep(wait * (i + 1))
    raise last  # pragma: no cover


def clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + " …[truncated]"
