"""CrewAI LLM factory. Reuses the app's existing Groq key + model setting (no new provider needed)."""
from __future__ import annotations

import os

from config import DEFAULT_GROQ_MODEL, GROQ_MODELS, get_secret


def crewai_available() -> tuple[bool, str]:
    try:
        import crewai  # noqa: F401
        return True, getattr(crewai, "__version__", "installed")
    except Exception as exc:  # noqa: BLE001
        return False, f"CrewAI is not importable: {exc}"


def make_llm(model: str | None = None, temperature: float = 0.1, max_tokens: int = 2200):
    """Return a crewai.LLM that routes to Groq (via LiteLLM). Low temperature for factual work."""
    from crewai import LLM

    key = (get_secret("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to Streamlit secrets.")
    os.environ["GROQ_API_KEY"] = key  # LiteLLM reads this variable
    chosen = model if model in GROQ_MODELS else DEFAULT_GROQ_MODEL
    kwargs = dict(model=f"groq/{chosen}", api_key=key, temperature=temperature, max_tokens=max_tokens)
    try:
        return LLM(**kwargs, reasoning_effort="low")   # gpt-oss supports it; keeps token use down
    except TypeError:
        return LLM(**kwargs)
