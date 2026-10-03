"""One small CrewAI crew per stage (1 agent + 1 task). Small crews keep each Groq request inside token limits
and let the UI show progress / retry a single stage instead of re-running everything."""
from __future__ import annotations

from typing import Callable, Sequence

from crew_agents.llm import make_llm
from crew_agents.util import with_backoff

GROUNDING_RULES = (
    "RULES: (1) Use ONLY facts returned by your tools. (2) For every number, date, requirement or recognition claim, "
    "give the source URL and a short VERBATIM quote copied from a page you opened with read_webpage. "
    "(3) If you cannot find or verify something, output null / 'NOT FOUND' - never estimate or recall from memory. "
    "(4) Prefer official sites (regulators, the institution) over blogs, calculators and aggregators. "
    "(5) Return exactly the JSON requested, no extra commentary."
)


def run_stage(*, role: str, goal: str, backstory: str, description: str, expected_output: str,
              tools: Sequence, model: str | None = None, max_iter: int = 5, max_tokens: int = 2200,
              on_wait: Callable[[str], None] | None = None) -> str:
    from crewai import Agent, Crew, Process, Task

    def _go() -> str:
        agent = Agent(role=role, goal=goal, backstory=backstory + "\n" + GROUNDING_RULES, tools=list(tools),
                      llm=make_llm(model, max_tokens=max_tokens), allow_delegation=False, verbose=False,
                      max_iter=max_iter, respect_context_window=True)
        task = Task(description=description, expected_output=expected_output, agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False, max_rpm=20, cache=True)
        out = crew.kickoff()
        return str(getattr(out, "raw", out) or "")

    return with_backoff(_go, on_wait=on_wait)
