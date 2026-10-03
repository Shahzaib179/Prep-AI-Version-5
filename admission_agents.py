from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool

from config import DEFAULT_GROQ_MODEL, get_secret
from db import save_agent_session
from groq_service import GroqServiceError
from web_search import search_web
from memory import LongTermMemory

TRUSTED_DOMAINS = (
    ".gov.pk", ".edu.pk", ".gov", ".edu", "hec.gov.pk", "pec.org.pk",
    "pmdc.pk", "nums.edu.pk", "nust.edu.pk", "uet.edu.pk", "nts.org.pk",
    "uhs.edu.pk", "pu.edu.pk", "comsats.edu.pk"
)


def _clean(s: Any) -> str:
    return str(s or "").strip()


def _domain(url: str) -> str:
    m = re.search(r"https?://([^/]+)", url or "")
    return m.group(1).lower() if m else ""


class AdmissionWebResearchTool(BaseTool):
    name: str = "admission_web_research"
    description: str = (
        "Search the free web for current admission rules, merit formulas, previous merit/cutoffs, "
        "eligibility, scholarships and accreditation. Prefer official university/government/regulator sources. "
        "Returns title, URL, snippet and whether the source is trusted."
    )

    def _run(self, query: str, max_results: int = 8) -> str:
        rows = search_web(query, max_results=max_results)
        rows = sorted(rows, key=lambda r: (bool(r.get("trusted")), _domain(r.get("url", "")) in TRUSTED_DOMAINS), reverse=True)
        return json.dumps(rows[:max_results], ensure_ascii=False)


class AdmissionPageExtractTool(BaseTool):
    name: str = "admission_page_extract"
    description: str = "Extract readable text from a public admission webpage URL using the free DDGS extractor. Use this to verify important facts from a search result."

    def _run(self, url: str) -> str:
        from ddgs import DDGS
        page = DDGS().extract(url, fmt="text_plain")
        content = _clean(page.get("content", ""))
        return content[:18000] if content else "No extractable content was returned."


class MeritCalculatorTool(BaseTool):
    name: str = "merit_calculator"
    description: str = (
        "Calculate a weighted admission aggregate. Input JSON must contain scores as percentages and weights as percentages, "
        "for example {'scores': {'test': 85, 'hssc': 90, 'ssc': 95}, 'weights': {'test': 50, 'hssc': 40, 'ssc': 10}}. "
        "The weights must total 100."
    )

    def _run(self, payload: str) -> str:
        try:
            data = json.loads(payload)
            scores = data["scores"]
            weights = data["weights"]
            total_weight = sum(float(v) for v in weights.values())
            if abs(total_weight - 100) > 0.01:
                return json.dumps({"error": f"Weights must total 100; received {total_weight:.2f}."})
            contributions = {}
            total = 0.0
            for key, weight in weights.items():
                if key not in scores:
                    return json.dumps({"error": f"Missing score for '{key}'."})
                score = float(scores[key])
                if not 0 <= score <= 100:
                    return json.dumps({"error": f"Score for '{key}' must be between 0 and 100."})
                contribution = score * float(weight) / 100
                contributions[key] = round(contribution, 4)
                total += contribution
            return json.dumps({"aggregate": round(total, 4), "contributions": contributions, "weights": weights, "scores": scores})
        except Exception as exc:
            return json.dumps({"error": f"Invalid calculator input: {exc}"})


@dataclass
class AgentResult:
    answer: str
    sources: list[dict[str, Any]]
    raw: str = ""


def _normalize_groq_model(model: str | None = None) -> str:
    """Return the exact Groq model identifier expected by the API.

    Older saved settings may contain ``gpt-oss-20b`` without the provider
    prefix. Groq expects the OpenAI-compatible model IDs below.
    """
    chosen = _clean(model or DEFAULT_GROQ_MODEL)
    aliases = {
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "openai/gpt-oss-20b": "openai/gpt-oss-20b",
        "groq/gpt-oss-20b": "openai/gpt-oss-20b",
        "groq/openai/gpt-oss-20b": "openai/gpt-oss-20b",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "openai/gpt-oss-120b": "openai/gpt-oss-120b",
        "groq/gpt-oss-120b": "openai/gpt-oss-120b",
        "groq/openai/gpt-oss-120b": "openai/gpt-oss-120b",
    }
    return aliases.get(chosen, chosen)


def _make_llm(model: str | None = None) -> LLM:
    key = _clean(get_secret("GROQ_API_KEY"))
    if not key:
        raise GroqServiceError("GROQ_API_KEY is missing. Add it to Streamlit Secrets.")
    chosen = _normalize_groq_model(model)
    return LLM(
        model=chosen,
        custom_openai=True,
        base_url="https://api.groq.com/openai/v1",
        api_key=key,
        temperature=0.15,
        max_tokens=5000,
    )


def _run_crew(agent: Agent, task_description: str) -> str:
    task = Task(description=task_description, expected_output="A grounded, student-friendly response with a Sources section containing the exact URLs used.", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return str(result.raw if hasattr(result, "raw") else result)


class MeritAggregateAgent:
    name = "Merit Aggregate Agent"

    def run(self, student_id: str, exam: str, institution: str, program: str, scores: dict[str, float], query_notes: str, model: str | None = None) -> AgentResult:
        llm = _make_llm(model)
        tools = [AdmissionWebResearchTool(), AdmissionPageExtractTool(), MeritCalculatorTool()]
        agent = Agent(
            role="Pakistan Admission Merit and Eligibility Analyst",
            goal="Calculate admission aggregates accurately and turn verified admission data into cautious, source-grounded recommendations.",
            backstory=(
                "You are an admission analyst. You never invent merit formulas, previous-year closing merits, eligibility rules, "
                "or scholarship facts. You research official sources first, cite URLs, and clearly label estimates. "
                "A previous-year closing merit is historical evidence, not a guaranteed cutoff for the next cycle."
            ),
            llm=llm,
            tools=tools,
            allow_delegation=False,
            verbose=False,
        )
        score_json = json.dumps(scores, ensure_ascii=False)
        task = f"""
Analyze this student's admission merit request.

Exam/test: {exam}
Institution: {institution or 'Not specified'}
Program: {program or 'Not specified'}
Student scores (percentages or marks as explicitly labeled): {score_json}
Extra notes: {query_notes or 'None'}

WORKFLOW — follow in this order:
1. Research the CURRENT official admission policy/formula for the named exam/institution/program. If the exam is institution-dependent (for example ECAT or NTS), do not assume one universal formula; identify the relevant institution or explain what is missing.
2. Research the MOST RECENT completed admission cycle's published merit/closing information. Prefer official merit lists, prospectuses, admission policy pages, or official university pages. If only third-party historical data exists, label it as secondary and do not present it as official.
3. Use the merit_calculator tool for the final arithmetic once the verified weights are known. Never calculate a weighted aggregate mentally when the tool can do it.
4. Check basic eligibility from official sources when available.
5. Give a recommendation using these labels: Likely competitive, Borderline/competitive, Unlikely based on historical evidence, or Cannot assess. Do not promise admission.
6. State every important assumption and every missing input.
7. End with a Sources section listing the exact URLs used and what each source supports.

Important grounding rule: if an official source does not support a fact, say “Not verified from an official source” instead of filling the gap from model knowledge.
"""
        raw = _run_crew(agent, task)
        save_agent_session(student_id, self.name, f"{exam} | {institution} | {program} | {score_json}", raw)
        LongTermMemory(student_id).add(f"Merit analysis for {exam}, {institution}, {program}: {raw[:5000]}", "merit_agent_response", institution, program, importance=0.8, confidence=0.9)
        return AgentResult(answer=raw, sources=[], raw=raw)


class PathFinderAgent:
    name = "Path Finder Agent"

    def run(self, student_id: str, education_level: str, background: str, marks: str, interests: str, preferred_fields: str, location: str, budget: str, goals: str, model: str | None = None) -> AgentResult:
        llm = _make_llm(model)
        tools = [AdmissionWebResearchTool(), AdmissionPageExtractTool()]
        agent = Agent(
            role="Pakistan Education and Career Path Finder",
            goal="Build evidence-grounded education and career pathways using verified university, regulator, scholarship and program information.",
            backstory=(
                "You are a cautious education counselor and research analyst. You never invent eligibility, accreditation, "
                "scholarship amounts, deadlines, fees, job prospects, or program availability. You verify important claims "
                "with official sources and clearly separate verified facts, historical evidence, and suggestions."
            ),
            llm=llm,
            tools=tools,
            allow_delegation=False,
            verbose=False,
        )
        task = f"""
Create a personalized education/career roadmap for this student.

Education level: {education_level}
Academic background: {background}
Marks/scores: {marks}
Interests/strengths: {interests}
Preferred fields: {preferred_fields}
Preferred location: {location or 'Pakistan / flexible'}
Budget/financial constraints: {budget or 'Not specified'}
Goals: {goals or 'Not specified'}

Research requirements:
1. Identify 3–6 realistic career directions that match the student's background and interests.
2. For each direction, identify relevant Pakistani university programs and entry routes.
3. Verify eligibility using official university sources and/or HEC/regulatory sources.
4. Check HEC recognition for institutions/campuses. For regulated fields, check the relevant professional accreditation body where applicable (for example PEC for engineering). Do not call a program “accredited” unless the relevant regulator supports that claim.
5. Research scholarships/financial-aid opportunities from official university, HEC, government, or scholarship-provider sources. If none is verified, say so.
6. Provide a practical roadmap: next 30 days, next 3–6 months, admission/application stage, and first-year development.
7. Rank options by fit, eligibility confidence, affordability evidence, and career alignment. Do not fabricate a salary or job-demand statistic.
8. Explicitly list “What still needs verification” for anything that changes frequently (fees, deadlines, seats, closing merit, scholarship dates, accreditation status).
9. End with a Sources section containing exact URLs and the claim supported by each URL.

Grounding rule: every specific eligibility, accreditation, scholarship, deadline, fee, or admission requirement must be traceable to a source. If a fact cannot be verified, say “Not verified.”
"""
        raw = _run_crew(agent, task)
        save_agent_session(student_id, self.name, f"Path Finder | {education_level} | {background} | {preferred_fields}", raw)
        LongTermMemory(student_id).add(f"Path Finder analysis: {raw[:5000]}", "path_finder_response", preferred_fields, location, importance=0.8, confidence=0.9)
        return AgentResult(answer=raw, sources=[], raw=raw)
