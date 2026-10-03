# Prep AI V5 — CrewAI agents, tools and platforms

## 1. Design rules (why it is built this way)
1. **Code does the maths, the LLM does the reading.** Merit aggregates and eligibility checks are plain Python (`formulas.py`, `eligibility.py`) with unit tests.
2. **Evidence or it did not happen.** Every researched number/claim must come with `{source_url, verbatim quote}`. After the agent answers, Python re-checks that the quote is really in the page that was fetched (`PAGE_CACHE`) and that the number is inside the quote. Failures are shown as *unverified* or dropped (`evidence.py`).
3. **Formulas are data with a status.** Secondary websites disagree (ECAT is quoted as 17/50/33, 25/45/30, 10/40/50…; NUMS as 10/40/50 or 0/50/50). Each profile is `provisional` until confirmed; variants are shown side by side; weights can be overridden.
4. **Small crews, one stage each.** Groq free tiers have tight tokens-per-minute limits, so each agent is its own 1-agent crew with capped tool output and automatic wait-and-retry on HTTP 429.
5. **Graceful degradation.** No CrewAI / no Groq key / agent failure → the calculator and your CSV merit list still work.

## 2. Agents

### Merit Advisor (`crew_agents/merit/`)
| Agent | Type | Job | Tools |
|---|---|---|---|
| Formula Verifier | CrewAI agent | Confirms weights on the official prospectus; flags conflicts | `official_source_search`, `web_search`, `read_webpage`, `formula_registry` |
| Merit Calculator | Python | Aggregate for MDCAT, NUMS, ECAT, NUST NET, FAST, NTS-generic, custom; variants; eligibility floors | — |
| Merit History Scout | CrewAI agent | Finds previous-year closing merits; each row must pass quote verification | `official_source_search`, `web_search`, `read_webpage` |
| Admission Recommender | Python | Safe / Target / Reach / Unlikely from margin vs last cut-off + year-to-year trend | SQLite `closing_merits` |
| Admission Advisor | CrewAI agent | Explains the computed table; may not add numbers | none |

### Path Finder (`crew_agents/pathfinder/`)
| Agent | Job | Tools |
|---|---|---|
| Career Explorer | 4–5 careers that fit interests/strengths, typical route | `career_data_esco`, `web_search`, `read_webpage` |
| Program & University Scout | Real programs + requirements from official pages | `official_source_search`, `web_search`, `read_webpage` |
| Eligibility & Accreditation Verifier | Deterministic eligibility (PASS/FAIL/UNKNOWN) + HEC/PMDC/PEC/NCEAC/NBEAC… recognition with evidence | `accreditation_registry`, `official_source_search`, `read_webpage`, `eligibility_check` (Python) |
| Scholarship Scout | Funding options from official provider pages; deadline/amount only if quoted | `scholarship_registry`, `official_source_search`, `web_search`, `read_webpage` |
| Roadmap Planner | Now / 3 months / 6–12 months / long-term plan from verified items only | none |

## 3. Tool details
| Tool | Class | Backed by | Cost / key | Notes |
|---|---|---|---|---|
| `web_search` | `FreeWebSearchTool` | `ddgs` (DuckDuckGo) | free, no key | Ranks official/institutional domains first; throttled to ~1 req/s; snippets are leads, not proof |
| `read_webpage` | `ReadWebpageTool` | `requests` + `beautifulsoup4` + `pypdf` | free | HTML and text-PDF; returns verbatim excerpts around keywords; SSRF guard (public hosts only), 6 MB / 20 s limits. Scanned PDFs need OCR (not included) |
| `official_source_search` | `OfficialSourceSearchTool` | DuckDuckGo `site:` filter | free | HEC, PMDC, PEC, PNMC, PCP, PVMC, PCATP, PBC, NUST, NUMS, UET, FAST, any `.gov.pk` |
| `formula_registry` | `FormulaRegistryTool` | `merit/formulas.py` | local | Hypothesis for the verifier |
| `career_data_esco` | `CareerDataTool` | ESCO REST API (European Commission) | free, no key | Occupation definitions only — no Pakistani salaries |
| `accreditation_registry` | `AccreditationRegistryTool` | `data/knowledge/regulators.json` | local | Says *where* to verify, never asserts a program is recognised |
| `eligibility_check` | `EligibilityCheckTool` | `pathfinder/eligibility.py` | local | Missing facts → UNKNOWN |
| `scholarship_registry` | `ScholarshipRegistryTool` | `data/knowledge/scholarships_seed.json` | local | Official domains only; no amounts/deadlines stored |

## 4. Open-source libraries and platforms
| Need | Choice | Why |
|---|---|---|
| Agent framework | **CrewAI** (already pinned) | Role-based agents, tools, sequential crews |
| LLM routing | **LiteLLM** via CrewAI (`groq/openai/gpt-oss-120b` or `-20b`) | Reuses your Groq key + Settings page |
| LLM host | **Groq** free/developer tier | Fast; watch tokens-per-minute |
| Web search | **ddgs** | No key |
| Scraping | **requests + BeautifulSoup4 + pypdf** | Light, no browser needed |
| UI | **Streamlit** (existing) | Two new sidebar pages |
| Storage | **SQLite** (existing) + new tables `closing_merits`, `merit_runs` | No new service |
| Reports | **ReportLab** (existing) | PDF export |

### Optional upgrades (not installed)
| Upgrade | When |
|---|---|
| **Playwright** (open source) | JavaScript-only admission portals |
| **pytesseract + Tesseract OCR / OCRmyPDF** | Scanned merit-list PDFs |
| **trafilatura** | Cleaner article extraction |
| **SearXNG** (self-hosted) | If DuckDuckGo rate-limits you; point `web_tools.search` at it |
| **O*NET Web Services** (free key) | US-centric career detail |
| **College Scorecard API** (free key) | US universities |
| **Serper / Tavily free tiers** | Better search quality; need a key |

## 5. Known limits
* Closing merits are public only if institutions publish them; many appear only in PDF images or on aggregator sites (tier 3, flagged as unofficial).
* The stored formulas are **provisional** until you confirm them; MDCAT 10/40/50 is consistent across sources, NUST 75/15/10 is consistent, ECAT and NUMS conflict.
* Past cut-offs never guarantee a future result; the app states this.
* The live web tools could not be exercised in the build sandbox (no network); they are covered by unit tests with mocked pages. Run one real query per page before sharing the app.
