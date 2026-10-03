"""Free web research primitives (no paid API keys): DuckDuckGo via `ddgs`, plain HTTP fetch,
HTML/PDF text extraction. Pure functions so they can be tested without CrewAI.

Safety / cost controls
  * only http(s); private/loopback/link-local hosts are refused (SSRF guard)
  * hard size + time limits, results truncated for Groq's token limits
  * every fetched page is stored in PAGE_CACHE so claims can be verified afterwards (evidence.py)
"""
from __future__ import annotations

import io
import ipaddress
import re
import socket
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from crew_agents.evidence import TIER_LABEL, trust_tier

UA = "PrepAI-Research/1.0 (+student education tool; polite, low-rate)"
MAX_BYTES = 6_000_000
TIMEOUT = 20

PAGE_CACHE: dict[str, str] = {}      # url -> full extracted text (this process)
_LAST_CALL = {"t": 0.0}


def reset_cache() -> None:
    PAGE_CACHE.clear()


def _throttle(min_gap: float = 1.0) -> None:
    gap = time.time() - _LAST_CALL["t"]
    if gap < min_gap:
        time.sleep(min_gap - gap)
    _LAST_CALL["t"] = time.time()


def _is_public_host(host: str) -> bool:
    try:
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True
    except Exception:
        return False


def search(query: str, max_results: int = 6) -> list[dict[str, str]]:
    """DuckDuckGo text search, official/institutional domains ranked first."""
    from ddgs import DDGS
    _throttle()
    rows: list[dict[str, str]] = []
    try:
        with DDGS() as d:
            for r in d.text(query, max_results=max_results, region="pk-en"):
                url = r.get("href", "")
                rows.append({"title": r.get("title", ""), "url": url, "snippet": r.get("body", ""),
                             "tier": str(trust_tier(url))})
    except Exception as exc:  # network / rate limit - report, never crash a crew
        return [{"title": "SEARCH_ERROR", "url": "", "snippet": f"Search failed: {exc}", "tier": "3"}]
    rows.sort(key=lambda x: int(x["tier"]))
    return rows


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "svg"]):
        tag.decompose()
    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr", "td", "th", "caption"]):
        if el.name in ("td", "th") and el.find_parent("tr"):
            continue
        if el.name == "tr":
            cells = [c.get_text(" ", strip=True) for c in el.find_all(["td", "th"])]
            txt = " | ".join(c for c in cells if c)
        else:
            txt = el.get_text(" ", strip=True)
        if txt:
            lines.append(txt)
    return "\n".join(lines)


def _pdf_to_text(data: bytes, max_pages: int = 40) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for i, page in enumerate(reader.pages[:max_pages]):
        parts.append(f"[page {i + 1}]\n" + (page.extract_text() or ""))
    return "\n".join(parts)


def fetch_page(url: str) -> tuple[str, str]:
    """Return (text, error). Text is cached in PAGE_CACHE under the URL."""
    if url in PAGE_CACHE:
        return PAGE_CACHE[url], ""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return "", "Only http(s) URLs are allowed."
    if not _is_public_host(p.hostname):
        return "", "Refused: host is not a public internet address."
    _throttle()
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT, stream=True)
        r.raise_for_status()
        data = r.raw.read(MAX_BYTES + 1, decode_content=True)
        if len(data) > MAX_BYTES:
            return "", "Page too large (limit 6 MB)."
        ctype = r.headers.get("content-type", "").lower()
        if "pdf" in ctype or url.lower().split("?")[0].endswith(".pdf"):
            text = _pdf_to_text(data)
        else:
            text = _html_to_text(data.decode(r.encoding or "utf-8", errors="replace"))
    except Exception as exc:
        return "", f"Fetch failed: {exc}"
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return "", "No readable text (possibly a scanned PDF or JavaScript-only page)."
    PAGE_CACHE[url] = text
    return text, ""


def find_in_text(text: str, keywords: list[str], window: int = 350, max_hits: int = 6) -> list[str]:
    """Return short windows around keyword hits - keeps LLM prompts small."""
    low = text.lower()
    hits: list[tuple[int, str]] = []
    used: list[tuple[int, int]] = []
    for kw in keywords:
        k = kw.lower().strip()
        if not k:
            continue
        for m in re.finditer(re.escape(k), low):
            s, e = max(0, m.start() - window), min(len(text), m.end() + window)
            if any(s < ue and e > us for us, ue in used):
                continue
            used.append((s, e))
            hits.append((s, text[s:e].replace("\n", " ")))
            if len(hits) >= max_hits:
                break
        if len(hits) >= max_hits:
            break
    hits.sort(key=lambda x: x[0])
    return [h[1] for h in hits]


def search_report(query: str, max_results: int = 6) -> str:
    rows = search(query, max_results)
    if not rows:
        return "No results."
    out = []
    for r in rows:
        out.append(f"- [{TIER_LABEL[int(r['tier'])]}] {r['title']}\n  URL: {r['url']}\n  {r['snippet'][:240]}")
    return "\n".join(out)


def read_report(url: str, keywords: str = "", max_chars: int = 3500) -> str:
    text, err = fetch_page(url)
    if err:
        return f"ERROR: {err}"
    kws = [k.strip() for k in re.split(r"[,;|]", keywords) if k.strip()]
    if kws:
        hits = find_in_text(text, kws)
        if hits:
            body = "\n---\n".join(hits)
            return f"SOURCE: {url} ({TIER_LABEL[trust_tier(url)]})\nMATCHING EXCERPTS (verbatim):\n{body[:max_chars]}"
        return f"SOURCE: {url}\nNo excerpt matched {kws}. First part of page:\n{text[:1200]}"
    return f"SOURCE: {url} ({TIER_LABEL[trust_tier(url)]})\n{text[:max_chars]}"
