"""Anti-hallucination helpers: every number/fact an agent reports must be traceable to fetched text."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict


@dataclass
class Evidence:
    value: str            # the claim, e.g. "89.4"
    source_url: str
    quote: str            # short verbatim snippet from the source page
    verified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def number_in_text(number: float | str, text: str, tolerance: float = 0.005) -> bool:
    """True if `number` appears in `text` (as printed, or within `tolerance` of any decimal in it)."""
    t = text or ""
    try:
        target = float(str(number).replace(",", "").replace("%", ""))
    except ValueError:
        return _norm(str(number)) in _norm(t)
    for m in re.finditer(r"\d[\d,]*\.?\d*", t):
        try:
            if abs(float(m.group().replace(",", "")) - target) <= tolerance:
                return True
        except ValueError:
            continue
    return False


def verify_evidence(value: str | float, quote: str, page_text: str) -> bool:
    """A claim is verified only if the quote is really in the page AND the value is in the quote."""
    if not quote or not page_text:
        return False
    if _norm(quote) not in _norm(page_text):
        return False
    return number_in_text(value, quote)


TRUST_TIERS = (
    (1, (".gov.pk", "hec.gov.pk", "pmdc.pk", "pec.org.pk", "pnc.org.pk", "pbc.org.pk", "numspak.edu.pk", "nts.org.pk", "pnmc.gov.pk", "pcp.org.pk", "pvmc.gov.pk", "pcatp.org.pk", "pakistanbarcouncil.org", "usefp.org", "chevening.org", "daad.de")),
    (1, (".edu.pk", "uhs.edu.pk", "nust.edu.pk", "uet.edu.pk", "admission.uet.edu.pk", "nu.edu.pk")),
    (2, (".gov", ".edu", ".ac.uk", ".int", "unesco.org", "onetonline.org", "esco.ec.europa.eu")),
)


def trust_tier(url: str) -> int:
    """1 = official/institutional, 2 = other authoritative, 3 = everything else (blogs, calculators, forums)."""
    u = (url or "").lower()
    for tier, hints in TRUST_TIERS:
        if any(h in u for h in hints):
            return tier
    return 3


TIER_LABEL = {1: "Official/institutional", 2: "Authoritative", 3: "Unofficial (verify)"}
