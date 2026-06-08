"""
Yahoo headline red-flag screen for discovery eligibility.

Two-stage: regex keyword screen → Gemini gatekeeper on hits only.
Regex-only hits do not block; LLM BLOCK (risk >= 70) blocks discovery.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.services.financial.yahoo import headline_lookback_slot, latest_headlines

logger = logging.getLogger(__name__)

BLOCK_BANNER = "***** BLOCKED *****"

# News-catalyst advisors: skip generic Yahoo headline gate (they screen their own feed).
HEADLINE_SCREEN_SKIP_ADVISORS = frozenset({"Pharm", "Bizfeed", "FDA", "User"})

HEADLINE_SYMBOL_ALIASES: Dict[str, str] = {
    "ZI": "GTM",
}

RED_FLAG_PATTERNS: List[Tuple[str, str]] = [
    ("bankruptcy", r"\bbankruptcy\b|\bchapter\s+11\b"),
    ("investigation", r"\binvestigation\b|\bsec probe\b|\bsubpoena\b|\bDOJ\b|\blitigation\b"),
    ("fraud", r"\bfraud\b|\baccounting irregularit"),
    ("restatement", r"\brestate(s|ment)\b|\brestate(s|ment) financial"),
    ("delisting", r"\bdelist(ing|ed)?\b|\bnon[- ]compliance notice\b"),
    ("offering", r"\b(public|secondary|registered) offering\b|\bstock offering\b|\bat-the-market\b|\bATM offering\b"),
    ("dilution", r"\bdilut(e|ion|ive)\b"),
    ("clinical_failure", r"\bclinical hold\b|\btrial (fail|halt|suspend|discontinu)\b|\bphase\s+[123].*(fail|miss|halt)\b"),
    ("fda_negative", r"\bcomplete response letter\b|\bcrl\b|\bfda reject"),
    ("leadership_exit", r"\bceo (resign|depart|step|exit|terminate|oust)\b|\bcfo (resign|depart|exit)\b"),
    (
        "outlook_cut",
        r"\bcut(s|ting)?\b.*\boutlook\b|\boutlook cut\b|\bcutting \d{4} revenue outlook\b"
        r"|\breduced (full[- ]year )?outlook\b",
    ),
    (
        "guidance_cut",
        r"\bcut(s|ting)? guidance\b|\blower(s|ed|ing)? (full[- ]year )?guidance\b"
        r"|\bwithdraw(s|ing)? guidance\b|\bguidance cut\b",
    ),
    ("downgrade", r"\bdowngrade(s|d)?\b|\bcut(s|ting)? (to )?(hold|sell|underperform|neutral)\b"),
    ("going_concern", r"\bgoing concern\b"),
    ("default_covenant", r"\b(default|covenant breach|liquidity crunch)\b"),
    ("halt", r"\btrading halt\b|\bsec suspend"),
    ("layoffs", r"\blayoff(s)?\b|\bworkforce reduction\b|\bjob cuts\b"),
    ("lawsuit", r"\bclass action\b|\bshareholder lawsuit\b|\bsecurities lawsuit\b"),
    ("recall", r"\bproduct recall\b|\brecall(s)? (all|product)\b"),
    ("audit", r"\baudit (fail|issue|concern|qualification)\b|\bqualified opinion\b"),
    (
        "earnings_miss",
        r"\bearnings miss\b|\b(results|report|quarter) miss\b"
        r"|\bmiss(ed|es)? (estimates|consensus|expectations|revenue targets|analyst)\b"
        r"|\bwhen \w+ missed\b|\b\w+ missed and\b",
    ),
    (
        "share_plunge",
        r"\bshares? (are )?(plunging|tumbling|crashing|sink(ing)?|falling)\b"
        r"|\b(is )?down \d+\.?\d*%"
        r"|\bshed \d+\.?\d*%"
        r"|\bstock(s)? trade down\b",
    ),
    (
        "monetization_risk",
        r"\bmonetization (doubts|concerns|uncertainty)\b|\bdelay raises\b|\bcapex concern",
    ),
]

_COMPILED: List[Tuple[str, re.Pattern[str]]] = [
    (label, re.compile(pat, re.IGNORECASE)) for label, pat in RED_FLAG_PATTERNS
]

LLM_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

LLM_BLOCK_RISK_THRESHOLD = 70

LLM_SYSTEM = """You are a quantitative risk gatekeeper for short-term equity entries (1–3 session horizon).

Question: Does this news indicate elevated probability (>5%) of meaningful downside over the next 1-3 sessions?

Output STRICT JSON only:
{
  "action": "EXECUTE" | "BLOCK",
  "risk_score": 0-100,
  "severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "main_drivers": ["short bullet", "..."],
  "counterarguments": ["why risk may be overstated", "..."],
  "reasoning": "One clear sentence summary",
  "risk_type": "NOISE" | "EARNINGS_RISK" | "STRUCTURAL_REGIME_CHANGE"
}

Scoring guide:
- 80-100 CRITICAL/HIGH: guidance/outlook cuts with price collapse, fraud probes, offerings, halts, clinical failure.
- 60-80 HIGH/MEDIUM: downgrades + guidance cuts without catastrophe (e.g. ZoomInfo-style).
- 30-50 MEDIUM/LOW: product delays, monetization doubts, known concerns — concerning but not crash catalysts (Meta-style).
- 0-30 LOW: nuance headlines where scary words appear but context is mixed (e.g. earnings miss alongside strong AI revenue).

BLOCK when risk_score >= 70 or structural tail-risk is clear.
EXECUTE when risk_score < 70 and headlines are noise, already-priced concerns, or mixed/contextual."""


@dataclass
class KeywordHit:
    label: str
    headline: str
    match: str


@dataclass
class HeadlineScreenResult:
    symbol: str
    headlines: List[str] = field(default_factory=list)
    keyword_hits: List[KeywordHit] = field(default_factory=list)
    stage: str = "clear"
    allowed: bool = True
    llm: Optional[Dict[str, Any]] = None
    reason: str = ""
    lookback: str = ""
    resolved_symbol: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "resolved_symbol": self.resolved_symbol,
            "headlines": self.headlines,
            "keyword_hits": [
                {"label": h.label, "headline": h.headline, "match": h.match}
                for h in self.keyword_hits
            ],
            "stage": self.stage,
            "allowed": self.allowed,
            "llm": self.llm,
            "reason": self.reason,
            "lookback": self.lookback,
        }


def headline_screen_skip_for_advisor(python_class: str) -> bool:
    return (python_class or "") in HEADLINE_SCREEN_SKIP_ADVISORS


def fetch_headlines(
    symbol: str,
    *,
    limit: int = 5,
    max_age_hours: Optional[float] = None,
    max_age_days: Optional[int] = None,
) -> Tuple[str, List[str], str]:
    """Return (resolved_symbol, headlines, lookback_label)."""
    sym = (symbol or "").strip().upper()
    if max_age_hours is not None:
        lookback = f"{max_age_hours:g}h"
        kwargs: Dict[str, Any] = {"limit": limit, "max_age_hours": max_age_hours, "max_age_days": 0}
    elif max_age_days is not None:
        lookback = f"{max_age_days}d"
        kwargs = {"limit": limit, "max_age_days": max_age_days}
    else:
        max_age_hours, slot = headline_lookback_slot()
        lookback = f"{max_age_hours:g}h auto ({slot})"
        kwargs = {"limit": limit, "max_age_hours": max_age_hours, "max_age_days": 0}

    headlines = latest_headlines(sym, **kwargs)
    empty = (
        not headlines
        or (len(headlines) == 1 and headlines[0].startswith("No headlines"))
        or (len(headlines) == 1 and headlines[0].startswith("No recent"))
    )
    alias = HEADLINE_SYMBOL_ALIASES.get(sym)
    if empty and alias:
        alt = latest_headlines(alias, **kwargs)
        if alt and not (len(alt) == 1 and alt[0].startswith("No ")):
            return alias, alt, lookback
    return sym, headlines, lookback


def scan_keywords(headlines: Sequence[str]) -> List[KeywordHit]:
    hits: List[KeywordHit] = []
    seen: set[Tuple[str, str]] = set()
    for headline in headlines:
        if not headline or headline.startswith("No recent") or headline.startswith("Error"):
            continue
        for label, pattern in _COMPILED:
            m = pattern.search(headline)
            if m is None:
                continue
            key = (label, headline)
            if key in seen:
                continue
            seen.add(key)
            hits.append(KeywordHit(label=label, headline=headline, match=m.group(0)))
    return hits


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def run_llm_gatekeeper(
    symbol: str,
    headlines: Sequence[str],
    *,
    trigger: str = "",
    keyword_hits: Sequence[KeywordHit],
    requested_symbol: str = "",
) -> Optional[Dict[str, Any]]:
    api_key = os.environ.get("GEMINI_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("Headline screen LLM skip: GEMINI_KEY not set")
        return None

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("Headline screen LLM skip: google-genai not installed")
        return None

    user_payload = {
        "ticker": symbol,
        "requested_ticker": requested_symbol or symbol,
        "technical_trigger": trigger or "discovery entry signal",
        "live_headlines": list(headlines),
        "regex_flags": [{"label": h.label, "headline": h.headline} for h in keyword_hits],
    }
    prompt = f"{LLM_SYSTEM}\n\nInput:\n{json.dumps(user_payload, indent=2)}"

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(temperature=0.0, top_p=1.0)

    for model in LLM_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            parsed = _extract_json(response.text or "")
            if parsed:
                time.sleep(0.5)
                return parsed
            logger.warning("Headline screen LLM: could not parse JSON from %s", model)
        except Exception as exc:
            logger.warning("Headline screen LLM: %s failed (%s)", model, exc)
    return None


def apply_llm_verdict(result: HeadlineScreenResult, llm: Dict[str, Any]) -> None:
    result.llm = llm
    action = str(llm.get("action", "")).strip().upper()
    risk_score = llm.get("risk_score")
    try:
        if risk_score is not None and float(risk_score) >= LLM_BLOCK_RISK_THRESHOLD and action != "EXECUTE":
            action = "BLOCK"
    except (TypeError, ValueError):
        pass

    if action == "BLOCK":
        result.stage = "llm_block"
        result.allowed = False
        result.reason = str(llm.get("reasoning") or "LLM BLOCK")
    elif action == "EXECUTE":
        result.stage = "llm_allow"
        result.allowed = True
        result.reason = str(llm.get("reasoning") or "LLM EXECUTE")
    else:
        result.stage = "llm_error"
        result.allowed = True
        result.reason = f"Unexpected LLM action: {action!r}"


def screen_headlines_for_discovery(
    symbol: str,
    *,
    advisor: str = "",
    trigger: str = "",
    limit: int = 5,
    use_llm: bool = True,
) -> HeadlineScreenResult:
    """
    Screen a symbol for discovery. Blocks only on LLM BLOCK after regex hit.
    Regex-only hits without LLM do not block (fail open if LLM unavailable).
    """
    requested = (symbol or "").strip().upper()
    resolved, headlines, lookback = fetch_headlines(requested, limit=limit)
    hits = scan_keywords(headlines)

    result = HeadlineScreenResult(
        symbol=requested,
        headlines=headlines,
        lookback=lookback,
        resolved_symbol=resolved,
    )
    if resolved != requested:
        result.reason = f"Headlines from alias {resolved}"

    if not hits:
        result.stage = "clear"
        result.allowed = True
        if not result.reason:
            result.reason = "No red-flag keywords in recent headlines"
        return result

    result.keyword_hits = hits
    result.stage = "keyword_hit"
    result.allowed = True
    labels = sorted({h.label for h in hits})
    prefix = f"Headlines from alias {resolved}; " if resolved != requested else ""
    result.reason = f"{prefix}Keyword hit: {', '.join(labels)}"

    if not use_llm:
        return result

    llm = run_llm_gatekeeper(
        resolved,
        headlines,
        trigger=trigger or f"{advisor} discovery" if advisor else "discovery entry signal",
        keyword_hits=hits,
        requested_symbol=requested,
    )
    if llm is None:
        result.stage = "llm_error"
        result.reason = "LLM unavailable; regex hits not blocking discovery"
        logger.warning(
            "%s: headline regex hit (%s) but LLM unavailable — allowing discovery",
            requested,
            ", ".join(labels),
        )
        return result

    apply_llm_verdict(result, llm)
    return result


def log_headline_blocked(advisor_name: str, symbol: str, result: HeadlineScreenResult) -> None:
    """Loud log when headline screen blocks discovery."""
    hits = ", ".join(sorted({h.label for h in result.keyword_hits})) or "none"
    risk = (result.llm or {}).get("risk_score", "?")
    severity = (result.llm or {}).get("severity", "?")
    reasoning = result.reason or "headline red-flag"
    headline_lines = "\n".join(f"    • {h}" for h in result.headlines[:5])

    logger.warning(
        "\n"
        "%s\n"
        "%s  %s: %s — HEADLINE SCREEN BLOCKED DISCOVERY\n"
        "  lookback:   %s\n"
        "  hits:       %s\n"
        "  risk_score: %s  severity: %s\n"
        "  reason:     %s\n"
        "  headlines:\n%s\n"
        "%s\n",
        BLOCK_BANNER,
        BLOCK_BANNER,
        advisor_name,
        symbol,
        result.lookback,
        hits,
        risk,
        severity,
        reasoning,
        headline_lines or "    (none)",
        BLOCK_BANNER,
    )
