"""
BIZFEED advisor stub — corporate wire RSS (no overlap with PHARM feed list).

Purpose:
- Poll multiple corporate RSS endpoints (PR Newswire, GlobeNewswire subject feeds,
  Newswire.com newsroom + beats). Still excludes PHARM feeds (PR health/biotech, etc.).
- Most publishers cap each RSS file (~20 items); total volume grows with feed count.
- Rows are tagged with `categories` (earnings, ma, corporate_action, buyback_dividend,
  executive, guidance, offering_dilution) via substring match on title+summary.
- Uncategorized rows are dismissed from the catalyst path but still sampled in logs so
  you can spot missing keywords.
- All tagged catalyst rows are sent to ask_llm (Gemini, then DeepSeek fallback) each run (optional BIZFEED_LLM_MAX cap). LLM output includes
  PHARM-style significance/surprise; a Discovery is created after positive+bullish gate, composite score floor,
  national listing, allow_discovery, and a passing health_check (same pattern as PHARM).
- With no previous SmartAnalysis timestamp, the feed window defaults to the last 1 day.

Run:
    python manage.py run_bizfeed

Optional env:
    BIZFEED_LIMIT_PER_FEED   (default 50) — max items stored per RSS URL
    BIZFEED_LOG_ROW_SAMPLE        (default 60) — how many rows to log per discover run
    BIZFEED_LOG_ONLY_CATEGORIES      (optional) — comma list (e.g. earnings,ma); catalyst sample logs only rows matching any tag
    BIZFEED_LOG_UNCATEGORIZED_SAMPLE (default 25) — how many no-tag rows to log for monitoring (0 = count only)
    BIZFEED_LLM_MAX             (optional) — if set to a positive integer, cap LLM calls per run; default = all tagged catalyst rows
    BIZFEED_SKIP_NATIONAL_EXCHANGE_CHECK — set 1/true to log OTC/non-national tickers instead of rejecting (debug only)
    BIZFEED_DISCOVER         — default 1; set 0/false/off to skip Discovery creation (LLM + listing logs still run)
    BIZFEED_FETCH_ARTICLE    — default 1; set 0/false/off to skip HTTP fetch of link HTML for LLM input
    BIZFEED_ARTICLE_MAX_CHARS — max chars of fetched body in LLM INPUT (default 12000)
    BIZFEED_LLM_LOG_DETAIL   — set 1/true/on to log materiality/significance/surprise reasons and catalyst_facts
    BIZFEED_EARNINGS_DENSITY_MIN — minimum metric-density hits before LLM on earnings rows (default 5)

After a valid LLM JSON + national listing: allow_discovery(symbol, 24h) then discovered() (same pattern as PHARM).

Requires:
    - feedparser
    - Optional: cloudscraper (recommended for Newswire.com on many hosts)

Database:
    - Create an Advisor row with name \"BIZFEED\" and python_class \"Bizfeed\"
      (same pattern as PHARM). If missing, run_bizfeed_standalone() still runs a
      network preview and reports row count.
"""

from __future__ import annotations

import html
import logging
import os
from collections import Counter
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from core.models import NATIONAL_EXCHANGES
from core.services.advisors.advisor import (
    AdvisorBase,
    FEED_DISCOVERY_COMPOSITE_MIN,
    discovery_trade_explanation_lead,
    feed_discovery_weight_from_parsed,
    register,
)
from core.services.advisors.pharm import _entry_datetime_utc, _parse_since_utc, json_dumps_compact

try:
    import feedparser
except Exception:  # pragma: no cover - optional dependency guard
    feedparser = None

logger = logging.getLogger(__name__)

# Default feeds — exclude PHARM channels (no PR health/biotech, FDA, BioSpace, Endpoints, etc.).
BIZFEED_DEFAULT_FEEDS: tuple[tuple[str, str], ...] = (
    ("prnewswire_all_news", "https://www.prnewswire.com/rss/all-news-list.rss"),
    ("prnewswire_news_releases", "https://www.prnewswire.com/rss/news-releases-list.rss"),
    (
        "globenewswire_all_news",
        "https://www.globenewswire.com/RssFeed/orgclass/1/"
        "feedTitle/GlobeNewswire%20-%20All%20News",
    ),
    (
        "globenewswire_earnings",
        "https://www.globenewswire.com/RssFeed/subjectcode/"
        "13-Earnings%20Releases%20and%20Operating%20Results/feedTitle/"
        "GlobeNewswire%20-%20Earnings%20Releases%20and%20Operating%20Results",
    ),
    (
        "globenewswire_ma",
        "https://www.globenewswire.com/RssFeed/subjectcode/"
        "27-Mergers%20and%20Acquisitions/feedTitle/"
        "GlobeNewswire%20-%20Mergers%20and%20Acquisitions",
    ),
    ("newswire_newsroom", "https://www.newswire.com/newsroom/rss"),
    (
        "newswire_banking_financial",
        "https://www.newswire.com/newsroom/rss/beat/banking-and-financial-services",
    ),
    (
        "newswire_healthcare_pharma",
        "https://www.newswire.com/newsroom/rss/beat/healthcare-and-pharmaceutical",
    ),
    (
        "newswire_technology",
        "https://www.newswire.com/newsroom/rss/beat/computers-technology-and-internet",
    ),
)

# Per-feed cap (each RSS file may return fewer items than this).
_DEFAULT_LIMIT_PER_FEED = 50
_LOG_ROW_SAMPLE = 60
_LOG_UNCATEGORIZED_SAMPLE = 25

# Catalyst buckets (title + summary heuristics). Order = reporting priority.
BIZFEED_CATEGORY_ORDER: tuple[str, ...] = (
    "earnings",
    "ma",
    "corporate_action",
    "buyback_dividend",
    "executive",
    "guidance",
    "offering_dilution",
)

# Multi-word / distinctive phrases only where possible; refine with LLM later.
BIZFEED_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "earnings": (
        "reports earnings",
        "earnings release",
        "financial results",
        "quarterly results",
        "results for the quarter",
        "results for the first quarter",
        "results for the second quarter",
        "results for the third quarter",
        "results for the fourth quarter",
        "results of operations",
        "earnings per share",
        "earnings call",
        "eps of",
        "eps of $",
        "non-gaap eps",
        " gaap eps",
        "fiscal first quarter",
        "fiscal second quarter",
        "fiscal third quarter",
        "fiscal fourth quarter",
        "preliminary financial",
        "preliminary results",
    ),
    "ma": (
        "merger agreement",
        "mergers and acquisitions",
        "to acquire ",
        "will acquire ",
        "acquisition of ",
        "acquires ",
        "definitive agreement",
        "tender offer",
        "buyout",
        "strategic combination",
        "business combination",
        "take-private",
        "going private",
    ),
    "corporate_action": (
        "stock split",
        "reverse stock split",
        "reverse split",
        "spin-off",
        "spinoff",
        "name change",
        "ticker change",
        "symbol change",
        "recapitalization",
        "reorganization plan",
        "listing on",
        "delist",
        "bankruptcy",
        "chapter 11",
    ),
    "buyback_dividend": (
        "share repurchase",
        "stock repurchase",
        "share buyback",
        "stock buyback",
        "buyback program",
        "declares dividend",
        "quarterly dividend",
        "special dividend",
        "cash dividend",
        "dividend of $",
        "increases dividend",
        "dividend increase",
        "authorized repurchase",
    ),
    "executive": (
        "chief executive officer",
        "chief financial officer",
        "chief operating officer",
        " appoints ",
        " appointed ",
        "names new ceo",
        "names new cfo",
        "named chief executive",
        "named chief financial",
        "ceo succession",
        "cfo to retire",
        "resigns as chief",
        "steps down as ceo",
        "steps down as cfo",
        "interim chief executive",
        "interim chief financial",
        "president and ceo",
    ),
    "guidance": (
        "revises guidance",
        "raises guidance",
        "lowers guidance",
        "withdraws guidance",
        "updates guidance",
        "profit warning",
        "pre-announcement",
        "preliminary revenue",
        "expects revenue",
        "expect revenue",
        "outlook for fiscal",
        "financial outlook",
    ),
    "offering_dilution": (
        "public offering",
        "secondary offering",
        "follow-on offering",
        "underwritten offering",
        "registered direct",
        "convertible senior notes",
        "convertible notes offering",
        "at-the-market",
        "at the market offering",
        "common stock offering",
        "priced its offering",
        "launch of offering",
    ),
}

_CATEGORY_RANK = {k: i for i, k in enumerate(BIZFEED_CATEGORY_ORDER)}

# Regex patterns complement keyword lists for common phrasing variants.
BIZFEED_CATEGORY_REGEX: dict[str, tuple[str, ...]] = {
    "earnings": (
        r"\breports?\s+(?:[a-z]+\s+){0,3}(?:first|second|third|fourth|q[1-4])\s+(?:quarter|qtr)\s+(?:fiscal\s+)?(?:\d{4}\s+)?results?\b",
        r"\b(?:announces?|reported?)\s+(?:[a-z]+\s+){0,2}(?:unaudited\s+)?(?:first|second|third|fourth|q[1-4])\s+(?:quarter|qtr)\s+(?:\d{4}\s+)?(?:financial\s+)?results?\b",
        r"\b(?:q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\b.*\b(?:revenue|eps|earnings|financial results?)\b",
    ),
    "ma": (
        r"\b(?:definitive\s+)?(?:merger|business combination)\s+agreement\b",
        r"\bto\s+list\s+on\s+the\s+nasdaq\s+through\s+a\s+merger\b",
        r"\bacquires?\b",
    ),
}

_BIZFEED_MA_STRONG_TITLE_MARKERS: tuple[str, ...] = (
    "acquires ",
    "acquisition of",
    "to acquire ",
    "will acquire ",
    "merger agreement",
    "business combination",
    "definitive agreement",
    "to list on the nasdaq through a merger",
    "tender offer",
    "going private",
    "take-private",
)

_BIZFEED_EARNINGS_DENSITY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("percent", r"\b\d+(?:\.\d+)?\s*%"),
    ("money", r"\b(?:usd|us\$|\$|rmb|eur|gbp|cad|aud)\s*\d"),
    ("scale", r"\b\d+(?:\.\d+)?\s*(?:million|billion|bn|m)\b"),
    ("yoy_qoq", r"\b(?:yoy|qoq|year[- ]over[- ]year|quarter[- ]over[- ]quarter)\b"),
    ("period", r"\b(?:q[1-4]|first quarter|second quarter|third quarter|fourth quarter|fiscal)\b"),
    ("revenue", r"\b(?:revenue|revenues|sales)\b"),
    ("profit_loss", r"\b(?:net income|net loss|operating income|operating loss)\b"),
    ("margin", r"\b(?:gross margin|operating margin|ebitda margin)\b"),
    ("eps", r"\b(?:eps|earnings per share)\b"),
    ("guide", r"\b(?:guidance|outlook)\b"),
)

# --- LLM: shared schema + per-category task text (routed by categories[0]) ---

TASK_BIZFEED_EARNINGS = "BIZFEED_TASK_EARNINGS"
TASK_BIZFEED_MA = "BIZFEED_TASK_MA"
TASK_BIZFEED_CORPORATE_ACTION = "BIZFEED_TASK_CORPORATE_ACTION"
TASK_BIZFEED_BUYBACK_DIVIDEND = "BIZFEED_TASK_BUYBACK_DIVIDEND"
TASK_BIZFEED_EXECUTIVE = "BIZFEED_TASK_EXECUTIVE"
TASK_BIZFEED_GUIDANCE = "BIZFEED_TASK_GUIDANCE"
TASK_BIZFEED_OFFERING_DILUTION = "BIZFEED_TASK_OFFERING_DILUTION"

BIZFEED_TASK_KEY_BY_CATEGORY: dict[str, str] = {
    "earnings": TASK_BIZFEED_EARNINGS,
    "ma": TASK_BIZFEED_MA,
    "corporate_action": TASK_BIZFEED_CORPORATE_ACTION,
    "buyback_dividend": TASK_BIZFEED_BUYBACK_DIVIDEND,
    "executive": TASK_BIZFEED_EXECUTIVE,
    "guidance": TASK_BIZFEED_GUIDANCE,
    "offering_dilution": TASK_BIZFEED_OFFERING_DILUTION,
}

TEXT_BIZFEED_EARNINGS = """Focus on reported financial period (quarter/fiscal year), EPS or revenue figures only if explicitly stated,
whether management characterizes results as strong/weak, and any forward-looking or guidance language tied to this release.
Do not infer numbers that are not in the article.

If the article only schedules a future earnings release or conference call (date/time/dial-in) and does NOT report actual results,
figures, or guidance, treat it as routine investor-relations noise: materiality_score <=2, event_outcome unclear, impact_direction
not_inferable, unless the text explicitly flags a material change or restatement."""

TEXT_BIZFEED_MA = """Focus on acquirer/target names, consideration form (cash/stock), announced deal value if stated, expected closing,
and whether the article frames the deal as clearly beneficial, risky, or uncertain for shareholders. Quote only stated facts.

Distinguish corporate M&A (control of a company/business) from routine operational buys (e.g. real estate parcels, property count,
“footprint” expansion) unless the article ties them clearly to equity value or consolidated financials. For those operational stories,
keep materiality_score typically <=3 unless the text states material size, financing, or EPS impact.

If price, exchange ratio, or enterprise value is not stated, do not assume a bullish equity reaction: prefer impact_direction neutral
or not_inferable and lower impact_magnitude unless the article gives explicit value-creation language tied to shareholders."""

TEXT_BIZFEED_CORPORATE_ACTION = """Focus on the corporate action type (split, spin-off, name/ticker change, listing, restructuring, distress),
effective dates if given, and management rationale if stated. Note if dilution or control change is explicitly mentioned."""

TEXT_BIZFEED_BUYBACK_DIVIDEND = """Focus on share repurchase: new or expanded authorization, dollar or share cap, remaining authorization, timing.
For dividends: declared amount per share, payment date, regular vs special, and any stated increase or cut. Use nulls when figures are not stated."""

TEXT_BIZFEED_EXECUTIVE = """Focus on role (CEO/CFO/other), named person, effective date, interim vs permanent, succession context,
and whether departure is framed as voluntary, performance-related, or unclear. Do not speculate on stock impact."""

TEXT_BIZFEED_GUIDANCE = """Focus on what outlook metric changed (revenue, EPS, margin, etc.), direction of change, timeframe,
and whether the article cites external reasons (macro, demand). Treat vague optimism without numbers as limited evidence."""

TEXT_BIZFEED_OFFERING_DILUTION = """Focus on offering type (primary, secondary, ATM, convertible), stated size or price if any,
use of proceeds, and any explicit dilution language. Do not estimate float impact unless the article gives numbers."""

BIZFEED_BASE_LLM_PROMPT_TEMPLATE = """You are a corporate equity research analyst reviewing a press release or wire story.

TASK — apply this lens to the structured questions below (still return the full JSON schema):
__TASK_TEXT__

INPUT:
__ARTICLE_JSON__

Return ONLY valid JSON (no markdown, no code fences, no surrounding prose).
Include every top-level key in the schema below; use null only where the schema allows (never omit keys).

OUTPUT JSON SCHEMA (all fields required; use null only where the schema allows):
{
  "company_name": "string|null",
  "ticker_symbol": "string|null",

  "primary_catalyst": "earnings|ma|corporate_action|buyback_dividend|executive|guidance|offering_dilution",

  "materiality_score": 1-5,
  "is_material": true/false,
  "materiality_reason": "string (max 2 sentences)",

  "significance_score": 1-5,
  "is_significant": true/false,
  "significance_reason": "string|null (max 2 sentences)",

  "surprise_score": 1-5,
  "is_surprise": true/false,
  "surprise_reason": "string|null (max 2 sentences)",

  "event_outcome": "positive|negative|mixed|unclear",
  "impact_direction": "bullish|bearish|neutral|not_inferable",
  "impact_magnitude": 1-5,

  "catalyst_facts": "string|null",
  "summary": "string (1-2 sentences, article evidence only)",

  "confidence": 0.0-1.0,
  "insufficient_event_info": true/false
}

SCORING (be conservative; lean lower when uncertain):
- materiality_score: 1 = routine/low market relevance; 3 = meaningful; 5 = highly consequential for equity value.
- impact_magnitude: strength of directional implication from the article text only (not a forecast of the stock price).

DEFAULT STANCE (mandatory — we want skepticism, not cheerleading):
- Treat the null hypothesis as "no tradeable equity signal." Do NOT default to positive or bullish to sound balanced or helpful.
- Unless the article states concrete facts that clearly support a favorable or adverse read for owners (numbers, binding terms, final regulatory/legal outcomes, explicit guidance change),
  prefer event_outcome="unclear" and impact_direction="not_inferable", with impact_magnitude <= 2 and materiality_score <= 3.
- Use event_outcome="positive" only when the text explicitly describes a clearly favorable development for shareholders (not mere corporate self-praise).
- Use impact_direction="bullish" only when those same facts support a directional view; otherwise use "neutral" (explicitly flat/no change) or "not_inferable".
- Boilerplate partnerships, vague "strategic", property count expansions, or hires without P&L context should rarely exceed materiality_score 3 or impact_magnitude 3.
- When you choose unclear/not_inferable/low scores, lower confidence accordingly (e.g. <= 0.55).

BOOLEAN CONSISTENCY:
- is_material must be true only if materiality_score >= 4
- is_significant must be true only if significance_score >= 4
- is_surprise must be true only if surprise_score >= 4
- Set the three is_* booleans from the scores above; do not contradict your own 1–5 scores.

SIGNIFICANCE / SURPRISE (PHARM-aligned; evidence from article only):
- significance_score: 1 = minor/routine; 3 = meaningful for value; 5 = highly consequential for equity value.
- surprise_score: 1 = fully expected or priced in per article cues; 3 = some uncertainty; 5 = materially unexpected vs expectations implied in the text (timing, size, reversal).

DISCOVERY GATE (downstream — hard requirements for actionable picks):
- A Discovery is recorded only when event_outcome="positive" AND impact_direction="bullish".
- significance_score, surprise_score, and is_significant / is_surprise are used for ranking/weighting downstream, not as hard gates.
- Do not force positive/bullish. The DEFAULT STANCE above still applies: most wires should fail the hard gate.

PRIMARY_CATALYST:
- Prefer primary_catalyst equal to INPUT.primary_category. Only override if the article clearly contradicts that label; mention the mismatch in summary.

EVIDENCE RULE:
- Base every field ONLY on INPUT. Do not use outside knowledge, tickers not in the text, or imagined figures.
- When INPUT.article_body is present, treat it as the primary source (full press release fetched from link); use title/summary as supplemental.
- When article_body is absent, rely on title and summary only.
- ticker_symbol: use a bare symbol only (e.g. AIRS, MSBI). If the text shows an exchange prefix like "NASDAQ: AIRS" or "NYSE: X",
  strip the prefix and return just the letters; if no symbol is stated, use null. Never invent a ticker.
- If the article omits key facts, set insufficient_event_info=true and use nulls where allowed.
- impact_direction must be "not_inferable" when the text does not support bullish/bearish/neutral with reasonable confidence.

catalyst_facts: one short sentence of the most concrete extracted facts for THIS task lens (amounts, dates, parties), or null if none stated.
"""

BIZFEED_LLM_OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        "company_name",
        "ticker_symbol",
        "primary_catalyst",
        "materiality_score",
        "is_material",
        "materiality_reason",
        "significance_score",
        "is_significant",
        "significance_reason",
        "surprise_score",
        "is_surprise",
        "surprise_reason",
        "event_outcome",
        "impact_direction",
        "impact_magnitude",
        "catalyst_facts",
        "summary",
        "confidence",
        "insufficient_event_info",
    }
)

BIZFEED_LLM_PRIMARY_CATALYST_VALUES = frozenset(BIZFEED_CATEGORY_ORDER)
BIZFEED_LLM_EVENT_OUTCOME_VALUES = frozenset({"positive", "negative", "mixed", "unclear"})
BIZFEED_LLM_IMPACT_DIRECTION_VALUES = frozenset({"bullish", "bearish", "neutral", "not_inferable"})


def _bizfeed_task_text_for_key(task_key: str) -> str:
    return {
        TASK_BIZFEED_EARNINGS: TEXT_BIZFEED_EARNINGS,
        TASK_BIZFEED_MA: TEXT_BIZFEED_MA,
        TASK_BIZFEED_CORPORATE_ACTION: TEXT_BIZFEED_CORPORATE_ACTION,
        TASK_BIZFEED_BUYBACK_DIVIDEND: TEXT_BIZFEED_BUYBACK_DIVIDEND,
        TASK_BIZFEED_EXECUTIVE: TEXT_BIZFEED_EXECUTIVE,
        TASK_BIZFEED_GUIDANCE: TEXT_BIZFEED_GUIDANCE,
        TASK_BIZFEED_OFFERING_DILUTION: TEXT_BIZFEED_OFFERING_DILUTION,
    }[task_key]


def _bizfeed_llm_task_key_for_row(row: dict[str, Any]) -> str:
    cats = row.get("categories") or []
    primary = cats[0] if cats else "earnings"
    return BIZFEED_TASK_KEY_BY_CATEGORY.get(primary, TASK_BIZFEED_EARNINGS)


def _bizfeed_score_as_int_1_5(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and 1 <= value <= 5:
        return value
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 5 else None


def _bizfeed_align_score_booleans(out: dict[str, Any]) -> None:
    """
    Ollama often sets is_* wrong vs 1–5 scores. PHARM rules: true iff score >= 4.
    Overwrite booleans from scores when the score is valid so JSON validation stays consistent.
    """
    pairs = (
        ("materiality_score", "is_material"),
        ("significance_score", "is_significant"),
        ("surprise_score", "is_surprise"),
    )
    for score_key, bool_key in pairs:
        s = _bizfeed_score_as_int_1_5(out.get(score_key))
        if s is None:
            continue
        want = s >= 4
        cur = out.get(bool_key)
        if isinstance(cur, bool) and cur != want:
            logger.info(
                "BIZFEED_LLM: aligned %s %s -> %s (from %s=%s)",
                bool_key,
                cur,
                want,
                score_key,
                s,
            )
        out[bool_key] = want


def _bizfeed_llm_normalize_parsed(parsed: Any) -> Any:
    """
    Best-effort repair before validation: backfill summary, strip exchange prefixes,
    align is_* flags with numeric scores (model drift).
    """
    if not isinstance(parsed, dict):
        return parsed
    out = dict(parsed)
    summ = out.get("summary")
    if summ is None or (isinstance(summ, str) and not summ.strip()):
        fb = out.get("catalyst_facts") if isinstance(out.get("catalyst_facts"), str) else ""
        mr = out.get("materiality_reason") if isinstance(out.get("materiality_reason"), str) else ""
        fallback = (fb.strip() or mr.strip() or "").strip()
        if fallback:
            out["summary"] = fallback[:800]
            logger.info("BIZFEED_LLM: backfilled summary from catalyst_facts/materiality_reason")
        else:
            out["summary"] = "No summary returned; insufficient text in other fields to backfill."
            logger.info("BIZFEED_LLM: filled empty summary with placeholder (model omitted summary)")

    ts = out.get("ticker_symbol")
    if isinstance(ts, str):
        t = ts.strip()
        m = re.match(r"^(?:NASDAQ|NYSE|AMEX|ARCA|BATS|OTC|PINK|OTCQB|OTCQX)\s*:\s*(.+)$", t, re.I)
        if m:
            t = m.group(1).strip()
        out["ticker_symbol"] = t[:12] if t else None

    _bizfeed_align_score_booleans(out)
    return out


def _build_bizfeed_llm_prompt(*, task_key: str, article_json: dict[str, Any]) -> str:
    task_text = _bizfeed_task_text_for_key(task_key)
    payload = dict(article_json or {})
    for k, v in list(payload.items()):
        if isinstance(v, str) and len(v) > 6000:
            payload[k] = v[:6000]
    return (
        BIZFEED_BASE_LLM_PROMPT_TEMPLATE.replace("__TASK_TEXT__", task_text.strip()).replace(
            "__ARTICLE_JSON__", json_dumps_compact(payload)
        )
    )


def _bizfeed_llm_hard_failures(parsed: Any) -> list[str]:
    out: list[str] = []
    if parsed is None:
        return ["parsed JSON is null (model returned nothing or parse failed)"]
    if not isinstance(parsed, dict):
        return [f"parsed JSON must be an object, got {type(parsed).__name__}"]

    missing = sorted(BIZFEED_LLM_OUTPUT_KEYS - parsed.keys())
    if missing:
        out.append(f"missing keys: {', '.join(missing)}")
    extra = sorted(parsed.keys() - BIZFEED_LLM_OUTPUT_KEYS)
    if extra:
        out.append(f"unexpected keys: {', '.join(extra)}")

    def _nullable_str(v: Any, field: str) -> None:
        if v is None:
            return
        if not isinstance(v, str):
            out.append(f"{field} must be string or null, got {type(v).__name__}")

    for key in (
        "company_name",
        "ticker_symbol",
        "materiality_reason",
        "significance_reason",
        "surprise_reason",
        "summary",
        "catalyst_facts",
    ):
        _nullable_str(parsed.get(key), key)

    pc = parsed.get("primary_catalyst")
    if not isinstance(pc, str) or pc not in BIZFEED_LLM_PRIMARY_CATALYST_VALUES:
        out.append(
            f"primary_catalyst must be one of {sorted(BIZFEED_LLM_PRIMARY_CATALYST_VALUES)}, got {pc!r}"
        )

    for key in ("is_material", "is_significant", "is_surprise", "insufficient_event_info"):
        if not isinstance(parsed.get(key), bool):
            out.append(f"{key} must be boolean, got {type(parsed.get(key)).__name__}")

    def _score_1_5(v: Any, field: str) -> int | None:
        if isinstance(v, bool) or v is None:
            out.append(f"{field} must be integer 1-5, got {type(v).__name__}")
            return None
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if not isinstance(v, int):
            out.append(f"{field} must be integer 1-5, got {type(v).__name__}")
            return None
        if v < 1 or v > 5:
            out.append(f"{field} must be between 1 and 5, got {v}")
            return None
        return v

    mat = _score_1_5(parsed.get("materiality_score"), "materiality_score")
    sig = _score_1_5(parsed.get("significance_score"), "significance_score")
    sur = _score_1_5(parsed.get("surprise_score"), "surprise_score")
    mag = _score_1_5(parsed.get("impact_magnitude"), "impact_magnitude")

    conf = parsed.get("confidence")
    if isinstance(conf, bool):
        out.append("confidence must be number 0.0-1.0, got bool")
    elif conf is None:
        out.append("confidence must be number 0.0-1.0, got null")
    else:
        try:
            cf = float(conf)
        except (TypeError, ValueError):
            out.append(f"confidence must be number 0.0-1.0, got {type(conf).__name__}")
        else:
            if cf < 0.0 or cf > 1.0:
                out.append(f"confidence must be between 0.0 and 1.0, got {cf}")

    eo = parsed.get("event_outcome")
    if not isinstance(eo, str) or eo not in BIZFEED_LLM_EVENT_OUTCOME_VALUES:
        out.append(
            f"event_outcome must be one of {sorted(BIZFEED_LLM_EVENT_OUTCOME_VALUES)}, got {eo!r}"
        )

    imp = parsed.get("impact_direction")
    if not isinstance(imp, str) or imp not in BIZFEED_LLM_IMPACT_DIRECTION_VALUES:
        out.append(
            f"impact_direction must be one of {sorted(BIZFEED_LLM_IMPACT_DIRECTION_VALUES)}, got {imp!r}"
        )

    if mat is not None and isinstance(parsed.get("is_material"), bool):
        want = mat >= 4
        if parsed["is_material"] != want:
            out.append(
                f"is_material ({parsed['is_material']}) inconsistent with materiality_score ({mat}); "
                f"expected is_material={want}"
            )

    if sig is not None and isinstance(parsed.get("is_significant"), bool):
        want_sig = sig >= 4
        if parsed["is_significant"] != want_sig:
            out.append(
                f"is_significant ({parsed['is_significant']}) inconsistent with significance_score ({sig}); "
                f"expected is_significant={want_sig}"
            )

    if sur is not None and isinstance(parsed.get("is_surprise"), bool):
        want_sur = sur >= 4
        if parsed["is_surprise"] != want_sur:
            out.append(
                f"is_surprise ({parsed['is_surprise']}) inconsistent with surprise_score ({sur}); "
                f"expected is_surprise={want_sur}"
            )

    return out


def _bizfeed_discovery_gate_fail(parsed: dict[str, Any]) -> str | None:
    """Hard gate: positive outcome and bullish direction only (scores feed weight, not this gate)."""
    company = parsed.get("company_name")
    company_txt = company.strip() if isinstance(company, str) and company.strip() else "unknown"

    if parsed.get("event_outcome") == "negative":
        return f"company={company_txt}, event_outcome is negative -> gate fail"
    if parsed.get("impact_direction") == "bearish":
        return f"company={company_txt}, impact_direction is bearish -> gate fail"

    if parsed.get("event_outcome") != "positive":
        return f"company={company_txt}, event_outcome is not positive -> gate fail"
    if parsed.get("impact_direction") != "bullish":
        return f"company={company_txt}, impact_direction is not bullish -> gate fail"

    return None


def _bizfeed_llm_log_detail_enabled() -> bool:
    return (os.getenv("BIZFEED_LLM_LOG_DETAIL") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _bizfeed_log_llm_result(
    *,
    index: int,
    batch_size: int,
    task_key: str,
    model: str | None,
    row: dict[str, Any],
    parsed: dict[str, Any] | None,
    json_fails: list[str] | None = None,
    gate_fail: str | None = None,
) -> None:
    """Log LLM verdict for every catalyst row (including gate/json failures)."""
    link = (row.get("link") or "").strip()
    title = (row.get("title") or "").strip()
    prefix = f"BIZFEED_LLM result {index}/{batch_size}"

    if parsed is None:
        logger.info(
            "%s | model=%s | status=parse_failed | task=%s | title=%r | link=%s",
            prefix,
            model or "—",
            task_key,
            title[:120],
            link or "—",
        )
        for reason in json_fails or []:
            logger.info("%s | json: %s", prefix, reason)
        return

    if json_fails:
        status = "json_invalid"
    elif gate_fail:
        status = "gate_fail"
    else:
        status = "gate_pass"

    conf = parsed.get("confidence")
    conf_txt = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else conf

    logger.info(
        "%s | model=%s | status=%s | task=%s | company=%r ticker=%r | "
        "outcome=%s dir=%s mag=%s | mat=%s sig=%s sur=%s conf=%s insufficient=%s | "
        "summary=%s | title=%r | link=%s",
        prefix,
        model or "—",
        status,
        task_key,
        parsed.get("company_name"),
        parsed.get("ticker_symbol"),
        parsed.get("event_outcome"),
        parsed.get("impact_direction"),
        parsed.get("impact_magnitude"),
        parsed.get("materiality_score"),
        parsed.get("significance_score"),
        parsed.get("surprise_score"),
        conf_txt,
        parsed.get("insufficient_event_info"),
        (parsed.get("summary") or "")[:320],
        title[:120],
        link or "—",
    )

    if gate_fail:
        logger.info("%s | gate: %s", prefix, gate_fail)

    if json_fails:
        for reason in json_fails:
            logger.info("%s | json: %s", prefix, reason)

    if _bizfeed_llm_log_detail_enabled() and isinstance(parsed, dict):
        for field in (
            "materiality_reason",
            "significance_reason",
            "surprise_reason",
            "catalyst_facts",
        ):
            val = parsed.get(field)
            if isinstance(val, str) and val.strip():
                logger.info("%s | %s: %s", prefix, field, val.strip()[:600])


def _bizfeed_skip_national_exchange_check() -> bool:
    return (os.getenv("BIZFEED_SKIP_NATIONAL_EXCHANGE_CHECK") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _bizfeed_sanitize_symbol(raw: str | None) -> str | None:
    if not isinstance(raw, str):
        return None
    t = raw.strip().upper()
    for prefix in ("NASDAQ:", "NYSE:", "AMEX:", "ARCA:", "BATS:"):
        if t.startswith(prefix):
            t = t[len(prefix) :].strip()
    parts = t.split()
    t = parts[0] if parts else ""
    t = re.sub(r"[^A-Z0-9.]", "", t)
    return t or None


def _bizfeed_yahoo_exchange_code(symbol: str) -> tuple[str | None, str]:
    """
    Fetch Yahoo `exchange` code for symbol.
    Returns (code, detail) where detail is 'ok', 'missing_exchange', or 'error:...'.
    """
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).info or {}
        code = info.get("exchange")
        if not code:
            return None, "missing_exchange"
        return str(code).strip(), "ok"
    except Exception as exc:
        return None, f"error:{exc.__class__.__name__}"


def _bizfeed_discover_enabled() -> bool:
    raw = (os.getenv("BIZFEED_DISCOVER") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _bizfeed_format_explanation_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        s = f"{value:.6f}".rstrip("0").rstrip(".")
        return s or "0"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _bizfeed_build_discovery_explanation(
    parsed: dict[str, Any],
    *,
    primary_category: str | None,
    article_link: str,
    resolved_symbol: str,
    yahoo_exchange: str,
) -> str:
    company = parsed.get("company_name")
    c = company.strip() if isinstance(company, str) and company.strip() else "Unknown"
    pc = (primary_category or "catalyst").strip().lower()
    headline = f"BIZFEED {pc} - {c}"
    trade_lead = discovery_trade_explanation_lead(headline)

    def _clean_sentence(value: Any, *, max_len: int = 180) -> str | None:
        if not isinstance(value, str):
            return None
        t = value.strip()
        if not t:
            return None
        while t.endswith("."):
            t = t[:-1].rstrip()
        if not t:
            return None
        if len(t) > max_len:
            t = t[: max_len - 3].rstrip() + "..."
        return t

    segments: list[str] = []
    if article_link:
        segments.append(discovery_trade_explanation_lead(f"Article: {headline}"))
        segments.append(article_link)
    else:
        segments.append(trade_lead)
    if trade_lead != headline:
        segments.append(f"headline: {headline}")

    field_specs: tuple[tuple[str, str], ...] = (
        ("summary", "summary"),
        ("materiality_reason", "materiality"),
        ("significance_reason", "significance"),
        ("surprise_reason", "surprise"),
        ("catalyst_facts", "facts"),
    )
    for key, label in field_specs:
        cleaned = _clean_sentence(parsed.get(key))
        if cleaned:
            segments.append(f"{label}: {cleaned}")

    scores = (
        "scores: "
        f"outcome={_bizfeed_format_explanation_value(parsed.get('event_outcome'))} "
        f"dir={_bizfeed_format_explanation_value(parsed.get('impact_direction'))} "
        f"mag={_bizfeed_format_explanation_value(parsed.get('impact_magnitude'))} "
        f"sig={_bizfeed_format_explanation_value(parsed.get('significance_score'))} "
        f"sur={_bizfeed_format_explanation_value(parsed.get('surprise_score'))} "
        f"conf={_bizfeed_format_explanation_value(parsed.get('confidence'))} "
        f"ticker={_bizfeed_format_explanation_value(parsed.get('ticker_symbol'))} "
        f"resolved={resolved_symbol} "
        f"exchange={yahoo_exchange}"
    )
    segments.append(scores)
    return " | ".join(segments)


def resolve_bizfeed_listed_symbol(
    *,
    ticker_hint: str | None,
    company_name: str | None,
) -> tuple[str | None, str | None, str]:
    """
    Resolve to a tradable US national-listing symbol for SoulTrader's universe.

    Tries sanitized LLM `ticker_hint` first, then `match_company_to_symbol(company_name)`.
    Each candidate is checked via Yahoo Finance `exchange` against NATIONAL_EXCHANGES
    (same rule as Stock.create).

    Returns (symbol, yahoo_exchange_code, provenance). On failure, symbol and code are None
    and provenance explains the last failure.
    """
    from core.services.advisors.fda import match_company_to_symbol

    skip_nat = _bizfeed_skip_national_exchange_check()
    ordered: list[tuple[str, str]] = []
    dedupe_keys: set[str] = set()

    hint = _bizfeed_sanitize_symbol(ticker_hint)
    if hint and hint not in dedupe_keys:
        dedupe_keys.add(hint)
        ordered.append((hint, "llm_ticker"))

    cn = company_name.strip() if isinstance(company_name, str) else ""
    if cn:
        resolved, how = match_company_to_symbol(cn)
        if resolved:
            sym2 = _bizfeed_sanitize_symbol(resolved) or str(resolved).strip().upper()
            if sym2 and sym2 not in dedupe_keys:
                dedupe_keys.add(sym2)
                ordered.append((sym2, f"company_match:{how or '?'}"))

    for sym, prov in ordered:
        code, detail = _bizfeed_yahoo_exchange_code(sym)
        if not code:
            logger.info(
                "BIZFEED listing check | symbol=%s source=%s yahoo=%s",
                sym,
                prov,
                detail,
            )
            continue
        if code not in NATIONAL_EXCHANGES:
            logger.info(
                "BIZFEED listing check | symbol=%s source=%s exchange=%s not in national set",
                sym,
                prov,
                code,
            )
            if not skip_nat:
                continue
        logger.info(
            "BIZFEED listing check | OK symbol=%s exchange=%s source=%s%s",
            sym,
            code,
            prov,
            " (national check skipped)" if skip_nat and code not in NATIONAL_EXCHANGES else "",
        )
        return sym, code, prov

    return None, None, "no_national_listing"


def _dedupe_bizfeed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("link") or "").strip().lower()
        if not key:
            key = f"{row.get('published_at', '')}|{(row.get('title') or '').strip().lower()}"
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def _strict_newswire_headers() -> dict[str, str]:
    referer = "https://www.newswire.com/newsroom"
    return {
        "User-Agent": _BROWSER_UA,
        "Accept": "application/rss+xml, application/xml, application/atom+xml, text/xml, */*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Referer": referer,
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }


def _needs_strict_fetch(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == "newswire.com" or host.endswith(".newswire.com")


def _parse_rss_url(url: str):
    """Same strategy as test_bizfeed.py (Newswire strict fetch + cloudscraper first)."""
    if feedparser is None:
        raise RuntimeError("feedparser is not installed")

    if not _needs_strict_fetch(url):
        return feedparser.parse(url, agent=_BROWSER_UA)

    headers = _strict_newswire_headers()

    def _attach_status(parsed: Any, code: int) -> None:
        parsed.status = code

    try:
        import cloudscraper

        scraper = cloudscraper.create_scraper()
        r = scraper.get(url, headers=headers, timeout=45)
        alt = feedparser.parse(r.content)
        _attach_status(alt, r.status_code)
        if r.status_code == 200 and (alt.entries or []):
            return alt
    except ImportError:
        pass
    except Exception:
        pass

    body: bytes = b""
    status: int | None = None
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read()
        status = e.code
    except Exception:
        pass

    parsed = feedparser.parse(body) if body else feedparser.parse(b"")
    if status is not None:
        parsed.status = status
    if status == 200 and (parsed.entries or []):
        return parsed

    try:
        import requests

        r = requests.get(url, headers=headers, timeout=45)
        alt = feedparser.parse(r.content)
        _attach_status(alt, r.status_code)
        if r.status_code == 200 and (alt.entries or []):
            return alt
        if status != 200 and r.status_code == 200:
            return alt
    except Exception:
        pass

    return parsed


def _clean_title(title: str) -> str:
    text = html.unescape(title or "")
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.split()).strip()


def _bizfeed_earnings_density_min() -> int:
    raw = (os.getenv("BIZFEED_EARNINGS_DENSITY_MIN") or "5").strip()
    try:
        return max(1, min(12, int(raw)))
    except ValueError:
        return 5


def _bizfeed_metric_density(text: str) -> tuple[int, list[str]]:
    hits: list[str] = []
    blob = (text or "").lower()
    if not blob:
        return 0, hits
    for label, pattern in _BIZFEED_EARNINGS_DENSITY_PATTERNS:
        if re.search(pattern, blob, flags=re.I):
            hits.append(label)
    return len(hits), hits


# Truncate fetched HTML text before boilerplate / related-articles blocks (saves tokens).
_BIZFEED_ARTICLE_TRUNC_MARKERS: tuple[str, ...] = (
    "Safe Harbor Statement",
    "Safe Harbor statement",
    "Forward-looking statements",
    "Also from this source",
    "More Releases From This Source",
    "Explore\n",
    "Request a Demo",
    "SOURCE ",
)

# CSS selectors for common wire hosts (first match with enough text wins).
_BIZFEED_ARTICLE_BODY_SELECTORS: tuple[str, ...] = (
    "section.release-body",
    "div.release-body",
    "div.article-body",
    "div#main-content article",
    "article",
    "main",
    "div.field--name-body",
    "div.region-content",
)


def _bizfeed_fetch_article_enabled() -> bool:
    return (os.getenv("BIZFEED_FETCH_ARTICLE") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _bizfeed_article_max_chars() -> int:
    raw = (os.getenv("BIZFEED_ARTICLE_MAX_CHARS") or "12000").strip()
    try:
        return max(500, min(20000, int(raw)))
    except ValueError:
        return 12000


def _bizfeed_trim_article_boilerplate(text: str) -> str:
    out = " ".join((text or "").split())
    if not out:
        return ""
    lower = out.lower()
    cut = len(out)
    for marker in _BIZFEED_ARTICLE_TRUNC_MARKERS:
        idx = lower.find(marker.lower())
        if idx >= 200:
            cut = min(cut, idx)
    return out[:cut].strip()


def _bizfeed_http_get(url: str, *, timeout: float = 25.0) -> tuple[int, bytes]:
    headers = dict(_strict_newswire_headers())
    headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    try:
        import requests

        r = requests.get(url, headers=headers, timeout=timeout)
        return int(r.status_code), r.content or b""
    except Exception:
        pass
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), resp.read()


def fetch_bizfeed_article_body(url: str) -> tuple[str | None, str]:
    """
    Fetch press-release HTML and return plain text for LLM INPUT.article_body.

    Returns (text_or_none, log_detail) e.g. ("8421 chars", "fetch_disabled").
    """
    link = (url or "").strip()
    if not link:
        return None, "no_link"
    if not _bizfeed_fetch_article_enabled():
        return None, "fetch_disabled"
    parsed = urlparse(link)
    if parsed.scheme not in ("http", "https"):
        return None, "bad_scheme"

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None, "bs4_missing"

    try:
        status, body = _bizfeed_http_get(link)
    except Exception as exc:
        return None, f"http_error:{exc.__class__.__name__}"

    if status != 200 or not body:
        return None, f"http_{status}"

    soup = BeautifulSoup(body, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    extracted: str | None = None
    for sel in _BIZFEED_ARTICLE_BODY_SELECTORS:
        node = soup.select_one(sel)
        if node is None:
            continue
        txt = " ".join(node.get_text(" ", strip=True).split())
        if len(txt) >= 200:
            extracted = txt
            break

    if not extracted:
        txt = " ".join(soup.get_text(" ", strip=True).split())
        extracted = txt if len(txt) >= 200 else None

    if not extracted:
        return None, "body_too_short"

    trimmed = _bizfeed_trim_article_boilerplate(extracted)
    cap = _bizfeed_article_max_chars()
    if len(trimmed) > cap:
        trimmed = trimmed[:cap].rstrip() + "…"

    return trimmed, f"ok_{len(trimmed)}_chars"


def _bizfeed_article_body_for_row(
    row: dict[str, Any],
    cache: dict[str, tuple[str | None, str]],
) -> tuple[str | None, str]:
    link = (row.get("link") or "").strip()
    if not link:
        return None, "no_link"
    if link in cache:
        return cache[link]
    text, detail = fetch_bizfeed_article_body(link)
    cache[link] = (text, detail)
    return text, detail


def _bizfeed_build_llm_article_payload(
    row: dict[str, Any],
    cache: dict[str, tuple[str | None, str]],
) -> tuple[dict[str, Any], str]:
    """LLM INPUT dict plus fetch status string for logs."""
    primary = (row.get("categories") or [None])[0]
    payload: dict[str, Any] = {
        "source": row.get("source"),
        "published_at": row.get("published_at"),
        "title": row.get("title"),
        "summary": row.get("summary"),
        "link": row.get("link"),
        "categories": row.get("categories"),
        "primary_category": primary,
    }
    body, detail = _bizfeed_article_body_for_row(row, cache)
    if body:
        payload["article_body"] = body
    return payload, detail


def classify_bizfeed_row(row: dict[str, Any]) -> list[str]:
    """
    Tag a row by catalyst type (title + summary, case-insensitive substring match).
    Returns category keys in BIZFEED_CATEGORY_ORDER (may be empty).
    """
    title = (row.get("title") or "").strip()
    summary = (row.get("summary") or "").strip()
    title_hay = title.lower()
    hay = f"{title} {summary}".lower()
    matched: list[str] = []

    for cat in BIZFEED_CATEGORY_ORDER:
        kw = BIZFEED_CATEGORY_KEYWORDS.get(cat, ())
        rx = BIZFEED_CATEGORY_REGEX.get(cat, ())
        kw_hit = any(p in hay for p in kw)
        rx_hit = any(re.search(p, hay, flags=re.I) for p in rx)
        if kw_hit or rx_hit:
            matched.append(cat)

    ma_title_strong = (
        any(p in title_hay for p in _BIZFEED_MA_STRONG_TITLE_MARKERS)
        or any(re.search(p, title_hay, flags=re.I) for p in BIZFEED_CATEGORY_REGEX.get("ma", ()))
    )
    # Avoid M&A tagging when only body/summary casually mentions a transaction.
    if "ma" in matched and not ma_title_strong:
        matched = [c for c in matched if c != "ma"]

    # If this is clearly an earnings headline, avoid misrouting to M&A due body mentions.
    earnings_title = (
        any(p in title_hay for p in BIZFEED_CATEGORY_KEYWORDS["earnings"])
        or any(re.search(p, title_hay, flags=re.I) for p in BIZFEED_CATEGORY_REGEX["earnings"])
    )
    if "earnings" in matched and "ma" in matched:
        if earnings_title and not ma_title_strong:
            matched = [c for c in matched if c != "ma"]

    matched.sort(key=lambda c: _CATEGORY_RANK.get(c, 99))
    return matched


def tag_bizfeed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mutate rows in place: set `categories` from classify_bizfeed_row."""
    for r in rows:
        r["categories"] = classify_bizfeed_row(r)
    return rows


def fetch_bizfeed_rows(
    *,
    limit_per_feed: int = 25,
    since: datetime | str | None = None,
) -> list[dict[str, Any]]:
    """
    Pull normalized rows from each default BIZFEED source.
    `since` uses the same parsing rules as PHARM (_parse_since_utc).
    """
    if feedparser is None:
        raise RuntimeError("feedparser is not installed. Install via: pip install feedparser")

    since_dt = _parse_since_utc(since)
    rows: list[dict[str, Any]] = []
    logger.info("BIZFEED polling %d feeds (cap=%d items each)", len(BIZFEED_DEFAULT_FEEDS), limit_per_feed)

    for source, url in BIZFEED_DEFAULT_FEEDS:
        try:
            parsed = _parse_rss_url(url)
        except Exception as exc:
            logger.warning("BIZFEED feed %s failed: %s", source, exc)
            continue

        raw_entries = list(parsed.entries or [])
        if not raw_entries and _needs_strict_fetch(url):
            logger.warning(
                "BIZFEED feed %s returned 0 items (HTTP %r); Newswire beats often need cloudscraper + US-friendly IP",
                source,
                getattr(parsed, "status", None),
            )

        n = 0
        for entry in raw_entries:
            dt = _entry_datetime_utc(entry)
            if since_dt is not None and dt < since_dt:
                continue
            rows.append(
                {
                    "source": source,
                    "source_url": url,
                    "published_at": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "title": _clean_title(entry.get("title") or ""),
                    "summary": " ".join(
                        (entry.get("summary") or entry.get("description") or "").strip().split()
                    ),
                    "link": (entry.get("link") or "").strip(),
                }
            )
            n += 1
            if n >= limit_per_feed:
                break
        logger.info("BIZFEED feed %s rows=%d (cap=%d)", source, n, limit_per_feed)

    return rows


class Bizfeed(AdvisorBase):
    """Stub advisor for corporate RSS → keyword routing → LLM (incremental)."""

    def discover(self, sa):
        logger.info("BIZFEED discover starting")
        prev_ts = self.get_previous_sa_timestamp(sa, username=getattr(sa, "username", None))
        since_dt = _parse_since_utc(prev_ts)
        if since_dt is None:
            since_dt = datetime.now(UTC) - timedelta(days=1)
            logger.info(
                "BIZFEED no previous SA found; fallback since=%s (1-day lookback)",
                since_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )

        cap = int(os.getenv("BIZFEED_LIMIT_PER_FEED", str(_DEFAULT_LIMIT_PER_FEED)))
        log_sample = int(os.getenv("BIZFEED_LOG_ROW_SAMPLE", str(_LOG_ROW_SAMPLE)))
        uncat_log_cap = int(os.getenv("BIZFEED_LOG_UNCATEGORIZED_SAMPLE", str(_LOG_UNCATEGORIZED_SAMPLE)))

        rows = fetch_bizfeed_rows(limit_per_feed=max(1, cap), since=since_dt)
        raw_n = len(rows)
        rows = _dedupe_bizfeed_rows(rows)
        tag_bizfeed_rows(rows)
        logger.info(
            "BIZFEED collected rows raw=%d unique=%d (feeds=%d)",
            raw_n,
            len(rows),
            len(BIZFEED_DEFAULT_FEEDS),
        )

        catalyst_rows = [r for r in rows if r.get("categories")]
        uncat_rows = [r for r in rows if not r.get("categories")]
        logger.info(
            "BIZFEED catalyst rows (tagged): %d | dismissed (no tag): %d — only tagged rows use the catalyst path",
            len(catalyst_rows),
            len(uncat_rows),
        )

        cat_counts = Counter(c for r in catalyst_rows for c in r.get("categories", []))
        ordered_counts = {k: cat_counts[k] for k in BIZFEED_CATEGORY_ORDER if cat_counts[k]}
        if ordered_counts:
            logger.info("BIZFEED category counts: %s", ordered_counts)

        def _sort_catalyst_key(r: dict[str, Any]) -> int:
            cats = r.get("categories") or []
            return _CATEGORY_RANK.get(cats[0], 99)

        sorted_catalyst = sorted(catalyst_rows, key=_sort_catalyst_key)
        only_raw = os.getenv("BIZFEED_LOG_ONLY_CATEGORIES", "").strip()
        if only_raw:
            want = {x.strip().lower() for x in only_raw.split(",") if x.strip()}
            filtered = [r for r in sorted_catalyst if want & set(r.get("categories") or [])]
            if filtered:
                sorted_catalyst = filtered
            else:
                logger.info(
                    "BIZFEED_LOG_ONLY_CATEGORIES=%r matched 0 catalyst rows; logging full catalyst sample",
                    only_raw,
                )

        for r in sorted_catalyst[: max(0, log_sample)]:
            tags = ",".join(r.get("categories") or [])
            logger.info(
                "BIZFEED row | %s | %s | tags=%s | %s",
                r["source"],
                r["published_at"],
                tags,
                (r["title"] or "")[:120],
            )

        if uncat_rows:
            uncat_sorted = sorted(
                uncat_rows,
                key=lambda r: r.get("published_at") or "",
                reverse=True,
            )
            if uncat_log_cap > 0:
                logger.info(
                    "BIZFEED uncat monitor (dismissed, newest first; cap=%d) — watch for missing keywords:",
                    uncat_log_cap,
                )
                for r in uncat_sorted[: max(0, uncat_log_cap)]:
                    logger.info(
                        "BIZFEED uncat | %s | %s | %s",
                        r["source"],
                        r["published_at"],
                        (r["title"] or "")[:120],
                    )
            else:
                logger.info(
                    "BIZFEED uncat: %d rows dismissed (BIZFEED_LOG_UNCATEGORIZED_SAMPLE=0; no title sample)",
                    len(uncat_rows),
                )

        llm_pool = list(catalyst_rows)
        llm_pool.sort(key=lambda r: r.get("published_at") or "", reverse=True)
        cap_raw = (os.getenv("BIZFEED_LLM_MAX") or "").strip()
        if cap_raw:
            try:
                cap_n = int(cap_raw)
            except ValueError:
                cap_n = 0
            if cap_n > 0:
                llm_batch = llm_pool[:cap_n]
                logger.info(
                    "BIZFEED_LLM | batch=%d of %d catalyst rows (BIZFEED_LLM_MAX=%d)",
                    len(llm_batch),
                    len(llm_pool),
                    cap_n,
                )
            else:
                llm_batch = llm_pool
                logger.info(
                    "BIZFEED_LLM | batch=%d catalyst rows (BIZFEED_LLM_MAX=%r ignored; not positive)",
                    len(llm_batch),
                    cap_raw,
                )
        else:
            llm_batch = llm_pool
            logger.info("BIZFEED_LLM | batch=%d catalyst rows (no cap)", len(llm_batch))

        article_body_cache: dict[str, tuple[str | None, str]] = {}
        fetch_on = _bizfeed_fetch_article_enabled()
        logger.info(
            "BIZFEED article fetch for LLM: %s (max_chars=%d)",
            "on" if fetch_on else "off",
            _bizfeed_article_max_chars(),
        )
        logger.info(
            "BIZFEED earnings density gate: min_hits=%d",
            _bizfeed_earnings_density_min(),
        )

        for i, row in enumerate(llm_batch, start=1):
            task_key = _bizfeed_llm_task_key_for_row(row)
            primary = (row.get("categories") or [None])[0]
            article_payload, fetch_detail = _bizfeed_build_llm_article_payload(row, article_body_cache)
            body_len = len(article_payload.get("article_body") or "")
            summ_len = len(article_payload.get("summary") or "")
            logger.info(
                "BIZFEED_LLM input | %s | fetch=%s | summary=%d chars | article_body=%d chars",
                (row.get("link") or "")[:80],
                fetch_detail,
                summ_len,
                body_len,
            )

            if primary == "earnings":
                density_blob = " ".join(
                    [
                        str(row.get("title") or ""),
                        str(row.get("summary") or ""),
                        str(article_payload.get("article_body") or ""),
                    ]
                )
                density_count, density_hits = _bizfeed_metric_density(density_blob)
                if density_count < _bizfeed_earnings_density_min():
                    logger.info(
                        "BIZFEED density skip | task=%s | hits=%d < %d | labels=%s | title=%.120s",
                        task_key,
                        density_count,
                        _bizfeed_earnings_density_min(),
                        ",".join(density_hits) or "none",
                        (row.get("title") or "")[:120],
                    )
                    continue
                logger.info(
                    "BIZFEED density pass | task=%s | hits=%d | labels=%s",
                    task_key,
                    density_count,
                    ",".join(density_hits),
                )

            prompt = _build_bizfeed_llm_prompt(task_key=task_key, article_json=article_payload)
            llm_model, parsed = self.ask_llm(prompt)
            parsed = _bizfeed_llm_normalize_parsed(parsed)
            fails = _bizfeed_llm_hard_failures(parsed)
            gate_fail = _bizfeed_discovery_gate_fail(parsed) if not fails else None
            _bizfeed_log_llm_result(
                index=i,
                batch_size=len(llm_batch),
                task_key=task_key,
                model=llm_model,
                row=row,
                parsed=parsed if isinstance(parsed, dict) else None,
                json_fails=fails or None,
                gate_fail=gate_fail,
            )
            if fails:
                continue
            if gate_fail:
                continue
            exp_pc = primary
            got_pc = parsed.get("primary_catalyst")
            if exp_pc and got_pc != exp_pc:
                logger.info(
                    "BIZFEED_LLM note: primary_catalyst model=%r vs heuristic=%r (title=%.80s)",
                    got_pc,
                    exp_pc,
                    (row.get("title") or "")[:80],
                )
            cn_res = parsed.get("company_name")
            if isinstance(cn_res, str):
                cn_res = cn_res.strip() or None
            else:
                cn_res = None
            listed_sym, listed_ex, listed_src = resolve_bizfeed_listed_symbol(
                ticker_hint=parsed.get("ticker_symbol"),
                company_name=cn_res,
            )
            if not listed_sym:
                logger.info(
                    "BIZFEED_LLM listed | rejected company=%r ticker_hint=%r detail=%s",
                    cn_res,
                    parsed.get("ticker_symbol"),
                    listed_src,
                )
                continue

            logger.info(
                "BIZFEED_LLM listed | sym=%s yahoo_exchange=%s via=%s",
                listed_sym,
                listed_ex,
                listed_src,
            )
            if not _bizfeed_discover_enabled():
                logger.info(
                    "BIZFEED skip discovery: BIZFEED_DISCOVER off (listed %s; set BIZFEED_DISCOVER=1 to persist)",
                    listed_sym,
                )
                continue

            if not self.allow_discovery(listed_sym, period=24):
                logger.info("BIZFEED skip discovery: allow_discovery false for %s", listed_sym)
                continue

            weight, raw_comp, wdetail = feed_discovery_weight_from_parsed(parsed, listed_sym)
            if weight is None:
                logger.info(
                    "BIZFEED skip discovery: composite raw=%s < %s | %s | %s",
                    raw_comp,
                    FEED_DISCOVERY_COMPOSITE_MIN,
                    listed_sym,
                    wdetail,
                )
                continue
            logger.info("BIZFEED discovery weight | %s | %s -> weight=%s", listed_sym, wdetail, weight)

            explanation = _bizfeed_build_discovery_explanation(
                parsed,
                primary_category=primary,
                article_link=(row.get("link") or "").strip(),
                resolved_symbol=listed_sym,
                yahoo_exchange=listed_ex or "",
            )

            stock = self.discovered(
                sa,
                listed_sym,
                explanation,
                None,
                weight=weight,
            )
            if stock:
                logger.info("BIZFEED discovered %s (%s)", listed_sym, primary or "")
            else:
                logger.warning("BIZFEED discovered() returned None for %s", listed_sym)


def run_bizfeed_standalone():
    """
    Entry point for `manage.py run_bizfeed`.
    If Advisor \"BIZFEED\" is missing, runs a short network preview only.
    """
    from core.services import advisors as advisor_modules
    from core.models import Advisor, SmartAnalysis

    if feedparser is None:
        return None, "feedparser is not installed"

    try:
        advisor_row = Advisor.objects.get(name="BIZFEED")
    except Advisor.DoesNotExist:
        try:
            preview = tag_bizfeed_rows(fetch_bizfeed_rows(limit_per_feed=3, since=None))
        except Exception as exc:
            return None, f"BIZFEED preview fetch failed: {exc}"
        return (
            f"BIZFEED Advisor row not found; preview fetched {len(preview)} sample rows. "
            "Create Advisor name=BIZFEED, python_class=Bizfeed, then re-run.",
            None,
        )

    module_name = advisor_row.python_class.lower()
    module = getattr(advisor_modules, module_name)
    python_class = getattr(module, advisor_row.python_class)

    sa = SmartAnalysis(username="run_bizfeed")
    sa.save()
    impl = python_class(advisor_row)
    impl.discover(sa)

    return "BIZFEED discover() completed", None


register(name="BIZFEED", python_class="Bizfeed")
