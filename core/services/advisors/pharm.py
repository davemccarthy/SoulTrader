"""
PHARM advisor stub.

Purpose:
- Placeholder advisor for pharma/regulatory RSS + trial catalyst pipeline.
- Mirrors the standalone entry pattern used by EDGAR.

Run:
    python manage.py run_pharm

Optional env:
    PHARM_EVENT_CLASS=approval|trial|general  (default approval) — which shortlist class gets LLM passes
    PHARM_DEBUG_SHORTLIST_IDX=N               — only process shortlist row N if it matches PHARM_EVENT_CLASS

Rows with heuristic score < 0 are never sent to the LLM (still listed in the final shortlist log).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import html
import logging
import os
import re
from typing import Any

from core.services.advisors.advisor import (
    AdvisorBase,
    FEED_DISCOVERY_COMPOSITE_MIN,
    discovery_trade_explanation_lead,
    feed_discovery_weight_from_parsed,
    register,
)

try:
    import feedparser
except Exception:  # pragma: no cover - optional dependency guard
    feedparser = None

logger = logging.getLogger(__name__)

FDA_FEED_URL = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
PRNEWSWIRE_HEALTH_FEED = "https://www.prnewswire.com/rss/health-latest-news-list.rss"
# Narrow biotech slice (20 items); health-latest is the same cap but mostly non-biotech noise.
PRNEWSWIRE_BIOTECH_FEED = "https://www.prnewswire.com/rss/health-latest-news/biotechnology-list.rss"
BUSINESSWIRE_FEED = "https://www.businesswire.com/portal/site/home/rss/"
GLOBENEWSWIRE_PHARMA_FEED = "https://www.globenewswire.com/RssFeed/subjectcode/20-Health/feedTitle/GlobeNewswire%20-%20Health"
ENDPOINTS_FEED = "https://endpts.com/feed/"
BIOSPACE_DRUG_DEVELOPMENT_FEED = "https://www.biospace.com/drug-development.rss"
BIOSPACE_FDA_FEED = "https://www.biospace.com/FDA.rss"
FIERCEBIOTECH_FEED = "https://www.fiercebiotech.com/rss/xml"
GEN_FEED = "http://feeds.feedburner.com/GenGeneticEngineeringAndBiotechnologyNews"


def _parse_since_utc(since: datetime | str | None) -> datetime | None:
    if since is None:
        return None
    if isinstance(since, datetime):
        return since.replace(tzinfo=UTC) if since.tzinfo is None else since.astimezone(UTC)
    if isinstance(since, str):
        raw = since.strip()
        if not raw:
            return None
        candidate = raw.replace("Z", "+00:00")
        if " " in candidate and "T" not in candidate:
            candidate = candidate.replace(" ", "T", 1)
        parsed = datetime.fromisoformat(candidate)
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    raise TypeError("'since' must be datetime, str, or None")


def _entry_datetime_utc(entry: Any) -> datetime:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=UTC)
        except Exception:
            pass
    raw = (entry.get("published") or entry.get("updated") or "").strip()
    if raw:
        for fmt in ("%b %d, %Y %I:%M%p", "%b %d, %Y %I:%M %p"):
            try:
                return datetime.strptime(raw, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
    return datetime.now(UTC)


def _clean_title(title: str) -> str:
    text = html.unescape(title or "")
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.split()).strip()


def _rrs_fetch(
    *,
    source: str,
    url: str,
    limit: int = 25,
    since: datetime | str | None = None,
) -> list[dict[str, str]]:
    if feedparser is None:
        raise RuntimeError("feedparser is not installed. Install via: pip install feedparser")

    since_dt = _parse_since_utc(since)
    parsed = feedparser.parse(url)
    rows: list[dict[str, str]] = []
    max_rows = max(0, int(limit))

    for entry in list(parsed.entries or []):
        entry_dt = _entry_datetime_utc(entry)
        if since_dt is not None and entry_dt < since_dt:
            continue
        rows.append(
            {
                "source": source,
                "source_url": url,
                "published_at": entry_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "title": _clean_title(entry.get("title") or ""),
                "summary": " ".join((entry.get("summary") or entry.get("description") or "").strip().split()),
                "link": (entry.get("link") or "").strip(),
            }
        )
        if len(rows) >= max_rows:
            break
    return rows


def rrs_fda(*, limit: int = 25, since: datetime | str | None = None, url: str = FDA_FEED_URL) -> list[dict[str, str]]:
    return _rrs_fetch(source="fda_press_releases", url=url, limit=limit, since=since)


def rrs_prnewswire_health(
    *, limit: int = 25, since: datetime | str | None = None, url: str = PRNEWSWIRE_HEALTH_FEED
) -> list[dict[str, str]]:
    return _rrs_fetch(source="prnewswire_health", url=url, limit=limit, since=since)


def rrs_prnewswire_biotech(
    *, limit: int = 25, since: datetime | str | None = None, url: str = PRNEWSWIRE_BIOTECH_FEED
) -> list[dict[str, str]]:
    return _rrs_fetch(source="prnewswire_biotech", url=url, limit=limit, since=since)


def rrs_businesswire(*, limit: int = 25, since: datetime | str | None = None, url: str = BUSINESSWIRE_FEED) -> list[dict[str, str]]:
    return _rrs_fetch(source="businesswire_general", url=url, limit=limit, since=since)


def rrs_globenewswire_pharma(
    *, limit: int = 25, since: datetime | str | None = None, url: str = GLOBENEWSWIRE_PHARMA_FEED
) -> list[dict[str, str]]:
    return _rrs_fetch(source="globenewswire_pharma", url=url, limit=limit, since=since)


def rrs_endpoints(*, limit: int = 25, since: datetime | str | None = None, url: str = ENDPOINTS_FEED) -> list[dict[str, str]]:
    return _rrs_fetch(source="endpoints", url=url, limit=limit, since=since)


def rrs_biospace_drug_development(
    *, limit: int = 25, since: datetime | str | None = None, url: str = BIOSPACE_DRUG_DEVELOPMENT_FEED
) -> list[dict[str, str]]:
    return _rrs_fetch(source="biospace_drug_development", url=url, limit=limit, since=since)


def rrs_biospace_fda(*, limit: int = 25, since: datetime | str | None = None, url: str = BIOSPACE_FDA_FEED) -> list[dict[str, str]]:
    return _rrs_fetch(source="biospace_fda", url=url, limit=limit, since=since)


def rrs_fiercebiotech(*, limit: int = 25, since: datetime | str | None = None, url: str = FIERCEBIOTECH_FEED) -> list[dict[str, str]]:
    return _rrs_fetch(source="fiercebiotech_all", url=url, limit=limit, since=since)


def rrs_gen(*, limit: int = 25, since: datetime | str | None = None, url: str = GEN_FEED) -> list[dict[str, str]]:
    return _rrs_fetch(source="gen", url=url, limit=limit, since=since)


PHARM_GREEN_TERMS_STRONG = (
    "fda approves",
    "fda approval",
    "approved by the fda",
    "fda nod",
    "fda greenlight",
    "breakthrough therapy",
    "pivotal trial",
    "phase 3",
    "topline",
    "readout",
    "nda",
    "bla",
    "pdufa",
)
PHARM_GREEN_TERMS_WEAK = (
    "phase 2",
    "phase 2b",
    "phase 1",
    "phase 1b",
    "first patient dosed",
    "initiates",
    "met endpoint",
    "statistically significant",
    "positive results",
)
PHARM_RED_TERMS_STRONG = (
    "fda refusal",
    "clinical hold",
    "bankruptcy",
    "wind-down",
    "fails pivotal trial",
    "misses phase 3",
    "deaths",
    "liver injuries",
)
PHARM_RED_TERMS_WEAK = (
    "doubts",
    "concerns",
    "mixed data",
    "delay",
    "delays decision",
    "extends review",
    "warns",
    "risk",
    "miss the mark",
    "setback",
)

PHARM_SUPPRESS_TERMS = (
    "scholarship",
    "award",
    "festival",
    "leadership team",
    "appoints",
    "reports march",
    "quarterly cash dividend",
    "sales",
    "fundraising tracker",
    "layoff tracker",
)

# Same score bar for every feed (no per-RSS bias).
PHARM_SHORTLIST_SCORE_MIN = 2

PHARM_NEGATIVE_DOMINANT_TERMS = (
    "fails",
    "failed",
    "misses",
    "missed",
    "bankruptcy",
    "clinical hold",
    "fda refusal",
    "deaths",
)

APPROVAL_CLASS_TERMS = (
    "fda approves",
    "fda approval",
    "approved by the fda",
    "fda nod",
    "fda greenlight",
    "nda",
    "bla",
    "pdufa",
    "grants approval",
)

TRIAL_CLASS_TERMS = (
    "phase 1",
    "phase 1b",
    "phase 2",
    "phase 2b",
    "phase 3",
    "pivotal trial",
    "readout",
    "topline",
    "first patient dosed",
    "met endpoint",
    "fails pivotal trial",
    "misses phase 3",
)

# PHARM uses a single base prompt template with 3 task variants (approval/trial/general).
# We keep short task keys for logging/routing; the actual prompt text is built from TASK_* + BASE template.
TASK_APPROVAL_KEY = "PHARM_APPROVAL_TASK"
TASK_TRIAL_KEY = "PHARM_TRIAL_TASK"
TASK_GENERAL_KEY = "PHARM_GENERAL_TASK"

PHARM_BASE_LLM_PROMPT_TEMPLATE = """You are a biotech equity research analyst.

Task:
__TASK_TEXT__

INPUT:
__ARTICLE_JSON__

Return ONLY valid JSON (no markdown, no surrounding quotes).
OUTPUT JSON SCHEMA (all fields required):
{
  "company_name": "string|null",

  "regulator": "FDA|EMA|MHRA|other|null",
  "drug_name": "string|null",
  "indication": "string|null",
  "approval_type": "full_approval|accelerated_approval|label_expansion|crl|complete_response|other|null",

  "is_significant": true/false,
  "significance_score": 1-5,
  "significance_reason": "string",

  "is_surprise": true/false,
  "surprise_score": 1-5,
  "surprise_reason": "string",

  "event_outcome": "positive|negative|mixed|unclear",
  "impact_direction": "bullish|bearish|neutral",
  "impact_magnitude": 1-5,

  "summary": "1-2 sentence plain-English summary",

  "confidence": 0.0-1.0,
  "insufficient_event_info": true/false
}

SCORING RULES (be conservative; lean lower when uncertain):
Significance (impact to company value):
1 = minor/niche; 3 = meaningful but not transformational; 5 = transformational

Surprise (relative to market expectations):
1 = fully expected/routine; 3 = some uncertainty; 5 = materially unexpected positive outcome

BOOLEAN CONSISTENCY:
- is_significant must be true only if significance_score >= 4
- is_surprise must be true only if surprise_score >= 4

DISCOVERY (downstream): A stock is discovered only when event_outcome="positive" AND impact_direction="bullish".
Significance/surprise scores (and booleans) inform Discovery weight with tape position, not the hard gate.

DIRECTIONALITY RULE (critical):
- You MUST classify event_outcome and impact_direction from article evidence.
- Negative clinical/regulatory outcomes (e.g., "failed primary endpoint", "misses phase 3", CRL, clinical hold, bankruptcy)
  should map to event_outcome="negative" and impact_direction="bearish" unless explicit conflicting evidence is provided.
- Positive catalyst outcomes (e.g., clear approval wins, strong successful pivotal readouts) should map to
  event_outcome="positive" and impact_direction="bullish".
- If mixed, use event_outcome="mixed" and impact_direction="neutral".

EVIDENCE RULE:
- Base judgments ONLY on the provided INPUT content. Do not use external knowledge.

REASON LIMITS:
- significance_reason and surprise_reason: max 2 sentences each.

If key event fields are missing, use null and insufficient_event_info=true.
"""

TASK_APPROVAL_TEXT = """Analyze a pharma/regulatory approval news item and determine:
1) Whether the approval is significant for the sponsoring company.
2) Whether the approval is a surprise vs expectations (timing/controversy/delay signals).

In your analysis, focus on regulator context (e.g., FDA/EMA), approval type hints (full vs accelerated vs label expansion), and whether the article implies meaningful commercial impact.
"""

TASK_TRIAL_TEXT = """Analyze a pharma clinical trial/readout news item (e.g., Phase 1/2/3, topline, readout, first patient dosed) and determine:
1) Whether the results are significant for the sponsoring company.
2) Whether the outcome is a surprise vs what the article implies was expected (e.g., delayed review, prior concerns, “misses” vs “positive readout”).

Prioritize explicit efficacy/safety/endpoint language over boilerplate.
"""

TASK_GENERAL_TEXT = """Analyze a pharma biotech catalyst news item that is NOT clearly an approval and NOT clearly a clinical trial readout (e.g., holds, partnership/commercial milestones, failures, bankruptcies) and determine:
1) Whether it is significant for the sponsoring company.
2) Whether the situation is a surprise vs expectations implied by the article language.

Be conservative: if the article sounds like generic business ops, set low scores and insufficient_info=true.
"""

# Expected keys from PHARM_BASE_LLM_PROMPT_TEMPLATE (all must be present on the parsed dict).
PHARM_LLM_OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        "company_name",
        "regulator",
        "drug_name",
        "indication",
        "approval_type",
        "is_significant",
        "significance_score",
        "significance_reason",
        "is_surprise",
        "surprise_score",
        "surprise_reason",
        "event_outcome",
        "impact_direction",
        "impact_magnitude",
        "summary",
        "confidence",
        "insufficient_event_info",
    }
)

PHARM_LLM_REGULATOR_VALUES = frozenset({"FDA", "EMA", "MHRA", "other"})
PHARM_LLM_APPROVAL_TYPE_VALUES = frozenset(
    {
        "full_approval",
        "accelerated_approval",
        "label_expansion",
        "crl",
        "complete_response",
        "other",
    }
)
PHARM_LLM_EVENT_OUTCOME_VALUES = frozenset({"positive", "negative", "mixed", "unclear"})
PHARM_LLM_IMPACT_DIRECTION_VALUES = frozenset({"bullish", "bearish", "neutral"})


def _pharm_llm_hard_failures(parsed: Any) -> list[str]:
    """
    Structural / consistency checks on Ollama/Gemini JSON. Empty list => OK for downstream use.
    """
    out: list[str] = []
    if parsed is None:
        return ["parsed JSON is null (model returned nothing or parse failed)"]
    if not isinstance(parsed, dict):
        return [f"parsed JSON must be an object, got {type(parsed).__name__}"]

    missing = sorted(PHARM_LLM_OUTPUT_KEYS - parsed.keys())
    if missing:
        out.append(f"missing keys: {', '.join(missing)}")

    extra = sorted(parsed.keys() - PHARM_LLM_OUTPUT_KEYS)
    if extra:
        out.append(f"unexpected keys: {', '.join(extra)}")

    def _nullable_str(v: Any, field: str) -> None:
        if v is None:
            return
        if not isinstance(v, str):
            out.append(f"{field} must be string or null, got {type(v).__name__}")

    for key in (
        "company_name",
        "regulator",
        "drug_name",
        "indication",
        "approval_type",
        "significance_reason",
        "surprise_reason",
        "summary",
    ):
        _nullable_str(parsed.get(key), key)

    for key in ("is_significant", "is_surprise", "insufficient_event_info"):
        v = parsed.get(key)
        if not isinstance(v, bool):
            out.append(f"{key} must be boolean, got {type(v).__name__}")

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

    reg = parsed.get("regulator")
    if reg is not None and (not isinstance(reg, str) or reg not in PHARM_LLM_REGULATOR_VALUES):
        out.append(f"regulator must be null or one of {sorted(PHARM_LLM_REGULATOR_VALUES)}, got {reg!r}")

    ap = parsed.get("approval_type")
    if ap is not None and (not isinstance(ap, str) or ap not in PHARM_LLM_APPROVAL_TYPE_VALUES):
        out.append(
            f"approval_type must be null or one of {sorted(PHARM_LLM_APPROVAL_TYPE_VALUES)}, got {ap!r}"
        )

    eo = parsed.get("event_outcome")
    if not isinstance(eo, str) or eo not in PHARM_LLM_EVENT_OUTCOME_VALUES:
        out.append(
            f"event_outcome must be one of {sorted(PHARM_LLM_EVENT_OUTCOME_VALUES)}, got {eo!r}"
        )

    imp = parsed.get("impact_direction")
    if not isinstance(imp, str) or imp not in PHARM_LLM_IMPACT_DIRECTION_VALUES:
        out.append(
            f"impact_direction must be one of {sorted(PHARM_LLM_IMPACT_DIRECTION_VALUES)}, got {imp!r}"
        )

    # Boolean vs score consistency (prompt contract).
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

    # Outcome/direction pairing is enforced in _pharm_first_discovery_gate_fail (business gate), not here,
    # so e.g. positive+bearish logs as "impact_direction is bearish -> fail" instead of PHARM_JSON_INVALID.

    return out


def _pharm_first_discovery_gate_fail(parsed: dict[str, Any]) -> str | None:
    """Hard gate: positive outcome and bullish direction (scores feed weight, not this gate)."""
    company = parsed.get("company_name")
    drug_name = parsed.get("drug_name")
    company_txt = company.strip() if isinstance(company, str) and company.strip() else "unknown"
    drug_txt = drug_name.strip() if isinstance(drug_name, str) and drug_name.strip() else "unknown"

    if parsed.get("event_outcome") == "negative":
        return f"company={company_txt}, drug_name={drug_txt}, event_outcome is negative -> fail"
    if parsed.get("impact_direction") == "bearish":
        return f"company={company_txt}, drug_name={drug_txt}, impact_direction is bearish -> fail"

    if parsed.get("event_outcome") != "positive":
        return f"company={company_txt}, drug_name={drug_txt}, event_outcome is not positive -> fail"
    if parsed.get("impact_direction") != "bullish":
        return f"company={company_txt}, drug_name={drug_txt}, impact_direction is not bullish -> fail"

    return None


def _pharm_discovery_headline(parsed: dict[str, Any], event_class: str) -> str:
    """First line for UI tables: {company}'s {drug} {event_class} (drug omitted if missing)."""
    company = parsed.get("company_name")
    drug = parsed.get("drug_name")
    c = company.strip() if isinstance(company, str) and company.strip() else "Unknown"
    d = drug.strip() if isinstance(drug, str) and drug.strip() else ""
    ec = (event_class or "general").strip().lower()
    if d:
        return f"Pharm {c}'s {d} {ec}"
    return f"{c}'s {ec}"


def _pharm_discovery_detail_paragraph(parsed: dict[str, Any]) -> str:
    """Single paragraph: summary, significance_reason, surprise_reason (non-empty parts only)."""
    pieces: list[str] = []
    for key in ("summary", "significance_reason", "surprise_reason"):
        raw = parsed.get(key)
        if not isinstance(raw, str):
            continue
        s = raw.strip()
        if not s:
            continue
        while s.endswith("."):
            s = s[:-1].rstrip()
        if s:
            pieces.append(s)
    if not pieces:
        return ""
    return ". ".join(pieces) + "."


# LLM fields appended to Discovery.explanation after headline + narrative (fixed order for UI).
PHARM_EXPLANATION_METADATA_KEYS: tuple[str, ...] = (
    "drug_name",
    "confidence",
    "event_outcome",
    "impact_direction",
    "impact_magnitude",
    "indication",
    "insufficient_event_info",
    "is_significant",
    "is_surprise",
    "regulator",
    "significance_score",
    "surprise_score",
)


def _pharm_format_explanation_field_value(value: Any) -> str:
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


def _pharm_discovery_metadata_lines(parsed: dict[str, Any]) -> str:
    lines = [f"{key}: {_pharm_format_explanation_field_value(parsed.get(key))}" for key in PHARM_EXPLANATION_METADATA_KEYS]
    return "\n".join(lines)


def _pharm_brief_trade_summary(listed_ticker: str, event_class: str) -> str:
    """
    One-line summary for Trade.explanation (analyze_discovery uses split(' | ')[0]).
    Keep short; full headline + narrative live in the body after ' | '.
    """
    sym = (listed_ticker or "").strip().upper()[:12] or "?"
    ec = (event_class or "general").strip().lower()[:32]
    return f"{sym} · PHARM {ec}"


def _pharm_build_discovery_explanation(
    parsed: dict[str, Any], event_class: str, *, listed_ticker: str, article_link: str
) -> str:
    headline = _pharm_discovery_headline(parsed, event_class)
    article_title = discovery_trade_explanation_lead(f"Article: {headline}")

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

    parts: list[str] = [article_title]
    if article_link:
        parts.append(article_link)
    parts.append(f"headline: {headline}")

    for key, label in (
        ("summary", "summary"),
        ("significance_reason", "significance"),
        ("surprise_reason", "surprise"),
    ):
        cleaned = _clean_sentence(parsed.get(key))
        if cleaned:
            parts.append(f"{label}: {cleaned}")

    scores = (
        "scores: "
        f"outcome={_pharm_format_explanation_field_value(parsed.get('event_outcome'))} "
        f"dir={_pharm_format_explanation_field_value(parsed.get('impact_direction'))} "
        f"mag={_pharm_format_explanation_field_value(parsed.get('impact_magnitude'))} "
        f"sig={_pharm_format_explanation_field_value(parsed.get('significance_score'))} "
        f"sur={_pharm_format_explanation_field_value(parsed.get('surprise_score'))} "
        f"conf={_pharm_format_explanation_field_value(parsed.get('confidence'))} "
        f"drug={_pharm_format_explanation_field_value(parsed.get('drug_name'))} "
        f"reg={_pharm_format_explanation_field_value(parsed.get('regulator'))} "
        f"ticker={listed_ticker}"
    )
    parts.append(scores)
    return " | ".join(parts)


def _score_row(row: dict[str, str]) -> dict[str, Any]:
    title = row.get("title") or ""
    summary = row.get("summary") or ""
    blob = f"{title} {summary}".lower()

    green_hits: list[str] = []
    red_hits: list[str] = []
    score = 0

    for term in PHARM_GREEN_TERMS_STRONG:
        if term in blob:
            green_hits.append(term)
            score += 2
    for term in PHARM_GREEN_TERMS_WEAK:
        if term in blob:
            green_hits.append(term)
            score += 1
    for term in PHARM_RED_TERMS_STRONG:
        if term in blob:
            red_hits.append(term)
            score -= 3
    for term in PHARM_RED_TERMS_WEAK:
        if term in blob:
            red_hits.append(term)
            score -= 1

    suppressed = any(term in blob for term in PHARM_SUPPRESS_TERMS)
    negative_dominant = any(term in blob for term in PHARM_NEGATIVE_DOMINANT_TERMS)
    if negative_dominant and score > 0:
        score = 0

    return {
        **row,
        "green_hits": green_hits,
        "red_hits": red_hits,
        "suppressed": suppressed,
        "negative_dominant": negative_dominant,
        "score": score,
    }


def _classify_event(row: dict[str, Any]) -> str:
    title_blob = (row.get("title") or "").lower()
    summary_blob = (row.get("summary") or "").lower()

    # Title-first negative-risk overrides to avoid misrouting ops/safety stories.
    title_general_overrides = (
        "bankruptcy",
        "clinical hold",
        "fda refusal",
        "deaths",
        "liver injuries",
        "cancels",
        "cancelled",
        "canceled",
        "layoff",
        "layoffs",
        "fund",
    )
    if any(term in title_blob for term in title_general_overrides):
        return "general"

    # Primary classification from title text.
    title_approval_hits = sum(1 for term in APPROVAL_CLASS_TERMS if term in title_blob)
    title_trial_hits = sum(1 for term in TRIAL_CLASS_TERMS if term in title_blob)
    if title_approval_hits > title_trial_hits and title_approval_hits > 0:
        return "approval"
    if title_trial_hits > title_approval_hits and title_trial_hits > 0:
        return "trial"
    if title_approval_hits > 0 and title_trial_hits > 0:
        return "approval"

    # Fallback: use summary only when title is inconclusive.
    summary_approval_hits = sum(1 for term in APPROVAL_CLASS_TERMS if term in summary_blob)
    summary_trial_hits = sum(1 for term in TRIAL_CLASS_TERMS if term in summary_blob)
    if summary_approval_hits > summary_trial_hits and summary_approval_hits > 0:
        return "approval"
    if summary_trial_hits > summary_approval_hits and summary_trial_hits > 0:
        return "trial"
    if summary_approval_hits > 0 and summary_trial_hits > 0:
        return "approval"

    return "general"


def _prompt_for_class(event_class: str) -> str:
    if event_class == "approval":
        return TASK_APPROVAL_KEY
    if event_class == "trial":
        return TASK_TRIAL_KEY
    return TASK_GENERAL_KEY


def _build_pharm_llm_prompt(*, task_key: str, article_json: dict[str, Any]) -> str:
    """
    Build the (future) PHARM LLM prompt for a given task key and article input JSON.
    Note: this function does not call the LLM yet; it only formats the prompt.
    """
    if task_key == TASK_APPROVAL_KEY:
        task_text = TASK_APPROVAL_TEXT
    elif task_key == TASK_TRIAL_KEY:
        task_text = TASK_TRIAL_TEXT
    else:
        task_text = TASK_GENERAL_TEXT

    # Keep article JSON compact; some feeds have very long descriptions.
    payload = dict(article_json or {})
    # Best-effort truncation for long text fields (do not assume exact keys).
    for k, v in list(payload.items()):
        if isinstance(v, str) and len(v) > 6000:
            payload[k] = v[:6000]

    return (
        PHARM_BASE_LLM_PROMPT_TEMPLATE
        .replace("__TASK_TEXT__", task_text.strip())
        .replace("__ARTICLE_JSON__", json_dumps_compact(payload))
    )


def json_dumps_compact(obj: Any) -> str:
    import json

    # Ensure ASCII for predictable downstream parsing/logs.
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def _is_shortlist_candidate(scored: dict[str, Any]) -> bool:
    if bool(scored.get("suppressed")):
        return False

    title_blob = ((scored.get("title") or "") + " " + (scored.get("summary") or "")).lower()

    # High-impact phrases: include regardless of feed source (not RSS-specific).
    hard_triggers = (
        "fda approves",
        "fda nod",
        "fda greenlight",
        "clinical hold",
        "bankruptcy",
        "fails pivotal trial",
        "misses phase 3",
        "deaths",
    )
    if any(t in title_blob for t in hard_triggers):
        return True

    return int(scored.get("score") or 0) >= PHARM_SHORTLIST_SCORE_MIN


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("link") or "").strip().lower()
        if not key:
            key = f"{row.get('published_at','')}|{(row.get('title') or '').strip().lower()}"
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


class Pharm(AdvisorBase):
    """Stub advisor for future pharma catalyst discovery logic."""

    def discover(self, sa):
        logger.info("PHARM phase-1 shortlist run starting")
        # Mirror other advisors: use the previous SmartAnalysis start time as the lookback lower bound.
        # Falls back to a fixed window when there is no previous SA (e.g. standalone runs with an unsaved SmartAnalysis).
        prev_ts = self.get_previous_sa_timestamp(sa, username=getattr(sa, "username", None))
        since_dt = _parse_since_utc(prev_ts)
        if since_dt is None:
            since_dt = datetime.now(UTC) - timedelta(days=7)
            logger.info("PHARM no previous SA found; fallback since=%s", since_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        since_iso = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info("PHARM feed window since=%s", since_iso)
        feed_calls: tuple[tuple[str, Any], ...] = (
            ("fda_press_releases", rrs_fda),
            ("prnewswire_health", rrs_prnewswire_health),
            ("prnewswire_biotech", rrs_prnewswire_biotech),
            ("biospace_fda", rrs_biospace_fda),
            ("biospace_drug_development", rrs_biospace_drug_development),
            ("endpoints", rrs_endpoints),
            ("fiercebiotech_all", rrs_fiercebiotech),
        )

        all_rows: list[dict[str, Any]] = []
        for source_name, fn in feed_calls:
            try:
                rows = fn(limit=25, since=since_iso)
                logger.info("PHARM feed %s rows=%d", source_name, len(rows))
                all_rows.extend(rows)
            except Exception as exc:
                logger.warning("PHARM feed %s failed: %s", source_name, exc)

        unique_rows = _dedupe_rows(all_rows)
        scored_rows = [_score_row(r) for r in unique_rows]
        shortlist = [r for r in scored_rows if _is_shortlist_candidate(r)]
        for row in shortlist:
            event_class = _classify_event(row)
            row["event_class"] = event_class
            row["prompt_key"] = _prompt_for_class(event_class)
        shortlist.sort(key=lambda r: (int(r.get("score") or 0), r.get("published_at") or ""), reverse=True)

        event_class_filter = os.getenv("PHARM_EVENT_CLASS", "approval").strip().lower()
        if event_class_filter not in ("approval", "trial", "general"):
            logger.warning("PHARM invalid PHARM_EVENT_CLASS=%r; using approval", event_class_filter)
            event_class_filter = "approval"
        task_default_for_class = {
            "approval": TASK_APPROVAL_KEY,
            "trial": TASK_TRIAL_KEY,
            "general": TASK_GENERAL_KEY,
        }[event_class_filter]

        class_match_rows = [
            (shortlist_idx, row)
            for shortlist_idx, row in enumerate(shortlist, start=1)
            if row.get("event_class") == event_class_filter
        ]
        neg_score_skipped = sum(
            1 for _, row in class_match_rows if int(row.get("score") or 0) < 0
        )
        if neg_score_skipped:
            logger.info(
                "PHARM skip LLM for %d %s shortlist row(s) with score < 0",
                neg_score_skipped,
                event_class_filter,
            )
        llm_shortlist = [
            (shortlist_idx, row)
            for shortlist_idx, row in class_match_rows
            if int(row.get("score") or 0) >= 0
        ]
        target_shortlist_idx_raw = os.getenv("PHARM_DEBUG_SHORTLIST_IDX", "").strip()
        if target_shortlist_idx_raw:
            try:
                target_shortlist_idx = int(target_shortlist_idx_raw)
            except ValueError:
                target_shortlist_idx = None
                logger.warning("PHARM invalid PHARM_DEBUG_SHORTLIST_IDX=%r (must be integer)", target_shortlist_idx_raw)
            if target_shortlist_idx is not None:
                llm_shortlist = [
                    (shortlist_idx, row)
                    for shortlist_idx, row in llm_shortlist
                    if shortlist_idx == target_shortlist_idx
                ]
                logger.info(
                    "PHARM debug filter active: PHARM_DEBUG_SHORTLIST_IDX=%d "
                    "PHARM_EVENT_CLASS=%s (matches=%d)",
                    target_shortlist_idx,
                    event_class_filter,
                    len(llm_shortlist),
                )
        if llm_shortlist:
            logger.info(
                "PHARM processing %s candidates: count=%d",
                event_class_filter,
                len(llm_shortlist),
            )

        for idx, (shortlist_idx, candidate) in enumerate(llm_shortlist, start=1):
            logger.info(
                "PHARM %s candidate %d/%d (shortlist %d/%d) | %s | %s",
                event_class_filter,
                idx,
                len(llm_shortlist),
                shortlist_idx,
                len(shortlist),
                candidate.get("published_at") or "",
                candidate.get("title") or "",
            )
            task_key = str(candidate.get("prompt_key") or task_default_for_class)
            article_payload = {
                "source": candidate.get("source"),
                "published_at": candidate.get("published_at"),
                "title": candidate.get("title"),
                "summary": candidate.get("summary"),
                "link": candidate.get("link"),
                "event_class": candidate.get("event_class"),
                "score": candidate.get("score"),
            }
            prompt = _build_pharm_llm_prompt(task_key=task_key, article_json=article_payload)
            _model, parsed = self.ask_ollama(prompt)

            hard_fails = _pharm_llm_hard_failures(parsed)
            if hard_fails:
                for reason in hard_fails:
                    logger.info("PHARM_JSON_INVALID: %s", reason)
                continue

            business_fail = _pharm_first_discovery_gate_fail(parsed)
            if business_fail:
                logger.info(business_fail)
                continue

            # Deterministic company -> ticker resolution (shared with FDA advisor).
            company_name = (parsed or {}).get("company_name") if isinstance(parsed, dict) else None
            if isinstance(company_name, str):
                company_name = company_name.strip() or None
            else:
                company_name = None

            resolved_ticker = None
            if company_name:
                try:
                    # Reuse existing deterministic resolver used by FDA advisor.
                    from core.services.advisors.fda import match_company_to_symbol

                    resolved_ticker, _ = match_company_to_symbol(company_name)
                except Exception as exc:
                    logger.warning("PHARM ticker resolution failed for %r: %s", company_name, exc)

            if not resolved_ticker:
                logger.info("PHARM skip discovery: no ticker for company_name=%r", company_name)
                continue

            if not self.allow_discovery(resolved_ticker, period=24):
                logger.info("PHARM skip discovery: allow_discovery false for %s", resolved_ticker)
                continue

            weight, raw_comp, wdetail = feed_discovery_weight_from_parsed(parsed, resolved_ticker)
            if weight is None:
                logger.info(
                    "PHARM skip discovery: composite raw=%s < %s | %s | %s",
                    raw_comp,
                    FEED_DISCOVERY_COMPOSITE_MIN,
                    resolved_ticker,
                    wdetail,
                )
                continue
            logger.info("PHARM discovery weight | %s | %s -> weight=%s", resolved_ticker, wdetail, weight)

            row_event_class = str(candidate.get("event_class") or "general")
            explanation = _pharm_build_discovery_explanation(
                parsed,
                row_event_class,
                listed_ticker=resolved_ticker,
                article_link=(candidate.get("link") or "").strip(),
            )

            sell_instructions = [
                ("PERCENTAGE_DIMINISHING", 1.30, 7),
                ("PERCENTAGE_AUGMENTING", 0.90, 20),
                ("DESCENDING_TREND", -0.20, None),
                ("NOT_TRENDING", None, None),
            ]

            stock = self.discovered(
                sa,
                resolved_ticker,
                explanation,
                sell_instructions,
                weight=weight,
            )
            if stock:
                logger.info("PHARM discovered %s (%s)", resolved_ticker, row_event_class)
            else:
                logger.warning("PHARM discovered() returned None for %s", resolved_ticker)

        logger.info(
            "PHARM shortlist complete: total=%d unique=%d shortlisted=%d",
            len(all_rows),
            len(unique_rows),
            len(shortlist),
        )
        return


def run_pharm_standalone():
    """
    Minimal entry point for the `run_pharm` management command.

    No params yet; resolves advisor class via Advisor.python_class and runs discover()
    with a minimal SmartAnalysis() instance.
    """
    from core.services import advisors as advisor_modules
    from core.models import Advisor, SmartAnalysis

    try:
        advisor_row = Advisor.objects.get(name="PHARM")
    except Advisor.DoesNotExist:
        return None, "PHARM advisor not found in Advisor table"

    module_name = advisor_row.python_class.lower()
    module = getattr(advisor_modules, module_name)
    PythonClass = getattr(module, advisor_row.python_class)

    sa = SmartAnalysis()
    impl = PythonClass(advisor_row)
    impl.discover(sa)

    return "PHARM discover() stub completed", None


register(name="PHARM", python_class="Pharm")
