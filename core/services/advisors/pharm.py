"""
PHARM advisor stub.

Purpose:
- Placeholder advisor for pharma/regulatory RSS + trial catalyst pipeline.
- Mirrors the standalone entry pattern used by EDGAR.

Run:
    python manage.py run_pharm
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import html
import logging
import re
from typing import Any

from core.services.advisors.advisor import AdvisorBase, register

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
  "ticker": "string|null",

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
  "insufficient_event_info": true/false,
  "ticker_unresolved": true/false
}

SCORING RULES (be conservative; lean lower when uncertain):
Significance (impact to company value):
1 = minor/niche; 3 = meaningful but not transformational; 5 = transformational

Surprise (relative to market expectations):
1 = fully expected/routine; 3 = some uncertainty; 5 = materially unexpected positive outcome

BOOLEAN CONSISTENCY:
- is_significant must be true only if significance_score >= 4
- is_surprise must be true only if surprise_score >= 4

TICKER RULE (reduce hallucinations):
- Only output a ticker if it is explicitly implied by the provided article content.
- If the ticker is not reliably present in the provided content, set ticker to null and ticker_unresolved=true.

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


def json_dumps_pretty(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=True, indent=2, sort_keys=True)


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

        # Temporary debug path: run Gemini on the first shortlisted row only.
        if shortlist:
            first = shortlist[0]
            task_key = str(first.get("prompt_key") or TASK_GENERAL_KEY)
            article_payload = {
                "source": first.get("source"),
                "published_at": first.get("published_at"),
                "title": first.get("title"),
                "summary": first.get("summary"),
                "link": first.get("link"),
                "event_class": first.get("event_class"),
                "score": first.get("score"),
            }
            prompt = _build_pharm_llm_prompt(task_key=task_key, article_json=article_payload)
            print("\n--- PHARM GEMINI PROMPT START ---")
            print(prompt)
            print("--- PHARM GEMINI PROMPT END ---\n")
            model, parsed = self.ask_gemini(prompt, timeout=120.0, use_search=False)
            print("\n--- PHARM GEMINI RESPONSE START ---")
            print(f"model={model or 'None'}")
            print(json_dumps_pretty(parsed) if parsed is not None else "null")
            print("--- PHARM GEMINI RESPONSE END ---\n")

        logger.info(
            "PHARM shortlist complete: total=%d unique=%d shortlisted=%d",
            len(all_rows),
            len(unique_rows),
            len(shortlist),
        )
        for idx, row in enumerate(shortlist, start=1):
            logger.info(
                "[%d/%d] score=%+d | class=%s | prompt=%s | %s | %s | %s",
                idx,
                len(shortlist),
                int(row.get("score") or 0),
                row.get("event_class") or "general",
                row.get("prompt_key") or TASK_GENERAL_KEY,
                row.get("source") or "",
                row.get("published_at") or "",
                row.get("title") or "",
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
