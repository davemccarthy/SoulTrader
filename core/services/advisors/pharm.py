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

PRN_BIO_ENTITY_CUES = (
    "therapeutics",
    "biotech",
    "biosciences",
    "pharma",
    "oncology",
    "rare disease",
    "gene therapy",
    "clinical",
    "drug",
    "trial",
)

PRN_CLINICAL_REG_CUES = (
    "phase 1",
    "phase 2",
    "phase 3",
    "trial",
    "topline",
    "readout",
    "fda",
    "nda",
    "bla",
    "pdufa",
    "approval",
)

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

PROMPT_APPROVAL = "PHARM_APPROVAL_PROMPT"
PROMPT_TRIAL = "PHARM_TRIAL_PROMPT"
PROMPT_GENERAL = "PHARM_GENERAL_PROMPT"


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
    blob = f"{row.get('title') or ''} {row.get('summary') or ''}".lower()
    approval_hits = sum(1 for term in APPROVAL_CLASS_TERMS if term in blob)
    trial_hits = sum(1 for term in TRIAL_CLASS_TERMS if term in blob)
    if approval_hits > trial_hits and approval_hits > 0:
        return "approval"
    if trial_hits > approval_hits and trial_hits > 0:
        return "trial"
    if approval_hits > 0 and trial_hits > 0:
        # Prefer approval when both are present (e.g., "FDA nod after phase 3")
        return "approval"
    return "general"


def _prompt_for_class(event_class: str) -> str:
    if event_class == "approval":
        return PROMPT_APPROVAL
    if event_class == "trial":
        return PROMPT_TRIAL
    return PROMPT_GENERAL


def _is_shortlist_candidate(scored: dict[str, Any]) -> bool:
    if bool(scored.get("suppressed")):
        return False

    title_blob = ((scored.get("title") or "") + " " + (scored.get("summary") or "")).lower()
    source = (scored.get("source") or "").strip().lower()

    if source == "prnewswire_health":
        has_entity_cue = any(t in title_blob for t in PRN_BIO_ENTITY_CUES)
        has_clin_reg_cue = any(t in title_blob for t in PRN_CLINICAL_REG_CUES)
        if not (has_entity_cue and has_clin_reg_cue):
            return False

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

    threshold = 2
    if source == "prnewswire_health":
        threshold = 3
    return int(scored.get("score") or 0) >= threshold


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
        since_dt = datetime.now(UTC) - timedelta(days=7)
        since_iso = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info("PHARM feed window since=%s", since_iso)
        feed_calls: tuple[tuple[str, Any], ...] = (
            ("fda_press_releases", rrs_fda),
            ("prnewswire_health", rrs_prnewswire_health),
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
                row.get("prompt_key") or PROMPT_GENERAL,
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
