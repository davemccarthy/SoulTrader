"""
Edgar advisor (ED-8): 8-K earnings–related filings, filters 1–6, analyze_8k.

Test independently: python manage.py run_edgar

Production: discover() will fetch 8-Ks, run filters (1–6, sector, reception),
and call self.discovered(...) for passing stocks.
"""

import logging
import os
import html
import re
import time
from decimal import Decimal
from collections import Counter
from datetime import date, datetime, timedelta, timezone as dt_timezone
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Tuple
import requests
import yfinance as yf
from django.conf import settings
from edgar import Company, find, set_identity, get_latest_filings
from datetime import date, datetime, time as dt_time, timedelta, timezone as dt_timezone
from core.services.advisors.advisor import AdvisorBase, register

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SEC identity (mandatory)
# ---------------------------------------------------------------------------
set_identity("Dave McCarthy dave@klynt.com")

# ---------------------------------------------------------------------------
# 8-K helpers and constants
# ---------------------------------------------------------------------------
_CIK_TO_TICKER_CACHE: Dict[str, Optional[str]] = {}


def cik_to_ticker(cik: str) -> Optional[str]:
    """Map CIK to ticker symbol using edgar Company helper."""
    cik = str(cik).zfill(10)
    if cik in _CIK_TO_TICKER_CACHE:
        return _CIK_TO_TICKER_CACHE[cik]
    try:
        company = Company(cik)
        ticker = company.get_ticker()
        _CIK_TO_TICKER_CACHE[cik] = ticker
        return ticker
    except Exception:
        _CIK_TO_TICKER_CACHE[cik] = None
        return None


def _filing_date_or_none(filing) -> Optional[date]:
    """
    Normalize filing.filing_date into a date, or return None if unusable.
    """
    fd = getattr(filing, "filing_date", None)
    if fd is None:
        return None
    if isinstance(fd, date):
        return fd
    if isinstance(fd, str):
        try:
            return date.fromisoformat(fd[:10])
        except ValueError:
            return None
    if hasattr(fd, "date") and callable(getattr(fd, "date")):
        try:
            return fd.date()
        except Exception:
            return None
    try:
        return date.fromisoformat(str(fd)[:10])
    except Exception:
        return None


def _filing_datetime_utc(dt):
    """Normalize a datetime to UTC. Naive datetimes are assumed US/Eastern (SEC)."""
    if dt is None:
        return None
    et = ZoneInfo("America/New_York")
    utc = dt_timezone.utc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=et).astimezone(utc)
    else:
        dt = dt.astimezone(utc)
    return dt


def _filing_datetime(filing):
    """
    Return filing datetime in UTC for sorting/dedupe, or None.
    Uses header.acceptance_datetime / accepted if available (assumed US/Eastern if naive),
    else filing_date at start-of-day Eastern; normalizes to UTC.
    """
    header = getattr(filing, "header", None)
    if header is not None:
        acc = getattr(header, "acceptance_datetime", None) or getattr(header, "accepted", None)
        if acc is not None:
            if hasattr(acc, "hour"):  # datetime-like
                return _filing_datetime_utc(acc)
            if hasattr(acc, "strftime") and hasattr(acc, "date"):
                return _filing_datetime_utc(acc)
            if isinstance(acc, str) and len(acc) >= 12:  # SEC YYYYMMDDHHMMSS
                try:
                    dt = datetime(
                        int(acc[:4]), int(acc[4:6]), int(acc[6:8]),
                        int(acc[8:10]), int(acc[10:12]), int(acc[12:14]) if len(acc) >= 14 else 0,
                    )
                    return _filing_datetime_utc(dt)
                except (ValueError, TypeError):
                    pass
    fd = getattr(filing, "filing_date", None)
    if fd is None:
        return None
    if hasattr(fd, "date"):
        d = fd.date() if callable(getattr(fd, "date")) else fd
    elif isinstance(fd, str):
        try:
            d = date.fromisoformat(fd[:10])
        except ValueError:
            return None
    else:
        d = fd
    if isinstance(d, date) and not isinstance(d, datetime):
        dt = datetime.combine(d, datetime.min.time())
    else:
        dt = d
    return _filing_datetime_utc(dt)


def _get_ex99_text(filing) -> str:
    """Return concatenated EX-99.x exhibit text from filing, or empty string."""
    parts = []
    if hasattr(filing, "exhibits") and filing.exhibits:
        for ex in filing.exhibits:
            if "99." in str(ex).lower():
                try:
                    parts.append(ex.text())
                except Exception:
                    continue
    return " ".join(parts)


# Filing filter (Filter 2): red/green flags
RED_FLAGS_SEVERE = {
    "chapter 11": -10, "bankruptcy filing": -10, "receivership": -10,
    "substantial doubt": -8, "ability to continue as a going concern": -8,
    "going concern": -8, "item 4.02": -8, "non-reliance on financial statements": -8,
    "restatement of previously issued financial statements": -7,
    "should no longer be relied upon": -7,
}
RED_FLAGS_MODERATE = {
    "resignation": -2, "ceo termination": -2, "cfo termination": -2,
    "termination of employment": -2, "departure": -2, "material weakness": -2,
    "internal control deficiency": -2, "sec investigation": -2, "sec inquiry": -2,
    "class action": -1, "material litigation": -2, "significant litigation": -2,
    "weak guidance": -2, "cautious outlook": -2, "cautious guidance": -2,
    "cautious stance": -2, "projected decline": -2, "expects decline": -2,
    "lower outlook": -2, "lowered expectations": -2, "decline in sales": -1,
    "anticipated softness": -2, "tempered outlook": -2, "mid-single-digit decline": -2,
    "single-digit decline": -1, "macroeconomic headwinds": -1, "promotional environment": -1,
    "disciplined inventory management": -1, "challenging macroeconomic environment": -2,
    "strategic alternatives": -2, "restructuring charges": -2, "liquidity constraints": -2,
    "breach of loan covenants": -2, "softer start": -2, "down approximately": -1,
    "net sales to be down": -1,
}
GREEN_FLAGS = {
    "raised guidance": +3, "increased outlook": +3, "increased guidance": +3,
    "initiated guidance": +1, "provided guidance for": +1, "dividend increase": +2,
    "increased dividend": +2, "share repurchase authorization": +2, "buyback program": +2,
    "above expectations": +2, "exceeded estimates": +2, "record revenue": +2,
    "record ebitda": +1, "record earnings": +1, "stronger than expected results": +2,
    "higher than expected revenue": +2, "return to profitability": +2, "driven by ai": +1,
    "eliminated debt": +1, "positive operating cash flow": +1, "strong operating leverage": +1,
    "meaningful margin expansion": +2, "revenue growth": +1, "revenue increase": +1,
    "exceptional revenue growth": +3, "robust cash generation": +1,
    "improved operating leverage": +1, "scalability and cost efficiency": +1,
    "successfully launched": +1, "growing demand ": +1, "adjusted ebitda profitability": +1,
    "backlog grew": +1, "in line with previous estimate": +1, "in line with filed estimates": +1,
    "listing compliance": +1, "compliance deadline": +1, "debt reduction": +1,
    "reduction of total debt": +1, "filed prior to": +1, "strongest year": +2,
    "strongest quater": +2,
}

# Filing filter (Filter 4): earnings release structure
EARNINGS_CONTEXT = ["earnings", "financial results", "results of operations", "quarter ended", "fiscal quarter"]
TABLE_HINTS = ["gaap", "non-gaap", "q/q", "y/y", "%"]
EPS_NUMBER_PATTERN = re.compile(r"(earnings per share|eps|diluted).{0,60}?(-?\d+\.\d+)", re.IGNORECASE)


def _filter4_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _filter4_has_earnings_context(text: str) -> bool:
    return any(k in text for k in EARNINGS_CONTEXT)


def _filter4_has_eps_evidence(text: str) -> bool:
    return bool(EPS_NUMBER_PATTERN.search(text))


def _filter4_numeric_density(text: str) -> int:
    return len(re.findall(r"-?\$?\d+(?:,\d{3})*(?:\.\d+)?", text))


def _filter4_has_table_structure(text: str) -> bool:
    return sum(1 for k in TABLE_HINTS if k in text) >= 2


def _filter4_has_comparables(text: str) -> bool:
    words = re.findall(r"(revenue|earnings per share|eps|net income)", text, re.IGNORECASE)
    counts = Counter(w.lower() for w in words)
    return any(v >= 2 for v in counts.values())


def _earnings_release_filter(raw_text: str) -> dict:
    """EX-99 structure check. Returns dict with pass, reason, and optional checks."""
    text = _filter4_normalize(raw_text)
    checks = {
        "earnings_context": _filter4_has_earnings_context(text),
        "eps_evidence": _filter4_has_eps_evidence(text),
        "numeric_density": _filter4_numeric_density(text),
        "table_structure": _filter4_has_table_structure(text),
        "comparables": _filter4_has_comparables(text),
    }
    if not checks["earnings_context"]:
        return {"pass": False, "reason": "no earnings context", "checks": checks}
    if not checks["eps_evidence"]:
        return {"pass": False, "reason": "no EPS evidence", "checks": checks}
    if checks["numeric_density"] < 15:
        return {"pass": False, "reason": "low numeric density", "checks": checks}
    score = sum([checks["table_structure"], checks["comparables"]])
    return {"pass": score >= 1, "score": score, "reason": "ok" if score >= 1 else "soft fail", "checks": checks}


# Financial filter (Filter 3)
FILTER3_PE_MAX = 100
FILTER3_MIN_CAP = 300e6
FILTER3_PRICE_MIN = 5.0
FILTER3_PRICE_MAX = 200.0
FILTER3_PREV_DAY_GAIN_FAIL_PCT = 10.0
# Single sector/industry list: (sector_substring, industry_substring_or_None, "hard_fail"|"weight").
# hard_fail: filter_financials returns False. weight: weigh_results applies -0.2.
# If industry is None: match when sector_substring in sector or industry. If industry set: match when both in sector and industry.
SECTOR_LIST = (
    ("cannabis", None, "hard_fail"),
    ("consumer cyclical", None, "weight"),
    ("real estate", None, "weight"),
    ("utilities", None, "weight"),
    ("technology", "software", "weight"),
    ("financial services", "insurance", "weight"),
)

# EPS beat strength thresholds (on eps_score)
BEAT_THRESHOLD = 8.0
STRONG_BEAT_THRESHOLD = 20.0
SCORE_THRESHOLD = 60

# ---------------------------------------------------------------------------
# EPS helpers (Filter 6-style logic)
# ---------------------------------------------------------------------------


def _latest_quarter_end_for_date(d: date) -> date:
    """Quarter-end date corresponding to the quarter this filing reports."""
    y, m = d.year, d.month
    if m in (1, 2, 3):
        return date(y - 1, 12, 31)
    if m in (4, 5, 6):
        return date(y, 3, 31)
    if m in (7, 8, 9):
        return date(y, 6, 30)
    return date(y, 9, 30)


def quarter_label_for_filing_date(filing_date: date) -> str:
    """
    Human-readable quarter label (e.g. 'Q4 2025') for this filing date,
    using the same quarter-end mapping as _latest_quarter_end_for_date.
    """
    qe = _latest_quarter_end_for_date(filing_date)
    if qe.month == 3:
        q = "Q1"
    elif qe.month == 6:
        q = "Q2"
    elif qe.month == 9:
        q = "Q3"
    else:
        q = "Q4"
    return f"{q} {qe.year}"


def get_eps_for_report_quarter(ticker: str, report_date: date) -> Optional[Dict]:
    """
    Fetch Alpha Vantage EARNINGS; return record for the quarter that matches the filing's
    report quarter (by fiscalDateEnding). Returns None on error or no match.
    """
    api_key = getattr(settings, "ALPHAVANTAGE_API_KEY", None) or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return None
    latest_quarter_end = _latest_quarter_end_for_date(report_date)
    target = latest_quarter_end.isoformat()
    url = "https://www.alphavantage.co/query"
    params = {"function": "EARNINGS", "symbol": ticker.upper(), "apikey": api_key}
    try:
        time.sleep(0.25)
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            logger.warning("Alpha Vantage rate limit (429): ticker=%s", ticker)
            return None
        resp.raise_for_status()
        data = resp.json()
        if "Error Message" in data or "Note" in data:
            return None
        if "Information" in data:
            logger.info("Alpha Vantage (Information) for %s: %s", ticker, data["Information"][:250])
            return None
        quarterly = data.get("quarterlyEarnings") or []
        for r in quarterly:
            fd = r.get("fiscalDateEnding") or ""
            if fd[:10] == target if len(fd) >= 10 else fd == target:
                return r
        return None
    except Exception:
        return None


def _extract_eps_from_xbrl(filing) -> Optional[float]:
    """Try diluted EPS from 8-K XBRL if available."""
    try:
        xbrl = filing.xbrl()
    except Exception:
        return None
    if not xbrl:
        return None
    concepts = ("EarningsPerShareDiluted", "EarningsPerShare", "EarningsPerShareBasic")
    for concept in concepts:
        try:
            facts = xbrl.query().by_concept(concept, exact=False).execute()
            if not facts:
                continue
            duration_facts = [f for f in facts if f.get("period_type") == "duration"]
            if not duration_facts:
                duration_facts = facts
            sorted_facts = sorted(
                duration_facts,
                key=lambda f: f.get("period_end", "") or "",
                reverse=True,
            )
            val = sorted_facts[0].get("numeric_value") or sorted_facts[0].get("value")
            if val is not None:
                return float(val)
        except Exception:
            continue
    return None


_EPS_PATTERNS = [
    r"(?:diluted|basic)\s+EPS\s+(?:of\s+|were\s+)?\$?\s*([\d,]+\.?\d{2,})",
    r"(?:GAAP\s+)?EPS\s+(?:of\s+|were\s+)?\$?\s*([\d,]+\.?\d{2,})",
    r"earnings\s+per\s+(?:common\s+)?share\s+(?:\(?EPS\)?)?\s+(?:of\s+|were\s+)?\$?\s*([\d,]+\.?\d{2,})",
    r"\$([\d,]+\.?\d{2,})\s+per\s+(?:diluted|basic)?\s*(?:share|diluted share)",
    r"(?:Diluted|diluted)\s+.*?(?:per\s+share|operations).*?\$\s*([\d,]+\.?\d{2,})",
    r"Continuing\s+operations\s+\$\s*([\d,]+\.?\d{2,})",
]
_NONGAAP_PATTERNS = [
    r"(?:non[-\s]?GAAP|adjusted)\s+EPS\s+(?:of\s+)?\$?\s*([\d,]+\.?\d{2,})",
]


def _normalize_exhibit_text(raw: str) -> str:
    """Strip HTML and normalize whitespace for EPS regex matching."""
    if not raw:
        return ""
    s = raw
    try:
        s = html.unescape(s)
    except Exception:
        pass
    s = re.sub(r"<script[^>]*>.*?</script>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<style[^>]*>.*?</style>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_eps_from_text(text: str) -> Optional[float]:
    """Extract one EPS value from exhibit 99 text; prefer diluted/GAAP, then non-GAAP."""
    text = _normalize_exhibit_text(text)
    if not text:
        return None
    for pattern in _EPS_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if abs(val) < 1e6 and val != 0:
                    return val
            except ValueError:
                pass
    for pattern in _NONGAAP_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if abs(val) < 1e6 and val != 0:
                    return val
            except ValueError:
                pass
    return None


def get_actual_eps_from_8k(filing, verbose: bool = False) -> Tuple[Optional[float], str]:
    """
    Get reported EPS from 8-K: try XBRL first, then exhibit 99 text.
    Returns (eps_value, source) where source is "xbrl" or "text" or "none".
    """
    eps = _extract_eps_from_xbrl(filing)
    if eps is not None:
        if verbose:
            logger.info("  [8-K XBRL] EPS %.2f", eps)
        return eps, "xbrl"
    text = _get_ex99_text(filing)
    eps = _extract_eps_from_text(text)
    if eps is not None:
        if verbose:
            logger.info("  [8-K text] EPS %.2f", eps)
        return eps, "text"
    return None, "none"


# ---------------------------------------------------------------------------
# EX-99.1 LLM (4-ducks: past, future, expectation, market)
# ---------------------------------------------------------------------------

FOUR_DUCKS_PROMPT_LABELS = """You are an equity analyst.
Given an earnings press release (EX-99.1 from an 8-K), evaluate the following four metrics independently.
Use only the information in the document.
Ignore stock price movement.
Ignore analyst consensus unless explicitly mentioned in the text.
Do not speculate beyond the text.

TASK
For each of the four metrics below, output a single label: "negative", "neutral", "positive", or "strong_positive".

- strong_positive: Clearly strong; multiple positive themes, confident tone, materially better than prior/expectations.
- positive: Net positive; more supportive than concerning; above average or improved.
- neutral: Mixed or ambiguous; balanced positives and negatives, or no clear signal.
- negative: Net negative; more concerning than supportive; weak vs prior or vs expectations.

1) Past Performance
Assess historical results versus prior periods. Consider revenue, profitability, margins, cash flow, EPS trends, and balance sheet quality. Output one label.

2) Future Performance
Assess forward-looking guidance and management commentary. Consider growth outlook, margins, demand environment, confidence vs caution, and risks mentioned. Output one label.

3) Expectation Gap
Assess whether results and commentary are better or worse than a reasonable pre-announcement expectation. Use the tone and content of the release to infer positive surprise vs negative surprise. Output one label.

4) Market reaction
How does the market normally react typically to this kind of earnings press release. Consider the company's financial history and sector. Output one label.

OUTPUT FORMAT (STRICT):
Respond with only a single valid JSON object, no other text. Use this structure:

{
  "past_performance": "negative" | "neutral" | "positive" | "strong_positive",
  "future_performance": "negative" | "neutral" | "positive" | "strong_positive",
  "expectation_gap": "negative" | "neutral" | "positive" | "strong_positive",
  "market_reaction": "negative" | "neutral" | "positive" | "strong_positive",
  "justifications": {
    "past_performance": "<1-2 sentences>",
    "future_performance": "<1-2 sentences>",
    "expectation_gap": "<1-2 sentences>",
    "market_reaction": "<1-2 sentences>"
  }
}

For past_performance, future_performance, expectation_gap, and market_reaction use exactly one of: "negative", "neutral", "positive", "strong_positive".
Replace the justification placeholder strings with brief text (1–2 sentences per metric).

----------------------------------------
BEGIN EX-99.1
----------------------------------------

<<<EX99_1_TEXT>>>

----------------------------------------
END EX-99.1
----------------------------------------
"""


# ---------------------------------------------------------------------------
# Media reaction LLM prompt template
# ---------------------------------------------------------------------------

MEDIA_REACTION_PROMPT_TEMPLATE = """Analyze {company} ({ticker}) {quarter} earnings release and, if available, earnings call transcript from the perspective of a professional buy-side equity analyst.

Search the web for recent business and financial news and analysis about this event.

Use reputable sources such as Bloomberg, Reuters, CNBC, Financial Times, Wall Street Journal, Barron's, MarketWatch, and major broker research (e.g., Goldman Sachs, Morgan Stanley, JPMorgan) where available.

Your tasks:
1. Assess the overall sentiment of coverage toward the earnings and outlook.
2. Determine whether the company beat or missed consensus expectations on EPS and Revenue (when such information is available).
3. Identify key positive themes and significant red flags mentioned across articles and broker notes.

Respond with STRICT JSON only. No other text before or after:
{{
  "sentiment": "strong_positive" | "positive" | "mixed" | "negative" | "no_coverage",
  "eps": "strong_beat" | beat" | "miss" | "other" | "unknown",
  "revenue": "strong_beat" | "beat" | "miss" | "other" | "unknown",
  "broker_reactions": "buy" | "strong_buy" | "moderate_buy" | "hold" | "sell" | "other (specify)",
  "headlines": [
    "<short positive headline or quote>",
    "<another positive headline or quote>"
  ],
  "red_flags": [
    "<short negative/red-flag headline or quote>",
    "<another negative/red-flag headline or quote>"
  ],
  "summary": "<2–3 sentences explaining why sentiment is positive/neutral/negative, citing key drivers such as guidance, margins, demand trends, cash flow, or leverage.>"
}}

If you find no relevant coverage in that window, set:
- \"sentiment\": \"no_coverage\",
- \"eps_result\": \"unknown\",
- \"revenue_result\": \"unknown\",
- \"broker_reactions\": \"unknown\"
"""


# ---------------------------------------------------------------------------
# ED-8 advisor class and command entry
# ---------------------------------------------------------------------------

class Edgar(AdvisorBase):
    """Advisor for 8-K earnings filings (basic filters only in this step)."""

    def filter_filing(self, filing) -> bool:
        """
        Basic filing filter (8-K content): Filters 1, 2, 4.
        Logs explicit failure reason and returns False on fail.
        """
        cik = str(getattr(filing, "cik", "") or "")
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)
        accession = (
            getattr(filing, "accession_no", None)
            or getattr(filing, "accession_number", None)
            or ""
        )

        def _fail(reason: str) -> bool:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s failed: %s",
                ticker or "N/A",
                cik or "N/A",
                accession,
                reason,
            )
            return False

        form = getattr(filing, "form", None)
        if form != "8-K":
            return _fail("not 8-K")

        if not (hasattr(filing, "exhibits") and filing.exhibits):
            return _fail("no exhibits")
        if not any("99." in str(ex).lower() for ex in filing.exhibits):
            return _fail("no exhibit 99.x")

        filing_text = (filing.text() or "").lower() if hasattr(filing, "text") else ""

        # Prefer the formal earnings announcement item over loose keyword heuristics.
        has_item_202 = (
            "item 2.02" in filing_text
            or "results of operations and financial condition" in filing_text
        )
        if not has_item_202:
            return _fail("no item 2.02 / results of operations and financial condition")

        # Filter 2: red/green flags
        exhibit_99_text = _get_ex99_text(filing)
        combined_text = (filing_text + " " + (exhibit_99_text or "")).lower()
        exhibit_99_lower = (exhibit_99_text or "").lower()
        score = 0
        for keyword, penalty in RED_FLAGS_SEVERE.items():
            if keyword in combined_text:
                score += penalty
        for keyword, penalty in RED_FLAGS_MODERATE.items():
            if keyword in combined_text:
                score += penalty
        for keyword, bonus in GREEN_FLAGS.items():
            if keyword in exhibit_99_lower:
                score += bonus
        if score < 0:
            return _fail("red flags (score < 0)")

        # Filter 4: EX-99 earnings release structure
        if not exhibit_99_text or not exhibit_99_text.strip():
            return _fail("no EX-99 text")
        result = _earnings_release_filter(exhibit_99_text)
        if not result["pass"]:
            return _fail(result["reason"])

        return True

    def filter_financials(self, filing) -> bool:
        """
        Basic financial filter: price band, cap, P/E, previous-day gain, sector hard fail.
        Logs explicit failure reason and returns False on fail.
        """
        cik = str(getattr(filing, "cik", "") or "")
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)
        accession = (
            getattr(filing, "accession_no", None)
            or getattr(filing, "accession_number", None)
            or ""
        )

        def _fail(reason: str) -> bool:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s failed: %s",
                ticker or "N/A",
                cik or "N/A",
                accession,
                reason,
            )
            return False

        if not ticker:
            return _fail("no ticker")

        try:
            time.sleep(0.05)
            stock = yf.Ticker(ticker)
            info = stock.info or {}
        except Exception as e:
            logger.warning("filter_financials: yfinance error for %s: %s", ticker, e)
            return _fail("yfinance error")

        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe is not None and isinstance(pe, (int, float)) and pe > FILTER3_PE_MAX:
            return _fail(f"overvalued (P/E > {FILTER3_PE_MAX:d})")

        cap = info.get("marketCap")
        if cap is not None and isinstance(cap, (int, float)) and cap < FILTER3_MIN_CAP:
            return _fail("market cap too low (< $300M)")

        price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
        if price is not None and isinstance(price, (int, float)):
            if price < FILTER3_PRICE_MIN:
                return _fail(f"price below band (< ${FILTER3_PRICE_MIN:.0f})")
            if price > FILTER3_PRICE_MAX:
                return _fail(f"price above band (> ${FILTER3_PRICE_MAX:.0f})")

        # Sector/industry: hard fail from SECTOR_LIST
        sector = (info.get("sector") or "").strip().lower()
        industry = (info.get("industry") or "").strip().lower()
        for entry in SECTOR_LIST:
            if len(entry) >= 3 and entry[2] == "hard_fail":
                sector_str, ind_str = entry[0], entry[1]
                if ind_str is None:
                    if sector_str in sector or sector_str in industry:
                        return _fail(f"sector/industry hard fail ({sector_str})")
                else:
                    if (sector_str in sector or sector_str in industry) and (ind_str in sector or ind_str in industry):
                        return _fail(f"sector/industry hard fail ({entry[0]}, {entry[1]})")

        # Recent 5-day big price gain (>= 10%) → hard fail
        filing_date = _filing_date_or_none(filing)
        if filing_date is not None and price is not None:
            try:
                start = filing_date - timedelta(days=14)
                end = filing_date + timedelta(days=2)
                hist = stock.history(start=start, end=end, auto_adjust=True)
                if hist is not None and not hist.empty:
                    before_filing = hist[[(x.date() if hasattr(x, "date") else x) < filing_date for x in hist.index]]
                    # Require at least 6 prior closes to compute a 5-day move
                    if len(before_filing) >= 6:
                        close_prev = float(before_filing["Close"].iloc[-1])
                        close_5d_ago = float(before_filing["Close"].iloc[-6])
                        if close_5d_ago > 0:
                            pct_5d = (close_prev - close_5d_ago) / close_5d_ago * 100.0
                            if pct_5d >= FILTER3_PREV_DAY_GAIN_FAIL_PCT:
                                pass#return _fail(f"5-day gain >= {FILTER3_PREV_DAY_GAIN_FAIL_PCT:.0f}%")
            except Exception:
                # If history fails we don't block on this check
                pass

        return True

    def evaluate_eps_beat(self, filing) -> Optional[str]:
        """
        Evaluate EPS performance for this 8-K's quarter using Alpha Vantage EARNINGS
        plus 8-K fallback. Returns one of:

            None          -> no usable EPS intel
            "miss"        -> EPS miss vs consensus
            "beat"        -> EPS beat
            "strong_beat" -> EPS beat with stronger score (threshold TBD)

        Under the hood we compute an eps_score similar to the old Filter 6:
            eps_score = min(surprise_pct, 50) * sqrt(abs(EPS))

        For now we use a simple threshold to distinguish beat vs strong_beat.
        """
        cik = str(getattr(filing, "cik", "") or "")
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)
        accession = (
            getattr(filing, "accession_no", None)
            or getattr(filing, "accession_number", None)
            or ""
        )
        if not ticker:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s EPS: no ticker",
                "N/A",
                cik or "N/A",
                accession,
            )
            return None

        report_date = _filing_date_or_none(filing)
        if report_date is None:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s EPS: no filing_date",
                ticker,
                cik or "N/A",
                accession,
            )
            return None

        record = get_eps_for_report_quarter(ticker, report_date)
        if not record:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s EPS: no Alpha Vantage record",
                ticker,
                cik or "N/A",
                accession,
            )
            return None

        # Primary: AV surprisePercentage + reportedEPS
        surprise_str = record.get("surprisePercentage")
        reported_str = record.get("reportedEPS")
        try:
            surprise_val = float(surprise_str)
            reported_val = float(reported_str)
            if reported_val != 0:
                surprise_pct = surprise_val
                eps_score = min(surprise_pct, 50.0) * (abs(reported_val) ** 0.5)
                status: Optional[str]
                if surprise_pct <= 0:
                    status = "miss"
                else:
                    if eps_score >= STRONG_BEAT_THRESHOLD:
                        status = "strong_beat"
                    elif eps_score >= BEAT_THRESHOLD:
                        status = "beat"
                    else:
                        # Small positive surprise but below beat threshold: treat as weak/no EPS edge
                        status = "miss"
                logger.info(
                    "ticker=%s, CIK=%s, accession=%s EPS (%s): surprise=%+.2f%% "
                    "reported=%s score=%.1f -> %s",
                    ticker,
                    cik or "N/A",
                    accession,
                    "alpha_vantage",
                    surprise_pct,
                    reported_str,
                    eps_score,
                    status,
                )
                return status
        except (TypeError, ValueError):
            # fall through to 8-K+AV fallback
            pass

        # Fallback: 8-K actual EPS + AV estimatedEPS
        actual_eps, source = get_actual_eps_from_8k(filing, verbose=False)
        estimated_eps = record.get("estimatedEPS")
        try:
            estimated_eps = float(estimated_eps) if estimated_eps not in (None, "", "None") else None
        except (TypeError, ValueError):
            estimated_eps = None

        if actual_eps is not None and estimated_eps is not None and estimated_eps != 0:
            surprise_pct = ((actual_eps - estimated_eps) / abs(estimated_eps)) * 100.0
            eps_score = min(surprise_pct, 50.0) * (abs(actual_eps) ** 0.5)
            if surprise_pct <= 0:
                status = "miss"
            else:
                if eps_score >= STRONG_BEAT_THRESHOLD:
                    status = "strong_beat"
                elif eps_score >= BEAT_THRESHOLD:
                    status = "beat"
                else:
                    status = "weak_beat"
            logger.info(
                "ticker=%s, CIK=%s, accession=%s EPS (%s+AV): actual=%.2f "
                "estimate=%.2f surprise=%+.2f%% score=%.1f -> %s",
                ticker,
                cik or "N/A",
                accession,
                source,
                actual_eps,
                estimated_eps,
                surprise_pct,
                eps_score,
                status or "none",
            )
            return status

        logger.info(
            "ticker=%s, CIK=%s, accession=%s EPS: missing EPS values "
            "(AV reported/estimate and 8-K fallback unusable)",
            ticker,
            cik or "N/A",
            accession,
        )
        return None

    def analyse_ex99_llm(self, filing) -> Dict[str, Optional[object]]:
        """
        Run EX-99.1 LLM (3-ducks labels prompt) on the filing.

        Returns a dict:
            {
              "eps": None,
              "revenue": None,
              "past_performance": "negative"|"neutral"|"positive"|"strong_positive" or None,
              "guidance": "negative"|"neutral"|"positive"|"strong_positive" or None,
              "expectation": "negative"|"neutral"|"positive"|"strong_positive" or None,
              "justifications": dict with keys past_performance, guidance, expectation (str or None) for discovery explanation,
            }

        Beat vs. consensus is now handled only by the media LLM (media_reaction_llm).
        """
        cik = str(getattr(filing, "cik", "") or "")
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)
        accession = (
            getattr(filing, "accession_no", None)
            or getattr(filing, "accession_number", None)
            or ""
        )

        result_dict: Dict[str, Optional[object]] = {
            "eps": None,
            "revenue": None,
            "past_performance": None,
            "guidance": None,
            "expectation": None,
            "market_reaction": None,
            "justifications": None,
        }

        text = _get_ex99_text(filing)
        if not text or not text.strip():
            logger.info(
                "ticker=%s, CIK=%s, accession=%s EX99 LLM: no EX-99.x text",
                ticker or "N/A",
                cik or "N/A",
                accession,
            )
            return result_dict

        prompt = FOUR_DUCKS_PROMPT_LABELS.replace("<<<EX99_1_TEXT>>>", text.strip())
        model, parsed = self.ask_gemini(prompt, timeout=120.0)
        if not parsed:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s EX99 LLM: no result from Gemini",
                ticker or "N/A",
                cik or "N/A",
                accession,
            )
            # No LLM - try again later
            self.watch(ticker, explanation=f"{accession}")
            return result_dict

        # Map label strings (negative | neutral | positive | strong_positive)
        _allowed = {"negative", "neutral", "positive", "strong_positive"}
        for key_src, key_dst in (
            ("past_performance", "past_performance"),
            ("future_performance", "guidance"),
            ("expectation_gap", "expectation"),
            ("market_reaction", "market_reaction"),
        ):
            val = parsed.get(key_src)
            if isinstance(val, str):
                label = val.strip().lower()
                result_dict[key_dst] = label if label in _allowed else None
            else:
                result_dict[key_dst] = None

        # Justifications for discovery explanation (past_performance, guidance, expectation, market_reaction)
        j = parsed.get("justifications") or {}
        if isinstance(j, dict):
            result_dict["justifications"] = {
                "past_performance": j.get("past_performance") if isinstance(j.get("past_performance"), str) else None,
                "guidance": j.get("future_performance") if isinstance(j.get("future_performance"), str) else None,
                "expectation": j.get("expectation_gap") if isinstance(j.get("expectation_gap"), str) else None,
                "market_reaction": j.get("market_reaction") if isinstance(j.get("market_reaction"), str) else None,
            }
        else:
            result_dict["justifications"] = None

        # Check for hard fails
        passed = True

        g = result_dict.get("guidance")
        e = result_dict.get("expectation")
        p = result_dict.get("past_performance")
        m = result_dict.get("market_reaction")

        # Hard-fail: Any metric labeled negative
        for name, v in (("guidance", g), ("expectation", e), ("past_performance", p), ("market_reaction", m)):
            if v == "negative":
                passed = False

        # Hard-fail: neutral guidance or market is not good enough
        for name, v in (("guidance", g), ("market_reaction", m)):
            if v == "negative":
                passed = False

        logger.info(
            "ticker=%s, CIK=%s, accession=%s EX99 LLM: model=%s "
            "past=%s guidance=%s expectation=%s %s",
            ticker or "N/A",
            cik or "N/A",
            accession,
            model or "N/A",
            result_dict["past_performance"] or "N/A",
            result_dict["guidance"] or "N/A",
            result_dict["expectation"] or "N/A",
            "-> pass"  if passed else "-> fail"
        )

        if passed:
            return result_dict

        return None

    def media_reaction_llm(self, filing) -> Optional[Dict[str, Optional[object]]]:
        """
        Run media-reaction LLM over business/financial coverage around the filing.
        """
        cik = str(getattr(filing, "cik", "") or "")
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)
        accession = (
            getattr(filing, "accession_no", None)
            or getattr(filing, "accession_number", None)
            or ""
        )

        # Filing date → used for quarter label only.
        filing_date = _filing_date_or_none(filing) or date.today()

        # Prompt: single template with EPS section substituted depending on eps_beat availability.
        company = getattr(filing, "company_name", None) or getattr(filing, "company", None) or (ticker or cik or "Unknown")
        quarter = quarter_label_for_filing_date(filing_date)

        media_prompt = MEDIA_REACTION_PROMPT_TEMPLATE.format(
            company=company,
            ticker=ticker or "",
            quarter=quarter,
        )
        print("-------")
        print(media_prompt)
        print("-------")

        model, parsed = self.ask_gemini(media_prompt, timeout=120.0, use_search=True)
        if not parsed or not isinstance(parsed, dict):
            logger.info(
                "ticker=%s, CIK=%s, accession=%s media LLM: no result from Gemini",
                ticker or "N/A",
                cik or "N/A",
                accession,
            )
            # No LLM - try again later
            self.watch(ticker, explanation=f"{accession}")
            return None

        print("-------")
        print(parsed)
        print("-------")

        sentiment = parsed.get("sentiment")
        eps = parsed.get("eps")
        revenue = parsed.get("revenue")
        broker = parsed.get("broker_reactions")
        headlines = parsed.get("headlines")
        red_flags = parsed.get("red_flags")
        summary = parsed.get("summary")

        # Media-driven hard fail or watch:
        if sentiment in ["no_coverage", "mixed", "negative"] or eps in ["miss", "unknown"] or broker in ["hold" ,"sell", "unknown"]:
            logger.info(
                "ticker=%s, accession=%s media LLM: "
                "(eps=%s, revenue=%s, sentiment=%s, broker=%s) -> fail",
                ticker,
                accession,
                eps,
                revenue,
                sentiment or "N/A",
                broker or "N/A"
            )
            # No coverage or broker unknown - try again later
            if sentiment == "no_coverage" or broker == "unknown":
                self.watch(ticker, explanation=f"{accession}")
            return None

        result: Dict[str, Optional[object]] = {
            "sentiment": sentiment,
            "eps": eps,
            "revenue": revenue,
            "broker": broker,
            "headlines": headlines if isinstance(headlines, list) else [],
            "red_flags": red_flags if isinstance(red_flags, list) else [],
            "summary": summary if isinstance(summary, str) else None
        }

        logger.info(
            "ticker=%s, accession=%s media LLM: "
            "(eps=%s, revenue=%s, sentiment=%s, broker=%s) -> pass",
            ticker,
            accession,
            eps,
            revenue,
            sentiment or "N/A",
            broker or "N/A"
        )

        return result

    def build_explanation(self, filing, advanced: dict) -> str:

        cik = str(getattr(filing, "cik", "") or "")
        accession = (
            getattr(filing, "accession_no", None)
            or getattr(filing, "accession_number", None)
            or ""
        )

        """Build discovery explanation string from advanced result (weight, ex99, media)."""
        weight = advanced.get("weight")
        if weight is None or not isinstance(weight, (int, float)):
            weight = 1.0

        ex99 = advanced.get("ex99") or {}
        media = advanced.get("media") or {}
        parts = [f"8-K {accession}: {weight:.2f} | https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude | "]

        justifications = ex99.get("justifications") or {}
        if isinstance(justifications, dict):
            j_parts = [str(v).strip() for k, v in justifications.items() if v and str(v).strip()]
            if j_parts:
                parts.append("EX-99: " + " | ".join(j_parts))

        parts.append(f"PAST: {ex99.get('past_performance')}")
        parts.append(f"GUIDANCE: {ex99.get('guidance')}")
        parts.append(f"EXPECTATION: {ex99.get('expectation')}")
        parts.append(f"MARGET: {ex99.get('market_reaction')}")

        summary = media.get("summary")
        if isinstance(summary, str) and summary.strip():
            parts.append("MEDIA: " + summary.strip())

        parts.append(f"SENTIMENT: {media.get('sentiment')}")
        parts.append(f"EPS: {media.get('eps')}")
        parts.append(f"REVENUE: {media.get('revenue')}")
        parts.append(f"BROKER: {media.get('broker')}")

        headlines = media.get("headlines") or []
        if isinstance(headlines, list) and headlines:
            snippets = [str(h).strip() for h in headlines[:2] if h and str(h).strip()]
            if snippets:
                parts.append("HEADLINES: | " + " | ".join(snippets))

        red_flags = media.get("red_flags") or []
        if isinstance(red_flags, list) and red_flags:
            snippets = [str(f).strip() for f in red_flags[:2] if f and str(f).strip()]
            if snippets:
                parts.append("RED FLAGS: | " + " | ".join(snippets))

        bonuses = advanced.get("bonuses") or []
        penalties = advanced.get("penalties") or []

        if bonuses:
            parts.append("BONUSES: | " + " | ".join(bonuses))

        if penalties:
            parts.append("PENALTIES: | " + " | ".join(penalties))

        return " | ".join(filter(None, parts))

    def score_results(self, filing, ex99, media):

        cik = str(getattr(filing, "cik", "") or "")
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)

        bonuses: list[str] = []
        penalties: list[str] = []
        score = 0.0

        # Filing time (ET): 6:30–7:30 inclusive +0.1, 16:00–20:00 inclusive -0.1
        filing_dt = _filing_datetime(filing)
        if filing_dt is not None:
            et = ZoneInfo("America/New_York")
            t_et = filing_dt.astimezone(et).time()
            if dt_time(6, 30) <= t_et <= dt_time(7, 30):
                bonuses.append("Pre-market filing (+0.1)")
                score += 0.1
            elif dt_time(16, 0) <= t_et <= dt_time(20, 0):
                penalties.append("After-hours filing (-0.1)")
                score -= 0.1

        # 4 ducks
        past_perform = ex99['past_performance']
        expectation = ex99['expectation']
        guidance = ex99['guidance']
        market = ex99['market_reaction']

        if past_perform == "strong_positive":
            bonuses.append("Strong past (+0.1)")
            score += 0.1
        elif past_perform == "neutral":
            penalties.append("Neutral past (-0.1)")
            score -= 0.1

        if expectation == "strong_positive":
            bonuses.append("Strong expectation (+0.1)")
            score += 0.1
        elif expectation == "neutral":
            penalties.append("Neutral expectation (-0.1)")
            score -= 0.2

        if guidance == "strong_positive":
            bonuses.append("Strong guidance (+0.2)")
            score += 0.2

        if market == "strong_positive":
            bonuses.append("Strong market (+0.2)")
            score += 0.2

        # Media
        sentiment = media.get("sentiment")
        eps = media.get("eps")
        revenue =  media.get("revenue")
        broker = media.get("broker")

        if sentiment == "strong_positive":
            bonuses.append("Strong reaction (+0.2)")
            score += 0.2

        if eps == "strong_beat":
            bonuses.append("Strong eps (+0.2)")
            score += 0.2
        elif eps == "weak_beat":
            penalties.append("Weak eps (-0.2)")
            score -= 0.2

        if revenue == "strong_beat":
            bonuses.append("Strong revenue (+0.1)")
            score += 0.1
        elif revenue == "weak_beat":
            penalties.append("Weak revenue (-0.1)")
            score -= 0.1
        elif revenue == "neutral":
            penalties.append("Neutral revenue (-0.2)")
            score -= 0.2
        elif revenue == "miss":
            penalties.append("Neutral revenue (-0.3)")
            score -= 0.3

        if broker == "strong_buy":
            bonuses.append("Strong buy (+0.2)")
            score += 0.2
        elif broker == "weak_buy":
            penalties.append("Weak buy (-0.2)")
            score -= 0.2

        # Valuation
        valuation = self.evaluate_stock(ticker)

        if valuation >= 1.10:
            penalties.append("Overvalued (-0.2)")
            score -= 0.2
        elif valuation < 0.90:
            bonuses.append("Undervalued (+0.2)")
            score += 0.2

        # Sector (BAD_SECTOR / WATCH -0.2) and 52-week high/low
        try:
            time.sleep(0.05)
            stock = yf.Ticker(ticker)
            info = stock.info or {}
        except Exception:
            info = {}

        # Sector check
        sector = (info.get("sector") or "").strip().lower()
        industry = (info.get("industry") or "").strip().lower()
        for entry in SECTOR_LIST:
            if len(entry) >= 3 and entry[2] == "weight":
                sector_str, ind_str = entry[0], entry[1]
                if ind_str is None:
                    if sector_str in sector or sector_str in industry:
                        penalties.append(f"Bad sector: {sector_str} (-0.2)")
                        score -= 0.2
                        break
                else:
                    if sector_str in sector and ind_str in industry:
                        penalties.append(f"Bad sector: {sector_str} (-0.2)")
                        score -= 0.2
                        break

        # 52-wwek high / low
        price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
        high_52 = info.get("fiftyTwoWeekHigh")
        low_52 = info.get("fiftyTwoWeekLow")
        if price is not None and isinstance(price, (int, float)) and price > 0:
            if low_52 is not None and isinstance(low_52, (int, float)) and low_52 > 0:
                if price <= low_52 * 1.10:
                    bonuses.append("52-week low (+0.2)")
                    score += 0.2
            if high_52 is not None and isinstance(high_52, (int, float)) and high_52 > 0:
                if price >= high_52 * 0.90:
                    penalties.append("52-week high (-0.2)")
                    score -= 0.2

        # 2-week high: near recent high -> -0.1
        try:
            hist = stock.history(period="2wk")
            if hist is not None and not hist.empty and price is not None and price > 0:
                high_2w = float(hist["High"].max())
                if high_2w > 0 and price >= high_2w * 0.95:
                    penalties.append("2-week high (-0.1)")
                    score -= 0.1
        except Exception:
            pass

        return score, bonuses, penalties

    def analyze_8k_basic(self, filing) -> bool:
        """
        Run basic filing + financial filters.
        Return True if pass.
        """
        if not self.filter_filing(filing):
            return False

        if not self.filter_financials(filing):
            return False

        return True

    def analyze_8k_advanced(self, filing):

        # Try and see if EPS intel in AV or Filing
        eps_beat = self.evaluate_eps_beat(filing)

        # First hard-fail
        if eps_beat == "miss":
            return None

        # First AI - anaylase ex99.1 8-K attachment
        if (ex99 := self.analyse_ex99_llm(filing)) is None:
            return None
        
        # Second AI - media_reaction_llm
        if (media := self.media_reaction_llm(filing)) is None:
            return None

        score, bonuses, penalties = self.score_results(filing, ex99, media)

        # Hard-fail on low score
        if score <= 0:
            return None

        # Good to go: weight = half / normal / double triage
        return {
            "weight": score + 0.5,
            "bonuses": bonuses,
            "penalties": penalties,
            "ex99": ex99,
            "media": media,
        }

    def analyze_8k(self, filing, sa) -> bool:

        sell_instructions = [
            ("PERCENTAGE_DIMINISHING", 1.20, 14),
            ("PERCENTAGE_AUGMENTING", 0.90, 14),
            ("PEAKED", 7.0, None),
            ("PROFIT_FLAT", 0.5, 4),
            ("DESCENDING_TREND", -0.20, None),
        ]

        cik = str(getattr(filing, "cik", "") or "")
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)

        if not ticker:
            logger.info(f"{ticker} - no tradable ticker")
            return False

        # Check if already discovered - rediscover if >1 days ago
        if not self.allow_discovery(ticker, period=24):
            return False

        # Check the basics
        if not self.analyze_8k_basic(filing):
            return False

        # Now call in the robot help
        if (advanced :=  self.analyze_8k_advanced(filing)) is None:
            return False

        # Make sense of it all
        explanation = self.build_explanation(filing, advanced)

        print(explanation)

        # Keeper
        self.discovered(sa, ticker, explanation, sell_instructions, weight=Decimal(advanced["weight"]), check_health=False)
        return True

    def discover(self, sa):

        logger.info("Fetching latest filings...")

        prev_ts = self.get_previous_sa_timestamp(sa)

        try:
            latest = get_latest_filings()
            filings = list(latest)
        except Exception as e:
            logger.warning("❌ Error converting latest filings to list: %s", e)
            return

        """
        filing1 = find("0001628280-26-020491")
        filing2 = find("0001193125-26-119844")

        filings = [filing1, filing2]
        """
        # Filter latest to 8-Ks only
        filings_8k = [f for f in filings if getattr(f, "form", None) == "8-K"]
        if not filings_8k:
            logger.info("No latest 8-K filings found.")
            return

        logger.info("Found %d 8-K filings. Running basic inspection (filing + financial filters)...", len(filings_8k))

        for filing in filings_8k:
            try:
                filing_dt = _filing_datetime(filing)
                if prev_ts is not None and filing_dt is not None and filing_dt < prev_ts:
                    accession = getattr(filing, "accession_no", None) or getattr(filing, "accession_number", None) or ""
                    logger.warning("Filing %s (filing_time=%s) is before prev SA %s — skipping",
                                   accession, filing_dt, prev_ts)
                    #continue

                self.analyze_8k(filing, sa)

            except Exception as e:
                logger.error("⚠️ Error inspecting filing: %s", e)

        # Filter wathced 8-Ks
        watched_8ks = self.watchlist()

        logger.info("Found %d 8-K watched filings.", len(watched_8ks))

        for item in watched_8ks:
            try:
                filing = find(item.explanation)
                item.status = "Excluded"
                item.save()

                self.analyze_8k(filing, sa)

            except Exception as e:
                logger.error("⚠️ Error inspecting filing: %s", e)

        return

def run_edgar_standalone():
    """
    Minimal entry point for the `run_edgar` management command.

    No params yet; instantiates the advisor via python_class (same as production)
    and calls discover(sa) with a minimal SmartAnalysis().
    """
    from core.services import advisors as advisor_modules
    from core.models import Advisor, SmartAnalysis

    try:
        advisor_row = Advisor.objects.get(name="EDDIE-8")
    except Advisor.DoesNotExist:
        return None, "ED-8 advisor not found in Advisor table"

    # Resolve class via python_class (same pattern as smartanalyse)
    module_name = advisor_row.python_class.lower()
    module = getattr(advisor_modules, module_name)
    PythonClass = getattr(module, advisor_row.python_class)

    sa = SmartAnalysis()
    impl = PythonClass(advisor_row)
    impl.discover(sa)

    return "EDDIE-8 discover() stub completed", None


register(name="EDDIE-8", python_class="Edgar")

