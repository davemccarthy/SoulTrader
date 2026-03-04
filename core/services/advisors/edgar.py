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
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple
import requests
import yfinance as yf
from django.conf import settings
from edgar import Company, find, set_identity, get_latest_filings

from core.services.advisors.advisor import AdvisorBase, register

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SEC identity (mandatory)
# ---------------------------------------------------------------------------
set_identity("David McCarthy david@example.com")

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


# Filing filter (Filter 1): earnings 8-K / exhibit 99
EARNINGS_KEYWORDS = [
    "results of operations", "financial condition", "period ended",
    "three months ended", "six months ended", "nine months ended",
    "fiscal quarter", "fiscal year", "furnished pursuant to",
    "press release announcing", "press release regarding", "attached as exhibit 99",
    "included as exhibit", "announced its results", "reported results",
    "released results", "announced financial results", "reported financial results",
    "provided guidance", "updated guidance", "outlook for", "expects for",
    "anticipated results", "quarter ended", "year ended", "period results",
    "financial results", "earnings release", "quarterly results", "full year results",
    "earnings per share", "eps", "net income per share", "guidance", "outlook",
    "expects", "forecast",
]
REG_FD_KEYWORDS = ["furnished pursuant to regulation fd", "Item 3.01", "Item 1.03"]

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
# Sector/industry hard fail (e.g. cannabis); each entry is (substring,) or (sector_substring, industry_substring)
SECTOR_INDUSTRY_HARD_FAIL = (
    ("cannabis",),
)

# EPS beat strength thresholds (on eps_score)
BEAT_THRESHOLD = 8.0
STRONG_BEAT_THRESHOLD = 20.0


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
# EX-99.1 LLM (3-ducks counts-style) prompt and helpers
# ---------------------------------------------------------------------------

THREE_DUCKS_PROMPT_COUNTS = """You are an equity analyst.
Given an earnings press release (EX-99.1 from an 8-K), evaluate the following three metrics independently.
Use only the information in the document.
Ignore stock price movement.
Ignore analyst consensus unless explicitly mentioned in the text.
Do not speculate beyond the text.

TASK
For each metric, count evidence points: how many support a positive reading vs how many support a negative reading.
Output only integer counts (positive, negative). When evidence is mixed, both counts can be non-zero.
Use 0 when there is no relevant evidence for that side.

1) Past Performance
Count positive vs negative evidence from historical results versus prior periods.
Consider revenue, profitability, margins, cash flow, EPS trends, and balance sheet quality.

2) Future Performance
Count positive vs negative evidence from forward-looking guidance and management commentary.
Consider growth outlook, margins, demand environment, confidence vs caution, and risks mentioned.

3) Expectation Gap
Count positive vs negative evidence for whether results and commentary are better or worse than a reasonable pre-announcement expectation.
Use the tone and content of the release to infer positive surprise vs negative surprise relative to prior expectations.

4) Beat (AKA surprise)
Note if the release states whether the company beat or missed expectations for EPS and/or revenue.
When there is evidence of a beat, also qualitatively distinguish whether it was a weak beat, normal beat, or strong beat based on the language in the release.

OUTPUT FORMAT (STRICT):
Respond with only a single valid JSON object, no other text. Use this structure:

{
  "past_performance": { "positive": <integer>, "negative": <integer> },
  "future_performance": { "positive": <integer>, "negative": <integer> },
  "expectation_gap": { "positive": <integer>, "negative": <integer> },
  "headline_beat": {
    "eps": "weak_beat"|"beat"|"strong_beat"|"miss"|"unknown",
    "revenue": "weak_beat"|"beat"|"strong_beat"|"miss"|"unknown"
  },
  "justifications": {
    "past_performance": "<1-2 sentences>",
    "future_performance": "<1-2 sentences>",
    "expectation_gap": "<1-2 sentences>"
  }
}

For headline_beat, use:
- "strong_beat" when the release clearly emphasizes a large upside surprise (e.g. "significantly exceeded expectations" or "well above consensus").
- "beat" for a normal, clearly positive surprise without language suggesting it was exceptionally large.
- "weak_beat" when the beat appears small, marginal, or only mildly emphasized.
- "miss" when the release clearly indicates that EPS or revenue fell short of expectations.
- "unknown" when the release does not clearly state whether expectations were beaten or missed.

Replace <integer> with non-negative integers (0 or more) for past_performance, future_performance, and expectation_gap.
For headline_beat use one of the specified strings for each of eps and revenue.
Replace the placeholder strings with brief justifications (1–2 sentences per metric).

----------------------------------------
BEGIN EX-99.1
----------------------------------------

<<<EX99_1_TEXT>>>

----------------------------------------
END EX-99.1
----------------------------------------
"""


def _ratio_score(positive: int, negative: int) -> float:
    """Convert (positive, negative) counts to a single score in [-1, 1]."""
    total = positive + negative
    if total == 0:
        return 0.0
    return (positive - negative) / total


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

        if not ticker:
            return _fail("no tradable ticker")

        if not (hasattr(filing, "exhibits") and filing.exhibits):
            return _fail("no exhibits")
        if not any("99." in str(ex).lower() for ex in filing.exhibits):
            return _fail("no exhibit 99.x")

        filing_text = (filing.text() or "").lower() if hasattr(filing, "text") else ""
        if "item 9.0" not in filing_text:
            return _fail("no item 9.0")
        for kw in REG_FD_KEYWORDS:
            if kw in filing_text:
                return _fail("Reg FD / non-earnings 8-K")
        if not any(kw in filing_text for kw in EARNINGS_KEYWORDS):
            return _fail("not earnings 8-K")

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

        # Sector/industry hard fail (e.g. cannabis)
        sector = (info.get("sector") or "").strip().lower()
        industry = (info.get("industry") or "").strip().lower()
        for entry in SECTOR_INDUSTRY_HARD_FAIL:
            if len(entry) == 1:
                if entry[0] in sector or entry[0] in industry:
                    return _fail(f"sector/industry hard fail ({entry[0]})")
            elif len(entry) >= 2:
                if (entry[0] in sector or entry[0] in industry) and (entry[1] in sector or entry[1] in industry):
                    return _fail(f"sector/industry hard fail ({entry})")

        # Previous trading day big price gain (>= 10%) → hard fail
        fd = getattr(filing, "filing_date", None)
        if fd is not None and price is not None:
            try:
                if isinstance(fd, date):
                    filing_date = fd
                elif isinstance(fd, str):
                    filing_date = date.fromisoformat(fd[:10])
                elif hasattr(fd, "date") and callable(getattr(fd, "date")):
                    filing_date = fd.date()
                else:
                    filing_date = date.fromisoformat(str(fd)[:10])
                start = filing_date - timedelta(days=14)
                end = filing_date + timedelta(days=2)
                hist = stock.history(start=start, end=end, auto_adjust=True)
                if hist is not None and not hist.empty:
                    before_filing = hist[[(x.date() if hasattr(x, "date") else x) < filing_date for x in hist.index]]
                    if len(before_filing) >= 2:
                        close_prev = float(before_filing["Close"].iloc[-1])
                        close_prev2 = float(before_filing["Close"].iloc[-2])
                        if close_prev2 > 0:
                            prev_day_pct = (close_prev - close_prev2) / close_prev2 * 100.0
                            if prev_day_pct >= FILTER3_PREV_DAY_GAIN_FAIL_PCT:
                                return _fail(f"previous day gain >= {FILTER3_PREV_DAY_GAIN_FAIL_PCT:.0f}%")
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

        fd = getattr(filing, "filing_date", None)
        if fd is None:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s EPS: no filing_date",
                ticker,
                cik or "N/A",
                accession,
            )
            return None
        if isinstance(fd, str):
            try:
                report_date = date.fromisoformat(fd[:10])
            except ValueError:
                return None
        else:
            report_date = getattr(fd, "date", lambda: fd)() if hasattr(fd, "date") else fd

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

    def analyse_ex99_llm(self, filing, eps_beat: Optional[str]) -> Dict[str, Optional[object]]:
        """
        Run EX-99.1 LLM (3-ducks counts prompt) on the filing.

        Returns a dict:
            {
              "eps": <weak_beat|beat|strong_beat|miss|unknown|None>,
              "revenue": <same set as eps>,
              "past_performance": float in [-1, 1] or None,
              "guidance": float in [-1, 1] or None,
              "expectation": float in [-1, 1] or None,
              "justifications": dict with keys past_performance, guidance, expectation (str or None) for discovery explanation,
            }

        If eps_beat is not None, the returned "eps" field will be that value
        (LLM headline_beat.eps is ignored). When eps_beat is None, we use the
        LLM headline_beat.eps mapping.
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

        prompt = THREE_DUCKS_PROMPT_COUNTS.replace("<<<EX99_1_TEXT>>>", text.strip())
        model, parsed = self.ask_gemini(prompt, timeout=120.0)
        if not parsed:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s EX99 LLM: no result from Gemini",
                ticker or "N/A",
                cik or "N/A",
                accession,
            )
            return result_dict

        # Map counts to ratio scores
        for key_src, key_dst in (
            ("past_performance", "past_performance"),
            ("future_performance", "guidance"),
            ("expectation_gap", "expectation"),
        ):
            val = parsed.get(key_src)
            if isinstance(val, dict):
                try:
                    p = int(val.get("positive", 0) or 0)
                    n = int(val.get("negative", 0) or 0)
                    result_dict[key_dst] = _ratio_score(p, n)
                except (TypeError, ValueError):
                    result_dict[key_dst] = None

        # Headline beat mapping
        hb = parsed.get("headline_beat") or {}
        eps_label = hb.get("eps") if isinstance(hb, dict) else None
        rev_label = hb.get("revenue") if isinstance(hb, dict) else None

        def _norm_label(label: Optional[str]) -> Optional[str]:
            if not isinstance(label, str):
                return "unknown"
            label = label.strip().lower()
            if label in {"weak_beat", "beat", "strong_beat", "miss", "unknown"}:
                return label
            return "unknown"

        if eps_beat is not None:
            # Use numeric EPS intel as the authoritative EPS label.
            result_dict["eps"] = eps_beat
        else:
            result_dict["eps"] = _norm_label(eps_label)

        result_dict["revenue"] = _norm_label(rev_label)

        # Justifications for discovery explanation (past_performance, guidance, expectation)
        j = parsed.get("justifications") or {}
        if isinstance(j, dict):
            result_dict["justifications"] = {
                "past_performance": j.get("past_performance") if isinstance(j.get("past_performance"), str) else None,
                "guidance": j.get("future_performance") if isinstance(j.get("future_performance"), str) else None,
                "expectation": j.get("expectation_gap") if isinstance(j.get("expectation_gap"), str) else None,
            }
        else:
            result_dict["justifications"] = None

        # Chcek for hard fails
        passed = True

        g = result_dict.get("guidance")
        e = result_dict.get("expectation")
        p = result_dict.get("past_performance")

        # Any negative ratio score
        for name, v in (("guidance", g), ("expectation", e), ("past_performance", p)):
            if isinstance(v, (int, float)) and v < 0:
                passed = False

        # Revenue miss
        if (rev := result_dict.get("revenue")) == "miss":
            passed = False

        logger.info(
            "ticker=%s, CIK=%s, accession=%s EX99 LLM: model=%s "
            "past=%s guidance=%s expectation=%s eps=%s revenue=%s %s",
            ticker or "N/A",
            cik or "N/A",
            accession,
            model or "N/A",
            f"{result_dict['past_performance']:.3f}" if isinstance(result_dict["past_performance"], (int, float)) else "N/A",
            f"{result_dict['guidance']:.3f}" if isinstance(result_dict["guidance"], (int, float)) else "N/A",
            f"{result_dict['expectation']:.3f}" if isinstance(result_dict["expectation"], (int, float)) else "N/A",
            result_dict["eps"],
            result_dict["revenue"],
            "-> pass"  if passed else "-> fail"
        )

        if passed:
            return result_dict

        return None


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

    def analyze_8k_advanced(self, filing) -> bool:

        # Try and see if EPS intel in AV or Filing
        eps_beat = self.evaluate_eps_beat(filing)

        # First hard-fail
        if eps_beat == "miss":
            return False

        # First AI - anaylase ex99.1 8-K attachment
        if (ex99 := self.analyse_ex99_llm(filing, eps_beat)) is None:
            return False

        print(ex99)
        return True

    def discover(self, sa):
        return
        logger.info("Fetching latest filings...")
        latest = get_latest_filings()
        """
        try:
            filings = list(latest)
        except Exception as e:
            logger.warning("❌ Error converting latest filings to list: %s", e)
            return

        """
        filing1 = find("0001997859-26-000015")
        filing2 = find("0001437749-26-006392")
        filing3 = find("0001628280-26-013191")

        filings = [filing1]

        # Filter to 8-Ks only
        filings_8k = [f for f in filings if getattr(f, "form", None) == "8-K"]
        if not filings_8k:
            logger.info("No latest 8-K filings found.")
            return

        logger.info("Found %d 8-K filings. Running basic inspection (filing + financial filters)...", len(filings_8k))

        for filing in filings_8k:
            try:
                if not self.analyze_8k_basic(filing):
                    continue

                if not self.analyze_8k_advanced(filing):
                    continue

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

