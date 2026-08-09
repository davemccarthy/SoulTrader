"""
Edgar advisor (EDDIE-8): two-stage 8-K earnings pipeline + Form 4 enrichment.

Stage 1 — Item 2.02 8-K → EX-99 LLM + score → watch() (Pending).
Stage 1b — media LLM after delay (AH +2h / pre/RTH +1h) → media_gate pass/fail.
Stage 2 — after RTH open (+15m): Goldilocks tape → discovered() on pass setups.

Test independently:
    python manage.py run_edgar
    python manage.py run_edgar --accession 0001193125-26-084267
    python manage.py run_edgar --watch SHOP,WULF,KYMR
    python manage.py run_edgar --open-check-only
    python manage.py run_edgar --open-check-only --force-open-check --dry-run

Open-check (Goldilocks −3%…+8% vs prior close): needs media_gate=pass and
known EPS beat. Cliffs Excluded; rockets stay Pending (no chase).
Form 4 helpers live in ``core.services.sec.form4``.
"""

import logging
import os
import html
import re
import time
from collections import Counter
from decimal import Decimal
from datetime import date, datetime, time as dt_time, timedelta, timezone as dt_timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple
import requests
import yfinance as yf
from django.conf import settings
from edgar import Company, find, set_identity, get_latest_filings

from core.services.advisors.advisor import AdvisorBase, register
from core.services.sec.form4 import get_form4_intel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SEC identity (mandatory)
# ---------------------------------------------------------------------------
set_identity("Dave McCarthy dave@klynt.com")

# ---------------------------------------------------------------------------
# 8-K helpers and constants
# ---------------------------------------------------------------------------
_CIK_TO_TICKER_CACHE: Dict[str, Optional[str]] = {}

_ET = ZoneInfo("US/Eastern")
_RTH_OPEN = dt_time(9, 30)
_RTH_CLOSE = dt_time(16, 0)

# Open-check: Goldilocks band on media-pass watches; SA hits :00/:15 → need ≥15m.
# Legacy grind constants kept for classify_open_bucket() reference helpers.
OPEN_CHECK_MIN_MINUTES = 15
_GOLDILOCKS_LO = -3.0  # exclusive of cliff: vs_close <= LO → cliff
_GOLDILOCKS_HI = 8.0  # exclusive of rocket: vs_close >= HI → rocket
# SEC current-feed lag vs acceptance time: only skip filings older than
# (prev_SA_started - this window). Bare prev_ts cut drops laggy 8-Ks (e.g. SLVM).
FILING_FEED_LAG_MINUTES = 45
# Stage 1b media delay from filing acceptance (ET): AH ≥16:00 → +2h; else +1h.
MEDIA_DELAY_POST_MARKET_HOURS = 2
MEDIA_DELAY_PRE_MARKET_HOURS = 1
MEDIA_BATCH_MAX = 5  # max media LLM calls per SA / discover pass
MEDIA_MAX_ATTEMPTS = 3  # parse/LLM failures before exclude
_GRIND_GAP_LO = 2.0
_GRIND_GAP_HI = 8.0
_GRIND_MAX_PB = 2.0
_GRIND_MIN_VS_VWAP = -0.25
_HEALTHY_DIP_LO = 1.5
_HEALTHY_DIP_HI = 5.0
_ROCKET_VS_CLOSE = 8.0
_ROCKET_MAX_PB = 2.0


def _parse_meta_datetime(raw: Any) -> Optional[datetime]:
    """Parse ISO datetime from watch meta (UTC preferred)."""
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=dt_timezone.utc)
        return raw.astimezone(dt_timezone.utc)
    if isinstance(raw, str) and raw.strip():
        try:
            dt = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=dt_timezone.utc)
        return dt.astimezone(dt_timezone.utc)
    return None


def _watch_filing_dt(w) -> Optional[datetime]:
    """Filing acceptance UTC from meta.filing_dt, else watch.created."""
    meta = w.meta if isinstance(getattr(w, "meta", None), dict) else {}
    dt = _parse_meta_datetime(meta.get("filing_dt"))
    if dt is not None:
        return dt
    created = getattr(w, "created", None)
    if isinstance(created, datetime):
        if created.tzinfo is None:
            return created.replace(tzinfo=dt_timezone.utc)
        return created.astimezone(dt_timezone.utc)
    return None


def media_delay_hours(filing_dt: datetime) -> int:
    """
    Hours to wait after filing before media LLM.

    Post-market (acceptance ET ≥ 16:00) → +2h; pre-market / RTH → +1h.
    """
    et = filing_dt.astimezone(_ET)
    if et.time() >= _RTH_CLOSE:
        return MEDIA_DELAY_POST_MARKET_HOURS
    return MEDIA_DELAY_PRE_MARKET_HOURS


def media_due_at(filing_dt: datetime) -> datetime:
    """UTC datetime when Stage 1b media check is allowed."""
    return filing_dt + timedelta(hours=media_delay_hours(filing_dt))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except (TypeError, ValueError):
        return None


def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return 100.0 * (a - b) / b


def _session_vwap(hist) -> Optional[float]:
    if hist is None or getattr(hist, "empty", True) or "Close" not in hist.columns:
        return None
    vol = hist["Volume"] if "Volume" in hist.columns else None
    if vol is None or float(vol.sum()) <= 0:
        return _safe_float(hist["Close"].iloc[-1])
    typical = (hist["High"] + hist["Low"] + hist["Close"]) / 3.0
    total_vol = float(vol.sum())
    if total_vol <= 0:
        return None
    return float((typical * vol.astype(float)).sum() / total_vol)


def earnings_open_tape(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Live open-tape snapshot vs prior close (gap, vs_close, pullback, VWAP).

    Uses yfinance 1m bars with prepost; prior close from fast_info when available.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return None
    try:
        t = yf.Ticker(sym)
        prior = None
        last = None
        try:
            fi = t.fast_info
            if isinstance(fi, dict):
                prior = _safe_float(fi.get("previousClose") or fi.get("previous_close"))
                last = _safe_float(fi.get("lastPrice") or fi.get("last_price"))
            else:
                prior = _safe_float(getattr(fi, "previous_close", None))
                last = _safe_float(getattr(fi, "last_price", None))
        except Exception:
            pass

        hist = t.history(period="1d", interval="1m", prepost=True)
        if hist is None or hist.empty:
            hist = t.history(period="1d", interval="1m")
        if hist is None or hist.empty:
            if prior is None or last is None:
                return None
            return {
                "prior_close": prior,
                "open_px": None,
                "last": last,
                "high": last,
                "gap_pct": None,
                "vs_close_pct": _pct(last, prior),
                "vs_open_pct": None,
                "pullback_pct": None,
                "vwap": None,
                "vs_vwap_pct": None,
            }

        idx = hist.index
        try:
            idx_et = idx.tz_convert(_ET) if getattr(idx, "tz", None) else idx
        except Exception:
            idx_et = idx
        times = [ts.time() if hasattr(ts, "time") else ts for ts in idx_et]
        rth_mask = [_RTH_OPEN <= tm <= _RTH_CLOSE for tm in times]
        rth = hist.loc[rth_mask] if any(rth_mask) else hist.iloc[0:0]

        open_px = None
        if not rth.empty and "Open" in rth.columns:
            open_px = _safe_float(rth["Open"].iloc[0])
        if open_px is None and "Open" in hist.columns:
            open_px = _safe_float(hist["Open"].iloc[0])

        high = _safe_float(hist["High"].max()) if "High" in hist.columns else None
        bar_last = _safe_float(hist["Close"].iloc[-1]) if "Close" in hist.columns else None
        if last is None:
            last = bar_last
        if prior is None:
            daily = t.history(period="10d", interval="1d", auto_adjust=True)
            if daily is not None and not daily.empty and "Close" in daily.columns:
                closes = daily["Close"].astype(float).dropna()
                if len(closes) >= 2:
                    # Last completed session close when today already has a bar.
                    prior = _safe_float(closes.iloc[-2])
                elif len(closes) == 1:
                    prior = _safe_float(closes.iloc[-1])

        vwap = _session_vwap(hist if hist is not None else rth)
        pb = None
        if last is not None and high is not None and high > 0:
            pb = max(0.0, 100.0 * (high - last) / high)

        return {
            "prior_close": prior,
            "open_px": open_px,
            "last": last,
            "high": high,
            "gap_pct": _pct(open_px, prior),
            "vs_close_pct": _pct(last, prior),
            "vs_open_pct": _pct(last, open_px),
            "pullback_pct": pb,
            "vwap": vwap,
            "vs_vwap_pct": _pct(last, vwap),
        }
    except Exception as e:
        logger.warning("earnings_open_tape(%s) failed: %s", sym, e)
        return None


def classify_goldilocks_bucket(tape: Dict[str, Any]) -> str:
    """
    Goldilocks open-check (mimics remote backtest bands): cliff | goldilocks | rocket | unclear.

    cliff:      vs prior close ≤ −3% → dismiss
    goldilocks: −3% < vs < +8% → punt at live quote (soft red included)
    rocket:     vs ≥ +8% → no chase
    unclear:    no vs_close
    """
    vs_close = tape.get("vs_close_pct")
    if vs_close is None:
        return "unclear"
    if vs_close <= _GOLDILOCKS_LO:
        return "cliff"
    if vs_close >= _GOLDILOCKS_HI:
        return "rocket"
    return "goldilocks"


def classify_open_bucket(tape: Dict[str, Any]) -> str:
    """
    Legacy four-way open-check: down | grinder | rocket | unclear.

    Prefer classify_goldilocks_bucket() for the live earnings path.
    """
    vs_close = tape.get("vs_close_pct")
    gap = tape.get("gap_pct")
    vs_open = tape.get("vs_open_pct")
    pb = tape.get("pullback_pct")
    vs_vwap = tape.get("vs_vwap_pct")

    if vs_close is None:
        return "unclear"
    if vs_close <= 0.0:
        return "down"

    if vs_close >= _ROCKET_VS_CLOSE and (pb is None or pb < _ROCKET_MAX_PB):
        return "rocket"

    # Healthy dip reclaim (gap–dip–rip forming with VWAP hold).
    if (
        pb is not None
        and _HEALTHY_DIP_LO <= pb <= _HEALTHY_DIP_HI
        and vs_vwap is not None
        and vs_vwap >= -0.5
        and vs_close >= 1.0
    ):
        return "grinder"

    # Classic gap & grind.
    if (
        gap is not None
        and _GRIND_GAP_LO <= gap <= _GRIND_GAP_HI
        and pb is not None
        and pb < _GRIND_MAX_PB
        and vs_vwap is not None
        and vs_vwap >= _GRIND_MIN_VS_VWAP
        and vs_open is not None
        and vs_open >= -0.5
        and vs_close >= 1.0
    ):
        return "grinder"

    # Softer grind: +2–8% vs prior, above VWAP, shallow PB.
    if (
        2.0 <= vs_close <= 8.0
        and vs_vwap is not None
        and vs_vwap >= _GRIND_MIN_VS_VWAP
        and (pb is None or pb < 3.0)
    ):
        return "grinder"

    return "unclear"


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


def _extract_item_section_text(filing_text: str, item_number: str) -> str:
    """Best-effort extraction of a specific 8-K item section."""
    if not filing_text or not item_number:
        return ""
    text = re.sub(r"\s+", " ", filing_text)
    start_re = re.compile(rf"\bitem\s*{re.escape(item_number)}\b", re.IGNORECASE)
    next_item_re = re.compile(r"\bitem\s*\d+\.\d+\b", re.IGNORECASE)

    start_match = start_re.search(text)
    if not start_match:
        return ""
    remainder = text[start_match.end():]
    end_match = next_item_re.search(remainder)
    if not end_match:
        return remainder.strip()
    return remainder[:end_match.start()].strip()


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
# EPS evidence in EX-99 text. Allow keyword→number and "$x.xx per (diluted) share"
# (SLVM-style), plus parenthetical losses like (0.28).
EPS_NUMBER_PATTERN = re.compile(
    r"(?:"
    r"(?:earnings(?:\s*\(\s*loss\s*\))?\s+per\s+share|\beps\b|\bdiluted\b)"
    r".{0,60}?"
    r"\(?-?\$?\d+\.\d{1,4}\)?"
    r"|"
    r"\(?-?\$?\d+\.\d{1,4}\)?"
    r".{0,24}?"
    r"per\s+(?:diluted\s+|basic\s+)?share"
    r")",
    re.IGNORECASE,
)


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

# Financial filter (Filter 3)
FILTER3_PE_MAX = 100
FILTER3_MIN_CAP = 150e6
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
BEAT_THRESHOLD = 2.0
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

MEDIA_REACTION_PROMPT_TEMPLATE = """Analyze {company} ({ticker}) {quarter} earnings release from the perspective of a professional buy-side equity analyst.

Search the web for recent business and financial news and analysis about this event.

Use reputable sources such as Bloomberg, Reuters, CNBC, Financial Times, Wall Street Journal, Barron's, MarketWatch, and major broker research (e.g., Goldman Sachs, Morgan Stanley, JPMorgan) where available.

Your tasks:
1. Assess the overall sentiment of coverage toward the earnings and outlook.
2. Determine whether the company beat or missed consensus expectations on EPS and Revenue (when such information is available).
3. Identify key positive themes and significant red flags (including analyst downgrades or cautious notes) across articles and broker research.

Do not summarize stale pre-earnings consensus ratings; focus on post-release news and fresh commentary.

Respond with STRICT JSON only. No other text before or after:
{{
  "sentiment": "strong_positive" | "positive" | "mixed" | "negative" | "no_coverage",
  "eps": "strong_beat" | "beat" | "miss" | "other" | "unknown",
  "revenue": "strong_beat" | "beat" | "miss" | "other" | "unknown",
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
- \"eps\": \"unknown\",
- \"revenue\": \"unknown\"
"""

PHARMA_ITEM_PROMPT_TEMPLATE = """You are a biotech/pharma event analyst.

Classify the 8-K disclosure using only Item 7.01 and Item 8.01 text.
Do not use outside knowledge. Do not infer facts not present in the text.

Return STRICT JSON only:
{{
  "category": "regulatory" | "clinical" | "commercial" | "legal" | "mixed" | "other",
  "sentiment": "strong_positive" | "positive" | "neutral" | "negative",
  "materiality": "high" | "medium" | "low",
  "key_event": "<short phrase>",
  "timeline": "<short phrase or unknown>",
  "red_flags": ["<risk 1>", "<risk 2>"],
  "summary": "<1-2 sentence summary>"
}}

If information is missing, use "unknown" for timeline and [] for red_flags.

Item 7.01:
<<<ITEM_701_TEXT>>>

Item 8.01:
<<<ITEM_801_TEXT>>>
"""

# Pharma regex scoring thresholds (item-text only; no EX-99 dependency)
PHARMA_SCORE_REJECT_MAX = 2
PHARMA_SCORE_LLM_MIN = 5

# Event regex scoring thresholds (log-only rollout)
EVENT_SCORE_REJECT_MAX = 2
EVENT_SCORE_PASS_MIN = 5

# First pipe segment of discovery explanation: optional lead from media LLM headlines
_EXPLANATION_HEADLINE_MAX_LEN = 180

# Media gate: require a known EPS beat (Fri autopsy: unknown only hit losers TDS/AD).
_MEDIA_EPS_PASS = frozenset({"beat", "strong_beat"})


def media_passes_gate(media: Optional[Dict[str, Any]]) -> bool:
    """
    True when media reaction is buy-eligible.

    Hard fails: no media, sentiment in {no_coverage, mixed, negative}, eps miss,
    or eps not in {beat, strong_beat} (unknown/other fail — no confirmed beat).
    """
    if not isinstance(media, dict) or not media:
        return False
    sentiment = media.get("sentiment")
    eps = media.get("eps")
    if sentiment in ("no_coverage", "mixed", "negative"):
        return False
    if eps == "miss" or eps not in _MEDIA_EPS_PASS:
        return False
    return True


def _media_gate_explanation_segment(media: Optional[Dict[str, Any]]) -> str:
    """Short UI segment: media sentiment/eps/revenue (e.g. media +/beat/beat)."""
    if not isinstance(media, dict) or not media:
        return ""
    sent = str(media.get("sentiment") or "?")
    eps = str(media.get("eps") or "?")
    rev = str(media.get("revenue") or "?")
    sent_short = {
        "strong_positive": "++",
        "positive": "+",
        "mixed": "~",
        "negative": "-",
        "no_coverage": "nc",
    }.get(sent, sent[:8])
    return f"media {sent_short}/{eps}/{rev}"


def _first_media_headline_for_explanation(media: dict) -> str:
    """First non-empty headline, safe for pipe-delimited explanation strings."""
    raw = media.get("headlines")
    if not isinstance(raw, list):
        return ""
    for item in raw:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        s = s.replace("|", "·").replace("\n", " ")
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            continue
        if len(s) > _EXPLANATION_HEADLINE_MAX_LEN:
            s = s[: _EXPLANATION_HEADLINE_MAX_LEN - 1].rstrip() + "…"
        return s
    return ""


_EXPLANATION_SEGMENT_MAX_LEN = 500
_EXPLANATION_LIST_ITEM_MAX = 4


def _sanitize_explanation_segment(text: str, max_len: int = _EXPLANATION_SEGMENT_MAX_LEN) -> str:
    """Flatten text for a single pipe-delimited discovery explanation segment."""
    s = str(text or "").strip()
    if not s:
        return ""
    s = s.replace("|", "·").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


def _edgar_detail_explanation_segments(
    ex99: dict,
    media: dict,
    form4: dict,
    bonuses: list,
    penalties: list,
) -> List[str]:
    """Extra explanation segments after the lead (EX-99, media, scoring, Form4)."""
    segments: List[str] = []

    ex99_pairs = []
    for key in ("expectation", "guidance", "past_performance", "market_reaction"):
        value = ex99.get(key)
        if value is not None and str(value).strip():
            ex99_pairs.append(f"{key}={value}")
    if ex99_pairs:
        segments.append(f"EX-99: {' '.join(ex99_pairs)}")

    justifications = ex99.get("justifications") if isinstance(ex99.get("justifications"), dict) else {}
    note_parts = []
    for key, label in (
        ("past_performance", "Past Performance"),
        ("guidance", "Guidance"),
        ("expectation", "Expectation"),
        ("market_reaction", "Market Reaction"),
    ):
        text = justifications.get(key)
        if text:
            clean = _sanitize_explanation_segment(text, max_len=220)
            if clean:
                note_parts.append(f"{label}: {clean}")
    if note_parts:
        segments.append(f"EX-99 notes: {' | '.join(note_parts)}")

    media_pairs = []
    for key in ("sentiment", "eps", "revenue"):
        value = media.get(key)
        if value is not None and str(value).strip():
            media_pairs.append(f"{key}={value}")
    if media_pairs:
        segments.append(f"Media: {' '.join(media_pairs)}")

    summary = _sanitize_explanation_segment(media.get("summary") or "")
    if summary:
        segments.append(f"Media summary: {summary}")

    headlines = media.get("headlines") if isinstance(media.get("headlines"), list) else []
    headline_bits = []
    for item in headlines[:_EXPLANATION_LIST_ITEM_MAX]:
        clean = _sanitize_explanation_segment(item, max_len=160)
        if clean:
            headline_bits.append(clean)
    if headline_bits:
        segments.append(f"Headlines: {'; '.join(headline_bits)}")

    red_flags = media.get("red_flags") if isinstance(media.get("red_flags"), list) else []
    flag_bits = []
    for item in red_flags[:_EXPLANATION_LIST_ITEM_MAX]:
        clean = _sanitize_explanation_segment(item, max_len=160)
        if clean:
            flag_bits.append(clean)
    if flag_bits:
        segments.append(f"Red flags: {'; '.join(flag_bits)}")

    if bonuses:
        bits = [_sanitize_explanation_segment(b, max_len=120) for b in bonuses]
        bits = [b for b in bits if b]
        if bits:
            segments.append(f"Bonuses: {'; '.join(bits)}")

    if penalties:
        bits = [_sanitize_explanation_segment(p, max_len=120) for p in penalties]
        bits = [b for b in bits if b]
        if bits:
            segments.append(f"Penalties: {'; '.join(bits)}")

    if isinstance(form4, dict) and form4:
        form4_bits = []
        kind = form4.get("watch_kind")
        if kind:
            form4_bits.append(f"kind={kind}")
        total = form4.get("total")
        if total is not None:
            form4_bits.append(f"total={total}")
        if form4_bits:
            segments.append(f"Form4: {' '.join(form4_bits)}")

    return segments


def resolve_latest_8k(symbol: str):
    """
    Latest 8-K for ticker via edgar Company. Prefer Item 2.02 (earnings) when present.
    Returns filing or None.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return None
    try:
        company = Company(sym)
        filings = list(company.get_filings(form="8-K").head(8))
    except Exception as e:
        logger.warning("resolve_latest_8k(%s): %s", sym, e)
        return None
    if not filings:
        return None

    def _items(f) -> str:
        raw = getattr(f, "items", None) or ""
        return str(raw)

    earnings = [f for f in filings if "2.02" in _items(f)]
    return earnings[0] if earnings else filings[0]


# ---------------------------------------------------------------------------
# ED-8 advisor class and command entry
# ---------------------------------------------------------------------------

class Edgar(AdvisorBase):
    """Advisor for 8-K earnings filings (basic filters only in this step)."""

    FORM4_WATCH_MIN_TOTAL = 5.0
    FORM4_WATCH_MAX_SELL_TOTAL = -5.0
    FORM4_WATCH_DAYS = 30

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
            return _fail(f"market cap too low (< ${FILTER3_MIN_CAP / 1e6:.0f}M)")

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

    def has_item(self, filing, item: str) -> bool:
        item_norm = str(item).strip().lower()  # e.g. "2.02"

        # 1) structured items field
        items = getattr(filing, "items", None)
        if isinstance(items, str):
            if item_norm in items.lower():
                return True
        elif isinstance(items, (list, tuple, set)):
            for it in items:
                if item_norm in str(it).lower():
                    return True

        # 2) fallback on filing text
        try:
            text = (filing.text() or "").lower()
        except Exception:
            text = ""

        # escape dots etc, then allow optional "item" prefix
        item_re = re.escape(item_norm)  # "2\\.02"
        return bool(re.search(rf"\bitem\s*{item_re}\b", text))

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
        model, parsed = self.ask_deepseek(prompt)
        
        if not parsed:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s EX99 LLM: no result from LLM",
                ticker or "N/A",
                cik or "N/A",
                accession,
            )
            # No LLM - try again later
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

        model, parsed = self.ask_deepseek(media_prompt)

        if not parsed or not isinstance(parsed, dict):
            logger.info(
                "ticker=%s, CIK=%s, accession=%s media LLM: no result from LLM(s)",
                ticker or "N/A",
                cik or "N/A",
                accession,
            )
            return None

        if parsed.get("sentiment") == "no_coverage":
            logger.info(
                "ticker=%s, CIK=%s, accession=%s media LLM: no_coverage - retrying",
                ticker or "N/A",
                cik or "N/A",
                accession,
            )
            model, parsed = self.ask_gemini(media_prompt, use_search=True)

        if not parsed or not isinstance(parsed, dict):
            logger.info(
                "ticker=%s, CIK=%s, accession=%s media LLM: no result from LLM(s)",
                ticker or "N/A",
                cik or "N/A",
                accession,
            )
            return None

        print("-------")
        print(parsed)
        print("-------")

        sentiment = parsed.get("sentiment")
        eps = parsed.get("eps")
        revenue = parsed.get("revenue")
        headlines = parsed.get("headlines")
        red_flags = parsed.get("red_flags")
        summary = parsed.get("summary")

        result: Dict[str, Optional[object]] = {
            "sentiment": sentiment,
            "eps": eps,
            "revenue": revenue,
            "headlines": headlines if isinstance(headlines, list) else [],
            "red_flags": red_flags if isinstance(red_flags, list) else [],
            "summary": summary if isinstance(summary, str) else None
        }

        # Caller applies media_passes_gate; always return parsed result when available.
        if not media_passes_gate(result):
            logger.info(
                "ticker=%s, accession=%s media LLM: "
                "(eps=%s, revenue=%s, sentiment=%s) -> fail",
                ticker,
                accession,
                eps,
                revenue,
                sentiment or "N/A",
            )
        else:
            logger.info(
                "ticker=%s, accession=%s media LLM: "
                "(eps=%s, revenue=%s, sentiment=%s) -> pass",
                ticker,
                accession,
                eps,
                revenue,
                sentiment or "N/A",
            )

        return result

    def match_form4(self, filing):

        cik = str(getattr(filing, "cik", "") or "")
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)
        accession = (
            getattr(filing, "accession_no", None)
            or getattr(filing, "accession_number", None)
            or ""
        )
        filing_dt = _filing_datetime(filing)

        if not ticker:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s Form4 match: no ticker",
                "N/A",
                cik or "N/A",
                accession,
            )
            return None

        end_date = filing_dt.date() if filing_dt is not None else None
        try:
            result = get_form4_intel(
                ticker,
                days=self.FORM4_WATCH_DAYS,
                end_date=end_date,
                limit=100,
            )
        except Exception as e:
            logger.warning(
                "ticker=%s, accession=%s Form4 on-demand lookup failed: %s",
                ticker,
                accession,
                e,
            )
            return None

        if not result.get("entry_count"):
            logger.info(
                "ticker=%s, accession=%s Form4 match: none (filings=%s parsed=%s)",
                ticker,
                accession,
                result.get("filing_count"),
                result.get("parsed_count"),
            )
            return None

        form4_total = 0.0
        try:
            form4_total = float(result.get("total") or 0.0)
        except (TypeError, ValueError):
            form4_total = 0.0

        if (
            form4_total < self.FORM4_WATCH_MIN_TOTAL
            and form4_total > self.FORM4_WATCH_MAX_SELL_TOTAL
        ):
            logger.info(
                "ticker=%s, accession=%s Form4 match: below threshold total=%.2f entries=%s buy=%s sell=%s",
                ticker,
                accession,
                form4_total,
                result.get("entry_count"),
                result.get("buy_count"),
                result.get("sell_count"),
            )
            return None

        result["watch_kind"] = "form4_signal" if form4_total >= 0 else "form4_sell"

        logger.info(
            "ticker=%s, accession=%s Form4 match: entries=%s net_total=%.2f buy=%s sell=%s latest=%s range=%s..%s",
            ticker,
            accession,
            result.get("entry_count"),
            form4_total,
            result.get("buy_count"),
            result.get("sell_count"),
            result.get("form4_accession") or "N/A",
            result.get("start_date"),
            result.get("end_date"),
        )
        return result

    def build_info(self, filing, advanced: dict) -> List[str]:
        """Build pipe-delimited discovery explanation segments (segment 0 = trade lead)."""
        cik = str(getattr(filing, "cik", "") or "")
        accession = (
            getattr(filing, "accession_no", None)
            or getattr(filing, "accession_number", None)
            or ""
        )

        weight = advanced.get("weight")
        if weight is None or not isinstance(weight, (int, float)):
            weight = 1.0

        ex99 = advanced.get("ex99") or {}
        media = advanced.get("media") or {}
        form4 = advanced.get("form4") or {}
        bonuses = advanced.get("bonuses") or []
        penalties = advanced.get("penalties") or []

        headline = _first_media_headline_for_explanation(media)
        if headline:
            lead = (
                f"{headline} | Accession: {accession} | "
                f"Weight:{weight:.2f} | https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude "
            )
        else:
            lead = (
                f"8-K earnings filing | Accession: {accession} | Weight:{weight:.2f} | "
                f"https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude "
            )

        parts = [lead]
        parts.extend(
            _edgar_detail_explanation_segments(ex99, media, form4, bonuses, penalties)
        )
        return parts


    def score_results(self, filing, ex99, media, form4):

        cik = str(getattr(filing, "cik", "") or "")
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)

        bonuses: list[str] = []
        penalties: list[str] = []
        score = 0.0

        # Filing time (ET): 6:30–7:30 inclusive +0.0, 16:00–20:00 inclusive -0.1
        filing_dt = _filing_datetime(filing)
        if filing_dt is not None:
            et = ZoneInfo("America/New_York")
            t_et = filing_dt.astimezone(et).time()
            if dt_time(6, 30) <= t_et <= dt_time(7, 30):
                score += 0.0
            elif dt_time(16, 0) <= t_et <= dt_time(20, 0):
                penalties.append("After-hours filing (-0.1)")
                score -= 0.1

        # 4 ducks
        past_perform = ex99['past_performance']
        expectation = ex99['expectation']
        guidance = ex99['guidance']
        market = ex99['market_reaction']

        if past_perform == "strong_positive":
            score += 0.0
        elif past_perform == "neutral":
            penalties.append("Neutral past (-0.1)")
            score -= 0.1

        if expectation == "strong_positive":
            bonuses.append("Strong expectation (+0.1)")
            score += 0.1
        elif expectation == "neutral":
            penalties.append("Neutral expectation (-0.1)")
            score -= 0.1

        if guidance == "strong_positive":
            bonuses.append("Strong guidance (+0.1)")
            score += 0.1

        if market == "strong_positive":
            bonuses.append("Strong market (+0.1)")
            score += 0.1

        # Media
        sentiment = media.get("sentiment")
        eps = media.get("eps")
        revenue =  media.get("revenue")

        if sentiment == "strong_positive":
            bonuses.append("Strong reaction (+0.1)")
            score += 0.1

        if eps == "strong_beat":
            bonuses.append("Strong eps (+0.1)")
            score += 0.1
        elif eps in ["weak_beat","unknown"]:
            penalties.append("Weak or unknown eps (-0.2)")
            score -= 0.2

        if revenue == "strong_beat":
            bonuses.append("Strong revenue (+0.05)")
            score += 0.05
        elif revenue == "weak_beat":
            penalties.append("Weak revenue (-0.1)")
            score -= 0.1
        elif revenue == "neutral":
            penalties.append("Neutral revenue (-0.2)")
            score -= 0.2
        elif revenue == "miss":
            penalties.append("Revenue miss (-0.3)")
            score -= 0.3

        # Form4 (optional, non-blocking): bullish insider cluster can help,
        # significant selling can add risk.
        form4_total = None
        form4_kind = None
        if isinstance(form4, dict):
            form4_kind = str(form4.get("watch_kind") or "")
            raw_total = form4.get("total")
            try:
                form4_total = float(raw_total) if raw_total is not None else None
            except (TypeError, ValueError):
                form4_total = None

        if form4_total is not None:
            if form4_kind == "form4_sell" or form4_total <= self.FORM4_WATCH_MAX_SELL_TOTAL:
                if form4_total <= -8.0:
                    penalties.append("Strong Form4 sell pressure (-0.30)")
                    score -= 0.30
                else:
                    penalties.append("Form4 sell pressure (-0.20)")
                    score -= 0.20
            elif form4_kind == "form4_signal" or form4_total >= self.FORM4_WATCH_MIN_TOTAL:
                if form4_total >= 8.0:
                    bonuses.append("Strong Form4 buy signal (+0.15)")
                    score += 0.15
                else:
                    bonuses.append("Form4 buy signal (+0.1)")
                    score += 0.1

        # Valuation
        valuation = self.evaluate_stock(ticker)

        if valuation > 1.0:
            over = valuation - 1.0
            penalties.append(f"Overvalued (ratio {valuation:.2f}, {-over:+.2f})")
            score -= over
        elif valuation < 1.0:
            under = 1.0 - valuation
            bonuses.append(f"Undervalued (ratio {valuation:.2f}, {under:+.2f})")
            score += under

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

        # Consensus
        if consensus := self.stock_consensus(ticker):
            upside_to_mean_pct = float(consensus.get("upside_to_mean_pct") or 0.0)
            upside_to_low_pct = float(consensus.get("upside_to_low_pct") or 0.0)
            recommendation_key = str(consensus.get("recommendation_key") or "")

            if upside_to_mean_pct < 0:
                penalties.append("Above consensus mean (-0.1)")
                score -= 0.1

            target_high = consensus.get("target_high")
            current_price = consensus.get("current_price")
            if target_high is not None and current_price is not None:
                try:
                    price = float(current_price)
                    high = float(target_high)
                    if price > 0 and high > 0:
                        upside_to_high_pct = (high - price) / price * 100.0
                        if upside_to_high_pct < 0:
                            penalties.append("Above consensus high (-0.1)")
                            score -= 0.1
                except (TypeError, ValueError):
                    pass

            if upside_to_low_pct > 0:
                bonuses.append("Below consensus low (+0.1)")
                score += 0.1

            if recommendation_key == "hold":
                penalties.append("Consensus: hold (-0.1)")
                score -= 0.1

            if recommendation_key == "sell":
                penalties.append("Consensus: sell (-0.2)")
                score -= 0.2

        return score, bonuses, penalties

    def analyze_8k_earnings(self, filing, cik, ticker, accession, sa):
        """
        Item 2.02 earnings path.

        Returns True when this filing is owned as earnings (pass → watch, or
        rejected with a logged reason). Returns False only when not an
        earnings filing (no Item 2.02) so callers may try pharma/event.
        """
        # Verify item 2.02
        if not self.has_item(filing, "2.02"):
            logger.info(
                "ticker=%s, CIK=%s, accession=%s no item 2.02 -> not earnings filing",
                ticker,
                cik,
                accession,
            )
            return False

        # Verify earnings text
        exhibit_99_text = _get_ex99_text(filing)
        if not (exhibit_99_text or "").strip():
            logger.info(
                "ticker=%s, CIK=%s, accession=%s earnings text gate fail "
                "(no EX-99 text) -> fail",
                ticker,
                cik,
                accession,
            )
            return True

        text = _filter4_normalize(exhibit_99_text)
        checks = {
            "earnings_context": _filter4_has_earnings_context(text),
            "eps_evidence": _filter4_has_eps_evidence(text),
            "numeric_density": _filter4_numeric_density(text),
            "table_structure": _filter4_has_table_structure(text),
            "comparables": _filter4_has_comparables(text),
        }
        if not checks["earnings_context"]:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s earnings text gate fail "
                "(earnings_context) -> fail",
                ticker,
                cik,
                accession,
            )
            return True
        if not checks["eps_evidence"]:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s earnings text gate fail "
                "(eps_evidence) -> fail",
                ticker,
                cik,
                accession,
            )
            return True
        if checks["numeric_density"] < 15:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s earnings text gate fail "
                "(numeric_density=%s < 15) -> fail",
                ticker,
                cik,
                accession,
                checks["numeric_density"],
            )
            return True

        score = sum([checks["table_structure"], checks["comparables"]])

        # Check soft fail
        if score < 1:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s earnings soft score -> fail "
                "(table=%s comparables=%s)",
                ticker,
                cik,
                accession,
                checks["table_structure"],
                checks["comparables"],
            )
            return True

        # Try and see if EPS intel in AV or Filing
        eps_beat = self.evaluate_eps_beat(filing)

        # First hard-fail
        if eps_beat == "miss":
            logger.info(
                "ticker=%s, CIK=%s, accession=%s eps miss -> fail",
                ticker,
                cik,
                accession,
            )
            return True

        # First AI - analyse ex99.1 8-K attachment
        if (ex99 := self.analyse_ex99_llm(filing)) is None:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s EX-99 LLM -> fail",
                ticker,
                cik,
                accession,
            )
            return True

        # Media LLM skipped (too soon after filing; open-tape stage gates buys).
        media: Dict[str, Any] = {}

        # Form4 association
        form4 = self.match_form4(filing)

        score, bonuses, penalties = self.score_results(filing, ex99, media, form4)

        # Minimize score impact
        weight = (score / 2) + 1.0

        explanation = " | ".join(
            self.build_info(
                filing,
                {
                    "weight": weight,
                    "bonuses": bonuses,
                    "penalties": penalties,
                    "ex99": ex99,
                    "media": media,
                    "form4": form4,
                },
            )
        )

        # Watch only — Stage 1b media + Stage 2 open-check promote later.
        filing_dt = _filing_datetime(filing)
        self.watch(
            ticker,
            explanation=explanation,
            days=2,
            meta={
                "source": "edgar_earnings",
                "accession": accession,
                "filing_dt": filing_dt.isoformat() if filing_dt else None,
                "weight": float(weight),
                "sa_id": getattr(sa, "id", None),
            },
            status="Pending",
        )
        return True

    def analyze_8k_pharma(self, filing, cik, ticker, accession, sa):

        # Verify no item 2.02 (earnings owns those; should not reach here if
        # analyze_8k_earnings returned True for a 2.02 filing).
        if self.has_item(filing, "2.02"):
            logger.debug(
                "ticker=%s, CIK=%s, accession=%s has item 2.02 -> skip pharma",
                ticker,
                cik,
                accession,
            )
            return False

        has_701 = self.has_item(filing, "7.01")
        has_801 = self.has_item(filing, "8.01")
        if not (has_701 or has_801):
            logger.info("ticker=%s, CIK=%s, accession=%s no item 7.01 or 8.01 -> not pharma filing ", ticker, cik, accession)
            return False

        filing_text = ""
        try:
            filing_text = filing.text() or ""
        except Exception:
            filing_text = ""

        item_701_text = _extract_item_section_text(filing_text, "7.01") if has_701 else ""
        item_801_text = _extract_item_section_text(filing_text, "8.01") if has_801 else ""
        text = _filter4_normalize(f"{item_701_text} {item_801_text}")
        if not text:
            logger.info("ticker=%s, CIK=%s, accession=%s no item 7.01 or 8.01 text -> not pharma filing", ticker, cik,accession)
            return False

        weighted_patterns = {
            "regulatory": [
                (4, r"\b(?:ind|investigational new drug|nda|new drug application|bla|biologics license application|sbla|supplemental nda|pdufa|complete response letter|crl)\b"),
                (4, r"\b(?:ema|european medicines agency|chmp|committee for medicinal products for human use)\b"),
                (4, r"\b(?:marketing authoriz\w+|conditional marketing authoriz\w+)\b"),
                (2, r"\b(?:fda|u\.?\s*s\.?\s*food and drug administration|mhra|pmda)\b"),
                (2, r"\b(?:approval|approved|clearance|authoriz\w+|label expansion|positive opinion|recommend\w+)\b"),
                (2, r"\b(?:fast track|breakthrough therapy|orphan drug|priority review|clinical hold|advisory committee)\b"),
            ],
            "clinical": [
                (3, r"\bphase\s*(?:1|2|3|i|ii|iii)\b"),
                (3, r"\b(?:topline|top-line)\s+(?:data|results?)\b"),
                (3, r"\b(?:primary endpoint|secondary endpoint|interim data|efficacy|safety|registrational trial|pivotal study)\b"),
                (2, r"\b(?:dose escalation|dose expansion|patient enrollment|randomized|placebo-controlled|double-blind|open-label|proof-of-concept|cohort)\b"),
                (2, r"\b(?:met|missed|did not meet)\s+(?:its\s+)?primary endpoint\b"),
            ],
            "commercial_legal": [
                (2, r"\b(?:exclusive license|exclusive licence|license agreement|licensing agreement)\b"),
                (2, r"\b(?:collaboration|partnership|milestone|royalt(?:y|ies)|commercializ\w+)\b"),
                (2, r"\b(?:patent|litigation|settlement)\b"),
            ],
            "product_therapeutic": [
                (2, r"\b(?:monotherapy|treatment of|relapsed|refractory)\b"),
                (1, r"\b(?:candidate|pipeline|therapeutic|biologic|small molecule|monoclonal antibody)\b"),
                (1, r"\b(?:gene therapy|cell therapy|mrna|car-?t|antibody-drug conjugate|vaccine|immunotherapy)\b"),
                (1, r"\b(?:oncology|rare disease|autoimmune|cns|antiviral)\b"),
            ],
            "filing_context": [
                (1, r"\bexhibit\s*99\.1\b"),
                (1, r"\bcorporate presentation\b"),
            ],
        }

        pharma_score = 0
        matched_signals = {}
        has_primary_signal = False  # regulatory/clinical only
        for bucket, weighted in weighted_patterns.items():
            bucket_hits = []
            for weight, pattern in weighted:
                m = re.search(pattern, text, re.IGNORECASE)
                if not m:
                    continue
                snippet = m.group(0).strip()
                bucket_hits.append({"weight": weight, "match": snippet})
                pharma_score += weight
                if bucket in ("regulatory", "clinical"):
                    has_primary_signal = True
            if bucket_hits:
                matched_signals[bucket] = bucket_hits

        if not has_primary_signal:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s no primary regulatory/clinical signal -> not pharma filing",
                ticker,
                cik,
                accession,
            )
            return False

        if pharma_score <= PHARMA_SCORE_REJECT_MAX:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s pharma score=%d <= %d -> reject",
                ticker,
                cik,
                accession,
                pharma_score,
                PHARMA_SCORE_REJECT_MAX,
            )
            return False

        logger.info(
            "ticker=%s, CIK=%s, accession=%s pharma score=%d claimed via items (7.01=%s, 8.01=%s) buckets=%s",
            ticker,
            cik,
            accession,
            pharma_score,
            has_701,
            has_801,
            {k: len(v) for k, v in matched_signals.items()},
        )

        if pharma_score < PHARMA_SCORE_LLM_MIN:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s pharma score=%d below LLM threshold=%d -> no LLM/discovery",
                ticker,
                cik,
                accession,
                pharma_score,
                PHARMA_SCORE_LLM_MIN,
            )
            return True

        # Optional LLM interpretation for pharma/event filings (item-driven only).
        prompt = (
            PHARMA_ITEM_PROMPT_TEMPLATE
            .replace("<<<ITEM_701_TEXT>>>", item_701_text[:8000] if item_701_text else "")
            .replace("<<<ITEM_801_TEXT>>>", item_801_text[:8000] if item_801_text else "")
        )
        logger.info(
            "ticker=%s, CIK=%s, accession=%s pharma LLM prompt items (7.01_len=%d, 8.01_len=%d)",
            ticker,
            cik,
            accession,
            len(item_701_text),
            len(item_801_text),
        )
        model, parsed = self.ask_llm(prompt)
        if isinstance(parsed, dict):
            sentiment = str(parsed.get("sentiment") or "").strip().lower()
            category = str(parsed.get("category") or "other").strip().lower()
            materiality = str(parsed.get("materiality") or "low").strip().lower()
            key_event = str(parsed.get("key_event") or "").strip()
            summary = str(parsed.get("summary") or "").strip()
            logger.info(
                "ticker=%s, CIK=%s, accession=%s pharma LLM: model=%s category=%s sentiment=%s materiality=%s",
                ticker,
                cik,
                accession,
                model or "N/A",
                category,
                sentiment,
                materiality,
            )

            weight = None
            if sentiment == "strong_positive":
                weight = Decimal("1.10")
            elif sentiment == "positive":
                weight = Decimal("1.00")

            if weight is not None:
                explanation = (
                    f"8-K pharma | Accession: {accession} | Weight:{float(weight):.2f} | "
                    f"https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude "
                    f"| Category:{category} Sentiment:{sentiment} Materiality:{materiality}"
                )
                if key_event:
                    explanation += f" | Event:{key_event}"
                if summary:
                    explanation += f" | Summary:{summary}"
                self.discovered(sa, ticker, explanation, weight=weight)
            else:
                logger.info(
                    "ticker=%s, CIK=%s, accession=%s pharma sentiment=%s -> no discovery",
                    ticker,
                    cik,
                    accession,
                    sentiment or "unknown",
                )
        else:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s pharma LLM: no valid JSON result",
                ticker,
                cik,
                accession,
            )
        return True

    def analyze_8k_event(self, filing, cik, ticker, accession, sa):
        """
        Event-category triage (log-only for now).
        Uses item-gated weighted regex scoring and emits WOULD_DISCOVER style logs.
        """
        _ = sa

        # Earnings ownership always wins.
        if self.has_item(filing, "2.02"):
            return False

        # Event-relevant 8-K items.
        event_items = ("1.01", "2.01", "2.03", "2.04", "3.02", "7.01", "8.01")
        item_flags = {item: self.has_item(filing, item) for item in event_items}
        if not any(item_flags.values()):
            return False

        filing_text = ""
        try:
            filing_text = filing.text() or ""
        except Exception:
            filing_text = ""

        sections: list[str] = []
        for item in event_items:
            if item_flags.get(item):
                sec = _extract_item_section_text(filing_text, item)
                if sec:
                    sections.append(sec)
        text = _filter4_normalize(" ".join(sections))
        if not text:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s event: no item section text -> reject",
                ticker,
                cik,
                accession,
            )
            return False

        weighted_patterns = {
            "mna": [
                (5, r"\b(?:merger agreement|definitive merger agreement|business combination)\b"),
                (4, r"\b(?:acquisition|strategic alternatives|sale process|take-private|special committee)\b"),
                (3, r"\b(?:asset purchase agreement|letter of intent|go-shop)\b"),
            ],
            "contracts": [
                (5, r"\b(?:government contract|federal contract|defense contract)\b"),
                (4, r"\b(?:multi-year agreement|enterprise agreement|exclusive partnership)\b"),
                (3, r"\b(?:commercial agreement|strategic partnership|distribution agreement|joint venture)\b"),
                (2, r"\b(?:customer agreement|supplier agreement|purchase order)\b"),
            ],
            "financing": [
                (5, r"\b(?:covenant waiver|event of default|liquidity crisis)\b"),
                (4, r"\b(?:debt refinancing|amended credit facility|credit agreement|term loan|senior secured notes)\b"),
                (3, r"\b(?:convertible notes|private placement|pipe financing|registered direct offering)\b"),
                (2, r"\b(?:liquidity facility|amendment and restatement)\b"),
            ],
        }

        event_score = 0
        matched_signals = {}
        for bucket, weighted in weighted_patterns.items():
            bucket_hits = []
            for weight, pattern in weighted:
                m = re.search(pattern, text, re.IGNORECASE)
                if not m:
                    continue
                snippet = m.group(0).strip()
                bucket_hits.append({"weight": weight, "match": snippet})
                event_score += weight
            if bucket_hits:
                matched_signals[bucket] = bucket_hits

        if event_score <= EVENT_SCORE_REJECT_MAX:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s event score=%d <= %d -> reject",
                ticker,
                cik,
                accession,
                event_score,
                EVENT_SCORE_REJECT_MAX,
            )
            return False

        # Additional anti-noise constraints.
        contracts_ok = True
        if "contracts" in matched_signals:
            contracts_ok = bool(
                re.search(
                    r"\b(?:multi-year|exclusive|government|enterprise)\b",
                    text,
                    re.IGNORECASE,
                )
            )
        financing_ok = True
        if "financing" in matched_signals:
            financing_ok = bool(
                re.search(
                    r"\b(?:credit facility|covenant|refinancing|pipe financing)\b",
                    text,
                    re.IGNORECASE,
                )
            )
        mna_ok = True
        if "mna" in matched_signals:
            mna_ok = bool(
                re.search(
                    r"\b(?:strategic alternatives|merger|acquisition)\b",
                    text,
                    re.IGNORECASE,
                )
            )

        if not (contracts_ok and financing_ok and mna_ok):
            logger.info(
                "ticker=%s, CIK=%s, accession=%s event score=%d anti-noise gate -> reject",
                ticker,
                cik,
                accession,
                event_score,
            )
            return False

        if event_score < EVENT_SCORE_PASS_MIN:
            logger.info(
                "ticker=%s, CIK=%s, accession=%s event score=%d below pass threshold=%d -> no-op",
                ticker,
                cik,
                accession,
                event_score,
                EVENT_SCORE_PASS_MIN,
            )
            return True

        logger.info(
            "*8K_EVENT_PASS ticker=%s CIK=%s accession=%s score=%d items=%s buckets=%s would_discover=true",
            ticker,
            cik,
            accession,
            event_score,
            {k: v for k, v in item_flags.items() if v},
            {k: len(v) for k, v in matched_signals.items()},
        )
        return True

    def analyze_8k(self, filing, sa, *, force: bool = False):

        cik = str(getattr(filing, "cik", "") or "")
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)
        accession = (getattr(filing, "accession_no", None) or getattr(filing, "accession_number", None) or "")

        if not ticker:
            logger.info(f"{ticker} - no tradable ticker")
            return

        # Check if already discovered - rediscover if >1 days ago
        if not force and not self.allow_discovery(ticker, period=24):
            return

        # Check the basics
        if not self.filter_filing(filing):
            return

        # Check financials
        if not self.filter_financials(filing):
            return

        # Earnings 8-k
        if self.analyze_8k_earnings(filing, cik, ticker, accession, sa):
            return

        # Pharma 8-k
        if self.analyze_8k_pharma(filing, cik, ticker, accession, sa):
            return

        # Event 8-K
        if self.analyze_8k_event(filing, cik, ticker, accession, sa):
            return

        # Unknown category (no earnings / pharma / event ownership)
        logger.info(
            "ticker=%s, CIK=%s, accession=%s no earnings/pharma/event match -> fail",
            ticker,
            cik,
            accession,
        )

    def analyze_tickers(self, sa, symbols: List[str]) -> Dict[str, int]:
        """
        Stage 1 pass/fail on latest 8-Ks for a manual ticker list.

        Resolves each ticker's newest earnings-ish 8-K (prefers Item 2.02),
        then runs full analyze_8k (filters + EX-99 LLM → watch on pass).
        Bypasses 24h discovery cooldown so Bizfeed-discovered names still get scored.
        """
        counts = {"analyzed": 0, "no_filing": 0, "errors": 0}
        # Drop prior MANUAL seed watches for these symbols so pass/fail isn't muddied.
        self._exclude_manual_watches(symbols)

        for raw in symbols:
            symbol = (raw or "").strip().upper()
            if not symbol:
                continue
            try:
                filing = resolve_latest_8k(symbol)
                if filing is None:
                    logger.info("Analyze ticker %s: no 8-K found", symbol)
                    counts["no_filing"] += 1
                    continue
                accession = (
                    getattr(filing, "accession_no", None)
                    or getattr(filing, "accession_number", None)
                    or ""
                )
                items = getattr(filing, "items", None)
                logger.info(
                    "Analyze ticker %s: accession=%s items=%s filing_date=%s",
                    symbol,
                    accession,
                    items,
                    getattr(filing, "filing_date", None),
                )
                self.analyze_8k(filing, sa, force=True)
                counts["analyzed"] += 1
            except Exception as e:
                counts["errors"] += 1
                logger.error("Analyze ticker %s failed: %s", symbol, e)

        logger.info("Analyze tickers done: %s", counts)
        return counts

    def _exclude_manual_watches(self, symbols: List[str]) -> int:
        """Mark meta.manual Pending watches Excluded for the given symbols."""
        from core.models import Watchlist

        want = {(s or "").strip().upper() for s in symbols if (s or "").strip()}
        if not want:
            return 0
        n = 0
        for w in self.watchlist().select_related("stock"):
            sym = (getattr(getattr(w, "stock", None), "symbol", None) or "").strip().upper()
            meta = w.meta if isinstance(w.meta, dict) else {}
            if sym in want and meta.get("manual"):
                w.status = "Excluded"
                w.explanation = f"Replaced by Stage-1 analyze | {w.explanation}"[:500]
                w.save(update_fields=["status", "explanation"])
                n += 1
                logger.info("Excluded manual watch %s (id=%s)", sym, w.id)
        return n

    def seed_earnings_watches(self, symbols: List[str]) -> Dict[str, int]:
        """
        Manually seed Pending edgar_earnings watches (open-check tape only).

        Skips symbols that already have a non-expired Pending edgar_earnings watch.
        Prefer analyze_tickers() when you need Stage-1 pass/fail.
        """
        counts = {"added": 0, "skipped": 0, "errors": 0}
        pending = {
            (getattr(getattr(w, "stock", None), "symbol", None) or "").strip().upper()
            for w in self.watchlist()
            if isinstance(getattr(w, "meta", None), dict)
            and w.meta.get("source") == "edgar_earnings"
        }

        for raw in symbols:
            symbol = (raw or "").strip().upper()
            if not symbol:
                continue
            if symbol in pending:
                logger.info("Seed watch %s: already Pending — skip", symbol)
                counts["skipped"] += 1
                continue
            try:
                entry = self.watch(
                    symbol,
                    explanation=f"MANUAL earnings watch | {symbol}",
                    days=2,
                    meta={
                        "source": "edgar_earnings",
                        "manual": True,
                        "weight": 1.0,
                    },
                    status="Pending",
                )
                if entry is None:
                    logger.warning("Seed watch %s: watch() returned None", symbol)
                    counts["errors"] += 1
                    continue
                pending.add(symbol)
                counts["added"] += 1
                logger.info("Seed watch %s: Pending edgar_earnings", symbol)
            except Exception as e:
                counts["errors"] += 1
                logger.error("Seed watch %s failed: %s", symbol, e)

        logger.info("Seed watches done: %s", counts)
        return counts

    def process_earnings_media(
        self, sa, *, dry_run: bool = False, force: bool = False
    ) -> Dict[str, int]:
        """
        Stage 1b: media LLM on Pending edgar_earnings watches that are due.

        Due when now >= filing_dt + delay (AH ≥16:00 ET → +2h; else +1h).
        Caps at MEDIA_BATCH_MAX calls per pass. Pass → media_gate=pass;
        fail → media_gate=fail + Excluded; LLM/parse miss → retry up to
        MEDIA_MAX_ATTEMPTS then exclude.
        """
        counts = {
            "pending": 0,
            "due": 0,
            "pass": 0,
            "fail": 0,
            "retry": 0,
            "skipped": 0,
            "errors": 0,
        }
        now = datetime.now(dt_timezone.utc)
        watches = [
            w
            for w in self.watchlist()
            if isinstance(getattr(w, "meta", None), dict)
            and w.meta.get("source") == "edgar_earnings"
        ]
        counts["pending"] = len(watches)

        due = []
        for w in watches:
            meta = w.meta or {}
            gate = str(meta.get("media_gate") or "").strip().lower()
            if gate in ("pass", "fail"):
                counts["skipped"] += 1
                continue
            accession = str(meta.get("accession") or "").strip()
            if not accession:
                # Tape-only / manual seeds have no filing — cannot run media.
                counts["skipped"] += 1
                continue
            filing_dt = _watch_filing_dt(w)
            if filing_dt is None:
                counts["skipped"] += 1
                continue
            due_at = media_due_at(filing_dt)
            if not force and now < due_at:
                counts["skipped"] += 1
                continue
            attempts = int(meta.get("media_attempts") or 0)
            if attempts >= MEDIA_MAX_ATTEMPTS:
                counts["skipped"] += 1
                continue
            due.append((w, accession, filing_dt, due_at, attempts))

        due.sort(key=lambda row: row[2])  # oldest filing first
        batch = due[:MEDIA_BATCH_MAX]
        counts["due"] = len(batch)
        logger.info(
            "Media gate: %d Pending earnings, %d due (batch=%d%s)%s",
            counts["pending"],
            len(due),
            len(batch),
            f", force" if force else "",
            " DRY-RUN" if dry_run else "",
        )
        if not batch:
            return counts

        for w, accession, filing_dt, due_at, attempts in batch:
            symbol = (
                getattr(getattr(w, "stock", None), "symbol", None) or ""
            ).strip().upper()
            try:
                filing = find(accession)
            except Exception as e:
                counts["errors"] += 1
                logger.warning(
                    "Media %s: find(%s) failed: %s", symbol or "?", accession, e
                )
                continue
            if filing is None:
                counts["errors"] += 1
                logger.warning(
                    "Media %s: find(%s) returned None", symbol or "?", accession
                )
                continue

            delay_h = media_delay_hours(filing_dt)
            logger.info(
                "Media %s: accession=%s filing_dt=%s delay=%sh due_at=%s",
                symbol or "?",
                accession,
                filing_dt.isoformat(),
                delay_h,
                due_at.isoformat(),
            )

            try:
                media = self.media_reaction_llm(filing)
            except Exception as e:
                counts["errors"] += 1
                logger.error("Media %s LLM failed: %s", symbol or "?", e)
                media = None

            meta = dict(w.meta or {})
            meta["media_attempts"] = attempts + 1
            meta["media_attempted_at"] = now.isoformat()

            if media is None:
                counts["retry"] += 1
                if meta["media_attempts"] >= MEDIA_MAX_ATTEMPTS:
                    meta["media_gate"] = "fail"
                    expl = (
                        f"MEDIA fail (no parse after {MEDIA_MAX_ATTEMPTS} tries) | "
                        f"{w.explanation or ''}"
                    )[:500]
                    if dry_run:
                        logger.info(
                            "Media %s: DRY-RUN would exclude (no parse)", symbol
                        )
                    else:
                        w.meta = meta
                        w.status = "Excluded"
                        w.explanation = expl
                        w.save(update_fields=["meta", "status", "explanation"])
                        counts["fail"] += 1
                        logger.info(
                            "Media %s: no parse after %d tries → Excluded",
                            symbol,
                            MEDIA_MAX_ATTEMPTS,
                        )
                elif dry_run:
                    logger.info("Media %s: DRY-RUN would retry (no parse)", symbol)
                else:
                    w.meta = meta
                    w.save(update_fields=["meta"])
                    logger.info(
                        "Media %s: no parse — retry later (attempt %d/%d)",
                        symbol,
                        meta["media_attempts"],
                        MEDIA_MAX_ATTEMPTS,
                    )
                continue

            meta["media"] = media
            if media_passes_gate(media):
                meta["media_gate"] = "pass"
                counts["pass"] += 1
                if dry_run:
                    logger.info(
                        "Media %s: DRY-RUN would PASS (%s)",
                        symbol,
                        _media_gate_explanation_segment(media),
                    )
                else:
                    w.meta = meta
                    w.save(update_fields=["meta"])
                    logger.info(
                        "Media %s: PASS (%s)",
                        symbol,
                        _media_gate_explanation_segment(media),
                    )
            else:
                meta["media_gate"] = "fail"
                counts["fail"] += 1
                expl = (
                    f"MEDIA fail ({_media_gate_explanation_segment(media) or 'gate'}) | "
                    f"{w.explanation or ''}"
                )[:500]
                if dry_run:
                    logger.info(
                        "Media %s: DRY-RUN would FAIL/exclude (%s)",
                        symbol,
                        _media_gate_explanation_segment(media),
                    )
                else:
                    w.meta = meta
                    w.status = "Excluded"
                    w.explanation = expl
                    w.save(update_fields=["meta", "status", "explanation"])
                    logger.info(
                        "Media %s: FAIL → Excluded (%s)",
                        symbol,
                        _media_gate_explanation_segment(media),
                    )

        logger.info("Media gate done: %s", counts)
        return counts

    def process_earnings_watches(
        self, sa, *, force: bool = False, dry_run: bool = False
    ) -> Dict[str, int]:
        """
        Goldilocks open-check on Pending edgar_earnings watches with media_gate=pass
        and a known EPS beat (meta.media eps in beat/strong_beat).

        Buckets (vs prior close):
          cliff      (≤ −3%) → Excluded
          goldilocks (−3%…+8%) → discovered() at live quote, watch → Executed
          rocket     (≥ +8%) → leave Pending (no chase)
          unclear    → leave Pending

        Requires market open ≥ OPEN_CHECK_MIN_MINUTES (15) unless force=True.
        dry_run: classify + log only.
        """
        from core.services.market.session import market_open

        counts = {
            "checked": 0,
            "cliff": 0,
            "goldilocks": 0,
            "rocket": 0,
            "unclear": 0,
            "skipped": 0,
            "no_media_pass": 0,
            "errors": 0,
        }

        mins = market_open()
        # market_open(): None = closed/AH/weekend; negative = premarket (mins until open);
        # positive = minutes since 9:30. Need mins >= OPEN_CHECK_MIN_MINUTES.
        if not force:
            if mins is None:
                logger.info("Open-check: market closed — skip")
                return counts
            if mins < OPEN_CHECK_MIN_MINUTES:
                logger.info(
                    "Open-check: market_open=%sm (need >= %sm) — skip",
                    mins,
                    OPEN_CHECK_MIN_MINUTES,
                )
                return counts
        else:
            logger.info(
                "Open-check: FORCE (market_open=%s)%s",
                mins,
                " DRY-RUN" if dry_run else "",
            )

        watches = [
            w
            for w in self.watchlist()
            if isinstance(getattr(w, "meta", None), dict)
            and w.meta.get("source") == "edgar_earnings"
        ]
        if not watches:
            logger.info("Open-check: no Pending edgar_earnings watches")
            return counts

        eligible = []
        for w in watches:
            meta = w.meta or {}
            if str(meta.get("media_gate") or "").strip().lower() != "pass":
                counts["no_media_pass"] += 1
                continue
            # Defense in depth: old tags may say pass with eps=unknown.
            if not media_passes_gate(meta.get("media")):
                counts["no_media_pass"] += 1
                logger.info(
                    "Open-check %s: media_gate=pass but eps beat missing — skip",
                    getattr(getattr(w, "stock", None), "symbol", "?"),
                )
                continue
            eligible.append(w)

        logger.info(
            "Open-check: %d Pending earnings (%d media-eligible, %d skipped no-pass)",
            len(watches),
            len(eligible),
            counts["no_media_pass"],
        )
        if not eligible:
            return counts

        for w in eligible:
            symbol = getattr(getattr(w, "stock", None), "symbol", None) or ""
            symbol = symbol.strip().upper()
            if not symbol:
                counts["skipped"] += 1
                continue

            meta = dict(w.meta or {})
            # Already promoted this session — don't double-discover.
            if meta.get("open_bucket") == "goldilocks" and w.status == "Executed":
                counts["skipped"] += 1
                continue

            try:
                tape = earnings_open_tape(symbol)
                if not tape or tape.get("vs_close_pct") is None:
                    logger.info("Open-check %s: no tape — unclear", symbol)
                    bucket = "unclear"
                    tape = tape or {}
                else:
                    bucket = classify_goldilocks_bucket(tape)

                counts["checked"] += 1
                counts[bucket] = counts.get(bucket, 0) + 1

                vs = tape.get("vs_close_pct")
                gap = tape.get("gap_pct")
                pb = tape.get("pullback_pct")
                vs_vwap = tape.get("vs_vwap_pct")
                logger.info(
                    "Open-check %s → %s (vs_close=%.2f%% gap=%s pb=%s vs_vwap=%s last=%s)",
                    symbol,
                    bucket,
                    vs if vs is not None else float("nan"),
                    f"{gap:.2f}%" if gap is not None else "n/a",
                    f"{pb:.2f}%" if pb is not None else "n/a",
                    f"{vs_vwap:.2f}%" if vs_vwap is not None else "n/a",
                    tape.get("last"),
                )

                meta["open_bucket"] = bucket
                meta["open_check"] = {
                    "mode": "goldilocks",
                    "vs_close_pct": vs,
                    "gap_pct": gap,
                    "vs_open_pct": tape.get("vs_open_pct"),
                    "pullback_pct": pb,
                    "vs_vwap_pct": vs_vwap,
                    "last": tape.get("last"),
                    "prior_close": tape.get("prior_close"),
                    "forced": bool(force),
                    "dry_run": bool(dry_run),
                }

                if dry_run:
                    logger.info("Open-check %s DRY-RUN — no write/discover", symbol)
                    continue

                w.meta = meta

                if bucket == "cliff":
                    w.status = "Excluded"
                    note = (
                        f"OPEN cliff ({vs:+.1f}% vs close)"
                        if vs is not None
                        else "OPEN cliff"
                    )
                    w.explanation = f"{note} | {w.explanation}"[:500]
                    w.save(update_fields=["status", "meta", "explanation"])
                    continue

                if bucket == "rocket":
                    note = (
                        f"OPEN rocket ({vs:+.1f}% vs close) — no chase"
                        if vs is not None
                        else "OPEN rocket — no chase"
                    )
                    w.explanation = f"{note} | {w.explanation}"[:500]
                    w.save(update_fields=["meta", "explanation"])
                    continue

                if bucket == "unclear":
                    w.save(update_fields=["meta"])
                    continue

                # goldilocks → discover at live quote
                weight = float(meta.get("weight") or 1.0)
                stock = w.stock
                try:
                    stock.refresh()
                except Exception as e:
                    logger.warning("Open-check %s refresh failed: %s", symbol, e)

                live = _safe_float(stock.price) or tape.get("last")
                media_seg = _media_gate_explanation_segment(meta.get("media"))
                lead = (
                    f"OPEN goldilocks @ {live:.2f} ({vs:+.1f}% vs close)"
                    if live is not None and vs is not None
                    else "OPEN goldilocks"
                )
                parts = [lead]
                if media_seg:
                    parts.append(media_seg)
                if w.explanation:
                    parts.append(w.explanation)
                expl = " | ".join(parts)
                self.discovered(sa, symbol, expl[:500], weight=weight)
                w.status = "Executed"
                w.explanation = expl[:500]
                w.save(update_fields=["status", "meta", "explanation"])

            except Exception as e:
                counts["errors"] += 1
                logger.error("Open-check %s failed: %s", symbol, e)

        logger.info("Open-check done: %s", counts)
        return counts

    def _accession_already_handled(self, accession: str) -> bool:
        """True if this advisor already has a watch tied to this 8-K accession."""
        acc = (accession or "").strip()
        if not acc:
            return False
        from core.models import Watchlist

        return Watchlist.objects.filter(
            advisor=self.advisor,
            meta__accession=acc,
        ).exists()

    def discover(
        self,
        sa,
        accessions: Optional[List[str]] = None,
        *,
        open_check_only: bool = False,
        force_open_check: bool = False,
        dry_run_open_check: bool = False,
        seed_watches: Optional[List[str]] = None,
        analyze_symbols: Optional[List[str]] = None,
    ):
        """
        Process latest 8-Ks (or accessions), then media gate, then open-check.

        When accessions is set, prev-SA time filter is skipped.
        Live feed uses a FILING_FEED_LAG_MINUTES lookback vs prev SA so
        acceptance-time vs feed-lag does not permanently drop filings.
        open_check_only: skip filing fetch; still runs media + open-check.
        force_open_check: bypass RTH/+15m gate (local lab only).
        dry_run_open_check: classify only; no watch writes / discover.
        seed_watches: optional tickers → Pending watches (tape-only, no Stage 1).
        analyze_symbols: optional tickers → full Stage 1 on latest 8-Ks.
        """
        if seed_watches:
            self.seed_earnings_watches(seed_watches)

        if analyze_symbols:
            self.analyze_tickers(sa, analyze_symbols)

        if not open_check_only and (accessions or not analyze_symbols):
            prev_ts = None
            if accessions:
                logger.info("Fetching %d accession(s) via find()...", len(accessions))
                filings = []
                for raw in accessions:
                    acc = (raw or "").strip()
                    if not acc:
                        continue
                    try:
                        filing = find(acc)
                    except Exception as e:
                        logger.warning("find(%s) failed: %s", acc, e)
                        continue
                    if filing is None:
                        logger.warning("find(%s) returned None", acc)
                        continue
                    filings.append(filing)
            else:
                logger.info(
                    "Fetching latest 8-K filings (page_size=100, lag=%sm)...",
                    FILING_FEED_LAG_MINUTES,
                )
                prev_ts = self.get_previous_sa_timestamp(sa)
                try:
                    latest = get_latest_filings(form="8-K", page_size=100)
                    filings = list(latest)
                except Exception as e:
                    logger.warning("❌ Error converting latest filings to list: %s", e)
                    filings = []

            filings_8k = [f for f in filings if getattr(f, "form", None) == "8-K"]
            if not filings_8k:
                logger.info("No 8-K filings to process.")
            else:
                logger.info(
                    "Found %d 8-K filings. Running basic inspection "
                    "(filing + financial filters)...",
                    len(filings_8k),
                )
                cutoff = None
                if prev_ts is not None:
                    cutoff = prev_ts - timedelta(minutes=FILING_FEED_LAG_MINUTES)

                for filing in filings_8k:
                    try:
                        accession = (
                            getattr(filing, "accession_no", None)
                            or getattr(filing, "accession_number", None)
                            or ""
                        )
                        if accession and self._accession_already_handled(accession):
                            logger.info(
                                "Filing 8-K %s already handled (watch exists) — skipping",
                                accession,
                            )
                            continue

                        filing_dt = _filing_datetime(filing)
                        if (
                            cutoff is not None
                            and filing_dt is not None
                            and filing_dt < cutoff
                        ):
                            logger.warning(
                                "Filing 8-K %s (filing_time=%s) older than "
                                "prev SA lookback %s (prev_sa=%s lag=%sm) — skipping",
                                accession,
                                filing_dt,
                                cutoff,
                                prev_ts,
                                FILING_FEED_LAG_MINUTES,
                            )
                            continue

                        self.analyze_8k(filing, sa)

                    except Exception as e:
                        logger.error("⚠️ Error inspecting filing: %s", e)

        self.process_earnings_media(sa, dry_run=dry_run_open_check)
        self.process_earnings_watches(
            sa, force=force_open_check, dry_run=dry_run_open_check
        )
        return


def run_edgar_standalone(
    accessions: Optional[List[str]] = None,
    *,
    open_check_only: bool = False,
    force_open_check: bool = False,
    dry_run_open_check: bool = False,
    seed_watches: Optional[List[str]] = None,
    analyze_symbols: Optional[List[str]] = None,
):
    """
    Entry point for the `run_edgar` management command.

    accessions: optional SEC accession numbers instead of get_latest_filings().
    analyze_symbols: optional tickers → full Stage 1 on latest 8-Ks.
    seed_watches: optional tickers to seed as Pending watches (tape-only).
    """
    from core.services import advisors as advisor_modules
    from core.models import Advisor, SmartAnalysis

    try:
        advisor_row = Advisor.objects.get(name="EDDIE-8")
    except Advisor.DoesNotExist:
        return None, "ED-8 advisor not found in Advisor table"

    module_name = advisor_row.python_class.lower()
    module = getattr(advisor_modules, module_name)
    PythonClass = getattr(module, advisor_row.python_class)

    sa = SmartAnalysis()
    impl = PythonClass(advisor_row)
    impl.discover(
        sa,
        accessions=accessions,
        open_check_only=open_check_only,
        force_open_check=force_open_check,
        dry_run_open_check=dry_run_open_check,
        seed_watches=seed_watches,
        analyze_symbols=analyze_symbols,
    )

    if analyze_symbols:
        return (
            f"EDDIE-8 Stage-1 analyze completed for {len(analyze_symbols)} ticker(s)",
            None,
        )
    if seed_watches and open_check_only:
        return (
            f"EDDIE-8 seeded {len(seed_watches)} watch(es) + open-check completed",
            None,
        )
    if open_check_only:
        return "EDDIE-8 open-check completed", None
    if accessions:
        return f"EDDIE-8 discover() completed for {len(accessions)} accession(s)", None
    if seed_watches:
        return f"EDDIE-8 discover() completed (seeded {len(seed_watches)} watch(es))", None
    return "EDDIE-8 discover() completed", None


register(name="EDDIE-8", python_class="Edgar")

