import os
import re
import time
from collections import Counter
from pathlib import Path

import requests
import yfinance as yf
from dotenv import load_dotenv
from edgar import set_identity, get_filings, get_latest_filings, find, Company
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Tuple
import html
import argparse

try:
    import pysentiment2 as ps
except ImportError:
    ps = None


# Load .env file if it exists (same as Django settings)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# -----------------------------
# 1️⃣ Set SEC identity (mandatory!)
# -----------------------------
# Replace with your real info for SEC compliance
set_identity("David McCarthy david@example.com")


# -----------------------------
# 2️⃣ CIK to ticker conversion (lightweight copy)
# -----------------------------
_CIK_TO_TICKER_CACHE: Dict[str, Optional[str]] = {}

def vprint(verbose, text):
    if verbose:
        print(text)


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


# -----------------------------
# 3️⃣ FILTER1: basic earnings-ness / tradability check
# -----------------------------
EARNINGS_KEYWORDS = [
    # 1. Core SEC-canonical phrases (highest recall, lowest risk)
    "results of operations",
    "financial condition",
    "period ended",
    "three months ended",
    "six months ended",
    "nine months ended",
    "fiscal quarter",
    "fiscal year",
    # 2. Filing-intent language (very important, very boring)
    "furnished pursuant to",
    "press release announcing",
    "press release regarding",
    "attached as exhibit 99",
    "included as exhibit",
    # 3. Earnings-context verbs (light semantic lift)
    "announced its results",
    "reported results",
    "released results",
    "announced financial results",
    "reported financial results",
    # 4. Guidance / outlook indicators (main-text safe)
    "provided guidance",
    "updated guidance",
    "outlook for",
    "expects for",
    "anticipated results",
    # 5. Low-precision but high-coverage phrases (use as OR-only)
    "quarter ended",
    "year ended",
    "period results",
    "financial results",
    # Original keywords (kept for backward compatibility)
    "earnings release",
    "quarterly results",
    "full year results",
    "earnings per share",
    "eps",
    "net income per share",
    "guidance",
    "outlook",
    "expects",
    "forecast",
]

REG_FD_KEYWORDS = [
    "furnished pursuant to regulation fd",
    "Item 3.01", # Delisting
    "Item 1.03", # Bankruptcy
]

# -----------------------------
# FILTER2: red/green flags for risk and quality scoring
# -----------------------------
# Red flags: severe penalties (bankruptcy, going concern, restatement, Item 4.02)
RED_FLAGS_SEVERE = {
    "chapter 11": -10,
    "bankruptcy filing": -10,
    "receivership": -10,
    "substantial doubt": -8,
    "ability to continue as a going concern": -8,
    "going concern": -8,
    "item 4.02": -8,
    "non-reliance on financial statements": -8,
    "restatement of previously issued financial statements": -7,
    "should no longer be relied upon": -7,
}

# Red flags: moderate penalties (management changes, material weakness, etc.)
RED_FLAGS_MODERATE = {
    # Management changes
    "resignation": -2,
    "ceo termination": -2,
    "cfo termination": -2,
    "termination of employment": -2,
    "departure": -2,
    "material weakness": -2,
    "internal control deficiency": -2,
    "sec investigation": -2,
    "sec inquiry": -2,
    "class action": -1,
    "material litigation": -2,
    "significant litigation": -2,
    # Weak / cautious guidance (JILL-type)
    "weak guidance": -2,
    "cautious outlook": -2,
    "cautious guidance": -2,
    "cautious stance": -2,
    "projected decline": -2,
    "expects decline": -2,
    "lower outlook": -2,
    "lowered expectations": -2,
    "decline in sales": -1,
    "anticipated softness": -2,
    "tempered outlook": -2,
    "mid-single-digit decline": -2,
    "single-digit decline": -1,
    "macroeconomic headwinds": -1,
    "promotional environment": -1,
    # Retail / outlook: in 8-K Outlook section often signals slower sales expectation
    "disciplined inventory management": -1,
    # Broader risk / defensive language
    "challenging macroeconomic environment": -2,
    "strategic alternatives": -2,
    "restructuring charges": -2,
    "liquidity constraints": -2,
    "breach of loan covenants": -2,
    "softer start": -2,
    "down approximately": -1,
    "net sales to be down": -1
}

# Green flags: high-quality signals (specific, hard to fake)
GREEN_FLAGS = {
    "raised guidance": +3,
    "increased outlook": +3,
    "increased guidance": +3,
    "initiated guidance": +1,
    "provided guidance for": +1,
    "dividend increase": +2,
    "increased dividend": +2,
    "share repurchase authorization": +2,
    "buyback program": +2,
    "above expectations": +2,
    "exceeded estimates": +2,
    "record revenue": +2,
    "record ebitda": +1,
    "record earnings": +1,
    # ASYS-type: beat + return to profit + AI/cash narrative
    "stronger than expected results": +2,
    "higher than expected revenue": +2,
    "return to profitability": +2,
    "driven by ai": +1,
    "eliminated debt": +1,
    "positive operating cash flow": +1,
    "strong operating leverage": +1,
    # XZO-type: growth / margin / cash (balanced, not ASYS-specific)
    "meaningful margin expansion": +2,
    "revenue growth": +1,
    "revenue increase": +1,
    "exceptional revenue growth": +3,
    "robust cash generation": +1,
    "improved operating leverage": +1,
    "scalability and cost efficiency": +1,
    "successfully launched": +1,
    "growing demand ": +1,
    "adjusted ebitda profitability": +1,
    "backlog grew": +1,
    # RILY-type: in-line results / compliance / debt improvement
    "in line with previous estimate": +1,
    "in line with filed estimates": +1,
    "listing compliance": +1,
    "compliance deadline": +1,
    "debt reduction": +1,
    "reduction of total debt": +1,
    "filed prior to": +1,
    "strongest year": +2,
    "strongest quater": +2
}

def compute_filter1_pass(filing, verbose: bool = False) -> bool:
    """
    FILTER1: very early, very cheap triage.

    v2 behaviour:
    - Require form == 8-K.
    - Require tradable ticker (filing.ticker or CIK->ticker).
    - Compute an 'earnings-ness' score from:
        * Item 2.02
        * 99.x exhibit
        * earnings/guidance keywords in text
        * Reg FD boilerplate with no numbers (negative)
    - Return:
        0  = failed FILTER1
        10 = passed FILTER1 (for now we just use 10 as 'level 1' score)
    """
    form = getattr(filing, "form", None)
    if form != "8-K":
        vprint(verbose, "FILTER1: form not 8-K")
        return False

    # Resolve ticker
    ticker = getattr(filing, "ticker", None)
    if not ticker:
        ticker = cik_to_ticker(getattr(filing, "cik", ""))
    if not ticker:
        vprint(verbose, "FILTER1: no tradable ticker → fail")
        return False

    # No exhibits
    if not (hasattr(filing, "exhibits") and filing.exhibits):
        vprint(verbose, "FILTER1: no exhibits → fail")
        return False

    # Exhibit 99
    has_exhibit_99 = False

    for ex in filing.exhibits:
        ex_str = str(ex).lower()

        if "99." in ex_str:
            has_exhibit_99 = True
            break

    # No Exhibit 99.x
    if not has_exhibit_99:
        vprint(verbose, "FILTER1: no exhibit 99.x → fail")
        return False

    # Examine filing text
    filing_text = filing.text().lower()

    # Need item 9.0x
    if "item 9.0" not in filing_text:
        vprint(verbose, "FILTER1: item 9.0.x → fail")
        return False

    # Bad words
    for kw in REG_FD_KEYWORDS:
        if kw in filing_text:
            vprint(verbose, f"FILTER1: Reg keyword {kw} in main text → fail")
            return False

    # Good words
    has_earnings_keyword = False
    for kw in EARNINGS_KEYWORDS:
        if kw in filing_text:
            vprint(verbose, f"FILTER1: Found earnings key phrase: '{kw}'")
            has_earnings_keyword = True
            break

    if not has_earnings_keyword:
        vprint(verbose, "FILTER1: No earnings key phrase in main text")
        return 0

    # Return score
    vprint(verbose, "FILTER1: (pass)")
    return True


# -----------------------------
# FILTER2: red/green flags scoring
# -----------------------------
def compute_filter2_pass(filing, verbose: bool = False) -> bool:
    """
    FILTER2: red/green flags for risk and quality adjustment.
    
    Returns signed integer score:
    - Negative: penalties (red flags)
    - Positive: bonuses (green flags)
    - Zero: neutral
    
    Red flags searched in: filing text + exhibit 99 text (combined)
    Green flags searched in: exhibit 99 text only
    """
    score = 0
    
    # Retrieve ALL 99.x exhibit text (so we include 99.1 Outlook and 99.2 etc.)
    exhibit_99_parts = []
    if hasattr(filing, "exhibits") and filing.exhibits:
        for ex in filing.exhibits:
            ex_str = str(ex).lower()
            if "99." in ex_str:
                try:
                    exhibit_99_parts.append(ex.text())
                except Exception:
                    pass
    exhibit_99_text = " ".join(exhibit_99_parts) if exhibit_99_parts else None
    
    # Get filing text
    filing_text = ""
    try:
        filing_text = filing.text() or ""
    except Exception:
        filing_text = ""
    
    # Combine texts for red flag search (lowercase)
    combined_text = (filing_text + " " + (exhibit_99_text or "")).lower()
    exhibit_99_lower = (exhibit_99_text or "").lower()
    
    # Apply red flag penalties (severe)
    for keyword, penalty in RED_FLAGS_SEVERE.items():
        if keyword in combined_text:
            vprint(verbose, f"FILTER2: severe red flag '{keyword}' → {penalty}")
            score += penalty
    
    # Apply red flag penalties (moderate)
    for keyword, penalty in RED_FLAGS_MODERATE.items():
        if keyword in combined_text:
            vprint(verbose, f"FILTER2: moderate red flag '{keyword}' → {penalty}")
            score += penalty
    
    # Apply green flag bonuses (exhibit 99 only, no cap)
    for keyword, bonus in GREEN_FLAGS.items():
        if keyword in exhibit_99_lower:
            vprint(verbose, f"FILTER2: green flag '{keyword}' → +{bonus}")
            score += bonus

    vprint(verbose, f"FILTER2: {'(pass)' if score >= 0 else 'fail'} {score}")
    return score >= 0


# ----------------
# FILTER3: generic sanity checks (valuation / cap / price band)
# -----------------------------
FILTER3_PE_MAX = 100  # Overvalued definitive: fail if trailing P/E > this
FILTER3_MIN_CAP = 300e6  # Exclude nano/micro: fail if market_cap < 300M
FILTER3_MAX_CAP = 200e9  # Exclused mega cap
FILTER3_PRICE_MIN = 5.0
FILTER3_PRICE_MAX = 1000.0

# Sector beware (vprint WARNING only; matches yfinance sector/industry wording)
FILTER3_UNDERPERFORM_SECTORS = ("consumer cyclical", "real estate", "utilities")  # 🔴 Underperform
FILTER3_WATCH_SECTOR_INDUSTRY = (  # 🟡 Neutral/cautious or Watch
    ("technology", "software"),   # Tech (software): AI disruption, capex concerns
    ("financial services", "insurance"),
)


def compute_filter3_pass(filing, verbose: bool = False) -> bool:
    """
    FILTER3: valuation, cap, and price band.
    - P/E: fail if trailing P/E > FILTER3_PE_MAX (overvalued definitive).
    - Cap: fail if market_cap < FILTER3_MIN_CAP (nano/micro) or > FILTER3_MAX_CAP (mega).
    - Price: fail if price < FILTER3_PRICE_MIN or > FILTER3_PRICE_MAX.
    Returns True if pass, False if fail. If no ticker or yfinance data missing, passes (no block).
    """
    ticker = getattr(filing, "ticker", None)
    if not ticker:
        ticker = cik_to_ticker(getattr(filing, "cik", ""))

    try:
        time.sleep(0.05)  # Rate limit yfinance
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception as e:
        vprint(verbose, f"FILTER3: yfinance error → pass (skip checks): {e}")
        return True

    # P/E: overvalued definitive
    pe = info.get("trailingPE") or info.get("forwardPE")
    if pe is not None and isinstance(pe, (int, float)):
        if pe > FILTER3_PE_MAX:
            vprint(verbose, f"FILTER3: fail P/E {pe:.1f} > {FILTER3_PE_MAX} (overvalued)")
            return False

    # Cap: exclude nano/micro
    cap = info.get("marketCap")
    if cap is not None and isinstance(cap, (int, float)):
        if cap < FILTER3_MIN_CAP:
            vprint(verbose, f"FILTER3: fail market_cap {cap/1e6:.1f}M < {FILTER3_MIN_CAP/1e6:.0f}M (nano/micro)")
            return False

        if cap > FILTER3_MAX_CAP:
            vprint(verbose, f"FILTER3: fail market_cap {cap / 1e6:.1f}M > {FILTER3_MAX_CAP / 1e6:.0f}M (mega)")
            return False

    # Price band: $5–$500
    price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
    if price is not None and isinstance(price, (int, float)):
        if price < FILTER3_PRICE_MIN:
            vprint(verbose, f"FILTER3: fail price ${price:.2f} < ${FILTER3_PRICE_MIN} (below band)")
            return False
        if price > FILTER3_PRICE_MAX:
            vprint(verbose, f"FILTER3: fail price ${price:.2f} > ${FILTER3_PRICE_MAX} (above band)")
            return False

    # Warnings (vprint only): 90%+ of 52-week high, near 52-week low, 90%+ of 2-week lead-up high
    fifty_two_high = info.get("fiftyTwoWeekHigh")
    fifty_two_low = info.get("fiftyTwoWeekLow")
    if verbose and price is not None:
        if (
            fifty_two_high is not None
            and isinstance(fifty_two_high, (int, float))
            and fifty_two_high > 0
        ):
            pct_52_high = 100.0 * price / fifty_two_high
            if pct_52_high >= 90.0:
                vprint(
                    verbose,
                    f"********** WARNING: Price at {pct_52_high:.1f}% of 52-week high "
                    f"(price ${price:.2f}, 52w high ${fifty_two_high:.2f}) **********",
                )
        if (
            fifty_two_low is not None
            and isinstance(fifty_two_low, (int, float))
            and fifty_two_low > 0
        ):
            pct_52_low = 100.0 * price / fifty_two_low
            # Within ~10% of 52-week low → highlight as opportunity
            if pct_52_low <= 110.0:
                vprint(
                    verbose,
                    f"🟢 OPPORTUNITY: Price at {pct_52_low:.1f}% of 52-week low "
                    f"(price ${price:.2f}, 52w low ${fifty_two_low:.2f})",
                )

    fd = getattr(filing, "filing_date", None)
    if verbose and price is not None and fd is not None:
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
                two_week_high = float(hist["High"].max())
                last_close = float(hist["Close"].iloc[-1])
                if two_week_high > 0 and last_close >= 0.90 * two_week_high:
                    pct_2w = 100.0 * last_close / two_week_high
                    vprint(
                        verbose,
                        f"********** WARNING: Filing price at {pct_2w:.1f}% of 2-week high "
                        f"(filing close ${last_close:.2f}, 2w high ${two_week_high:.2f}) **********",
                    )

                # Price vs averages: relative to 2-week and 52-week averages (100% = at average)
                two_week_avg = float(hist["Close"].mean())
                rel_2w = 100.0 * last_close / two_week_avg if two_week_avg > 0 else None

                rel_52w = None
                avg_52w = None
                try:
                    hist_52w = stock.history(
                        start=filing_date - timedelta(days=365),
                        end=filing_date,
                        auto_adjust=True,
                    )
                    if hist_52w is not None and not hist_52w.empty:
                        avg_52w = float(hist_52w["Close"].mean())
                        if avg_52w > 0:
                            rel_52w = 100.0 * last_close / avg_52w
                except Exception:
                    hist_52w = None

                parts = []
                if rel_2w is not None:
                    parts.append(
                        f"2w={rel_2w:.1f}% (last_close ${last_close:.2f}, 2w avg ${two_week_avg:.2f})"
                    )
                if rel_52w is not None and avg_52w is not None:
                    parts.append(
                        f"52w={rel_52w:.1f}% (last_close ${last_close:.2f}, 52w avg ${avg_52w:.2f})"
                    )
                if parts:
                    vprint(verbose, "Price vs averages: " + "; ".join(parts))
        except Exception as e:
            vprint(verbose, f"FILTER3: (skip 2-week high check: {e})")

    # Warnings (vprint only): sector/industry in beware lists
    if verbose:
        sector = (info.get("sector") or "").strip().lower()
        industry = (info.get("industry") or "").strip().lower()
        if sector and sector in FILTER3_UNDERPERFORM_SECTORS:
            vprint(
                verbose,
                f"********** WARNING: Sector in underperform list "
                f"(sector={info.get('sector') or 'N/A'}, industry={info.get('industry') or 'N/A'}) **********",
            )
        for watch_sector, watch_industry in FILTER3_WATCH_SECTOR_INDUSTRY:
            if sector and industry and watch_sector in sector and watch_industry in industry:
                vprint(
                    verbose,
                    f"********** WARNING: Sector/industry in watch list "
                    f"(sector={info.get('sector') or 'N/A'}, industry={info.get('industry') or 'N/A'}) **********",
                )
                break

    vprint(verbose, "FILTER3: (pass) (P/E, cap, price band ok)")
    return True

# ----------------
# FILTER4: Earnings release filter (structure + evidence, no brittle parsing)
# -----------------------------
EARNINGS_CONTEXT = [
    "earnings",
    "financial results",
    "results of operations",
    "quarter ended",
    "fiscal quarter",
]

EPS_KEYWORDS = [
    "earnings per share",
    "eps",
    "diluted",
]

REVENUE_KEYWORDS = [
    "revenue",
    "revenues",
    "net sales",
    "total revenue",
]

TABLE_HINTS = [
    "gaap",
    "non-gaap",
    "q/q",
    "y/y",
    "%",
]

EPS_NUMBER_PATTERN = re.compile(
    r"(earnings per share|eps|diluted).{0,60}?(-?\d+\.\d+)",
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
    hits = sum(1 for k in TABLE_HINTS if k in text)
    return hits >= 2


def _filter4_has_comparables(text: str) -> bool:
    words = re.findall(r"(revenue|earnings per share|eps|net income)", text, re.IGNORECASE)
    counts = Counter(w.lower() for w in words)
    return any(v >= 2 for v in counts.values())


def earnings_release_filter(raw_text: str) -> dict:
    """
    Given raw EX-99.1 text, decide: is this very likely a real earnings release with numeric comparables?
    Returns dict with pass, reason, checks (and score when pass).
    """
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

    score = sum([
        checks["table_structure"],
        checks["comparables"],
    ])
    return {
        "pass": score >= 1,
        "score": score,
        "reason": "ok" if score >= 1 else "soft fail",
        "checks": checks,
    }


def compute_filter4_pass(filing, verbose: bool = False) -> bool:
    """
    FILTER4: earnings release filter (structure + evidence).
    Gets 99.x exhibit text, runs earnings_release_filter; returns 1 if pass, 0 otherwise.
    """
    exhibit_99_parts = []
    if hasattr(filing, "exhibits") and filing.exhibits:
        for ex in filing.exhibits:
            ex_str = str(ex).lower()
            if "99." in ex_str:
                try:
                    exhibit_99_parts.append(ex.text())
                except Exception:
                    pass
    text = " ".join(exhibit_99_parts) if exhibit_99_parts else ""
    if not text:
        vprint(verbose, "FILTER4: no 99.x exhibit text → fail")
        return False

    result = earnings_release_filter(text)
    if not result["pass"]:
        vprint(verbose, f"FILTER4: {result['reason']} → fail")
        if verbose and result.get("checks"):
            vprint(verbose, f"FILTER4: checks={result['checks']}")
        return False

    if verbose and result.get("checks"):
        vprint(verbose, f"FILTER4: (pass) checks={result['checks']}")

    return True


# ----------------
# FILTER5: LM guidance sentiment (ported from edgar.py for backtest)
# -----------------------------

# Section headings / openers that start the disclaimer block (remove before parsing).
# Regex catches "Forward-Looking Statements", "Forward Looking Statements", "Forward - Looking Statements", etc.
_FORWARD_LOOKING_PATTERN = re.compile(
    r"forward\s*[-]?\s*looking\s+statements",
    re.IGNORECASE,
)
_FORWARD_LOOKING_STARTS = [
    "Cautionary Statement",
    "Safe Harbor",
    "This press release contains forward-looking statements",
    "This release contains forward-looking statements",
]


def _filter5_strip_forward_looking_section(text: str) -> str:
    """
    Remove the Forward-Looking Statements (disclaimer) section from EX-99.1 text
    before parsing, so its negative wording does not affect FILTER5 LM.
    Uses earliest match of regex + phrase list; removes from there to end of text.
    Note: ex99_1_dump.txt is written from raw exhibit text, so it will still contain
    this section; FILTER5 uses the stripped text in memory only.
    """
    if not text or not text.strip():
        return text
    text_lower = text.lower()
    earliest = len(text)
    m = _FORWARD_LOOKING_PATTERN.search(text)
    if m:
        earliest = min(earliest, m.start())
    for phrase in _FORWARD_LOOKING_STARTS:
        idx = text_lower.find(phrase.lower())
        if idx != -1 and idx < earliest:
            earliest = idx
    if earliest == len(text):
        return text
    return text[:earliest].strip()


GUIDANCE_KEYWORDS = [
    "guidance", "outlook", "financial outlook", "forecast", "projected", "projection",
    "expects", "expect", "we expect", "anticipate", "target", "targets", "goal", "goals",
    "estimate", "estimates", "future", "next quarter", "next year",
]

BOILERPLATE_PHRASES = [
    "forward-looking statements",
    "management believes that",
    "we do not provide a forward-looking reconciliation",
    "cannot be estimated at this time without unreasonable efforts",
    "cannot, without unreasonable effort",
    "not be meaningful to investors",
    "range of values so broad",
    "summarized in the table below",
]


def _filter5_split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text)
    # Split on sentence end (.?!) or closing quote followed by space (so quoted text isn't merged with table headers).
    parts = re.split(r"(?<=[\.\?\!])\s+|(?<=\")\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _filter5_looks_like_table_fragment(sentence: str) -> bool:
    """Exclude table rows / financial grids from LM scoring (e.g. 'Q1 2026 ... $168.0 - $174.0 ... Non-GAAP')."""
    s = sentence.strip()
    if not s:
        return True
    # Many digits + dollar amounts + financial labels → likely a table row
    digit_count = len(re.findall(r"\d", s))
    dollar_count = len(re.findall(r"\$\d", s))
    lower = s.lower()
    if digit_count >= 12 and ("$" in s or "million" in lower or "non-gaap" in lower):
        return True
    if dollar_count >= 2 and ("non-gaap" in lower or "revenue" in lower or "operating income" in lower):
        return True
    return False


def _filter5_extract_guidance_sentences(text: str) -> list:
    sentences = _filter5_split_sentences(text)
    lower_kw = [k.lower() for k in GUIDANCE_KEYWORDS]
    return [s for s in sentences if any(k in s.lower() for k in lower_kw)]


def _filter5_is_boilerplate(sentence: str) -> bool:
    lower = sentence.lower()
    return any(phrase in lower for phrase in BOILERPLATE_PHRASES)


def _filter5_compute_lm_guidance(text: str):
    """
    Run LM on guidance sentences. Returns None if ps missing / no guidance after boilerplate.
    Else dict with passed, n_sentences, total_pos, total_neg, avg_polarity, net_polarity.
    """
    if ps is None or not (text or "").strip():
        return None
    guidance = _filter5_extract_guidance_sentences(text)
    candidates = [
        s for s in guidance
        if not _filter5_is_boilerplate(s) and not _filter5_looks_like_table_fragment(s)
    ]
    if not candidates:
        return None
    try:
        lm = ps.LM()
        total_pos = total_neg = 0
        polarities = []
        for s in candidates:
            tokens = lm.tokenize(s)
            score = lm.get_score(tokens)
            total_pos += score.get("Positive", 0)
            total_neg += score.get("Negative", 0)
            polarities.append(score.get("Polarity", 0.0))
        avg_polarity = sum(polarities) / len(polarities) if polarities else 0.0
        denom = total_pos + total_neg
        net_polarity = (total_pos - total_neg) / denom if denom > 0 else 0.0
        # Pass when pos >= neg, or when LM signal is too weak to conclude negative (e.g. few tokens)
        passed = not (total_neg > total_pos) or denom < 4
        return {
            "passed": passed,
            "n_sentences": len(candidates),
            "total_pos": total_pos,
            "total_neg": total_neg,
            "avg_polarity": round(avg_polarity, 4),
            "net_polarity": round(net_polarity, 4),
        }
    except Exception:
        return None


def compute_filter5_pass(filing, verbose: bool = False) -> bool:
    """
    FILTER5: LM guidance sentiment (same logic as edgar.py).
    Pass when neg does not beat pos; skip (pass) when no text / no guidance / ps unavailable.
    """
    exhibit_99_parts = []
    if hasattr(filing, "exhibits") and filing.exhibits:
        for ex in filing.exhibits:
            if "99." in str(ex).lower():
                try:
                    exhibit_99_parts.append(ex.text())
                except Exception:
                    pass
    text = " ".join(exhibit_99_parts) if exhibit_99_parts else ""
    if not text:
        vprint(verbose, "FILTER5: (fail) no 99.x exhibit text")
        return False

    text = _filter5_strip_forward_looking_section(text)

    guidance_sents = _filter5_extract_guidance_sentences(text)
    if not guidance_sents:
        vprint(verbose, "FILTER5: (fail) no guidance sentences")
        return False

    non_boilerplate = [
        s for s in guidance_sents
        if not _filter5_is_boilerplate(s) and not _filter5_looks_like_table_fragment(s)
    ]
    vprint(verbose, f"FILTER5: guidance {len(guidance_sents)} total, {len(non_boilerplate)} after boilerplate + table filter")

    result = _filter5_compute_lm_guidance(text)
    if result is None:
        vprint(verbose, "FILTER5: (pass) no guidance after boilerplate or pysentiment2 unavailable")
        return False

    vprint(verbose, f"FILTER5: ({'pass' if result['passed'] else 'fail'}) n_sentences={result['n_sentences']} total_pos={result['total_pos']} total_neg={result['total_neg']} net_polarity={result['net_polarity']:+.3f}")
    if not result["passed"] and non_boilerplate:
        vprint(verbose, "FILTER5 failing sentences:")
        for i, s in enumerate(non_boilerplate, 1):
            vprint(verbose, f"  [{i}] {s[:200]}{'...' if len(s) > 200 else ''}")
    return result["passed"]


# ----------------
# FILTER6: retrieve EPS beat (Alpha Vantage, latest quarter for filing date)
# -----------------------------
def _latest_quarter_end_for_date(d: date) -> date:
    """Return quarter-end date for the quarter this filing date reports. Jan/Feb/Mar → prev Dec 31; Apr/Jun → Mar 31; etc."""
    y, m = d.year, d.month
    if m in (1, 2, 3):
        return date(y - 1, 12, 31)
    if m in (4, 5, 6):
        return date(y, 3, 31)
    if m in (7, 8, 9):
        return date(y, 6, 30)
    return date(y, 9, 30)


# -------- 8-K actual EPS (for Filter 6 fallback when AV reportedEPS/surprise missing) --------

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
            vprint(verbose, f"  [8-K XBRL] EPS ${eps:.2f}")
        return eps, "xbrl"
    text = _get_ex99_text(filing)
    eps = _extract_eps_from_text(text)
    if eps is not None:
        if verbose:
            vprint(verbose, f"  [8-K text] EPS ${eps:.2f}")
        return eps, "text"
    return None, "none"


def get_eps_for_report_quarter(ticker: str, report_date: date) -> Optional[Dict]:
    """
    Fetch Alpha Vantage EARNINGS; find the quarterly record for the latest quarter (by filing date).
    Returns the quarter-matched record even when surprisePercentage/reportedEPS is missing (same as edgar).
    """
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return None
    latest_quarter_end = _latest_quarter_end_for_date(report_date)
    target = latest_quarter_end.isoformat()  # "YYYY-MM-DD"
    url = "https://www.alphavantage.co/query"
    params = {"function": "EARNINGS", "symbol": ticker.upper(), "apikey": api_key}
    try:
        time.sleep(0.25)
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "Error Message" in data:
            return None
        if "Note" in data:
            print("Alpha Vantage (Note) for %s: %s" % (ticker, data["Note"][:250]))
            return None
        if "Information" in data:
            print("Alpha Vantage (Information) for %s: %s" % (ticker, data["Information"][:250]))
            get_actual_eps_from_8k().sleep(10)
            return None
        quarterly = data.get("quarterlyEarnings") or []
        for r in quarterly:
            fd = r.get("fiscalDateEnding") or ""
            if fd[:10] == target if len(fd) >= 10 else fd == target:
                return r
        return None
    except Exception:
        return None


def compute_filter6_value(filing, verbose: bool = False) -> Optional[Dict]:
    """
    Fetch EPS for the filing's latest quarter (by filing date). Returns AV record or synthetic record
    when surprise is computed from 8-K actual + AV estimate (same logic as edgar Filter 6).
    """
    ticker = getattr(filing, "ticker", None) or cik_to_ticker(str(getattr(filing, "cik", "")))
    if not ticker:
        return None
    fd = getattr(filing, "filing_date", None)
    if fd is None:
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
        return None
    # Primary: AV reported EPS + surprise
    surprise_str = record.get("surprisePercentage")
    reported_str = record.get("reportedEPS")
    try:
        surprise_val = float(surprise_str)
        reported_val = float(reported_str)
        if reported_val != 0:
            eps_score = min(surprise_val, 50) * (reported_val ** 0.5)
            if verbose:
                vprint(verbose, f"FILTER6: {ticker} EPS surprise={surprise_str}% reported={reported_str} score={eps_score}")
            return record
    except (TypeError, ValueError):
        pass
    # Fallback: 8-K actual + AV estimatedEPS (same as edgar)
    actual_eps, source = get_actual_eps_from_8k(filing, verbose=verbose)
    estimated_eps = record.get("estimatedEPS")
    try:
        estimated_eps = float(estimated_eps) if estimated_eps not in (None, "", "None") else None
    except (TypeError, ValueError):
        estimated_eps = None
    if actual_eps is not None and estimated_eps is not None and estimated_eps != 0:
        surprise_pct = ((actual_eps - estimated_eps) / abs(estimated_eps)) * 100
        eps_score = min(surprise_pct, 50) * (actual_eps ** 0.5)
        if verbose:
            vprint(verbose, f"FILTER6: {ticker} 8-K actual ${actual_eps:.2f} ({source}) + AV estimate ${estimated_eps:.2f} → surprise {surprise_pct:+.2f}% score {eps_score}")
        # Synthetic record so downstream (track_candidates, etc.) sees surprisePercentage and reportedEPS
        out = dict(record)
        out["surprisePercentage"] = surprise_pct
        out["reportedEPS"] = str(actual_eps)
        return out
    return record

def _filing_datetime(filing):
    """
    Return filing datetime for sorting/display, or None.
    Uses header.acceptance_datetime / accepted if available, else filing_date (as start-of-day).
    """
    header = getattr(filing, "header", None)
    if header is not None:
        acc = getattr(header, "acceptance_datetime", None) or getattr(header, "accepted", None)
        if acc is not None:
            if hasattr(acc, "hour"):  # datetime-like
                return acc
            if hasattr(acc, "strftime") and hasattr(acc, "date"):
                return acc
            if isinstance(acc, str) and len(acc) >= 12:  # SEC YYYYMMDDHHMMSS
                try:
                    return datetime(
                        int(acc[:4]), int(acc[4:6]), int(acc[6:8]),
                        int(acc[8:10]), int(acc[10:12]), int(acc[12:14]) if len(acc) >= 14 else 0,
                    )
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
        return datetime.combine(d, datetime.min.time())
    return d


# -----------------------------
# 4️⃣ Core 8-K inspection entry point
# -----------------------------
def analyze_8k(filing, verbose: bool = False):
    """
    Basic 8-K inspection.

    Run FILTER1–5 and FILTER6 (EPS). If all pass, return (ticker, cik, accession, eps).
    eps is the Alpha Vantage record for reportedDate == filing date, or None.
    Otherwise return None.
    """

    t = _filing_datetime(filing)

    # Handle both accession_no and accession_number attributes
    accession = getattr(filing, "accession_no", None) or getattr(
        filing, "accession_number", None
    )

    # Resolve ticker via filing or CIK mapping
    cik = str(getattr(filing, "cik", None))
    ticker = getattr(filing, "ticker", None)

    if not ticker:
        ticker = cik_to_ticker(cik)

    if not ticker:
        return None

    print(
        f"Inpecting: ticker={ticker or 'N/A'}, "
        f"CIK={cik}, "
        f"accession={accession}"
    )

    # First filter (basics)
    if not compute_filter1_pass(filing, verbose):
        return None

    # Second filter (red/green flags)
    if not compute_filter2_pass(filing, verbose):
        return None

    # Fourth filter (delta): verify financial data exists in 99.x (order 1 → 4 → 2 → 3)
    if not compute_filter4_pass(filing, verbose):
        return None

    # Third filter (stock health)
    if not compute_filter3_pass(filing, verbose):
        return None

    # Fifth filter (non-LLM parser) → pass only, no score
    if not compute_filter5_pass(filing, verbose):
        return None

    # Sixth filter (EPS beats): fetch once, attach to candidate; production may eliminate if None
    eps = compute_filter6_value(filing, verbose)

    # SEC filing link
    vprint(verbose,f"SEC: https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude")

    # TODO: more filters (e.g. filter7 LLM)
    filing_dt = _filing_datetime(filing)
    return ticker, cik, accession, eps, filing_dt


# -----------------------------
# 5️⃣ Fetch 8-Ks for a specific date
# -----------------------------
def get_8ks_for_date(target_date: date):
    """
    Fetch 8-K filings for a single calendar date.

    We can later expand this to a 0–2 day window, but for
    the first framework run we keep it simple: exact date.
    """
    date_str = target_date.isoformat()
    filings = get_filings(form="8-K", filing_date=date_str)
    return filings


# Accession pattern: 0001104659-26-015732
_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")


def get_8ks_from_file(filepath) -> list[tuple[str, str, str]]:
    """
    Read a file of pasted table lines (symbol, CIK, accession, ...) and return
    list of (symbol, cik, accession). Skips lines that don't have an
    accession-like third column.
    """
    path = Path(filepath) if not isinstance(filepath, Path) else filepath
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("-") or line.startswith("="):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        symbol, cik, accession = parts[0], parts[1], parts[2]
        if _ACCESSION_RE.match(accession):
            rows.append((symbol, cik, accession))
    return rows


def get_eps_for_report_date(ticker: str, report_date: date) -> Optional[Dict]:
    """
    Fetch Alpha Vantage EARNINGS for ticker and return the quarter whose
    reportedDate matches report_date (backtest date). Tries exact date, then
    report_date - 1 day (filing_date can be UTC vs AV reportedDate).
    Returns None if no match or unavailable.
    """
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return None
    url = "https://www.alphavantage.co/query"
    params = {"function": "EARNINGS", "symbol": ticker.upper(), "apikey": api_key}
    try:
        time.sleep(0.25)  # Rate limiting
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "Error Message" in data:
            return None
        if "Note" in data:
            print("Alpha Vantage (Note) for %s: %s" % (ticker, data["Note"][:250]))
            return None
        if "Information" in data:
            print("Alpha Vantage (Information) for %s: %s" % (ticker, data["Information"][:250]))
            return None
        quarterly = data.get("quarterlyEarnings") or []
        for d in (report_date, report_date - timedelta(days=1)):
            target = d.isoformat()
            for record in quarterly:
                if record.get("reportedDate") == target:
                    return record
        return None
    except Exception:
        return None


# -----------------------------
# 5️⃣ Track how stock price reacted (if any)
# -----------------------------
def track_candidates(start_date: date, candidates):
    """
    Track stock price reactions to 8-K filings.
    Shows filing date price, 1d %, and 7d % changes.
    """
    if not candidates:
        print("No candidates to track.")
        return

    # Sort by filing time (earliest first); candidates with no time last
    def _sort_key(c):
        if len(c) >= 5 and c[4] is not None:
            dt = c[4]
            try:
                return (0, dt.timestamp())
            except (AttributeError, OSError):
                return (0, dt)
        return (1, float("inf"))

    candidates = sorted(candidates, key=_sort_key)

    print(f"\n{'=' * 120}")
    print(f"Price Reaction Analysis (Filing Date: {start_date})")
    print(f"{'=' * 120}")

    # Table header (1d = close on filing date, 2d = close on filing date + 1)
    # EPS $ and EPS # removed to make space for 2w % and 52w % (relative to averages)
    header = (
        f"{'Time':<6} {'Symbol':<8} {'CIK':<8} {'Accession':<22} "
        f"{'EPS %':<10} {'Filing $':<10} {'1d %':<8} {'2d %':<8} "
        f"{'7d %':<8} {'2w %':<8} {'52w %':<8}"
    )
    print(header)
    print("-" * 120)

    for candidate in candidates:
        # Support 5-tuple (with filing_dt) or 4-tuple (legacy)
        ticker = candidate[0] or ""
        cik = (candidate[1] or "")[:8]
        accession = (candidate[2] or "")[:22]
        eps_record = candidate[3] if len(candidate) > 3 else None
        filing_dt = candidate[4] if len(candidate) >= 5 else None
        time_str = filing_dt.strftime("%H:%M") if (filing_dt is not None and hasattr(filing_dt, "hour")) else (
            filing_dt.strftime("%Y-%m-%d") if (filing_dt is not None and hasattr(filing_dt, "strftime")) else "N/A"
        )

        if not ticker:
            row = (
                f"{time_str:<6} {'N/A':<8} {cik:<8} {accession:<22} "
                f"{'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<8} "
                f"{'N/A':<8} {'N/A':<8} {'N/A':<8}"
            )
            print(row)
            continue

        # EPS % (surprise) and EPS¢ (beat in cents) from analyze_8k
        eps_str = "N/A"
        # EPS $ and EPS # no longer shown in the candidates table

        pct = None
        if eps_record:
            if eps_record.get("surprisePercentage") not in (None, "", "None"):
                try:
                    pct = float(eps_record["surprisePercentage"])
                    eps_str = f"{pct:+.1f}%"
                except (TypeError, ValueError):
                    pass
            reported = eps_record.get("reportedEPS")
            if reported not in (None, "", "None"):
                try:
                    r = float(reported)
                    eps_dollar_str = f"${r:.2f}"
                    if pct is not None:
                        s = min(pct, 50) * (r ** 0.5)
                        eps_score_str = f"{int(s)}"
                except (TypeError, ValueError):
                    pass
        
        try:
            # Fetch historical prices
            start_lookup = start_date - timedelta(days=5)
            end_lookup = start_date + timedelta(days=10)
            time.sleep(0.05)
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_lookup, end=end_lookup)

            if hist is None or hist.empty or "Close" not in hist.columns:
                row = (
                    f"{time_str:<6} {ticker:<8} {cik:<8} {accession:<22} "
                    f"{eps_str:<10} {'N/A':<10} {'N/A':<8} {'N/A':<8} "
                    f"{'N/A':<8} {'N/A':<8} {'N/A':<8}"
                )
                print(row)
                continue

            # Index dates: handle timezone-aware or naive
            def _index_dates(df):
                try:
                    return [t.date() if hasattr(t, "date") and callable(getattr(t, "date")) else t for t in df.index]
                except Exception:
                    return []

            index_dates = _index_dates(hist)
            if not index_dates:
                row = (
                    f"{time_str:<6} {ticker:<8} {cik:<8} {accession:<22} "
                    f"{eps_str:<10} {'N/A':<10} {'N/A':<8} {'N/A':<8} "
                    f"{'N/A':<8} {'N/A':<8} {'N/A':<8}"
                )
                print(row)
                continue

            # Prior close = last close before start_date (baseline for 1d/2d/7d %)
            prior_close = None
            for i, d in enumerate(index_dates):
                if d < start_date:
                    prior_close = float(hist.iloc[i]["Close"])
                else:
                    break

            # Filing $ = close on filing date (same calendar day)
            close_1d = None
            for i, d in enumerate(index_dates):
                if d >= start_date:
                    close_1d = float(hist.iloc[i]["Close"])
                    break
            filing_price = close_1d  # display as Filing $
            if filing_price is None and index_dates and index_dates[0] > start_date:
                filing_price = float(hist.iloc[0]["Close"])

            # If we still don't have a filing_price (e.g. running before the filing-day bar exists),
            # fall back to the latest available close in the lookup window so we can still
            # compute relative metrics (2w %, 52w %) using a "last close" anchor.
            if filing_price is None:
                try:
                    filing_price = float(hist["Close"].iloc[-1])
                except Exception:
                    filing_price = None

            # 2d = close on filing date + 1
            after_2d_date = start_date + timedelta(days=1)
            close_2d = None
            for i, d in enumerate(index_dates):
                if d >= after_2d_date:
                    close_2d = float(hist.iloc[i]["Close"])
                    break

            after_7d_date = start_date + timedelta(days=7)
            close_7d = None
            for i, d in enumerate(index_dates):
                if d >= after_7d_date:
                    close_7d = float(hist.iloc[i]["Close"])
                    break

            filing_price_str = f"${filing_price:.2f}" if filing_price is not None else "N/A"
            # 1d % = (filing day close - previous day close) / previous day close
            if prior_close is not None and close_1d is not None:
                change_1d = ((close_1d - prior_close) / prior_close) * 100
                change_1d_str = f"{change_1d:+.1f}%"
            else:
                change_1d_str = "N/A"
            if prior_close is not None and close_2d is not None:
                change_2d = ((close_2d - prior_close) / prior_close) * 100
                change_2d_str = f"{change_2d:+.1f}%"
            else:
                change_2d_str = "N/A"
            if prior_close is not None and close_7d is not None:
                change_7d = ((close_7d - prior_close) / prior_close) * 100
                change_7d_str = f"{change_7d:+.1f}%"
            else:
                change_7d_str = "N/A"

            # Relative to 2-week and 52-week highs, using filing_price as anchor
            two_week_pct_str = "N/A"
            fifty_two_pct_str = "N/A"
            if filing_price is not None:
                try:
                    hist_2w = stock.history(
                        start=start_date - timedelta(days=14),
                        end=start_date + timedelta(days=2),
                        auto_adjust=True,
                    )
                    if hist_2w is not None and not hist_2w.empty and "High" in hist_2w.columns:
                        high_2w = float(hist_2w["High"].max())
                        if high_2w > 0:
                            two_week_pct = 100.0 * filing_price / high_2w
                            two_week_pct_str = f"{two_week_pct:.1f}%"
                except Exception:
                    pass
                try:
                    hist_52w = stock.history(
                        start=start_date - timedelta(days=365),
                        end=start_date,
                        auto_adjust=True,
                    )
                    if hist_52w is not None and not hist_52w.empty and "High" in hist_52w.columns:
                        high_52w = float(hist_52w["High"].max())
                        if high_52w > 0:
                            fifty_two_pct = 100.0 * filing_price / high_52w
                            fifty_two_pct_str = f"{fifty_two_pct:.1f}%"
                except Exception:
                    pass

            row = (
                f"{time_str:<6} {ticker:<8} {cik:<8} {accession:<22} "
                f"{eps_str:<10} {filing_price_str:<10} {change_1d_str:<8} {change_2d_str:<8} "
                f"{change_7d_str:<8} {two_week_pct_str:<8} {fifty_two_pct_str:<8}"
            )
            print(row)

        except Exception as e:
            row = (
                f"{time_str:<6} {ticker:<8} {cik:<8} {accession:<22} "
                f"{eps_str:<10} {'N/A':<10} {'N/A':<8} {'N/A':<8} "
                f"{'N/A':<8} {'N/A':<8} {'N/A':<8}"
            )
            print(row)
    
    print(f"{'=' * 120}\n")

# -----------------------------
# 6️⃣ CLI helpers
# -----------------------------
EX99_1_DUMP_PATH = BASE_DIR / "ex99_1_dump.txt"


def dump_ex99_1(filing) -> None:
    """
    Extract EX-99.1 exhibit text and write to ex99_1_dump.txt.
    Only used for --analyse (single accession); lets LLM/test script read the same content.
    """
    text = ""
    if hasattr(filing, "exhibits") and filing.exhibits:
        for ex in filing.exhibits:
            ex_str = str(ex).lower()
            # Match exhibit 99.1 (word boundary to avoid 99.10, 99.11, etc.)
            if re.search(r"99\.1\b", ex_str):
                try:
                    text = ex.text()
                    break
                except Exception:
                    pass
    path = EX99_1_DUMP_PATH
    path.write_text(text or "", encoding="utf-8")
    print(f"Wrote EX-99.1 text to {path} ({len(text)} chars)")


def run_for_date(date_str: str):
    """CLI handler for --date: inspect all 8-Ks on a given day."""
    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        print(f"❌ Invalid date format: {date_str}. Use YYYY-MM-DD")
        return

    print(f"Collecting 8-K filings for date: {target_date}")

    filings = list(get_8ks_for_date(target_date) or [])
    # TODO: Filter by market times (exclude filings accepted during market hours).
    candidates: list[tuple[str, str, str, Optional[Dict], Optional[datetime]]] = []  # (ticker, cik, accession, eps, filing_dt)

    if not filings:
        print("No 8-K filings found for that date.")
        return

    print(f"Found {len(filings)} 8-K filings. Running FILTER1 + basic inspection...")
    for filing in filings:
        try:
            candidate = analyze_8k(filing)

            if candidate is not None:
                candidates.append(candidate)

        except Exception as e:
            print(f"  ⚠️  Error inspecting filing: {e}")

    print(f"Passed {len(candidates)} candidates...")

    track_candidates(target_date, candidates)


def run_for_accession(accession_number: str):
    """CLI handler for --analyse: inspect a single 8-K by accession reference."""
    print(f"Looking up filing: {accession_number}")
    try:
        filing = find(accession_number)
    except Exception as e:
        print(f"❌ Error looking up filing: {e}")
        return

    if not filing:
        print(f"❌ Could not find filing: {accession_number}")
        return

    # Filing time: use header acceptance datetime if available, else date only
    filed_str = str(getattr(filing, "filing_date", "N/A"))
    header = getattr(filing, "header", None)
    if header is not None:
        acc = getattr(header, "acceptance_datetime", None) or getattr(header, "accepted", None)
        if acc is not None:
            if hasattr(acc, "strftime"):
                filed_str = acc.strftime("%Y-%m-%d %H:%M")
            elif isinstance(acc, str) and len(acc) >= 12:
                # SEC format YYYYMMDDHHMMSS
                filed_str = f"{acc[:4]}-{acc[4:6]}-{acc[6:8]} {acc[8:10]}:{acc[10:12]}"
    print(
        f"✓ Found filing: {getattr(filing, 'company', 'N/A')} - "
        f"{getattr(filing, 'form', 'N/A')} filed {filed_str} ET"
    )

    if getattr(filing, "form", None) != "8-K":
        print(
            f"⚠️  Warning: Filing form is {getattr(filing, 'form', None)}, not 8-K. "
            "Inspecting anyway..."
        )

    dump_ex99_1(filing)

    try:
        analyze_8k(filing, True)
    except Exception as e:
        print(f"❌ Error inspecting filing {accession_number}: {e}")


def run_for_file(filepath: str):
    """CLI handler for --file: read pasted table lines (symbol, CIK, accession), fetch each filing, run inspect + track."""
    rows = get_8ks_from_file(filepath)
    if not rows:
        print(f"No accessions found in {filepath}")
        return
    print(f"Found {len(rows)} accessions in file. Fetching filings and running inspection...")
    candidates: list[tuple[str, str, str, Optional[Dict], Optional[datetime]]] = []
    for symbol, cik, accession in rows:
        try:
            filing = find(accession)
            if not filing:
                print(f"  ⚠️  Could not find filing: {accession}")
                continue
            candidate = analyze_8k(filing)
            if candidate is not None:
                candidates.append(candidate)
        except Exception as e:
            print(f"  ⚠️  Error inspecting {accession}: {e}")
    if candidates:
        print(f"Passed {len(candidates)} candidates...")
        track_candidates(date.today(), candidates)
    else:
        print("No candidates passed.")


def run_latest():
    """
    CLI handler for no-arg run:
    use get_latest_filings via edgar, filter to 8-Ks,
    and apply FILTER1 + basic inspection.
    """
    print("Fetching latest filings...")
    latest = get_latest_filings()

    try:
        filings = list(latest)
    except Exception as e:
        print(f"❌ Error converting latest filings to list: {e}")
        return

    # Filter to 8-Ks only
    filings_8k = [f for f in filings if getattr(f, "form", None) == "8-K"]
    if not filings_8k:
        print("No latest 8-K filings found.")
        return

    print(f"Found {len(filings_8k)} latest 8-K filings. Running FILTER1 + basic inspection...")
    candidates: list[tuple[str, str, str, Optional[Dict], Optional[datetime]]] = []
    for filing in filings_8k:
        try:
            candidate = analyze_8k(filing)
            if candidate is not None:
                candidates.append(candidate)
        except Exception as e:
            print(f"  ⚠️  Error inspecting latest filing: {e}")

    if candidates:
        print(f"Passed {len(candidates)} candidates...")
        track_candidates(date.today(), candidates)


# -----------------------------
# 7️⃣ Main entry point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect 8-K filings (framework + FILTER1, no EPS/LLM yet)"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Date in YYYY-MM-DD format to list 8-K filings for inspection",
    )
    parser.add_argument(
        "--analyse",
        type=str,
        help="Inspect a specific filing by accession number (e.g., 0001660280-25-000128)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to file of pasted table lines (symbol, CIK, accession) to process",
    )

    args = parser.parse_args()

    if args.analyse:
        run_for_accession(args.analyse)
    elif args.file:
        run_for_file(args.file)
    elif args.date:
        run_for_date(args.date)
    else:
        run_latest()

