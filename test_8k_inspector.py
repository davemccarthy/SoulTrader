import os
import re
import time
from collections import Counter
from pathlib import Path

import requests
import yfinance as yf
from dotenv import load_dotenv
from edgar import set_identity, get_filings, get_latest_filings, find, Company
from datetime import date, timedelta
from typing import Optional, Dict
import argparse


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
FILTER3_PRICE_MIN = 5.0
FILTER3_PRICE_MAX = 1000.0


def compute_filter3_pass(filing, verbose: bool = False) -> bool:
    """
    FILTER3: valuation, cap, and price band.
    - P/E: fail if trailing P/E > FILTER3_PE_MAX (overvalued definitive).
    - Cap: fail if market_cap < FILTER3_MIN_CAP (nano/micro).
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

    # Price band: $5–$500
    price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
    if price is not None and isinstance(price, (int, float)):
        if price < FILTER3_PRICE_MIN:
            vprint(verbose, f"FILTER3: fail price ${price:.2f} < ${FILTER3_PRICE_MIN} (below band)")
            return False
        if price > FILTER3_PRICE_MAX:
            vprint(verbose, f"FILTER3: fail price ${price:.2f} > ${FILTER3_PRICE_MAX} (above band)")
            return False

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
# FILTER5: Non-LLM delta parser (2/3: past, guidance, EPS)
# -----------------------------
def _filter5_parse_number(raw_val: str, scale: Optional[str] = None) -> float:
    """Convert string like '3,025' with optional million/billion into float magnitude."""
    num = float((raw_val or "0").replace(",", ""))
    if scale:
        scale = scale.lower()
        if scale == "billion":
            num *= 1e9
        elif scale == "million":
            num *= 1e6
    return num


def _filter5_extract_metrics(text: str) -> dict:
    """
    Extract revenue, net income, EPS, guidance/backlog with optional % change.
    Returns dict of lists: revenue, net_income, eps, guidance.
    """
    number_pattern = r"\$?([\d,]+(?:\.\d+)?)\s*(million|billion)?"
    metric_patterns = {
        "revenue": re.compile(
            rf"(revenue|revenues).*?{number_pattern}.*?(?:up|increase|down|decrease)?\s*(?:of\s*)?(\d+)?\s*%?",
            re.IGNORECASE | re.DOTALL,
        ),
        "net_income": re.compile(
            rf"(net\s+income).*?{number_pattern}.*?(?:up|increase|down|decrease)?\s*(?:of\s*)?(\d+)?\s*%?",
            re.IGNORECASE | re.DOTALL,
        ),
        "eps": re.compile(
            rf"(eps|earnings\s+per\s+share|diluted).*?{number_pattern}.*?(?:up|increase|down|decrease)?\s*(?:of\s*)?(\d+)?\s*%?",
            re.IGNORECASE | re.DOTALL,
        ),
    }
    metrics = {"revenue": [], "net_income": [], "eps": [], "guidance": []}

    for key, pat in metric_patterns.items():
        for m in pat.finditer(text):
            g = m.groups()
            raw_val = g[1] if len(g) > 1 else None
            scale = g[2] if len(g) > 2 else None
            pct_change = g[3] if len(g) > 3 and g[3] else None
            if raw_val is None:
                continue
            try:
                val = _filter5_parse_number(raw_val, scale)
            except (ValueError, TypeError):
                continue
            pct = float(pct_change) if pct_change else None
            metrics[key].append({"value": val, "pct_change": pct})

    guidance_pat = re.compile(
        rf"(guidance|backlog|remaining\s+performance\s+obligations).*?{number_pattern}",
        re.IGNORECASE | re.DOTALL,
    )
    for m in guidance_pat.finditer(text):
        g = m.groups()
        raw_val = g[1] if len(g) > 1 else None
        scale = g[2] if len(g) > 2 else None
        if raw_val is None:
            continue
        try:
            val = _filter5_parse_number(raw_val, scale)
        except (ValueError, TypeError):
            continue
        metrics["guidance"].append({"value": val})

    return metrics


def _filter5_compute_parts(text: str, metrics: dict, verbose: bool) -> dict:
    """
    Return numeric parts for FILTER5 score. Each part in {-1, 0, +1}.
    REVENUE = past performance (rev/NI/EPS): +1 good, 0 neutral, -1 bad.
    GUIDANCE = +1 present, 0 absent.
    EXPECTATION = tone: +1 SURPASS, 0 NEUTRAL, -1 SHORTFALL.
    """
    rev_up = any((m.get("pct_change") or 0) > 0 for m in metrics.get("revenue", []))
    ni_up = any((m.get("pct_change") or 0) > 0 for m in metrics.get("net_income", []))
    rev_down = any((m.get("pct_change") or 0) < 0 for m in metrics.get("revenue", []))
    ni_down = any((m.get("pct_change") or 0) < 0 for m in metrics.get("net_income", []))
    has_eps = len(metrics.get("eps", [])) > 0
    eps_down = any((m.get("pct_change") or 0) < 0 for m in metrics.get("eps", [])) if has_eps else False

    # Past performance: -1 if EPS down or rev/NI down; +1 if rev up and NI up and EPS not down; else 0
    if (has_eps and eps_down) or rev_down or ni_down:
        revenue = -1
    elif rev_up and ni_up and not eps_down:
        revenue = 1
    else:
        revenue = 0

    # Guidance: +1 if present, 0 if absent
    guidance = 1 if len(metrics.get("guidance", [])) > 0 else 0

    # Expectation (tone): +1 SURPASS, -1 SHORTFALL, 0 NEUTRAL
    negative = ["slightly", "shift", "timing", "unchanged", "factored", "remains"]
    positive = ["ahead of expectations", "accelerat", "raised", "better than"]
    neg_hits = sum(p in text for p in negative)
    pos_hits = sum(p in text for p in positive)
    if pos_hits > neg_hits:
        expectation = 1
    elif neg_hits > pos_hits:
        expectation = -1
    else:
        expectation = 0

    return {"REVENUE": revenue, "GUIDANCE": guidance, "EXPECTATION": expectation}


FILTER5_PASS_THRESHOLD = 1  # Pass if total score >= this (internal only; no score returned)


def compute_filter5_pass(filing, verbose: bool = False) -> bool:
    """
    FILTER5: non-LLM delta parser (2/3 ducks).
    Gets 99.x text, extracts metrics, computes REVENUE/GUIDANCE/EXPECTATION parts (-1/0/+1).
    Returns True if score >= FILTER5_PASS_THRESHOLD (pass), False otherwise. No score exposed.
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
        vprint(verbose, "FILTER5: no 99.x exhibit text → pass (skip)")
        return True

    metrics = _filter5_extract_metrics(text)

    total_extracted = sum(len(metrics.get(k, [])) for k in ("revenue", "net_income", "eps", "guidance"))
    if total_extracted == 0:
        vprint(verbose, "FILTER5: no metrics parsed → pass (skip)")
        print("FILTER5: no metrics parsed → pass (skip)")
        return True

    parts = _filter5_compute_parts(text.lower(), metrics, verbose)
    score = parts["REVENUE"] + parts["GUIDANCE"] + parts["EXPECTATION"]
    passed = score >= FILTER5_PASS_THRESHOLD

    if passed:
        vprint(verbose, f"FILTER5: (pass) {parts}")
    else:
        vprint(verbose, f"FILTER5: (fail) {parts}")
    return passed

# ----------------
# FILTER6: retrieve EPS beat (Alpha Vantage, quarter where reportedDate == filing date)
# -----------------------------
def compute_filter6_value(filing, verbose: bool = False) -> Optional[Dict]:
    """
    Fetch EPS for the filing's date (reportedDate match). Returns Alpha Vantage record or None.
    In production (edgar advisor) no EPS could eliminate; in test script we let through.
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
    record = get_eps_for_report_date(ticker, report_date)
    if verbose and record:
        vprint(verbose, f"FILTER6: {ticker} EPS surprise={record.get('surprisePercentage')}%")
    return record

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
    return ticker, cik, accession, eps


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


def get_eps_for_report_date(ticker: str, report_date: date) -> Optional[Dict]:
    """
    Fetch Alpha Vantage EARNINGS for ticker and return the quarter whose
    reportedDate matches report_date (backtest date). Enables clean backtest:
    EPS shown is the one that was reported on that date.
    Returns None if no match or unavailable. (One-day-off reportedDate can be handled later.)
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
        if "Error Message" in data or "Note" in data or "Information" in data:
            return None
        quarterly = data.get("quarterlyEarnings") or []
        target = report_date.isoformat()
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
    
    print(f"\n{'=' * 110}")
    print(f"Price Reaction Analysis (Filing Date: {start_date})")
    print(f"{'=' * 110}")
    
    # Table header (EPS after Accession, before Filing $)
    header = f"{'Symbol':<8} {'CIK':<8} {'Accession':<22} {'EPS':<10} {'Filing $':<10} {'1d %':<8} {'7d %':<8}"
    print(header)
    print("-" * 110)
    
    results = []
    
    for candidate in candidates:
        ticker, cik, accession, eps_record = candidate  # (ticker, cik, accession, eps)

        if not ticker:
            # Skip if no ticker
            row = f"{'N/A':<8} {cik[:8]:<8} {accession[:22]:<22} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<8}"
            print(row)
            continue

        # EPS from analyze_8k (no second fetch)
        if eps_record and eps_record.get("surprisePercentage") not in (None, "", "None"):
            try:
                pct = float(eps_record["surprisePercentage"])
                eps_str = f"{pct:+.1f}%"
            except (TypeError, ValueError):
                eps_str = "N/A"
        else:
            eps_str = "N/A"
        
        try:
            # Fetch historical prices
            # Get data from 5 days before filing to 10 days after (buffer for weekends/holidays)
            start_lookup = start_date - timedelta(days=5)
            end_lookup = start_date + timedelta(days=10)
            
            # Rate limiting
            time.sleep(0.05)
            
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_lookup, end=end_lookup)
            
            if hist.empty:
                # No price data available
                row = f"{ticker:<8} {cik[:8]:<8} {accession[:22]:<22} {eps_str:<10} {'N/A':<10} {'N/A':<8} {'N/A':<8}"
                print(row)
                continue
            
            # Get filing date price (or nearest trading day before filing)
            filing_prices = hist[hist.index.date <= start_date]
            if filing_prices.empty:
                filing_price = None
            else:
                filing_price = filing_prices.iloc[-1]["Close"]
            
            # Get 1d price (next trading day after filing)
            after_1d_date = start_date + timedelta(days=1)
            after_1d_prices = hist[hist.index.date >= after_1d_date]
            after_1d_price = after_1d_prices.iloc[0]["Close"] if not after_1d_prices.empty else None
            
            # Get 7d price (7 calendar days after filing, find next trading day)
            after_7d_date = start_date + timedelta(days=7)
            after_7d_prices = hist[hist.index.date >= after_7d_date]
            after_7d_price = after_7d_prices.iloc[0]["Close"] if not after_7d_prices.empty else None
            
            # Calculate percentage changes
            filing_price_str = f"${filing_price:.2f}" if filing_price else "N/A"
            
            if filing_price and after_1d_price:
                change_1d = ((after_1d_price - filing_price) / filing_price) * 100
                change_1d_str = f"{change_1d:+.1f}%"
            else:
                change_1d_str = "N/A"
            
            if filing_price and after_7d_price:
                change_7d = ((after_7d_price - filing_price) / filing_price) * 100
                change_7d_str = f"{change_7d:+.1f}%"
            else:
                change_7d_str = "N/A"
            
            # Print row (EPS after Accession, before Filing $)
            row = f"{ticker:<8} {cik[:8]:<8} {accession[:22]:<22} {eps_str:<10} {filing_price_str:<10} {change_1d_str:<8} {change_7d_str:<8}"
            print(row)
            
        except Exception as e:
            # Error fetching price data
            row = f"{ticker:<8} {cik[:8]:<8} {accession[:22]:<22} {eps_str:<10} {'Error':<10} {'N/A':<8} {'N/A':<8}"
            print(row)
    
    print(f"{'=' * 110}\n")

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
    candidates: list[tuple[str, str, str, Optional[Dict]]] = []  # (ticker, cik, accession, eps)

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

    print(
        f"✓ Found filing: {getattr(filing, 'company', 'N/A')} - "
        f"{getattr(filing, 'form', 'N/A')} on {getattr(filing, 'filing_date', 'N/A')}"
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
    for filing in filings_8k:
        try:
            analyze_8k(filing)
        except Exception as e:
            print(f"  ⚠️  Error inspecting latest filing: {e}")


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

    args = parser.parse_args()

    if args.analyse:
        run_for_accession(args.analyse)
    elif args.date:
        run_for_date(args.date)
    else:
        run_latest()

