import re

from edgar import set_identity, get_filings, get_latest_filings, find, Company
from datetime import date, timedelta
from typing import Optional, Dict
from pathlib import Path
from dotenv import load_dotenv
import argparse
import yfinance as yf
import time


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

def compute_filter1_pass(filing, verbose: bool = False) -> int:
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
        return 0

    # Resolve ticker
    ticker = getattr(filing, "ticker", None)
    if not ticker:
        ticker = cik_to_ticker(getattr(filing, "cik", ""))
    if not ticker:
        vprint(verbose, "FILTER1: no tradable ticker → fail")
        return 0

    # No exhibits
    if not (hasattr(filing, "exhibits") and filing.exhibits):
        vprint(verbose, "FILTER1: no exhibits → fail")
        return 0

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
        return 0

    # Examine filing text
    filing_text = filing.text().lower()

    # Need item 9.0x
    if "item 9.0" not in filing_text:
        vprint(verbose, "FILTER1: item 9.0.x → fail")
        return 0

    # Bad words
    for kw in REG_FD_KEYWORDS:
        if kw in filing_text:
            vprint(verbose, f"FILTER1: Reg keyword {kw} in main text → fail")
            return 0

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
    vprint(verbose, "FILTER1: pass")
    return 1


# -----------------------------
# FILTER2: red/green flags scoring
# -----------------------------
def compute_filter2_pass(filing, verbose: bool = False) -> int:
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

    vprint(verbose, f"FILTER2: score {score}")
    return score


# ----------------
# FILTER3: generic sanity checks (valuation / size / momentum)
# -----------------------------
def compute_filter3_pass(filing, verbose: bool = False) -> int:
    """
    Planned checks (stub for now; always passes):
    - Over-valued: e.g. P/E or other valuation above threshold
    - Nano / micro cap: exclude very small market cap (illiquid, high spread)
    - Price already moved too much: e.g. big run-up before 8-K (don't chase)
    - Notional vs actual share price

    Returns: score neutral
    """

    # Stub: no checks implemented yet
    vprint(verbose, f"FILTER3: (stub)")
    return 0

# ----------------
# FILTER4: preliminary delta check (minimal: verify financial data exists + revenue current/prior)
# -----------------------------
FILTER4_FINANCIAL_TERMS = (
    "revenue",
    "net sales",
    "eps",
    "earnings per share",
    "net income",
    "operating income",
)

_FILTER4_NUMBER_RE = re.compile(
    r"[\d,]+\.?\d*\s*(million|billion)?",
    re.IGNORECASE,
)

# Revenue: number with optional $ and million/billion (for parsing magnitude)
_FILTER4_REVENUE_NUM_RE = re.compile(
    r"\$?\s*([\d,]+\.?\d*)\s*(million|billion)?",
    re.IGNORECASE,
)

# Prior-year / comparison phrases (for finding prior period number)
_FILTER4_PRIOR_PHRASES = (
    "compared to",
    "prior year",
    "prior quarter",
    "versus",
    "last year",
    "year ago",
    "compared with",
)

# Last extraction result for downstream (revenue current/prior/direction)
FILTER4_LAST_RESULT: Optional[dict] = None


def _filter4_parse_number(match) -> Optional[float]:
    """Convert regex match (digits + optional million/billion) to float magnitude."""
    if not match:
        return None
    s = match.group(1).replace(",", "")
    try:
        val = float(s)
    except ValueError:
        return None
    unit = (match.group(2) or "").lower()
    if unit == "billion":
        val *= 1e9
    elif unit == "million":
        val *= 1e6
    return val


def _filter4_find_revenue_current(text: str) -> Optional[float]:
    """Find first revenue-like number after 'revenue' or 'net sales'."""
    text_lower = text.lower()
    for trigger in ("revenue", "net sales", "total revenue"):
        idx = text_lower.find(trigger)
        if idx == -1:
            continue
        # Search in next ~250 chars for a number
        segment = text[idx : idx + 250]
        m = _FILTER4_REVENUE_NUM_RE.search(segment)
        if m:
            return _filter4_parse_number(m)
    return None


def _filter4_find_revenue_prior(text: str) -> Optional[float]:
    """Find a prior-period number near comparison phrases."""
    text_lower = text.lower()
    for phrase in _FILTER4_PRIOR_PHRASES:
        idx = text_lower.find(phrase)
        if idx == -1:
            continue
        # Search in window around phrase (±150 chars) for a number
        start = max(0, idx - 80)
        end = min(len(text), idx + 150)
        segment = text[start:end]
        m = _FILTER4_REVENUE_NUM_RE.search(segment)
        if m:
            return _filter4_parse_number(m)
    return None


def compute_filter4_pass(filing, verbose: bool = False) -> int:
    """
    FILTER4: preliminary delta check.
    Verifies 99.x exhibit has financial data; extracts revenue current/prior and direction (UP/DOWN/FLAT).
    Result stored in FILTER4_LAST_RESULT for downstream. Returns 1 if financial data present, 0 otherwise.
    """
    global FILTER4_LAST_RESULT
    FILTER4_LAST_RESULT = None

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
        return 0

    text_lower = text.lower()
    has_term = any(term in text_lower for term in FILTER4_FINANCIAL_TERMS)
    has_number = bool(_FILTER4_NUMBER_RE.search(text))

    if not (has_term and has_number):
        vprint(verbose, "FILTER4: no financial metrics / numbers in 99.x → fail")
        return 0

    # Step 1 & 2: extract revenue current and prior; compute direction
    revenue_current = _filter4_find_revenue_current(text)
    revenue_prior = _filter4_find_revenue_prior(text)

    direction = None
    if revenue_current is not None and revenue_prior is not None and revenue_prior != 0:
        if revenue_current > revenue_prior:
            direction = "UP"
        elif revenue_current < revenue_prior:
            direction = "DOWN"
        else:
            direction = "FLAT"

    result = {
        "has_financials": True,
        "revenue": {
            "current": revenue_current,
            "prior": revenue_prior,
            "direction": direction,
        },
    }
    FILTER4_LAST_RESULT = result

    vprint(verbose, "FILTER4: financial data present → pass")
    vprint(verbose, f"FILTER4: revenue current={revenue_current}, prior={revenue_prior}, direction={direction}")

    return 1


# -----------------------------
# 4️⃣ Core 8-K inspection entry point
# -----------------------------
def analyze_8k(filing, verbose: bool = False) -> (str, int):
    """
    Basic 8-K inspection.

    First pass: no regex, no LLM – just identify the filing,
    run FILTER1, and (for now) print a minimal inspection line
    for those that pass.
    """
    score = 0

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
    score += compute_filter1_pass(filing, verbose)

    # TODO: table of filter scores
    if score < 1:
        return None

    # Fourth filter (delta): verify financial data exists in 99.x (order 1 → 4 → 2 → 3)
    if compute_filter4_pass(filing, verbose) < 1:
        return None

    # Second filter (red/green flags)
    score += compute_filter2_pass(filing, verbose)

    if score < 1:
        pass
        #return None

    # Third filter (stock health)
    score += compute_filter3_pass(filing, verbose)

    # SEC filing link
    vprint(verbose,f"SEC: https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude")

    # TODO: more filters
    return ticker, cik, accession, score


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
    
    print(f"\n{'=' * 100}")
    print(f"Price Reaction Analysis (Filing Date: {start_date})")
    print(f"{'=' * 100}")
    
    # Table header
    header = f"{'Symbol':<8} {'CIK':<8} {'Accession':<22} {'Score':<7} {'Filing $':<10} {'1d %':<8} {'7d %':<8}"
    print(header)
    print("-" * 100)
    
    results = []
    
    for candidate in candidates:
        ticker, cik, accession, score = candidate  # Unpack
        
        if not ticker:
            # Skip if no ticker
            row = f"{'N/A':<8} {cik[:8]:<8}  {accession[:22]:<22} {score:<7} {'N/A':<10} {'N/A':<8} {'N/A':<8}"
            print(row)
            continue
        
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
                row = f"{ticker:<8} {cik[:8]:<8} {accession[:22]:<22} {score:<7} {'N/A':<10} {'N/A':<8} {'N/A':<8}"
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
            
            # Print row
            row = f"{ticker:<8} {cik[:8]:<8} {accession[:22]:<22} {score:<7} {filing_price_str:<10} {change_1d_str:<8} {change_7d_str:<8}"
            print(row)
            
        except Exception as e:
            # Error fetching price data
            row = f"{ticker:<8} {cik[:8]:<8} {accession[:22]:<22} {score:<7} {'Error':<10} {'N/A':<8} {'N/A':<8}"
            print(row)
    
    print(f"{'=' * 100}\n")

# -----------------------------
# 6️⃣ CLI helpers
# -----------------------------
def run_for_date(date_str: str):
    """CLI handler for --date: inspect all 8-Ks on a given day."""
    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        print(f"❌ Invalid date format: {date_str}. Use YYYY-MM-DD")
        return

    print(f"Collecting 8-K filings for date: {target_date}")

    filings = list(get_8ks_for_date(target_date) or [])
    candidates: list[tuple[str, str, int]] = []

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

