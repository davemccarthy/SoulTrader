from edgar import set_identity, get_filings, get_latest_filings, find, Company
from datetime import date
from typing import Optional, Dict
from pathlib import Path
from dotenv import load_dotenv
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
    "regulation fd",
]

# -----------------------------
# FILTER2: red/green flags for risk and quality scoring
# -----------------------------
# Red flags: severe penalties (bankruptcy, going concern, restatement)
RED_FLAGS_SEVERE = {
    "chapter 11": -10,
    "bankruptcy filing": -10,
    "receivership": -10,
    "substantial doubt": -8,
    "ability to continue as a going concern": -8,
    "going concern": -8,
    "restatement of previously issued financial statements": -7,
    "should no longer be relied upon": -7,
}

# Red flags: moderate penalties (management changes, material weakness, etc.)
RED_FLAGS_MODERATE = {
    # Management changes (check for CEO/CFO + resignation/termination/departure)
    "resignation": -2,  # Will catch CEO/CFO resignations in context
    "termination": -2,
    "departure": -2,
    "material weakness": -2,
    "internal control deficiency": -2,
    "sec investigation": -2,
    "sec inquiry": -2,
    "class action": -1,
    "litigation": -1,
    "restructuring": -3,  # non-bankruptcy restructuring
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
    "record revenue": +1,
    "record ebitda": +1,
    "record earnings": +1,
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
    for ex in filing.exhibits:
        ex_str = str(ex).lower()

        if "99." in ex_str:  # or "99.1" if you want to be stricter
            exhibit_99_text = ex.text()
            length = len(exhibit_99_text)

            digit_count = sum(ch.isdigit() for ch in exhibit_99_text)
            digit_ratio = digit_count / length if length > 0 else 0.0

            if length < 5000 or digit_ratio < 0.02:
                vprint(verbose, "FILTER1: exhibit_99 not creditable → fail")
                return 0

    filing_text = filing.text().lower()

    # Bad words
    for kw in REG_FD_KEYWORDS:
        if kw in filing_text:
            vprint(verbose, f"FILTER1: Reg keyword {kw} in main text → fail")
            return 0

    # Good words
    has_earnings_keyword = False
    for kw in EARNINGS_KEYWORDS:
        if kw in filing_text:
            has_earnings_keyword = True
            break

    if not has_earnings_keyword:
        vprint(verbose, "FILTER1: No earnings key phrase in main text")
        return 0

    # Return score
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
    
    # Retrieve exhibit 99 text (duplicate logic from FILTER1)
    exhibit_99_text = None
    if hasattr(filing, "exhibits") and filing.exhibits:
        for ex in filing.exhibits:
            ex_str = str(ex).lower()
            if "99." in ex_str:
                try:
                    exhibit_99_text = ex.text()
                    break
                except Exception:
                    exhibit_99_text = None
    
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

    return score


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
    cik = getattr(filing, "cik", None)
    ticker = getattr(filing, "ticker", None)

    if not ticker:
        ticker = cik_to_ticker(cik)

    if not ticker:
        return None

    print(
        f"Inpecting: ticker={ticker or 'N/A'}, "
        f"CIK={cik}, "
        f"accession={accession}",
    )

    # First filter (basics)
    score += compute_filter1_pass(filing, verbose)

    # TODO: table of filter scores
    if score < 1:
        return None

    # Second filter (red/green flags)
    score += compute_filter2_pass(filing, verbose)

    if score < 2:
        return None

    # TODO: more filters
    return ticker, accession, score


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
    # A look into past's future
    for candidate in candidates:
        ticker, accession, score = candidate  # Unpack
        print(f"{ticker}, {accession}, {score}")

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

