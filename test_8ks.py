from edgar import set_identity, get_filings, get_latest_filings, find, Company
from datetime import date, timedelta
from typing import Optional
import re
import argparse

# CIK to ticker cache
_CIK_TO_TICKER_CACHE = {}

# -----------------------------
# 1️⃣ Set SEC identity (mandatory!)
# -----------------------------
# Replace with your real info for SEC compliance
set_identity("David McCarthy david@example.com")

# -----------------------------
# 2️⃣ EPS regex extraction
# -----------------------------
EPS_PATTERNS = [
    r"earnings per share of\s+\$?(-?\d+\.?\d*)",
    r"eps of\s+\$?(-?\d+\.?\d*)",
    r"adjusted earnings per share of\s+\$?(-?\d+\.?\d*)",
    r"adjusted eps of\s+\$?(-?\d+\.?\d*)",
]

def extract_eps_values(text: str):
    eps_values = []
    for pattern in EPS_PATTERNS:
        for match in re.findall(pattern, text, re.IGNORECASE):
            try:
                eps_values.append(float(match))
            except ValueError:
                pass
    return eps_values

# -----------------------------
# 3️⃣ CIK to ticker conversion
# -----------------------------
def cik_to_ticker(cik: str) -> Optional[str]:
    """Map CIK to ticker symbol using edgartools Company."""
    cik = str(cik).zfill(10)
    
    if cik in _CIK_TO_TICKER_CACHE:
        return _CIK_TO_TICKER_CACHE[cik]
    
    try:
        company = Company(cik)
        ticker = company.get_ticker()
        _CIK_TO_TICKER_CACHE[cik] = ticker
        return ticker
    except Exception as e:
        _CIK_TO_TICKER_CACHE[cik] = None
        return None

# -----------------------------
# 4️⃣ Detect earnings-related 8-K
# -----------------------------
def is_earnings_8k(filing) -> bool:
    """Check if 8-K is earnings related."""
    text = filing.text().lower()
    if "item 2.02" in text or "results of operations" in text:
        return True
    # Check exhibits for 99.1
    if hasattr(filing, "items"):
        for item in filing.items:
            if "exhibit 99.1" in item.lower():
                return True
    return False


# -----------------------------
# 5️⃣ Fetch 8-Ks for 0–2 day window
# -----------------------------
def get_8ks_0_to_2_days(target_date: date):
    start = target_date.isoformat()
    end   = (target_date + timedelta(days=2)).isoformat()

    # Fetch 8-K filings in the 0–2 day window
    filings = get_filings(form="8-K", filing_date=f"{start}:{end}")
    return filings

# -----------------------------
# 6️⃣ Analyze EPS in 8-K
# -----------------------------
def analyze_8k_eps(filing):
    """Analyze a single 8-K for earnings-related info including exhibits (v5.10.1 compatible)."""

    if not is_earnings_8k(filing):
        return None

    try:
        text = filing.text()
    except AttributeError:
        return None

    # Handle both accession_no and accession_number attributes
    accession = getattr(filing, "accession_no", None) or getattr(filing, "accession_number", None)
    print(f"Analyzing filing CIK={filing.cik} | Accession={accession}")

    eps_values = extract_eps_values(text)

    # Try exhibits if main text has nothing
    if not eps_values and hasattr(filing, "exhibits"):
        for att in filing.exhibits:  # Attachment objects
            att_str = str(att).lower()  # use string representation
            if "99.1" in att_str:
                try:
                    exhibit_text = att.text()
                    eps_values += extract_eps_values(exhibit_text)
                except Exception as e:
                    print(f"Failed to fetch exhibit {att_str} for CIK={filing.cik}: {e}")

    if not eps_values:
        return None

    # Progress log
    print(f"Analyzing filing CIK={filing.cik} | Accession={accession} | EPS found: {eps_values}")

    # Get ticker - try from filing first, then resolve from CIK
    ticker = getattr(filing, "ticker", None)
    if not ticker:
        ticker = cik_to_ticker(filing.cik)
        if ticker:
            print(f"  Resolved ticker: {ticker}")

    return {
        "cik": filing.cik,
        "ticker": ticker,
        "filing_date": filing.filing_date,
        "accession": accession,
        "eps_values": eps_values,
        "max_eps": max(eps_values),
        "min_eps": min(eps_values),
        "eps_count": len(eps_values),
    }


# -----------------------------
# 7️⃣ Backtest runner
# -----------------------------
def backtest_eps_surprises(start_date: date, end_date: date):
    results = []
    current = start_date

    while current <= end_date:
        eight_ks = get_8ks_0_to_2_days(current)
        for filing in eight_ks:
            analysis = analyze_8k_eps(filing)
            if analysis:
                results.append(analysis)
        current += timedelta(days=1)

    return results

# -----------------------------
# 8️⃣ Fetch latest 8-Ks
# -----------------------------
def get_latest_8ks():
    """Fetch latest 8-K filings using get_latest_filings()."""
    print("Fetching latest filings...")
    all_latest = get_latest_filings()
    
    # Convert to list if needed
    if not isinstance(all_latest, list):
        try:
            all_latest = list(all_latest)
        except Exception as e:
            print(f"Error converting latest filings to list: {e}")
            all_latest = []
    
    # Filter for 8-Ks
    filings_8k = []
    for f in all_latest:
        try:
            form = getattr(f, 'form', None)
            if form == "8-K":
                filings_8k.append(f)
        except Exception as e:
            continue
    
    print(f"Found {len(filings_8k)} latest 8-K filings")
    return filings_8k

# -----------------------------
# 9️⃣ Analyze by filing reference
# -----------------------------
def analyze_filing_by_reference(accession_number: str):
    """Analyze a specific 8-K filing by accession number."""
    print(f"Looking up filing: {accession_number}")
    try:
        filing = find(accession_number)
        if not filing:
            print(f"❌ Could not find filing: {accession_number}")
            return None
        
        print(f"✓ Found filing: {filing.company} - {filing.form} on {filing.filing_date}")
        
        if filing.form != "8-K":
            print(f"⚠️  Warning: Filing is {filing.form}, not 8-K. Analyzing anyway...")
        
        analysis = analyze_8k_eps(filing)
        if analysis:
            print("\n" + "=" * 80)
            print("ANALYSIS RESULTS:")
            print("=" * 80)
            for key, value in analysis.items():
                print(f"  {key}: {value}")
            return analysis
        else:
            print("❌ No EPS values found in filing")
            return None
            
    except Exception as e:
        print(f"❌ Error analyzing filing {accession_number}: {e}")
        import traceback
        traceback.print_exc()
        return None

# -----------------------------
# 🔟 Main entry point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze 8-K filings for EPS values")
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format for backtesting")
    parser.add_argument("--filing", type=str, help="Analyze specific filing by accession number (e.g., 0001660280-25-000128)")
    
    args = parser.parse_args()
    
    if args.filing:
        # Analyze single filing by reference
        analyze_filing_by_reference(args.filing)
    
    elif args.date:
        # Backtest single date
        try:
            target_date = date.fromisoformat(args.date)
            print(f"Backtesting 8-K filings for date: {target_date}")
            results = backtest_eps_surprises(target_date, target_date)
            
            print(f"\n{'=' * 80}")
            print(f"Found {len(results)} earnings-related 8-K filings with EPS values.")
            print(f"{'=' * 80}")
            for r in results:
                print(f"\nTicker: {r.get('ticker', 'N/A')} | CIK: {r.get('cik', 'N/A')}")
                print(f"  Filing Date: {r.get('filing_date', 'N/A')}")
                print(f"  Accession: {r.get('accession', 'N/A')}")
                print(f"  EPS Values: {r.get('eps_values', [])}")
                print(f"  Max EPS: {r.get('max_eps', 'N/A')}")
        except ValueError:
            print(f"❌ Invalid date format: {args.date}. Use YYYY-MM-DD")
    
    else:
        # Fetch latest filings and analyze
        print("Fetching latest 8-K filings...")
        latest_8ks = get_latest_8ks()
        
        if not latest_8ks:
            print("No latest 8-K filings found")
            exit(0)
        
        print(f"\nAnalyzing {len(latest_8ks)} latest 8-K filings...")
        results = []
        for filing in latest_8ks:
            analysis = analyze_8k_eps(filing)
            if analysis:
                results.append(analysis)
        
        print(f"\n{'=' * 80}")
        print(f"Found {len(results)} earnings-related 8-K filings with EPS values.")
        print(f"{'=' * 80}")
        for r in results:
            print(f"\nTicker: {r.get('ticker', 'N/A')} | CIK: {r.get('cik', 'N/A')}")
            print(f"  Filing Date: {r.get('filing_date', 'N/A')}")
            print(f"  Accession: {r.get('accession', 'N/A')}")
            print(f"  EPS Values: {r.get('eps_values', [])}")
            print(f"  Max EPS: {r.get('max_eps', 'N/A')}")

