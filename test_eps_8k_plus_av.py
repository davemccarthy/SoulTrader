"""
Alternative EPS: actual from 8-K (XBRL or exhibit 99.1 text) + estimate from Alpha Vantage → compute surprise.

Use when Alpha Vantage hasn't yet populated reportedEPS (real-time) or to cross-check.
Plan: get actual from 8-K XBRL (e.g. EarningsPerShareDiluted) or exhibit text; estimate from AV; surprise = (actual - est) / |est| * 100.

Usage:
    python test_eps_8k_plus_av.py --accession 0001751008-26-000005
    python test_eps_8k_plus_av.py --accession 0000851205-26-000009 --verbose
"""

import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

import requests

import html
from dotenv import load_dotenv
from edgar import set_identity, find

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
set_identity("SoulTrader eps-test user@example.com")


# -------- 8-K actual EPS --------

def get_exhibit_99_text(filing) -> str:
    """Get concatenated text from exhibit 99.x parts."""
    text_parts = []
    if hasattr(filing, "exhibits") and filing.exhibits:
        for ex in filing.exhibits:
            if "99." in str(ex).lower():
                try:
                    text_parts.append(ex.text())
                except Exception:
                    pass
    return " ".join(text_parts) if text_parts else ""


def extract_eps_from_xbrl(filing) -> Optional[float]:
    """Try to get diluted EPS from 8-K XBRL (if available). Many 8-Ks have no or minimal XBRL."""
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
            # Prefer duration (quarter), most recent period_end
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


# Regexes for EPS in exhibit 99 text (prefer diluted / quarter EPS)
EPS_PATTERNS = [
    r"(?:diluted|basic)\s+EPS\s+(?:of\s+|were\s+)?\$?\s*([\d,]+\.?\d{2,})",
    r"(?:GAAP\s+)?EPS\s+(?:of\s+|were\s+)?\$?\s*([\d,]+\.?\d{2,})",
    r"earnings\s+per\s+(?:common\s+)?share\s+(?:\(?EPS\)?)?\s+(?:of\s+|were\s+)?\$?\s*([\d,]+\.?\d{2,})",
    r"\$([\d,]+\.?\d{2,})\s+per\s+(?:diluted|basic)?\s*(?:share|diluted share)",
    r"(?:Diluted|diluted)\s+.*?(?:per\s+share|operations).*?\$\s*([\d,]+\.?\d{2,})",
    r"Continuing\s+operations\s+\$\s*([\d,]+\.?\d{2,})",  # Table row for diluted EPS
]
NONGAAP_PATTERNS = [
    r"(?:non[-\s]?GAAP|adjusted)\s+EPS\s+(?:of\s+)?\$?\s*([\d,]+\.?\d{2,})",
]


def _normalize_exhibit_text(raw: str) -> str:
    """Strip HTML and normalize whitespace so regex can match."""
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


def extract_eps_from_text(text: str, verbose: bool = False) -> Optional[float]:
    """Extract one EPS value from exhibit 99 text; prefer diluted/GAAP, then non-GAAP."""
    text = _normalize_exhibit_text(text)
    if not text:
        return None
    # Prefer diluted/GAAP first
    for pattern in EPS_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if abs(val) < 1e6 and val != 0:  # Sanity
                    if verbose:
                        print(f"  [8-K text] EPS ${val:.2f} (pattern: {pattern[:40]}...)")
                    return val
            except ValueError:
                pass
    for pattern in NONGAAP_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if abs(val) < 1e6 and val != 0:
                    if verbose:
                        print(f"  [8-K text] Adjusted/Non-GAAP EPS ${val:.2f}")
                    return val
            except ValueError:
                pass
    return None


def get_actual_eps_from_8k(filing, verbose: bool = False) -> Tuple[Optional[float], str]:
    """
    Get reported EPS from 8-K: try XBRL first, then exhibit 99 text.
    Returns (eps_value, source) where source is "xbrl" or "text" or "none".
    """
    eps = extract_eps_from_xbrl(filing)
    if eps is not None:
        if verbose:
            print(f"  [8-K XBRL] EPS ${eps:.2f}")
        return eps, "xbrl"
    text = get_exhibit_99_text(filing)
    eps = extract_eps_from_text(text, verbose=verbose)
    if eps is not None:
        return eps, "text"
    return None, "none"


# -------- Alpha Vantage estimate --------

def get_av_earnings_for_date(
    ticker: str, report_date: date, verbose: bool = False
) -> Optional[dict]:
    """
    Fetch AV EARNINGS; return the quarterly record whose reportedDate matches report_date.
    Tries report_date then ±1, ±2, ±3 days (filing_date can be UTC vs local or off by a day).
    """
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        if verbose:
            print("  (ALPHAVANTAGE_API_KEY not set)")
        return None
    url = "https://www.alphavantage.co/query"
    params = {"function": "EARNINGS", "symbol": ticker.upper(), "apikey": api_key}
    try:
        time.sleep(0.25)
        resp = requests.get(url, params=params, timeout=15)
        if not resp.ok:
            print(f"  Alpha Vantage returned HTTP {resp.status_code} (e.g. rate limit 429 or 4xx).")
            if verbose and resp.text:
                try:
                    j = resp.json()
                    if "Note" in j:
                        print(f"  Note: {j['Note'][:200]}")
                    elif "Error Message" in j:
                        print(f"  Error: {j['Error Message'][:200]}")
                except Exception:
                    print(f"  Body: {resp.text[:200]}")
            return None
        resp.raise_for_status()
        data = resp.json()
        if "Error Message" in data or "Note" in data or "Information" in data:
            if verbose:
                print(f"  AV API: {list(data.keys())}")
            else:
                print(f"  Alpha Vantage API: no data (Note/Error/Information in response).")
            return None
        quarterly = data.get("quarterlyEarnings") or []
        if verbose and quarterly:
            sample = [r.get("reportedDate") for r in quarterly[:5]]
            print(f"  AV quarterly reportedDates (first 5): {sample}")
        # Try exact, then ±1, ±2, ±3 days
        for delta in (0, -1, 1, -2, 2, -3, 3):
            d = report_date + timedelta(days=delta)
            target = d.isoformat()
            for record in quarterly:
                if record.get("reportedDate") == target:
                    if verbose and delta != 0:
                        print(f"  Matched AV on report_date {delta:+d} day(s): {target}")
                    return record
        return None
    except requests.RequestException as e:
        print(f"  Alpha Vantage request failed: {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"  AV error: {e}")
        return None


def get_ticker_and_date(filing) -> Tuple[Optional[str], Optional[date]]:
    """Resolve ticker and filing date from filing."""
    cik = str(getattr(filing, "cik", "") or "").zfill(10)
    ticker = getattr(filing, "ticker", None)
    if not ticker:
        try:
            from edgar import Company
            company = Company(cik)
            ticker = company.get_ticker()
        except Exception:
            pass
    fd = getattr(filing, "filing_date", None)
    if fd is None:
        report_date = None
    elif isinstance(fd, date):
        report_date = fd
    elif isinstance(fd, str):
        try:
            report_date = date.fromisoformat(fd[:10])
        except ValueError:
            report_date = None
    else:
        report_date = getattr(fd, "date", lambda: None)()
    return ticker, report_date


# -------- Main --------

def run(accession_number: str, verbose: bool = False):
    """Load 8-K by accession; get actual EPS from 8-K, estimate from AV; compute and print surprise."""
    print(f"Looking up filing: {accession_number}")
    try:
        filing = find(accession_number)
    except Exception as e:
        print(f"Error: {e}")
        return
    if not filing:
        print("Filing not found.")
        return

    company = getattr(filing, "company", "N/A")
    print(f"Filing: {company} ({getattr(filing, 'form', 'N/A')})")

    ticker, report_date = get_ticker_and_date(filing)
    if not ticker:
        print("Could not resolve ticker.")
        return
    if not report_date:
        print("Could not resolve filing date.")
        return
    print(f"Ticker: {ticker}  Filing date: {report_date}")
    if verbose:
        raw_fd = getattr(filing, "filing_date", None)
        print(f"  (raw filing_date from edgar: {raw_fd!r})")

    # Actual from 8-K
    actual_eps, source = get_actual_eps_from_8k(filing, verbose=verbose)
    if actual_eps is None:
        print("No EPS found in 8-K (XBRL or exhibit 99 text).")
    else:
        print(f"8-K actual EPS: ${actual_eps:.2f} (source: {source})")

    # Estimate from Alpha Vantage
    av_record = get_av_earnings_for_date(ticker, report_date, verbose=verbose)
    if not av_record:
        print("No Alpha Vantage earnings record for this filing date.")
        if verbose:
            print(f"  (Tried filing_date={report_date} and ±3 days; check AV has {ticker} for that quarter.)")
        if actual_eps is not None:
            print("(We have 8-K actual but no AV estimate to compute surprise.)")
        return

    estimated_eps = av_record.get("estimatedEPS")
    try:
        estimated_eps = float(estimated_eps) if estimated_eps not in (None, "", "None") else None
    except (TypeError, ValueError):
        estimated_eps = None

    reported_av = av_record.get("reportedEPS")
    try:
        reported_av = float(reported_av) if reported_av not in (None, "", "None") else None
    except (TypeError, ValueError):
        reported_av = None

    surprise_av = av_record.get("surprisePercentage")
    try:
        surprise_av = float(surprise_av) if surprise_av not in (None, "", "None") else None
    except (TypeError, ValueError):
        surprise_av = None

    print(f"Alpha Vantage: estimated EPS ${estimated_eps:.2f}" if estimated_eps is not None else "Alpha Vantage: no estimate")
    if reported_av is not None:
        print(f"Alpha Vantage: reported EPS ${reported_av:.2f} (surprise: {surprise_av}%)" if surprise_av is not None else f"Alpha Vantage: reported EPS ${reported_av:.2f}")

    # Compute surprise from 8-K actual + AV estimate (alternative when AV reported not yet in)
    if actual_eps is not None and estimated_eps is not None and estimated_eps != 0:
        surprise_pct = ((actual_eps - estimated_eps) / abs(estimated_eps)) * 100
        print(f"Computed surprise (8-K actual vs AV estimate): {surprise_pct:+.2f}%")
        if verbose and reported_av is not None:
            print(f"(AV reported ${reported_av:.2f} → AV surprise {surprise_av}%)")
    elif actual_eps is not None:
        print("Cannot compute surprise: no AV estimate for this quarter.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EPS from 8-K (XBRL/text) + AV estimate → surprise")
    parser.add_argument("--accession", "-a", required=True, help="8-K accession (e.g. 0001751008-26-000005)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    run(args.accession, verbose=args.verbose)
