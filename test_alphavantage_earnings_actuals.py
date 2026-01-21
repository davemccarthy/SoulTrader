"""
Test script for Alpha Vantage EARNINGS endpoint (actuals vs estimates).

Usage:
    python test_alphavantage_earnings_actuals.py --ticker IBM [--api-key YOUR_KEY] [--fiscal-year 2025] [--fiscal-quarter 3]
"""

import os
import sys
import requests
import json
import time
import argparse
from datetime import datetime, date
from typing import Optional, Dict, List


def is_pending_earnings(earnings: Dict) -> bool:
    """
    Check if earnings data appears to be pending (not yet populated).
    
    Alpha Vantage may show reportedEPS as 0 or None before actual earnings
    are released, even if reportedDate is today. This can cause false
    "-100% miss" signals.
    
    Returns:
        True if earnings appear to be pending/placeholder data
    """
    reported_eps = earnings.get("reportedEPS")
    reported_date_str = earnings.get("reportedDate")
    estimated_eps = earnings.get("estimatedEPS")
    
    # Check if reportedEPS is 0, None, empty, or "None"
    eps_is_zero_or_missing = (
        reported_eps is None or
        reported_eps == "" or
        reported_eps == "None" or
        (isinstance(reported_eps, str) and reported_eps.strip() == "0") or
        (isinstance(reported_eps, (int, float)) and reported_eps == 0)
    )
    
    if not eps_is_zero_or_missing:
        return False
    
    # Check if there's an estimate (suggests we expect real data)
    has_estimate = estimated_eps and estimated_eps != "None" and estimated_eps != "0"
    
    if not has_estimate:
        return False  # No estimate, so 0 might be legitimate
    
    # Check if reportedDate is recent (within last 7 days)
    if reported_date_str:
        try:
            reported_date = datetime.strptime(reported_date_str, "%Y-%m-%d").date()
            days_ago = (date.today() - reported_date).days
            if days_ago <= 7:
                return True  # Recent report date + zero EPS = likely pending
        except:
            pass
    
    return False


def get_earnings_actuals(ticker: str, api_key: str, debug: bool = True) -> Optional[Dict]:
    """
    Get earnings actuals vs estimates from Alpha Vantage EARNINGS endpoint.
    
    Args:
        ticker: Stock ticker symbol
        api_key: Alpha Vantage API key
        debug: Print debug information
        
    Returns:
        Dictionary with earnings data, or None if error
    """
    base_url = "https://www.alphavantage.co/query"
    
    params = {
        "function": "EARNINGS",
        "symbol": ticker.upper(),
        "apikey": api_key
    }
    
    if debug:
        print(f"\n{'='*80}")
        print(f"Testing Alpha Vantage EARNINGS (actuals vs estimates) for {ticker}")
        print(f"{'='*80}")
    
    try:
        time.sleep(0.2)  # Rate limiting
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            print(f"❌ API Error: {data['Error Message']}")
            return None
        
        if "Note" in data:
            print(f"⚠️  API Note: {data['Note']}")
            return None
        
        if "Information" in data:
            print(f"ℹ️  API Information: {data['Information']}")
            return None
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def find_matching_quarterly_earnings(data: Dict, fiscal_year: int, fiscal_quarter: int, debug: bool = True) -> Optional[Dict]:
    """
    Find quarterly earnings matching a specific fiscal period.
    
    Args:
        data: Raw API response
        fiscal_year: Fiscal year to match
        fiscal_quarter: Fiscal quarter to match (1-4)
        debug: Print matching logic
        
    Returns:
        Matching earnings record, or None
    """
    if not data or "quarterlyEarnings" not in data:
        return None
    
    quarterly = data["quarterlyEarnings"]
    
    if debug:
        print(f"\n{'='*80}")
        print(f"Searching for FY{fiscal_year} Q{fiscal_quarter} earnings")
        print(f"{'='*80}")
        print(f"Found {len(quarterly)} quarterly earnings records")
    
    for i, earnings in enumerate(quarterly):
        date_str = earnings.get("fiscalDateEnding")
        if not date_str:
            continue
        
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            est_year = date_obj.year
            est_month = date_obj.month
            est_quarter = (est_month - 1) // 3 + 1
            
            if debug and i < 5:
                reported = earnings.get("reportedEPS", "N/A")
                estimated = earnings.get("estimatedEPS", "N/A")
                surprise = earnings.get("surprisePercentage", "N/A")
                print(f"Record {i+1}: {date_str} (FY{est_year} Q{est_quarter}) - Reported: ${reported}, Est: ${estimated}, Surprise: {surprise}%")
            
            if est_year == fiscal_year and est_quarter == fiscal_quarter:
                if debug:
                    print(f"\n✓ Found match!")
                return earnings
        except Exception as e:
            if debug:
                print(f"Record {i+1}: Error parsing date '{date_str}': {e}")
            continue
    
    if debug:
        print(f"⚠️  No matching earnings found")
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Test Alpha Vantage EARNINGS endpoint")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument("--api-key", type=str, help="Alpha Vantage API key (or set ALPHAVANTAGE_API_KEY env var)")
    parser.add_argument("--fiscal-year", type=int, help="Fiscal year to search for")
    parser.add_argument("--fiscal-quarter", type=int, choices=[1, 2, 3, 4], help="Fiscal quarter to search for")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("❌ Error: API key required")
        sys.exit(1)
    
    # Test the endpoint
    data = get_earnings_actuals(args.ticker, api_key, debug=not args.quiet)
    
    if not data:
        print("\n❌ Failed to get earnings data")
        sys.exit(1)
    
    # Show structure
    if not args.quiet:
        print(f"\nResponse keys: {list(data.keys())}")
    
    # Show quarterly earnings summary
    if "quarterlyEarnings" in data:
        quarterly = data["quarterlyEarnings"]
        print(f"\n{'='*80}")
        print(f"QUARTERLY EARNINGS ({len(quarterly)} records):")
        print(f"{'='*80}")
        
        # Show most recent 10
        for i, earnings in enumerate(quarterly[:10], 1):
            date_str = earnings.get("fiscalDateEnding", "N/A")
            reported = earnings.get("reportedEPS", "N/A")
            estimated = earnings.get("estimatedEPS", "N/A")
            surprise = earnings.get("surprisePercentage", "N/A")
            reported_date = earnings.get("reportedDate", "N/A")
            
            # Check for pending data
            if is_pending_earnings(earnings):
                print(f"\n{i}. {date_str} (Reported: {reported_date})")
                print(f"   ╔════════════════════════════════════════════════════════════╗")
                print(f"   ║  ⚠️  WARNING: EARNINGS DATA APPEARS PENDING!               ║")
                print(f"   ║  Reported EPS is 0/null but estimate exists.              ║")
                print(f"   ║  This is likely placeholder data - NOT a real miss!       ║")
                print(f"   ║  Wait for Alpha Vantage to update with actual results.    ║")
                print(f"   ╚════════════════════════════════════════════════════════════╝")
                print(f"   Reported EPS: ${reported} (PENDING)")
                print(f"   Estimated EPS: ${estimated}")
                print(f"   Surprise: {surprise}% (UNRELIABLE)")
            else:
                print(f"\n{i}. {date_str} (Reported: {reported_date})")
                print(f"   Reported EPS: ${reported}")
                print(f"   Estimated EPS: ${estimated}")
                print(f"   Surprise: {surprise}%")
    
    # If fiscal period specified, find matching record
    if args.fiscal_year and args.fiscal_quarter:
        matching = find_matching_quarterly_earnings(
            data,
            args.fiscal_year,
            args.fiscal_quarter,
            debug=not args.quiet
        )
        
        if matching:
            print(f"\n{'='*80}")
            print(f"MATCHING EARNINGS:")
            print(f"{'='*80}")
            print(json.dumps(matching, indent=2))
            
            # Calculate beat/miss
            reported = matching.get("reportedEPS")
            estimated = matching.get("estimatedEPS")
            surprise_pct = matching.get("surprisePercentage")
            
            if reported and estimated:
                try:
                    reported_float = float(reported)
                    estimated_float = float(estimated)
                    beat_miss = ((reported_float - estimated_float) / estimated_float * 100) if estimated_float != 0 else 0
                    print(f"\n📊 Summary:")
                    print(f"   Reported: ${reported_float:.2f}")
                    print(f"   Estimated: ${estimated_float:.2f}")
                    print(f"   Beat/Miss: {beat_miss:+.2f}%")
                    if surprise_pct:
                        print(f"   Surprise: {surprise_pct}%")
                except:
                    pass


if __name__ == "__main__":
    main()
