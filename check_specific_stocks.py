#!/usr/bin/env python3
"""Check specific stocks (NE, NVDL) to see exactly why they're filtered on specific dates."""

import sys
sys.path.insert(0, '/Users/davidmccarthy/Development/CursorAI/Django/soultrader')

from test_oscilla import (
    generate_trading_candidates, get_historical_data_yfinance,
    MIN_PRICE, MAX_PRICE, MIN_AVG_VOLUME, REL_VOLUME_MIN, REL_VOLUME_MAX,
    LOOKBACK_DAYS, MIN_RR
)
import pandas as pd

# Stocks and dates to check
stocks_to_check = [
    ("NE", "2025-04-08"),  # First weekly date after 2025-04-01 entry
    ("NE", "2025-04-15"),  # Second weekly date
    ("NVDL", "2025-04-23"),  # Date after 2025-04-15 stop loss
]

print("=" * 80)
print("DIAGNOSTIC: Checking why specific stocks are filtered")
print("=" * 80)
print()

for ticker, test_date in stocks_to_check:
    print(f"\n{'='*80}")
    print(f"Checking {ticker} on {test_date}")
    print(f"{'='*80}")
    print()
    
    # Run candidate generation with verbose to see diagnostics
    print(f"Running candidate generation with diagnostics...")
    print()
    
    try:
        candidates_df = generate_trading_candidates(
            reference_date=test_date,
            max_stocks=1000,  # Check many stocks to ensure we see it if it passes
            verbose=True
        )
        
        # Check if ticker appears (handle empty dataframe)
        if candidates_df.empty:
            print(f"\n✗ No candidates found for {test_date}")
            print("  (Likely needs Polygon API key or no data available)")
            continue
            
        ticker_found = candidates_df[candidates_df['ticker'] == ticker] if 'ticker' in candidates_df.columns else pd.DataFrame()
        
        if not ticker_found.empty:
            print(f"\n✓✓✓ {ticker} WAS SELECTED on {test_date}! ✓✓✓")
            print()
            print("Metrics:")
            for col in ticker_found.columns:
                print(f"  {col}: {ticker_found[col].iloc[0]}")
        else:
            print(f"\n✗ {ticker} was NOT selected on {test_date}")
            print()
            print("This means it was filtered out in one of these stages:")
            print("1. Initial price/volume filters (build_candidates)")
            print("2. Wavelet analysis filters (wavelet_trade_engine)")
            print()
            print("Check the verbose output above for the specific filter that rejected it.")
            print("(Look for lines starting with the ticker symbol)")
        
    except Exception as e:
        print(f"Error checking {ticker}: {e}")
        import traceback
        traceback.print_exc()
    
    print()

print("=" * 80)
print("Summary")
print("=" * 80)
print("If stocks were filtered, the verbose output above shows the exact reason.")
print("Common reasons:")
print("  - Insufficient historical data")
print("  - Avg volume < MIN_AVG_VOLUME")
print("  - Relative volume outside [REL_VOLUME_MIN, REL_VOLUME_MAX]")
print("  - Wave position < -3.0 or > 0.35")
print("  - R:R ratio < MIN_RR")
print("  - Invalid stop calculation")

