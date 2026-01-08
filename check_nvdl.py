#!/usr/bin/env python3
"""Check if NVDL would qualify on 2025-04-23 after hitting stop loss on 2025-04-15."""

import sys
sys.path.insert(0, '/Users/davidmccarthy/Development/CursorAI/Django/soultrader')

from test_oscilla import generate_trading_candidates, wavelet_trade_engine, get_historical_data_yfinance
from datetime import datetime, timedelta
import pandas as pd

# NVDL hit stop loss on 2025-04-15 (-13.48% in 1 day)
# Check if it would qualify on 2025-04-23 (8 days later)
test_date = "2025-04-23"
entry_date = "2025-04-15"

print("=" * 80)
print(f"Checking NVDL re-entry on {test_date} (after stop loss on {entry_date})")
print("=" * 80)
print()

# Direct wavelet analysis (more reliable than full candidate generation)
print("Running direct wavelet analysis on NVDL...")
print()

try:
    from test_oscilla import MIN_RR, MIN_PRICE, MAX_PRICE, MIN_AVG_VOLUME
    ref_dt = datetime.strptime(test_date, "%Y-%m-%d")
    start_date = (ref_dt - timedelta(days=150)).strftime("%Y-%m-%d")
    
    print(f"Fetching historical data from {start_date} to {test_date}...")
    df_price = get_historical_data_yfinance("NVDL", start_date, test_date)
    
    if df_price.empty:
        print(f"✗ No historical data available")
        print("  (This could be due to certificate issues or data availability)")
    elif len(df_price) < 64:
        print(f"✗ Not enough historical data: {len(df_price)} points (need 64)")
    else:
        print(f"✓ Got {len(df_price)} data points for analysis")
        
        # Check basic price filter first
        current_price = df_price['close'].iloc[-1]
        print(f"\nCurrent price on {test_date}: ${current_price:.2f}")
        
        if MIN_PRICE <= current_price <= MAX_PRICE:
            print(f"✓ Price ${current_price:.2f} passes price filter (${MIN_PRICE:.2f} - ${MAX_PRICE:.2f})")
        else:
            print(f"✗ Price ${current_price:.2f} FAILS price filter (${MIN_PRICE:.2f} - ${MAX_PRICE:.2f})")
        
        print()
        print("Running wavelet analysis...")
        print("-" * 80)
        
        wave_result = wavelet_trade_engine(df_price["close"], min_rr=MIN_RR)
        
        if wave_result.get("accepted", False):
            print("\n✓✓✓ NVDL would be ACCEPTED by wavelet analysis! ✓✓✓")
            print()
            print("Metrics:")
            print(f"  Wave position: {wave_result.get('wave_position', 'N/A')}")
            print(f"  Consistency: {wave_result.get('consistency', 'N/A')}")
            print(f"  R:R ratio: {wave_result.get('reward_risk', 'N/A')}")
            print(f"  Dominant period: {wave_result.get('dominant_period_days', 'N/A')} days")
            print()
            print("Trade levels:")
            print(f"  Buy: ${wave_result.get('buy', 0):.2f}")
            print(f"  Stop: ${wave_result.get('stop', 0):.2f}")
            print(f"  Target: ${wave_result.get('target', 0):.2f}")
            print()
            print("This supports allowing re-entry after stop loss!")
        else:
            print("\n✗ NVDL would be REJECTED by wavelet analysis")
            print()
            print(f"Reason: {wave_result.get('reason', 'Unknown')}")
            
            if 'wave_position' in wave_result:
                print(f"Wave position: {wave_result.get('wave_position', 'N/A')}")
            if 'consistency' in wave_result:
                print(f"Consistency: {wave_result.get('consistency', 'N/A')}")
            if 'rr' in wave_result:
                print(f"R:R ratio: {wave_result.get('rr', 'N/A')} (required: {MIN_RR})")
            
            if 'log' in wave_result:
                print()
                print("Analysis log:")
                for log_msg in wave_result['log']:
                    print(f"  {log_msg}")

except Exception as e:
    print(f"Error in analysis: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("Implications:")
print("=" * 80)
print("If NVDL would have qualified on 2025-04-23 and made a significant gain,")
print("this supports allowing re-entry of stocks after stop loss.")
print()
print("Considerations:")
print("1. Add cooldown period (e.g., 7+ days after stop loss)")
print("2. Track previously traded stocks in backtest")
print("3. Allow re-entry if stock still passes all filters")
print("4. Evaluate if this improves overall backtest performance")

