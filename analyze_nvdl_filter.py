#!/usr/bin/env python3
"""Analyze why NVDL wasn't picked up on 2025-04-23 despite being profitable."""

import sys
sys.path.insert(0, '/Users/davidmccarthy/Development/CursorAI/Django/soultrader')

from test_oscilla import (
    build_candidates, wavelet_trade_engine, get_historical_data_yfinance,
    MIN_PRICE, MAX_PRICE, MIN_AVG_VOLUME, REL_VOLUME_MIN, REL_VOLUME_MAX,
    LOOKBACK_DAYS, MIN_RR, MAX_WAVE_POSITION, MIN_CONSISTENCY
)
from datetime import datetime, timedelta
import pandas as pd

test_date = "2025-04-23"
entry_date = "2025-04-15"

print("=" * 80)
print(f"Analyzing why NVDL wasn't selected on {test_date}")
print("=" * 80)
print()

# Check each filter stage
print("STAGE 1: Initial Candidate Filters (build_candidates)")
print("-" * 80)

# Note: Can't use build_candidates without Polygon API, so we'll check manually
print("(Skipping Polygon API check - will check filters manually)")
print()

print("STAGE 2: Individual Filter Checks")
print("-" * 80)
print()

try:
    # Get historical data to check current metrics
    ref_dt = datetime.strptime(test_date, "%Y-%m-%d")
    start_date = (ref_dt - timedelta(days=LOOKBACK_DAYS * 2)).strftime("%Y-%m-%d")
    end_date = (ref_dt + timedelta(days=5)).strftime("%Y-%m-%d")
    
    print(f"Fetching data from {start_date} to {end_date}...")
    df_price = get_historical_data_yfinance("NVDL", start_date, end_date)
    
    if df_price.empty:
        print("✗ No price data available")
        sys.exit(1)
    
    # Filter 1: Price range
    current_price = df_price['close'].iloc[-1]
    print(f"\n1. Price Filter:")
    print(f"   Current price: ${current_price:.2f}")
    print(f"   Required range: ${MIN_PRICE:.2f} - ${MAX_PRICE:.2f}")
    if MIN_PRICE <= current_price <= MAX_PRICE:
        print(f"   ✓ PASSES")
    else:
        print(f"   ✗ FAILS")
        print(f"   → NVDL excluded at this stage")
        sys.exit(0)
    
    # Filter 2: Average volume
    df_recent = df_price.tail(LOOKBACK_DAYS) if len(df_price) >= LOOKBACK_DAYS else df_price
    avg_volume = df_recent['volume'].mean()
    today_volume = df_price['volume'].iloc[-1]
    rel_volume = today_volume / avg_volume if avg_volume > 0 else 0
    
    print(f"\n2. Volume Filters:")
    print(f"   Average volume (last {LOOKBACK_DAYS} days): {avg_volume:,.0f}")
    print(f"   Today's volume: {today_volume:,.0f}")
    print(f"   Relative volume: {rel_volume:.2f}")
    print(f"   Required avg volume: >= {MIN_AVG_VOLUME:,}")
    print(f"   Required rel volume: {REL_VOLUME_MIN:.1f} - {REL_VOLUME_MAX:.1f}")
    
    volume_pass = avg_volume >= MIN_AVG_VOLUME
    rel_vol_pass = REL_VOLUME_MIN <= rel_volume <= REL_VOLUME_MAX
    
    if volume_pass:
        print(f"   ✓ Avg volume PASSES")
    else:
        print(f"   ✗ Avg volume FAILS")
        print(f"   → NVDL excluded at this stage")
    
    if rel_vol_pass:
        print(f"   ✓ Rel volume PASSES")
    else:
        print(f"   ✗ Rel volume FAILS")
        print(f"   → NVDL excluded at this stage")
    
    if not (volume_pass and rel_vol_pass):
        sys.exit(0)
    
    print(f"\n3. Wavelet Analysis Filters:")
    print("-" * 80)
    
    # Get enough data for wavelet (need ~120 calendar days)
    wavelet_start = (ref_dt - timedelta(days=150)).strftime("%Y-%m-%d")
    df_wavelet = get_historical_data_yfinance("NVDL", wavelet_start, test_date)
    
    if df_wavelet.empty or len(df_wavelet) < 64:
        print(f"✗ Insufficient data: {len(df_wavelet)} points (need 64)")
        sys.exit(0)
    
    print(f"   Got {len(df_wavelet)} data points for wavelet analysis")
    print()
    
    wave_result = wavelet_trade_engine(df_wavelet["close"], min_rr=MIN_RR)
    
    if wave_result.get("accepted", False):
        print("   ✓✓✓ ACCEPTED by wavelet analysis")
        print()
        print(f"   Wave position: {wave_result.get('wave_position', 'N/A')}")
        print(f"   Consistency: {wave_result.get('consistency', 'N/A')}")
        print(f"   R:R ratio: {wave_result.get('reward_risk', 'N/A')}")
        print(f"   Buy: ${wave_result.get('buy', 0):.2f}")
        print(f"   Stop: ${wave_result.get('stop', 0):.2f}")
        print(f"   Target: ${wave_result.get('target', 0):.2f}")
        print()
        print("   → NVDL SHOULD have been selected! Check Polygon API or candidate generation.")
    else:
        print("   ✗ REJECTED by wavelet analysis")
        print()
        reason = wave_result.get('reason', 'Unknown')
        print(f"   Reason: {reason}")
        print()
        
        # Show all available metrics
        if 'wave_position' in wave_result:
            wp = wave_result['wave_position']
            print(f"   Wave position: {wp}")
            if MAX_WAVE_POSITION > -999 and wp < MAX_WAVE_POSITION:
                print(f"   ✗ FAILS: wave_position {wp} < MAX_WAVE_POSITION {MAX_WAVE_POSITION}")
            elif wp > 0.35:
                print(f"   ✗ FAILS: wave_position {wp} > 0.35 (too close to peak)")
            else:
                print(f"   ✓ PASSES wave position filter")
        
        if 'consistency' in wave_result:
            cons = wave_result['consistency']
            print(f"   Consistency: {cons}")
            if MIN_CONSISTENCY > 0 and cons < MIN_CONSISTENCY:
                print(f"   ✗ FAILS: consistency {cons} < MIN_CONSISTENCY {MIN_CONSISTENCY}")
            else:
                print(f"   ✓ PASSES consistency filter")
        
        if 'rr' in wave_result:
            rr = wave_result['rr']
            print(f"   R:R ratio: {rr}")
            if rr < MIN_RR:
                print(f"   ✗ FAILS: R:R {rr:.2f} < MIN_RR {MIN_RR}")
            else:
                print(f"   ✓ PASSES R:R filter")
        
        if 'log' in wave_result:
            print()
            print("   Analysis log:")
            for log_msg in wave_result['log']:
                print(f"     {log_msg}")
        
        print()
        print(f"   → NVDL excluded by: {reason}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("Summary:")
print("=" * 80)
print("If NVDL would have been profitable on 2025-04-23 but wasn't selected,")
print("this indicates a filter is excluding profitable opportunities.")
print()
print("Possible solutions:")
print("1. Relax the filter that's excluding it (if it's too strict)")
print("2. Add re-entry logic that uses different/adjusted criteria")
print("3. Track that this was a previously traded stock and apply cooldown logic")































