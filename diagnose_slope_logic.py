#!/usr/bin/env python3
"""
Analyze the slope calculation logic to understand why so many stocks are rejected.
"""

import numpy as np
import pandas as pd

def analyze_slope_calculation():
    """
    The current slope calculation is:
    slope = smoothed.iloc[-1] - smoothed.iloc[-3]
    
    This compares smoothed[-1] (today) vs smoothed[-3] (3 days ago).
    But smoothed is calculated with a rolling window of half_period (often 10-30 days).
    
    If smoothed uses a 15-day window, then:
    - smoothed[-1] = average of prices[-8 to +7] (centered)
    - smoothed[-3] = average of prices[-11 to +4] (centered)
    
    Since these windows overlap significantly, the difference between them
    is only looking at the difference between the most recent 3 days of the
    smoothed average, not the raw price action.
    """
    
    print("=" * 80)
    print("SLOPE CALCULATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Simulate a stock that just turned up
    # Days 1-10: declining
    # Days 11-15: turning up
    np.random.seed(42)
    declining = np.linspace(100, 90, 10)
    turning_up = 90 + np.array([0.5, 1.0, 2.0, 3.0, 5.0])  # Strong up move
    price_series = pd.Series(np.concatenate([declining, turning_up]))
    
    print("📊 Simulated Price Series (stock turning up at end):")
    print(f"   Days 1-10: Declining from $100 to $90")
    print(f"   Days 11-15: Turning up: {turning_up}")
    print(f"   Last 3 prices: {price_series.iloc[-3:].values}")
    print()
    
    # Test with different half_periods
    for half_period in [10, 15, 20]:
        smoothed = price_series.rolling(window=half_period, min_periods=3, center=True).mean()
        slope = smoothed.iloc[-1] - smoothed.iloc[-3]
        
        # Also check raw price change
        raw_slope = price_series.iloc[-1] - price_series.iloc[-3]
        
        print(f"Half Period: {half_period} days")
        print(f"   Smoothed[-3]: ${smoothed.iloc[-3]:.2f}")
        print(f"   Smoothed[-1]: ${smoothed.iloc[-1]:.2f}")
        print(f"   Smoothed Slope: {slope:.5f}")
        print(f"   Raw Price Slope: {raw_slope:.5f} (actual 3-day change)")
        print(f"   ⚠️  {'Smoothed slope is NEGATIVE even though price is rising!' if slope < 0 else 'Smoothed slope is positive ✓'}")
        print()
    
    print("=" * 80)
    print("PROBLEM IDENTIFIED")
    print("=" * 80)
    print()
    print("The issue: Using a 3-day window on a smoothed series with a 10-30 day smoothing")
    print("window creates significant lag. Even when raw price is turning up, the smoothed")
    print("average can still be declining because:")
    print()
    print("1. The smoothing window includes many older declining days")
    print("2. Only the most recent 3 days matter for the slope, but they're heavily")
    print("   influenced by the older declining days still in the smoothing window")
    print("3. A 3-day difference on a 15-day smoothed average has a lot of lag")
    print()
    print("SOLUTION OPTIONS:")
    print()
    print("Option 1: Use a longer window for slope calculation")
    print("   slope = smoothed.iloc[-1] - smoothed.iloc[-5]  # 5 days instead of 3")
    print()
    print("Option 2: Normalize slope by wave_range")
    print("   slope_pct = slope / wave_range  # Makes it relative to volatility")
    print()
    print("Option 3: Use raw price slope instead of smoothed")
    print("   slope = price_series.iloc[-1] - price_series.iloc[-3]")
    print()
    print("Option 4: Check if smoothed is still declining but actual price is rising")
    print("   smoothed_slope = smoothed.iloc[-1] - smoothed.iloc[-3]")
    print("   raw_slope = price_series.iloc[-1] - price_series.iloc[-3]")
    print("   if raw_slope > 0 and smoothed_slope > -threshold: accept")
    print()

if __name__ == "__main__":
    analyze_slope_calculation()





