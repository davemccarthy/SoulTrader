#!/usr/bin/env python3
"""
Diagnostic script to analyze rejected stocks and compare smoothed vs actual price action.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import ssl
import sys

# Disable SSL verification for yfinance
ssl._create_default_https_context = ssl._create_unverified_context

def get_historical_data_yfinance(symbol, start_date, end_date):
    """Fetch historical data from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        if 'date' not in df.columns and 'datetime' in df.columns:
            df['date'] = df['datetime']
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def calculate_smoothed_and_slope(price_series, half_period):
    """Calculate smoothed price and slope."""
    smoothed = price_series.rolling(window=half_period, min_periods=3, center=True).mean()
    if len(smoothed) < 3:
        return None, None, None
    
    # Slope over last 3 days of smoothed price
    slope = smoothed.iloc[-1] - smoothed.iloc[-3]
    
    # Also calculate recent actual price trend
    recent_actual = price_series.iloc[-5:].values
    up_days = sum(1 for i in range(1, len(recent_actual)) if recent_actual[i] > recent_actual[i-1])
    
    return smoothed, slope, up_days

def analyze_stock(symbol, test_date, lookback_days=100):
    """Analyze a stock's price action around the test date."""
    test_dt = datetime.strptime(test_date, "%Y-%m-%d")
    start_dt = test_dt - timedelta(days=lookback_days)
    end_dt = test_dt + timedelta(days=5)
    
    df = get_historical_data_yfinance(symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    if df.empty or len(df) < 20:
        return None
    
    # Find the row closest to test_date
    df['date'] = pd.to_datetime(df['date']).dt.date
    test_date_only = test_dt.date()
    
    test_row_idx = None
    for i, row_date in enumerate(df['date']):
        if row_date == test_date_only:
            test_row_idx = i
            break
    
    if test_row_idx is None or test_row_idx < 20:
        return None
    
    # Get price series up to test date
    price_series = df.iloc[:test_row_idx+1]['close']
    
    # Estimate half_period (simplified - in real code it's based on wavelet analysis)
    # Use a reasonable default (will vary per stock in reality)
    half_period = 15  # Default, actual would come from wavelet analysis
    
    smoothed, slope, up_days = calculate_smoothed_and_slope(price_series, half_period)
    if smoothed is None:
        return None
    
    # Get recent price action details
    recent_data = df.iloc[test_row_idx-4:test_row_idx+1][['date', 'open', 'high', 'low', 'close']].copy()
    recent_data['price_change'] = recent_data['close'].pct_change() * 100
    recent_data['day_direction'] = recent_data['close'] > recent_data['close'].shift(1)
    
    return {
        'symbol': symbol,
        'test_date': test_date,
        'test_price': float(price_series.iloc[-1]),
        'smoothed_price': float(smoothed.iloc[-1]) if len(smoothed) > 0 else None,
        'slope': float(slope) if slope is not None else None,
        'up_days_last_5': up_days,
        'recent_data': recent_data,
        'smoothed_series': smoothed,
        'price_series': price_series,
    }

def main():
    # Stocks rejected for negative slope on 2025-04-15
    rejected_stocks = [
        'APG',   # slope: -0.04833
        'JCI',   # slope: -0.4404
        'CENX',  # slope: -0.58222
        'SPYV',  # slope: -0.26115
        'CPNG',  # slope: -0.0352
        'GNTX',  # slope: -0.46092
        'PYPL',  # slope: -1.16797
        'AA',    # slope: -1.24257
        'SOFI',  # slope: -0.14875 (was profitable in original backtest!)
    ]
    
    test_date = "2025-04-15"
    
    print("=" * 80)
    print(f"DIAGNOSTIC: Analyzing Rejected Stocks - {test_date}")
    print("=" * 80)
    print()
    
    results = []
    
    for symbol in rejected_stocks:
        print(f"Analyzing {symbol}...")
        result = analyze_stock(symbol, test_date)
        if result:
            results.append(result)
            print(f"  ✓ Data retrieved")
        else:
            print(f"  ✗ Failed to retrieve data")
        print()
    
    # Print detailed analysis
    print("=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    print()
    
    for result in results:
        symbol = result['symbol']
        print(f"📊 {symbol}")
        print(f"   Test Date: {result['test_date']}")
        print(f"   Test Price: ${result['test_price']:.2f}")
        print(f"   Smoothed Price: ${result['smoothed_price']:.2f}" if result['smoothed_price'] else "   Smoothed Price: N/A")
        print(f"   Slope (smoothed[-1] - smoothed[-3]): {result['slope']:.5f}")
        print(f"   Up Days (last 5 days): {result['up_days_last_5']}/4")
        print()
        print("   Recent Price Action (5 days leading up to test date):")
        print("   " + "-" * 70)
        recent = result['recent_data'].copy()
        for idx, row in recent.iterrows():
            direction = "↑" if row['day_direction'] else "↓"
            change_pct = f"{row['price_change']:+.2f}%" if pd.notna(row['price_change']) else "N/A"
            print(f"   {row['date']}: Close=${row['close']:.2f} {direction} ({change_pct})")
        print()
        
        # Compare actual vs smoothed
        price_changes = recent['price_change'].dropna()
        avg_actual_change = price_changes.mean()
        print(f"   Average Daily Change (last 5 days): {avg_actual_change:+.2f}%")
        print(f"   Smoothed Slope: {result['slope']:.5f}")
        print(f"   ⚠️  Mismatch: Actual price {'rising' if avg_actual_change > 0 else 'falling'}, but smoothed slope is {'positive' if result['slope'] > 0 else 'negative'}")
        print()
        print("=" * 80)
        print()

if __name__ == "__main__":
    main()





