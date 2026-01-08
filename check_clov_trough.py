#!/usr/bin/env python
"""Check if CLOV was actually at a trough on 2025-12-10"""
import sys
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from test_cyclical_patterns import find_local_extrema, calculate_support_resistance

symbol = 'CLOV'
analysis_date = datetime(2025, 12, 10)

# Get historical data (same as test script)
lookback_days = 180
window = 20
start_date = analysis_date - timedelta(days=lookback_days + 30)
end_date = analysis_date + timedelta(days=10)  # Some forward data

print(f"Analyzing {symbol} on {analysis_date.strftime('%Y-%m-%d')}\n")

ticker = yf.Ticker(symbol)
df = ticker.history(
    start=start_date.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    interval='1d'
)

if df.empty:
    print("No data found")
    sys.exit(1)

# Find the analysis date in the dataframe
analysis_price = None
analysis_idx = None
for i, idx in enumerate(df.index):
    if hasattr(idx, 'date'):
        idx_date = idx.date()
    elif hasattr(idx, 'to_pydatetime'):
        idx_date = idx.to_pydatetime().date()
    else:
        continue
    
    if idx_date == analysis_date.date():
        analysis_idx = i
        analysis_price = float(df['Close'].iloc[i])
        break

if analysis_price is None:
    print(f"Could not find price data for {analysis_date.date()}")
    sys.exit(1)

# Find troughs
peaks, troughs = find_local_extrema(df, window=window)
support, resistance = calculate_support_resistance(df, peaks, troughs)

print(f"Price on {analysis_date.date()}: ${analysis_price:.2f}")
print(f"Support: ${support:.2f}")
print(f"Resistance: ${resistance:.2f}")
print(f"Price range: ${resistance - support:.2f}\n")

# Calculate position_pct
price_range = resistance - support
position_pct = (analysis_price - support) / price_range if price_range > 0 else 0.5
print(f"Position %: {position_pct * 100:.1f}% (0% = at support, 100% = at resistance)\n")

# Check if actually at a trough
print("Recent troughs (within 30 days of analysis date):")
troughs_near_date = []
for trough_idx in troughs:
    days_from_date = abs(trough_idx - analysis_idx)
    if days_from_date <= 30:
        trough_date = df.index[trough_idx]
        trough_low = float(df['Low'].iloc[trough_idx])
        trough_close = float(df['Close'].iloc[trough_idx])
        troughs_near_date.append({
            'idx': trough_idx,
            'date': trough_date,
            'low': trough_low,
            'close': trough_close,
            'days_from_date': days_from_date,
            'price_diff_pct': ((analysis_price - trough_low) / trough_low) * 100
        })

troughs_near_date.sort(key=lambda x: x['days_from_date'])

if troughs_near_date:
    for t in troughs_near_date[:5]:  # Show 5 closest
        date_str = t['date'].strftime('%Y-%m-%d') if hasattr(t['date'], 'strftime') else str(t['date'])
        print(f"  {date_str}: Low ${t['low']:.2f}, Close ${t['close']:.2f}, {t['days_from_date']} days away, {t['price_diff_pct']:+.1f}% from analysis price")
    
    # Check if within 2% of nearest trough
    nearest = troughs_near_date[0]
    if abs(nearest['price_diff_pct']) <= 2:
        print(f"\n✅ Analysis price is within 2% of nearest trough ({nearest['days_from_date']} days away)")
    else:
        print(f"\n❌ Analysis price is {nearest['price_diff_pct']:+.1f}% from nearest trough ({nearest['days_from_date']} days away)")
        print(f"   Analysis price: ${analysis_price:.2f}")
        print(f"   Nearest trough low: ${nearest['low']:.2f}")
else:
    print("  No troughs found within 30 days")

print("\n" + "="*80)
print("Price data around analysis date (±10 days):")
print("="*80)
print(f"{'Date':<12} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Volume':<12}")
print("-"*80)

start_show = max(0, analysis_idx - 10)
end_show = min(len(df), analysis_idx + 11)

for i in range(start_show, end_show):
    idx = df.index[i]
    date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx.date())
    marker = " <-- ANALYSIS DATE" if i == analysis_idx else ""
    print(f"{date_str:<12} ${df['Open'].iloc[i]:<7.2f} ${df['High'].iloc[i]:<7.2f} "
          f"${df['Low'].iloc[i]:<7.2f} ${df['Close'].iloc[i]:<7.2f} {int(df['Volume'].iloc[i]):<12}{marker}")






