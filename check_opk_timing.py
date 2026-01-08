#!/usr/bin/env python
"""Check OPK price history around Dec 8-10 to see buy timing"""
import yfinance as yf
from datetime import datetime, timedelta

symbol = 'OPK'
start_date = datetime(2025, 12, 1)
end_date = datetime(2025, 12, 20)

print(f"Checking {symbol} price history from {start_date.date()} to {end_date.date()}\n")

ticker = yf.Ticker(symbol)
df = ticker.history(
    start=start_date.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    interval='1d'
)

print(f"{'Date':<12} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Volume':<12}")
print("-" * 80)

for idx in df.index:
    date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx.date())
    marker = ""
    if date_str == "2025-12-08":
        marker = " <-- Dec 8 (user says good buy)"
    elif date_str == "2025-12-09":
        marker = " <-- Dec 9 (user says good buy)"
    elif date_str == "2025-12-10":
        marker = " <-- Dec 10 (BUY signal, but user says price too high)"
    
    print(f"{date_str:<12} ${df.loc[idx, 'Open']:<7.2f} ${df.loc[idx, 'High']:<7.2f} "
          f"${df.loc[idx, 'Low']:<7.2f} ${df.loc[idx, 'Close']:<7.2f} "
          f"{int(df.loc[idx, 'Volume']):<12}{marker}")

# Calculate price changes
dec8_price = float(df.loc[df.index[df.index.strftime('%Y-%m-%d') == '2025-12-08'][0], 'Close']) if len(df.index[df.index.strftime('%Y-%m-%d') == '2025-12-08']) > 0 else None
dec9_price = float(df.loc[df.index[df.index.strftime('%Y-%m-%d') == '2025-12-09'][0], 'Close']) if len(df.index[df.index.strftime('%Y-%m-%d') == '2025-12-09']) > 0 else None
dec10_price = float(df.loc[df.index[df.index.strftime('%Y-%m-%d') == '2025-12-10'][0], 'Close']) if len(df.index[df.index.strftime('%Y-%m-%d') == '2025-12-10']) > 0 else None

if dec8_price and dec9_price and dec10_price:
    print("\nPrice changes:")
    print(f"  Dec 8 to Dec 9: {((dec9_price - dec8_price) / dec8_price * 100):+.2f}%")
    print(f"  Dec 9 to Dec 10: {((dec10_price - dec9_price) / dec9_price * 100):+.2f}%")
    print(f"  Dec 8 to Dec 10: {((dec10_price - dec8_price) / dec8_price * 100):+.2f}%")






