#!/usr/bin/env python
"""Quick script to check JEPI price for 2025-06-04"""

import yfinance as yf
from datetime import datetime

ticker = "JEPI"
check_date = "2025-06-04"

# Get historical data (default - adjusted prices)
ticker_obj = yf.Ticker(ticker)
hist = ticker_obj.history(start="2025-06-01", end="2025-06-05")

print(f"\nChecking {ticker} for {check_date}")
print("=" * 60)

if not hist.empty:
    # Find the row for 2025-06-04
    target_date = datetime.strptime(check_date, "%Y-%m-%d").date()
    
    for idx, row in hist.iterrows():
        if idx.date() == target_date:
            print(f"\nDate: {idx.date()}")
            print(f"Close (adjusted): ${row['Close']:.2f}")
            print(f"High (adjusted): ${row['High']:.2f}")
            print(f"Low (adjusted): ${row['Low']:.2f}")
            print(f"Volume: {row['Volume']:,.0f}")
            break
    else:
        print(f"\n⚠️  No data found for {check_date}")
        print("\nAvailable dates:")
        for idx in hist.index:
            print(f"  {idx.date()}: Close = ${hist.loc[idx, 'Close']:.2f}")
else:
    print("❌ No historical data returned")

# Note: yfinance returns adjusted prices by default
# The 'adjusted' parameter is not available in this yfinance version
print("\n" + "=" * 60)
print("Note: yfinance returns adjusted prices by default")
print("(The 'adjusted' parameter is not available in this yfinance version)")
print("To get unadjusted prices, you would need to use a different data source")

print("\n" + "=" * 60)

