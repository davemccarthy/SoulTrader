#!/usr/bin/env python
"""
Grab today's most undervalued stocks from Yahoo Finance screener
and save to a file for tomorrow's comparison.
"""

import sys
from datetime import datetime

try:
    from yahooquery import Screener
except ImportError:
    print("yahooquery not installed. Install with: pip install yahooquery")
    sys.exit(1)

def grab_undervalued_stocks():
    """Fetch undervalued growth stocks from Yahoo Finance screener."""
    screener = Screener()
    screener_id = 'undervalued_growth_stocks'
    
    print(f"Fetching '{screener_id}' screener from Yahoo Finance...")
    
    try:
        # Get all available stocks (typically 176 total)
        result = screener.get_screeners([screener_id], count=200)
        
        if screener_id not in result:
            print(f"Error: Screener '{screener_id}' not found in results")
            return None
        
        screener_data = result[screener_id]
        quotes = screener_data.get('quotes', [])
        
        if not quotes:
            print("No stocks found in screener results")
            return None
        
        print(f"✓ Retrieved {len(quotes)} stocks")
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"yahooquery_stocks_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            # Header
            f.write(f"Symbol,Name,Price\n")
            
            # Write each stock
            for quote in quotes:
                symbol = quote.get('symbol', 'N/A')
                name = quote.get('shortName') or quote.get('longName', 'N/A')
                price = quote.get('regularMarketPrice', 'N/A')
                
                # Clean name (remove commas)
                name = name.replace(',', '')
                
                # Format price
                if isinstance(price, (int, float)):
                    price_str = f"{price:.2f}"
                else:
                    price_str = str(price)
                
                f.write(f"{symbol},{name},{price_str}\n")
        
        print(f"✓ Saved {len(quotes)} stocks to {filename}")
        print(f"\nTop 10 stocks:")
        for i, quote in enumerate(quotes[:10], 1):
            symbol = quote.get('symbol', 'N/A')
            name = quote.get('shortName') or quote.get('longName', 'N/A')
            price = quote.get('regularMarketPrice', 'N/A')
            price_str = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
            print(f"  {i:2d}. {symbol:<6} {name[:45]:<45} {price_str:>10}")
        
        return filename
        
    except Exception as e:
        print(f"Error fetching screener: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*80)
    print("YAHOO FINANCE SCREENER - UNDERVALUED GROWTH STOCKS")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    filename = grab_undervalued_stocks()
    
    if filename:
        print(f"\n{'='*80}")
        print(f"SUCCESS: Snapshot saved to {filename}")
        print(f"Run tomorrow: python compare_yahooquery_prices.py {filename}")
        print(f"{'='*80}")
    else:
        print("\nFailed to grab stocks")

if __name__ == "__main__":
    main()







