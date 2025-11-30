#!/usr/bin/env python
"""
Compare yesterday's top 10 yahooquery screener results with today's prices.
"""

import csv
import sys
import yfinance as yf

def load_top10_snapshot(filename):
    """Load top 10 stocks from yesterday's snapshot."""
    stocks = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if count >= 10:
                break
            try:
                symbol = row['Symbol'].strip()
                name = row['Name'].strip()
                yesterday_price = float(row['Price'].strip())
                stocks.append({
                    'symbol': symbol,
                    'name': name,
                    'yesterday_price': yesterday_price
                })
                count += 1
            except (ValueError, KeyError) as e:
                print(f"Error parsing row: {row}, error: {e}")
                continue
    return stocks

def fetch_today_prices(stocks):
    """Fetch today's prices for the stocks."""
    symbols = [s['symbol'] for s in stocks]
    
    print(f"Fetching prices for top 10 stocks: {', '.join(symbols)}")
    
    try:
        tickers = yf.Tickers(" ".join(symbols))
        for stock in stocks:
            try:
                ticker = tickers.tickers[stock['symbol']]
                info = ticker.fast_info
                today_price = info.get('lastPrice') or info.get('regularMarketPrice')
                
                if today_price:
                    stock['today_price'] = float(today_price)
                    stock['change'] = stock['today_price'] - stock['yesterday_price']
                    stock['change_pct'] = (stock['change'] / stock['yesterday_price']) * 100 if stock['yesterday_price'] > 0 else 0
                else:
                    stock['today_price'] = None
                    stock['change'] = None
                    stock['change_pct'] = None
                    print(f"  Warning: No price found for {stock['symbol']}")
            except Exception as e:
                print(f"  Error fetching {stock['symbol']}: {e}")
                stock['today_price'] = None
                stock['change'] = None
                stock['change_pct'] = None
    except Exception as e:
        print(f"Error fetching prices: {e}")
    
    return stocks

def analyze_results(stocks):
    """Analyze the price changes for top 10."""
    valid_stocks = [s for s in stocks if s.get('today_price') is not None]
    
    if not valid_stocks:
        print("No valid price comparisons found")
        return
    
    total_change = sum(s['change'] for s in valid_stocks)
    total_change_pct = sum(s['change_pct'] for s in valid_stocks)
    avg_change = total_change / len(valid_stocks)
    avg_change_pct = total_change_pct / len(valid_stocks)
    
    gains = [s for s in valid_stocks if s['change'] > 0]
    losses = [s for s in valid_stocks if s['change'] < 0]
    unchanged = [s for s in valid_stocks if s['change'] == 0]
    
    print("\n" + "="*80)
    print("TOP 10 YAHOOQUERY PRICE COMPARISON RESULTS")
    print("="*80)
    print(f"\nTotal stocks analyzed: {len(valid_stocks)}")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Average change: ${avg_change:.4f} ({avg_change_pct:+.2f}%)")
    print(f"  Total change: ${total_change:.2f}")
    
    print(f"\n📈 GAINS:")
    print(f"  Stocks up: {len(gains)} ({len(gains)/len(valid_stocks)*100:.1f}%)")
    if gains:
        avg_gain = sum(s['change'] for s in gains) / len(gains)
        avg_gain_pct = sum(s['change_pct'] for s in gains) / len(gains)
        print(f"  Average gain: ${avg_gain:.4f} ({avg_gain_pct:+.2f}%)")
        best = max(gains, key=lambda x: x['change_pct'])
        print(f"  Best performer: {best['symbol']} ({best['change_pct']:+.2f}%, ${best['change']:+.4f})")
    
    print(f"\n📉 LOSSES:")
    print(f"  Stocks down: {len(losses)} ({len(losses)/len(valid_stocks)*100:.1f}%)")
    if losses:
        avg_loss = sum(s['change'] for s in losses) / len(losses)
        avg_loss_pct = sum(s['change_pct'] for s in losses) / len(losses)
        print(f"  Average loss: ${avg_loss:.4f} ({avg_loss_pct:+.2f}%)")
        worst = min(losses, key=lambda x: x['change_pct'])
        print(f"  Worst performer: {worst['symbol']} ({worst['change_pct']:+.2f}%, ${worst['change']:+.4f})")
    
    print(f"\n➡️  UNCHANGED:")
    print(f"  Stocks unchanged: {len(unchanged)}")
    
    # Show all stocks sorted by performance
    sorted_stocks = sorted(valid_stocks, key=lambda x: x['change_pct'], reverse=True)
    
    print(f"\n📋 ALL TOP 10 STOCKS (sorted by performance):")
    for i, stock in enumerate(sorted_stocks, 1):
        change_emoji = "📈" if stock['change'] > 0 else "📉" if stock['change'] < 0 else "➡️"
        print(f"  {i:2d}. {change_emoji} {stock['symbol']:<6} {stock['name'][:45]:<45} "
              f"${stock['yesterday_price']:>7.2f} → ${stock['today_price']:>7.2f} "
              f"({stock['change_pct']:+.2f}%, ${stock['change']:+.4f})")

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_top10_yahooquery.py <snapshot_file>")
        print("Example: python compare_top10_yahooquery.py yahooquery_stocks_20251119_074225.txt")
        sys.exit(1)
    
    snapshot_file = sys.argv[1]
    
    print(f"Loading top 10 stocks from snapshot: {snapshot_file}")
    stocks = load_top10_snapshot(snapshot_file)
    print(f"Loaded {len(stocks)} stocks")
    
    stocks = fetch_today_prices(stocks)
    analyze_results(stocks)

if __name__ == "__main__":
    main()







