#!/usr/bin/env python
"""
Compare and contrast the top 10 and bottom 10 performers from yahooquery results.
"""

import csv
import sys
import yfinance as yf
from collections import defaultdict

def load_snapshot_and_calculate_changes(filename):
    """Load snapshot and calculate price changes."""
    stocks = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                symbol = row['Symbol'].strip()
                name = row['Name'].strip()
                yesterday_price = float(row['Price'].strip())
                stocks.append({
                    'symbol': symbol,
                    'name': name,
                    'yesterday_price': yesterday_price
                })
            except (ValueError, KeyError) as e:
                continue
    
    # Fetch today's prices
    symbols = [s['symbol'] for s in stocks]
    print(f"Fetching prices for {len(symbols)} stocks...")
    
    batch_size = 50
    price_dict = {}
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}...")
        
        try:
            tickers = yf.Tickers(" ".join(batch))
            for symbol in batch:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.fast_info
                    today_price = info.get('lastPrice') or info.get('regularMarketPrice')
                    if today_price:
                        price_dict[symbol] = float(today_price)
                except:
                    pass
        except:
            pass
    
    # Calculate changes
    for stock in stocks:
        if stock['symbol'] in price_dict:
            stock['today_price'] = price_dict[stock['symbol']]
            stock['change'] = stock['today_price'] - stock['yesterday_price']
            stock['change_pct'] = (stock['change'] / stock['yesterday_price']) * 100 if stock['yesterday_price'] > 0 else 0
    
    # Filter valid stocks
    valid_stocks = [s for s in stocks if 'today_price' in s]
    valid_stocks.sort(key=lambda x: x['change_pct'], reverse=True)
    
    return valid_stocks

def fetch_stock_metrics(symbols):
    """Fetch additional metrics for analysis."""
    print(f"\nFetching detailed metrics for {len(symbols)} stocks...")
    
    metrics = {}
    batch_size = 20  # Smaller batches for detailed data
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}...")
        
        for symbol in batch:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                metrics[symbol] = {
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
                    'peg_ratio': info.get('pegRatio'),
                    'price_to_book': info.get('priceToBook'),
                    'dividend_yield': info.get('dividendYield'),
                    'volume': info.get('volume'),
                    'avg_volume': info.get('averageVolume'),
                    'beta': info.get('beta'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'price': info.get('currentPrice') or info.get('regularMarketPrice'),
                }
            except Exception as e:
                metrics[symbol] = {}
    
    return metrics

def analyze_groups(top10, bottom10, metrics):
    """Compare top 10 vs bottom 10."""
    print("\n" + "="*80)
    print("TOP 10 vs BOTTOM 10 PERFORMER ANALYSIS")
    print("="*80)
    
    # Price characteristics
    top10_prices = [s['yesterday_price'] for s in top10]
    bottom10_prices = [s['yesterday_price'] for s in bottom10]
    
    print(f"\n💰 PRICE CHARACTERISTICS:")
    print(f"  Top 10 average price: ${sum(top10_prices)/len(top10_prices):.2f}")
    print(f"  Bottom 10 average price: ${sum(bottom10_prices)/len(bottom10_prices):.2f}")
    print(f"  Top 10 price range: ${min(top10_prices):.2f} - ${max(top10_prices):.2f}")
    print(f"  Bottom 10 price range: ${min(bottom10_prices):.2f} - ${max(bottom10_prices):.2f}")
    
    # Market cap
    top10_caps = [metrics.get(s['symbol'], {}).get('market_cap', 0) for s in top10 if metrics.get(s['symbol'], {}).get('market_cap')]
    bottom10_caps = [metrics.get(s['symbol'], {}).get('market_cap', 0) for s in bottom10 if metrics.get(s['symbol'], {}).get('market_cap')]
    
    if top10_caps and bottom10_caps:
        avg_top_cap = sum(top10_caps) / len(top10_caps) / 1e9  # Convert to billions
        avg_bottom_cap = sum(bottom10_caps) / len(bottom10_caps) / 1e9
        print(f"\n📊 MARKET CAP:")
        print(f"  Top 10 average: ${avg_top_cap:.2f}B")
        print(f"  Bottom 10 average: ${avg_bottom_cap:.2f}B")
    
    # P/E Ratio
    top10_pe = [metrics.get(s['symbol'], {}).get('pe_ratio') for s in top10 if metrics.get(s['symbol'], {}).get('pe_ratio')]
    bottom10_pe = [metrics.get(s['symbol'], {}).get('pe_ratio') for s in bottom10 if metrics.get(s['symbol'], {}).get('pe_ratio')]
    
    if top10_pe and bottom10_pe:
        avg_top_pe = sum(top10_pe) / len(top10_pe)
        avg_bottom_pe = sum(bottom10_pe) / len(bottom10_pe)
        print(f"\n📈 P/E RATIO:")
        print(f"  Top 10 average: {avg_top_pe:.2f}")
        print(f"  Bottom 10 average: {avg_bottom_pe:.2f}")
    
    # PEG Ratio
    top10_peg = [metrics.get(s['symbol'], {}).get('peg_ratio') for s in top10 if metrics.get(s['symbol'], {}).get('peg_ratio')]
    bottom10_peg = [metrics.get(s['symbol'], {}).get('peg_ratio') for s in bottom10 if metrics.get(s['symbol'], {}).get('peg_ratio')]
    
    if top10_peg and bottom10_peg:
        avg_top_peg = sum(top10_peg) / len(top10_peg)
        avg_bottom_peg = sum(bottom10_peg) / len(bottom10_peg)
        print(f"\n📊 PEG RATIO:")
        print(f"  Top 10 average: {avg_top_peg:.2f}")
        print(f"  Bottom 10 average: {avg_bottom_peg:.2f}")
    
    # Price to Book
    top10_pb = [metrics.get(s['symbol'], {}).get('price_to_book') for s in top10 if metrics.get(s['symbol'], {}).get('price_to_book')]
    bottom10_pb = [metrics.get(s['symbol'], {}).get('price_to_book') for s in bottom10 if metrics.get(s['symbol'], {}).get('price_to_book')]
    
    if top10_pb and bottom10_pb:
        avg_top_pb = sum(top10_pb) / len(top10_pb)
        avg_bottom_pb = sum(bottom10_pb) / len(bottom10_pb)
        print(f"\n📚 PRICE TO BOOK:")
        print(f"  Top 10 average: {avg_top_pb:.2f}")
        print(f"  Bottom 10 average: {avg_bottom_pb:.2f}")
    
    # Beta
    top10_beta = [metrics.get(s['symbol'], {}).get('beta') for s in top10 if metrics.get(s['symbol'], {}).get('beta')]
    bottom10_beta = [metrics.get(s['symbol'], {}).get('beta') for s in bottom10 if metrics.get(s['symbol'], {}).get('beta')]
    
    if top10_beta and bottom10_beta:
        avg_top_beta = sum(top10_beta) / len(top10_beta)
        avg_bottom_beta = sum(bottom10_beta) / len(bottom10_beta)
        print(f"\n📉 BETA (Volatility):")
        print(f"  Top 10 average: {avg_top_beta:.2f}")
        print(f"  Bottom 10 average: {avg_bottom_beta:.2f}")
    
    # Sector analysis
    top10_sectors = defaultdict(int)
    bottom10_sectors = defaultdict(int)
    
    for s in top10:
        sector = metrics.get(s['symbol'], {}).get('sector', 'Unknown')
        top10_sectors[sector] += 1
    
    for s in bottom10:
        sector = metrics.get(s['symbol'], {}).get('sector', 'Unknown')
        bottom10_sectors[sector] += 1
    
    print(f"\n🏭 SECTOR DISTRIBUTION:")
    print(f"  Top 10 sectors:")
    for sector, count in sorted(top10_sectors.items(), key=lambda x: x[1], reverse=True):
        print(f"    {sector}: {count}")
    print(f"  Bottom 10 sectors:")
    for sector, count in sorted(bottom10_sectors.items(), key=lambda x: x[1], reverse=True):
        print(f"    {sector}: {count}")
    
    # Volume
    top10_vol = [metrics.get(s['symbol'], {}).get('volume', 0) for s in top10 if metrics.get(s['symbol'], {}).get('volume')]
    bottom10_vol = [metrics.get(s['symbol'], {}).get('volume', 0) for s in bottom10 if metrics.get(s['symbol'], {}).get('volume')]
    
    if top10_vol and bottom10_vol:
        avg_top_vol = sum(top10_vol) / len(top10_vol) / 1e6  # Convert to millions
        avg_bottom_vol = sum(bottom10_vol) / len(bottom10_vol) / 1e6
        print(f"\n📊 VOLUME:")
        print(f"  Top 10 average: {avg_top_vol:.2f}M")
        print(f"  Bottom 10 average: {avg_bottom_vol:.2f}M")
    
    print(f"\n📋 TOP 10 PERFORMERS:")
    for i, stock in enumerate(top10, 1):
        change_emoji = "📈"
        print(f"  {i:2d}. {change_emoji} {stock['symbol']:<6} {stock['name'][:40]:<40} "
              f"${stock['yesterday_price']:>7.2f} → ${stock['today_price']:>7.2f} "
              f"({stock['change_pct']:+.2f}%)")
    
    print(f"\n📋 BOTTOM 10 PERFORMERS:")
    for i, stock in enumerate(bottom10, 1):
        change_emoji = "📉"
        print(f"  {i:2d}. {change_emoji} {stock['symbol']:<6} {stock['name'][:40]:<40} "
              f"${stock['yesterday_price']:>7.2f} → ${stock['today_price']:>7.2f} "
              f"({stock['change_pct']:+.2f}%)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_top_bottom_performers.py <snapshot_file>")
        print("Example: python compare_top_bottom_performers.py yahooquery_stocks_20251119_074225.txt")
        sys.exit(1)
    
    snapshot_file = sys.argv[1]
    
    print(f"Loading snapshot: {snapshot_file}")
    stocks = load_snapshot_and_calculate_changes(snapshot_file)
    
    if len(stocks) < 20:
        print(f"Not enough stocks ({len(stocks)}) to compare top/bottom 10")
        sys.exit(1)
    
    top10 = stocks[:10]
    bottom10 = stocks[-10:]
    
    all_symbols = [s['symbol'] for s in top10 + bottom10]
    metrics = fetch_stock_metrics(all_symbols)
    
    analyze_groups(top10, bottom10, metrics)

if __name__ == "__main__":
    main()







