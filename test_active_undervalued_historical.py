#!/usr/bin/env python
"""
Test undervalued approach on historical snapshot data.

NOTE: This uses snapshot files from the "undervalued_growth_stocks" screener,
NOT "most active" stocks. The actual approach in find_active_undervalued.py uses:
1. Most active stocks (by volume)
2. Filter for upward trend
3. Calculate notional prices
4. Filter for undervalued (ratio <= 0.66)

This test approximates that by:
1. Using stocks from snapshot (undervalued screener, not most active)
2. Filtering for upward trend (positive change from previous day)
3. Calculating notional prices
4. Filtering for undervalued (ratio <= 0.66)
5. Verifying against next day's snapshot

The results may differ because we're not using "most active" stocks.
"""

import csv
import sys
import glob
from datetime import datetime

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("ERROR: yfinance required. Install with: pip install yfinance")
    sys.exit(1)

# Import the notional price calculation functions
from calculate_notional_prices import calculate_best_notional_price

def load_snapshot(filename):
    """Load stocks from snapshot file."""
    stocks = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                symbol = row['Symbol'].strip()
                name = row['Name'].strip()
                price = float(row['Price'].strip())
                stocks.append({
                    'symbol': symbol,
                    'name': name,
                    'price': price,
                })
            except (ValueError, KeyError):
                continue
    return stocks

def calculate_price_changes(today_stocks, yesterday_stocks):
    """Calculate price changes to identify upward trends."""
    yesterday_prices = {s['symbol']: s['price'] for s in yesterday_stocks}
    
    for stock in today_stocks:
        symbol = stock['symbol']
        today_price = stock['price']
        yesterday_price = yesterday_prices.get(symbol)
        
        if yesterday_price and yesterday_price > 0:
            change_pct = ((today_price - yesterday_price) / yesterday_price) * 100
            stock['change_pct'] = change_pct
        else:
            stock['change_pct'] = 0.0  # No previous data, assume neutral
    
    return today_stocks

def test_snapshot_pair(today_file, tomorrow_file, yesterday_file=None):
    """Test one pair of snapshots."""
    print(f"\n{'='*80}")
    print(f"Testing: {today_file} → {tomorrow_file}")
    if yesterday_file:
        print(f"Using {yesterday_file} for trend calculation")
    print(f"{'='*80}")
    
    # Load today's stocks
    # NOTE: These are from "undervalued_growth_stocks" screener, not "most active"
    today_stocks = load_snapshot(today_file)
    print(f"Loaded {len(today_stocks)} stocks from today's snapshot")
    print("  ⚠️  NOTE: These are from 'undervalued_growth_stocks' screener, not 'most active' stocks")
    
    # Calculate upward trends if we have yesterday's data
    if yesterday_file:
        yesterday_stocks = load_snapshot(yesterday_file)
        today_stocks = calculate_price_changes(today_stocks, yesterday_stocks)
        upward_stocks = [s for s in today_stocks if s.get('change_pct', 0) > 0]
        print(f"Filtered to {len(upward_stocks)} stocks with upward trend")
    else:
        upward_stocks = today_stocks
        print("No yesterday data - using all stocks (no trend filter)")
    
    # Load tomorrow's prices for verification
    tomorrow_stocks = load_snapshot(tomorrow_file)
    tomorrow_prices = {s['symbol']: s['price'] for s in tomorrow_stocks}
    print(f"Loaded {len(tomorrow_prices)} stocks from tomorrow's snapshot")
    
    # Calculate notional prices for upward trending stocks
    print("Calculating notional prices...")
    results = []
    batch_size = 20
    
    for i in range(0, len(upward_stocks), batch_size):
        batch = upward_stocks[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1} ({len(batch)} stocks)...")
        
        for stock in batch:
            if stock['symbol'] not in tomorrow_prices:
                continue
            
            try:
                ticker = yf.Ticker(stock['symbol'])
                info = ticker.info
                
                method, notional_price = calculate_best_notional_price(info)
                
                if notional_price and notional_price > 0:
                    today_price = stock['price']
                    tomorrow_price = tomorrow_prices[stock['symbol']]
                    
                    discount_ratio = today_price / notional_price
                    actual_change_pct = ((tomorrow_price - today_price) / today_price) * 100 if today_price > 0 else 0
                    
                    results.append({
                        'symbol': stock['symbol'],
                        'name': stock['name'],
                        'today_price': today_price,
                        'tomorrow_price': tomorrow_price,
                        'notional_price': notional_price,
                        'discount_ratio': discount_ratio,
                        'actual_change_pct': actual_change_pct,
                        'method': method,
                        'change_pct': stock.get('change_pct', 0),
                    })
            except Exception as e:
                continue
    
    if not results:
        print("No results generated")
        return None
    
    # Filter for undervalued (ratio <= 0.66)
    undervalued = [r for r in results if r['discount_ratio'] <= 0.66]
    
    if not undervalued:
        print("No undervalued stocks found (ratio <= 0.66)")
        return None
    
    # Analyze results
    undervalued_gainers = [r for r in undervalued if r['actual_change_pct'] > 0]
    all_gainers = [r for r in results if r['actual_change_pct'] > 0]
    
    undervalued_avg = sum(r['actual_change_pct'] for r in undervalued) / len(undervalued)
    all_avg = sum(r['actual_change_pct'] for r in results) / len(results)
    
    print(f"\n📊 RESULTS:")
    print(f"  Total stocks analyzed: {len(results)}")
    print(f"  Undervalued stocks (ratio <= 0.66): {len(undervalued)}")
    print(f"  Undervalued gainers: {len(undervalued_gainers)}/{len(undervalued)} ({len(undervalued_gainers)/len(undervalued)*100:.1f}%)")
    print(f"  Undervalued avg change: {undervalued_avg:+.2f}%")
    print(f"  All stocks avg change: {all_avg:+.2f}%")
    print(f"  Outperformance: {undervalued_avg - all_avg:+.2f}%")
    
    return {
        'today_file': today_file,
        'tomorrow_file': tomorrow_file,
        'total_analyzed': len(results),
        'undervalued_count': len(undervalued),
        'undervalued_gainers': len(undervalued_gainers),
        'undervalued_total': len(undervalued),
        'undervalued_avg': undervalued_avg,
        'all_avg': all_avg,
        'outperformance': undervalued_avg - all_avg,
    }

def find_snapshot_triplets():
    """Find triplets of consecutive snapshot files (yesterday, today, tomorrow)."""
    snapshot_files = sorted(glob.glob("yahooquery_stocks_*.txt"))
    
    triplets = []
    for i in range(len(snapshot_files) - 2):
        yesterday_file = snapshot_files[i]
        today_file = snapshot_files[i + 1]
        tomorrow_file = snapshot_files[i + 2]
        
        triplets.append((yesterday_file, today_file, tomorrow_file))
    
    return triplets

def main():
    print("="*80)
    print("UNDERVALUED APPROACH - HISTORICAL TESTING")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n⚠️  IMPORTANT NOTE:")
    print("   This test uses snapshot files from 'undervalued_growth_stocks' screener.")
    print("   The actual find_active_undervalued.py uses 'most active' stocks by volume.")
    print("   Results may differ due to different data sources.")
    print("="*80)
    
    # Find snapshot triplets (yesterday, today, tomorrow)
    triplets = find_snapshot_triplets()
    
    if not triplets:
        print("No snapshot triplets found. Need at least 3 snapshot files.")
        print("Files should be named: yahooquery_stocks_YYYYMMDD_HHMMSS.txt")
        sys.exit(1)
    
    print(f"\nFound {len(triplets)} snapshot triplets to test")
    
    all_results = []
    
    for yesterday_file, today_file, tomorrow_file in triplets:
        result = test_snapshot_pair(today_file, tomorrow_file, yesterday_file)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("\nNo results to summarize")
        sys.exit(1)
    
    # Summarize all results
    print("\n" + "="*80)
    print("HISTORICAL TEST SUMMARY")
    print("="*80)
    
    total_days = len(all_results)
    total_undervalued = sum(r['undervalued_count'] for r in all_results)
    total_undervalued_gainers = sum(r['undervalued_gainers'] for r in all_results)
    
    avg_undervalued_avg = sum(r['undervalued_avg'] for r in all_results) / total_days
    avg_all_avg = sum(r['all_avg'] for r in all_results) / total_days
    avg_outperformance = sum(r['outperformance'] for r in all_results) / total_days
    
    print(f"\n📊 OVERALL STATISTICS ({total_days} days):")
    print(f"  Total undervalued stocks: {total_undervalued}")
    print(f"  Undervalued gainers: {total_undervalued_gainers}/{total_undervalued} ({total_undervalued_gainers/total_undervalued*100:.1f}%)")
    print(f"  Average undervalued performance: {avg_undervalued_avg:+.2f}%")
    print(f"  Average all stocks performance: {avg_all_avg:+.2f}%")
    print(f"  Average outperformance: {avg_outperformance:+.2f}%")
    
    winning_days = sum(1 for r in all_results if r['outperformance'] > 0)
    print(f"\n📈 CONSISTENCY:")
    print(f"  Days undervalued outperformed: {winning_days}/{total_days} ({winning_days/total_days*100:.1f}%)")
    
    print(f"\n📅 PER-DAY BREAKDOWN:")
    print(f"{'Date':<15} {'Undervalued':<12} {'Gainers':<10} {'Avg%':<10} {'All Avg%':<10} {'Outperf':<10}")
    print("-" * 80)
    for r in all_results:
        date_str = r['today_file'].replace('yahooquery_stocks_', '').replace('.txt', '')[:8]
        print(f"{date_str:<15} {r['undervalued_count']:<12} "
              f"{r['undervalued_gainers']}/{r['undervalued_total']:<8} "
              f"{r['undervalued_avg']:>+8.2f}% {r['all_avg']:>+8.2f}% {r['outperformance']:>+8.2f}%")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

