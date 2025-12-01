#!/usr/bin/env python
"""
Backtest notional price predictions on historical data.
Tests the approach on multiple days to see if undervalued stocks (ratio < 0.66) 
consistently outperform.
"""

import csv
import sys
import glob
from datetime import datetime
from collections import defaultdict

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("ERROR: yfinance required. Install with: pip install yfinance")
    sys.exit(1)

# Import the notional price calculation functions
from calculate_notional_prices import (
    calculate_notional_price_pe,
    calculate_notional_price_ev_ebitda,
    calculate_notional_price_dcf,
    calculate_notional_price_revenue,
    calculate_notional_price_book,
    calculate_best_notional_price
)

def load_snapshot_prices(filename):
    """Load prices from snapshot file."""
    prices = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                symbol = row['Symbol'].strip()
                prices[symbol] = float(row['Price'].strip())
            except (ValueError, KeyError):
                continue
    return prices

def analyze_snapshot_pair(today_file, tomorrow_file):
    """Analyze one pair of snapshots (today's predictions vs tomorrow's results)."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {today_file} → {today_file}")
    print(f"{'='*80}")
    
    # Load today's prices
    today_prices = load_snapshot_prices(today_file)
    print(f"Loaded {len(today_prices)} stocks from today's snapshot")
    
    # Load tomorrow's prices
    tomorrow_prices = load_snapshot_prices(tomorrow_file)
    print(f"Loaded {len(tomorrow_prices)} stocks from tomorrow's snapshot")
    
    # Calculate notional prices for today's stocks
    print("Calculating notional prices...")
    results = []
    symbols = list(today_prices.keys())
    
    batch_size = 20
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1} ({len(batch)} stocks)...")
        
        for symbol in batch:
            if symbol not in tomorrow_prices:
                continue
            
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Calculate notional price
                method, notional_price = calculate_best_notional_price(info)
                
                if notional_price and notional_price > 0:
                    today_price = today_prices[symbol]
                    tomorrow_price = tomorrow_prices[symbol]
                    
                    discount_ratio = today_price / notional_price
                    actual_change_pct = ((tomorrow_price - today_price) / today_price) * 100 if today_price > 0 else 0
                    
                    results.append({
                        'symbol': symbol,
                        'today_price': today_price,
                        'tomorrow_price': tomorrow_price,
                        'notional_price': notional_price,
                        'discount_ratio': discount_ratio,
                        'actual_change_pct': actual_change_pct,
                        'method': method,
                    })
            except Exception as e:
                continue
    
    if not results:
        print("No results generated")
        return None
    
    # Analyze results
    undervalued = [r for r in results if r['discount_ratio'] < 0.66]
    all_stocks = results
    
    if not undervalued:
        print("No undervalued stocks found (ratio < 0.66)")
        return None
    
    undervalued_gainers = [r for r in undervalued if r['actual_change_pct'] > 0]
    all_gainers = [r for r in all_stocks if r['actual_change_pct'] > 0]
    
    undervalued_avg = sum(r['actual_change_pct'] for r in undervalued) / len(undervalued)
    all_avg = sum(r['actual_change_pct'] for r in all_stocks) / len(all_stocks)
    
    print(f"\n📊 RESULTS:")
    print(f"  Undervalued stocks: {len(undervalued)}")
    print(f"  Undervalued gainers: {len(undervalued_gainers)}/{len(undervalued)} ({len(undervalued_gainers)/len(undervalued)*100:.1f}%)")
    print(f"  Undervalued avg change: {undervalued_avg:+.2f}%")
    print(f"  All stocks avg change: {all_avg:+.2f}%")
    print(f"  Outperformance: {undervalued_avg - all_avg:+.2f}%")
    
    return {
        'undervalued_count': len(undervalued),
        'undervalued_gainers': len(undervalued_gainers),
        'undervalued_total': len(undervalued),
        'undervalued_avg': undervalued_avg,
        'all_avg': all_avg,
        'outperformance': undervalued_avg - all_avg,
        'all_gainers': len(all_gainers),
        'all_total': len(all_stocks),
    }

def find_snapshot_pairs():
    """Find pairs of consecutive snapshot files."""
    snapshot_files = sorted(glob.glob("yahooquery_stocks_*.txt"))
    
    pairs = []
    for i in range(len(snapshot_files) - 1):
        today_file = snapshot_files[i]
        tomorrow_file = snapshot_files[i + 1]
        
        # Extract dates from filenames
        today_date = today_file.split('_')[2].split('.')[0]  # yyyymmdd
        tomorrow_date = tomorrow_file.split('_')[2].split('.')[0]
        
        pairs.append((today_file, tomorrow_file, today_date, tomorrow_date))
    
    return pairs

def main():
    print("="*80)
    print("NOTIONAL PRICE BACKTESTING")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find snapshot pairs
    pairs = find_snapshot_pairs()
    
    if not pairs:
        print("No snapshot pairs found. Need at least 2 snapshot files.")
        print("Files should be named: yahooquery_stocks_YYYYMMDD_HHMMSS.txt")
        sys.exit(1)
    
    print(f"\nFound {len(pairs)} snapshot pairs to analyze")
    
    all_results = []
    
    for today_file, tomorrow_file, today_date, tomorrow_date in pairs:
        result = analyze_snapshot_pair(today_file, tomorrow_file)
        if result:
            result['today_date'] = today_date
            result['tomorrow_date'] = tomorrow_date
            all_results.append(result)
    
    if not all_results:
        print("\nNo results to summarize")
        sys.exit(1)
    
    # Summarize all results
    print("\n" + "="*80)
    print("BACKTEST SUMMARY")
    print("="*80)
    
    total_days = len(all_results)
    total_undervalued = sum(r['undervalued_count'] for r in all_results)
    total_undervalued_gainers = sum(r['undervalued_gainers'] for r in all_results)
    total_all_gainers = sum(r['all_gainers'] for r in all_results)
    total_all = sum(r['all_total'] for r in all_results)
    
    avg_undervalued_avg = sum(r['undervalued_avg'] for r in all_results) / total_days
    avg_all_avg = sum(r['all_avg'] for r in all_results) / total_days
    avg_outperformance = sum(r['outperformance'] for r in all_results) / total_days
    
    print(f"\n📊 OVERALL STATISTICS ({total_days} days):")
    print(f"  Total undervalued stocks analyzed: {total_undervalued}")
    print(f"  Undervalued gainers: {total_undervalued_gainers}/{total_undervalued} ({total_undervalued_gainers/total_undervalued*100:.1f}%)")
    print(f"  All stocks gainers: {total_all_gainers}/{total_all} ({total_all_gainers/total_all*100:.1f}%)")
    print(f"  Average undervalued performance: {avg_undervalued_avg:+.2f}%")
    print(f"  Average all stocks performance: {avg_all_avg:+.2f}%")
    print(f"  Average outperformance: {avg_outperformance:+.2f}%")
    
    # Count winning days
    winning_days = sum(1 for r in all_results if r['outperformance'] > 0)
    print(f"\n📈 CONSISTENCY:")
    print(f"  Days undervalued outperformed: {winning_days}/{total_days} ({winning_days/total_days*100:.1f}%)")
    
    # Show per-day breakdown
    print(f"\n📅 PER-DAY BREAKDOWN:")
    print(f"{'Date':<12} {'Undervalued':<12} {'Gainers':<10} {'Avg%':<10} {'All Avg%':<10} {'Outperf':<10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['today_date']:<12} {r['undervalued_count']:<12} "
              f"{r['undervalued_gainers']}/{r['undervalued_total']:<8} "
              f"{r['undervalued_avg']:>+8.2f}% {r['all_avg']:>+8.2f}% {r['outperformance']:>+8.2f}%")
    
    print("\n" + "="*80)
    print("💡 INTERPRETATION:")
    if avg_outperformance > 0:
        print(f"  ✅ Notional price approach shows {avg_outperformance:.2f}% average outperformance")
        print(f"  ✅ Undervalued stocks beat market {winning_days}/{total_days} days ({winning_days/total_days*100:.1f}%)")
    else:
        print(f"  ❌ Notional price approach underperformed by {abs(avg_outperformance):.2f}% on average")
    print("="*80)

if __name__ == "__main__":
    main()





















