#!/usr/bin/env python
"""
Verify notional price predictions against actual performance.
"""

import csv
import sys

def load_notional_predictions(filename):
    """Load notional price predictions."""
    predictions = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                symbol = row['Symbol'].strip()
                predictions[symbol] = {
                    'symbol': symbol,
                    'name': row['Name'].strip(),
                    'actual_price': float(row['ActualPrice'].strip()),
                    'notional_price': float(row['NotionalPrice'].strip()),
                    'discount_ratio': float(row['DiscountRatio'].strip()),
                    'upside_pct': float(row['UpsidePct'].strip()),
                    'method': row['Method'].strip(),
                }
            except (ValueError, KeyError):
                continue
    return predictions

def load_actual_prices(filename):
    """Load actual prices from snapshot."""
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

def verify_predictions(predictions, actual_prices):
    """Verify predictions against actual performance."""
    results = []
    
    for symbol, pred in predictions.items():
        if symbol not in actual_prices:
            continue
        
        pred_price = pred['actual_price']
        actual_price = actual_prices[symbol]
        
        actual_change = actual_price - pred_price
        actual_change_pct = (actual_change / pred_price) * 100 if pred_price > 0 else 0
        
        results.append({
            'symbol': symbol,
            'name': pred['name'],
            'pred_price': pred_price,
            'actual_price': actual_price,
            'notional_price': pred['notional_price'],
            'discount_ratio': pred['discount_ratio'],
            'predicted_upside': pred['upside_pct'],
            'actual_change_pct': actual_change_pct,
            'method': pred['method'],
        })
    
    return results

def analyze_results(results):
    """Analyze verification results."""
    if not results:
        print("No results to analyze")
        return
    
    # Sort by discount ratio (most undervalued first)
    results.sort(key=lambda x: x['discount_ratio'])
    
    # Filter undervalued stocks (discount_ratio < 0.66)
    undervalued = [r for r in results if r['discount_ratio'] < 0.66]
    
    print("\n" + "="*100)
    print("NOTIONAL PRICE PREDICTIONS - VERIFICATION")
    print("="*100)
    print(f"\nTotal stocks analyzed: {len(results)}")
    print(f"Undervalued stocks (ratio < 0.66): {len(undervalued)} ({len(undervalued)/len(results)*100:.1f}%)")
    
    if undervalued:
        print(f"\n🎯 UNDERVALUED STOCKS PERFORMANCE:")
        gainers = [r for r in undervalued if r['actual_change_pct'] > 0]
        losers = [r for r in undervalued if r['actual_change_pct'] < 0]
        
        print(f"  Gainers: {len(gainers)}/{len(undervalued)} ({len(gainers)/len(undervalued)*100:.1f}%)")
        print(f"  Losers: {len(losers)}/{len(undervalued)} ({len(losers)/len(undervalued)*100:.1f}%)")
        
        if gainers:
            avg_gain = sum(r['actual_change_pct'] for r in gainers) / len(gainers)
            print(f"  Average gain: {avg_gain:+.2f}%")
        
        if losers:
            avg_loss = sum(r['actual_change_pct'] for r in losers) / len(losers)
            print(f"  Average loss: {avg_loss:+.2f}%")
        
        avg_change = sum(r['actual_change_pct'] for r in undervalued) / len(undervalued)
        print(f"  Average change: {avg_change:+.2f}%")
        
        print(f"\n📊 UNDERVALUED STOCKS DETAIL:")
        print(f"{'Symbol':<8} {'Name':<35} {'Ratio':<8} {'PredUpside':<12} {'Actual%':<10} {'Method':<12}")
        print("-" * 100)
        
        for r in undervalued:
            change_emoji = "📈" if r['actual_change_pct'] > 0 else "📉" if r['actual_change_pct'] < 0 else "➡️"
            print(f"{change_emoji} {r['symbol']:<6} {r['name'][:33]:<35} "
                  f"{r['discount_ratio']:>6.2f} {r['predicted_upside']:>+9.1f}% {r['actual_change_pct']:>+8.1f}% {r['method']:<12}")
    
    # Compare with all stocks
    all_gainers = [r for r in results if r['actual_change_pct'] > 0]
    all_avg = sum(r['actual_change_pct'] for r in results) / len(results)
    
    print(f"\n📊 ALL STOCKS PERFORMANCE:")
    print(f"  Gainers: {len(all_gainers)}/{len(results)} ({len(all_gainers)/len(results)*100:.1f}%)")
    print(f"  Average change: {all_avg:+.2f}%")
    
    if undervalued:
        print(f"\n💡 COMPARISON:")
        print(f"  Undervalued stocks avg: {avg_change:+.2f}%")
        print(f"  All stocks avg: {all_avg:+.2f}%")
        print(f"  Difference: {avg_change - all_avg:+.2f}%")
        
        if avg_change > all_avg:
            print(f"  ✅ Undervalued stocks outperformed by {avg_change - all_avg:.2f}%")
        else:
            print(f"  ❌ Undervalued stocks underperformed by {all_avg - avg_change:.2f}%")
    
    print("\n" + "="*100)

def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_notional_predictions.py <notional_prices_file> <actual_prices_file>")
        print("Example: python verify_notional_predictions.py notional_prices_20251122_112152.csv yahooquery_stocks_20251121_145532.txt")
        sys.exit(1)
    
    notional_file = sys.argv[1]
    actual_file = sys.argv[2]
    
    print(f"Loading notional predictions from: {notional_file}")
    predictions = load_notional_predictions(notional_file)
    print(f"Loaded {len(predictions)} predictions")
    
    print(f"Loading actual prices from: {actual_file}")
    actual_prices = load_actual_prices(actual_file)
    print(f"Loaded {len(actual_prices)} actual prices")
    
    results = verify_predictions(predictions, actual_prices)
    
    if not results:
        print("No matching stocks found")
        sys.exit(1)
    
    analyze_results(results)

if __name__ == "__main__":
    main()




















































