#!/usr/bin/env python
"""
Verify active undervalued stock predictions against actual performance.
Run this on Monday (or next trading day) to see how the predictions performed.
"""

import csv
import sys
from datetime import datetime

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("ERROR: yfinance required. Install with: pip install yfinance")
    sys.exit(1)

def load_predictions(filename):
    """Load predictions from CSV file."""
    predictions = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                symbol = row['Symbol'].strip()
                predictions[symbol] = {
                    'symbol': symbol,
                    'name': row['Name'].strip(),
                    'prediction_price': float(row['ActualPrice'].strip()),
                    'notional_price': float(row['NotionalPrice'].strip()),
                    'discount_ratio': float(row['DiscountRatio'].strip()),
                    'upside_pct': float(row['UpsidePct'].strip()),
                    'today_change_pct': float(row['TodayChangePct'].strip()),
                    'method': row['Method'].strip(),
                }
            except (ValueError, KeyError) as e:
                print(f"Error parsing row: {row}, error: {e}")
                continue
    return predictions

def fetch_current_prices(symbols):
    """Fetch current prices for symbols."""
    print(f"Fetching current prices for {len(symbols)} stocks...")
    
    price_dict = {}
    batch_size = 50
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1} ({len(batch)} stocks)...")
        
        try:
            tickers = yf.Tickers(" ".join(batch))
            for symbol in batch:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.fast_info
                    price = info.get('lastPrice') or info.get('regularMarketPrice')
                    if price:
                        price_dict[symbol] = float(price)
                except:
                    pass
        except:
            pass
    
    return price_dict

def analyze_results(predictions, current_prices):
    """Analyze prediction accuracy."""
    results = []
    
    for symbol, pred in predictions.items():
        if symbol not in current_prices:
            continue
        
        pred_price = pred['prediction_price']
        current_price = current_prices[symbol]
        
        actual_change = current_price - pred_price
        actual_change_pct = (actual_change / pred_price) * 100 if pred_price > 0 else 0
        
        results.append({
            'symbol': symbol,
            'name': pred['name'],
            'prediction_price': pred_price,
            'current_price': current_price,
            'notional_price': pred['notional_price'],
            'discount_ratio': pred['discount_ratio'],
            'predicted_upside': pred['upside_pct'],
            'actual_change_pct': actual_change_pct,
            'method': pred['method'],
        })
    
    return results

def display_results(results, prediction_date):
    """Display verification results."""
    if not results:
        print("No results to display")
        return
    
    # Sort by discount ratio (most undervalued first)
    results.sort(key=lambda x: x['discount_ratio'])
    
    gainers = [r for r in results if r['actual_change_pct'] > 0]
    losers = [r for r in results if r['actual_change_pct'] < 0]
    
    avg_change = sum(r['actual_change_pct'] for r in results) / len(results)
    avg_predicted_upside = sum(r['predicted_upside'] for r in results) / len(results)
    
    print("\n" + "="*100)
    print("ACTIVE UNDERVALUED STOCKS - PREDICTION VERIFICATION")
    print("="*100)
    print(f"\nPrediction Date: {prediction_date}")
    print(f"Verification Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total stocks analyzed: {len(results)}")
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"  Gainers: {len(gainers)}/{len(results)} ({len(gainers)/len(results)*100:.1f}%)")
    print(f"  Losers: {len(losers)}/{len(results)} ({len(losers)/len(results)*100:.1f}%)")
    print(f"  Average actual change: {avg_change:+.2f}%")
    print(f"  Average predicted upside: {avg_predicted_upside:.2f}%")
    
    if gainers:
        avg_gain = sum(r['actual_change_pct'] for r in gainers) / len(gainers)
        best = max(gainers, key=lambda x: x['actual_change_pct'])
        print(f"\n📈 GAINERS:")
        print(f"  Average gain: {avg_gain:+.2f}%")
        print(f"  Best performer: {best['symbol']} ({best['actual_change_pct']:+.2f}%)")
    
    if losers:
        avg_loss = sum(r['actual_change_pct'] for r in losers) / len(losers)
        worst = min(losers, key=lambda x: x['actual_change_pct'])
        print(f"\n📉 LOSERS:")
        print(f"  Average loss: {avg_loss:+.2f}%")
        print(f"  Worst performer: {worst['symbol']} ({worst['actual_change_pct']:+.2f}%)")
    
    print(f"\n🏆 DETAILED RESULTS:")
    print(f"{'Rank':<6} {'Symbol':<8} {'Name':<30} {'Pred':<10} {'Current':<10} {'Change%':<10} {'Upside%':<10} {'Method':<12}")
    print("-" * 100)
    
    for i, result in enumerate(results, 1):
        change_emoji = "📈" if result['actual_change_pct'] > 0 else "📉" if result['actual_change_pct'] < 0 else "➡️"
        name = result['name'][:28] + '..' if len(result['name']) > 30 else result['name']
        print(f"{i:<6} {change_emoji} {result['symbol']:<6} {name:<30} "
              f"${result['prediction_price']:>8.2f} ${result['current_price']:>8.2f} "
              f"{result['actual_change_pct']:>+8.1f}% {result['predicted_upside']:>+8.1f}% {result['method']:<12}")
    
    print("\n" + "="*100)
    print("💡 INTERPRETATION:")
    if avg_change > 0:
        print(f"  ✅ Predictions showed positive average return of {avg_change:.2f}%")
    else:
        print(f"  ❌ Predictions showed negative average return of {avg_change:.2f}%")
    print(f"  - {len(gainers)}/{len(results)} stocks gained ({len(gainers)/len(results)*100:.1f}% win rate)")
    print("="*100)

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_active_undervalued.py <predictions_file>")
        print("Example: python verify_active_undervalued.py active_undervalued_20251122_115714.csv")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    
    # Extract date from filename if possible
    prediction_date = predictions_file.replace('active_undervalued_', '').replace('.csv', '')
    if len(prediction_date) == 15:  # YYYYMMDD_HHMMSS format
        try:
            dt = datetime.strptime(prediction_date, '%Y%m%d_%H%M%S')
            prediction_date = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
    
    print(f"Loading predictions from: {predictions_file}")
    predictions = load_predictions(predictions_file)
    print(f"Loaded {len(predictions)} predictions")
    
    if not predictions:
        print("No predictions loaded")
        sys.exit(1)
    
    symbols = list(predictions.keys())
    current_prices = fetch_current_prices(symbols)
    print(f"Fetched prices for {len(current_prices)} stocks")
    
    if not current_prices:
        print("No current prices available")
        sys.exit(1)
    
    results = analyze_results(predictions, current_prices)
    
    if not results:
        print("No matching stocks found")
        sys.exit(1)
    
    display_results(results, prediction_date)
    
    # Save verification results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"verification_active_undervalued_{timestamp}.csv"
    
    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Symbol', 'Name', 'PredictionPrice', 'CurrentPrice', 
                                               'Change', 'ChangePct', 'PredictedUpside', 'DiscountRatio', 'Method'])
        writer.writeheader()
        for result in sorted(results, key=lambda x: x['discount_ratio']):
            writer.writerow({
                'Symbol': result['symbol'],
                'Name': result['name'],
                'PredictionPrice': f"{result['prediction_price']:.2f}",
                'CurrentPrice': f"{result['current_price']:.2f}",
                'Change': f"{result['current_price'] - result['prediction_price']:.4f}",
                'ChangePct': f"{result['actual_change_pct']:.2f}",
                'PredictedUpside': f"{result['predicted_upside']:.2f}",
                'DiscountRatio': f"{result['discount_ratio']:.4f}",
                'Method': result['method'],
            })
    
    print(f"\n💾 Verification results saved to: {output_file}")

if __name__ == "__main__":
    main()




















