#!/usr/bin/env python
"""
Verify prediction accuracy by comparing predicted gainers with actual results.
"""

import csv
import sys
from datetime import datetime
from collections import defaultdict

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

def load_predictions(filename):
    """Load predictions from file."""
    predictions = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                symbol = row['Symbol'].strip()
                predictions[symbol] = {
                    'symbol': symbol,
                    'name': row['Name'].strip(),
                    'prediction_price': float(row['Price'].strip()),
                    'prediction_score': float(row['PredictionScore'].strip()),
                    'market_cap': float(row['MarketCap'].strip()) if row['MarketCap'] else 0,
                    'volume': float(row['Volume'].strip()) if row['Volume'] else 0,
                    'beta': float(row['Beta'].strip()) if row['Beta'] else 1.0,
                }
            except (ValueError, KeyError) as e:
                print(f"Error parsing prediction row: {row}, error: {e}")
                continue
    return predictions

def load_snapshot_prices(filename):
    """Load actual prices from snapshot file."""
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

def fetch_current_prices(symbols):
    """Fetch current prices for symbols."""
    if not HAS_YFINANCE:
        print("ERROR: yfinance not available. Please provide a snapshot file.")
        return {}
    
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

def calculate_accuracy(predictions, actual_prices, prediction_prices):
    """Calculate prediction accuracy metrics."""
    results = []
    
    for symbol, pred_data in predictions.items():
        if symbol not in actual_prices:
            continue
        
        pred_price = pred_data['prediction_price']
        actual_price = actual_prices[symbol]
        
        # Calculate actual change
        actual_change = actual_price - pred_price
        actual_change_pct = (actual_change / pred_price) * 100 if pred_price > 0 else 0
        
        results.append({
            'symbol': symbol,
            'name': pred_data['name'],
            'prediction_price': pred_price,
            'actual_price': actual_price,
            'actual_change': actual_change,
            'actual_change_pct': actual_change_pct,
            'prediction_score': pred_data['prediction_score'],
            'rank': None,  # Will be set later
        })
    
    # Sort by prediction score (highest first)
    results.sort(key=lambda x: x['prediction_score'], reverse=True)
    
    # Assign ranks
    for i, result in enumerate(results, 1):
        result['rank'] = i
    
    return results

def analyze_accuracy(results):
    """Analyze prediction accuracy."""
    if not results:
        print("No results to analyze")
        return
    
    # Overall statistics
    total_stocks = len(results)
    gainers = [r for r in results if r['actual_change'] > 0]
    losers = [r for r in results if r['actual_change'] < 0]
    unchanged = [r for r in results if r['actual_change'] == 0]
    
    avg_change = sum(r['actual_change'] for r in results) / total_stocks
    avg_change_pct = sum(r['actual_change_pct'] for r in results) / total_stocks
    
    # Top 20 predictions vs actual
    top20 = results[:20]
    top20_gainers = [r for r in top20 if r['actual_change'] > 0]
    top20_avg_change = sum(r['actual_change'] for r in top20) / len(top20)
    top20_avg_change_pct = sum(r['actual_change_pct'] for r in top20) / len(top20)
    
    # Top 10 predictions
    top10 = results[:10]
    top10_gainers = [r for r in top10 if r['actual_change'] > 0]
    top10_avg_change = sum(r['actual_change'] for r in top10) / len(top10)
    top10_avg_change_pct = sum(r['actual_change_pct'] for r in top10) / len(top10)
    
    print("\n" + "="*80)
    print("PREDICTION ACCURACY VERIFICATION")
    print("="*80)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total stocks analyzed: {total_stocks}")
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  Average change: ${avg_change:.4f} ({avg_change_pct:+.2f}%)")
    print(f"  Gainers: {len(gainers)} ({len(gainers)/total_stocks*100:.1f}%)")
    print(f"  Losers: {len(losers)} ({len(losers)/total_stocks*100:.1f}%)")
    print(f"  Unchanged: {len(unchanged)}")
    
    if gainers:
        avg_gain = sum(r['actual_change'] for r in gainers) / len(gainers)
        avg_gain_pct = sum(r['actual_change_pct'] for r in gainers) / len(gainers)
        best = max(gainers, key=lambda x: x['actual_change_pct'])
        print(f"\n📈 GAINERS:")
        print(f"  Average gain: ${avg_gain:.4f} ({avg_gain_pct:+.2f}%)")
        print(f"  Best performer: {best['symbol']} ({best['actual_change_pct']:+.2f}%, ${best['actual_change']:+.4f})")
    
    if losers:
        avg_loss = sum(r['actual_change'] for r in losers) / len(losers)
        avg_loss_pct = sum(r['actual_change_pct'] for r in losers) / len(losers)
        worst = min(losers, key=lambda x: x['actual_change_pct'])
        print(f"\n📉 LOSERS:")
        print(f"  Average loss: ${avg_loss:.4f} ({avg_loss_pct:+.2f}%)")
        print(f"  Worst performer: {worst['symbol']} ({worst['actual_change_pct']:+.2f}%, ${worst['actual_change']:+.4f})")
    
    print(f"\n🎯 TOP 10 PREDICTIONS PERFORMANCE:")
    print(f"  Gainers: {len(top10_gainers)}/{len(top10)} ({len(top10_gainers)/len(top10)*100:.1f}%)")
    print(f"  Average change: ${top10_avg_change:.4f} ({top10_avg_change_pct:+.2f}%)")
    print(f"\n  Top 10 Details:")
    for i, result in enumerate(top10, 1):
        change_emoji = "📈" if result['actual_change'] > 0 else "📉" if result['actual_change'] < 0 else "➡️"
        print(f"    {i:2d}. {change_emoji} {result['symbol']:<6} {result['name'][:35]:<35} "
              f"${result['prediction_price']:>7.2f} → ${result['actual_price']:>7.2f} "
              f"({result['actual_change_pct']:+.2f}%) Score: {result['prediction_score']:.4f}")
    
    print(f"\n🎯 TOP 20 PREDICTIONS PERFORMANCE:")
    print(f"  Gainers: {len(top20_gainers)}/{len(top20)} ({len(top20_gainers)/len(top20)*100:.1f}%)")
    print(f"  Average change: ${top20_avg_change:.4f} ({top20_avg_change_pct:+.2f}%)")
    
    # Accuracy by prediction score quartiles
    quartile_size = total_stocks // 4
    quartiles = [
        ("Top Quartile (Best Predictions)", results[:quartile_size]),
        ("2nd Quartile", results[quartile_size:quartile_size*2]),
        ("3rd Quartile", results[quartile_size*2:quartile_size*3]),
        ("Bottom Quartile (Worst Predictions)", results[quartile_size*3:]),
    ]
    
    print(f"\n📊 PERFORMANCE BY PREDICTION SCORE QUARTILES:")
    for label, quartile_results in quartiles:
        if not quartile_results:
            continue
        quartile_gainers = [r for r in quartile_results if r['actual_change'] > 0]
        quartile_avg = sum(r['actual_change_pct'] for r in quartile_results) / len(quartile_results)
        print(f"  {label}:")
        print(f"    Gainers: {len(quartile_gainers)}/{len(quartile_results)} ({len(quartile_gainers)/len(quartile_results)*100:.1f}%)")
        print(f"    Average change: {quartile_avg:+.2f}%")
    
    # Show actual top performers vs predicted
    results_by_actual = sorted(results, key=lambda x: x['actual_change_pct'], reverse=True)
    actual_top10 = results_by_actual[:10]
    
    print(f"\n🏆 ACTUAL TOP 10 PERFORMERS (vs their prediction ranks):")
    for i, result in enumerate(actual_top10, 1):
        rank_indicator = "✅" if result['rank'] <= 20 else "⚠️" if result['rank'] <= 50 else "❌"
        print(f"    {i:2d}. {rank_indicator} {result['symbol']:<6} {result['name'][:35]:<35} "
              f"({result['actual_change_pct']:+.2f}%) - Predicted rank: #{result['rank']}")
    
    print("\n" + "="*80)
    print("💡 INTERPRETATION:")
    print("   ✅ = Predicted in top 20")
    print("   ⚠️  = Predicted in top 50")
    print("   ❌ = Predicted below top 50")
    print("="*80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_predictions.py <predictions_file> [snapshot_file]")
        print("Example: python verify_predictions.py predictions_20251120_131112.txt yahooquery_stocks_20251121_145532.txt")
        print("\nIf snapshot_file is not provided, will fetch current prices from Yahoo Finance")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    snapshot_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Loading predictions from: {predictions_file}")
    predictions = load_predictions(predictions_file)
    print(f"Loaded {len(predictions)} predictions")
    
    if snapshot_file:
        print(f"Loading actual prices from snapshot: {snapshot_file}")
        actual_prices = load_snapshot_prices(snapshot_file)
        print(f"Loaded prices for {len(actual_prices)} stocks from snapshot")
    else:
        if not HAS_YFINANCE:
            print("ERROR: yfinance not available and no snapshot file provided.")
            print("Please provide a snapshot file or install yfinance: pip install yfinance")
            sys.exit(1)
        print("Fetching current prices from Yahoo Finance...")
        symbols = list(predictions.keys())
        actual_prices = fetch_current_prices(symbols)
        print(f"Fetched prices for {len(actual_prices)} stocks")
    
    # Get prediction prices
    prediction_prices = {symbol: pred['prediction_price'] for symbol, pred in predictions.items()}
    
    # Calculate results
    results = calculate_accuracy(predictions, actual_prices, prediction_prices)
    
    if not results:
        print("No matching stocks found between predictions and actual prices")
        sys.exit(1)
    
    # Analyze
    analyze_accuracy(results)
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"verification_results_{timestamp}.csv"
    
    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Rank', 'Symbol', 'Name', 'PredictionPrice', 'ActualPrice', 
                                               'Change', 'ChangePct', 'PredictionScore'])
        writer.writeheader()
        for result in results:
            writer.writerow({
                'Rank': result['rank'],
                'Symbol': result['symbol'],
                'Name': result['name'],
                'PredictionPrice': f"{result['prediction_price']:.2f}",
                'ActualPrice': f"{result['actual_price']:.2f}",
                'Change': f"{result['actual_change']:.4f}",
                'ChangePct': f"{result['actual_change_pct']:.2f}",
                'PredictionScore': f"{result['prediction_score']:.4f}",
            })
    
    print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()

