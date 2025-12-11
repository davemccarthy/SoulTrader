#!/usr/bin/env python
"""
Analyze if predicted stocks were already gainers on the previous day.
This helps understand if mean reversion explains the poor performance.
"""

import csv
import sys

def load_prices(filename):
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

def load_predictions(filename):
    """Load predictions."""
    predictions = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                symbol = row['Symbol'].strip()
                predictions[symbol] = {
                    'symbol': symbol,
                    'name': row['Name'].strip(),
                    'pred_price': float(row['Price'].strip()),
                    'score': float(row['PredictionScore'].strip()),
                }
            except (ValueError, KeyError):
                continue
    return predictions

def main():
    if len(sys.argv) < 4:
        print("Usage: python analyze_prediction_context.py <predictions_file> <day_before_file> <prediction_day_file>")
        print("Example: python analyze_prediction_context.py predictions_20251120_131112.txt yahooquery_stocks_20251119_074126.txt yahooquery_stocks_20251120_125440.txt")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    day_before_file = sys.argv[2]  # Nov 19
    prediction_day_file = sys.argv[3]  # Nov 20
    
    print("Loading predictions...")
    predictions = load_predictions(predictions_file)
    print(f"Loaded {len(predictions)} predictions")
    
    print("Loading prices...")
    day_before_prices = load_prices(day_before_file)
    prediction_day_prices = load_prices(prediction_day_file)
    print(f"Loaded prices for {len(day_before_prices)} stocks from day before")
    print(f"Loaded prices for {len(prediction_day_prices)} stocks from prediction day")
    
    # Analyze each predicted stock
    results = []
    for symbol, pred in predictions.items():
        if symbol not in day_before_prices or symbol not in prediction_day_prices:
            continue
        
        day_before_price = day_before_prices[symbol]
        prediction_day_price = pred['pred_price']
        next_day_price = prediction_day_prices.get(symbol)
        
        # Calculate change from day before to prediction day
        change_day_before_to_pred = ((prediction_day_price - day_before_price) / day_before_price) * 100 if day_before_price > 0 else 0
        
        # Calculate change from prediction day to next day
        change_pred_to_next = ((next_day_price - prediction_day_price) / prediction_day_price) * 100 if prediction_day_price > 0 and next_day_price else None
        
        results.append({
            'symbol': symbol,
            'name': pred['name'],
            'day_before_price': day_before_price,
            'pred_price': prediction_day_price,
            'next_day_price': next_day_price,
            'change_day_before_to_pred': change_day_before_to_pred,
            'change_pred_to_next': change_pred_to_next,
            'score': pred['score'],
        })
    
    # Sort by prediction score (most undervalued first)
    results.sort(key=lambda x: x['score'])
    
    # Analyze
    print("\n" + "="*100)
    print("PREDICTION CONTEXT ANALYSIS")
    print("="*100)
    print(f"\nTotal predicted stocks analyzed: {len(results)}")
    
    # How many were gainers on Nov 19 (day before prediction)?
    gainers_day_before = [r for r in results if r['change_day_before_to_pred'] > 0]
    losers_day_before = [r for r in results if r['change_day_before_to_pred'] < 0]
    
    print(f"\n📊 PERFORMANCE ON DAY BEFORE PREDICTION (Nov 19):")
    print(f"  Gainers: {len(gainers_day_before)} ({len(gainers_day_before)/len(results)*100:.1f}%)")
    print(f"  Losers: {len(losers_day_before)} ({len(losers_day_before)/len(results)*100:.1f}%)")
    
    if gainers_day_before:
        avg_gain = sum(r['change_day_before_to_pred'] for r in gainers_day_before) / len(gainers_day_before)
        print(f"  Average gain: {avg_gain:+.2f}%")
    
    if losers_day_before:
        avg_loss = sum(r['change_day_before_to_pred'] for r in losers_day_before) / len(losers_day_before)
        print(f"  Average loss: {avg_loss:+.2f}%")
    
    # How did they perform the next day?
    print(f"\n📊 PERFORMANCE ON PREDICTION DAY TO NEXT DAY (Nov 20 → Nov 21):")
    next_day_results = [r for r in results if r['change_pred_to_next'] is not None]
    next_day_gainers = [r for r in next_day_results if r['change_pred_to_next'] > 0]
    next_day_losers = [r for r in next_day_results if r['change_pred_to_next'] < 0]
    
    print(f"  Gainers: {len(next_day_gainers)} ({len(next_day_gainers)/len(next_day_results)*100:.1f}%)")
    print(f"  Losers: {len(next_day_losers)} ({len(next_day_losers)/len(next_day_results)*100:.1f}%)")
    
    if next_day_results:
        avg_next_day = sum(r['change_pred_to_next'] for r in next_day_results) / len(next_day_results)
        print(f"  Average change: {avg_next_day:+.2f}%")
    
    # Key insight: Did stocks that were gainers on Nov 19 continue to gain on Nov 20?
    print(f"\n💡 MEAN REVERSION ANALYSIS:")
    gainers_that_continued = [r for r in gainers_day_before if r['change_pred_to_next'] and r['change_pred_to_next'] > 0]
    gainers_that_reverted = [r for r in gainers_day_before if r['change_pred_to_next'] and r['change_pred_to_next'] < 0]
    
    print(f"  Stocks that gained on Nov 19:")
    print(f"    Continued gaining on Nov 20: {len(gainers_that_continued)}/{len([r for r in gainers_day_before if r['change_pred_to_next'] is not None])} ({len(gainers_that_continued)/len([r for r in gainers_day_before if r['change_pred_to_next'] is not None])*100:.1f}%)")
    print(f"    Reverted (lost) on Nov 20: {len(gainers_that_reverted)}/{len([r for r in gainers_day_before if r['change_pred_to_next'] is not None])} ({len(gainers_that_reverted)/len([r for r in gainers_day_before if r['change_pred_to_next'] is not None])*100:.1f}%)")
    
    if gainers_that_continued:
        avg_continued = sum(r['change_pred_to_next'] for r in gainers_that_continued) / len(gainers_that_continued)
        print(f"    Average gain for continuers: {avg_continued:+.2f}%")
    
    if gainers_that_reverted:
        avg_reverted = sum(r['change_pred_to_next'] for r in gainers_that_reverted) / len(gainers_that_reverted)
        print(f"    Average loss for reverters: {avg_reverted:+.2f}%")
    
    # Show top predicted stocks and their context
    print(f"\n📋 TOP 20 PREDICTED STOCKS - CONTEXT:")
    print(f"{'Rank':<6} {'Symbol':<8} {'Nov19→Nov20':<12} {'Nov20→Nov21':<12} {'Score':<8}")
    print("-" * 100)
    
    for i, result in enumerate(results[:20], 1):
        day_before_str = f"{result['change_day_before_to_pred']:+.2f}%"
        next_day_str = f"{result['change_pred_to_next']:+.2f}%" if result['change_pred_to_next'] is not None else "N/A"
        print(f"{i:<6} {result['symbol']:<8} {day_before_str:<12} {next_day_str:<12} {result['score']:<8.4f}")
    
    print("\n" + "="*100)
    print("💡 INTERPRETATION:")
    print("   If most predicted stocks were gainers on Nov 19, mean reversion explains")
    print("   why they didn't continue to be the biggest gainers on Nov 20.")
    print("="*100)

if __name__ == "__main__":
    main()















































