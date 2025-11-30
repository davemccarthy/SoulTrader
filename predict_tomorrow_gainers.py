#!/usr/bin/env python
"""
Predict tomorrow's gainers based on today's yahooquery data.
Uses insights from yesterday's analysis:
1. Higher volume = more likely to gain
2. Larger market cap = more likely to gain  
3. Higher volatility (beta) = more likely to gain

DISCLAIMER: This is for fun/educational purposes only!
"""

import csv
import sys
import yfinance as yf
from datetime import datetime

def load_today_snapshot(filename):
    """Load today's stock snapshot."""
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
                    'price': price
                })
            except (ValueError, KeyError):
                continue
    return stocks

def fetch_prediction_metrics(stocks):
    """Fetch metrics needed for prediction."""
    print(f"Fetching metrics for {len(stocks)} stocks...")
    
    symbols = [s['symbol'] for s in stocks]
    metrics = {}
    
    batch_size = 20
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1} ({len(batch)} stocks)...")
        
        for symbol in batch:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                market_cap = info.get('marketCap', 0) or 0
                volume = info.get('volume', 0) or info.get('averageVolume', 0) or 0
                beta = info.get('beta', 1.0) or 1.0
                
                metrics[symbol] = {
                    'market_cap': market_cap,
                    'volume': volume,
                    'beta': beta,
                }
            except Exception as e:
                metrics[symbol] = {
                    'market_cap': 0,
                    'volume': 0,
                    'beta': 1.0,
                }
    
    return metrics

def calculate_prediction_scores(stocks, metrics):
    """Calculate prediction scores based on insights."""
    # Normalize metrics for scoring
    market_caps = [metrics.get(s['symbol'], {}).get('market_cap', 0) for s in stocks]
    volumes = [metrics.get(s['symbol'], {}).get('volume', 0) for s in stocks]
    betas = [metrics.get(s['symbol'], {}).get('beta', 1.0) for s in stocks]
    
    max_cap = max(market_caps) if market_caps else 1
    max_vol = max(volumes) if volumes else 1
    max_beta = max(betas) if betas else 1
    
    for stock in stocks:
        symbol = stock['symbol']
        stock_metrics = metrics.get(symbol, {})
        
        market_cap = stock_metrics.get('market_cap', 0)
        volume = stock_metrics.get('volume', 0)
        beta = stock_metrics.get('beta', 1.0)
        
        # Normalize to 0-1 scale
        cap_score = (market_cap / max_cap) if max_cap > 0 else 0
        vol_score = (volume / max_vol) if max_vol > 0 else 0
        beta_score = (beta / max_beta) if max_beta > 0 else 0
        
        # Weighted combination based on insights:
        # - Volume: 40% (strongest indicator from yesterday)
        # - Market Cap: 35% (second strongest)
        # - Beta: 25% (volatility helps)
        prediction_score = (
            vol_score * 0.40 +
            cap_score * 0.35 +
            beta_score * 0.25
        )
        
        stock['prediction_score'] = prediction_score
        stock['market_cap'] = market_cap
        stock['volume'] = volume
        stock['beta'] = beta
        stock['cap_score'] = cap_score
        stock['vol_score'] = vol_score
        stock['beta_score'] = beta_score
    
    return stocks

def display_predictions(stocks):
    """Display top predictions."""
    # Sort by prediction score
    stocks.sort(key=lambda x: x['prediction_score'], reverse=True)
    
    print("\n" + "="*80)
    print("TOMORROW'S PREDICTED TOP GAINERS (Based on Yesterday's Patterns)")
    print("="*80)
    print("\n⚠️  DISCLAIMER: This is for educational/fun purposes only!")
    print("   Past performance patterns don't guarantee future results.\n")
    
    print("📊 PREDICTION METHODOLOGY:")
    print("   Based on yesterday's top 10 vs bottom 10 analysis:")
    print("   - Volume (40% weight): Winners had 3.6x more volume")
    print("   - Market Cap (35% weight): Winners were 2x larger")
    print("   - Beta/Volatility (25% weight): Winners had higher beta")
    print()
    
    print("🏆 TOP 20 PREDICTED GAINERS:")
    print(f"{'Rank':<6} {'Symbol':<8} {'Name':<40} {'Score':<8} {'Cap':<12} {'Vol':<12} {'Beta':<8} {'Price':<10}")
    print("-" * 100)
    
    for i, stock in enumerate(stocks[:20], 1):
        cap_str = f"${stock['market_cap']/1e9:.2f}B" if stock['market_cap'] > 1e9 else f"${stock['market_cap']/1e6:.0f}M"
        vol_str = f"{stock['volume']/1e6:.1f}M" if stock['volume'] > 1e6 else f"{stock['volume']/1e3:.0f}K"
        name = stock['name'][:38] + '..' if len(stock['name']) > 40 else stock['name']
        
        print(f"{i:<6} {stock['symbol']:<8} {name:<40} "
              f"{stock['prediction_score']:.4f} {cap_str:<12} {vol_str:<12} "
              f"{stock['beta']:.2f} ${stock['price']:.2f}")
    
    print("\n" + "="*80)
    print("💡 REMINDER: Check back tomorrow to see how accurate these predictions were!")
    print("="*80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_tomorrow_gainers.py <today_snapshot_file>")
        print("Example: python predict_tomorrow_gainers.py yahooquery_stocks_20251120_125440.txt")
        sys.exit(1)
    
    snapshot_file = sys.argv[1]
    
    print(f"Loading today's snapshot: {snapshot_file}")
    stocks = load_today_snapshot(snapshot_file)
    print(f"Loaded {len(stocks)} stocks")
    
    metrics = fetch_prediction_metrics(stocks)
    stocks = calculate_prediction_scores(stocks, metrics)
    display_predictions(stocks)
    
    # Save predictions for tomorrow's verification
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prediction_file = f"predictions_{timestamp}.txt"
    
    with open(prediction_file, 'w') as f:
        f.write("Symbol,Name,Price,PredictionScore,MarketCap,Volume,Beta\n")
        for stock in sorted(stocks, key=lambda x: x['prediction_score'], reverse=True):
            f.write(f"{stock['symbol']},{stock['name']},{stock['price']:.2f},"
                   f"{stock['prediction_score']:.4f},{stock['market_cap']},{stock['volume']},{stock['beta']:.2f}\n")
    
    print(f"\n💾 Predictions saved to: {prediction_file}")
    print(f"   Compare tomorrow with: python compare_yahooquery_prices.py {snapshot_file}")

if __name__ == "__main__":
    main()







