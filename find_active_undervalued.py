#!/usr/bin/env python
"""
Find undervalued stocks from most active stocks with upward trends.
1. Get most active stocks from Yahoo Finance
2. Filter for upward trend (positive % change)
3. Calculate notional prices
4. Filter for actual/notional <= 0.66 (undervalued)
"""

import sys
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

def get_most_active_stocks(limit=200):
    """Get most active stocks from Yahoo Finance using yf.screen()."""
    print(f"Fetching {limit} most active stocks from Yahoo Finance...")
    
    try:
        # Use yfinance screen function with a query for most active stocks
        # Similar to how yahoo.py advisor does it
        from yfinance.screener import EquityQuery as YfEquityQuery
        
        # Create query for stocks from US exchanges with minimum price
        most_active_query = YfEquityQuery(
            "and",
            [
                YfEquityQuery("is-in", ["exchange", "NMS", "NYQ"]),  # US exchanges only
                YfEquityQuery("gt", ["intradayprice", 1.0]),  # Minimum price $1 to filter out penny stocks
            ],
        )
        
        # Use yf.screen() like yahoo.py does
        # Yahoo limits query size to 250
        max_size = min(limit * 2, 250)
        response = yf.screen(
            most_active_query,
            offset=0,
            size=max_size,
            sortField="intradayprice",  # Sort by price as fallback
            sortAsc=True,
        )
        
        quotes = response.get("quotes", [])
        print(f"✓ Retrieved {len(quotes)} stocks from screener")
        
        stocks = []
        for quote in quotes:
            symbol = quote.get('symbol')
            name = quote.get('shortName') or quote.get('longName', 'N/A')
            price = quote.get('regularMarketPrice') or quote.get('intradayprice')
            previous_close = quote.get('regularMarketPreviousClose')
            volume = quote.get('volume') or quote.get('regularMarketVolume') or 0
            
            if symbol and price:
                # Calculate change percentage
                change_pct = 0.0
                if previous_close and previous_close > 0:
                    change_pct = ((price - previous_close) / previous_close) * 100
                
                stocks.append({
                    'symbol': symbol,
                    'name': name,
                    'price': float(price),
                    'change_pct': change_pct,
                    'volume': float(volume) if volume else 0.0,
                })
        
        # Sort by volume (most active first) and return top N
        stocks.sort(key=lambda x: x['volume'], reverse=True)
        return stocks[:limit]
        
    except Exception as e:
        print(f"Error fetching most active stocks: {e}")
        import traceback
        traceback.print_exc()
        return []

def filter_upward_trend(stocks):
    """Filter stocks with upward trend (positive % change)."""
    upward = [s for s in stocks if s['change_pct'] > 0]
    print(f"✓ Filtered to {len(upward)} stocks with upward trend (positive % change)")
    return upward

def calculate_notional_prices(stocks):
    """Calculate notional prices for stocks."""
    print(f"\nCalculating notional prices for {len(stocks)} stocks...")
    
    results = []
    batch_size = 20
    
    for i in range(0, len(stocks), batch_size):
        batch = stocks[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1} ({len(batch)} stocks)...")
        
        for stock in batch:
            try:
                ticker = yf.Ticker(stock['symbol'])
                info = ticker.info
                
                # Calculate notional price
                method, notional_price = calculate_best_notional_price(info)
                
                if notional_price and notional_price > 0:
                    actual_price = stock['price']
                    discount_ratio = actual_price / notional_price
                    discount_pct = (1 - discount_ratio) * 100
                    upside = notional_price - actual_price
                    upside_pct = (upside / actual_price) * 100 if actual_price > 0 else 0
                    
                    results.append({
                        'symbol': stock['symbol'],
                        'name': stock['name'],
                        'actual_price': actual_price,
                        'notional_price': notional_price,
                        'discount_ratio': discount_ratio,
                        'discount_pct': discount_pct,
                        'upside_pct': upside_pct,
                        'change_pct': stock['change_pct'],
                        'method': method,
                        'market_cap': info.get('marketCap', 0),
                        'volume': info.get('volume', 0),
                        'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
                    })
            except Exception as e:
                continue
    
    print(f"✓ Calculated notional prices for {len(results)} stocks")
    return results

def filter_undervalued(results, threshold=0.66):
    """Filter stocks where actual/notional <= threshold."""
    undervalued = [r for r in results if r['discount_ratio'] <= threshold]
    print(f"✓ Found {len(undervalued)} undervalued stocks (ratio <= {threshold})")
    return undervalued

def display_results(undervalued):
    """Display the results."""
    if not undervalued:
        print("\n❌ No undervalued stocks found matching criteria")
        return
    
    # Sort by discount ratio (most undervalued first)
    undervalued.sort(key=lambda x: x['discount_ratio'])
    
    print("\n" + "="*100)
    print("ACTIVE UNDERVALUED STOCKS (Upward Trend + Notional Price Filter)")
    print("="*100)
    print(f"\nTotal stocks found: {len(undervalued)}")
    print(f"Criteria:")
    print(f"  ✓ Most active stocks")
    print(f"  ✓ Upward trend (positive % change)")
    print(f"  ✓ Undervalued (actual/notional <= 0.66)")
    
    print(f"\n📊 SUMMARY:")
    avg_discount = sum(r['discount_pct'] for r in undervalued) / len(undervalued)
    avg_upside = sum(r['upside_pct'] for r in undervalued) / len(undervalued)
    avg_change = sum(r['change_pct'] for r in undervalued) / len(undervalued)
    print(f"  Average discount: {avg_discount:.2f}%")
    print(f"  Average upside potential: {avg_upside:.2f}%")
    print(f"  Average today's change: {avg_change:+.2f}%")
    
    print(f"\n🏆 UNDERVALUED STOCKS:")
    print(f"{'Rank':<6} {'Symbol':<8} {'Name':<35} {'Price':<10} {'Notional':<10} {'Ratio':<8} {'Upside%':<10} {'Today%':<10} {'Method':<12}")
    print("-" * 100)
    
    for i, stock in enumerate(undervalued, 1):
        name = stock['name'][:33] + '..' if len(stock['name']) > 35 else stock['name']
        print(f"{i:<6} {stock['symbol']:<8} {name:<35} "
              f"${stock['actual_price']:>8.2f} ${stock['notional_price']:>8.2f} "
              f"{stock['discount_ratio']:>6.2f} {stock['upside_pct']:>+8.1f}% "
              f"{stock['change_pct']:>+8.2f}% {stock['method']:<12}")
    
    # Show distribution by method
    method_counts = {}
    for r in undervalued:
        method = r['method']
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print(f"\n📈 VALUATION METHODS USED:")
    for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method}: {count} stocks ({count/len(undervalued)*100:.1f}%)")
    
    print("\n" + "="*100)
    print("💡 INTERPRETATION:")
    print("   - These stocks are:")
    print("     • Most active (high volume/liquidity)")
    print("     • Upward trending (positive momentum)")
    print("     • Undervalued (actual price < 66% of notional/fair value)")
    print("   - Upside% shows potential gain if stock reaches notional price")
    print("="*100)

def save_results(undervalued):
    """Save results to CSV file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"active_undervalued_{timestamp}.csv"
    
    import csv
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Symbol', 'Name', 'ActualPrice', 'NotionalPrice', 
                                               'DiscountRatio', 'DiscountPct', 'UpsidePct',
                                               'TodayChangePct', 'Method', 'MarketCap', 'Volume', 'PERatio'])
        writer.writeheader()
        for stock in sorted(undervalued, key=lambda x: x['discount_ratio']):
            writer.writerow({
                'Symbol': stock['symbol'],
                'Name': stock['name'],
                'ActualPrice': f"{stock['actual_price']:.2f}",
                'NotionalPrice': f"{stock['notional_price']:.2f}",
                'DiscountRatio': f"{stock['discount_ratio']:.4f}",
                'DiscountPct': f"{stock['discount_pct']:.2f}",
                'UpsidePct': f"{stock['upside_pct']:.2f}",
                'TodayChangePct': f"{stock['change_pct']:.2f}",
                'Method': stock['method'],
                'MarketCap': stock['market_cap'],
                'Volume': stock['volume'],
                'PERatio': f"{stock['pe_ratio']:.2f}" if stock['pe_ratio'] else '',
            })
    
    print(f"\n💾 Results saved to: {filename}")
    return filename

def main():
    print("="*80)
    print("ACTIVE UNDERVALUED STOCKS FINDER")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Get most active stocks
    stocks = get_most_active_stocks(limit=200)
    
    if not stocks:
        print("No stocks retrieved")
        sys.exit(1)
    
    # Step 2: Filter for upward trend
    upward_stocks = filter_upward_trend(stocks)
    
    if not upward_stocks:
        print("No stocks with upward trend found")
        sys.exit(1)
    
    # Step 3: Calculate notional prices
    results = calculate_notional_prices(upward_stocks)
    
    if not results:
        print("No notional prices calculated")
        sys.exit(1)
    
    # Step 4: Filter for undervalued (actual/notional <= 0.66)
    undervalued = filter_undervalued(results, threshold=0.66)
    
    # Step 5: Display results
    display_results(undervalued)
    
    # Step 6: Save results
    if undervalued:
        save_results(undervalued)

if __name__ == "__main__":
    main()

