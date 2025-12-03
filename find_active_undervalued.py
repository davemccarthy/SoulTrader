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
        
        # Sort by volume (most active first) and assign volume rank
        stocks.sort(key=lambda x: x['volume'], reverse=True)
        
        # Assign volume rank (1 = highest volume, 100 = lowest volume in top 100)
        for i, stock in enumerate(stocks[:limit], 1):
            stock['volume_rank'] = i
        
        # Debug: Show top 10 by volume
        print(f"\n  Top 10 stocks by volume:")
        for i, stock in enumerate(stocks[:10], 1):
            dollar_volume = stock['volume'] * stock['price']
            print(f"    {i:2d}. {stock['symbol']:<8} Volume: {stock['volume']:>12,.0f} shares, ${dollar_volume:>12,.0f} dollar volume, Price: ${stock['price']:.2f}")
        
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
                    
                    # Get fundamental metrics (matching Yahoo advisor)
                    profit_margin = info.get('profitMargins', 0) or 0
                    yearly_change = (info.get('fiftyTwoWeekChangePercent', 0) or 0) * 100
                    market_cap = info.get('marketCap', 0) or 0
                    
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
                        'profit_margin': profit_margin,
                        'yearly_change_pct': yearly_change,
                        'market_cap': market_cap,
                        'volume': info.get('volume', 0),
                        'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
                        'volume_rank': stock.get('volume_rank', 0),  # Preserve volume rank
                    })
                else:
                    # Debug: log stocks that didn't get a notional price
                    pass  # Could add logging here if needed
            except Exception as e:
                # Debug: log errors
                pass  # Could add logging here if needed
                continue
    
    print(f"✓ Calculated notional prices for {len(results)} stocks")
    return results

def calculate_recent_trend(symbol, days=5):
    """Calculate price trend over the last N trading days."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{days}d", interval="1d")
        
        if hist.empty or len(hist) < 2:
            return None
        
        # Get first and last close prices
        first_close = hist['Close'].iloc[0]
        last_close = hist['Close'].iloc[-1]
        
        # Calculate percentage change
        trend_pct = ((last_close - first_close) / first_close) * 100
        return trend_pct
    except Exception as e:
        return None

def filter_fundamentals(results):
    """Filter stocks by fundamental metrics (matching Yahoo advisor)."""
    MIN_PROFIT_MARGIN = 0.0
    MAX_YEARLY_LOSS = -50.0
    MIN_MARKET_CAP = 25_000_000  # Lowered from 100M to 25M to be less strict
    
    filtered = []
    excluded_profit = 0
    excluded_loss = 0
    excluded_cap = 0
    
    for stock in results:
        profit_margin = stock.get('profit_margin', 0) or 0
        yearly_change = stock.get('yearly_change_pct', 0) or 0
        market_cap = stock.get('market_cap', 0) or 0
        
        # Check profit margin
        if profit_margin < MIN_PROFIT_MARGIN:
            excluded_profit += 1
            continue
        
        # Check yearly loss
        if yearly_change < MAX_YEARLY_LOSS:
            excluded_loss += 1
            continue
        
        # Check market cap
        if market_cap < MIN_MARKET_CAP:
            excluded_cap += 1
            continue
        
        filtered.append(stock)
    
    print(f"✓ Filtered to {len(filtered)} stocks passing fundamental filters")
    print(f"  Excluded: {excluded_profit} (profit margin), {excluded_loss} (yearly loss), {excluded_cap} (market cap)")
    return filtered

def filter_recent_trend(results, max_trend_pct=5.0):
    """Filter out stocks with strong upward trends over recent days."""
    filtered = []
    
    for stock in results:
        symbol = stock.get('symbol')
        if not symbol:
            continue
        
        # Calculate trend over last 5 trading days
        trend_pct = calculate_recent_trend(symbol, days=5)
        
        # If we can't calculate trend, include the stock (don't exclude due to data issues)
        if trend_pct is None:
            filtered.append(stock)
            continue
        
        # Filter out stocks with strong upward trends (> max_trend_pct)
        # Keep stocks with flat (0-2%) or slightly positive (2-5%) trends
        if trend_pct > max_trend_pct:
            continue
        
        # Store trend for reference
        stock['recent_trend_pct'] = trend_pct
        filtered.append(stock)
    
    print(f"✓ Filtered to {len(filtered)} stocks with recent trend <= {max_trend_pct}% (avoiding stocks that already ran up)")
    return filtered

def calculate_weights_and_filter(results, threshold=0.75, max_rank=100):
    """
    Calculate weights and filter stocks:
    - Volume weight: rank 100 = 1.0, rank 1 = 0.01 (inverse - lower volume rank = lower weight)
    - Ratio weight: ratio 1.0 = 1.0, ratio 0.75 = 0.75 (linear - lower ratio = higher weight)
    - Combined weight = volume_weight + ratio_weight
    - Filter: ratio <= threshold
    """
    # First, filter by threshold
    filtered = [r for r in results if r['discount_ratio'] <= threshold]
    
    # Debug: Show stocks that were excluded
    excluded = [r for r in results if r['discount_ratio'] > threshold]
    if excluded:
        excluded.sort(key=lambda x: x['discount_ratio'])
        print(f"  Excluded {len(excluded)} stocks with ratio > {threshold}")
        print(f"  Top 5 excluded (lowest ratios, closest to threshold):")
        for stock in excluded[:5]:
            print(f"    {stock['symbol']}: ratio={stock['discount_ratio']:.3f} (actual=${stock['actual_price']:.2f}, notional=${stock['notional_price']:.2f})")
    
    if not filtered:
        print(f"✓ Found 0 stocks with ratio <= {threshold}")
        return []
    
    # Calculate weights for each stock
    for stock in filtered:
        # Volume weight: rank 100 = 1.0, rank 1 = 0.01
        # Formula: weight = 0.01 + (max_rank - rank) / max_rank * (1.0 - 0.01)
        #          weight = 0.01 + (100 - rank) / 100 * 0.99
        volume_rank = stock.get('volume_rank', max_rank)  # Default to max_rank if missing
        if volume_rank > max_rank:
            volume_rank = max_rank
        volume_weight = 0.01 + (max_rank - volume_rank) / max_rank * 0.99
        
        # Ratio weight: ratio 1.0 = 1.0, ratio 0.75 = 0.75
        # Lower ratio (more undervalued) = higher weight
        # weight = 1.75 - ratio (so ratio 0.75 -> 1.0, ratio 1.0 -> 0.75)
        ratio = stock['discount_ratio']
        if ratio < 0.75:
            ratio_weight = 1.0  # Cap at 1.0 for ratios below 0.75
        elif ratio > 1.0:
            ratio_weight = 0.75  # Cap at 0.75 for ratios above 1.0
        else:
            ratio_weight = 1.75 - ratio  # Linear: 0.75->1.0, 1.0->0.75
        
        stock['volume_weight'] = volume_weight
        stock['ratio_weight'] = ratio_weight
        stock['combined_weight'] = volume_weight + ratio_weight
    
    # Sort by combined weight (highest first)
    filtered.sort(key=lambda x: x['combined_weight'], reverse=True)
    
    print(f"✓ Found {len(filtered)} stocks with ratio <= {threshold}")
    print(f"  Weight range: {min(s['combined_weight'] for s in filtered):.3f} - {max(s['combined_weight'] for s in filtered):.3f}")
    return filtered

def filter_undervalued(results, threshold=0.66):
    """Filter stocks where actual/notional <= threshold."""
    undervalued = [r for r in results if r['discount_ratio'] <= threshold]
    excluded = len(results) - len(undervalued)
    
    # Show some examples of excluded stocks (those with ratio > threshold)
    if excluded > 0 and len(results) > 0:
        excluded_stocks = [r for r in results if r['discount_ratio'] > threshold]
        excluded_stocks.sort(key=lambda x: x['discount_ratio'], reverse=True)
        print(f"  Excluded {excluded} stocks with ratio > {threshold}")
        if len(excluded_stocks) > 0:
            print(f"  Top 5 excluded (highest ratios):")
            for stock in excluded_stocks[:5]:
                print(f"    {stock['symbol']}: ratio={stock['discount_ratio']:.3f} (actual=${stock['actual_price']:.2f}, notional=${stock['notional_price']:.2f})")
    
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
    print("ACTIVE UNDERVALUED STOCKS (Notional Price Filter)")
    print("="*100)
    print(f"\nTotal stocks found: {len(undervalued)}")
    print(f"Criteria:")
    print(f"  ✓ Most active stocks")
    # print(f"  ✓ Upward trend (positive % change)")  # TEMPORARILY DISABLED
    print(f"  ✓ Fundamental filters (profit margin, yearly loss, market cap)")
    print(f"  ✓ Recent trend filter (avoiding stocks with >5% gain over last 5 days)")
    print(f"  ✓ Undervalued (actual/notional <= 0.66)")
    
    print(f"\n📊 SUMMARY:")
    avg_discount = sum(r['discount_pct'] for r in undervalued) / len(undervalued)
    avg_upside = sum(r['upside_pct'] for r in undervalued) / len(undervalued)
    avg_change = sum(r['change_pct'] for r in undervalued) / len(undervalued)
    print(f"  Average discount: {avg_discount:.2f}%")
    print(f"  Average upside potential: {avg_upside:.2f}%")
    print(f"  Average today's change: {avg_change:+.2f}%")
    
    print(f"\n🏆 UNDERVALUED STOCKS:")
    print(f"{'Rank':<6} {'Symbol':<8} {'Name':<35} {'Price':<10} {'Notional':<10} {'Ratio':<8} {'Upside%':<10} {'5d Trend%':<11} {'Method':<12}")
    print("-" * 110)
    
    for i, stock in enumerate(undervalued, 1):
        name = stock['name'][:33] + '..' if len(stock['name']) > 35 else stock['name']
        recent_trend = stock.get('recent_trend_pct', 'N/A')
        trend_str = f"{recent_trend:+.1f}%" if isinstance(recent_trend, (int, float)) else "N/A"
        print(f"{i:<6} {stock['symbol']:<8} {name:<35} "
              f"${stock['actual_price']:>8.2f} ${stock['notional_price']:>8.2f} "
              f"{stock['discount_ratio']:>6.2f} {stock['upside_pct']:>+8.1f}% "
              f"{trend_str:>10} {stock['method']:<12}")
    
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
            pe_ratio = stock.get('pe_ratio')
            if pe_ratio is not None:
                try:
                    pe_ratio_str = f"{float(pe_ratio):.2f}"
                except (ValueError, TypeError):
                    pe_ratio_str = ''
            else:
                pe_ratio_str = ''
            
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
                'PERatio': pe_ratio_str,
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
    # TEMPORARILY COMMENTED OUT to match Yahoo advisor behavior
    # upward_stocks = filter_upward_trend(stocks)
    # 
    # if not upward_stocks:
    #     print("No stocks with upward trend found")
    #     sys.exit(1)
    
    # Step 3: Calculate notional prices
    results = calculate_notional_prices(stocks)  # Use stocks instead of upward_stocks
    
    if not results:
        print("No notional prices calculated")
        sys.exit(1)
    
    # Step 4: Filter by fundamentals (matching Yahoo advisor)
    fundamental_filtered = filter_fundamentals(results)
    
    if not fundamental_filtered:
        print("No stocks passed fundamental filters")
        sys.exit(1)
    
    # Step 5: Filter out stocks with strong recent upward trends
    trend_filtered = filter_recent_trend(fundamental_filtered, max_trend_pct=5.0)
    
    if not trend_filtered:
        print("No stocks passed recent trend filter")
        sys.exit(1)
    
    # Step 6: Filter for undervalued (actual/notional <= 0.66)
    undervalued = filter_undervalued(trend_filtered, threshold=0.66)
    
    # Step 7: Display results
    display_results(undervalued)
    
    # Step 8: Save results
    if undervalued:
        save_results(undervalued)

if __name__ == "__main__":
    main()

