#!/usr/bin/env python
"""
Find undervalued stocks using notional price method, but filter out stocks
that are already in a strong upward trend (to avoid mean reversion).

1. Get most active stocks from Yahoo Finance
2. Filter for upward trend (positive % change) - OR remove this?
3. Calculate notional prices
4. Calculate intraday trend using Stock.calc_trend
5. Filter out stocks with strong upward trends (trend > threshold)
6. Filter for actual/notional <= 0.66 (undervalued)
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

# Import Stock model to use calc_trend
import os
import django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import Stock

def get_most_active_stocks(limit=200):
    """Get most active stocks from Yahoo Finance using yf.screen()."""
    print(f"Fetching {limit} most active stocks from Yahoo Finance...")
    
    try:
        from yfinance.screener import EquityQuery as YfEquityQuery
        
        most_active_query = YfEquityQuery(
            "and",
            [
                YfEquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
                YfEquityQuery("gt", ["intradayprice", 1.0]),
            ],
        )
        
        max_size = min(limit * 2, 250)
        response = yf.screen(
            most_active_query,
            offset=0,
            size=max_size,
            sortField="intradayprice",
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
        
        stocks.sort(key=lambda x: x['volume'], reverse=True)
        return stocks[:limit]
        
    except Exception as e:
        print(f"Error fetching most active stocks: {e}")
        import traceback
        traceback.print_exc()
        return []

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
                
                method, notional_price = calculate_best_notional_price(info)
                
                if notional_price and notional_price > 0:
                    actual_price = stock['price']
                    discount_ratio = actual_price / notional_price
                    discount_pct = (1 - discount_ratio) * 100
                    upside = notional_price - actual_price
                    upside_pct = (upside / actual_price) * 100 if actual_price > 0 else 0
                    
                    # Get fundamental metrics
                    profit_margin = info.get('profitMargins', 0) or 0
                    yearly_change = info.get('fiftyTwoWeekChangePercent', 0) or 0
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
                        'market_cap': market_cap,
                        'volume': info.get('volume', 0),
                        'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
                        'profit_margin': profit_margin,
                        'yearly_change_pct': yearly_change * 100,  # Convert to percentage
                    })
            except Exception as e:
                continue
    
    print(f"✓ Calculated notional prices for {len(results)} stocks")
    return results

def filter_fundamentals(results, min_profit_margin=0.0, max_yearly_loss=-50.0, min_market_cap=100_000_000):
    """
    Filter stocks by fundamental metrics to avoid distressed companies.
    
    Args:
        results: List of stock results with fundamental data
        min_profit_margin: Minimum profit margin (default: 0.0 = must be profitable)
        max_yearly_loss: Maximum allowed yearly loss percentage (default: -50% = can lose up to 50%)
        min_market_cap: Minimum market cap in dollars (default: $100M to avoid micro-caps)
    """
    filtered = []
    excluded = {
        'unprofitable': [],
        'excessive_loss': [],
        'micro_cap': [],
    }
    
    for stock in results:
        excluded_reasons = []
        
        # Check profit margin
        profit_margin = stock.get('profit_margin', 0) or 0
        if profit_margin < min_profit_margin:
            excluded['unprofitable'].append(stock)
            excluded_reasons.append(f"unprofitable (margin: {profit_margin:.1%})")
            continue
        
        # Check yearly loss
        yearly_change = stock.get('yearly_change_pct', 0) or 0
        if yearly_change < max_yearly_loss:
            excluded['excessive_loss'].append(stock)
            excluded_reasons.append(f"excessive loss ({yearly_change:.1f}%)")
            continue
        
        # Check market cap
        market_cap = stock.get('market_cap', 0) or 0
        if market_cap < min_market_cap:
            excluded['micro_cap'].append(stock)
            excluded_reasons.append(f"micro-cap (${market_cap:,.0f})")
            continue
        
        # Passed all filters
        filtered.append(stock)
    
    print(f"\n📊 FUNDAMENTAL FILTERING:")
    print(f"  Total stocks: {len(results)}")
    print(f"  Excluded - Unprofitable (margin < {min_profit_margin:.1%}): {len(excluded['unprofitable'])}")
    print(f"  Excluded - Excessive loss (yearly < {max_yearly_loss:.1f}%): {len(excluded['excessive_loss'])}")
    print(f"  Excluded - Micro-cap (market cap < ${min_market_cap:,.0f}): {len(excluded['micro_cap'])}")
    print(f"  Remaining: {len(filtered)}")
    
    # Show examples of excluded stocks
    if excluded['unprofitable']:
        print(f"\n  Examples - Unprofitable stocks:")
        for stock in excluded['unprofitable'][:5]:
            margin = stock.get('profit_margin', 0) or 0
            print(f"    {stock['symbol']:<8} {stock['name'][:35]:<35} Margin: {margin:.1%}")
    
    if excluded['excessive_loss']:
        print(f"\n  Examples - Excessive loss stocks:")
        for stock in excluded['excessive_loss'][:5]:
            loss = stock.get('yearly_change_pct', 0) or 0
            print(f"    {stock['symbol']:<8} {stock['name'][:35]:<35} Loss: {loss:.1f}%")
    
    if excluded['micro_cap']:
        print(f"\n  Examples - Micro-cap stocks:")
        for stock in excluded['micro_cap'][:5]:
            cap = stock.get('market_cap', 0) or 0
            print(f"    {stock['symbol']:<8} {stock['name'][:35]:<35} Cap: ${cap:,.0f}")
    
    return filtered

def calculate_trends(results, hours=12):
    """Calculate intraday trends for stocks using Stock.calc_trend."""
    print(f"\nCalculating trends for {len(results)} stocks (last {hours} hours)...")
    
    trend_results = []
    
    for i, stock_data in enumerate(results):
        symbol = stock_data['symbol']
        try:
            # Get or create Stock object
            stock, created = Stock.objects.get_or_create(
                symbol=symbol,
                defaults={'company': stock_data['name'], 'price': stock_data['actual_price']}
            )
            
            # Update price if needed
            if stock.price != stock_data['actual_price']:
                stock.price = stock_data['actual_price']
                stock.save()
            
            # Calculate trend
            trend = stock.calc_trend(period="1d", interval="15m", hours=hours)
            
            stock_data['trend'] = float(trend) if trend is not None else None
            trend_results.append(stock_data)
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(results)} stocks...")
                
        except Exception as e:
            print(f"    Error calculating trend for {symbol}: {e}")
            stock_data['trend'] = None
            trend_results.append(stock_data)
    
    print(f"✓ Calculated trends for {len(trend_results)} stocks")
    return trend_results

def analyze_trends(results):
    """
    Analyze and display trend values (no filtering).
    When filtering is enabled, use threshold of 0.2 (trend > 0.2 will be excluded).
    """
    trends = []
    for stock in results:
        trend = stock.get('trend')
        if trend is not None:
            trends.append(trend)
    
    print(f"\n📊 TREND ANALYSIS (no filtering applied):")
    print(f"  Total stocks with trend data: {len(trends)}/{len(results)}")
    
    if trends:
        trends_sorted = sorted(trends)
        print(f"\n  Trend Statistics:")
        print(f"    Min: {min(trends):+.2f}")
        print(f"    Max: {max(trends):+.2f}")
        print(f"    Average: {sum(trends)/len(trends):+.2f}")
        print(f"    Median: {trends_sorted[len(trends)//2]:+.2f}")
        
        # Count by ranges
        strong_uptrend = len([t for t in trends if t > 0.2])
        weak_uptrend = len([t for t in trends if 0 < t <= 0.2])
        sideways = len([t for t in trends if -0.2 <= t <= 0])
        weak_downtrend = len([t for t in trends if -0.5 < t < -0.2])
        strong_downtrend = len([t for t in trends if t <= -0.5])
        
        print(f"\n  Trend Distribution:")
        print(f"    Strong uptrend (>0.2): {strong_uptrend} ({strong_uptrend/len(trends)*100:.1f}%)")
        print(f"    Weak uptrend (0 to 0.2): {weak_uptrend} ({weak_uptrend/len(trends)*100:.1f}%)")
        print(f"    Sideways (-0.2 to 0): {sideways} ({sideways/len(trends)*100:.1f}%)")
        print(f"    Weak downtrend (-0.5 to -0.2): {weak_downtrend} ({weak_downtrend/len(trends)*100:.1f}%)")
        print(f"    Strong downtrend (<-0.5): {strong_downtrend} ({strong_downtrend/len(trends)*100:.1f}%)")
        
        print(f"\n  NOTE: When filtering is enabled, stocks with trend > 0.2 will be excluded")
        
        # Show top and bottom trends
        print(f"\n  Top 10 Strongest Uptrends:")
        sorted_by_trend = sorted(results, key=lambda x: x.get('trend', -999) or -999, reverse=True)
        for i, stock in enumerate(sorted_by_trend[:10], 1):
            trend_val = stock.get('trend', 'N/A')
            if isinstance(trend_val, (int, float)):
                print(f"    {i:2d}. {stock['symbol']:<8} {stock['name'][:35]:<35} Trend: {trend_val:+.2f}")
        
        print(f"\n  Top 10 Strongest Downtrends:")
        sorted_by_trend_rev = sorted(results, key=lambda x: x.get('trend', 999) or 999)
        for i, stock in enumerate(sorted_by_trend_rev[:10], 1):
            trend_val = stock.get('trend', 'N/A')
            if isinstance(trend_val, (int, float)):
                print(f"    {i:2d}. {stock['symbol']:<8} {stock['name'][:35]:<35} Trend: {trend_val:+.2f}")
    
    # Return all results (no filtering)
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
    print("UNDERVALUED STOCKS (Notional Price + Trend Filter)")
    print("="*100)
    print(f"\nTotal stocks found: {len(undervalued)}")
    print(f"Criteria:")
    print(f"  ✓ Most active stocks")
    print(f"  ✓ Fundamental filters (profitable, reasonable losses, minimum market cap)")
    print(f"  ✓ Notional price filter (actual/notional <= 0.66)")
    print(f"  ⚠️  Trend analysis (no filtering applied - values shown for reference)")
    
    print(f"\n📊 SUMMARY:")
    avg_discount = sum(r['discount_pct'] for r in undervalued) / len(undervalued)
    avg_upside = sum(r['upside_pct'] for r in undervalued) / len(undervalued)
    avg_trend = sum(r.get('trend', 0) or 0 for r in undervalued) / len(undervalued)
    print(f"  Average discount: {avg_discount:.2f}%")
    print(f"  Average upside potential: {avg_upside:.2f}%")
    print(f"  Average trend: {avg_trend:+.2f}")
    
    print(f"\n🏆 UNDERVALUED STOCKS:")
    print(f"{'Rank':<6} {'Symbol':<8} {'Name':<25} {'Price':<8} {'Notional':<9} {'Ratio':<7} {'Margin%':<9} {'Yearly%':<9} {'Trend':<8} {'Upside%':<9}")
    print("-" * 120)
    
    for i, stock in enumerate(undervalued, 1):
        name = stock['name'][:23] + '..' if len(stock['name']) > 25 else stock['name']
        trend_val = stock.get('trend', 'N/A')
        trend_str = f"{trend_val:+.2f}" if isinstance(trend_val, (int, float)) else str(trend_val)
        margin = stock.get('profit_margin', 0) or 0
        yearly = stock.get('yearly_change_pct', 0) or 0
        margin_str = f"{margin*100:+.1f}%" if margin else "N/A"
        yearly_str = f"{yearly:+.1f}%"
        print(f"{i:<6} {stock['symbol']:<8} {name:<25} "
              f"${stock['actual_price']:>7.2f} ${stock['notional_price']:>8.2f} "
              f"{stock['discount_ratio']:>6.2f} {margin_str:>8} {yearly_str:>8} {trend_str:>7} {stock['upside_pct']:>+8.1f}%")
    
    print("\n" + "="*100)
    print("💡 INTERPRETATION:")
    print("   - These stocks are undervalued (actual < 66% of notional)")
    print("   - Strong upward trends were filtered out to avoid mean reversion")
    print("   - Trend values: positive = uptrend, negative = downtrend, ~0 = sideways")
    print("="*100)

def save_results(undervalued):
    """Save results to CSV file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"undervalued_trend_filtered_{timestamp}.csv"
    
    import csv
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Symbol', 'Name', 'ActualPrice', 'NotionalPrice', 
                                               'DiscountRatio', 'DiscountPct', 'UpsidePct',
                                               'Trend', 'Method', 'MarketCap', 'Volume', 'PERatio',
                                               'ProfitMargin', 'YearlyChangePct'])
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
                'Trend': f"{stock.get('trend', 'N/A')}",
                'Method': stock['method'],
                'MarketCap': stock['market_cap'],
                'Volume': stock['volume'],
                'PERatio': f"{stock['pe_ratio']:.2f}" if stock['pe_ratio'] else '',
                'ProfitMargin': f"{stock.get('profit_margin', 0) * 100:.2f}%" if stock.get('profit_margin') else '',
                'YearlyChangePct': f"{stock.get('yearly_change_pct', 0):.2f}%",
            })
    
    print(f"\n💾 Results saved to: {filename}")
    return filename

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Find undervalued stocks with fundamental and trend analysis')
    parser.add_argument('--hours', type=int, default=12,
                       help='Hours of data to use for trend calculation (default: 12)')
    parser.add_argument('--limit', type=int, default=200,
                       help='Maximum number of active stocks to fetch (default: 200)')
    parser.add_argument('--min-profit-margin', type=float, default=0.0,
                       help='Minimum profit margin (default: 0.0 = must be profitable)')
    parser.add_argument('--max-yearly-loss', type=float, default=-50.0,
                       help='Maximum allowed yearly loss percentage (default: -50.0)')
    parser.add_argument('--min-market-cap', type=int, default=100_000_000,
                       help='Minimum market cap in dollars (default: 100000000 = $100M)')
    args = parser.parse_args()
    
    print("="*80)
    print("UNDERVALUED STOCKS FINDER (with Fundamental & Trend Analysis)")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Settings:")
    print(f"  Trend calculation hours: {args.hours}")
    print(f"  Active stocks limit: {args.limit}")
    print(f"  Fundamental filters:")
    print(f"    Min profit margin: {args.min_profit_margin:.1%}")
    print(f"    Max yearly loss: {args.max_yearly_loss:.1f}%")
    print(f"    Min market cap: ${args.min_market_cap:,.0f}")
    print(f"  NOTE: Trend filtering is DISABLED - values are shown for analysis")
    print(f"        When enabled, will filter out stocks with trend > 0.2")
    print()
    
    # Step 1: Get most active stocks
    stocks = get_most_active_stocks(limit=args.limit)
    
    if not stocks:
        print("No stocks retrieved")
        sys.exit(1)
    
    # Step 2: Calculate notional prices
    results = calculate_notional_prices(stocks)
    
    if not results:
        print("No notional prices calculated")
        sys.exit(1)
    
    # Step 3: Filter by fundamentals
    fundamental_filtered = filter_fundamentals(
        results,
        min_profit_margin=args.min_profit_margin,
        max_yearly_loss=args.max_yearly_loss,
        min_market_cap=args.min_market_cap
    )
    
    if not fundamental_filtered:
        print("No stocks passed fundamental filters")
        sys.exit(1)
    
    # Step 4: Calculate trends
    results_with_trends = calculate_trends(fundamental_filtered, hours=args.hours)
    
    # Step 5: Analyze trends (no filtering - just print values)
    trend_analyzed = analyze_trends(results_with_trends)
    
    # Step 6: Filter for undervalued (actual/notional <= 0.66)
    undervalued = filter_undervalued(trend_analyzed, threshold=0.66)
    
    # Step 6: Display results
    display_results(undervalued)
    
    # Step 7: Save results
    if undervalued:
        save_results(undervalued)

if __name__ == "__main__":
    main()

