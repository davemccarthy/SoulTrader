#!/usr/bin/env python
"""
Calculate notional/fair value share prices using multiple valuation methods
and compare with actual prices to identify undervalued stocks.
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

def calculate_notional_price_dcf(ticker_info):
    """
    Calculate notional price using DCF method (simplified).
    Uses free cash flow and growth assumptions.
    """
    try:
        fcf = ticker_info.get('freeCashflow') or ticker_info.get('operatingCashflow')
        shares = ticker_info.get('sharesOutstanding')
        
        if not fcf or not shares or fcf <= 0 or shares <= 0:
            return None
        
        # Simplified DCF: assume 5% growth, 10% discount rate, 10x terminal multiple
        growth_rate = 0.05
        discount_rate = 0.10
        terminal_multiple = 10.0
        years = 5
        
        # Project FCF for 5 years
        pv_fcf = 0
        for year in range(1, years + 1):
            future_fcf = fcf * ((1 + growth_rate) ** year)
            pv_fcf += future_fcf / ((1 + discount_rate) ** year)
        
        # Terminal value
        terminal_fcf = fcf * ((1 + growth_rate) ** years)
        terminal_value = terminal_fcf * terminal_multiple
        pv_terminal = terminal_value / ((1 + discount_rate) ** years)
        
        equity_value = pv_fcf + pv_terminal
        notional_price = equity_value / shares
        
        return notional_price
    except:
        return None

def calculate_notional_price_pe(ticker_info, sector_pe=None):
    """
    Calculate notional price using P/E multiple method.
    Uses company earnings and sector/industry P/E ratio.
    IMPORTANT: Uses sector/industry average, NOT company's own P/E.
    """
    try:
        eps = ticker_info.get('trailingEps') or ticker_info.get('forwardEps')
        shares = ticker_info.get('sharesOutstanding')
        
        if not eps or not shares or eps <= 0 or shares <= 0:
            return None
        
        # DON'T use company's own P/E - that would just give us current price
        # Instead, use sector average or market average
        # Sector P/E averages (refined based on current market conditions):
        sector_pe_map = {
            'Technology': 28.0,  # Higher for growth tech
            'Healthcare': 22.0,  # Stable, slightly above market
            'Financial Services': 13.0,  # Lower due to interest rate sensitivity
            'Consumer Cyclical': 19.0,  # Moderate growth
            'Consumer Defensive': 21.0,  # Stable, defensive premium
            'Energy': 11.0,  # Volatile, lower multiples
            'Industrials': 19.0,  # Moderate growth
            'Basic Materials': 16.0,  # Cyclical, moderate
            'Real Estate': 18.0,  # REITs typically lower
            'Utilities': 16.0,  # Stable but lower growth
            'Communication Services': 16.0,  # Mixed, moderate
        }
        
        sector = ticker_info.get('sector', '')
        pe_ratio = sector_pe_map.get(sector) or sector_pe or 18.0  # Default to market average
        
        notional_price = eps * pe_ratio
        return notional_price
    except:
        return None

def calculate_notional_price_ev_ebitda(ticker_info, sector_multiple=None):
    """
    Calculate notional price using EV/EBITDA multiple method.
    """
    try:
        ebitda = ticker_info.get('ebitda')
        shares = ticker_info.get('sharesOutstanding')
        market_cap = ticker_info.get('marketCap')
        total_debt = ticker_info.get('totalDebt') or 0
        cash = ticker_info.get('totalCash') or 0
        
        if not ebitda or not shares or ebitda <= 0 or shares <= 0:
            return None
        
        # Use company's own EV/EBITDA if available, otherwise use sector average
        ev_ebitda = ticker_info.get('enterpriseToEbitda')
        
        if not ev_ebitda:
            # Use sector multiple if provided, otherwise default to 10x
            ev_ebitda = sector_multiple or 10.0
        
        # Calculate Enterprise Value
        enterprise_value = ebitda * ev_ebitda
        
        # Convert to Equity Value
        net_debt = total_debt - cash
        equity_value = enterprise_value - net_debt
        
        # If we have market cap, use it to validate/adjust
        if market_cap and market_cap > 0:
            # Use a blend: 70% calculated, 30% market-based
            equity_value = equity_value * 0.7 + market_cap * 0.3
        
        notional_price = equity_value / shares
        return notional_price
    except:
        return None

def calculate_notional_price_revenue(ticker_info, sector_multiple=None):
    """
    Calculate notional price using revenue multiple method.
    Good for growth companies without earnings.
    """
    try:
        # Get revenue per share if available, otherwise calculate from total revenue
        revenue_per_share = ticker_info.get('revenuePerShare')
        shares = ticker_info.get('sharesOutstanding')
        total_revenue = ticker_info.get('totalRevenue')
        
        if not revenue_per_share and total_revenue and shares and shares > 0:
            revenue_per_share = total_revenue / shares
        
        if not revenue_per_share or revenue_per_share <= 0:
            return None
        
        # Use price-to-sales ratio (default 2.0 for growth companies)
        # But validate it's reasonable (between 0.5 and 20)
        ps_ratio = ticker_info.get('priceToSalesTrailing12Months')
        if not ps_ratio or ps_ratio < 0.5 or ps_ratio > 20:
            ps_ratio = sector_multiple or 2.0
        
        notional_price = revenue_per_share * ps_ratio
        
        # Sanity check: notional price should be reasonable (not billions)
        if notional_price > 10000:  # If > $10,000, something is wrong
            return None
        
        return notional_price
    except:
        pass
    
    return None

def calculate_notional_price_book(ticker_info):
    """
    Calculate notional price using price-to-book method.
    Good for financial companies.
    """
    try:
        book_value = ticker_info.get('bookValue')
        shares = ticker_info.get('sharesOutstanding')
        
        if not book_value or not shares or book_value <= 0 or shares <= 0:
            return None
        
        book_per_share = book_value / shares
        
        # Use company's P/B if available, otherwise use sector average (default 1.5)
        pb_ratio = ticker_info.get('priceToBook') or 1.5
        
        notional_price = book_per_share * pb_ratio
        return notional_price
    except:
        return None

def calculate_best_notional_price(ticker_info):
    """
    Calculate notional price using the best available method.
    Tries methods in order of reliability.
    """
    methods = []
    actual_price = ticker_info.get('currentPrice') or ticker_info.get('regularMarketPrice') or 0
    
    # Try EV/EBITDA first (most reliable for most companies)
    ev_ebitda_price = calculate_notional_price_ev_ebitda(ticker_info)
    if ev_ebitda_price and ev_ebitda_price > 0 and ev_ebitda_price < actual_price * 10:  # Sanity check
        methods.append(('EV/EBITDA', ev_ebitda_price))
    
    # Try P/E (good for profitable companies)
    pe_price = calculate_notional_price_pe(ticker_info)
    if pe_price and pe_price > 0 and pe_price < actual_price * 10:  # Sanity check
        methods.append(('P/E', pe_price))
    
    # Try DCF (most rigorous but requires assumptions)
    dcf_price = calculate_notional_price_dcf(ticker_info)
    if dcf_price and dcf_price > 0 and dcf_price < actual_price * 10:  # Sanity check
        methods.append(('DCF', dcf_price))
    
    # Try P/B (for financial companies)
    pb_price = calculate_notional_price_book(ticker_info)
    if pb_price and pb_price > 0 and pb_price < actual_price * 10:  # Sanity check
        methods.append(('P/B', pb_price))
    
    # Try Revenue multiple last (for growth companies, less reliable)
    revenue_price = calculate_notional_price_revenue(ticker_info)
    if revenue_price and revenue_price > 0 and revenue_price < actual_price * 10:  # Sanity check
        methods.append(('Revenue', revenue_price))
    
    if not methods:
        return None, None
    
    # Use the method with the most reasonable price (closest to actual if available)
    if actual_price and actual_price > 0:
        # Choose method closest to actual (most conservative)
        # But prefer methods that are higher than actual (undervalued)
        valid_methods = [m for m in methods if m[1] > actual_price * 0.5 and m[1] < actual_price * 5]
        if valid_methods:
            best_method = min(valid_methods, key=lambda x: abs(x[1] - actual_price))
        else:
            best_method = min(methods, key=lambda x: abs(x[1] - actual_price))
    else:
        # Use first available method
        best_method = methods[0]
    
    return best_method[0], best_method[1]

def analyze_stocks(snapshot_file):
    """Analyze stocks from snapshot and calculate notional prices."""
    print(f"Loading stocks from: {snapshot_file}")
    
    stocks = []
    with open(snapshot_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                symbol = row['Symbol'].strip()
                name = row['Name'].strip()
                actual_price = float(row['Price'].strip())
                stocks.append({
                    'symbol': symbol,
                    'name': name,
                    'actual_price': actual_price
                })
            except (ValueError, KeyError):
                continue
    
    print(f"Loaded {len(stocks)} stocks")
    print(f"\nFetching detailed financial data...")
    
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
                    actual_price = stock['actual_price']
                    discount_ratio = actual_price / notional_price
                    discount_pct = (1 - discount_ratio) * 100
                    
                    # Calculate potential upside
                    upside = notional_price - actual_price
                    upside_pct = (upside / actual_price) * 100 if actual_price > 0 else 0
                    
                    results.append({
                        'symbol': stock['symbol'],
                        'name': stock['name'],
                        'actual_price': actual_price,
                        'notional_price': notional_price,
                        'method': method,
                        'discount_ratio': discount_ratio,
                        'discount_pct': discount_pct,
                        'upside': upside,
                        'upside_pct': upside_pct,
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
                        'peg_ratio': info.get('pegRatio'),
                    })
            except Exception as e:
                print(f"    Error processing {stock['symbol']}: {e}")
                continue
    
    return results

def display_results(results):
    """Display analysis results."""
    if not results:
        print("No results to display")
        return
    
    # Sort by discount (most undervalued first)
    results.sort(key=lambda x: x['discount_ratio'])
    
    # Filter for undervalued stocks (actual/notional < 0.66 or discount > 33%)
    undervalued = [r for r in results if r['discount_ratio'] < 0.66]
    
    print("\n" + "="*100)
    print("NOTIONAL PRICE ANALYSIS - UNDERVALUED STOCKS")
    print("="*100)
    print(f"\nTotal stocks analyzed: {len(results)}")
    print(f"Undervalued stocks (actual/notional < 0.66): {len(undervalued)} ({len(undervalued)/len(results)*100:.1f}%)")
    
    print(f"\n📊 SUMMARY:")
    # Filter out extreme outliers for average calculation
    valid_results = [r for r in results if -100 < r['discount_pct'] < 100]
    if valid_results:
        avg_discount = sum(r['discount_pct'] for r in valid_results) / len(valid_results)
        avg_upside = sum(r['upside_pct'] for r in valid_results) / len(valid_results)
        print(f"  Average discount: {avg_discount:.2f}%")
        print(f"  Average upside potential: {avg_upside:.2f}%")
    else:
        print(f"  (Unable to calculate averages - extreme values detected)")
    
    print(f"\n🏆 TOP 20 MOST UNDERVALUED (actual/notional < 0.66):")
    print(f"{'Rank':<6} {'Symbol':<8} {'Name':<35} {'Actual':<10} {'Notional':<10} {'Ratio':<8} {'Upside%':<10} {'Method':<12}")
    print("-" * 100)
    
    for i, stock in enumerate(undervalued[:20], 1):
        print(f"{i:<6} {stock['symbol']:<8} {stock['name'][:33]:<35} "
              f"${stock['actual_price']:>8.2f} ${stock['notional_price']:>8.2f} "
              f"{stock['discount_ratio']:>6.2f} {stock['upside_pct']:>+8.1f}% {stock['method']:<12}")
    
    # Show distribution by method
    method_counts = {}
    for r in results:
        method = r['method']
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print(f"\n📈 VALUATION METHODS USED:")
    for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method}: {count} stocks ({count/len(results)*100:.1f}%)")
    
    print("\n" + "="*100)
    print("💡 INTERPRETATION:")
    print("   - Discount Ratio < 0.66 means stock is undervalued by >33%")
    print("   - Upside% shows potential gain if stock reaches notional price")
    print("   - Method shows which valuation approach was used")
    print("="*100)

def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_notional_prices.py <snapshot_file>")
        print("Example: python calculate_notional_prices.py yahooquery_stocks_20251121_145532.txt")
        sys.exit(1)
    
    snapshot_file = sys.argv[1]
    
    results = analyze_stocks(snapshot_file)
    
    if not results:
        print("No results generated")
        sys.exit(1)
    
    display_results(results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"notional_prices_{timestamp}.csv"
    
    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Symbol', 'Name', 'ActualPrice', 'NotionalPrice', 
                                               'Method', 'DiscountRatio', 'DiscountPct', 
                                               'Upside', 'UpsidePct', 'MarketCap', 'PERatio', 'PEGRatio'])
        writer.writeheader()
        for result in sorted(results, key=lambda x: x['discount_ratio']):
            writer.writerow({
                'Symbol': result['symbol'],
                'Name': result['name'],
                'ActualPrice': f"{result['actual_price']:.2f}",
                'NotionalPrice': f"{result['notional_price']:.2f}",
                'Method': result['method'],
                'DiscountRatio': f"{result['discount_ratio']:.4f}",
                'DiscountPct': f"{result['discount_pct']:.2f}",
                'Upside': f"{result['upside']:.2f}",
                'UpsidePct': f"{result['upside_pct']:.2f}",
                'MarketCap': result['market_cap'],
                'PERatio': f"{result['pe_ratio']:.2f}" if result['pe_ratio'] else '',
                'PEGRatio': f"{result['peg_ratio']:.2f}" if result['peg_ratio'] else '',
            })
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()

