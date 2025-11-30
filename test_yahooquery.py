#!/usr/bin/env python
"""
Test script to explore yahooquery's Screener functionality.

This script tests different approaches to access Yahoo Finance screeners
using the yahooquery library.
"""

import sys

try:
    from yahooquery import Screener, Ticker
except ImportError:
    print("yahooquery not installed. Install with: pip install yahooquery")
    sys.exit(1)


def test_screener_basics():
    """Test basic Screener class functionality."""
    print("=" * 60)
    print("TEST 1: Basic Screener Class")
    print("=" * 60)
    
    screener = Screener()
    
    # Check what methods/attributes are available
    print("\nScreener object attributes/methods:")
    attrs = [attr for attr in dir(screener) if not attr.startswith('_')]
    for attr in attrs:
        print(f"  - {attr}")
    
    return screener


def test_available_screeners(screener):
    """Test if we can list available screeners."""
    print("\n" + "=" * 60)
    print("TEST 2: List Available Screeners")
    print("=" * 60)
    
    # available_screeners is a property, not a method
    if hasattr(screener, 'available_screeners'):
        try:
            result = screener.available_screeners
            print(f"\n✓ available_screeners property exists:")
            print(f"  Type: {type(result)}")
            if isinstance(result, (list, dict)):
                print(f"  Length/Keys: {len(result)}")
                if isinstance(result, dict):
                    print(f"  Keys (first 20): {list(result.keys())[:20]}")
                elif isinstance(result, list):
                    print(f"  First few items: {result[:10]}")
            else:
                print(f"  Value: {result}")
        except Exception as e:
            print(f"\n✗ available_screeners property raised error: {e}")
    else:
        print("\n✗ available_screeners property does not exist")


def test_specific_screener(screener):
    """Test accessing the 'undervalued_growth_stocks' screener."""
    print("\n" + "=" * 60)
    print("TEST 3: Access Specific Screener (undervalued_growth_stocks)")
    print("=" * 60)
    
    screener_id = 'undervalued_growth_stocks'
    
    print(f"\nTrying screener ID: '{screener_id}'")
    try:
        # get_screeners requires screen_ids as a list
        # Get all available stocks (176 total) for complete results
        result = screener.get_screeners([screener_id], count=176)
        print(f"  ✓ get_screeners(['{screener_id}'], count=176) succeeded!")
        print(f"    Type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"    Keys: {list(result.keys())}")
            if screener_id in result:
                screener_data = result[screener_id]
                print(f"    Data type: {type(screener_data)}")
                
                if isinstance(screener_data, dict):
                    print(f"    Data keys: {list(screener_data.keys())[:10]}")
                    # Check for quotes or similar
                    if 'quotes' in screener_data:
                        quotes = screener_data['quotes']
                        print(f"    Number of quotes: {len(quotes)}")
                        if 'total' in screener_data:
                            print(f"    Total available: {screener_data['total']}")
                        if len(quotes) > 0:
                            print(f"\n    First quote keys (showing first 30): {list(quotes[0].keys())[:30]}")
                            print(f"\n    Sample quote (key fields):")
                            first = quotes[0]
                            # Try various field name variations
                            key_fields = [
                                'symbol', 'shortName', 'longName', 'regularMarketPrice', 
                                'pegRatio', 'pegratio_5y', 'pegRatio5Y',
                                'epsGrowth', 'epsgrowth', 'earningsGrowth', 'earningsQuarterlyGrowth',
                                'trailingPE', 'forwardPE', 'priceToBook', 'priceToSalesTrailing12Months'
                            ]
                            for key in key_fields:
                                if key in first:
                                    print(f"      {key}: {first[key]}")
                            
                            # Show all available keys for debugging
                            print(f"\n    All available keys in quote ({len(quotes[0].keys())} total):")
                            all_keys = sorted(quotes[0].keys())
                            for i in range(0, len(all_keys), 5):
                                print(f"      {', '.join(all_keys[i:i+5])}")
                            
                            # Show all stocks
                            print(f"\n    All {len(quotes)} stocks:")
                            print(f"    {'Symbol':<10} {'Name':<40} {'Price':>10} {'PEG':>8} {'EPS Growth':>12} {'P/E':>8}")
                            print(f"    {'-'*10} {'-'*40} {'-'*10} {'-'*8} {'-'*12} {'-'*8}")
                            for quote in quotes:
                                symbol = quote.get('symbol', 'N/A')
                                name = quote.get('shortName', quote.get('longName', 'N/A'))
                                if len(name) > 38:
                                    name = name[:35] + '...'
                                price = quote.get('regularMarketPrice', 'N/A')
                                
                                # Try multiple field name variations
                                peg = (quote.get('pegRatio') or 
                                      quote.get('pegratio_5y') or 
                                      quote.get('pegRatio5Y') or 
                                      'N/A')
                                
                                eps_growth = (quote.get('epsGrowth') or 
                                            quote.get('earningsGrowth') or 
                                            quote.get('earningsQuarterlyGrowth') or 
                                            'N/A')
                                
                                pe = quote.get('trailingPE', 'N/A')
                                
                                price_str = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
                                peg_str = f"{peg:.2f}" if isinstance(peg, (int, float)) else str(peg)
                                if isinstance(eps_growth, (int, float)):
                                    eps_str = f"{eps_growth:.1%}" if abs(eps_growth) < 10 else f"{eps_growth:.0%}"
                                else:
                                    eps_str = str(eps_growth)
                                pe_str = f"{pe:.1f}" if isinstance(pe, (int, float)) else str(pe)
                                
                                print(f"    {symbol:<10} {name:<40} {price_str:>10} {peg_str:>8} {eps_str:>12} {pe_str:>8}")
                elif isinstance(screener_data, list):
                    print(f"    Number of items: {len(screener_data)}")
                    if len(screener_data) > 0:
                        print(f"    First item type: {type(screener_data[0])}")
                        print(f"    First item: {screener_data[0]}")
                
        return result
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_screener_with_params(screener):
    """Test screener with different parameters."""
    print("\n" + "=" * 60)
    print("TEST 4: Screener with Different Count Parameters")
    print("=" * 60)
    
    screener_id = 'undervalued_growth_stocks'
    counts_to_try = [10, 50, 100]
    
    for count in counts_to_try:
        print(f"\nTrying with count={count}:")
        try:
            result = screener.get_screeners([screener_id], count=count)
            if isinstance(result, dict) and screener_id in result:
                screener_data = result[screener_id]
                if isinstance(screener_data, dict) and 'quotes' in screener_data:
                    num_quotes = len(screener_data['quotes'])
                    print(f"  ✓ Retrieved {num_quotes} quotes")
                else:
                    print(f"  ✓ Retrieved data (structure: {type(screener_data)})")
        except Exception as e:
            print(f"  ✗ Failed: {e}")


def test_screener_metadata(screener):
    """Explore screener metadata and criteria."""
    print("\n" + "=" * 60)
    print("TEST 5: Screener Metadata and Criteria")
    print("=" * 60)
    
    screener_id = 'undervalued_growth_stocks'
    result = screener.get_screeners([screener_id], count=1)  # Just get 1 for metadata
    
    if screener_id in result:
        data = result[screener_id]
        print(f"\nScreener: {screener_id}")
        print(f"  Title: {data.get('title', 'N/A')}")
        print(f"  Description: {data.get('description', 'N/A')}")
        print(f"  Canonical Name: {data.get('canonicalName', 'N/A')}")
        print(f"  Total available: {data.get('total', 'N/A')}")
        
        if 'criteriaMeta' in data:
            print(f"\n  Criteria Meta:")
            criteria = data['criteriaMeta']
            if isinstance(criteria, dict):
                for key, value in list(criteria.items())[:10]:
                    print(f"    {key}: {value}")
            elif isinstance(criteria, list):
                for item in criteria[:5]:
                    print(f"    {item}")
        
        if 'rawCriteria' in data:
            print(f"\n  Raw Criteria (first 500 chars):")
            raw = data['rawCriteria']
            print(f"    {str(raw)[:500]}...")


def test_pagination(screener):
    """Test if we can get more results using pagination."""
    print("\n" + "=" * 60)
    print("TEST 6: Pagination - Getting All Results")
    print("=" * 60)
    
    screener_id = 'undervalued_growth_stocks'
    
    # First, get total count
    result = screener.get_screeners([screener_id], count=1)
    if screener_id in result:
        total = result[screener_id].get('total', 0)
        print(f"Total stocks available: {total}")
        
        # Try to get all results
        print(f"\nAttempting to get all {total} stocks:")
        try:
            result_all = screener.get_screeners([screener_id], count=total)
            if screener_id in result_all:
                quotes = result_all[screener_id].get('quotes', [])
                print(f"  ✓ Retrieved {len(quotes)} stocks")
                print(f"  First 10 symbols: {[q.get('symbol') for q in quotes[:10]]}")
                print(f"  Last 10 symbols: {[q.get('symbol') for q in quotes[-10:]]}")
        except Exception as e:
            print(f"  ✗ Failed to get all results: {e}")


def test_other_screeners(screener):
    """Explore other potentially useful screeners."""
    print("\n" + "=" * 60)
    print("TEST 7: Other Useful Screeners")
    print("=" * 60)
    
    # Get list of available screeners
    available = screener.available_screeners
    print(f"Total available screeners: {len(available)}")
    
    # Look for interesting screeners
    interesting = [
        'undervalued_growth_stocks',
        'growth_stocks',
        'undervalued_stocks',
        'most_active',
        'day_gainers',
        'day_losers',
        'aggressive_small_caps',
        'small_cap_gainers',
        'penny_stocks',
        'high_short_interest',
    ]
    
    print(f"\nChecking for interesting screeners:")
    found = []
    for screener_id in interesting:
        if screener_id in available:
            found.append(screener_id)
            print(f"  ✓ {screener_id}")
        else:
            # Try partial matches
            matches = [s for s in available if screener_id in s or s in screener_id]
            if matches:
                print(f"  ~ {screener_id} (not exact, but found: {matches[:3]})")
    
    # Test a couple of other screeners
    test_screeners = ['most_active', 'day_gainers']
    for test_id in test_screeners:
        if test_id in available:
            print(f"\n  Testing '{test_id}' screener:")
            try:
                result = screener.get_screeners([test_id], count=5)
                if test_id in result:
                    quotes = result[test_id].get('quotes', [])
                    print(f"    ✓ Retrieved {len(quotes)} stocks")
                    if quotes:
                        print(f"    Sample: {quotes[0].get('symbol')} - {quotes[0].get('shortName', 'N/A')}")
            except Exception as e:
                print(f"    ✗ Failed: {e}")


def test_enrich_with_ticker(screener):
    """Test enriching screener results with detailed Ticker data."""
    print("\n" + "=" * 60)
    print("TEST 8: Enriching Results with Ticker Data")
    print("=" * 60)
    
    screener_id = 'undervalued_growth_stocks'
    result = screener.get_screeners([screener_id], count=5)
    
    if screener_id in result:
        quotes = result[screener_id].get('quotes', [])
        if quotes:
            # Get symbols
            symbols = [q.get('symbol') for q in quotes if q.get('symbol')]
            print(f"Testing enrichment for {len(symbols)} symbols: {symbols}")
            
            # Fetch detailed data using Ticker
            try:
                tickers = Ticker(symbols)
                
                # Try to get PEG and EPS Growth from various Ticker properties
                print(f"\nFetching detailed data...")
                info_data = {}
                for symbol in symbols:
                    try:
                        ticker = Ticker(symbol)
                        
                        # Try multiple data sources
                        summary = ticker.summary_detail.get(symbol, {}) if hasattr(ticker, 'summary_detail') else {}
                        key_stats = ticker.key_stats.get(symbol, {}) if hasattr(ticker, 'key_stats') else {}
                        financial_data = ticker.financial_data.get(symbol, {}) if hasattr(ticker, 'financial_data') else {}
                        info = ticker.info if hasattr(ticker, 'info') else {}
                        
                        # Try to find PEG and EPS Growth in any of these
                        peg = (summary.get('pegRatio') or 
                              key_stats.get('pegRatio') or 
                              financial_data.get('pegRatio') or
                              info.get('pegRatio') or
                              info.get('pegratio_5y'))
                        
                        earnings_growth = (summary.get('earningsGrowth') or
                                          key_stats.get('earningsGrowth') or
                                          info.get('earningsGrowth') or
                                          info.get('earningsQuarterlyGrowth') or
                                          info.get('epsGrowth'))
                        
                        revenue_growth = (summary.get('revenueGrowth') or
                                         key_stats.get('revenueGrowth') or
                                         info.get('revenueGrowth'))
                        
                        info_data[symbol] = {
                            'pegRatio': peg,
                            'earningsGrowth': earnings_growth,
                            'revenueGrowth': revenue_growth,
                        }
                    except Exception as e:
                        print(f"  Error fetching {symbol}: {e}")
                
                # Display enriched data
                print(f"\nEnriched data:")
                print(f"  {'Symbol':<10} {'PEG':>8} {'Earnings Growth':>18} {'Revenue Growth':>18}")
                print(f"  {'-'*10} {'-'*8} {'-'*18} {'-'*18}")
                for symbol in symbols:
                    data = info_data.get(symbol, {})
                    peg = data.get('pegRatio', 'N/A')
                    earnings = data.get('earningsGrowth', 'N/A')
                    revenue = data.get('revenueGrowth', 'N/A')
                    
                    peg_str = f"{peg:.2f}" if isinstance(peg, (int, float)) else str(peg)
                    if isinstance(earnings, (int, float)):
                        earnings_str = f"{earnings:.1%}" if abs(earnings) < 10 else f"{earnings:.0%}"
                    else:
                        earnings_str = str(earnings)
                    if isinstance(revenue, (int, float)):
                        revenue_str = f"{revenue:.1%}" if abs(revenue) < 10 else f"{revenue:.0%}"
                    else:
                        revenue_str = str(revenue)
                    
                    print(f"  {symbol:<10} {peg_str:>8} {earnings_str:>18} {revenue_str:>18}")
                    
            except Exception as e:
                print(f"  ✗ Failed to enrich: {e}")
                import traceback
                traceback.print_exc()


def test_direct_api_call():
    """Test if we can access the screener data directly via Ticker or other means."""
    print("\n" + "=" * 60)
    print("TEST 9: Alternative Approaches")
    print("=" * 60)
    
    # Check if Ticker has screener methods
    print("\nChecking Ticker class for screener methods:")
    ticker_attrs = [attr for attr in dir(Ticker) if 'screen' in attr.lower()]
    if ticker_attrs:
        print(f"  Found: {ticker_attrs}")
    else:
        print("  No screener-related methods found in Ticker")


def save_results_to_file(result, filename='yahooquery_stocks.txt'):
    """Save stock results to a flat file: Symbol, Name, Price."""
    if not result or 'undervalued_growth_stocks' not in result:
        print(f"\nNo results to save")
        return
    
    screener_data = result['undervalued_growth_stocks']
    quotes = screener_data.get('quotes', [])
    
    if not quotes:
        print(f"\nNo quotes to save")
        return
    
    # Save to file
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"yahooquery_stocks_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        # Header
        f.write(f"Symbol,Name,Price\n")
        
        # Write each stock
        for quote in quotes:
            symbol = quote.get('symbol', 'N/A')
            name = quote.get('shortName') or quote.get('longName', 'N/A')
            price = quote.get('regularMarketPrice', 'N/A')
            
            # Clean name (remove commas)
            name = name.replace(',', '')
            
            f.write(f"{symbol},{name},{price}\n")
    
    print(f"\n✓ Saved {len(quotes)} stocks to {filename}")


def main():
    """Run all tests."""
    print("Yahooquery Screener Exploration")
    print("=" * 60)
    
    screener = test_screener_basics()
    test_available_screeners(screener)
    result = test_specific_screener(screener)
    test_screener_with_params(screener)
    test_screener_metadata(screener)
    test_pagination(screener)
    test_other_screeners(screener)
    test_enrich_with_ticker(screener)
    test_direct_api_call()
    
    # Save results to file
    if result:
        save_results_to_file(result)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if result:
        print("✓ Successfully retrieved screener data!")
        print(f"  Result type: {type(result)}")
        if isinstance(result, dict) and 'undervalued_growth_stocks' in result:
            data = result['undervalued_growth_stocks']
            if isinstance(data, dict) and 'quotes' in data:
                quotes = data['quotes']
                print(f"  Number of stocks: {len(quotes)}")
                if len(quotes) > 0:
                    print(f"  First stock symbol: {quotes[0].get('symbol', 'N/A')}")
                    print(f"  First stock name: {quotes[0].get('shortName', 'N/A')}")
    else:
        print("✗ Could not retrieve screener data using standard methods")
        print("  May need to check yahooquery documentation or source code")
        print("  GitHub: https://github.com/dpguthrie/yahooquery")
    
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print("""
1. Screener Criteria (from metadata):
   - P/E ratio: 0-20 OR PEG < 1
   - EPS Growth: 25-50% OR 50-100% OR >100%
   - Exchange: NMS or NYQ
   - Sorted by: Volume (DESC)

2. Can get ALL stocks: Use count=176 (or total from metadata)

3. Available Screeners (383 total):
   - undervalued_growth_stocks (176 stocks)
   - day_gainers, day_losers
   - aggressive_small_caps
   - small_cap_gainers
   - most_actives

4. Data Available in Quotes:
   - symbol, shortName, longName
   - regularMarketPrice
   - trailingPE, forwardPE
   - priceToBook, marketCap
   - 80+ other fields
   - Note: PEG/EPS Growth not in quote data (already filtered by screener)

5. Enrichment: Can use Ticker for additional data, but PEG/EPS Growth
   may not be available in yahooquery's Ticker properties.
""")
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLE")
    print("=" * 60)
    print("""
from yahooquery import Screener

screener = Screener()
result = screener.get_screeners(['undervalued_growth_stocks'], count=176)

# Extract symbols
if 'undervalued_growth_stocks' in result:
    quotes = result['undervalued_growth_stocks'].get('quotes', [])
    symbols = [q['symbol'] for q in quotes if 'symbol' in q]
    print(f"Found {len(symbols)} stocks: {symbols[:10]}")
    
    # Access stock data
    for quote in quotes:
        symbol = quote.get('symbol')
        price = quote.get('regularMarketPrice')
        pe = quote.get('trailingPE')
        print(f"{symbol}: ${price:.2f}, P/E: {pe:.1f}")
""")


if __name__ == "__main__":
    main()

