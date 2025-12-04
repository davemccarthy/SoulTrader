"""
Test Health Check Algorithm

Evaluates the ChatGPT-provided stock health scoring algorithm.
Tests with various stocks to assess merit and accuracy.

Usage:
    python test_health_check.py                    # Test default stocks
    python test_health_check.py AAPL               # Test single stock
    python test_health_check.py AAPL MSFT GOOGL    # Test multiple stocks
"""

import yfinance as yf
import numpy as np
import argparse
from typing import Dict, Optional, List


def safe_div(a, b):
    """Safely divide two numbers, returning None for invalid operations."""
    try:
        if a is None or b in (0, None):
            return None
        return a / b
    except:
        return None


def get_buy_score(ticker: str) -> Optional[Dict]:
    """
    Calculate buy score for a stock using financial ratios and scoring metrics.
    
    Returns:
        Dictionary with buy_score (0-100), component scores, and financial metrics
        None if calculation fails
    """
    try:
        t = yf.Ticker(ticker)

        info = t.info
        bs = t.balance_sheet.transpose()
        is_ = t.financials.transpose()
        cf = t.cashflow.transpose()

        # --- Extract Key Fields ---
        try:
            latest_bs = bs.iloc[0]
            latest_is = is_.iloc[0]
            latest_cf = cf.iloc[0]
        except:
            raise ValueError("Missing financial statement data.")

        # Financial statement items
        total_assets = latest_bs.get("Total Assets")
        total_liab = latest_bs.get("Total Liab")
        current_assets = latest_bs.get("Total Current Assets")
        current_liab = latest_bs.get("Total Current Liabilities")
        inventory = latest_bs.get("Inventory", 0)
        long_term_debt = latest_bs.get("Long Term Debt") or 0
        short_term_debt = latest_bs.get("Short Long Term Debt") or 0

        revenue = latest_is.get("Total Revenue")
        gross_profit = latest_is.get("Gross Profit")
        net_income = latest_is.get("Net Income")
        ebit = latest_is.get("EBIT") or latest_is.get("Operating Income")

        op_cf = latest_cf.get("Total Cash From Operating Activities")

        # Market (price-based)
        price = info.get("currentPrice")
        eps = info.get("trailingEps")
        book_value = info.get("bookValue")
        market_cap = info.get("marketCap")
        enterprise_value = info.get("enterpriseValue")
        ebitda = info.get("ebitda")

        # --- Core Ratios ---
        ratios = {
            "current_ratio": safe_div(current_assets, current_liab),
            "quick_ratio": safe_div((current_assets or 0) - (inventory or 0), current_liab),
            "debt_to_equity": safe_div(
                long_term_debt + short_term_debt,
                (total_assets - total_liab) if total_assets and total_liab else None
            ),
            "net_margin": safe_div(net_income, revenue),
            "roa": safe_div(net_income, total_assets),
            "asset_turnover": safe_div(revenue, total_assets),
        }

        # --- Valuation Ratios (share-price dependent) ---
        valuation = {
            "pe": safe_div(price, eps),
            "pb": safe_div(price, book_value),
            "ev_ebitda": safe_div(enterprise_value, ebitda),
            "price_sales": safe_div(market_cap, revenue),
            "fcf_yield": safe_div(op_cf, market_cap)
        }

        # --- Simple Piotroski F-score (reduced version) ---
        try:
            prev_is = is_.iloc[1]
            prev_bs = bs.iloc[1]

            roa_cur = safe_div(net_income, total_assets)
            roa_prev = safe_div(prev_is.get("Net Income"), prev_bs.get("Total Assets"))
            ocf_pos = op_cf is not None and op_cf > 0
            roa_improve = roa_prev is not None and roa_cur is not None and roa_cur > roa_prev
            gross_cur = safe_div(gross_profit, revenue)
            gross_prev = safe_div(prev_is.get("Gross Profit"), prev_is.get("Total Revenue"))
            gross_improve = gross_cur is not None and gross_prev is not None and gross_cur > gross_prev

            piotroski = sum([
                1 if roa_cur and roa_cur > 0 else 0,
                1 if ocf_pos else 0,
                1 if roa_improve else 0,
                1 if gross_improve else 0
            ])
        except:
            piotroski = None

        # --- Altman Z Simplified ---
        try:
            # Working Capital / Total Assets
            working_capital = (current_assets or 0) - (current_liab or 0)
            A = safe_div(working_capital, total_assets)
            
            # Retained Earnings / Total Assets (try to get from balance sheet)
            retained_earnings = latest_bs.get("Retained Earnings") or latest_bs.get("Retained Earnings All Equity") or None
            B = safe_div(retained_earnings, total_assets)
            
            # EBIT / Total Assets
            C = safe_div(ebit, total_assets)
            
            # Market Cap / Total Liabilities (or Book Value of Equity if market cap not available)
            # Use total equity as fallback for D component
            total_equity = (total_assets or 0) - (total_liab or 0)
            if market_cap and total_liab:
                D = safe_div(market_cap, total_liab)
            elif total_equity and total_liab:
                # Fallback to book value if market cap unavailable
                D = safe_div(total_equity, total_liab)
            else:
                D = None
            
            # Sales / Total Assets (Asset Turnover)
            E = safe_div(revenue, total_assets)
            
            # Calculate Altman Z with proper handling of None values
            # If critical components are missing, return None
            if A is None and C is None and E is None:
                altman = None
            else:
                # Use 0 for missing optional components (B and D)
                altman = (
                    1.2 * (A or 0) + 
                    1.4 * (B or 0) + 
                    3.3 * (C or 0) + 
                    0.6 * (D or 0) + 
                    1.0 * (E or 0)
                )
                # Handle NaN values (can occur from division by zero or invalid calculations)
                if altman is not None and (np.isnan(altman) or np.isinf(altman)):
                    altman = None
        except Exception as e:
            altman = None

        # --- Convert each component into a 0-100 score ---
        # Business health
        health_score = np.mean([
            np.clip((ratios["current_ratio"] or 1) * 20, 0, 100),
            np.clip(100 - (ratios["debt_to_equity"] or 0) * 20, 0, 100),
            np.clip((ratios["net_margin"] or 0) * 500, 0, 100),
            np.clip((ratios["roa"] or 0) * 2000, 0, 100)
        ])

        # Valuation attractiveness
        valuation_score = np.mean([
            np.clip(50 - (valuation["pe"] or 50), 0, 100),
            np.clip(50 - (valuation["pb"] or 50), 0, 100),
            np.clip(50 - (valuation["ev_ebitda"] or 50), 0, 100),
            np.clip((valuation["fcf_yield"] or 0) * 500, 0, 100)
        ])

        # Piotroski normalized
        pscore = (piotroski / 4 * 100) if piotroski is not None else 50

        # Altman mapped to 0-100
        if altman is None:
            zscore = 50
        elif altman > 3:
            zscore = 90
        elif altman < 1.8:
            zscore = 20
        else:
            zscore = 50

        # --- Combined Buy Score ---
        buy_score = (
            health_score * 0.40 +
            valuation_score * 0.30 +
            pscore * 0.15 +
            zscore * 0.15
        )

        return {
            "ticker": ticker,
            "buy_score": round(buy_score, 1),
            "health_score": round(health_score, 1),
            "valuation_score": round(valuation_score, 1),
            "piotroski": piotroski,
            "altman_z": altman,
            "ratios": ratios,
            "valuation": valuation
        }
    except Exception as e:
        print(f"Error calculating buy score for {ticker}: {e}")
        return None


def test_single_stock(ticker: str, verbose: bool = True) -> Optional[Dict]:
    """Test the algorithm on a single stock."""
    print(f"\n{'='*60}")
    print(f"Testing: {ticker}")
    print(f"{'='*60}")
    
    result = get_buy_score(ticker)
    
    if result is None:
        print(f"❌ Failed to calculate score for {ticker}")
        return None
    
    if verbose:
        print(f"\n📊 Results:")
        print(f"  Buy Score: {result['buy_score']:.1f}/100")
        print(f"  ├─ Health Score: {result['health_score']:.1f}/100 (40% weight)")
        print(f"  ├─ Valuation Score: {result['valuation_score']:.1f}/100 (30% weight)")
        print(f"  ├─ Piotroski: {result['piotroski']}/4 (15% weight)")
        altman_val = result['altman_z']
        if altman_val is None or (isinstance(altman_val, float) and (np.isnan(altman_val) or np.isinf(altman_val))):
            altman_display = "N/A"
        else:
            altman_display = f"{altman_val:.2f}"
        print(f"  └─ Altman Z: {altman_display} (15% weight)")
        
        print(f"\n💼 Key Ratios:")
        ratios = result['ratios']
        print(f"  Current Ratio: {ratios.get('current_ratio', 'N/A'):.2f}" if ratios.get('current_ratio') else f"  Current Ratio: N/A")
        print(f"  Debt-to-Equity: {ratios.get('debt_to_equity', 'N/A'):.2f}" if ratios.get('debt_to_equity') else f"  Debt-to-Equity: N/A")
        print(f"  Net Margin: {ratios.get('net_margin', 'N/A'):.2%}" if ratios.get('net_margin') else f"  Net Margin: N/A")
        print(f"  ROA: {ratios.get('roa', 'N/A'):.2%}" if ratios.get('roa') else f"  ROA: N/A")
        
        print(f"\n💰 Valuation Metrics:")
        valuation = result['valuation']
        print(f"  P/E: {valuation.get('pe', 'N/A'):.2f}" if valuation.get('pe') else f"  P/E: N/A")
        print(f"  P/B: {valuation.get('pb', 'N/A'):.2f}" if valuation.get('pb') else f"  P/B: N/A")
        print(f"  EV/EBITDA: {valuation.get('ev_ebitda', 'N/A'):.2f}" if valuation.get('ev_ebitda') else f"  EV/EBITDA: N/A")
        print(f"  FCF Yield: {valuation.get('fcf_yield', 'N/A'):.2%}" if valuation.get('fcf_yield') else f"  FCF Yield: N/A")
    
    return result


def test_multiple_stocks(tickers: List[str]) -> List[Dict]:
    """Test the algorithm on multiple stocks and return sorted results."""
    print(f"\n{'='*60}")
    print(f"Testing {len(tickers)} stocks")
    print(f"{'='*60}")
    
    results = []
    for ticker in tickers:
        result = test_single_stock(ticker, verbose=False)
        if result:
            results.append(result)
        print(f"{ticker}: {result['buy_score']:.1f}/100" if result else f"{ticker}: FAILED")
    
    # Sort by buy_score descending
    results.sort(key=lambda x: x['buy_score'], reverse=True)
    
    print(f"\n{'='*60}")
    print("Ranked Results (by Buy Score):")
    print(f"{'='*60}")
    for i, result in enumerate(results, 1):
        altman_val = result['altman_z']
        if altman_val is None or (isinstance(altman_val, float) and (np.isnan(altman_val) or np.isinf(altman_val))):
            altman_str = "N/A"
        else:
            altman_str = f"{altman_val:.2f}"
        print(f"{i:2d}. {result['ticker']:6s} - Score: {result['buy_score']:5.1f} | "
              f"Health: {result['health_score']:5.1f} | "
              f"Valuation: {result['valuation_score']:5.1f} | "
              f"Piotroski: {result['piotroski']} | "
              f"Altman Z: {altman_str}")
    
    return results


def test_error_handling():
    """Test algorithm behavior with edge cases and invalid inputs."""
    print(f"\n{'='*60}")
    print("Testing Error Handling")
    print(f"{'='*60}")
    
    # Test invalid ticker
    print("\n1. Testing invalid ticker...")
    result = get_buy_score("INVALID_TICKER_XYZ")
    print(f"   Result: {'PASSED (returned None)' if result is None else 'FAILED (should return None)'}")
    
    # Test stocks that might have missing data
    edge_cases = ["BRK-A", "GOOGL"]  # These might have different data structures
    for ticker in edge_cases:
        print(f"\n2. Testing {ticker} (potential edge case)...")
        result = get_buy_score(ticker)
        print(f"   Result: {'SUCCESS' if result else 'FAILED'}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Health Check Algorithm for stock evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_health_check.py                    # Test default stock set
  python test_health_check.py AAPL               # Test single stock
  python test_health_check.py AAPL MSFT GOOGL    # Test multiple stocks
  python test_health_check.py --no-summary AAPL  # Skip summary, detailed only
        """
    )
    parser.add_argument(
        'symbols',
        nargs='*',
        help='Stock symbol(s) to test (e.g., AAPL MSFT GOOGL). If none provided, uses default test set.'
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip summary statistics when testing single stock'
    )
    parser.add_argument(
        '--no-error-test',
        action='store_true',
        help='Skip error handling tests'
    )
    
    args = parser.parse_args()
    
    # Determine which stocks to test
    if args.symbols:
        test_stocks = [s.upper() for s in args.symbols]  # Normalize to uppercase
    else:
        # Default test set with various stocks across different sectors
        test_stocks = [
            # Tech
            "AAPL", "MSFT", "GOOGL", "NVDA",
            # Finance
            "JPM", "BAC", "GS",
            # Consumer
            "AMZN", "WMT", "TGT",
            # Healthcare
            "JNJ", "PFE", "UNH",
            # Energy
            "XOM", "CVX",
            # Small cap / potentially volatile
            "TSLA", "AMD",
        ]
    
    print("=" * 60)
    print("Health Check Algorithm Evaluation")
    print("=" * 60)
    
    # If single stock, show detailed output only
    if len(test_stocks) == 1:
        print(f"\nTesting single stock: {test_stocks[0]}\n")
        result = test_single_stock(test_stocks[0], verbose=True)
        if not args.no_summary and result:
            print(f"\n{'='*60}")
            print("Summary")
            print(f"{'='*60}")
            print(f"Buy Score: {result['buy_score']:.1f}/100")
            print(f"Health Score: {result['health_score']:.1f}/100")
            print(f"Valuation Score: {result['valuation_score']:.1f}/100")
    else:
        # Multiple stocks - run batch test
        results = test_multiple_stocks(test_stocks)
        
        # Test error handling unless disabled
        if not args.no_error_test:
            test_error_handling()
        
        # Show detailed analysis for top 3 and bottom 3
        if len(results) >= 3:
            print(f"\n{'='*60}")
            print("Detailed Analysis: Top 3 Stocks")
            print(f"{'='*60}")
            for result in results[:3]:
                test_single_stock(result['ticker'], verbose=True)
        
        if len(results) > 3:
            print(f"\n{'='*60}")
            print("Detailed Analysis: Bottom 3 Stocks")
            print(f"{'='*60}")
            for result in results[-3:]:
                test_single_stock(result['ticker'], verbose=True)
        
        # Summary statistics
        if not args.no_summary:
            print(f"\n{'='*60}")
            print("Evaluation Summary")
            print(f"{'='*60}")
            print(f"Total stocks tested: {len(test_stocks)}")
            print(f"Successful calculations: {len(results)}")
            print(f"Failed calculations: {len(test_stocks) - len(results)}")
            if results:
                print(f"Average buy score: {np.mean([r['buy_score'] for r in results]):.1f}")
                print(f"Score range: {min(r['buy_score'] for r in results):.1f} - {max(r['buy_score'] for r in results):.1f}")
    
    print("\n" + "=" * 60)

