#!/usr/bin/env python
"""
Test script for the Yahoo advisor's discover() method logic.

This script tests the actual discovery logic used by the Yahoo advisor:
- Active stock fetching
- Notional price calculation (DCF, P/E, EV/EBITDA, etc.)
- Fundamental filtering
- Recent trend filtering
- Undervalued stock selection

This version mimics the logic without using AdvisorBase to avoid Django dependencies.

Usage:
    python test_yahoo.py
    python test_yahoo.py --limit 50
    python test_yahoo.py --verbose
"""

import argparse
import logging
import sys

import yfinance as yf
from yfinance.screener import EquityQuery as YfEquityQuery

# Constants from yahoo.py
MAX_PRICE = 5.0
UNDERVALUED_RATIO_THRESHOLD = 0.70  # Changed from 0.66 to capture more opportunities (e.g., WDH at 31.2% discount)
MIN_PROFIT_MARGIN = 0.0
MAX_YEARLY_LOSS = -50.0
MIN_MARKET_CAP = 25_000_000
MAX_RECENT_TREND_PCT = 5.0

# Additional quality filters (Phase 1)
MIN_REVENUE_GROWTH = -5.0  # Reject if revenue decline >5% (catches declining companies)
MAX_30_DAY_DECLINE = -15.0  # Reject if down >15% in last 30 days (catches sustained declines)

# Average low filter (TEMPORARY - experimental)
# Set to None to disable, or a value like 1.10 to allow price up to 10% above 50-day SMA
MAX_PRICE_VS_SMA50 = 1.10  # TEMPORARY: Filter stocks trading >10% above 50-day SMA (None = disabled)

logger = logging.getLogger(__name__)


def get_active_stocks(limit=200, min_volume=None, max_volume=None, sort_volume_asc=False):
    """
    Get stocks from Yahoo Finance with configurable volume filtering (mimics _get_active_stocks).
    
    Args:
        limit: Maximum number of stocks to return
        min_volume: Optional minimum daily volume (filters out illiquid stocks)
        max_volume: Optional maximum daily volume (filters out over-traded stocks)
        sort_volume_asc: If True, sort by volume ascending (lowest first, for less discovered stocks).
                        If False, sort descending (highest first, current default behavior)
    """
    try:
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
        stocks = []
        
        for quote in quotes:
            symbol = quote.get('symbol')
            name = quote.get('shortName') or quote.get('longName', 'N/A')
            price = quote.get('regularMarketPrice') or quote.get('intradayprice')
            volume = quote.get('volume') or quote.get('regularMarketVolume') or 0
            
            if symbol and price:
                stocks.append({
                    'symbol': symbol,
                    'name': name,
                    'price': float(price),
                    'volume': float(volume) if volume else 0.0,
                })
        
        # Filter by volume range if specified
        if min_volume is not None or max_volume is not None:
            stocks = [
                s for s in stocks 
                if (min_volume is None or s['volume'] >= min_volume) and
                   (max_volume is None or s['volume'] <= max_volume)
            ]
        
        # Sort by volume (ascending or descending based on param)
        stocks.sort(key=lambda x: x['volume'], reverse=not sort_volume_asc)
        return stocks[:limit]
        
    except Exception as e:
        logger.warning(f"Error fetching active stocks: {e}")
        return []


def calculate_notional_price_dcf(ticker_info):
    """Calculate notional price using DCF method (simplified)."""
    try:
        fcf = ticker_info.get('freeCashflow') or ticker_info.get('operatingCashflow')
        shares = ticker_info.get('sharesOutstanding')
        
        if fcf is None and ticker_info.get('operatingCashflow') is None:
            return None, "Missing FCF data"
        if shares is None:
            return None, "Missing shares outstanding data"
        if fcf is not None and fcf <= 0:
            return None, "Invalid FCF data (negative or zero)"
        if shares <= 0:
            return None, "Invalid shares outstanding data (zero or negative)"
        
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
        
        return notional_price, None
    except Exception as e:
        return None, f"Error: {str(e)}"


def calculate_notional_price_pe(ticker_info):
    """Calculate notional price using P/E multiple method."""
    try:
        eps = ticker_info.get('trailingEps') or ticker_info.get('forwardEps')
        shares = ticker_info.get('sharesOutstanding')
        
        if eps is None and ticker_info.get('forwardEps') is None:
            return None, "Missing EPS data"
        if shares is None:
            return None, "Missing shares outstanding data"
        if eps is not None and eps <= 0:
            return None, "Invalid EPS data (negative or zero - company may be unprofitable)"
        if shares <= 0:
            return None, "Invalid shares outstanding data (zero or negative)"
        
        # Use sector average P/E, not company's own P/E
        sector_pe_map = {
            'Technology': 28.0,
            'Healthcare': 22.0,
            'Financial Services': 13.0,
            'Consumer Cyclical': 19.0,
            'Consumer Defensive': 21.0,
            'Energy': 11.0,
            'Industrials': 19.0,
            'Basic Materials': 16.0,
            'Real Estate': 18.0,
            'Utilities': 16.0,
            'Communication Services': 16.0,
        }
        
        sector = ticker_info.get('sector', '')
        pe_ratio = sector_pe_map.get(sector, 18.0)  # Default to market average
        
        notional_price = eps * pe_ratio
        return notional_price, None
    except Exception as e:
        return None, f"Error: {str(e)}"


def calculate_notional_price_ev_ebitda(ticker_info):
    """Calculate notional price using EV/EBITDA multiple method."""
    try:
        ebitda = ticker_info.get('ebitda')
        shares = ticker_info.get('sharesOutstanding')
        market_cap = ticker_info.get('marketCap')
        total_debt = ticker_info.get('totalDebt') or 0
        cash = ticker_info.get('totalCash') or 0
        
        if ebitda is None:
            return None, "Missing EBITDA data"
        if shares is None:
            return None, "Missing shares outstanding data"
        if ebitda <= 0:
            return None, "Invalid EBITDA data (negative or zero - company may be unprofitable)"
        if shares <= 0:
            return None, "Invalid shares outstanding data (zero or negative)"
        
        # Use company's own EV/EBITDA if available, otherwise use sector average
        ev_ebitda = ticker_info.get('enterpriseToEbitda') or 10.0
        
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
        return notional_price, None
    except Exception as e:
        return None, f"Error: {str(e)}"


def calculate_notional_price_revenue(ticker_info):
    """Calculate notional price using revenue multiple method."""
    try:
        revenue_per_share = ticker_info.get('revenuePerShare')
        shares = ticker_info.get('sharesOutstanding')
        total_revenue = ticker_info.get('totalRevenue')
        
        if not revenue_per_share and total_revenue and shares and shares > 0:
            revenue_per_share = total_revenue / shares
        
        if revenue_per_share is None:
            if total_revenue is None:
                return None, "Missing revenue data"
            if shares is None or shares <= 0:
                return None, "Missing or invalid shares outstanding data"
        if revenue_per_share and revenue_per_share <= 0:
            return None, "Invalid revenue data (negative or zero)"
        
        # Use price-to-sales ratio (default 2.0 for growth companies)
        ps_ratio = ticker_info.get('priceToSalesTrailing12Months')
        if not ps_ratio or ps_ratio < 0.5 or ps_ratio > 20:
            ps_ratio = 2.0
        
        notional_price = revenue_per_share * ps_ratio
        
        # Sanity check
        if notional_price > 10000:
            return None, "Notional price too high (sanity check failed)"
        
        return notional_price, None
    except Exception as e:
        return None, f"Error: {str(e)}"


def calculate_notional_price_book(ticker_info):
    """Calculate notional price using price-to-book method."""
    try:
        book_value = ticker_info.get('bookValue')
        shares = ticker_info.get('sharesOutstanding')
        
        if book_value is None:
            return None, "Missing book value data"
        if shares is None:
            return None, "Missing shares outstanding data"
        if book_value <= 0:
            return None, "Invalid book value data (negative or zero)"
        if shares <= 0:
            return None, "Invalid shares outstanding data (zero or negative)"
        
        book_per_share = book_value / shares
        
        # Use company's P/B if available, otherwise use sector average (default 1.5)
        pb_ratio = ticker_info.get('priceToBook') or 1.5
        
        notional_price = book_per_share * pb_ratio
        return notional_price, None
    except Exception as e:
        return None, f"Error: {str(e)}"


def calculate_best_notional_price(ticker_info, return_reasons=False):
    """Calculate notional price using the best available method."""
    methods = []
    reasons = {}
    actual_price = ticker_info.get('currentPrice') or ticker_info.get('regularMarketPrice') or 0
    
    # Try EV/EBITDA first (most reliable for most companies)
    ev_ebitda_price, reason = calculate_notional_price_ev_ebitda(ticker_info)
    if reason:
        reasons['EV/EBITDA'] = reason
    if ev_ebitda_price and ev_ebitda_price > 0 and ev_ebitda_price < actual_price * 10:
        methods.append(('EV/EBITDA', ev_ebitda_price))
    
    # Try P/E (good for profitable companies)
    pe_price, reason = calculate_notional_price_pe(ticker_info)
    if reason:
        reasons['P/E'] = reason
    if pe_price and pe_price > 0 and pe_price < actual_price * 10:
        methods.append(('P/E', pe_price))
    
    # Try DCF (most rigorous but requires assumptions)
    dcf_price, reason = calculate_notional_price_dcf(ticker_info)
    if reason:
        reasons['DCF'] = reason
    if dcf_price and dcf_price > 0 and dcf_price < actual_price * 10:
        methods.append(('DCF', dcf_price))
    
    # Try P/B (for financial companies)
    pb_price, reason = calculate_notional_price_book(ticker_info)
    if reason:
        reasons['P/B'] = reason
    if pb_price and pb_price > 0 and pb_price < actual_price * 10:
        methods.append(('P/B', pb_price))
    
    # Try Revenue multiple last (for growth companies, less reliable)
    revenue_price, reason = calculate_notional_price_revenue(ticker_info)
    if reason:
        reasons['Revenue'] = reason
    if revenue_price and revenue_price > 0 and revenue_price < actual_price * 10:
        methods.append(('Revenue', revenue_price))
    
    if not methods:
        if return_reasons:
            return None, None, reasons
        return None, None
    
    # Use the method with the most reasonable price
    if actual_price and actual_price > 0:
        valid_methods = [m for m in methods if m[1] > actual_price * 0.5 and m[1] < actual_price * 5]
        if valid_methods:
            best_method = min(valid_methods, key=lambda x: abs(x[1] - actual_price))
        else:
            best_method = min(methods, key=lambda x: abs(x[1] - actual_price))
    else:
        best_method = methods[0]
    
    if return_reasons:
        return best_method[0], best_method[1], reasons
    return best_method[0], best_method[1]


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
        logger.debug(f"Could not calculate recent trend for {symbol}: {e}")
        return None


def filter_fundamentals(results):
    """Filter stocks by fundamental metrics."""
    filtered = []
    
    for stock in results:
        profit_margin = stock.get('profit_margin')
        yearly_change = stock.get('yearly_change_pct', 0) or 0
        market_cap = stock.get('market_cap', 0) or 0
        revenue_growth = stock.get('revenue_growth')
        
        # Check profit margin - None means missing data (unprofitable companies often have None)
        if profit_margin is None or profit_margin < MIN_PROFIT_MARGIN:
            continue
        
        # Check yearly loss
        if yearly_change < MAX_YEARLY_LOSS:
            continue
        
        # Check market cap
        if market_cap < MIN_MARKET_CAP:
            continue
        
        # Check revenue growth - reject if revenue declining >5% (Phase 1 filter)
        if revenue_growth is not None and revenue_growth < MIN_REVENUE_GROWTH:
            logger.debug(f"Filtered out {stock.get('symbol', 'unknown')} - revenue declining: {revenue_growth:.2f}%")
            continue
        
        filtered.append(stock)
    
    return filtered


def filter_recent_trend(results):
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
        
        # Filter out stocks with strong upward trends (> MAX_RECENT_TREND_PCT)
        # Keep stocks with flat (0-2%) or slightly positive (2-5%) trends
        if trend_pct > MAX_RECENT_TREND_PCT:
            logger.debug(f"Filtered out {symbol} - recent trend too strong: {trend_pct:.2f}%")
            continue
        
        # Store trend for reference
        stock['recent_trend_pct'] = trend_pct
        filtered.append(stock)
    
    return filtered


def filter_longer_term_trend(results):
    """Filter out stocks with sustained declines over longer periods (Phase 1 filter)."""
    filtered = []
    
    for stock in results:
        symbol = stock.get('symbol')
        if not symbol:
            continue
        
        # Calculate trend over last 30 trading days
        trend_pct = calculate_recent_trend(symbol, days=30)
        
        # If we can't calculate trend, include the stock (don't exclude due to data issues)
        if trend_pct is None:
            filtered.append(stock)
            continue
        
        # Filter out stocks with sustained declines (> MAX_30_DAY_DECLINE)
        if trend_pct < MAX_30_DAY_DECLINE:
            logger.debug(f"Filtered out {symbol} - 30-day decline too severe: {trend_pct:.2f}%")
            continue
        
        # Store 30-day trend for reference
        stock['thirty_day_trend_pct'] = trend_pct
        filtered.append(stock)
    
    return filtered


def check_average_low(symbol, current_price):
    """
    Check if stock is trading near/below its 50-day SMA (TEMPORARY filter).
    Returns True if price is at or below threshold vs SMA50, or if filter is disabled.
    """
    # If filter is disabled, always return True (include stock)
    if MAX_PRICE_VS_SMA50 is None:
        return True
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='60d', interval='1d')  # Need 60d to calc 50-day SMA
        
        if hist.empty or len(hist) < 50:
            # If we can't calculate, include the stock (don't exclude due to data issues)
            return True
        
        # Calculate 50-day SMA
        import pandas as pd
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
        sma50 = hist['SMA50'].iloc[-1]
        
        if pd.isna(sma50) or sma50 <= 0:
            return True  # Include if can't calculate
        
        # Check if current price is within threshold of SMA50
        price_ratio = current_price / sma50
        return price_ratio <= MAX_PRICE_VS_SMA50
        
    except Exception as e:
        logger.debug(f"Could not check average low for {symbol}: {e}")
        return True  # Include if error (don't exclude due to data issues)


def filter_average_low(results):
    """Filter stocks to only include those trading near/below their 50-day average (TEMPORARY)."""
    # If filter is disabled, return all results unchanged
    if MAX_PRICE_VS_SMA50 is None:
        return results
    
    filtered = []
    
    for stock in results:
        symbol = stock.get('symbol')
        current_price = stock.get('actual_price', 0)
        
        if not symbol or not current_price:
            continue
        
        if check_average_low(symbol, current_price):
            filtered.append(stock)
        else:
            logger.debug(f"Filtered out {symbol} - trading too far above 50-day SMA")
    
    return filtered


def test_get_active_stocks(limit=50):
    """Test fetching active stocks."""
    print("=" * 60)
    print("TEST 1: Get Active Stocks")
    print("=" * 60)
    
    try:
        stocks = get_active_stocks(limit=limit)
        print(f"✓ Retrieved {len(stocks)} active stocks")
        
        if stocks:
            print(f"\nFirst 10 stocks:")
            print(f"{'Symbol':<10} {'Name':<40} {'Price':>10} {'Volume':>15}")
            print("-" * 75)
            for stock in stocks[:10]:
                print(f"{stock['symbol']:<10} {stock['name'][:38]:<40} ${stock['price']:>9.2f} {stock['volume']:>15,.0f}")
        
        return stocks
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_notional_price_calculation(stocks, limit=5):
    """Test notional price calculation for multiple discovered stocks."""
    print("\n" + "=" * 60)
    print(f"TEST 2: Notional Price Calculation (First {limit} Discovered Stocks)")
    print("=" * 60)
    
    if not stocks:
        print("⚠ No stocks to test")
        return []
    
    results = []
    test_stocks = stocks[:limit]
    
    for stock in test_stocks:
        symbol = stock['symbol']
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            method, notional_price = calculate_best_notional_price(info)
            
            if notional_price:
                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                discount_ratio = current_price / notional_price if notional_price > 0 else None
                discount_pct = (1 - discount_ratio) * 100 if discount_ratio else None
                
                results.append({
                    'symbol': symbol,
                    'method': method,
                    'current_price': current_price,
                    'notional_price': notional_price,
                    'discount_ratio': discount_ratio,
                    'discount_pct': discount_pct,
                })
            else:
                results.append({
                    'symbol': symbol,
                    'method': None,
                    'current_price': None,
                    'notional_price': None,
                    'discount_ratio': None,
                    'discount_pct': None,
                })
        except Exception as e:
            logger.debug(f"Error calculating notional price for {symbol}: {e}")
            results.append({
                'symbol': symbol,
                'method': None,
                'current_price': None,
                'notional_price': None,
                'discount_ratio': None,
                'discount_pct': None,
            })
    
    # Display results
    successful = [r for r in results if r['notional_price'] is not None]
    print(f"✓ Calculated notional prices for {len(successful)}/{len(results)} stocks\n")
    
    if successful:
        print(f"{'Symbol':<10} {'Method':<12} {'Current':>10} {'Notional':>10} {'Discount':>10}")
        print("-" * 52)
        for r in successful:
            print(f"{r['symbol']:<10} {r['method']:<12} ${r['current_price']:>9.2f} "
                  f"${r['notional_price']:>9.2f} {r['discount_pct']:>9.1f}%")
    
    failed = [r for r in results if r['notional_price'] is None]
    if failed:
        print(f"\n✗ Could not calculate notional price for {len(failed)} stocks:")
        for r in failed:
            print(f"  {r['symbol']}")
    
    return results


def test_fundamental_filtering(sample_results):
    """Test fundamental filtering logic."""
    print("\n" + "=" * 60)
    print("TEST 3: Fundamental Filtering")
    print("=" * 60)
    
    if not sample_results:
        print("⚠ No sample results to test")
        return []
    
    try:
        # Show why stocks are failing
        print(f"\nChecking {len(sample_results)} stocks against fundamental filters...")
        print(f"\nFilter criteria:")
        print(f"  MIN_PROFIT_MARGIN: {MIN_PROFIT_MARGIN}")
        print(f"  MAX_YEARLY_LOSS: {MAX_YEARLY_LOSS}%")
        print(f"  MIN_MARKET_CAP: ${MIN_MARKET_CAP:,.0f}")
        
        # Show first 5 stocks and why they pass/fail, plus NRDY specifically
        print(f"\nFirst 5 stocks filter check:")
        stocks_to_check = sample_results[:5]
        # Also check NRDY if it's not in the first 5
        nrdy_stock = next((s for s in sample_results if s['symbol'] == 'NRDY'), None)
        if nrdy_stock and nrdy_stock not in stocks_to_check:
            stocks_to_check.append(nrdy_stock)
            print("(Also checking NRDY since it was expected to pass)")
        
        for stock in stocks_to_check:
            pm = stock.get('profit_margin')
            yc = stock.get('yearly_change_pct', 0) or 0
            mc = stock.get('market_cap', 0) or 0
            
            pm_check = pm is not None and pm >= MIN_PROFIT_MARGIN
            yc_check = yc >= MAX_YEARLY_LOSS
            mc_check = mc >= MIN_MARKET_CAP
            
            pm_str = f"{pm:.2%}" if pm is not None else "None"
            status = "✓" if (pm_check and yc_check and mc_check) else "✗"
            reasons = []
            if not pm_check:
                reasons.append(f"pm={pm_str}")
            if not yc_check:
                reasons.append(f"yc={yc:.1f}%")
            if not mc_check:
                reasons.append(f"mc=${mc:,.0f}")
            reason_str = ", ".join(reasons) if reasons else "PASSES"
            print(f"  {status} {stock['symbol']}: {reason_str}")
        
        filtered = filter_fundamentals(sample_results)
        print(f"\n✓ Filtered {len(sample_results)} stocks down to {len(filtered)}")
        
        if filtered:
            print(f"\nSample of filtered stocks:")
            for stock in filtered[:5]:
                print(f"  {stock['symbol']}: margin={stock['profit_margin']:.2%}, "
                      f"yearly={stock['yearly_change_pct']:+.1f}%, "
                      f"mcap=${stock['market_cap']:,.0f}")
        
        return filtered
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_recent_trend_filtering(sample_results):
    """Test recent trend filtering logic."""
    print("\n" + "=" * 60)
    print("TEST 4: Recent Trend Filtering")
    print("=" * 60)
    
    if not sample_results:
        print("⚠ No sample results to test")
        return []
    
    try:
        filtered = filter_recent_trend(sample_results)
        print(f"✓ Filtered {len(sample_results)} stocks down to {len(filtered)}")
        
        print(f"\nFilter criteria:")
        print(f"  MAX_RECENT_TREND_PCT: {MAX_RECENT_TREND_PCT}% (excludes stocks with >{MAX_RECENT_TREND_PCT}% gain in last 5 days)")
        
        # Show trend info for filtered stocks
        if filtered:
            print(f"\nTrend information for filtered stocks:")
            for stock in filtered[:5]:
                trend = stock.get('recent_trend_pct')
                if trend is not None:
                    print(f"  {stock['symbol']}: {trend:+.2f}%")
                else:
                    print(f"  {stock['symbol']}: trend data unavailable")
        
        return filtered
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_undervalued_selection(sample_results):
    """Test undervalued stock selection logic."""
    print("\n" + "=" * 60)
    print("TEST 5: Undervalued Stock Selection")
    print("=" * 60)
    
    if not sample_results:
        print("⚠ No sample results to test")
        return []
    
    try:
        undervalued = [
            r for r in sample_results 
            if r['discount_ratio'] <= UNDERVALUED_RATIO_THRESHOLD
        ]
        
        print(f"✓ Filtered {len(sample_results)} stocks down to {len(undervalued)} undervalued stocks")
        print(f"\nFilter criteria:")
        print(f"  UNDERVALUED_RATIO_THRESHOLD: {UNDERVALUED_RATIO_THRESHOLD} (actual/notional <= {UNDERVALUED_RATIO_THRESHOLD})")
        
        if undervalued:
            print(f"\nUndervalued stocks:")
            print(f"{'Symbol':<10} {'Actual':>10} {'Notional':>10} {'Discount':>10} {'Method':<12}")
            print("-" * 52)
            for stock in undervalued[:10]:
                discount_pct = (1 - stock['discount_ratio']) * 100
                print(f"{stock['symbol']:<10} ${stock['actual_price']:>9.2f} "
                      f"${stock['notional_price']:>9.2f} {discount_pct:>9.1f}% "
                      f"{stock['method']:<12}")
        
        return undervalued
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_notional_price_methods(stocks, limit=5):
    """Test individual notional price calculation methods for stocks that passed all filters."""
    print("\n" + "=" * 60)
    count_str = f"({len(stocks)} stock{'s' if len(stocks) != 1 else ''} that passed all filters)" if stocks else ""
    print(f"TEST 6: Individual Notional Price Methods {count_str}")
    print("=" * 60)
    
    if not stocks:
        print("⚠ No stocks to test")
        return []
    
    test_stocks = stocks[:limit]
    all_results = []
    
    for stock_data in test_stocks:
        symbol = stock_data['symbol']
        print(f"\n{symbol}:")
        print("-" * 40)
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            print(f"Current Price: ${current_price:.2f}")
            
            methods = []
            method_results = {}
            
            # Test DCF
            dcf_price, dcf_reason = calculate_notional_price_dcf(info)
            if dcf_price:
                methods.append(('DCF', dcf_price))
                method_results['DCF'] = dcf_price
                print(f"  ✓ DCF: ${dcf_price:.2f}")
            else:
                reason_str = f" ({dcf_reason})" if dcf_reason else ""
                method_results['DCF'] = None
                print(f"  ✗ DCF: Not available{reason_str}")
            
            # Test P/E
            pe_price, pe_reason = calculate_notional_price_pe(info)
            if pe_price:
                methods.append(('P/E', pe_price))
                method_results['P/E'] = pe_price
                print(f"  ✓ P/E: ${pe_price:.2f}")
            else:
                reason_str = f" ({pe_reason})" if pe_reason else ""
                method_results['P/E'] = None
                print(f"  ✗ P/E: Not available{reason_str}")
            
            # Test EV/EBITDA
            ev_ebitda_price, ev_reason = calculate_notional_price_ev_ebitda(info)
            if ev_ebitda_price:
                methods.append(('EV/EBITDA', ev_ebitda_price))
                method_results['EV/EBITDA'] = ev_ebitda_price
                print(f"  ✓ EV/EBITDA: ${ev_ebitda_price:.2f}")
            else:
                reason_str = f" ({ev_reason})" if ev_reason else ""
                method_results['EV/EBITDA'] = None
                print(f"  ✗ EV/EBITDA: Not available{reason_str}")
            
            # Test Revenue
            revenue_price, rev_reason = calculate_notional_price_revenue(info)
            if revenue_price:
                methods.append(('Revenue', revenue_price))
                method_results['Revenue'] = revenue_price
                print(f"  ✓ Revenue: ${revenue_price:.2f}")
            else:
                reason_str = f" ({rev_reason})" if rev_reason else ""
                method_results['Revenue'] = None
                print(f"  ✗ Revenue: Not available{reason_str}")
            
            # Test P/B
            pb_price, pb_reason = calculate_notional_price_book(info)
            if pb_price:
                methods.append(('P/B', pb_price))
                method_results['P/B'] = pb_price
                print(f"  ✓ P/B: ${pb_price:.2f}")
            else:
                reason_str = f" ({pb_reason})" if pb_reason else ""
                method_results['P/B'] = None
                print(f"  ✗ P/B: Not available{reason_str}")
            
            # Show best method
            if methods:
                method, notional_price = calculate_best_notional_price(info)
                print(f"  → Best method: {method} (${notional_price:.2f})")
            
            all_results.append({
                'symbol': symbol,
                'methods': method_results,
                'best_method': method if methods else None,
                'best_price': notional_price if methods else None,
            })
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            logger.debug(f"Error testing methods for {symbol}: {e}")
            all_results.append({
                'symbol': symbol,
                'methods': {},
                'best_method': None,
                'best_price': None,
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'Symbol':<10} {'DCF':<8} {'P/E':<8} {'EV/EBITDA':<10} {'Revenue':<8} {'P/B':<8} {'Best':<12}")
    print("-" * 72)
    for r in all_results:
        dcf = "✓" if r['methods'].get('DCF') else "✗"
        pe = "✓" if r['methods'].get('P/E') else "✗"
        ev = "✓" if r['methods'].get('EV/EBITDA') else "✗"
        rev = "✓" if r['methods'].get('Revenue') else "✗"
        pb = "✓" if r['methods'].get('P/B') else "✗"
        best = f"{r['best_method']}" if r['best_method'] else "N/A"
        print(f"{r['symbol']:<10} {dcf:<8} {pe:<8} {ev:<10} {rev:<8} {pb:<8} {best:<12}")
    
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Test Yahoo advisor discovery logic")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit for active stocks test (default: 20)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="AAPL",
        help="Symbol to test notional price calculation (default: AAPL)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--min-volume",
        type=int,
        default=None,
        help="Minimum daily volume threshold (filters out illiquid stocks)"
    )
    parser.add_argument(
        "--max-volume",
        type=int,
        default=None,
        help="Maximum daily volume threshold (filters out over-traded stocks)"
    )
    parser.add_argument(
        "--low-volume",
        action="store_true",
        help="Sort by volume ascending (lowest first) to find less discovered stocks"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(message)s"
    )
    
    print("Yahoo Advisor Discovery Pipeline")
    print("=" * 60)
    print(f"Testing the actual discover() method logic (no Django, no database)")
    print()
    
    # STEP 1: Get stocks based on volume strategy
    volume_strategy = "Low Volume (Less Discovered)" if args.low_volume else "High Volume (Most Active)"
    print(f"STEP 1: Get Stocks - {volume_strategy}")
    print("-" * 60)
    
    volume_filters = []
    if args.min_volume:
        volume_filters.append(f"min_volume >= {args.min_volume:,}")
    if args.max_volume:
        volume_filters.append(f"max_volume <= {args.max_volume:,}")
    if volume_filters:
        print(f"Volume filters: {', '.join(volume_filters)}")
    
    stocks = get_active_stocks(
        limit=args.limit,
        min_volume=args.min_volume,
        max_volume=args.max_volume,
        sort_volume_asc=args.low_volume
    )
    if not stocks:
        print("✗ No active stocks retrieved")
        return
    
    strategy_desc = "low volume" if args.low_volume else "high volume"
    print(f"✓ Retrieved {len(stocks)} stocks (sorted by {strategy_desc})")
    
    if stocks:
        volumes = [s['volume'] for s in stocks if s['volume'] > 0]
        if volumes:
            print(f"Volume range: {min(volumes):,.0f} - {max(volumes):,.0f} (avg: {sum(volumes)/len(volumes):,.0f})")
    
    print(f"\nTop {min(10, len(stocks))} stocks by volume:")
    for i, stock in enumerate(stocks[:10], 1):
        print(f"  {i:2d}. {stock['symbol']:<6} {stock['name'][:40]:<40} ${stock['price']:>6.2f}  Vol: {stock['volume']:>12,.0f}")
    
    # STEP 2: Calculate notional prices
    print(f"\n\nSTEP 2: Calculate Notional Prices")
    print("-" * 60)
    sample_results = []
    
    for stock in stocks:
        try:
            ticker = yf.Ticker(stock['symbol'])
            info = ticker.info
            
            method, notional_price = calculate_best_notional_price(info)
            if notional_price and notional_price > 0:
                actual_price = stock['price']
                discount_ratio = actual_price / notional_price
                
                # Store None if profitMargins is missing (unprofitable companies often have None)
                profit_margin = info.get('profitMargins')
                # fiftyTwoWeekChangePercent is already a percentage (e.g., -29.47 = -29.47%), don't multiply by 100
                yearly_change = (info.get('fiftyTwoWeekChangePercent', 0) or 0)
                market_cap = info.get('marketCap', 0) or 0
                # Revenue growth as percentage (e.g., -0.069 = -6.9%), multiply by 100 to get percentage
                revenue_growth = (info.get('revenueGrowth') * 100) if info.get('revenueGrowth') is not None else None
                
                sample_results.append({
                    'symbol': stock['symbol'],
                    'name': stock['name'],
                    'actual_price': actual_price,
                    'notional_price': notional_price,
                    'discount_ratio': discount_ratio,
                    'method': method,
                    'profit_margin': profit_margin,
                    'yearly_change_pct': yearly_change,
                    'market_cap': market_cap,
                    'revenue_growth': revenue_growth,  # Phase 1: revenue growth filter
                })
        except Exception as e:
            logger.debug(f"Error processing {stock['symbol']}: {e}")
            continue
    
    print(f"✓ Calculated notional prices for {len(sample_results)}/{len(stocks)} stocks")
    if sample_results:
        print(f"\nStocks with notional prices ({len(sample_results)} stocks):")
        for stock in sample_results[:10]:
            discount_pct = (1 - stock['discount_ratio']) * 100
            print(f"  {stock['symbol']:<6} ${stock['actual_price']:>6.2f} → ${stock['notional_price']:>6.2f} "
                  f"({discount_pct:>5.1f}% discount, {stock['method']})")
        if len(sample_results) > 10:
            print(f"  ... and {len(sample_results) - 10} more")
    
    if not sample_results:
        print("\n⚠ Pipeline stopped: No stocks with valid notional prices")
        return
    
    # STEP 3: Filter by fundamentals (profit margin, yearly change, market cap, revenue growth)
    print(f"\n\nSTEP 3: Filter by Fundamentals")
    print("-" * 60)
    print(f"Criteria: profit_margin >= {MIN_PROFIT_MARGIN:.0%}, yearly_change >= {MAX_YEARLY_LOSS:.1f}%, market_cap >= ${MIN_MARKET_CAP:,.0f}, revenue_growth >= {MIN_REVENUE_GROWTH:.1f}%")
    
    # Show why stocks are failing
    failing_reasons = {'profit_margin': 0, 'yearly_change': 0, 'market_cap': 0, 'revenue_growth': 0}
    for stock in sample_results:
        pm = stock.get('profit_margin')
        yc = stock.get('yearly_change_pct', 0) or 0
        mc = stock.get('market_cap', 0) or 0
        rg = stock.get('revenue_growth')
        
        if pm is None or pm < MIN_PROFIT_MARGIN:
            failing_reasons['profit_margin'] += 1
        if yc < MAX_YEARLY_LOSS:
            failing_reasons['yearly_change'] += 1
        if mc < MIN_MARKET_CAP:
            failing_reasons['market_cap'] += 1
        if rg is not None and rg < MIN_REVENUE_GROWTH:
            failing_reasons['revenue_growth'] += 1
    
    print(f"\nFailure reasons (stocks can fail multiple criteria):")
    print(f"  Profit margin < 0%: {failing_reasons['profit_margin']} stocks")
    print(f"  Yearly loss > 50%: {failing_reasons['yearly_change']} stocks")
    print(f"  Market cap < $25M: {failing_reasons['market_cap']} stocks")
    print(f"  Revenue decline > 5%: {failing_reasons['revenue_growth']} stocks (NEW)")
    
    fundamental_filtered = filter_fundamentals(sample_results)
    print(f"\n✓ Filtered {len(sample_results)} stocks down to {len(fundamental_filtered)}")
    
    if fundamental_filtered:
        print(f"\nStocks passing fundamental filters ({len(fundamental_filtered)} stocks):")
        for stock in fundamental_filtered:
            pm_str = f"{stock['profit_margin']:.2%}" if stock['profit_margin'] is not None else "N/A"
            print(f"  {stock['symbol']:<6} margin={pm_str:>7}, yearly={stock['yearly_change_pct']:>6.1f}%, "
                  f"mcap=${stock['market_cap']:>12,.0f}")
    else:
        print("\n⚠ Pipeline stopped: No stocks passed fundamental filters")
        return
    
    # STEP 4: Filter by recent trend
    print(f"\n\nSTEP 4: Filter by Recent Trend (5-day)")
    print("-" * 60)
    print(f"Criteria: recent_5day_trend <= {MAX_RECENT_TREND_PCT:.1f}% (exclude stocks that already ran up)")
    
    trend_filtered = filter_recent_trend(fundamental_filtered)
    print(f"✓ Filtered {len(fundamental_filtered)} stocks down to {len(trend_filtered)}")
    
    if not trend_filtered:
        print("\n⚠ Pipeline stopped: No stocks passed recent trend filter")
        return
    
    # STEP 4b: Filter by longer-term trend (30-day) - Phase 1 filter
    print(f"\n\nSTEP 4b: Filter by Longer-Term Trend (30-day)")
    print("-" * 60)
    print(f"Criteria: 30day_trend >= {MAX_30_DAY_DECLINE:.1f}% (exclude stocks with sustained declines)")
    
    longer_term_filtered = filter_longer_term_trend(trend_filtered)
    print(f"✓ Filtered {len(trend_filtered)} stocks down to {len(longer_term_filtered)}")
    
    if not longer_term_filtered:
        print("\n⚠ Pipeline stopped: No stocks passed longer-term trend filter")
        return
    
    # STEP 4c: Filter by average low (TEMPORARY - experimental)
    if MAX_PRICE_VS_SMA50 is not None:
        print(f"\n\nSTEP 4c: Filter by Average Low (50-day SMA) - TEMPORARY")
        print("-" * 60)
        print(f"Criteria: price <= {MAX_PRICE_VS_SMA50:.0%} of 50-day SMA (exclude stocks trading too far above average)")
        
        average_low_filtered = filter_average_low(longer_term_filtered)
        print(f"✓ Filtered {len(longer_term_filtered)} stocks down to {len(average_low_filtered)}")
        
        if average_low_filtered:
            print(f"\nStocks passing average low filter ({len(average_low_filtered)} stocks):")
            for stock in average_low_filtered:
                trend_5d = stock.get('recent_trend_pct', 0)
                trend_30d = stock.get('thirty_day_trend_pct', 0)
                trend_5d_str = f"{trend_5d:+.2f}%" if trend_5d is not None else "N/A"
                trend_30d_str = f"{trend_30d:+.2f}%" if trend_30d is not None else "N/A"
                discount_pct = (1 - stock['discount_ratio']) * 100
                print(f"  {stock['symbol']:<6} 5d={trend_5d_str:>7}, 30d={trend_30d_str:>7}, discount={discount_pct:>5.1f}%, method={stock['method']}")
        else:
            print("\n⚠ Pipeline stopped: No stocks passed average low filter")
            return
    else:
        average_low_filtered = longer_term_filtered
        print(f"\n⚠ Average low filter disabled (MAX_PRICE_VS_SMA50 = None)")
    
    # STEP 5: Filter for undervalued (discount ratio <= threshold)
    print(f"\n\nSTEP 5: Select Undervalued Stocks")
    print("-" * 60)
    print(f"Criteria: discount_ratio <= {UNDERVALUED_RATIO_THRESHOLD} (actual/notional <= {UNDERVALUED_RATIO_THRESHOLD})")
    
    undervalued_stocks = [
        r for r in average_low_filtered 
        if r['discount_ratio'] <= UNDERVALUED_RATIO_THRESHOLD
    ]
    print(f"✓ Filtered {len(average_low_filtered)} stocks down to {len(undervalued_stocks)} undervalued stocks")
    
    if undervalued_stocks:
        print(f"\n✓ FINAL RESULT: {len(undervalued_stocks)} stock(s) would be discovered:")
        print(f"\n{'Symbol':<8} {'Actual':>10} {'Notional':>10} {'Discount':>10} {'Upside':>10} {'Method':<12}")
        print("-" * 70)
        for stock in undervalued_stocks:
            discount_pct = (1 - stock['discount_ratio']) * 100
            upside = stock['notional_price'] - stock['actual_price']
            upside_pct = (upside / stock['actual_price']) * 100 if stock['actual_price'] > 0 else 0
            print(f"{stock['symbol']:<8} ${stock['actual_price']:>9.2f} ${stock['notional_price']:>9.2f} "
                  f"{discount_pct:>9.1f}% {upside_pct:>9.1f}% {stock['method']:<12}")
    else:
        print("\n⚠ Pipeline stopped: No undervalued stocks found")
    
    
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"✓ Pipeline completed (no Django, no database)")
    print(f"\nResults:")
    print(f"  Step 1: {len(stocks)} stocks retrieved")
    print(f"  Step 2: {len(sample_results)} stocks with notional prices")
    print(f"  Step 3: {len(fundamental_filtered) if 'fundamental_filtered' in locals() else 0} stocks passed fundamentals")
    print(f"  Step 4: {len(trend_filtered) if 'trend_filtered' in locals() else 0} stocks passed 5-day trend filter")
    print(f"  Step 4b: {len(longer_term_filtered) if 'longer_term_filtered' in locals() else 0} stocks passed 30-day trend filter")
    if MAX_PRICE_VS_SMA50 is not None:
        print(f"  Step 4c: {len(average_low_filtered) if 'average_low_filtered' in locals() else 0} stocks passed average low filter (TEMPORARY)")
    print(f"  Step 5: {len(undervalued_stocks) if 'undervalued_stocks' in locals() else 0} stocks would be discovered")


if __name__ == "__main__":
    main()
