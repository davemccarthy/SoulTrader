#!/usr/bin/env python
"""
Crazy Trading Method Backtest

Strategy:
1. On given date, simulate 10 buys of middle-of-the-road average volume/price stocks
2. Calculate 10% of stock price as target profit (e.g., $110 stock → want $10 profit = $120 target)
3. Track all stocks daily from back date:
   - If up +10% from average price → sell at profit (exit position)
   - If down -10% from average price → buy 10% more stock (average down)

This is a dollar-cost averaging down strategy with profit-taking at 10%.
"""

import sys
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import random

# Add Django project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    django.setup()
    from core.models import Stock
    HAS_DJANGO = True
except Exception as e:
    HAS_DJANGO = False
    print(f"⚠️  Django not available: {e}")
    print("   Running in standalone mode without Django models")

# Polygon API for stock discovery
try:
    from polygon import RESTClient
    HAS_POLYGON = True
except ImportError:
    HAS_POLYGON = False
    print("⚠️  Polygon not available. Install with: pip install polygon-api-client")

# Import wavelet analysis from test_oscilla.py
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_oscilla import (
        get_historical_data_yfinance as oscilla_get_historical_data,
        wavelet_trade_engine,
        MIN_RR as OSCILLA_MIN_RR,
        LOOKBACK_DAYS as OSCILLA_LOOKBACK_DAYS,
        TURN_CONFIRMATION_ENABLED as OSCILLA_TURN_CONFIRMATION,
        MIN_AVG_VOLUME as OSCILLA_MIN_AVG_VOLUME,
        REL_VOLUME_MIN as OSCILLA_REL_VOLUME_MIN,
        REL_VOLUME_MAX as OSCILLA_REL_VOLUME_MAX
    )
    HAS_OSCILLA = True
except ImportError as e:
    HAS_OSCILLA = False
    print(f"⚠️  Oscilla wavelet analysis not available: {e}")
    print("   Will use simple volume/price filtering instead")


def fetch_stocks_for_date(reference_date, min_price=10.0, max_price=100.0, min_volume=1_000_000):
    """
    Fetch stocks using Polygon's get_grouped_daily_aggs (1 API call for all stocks on a date).
    Similar to test_oscilla.py approach.
    
    Args:
        reference_date: Date string (YYYY-MM-DD) or date object
        min_price: Minimum stock price
        max_price: Maximum stock price
        min_volume: Minimum volume filter
    
    Returns:
        pandas DataFrame with columns: ticker, price, today_volume
    """
    if not HAS_POLYGON:
        print("    ❌ Polygon not available. Install with: pip install polygon-api-client")
        sys.exit(1)
    
    polygon_api_key = os.getenv('POLYGON_API_KEY')
    if not polygon_api_key:
        print("    ❌ POLYGON_API_KEY environment variable not set")
        sys.exit(1)
    
    try:
        if isinstance(reference_date, datetime):
            date_str = reference_date.strftime("%Y-%m-%d")
        elif hasattr(reference_date, 'strftime'):
            date_str = reference_date.strftime("%Y-%m-%d")
        else:
            date_str = str(reference_date)
        
        client = RESTClient(polygon_api_key)
        
        print(f"    Fetching stocks from Polygon for {date_str}...")
        aggs = client.get_grouped_daily_aggs(
            locale="us",
            date=date_str,
            adjusted=False
        )
        
        rows = []
        for agg in aggs:
            if (
                min_price <= agg.close <= max_price and
                agg.volume >= min_volume
            ):
                rows.append({
                    "ticker": agg.ticker,
                    "price": float(agg.close),
                    "today_volume": int(agg.volume)
                })
        
        df = pd.DataFrame(rows)
        
        if not df.empty:
            print(f"    Found {len(df)} stocks in ${min_price:.2f}-${max_price:.2f} price range with volume >= {min_volume:,}")
        
        return df
        
    except Exception as e:
        print(f"    ❌ Error fetching stocks from Polygon: {e}")
        sys.exit(1)


def find_stocks_simple_filter(df, count, random_seed=42):
    """Simple volume/price filtering fallback (original method)."""
    # Filter for middle-of-the-road stocks by volume
    df = df.sort_values("today_volume")
    total = len(df)
    
    # Take stocks from middle third of volume range (avoid too low or too high)
    start_idx = total // 3
    end_idx = (total * 2) // 3
    middle_stocks = df.iloc[start_idx:end_idx].copy()
    
    if middle_stocks.empty:
        middle_stocks = df
    
    # Also filter by price - prefer mid-range prices
    middle_stocks = middle_stocks.sort_values("price")
    price_total = len(middle_stocks)
    
    # Take from middle 60% of price range
    price_start = int(price_total * 0.2)
    price_end = int(price_total * 0.8)
    price_filtered = middle_stocks.iloc[price_start:price_end].copy() if price_total > 10 else middle_stocks
    
    if price_filtered.empty:
        price_filtered = middle_stocks
    
    # Shuffle selection (deterministic by default, random if seed is None)
    price_filtered = price_filtered.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    # Take sample
    selected = price_filtered.head(count) if len(price_filtered) > count else price_filtered
    symbols = selected["ticker"].tolist()
    
    return symbols


def find_middle_of_road_stocks(date, count=10, price_min=10.0, price_max=100.0, min_volume=1_000_000,
                               use_wavelet_filter=True, min_rr=1.8, lookback_days=40, etf_only=False, random_seed=42,
                               max_stocks_to_test=None):
    """
    Find stocks using Oscilla wavelet filter.
    Uses Polygon to fetch stocks, then filters for stocks that pass wavelet test.
    
    Args:
        date: Date to find stocks on
        count: Number of stocks to return
        price_min: Minimum price filter
        price_max: Maximum price filter
        min_volume: Minimum volume filter
        use_wavelet_filter: If True, use wavelet test (Oscilla method). If False, use simple volume/price filtering.
        min_rr: Minimum reward:risk ratio for wavelet filter (default: 1.8)
        lookback_days: Lookback days for wavelet analysis (default: 40)
        etf_only: If True, only return ETFs (default: False)
        random_seed: Random seed for reproducible selection (default: 42 for deterministic, None for random)
    
    Returns:
        List of stock symbols
    """
    asset_type = "ETFs" if etf_only else "stocks"
    print(f"  Finding {count} {asset_type} on {date}...")
    
    if isinstance(date, datetime):
        ref_date = date.date()
        ref_dt = date
    elif hasattr(date, 'date'):
        ref_date = date
        ref_dt = datetime.combine(ref_date, datetime.min.time())
    else:
        if isinstance(date, str):
            ref_date = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            ref_date = date
        ref_dt = datetime.combine(ref_date, datetime.min.time())
    
    # Fetch stocks using Polygon (1 API call)
    df = fetch_stocks_for_date(ref_date, price_min, price_max, min_volume)
    
    # If ETF-only mode, use known ETF list instead of wavelet filtering
    if etf_only:
        print("    Using known ETF list (wavelet filtering skipped for ETFs)...")
        known_etfs = get_known_etfs()
        
        # Filter known ETFs by price/volume from Polygon data if available
        if not df.empty:
            etf_df = df[df['ticker'].isin(known_etfs)].copy()
            if etf_df.empty:
                # No known ETFs in Polygon data, use known list directly
                print("    ⚠️  No known ETFs found in Polygon data, using known list directly")
                etf_list = known_etfs
            else:
                etf_list = etf_df['ticker'].tolist()
        else:
            # No Polygon data, use known list directly
            etf_list = known_etfs
        
        # Select from available ETFs (deterministic by default, random if seed is None)
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(etf_list)
        symbols = etf_list[:count]
        
        print(f"    ✅ Found {len(symbols)} ETFs from known list:")
        for symbol in symbols:
            print(f"      {symbol}")
        return symbols
    
    if df.empty:
        print(f"    ❌ No stocks found from Polygon for {date}")
        sys.exit(1)
    
    # If wavelet filter not available or disabled, use simple filtering
    if not HAS_OSCILLA or not use_wavelet_filter:
        print("    Using simple volume/price filtering...")
        symbols = find_stocks_simple_filter(df, count, random_seed)
        print(f"    ✅ Found {len(symbols)} stocks (randomized selection):")
        selected_df = df[df['ticker'].isin(symbols)]
        for _, row in selected_df.iterrows():
            print(f"      {row['ticker']}: ${row['price']:.2f}, volume={row['today_volume']:,}")
        return symbols
    
    # Use wavelet filter (Oscilla method)
    print("    Filtering stocks using Oscilla wavelet test...")
    
    # Calculate start date for historical data (need at least 64 trading days)
    min_calendar_days = int(64 * 7 / 5) + 30  # ~120 calendar days for 64 trading days
    wavelet_lookback_days = max(min_calendar_days, lookback_days * 2)
    start_date = (ref_dt - timedelta(days=wavelet_lookback_days)).strftime("%Y-%m-%d")
    ref_date_str = ref_date.strftime("%Y-%m-%d")
    
    print(f"    Using historical data from {start_date} to {ref_date_str} (need ≥64 trading days)")
    
    # Limit number of stocks to test if specified
    if max_stocks_to_test is not None and max_stocks_to_test > 0:
        df = df.head(max_stocks_to_test)
        print(f"    Limiting test to first {max_stocks_to_test} stocks (for faster testing)")
    
    # Test each stock with wavelet analysis
    passed_stocks = []
    failed_stocks = []  # Collect failed stocks with wave positions
    tested_count = 0
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        tested_count = i
        ticker = row["ticker"]
        
        if (i % 50 == 0):
            print(f"      Testing {i}/{len(df)} stocks... ({len(passed_stocks)} passed so far)")
        
        try:
            # Get historical data
            df_price = oscilla_get_historical_data(ticker, start_date, ref_date_str)
            
            if df_price.empty:
                continue
            
            if len(df_price) < 64:
                continue
            
            # Volume filtering: check average volume and relative volume (middle-of-the-road stocks)
            if 'volume' in df_price.columns and len(df_price) >= OSCILLA_LOOKBACK_DAYS:
                # Calculate average volume over lookback period
                df_price_sorted = df_price.sort_values("date").tail(OSCILLA_LOOKBACK_DAYS)
                avg_volume = df_price_sorted["volume"].mean()
                
                # Check minimum average volume
                if avg_volume < OSCILLA_MIN_AVG_VOLUME:
                    print(f"      {ticker}: ✗ Avg volume {avg_volume:,.0f} < MIN_AVG_VOLUME {OSCILLA_MIN_AVG_VOLUME:,}")
                    continue
                
                # Get today's volume from Polygon data
                today_volume = row.get("today_volume", 0)
                if today_volume > 0 and avg_volume > 0:
                    # Calculate relative volume (today vs average)
                    rel_volume = today_volume / avg_volume
                    
                    # Filter: only accept stocks with relative volume in range [REL_VOLUME_MIN, REL_VOLUME_MAX]
                    # This finds "middle-of-the-road" stocks with stable volume patterns
                    if rel_volume < OSCILLA_REL_VOLUME_MIN:
                        print(f"      {ticker}: ✗ Rel volume {rel_volume:.2f} < MIN {OSCILLA_REL_VOLUME_MIN:.1f} (avg={avg_volume:,.0f}, today={today_volume:,.0f})")
                        continue
                    elif rel_volume > OSCILLA_REL_VOLUME_MAX:
                        print(f"      {ticker}: ✗ Rel volume {rel_volume:.2f} > MAX {OSCILLA_REL_VOLUME_MAX:.1f} (avg={avg_volume:,.0f}, today={today_volume:,.0f})")
                        continue
            
            # Convert to pandas Series for wavelet_trade_engine
            price_series = pd.Series(df_price["close"].values, index=range(len(df_price)))
            low_series = pd.Series(df_price["low"].values, index=range(len(df_price)))
            
            # Run wavelet analysis
            wave_result = wavelet_trade_engine(
                price_series, 
                min_rr=min_rr, 
                low_series=low_series,
                turn_confirmation_enabled=OSCILLA_TURN_CONFIRMATION
            )
            
            # Collect failed stock info (always, for summary)
            if not wave_result.get("accepted", False):
                failed_info = {
                    "ticker": ticker,
                    "reason": wave_result.get("reason", "Unknown"),
                    "wave_position": wave_result.get("wave_position"),
                    "consistency": wave_result.get("consistency"),
                    "rr": wave_result.get("rr")
                }
                failed_stocks.append(failed_info)
            
            # If stock passes wavelet test, check additional filters
            if wave_result.get("accepted", False):
                # Filter wave_position: only accept between -50.0 and 0.0
                wave_position = wave_result.get("wave_position", 0)
                if wave_position < -50.0 or wave_position > 0.0:
                    # Track this as a filter rejection too
                    failed_info = {
                        "ticker": ticker,
                        "reason": f"Wave position out of range ({wave_position:.3f})",
                        "wave_position": wave_position,
                        "consistency": wave_result.get("consistency"),
                        "rr": wave_result.get("reward_risk")
                    }
                    failed_stocks.append(failed_info)
                    continue
                
                try:
                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.info
                    
                    # Check if company is profitable (gains >= 0)
                    trailing_eps = info.get('trailingEps')
                    net_income = info.get('netIncomeToCommon') or info.get('netIncome')
                    profit_margin = info.get('profitMargins')
                    
                    # Company is profitable if any of these are >= 0
                    is_profitable = (
                        (trailing_eps is not None and trailing_eps >= 0) or
                        (net_income is not None and net_income >= 0) or
                        (profit_margin is not None and profit_margin >= 0)
                    )
                    
                    # Skip unprofitable companies
                    if not is_profitable:
                        failed_info = {
                            "ticker": ticker,
                            "reason": "Not profitable",
                            "wave_position": wave_position,
                            "consistency": wave_result.get("consistency"),
                            "rr": wave_result.get("reward_risk")
                        }
                        failed_stocks.append(failed_info)
                        continue
                        
                except Exception as e:
                    # If we can't check, skip to be safe
                    continue
                
                # Stock passed wavelet test, wave_position filter, and profitability check
                passed_stocks.append({
                    "ticker": ticker,
                    "price": row["price"],
                    "today_volume": row["today_volume"],
                    "wave_position": wave_position,
                    "rr": wave_result.get("reward_risk", 0)
                })
                
                if len(passed_stocks) >= count * 3:  # Get 3x the needed amount for variety
                    break
        
        except Exception as e:
            # Only show error for first few failures to avoid spam
            if i <= 5:
                print(f"      ⚠️  {ticker}: Error in wavelet test - {str(e)[:80]}")
            continue
    
    # Print summary of failed stocks with wave positions
    if failed_stocks:
        print(f"\n{'='*80}")
        print(f"FAILED FILTER TEST SUMMARY ({len(failed_stocks)} stocks)")
        print(f"{'='*80}")
        print(f"{'Ticker':<10} {'Reason':<30} {'Wave Pos':<12} {'Consistency':<12} {'R:R':<8}")
        print("-" * 80)
        
        # Sort by wave_position if available (to see distribution)
        failed_with_wp = [f for f in failed_stocks if f.get("wave_position") is not None]
        failed_without_wp = [f for f in failed_stocks if f.get("wave_position") is None]
        
        # Sort failed_with_wp by wave_position
        failed_with_wp.sort(key=lambda x: x.get("wave_position", 0))
        
        # Print stocks with wave positions
        for stock in failed_with_wp:
            wp = stock.get("wave_position")
            wp_str = f"{wp:.3f}" if wp is not None else "N/A"
            consistency = stock.get("consistency")
            consistency_str = f"{consistency:.3f}" if consistency is not None else "N/A"
            rr = stock.get("rr")
            rr_str = f"{rr:.2f}" if rr is not None else "N/A"
            reason = stock.get("reason", "Unknown")[:28]  # Truncate long reasons
            print(f"{stock['ticker']:<10} {reason:<30} {wp_str:<12} {consistency_str:<12} {rr_str:<8}")
        
        # Print stocks without wave positions
        for stock in failed_without_wp:
            reason = stock.get("reason", "Unknown")[:28]
            print(f"{stock['ticker']:<10} {reason:<30} {'N/A':<12} {'N/A':<12} {'N/A':<8}")
        
        # Print statistics
        if failed_with_wp:
            wave_positions = [f.get("wave_position") for f in failed_with_wp if f.get("wave_position") is not None]
            if wave_positions:
                print(f"\nWave Position Statistics (for {len(wave_positions)} stocks with wave positions):")
                print(f"  Min: {min(wave_positions):.3f}")
                print(f"  Max: {max(wave_positions):.3f}")
                print(f"  Mean: {np.mean(wave_positions):.3f}")
                print(f"  Median: {np.median(wave_positions):.3f}")
        
        print(f"\n{'='*80}\n")
    
    if not passed_stocks:
        print(f"    ❌ No stocks passed wavelet test (tested {tested_count} stocks)")
        sys.exit(1)
    
    # Select from passed stocks (deterministic by default, random if seed is None)
    passed_df = pd.DataFrame(passed_stocks)
    passed_df = passed_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    selected = passed_df.head(count)
    symbols = selected["ticker"].tolist()
    
    print(f"    ✅ Found {len(symbols)} stocks that passed Oscilla wavelet test:")
    for _, row in selected.iterrows():
        wp = row.get('wave_position', 0)
        rr = row.get('rr', 0)
        print(f"      {row['ticker']}: ${row['price']:.2f}, wave_pos={wp:.2f}, R:R={rr:.2f}")
    
    return symbols


def get_known_etfs():
    """Known list of popular ETFs."""
    return [
        "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "VEA", "VWO", "EFA", "EEM",
        "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLC",
        "SMH", "SOXX", "IBB", "XBI", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF",
        "TQQQ", "SQQQ", "SPXL", "SPXS", "UDOW", "SDOW", "TNA", "TZA",
        "YANG", "FXI", "EWZ", "EWJ", "EWY", "EWH", "EPI", "INDA",
        "GDX", "GDXJ", "SLV", "GLD", "USO", "UCO", "BOIL", "KOLD",
        "HYG", "LQD", "TLT", "SHY", "IEF", "TIP", "AGG", "BND",
        "VUG", "VTV", "VYM", "SCHD", "DGRO", "SPHD", "SPYD", "DVY"
    ]


class Position:
    """Track a single stock position."""
    def __init__(self, symbol: str, buy_date: datetime, initial_price: float, initial_shares: int, 
                 profit_target_pct: float = 0.10, average_down_threshold_pct: float = 0.10,
                 diminishing_tp_days: int = 45):
        self.symbol = symbol
        self.buy_date = buy_date
        self.initial_price = initial_price  # Store initial price for profit calculation
        self.profit_target_pct = profit_target_pct  # Profit target as percentage (e.g., 0.10 = 10%)
        self.average_down_threshold_pct = average_down_threshold_pct  # Threshold for averaging down (e.g., 0.10 = 10% drop)
        self.diminishing_tp_days = diminishing_tp_days  # Days over which TP diminishes to break-even
        self.purchases = []  # List of (date, price, shares, cost)
        self.sell_date: Optional[datetime] = None
        self.sell_price: Optional[float] = None
        self.sell_reason: Optional[str] = None
        self.target_price_reached = False  # Track if we've ever hit the target price
        self.recent_prices = []  # Track recent prices for trend detection (last 5 days)
        
        # Initial purchase
        self.add_purchase(buy_date, initial_price, initial_shares)
    
    def add_purchase(self, date: datetime, price: float, shares: int):
        """Add a purchase (initial or average down)."""
        cost = price * shares
        self.purchases.append({
            'date': date,
            'price': price,
            'shares': shares,
            'cost': cost
        })
    
    def get_total_shares(self) -> int:
        """Get total shares owned."""
        return sum(p['shares'] for p in self.purchases)
    
    def get_total_cost(self) -> float:
        """Get total cost basis."""
        return sum(p['cost'] for p in self.purchases)
    
    def get_average_price(self) -> float:
        """Get average cost per share."""
        total_shares = self.get_total_shares()
        if total_shares == 0:
            return 0.0
        return self.get_total_cost() / total_shares
    
    def get_target_profit_amount(self) -> float:
        """
        Calculate the target profit amount per initial share.
        Target profit = profit_target_pct of initial stock price as dollar amount.
        Example: $110 stock with 10% target → $11 profit per share.
        Example: $110 stock with 5% target → $5.50 profit per share.
        """
        return self.initial_price * self.profit_target_pct
    
    def get_target_price(self, current_date: Optional[datetime] = None) -> float:
        """
        Get target sell price with diminishing target over time.
        Target diminishes from original target down to break-even (average_price) over diminishing_tp_days.
        
        Args:
            current_date: Current date to calculate days held (if None, uses buy_date for original target)
        
        Returns:
            Target price (diminishing over time if current_date provided)
        """
        target_profit_per_initial_share = self.get_target_profit_amount()
        initial_shares = self.purchases[0]['shares'] if self.purchases else 1
        total_target_profit = target_profit_per_initial_share * initial_shares  # Fixed profit amount
        
        total_shares = self.get_total_shares()
        total_cost = self.get_total_cost()
        
        if total_shares == 0:
            return 0.0
        
        # Calculate original target price (without diminishing)
        original_target_price = (total_cost + total_target_profit) / total_shares
        avg_price = self.get_average_price()
        
        # Apply diminishing target if current_date is provided
        if current_date is not None:
            days_held = (current_date.date() - self.buy_date.date()).days
            if days_held <= self.diminishing_tp_days:
                progress = float(days_held) / float(self.diminishing_tp_days) if self.diminishing_tp_days > 0 else 1.0
                # Diminish from original_target_price down to avg_price (break-even)
                target_price = original_target_price - progress * (original_target_price - avg_price)
            else:
                # After diminishing period, target = break-even
                target_price = avg_price
        else:
            # No date provided, return original target
            target_price = original_target_price
        
        return target_price
    
    def get_average_down_price(self) -> float:
        """Get price at which we should average down (threshold% below average)."""
        avg_price = self.get_average_price()
        return avg_price * (1.0 - self.average_down_threshold_pct)
    
    def is_closed(self) -> bool:
        """Check if position is closed."""
        return self.sell_date is not None
    
    def sell(self, date: datetime, price: float, reason: str):
        """Close the position."""
        self.sell_date = date
        self.sell_price = price
        self.sell_reason = reason
    
    def get_profit_loss(self) -> float:
        """Get profit/loss if closed."""
        if not self.is_closed():
            return 0.0
        total_cost = self.get_total_cost()
        total_value = self.sell_price * self.get_total_shares()
        return total_value - total_cost
    
    def get_profit_loss_pct(self) -> float:
        """Get profit/loss percentage if closed."""
        if not self.is_closed():
            return 0.0
        total_cost = self.get_total_cost()
        if total_cost == 0:
            return 0.0
        return (self.get_profit_loss() / total_cost) * 100


def get_daily_price(symbol: str, date: datetime) -> Optional[float]:
    """
    Get closing price for a stock on a specific date.
    Handles holidays and non-trading days by falling back to the most recent trading day.
    """
    try:
        ticker = yf.Ticker(symbol)
        start_date = date.date()
        end_date = start_date + timedelta(days=1)
        
        # Try to get price for the specific date
        hist = ticker.history(start=start_date, end=end_date, interval='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        
        # If no data for this date (holiday/weekend), get the most recent trading day
        # Look back up to 10 calendar days to find the last trading day
        for days_back in range(1, 11):
            lookup_date = start_date - timedelta(days=days_back)
            lookup_start = lookup_date
            lookup_end = lookup_date + timedelta(days=1)
            
            hist = ticker.history(start=lookup_start, end=lookup_end, interval='1d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        
        # If still no data after looking back 10 days, return None
        return None
        
    except Exception as e:
        # Silently handle errors (holidays, delistings, etc.)
        return None


def simulate_crazy_strategy(start_date: datetime, initial_cash: float = 10000.0, 
                           days_to_track: int = 90, profit_target_pct: float = 0.10, 
                           etf_only: bool = False, average_down_threshold_pct: float = 0.10, 
                           random_seed: int = 42, diminishing_tp_days: int = 45,
                           max_stocks_to_test: int = None) -> Dict:
    """
    Simulate the crazy trading strategy.
    
    Args:
        start_date: Date to start buying
        initial_cash: Starting cash amount
        days_to_track: Maximum days to track positions
        profit_target_pct: Profit target as percentage of initial price
        etf_only: If True, only trade ETFs
        average_down_threshold_pct: Threshold percentage drop to trigger averaging down, and % of cost basis to rebuy (default: 0.10 = 10%)
        random_seed: Random seed for reproducible selection (default: 42 for deterministic, None for random)
        diminishing_tp_days: Days over which TP diminishes to break-even (default: 45)
        max_stocks_to_test: Maximum number of stocks to test during filtering (None = test all, for faster testing)
    
    Returns:
        Dictionary with results
    """
    print("\n" + "="*80)
    print("CRAZY TRADING STRATEGY BACKTEST")
    print("="*80)
    print(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
    print(f"Initial Cash: ${initial_cash:,.2f}")
    print(f"Max Days to Track: {days_to_track}")
    print(f"Profit Target: {profit_target_pct*100:.1f}% of initial price per share (diminishing over {diminishing_tp_days} days)")
    print(f"Average Down Threshold: {average_down_threshold_pct*100:.1f}% drop (also used for rebuy amount)")
    if etf_only:
        print("Asset Type: ETFs only")
    print("="*80)
    
    # Find stocks to buy
    stocks = find_middle_of_road_stocks(start_date.date(), count=10, etf_only=etf_only, random_seed=random_seed,
                                       max_stocks_to_test=max_stocks_to_test)
    if len(stocks) < 10:
        print(f"⚠️  Warning: Only found {len(stocks)} stocks, continuing with {len(stocks)}")
    
    # Initialize positions
    positions: Dict[str, Position] = {}
    cash = initial_cash
    fixed_amount_per_stock = 3000.0  # Fixed $3,000 per stock
    
    # Buy initial positions
    print(f"\n📊 INITIAL PURCHASES ({start_date.strftime('%Y-%m-%d')}):")
    print("-" * 80)
    for symbol in stocks:
        price = get_daily_price(symbol, start_date)
        if price is None or price == 0:
            print(f"  ⚠️  {symbol}: No price data, skipping")
            continue
        
        # Use fixed amount per stock (or remaining cash if less)
        investment_amount = min(fixed_amount_per_stock, cash)
        shares = int(investment_amount / price)
        if shares == 0:
            print(f"  ⚠️  {symbol}: Can't afford any shares at ${price:.2f}")
            continue
        
        cost = shares * price
        
        cash -= cost
        positions[symbol] = Position(symbol, start_date, price, shares, profit_target_pct, 
                                     average_down_threshold_pct, diminishing_tp_days)
        print(f"  ✅ {symbol}: {shares} shares @ ${price:.2f} = ${cost:,.2f}")
    
    if not positions:
        print("  ❌ No positions opened!")
        return {'error': 'No positions opened'}
    
    print(f"\n  💵 Remaining Cash: ${cash:,.2f}")
    print(f"  📈 Total Positions: {len(positions)}")
    
    # Track positions daily
    current_date = start_date
    end_date = start_date + timedelta(days=days_to_track)
    
    print(f"\n📅 DAILY TRACKING (from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}):")
    print("-" * 80)
    
    day_count = 0
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        day_count += 1
        if day_count % 10 == 0:
            closed_count = sum(1 for p in positions.values() if p.is_closed())
            print(f"  Day {day_count}: {len(positions) - closed_count} positions open, {closed_count} closed")
        
        all_closed = True
        for symbol, position in positions.items():
            if position.is_closed():
                continue
            
            all_closed = False
            current_price = get_daily_price(symbol, current_date)
            if current_price is None or current_price == 0:
                continue
            
            avg_price = position.get_average_price()
            target_price = position.get_target_price(current_date)  # Diminishing target
            avg_down_price = position.get_average_down_price()
            
            # Track recent prices for trend detection (keep last 5 days)
            position.recent_prices.append(current_price)
            if len(position.recent_prices) > 5:
                position.recent_prices.pop(0)
            
            # Check for profit taking with enhanced exit conditions
            should_sell = False
            sell_reason_detail = ""
            
            # Check: target profit met (diminishing target)
            if current_price >= target_price and current_price >= avg_price:
                position.target_price_reached = True
                # Check for downward trend: compare current price to 2-day average
                if len(position.recent_prices) >= 3:
                    # Calculate average of last 2 days (excluding current_price which was just added)
                    last_2_prices = position.recent_prices[-3:-1]
                    avg_last_2_days = sum(last_2_prices) / len(last_2_prices)
                    if current_price < avg_last_2_days:
                        # Current price is below 2-day average - downward trend detected, sell
                        should_sell = True
                        days_held = (current_date.date() - position.buy_date.date()).days
                        sell_reason_detail = f"Target met (${current_price:.2f}) but downward trend detected (current < 2-day avg ${avg_last_2_days:.2f}, day {days_held})"
                    # If no downward trend, don't sell yet - wait for trend or drop below TP
                else:
                    # Not enough history yet (need at least 2 previous days) - don't sell yet, wait for more data
                    pass
            
            # Check: target was met before, but price has dropped below TP
            if not should_sell and position.target_price_reached and current_price < target_price:
                should_sell = True
                sell_reason_detail = f"Target was met previously, but price dropped below TP (${current_price:.2f} < ${target_price:.2f})"
            
            if should_sell:
                total_shares = position.get_total_shares()
                proceeds = current_price * total_shares
                total_cost = position.get_total_cost()
                actual_profit = proceeds - total_cost
                position.sell(current_date, current_price, f"{sell_reason_detail}, profit: ${actual_profit:.2f}")
                cash += proceeds
                if day_count % 10 == 0 or day_count <= 5:
                    pnl = position.get_profit_loss_pct()
                    print(f"    ✅ {symbol}: SOLD at ${current_price:.2f} (target: ${target_price:.2f}), P&L: {pnl:+.2f}% (${actual_profit:+.2f})")
                continue
            
            # Check for averaging down
            if current_price <= avg_down_price:
                # Calculate rebuy amount using average_down_threshold_pct of current cost basis
                current_cost = position.get_total_cost()
                additional_investment = current_cost * position.average_down_threshold_pct
                
                if cash >= additional_investment:
                    new_shares = int(additional_investment / current_price)
                    if new_shares > 0:
                        cost = new_shares * current_price
                        position.add_purchase(current_date, current_price, new_shares)
                        cash -= cost
                        new_avg = position.get_average_price()
                        print(f"    📉 {symbol}: Averaged down @ ${current_price:.2f}, new avg: ${new_avg:.2f}")
                else:
                    # Not enough cash to average down - always log this
                    print(f"    ⚠️  {symbol}: Price ${current_price:.2f} <= threshold ${avg_down_price:.2f}, but insufficient cash (need ${additional_investment:.2f}, have ${cash:.2f})")
        
        if all_closed:
            print(f"\n  ✅ All positions closed by day {day_count}")
            break
        
        current_date += timedelta(days=1)
    
    # Close any remaining positions at final price
    print(f"\n🔚 FINAL CLOSEOUT:")
    print("-" * 80)
    final_date = current_date - timedelta(days=1)  # Last date we checked
    for symbol, position in positions.items():
        if not position.is_closed():
            final_price = get_daily_price(symbol, final_date)
            if final_price is None:
                final_price = position.get_average_price()  # Use average if no price
            
            total_shares = position.get_total_shares()
            proceeds = final_price * total_shares
            position.sell(final_date, final_price, f"End of tracking period")
            cash += proceeds
            print(f"  📤 {symbol}: Closed at ${final_price:.2f}")
    
    # Calculate results
    total_cost = sum(p.get_total_cost() for p in positions.values())
    final_value = cash
    total_profit = final_value - initial_cash
    total_return = (total_profit / initial_cash) * 100
    
    winners = [p for p in positions.values() if p.get_profit_loss() > 0]
    losers = [p for p in positions.values() if p.get_profit_loss() < 0]
    
    # Calculate average days held
    days_held_list = []
    for position in positions.values():
        if position.sell_date:
            days_held = (position.sell_date.date() - position.buy_date.date()).days
        else:
            days_held = (final_date.date() - position.buy_date.date()).days
        days_held_list.append(days_held)
    avg_days_held = sum(days_held_list) / len(days_held_list) if days_held_list else 0
    
    print(f"\n📊 RESULTS:")
    print("=" * 80)
    print(f"Initial Cash: ${initial_cash:,.2f}")
    print(f"Final Cash: ${final_value:,.2f}")
    print(f"Total Profit/Loss: ${total_profit:+,.2f}")
    print(f"Total Return: {total_return:+.2f}%")
    print(f"\nPositions:")
    print(f"  Total: {len(positions)}")
    print(f"  Winners: {len(winners)} ({len(winners)/len(positions)*100:.1f}%)")
    print(f"  Losers: {len(losers)} ({len(losers)/len(positions)*100:.1f}%)")
    print(f"  Average Days Held: {avg_days_held:.1f}")
    
    if winners:
        avg_winner = sum(p.get_profit_loss_pct() for p in winners) / len(winners)
        print(f"  Average Winner: {avg_winner:+.2f}%")
    
    if losers:
        avg_loser = sum(p.get_profit_loss_pct() for p in losers) / len(losers)
        print(f"  Average Loser: {avg_loser:+.2f}%")
    
    # Show individual position details
    print(f"\n📋 POSITION DETAILS:")
    print("-" * 80)
    for symbol, position in sorted(positions.items()):
        pnl = position.get_profit_loss_pct()
        pnl_symbol = "✅" if pnl > 0 else "❌"
        sell_date_str = position.sell_date.strftime('%Y-%m-%d') if position.sell_date else 'N/A'
        sell_price_str = f"${position.sell_price:.2f}" if position.sell_price else 'N/A'
        
        # Calculate days held
        if position.sell_date:
            days_held = (position.sell_date.date() - position.buy_date.date()).days
        else:
            days_held = (final_date.date() - position.buy_date.date()).days
        
        print(f"  {pnl_symbol} {symbol}:")
        print(f"     Buy: {position.buy_date.strftime('%Y-%m-%d')} @ avg ${position.get_average_price():.2f}")
        print(f"     Sell: {sell_date_str} @ {sell_price_str}")
        print(f"     Days: {days_held}")
        
        # Show all individual purchases if there are multiple
        if len(position.purchases) > 1:
            print(f"     All Purchases ({len(position.purchases)}):")
            for i, purchase in enumerate(position.purchases, 1):
                print(f"       {i}. {purchase['date'].strftime('%Y-%m-%d')}: {purchase['shares']} shares @ ${purchase['price']:.2f} = ${purchase['cost']:.2f}")
        else:
            print(f"     Purchases: {len(position.purchases)}")
        
        print(f"     Total Shares: {position.get_total_shares()}")
        print(f"     P&L: {pnl:+.2f}% ({position.get_profit_loss():+,.2f})")
        print(f"     Reason: {position.sell_reason}")
        print()
    
    return {
        'initial_cash': initial_cash,
        'final_cash': final_value,
        'total_profit': total_profit,
        'total_return': total_return,
        'positions': len(positions),
        'winners': len(winners),
        'losers': len(losers),
        'positions_detail': positions,
    }


def main():
    parser = argparse.ArgumentParser(description='Backtest the crazy trading strategy')
    parser.add_argument('--date', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--cash', type=float, default=50000.0,
                       help='Initial cash amount (default: 50000)')
    parser.add_argument('--days', type=int, default=90,
                       help='Maximum days to track positions (default: 90)')
    parser.add_argument('--stocks', type=int, default=10,
                       help='Number of stocks to buy (default: 10)')
    parser.add_argument('--profit-target', type=float, default=0.10,
                       help='Profit target as percentage of initial price (default: 0.10 = 10%%, use 0.05 for 5%%)')
    parser.add_argument('--average-down-threshold', type=float, default=0.10,
                       help='Threshold percentage drop to trigger averaging down, and %% of cost basis to rebuy (default: 0.10 = 10%%, use 0.05 for 5%%)')
    parser.add_argument('--etf-only', action='store_true',
                       help='Only trade ETFs instead of stocks')
    parser.add_argument('--random', action='store_true',
                       help='Use random stock selection (default: deterministic/reproducible)')
    parser.add_argument('--diminishing-tp-days', type=int, default=45,
                       help='Days over which profit target diminishes to break-even (default: 45)')
    parser.add_argument('--max-stocks-to-test', type=int, default=None,
                       help='Maximum number of stocks to test during filtering (default: test all, use lower number for faster testing)')
    
    args = parser.parse_args()
    
    try:
        start_date = datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f"❌ Invalid date format: {args.date}. Use YYYY-MM-DD")
        sys.exit(1)
    
    # Make sure date is not in the future
    if start_date.date() > datetime.now().date():
        print(f"❌ Date cannot be in the future: {args.date}")
        sys.exit(1)
    
    # Validate profit target
    if args.profit_target <= 0 or args.profit_target > 1:
        print(f"❌ Profit target must be between 0 and 1 (e.g., 0.10 for 10%%, 0.05 for 5%%)")
        sys.exit(1)
    
    # Validate average down threshold
    if args.average_down_threshold <= 0 or args.average_down_threshold > 1:
        print(f"❌ Average down threshold must be between 0 and 1 (e.g., 0.10 for 10%%, 0.05 for 5%%)")
        sys.exit(1)
    
    # Use deterministic selection (seed=42) by default, random (seed=None) if --random flag is set
    random_seed = None if args.random else 42
    
    results = simulate_crazy_strategy(
        start_date=start_date,
        initial_cash=args.cash,
        days_to_track=args.days,
        profit_target_pct=args.profit_target,
        etf_only=args.etf_only,
        average_down_threshold_pct=args.average_down_threshold,
        random_seed=random_seed,
        diminishing_tp_days=args.diminishing_tp_days,
        max_stocks_to_test=args.max_stocks_to_test
    )
    
    if 'error' in results:
        print(f"\n❌ Error: {results['error']}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✅ BACKTEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
