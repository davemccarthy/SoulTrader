#!/usr/bin/env python
"""
Oscilla Trading - Intra-week Wave-based Trading Strategy

Oscilla Trading identifies liquid, stable-volume stocks exhibiting consistent cyclical behavior.
The system leverages wavelet analysis to detect historical peaks and troughs, defining optimal
entry points near troughs and calculating dynamic stop-loss and target prices based on recent
wave structure.

Focus: Boring, reliable stocks with stable volume patterns, not high-volume momentum stocks.
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from polygon import RESTClient
import pywt
from scipy.signal import find_peaks

# Suppress yfinance warnings for missing historical data
warnings.filterwarnings("ignore", message=".*possibly delisted.*")

# ------------------------------
# CONFIG
# ------------------------------
MIN_PRICE = 8
MAX_PRICE = 80  # Upper bound for stock price filter
MIN_VOLUME = 1_000_000  # Minimum volume filter for initial fetch
MIN_AVG_VOLUME = 2_000_000
REL_VOLUME_MIN = 0.8
REL_VOLUME_MAX = 1.3
LOOKBACK_DAYS = 20
MIN_RR = 1.8
MIN_STOP_BUFFER_PCT = 0.10  # Minimum 10% stop distance from entry (stop at 90% of entry price)
MAX_TARGET_PCT = 0.10 # Maximum target price as % gain (applies only when TARGET_DIMINISHING_ENABLED=False)
TARGET_DIMINISHING_MULTIPLIER = 0.75  # When diminishing enabled, cap target at this fraction of calculated target (0.75 = 75%)
# Diminishing Target / Augmenting Stop Configuration
TARGET_DIMINISHING_ENABLED = False   # Enable diminishing target over time (target → buy_price over max_days)
STOP_AUGMENTING_ENABLED = True     # Enable trailing stop over time (stop → buy_price over max_days)
# EXPERIMENTAL FILTERS (currently DISABLED - collecting more data for analysis):
MAX_WAVE_POSITION = -999  # Maximum (most negative) wave_position to accept (filters strong downtrends) - DISABLED
MIN_CONSISTENCY = 0.0  # Minimum consistency score (filters inconsistent wave patterns) - DISABLED
# Note: Set MAX_WAVE_POSITION = -999 to disable wave position filter
#       Set MIN_CONSISTENCY = 0.0 to disable consistency filter


def fetch_stocks_for_date(reference_date, min_price=MIN_PRICE, max_price=MAX_PRICE, min_volume=MIN_VOLUME):
    """
    Fetch stocks using Polygon's get_grouped_daily_aggs (1 API call for all stocks on a date).
    
    Args:
        reference_date: Date string (YYYY-MM-DD)
        min_price: Minimum stock price (default: $8)
        max_price: Maximum stock price (default: $80)
        min_volume: Minimum volume filter (default: 1,000,000)
    
    Returns:
        pandas DataFrame with columns: ticker, price, today_volume
    """
    polygon_api_key = os.getenv('POLYGON_API_KEY')
    if not polygon_api_key:
        print("❌ POLYGON_API_KEY environment variable not set")
        print("   Please set it with: export POLYGON_API_KEY='your_key_here'")
        return pd.DataFrame()
    
    try:
        client = RESTClient(polygon_api_key)
        
        print(f"   Fetching all stocks for {reference_date} using Polygon (1 API call)...")
        aggs = client.get_grouped_daily_aggs(
            locale="us",
            date=reference_date,
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
            print(f"   Found {len(df)} stocks in ${min_price:.2f}-${max_price:.2f} price range with volume >= {min_volume:,}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error fetching stocks from Polygon: {e}")
        return pd.DataFrame()


def get_historical_data_yfinance(ticker, start_date, end_date):
    """
    Get historical data using yfinance (rate-limit friendly).
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
    
    Returns:
        pandas DataFrame with columns: date, close, high, low, volume
    """
    try:
        # yfinance end date is exclusive, so add 1 day
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        end_date_exclusive = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        
        ticker_obj = yf.Ticker(ticker)
        # Suppress yfinance warnings for missing historical data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*possibly delisted.*")
            hist = ticker_obj.history(start=start_date, end=end_date_exclusive, raise_errors=False)
        
        if hist.empty:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            "date": hist.index,
            "close": hist['Close'].values,
            "high": hist['High'].values,
            "low": hist['Low'].values,
            "volume": hist['Volume'].values
        })
        
        return df.reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame()






def build_candidates(reference_date=None, min_price=MIN_PRICE, max_price=MAX_PRICE,
                    min_avg_volume=MIN_AVG_VOLUME, rel_volume_min=REL_VOLUME_MIN, 
                    rel_volume_max=REL_VOLUME_MAX, lookback_days=LOOKBACK_DAYS, 
                    max_stocks=None, verbose=False):
    """
    Build candidate stocks that meet volume and price criteria.
    Uses Polygon for initial stock discovery (1 API call), yfinance for historical data.
    
    Args:
        reference_date: Reference date string (YYYY-MM-DD), defaults to today
        min_price: Minimum stock price
        min_avg_volume: Minimum average volume
        rel_volume_min: Minimum relative volume (vs average)
        rel_volume_max: Maximum relative volume (vs average)
        lookback_days: Number of days to look back for analysis
        verbose: If True, print progress messages
    
    Returns:
        pandas DataFrame with candidate stocks
    """
    if reference_date is None:
        reference_date = datetime.today().strftime("%Y-%m-%d")
    
    # Validate date format
    try:
        ref_dt = datetime.strptime(reference_date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ Invalid date format: {reference_date}. Use YYYY-MM-DD")
        return pd.DataFrame()
    
    start_date = (ref_dt - timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")
    
    if verbose:
        print(f"📊 Building candidates for {reference_date}...")
        print(f"   Fetching stocks using Polygon...")
    
    # Get stocks using Polygon's get_grouped_daily_aggs (1 API call)
    df_daily = fetch_stocks_for_date(reference_date, min_price, max_price, MIN_VOLUME)
    
    if df_daily.empty:
        if verbose:
            print(f"⚠️  No data found for {reference_date} (likely weekend/holiday)")
        return pd.DataFrame()
    
    if verbose:
        print(f"   Found {len(df_daily)} stocks in ${min_price:.2f}-${max_price:.2f} price range")
    
    # Limit to max_stocks if specified (for testing/rate limit management)
    if max_stocks and len(df_daily) > max_stocks:
        df_daily = df_daily.head(max_stocks)
        if verbose:
            print(f"   Limiting to first {max_stocks} stocks for analysis...")
    
    if verbose:
        print(f"   Calculating average volume over {lookback_days} days for each stock...")
    
    candidates = []
    
    for i, (_, row) in enumerate(df_daily.iterrows(), 1):
        ticker = row["ticker"]
        today_volume = row["today_volume"]
        last_close = row["price"]
        
        if verbose and i % 50 == 0:
            print(f"   Processed {i}/{len(df_daily)} stocks...")
        
        # Get historical data to calculate average volume
        # yfinance is rate-limit friendly, no delay needed
        try:
            df_hist = get_historical_data_yfinance(ticker, start_date, reference_date)
            if df_hist.empty or len(df_hist) < lookback_days:
                continue
            
            df_hist = df_hist.sort_values("date").tail(lookback_days)
            avg_volume = df_hist["volume"].mean()
            
            if avg_volume < min_avg_volume:
                continue
            
            rel_volume = today_volume / avg_volume if avg_volume > 0 else 0
            
            if not (rel_volume_min <= rel_volume <= rel_volume_max):
                continue
            
            candidates.append({
                "ticker": ticker,
                "price": last_close,
                "avg_volume": int(avg_volume),
                "today_volume": int(today_volume),
                "rel_volume": round(rel_volume, 2)
            })
            
        except Exception:
            continue
    
    df_candidates = pd.DataFrame(candidates)
    if df_candidates.empty:
        if verbose:
            print("   No candidates found matching criteria")
        return df_candidates
    
    df_candidates["vol_score"] = 1 - abs(df_candidates["rel_volume"] - 1)
    
    if verbose:
        print(f"✓ Found {len(df_candidates)} candidate stocks with stable volume patterns")
    
    return df_candidates.sort_values("vol_score", ascending=False).reset_index(drop=True)


def wavelet_trade_engine(price_series, min_rr=MIN_RR):
    """
    Analyze price series using wavelet analysis to detect cyclical patterns and generate trade signals.
    
    Args:
        price_series: pandas Series of closing prices
        min_rr: Minimum reward:risk ratio required
    
    Returns:
        dict with trade analysis results, including 'accepted' boolean flag
    """
    log = []
    
    if price_series.empty or len(price_series) < 64:
        return {"accepted": False, "reason": "Insufficient data points", "log": ["Need at least 64 data points"]}
    
    # Detrend
    detrended = price_series - price_series.rolling(20, min_periods=1).mean()
    
    # Wavelet transform
    scales = np.arange(2, 64)
    coeffs, _ = pywt.cwt(detrended.values, scales, 'morl')
    power = np.abs(coeffs) ** 2
    
    # Find dominant scale
    avg_power_per_scale = power.mean(axis=1)
    dominant_scale_idx = np.argmax(avg_power_per_scale)
    dominant_scale = scales[dominant_scale_idx]  # Actual scale value (2-63)
    dominant_period = int(dominant_scale)
    
    # Use the index to get dominant power (fix for index bug)
    dominant_power = power[dominant_scale_idx]
    consistency = float(np.sum(dominant_power > 0.5 * dominant_power.max()) / len(dominant_power))
    
    log.append(f"Dominant period: {dominant_period} days, consistency: {consistency:.3f}")
    
    # Filter: Reject inconsistent wave patterns (can be disabled by setting MIN_CONSISTENCY = 0.0)
    if MIN_CONSISTENCY > 0 and consistency < MIN_CONSISTENCY:
        log.append(f"Rejected: consistency={consistency:.3f} < MIN_CONSISTENCY={MIN_CONSISTENCY} (inconsistent pattern)")
        return {"accepted": False, "reason": "Low consistency", "consistency": consistency, "log": log}
    
    # Smooth & structure
    half_period = max(3, dominant_period // 2)
    smoothed = price_series.rolling(half_period, center=True, min_periods=1).mean()
    peaks, _ = find_peaks(smoothed)
    troughs, _ = find_peaks(-smoothed)
    
    if len(peaks) < 2 or len(troughs) < 2:
        log.append(f"Insufficient peaks ({len(peaks)}) or troughs ({len(troughs)})")
        return {"accepted": False, "reason": "Insufficient peaks/troughs", "log": log}
    
    avg_peak = float(smoothed.iloc[peaks].mean())
    avg_trough = float(smoothed.iloc[troughs].mean())
    wave_range = avg_peak - avg_trough
    
    if wave_range <= 0:
        log.append("Invalid wave range (peak <= trough)")
        return {"accepted": False, "reason": "Invalid wave range", "log": log}
    
    last_trough = float(smoothed.iloc[troughs[-1]])
    current_price = float(price_series.iloc[-1])
    
    # Wave phase
    wave_position = (current_price - avg_trough) / wave_range
    log.append(f"Wave position: {wave_position:.3f} (0=trough, 1=peak)")
    
    if wave_position > 0.35:
        log.append(f"Rejected: wave_position={wave_position:.2f} (too high, want near trough)")
        return {"accepted": False, "reason": "Bad wave phase", "wave_position": wave_position, "log": log}
    
    # Filter: Reject extreme negative wave positions (strong downtrends)
    # Can be disabled by setting MAX_WAVE_POSITION to a very negative value (e.g., -999)
    if MAX_WAVE_POSITION > -999 and wave_position < MAX_WAVE_POSITION:
        log.append(f"Rejected: wave_position={wave_position:.3f} < MAX_WAVE_POSITION={MAX_WAVE_POSITION} (too far below avg trough = strong downtrend)")
        return {"accepted": False, "reason": "Extreme wave position (downtrend)", "wave_position": wave_position, "log": log}
    
    # Trade levels
    buy = current_price
    calculated_stop = last_trough - 0.15 * wave_range
    
    # Reject if calculated stop is invalid (at or above buy price - indicates invalid wave pattern)
    if calculated_stop >= buy:
        log.append(f"Rejected: Invalid stop calculation (stop ${calculated_stop:.2f} >= buy ${buy:.2f})")
        return {"accepted": False, "reason": "Invalid stop calculation", "log": log}
    
    # Use calculated stop for candidate filtering (min stop buffer applied later in backtesting/trading)
    stop = calculated_stop
    
    target = buy + 0.85 * (avg_peak - buy)
    risk = buy - stop
    reward = target - buy
    rr = reward / risk if risk > 0 else float('nan')
    
    log.append(f"Buy: ${buy:.2f}, Stop: ${stop:.2f}, Target: ${target:.2f}")
    log.append(f"Risk: ${risk:.2f}, Reward: ${reward:.2f}, R:R={rr:.2f}")
    
    if np.isnan(rr) or rr < min_rr:
        log.append(f"Rejected: R:R={rr:.2f} < {min_rr}")
        return {"accepted": False, "reason": "Insufficient R:R", "rr": rr, "log": log}
    
    log.append("Accepted: good wave phase and R:R")
    
    return {
        "accepted": True,
        "dominant_period_days": dominant_period,
        "consistency": round(consistency, 3),
        "wave_position": round(wave_position, 3),
        "buy": round(buy, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "reward_risk": round(rr, 2),
        "log": log
    }


def generate_trading_candidates(reference_date=None, min_price=MIN_PRICE, max_price=MAX_PRICE,
                                min_avg_volume=MIN_AVG_VOLUME, min_rr=MIN_RR,
                                lookback_days=LOOKBACK_DAYS, limit=None, max_stocks=None, verbose=False):
    """
    Generate trading candidates by analyzing candidates with wavelet trade engine.
    
    Args:
        reference_date: Analysis date string (YYYY-MM-DD), defaults to today
        min_price: Minimum stock price filter
        max_price: Maximum stock price filter
        min_avg_volume: Minimum average volume filter
        min_rr: Minimum reward:risk ratio
        lookback_days: Days of history to analyze
        limit: Maximum number of candidates to process (None = all)
        verbose: If True, show detailed output
    
    Returns:
        pandas DataFrame with trading candidates that passed all filters
    """
    if reference_date is None:
        reference_date = datetime.today().strftime("%Y-%m-%d")
    
    candidates_df = build_candidates(
        reference_date, min_price, max_price, min_avg_volume,
        REL_VOLUME_MIN, REL_VOLUME_MAX, lookback_days, max_stocks, verbose
    )
    
    if candidates_df.empty:
        print("No candidates found")
        return pd.DataFrame()
    
    if limit:
        candidates_df = candidates_df.head(limit)
        if verbose:
            print(f"   Limiting analysis to top {limit} candidates")
    
    if verbose:
        print(f"\n🔍 Analyzing {len(candidates_df)} candidates with wavelet trade engine...")
    
    results = []
    
    ref_dt = datetime.strptime(reference_date, "%Y-%m-%d")
    # For wavelet analysis, we need at least 64 trading days
    # Account for weekends/holidays: ~5 trading days per 7 calendar days
    # So 64 trading days ≈ 64 * 7/5 ≈ 90 calendar days, add buffer for safety
    min_calendar_days = int(64 * 7 / 5) + 30  # ~120 calendar days for 64 trading days
    wavelet_lookback_days = max(min_calendar_days, lookback_days * 2)
    start_date = (ref_dt - timedelta(days=wavelet_lookback_days)).strftime("%Y-%m-%d")
    
    for i, (_, row) in enumerate(candidates_df.iterrows(), 1):
        ticker = row["ticker"]
        
        if verbose and i % 10 == 0:
            print(f"   Analyzed {i}/{len(candidates_df)} candidates...")
        
        # yfinance is rate-limit friendly, no delay needed
        try:
            df_price = get_historical_data_yfinance(ticker, start_date, reference_date)
            if df_price.empty:
                if verbose:
                    print(f"   {ticker}: No historical data available")
                continue
            
            # Check if we have enough data points
            if len(df_price) < 64:
                if verbose:
                    print(f"   {ticker}: Only {len(df_price)} data points available (need 64)")
                continue
            
            wave_result = wavelet_trade_engine(df_price["close"], min_rr)
            
            if verbose and not wave_result.get("accepted", False):
                reason = wave_result.get("reason", "Unknown")
                print(f"   {ticker}: Rejected - {reason}")
                if wave_result.get("log"):
                    for log_msg in wave_result["log"][-2:]:  # Show last 2 log messages
                        print(f"      {log_msg}")
            
            if wave_result.get("accepted", False):
                result = {
                    "ticker": ticker,
                    "price": row["price"],
                    "avg_volume": row["avg_volume"],
                    "today_volume": row["today_volume"],
                    "rel_volume": row["rel_volume"],
                    **wave_result
                }
                results.append(result)
                if verbose:
                    print(f"   {ticker}: ✓ Accepted (R:R={wave_result.get('reward_risk', 0):.2f}, wave_pos={wave_result.get('wave_position', 0):.3f})")
        except Exception as e:
            if verbose:
                print(f"   Error analyzing {ticker}: {e}")
            continue
    
    if not results:
        print("No trading candidates passed wavelet analysis")
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    
    # Sort by reward_risk and consistency
    df_results = df_results.sort_values(
        ["reward_risk", "consistency"], 
        ascending=[False, False]
    ).reset_index(drop=True)
    
    return df_results


def format_results(df_results):
    """
    Format results DataFrame for display.
    
    Args:
        df_results: DataFrame with trading candidates
    
    Returns:
        Formatted string for display
    """
    if df_results.empty:
        return "No results to display"
    
    # Select columns to display
    display_cols = [
        "ticker", "price", "rel_volume", "dominant_period_days", 
        "consistency", "wave_position", "buy", "stop", "target", "reward_risk"
    ]
    
    # Ensure all columns exist
    available_cols = [col for col in display_cols if col in df_results.columns]
    df_display = df_results[available_cols].copy()
    
    # Format columns for display
    if "price" in df_display.columns:
        df_display["price"] = df_display["price"].apply(lambda x: f"${x:.2f}")
    if "buy" in df_display.columns:
        df_display["buy"] = df_display["buy"].apply(lambda x: f"${x:.2f}")
    if "stop" in df_display.columns:
        df_display["stop"] = df_display["stop"].apply(lambda x: f"${x:.2f}")
    if "target" in df_display.columns:
        df_display["target"] = df_display["target"].apply(lambda x: f"${x:.2f}")
    if "wave_position" in df_display.columns:
        df_display["wave_position"] = df_display["wave_position"].apply(lambda x: f"{x:.3f}")
    if "consistency" in df_display.columns:
        df_display["consistency"] = df_display["consistency"].apply(lambda x: f"{x:.3f}")
    if "reward_risk" in df_display.columns:
        df_display["reward_risk"] = df_display["reward_risk"].apply(lambda x: f"{x:.2f}")
    
    return df_display.to_string(index=False)


def backtest_signal(ticker, entry_date, buy_price, stop_price, target_price, max_days=40,
                    target_diminishing_enabled=False, stop_augmenting_enabled=False):
    """
    Backtest a single trading signal by checking if stop or target is hit first.
    
    Args:
        ticker: Stock ticker symbol
        entry_date: Entry date string (YYYY-MM-DD)
        buy_price: Entry price
        stop_price: Stop loss price (original)
        target_price: Target price (original)
        max_days: Maximum days to hold (also used for diminishing/augmenting period)
        target_diminishing_enabled: If True, target diminishes from original_target → buy_price over max_days
        stop_augmenting_enabled: If True, stop augments from original_stop → buy_price over max_days
    
    Returns:
        dict with results: {'outcome': 'win'|'loss'|'timeout', 'days_held': int, 
                           'exit_price': float, 'return_pct': float}
    """
    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    end_dt = entry_dt + timedelta(days=max_days + 5)  # Add buffer for weekends/holidays
    end_date = end_dt.strftime("%Y-%m-%d")
    
    try:
        # Get historical data for the period after entry
        df = get_historical_data_yfinance(ticker, entry_date, end_date)
        
        if df.empty or len(df) < 2:
            return {'outcome': 'no_data', 'days_held': 0, 'exit_price': buy_price, 'return_pct': 0.0}
        
        # Check each day after entry to see if stop or target is hit
        entry_date_only = entry_dt.date()
        
        for i, row in df.iterrows():
            # Get date from the row (could be datetime or date)
            row_date = row['date']
            if hasattr(row_date, 'date'):
                row_date_only = row_date.date()
            elif hasattr(row_date, 'to_pydatetime'):
                row_date_only = row_date.to_pydatetime().date()
            else:
                row_date_only = row_date
            
            # Skip entry day (we enter at close of entry day, check starts next day)
            if row_date_only <= entry_date_only:
                continue
            
            low = row['low']
            high = row['high']
            close = row['close']
            
            days_held = (row_date_only - entry_date_only).days
            
            # Calculate adjusted target and stop based on diminishing/augmenting logic
            if target_diminishing_enabled and days_held <= max_days:
                # Diminish target linearly from original_target → buy_price over max_days
                progress = days_held / max_days if max_days > 0 else 1.0
                current_target = target_price - progress * (target_price - buy_price)
            else:
                current_target = target_price
            
            if stop_augmenting_enabled and days_held <= max_days:
                # Augment stop linearly from original_stop → buy_price over max_days
                progress = days_held / max_days if max_days > 0 else 1.0
                current_stop = stop_price + progress * (buy_price - stop_price)
            else:
                current_stop = stop_price
            
            # Check if stop was hit (price went below stop) - check first as more conservative
            if low <= current_stop:
                exit_price = current_stop
                return_pct = ((exit_price - buy_price) / buy_price) * 100
                return {
                    'outcome': 'loss',
                    'days_held': days_held,
                    'exit_price': exit_price,
                    'return_pct': return_pct
                }
            
            # Check if target was hit (price went above target)
            if high >= current_target:
                exit_price = current_target
                return_pct = ((exit_price - buy_price) / buy_price) * 100
                return {
                    'outcome': 'win',
                    'days_held': days_held,
                    'exit_price': exit_price,
                    'return_pct': return_pct
                }
            
            # If we've exceeded max_days, exit at close
            if days_held >= max_days:
                exit_price = close
                return_pct = ((exit_price - buy_price) / buy_price) * 100
                return {
                    'outcome': 'timeout',
                    'days_held': days_held,
                    'exit_price': exit_price,
                    'return_pct': return_pct
                }
        
        # If we've gone through all data without hitting stop/target, exit at last close
        if len(df) > 1:
            last_close = df.iloc[-1]['close']
            last_date = df.iloc[-1]['date']
            if hasattr(last_date, 'date'):
                last_date_only = last_date.date()
            elif hasattr(last_date, 'to_pydatetime'):
                last_date_only = last_date.to_pydatetime().date()
            else:
                last_date_only = last_date
            days_held = (last_date_only - entry_date_only).days
        else:
            last_close = buy_price
            days_held = 0
        
        return_pct = ((last_close - buy_price) / buy_price) * 100
        
        return {
            'outcome': 'timeout',
            'days_held': days_held,
            'exit_price': last_close,
            'return_pct': return_pct
        }
        
    except Exception as e:
        return {'outcome': 'error', 'days_held': 0, 'exit_price': buy_price, 'return_pct': 0.0, 'error': str(e)}


def run_backtest(start_date, end_date=None, num_dates=10, max_stocks=None, verbose=False):
    """
    Run backtesting on historical dates.
    
    Args:
        start_date: Starting date string (YYYY-MM-DD)
        end_date: Ending date string (YYYY-MM-DD), defaults to start_date + num_dates weeks
        num_dates: Number of trading dates to test (default: 10)
        max_stocks: Maximum stocks to analyze per date (for speed)
        verbose: If True, show detailed output
    
    Returns:
        pandas DataFrame with backtest results
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    if end_date is None:
        # Test every 5 trading days (approximately weekly)
        end_dt = start_dt + timedelta(days=num_dates * 7)
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate test dates (approximately weekly)
    test_dates = []
    current_dt = start_dt
    while current_dt <= end_dt and len(test_dates) < num_dates:
        # Skip weekends (Saturday=5, Sunday=6)
        if current_dt.weekday() < 5:
            test_dates.append(current_dt.strftime("%Y-%m-%d"))
        current_dt += timedelta(days=7)
    
    if verbose:
        print(f"📊 Backtesting on {len(test_dates)} dates from {start_date} to {end_dt.strftime('%Y-%m-%d')}")
        print(f"   Test dates: {test_dates}\n")
    
    all_results = []
    
    for i, test_date in enumerate(test_dates, 1):
        if verbose:
            print(f"\n{'='*80}")
            print(f"Test Date {i}/{len(test_dates)}: {test_date}")
            print(f"{'='*80}")
        
        # Get trading candidates for this date
        candidates_df = generate_trading_candidates(
            reference_date=test_date,
            max_stocks=max_stocks,
            verbose=verbose
        )
        
        if candidates_df.empty:
            if verbose:
                print(f"   No candidates found for {test_date}")
            continue
        
        # Backtest each candidate
        for _, row in candidates_df.iterrows():
            ticker = row['ticker']
            buy_price = row['buy']
            calculated_stop_price = row['stop']
            target_price = row['target']
            
            # Apply minimum stop buffer for backtesting only (doesn't affect candidate filtering)
            min_stop_price = buy_price * (1 - MIN_STOP_BUFFER_PCT)
            stop_price = min(calculated_stop_price, min_stop_price)  # Use the wider stop
            
            # Apply target cap based on TARGET_DIMINISHING_ENABLED setting
            calculated_target_price = target_price
            if not TARGET_DIMINISHING_ENABLED:
                # Fixed 10% cap when diminishing disabled
                max_target_price = buy_price * (1 + MAX_TARGET_PCT)
                target_price = min(target_price, max_target_price)  # Cap at MAX_TARGET_PCT
            else:
                # Dynamic cap at TARGET_DIMINISHING_MULTIPLIER of calculated target when diminishing enabled
                target_gain = target_price - buy_price
                capped_target_gain = target_gain * TARGET_DIMINISHING_MULTIPLIER
                target_price = buy_price + capped_target_gain
            # Note: When TARGET_DIMINISHING_ENABLED=True, diminishing target will then reduce this capped target over time
            
            if verbose:
                dim_note = " (diminishing)" if TARGET_DIMINISHING_ENABLED else ""
                aug_note = " (augmenting)" if STOP_AUGMENTING_ENABLED else ""
                print(f"\n   Testing {ticker}: Buy=${buy_price:.2f}, Stop=${stop_price:.2f}{aug_note} (calc: ${calculated_stop_price:.2f}), Target=${target_price:.2f}{dim_note} (calc: ${calculated_target_price:.2f})")
            
            result = backtest_signal(
                ticker, test_date, buy_price, stop_price, target_price,
                max_days=40,
                target_diminishing_enabled=TARGET_DIMINISHING_ENABLED,
                stop_augmenting_enabled=STOP_AUGMENTING_ENABLED
            )
            
            result.update({
                'test_date': test_date,
                'ticker': ticker,
                'buy_price': buy_price,
                'stop_price': stop_price,
                'target_price': target_price,
                'reward_risk': row.get('reward_risk', 0),
                'consistency': row.get('consistency', 0),
                'wave_position': row.get('wave_position', 0)
            })
            
            all_results.append(result)
            
            if verbose:
                outcome_symbol = "✓" if result['outcome'] == 'win' else "✗" if result['outcome'] == 'loss' else "⏱"
                print(f"     {outcome_symbol} {result['outcome'].upper()}: Exit=${result['exit_price']:.2f} "
                      f"({result['return_pct']:+.2f}%) after {result['days_held']} days")
    
    if not all_results:
        print("No backtest results generated")
        return pd.DataFrame()
    
    df_backtest = pd.DataFrame(all_results)
    
    # Calculate dollar returns (assuming $1000 position per trade for comparison)
    POSITION_SIZE = 1000.0
    df_backtest['return_dollars'] = df_backtest.apply(
        lambda row: (row['return_pct'] / 100) * POSITION_SIZE, axis=1
    )
    
    # Calculate summary statistics (always show summary)
    print(f"\n{'='*80}")
    print("BACKTEST SUMMARY")
    print(f"{'='*80}")
    
    total_trades = len(df_backtest)
    wins = len(df_backtest[df_backtest['outcome'] == 'win'])
    losses = len(df_backtest[df_backtest['outcome'] == 'loss'])
    timeouts = len(df_backtest[df_backtest['outcome'] == 'timeout'])
    
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    avg_return_pct = df_backtest['return_pct'].mean()
    avg_return_wins_pct = df_backtest[df_backtest['outcome'] == 'win']['return_pct'].mean() if wins > 0 else 0
    avg_return_losses_pct = df_backtest[df_backtest['outcome'] == 'loss']['return_pct'].mean() if losses > 0 else 0
    
    avg_return_dollars = df_backtest['return_dollars'].mean()
    avg_return_wins_dollars = df_backtest[df_backtest['outcome'] == 'win']['return_dollars'].mean() if wins > 0 else 0
    avg_return_losses_dollars = df_backtest[df_backtest['outcome'] == 'loss']['return_dollars'].mean() if losses > 0 else 0
    total_return_dollars = df_backtest['return_dollars'].sum()
    
    avg_days_held = df_backtest['days_held'].mean()
    
    print(f"\nTotal Trades: {total_trades} (${POSITION_SIZE:,.0f} position per trade)")
    print(f"Wins: {wins} ({win_rate:.1f}%)")
    print(f"Losses: {losses} ({losses/total_trades*100:.1f}%)" if total_trades > 0 else "Losses: 0")
    print(f"Timeouts: {timeouts} ({timeouts/total_trades*100:.1f}%)" if total_trades > 0 else "Timeouts: 0")
    print(f"\nAverage Return: {avg_return_pct:.2f}% (${avg_return_dollars:+.2f})")
    if wins > 0:
        print(f"Average Return (Wins): {avg_return_wins_pct:.2f}% (${avg_return_wins_dollars:+.2f})")
    if losses > 0:
        print(f"Average Return (Losses): {avg_return_losses_pct:.2f}% (${avg_return_losses_dollars:+.2f})")
    print(f"Average Days Held: {avg_days_held:.1f} days")
    print(f"\nTotal Return: ${total_return_dollars:+.2f} (on ${total_trades * POSITION_SIZE:,.0f} total capital)")
    
    # Expected value calculation
    if wins > 0 and losses > 0:
        expected_value_pct = (win_rate/100 * avg_return_wins_pct) + ((100-win_rate)/100 * avg_return_losses_pct)
        expected_value_dollars = (win_rate/100 * avg_return_wins_dollars) + ((100-win_rate)/100 * avg_return_losses_dollars)
        print(f"\nExpected Value per Trade: {expected_value_pct:.2f}% (${expected_value_dollars:+.2f})")
    
    return df_backtest


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Oscilla Trading - Intra-week wave-based trading strategy'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Analysis date (YYYY-MM-DD), default: today'
    )
    parser.add_argument(
        '--min-price',
        type=float,
        default=MIN_PRICE,
        help=f'Minimum stock price filter (default: {MIN_PRICE})'
    )
    parser.add_argument(
        '--max-price',
        type=float,
        default=MAX_PRICE,
        help=f'Maximum stock price filter (default: {MAX_PRICE})'
    )
    parser.add_argument(
        '--min-avg-volume',
        type=int,
        default=MIN_AVG_VOLUME,
        help=f'Minimum average volume (default: {MIN_AVG_VOLUME:,})'
    )
    parser.add_argument(
        '--min-rr',
        type=float,
        default=MIN_RR,
        help=f'Minimum reward:risk ratio (default: {MIN_RR})'
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=LOOKBACK_DAYS,
        help=f'Days of history to analyze (default: {LOOKBACK_DAYS})'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of candidates to process (default: None = all)'
    )
    parser.add_argument(
        '--max-stocks',
        type=int,
        default=None,
        help='Maximum number of stocks to analyze from price-filtered list (default: None = all, use for testing)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run backtesting mode instead of regular analysis'
    )
    parser.add_argument(
        '--backtest-start-date',
        type=str,
        help='Start date for backtesting (YYYY-MM-DD). Required if --backtest is used.'
    )
    parser.add_argument(
        '--backtest-end-date',
        type=str,
        default=None,
        help='End date for backtesting (YYYY-MM-DD). If not specified, uses --backtest-num-dates from start date.'
    )
    parser.add_argument(
        '--backtest-num-dates',
        type=int,
        default=10,
        help='Number of trading dates to test in backtesting (default: 10, approximately weekly)'
    )
    
    args = parser.parse_args()
    
    # Handle backtesting mode
    if args.backtest:
        if not args.backtest_start_date:
            print("❌ Error: --backtest-start-date is required when using --backtest")
            sys.exit(1)
        
        df_backtest = run_backtest(
            start_date=args.backtest_start_date,
            end_date=args.backtest_end_date,
            num_dates=args.backtest_num_dates,
            max_stocks=args.max_stocks,
            verbose=args.verbose
        )
        
        if not df_backtest.empty:
            print("\n" + "=" * 80)
            print("BACKTEST RESULTS DETAIL")
            print("=" * 80)
            print(df_backtest.to_string(index=False))
        
        return
    
    # Set reference date - use today if not specified
    today = datetime.today()
    if args.date:
        reference_date = args.date
        try:
            ref_dt = datetime.strptime(reference_date, "%Y-%m-%d")
            # Warn if date is in the future or too far in the past
            if ref_dt > today:
                print(f"⚠️  Warning: Reference date {reference_date} is in the future")
            elif (today - ref_dt).days > 365:
                print(f"⚠️  Warning: Reference date {reference_date} is more than 1 year ago - many stocks may not have data")
        except ValueError:
            print(f"❌ Invalid date format: {reference_date}. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        reference_date = today.strftime("%Y-%m-%d")
    
    print("=" * 80)
    print("OSCILLA TRADING - Intra-week Wave-based Trading Strategy")
    print("=" * 80)
    print(f"Analysis date: {reference_date}")
    print(f"Current date: {today.strftime('%Y-%m-%d')}")
    print(f"Settings:")
    print(f"  Price range: ${args.min_price:.2f} - ${args.max_price:.2f}")
    print(f"  Min avg volume: {args.min_avg_volume:,}")
    print(f"  Min R:R ratio: {args.min_rr:.2f}")
    print(f"  Lookback days: {args.lookback_days}")
    if args.limit:
        print(f"  Candidate limit: {args.limit}")
    if args.max_stocks:
        print(f"  Max stocks to analyze: {args.max_stocks}")
    print()
    
    # Generate trading candidates
    df_results = generate_trading_candidates(
        reference_date=reference_date,
        min_price=args.min_price,
        max_price=args.max_price,
        min_avg_volume=args.min_avg_volume,
        min_rr=args.min_rr,
        lookback_days=args.lookback_days,
        limit=args.limit,
        max_stocks=args.max_stocks,
        verbose=args.verbose
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("TRADING CANDIDATES")
    print("=" * 80)
    
    if df_results.empty:
        print("No trading candidates found")
    else:
        print(f"\nFound {len(df_results)} trading candidates:\n")
        print(format_results(df_results))
        
        if args.verbose:
            print("\n" + "-" * 80)
            print("Summary Statistics:")
            print(f"  Average R:R ratio: {df_results['reward_risk'].mean():.2f}")
            print(f"  Average consistency: {df_results['consistency'].mean():.3f}")
            print(f"  Average wave position: {df_results['wave_position'].mean():.3f}")


def analyze_losing_trades():
    """
    Analyze losing trades to identify common patterns for filtering.
    This function examines trades that hit the -10% stop-loss.
    """
    losing_trades = [
        {"ticker": "ADMA", "reward_risk": 111.18, "consistency": 0.341, "wave_position": -1.548},
        {"ticker": "FRSH", "reward_risk": 29.26, "consistency": 0.439, "wave_position": -2.756},
        {"ticker": "XRAY", "reward_risk": 13.65, "consistency": 0.341, "wave_position": -9.208},
        {"ticker": "BOIL", "reward_risk": 6.40, "consistency": 0.232, "wave_position": -4.763},
        {"ticker": "LKQ", "reward_risk": 6.15, "consistency": 0.293, "wave_position": -1.458},
        {"ticker": "BRBR", "reward_risk": 4.59, "consistency": 0.232, "wave_position": -4.645},
        {"ticker": "TSDD", "reward_risk": 4.04, "consistency": 0.366, "wave_position": -3.850},
        {"ticker": "CZR", "reward_risk": 2.64, "consistency": 0.354, "wave_position": -0.406},
    ]
    
    df = pd.DataFrame(losing_trades)
    
    print("=" * 80)
    print("ANALYSIS OF LOSING TRADES (All hit -10% stop-loss)")
    print("=" * 80)
    print(f"\nTotal losing trades analyzed: {len(df)}")
    print("\n" + "-" * 80)
    print("STATISTICS:")
    print("-" * 80)
    print(f"Reward:Risk Ratio:")
    print(f"  Mean: {df['reward_risk'].mean():.2f}")
    print(f"  Median: {df['reward_risk'].median():.2f}")
    print(f"  Min: {df['reward_risk'].min():.2f}")
    print(f"  Max: {df['reward_risk'].max():.2f}")
    print(f"\nConsistency:")
    print(f"  Mean: {df['consistency'].mean():.3f}")
    print(f"  Median: {df['consistency'].median():.3f}")
    print(f"  Min: {df['consistency'].min():.3f}")
    print(f"  Max: {df['consistency'].max():.3f}")
    print(f"\nWave Position (negative = below avg trough):")
    print(f"  Mean: {df['wave_position'].mean():.3f}")
    print(f"  Median: {df['wave_position'].median():.3f}")
    print(f"  Min: {df['wave_position'].min():.3f}")
    print(f"  Max: {df['wave_position'].max():.3f}")
    
    print("\n" + "-" * 80)
    print("PATTERN ANALYSIS:")
    print("-" * 80)
    
    # Check for extreme wave positions (strong downtrends)
    extreme_wave = df[df['wave_position'] < -3.0]
    print(f"\n1. Extreme wave_position (< -3.0): {len(extreme_wave)}/{len(df)} trades ({100*len(extreme_wave)/len(df):.1f}%)")
    if len(extreme_wave) > 0:
        print(f"   Tickers: {', '.join(extreme_wave['ticker'].tolist())}")
        print(f"   Suggestion: Filter out wave_position < -3.0 (too far below average trough)")
    
    # Check for low consistency
    low_consistency = df[df['consistency'] < 0.35]
    print(f"\n2. Low consistency (< 0.35): {len(low_consistency)}/{len(df)} trades ({100*len(low_consistency)/len(df):.1f}%)")
    if len(low_consistency) > 0:
        print(f"   Tickers: {', '.join(low_consistency['ticker'].tolist())}")
        print(f"   Suggestion: Consider filtering consistency < 0.35 (inconsistent patterns)")
    
    # Check for low reward:risk (these should already be filtered, but check)
    low_rr = df[df['reward_risk'] < 3.0]
    print(f"\n3. Low R:R (< 3.0): {len(low_rr)}/{len(df)} trades ({100*len(low_rr)/len(df):.1f}%)")
    if len(low_rr) > 0:
        print(f"   Tickers: {', '.join(low_rr['ticker'].tolist())}")
        print(f"   Note: These passed MIN_RR={MIN_RR} filter but have low R:R")
    
    # Check combinations
    extreme_and_low_consistency = df[(df['wave_position'] < -3.0) & (df['consistency'] < 0.35)]
    print(f"\n4. Extreme wave_position AND low consistency: {len(extreme_and_low_consistency)}/{len(df)} trades ({100*len(extreme_and_low_consistency)/len(df):.1f}%)")
    if len(extreme_and_low_consistency) > 0:
        print(f"   Tickers: {', '.join(extreme_and_low_consistency['ticker'].tolist())}")
    
    print("\n" + "-" * 80)
    print("RECOMMENDED FILTERS:")
    print("-" * 80)
    print("Based on this analysis, consider adding these filters to wavelet_trade_engine():")
    print("  1. Filter wave_position < -3.0 (too far below average trough = strong downtrend)")
    print("  2. Filter consistency < 0.35 (inconsistent wave patterns)")
    print("  3. Consider requiring consistency >= 0.30 as a minimum")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze-losses":
        analyze_losing_trades()
    else:
        main()

