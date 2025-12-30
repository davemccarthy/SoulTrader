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
REL_VOLUME_MIN = 0.7  # Relaxed from 0.8 to allow stocks with slightly below-average volume (e.g., NVDL recovery cases)
REL_VOLUME_MAX = 1.3
LOOKBACK_DAYS = 40
MIN_RR = 1.8
# Turn Confirmation Configuration
TURN_CONFIRMATION_ENABLED = True  # Require turn confirmation (higher close + higher low) before entry
# Minimum stop buffer (set to 0.0 to disable, or e.g. 0.10 for 10% minimum stop)
MIN_STOP_BUFFER_PCT = 0.10  # Minimum stop distance from entry (0 = disabled, use calculated stop directly)
MAX_TARGET_PCT = 0.50  # Maximum target price as % gain (applies only when TARGET_DIMINISHING_ENABLED=False) - TESTING: Set to 50% to effectively disable target cap and test wave detection
TARGET_DIMINISHING_MULTIPLIER = 0.75  # When diminishing enabled, cap target at this fraction of calculated target (0.75 = 75%)
# Diminishing Target / Augmenting Stop Configuration
TARGET_DIMINISHING_ENABLED = True   # Enable diminishing target over time (target → buy_price over max_days)
STOP_AUGMENTING_ENABLED = False     # Enable trailing stop over time (stop → buy_price over max_days)
# Wave Exhaustion Exit Configuration (DISABLED - Plan C)
WAVE_EXIT_ENABLED = False  # Enable advanced peak detection as primary exit (wave descent detection)
WAVE_EXIT_MIN_PROFIT_PCT = 3.0  # Minimum profit % before allowing wave exit (increased to let stocks run more)
WAVE_TIME_EXPIRY_MULTIPLIER = 1.2  # Exit after this multiplier × dominant_period (wave cycle complete - fallback)
# Descending Trend Exit Configuration (DISABLED - Plan C, will be replaced by STOP_SLIDE)
DESC_TREND_ENABLED = False  # Enable descending trend detection (price drops below recent average by threshold)
DESC_TREND_LOOKBACK = 14  # Number of days to average for trend calculation
DESC_TREND_THRESHOLD = -3  # Sell if price drops this % below recent average (e.g., -3.0 = 3% below)
# STOP_SLIDE Configuration (Plan C - structural exit on loss of directional control)
STOP_SLIDE_ENABLED = False  # Enable STOP_SLIDE exit (price below smoothed with negative smoothed slope)
STOP_SLIDE_CONFIRM_BARS = 1  # Number of consecutive bars to confirm STOP_SLIDE signal
# Note: STOP_SLIDE uses wave-aware half_period with progressive min_periods for early activation
# Note: EXIT_WAVE_POSITION is deprecated - replaced by multi-signal peak detection algorithm
# EXPERIMENTAL FILTERS (currently DISABLED - collecting more data for analysis):
MAX_WAVE_POSITION = -999  # Maximum (most negative) wave_position to accept (filters strong downtrends) - DISABLED
MIN_CONSISTENCY = 0.0  # Minimum consistency score (filters inconsistent wave patterns) - DISABLED
# Note: Set MAX_WAVE_POSITION = -999 to disable wave position filter
#       Set MIN_CONSISTENCY = 0.0 to disable consistency filter


def get_last_trading_day():
    """
    Get the previous working day (Mon-Fri) for Polygon API.
    
    Only works Tue-Fri (skips Mon/Sat/Sun discoveries).
    Returns None if today is Mon/Sat/Sun or if last day was a holiday.
    
    Returns:
        date string (YYYY-MM-DD) or None
    """
    today = datetime.now().date()
    weekday = today.weekday()  # Monday=0, Sunday=6
    
    # Only run Tue-Fri (1-4)
    if weekday == 0:  # Monday
        print("⚠️  Skipping discovery on Monday")
        return None
    elif weekday >= 5:  # Saturday (5) or Sunday (6)
        print("⚠️  Skipping discovery on weekend")
        return None
    
    # Tue-Fri: previous working day is just yesterday
    # (if today is Tue, yesterday is Mon - both weekdays)
    previous_day = today - timedelta(days=1)
    
    # If yesterday was Sunday (previous_day.weekday() == 6), go back to Friday
    if previous_day.weekday() == 6:  # Yesterday was Sunday
        previous_day = previous_day - timedelta(days=2)  # Go to Friday
    # If yesterday was Saturday (previous_day.weekday() == 5), go back to Friday
    elif previous_day.weekday() == 5:  # Yesterday was Saturday
        previous_day = previous_day - timedelta(days=1)  # Go to Friday
    
    return previous_day.strftime("%Y-%m-%d")


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
    Always replaces the last entry's price with current price from yfinance.
    
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
        
        # Always replace last entry's price with current price from yfinance
        # This ensures we use the actual current price for analysis, not just historical closing price
        try:
            if len(df) > 0:
                info = ticker_obj.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                
                if current_price and current_price > 0:
                    # Replace the last row's prices with current prices
                    df.iloc[-1, df.columns.get_loc('close')] = current_price
                    
                    # Update high/low if available, otherwise use current_price
                    current_low = info.get('dayLow') or current_price
                    current_high = info.get('dayHigh') or current_price
                    df.iloc[-1, df.columns.get_loc('low')] = current_low
                    df.iloc[-1, df.columns.get_loc('high')] = current_high
                    
                    # Update volume if available
                    current_volume = info.get('volume')
                    if current_volume:
                        df.iloc[-1, df.columns.get_loc('volume')] = current_volume
        except Exception as e:
            # If current price fetch fails, continue with historical data only
            pass
        
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
        reference_date = get_last_trading_day()
        if reference_date is None:
            print("❌ No valid trading date available (Mon/weekend/holiday)")
            return pd.DataFrame()
    
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
    
    # Track specific stocks for detailed diagnostics (can be set via environment or config)
    diagnostic_tickers = ["NE", "NVDL"]  # Stocks to show detailed filtering info for
    
    for i, (_, row) in enumerate(df_daily.iterrows(), 1):
        ticker = row["ticker"]
        today_volume = row["today_volume"]
        last_close = row["price"]
        
        show_diagnostic = ticker in diagnostic_tickers
        
        if verbose and i % 50 == 0:
            print(f"   Processed {i}/{len(df_daily)} stocks...")
        
        # Get historical data to calculate average volume
        # yfinance is rate-limit friendly, no delay needed
        try:
            df_hist = get_historical_data_yfinance(ticker, start_date, reference_date)
            if df_hist.empty or len(df_hist) < lookback_days:
                if show_diagnostic and verbose:
                    print(f"   {ticker}: ✗ Insufficient historical data ({len(df_hist) if not df_hist.empty else 0} points, need {lookback_days})")
                continue
            
            df_hist = df_hist.sort_values("date").tail(lookback_days)
            avg_volume = df_hist["volume"].mean()
            
            if avg_volume < min_avg_volume:
                if show_diagnostic and verbose:
                    print(f"   {ticker}: ✗ Avg volume {avg_volume:,.0f} < MIN_AVG_VOLUME {min_avg_volume:,}")
                continue
            
            rel_volume = today_volume / avg_volume if avg_volume > 0 else 0
            
            if not (rel_volume_min <= rel_volume <= rel_volume_max):
                if show_diagnostic and verbose:
                    print(f"   {ticker}: ✗ Rel volume {rel_volume:.2f} not in range [{rel_volume_min:.1f}, {rel_volume_max:.1f}] (avg={avg_volume:,.0f}, today={today_volume:,.0f})")
                continue
            
            candidates.append({
                "ticker": ticker,
                "price": last_close,
                "avg_volume": int(avg_volume),
                "today_volume": int(today_volume),
                "rel_volume": round(rel_volume, 2)
            })
            
            if show_diagnostic and verbose:
                print(f"   {ticker}: ✓ Passed initial filters (price=${last_close:.2f}, rel_vol={rel_volume:.2f})")
            
        except Exception as e:
            if show_diagnostic and verbose:
                print(f"   {ticker}: ✗ Error in candidate building: {e}")
            continue
    
    df_candidates = pd.DataFrame(candidates)
    if df_candidates.empty:
        if verbose:
            print("   No candidates found matching criteria")
        return df_candidates
    
    # Deduplicate by ticker (keep first occurrence)
    if len(df_candidates) > 0:
        df_candidates = df_candidates.drop_duplicates(subset=['ticker'], keep='first')
        if verbose and len(candidates) > len(df_candidates):
            print(f"   Removed {len(candidates) - len(df_candidates)} duplicate tickers")
    
    df_candidates["vol_score"] = 1 - abs(df_candidates["rel_volume"] - 1)
    
    if verbose:
        print(f"✓ Found {len(df_candidates)} candidate stocks with stable volume patterns")
    
    return df_candidates.sort_values("vol_score", ascending=False).reset_index(drop=True)


def wavelet_trade_engine(price_series, min_rr=MIN_RR, low_series=None, turn_confirmation_enabled=None):
    """
    Analyze price series using wavelet analysis to detect cyclical patterns and generate trade signals.
    
    Args:
        price_series: pandas Series of closing prices
        min_rr: Minimum reward:risk ratio required
        low_series: pandas Series of low prices (optional, needed for turn confirmation)
        turn_confirmation_enabled: If True, require turn confirmation before accepting (default: TURN_CONFIRMATION_ENABLED)
    
    Returns:
        dict with trade analysis results, including 'accepted' boolean flag
    """
    if turn_confirmation_enabled is None:
        turn_confirmation_enabled = TURN_CONFIRMATION_ENABLED
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
    
    # Turn Confirmation Check (ChatGPT's recommendation: least-laggy turn condition)
    # Confirms seller exhaustion: higher close AND higher low (price has actually turned)
    if turn_confirmation_enabled and low_series is not None:
        if len(price_series) >= 2 and len(low_series) >= 2:
            current_close = float(price_series.iloc[-1])
            prev_close = float(price_series.iloc[-2])
            current_low = float(low_series.iloc[-1])
            prev_low = float(low_series.iloc[-2])
            
            # Turn confirmed: higher close AND higher low (seller exhaustion)
            turn_confirmed = (current_close > prev_close) and (current_low > prev_low)
            
            if not turn_confirmed:
                log.append(f"Rejected: Turn not confirmed (close: {current_close:.2f} vs {prev_close:.2f}, low: {current_low:.2f} vs {prev_low:.2f})")
                return {"accepted": False, "reason": "Turn not confirmed", "log": log}
            else:
                log.append(f"Turn confirmed: higher close ({current_close:.2f} > {prev_close:.2f}) and higher low ({current_low:.2f} > {prev_low:.2f})")
        else:
            if turn_confirmation_enabled:
                log.append("Rejected: Insufficient data for turn confirmation (need at least 2 bars)")
                return {"accepted": False, "reason": "Insufficient data for turn confirmation", "log": log}
    
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
        "half_period": half_period,  # Smoothing window for rollover detection
        "consistency": round(consistency, 3),
        "wave_position": round(wave_position, 3),
        "buy": round(buy, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "reward_risk": round(rr, 2),
        # Wave state for exit detection (wave exhaustion signals)
        "avg_trough": round(avg_trough, 2),
        "avg_peak": round(avg_peak, 2),
        "wave_range": round(wave_range, 2),
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
        reference_date = get_last_trading_day()
        if reference_date is None:
            print("❌ No valid trading date available (Mon/weekend/holiday)")
            return pd.DataFrame()
    
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
            
            wave_result = wavelet_trade_engine(df_price["close"], min_rr, low_series=df_price["low"])
            
            # Show diagnostic info for specific tickers or all rejected stocks in verbose mode
            diagnostic_tickers = ["NE", "NVDL"]
            show_full_diagnostic = ticker in diagnostic_tickers or verbose
            
            if not wave_result.get("accepted", False):
                reason = wave_result.get("reason", "Unknown")
                if show_full_diagnostic:
                    print(f"   {ticker}: ✗ Rejected - {reason}")
                    if wave_result.get("log"):
                        # Show all log messages for diagnostic tickers, last 3 for others
                        log_msgs = wave_result["log"]
                        num_to_show = len(log_msgs) if ticker in diagnostic_tickers else min(3, len(log_msgs))
                        for log_msg in log_msgs[-num_to_show:]:
                            print(f"      {log_msg}")
                
                # Show detailed metrics for diagnostic tickers
                if ticker in diagnostic_tickers:
                    if 'wave_position' in wave_result:
                        wp = wave_result['wave_position']
                        print(f"      Wave position: {wp:.3f} (filter: > -3.0 and <= 0.35)")
                    if 'consistency' in wave_result:
                        print(f"      Consistency: {wave_result['consistency']:.3f}")
                    if 'rr' in wave_result:
                        print(f"      R:R ratio: {wave_result['rr']:.2f} (required: >= {min_rr})")
            
            
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
    
    # Deduplicate by ticker (keep first occurrence - should already be unique, but safety check)
    if len(df_results) > 0:
        original_len = len(df_results)
        df_results = df_results.drop_duplicates(subset=['ticker'], keep='first')
        if verbose and original_len > len(df_results):
            print(f"   ⚠️  Removed {original_len - len(df_results)} duplicate tickers from results")
    
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


def detect_peak_formation_simple(
    price_series: pd.Series,
    smoothed_series: pd.Series,
    min_lookback: int = 3
):
    """
    Simple smoothed rollover detection (ChatGPT version):
    Detects when price crosses below a rising smoothed line.
    
    Args:
        price_series: Price series since entry
        smoothed_series: Smoothed price series (already calculated)
        min_lookback: Minimum days needed for analysis (default 3)
    
    Returns:
        dict with:
            peak_detected (bool): True if rollover detected
            signal_type (str): 'rollover' or None
            details (dict): Signal-specific details
    """
    if len(price_series) < min_lookback or len(smoothed_series) < min_lookback:
        return {
            "peak_detected": False,
            "signal_type": None,
            "details": {"reason": "insufficient_data"}
        }
    
    # Need at least 2 points to check for rollover
    if len(smoothed_series) < 2:
        return {
            "peak_detected": False,
            "signal_type": None,
            "details": {"reason": "insufficient_smoothed_data"}
        }
    
    # Get the last two points
    current_price = price_series.iloc[-1]
    current_smoothed = smoothed_series.iloc[-1]
    prev_smoothed = smoothed_series.iloc[-2]
    
    # Rollover: Price below smoothed line AND smoothed line was rising
    if current_price < current_smoothed and prev_smoothed < current_smoothed:
        return {
            "peak_detected": True,
            "signal_type": "rollover",
            "details": {
                "price": round(current_price, 2),
                "smoothed": round(current_smoothed, 2),
                "prev_smoothed": round(prev_smoothed, 2)
            }
        }
    
    return {
        "peak_detected": False,
        "signal_type": None,
        "details": {}
    }


def detect_wave_exhaustion(
    current_price: float,
    entry_wave_state: dict,
    price_series_since_entry: pd.Series,
    entry_date,
    current_date,
    buy_price: float,
    target_price: float,
    stop_price: float = None,
    config: dict = None
):
    """
    PRIMARY exit system: Detects wave peak/descent using simple smoothed rollover detection.
    
    Args:
        current_price: Current stock price
        entry_wave_state: dict with wave state at entry (avg_trough, avg_peak, wave_range, 
                          dominant_period_days, half_period)
        price_series_since_entry: pandas Series of prices since entry (for peak detection)
        entry_date: Entry date (datetime or date)
        current_date: Current date (datetime or date)
        buy_price: Entry price (for profit guard and progress calculation)
        target_price: Target price (for progress calculation)
        stop_price: Stop loss price (optional, for comparison - don't exit if worse than stop)
        config: dict with exit thresholds (optional, uses defaults if None)

    Returns:
        dict with keys:
            exhausted (bool): True if wave exhaustion detected
            signal (str or None): 'peak' | 'desc_trend' | 'time_expiry' | None
            progress (float): Progress to target (0.0 = entry, 1.0 = target)
            details (dict): Signal-specific details
    """
    # Use config parameter or defaults
    if config is None:
        config = {}

    time_expiry_mult = config.get("WAVE_TIME_EXPIRY_MULTIPLIER", WAVE_TIME_EXPIRY_MULTIPLIER)
    min_profit_pct = config.get("WAVE_EXIT_MIN_PROFIT_PCT", WAVE_EXIT_MIN_PROFIT_PCT)  # Use global constant default
    desc_trend_enabled = config.get("DESC_TREND_ENABLED", DESC_TREND_ENABLED)
    desc_trend_lookback = config.get("DESC_TREND_LOOKBACK", DESC_TREND_LOOKBACK)
    desc_trend_threshold = config.get("DESC_TREND_THRESHOLD", DESC_TREND_THRESHOLD)

    dominant_period = entry_wave_state["dominant_period_days"]
    half_period = entry_wave_state["half_period"]

    # Calculate current profit percentage
    return_pct = ((current_price - buy_price) / buy_price) * 100

    # Calculate progress to target (for reporting/analysis)
    target_range = target_price - buy_price
    if target_range <= 0:
        progress = 0.0
    else:
        progress = max(0.0, (current_price - buy_price) / target_range)

    # -----------------------------
    # 2️⃣ Descending Trend Detection (price drops below recent average) - CHECK BEFORE PROFIT GUARD
    # This should work even at a loss to protect against further declines
    # Improved: Only trigger if we're actually declining (not just a temporary dip)
    # -----------------------------
    if desc_trend_enabled and len(price_series_since_entry) >= desc_trend_lookback + 3:
        # Calculate recent average (last N days)
        recent_avg = price_series_since_entry.iloc[-desc_trend_lookback:].mean()
        
        # Calculate earlier average (to confirm we're actually declining, not just volatile)
        earlier_start = -desc_trend_lookback - 3
        earlier_end = -desc_trend_lookback
        if earlier_start < -len(price_series_since_entry):
            earlier_start = -len(price_series_since_entry)
        earlier_avg = price_series_since_entry.iloc[earlier_start:earlier_end].mean() if earlier_end < 0 else None
        
        if recent_avg > 0:
            trend_pct = ((current_price - recent_avg) / recent_avg) * 100
            
            # Only trigger if:
            # 1. Price drops below recent average by threshold percentage
            # 2. Recent average is declining (confirming actual trend, not just volatility)
            avg_declining = True
            if earlier_avg is not None and earlier_avg > 0:
                avg_change_pct = ((recent_avg - earlier_avg) / earlier_avg) * 100
                # Average should be declining (at least slightly negative) to confirm trend
                avg_declining = avg_change_pct < -0.5  # Average dropped at least 0.5%
            
            # Also check if exit price would be better than stop loss (if stop_price provided)
            better_than_stop = True
            if stop_price is not None:
                # Only trigger if exit price would be better than stop loss
                better_than_stop = current_price >= stop_price
            
            if trend_pct <= desc_trend_threshold and avg_declining and better_than_stop:
                return {
                    "exhausted": True,
                    "signal": "desc_trend",
                    "progress": round(progress, 3),
                    "details": {
                        "signal_type": "desc_trend"
                    }
                }

    # 🔒 Profit guard: Small minimum profit to avoid noise exits (applies to peak detection only)
    if return_pct < min_profit_pct:
        return {
            "exhausted": False,
            "signal": None,
            "progress": 0.0,
            "details": {"reason": f"profit_too_low ({return_pct:.2f}% < {min_profit_pct}%)"}
        }

    # -----------------------------
    # 🎯 PRIMARY: Simple Smoothed Rollover Detection (ChatGPT version)
    # -----------------------------
    if len(price_series_since_entry) >= half_period + 3:  # Need enough data
        smoothed = price_series_since_entry.rolling(
            window=half_period, min_periods=half_period
        ).mean()
        
        peak_result = detect_peak_formation_simple(
            price_series=price_series_since_entry,
            smoothed_series=smoothed,
            min_lookback=3
        )
        
        if peak_result["peak_detected"]:
            return {
                "exhausted": True,
                "signal": "peak",
                "progress": round(progress, 3),
                "details": {
                    "signal_type": peak_result["signal_type"]
                }
                }
    
    # -----------------------------
    # 3️⃣ Time-based cycle expiry (fallback)
    # -----------------------------
    # Ensure both dates are date objects for subtraction
    if hasattr(entry_date, 'date'):
        entry_date_only = entry_date.date()
    else:
        entry_date_only = entry_date
    
    if hasattr(current_date, 'date'):
        current_date_only = current_date.date()
    else:
        current_date_only = current_date
    
    days_held = (current_date_only - entry_date_only).days
    max_days = int(time_expiry_mult * dominant_period)

    if days_held >= max_days:
        return {
            "exhausted": True,
            "signal": "time_expiry",
            "progress": round(progress, 3),
            "details": {
                "signal_type": "time"
            }
        }

    # -----------------------------
    # Hold (no exit signals)
    # -----------------------------
    return {
        "exhausted": False,
        "signal": None,
        "progress": round(progress, 3),
        "details": {}
    }


def backtest_signal(ticker, entry_date, buy_price, stop_price, target_price, max_days=40,
                    target_diminishing_enabled=False, stop_augmenting_enabled=False,
                    wave_exit_enabled=False, entry_wave_state=None, half_period=None):
    """
    Backtest a single trading signal with target/stop exit logic.
    
    Args:
        ticker: Stock ticker symbol
        entry_date: Entry date string (YYYY-MM-DD)
        buy_price: Entry price
        stop_price: Stop loss price (original)
        target_price: Target price (original)
        max_days: Maximum days to hold (also used for diminishing/augmenting period)
        target_diminishing_enabled: If True, target diminishes from original_target → buy_price over max_days
        stop_augmenting_enabled: If True, stop augments from original_stop → buy_price over max_days
        wave_exit_enabled: (DEPRECATED - Plan C) No longer used, kept for compatibility
        entry_wave_state: (DEPRECATED - Plan C) No longer used, kept for compatibility
    
    Returns:
        dict with results: {'outcome': 'win'|'loss'|'timeout', 'days_held': int, 
                           'exit_price': float, 'return_pct': float, 'progress': float}
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
        
        # Track price history for STOP_SLIDE detection
        price_history_since_entry = []
        
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
            
            # Update price history for STOP_SLIDE detection
            price_history_since_entry.append(close)
            price_series = pd.Series(price_history_since_entry)
            
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
                # Calculate progress for reporting
                target_range = current_target - buy_price
                progress = max(0.0, (exit_price - buy_price) / target_range) if target_range > 0 else 0.0
                return {
                    'outcome': 'loss',
                    'days_held': days_held,
                    'exit_price': exit_price,
                    'return_pct': return_pct,
                    'exit_signal': None,
                    'progress': round(progress, 3),
                    'exit_details': {}
                }
            
            # -----------------------------
            # STOP_SLIDE: Loss of directional control (wave-aware, progressive)
            # -----------------------------
            if STOP_SLIDE_ENABLED and half_period is not None:
                # Only activate once trade is profitable (protects winners in progress)
                # Hard stop handles fast failures; STOP_SLIDE handles structural decay
                if close > buy_price and len(price_series) >= STOP_SLIDE_CONFIRM_BARS + 2:
                    window = half_period
                    min_p = max(3, window // 3)  # Progressive activation - allows smoothing to form early
                    
                    smoothed = price_series.rolling(
                        window=window,
                        min_periods=min_p
                    ).mean()
                    
                    # Need valid smoothed values for all recent bars
                    recent_price = price_series.iloc[-STOP_SLIDE_CONFIRM_BARS:]
                    recent_smoothed = smoothed.iloc[-STOP_SLIDE_CONFIRM_BARS:]
                    recent_slope = recent_smoothed.diff().dropna()
                    
                    # STOP_SLIDE conditions:
                    # 1) Price below smoothed (structural decline)
                    # 2) Smoothed slope turning negative (momentum loss)
                    # 3) Confirmed for N bars (avoid false signals)
                    if (
                        len(recent_slope) >= STOP_SLIDE_CONFIRM_BARS - 1 and
                        (recent_price < recent_smoothed).all() and
                        (recent_slope < 0).all()
                    ):
                        exit_price = close
                        return_pct = ((exit_price - buy_price) / buy_price) * 100
                        # Calculate progress for reporting
                        target_range = current_target - buy_price
                        progress = max(0.0, (exit_price - buy_price) / target_range) if target_range > 0 else 0.0
                        return {
                            'outcome': 'stop_slide',
                            'days_held': days_held,
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'exit_signal': None,
                            'progress': round(progress, 3),
                            'exit_details': {}
                        }
            
            # Check if target was hit (price went above target)
            if high >= current_target:
                exit_price = current_target
                return_pct = ((exit_price - buy_price) / buy_price) * 100
                # Progress = 1.0 when target is hit (100% of target range)
                target_range = current_target - buy_price
                progress = 1.0 if target_range > 0 else 0.0
                return {
                    'outcome': 'win',
                    'days_held': days_held,
                    'exit_price': exit_price,
                    'return_pct': return_pct,
                    'exit_signal': None,
                    'progress': round(progress, 3),
                    'exit_details': {}
                }
            
            # If we've exceeded max_days, exit at close
            if days_held >= max_days:
                exit_price = close
                return_pct = ((exit_price - buy_price) / buy_price) * 100
                # Calculate progress for reporting
                target_range = current_target - buy_price
                progress = max(0.0, (exit_price - buy_price) / target_range) if target_range > 0 else 0.0
                return {
                    'outcome': 'timeout',
                    'days_held': days_held,
                    'exit_price': exit_price,
                    'return_pct': return_pct,
                    'exit_signal': None,
                    'progress': round(progress, 3),
                    'exit_details': {}
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
        
        # Calculate progress for reporting (using original target)
        target_range = target_price - buy_price
        progress = max(0.0, (last_close - buy_price) / target_range) if target_range > 0 else 0.0
        
        return {
            'outcome': 'timeout',
            'days_held': days_held,
            'exit_price': last_close,
            'return_pct': return_pct,
            'exit_signal': None,
            'progress': round(progress, 3),
            'exit_details': {}
        }
        
    except Exception as e:
        return {
            'outcome': 'error',
            'days_held': 0,
            'exit_price': buy_price,
            'return_pct': 0.0,
            'exit_signal': None,
            'progress': 0.0,
            'exit_details': {},
            'error': str(e)
        }


def track_stock_reentry(tickers, start_date, end_date, verbose=False):
    """
    Track one or more stocks over a date range to see:
    - When they pass filters and become candidates
    - When they would be bought
    - When they hit stop/target/timeout
    - If/when they re-enter as candidates after a stop
    
    Args:
        tickers: List of ticker symbols (e.g., ['NE', 'NVDL']) or single ticker string
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        verbose: If True, show detailed filter rejection reasons
    
    Returns:
        pandas DataFrame with timeline of events
    """
    # Normalize tickers to list
    if isinstance(tickers, str):
        tickers = [tickers.upper()]
    else:
        tickers = [t.upper() for t in tickers]
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Validate dates
    today = datetime.today()
    if start_dt > today:
        print(f"⚠️  Warning: Start date {start_date} is in the future (today is {today.strftime('%Y-%m-%d')})")
        print(f"   Historical data won't be available for future dates")
    if end_dt > today:
        print(f"⚠️  Warning: End date {end_date} is in the future (today is {today.strftime('%Y-%m-%d')})")
        print(f"   Historical data won't be available for future dates")
    if end_dt < start_dt:
        print(f"❌ Error: End date {end_date} is before start date {start_date}")
        return pd.DataFrame()
    
    # Generate all trading dates in range
    test_dates = []
    current_dt = start_dt
    while current_dt <= end_dt:
        if current_dt.weekday() < 5:  # Skip weekends
            test_dates.append(current_dt.strftime("%Y-%m-%d"))
        current_dt += timedelta(days=1)
    
    if verbose:
        print(f"📊 Tracking {', '.join(tickers)} from {start_date} to {end_date}")
        print(f"   {len(test_dates)} trading days to check\n")
    
    timeline = []
    active_positions = {}  # ticker -> {entry_date, buy_price, stop_price, target_price, half_period, exit_date, exit_result}
    positions_to_remove = []
    
    for test_date in test_dates:
        test_dt = datetime.strptime(test_date, "%Y-%m-%d")
        
        # Check for active positions first (backtest if they exit)
        for ticker in list(active_positions.keys()):
            pos = active_positions[ticker]
            
            # Only backtest once per position (lazy evaluation)
            if 'exit_result' not in pos:
                result = backtest_signal(
                    ticker, pos['entry_date'], pos['buy_price'], pos['stop_price'], pos['target_price'],
                    max_days=40,
                    target_diminishing_enabled=TARGET_DIMINISHING_ENABLED,
                    stop_augmenting_enabled=STOP_AUGMENTING_ENABLED,
                    wave_exit_enabled=False,
                    entry_wave_state=None,
                    half_period=pos.get('half_period')
                )
                pos['exit_result'] = result
                
                # Calculate exit date
                if result['outcome'] != 'no_data':
                    entry_dt = datetime.strptime(pos['entry_date'], "%Y-%m-%d")
                    exit_dt = entry_dt + timedelta(days=result['days_held'])
                    pos['exit_date'] = exit_dt.strftime("%Y-%m-%d")
            
            # Check if position exited on or before this date
            if 'exit_date' in pos and pos['exit_date'] <= test_date:
                # Position exited - record it
                result = pos['exit_result']
                timeline.append({
                    'date': pos['exit_date'],
                    'ticker': ticker,
                    'event': 'exit',
                    'outcome': result['outcome'],
                    'entry_date': pos['entry_date'],
                    'buy_price': pos['buy_price'],
                    'exit_price': result['exit_price'],
                    'return_pct': result['return_pct'],
                    'days_held': result['days_held'],
                    'stop_price': pos['stop_price'],
                    'target_price': pos['target_price'],
                })
                # Mark for removal
                positions_to_remove.append(ticker)
        
        # Remove exited positions
        for ticker in positions_to_remove:
            del active_positions[ticker]
        positions_to_remove = []
        
        # Now check if any tracked tickers become candidates on this date
        for ticker in tickers:
            # Skip if already in an active position
            if ticker in active_positions:
                continue
            
            # Direct check: Get historical data and run wavelet analysis
            # (bypassing Polygon - using yfinance directly)
            try:
                # Calculate lookback period for wavelet analysis
                ref_dt = datetime.strptime(test_date, "%Y-%m-%d")
                wavelet_lookback_days = max(120, LOOKBACK_DAYS * 2)
                start_date_wavelet = (ref_dt - timedelta(days=wavelet_lookback_days)).strftime("%Y-%m-%d")
                
                # Get historical data using yfinance
                # Request data up to test_date (yfinance handles past dates)
                df_price = get_historical_data_yfinance(ticker, start_date_wavelet, test_date)
                
                # If empty, try requesting with a small buffer past test_date (yfinance sometimes needs this)
                if df_price.empty:
                    test_dt = datetime.strptime(test_date, "%Y-%m-%d")
                    end_date_with_buffer = (test_dt + timedelta(days=10)).strftime("%Y-%m-%d")
                    df_price = get_historical_data_yfinance(ticker, start_date_wavelet, end_date_with_buffer)
                    
                    # Filter to only dates <= test_date (to avoid lookahead bias)
                    if not df_price.empty and 'date' in df_price.columns:
                        df_price['date_dt'] = pd.to_datetime(df_price['date'])
                        test_dt_ts = pd.Timestamp(test_date)
                        df_price = df_price[df_price['date_dt'] <= test_dt_ts].copy()
                        if 'date_dt' in df_price.columns:
                            df_price = df_price.drop(columns=['date_dt'])
                        # Reset index after filtering
                        df_price = df_price.reset_index(drop=True)
                
                if df_price.empty:
                    if verbose:
                        print(f"   {ticker} ({test_date}): ✗ No data returned from yfinance (start: {start_date_wavelet}, end: {test_date})")
                    continue
                
                if len(df_price) < 64:
                    if verbose:
                        print(f"   {ticker} ({test_date}): ✗ Insufficient data ({len(df_price)} points, need 64)")
                    continue
                
                # Get current price for basic filters
                current_price = float(df_price['close'].iloc[-1])
                
                # Basic price filter (same as build_candidates would do)
                if current_price < MIN_PRICE or current_price > MAX_PRICE:
                    if verbose:
                        print(f"   {ticker} ({test_date}): ✗ Price ${current_price:.2f} outside range [${MIN_PRICE:.2f}, ${MAX_PRICE:.2f}]")
                    continue
                
                # Check volume filters (same as build_candidates)
                if 'volume' in df_price.columns and len(df_price) >= LOOKBACK_DAYS:
                    avg_volume = df_price['volume'].tail(LOOKBACK_DAYS).mean()
                    today_volume = df_price['volume'].iloc[-1]
                    
                    if avg_volume < MIN_AVG_VOLUME:
                        if verbose:
                            print(f"   {ticker} ({test_date}): ✗ Avg volume {avg_volume:,.0f} < {MIN_AVG_VOLUME:,}")
                        continue
                    
                    rel_volume = today_volume / avg_volume if avg_volume > 0 else 0
                    if rel_volume < REL_VOLUME_MIN or rel_volume > REL_VOLUME_MAX:
                        if verbose:
                            print(f"   {ticker} ({test_date}): ✗ Rel volume {rel_volume:.2f} not in range [{REL_VOLUME_MIN}, {REL_VOLUME_MAX}]")
                        continue
                
                # Run wavelet analysis (this is the real filter)
                wave_result = wavelet_trade_engine(df_price["close"], MIN_RR, low_series=df_price["low"])
                
                if wave_result.get("accepted", False):
                    # Stock passed all filters - it's a candidate!
                    buy_price = wave_result['buy']
                    calculated_stop_price = wave_result['stop']
                    target_price = wave_result['target']
                    
                    # Apply minimum stop buffer (if MIN_STOP_BUFFER_PCT=0, min_stop_price=buy_price so calculated_stop is always used)
                    min_stop_price = buy_price * (1 - MIN_STOP_BUFFER_PCT)
                    stop_price = min(calculated_stop_price, min_stop_price)  # Use wider stop (lower price)
                    
                    # Apply target cap (same as backtest)
                    if not TARGET_DIMINISHING_ENABLED:
                        max_target_price = buy_price * (1 + MAX_TARGET_PCT)
                        target_price = min(target_price, max_target_price)
                    
                    timeline.append({
                        'date': test_date,
                        'ticker': ticker,
                        'event': 'candidate',
                        'outcome': None,
                        'entry_date': test_date,
                        'buy_price': buy_price,
                        'exit_price': None,
                        'return_pct': None,
                        'days_held': None,
                        'stop_price': stop_price,
                        'target_price': target_price,
                        'reward_risk': wave_result.get('reward_risk'),
                        'consistency': wave_result.get('consistency'),
                        'wave_position': wave_result.get('wave_position'),
                    })
                    
                    # Add to active positions
                    active_positions[ticker] = {
                        'entry_date': test_date,
                        'buy_price': buy_price,
                        'stop_price': stop_price,
                        'target_price': target_price,
                        'half_period': wave_result.get('half_period')
                    }
                    
                    if verbose:
                        print(f"   {ticker} ({test_date}): ✓ Candidate (R:R={wave_result.get('reward_risk', 0):.2f}, wave_pos={wave_result.get('wave_position', 0):.3f})")
                else:
                    # Stock didn't pass wavelet filters
                    if verbose:
                        reason = wave_result.get('reason', 'Unknown')
                        print(f"   {ticker} ({test_date}): ✗ Rejected - {reason}")
                        # Show last few log messages for context
                        if wave_result.get('log'):
                            log_msgs = wave_result['log']
                            for log_msg in log_msgs[-3:]:
                                print(f"      {log_msg}")
                        
            except Exception as e:
                if verbose:
                    print(f"   {ticker} ({test_date}): ✗ Error - {e}")
                continue
    
    # Check any remaining active positions at end of range
    for ticker, pos in active_positions.items():
        # Backtest if not already done
        if 'exit_result' not in pos:
            result = backtest_signal(
                ticker, pos['entry_date'], pos['buy_price'], pos['stop_price'], pos['target_price'],
                max_days=40,
                target_diminishing_enabled=TARGET_DIMINISHING_ENABLED,
                stop_augmenting_enabled=STOP_AUGMENTING_ENABLED,
                wave_exit_enabled=False,
                entry_wave_state=None,
                half_period=pos.get('half_period')
            )
            pos['exit_result'] = result
        
        result = pos['exit_result']
        if result['outcome'] != 'no_data':
            # Calculate exit date
            entry_dt = datetime.strptime(pos['entry_date'], "%Y-%m-%d")
            exit_dt = entry_dt + timedelta(days=result['days_held'])
            
            timeline.append({
                'date': exit_dt.strftime("%Y-%m-%d"),
                'ticker': ticker,
                'event': 'exit',
                'outcome': result['outcome'],
                'entry_date': pos['entry_date'],
                'buy_price': pos['buy_price'],
                'exit_price': result['exit_price'],
                'return_pct': result['return_pct'],
                'days_held': result['days_held'],
                'stop_price': pos['stop_price'],
                'target_price': pos['target_price'],
            })
    
    # Convert to DataFrame and sort by date, then ticker
    if timeline:
        df = pd.DataFrame(timeline)
        df = df.sort_values(['date', 'ticker'])
        return df
    else:
        return pd.DataFrame()


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
            
            # Apply minimum stop buffer (if MIN_STOP_BUFFER_PCT=0, min_stop_price=buy_price so calculated_stop is always used)
            min_stop_price = buy_price * (1 - MIN_STOP_BUFFER_PCT)
            stop_price = min(calculated_stop_price, min_stop_price)  # Use wider stop (lower price)
            
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
            
            # Get half_period for STOP_SLIDE (if available in candidates)
            half_period_val = row.get('half_period')
            if half_period_val is not None:
                half_period_val = max(3, int(half_period_val))  # Ensure it's a valid integer
            
            result = backtest_signal(
                ticker, test_date, buy_price, stop_price, target_price,
                max_days=40,
                target_diminishing_enabled=TARGET_DIMINISHING_ENABLED,
                stop_augmenting_enabled=STOP_AUGMENTING_ENABLED,
                wave_exit_enabled=False,  # Disabled - Plan C
                entry_wave_state=None,
                half_period=half_period_val
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
                outcome_symbol = "✓" if result['outcome'] == 'win' else "✗" if result['outcome'] == 'loss' else "📉" if result['outcome'] == 'stop_slide' else "⏱"
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
    stop_slides = len(df_backtest[df_backtest['outcome'] == 'stop_slide'])
    timeouts = len(df_backtest[df_backtest['outcome'] == 'timeout'])
    
    profitable_trades = wins + stop_slides
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_return_pct = df_backtest['return_pct'].mean()
    avg_return_wins_pct = df_backtest[df_backtest['outcome'] == 'win']['return_pct'].mean() if wins > 0 else 0
    avg_return_stop_slides_pct = df_backtest[df_backtest['outcome'] == 'stop_slide']['return_pct'].mean() if stop_slides > 0 else 0
    avg_return_profitable_pct = df_backtest[df_backtest['outcome'].isin(['win', 'stop_slide'])]['return_pct'].mean() if profitable_trades > 0 else 0
    avg_return_losses_pct = df_backtest[df_backtest['outcome'] == 'loss']['return_pct'].mean() if losses > 0 else 0
    
    avg_return_dollars = df_backtest['return_dollars'].mean()
    avg_return_wins_dollars = df_backtest[df_backtest['outcome'] == 'win']['return_dollars'].mean() if wins > 0 else 0
    avg_return_stop_slides_dollars = df_backtest[df_backtest['outcome'] == 'stop_slide']['return_dollars'].mean() if stop_slides > 0 else 0
    avg_return_profitable_dollars = df_backtest[df_backtest['outcome'].isin(['win', 'stop_slide'])]['return_dollars'].mean() if profitable_trades > 0 else 0
    avg_return_losses_dollars = df_backtest[df_backtest['outcome'] == 'loss']['return_dollars'].mean() if losses > 0 else 0
    total_return_dollars = df_backtest['return_dollars'].sum()
    
    avg_days_held = df_backtest['days_held'].mean()
    
    print(f"\nTotal Trades: {total_trades} (${POSITION_SIZE:,.0f} position per trade)")
    print(f"Profitable Trades: {profitable_trades} ({win_rate:.1f}%)")
    print(f"  - Wins (target hit): {wins} ({wins/total_trades*100:.1f}%)")
    if stop_slides > 0:
        print(f"  - STOP_SLIDE Exits: {stop_slides} ({stop_slides/total_trades*100:.1f}%)")
    print(f"Losses: {losses} ({losses/total_trades*100:.1f}%)" if total_trades > 0 else "Losses: 0")
    print(f"Timeouts: {timeouts} ({timeouts/total_trades*100:.1f}%)" if total_trades > 0 else "Timeouts: 0")
    print(f"\nAverage Return: {avg_return_pct:.2f}% (${avg_return_dollars:+.2f})")
    if profitable_trades > 0:
        print(f"Average Return (Profitable): {avg_return_profitable_pct:.2f}% (${avg_return_profitable_dollars:+.2f})")
    if wins > 0:
        print(f"Average Return (Wins): {avg_return_wins_pct:.2f}% (${avg_return_wins_dollars:+.2f})")
    if stop_slides > 0:
        print(f"Average Return (STOP_SLIDE): {avg_return_stop_slides_pct:.2f}% (${avg_return_stop_slides_dollars:+.2f})")
    if losses > 0:
        print(f"Average Return (Losses): {avg_return_losses_pct:.2f}% (${avg_return_losses_dollars:+.2f})")
    print(f"Average Days Held: {avg_days_held:.1f} days")
    print(f"\nTotal Return: ${total_return_dollars:+.2f} (on ${total_trades * POSITION_SIZE:,.0f} total capital)")
    
    # Expected value calculation (using profitable vs loss rates)
    if profitable_trades > 0 and losses > 0:
        loss_rate = (losses / total_trades * 100) if total_trades > 0 else 0
        expected_value_pct = (win_rate/100 * avg_return_profitable_pct) + (loss_rate/100 * avg_return_losses_pct)
        expected_value_dollars = (win_rate/100 * avg_return_profitable_dollars) + (loss_rate/100 * avg_return_losses_dollars)
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
        help='Analysis date (YYYY-MM-DD), default: last trading day'
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
    parser.add_argument(
        '--track-stocks',
        type=str,
        nargs='+',
        help='Track specific stock(s) over a date range. Requires --start-date and --end-date.'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for tracking (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for tracking (YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    
    # Handle stock tracking mode
    if args.track_stocks:
        if not args.start_date or not args.end_date:
            print("❌ Error: --start-date and --end-date are required when using --track-stocks")
            sys.exit(1)
        
        df_timeline = track_stock_reentry(
            tickers=args.track_stocks,
            start_date=args.start_date,
            end_date=args.end_date,
            verbose=args.verbose
        )
        
        if df_timeline.empty:
            print(f"\n❌ No events found for {', '.join(args.track_stocks)} in the date range")
        else:
            print("\n" + "=" * 80)
            print(f"TIMELINE: {', '.join(args.track_stocks)}")
            print("=" * 80)
            print(f"\nDate Range: {args.start_date} to {args.end_date}")
            print(f"Total Events: {len(df_timeline)}\n")
            
            # Format output nicely
            display_cols = ['date', 'ticker', 'event', 'outcome', 'entry_date', 
                           'buy_price', 'exit_price', 'return_pct', 'days_held',
                           'stop_price', 'target_price']
            if 'reward_risk' in df_timeline.columns:
                display_cols.append('reward_risk')
            if 'consistency' in df_timeline.columns:
                display_cols.append('consistency')
            if 'wave_position' in df_timeline.columns:
                display_cols.append('wave_position')
            
            display_cols = [col for col in display_cols if col in df_timeline.columns]
            print(df_timeline[display_cols].to_string(index=False))
            
            # Summary statistics
            exits = df_timeline[df_timeline['event'] == 'exit']
            candidates = df_timeline[df_timeline['event'] == 'candidate']
            
            if not exits.empty:
                print(f"\n📊 Summary:")
                print(f"   Candidates found: {len(candidates)}")
                print(f"   Exits: {len(exits)}")
                print(f"   Average return: {exits['return_pct'].mean():.2f}%")
                print(f"   Wins: {len(exits[exits['return_pct'] > 0])}")
                print(f"   Losses: {len(exits[exits['return_pct'] <= 0])}")
                
                # Check for re-entries (candidate after an exit)
                reentries = 0
                for ticker in args.track_stocks:
                    ticker_events = df_timeline[df_timeline['ticker'] == ticker.upper()].sort_values('date')
                    for i in range(1, len(ticker_events)):
                        if (ticker_events.iloc[i-1]['event'] == 'exit' and 
                            ticker_events.iloc[i]['event'] == 'candidate'):
                            reentries += 1
                
                if reentries > 0:
                    print(f"   Re-entries: {reentries} ⭐")
        
        return
    
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
            # Reorder columns for better readability and handle exit_details formatting
            display_cols = [
                'outcome', 'days_held', 'exit_price', 'return_pct', 'exit_signal',
                'progress', 'exit_details', 'test_date', 'ticker', 'buy_price',
                'stop_price', 'target_price', 'reward_risk', 'consistency',
                'wave_position', 'return_dollars'
            ]
            # Only include columns that exist
            display_cols = [col for col in display_cols if col in df_backtest.columns]
            df_display = df_backtest[display_cols].copy()
            
            # Format exit_details dict for better display (show signal_type if available)
            if 'exit_details' in df_display.columns:
                def format_exit_details(x):
                    if isinstance(x, dict) and x:
                        if 'signal_type' in x:
                            return x.get('signal_type', '')
                        return str(x) if x else ''
                    return ''
                df_display['exit_details'] = df_display['exit_details'].apply(format_exit_details)
            
            # Keep progress as numeric (0.0-1.0) - pandas will format it nicely
            # Progress represents % of target range achieved (0.0 = entry, 1.0 = target hit)
            
            print(df_display.to_string(index=False))
        
        return
    
    # Set reference date - use last trading day if not specified
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
        reference_date = get_last_trading_day()
        if reference_date is None:
            print("❌ No valid trading date available (Mon/weekend/holiday)")
            sys.exit(1)
    
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

