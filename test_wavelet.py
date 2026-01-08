#!/usr/bin/env python
"""
Wavelet-based Cycle Trading Levels - Original Implementation

This is the original wavelet-based algorithm for detecting cyclical patterns
and determining buy-in, stop-loss, and target prices.

Based on the original wavelet cycle trade levels implementation.
"""
import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks
import yfinance as yf
from yfinance.screener import EquityQuery as YfEquityQuery
from datetime import datetime, timedelta
import argparse
import sys
import time
import os


def wavelet_cycle_trade_levels(
        price_series,
        scales=np.arange(2, 64),    # good for intra-week to ~2 month cycles
        trough_pct=0.2,             # buy-in above trough
        stop_pct=0.3,               # stop below trough
        peak_pct=0.1                # sell target before peak
    ):
    """
    Returns dominant cycle period, consistency score, average peak/trough,
    buy-in, stop-loss, target price, and reward: risk.
    """

    # ---------------------------------------------------------
    # 1. Detrend the price series for clearer cycle detection
    # ---------------------------------------------------------
    detrended = price_series - price_series.rolling(20, min_periods=1).mean()

    # ---------------------------------------------------------
    # 2. Wavelet transform to detect cycles
    # ---------------------------------------------------------
    coeffs, freqs = pywt.cwt(detrended.values, scales, 'morl')
    power = np.abs(coeffs) ** 2

    # ---------------------------------------------------------
    # 3. Dominant cycle = scale with highest average power
    # ---------------------------------------------------------
    avg_power_per_scale = power.mean(axis=1)
    dominant_scale_idx = np.argmax(avg_power_per_scale)  # Index into scales array
    dominant_scale = scales[dominant_scale_idx]  # Actual scale value
    dominant_period = int(dominant_scale)

    # ---------------------------------------------------------
    # 4. Consistency score: % of time the dominant cycle is strong
    # ---------------------------------------------------------
    dominant_power = power[dominant_scale_idx]  # Use index, not scale value
    threshold = 0.5 * dominant_power.max()
    consistency_score = float(np.sum(dominant_power > threshold) / len(dominant_power))

    # ---------------------------------------------------------
    # 5. Smooth series using half-cycle window to isolate waves
    # ---------------------------------------------------------
    half_period = max(3, dominant_period // 2)
    smoothed = price_series.rolling(half_period, center=True, min_periods=1).mean()

    # ---------------------------------------------------------
    # 6. Detect peaks & troughs
    # ---------------------------------------------------------
    peaks, _ = find_peaks(smoothed)
    troughs, _ = find_peaks(-smoothed)

    peak_vals = smoothed.iloc[peaks].values
    trough_vals = smoothed.iloc[troughs].values

    avg_peak = float(np.mean(peak_vals)) if len(peak_vals) else np.nan
    avg_trough = float(np.mean(trough_vals)) if len(trough_vals) else np.nan
    avg_range = float(avg_peak - avg_trough) if len(peak_vals) and len(trough_vals) else np.nan

    # ---------------------------------------------------------
    # 7. BUY-IN / STOP / TARGET (Fixed % Method)
    # ---------------------------------------------------------
    buy_in = avg_trough + trough_pct * avg_range
    stop_loss = avg_trough - stop_pct * avg_range
    target = avg_peak - peak_pct * avg_range

    # ---------------------------------------------------------
    # 8. Reward:Risk Ratio
    # ---------------------------------------------------------
    reward = target - buy_in
    risk = buy_in - stop_loss
    rr_ratio = reward / risk if risk > 0 else np.nan

    return {
        "dominant_period_days": dominant_period,
        "consistency_score": round(consistency_score, 4),
        "average_peak": avg_peak,
        "average_trough": avg_trough,
        "average_peak_trough_range": avg_range,
        "buy_in": buy_in,
        "stop_loss": stop_loss,
        "target": target,
        "reward_risk_ratio": rr_ratio
    }


# --------------------------------------------------------------
# EXAMPLE USAGE / TEST FUNCTION
# --------------------------------------------------------------
def test_symbol(symbol, lookback_days=180, end_date=None):
    """
    Test the wavelet cycle trade levels function on a stock symbol.
    
    Args:
        symbol: Stock symbol to analyze
        lookback_days: Days of history to analyze
        end_date: End date for analysis (datetime). If None, uses today.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {symbol}")
    print(f"{'='*80}\n")
    
    try:
        ticker = yf.Ticker(symbol)
        
        if end_date:
            start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
        else:
            df = ticker.history(period=f"{lookback_days}d", interval='1d')
        
        if df.empty:
            print(f"❌ No data found for {symbol}")
            return None
        
        price_series = df['Close']
        current_price = float(price_series.iloc[-1])
        
        print(f"Current price: ${current_price:.2f}")
        print(f"Data range: {df.index[0]} to {df.index[-1]}")
        print(f"Data points: {len(df)}\n")
        
        # Run wavelet analysis
        result = wavelet_cycle_trade_levels(
            price_series,
            scales=np.arange(2, 64),
            trough_pct=0.2,  # Buy-in 20% above trough
            stop_pct=0.3,    # Stop 30% below trough
            peak_pct=0.1     # Target 10% below peak
        )
        
        # Print results
        print("Wavelet Cycle Analysis Results:")
        print("-" * 80)
        for key, value in result.items():
            if isinstance(value, float):
                if np.isnan(value):
                    print(f"{key:30s}: N/A")
                else:
                    print(f"{key:30s}: {value:.4f}" if 'score' in key or 'ratio' in key else f"{key:30s}: {value:.2f}")
            else:
                print(f"{key:30s}: {value}")
        
        # Determine if current price is near buy-in level
        print("\n" + "-" * 80)
        buy_in = result['buy_in']
        stop_loss = result['stop_loss']
        target = result['target']
        
        if not np.isnan(buy_in):
            price_to_buy_in = ((current_price - buy_in) / buy_in) * 100
            print(f"\nCurrent price vs Buy-in level:")
            print(f"  Buy-in:        ${buy_in:.2f}")
            print(f"  Current price: ${current_price:.2f}")
            print(f"  Difference:    {price_to_buy_in:+.2f}%")
            
            # Use default tolerance of 2% for single symbol analysis
            tolerance = 2.0
            if abs(price_to_buy_in) <= tolerance:
                print(f"  ✅ Price is within {tolerance}% of buy-in level!")
            elif price_to_buy_in < -tolerance:
                print(f"  ⬇️  Price is {abs(price_to_buy_in):.2f}% below buy-in (not ready to buy)")
            else:
                print(f"  ⬆️  Price is {price_to_buy_in:.2f}% above buy-in (may have missed entry)")
            
            print(f"\nTrade levels:")
            print(f"  Stop loss: ${stop_loss:.2f} ({((stop_loss - buy_in) / buy_in * 100):.1f}% from buy-in)")
            print(f"  Target:   ${target:.2f} ({((target - buy_in) / buy_in * 100):.1f}% from buy-in)")
            if not np.isnan(result['reward_risk_ratio']):
                print(f"  Reward:Risk = {result['reward_risk_ratio']:.2f}:1")
        
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_high_volume_stocks(limit=50, min_price=1.0, historical_date=None):
    """
    Get high-volume stocks based on volume at a specific date.
    
    Args:
        limit: Maximum number of stocks to return
        min_price: Minimum stock price filter
        historical_date: Date to check historical volume (datetime). If None, uses current volume.
    
    Returns:
        List of stock symbols sorted by volume (highest first), or None if no data available (weekend/holiday)
    """
    if historical_date:
        print(f"📊 Fetching {limit} high-volume stocks based on volume at {historical_date.strftime('%Y-%m-%d')}...")
        result = get_historical_high_volume_stocks(limit, min_price, historical_date)
        # get_historical_high_volume_stocks returns None for weekends/holidays
        return result
    else:
        print(f"📊 Fetching {limit} high-volume stocks (current)...")
    
    try:
        most_active_query = YfEquityQuery(
            "and",
            [
                YfEquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
                YfEquityQuery("gt", ["intradayprice", min_price]),
            ]
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
            volume = quote.get('volume') or quote.get('regularMarketVolume') or 0
            price = quote.get('regularMarketPrice') or quote.get('intradayprice', 0)
            
            if symbol and price and price >= min_price:
                stocks.append({
                    'symbol': symbol,
                    'volume': float(volume) if volume else 0.0,
                    'price': float(price),
                })
        
        # Sort by volume (highest first)
        stocks.sort(key=lambda x: x['volume'], reverse=True)
        symbols = [s['symbol'] for s in stocks[:limit]]
        
        print(f"✓ Retrieved {len(symbols)} high-volume stocks")
        return symbols
        
    except Exception as e:
        print(f"⚠️  Screener failed: {e}")
        print("❌ Cannot retrieve stocks - screener unavailable")
        return []


def get_historical_high_volume_stocks(limit=50, min_price=1.0, historical_date=None):
    """
    Get high-volume stocks based on historical volume at a specific date using Polygon.io.
    This is for proper backtesting without look-ahead bias.
    
    Args:
        limit: Maximum number of stocks to return
        min_price: Minimum stock price filter (default: 1.0)
        historical_date: Date to check historical volume (datetime)
    
    Returns:
        List of stock symbols sorted by historical volume (highest first)
    """
    from polygon import RESTClient
    
    # Get API key from environment
    polygon_api_key = os.getenv('POLYGON_API_KEY')
    
    if not polygon_api_key:
        print("❌ POLYGON_API_KEY environment variable not set")
        print("   Please set it with: export POLYGON_API_KEY='your_key_here'")
        sys.exit(1)
    
    print(f"   Fetching historical volume data from Polygon.io for {historical_date.strftime('%Y-%m-%d')}...")
    
    try:
        # Add delay to avoid rate limiting (Polygon has strict rate limits)
        time.sleep(1)  # 1 second delay before API call
        
        client = RESTClient(polygon_api_key)
        
        # Polygon expects date in YYYY-MM-DD format
        trade_date = historical_date.strftime('%Y-%m-%d')
        
        # Retry logic for rate limiting (429 errors)
        max_retries = 3
        retry_delay = 2  # Start with 2 seconds
        aggs = None
        
        for attempt in range(max_retries):
            try:
                # Get all stocks with their volume for that date
                aggs = client.get_grouped_daily_aggs(
                    locale="us",
                    date=trade_date,
                    adjusted=False,
                    include_otc=False
                )
                break  # Success, exit retry loop
            except Exception as api_error:
                error_str = str(api_error).lower()
                # Check if it's a rate limit error (429)
                if '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                        print(f"   ⚠️  Rate limited (429). Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Last attempt failed
                        raise
                else:
                    # Not a rate limit error, re-raise immediately
                    raise
        
        # Check if we got data
        if aggs is None:
            raise Exception("Failed to get data from Polygon API after retries")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "ticker": a.ticker,
            "volume": a.volume,
            "close": a.close,
            "vwap": a.vwap
        } for a in aggs])
        
        if df.empty:
            print(f"⚠️  No data found for {trade_date} (likely weekend/holiday)")
            return None  # Return None to indicate no data - caller will skip to next day
        
        print(f"   Found {len(df)} stocks with data for {trade_date}")
        
        # Filter by minimum price
        df = df[df['close'] >= min_price]
        print(f"   After filtering by min_price >= ${min_price:.2f}: {len(df)} stocks")
        
        # Sort by volume (descending - highest volume first)
        df = df.sort_values('volume', ascending=False)
        
        # Get top N symbols
        symbols = df.head(limit)['ticker'].tolist()
        
        print(f"✓ Retrieved {len(symbols)} high-volume stocks from Polygon.io (sorted by {historical_date.strftime('%Y-%m-%d')} volume)")
        
        # Show top 5 for verification
        if len(df) > 0:
            print(f"   Top 5 by volume:")
            for i, row in df.head(5).iterrows():
                print(f"     {row['ticker']}: Volume={row['volume']:,.0f}, Close=${row['close']:.2f}")
        
        return symbols
        
    except Exception as e:
        print(f"❌ Polygon API failed: {e}")
        import traceback
        traceback.print_exc()
        return None  # Return None to indicate failure - caller can skip to next day


def check_after_days_performance(symbol, buy_price, buy_date, after_days, stop_loss, target_price):
    """
    Check what the price and performance would be after holding for a specific number of days.
    Emulates actual trading: exits early if stop loss or target price is hit.
    
    Args:
        symbol: Stock symbol
        buy_price: Price at which stock was bought
        buy_date: Date when stock was bought (datetime)
        after_days: Number of days to hold (if stop/target not hit)
        stop_loss: Stop loss price (exit immediately if price drops to/below this)
        target_price: Target price (exit immediately if price rises to/above this)
    
    Returns:
        dict with sell_price, return_pct, return_abs, sell_date, exit_reason, days_held
    """
    try:
        # Get price data from buy_date onwards
        end_date = buy_date + timedelta(days=after_days + 5)  # Extra buffer
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=buy_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d'
        )
        
        if df.empty:
            return {
                'sell_price': None, 'return_pct': None, 'return_abs': None, 
                'sell_date': None, 'exit_reason': None, 'days_held': None,
                'error': 'No data'
            }
        
        # Check each day to see if stop loss or target was hit
        buy_idx = None
        for i, idx in enumerate(df.index):
            if hasattr(idx, 'date'):
                idx_date = idx.date()
            elif hasattr(idx, 'to_pydatetime'):
                idx_date = idx.to_pydatetime().date()
            else:
                continue
            
            if idx_date == buy_date.date():
                buy_idx = i
                break
        
        if buy_idx is None:
            # Buy date not in data, start from first available day
            buy_idx = 0
        
        # Check each day from buy_date onwards
        target_end_date = buy_date + timedelta(days=after_days)
        days_held = 0
        exit_reason = 'AFTER_DAYS'  # Default: held for full period
        stop_loss_info = None
        
        for i in range(buy_idx + 1, len(df)):
            idx = df.index[i]
            low_price = float(df['Low'].iloc[i])
            high_price = float(df['High'].iloc[i])
            close_price = float(df['Close'].iloc[i])
            
            # Check if stop loss was hit (use low price for stop loss)
            # FIX: Only trigger stop loss if it's actually below buy price
            if stop_loss < buy_price and low_price <= stop_loss:
                sell_price = stop_loss  # Exit at stop loss price
                days_held = i - buy_idx
                exit_reason = 'STOP_LOSS'
                # Store detailed info about the stop loss trigger
                stop_loss_info = {
                    'date': idx,
                    'low_price': low_price,
                    'high_price': high_price,
                    'close_price': close_price,
                    'stop_loss_level': stop_loss
                }
                break
            
            # Check if target was hit (use high price for target)
            if high_price >= target_price:
                sell_price = target_price  # Exit at target price
                days_held = i - buy_idx
                exit_reason = 'TARGET'
                break
            
            # Check if we've reached the after_days date
            if hasattr(idx, 'date'):
                idx_date = idx.date()
            elif hasattr(idx, 'to_pydatetime'):
                idx_date = idx.to_pydatetime().date()
            else:
                continue
            
            if idx_date >= target_end_date.date():
                sell_price = close_price  # Exit at close price on after_days
                days_held = i - buy_idx
                exit_reason = 'AFTER_DAYS'
                break
        else:
            # Loop completed without break - use last available price
            sell_price = float(df['Close'].iloc[-1])
            days_held = len(df) - buy_idx - 1
            exit_reason = 'END_OF_DATA'
        
        return_pct = ((sell_price - buy_price) / buy_price) * 100
        return_abs = sell_price - buy_price
        
        # Get the actual sell date
        sell_date_idx = buy_idx + days_held if days_held > 0 else buy_idx
        if sell_date_idx < len(df):
            sell_date = df.index[sell_date_idx]
        else:
            sell_date = df.index[-1]
        
        return {
            'sell_price': sell_price,
            'return_pct': return_pct,
            'return_abs': return_abs,
            'sell_date': sell_date,
            'exit_reason': exit_reason,
            'days_held': days_held,
            'error': None,
            'stop_loss_info': stop_loss_info
        }
    except Exception as e:
        return {
            'sell_price': None, 'return_pct': None, 'return_abs': None, 
            'sell_date': None, 'exit_reason': None, 'days_held': None,
            'error': str(e)
        }


def run_backtest(historical_date, limit=50, min_price=1.0, lookback_days=180, 
                 trough_pct=0.2, stop_pct=0.3, peak_pct=0.1, after_days=None, 
                 min_range_pct=0.05, buy_tolerance_pct=2.0, verbose=True):
    """
    Run backtest on high-volume stocks for a specific date.
    
    Args:
        historical_date: Date to analyze (datetime)
        limit: Number of high-volume stocks to test
        min_price: Minimum stock price
        lookback_days: Days of history to analyze
        trough_pct: Buy-in percentage above trough
        stop_pct: Stop loss percentage below trough
        peak_pct: Target percentage below peak
        after_days: If specified, check performance after holding for this many days
        min_range_pct: Minimum average peak-trough range as percentage of average trough (default: 0.05 = 5%)
        buy_tolerance_pct: Tolerance percentage for buy signal - price must be within this % of buy-in (default: 2.0 = 2%)
        verbose: If True, print detailed output (default: True)
    """
    if verbose:
        print("="*80)
        print("WAVELET CYCLE TRADING - BACKTEST MODE")
        print(f"Analysis date: {historical_date.strftime('%Y-%m-%d')}")
        print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
        print("="*80)
        print(f"Settings:")
        print(f"  Stock limit: {limit}")
        print(f"  Lookback period: {lookback_days} days (~{lookback_days//30} months)")
        print(f"  Min price: ${min_price:.2f}")
        print(f"  Buy-in: {trough_pct*100:.0f}% above average trough")
        print(f"  Stop loss: {stop_pct*100:.0f}% below average trough")
        print(f"  Target: {peak_pct*100:.0f}% below average peak")
        print(f"  Min range filter: {min_range_pct*100:.1f}% of average trough (excludes low-volatility stocks)")
        print(f"  Buy tolerance: ±{buy_tolerance_pct:.1f}% from buy-in level")
        if after_days:
            print(f"  Hold period: {after_days} days")
        print()
    
    # Get high-volume stocks
    symbols = get_high_volume_stocks(limit=limit, min_price=min_price, historical_date=historical_date)
    
    if not symbols:
        if verbose:
            print("❌ No stocks retrieved")
        return [], []
    
    if verbose:
        print(f"\n🔍 Analyzing {len(symbols)} stocks for wavelet cycles...")
        print("   (This may take a few minutes)\n")
    
    results = []
    buy_signals = []
    
    for i, symbol in enumerate(symbols, 1):
        if verbose:
            print(f"  [{i}/{len(symbols)}] {symbol}...", end=' ', flush=True)
        
        try:
            ticker = yf.Ticker(symbol)
            start_date = historical_date - timedelta(days=lookback_days + 30)
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=historical_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if df.empty or len(df) < 64:  # Need enough data for wavelets
                if verbose:
                    print("✗ Insufficient data")
                continue
            
            price_series = df['Close']
            current_price = float(price_series.iloc[-1])
            
            # Run wavelet analysis
            result = wavelet_cycle_trade_levels(
                price_series,
                scales=np.arange(2, 64),
                trough_pct=trough_pct,
                stop_pct=stop_pct,
                peak_pct=peak_pct
            )
            
            # Check if valid results
            if np.isnan(result['buy_in']) or np.isnan(result['average_trough']):
                if verbose:
                    print("✗ No valid cycles detected")
                continue
            
            # Filter out stocks with very small price ranges (low volatility)
            avg_trough = result['average_trough']
            avg_range = result['average_peak_trough_range']
            if not np.isnan(avg_trough) and not np.isnan(avg_range) and avg_trough > 0:
                range_pct = (avg_range / avg_trough) * 100
                if range_pct < min_range_pct * 100:
                    if verbose:
                        print(f"✗ Low volatility (range {range_pct:.1f}% < {min_range_pct*100:.1f}%)")
                    continue
            
            # Determine if price is near buy-in level (within tolerance)
            buy_in = result['buy_in']
            price_to_buy_in_pct = ((current_price - buy_in) / buy_in) * 100
            
            result['symbol'] = symbol
            result['current_price'] = current_price
            result['price_to_buy_in_pct'] = price_to_buy_in_pct
            result['is_buy_signal'] = abs(price_to_buy_in_pct) <= buy_tolerance_pct
            
            results.append(result)
            
            if result['is_buy_signal']:
                buy_signals.append(result)
                if verbose:
                    print(f"✓ BUY signal (price ${current_price:.2f} vs buy-in ${buy_in:.2f}, {price_to_buy_in_pct:+.1f}%)")
            else:
                if verbose:
                    if price_to_buy_in_pct < -buy_tolerance_pct:
                        print(f"✗ Below buy-in ({price_to_buy_in_pct:+.1f}%)")
                    else:
                        print(f"✗ Above buy-in ({price_to_buy_in_pct:+.1f}%)")
        
        except Exception as e:
            if verbose:
                print(f"✗ Error: {str(e)[:30]}")
            continue
    
    # Track buy date for all signals (needed for multi-day aggregation)
    for sig in buy_signals:
        sig['buy_date'] = historical_date
    
    # If after_days specified, check performance (emulating stop loss and target)
    if after_days and buy_signals:
        if verbose:
            print(f"\n📊 Checking performance after {after_days} days (with stop loss & target exits)...")
        for sig in buy_signals:
            # Add debug output to verify calculations
            print(f"\n  DEBUG for {sig['symbol']}:")
            print(f"    avg_trough: ${sig['average_trough']:.2f}")
            print(f"    avg_range: ${sig['average_peak_trough_range']:.2f}")
            print(f"    buy_in (calculated): ${sig['buy_in']:.2f}")
            print(f"    current_price (used as buy price): ${sig['current_price']:.2f}")
            print(f"    stop_loss: ${sig['stop_loss']:.2f} (stop_pct={stop_pct})")
            print(f"    target: ${sig['target']:.2f}")
            print(f"    stop_loss distance from buy_in: ${sig['buy_in'] - sig['stop_loss']:.2f} ({((sig['buy_in'] - sig['stop_loss']) / sig['buy_in'] * 100):.2f}%)")
            print(f"    stop_loss distance from current_price: ${sig['current_price'] - sig['stop_loss']:.2f} ({((sig['current_price'] - sig['stop_loss']) / sig['current_price'] * 100):.2f}%)")
            
            # Use buy_in as the buy price for consistency with stop_loss calculation
            # (stop_loss is calculated relative to avg_trough, which buy_in is also based on)
            buy_price_to_use = sig['buy_in']
            print(f"    Using buy_in (${buy_price_to_use:.2f}) as buy price for backtest (vs current_price ${sig['current_price']:.2f})")
            
            perf = check_after_days_performance(
                sig['symbol'],
                buy_price_to_use,
                historical_date,
                after_days,
                sig['stop_loss'],
                sig['target']
            )
            sig['after_days_perf'] = perf
            sig['buy_price_used'] = buy_price_to_use  # Store for display in results
            
            # Show detailed stop loss trigger information if available
            if perf.get('stop_loss_info'):
                sl_info = perf['stop_loss_info']
                sl_date = sl_info['date']
                if hasattr(sl_date, 'strftime'):
                    date_str = sl_date.strftime('%Y-%m-%d')
                elif hasattr(sl_date, 'date'):
                    date_str = sl_date.date().strftime('%Y-%m-%d')
                else:
                    date_str = str(sl_date)
                print(f"\n    STOP LOSS TRIGGERED on {date_str}:")
                print(f"      Stop loss level: ${sl_info['stop_loss_level']:.2f}")
                print(f"      Day's Low: ${sl_info['low_price']:.2f}")
                print(f"      Day's High: ${sl_info['high_price']:.2f}")
                print(f"      Day's Close: ${sl_info['close_price']:.2f}")
                print(f"      Exit price (stop_loss): ${perf['sell_price']:.2f}")
    
    if verbose:
        # Print summary
        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)
        print(f"\nTotal stocks analyzed: {len(results)}")
        print(f"BUY signals found: {len(buy_signals)}")
    
    if verbose and buy_signals:
        if after_days:
            # Show results with after_days performance
            print(f"\n{'Symbol':<8} {'Buy $':<8} {'Sell $':<8} {'Return %':<10} {'Return $':<10} {'Exit':<12} {'Days':<6} {'R:R':<8}")
            print("-"*80)
            for sig in sorted(buy_signals, key=lambda x: x.get('after_days_perf', {}).get('return_pct', -999) or -999, reverse=True):
                perf = sig.get('after_days_perf', {})
                sell_price = perf.get('sell_price', 'N/A')
                return_pct = perf.get('return_pct', None)
                return_abs = perf.get('return_abs', None)
                exit_reason = perf.get('exit_reason', 'N/A')
                days_held = perf.get('days_held', 'N/A')
                
                if return_pct is not None:
                    return_pct_str = f"{return_pct:+.2f}%"
                    return_abs_str = f"${return_abs:+.2f}"
                else:
                    return_pct_str = "N/A"
                    return_abs_str = "N/A"
                
                rr = sig.get('reward_risk_ratio', 0)
                rr_str = f"{rr:.2f}:1" if not np.isnan(rr) else "N/A"
                
                sell_str = f"${sell_price:.2f}" if isinstance(sell_price, (int, float)) else sell_price
                days_str = f"{days_held}d" if isinstance(days_held, int) else str(days_held)
                buy_price_display = sig.get('buy_price_used', sig['current_price'])
                print(f"{sig['symbol']:<8} ${buy_price_display:<7.2f} {sell_str:<8} {return_pct_str:<10} {return_abs_str:<10} "
                      f"{exit_reason:<12} {days_str:<6} {rr_str:<8}")
            
            # Summary statistics
            valid_perfs = [s.get('after_days_perf', {}) for s in buy_signals if s.get('after_days_perf', {}).get('return_pct') is not None]
            if valid_perfs:
                returns = [p['return_pct'] for p in valid_perfs]
                winners = [r for r in returns if r > 0]
                losers = [r for r in returns if r <= 0]
                
                # Count exit reasons
                exit_reasons = {}
                for p in valid_perfs:
                    reason = p.get('exit_reason', 'UNKNOWN')
                    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                
                avg_return = sum(returns) / len(returns)
                win_rate = (len(winners) / len(returns)) * 100 if returns else 0
                
                # Average days held
                days_held_list = [p.get('days_held', 0) for p in valid_perfs if p.get('days_held') is not None]
                avg_days_held = sum(days_held_list) / len(days_held_list) if days_held_list else 0
                
                print(f"\n" + "-"*80)
                print(f"Summary (max {after_days} days holding period):")
                print(f"  Valid results: {len(valid_perfs)}/{len(buy_signals)}")
                print(f"  Winners: {len(winners)} ({len(winners)/len(returns)*100:.1f}%)")
                print(f"  Losers: {len(losers)} ({len(losers)/len(returns)*100:.1f}%)")
                print(f"  Average return: {avg_return:+.2f}%")
                print(f"  Average days held: {avg_days_held:.1f} days")
                if exit_reasons:
                    print(f"  Exit reasons:")
                    for reason, count in exit_reasons.items():
                        print(f"    {reason}: {count}")
                if winners:
                    avg_win = sum(winners) / len(winners)
                    print(f"  Average win: +{avg_win:.2f}%")
                if losers:
                    avg_loss = sum(losers) / len(losers)
                    print(f"  Average loss: {avg_loss:.2f}%")
        else:
            # Show results without after_days performance
            print(f"\n{'Symbol':<8} {'Price':<8} {'Buy-in':<8} {'Stop':<8} {'Target':<8} {'R:R':<8} {'Period':<8} {'Consist':<8}")
            print("-"*80)
            for sig in sorted(buy_signals, key=lambda x: x.get('reward_risk_ratio', 0), reverse=True):
                rr = sig.get('reward_risk_ratio', 0)
                rr_str = f"{rr:.2f}:1" if not np.isnan(rr) else "N/A"
                print(f"{sig['symbol']:<8} ${sig['current_price']:<7.2f} ${sig['buy_in']:<7.2f} "
                      f"${sig['stop_loss']:<7.2f} ${sig['target']:<7.2f} {rr_str:<8} "
                      f"{sig['dominant_period_days']:<8} {sig['consistency_score']:<8.3f}")
    
    return results, buy_signals


def print_aggregated_summary(all_results, all_buy_signals, after_days, num_days):
    """
    Print aggregated summary across multiple days of backtesting.
    
    Args:
        all_results: List of all results from all days
        all_buy_signals: List of all buy signals from all days
        after_days: Holding period for performance calculation
        num_days: Number of days tested
    """
    print("\n" + "="*80)
    print("AGGREGATED BACKTEST RESULTS")
    print("="*80)
    print(f"\nTest period: {num_days} days")
    print(f"Total stocks analyzed: {len(all_results)}")
    print(f"Total BUY signals found: {len(all_buy_signals)}")
    
    if not all_buy_signals:
        print("\n❌ No BUY signals found across all days")
        return
    
    # Deduplicate: keep only first occurrence (earliest buy date) of each symbol
    seen_symbols = {}
    deduplicated_signals = []
    
    # Sort by buy_date to ensure we process earliest first
    sorted_signals = sorted(all_buy_signals, key=lambda x: (x.get('buy_date') or datetime.max, x.get('symbol', '')))
    
    for sig in sorted_signals:
        symbol = sig.get('symbol')
        if symbol and symbol not in seen_symbols:
            seen_symbols[symbol] = True
            deduplicated_signals.append(sig)
    
    print(f"  (After deduplication: {len(deduplicated_signals)} unique symbols)")
    
    # Count signals per day (using deduplicated list)
    signals_per_day = {}
    for sig in deduplicated_signals:
        buy_date = sig.get('buy_date')
        if buy_date:
            date_str = buy_date.strftime('%Y-%m-%d')
            signals_per_day[date_str] = signals_per_day.get(date_str, 0) + 1
    
    print(f"\nSignals per day:")
    for date_str in sorted(signals_per_day.keys()):
        print(f"  {date_str}: {signals_per_day[date_str]} signals")
    
    if after_days:
        # Show aggregated results with after_days performance (using deduplicated signals)
        print(f"\n{'Symbol':<8} {'Buy Date':<12} {'Buy $':<8} {'Sell $':<8} {'Return %':<10} {'Return $':<10} {'Exit':<12} {'Days':<6} {'R:R':<8}")
        print("-"*90)
        for sig in sorted(deduplicated_signals, key=lambda x: x.get('after_days_perf', {}).get('return_pct', -999) or -999, reverse=True):
            perf = sig.get('after_days_perf', {})
            sell_price = perf.get('sell_price', 'N/A')
            return_pct = perf.get('return_pct', None)
            return_abs = perf.get('return_abs', None)
            exit_reason = perf.get('exit_reason', 'N/A')
            days_held = perf.get('days_held', 'N/A')
            buy_date = sig.get('buy_date')
            buy_date_str = buy_date.strftime('%Y-%m-%d') if buy_date else 'N/A'
            
            if return_pct is not None:
                return_pct_str = f"{return_pct:+.2f}%"
                return_abs_str = f"${return_abs:+.2f}"
            else:
                return_pct_str = "N/A"
                return_abs_str = "N/A"
            
            rr = sig.get('reward_risk_ratio', 0)
            rr_str = f"{rr:.2f}:1" if not np.isnan(rr) else "N/A"
            
            sell_str = f"${sell_price:.2f}" if isinstance(sell_price, (int, float)) else sell_price
            days_str = f"{days_held}d" if isinstance(days_held, int) else str(days_held)
            buy_price_display = sig.get('buy_price_used', sig['current_price'])
            print(f"{sig['symbol']:<8} {buy_date_str:<12} ${buy_price_display:<7.2f} {sell_str:<8} {return_pct_str:<10} {return_abs_str:<10} "
                  f"{exit_reason:<12} {days_str:<6} {rr_str:<8}")
        
        # Aggregated summary statistics (using deduplicated signals)
        valid_perfs = [s.get('after_days_perf', {}) for s in deduplicated_signals if s.get('after_days_perf', {}).get('return_pct') is not None]
        if valid_perfs:
            returns = [p['return_pct'] for p in valid_perfs]
            winners = [r for r in returns if r > 0]
            losers = [r for r in returns if r <= 0]
            
            # Count exit reasons
            exit_reasons = {}
            for p in valid_perfs:
                reason = p.get('exit_reason', 'UNKNOWN')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            avg_return = sum(returns) / len(returns)
            win_rate = (len(winners) / len(returns)) * 100 if returns else 0
            
            # Average days held
            days_held_list = [p.get('days_held', 0) for p in valid_perfs if p.get('days_held') is not None]
            avg_days_held = sum(days_held_list) / len(days_held_list) if days_held_list else 0
            
            print(f"\n" + "-"*90)
            print(f"Aggregated Summary (max {after_days} days holding period):")
            print(f"  Valid results: {len(valid_perfs)}/{len(deduplicated_signals)}")
            print(f"  Winners: {len(winners)} ({len(winners)/len(returns)*100:.1f}%)")
            print(f"  Losers: {len(losers)} ({len(losers)/len(returns)*100:.1f}%)")
            print(f"  Average return: {avg_return:+.2f}%")
            print(f"  Average days held: {avg_days_held:.1f} days")
            if exit_reasons:
                print(f"  Exit reasons:")
                for reason, count in exit_reasons.items():
                    print(f"    {reason}: {count}")
            if winners:
                avg_win = sum(winners) / len(winners)
                print(f"  Average win: +{avg_win:.2f}%")
            if losers:
                avg_loss = sum(losers) / len(losers)
                print(f"  Average loss: {avg_loss:.2f}%")
            
            # Calculate cumulative gain/loss (using deduplicated signals)
            total_return_abs = sum([p.get('return_abs', 0) for p in valid_perfs if p.get('return_abs') is not None])
            # Get buy prices for trades with valid performance data
            total_invested = sum([sig.get('buy_price_used', sig['current_price']) for sig in deduplicated_signals if sig.get('after_days_perf', {}).get('return_pct') is not None])
            
            if total_invested > 0:
                cumulative_return_pct = (total_return_abs / total_invested) * 100
                print(f"\n  Cumulative Performance:")
                print(f"    Total invested: ${total_invested:.2f}")
                print(f"    Total P&L: ${total_return_abs:+.2f}")
                print(f"    Cumulative return: {cumulative_return_pct:+.2f}%")
    else:
        # Show aggregated results without after_days performance (using deduplicated signals)
        print(f"\n{'Symbol':<8} {'Buy Date':<12} {'Price':<8} {'Buy-in':<8} {'Stop':<8} {'Target':<8} {'R:R':<8} {'Period':<8} {'Consist':<8}")
        print("-"*90)
        for sig in sorted(deduplicated_signals, key=lambda x: x.get('reward_risk_ratio', 0), reverse=True):
            rr = sig.get('reward_risk_ratio', 0)
            rr_str = f"{rr:.2f}:1" if not np.isnan(rr) else "N/A"
            buy_date = sig.get('buy_date')
            buy_date_str = buy_date.strftime('%Y-%m-%d') if buy_date else 'N/A'
            print(f"{sig['symbol']:<8} {buy_date_str:<12} ${sig['current_price']:<7.2f} ${sig['buy_in']:<7.2f} "
                  f"${sig['stop_loss']:<7.2f} ${sig['target']:<7.2f} {rr_str:<8} "
                  f"{sig['dominant_period_days']:<8} {sig['consistency_score']:<8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wavelet Cycle Trading Levels - Original Implementation')
    parser.add_argument('symbol', nargs='?', help='Stock symbol to analyze (default: CLOV)')
    parser.add_argument('--date', type=str, help='Analysis date (YYYY-MM-DD). For backtest mode.')
    parser.add_argument('--backtest', action='store_true', help='Enable backtest mode: test high-volume stocks on date')
    parser.add_argument('--limit', type=int, default=50, help='Number of high-volume stocks to analyze in backtest (default: 50)')
    parser.add_argument('--min-price', type=float, default=1.0, help='Minimum stock price filter (default: 1.0)')
    parser.add_argument('--lookback', type=int, default=180, help='Days of history to analyze (default: 180)')
    parser.add_argument('--trough-pct', type=float, default=0.2, help='Buy-in percentage above trough (default: 0.2 = 20%%)')
    parser.add_argument('--stop-pct', type=float, default=0.3, help='Stop loss percentage below trough (default: 0.3 = 30%%)')
    parser.add_argument('--peak-pct', type=float, default=0.1, help='Target percentage below peak (default: 0.1 = 10%%)')
    parser.add_argument('--min-range-pct', type=float, default=0.05, help='Minimum average peak-trough range as percentage of average trough to include stock (default: 0.05 = 5%%, filters out low-volatility stocks)')
    parser.add_argument('--buy-tolerance-pct', type=float, default=2.0, help='Tolerance percentage for buy signal - price must be within this %% of buy-in level (default: 2.0 = 2%%)')
    parser.add_argument('--after-days', type=int, default=None, help='Check performance after holding for this many days (default: None)')
    parser.add_argument('--days', type=int, default=1, help='Number of consecutive days to test (default: 1, for multi-day backtesting)')
    
    args = parser.parse_args()
    
    # Parse date if provided
    end_date = None
    if args.date:
        try:
            end_date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print(f"❌ Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    
    # Run backtest mode or single symbol analysis
    if args.backtest:
        if not end_date:
            print("❌ --backtest requires --date")
            sys.exit(1)
        
        if args.days > 1:
            # Multi-day backtest
            print("="*80)
            print("MULTI-DAY BACKTEST MODE")
            print(f"Testing {args.days} consecutive days starting from {end_date.strftime('%Y-%m-%d')}")
            print("="*80)
            
            all_results = []
            all_buy_signals = []
            
            days_tested = 0
            day_offset = 0
            max_attempts = args.days * 2  # Allow for weekends/holidays
            
            while days_tested < args.days and day_offset < max_attempts:
                test_date = end_date + timedelta(days=day_offset)
                day_of_week = test_date.weekday()  # 0=Monday, 6=Sunday
                
                print(f"\n{'='*80}")
                print(f"DAY {days_tested + 1}/{args.days}: {test_date.strftime('%Y-%m-%d')} ({test_date.strftime('%A')})")
                print(f"{'='*80}")
                
                # Add delay before API call to avoid rate limiting (Polygon has strict limits)
                if day_offset > 0:
                    print("   ⏳ Waiting 2 seconds to avoid rate limiting...")
                    time.sleep(2)
                else:
                    # First call also gets a delay
                    time.sleep(1)
                
                # Try to get stocks for this date
                symbols = get_high_volume_stocks(limit=args.limit, min_price=args.min_price, historical_date=test_date)
                
                # If no data (weekend/holiday), skip to next day
                if symbols is None or len(symbols) == 0:
                    print(f"   ⏭️  Skipping {test_date.strftime('%Y-%m-%d')} (no market data - weekend/holiday)")
                    day_offset += 1
                    continue
                
                # Run backtest for this date
                daily_results, daily_buy_signals = run_backtest(
                    historical_date=test_date,
                    limit=args.limit,
                    min_price=args.min_price,
                    lookback_days=args.lookback,
                    trough_pct=args.trough_pct,
                    stop_pct=args.stop_pct,
                    peak_pct=args.peak_pct,
                    after_days=args.after_days,
                    min_range_pct=args.min_range_pct,
                    buy_tolerance_pct=args.buy_tolerance_pct,
                    verbose=True  # Show per-day output
                )
                
                all_results.extend(daily_results)
                all_buy_signals.extend(daily_buy_signals)
                days_tested += 1
                day_offset += 1
            
            # Print aggregated summary across all days
            print(f"\n{'='*80}")
            print(f"Completed testing {days_tested} trading days (skipped weekends/holidays)")
            print(f"{'='*80}")
            print_aggregated_summary(all_results, all_buy_signals, args.after_days, days_tested)
        else:
            # Single day backtest (existing behavior)
            run_backtest(
                historical_date=end_date,
                limit=args.limit,
                min_price=args.min_price,
                lookback_days=args.lookback,
                trough_pct=args.trough_pct,
                stop_pct=args.stop_pct,
                peak_pct=args.peak_pct,
                after_days=args.after_days,
                min_range_pct=args.min_range_pct,
                buy_tolerance_pct=args.buy_tolerance_pct,
                verbose=True
            )
    else:
        # Single symbol analysis
        symbol = args.symbol.upper() if args.symbol else "CLOV"
        test_symbol(symbol, lookback_days=args.lookback, end_date=end_date)

