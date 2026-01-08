#!/usr/bin/env python
"""
Diagnostic script to analyze NTLA discovery on Jan 6, 2026
Checks smoothed slope, multi-day momentum, and what is_trough_entry() would say
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime, timedelta
import pywt
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", message=".*possibly delisted.*")
warnings.filterwarnings("ignore", message=".*no timezone found.*")
warnings.filterwarnings("ignore", message=".*no price data found.*")
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")

# Import from oscilla module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Minimal imports from oscilla
def get_historical_data_yfinance(ticker, start_date, end_date):
    """Get historical data using yfinance"""
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        # If end_date is in future, use today
        today = datetime.now().date()
        if end_dt.date() > today:
            end_dt = datetime.combine(today, datetime.min.time())
        end_date_exclusive = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        
        ticker_obj = yf.Ticker(ticker)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*possibly delisted.*")
            warnings.filterwarnings("ignore", message=".*no timezone found.*")
            warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")
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
        
        try:
            if len(df) > 0:
                info = ticker_obj.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if current_price and current_price > 0:
                    df.iloc[-1, df.columns.get_loc('close')] = current_price
                    current_low = info.get('dayLow') or current_price
                    current_high = info.get('dayHigh') or current_price
                    df.iloc[-1, df.columns.get_loc('low')] = current_low
                    df.iloc[-1, df.columns.get_loc('high')] = current_high
                    current_volume = info.get('volume')
                    if current_volume:
                        df.iloc[-1, df.columns.get_loc('volume')] = current_volume
        except Exception:
            pass
        
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()


def wavelet_trade_engine(price_series, min_rr=1.8, low_series=None, turn_confirmation_enabled=True):
    """Analyze price series using wavelet analysis"""
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
    dominant_scale = scales[dominant_scale_idx]
    dominant_period = int(dominant_scale)
    dominant_power = power[dominant_scale_idx]
    consistency = float(np.sum(dominant_power > 0.5 * dominant_power.max()) / len(dominant_power))
    
    log.append(f"Dominant period: {dominant_period} days, consistency: {consistency:.3f}")
    
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
    
    # Turn Confirmation Check
    if turn_confirmation_enabled and low_series is not None:
        if len(price_series) >= 2 and len(low_series) >= 2:
            current_close = float(price_series.iloc[-1])
            prev_close = float(price_series.iloc[-2])
            current_low = float(low_series.iloc[-1])
            prev_low = float(low_series.iloc[-2])
            
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
    
    if calculated_stop >= buy:
        log.append(f"Rejected: Invalid stop calculation (stop ${calculated_stop:.2f} >= buy ${buy:.2f})")
        return {"accepted": False, "reason": "Invalid stop calculation", "log": log}
    
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
        "half_period": half_period,
        "consistency": round(consistency, 3),
        "wave_position": round(wave_position, 3),
        "buy": round(buy, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "reward_risk": round(rr, 2),
        "avg_trough": round(avg_trough, 2),
        "avg_peak": round(avg_peak, 2),
        "wave_range": round(wave_range, 2),
        "log": log
    }


def is_trough_entry(
    price_series: pd.Series,
    avg_trough: float,
    wave_range: float,
    half_period: int,
    max_wave_position: float = 0.35,
    min_up_days: int = 2,
    min_slope: float = 0.0,
):
    """
    Determines whether the current price action represents a valid
    'coming out of trough' entry for Oscilla.

    Returns:
        (bool, dict): (is_valid_entry, diagnostics)
    """
    if len(price_series) < max(half_period, min_up_days + 2):
        return False, {"reason": "insufficient_data"}

    price = price_series.values

    # --- 1️⃣ Wave position (hard gate) ---
    wave_position = (price[-1] - avg_trough) / max(wave_range, 1e-6)

    if wave_position > max_wave_position:
        return False, {
            "reason": "too_late_in_wave",
            "wave_position": round(wave_position, 3)
        }

    # --- 2️⃣ Smoothed slope (directional confirmation) ---
    smoothed = (
        pd.Series(price)
        .rolling(window=half_period, min_periods=3)
        .mean()
    )

    if len(smoothed) < 3:
        return False, {"reason": "insufficient_smoothed_data"}

    slope = smoothed.iloc[-1] - smoothed.iloc[-3]

    if slope <= min_slope:
        return False, {
            "reason": "no_upward_slope",
            "slope": round(float(slope), 5)
        }

    # --- 3️⃣ Low-lag price structure (least lag check) ---
    up_days = 0
    for i in range(1, min_up_days + 1):
        if len(price) > i and price[-i] > price[-i - 1]:
            up_days += 1

    if up_days < min_up_days:
        return False, {
            "reason": "no_price_turn",
            "up_days": up_days
        }

    # --- ✅ Valid trough entry ---
    return True, {
        "wave_position": round(wave_position, 3),
        "slope": round(float(slope), 5),
        "up_days": up_days
    }


def diagnose_ntla():
    ticker = "NTLA"
    discovery_date = "2026-01-06"
    
    print(f"\n{'='*60}")
    print(f"NTLA Discovery Diagnostic - {discovery_date}")
    print(f"{'='*60}\n")
    
    # Get historical data (same as discovery would use)
    ref_dt = datetime.strptime(discovery_date, "%Y-%m-%d")
    min_calendar_days = int(64 * 7 / 5) + 30  # ~120 calendar days
    wavelet_start_date = (ref_dt - timedelta(days=min_calendar_days)).strftime("%Y-%m-%d")
    
    print(f"Fetching historical data from {wavelet_start_date} to {discovery_date}...")
    df_price = get_historical_data_yfinance(ticker, wavelet_start_date, discovery_date)
    
    if df_price.empty:
        print(f"ERROR: No historical data found for {ticker}")
        return
    
    print(f"Got {len(df_price)} data points\n")
    
    # Show recent price action
    print("Recent Price Action (last 10 days):")
    print("-" * 60)
    recent = df_price.tail(10)[['date', 'close', 'high', 'low', 'volume']]
    for _, row in recent.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: Close=${row['close']:.2f}, "
              f"High=${row['high']:.2f}, Low=${row['low']:.2f}, Vol={row['volume']:,.0f}")
    
    print("\n" + "="*60)
    print("Wavelet Analysis Results (as discovered)")
    print("="*60 + "\n")
    
    # Run the same wavelet analysis
    wave_result = wavelet_trade_engine(
        df_price["close"],
        min_rr=1.8,
        low_series=df_price["low"]
    )
    
    print("Wave Result:")
    print(f"  Accepted: {wave_result.get('accepted', False)}")
    if wave_result.get('accepted'):
        print(f"  Dominant Period: {wave_result.get('dominant_period_days')} days")
        print(f"  Half Period: {wave_result.get('half_period')} days")
        print(f"  Wave Position: {wave_result.get('wave_position')}")
        print(f"  Consistency: {wave_result.get('consistency')}")
        print(f"  R:R Ratio: {wave_result.get('reward_risk')}")
        print(f"  Buy: ${wave_result.get('buy')}")
        print(f"  Stop: ${wave_result.get('stop')}")
        print(f"  Target: ${wave_result.get('target')}")
        print(f"  Avg Trough: ${wave_result.get('avg_trough')}")
        print(f"  Avg Peak: ${wave_result.get('avg_peak')}")
        print(f"  Wave Range: ${wave_result.get('wave_range')}")
    
    print("\nFull Log:")
    for log_entry in wave_result.get('log', []):
        print(f"  {log_entry}")
    
    # Now check what is_trough_entry() would say
    print("\n" + "="*60)
    print("is_trough_entry() Diagnostic")
    print("="*60 + "\n")
    
    if wave_result.get('accepted'):
        avg_trough = wave_result.get('avg_trough')
        wave_range = wave_result.get('wave_range')
        half_period = wave_result.get('half_period')
        
        # Check trough entry
        is_valid, diagnostics = is_trough_entry(
            price_series=df_price["close"],
            avg_trough=avg_trough,
            wave_range=wave_range,
            half_period=half_period,
            max_wave_position=0.35,
            min_up_days=2,
            min_slope=0.0,
        )
        
        print(f"Would is_trough_entry() accept? {is_valid}")
        print(f"\nDiagnostics:")
        for key, value in diagnostics.items():
            print(f"  {key}: {value}")
        
        if not is_valid:
            print(f"\n❌ REJECTED: {diagnostics.get('reason')}")
            print("\nThis explains why it might still be a crest entry!")
        else:
            print(f"\n✅ WOULD ACCEPT")
        
        # Additional diagnostics
        print("\n" + "="*60)
        print("Additional Price Action Diagnostics")
        print("="*60 + "\n")
        
        price_values = df_price["close"].values
        recent_prices = price_values[-5:]
        
        print("Last 5 closes:")
        for i, price in enumerate(recent_prices):
            days_ago = len(recent_prices) - 1 - i
            change = ""
            if i > 0:
                pct_change = ((price - recent_prices[i-1]) / recent_prices[i-1]) * 100
                change = f" ({pct_change:+.2f}%)"
            print(f"  {days_ago} days ago: ${price:.2f}{change}")
        
        # Check consecutive up days
        consecutive_up = 0
        for i in range(len(price_values) - 1, 0, -1):
            if price_values[i] > price_values[i-1]:
                consecutive_up += 1
            else:
                break
        
        print(f"\nConsecutive up days: {consecutive_up}")
        if consecutive_up < 2:
            print("  ⚠️  Less than 2 consecutive up days - weak momentum!")
        
        # Calculate smoothed slope manually
        smoothed = df_price["close"].rolling(window=half_period, min_periods=3).mean()
        if len(smoothed) >= 3:
            slope = smoothed.iloc[-1] - smoothed.iloc[-3]
            print(f"\nSmoothed slope (current - 3 days ago): {slope:.5f}")
            if slope <= 0:
                print("  ⚠️  Negative or flat slope - wave not turning up!")
            else:
                print("  ✅ Positive slope")
        
        # Show where price is relative to recent range
        recent_10_days = price_values[-10:]
        recent_high = max(recent_10_days)
        recent_low = min(recent_10_days)
        recent_range = recent_high - recent_low
        current_price = price_values[-1]
        position_in_recent_range = (current_price - recent_low) / recent_range if recent_range > 0 else 0
        
        print(f"\nRecent 10-day range: ${recent_low:.2f} - ${recent_high:.2f}")
        print(f"Current price: ${current_price:.2f}")
        print(f"Position in recent range: {position_in_recent_range:.1%}")
        if position_in_recent_range > 0.7:
            print("  ⚠️  Near top of recent range - possible crest!")
        elif position_in_recent_range < 0.3:
            print("  ✅ Near bottom of recent range - possible trough")
        
    else:
        print("Wave analysis was not accepted - cannot check is_trough_entry()")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    diagnose_ntla()

