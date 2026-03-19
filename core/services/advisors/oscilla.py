
import logging
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import pywt
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from decimal import Decimal

from core.services.advisors.advisor import AdvisorBase, register

logger = logging.getLogger(__name__)

# Suppress yfinance warnings and logging for missing historical data
warnings.filterwarnings("ignore", message=".*possibly delisted.*")
warnings.filterwarnings("ignore", message=".*no timezone found.*")
warnings.filterwarnings("ignore", message=".*no price data found.*")
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")

# Suppress yfinance internal logger messages (set to CRITICAL to suppress ERROR level messages)
yfinance_logger = logging.getLogger("yfinance")
yfinance_logger.setLevel(logging.CRITICAL)
yfinance_logger.disabled = True  # Completely disable yfinance logger
history_logger = logging.getLogger("history")
history_logger.setLevel(logging.CRITICAL)
history_logger.disabled = True  # Completely disable history logger
quote_logger = logging.getLogger("quote")
quote_logger.setLevel(logging.CRITICAL)
quote_logger.disabled = True  # Completely disable quote logger (HTTP 404 errors)

# ------------------------------
# OSCILLA CONFIGURATION
# ------------------------------
MIN_PRICE = 8
MAX_PRICE = 80
MIN_VOLUME = 1_000_000  # Minimum volume filter for initial fetch
MIN_AVG_VOLUME = 2_000_000
REL_VOLUME_MIN = 0.7  # Relaxed from 0.8 to allow stocks with slightly below-average volume (e.g., NVDL recovery cases)
REL_VOLUME_MAX = 1.3
LOOKBACK_DAYS = 40  # Days of history needed for wavelet analysis (64 trading days minimum)
MIN_RR = 1.8  # Minimum reward:risk ratio
TURN_CONFIRMATION_ENABLED = True  # Require turn confirmation (higher close + higher low) before entry
# MIN_STOP_BUFFER_PCT = 0.10  # Minimum stop distance from entry (0 = disabled, use calculated stop directly)
MAX_WAVE_POSITION = -999  # Maximum (most negative) wave_position to accept (filters strong downtrends) - DISABLED
MIN_CONSISTENCY = 0.0  # Minimum consistency score (filters inconsistent wave patterns) - DISABLED
MAX_STOCKS = 1000  # Maximum number of stocks to process (for testing/comparison with test script, None = unlimited)


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
            logger.debug(f"Error fetching current price for {ticker}: {e}")
        
        return df.reset_index(drop=True)
    except Exception as e:
        logger.debug(f"Error fetching historical data for {ticker}: {e}")
        return pd.DataFrame()


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
    
    # Note: Internal wave_position filtering removed - filtering happens in discover() with range (-50.0 to 0.0)
    
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
    target = buy + 1.0 * (avg_peak - buy)  # Aim for full avg_peak (diminishing target provides protection)
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


class Oscilla(AdvisorBase):
    """
    Oscilla Trading Advisor - Intra-week wave-based trading strategy.
    
    Uses wavelet analysis to detect cyclical patterns and identify optimal entry points.
    """

    def discover(self, sa):
        """
        Discover stocks using wavelet-based trading strategy.
        
        Uses Polygon to get candidate stocks, then applies wavelet analysis to identify
        optimal entry points with calculated stop-loss and target prices.
        
        Args:
            sa: SmartAnalysis session
        """
        # Once-per-day guard: if there was a previous SA today, skip entirely
        prev = self.get_previous_sa_timestamp(sa, username=getattr(sa, "username", None))
        if prev and prev.date() == sa.started.date():
            logger.info("Oscilla: already ran for this user today, skipping.")
            return

        try:
            # Get reference date (last trading day) - calculate FIRST
            last_trading_date = AdvisorBase.get_last_trading_day()
            
            if not last_trading_date:
                logger.warning("Oscilla: No valid trading date available")
                return
            
            logger.info(f"Oscilla: Using reference date: {last_trading_date}")
            
            # Get filtered stocks from Polygon (pass test_date explicitly)
            df_stocks = AdvisorBase.get_filtered_stocks(
                sa=sa,
                min_price=MIN_PRICE,
                max_price=MAX_PRICE,
                min_volume=MIN_VOLUME,
                test_date=last_trading_date  # Pass explicitly to ensure consistency
            )
            
            if df_stocks.empty:
                logger.info("Oscilla: No stocks found from Polygon")
                return
            
            # Limit to MAX_STOCKS if set (for testing/comparison)
            if MAX_STOCKS and len(df_stocks) > MAX_STOCKS:
                df_stocks = df_stocks.head(MAX_STOCKS)
                logger.info(f"Oscilla: Limited to {MAX_STOCKS} stocks for analysis")
            
            logger.info(f"Oscilla: Analyzing {len(df_stocks)} stocks with wavelet analysis...")
            
            # For wavelet analysis, we need at least 64 trading days
            # Account for weekends/holidays: ~5 trading days per 7 calendar days
            # So 64 trading days ≈ 64 * 7/5 ≈ 90 calendar days, add buffer for safety
            ref_dt = datetime.strptime(last_trading_date, "%Y-%m-%d")
            min_calendar_days = int(64 * 7 / 5) + 30  # ~120 calendar days for 64 trading days
            wavelet_lookback_days = max(min_calendar_days, LOOKBACK_DAYS * 2)
            wavelet_start_date = (ref_dt - timedelta(days=wavelet_lookback_days)).strftime("%Y-%m-%d")
            
            # Now run wavelet analysis directly on stocks from Polygon (matching test_crazy.py)
            discoveries = 0
            filtered_no_data = 0
            filtered_insufficient_data = 0
            filtered_rel_volume = 0
            filtered_wavelet_rejected = 0
            filtered_wave_position = 0
            filtered_not_us = 0
            filtered_not_profitable = 0
            filtered_error = 0
            
            for i, (_, row) in enumerate(df_stocks.iterrows(), 1):
                ticker = row["ticker"]
                
                if i % 50 == 0:
                    logger.info(f"Oscilla: Wavelet analysis {i}/{len(df_stocks)} stocks...")
                
                try:
                    # Get historical data for wavelet analysis
                    df_price = get_historical_data_yfinance(ticker, wavelet_start_date, last_trading_date)
                    if df_price.empty:
                        filtered_no_data += 1
                        continue
                    
                    # Check if we have enough data points (need at least 64)
                    if len(df_price) < 64:
                        filtered_insufficient_data += 1
                        continue
                    
                    # Volume filtering: check relative volume (middle-of-the-road stocks)
                    # MIN_AVG_VOLUME check removed (too restrictive) - only using relative volume range
                    if 'volume' in df_price.columns and len(df_price) >= LOOKBACK_DAYS:
                        # Calculate average volume over lookback period
                        df_price_sorted = df_price.sort_values("date").tail(LOOKBACK_DAYS)
                        avg_volume = df_price_sorted["volume"].mean()
                        
                        # Get today's volume from Polygon data
                        today_volume = row.get("today_volume", 0)
                        if today_volume > 0 and avg_volume > 0:
                            # Calculate relative volume (today vs average)
                            rel_volume = today_volume / avg_volume
                            
                            # Filter: only accept stocks with relative volume in range [REL_VOLUME_MIN, REL_VOLUME_MAX]
                            # This finds "middle-of-the-road" stocks with stable volume patterns
                            if rel_volume < REL_VOLUME_MIN or rel_volume > REL_VOLUME_MAX:
                                filtered_rel_volume += 1
                                continue
                    
                    # Run wavelet analysis
                    wave_result = wavelet_trade_engine(
                        df_price["close"],
                        min_rr=MIN_RR,
                        low_series=df_price["low"]
                    )
                    
                    if not wave_result.get("accepted", False):
                        filtered_wavelet_rejected += 1
                        continue
                    
                    # Apply additional filters after wavelet passes
                    wave_position = wave_result.get("wave_position", 0)
                    
                    # Filter wave_position: only accept between -50.0 and 0.0
                    if wave_position < -50.0 or wave_position > 0.0:
                        filtered_wave_position += 1
                        continue
                    
                    # Verify US exchange and profitability
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        info = ticker_obj.info
                        
                        # Verify US exchange (NMS, NYQ, NAS, NYS)
                        exchange = info.get('exchange')
                        us_exchanges = ['NMS', 'NYQ', 'NAS', 'NYS']
                        if exchange not in us_exchanges:
                            filtered_not_us += 1
                            continue
                        
                        # Check profitability (at least one metric must be >= 0)
                        trailing_eps = info.get('trailingEps')
                        net_income = info.get('netIncomeToCommon') or info.get('netIncome')
                        profit_margin = info.get('profitMargins')
                        
                        is_profitable = (
                            (trailing_eps is not None and trailing_eps >= 0) or
                            (net_income is not None and net_income >= 0) or
                            (profit_margin is not None and profit_margin >= 0)
                        )
                        
                        if not is_profitable:
                            filtered_not_profitable += 1
                            continue
                            
                    except Exception as e:
                        filtered_error += 1
                        logger.debug(f"Oscilla: Error checking exchange/profitability for {ticker}: {e}")
                        continue
                    
                    # Create discovery with sell instructions
                    explanation = (
                        f"Wavelet pattern detected: period={wave_result['dominant_period_days']}d | "
                        f"R:R={wave_result['reward_risk']:.2f} | "
                        f"consistency={wave_result['consistency']:.3f} | "
                        f"wave_pos={wave_result['wave_position']:.3f}"
                    )
                    
                    # Create sell instructions: PROFIT_TARGET, PERCENTAGE_REBUY, and PROFIT_FLAT
                    # PROFIT_TARGET: val1=ratio (e.g., 0.10 for 10% profit on average spend)
                    # PERCENTAGE_REBUY: val1=drop %, val2=rebuy % (10% loss, rebuy 10% of cost basis)
                    # PROFIT_FLAT: val1=range threshold %, val2=evaluation days
                    sell_instructions = [
                        ("PROFIT_TARGET", Decimal('0.10'), None),  # 10% profit on average spend
                        ("PERCENTAGE_REBUY", Decimal('0.10'), Decimal('0.20')),  # 10% loss, rebuy 20%
                        ("PROFIT_FLAT", Decimal('0.02'), Decimal('3')),  # Sell if price range within 2% over 3 days
                        ("PROFIT_FLAT", Decimal('0.05'), Decimal('20')),  # Sell if price range within 5% over 20 days
                    ]

                    logger.info(f"Submitting {ticker} - R:R={wave_result['reward_risk']:.2f}, stop=${wave_result['stop']:.2f}, target=${wave_result['target']:.2f}, wave_pos={wave_position:.3f}")

                    # Create discovery
                    if self.discovered(sa=sa, symbol=ticker, explanation=explanation, sell_instructions=sell_instructions, weight=1.0) is not None:
                        discoveries += 1

                except Exception as e:
                    filtered_error += 1
                    logger.debug(f"Oscilla: Error analyzing {ticker}: {e}", exc_info=True)
                    continue
            
            logger.info(
                f"Oscilla: Discovery complete - {discoveries} candidates. "
                f"Filtered: no_data={filtered_no_data}, insufficient_data={filtered_insufficient_data}, "
                f"rel_volume={filtered_rel_volume}, wavelet_rejected={filtered_wavelet_rejected}, "
                f"wave_position={filtered_wave_position}, not_us={filtered_not_us}, "
                f"not_profitable={filtered_not_profitable}, errors={filtered_error}"
            )
            
        except Exception as e:
            logger.error(f"Oscilla: Error in discovery: {e}", exc_info=True)


register(name="Oscilla", python_class="Oscilla")


