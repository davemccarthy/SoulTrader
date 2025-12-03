"""
Intraday Momentum Trading Strategy - Discovery, Exit Tracking, and Evaluation

This script supports three modes for evaluating intraday momentum trading:

1. DISCOVERY MODE
   Scan active stocks for current entry signals based on:
   - Price > VWAP (intraday value)
   - Short EMA > Long EMA (momentum)
   - RSI < 75 (not overbought)
   - Volume > 1.5x average (surge)
   
   Usage: python test_intraday_trade.py discovery --max-stocks 100
   
   Trading Hours: By default, blocks during lunch lull (12:00-13:30 ET) and 
   off-market hours. Use --force to override.
   
   Best Times:
   - Market Open (09:30-10:30 ET) - Most volatile
   - Late Morning (10:30-12:00 ET) - Trend confirmation
   - Afternoon/Pre-Close (13:30-16:00 ET) - Second volatility spike
   
   Output: CSV file with discoveries including entry price, stop loss, take profit

2. TRACK-EXITS MODE
   Monitor previous discoveries for exit signals:
   - Stop loss hit (entry - 2*ATR)
   - Take profit hit (entry + 3*ATR)
   - EMA cross down (momentum reversal)
   - RSI overbought (>75)
   
   Usage: python test_intraday_trade.py track-exits --file intraday_discoveries_YYYYMMDD.csv
   
   Output: Summary of exits and active positions

3. EVALUATE MODE (Backtest)
   Backtest strategy on historical data for a single symbol.
   
   Usage: python test_intraday_trade.py evaluate --symbol AAPL --period 60d --plot

All modes are standalone and don't require database access.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime
import time
import os
import sys
from pytz import timezone as tz
from datetime import datetime as dt

# -----------------------
# Trading hours validation
# -----------------------
def get_current_et_time():
    """Get current time in Eastern Time (ET)."""
    et = tz('US/Eastern')
    return dt.now(et)

def get_best_trading_times_local():
    """
    Display best trading times in both ET and local timezone.
    Returns formatted string with both times.
    """
    et = tz('US/Eastern')
    local = tz('Europe/Dublin')  # Default, will auto-detect if possible
    
    # Try to get local timezone automatically
    try:
        local_tzname = dt.now().astimezone().tzinfo
        if hasattr(local_tzname, 'zone'):
            local = tz(local_tzname.zone)
    except:
        pass  # Use default
    
    # Best trading periods in ET
    best_times = [
        ("Market Open (BEST)", "09:30", "10:30"),
        ("Late Morning (GOOD)", "10:30", "12:00"),
        ("Afternoon/Pre-Close (GOOD)", "13:30", "16:00"),
    ]
    
    lines = ["\n📅 Best Trading Times (US Market):\n"]
    
    for period_name, start_et, end_et in best_times:
        # Parse ET times
        start_h, start_m = map(int, start_et.split(':'))
        end_h, end_m = map(int, end_et.split(':'))
        
        # Create ET datetime objects (using today as reference)
        today = dt.now(et).date()
        start_et_dt = et.localize(dt.combine(today, dt.min.time().replace(hour=start_h, minute=start_m)))
        end_et_dt = et.localize(dt.combine(today, dt.min.time().replace(hour=end_h, minute=end_m)))
        
        # Convert to local time
        start_local = start_et_dt.astimezone(local)
        end_local = end_et_dt.astimezone(local)
        
        lines.append(f"  {period_name}:")
        lines.append(f"    ET:  {start_et} - {end_et}")
        lines.append(f"    Local: {start_local.strftime('%H:%M')} - {end_local.strftime('%H:%M %Z')}")
        lines.append("")
    
    lines.append("  ⚠️  Avoid: 12:00-13:30 ET (Lunch lull - low liquidity)")
    
    # Calculate lunch lull in local time
    lunch_start_et = et.localize(dt.combine(today, dt.min.time().replace(hour=12, minute=0)))
    lunch_end_et = et.localize(dt.combine(today, dt.min.time().replace(hour=13, minute=30)))
    lunch_start_local = lunch_start_et.astimezone(local)
    lunch_end_local = lunch_end_et.astimezone(local)
    lines.append(f"    Local: {lunch_start_local.strftime('%H:%M')} - {lunch_end_local.strftime('%H:%M %Z')}\n")
    
    return "\n".join(lines)

def check_trading_hours(force=False):
    """
    Check if current time is within good trading hours for intraday momentum.
    
    Best times:
    - Market Open: 09:30-10:30 ET (most volatile)
    - Late Morning: 10:30-12:00 ET (trend confirmation)
    - Afternoon/Pre-Close: 13:30-16:00 ET (second volatility spike)
    
    Avoid:
    - Lunch Lull: 12:00-13:30 ET (low liquidity)
    - Before Market Open: < 09:30 ET
    - After Market Close: > 16:00 ET
    
    Returns: (is_good_time: bool, message: str)
    """
    if force:
        return True, "⏭️  Time check bypassed with --force flag"
    
    now_et = get_current_et_time()
    now_local = now_et.astimezone()  # Convert to local timezone
    current_time = now_et.time()
    weekday = now_et.weekday()  # 0=Monday, 6=Sunday
    
    # Market is closed on weekends
    if weekday >= 5:  # Saturday or Sunday
        local_str = now_local.strftime('%Y-%m-%d %H:%M:%S %Z')
        et_str = now_et.strftime('%Y-%m-%d %H:%M:%S %Z')
        return False, (
            f"❌ Market closed (weekend).\n"
            f"   Local time: {local_str}\n"
            f"   ET time: {et_str}"
        )
    
    hour = current_time.hour
    minute = current_time.minute
    time_minutes = hour * 60 + minute
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = 9 * 60 + 30  # 09:30
    market_close = 16 * 60  # 16:00
    
    # Before market open
    if time_minutes < market_open:
        local_str = now_local.strftime('%H:%M:%S %Z')
        et_str = now_et.strftime('%H:%M:%S %Z')
        return False, (
            f"⏰ Before market open.\n"
            f"   Local time: {local_str}\n"
            f"   ET time: {et_str}\n"
            f"   Market opens at 09:30 ET"
        )
    
    # After market close
    if time_minutes >= market_close:
        local_str = now_local.strftime('%H:%M:%S %Z')
        et_str = now_et.strftime('%H:%M:%S %Z')
        return False, (
            f"⏰ After market close.\n"
            f"   Local time: {local_str}\n"
            f"   ET time: {et_str}\n"
            f"   Market closed at 16:00 ET"
        )
    
    # Lunch lull (12:00-13:30) - avoid
    lunch_start = 12 * 60  # 12:00
    lunch_end = 13 * 60 + 30  # 13:30
    
    if lunch_start <= time_minutes < lunch_end:
        local_str = now_local.strftime('%H:%M:%S %Z')
        et_str = now_et.strftime('%H:%M:%S %Z')
        return False, (
            f"⚠️  Lunch lull period (12:00-13:30 ET).\n"
            f"   Local time: {local_str}\n"
            f"   ET time: {et_str}\n"
            f"   Low liquidity during this time. Use --force to override."
        )
    
    # Good trading hours
    # Market Open: 09:30-10:30
    if market_open <= time_minutes < (10 * 60 + 30):
        period = "Market Open"
        quality = "🔥 BEST - Most volatile"
    # Late Morning: 10:30-12:00
    elif (10 * 60 + 30) <= time_minutes < lunch_start:
        period = "Late Morning"
        quality = "✅ GOOD - Trend confirmation"
    # Afternoon/Pre-Close: 13:30-16:00
    else:
        period = "Afternoon/Pre-Close"
        quality = "✅ GOOD - Second volatility spike"
    
    local_str = now_local.strftime('%H:%M:%S %Z')
    et_str = now_et.strftime('%H:%M:%S %Z')
    return True, f"✅ {quality} ({period}).\n   Local time: {local_str}\n   ET time: {et_str}"

# -----------------------
# Indicator helpers
# -----------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def normalize_dataframe(df):
    """
    Normalize DataFrame from yfinance download/history.
    Handles MultiIndex columns and ensures proper column names.
    """
    df = df.copy()
    
    # Handle MultiIndex columns (yfinance sometimes returns these even for single symbols)
    if isinstance(df.columns, pd.MultiIndex):
        # If MultiIndex, take the first level or flatten appropriately
        if len(df.columns.levels) > 1:
            df.columns = df.columns.droplevel(1)
        else:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Ensure columns are accessible (handle case where columns might be tuples)
    df.columns = [str(col).split('.')[0] if '.' in str(col) else str(col) for col in df.columns]
    
    return df

def sma(series, window):
    return series.rolling(window).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    # Combine into DataFrame to avoid alignment issues, then take max across columns
    tr_df = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3})
    tr = tr_df.max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def vwap(df):
    # typical price * volume cumulative / volume cumulative per day
    tpv = (df['Close'] * df['Volume']).cumsum()
    vol_cum = df['Volume'].cumsum()
    return tpv / vol_cum

# -----------------------
# Signal generation
# -----------------------
def generate_signals(df,
                     ema_short=8,
                     ema_long=21,
                     rsi_period=14,
                     rsi_oversold=35,
                     rsi_overbought=75,
                     vol_multiplier=1.5):
    df = df.copy()
    df['EMA_S'] = ema(df['Close'], ema_short)
    df['EMA_L'] = ema(df['Close'], ema_long)
    df['RSI'] = rsi(df['Close'], rsi_period)
    df['ATR'] = atr(df, period=14)
    df['VWAP'] = vwap(df)
    df['Vol_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()

    # Conditions
    # Entry long: price > VWAP, EMA_S > EMA_L, RSI not overbought, volume surge
    df['enter_long'] = (
        (df['Close'] > df['VWAP']) &
        (df['EMA_S'] > df['EMA_L']) &
        (df['RSI'] < rsi_overbought) &
        (df['Volume'] > df['Vol_MA'] * vol_multiplier)
    )

    # Exit long: EMA_S crosses below EMA_L OR RSI very high OR price hits TP/SL (handled in backtest)
    df['exit_long_hint'] = (
        (df['EMA_S'] < df['EMA_L']) |
        (df['RSI'] > rsi_overbought)
    )

    return df

# -----------------------
# Simple backtester (discrete, single-position)
# -----------------------
def backtest(df,
             initial_capital=10000,
             risk_per_trade=0.01,   # fraction of capital risked per trade
             sl_atr_mult=2.0,
             tp_atr_mult=3.0,
             verbose=False):
    df = df.copy().reset_index(drop=True)
    capital = initial_capital
    position = 0.0           # shares held (0 or >0)
    entry_price = None
    entry_index = None
    equity_curve = []
    trades = []

    for i, row in df.iterrows():
        price = row['Close']
        atr = row['ATR'] if not np.isnan(row['ATR']) else 0.0

        # If no position, check entry
        if position == 0:
            if row.get('enter_long', False):
                # determine position size based on ATR stop
                stop_price = price - sl_atr_mult * atr
                if stop_price <= 0 or atr == 0:
                    # skip if ATR invalid
                    continue

                # dollars risk per trade
                dollars_risk = capital * risk_per_trade
                # shares such that (entry - stop) * shares = dollars_risk
                shares = dollars_risk / (price - stop_price)
                shares = np.floor(shares) if shares > 0 else 0
                if shares < 1:
                    continue

                position = shares
                entry_price = price
                entry_index = i
                stop_loss = stop_price
                take_profit = price + tp_atr_mult * atr
                trades.append({
                    'entry_index': i,
                    'entry_price': price,
                    'shares': shares,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'exit_index': None,
                    'exit_price': None,
                    'pnl': None
                })
                if verbose:
                    print(f"ENTER idx {i} price {price:.4f} shares {shares} stop {stop_loss:.4f} tp {take_profit:.4f}")

        else:
            # There is an open position — check exits
            t = trades[-1]
            exited = False

            # 1) stop loss
            if row['Low'] <= t['stop_loss']:
                exit_price = t['stop_loss']  # assume executed at stop
                exited = True
                reason = 'stop'
            # 2) take profit
            elif row['High'] >= t['take_profit']:
                exit_price = t['take_profit']
                exited = True
                reason = 'tp'
            # 3) exit hint (EMA cross or RSI overbought) - use close
            elif row.get('exit_long_hint', False):
                exit_price = price
                exited = True
                reason = 'signal_exit'

            if exited:
                pnl = (exit_price - t['entry_price']) * t['shares']
                capital += pnl
                t['exit_index'] = i
                t['exit_price'] = exit_price
                t['pnl'] = pnl
                t['reason'] = reason
                if verbose:
                    print(f"EXIT idx {i} price {exit_price:.4f} pnl {pnl:.2f} reason {reason}")
                position = 0
                entry_price = None
                entry_index = None

        equity_curve.append(capital + (position * price))

    df['equity'] = equity_curve
    # basic stats
    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    net_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0.0
    win_rate = len(wins) / total_trades if total_trades > 0 else np.nan
    avg_win = wins['pnl'].mean() if not wins.empty else 0.0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0.0

    stats = {
        'initial_capital': initial_capital,
        'final_equity': df['equity'].iloc[-1],
        'net_pnl': net_pnl,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }

    return df, trades_df, stats

# -----------------------
# Discovery mode: Scan for current entry signals
# -----------------------
def check_entry_signal(symbol, period="7d", interval="1h", 
                      ema_short=8, ema_long=21, rsi_period=14, 
                      rsi_overbought=75, vol_multiplier=1.5,
                      sl_atr_mult=2.0, tp_atr_mult=3.0,
                      max_price=None, return_diagnostics=False):
    """
    Check if a stock currently meets ALL entry conditions.
    Returns discovery dict if signals present, None otherwise.
    
    Args:
        max_price: Maximum price to consider (filters out expensive stocks early)
        return_diagnostics: If True, returns (discovery, diagnostics) tuple
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Quick price check first (if max_price specified) - avoid expensive API calls
        if max_price is not None:
            try:
                fast_info = ticker.fast_info
                current_price = fast_info.get('lastPrice') or fast_info.get('regularMarketPrice')
                if current_price and current_price > max_price:
                    if return_diagnostics:
                        return None, {'filtered': 'price_too_high', 'price': current_price, 'max_price': max_price}
                    return None  # Skip expensive stocks early
            except:
                pass  # Continue if price check fails
        
        df = ticker.history(period=period, interval=interval)
        
        # Normalize DataFrame columns (handle MultiIndex from yfinance)
        df = normalize_dataframe(df)
        
        # Need at least 40 bars for reliable indicators (EMA_L=21, Vol_MA=20, RSI/ATR=14 + buffer)
        if df.empty or len(df) < 40:
            if return_diagnostics:
                return None, {'filtered': 'insufficient_data', 'bars': len(df) if not df.empty else 0}
            return None
        
        # Calculate indicators
        df = generate_signals(df, ema_short=ema_short, ema_long=ema_long,
                            rsi_period=rsi_period, rsi_overbought=rsi_overbought,
                            vol_multiplier=vol_multiplier)
        
        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        atr_val = float(latest['ATR']) if not np.isnan(latest['ATR']) else 0.0
        
        # Check ALL entry conditions (must all be true)
        price_above_vwap = latest['Close'] > latest['VWAP']
        ema_bullish = latest['EMA_S'] > latest['EMA_L']
        rsi_ok = latest['RSI'] < rsi_overbought
        vol_ok = latest['Volume'] > latest['Vol_MA'] * vol_multiplier
        
        # Build diagnostics if requested
        diagnostics = {}
        if return_diagnostics or not (price_above_vwap and ema_bullish and rsi_ok and vol_ok):
            diagnostics = {
                'price_above_vwap': price_above_vwap,
                'ema_bullish': ema_bullish,
                'rsi_ok': rsi_ok,
                'vol_ok': vol_ok,
                'price': float(current_price),
                'vwap': float(latest['VWAP']),
                'ema_s': float(latest['EMA_S']),
                'ema_l': float(latest['EMA_L']),
                'rsi': float(latest['RSI']),
                'volume_ratio': float(latest['Volume'] / latest['Vol_MA']) if latest['Vol_MA'] > 0 else 0.0,
                'required_vol_mult': vol_multiplier,
            }
        
        if not (price_above_vwap and ema_bullish and rsi_ok and vol_ok):
            if return_diagnostics:
                return None, diagnostics
            return None
        
        # Calculate stop/target based on ATR
        stop_loss = current_price - (sl_atr_mult * atr_val)
        take_profit = current_price + (tp_atr_mult * atr_val)
        risk = current_price - stop_loss
        reward = take_profit - current_price
        risk_reward_ratio = reward / risk if risk > 0 else 0

        print(f"risk_reward_ratio = {risk_reward_ratio}")
        
        # Get company name
        try:
            info = ticker.info
            company_name = info.get('longName') or info.get('shortName') or symbol
            market_cap = info.get('marketCap', 0)
        except:
            company_name = symbol
            market_cap = 0
        
        # Build explanation
        explanation_parts = [
            f"Price ${current_price:.2f} > VWAP ${latest['VWAP']:.2f}",
            f"EMA {latest['EMA_S']:.2f} > {latest['EMA_L']:.2f}",
            f"RSI {latest['RSI']:.1f}",
            f"Volume {latest['Volume']/latest['Vol_MA']:.1f}x avg"
        ]
        
        result = {
            'symbol': symbol,
            'company': company_name,
            'entry_price': current_price,
            'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward_ratio,
            'explanation': " | ".join(explanation_parts),
            'rsi': float(latest['RSI']),
            'volume_ratio': float(latest['Volume'] / latest['Vol_MA']),
            'atr': atr_val,
            'vwap': float(latest['VWAP']),
            'ema_short': float(latest['EMA_S']),
            'ema_long': float(latest['EMA_L']),
            'market_cap': market_cap,
        }
        if return_diagnostics:
            return result, {}
        return result
    except Exception as e:
        if return_diagnostics:
            return None, {'error': str(e)}
        print(f"  Error checking {symbol}: {e}")
        return None

def discover_opportunities(max_stocks=50, min_market_cap=100_000_000, 
                          max_price=None, period="7d", interval="1h",
                          vol_multiplier=1.5):
    """
    Scan active stocks for intraday momentum entry opportunities.
    Returns list of discoveries sorted by volume ratio (strongest momentum first).
    """
    print("🔍 Fetching active stocks...")
    
    # Get most active stocks (using same pattern as Yahoo advisor)
    try:
        from yfinance.screener import EquityQuery as YfEquityQuery
        most_active_query = YfEquityQuery(
            "and",
            [
                YfEquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
                YfEquityQuery("gt", ["intradayprice", 1.0]),
            ] + ([YfEquityQuery("lt", ["intradayprice", max_price])] if max_price is not None else [])
        )
        
        max_size = min(max_stocks * 2, 250)  # Cap at 250 like Yahoo does
        response = yf.screen(
            most_active_query,
            offset=0,
            size=max_size,
            sortField="intradayprice",  # Use intradayprice, not volume (volume not valid sortField)
            sortAsc=True,
        )
        
        quotes = response.get("quotes", [])
        stocks = []
        
        for quote in quotes:
            symbol = quote.get('symbol')
            volume = quote.get('volume') or quote.get('regularMarketVolume') or 0
            
            if symbol:
                stocks.append({
                    'symbol': symbol,
                    'volume': float(volume) if volume else 0.0,
                })
        
        # Sort by volume AFTER getting results (like Yahoo does)
        stocks.sort(key=lambda x: x['volume'], reverse=True)
        symbols = [s['symbol'] for s in stocks[:max_stocks]]
        
    except Exception as e:
        print(f"⚠️  Screener failed, using fallback list: {e}")
        # Fallback: Top liquid + volatile stocks commonly used by short-term traders
        # Note: Some may be above max_price - will be filtered during signal check
        symbols = [
            "TSLA", "AMD", "PLTR",  # Top volatile favorites
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",  # Tech mega-caps (very liquid)
            "SPY", "QQQ", "IWM",  # ETFs (high volume)
            "NIO", "RIVN", "LCID",  # EV sector
            "SOFI", "SNAP", "HOOD", "RBLX"  # Growth/meme stocks
        ]
    
    print(f"📊 Evaluating {len(symbols)} stocks for entry signals...\n")
    
    discoveries = []
    skipped_no_data = 0
    skipped_no_signal = 0
    skipped_price = 0
    skipped_mcap = 0
    
    # Diagnostic counters
    diag_failed_price_vwap = 0
    diag_failed_ema = 0
    diag_failed_rsi = 0
    diag_failed_volume = 0
    diag_failed_multiple = 0
    diag_errors = 0
    all_diagnostics = []
    
    for i, symbol in enumerate(symbols, 1):
        if i % 10 == 0:
            print(f"   Processed {i}/{len(symbols)}...")
        
        discovery, diagnostics = check_entry_signal(symbol, period=period, interval=interval, 
                                                    max_price=max_price, vol_multiplier=vol_multiplier,
                                                    return_diagnostics=True)
        
        if discovery is None:
            skipped_no_signal += 1
            
            # Track diagnostic reasons (skip 'filtered' entries - those are pre-filters, not signal failures)
            if diagnostics and 'filtered' not in diagnostics:
                if 'error' in diagnostics:
                    diag_errors += 1
                else:
                    failures = []
                    if not diagnostics.get('price_above_vwap', True):
                        failures.append('Price<=VWAP')
                    if not diagnostics.get('ema_bullish', True):
                        failures.append('EMA')
                    if not diagnostics.get('rsi_ok', True):
                        failures.append('RSI')
                    if not diagnostics.get('vol_ok', True):
                        failures.append('Volume')
                    
                    # Count individual failures
                    if len(failures) == 1:
                        if 'Price<=VWAP' in failures:
                            diag_failed_price_vwap += 1
                        elif 'EMA' in failures:
                            diag_failed_ema += 1
                        elif 'RSI' in failures:
                            diag_failed_rsi += 1
                        elif 'Volume' in failures:
                            diag_failed_volume += 1
                    elif len(failures) > 1:
                        diag_failed_multiple += 1
                    
                    # Store first 10 for detailed display
                    if len(all_diagnostics) < 10:
                        all_diagnostics.append((symbol, diagnostics, failures))
            continue
        
        # Filter by market cap
        if discovery['market_cap'] > 0 and discovery['market_cap'] < min_market_cap:
            skipped_mcap += 1
            continue
        # Double-check price filter (in case price changed or wasn't checked)
        if max_price is not None and discovery['entry_price'] > max_price:
            skipped_price += 1
            continue
        
        discoveries.append(discovery)
        
        time.sleep(0.1)  # Rate limiting
    
    # Summary of why stocks were skipped
    if skipped_no_signal > 0 or skipped_mcap > 0 or skipped_price > 0:
        print(f"\n📊 Scan summary:")
        print(f"   Found: {len(discoveries)} opportunities")
        print(f"   Skipped - no signal: {skipped_no_signal}")
        if skipped_mcap > 0:
            print(f"   Skipped - market cap < ${min_market_cap:,}: {skipped_mcap}")
        if skipped_price > 0:
            print(f"   Skipped - price > ${max_price:.2f}: {skipped_price}")
        print(f"   Total evaluated: {len(symbols)}\n")
    
    # Diagnostic breakdown
    if skipped_no_signal > 0:
        print(f"\n🔍 Diagnostic breakdown of {skipped_no_signal} failed signals:")
        print(f"   ❌ Price <= VWAP: {diag_failed_price_vwap}")
        print(f"   ❌ EMA_S <= EMA_L: {diag_failed_ema}")
        print(f"   ❌ RSI >= 75: {diag_failed_rsi}")
        print(f"   ❌ Volume < {vol_multiplier}x avg: {diag_failed_volume}")
        print(f"   ❌ Multiple conditions failed: {diag_failed_multiple}")
        if diag_errors > 0:
            print(f"   ⚠️  Errors/data issues: {diag_errors}")
        
        # Show sample failures with details
        if all_diagnostics:
            print(f"\n📋 Sample failures (first {len(all_diagnostics)}):")
            for sym, diag, failures in all_diagnostics:
                fail_str = ", ".join(failures) if failures else "No data"
                vol_info = f"vol={diag.get('volume_ratio', 0):.2f}x (need {diag.get('required_vol_mult', vol_multiplier)}x)" if 'volume_ratio' in diag else ""
                rsi_info = f"RSI={diag.get('rsi', 0):.1f}" if 'rsi' in diag else ""
                price_info = f"${diag.get('price', 0):.2f} vs VWAP ${diag.get('vwap', 0):.2f}" if 'price' in diag else ""
                print(f"   {sym:6s}: {fail_str:20s} | {vol_info} {rsi_info} {price_info}")
            print()
    
    # Sort by volume ratio (highest first) - strongest volume surge = best momentum signal
    discoveries.sort(key=lambda x: x['volume_ratio'], reverse=True)
    return discoveries

# -----------------------
# Exit tracking mode: Check discoveries for exit signals
# -----------------------
def check_exit_signal(symbol, entry_price, stop_loss, take_profit, 
                     period="5d", interval="1h"):
    """
    Check if a stock with known entry/stop/target has triggered exit signals.
    Returns exit info dict if exit triggered, None otherwise.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        # Normalize DataFrame columns (handle MultiIndex from yfinance)
        df = normalize_dataframe(df)
        
        # Need at least 40 bars for reliable indicators (EMA_L=21, Vol_MA=20, RSI/ATR=14 + buffer)
        if df.empty or len(df) < 40:
            return None
        
        df = generate_signals(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        current_price = float(latest['Close'])
        current_low = float(latest['Low'])
        current_high = float(latest['High'])
        
        # Check exit conditions
        exit_reason = None
        exit_price = None
        
        # 1) Stop loss hit
        if current_low <= stop_loss:
            exit_reason = 'stop_loss'
            exit_price = stop_loss
        # 2) Take profit hit
        elif current_high >= take_profit:
            exit_reason = 'take_profit'
            exit_price = take_profit
        # 3) EMA cross down (momentum reversal)
        elif (latest['EMA_S'] < latest['EMA_L'] and 
              prev['EMA_S'] >= prev['EMA_L']):
            exit_reason = 'ema_cross_down'
            exit_price = current_price
        # 4) RSI overbought
        elif latest['RSI'] > 75:
            exit_reason = 'rsi_overbought'
            exit_price = current_price
        
        if exit_reason:
            pnl = exit_price - entry_price
            pnl_pct = (pnl / entry_price) * 100
            
            return {
                'symbol': symbol,
                'exit_reason': exit_reason,
                'exit_price': exit_price,
                'entry_price': entry_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
            }
        
        return None
    except Exception as e:
        print(f"  Error checking exit for {symbol}: {e}")
        return None

def track_exits(discoveries_file):
    """
    Read discoveries CSV and check each for exit signals.
    """
    if not os.path.exists(discoveries_file):
        print(f"❌ File not found: {discoveries_file}")
        return
    
    print(f"📂 Reading discoveries from: {discoveries_file}")
    df_discoveries = pd.read_csv(discoveries_file)
    print(f"📊 Found {len(df_discoveries)} discoveries to check\n")
    
    exits = []
    still_active = []
    
    for idx, row in df_discoveries.iterrows():
        symbol = row['symbol']
        entry_price = float(row['entry_price'])
        stop_loss = float(row['stop_loss'])
        take_profit = float(row['take_profit'])
        
        print(f"Checking {symbol}...", end=" ")
        exit_signal = check_exit_signal(symbol, entry_price, stop_loss, take_profit)
        
        if exit_signal:
            print(f"✅ EXIT: {exit_signal['exit_reason']} at ${exit_signal['exit_price']:.2f} ({exit_signal['pnl_pct']:+.2f}%)")
            exits.append(exit_signal)
        else:
            # Get current price from latest bar
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d", interval="1h")
                hist = normalize_dataframe(hist)
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                else:
                    current_price = entry_price
            except:
                current_price = entry_price
            
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            print(f"⏳ Active: ${current_price:.2f} ({pnl_pct:+.2f}%)")
            still_active.append({
                'symbol': symbol,
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl_pct': pnl_pct,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
        
        time.sleep(0.2)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EXIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total checked: {len(df_discoveries)}")
    print(f"Exits triggered: {len(exits)}")
    print(f"Still active: {len(still_active)}")
    
    if exits:
        df_exits = pd.DataFrame(exits)
        print(f"\nExit reasons:")
        print(df_exits['exit_reason'].value_counts())
        print(f"\nTotal P&L: ${df_exits['pnl'].sum():.2f}")
        print(f"Average P&L: {df_exits['pnl_pct'].mean():.2f}%")
        
        # Save exits
        exits_file = f"intraday_exits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_exits.to_csv(exits_file, index=False)
        print(f"\n💾 Saved exits to: {exits_file}")
    
    if still_active:
        active_file = f"intraday_active_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(still_active).to_csv(active_file, index=False)
        print(f"💾 Saved active positions to: {active_file}")

def exit_end_of_day(discoveries_file, min_profit_pct=0.0, track_overnight=False):
    """
    Exit all winning positions at end of trading day.
    
    Strategy: For intraday trading, it's often better to lock in profits
    and avoid overnight risk (gaps, news, pre-market moves).
    
    Args:
        discoveries_file: CSV file with discoveries
        min_profit_pct: Minimum profit % to exit (default: 0.0 = exit all winners)
        track_overnight: If True, track what happens to exited stocks overnight
    """
    if not os.path.exists(discoveries_file):
        print(f"❌ File not found: {discoveries_file}")
        return
    
    # Check if it's near market close
    et_now = get_current_et_time()
    current_hour = et_now.hour
    current_minute = et_now.minute
    time_minutes = current_hour * 60 + current_minute
    market_close = 16 * 60  # 16:00
    
    print(f"📂 Reading discoveries from: {discoveries_file}")
    df_discoveries = pd.read_csv(discoveries_file)
    print(f"📊 Found {len(df_discoveries)} discoveries to evaluate\n")
    
    # Check time
    if time_minutes < (15 * 60 + 30):  # Before 15:30
        print(f"⚠️  Warning: Market close is at 16:00 ET. Current time: {et_now.strftime('%H:%M %Z')}")
        print(f"   Consider running this after 15:30 ET for best results.\n")
    
    exits = []
    held = []  # Stocks to hold (losers or below min_profit)
    
    print("=" * 80)
    print(f"{'Symbol':<8} {'Entry':<8} {'Current':<10} {'Change':<10} {'Action':<15} {'Reason':<20}")
    print("=" * 80)
    
    for _, row in df_discoveries.iterrows():
        symbol = row['symbol']
        entry_price = float(row['entry_price'])
        stop_loss = float(row['stop_loss'])
        take_profit = float(row['take_profit'])
        
        try:
            ticker = yf.Ticker(symbol)
            fast_info = ticker.fast_info
            current_price = fast_info.get('lastPrice') or fast_info.get('regularMarketPrice')
            
            if current_price is None:
                # Try hourly data
                hist = ticker.history(period="1d", interval="1h")
                hist = normalize_dataframe(hist)
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                else:
                    print(f"{symbol:<8} {'N/A':<8} {'No data':<10} {'HOLD':<15} {'No price data':<20}")
                    held.append({
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'current_price': None,
                        'change_pct': None,
                        'reason': 'no_price_data'
                    })
                    continue
            
            current_price = float(current_price)
            change_pct = ((current_price - entry_price) / entry_price) * 100
            pnl = current_price - entry_price
            
            # Decision: Exit if profitable and above minimum profit threshold
            if change_pct >= min_profit_pct:
                action = "🟢 EXIT"
                reason = f"Lock profit ({change_pct:+.2f}%)"
                exits.append({
                    'symbol': symbol,
                    'company': row.get('company', symbol),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'pnl': pnl,
                    'pnl_pct': change_pct,
                    'exit_time': et_now.strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_reason': 'end_of_day_profit_lock',
                    'held_overnight': not track_overnight  # Will track if requested
                })
            else:
                action = "⏳ HOLD"
                if change_pct < 0:
                    reason = f"Below entry ({change_pct:+.2f}%)"
                else:
                    reason = f"Below min profit threshold ({min_profit_pct}%)"
                held.append({
                    'symbol': symbol,
                    'company': row.get('company', symbol),
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'pnl_pct': change_pct,
                    'reason': 'below_threshold' if change_pct >= 0 else 'loss'
                })
            
            print(f"{symbol:<8} ${entry_price:<7.2f} ${current_price:<9.2f} {change_pct:>+7.2f}%  {action:<15} {reason:<20}")
            
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            print(f"{symbol:<8} {'Error':<8} {str(e)[:30]:<10}")
            held.append({
                'symbol': symbol,
                'entry_price': entry_price,
                'current_price': None,
                'change_pct': None,
                'reason': f'error: {str(e)[:30]}'
            })
    
    print("=" * 80)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"END-OF-DAY EXIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total positions: {len(df_discoveries)}")
    print(f"Exited (winners): {len(exits)}")
    print(f"Held (losers/below threshold): {len(held)}")
    
    if exits:
        exits_df = pd.DataFrame(exits)
        total_pnl = exits_df['pnl'].sum()
        avg_pnl_pct = exits_df['pnl_pct'].mean()
        
        print(f"\n💰 Exited Positions:")
        print(f"   Total P&L: ${total_pnl:+.2f}")
        print(f"   Average gain: {avg_pnl_pct:+.2f}%")
        print(f"   Best: {exits_df.loc[exits_df['pnl_pct'].idxmax(), 'symbol']} ({exits_df['pnl_pct'].max():+.2f}%)")
        
        # Save exits
        exits_file = f"intraday_eod_exits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        exits_df.to_csv(exits_file, index=False)
        print(f"\n💾 Saved exits to: {exits_file}")
        
        # Track overnight if requested
        if track_overnight:
            print(f"\n📊 Tracking overnight performance for exited stocks...")
            track_overnight_performance(exits_file)
    
    if held:
        held_df = pd.DataFrame(held)
        if 'pnl_pct' in held_df.columns:
            avg_held = held_df['pnl_pct'].mean()
            print(f"\n⏳ Held Positions (will check tomorrow):")
            print(f"   Average current: {avg_held:+.2f}%")
        
        held_file = f"intraday_eod_held_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        held_df.to_csv(held_file, index=False)
        print(f"💾 Saved held positions to: {held_file}")
    
    print(f"\n💡 Tip: Run discovery mode tomorrow to find fresh opportunities!")

def track_overnight_performance(exits_file):
    """
    Track what happens to stocks that were exited at EOD.
    Check next day's open price vs exit price to see if we made the right call.
    """
    if not os.path.exists(exits_file):
        print(f"❌ File not found: {exits_file}")
        return
    
    df_exits = pd.read_csv(exits_file)
    
    print(f"\n{'='*60}")
    print(f"OVERNIGHT PERFORMANCE TRACKER")
    print(f"{'='*60}")
    print(f"Checking {len(df_exits)} exited stocks...\n")
    
    print(f"{'Symbol':<8} {'EOD Exit':<10} {'Next Open':<12} {'Overnight':<12} {'Verdict':<20}")
    print("=" * 80)
    
    results = []
    
    for _, row in df_exits.iterrows():
        symbol = row['symbol']
        exit_price = float(row['exit_price'])
        
        try:
            ticker = yf.Ticker(symbol)
            # Get next day's data
            hist = ticker.history(period="5d", interval="1d")
            hist = normalize_dataframe(hist)
            
            if len(hist) < 2:
                print(f"{symbol:<8} ${exit_price:<9.2f} {'Insufficient data':<12}")
                continue
            
            # Get the day after exit
            # Assuming exit was yesterday, get today's open
            next_open = float(hist['Open'].iloc[-1])  # Most recent day's open
            
            overnight_change = ((next_open - exit_price) / exit_price) * 100
            
            if overnight_change > 2:
                verdict = "✅ Good exit (gap up)"
            elif overnight_change < -2:
                verdict = "✅ Excellent exit (gap down)"
            else:
                verdict = "➡️  Flat (neutral)"
            
            print(f"{symbol:<8} ${exit_price:<9.2f} ${next_open:<11.2f} {overnight_change:>+8.2f}%  {verdict:<20}")
            
            results.append({
                'symbol': symbol,
                'eod_exit_price': exit_price,
                'next_open_price': next_open,
                'overnight_change_pct': overnight_change
            })
            
            time.sleep(0.2)
            
        except Exception as e:
            print(f"{symbol:<8} ${exit_price:<9.2f} {'Error':<12} {str(e)[:30]}")
    
    if results:
        results_df = pd.DataFrame(results)
        avg_overnight = results_df['overnight_change_pct'].mean()
        gaps_up = len(results_df[results_df['overnight_change_pct'] > 2])
        gaps_down = len(results_df[results_df['overnight_change_pct'] < -2])
        
        print("=" * 80)
        print(f"\n📊 Overnight Summary:")
        print(f"   Average overnight change: {avg_overnight:+.2f}%")
        print(f"   Gapped up (>2%): {gaps_up} stocks")
        print(f"   Gapped down (<-2%): {gaps_down} stocks")
        print(f"   This validates the EOD exit strategy: {'✅ Locking profits was smart!' if avg_overnight < 0 or gaps_down > gaps_up else '⚠️  Some stocks continued higher'}")
        
        overnight_file = f"intraday_overnight_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(overnight_file, index=False)
        print(f"💾 Saved overnight tracking to: {overnight_file}")

def track_prices(discoveries_file, top_n=5):
    """
    Track current prices for top N discoveries.
    Shows entry price, current price, % change, and distance to TP/SL.
    """
    if not os.path.exists(discoveries_file):
        print(f"❌ File not found: {discoveries_file}")
        return
    
    print(f"📂 Reading discoveries from: {discoveries_file}")
    df_discoveries = pd.read_csv(discoveries_file)
    
    # Take top N by volume_ratio (or first N if no volume_ratio column)
    if 'volume_ratio' in df_discoveries.columns:
        df_top = df_discoveries.nlargest(top_n, 'volume_ratio')
    else:
        df_top = df_discoveries.head(top_n)
    
    print(f"📊 Tracking top {len(df_top)} discoveries\n")
    print("=" * 80)
    print(f"{'Symbol':<8} {'Entry':<8} {'Current':<10} {'Change':<10} {'Status':<15} {'To TP':<10} {'To SL':<10}")
    print("=" * 80)
    
    results = []
    
    for _, row in df_top.iterrows():
        symbol = row['symbol']
        entry_price = float(row['entry_price'])
        stop_loss = float(row['stop_loss'])
        take_profit = float(row['take_profit'])
        
        try:
            ticker = yf.Ticker(symbol)
            fast_info = ticker.fast_info
            current_price = fast_info.get('lastPrice') or fast_info.get('regularMarketPrice')
            
            if current_price is None:
                print(f"{symbol:<8} {'N/A':<8} {'No data':<10}")
                continue
            
            current_price = float(current_price)
            change_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Calculate distance to TP/SL as percentage
            to_tp_pct = ((take_profit - current_price) / current_price) * 100 if current_price < take_profit else 0
            to_sl_pct = ((current_price - stop_loss) / current_price) * 100 if current_price > stop_loss else 0
            
            # Determine status
            if current_price >= take_profit:
                status = "🎯 TP HIT"
            elif current_price <= stop_loss:
                status = "🛑 SL HIT"
            elif current_price >= entry_price * 0.95 + take_profit * 0.05:  # Within 5% of TP
                status = "🟢 Near TP"
            elif current_price <= entry_price * 0.05 + stop_loss * 0.95:  # Within 5% of SL
                status = "🔴 Near SL"
            elif change_pct > 2:
                status = "📈 Up"
            elif change_pct < -2:
                status = "📉 Down"
            else:
                status = "➡️  Flat"
            
            print(f"{symbol:<8} ${entry_price:<7.2f} ${current_price:<9.2f} {change_pct:>+7.2f}%  {status:<15} {to_tp_pct:>+7.2f}%  {to_sl_pct:>+7.2f}%")
            
            results.append({
                'symbol': symbol,
                'entry_price': entry_price,
                'current_price': current_price,
                'change_pct': change_pct,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': status
            })
            
        except Exception as e:
            print(f"{symbol:<8} {'Error':<8} {str(e)[:30]:<10}")
        
        time.sleep(0.2)  # Rate limiting
    
    print("=" * 80)
    
    # Summary
    if results:
        results_df = pd.DataFrame(results)
        avg_change = results_df['change_pct'].mean()
        winners = len(results_df[results_df['change_pct'] > 0])
        losers = len(results_df[results_df['change_pct'] <= 0])
        
        print(f"\n📊 Summary:")
        print(f"   Average change: {avg_change:+.2f}%")
        print(f"   Winners: {winners} | Losers: {losers}")
        
        # Save updated prices
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prices_file = f"intraday_prices_{timestamp}.csv"
        results_df.to_csv(prices_file, index=False)
        print(f"   💾 Saved to: {prices_file}")

# -----------------------
# Command-line interface
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Intraday Momentum Trading: Discovery, Exit Tracking, and Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  discovery     - Scan active stocks for current entry signals (discover opportunities)
  track-prices  - Track current prices for top discoveries (quick price check)
  track-exits   - Check previous discoveries for exit signals (monitor positions)
  exit-eod      - Exit all winning positions at end of trading day (lock profits, avoid overnight risk)
  evaluate      - Backtest strategy on historical data for a single symbol

Examples:
  # Discover new opportunities (blocks during lunch lull 12:00-13:30 ET)
  python test_intraday_trade.py discovery --max-stocks 100

  # Track prices of top 5 discoveries
  python test_intraday_trade.py track-prices --file intraday_discoveries_20250101.csv --top 5

  # Track exits from previous discoveries
  python test_intraday_trade.py track-exits --file intraday_discoveries_20250101.csv

  # Exit all winners at end of day (lock profits, avoid overnight risk)
  python test_intraday_trade.py exit-eod --file intraday_discoveries_20250101.csv --track-overnight

  # Evaluate/backtest on historical data
  python test_intraday_trade.py evaluate --symbol AAPL --period 60d --plot

Trading Hours:
  Discovery mode blocks during lunch lull (12:00-13:30 ET) and off-market hours.
  Best times: 09:30-12:00 ET and 13:30-16:00 ET. Use --force to override.
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Discovery mode
    parser_discover = subparsers.add_parser('discovery', help='Scan for entry signals')
    parser_discover.add_argument('--max-stocks', type=int, default=50,
                                help='Maximum stocks to scan (default: 50)')
    parser_discover.add_argument('--max-price', type=float, default=None,
                                help='Maximum stock price to scan (default: None = no limit)')
    parser_discover.add_argument('--min-market-cap', type=int, default=100_000_000,
                                help='Minimum market cap (default: 100M)')
    parser_discover.add_argument('--period', type=str, default='7d',
                                help='Historical period for indicators (default: 7d = ~50 hourly bars)')
    parser_discover.add_argument('--interval', type=str, default='1h',
                                help='Data interval (default: 1h)')
    parser_discover.add_argument('--volume-threshold', type=float, default=1.5,
                                help='Volume surge multiplier - current volume must be > this × average (default: 1.5). Lower for quiet days (e.g., 0.5 for early hours)')
    parser_discover.add_argument('--force', action='store_true',
                                help='Force run even during lunch lull or off-hours (not recommended)')
    
    # Track exits mode
    parser_track = subparsers.add_parser('track-exits', help='Check discoveries for exits')
    parser_track.add_argument('--file', type=str, required=True,
                             help='CSV file with previous discoveries')
    parser_track.add_argument('--force', action='store_true',
                             help='Force run even during lunch lull or off-hours')
    
    # Track prices mode
    parser_prices = subparsers.add_parser('track-prices', help='Track current prices for top discoveries')
    parser_prices.add_argument('--file', type=str, required=True,
                              help='CSV file with discoveries (from discovery mode)')
    parser_prices.add_argument('--top', type=int, default=5,
                              help='Number of top stocks to track (default: 5)')
    
    # Exit EOD mode
    parser_eod = subparsers.add_parser('exit-eod', help='Exit all winning positions at end of trading day')
    parser_eod.add_argument('--file', type=str, required=True,
                           help='CSV file with discoveries (from discovery mode)')
    parser_eod.add_argument('--min-profit', type=float, default=0.0,
                           help='Minimum profit %% to exit (default: 0.0 = exit all winners)')
    parser_eod.add_argument('--track-overnight', action='store_true',
                           help='Track what happens to exited stocks overnight (check next day open)')
    
    # Evaluate mode (backtest)
    parser_eval = subparsers.add_parser('evaluate', help='Backtest on historical data')
    parser_eval.add_argument('--symbol', type=str, default='AAPL',
                            help='Stock symbol to backtest (default: AAPL)')
    parser_eval.add_argument('--period', type=str, default='60d',
                            help='Historical period (default: 60d)')
    parser_eval.add_argument('--interval', type=str, default='1h',
                            help='Data interval (default: 1h)')
    parser_eval.add_argument('--capital', type=float, default=10000,
                            help='Initial capital (default: 10000)')
    parser_eval.add_argument('--risk', type=float, default=0.01,
                            help='Risk per trade as fraction (default: 0.01)')
    parser_eval.add_argument('--plot', action='store_true',
                            help='Show plots')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    # ====================
    # DISCOVERY MODE
    # ====================
    if args.mode == 'discovery':
        print("=" * 60)
        print("INTRADAY MOMENTUM DISCOVERY SCANNER")
        print("=" * 60)
        
        # Show best trading times
        print(get_best_trading_times_local())
        
        # Check trading hours
        is_good_time, time_message = check_trading_hours(force=getattr(args, 'force', False))
        print(time_message)
        
        if not is_good_time:
            print("\n💡 Tip: Use --force flag to override time check if needed.")
            sys.exit(1)
        
        print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        discoveries = discover_opportunities(
            max_stocks=args.max_stocks,
            min_market_cap=args.min_market_cap,
            max_price=args.max_price,
            period=args.period,
            interval=args.interval,
            vol_multiplier=getattr(args, 'volume_threshold', 1.5)
        )
        
        print(f"\n✅ Found {len(discoveries)} opportunities\n")
        print("=" * 60)
        print("TOP DISCOVERIES (sorted by volume ratio - strongest momentum first)")
        print("=" * 60)
        
        # Display top discoveries
        for i, disc in enumerate(discoveries[:20], 1):
            print(f"\n{i}. {disc['symbol']} - {disc['company']}")
            print(f"   Entry: ${disc['entry_price']:.2f} | R:R {disc['risk_reward_ratio']:.2f}")
            print(f"   SL: ${disc['stop_loss']:.2f} | TP: ${disc['take_profit']:.2f}")
            print(f"   {disc['explanation']}")
        
        # Save to CSV
        if discoveries:
            df_discoveries = pd.DataFrame(discoveries)
            filename = f"intraday_discoveries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_discoveries.to_csv(filename, index=False)
            print(f"\n💾 Saved {len(discoveries)} discoveries to: {filename}")
        
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ====================
    # TRACK EXITS MODE
    # ====================
    elif args.mode == 'track-exits':
        print("=" * 60)
        print("EXIT SIGNAL TRACKER")
        print("=" * 60)
        
        # Check trading hours (warning only for exit tracking - less critical)
        is_good_time, time_message = check_trading_hours(force=getattr(args, 'force', False))
        if not is_good_time:
            print(f"⚠️  {time_message}")
            print("💡 Note: Exit tracking can run anytime. Use --force to suppress this warning.")
        else:
            print(time_message)
        
        print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        track_exits(args.file)
        
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ====================
    # TRACK-PRICES MODE
    # ====================
    elif args.mode == 'track-prices':
        print("=" * 60)
        print("INTRADAY DISCOVERY PRICE TRACKER")
        print("=" * 60)
        print(f"File: {args.file}")
        print(f"Tracking top {args.top} stocks\n")
        
        track_prices(args.file, top_n=args.top)
        
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ====================
    # EXIT EOD MODE
    # ====================
    elif args.mode == 'exit-eod':
        print("=" * 60)
        print("END-OF-DAY EXIT (Lock Profits)")
        print("=" * 60)
        print(f"Strategy: Exit winning positions before market close to avoid overnight risk")
        print(f"File: {args.file}")
        print(f"Minimum profit to exit: {args.min_profit}%")
        if args.track_overnight:
            print(f"Overnight tracking: ENABLED (will check next day's open)\n")
        else:
            print(f"Overnight tracking: DISABLED (use --track-overnight to enable)\n")
        
        exit_end_of_day(
            args.file,
            min_profit_pct=args.min_profit,
            track_overnight=args.track_overnight
        )
        
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ====================
    # EVALUATE MODE (Backtest)
    # ====================
    elif args.mode == 'evaluate':
        print("=" * 60)
        print("INTRADAY MOMENTUM BACKTEST")
        print("=" * 60)
        print(f"Symbol: {args.symbol}")
        print(f"Period: {args.period} | Interval: {args.interval}\n")

        # Download OHLCV
        df = yf.download(args.symbol, period=args.period, interval=args.interval)
        if df.empty:
            raise SystemExit("No data downloaded. Check symbol/interval/period or your network.")
        
        # Normalize DataFrame columns (handle MultiIndex from yfinance)
        df = normalize_dataframe(df)
        
        # Ensure we have the expected columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise SystemExit(f"Missing required columns: {missing_cols}. Available: {list(df.columns)}")

        print(f"📊 Loaded {len(df)} bars of data")

        # Compute signals
        df_signal = generate_signals(df,
                                     ema_short=8,
                                     ema_long=21,
                                     rsi_period=14,
                                     rsi_oversold=35,
                                     rsi_overbought=75,
                                     vol_multiplier=1.5)

        # Backtest
        df_back, trades_df, stats = backtest(df_signal,
                                             initial_capital=args.capital,
                                             risk_per_trade=args.risk,
                                             sl_atr_mult=2.0,
                                             tp_atr_mult=3.0,
                                             verbose=False)

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

        if not trades_df.empty:
            print(f"\n📈 Trade Breakdown:")
            print(f"  Total trades: {len(trades_df)}")
            print(f"  Wins: {len(trades_df[trades_df['pnl'] > 0])}")
            print(f"  Losses: {len(trades_df[trades_df['pnl'] <= 0])}")
            print(f"  TP hits: {len(trades_df[trades_df['reason'] == 'tp'])}")
            print(f"  SL hits: {len(trades_df[trades_df['reason'] == 'stop'])}")
            print(f"  Signal exits: {len(trades_df[trades_df['reason'] == 'signal_exit'])}")
            
            print(f"\n📊 Sample trades:")
            print(trades_df.head(10).to_string())
        
        # Plots
        if args.plot:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("⚠️  matplotlib not installed. Install with: pip install matplotlib")
                print("   Skipping plots...")
                args.plot = False
        
        if args.plot:
            fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax[0].plot(df_back.index, df_back['Close'], label='Close', linewidth=1.5)
            ax[0].plot(df_back.index, df_back['EMA_S'], label='EMA_S', alpha=0.7)
            ax[0].plot(df_back.index, df_back['EMA_L'], label='EMA_L', alpha=0.7)
            ax[0].plot(df_back.index, df_back['VWAP'], label='VWAP', linestyle='--', alpha=0.7)
            
            # Mark entry/exit points
            if not trades_df.empty:
                for _, trade in trades_df.iterrows():
                    entry_idx = trade['entry_index']
                    exit_idx = trade.get('exit_index')
                    if exit_idx and exit_idx < len(df_back):
                        ax[0].scatter(df_back.index[entry_idx], df_back['Close'].iloc[entry_idx], 
                                     color='green', marker='^', s=100, zorder=5)
                        ax[0].scatter(df_back.index[exit_idx], df_back['Close'].iloc[exit_idx], 
                                     color='red', marker='v', s=100, zorder=5)
            
            ax[0].legend()
            ax[0].set_title(f"{args.symbol} Price + Indicators")
            ax[0].grid(True, alpha=0.3)
            
            ax[1].plot(df_back.index, df_back['equity'], label='Equity', linewidth=2)
            ax[1].axhline(y=args.capital, color='gray', linestyle='--', label='Initial Capital', alpha=0.5)
            ax[1].legend()
            ax[1].set_title("Equity Curve")
            ax[1].set_xlabel("Time")
            ax[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

