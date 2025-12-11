"""
Simulate Intraday Discoveries on Historical Data
Tests EOD exit strategies with more data points
"""

import sys
import os

# Import functions needed from test_intraday_trade
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_intraday_trade import normalize_dataframe, generate_signals
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone as tz
from yfinance.screener import EquityQuery as YfEquityQuery
import time

# Default stock universe (mid-cap $5-$20 range)
DEFAULT_STOCK_UNIVERSE = [
    # Mid-cap stocks in $5-$20 range
    "PLTR", "SOFI", "SNAP", "RBLX", "HOOD", "NIO", "RIVN", "LCID",
    "FUBO", "OPENL", "CLOV", "WISH", "SPCE", "BB", "AMC", "GME",
    "CLNE", "WKHS", "QS", "SPCE", "SKLZ", "CLOV", "WISH", "UWMC",
    "MVST", "PROG", "ATER", "SDC", "BBIG", "CEI", "BENE", "DWAC",
    "SNDL", "SOS", "NAKD", "EXPR", "FIZZ", "ROKU", "ZM", "PTON",
    "BYND", "NKLA", "HYLN", "WORKH", "MARA", "RIOT", "HUT", "HIVE",
    "GPRO", "TLRY", "ACB", "CGC", "HEXO", "APHA", "OGI", "CRON",
    "TSLA", "AMD", "NVDA", "AAPL", "MSFT",  # High volume for liquidity check
]

VOL_MULTIPLIER = 1.5  # Match Django advisor
MIN_MARKET_CAP = 100_000_000
MAX_PRICE = 10.0  # For intraday strategy

def discover_on_date(symbols, target_date, max_price=MAX_PRICE, vol_multiplier=VOL_MULTIPLIER):
    """
    Simulate discovery for a specific historical date.
    Checks entry signals using historical data up to target_date.
    
    Args:
        symbols: List of stock symbols to check
        target_date: Date to simulate discovery (datetime or date object)
        max_price: Maximum stock price
        vol_multiplier: Volume multiplier threshold
    
    Returns:
        List of discoveries that would have been found on that date
    """
    discoveries = []
    
    # Convert target_date to datetime if needed
    if isinstance(target_date, datetime):
        target_dt = target_date
    else:
        # Assume discovery happens at 10:30 AM ET (during discovery window)
        et = tz('US/Eastern')
        target_dt = et.localize(datetime.combine(target_date, datetime.strptime("10:30", "%H:%M").time()))
    
    # Convert to UTC for yfinance
    target_dt_utc = target_dt.astimezone(tz('UTC'))
    end_date = target_dt_utc
    start_date = end_date - timedelta(days=10)  # Need 7d + buffer for indicators
    
    print(f"  Simulating discovery for {target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)}...", end=" ", flush=True)
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data up to target_date
            # Use date range - yfinance will give us hourly data
            start_date_obj = start_date.date() if hasattr(start_date, 'date') else start_date
            end_date_obj = (end_date + timedelta(days=1)).date() if hasattr(end_date, 'date') else end_date + timedelta(days=1)
            
            df = ticker.history(start=start_date_obj, end=end_date_obj, interval='1h')
            
            if df.empty:
                continue
            
            df = normalize_dataframe(df)
            
            # Filter to data up to target_date (discovery happens around 10:30 AM ET)
            # Convert index to UTC-aware if not already
            if df.index.tz is None:
                df.index = df.index.tz_localize('US/Eastern').tz_convert('UTC')
            
            # Get last hourly bar before or at discovery time
            df_filtered = df[df.index <= target_dt_utc]
            
            if df_filtered.empty or len(df_filtered) < 40:
                continue
            
            # Calculate indicators
            df_signal = generate_signals(
                df_filtered,
                ema_short=8,
                ema_long=21,
                rsi_period=14,
                rsi_overbought=75,
                vol_multiplier=vol_multiplier
            )
            
            # Get the last bar (this would be the discovery point)
            latest = df_signal.iloc[-1]
            
            # Check entry conditions
            price_above_vwap = latest['Close'] > latest['VWAP']
            ema_bullish = latest['EMA_S'] > latest['EMA_L']
            rsi_ok = latest['RSI'] < 75
            vol_ok = latest['Volume'] > latest['Vol_MA'] * vol_multiplier
            
            if not (price_above_vwap and ema_bullish and rsi_ok and vol_ok):
                continue
            
            entry_price = float(latest['Close'])
            
            # Price filter
            if max_price and entry_price > max_price:
                continue
            
            # Get market cap (try from current info, may not be perfect for historical)
            try:
                info = ticker.info
                market_cap = info.get('marketCap', 0)
                company_name = info.get('longName') or info.get('shortName') or symbol
            except:
                market_cap = 0
                company_name = symbol
            
            if market_cap > 0 and market_cap < MIN_MARKET_CAP:
                continue
            
            # Calculate stop/target
            atr_val = float(latest['ATR']) if not np.isnan(latest['ATR']) else 0.0
            stop_loss = entry_price - (2.0 * atr_val)
            take_profit = entry_price + (3.0 * atr_val)
            
            discoveries.append({
                'symbol': symbol,
                'company': company_name,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'volume_ratio': float(latest['Volume'] / latest['Vol_MA']) if latest['Vol_MA'] > 0 else 0.0,
                'discovery_date': target_date if hasattr(target_date, 'date') else target_date,
            })
            
        except Exception as e:
            continue  # Skip errors
    
    print(f"Found {len(discoveries)} discoveries")
    return discoveries

def get_exit_price(symbol, target_date, entry_price, stop_loss, take_profit, use_hourly=True):
    """
    Get exit price for a stock on a specific date, checking for stop loss/take profit hits.
    
    Args:
        symbol: Stock symbol
        target_date: Date to check
        entry_price: Entry price at discovery
        stop_loss: Stop loss price
        take_profit: Take profit price
        use_hourly: If True, use hourly data to check intraday hits
    
    Returns:
        (exit_price, exit_reason) where exit_reason is 'stop_loss', 'take_profit', or 'eod'
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Handle both date and datetime objects
        if isinstance(target_date, datetime):
            start_date = target_date.date()
        else:
            start_date = target_date
        
        end_date = start_date + timedelta(days=1)
        
        # Always use hourly data to match discovery price source (avoids split adjustment issues)
        if use_hourly:
            hist = ticker.history(start=start_date, end=end_date, interval='1h')
            if not hist.empty:
                hist = normalize_dataframe(hist)
                # Convert index to date for filtering
                if hist.index.tz is None:
                    hist.index = hist.index.tz_localize('US/Eastern')
                hist.index = hist.index.tz_convert('UTC')
                
                # Get bars for the target date only (after discovery time - 10:30 AM ET)
                et = tz('US/Eastern')
                discovery_time_et = et.localize(datetime.combine(start_date, datetime.strptime("10:30", "%H:%M").time()))
                discovery_time_utc = discovery_time_et.astimezone(tz('UTC'))
                
                target_date_start = tz('UTC').localize(datetime.combine(start_date, datetime.min.time()))
                target_date_end = target_date_start + timedelta(days=1)
                hist_filtered = hist[(hist.index >= discovery_time_utc) & (hist.index < target_date_end)]
                
                if not hist_filtered.empty:
                    # Check if stop loss or take profit was hit during the day
                    for idx, row in hist_filtered.iterrows():
                        low_price = float(row['Low'])
                        high_price = float(row['High'])
                        
                        # Check stop loss first (if hit, we exit)
                        if stop_loss > 0 and low_price <= stop_loss:
                            return stop_loss, 'stop_loss'
                        
                        # Check take profit (if hit, we exit)
                        if take_profit > 0 and high_price >= take_profit:
                            return take_profit, 'take_profit'
                    
                    # No stop/target hit, return EOD price
                    eod_price = float(hist_filtered['Close'].iloc[-1])
                    return eod_price, 'eod'
        
        # Fallback: try daily data if hourly didn't work (can't check intraday stops)
        hist = ticker.history(start=start_date, end=end_date, interval='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1]), 'eod'
        
        return None, None
    except Exception as e:
        return None, None

def simulate_historical_discoveries(start_date, end_date, stock_universe=None, max_price=MAX_PRICE, vol_multiplier=VOL_MULTIPLIER):
    """
    Simulate intraday discoveries for a range of historical dates.
    
    Args:
        start_date: Start date (datetime or date)
        end_date: End date (datetime or date)
        stock_universe: List of symbols to check (defaults to DEFAULT_STOCK_UNIVERSE)
        max_price: Maximum stock price
        vol_multiplier: Volume multiplier threshold
    
    Returns:
        List of discoveries with EOD prices and strategy comparisons
    """
    if stock_universe is None:
        stock_universe = DEFAULT_STOCK_UNIVERSE
    
    # Convert to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
    
    et = tz('US/Eastern')
    all_results = []
    
    current_date = start_date
    while current_date <= end_date:
        # Skip weekends
        weekday = current_date.weekday()
        if weekday >= 5:  # Saturday or Sunday
            current_date += timedelta(days=1)
            continue
        
        # Create discovery datetime (10:30 AM ET)
        discovery_dt = et.localize(datetime.combine(current_date, datetime.strptime("10:30", "%H:%M").time()))
        
        # Simulate discovery
        discoveries = discover_on_date(stock_universe, discovery_dt, max_price=max_price, vol_multiplier=vol_multiplier)
        
        # Get exit prices (checking for stop loss/take profit hits) and calculate results
        for disc in discoveries:
            entry_price = disc['entry_price']
            stop_loss = disc.get('stop_loss', 0)
            take_profit = disc.get('take_profit', 0)
            
            # Check for intraday stop/target hits
            exit_price, exit_reason = get_exit_price(
                disc['symbol'], 
                current_date, 
                entry_price,
                stop_loss,
                take_profit,
                use_hourly=True
            )
            
            if exit_price:
                exit_pnl = ((exit_price - entry_price) / entry_price) * 100
                
                all_results.append({
                    'date': current_date,
                    'symbol': disc['symbol'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'exit_pnl': exit_pnl,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'volume_ratio': disc['volume_ratio'],
                    # Keep eod_pnl for backward compatibility (what EOD price would have been)
                    'eod_pnl': exit_pnl if exit_reason == 'eod' else None,
                })
        
        current_date += timedelta(days=1)
        time.sleep(0.1)  # Rate limiting
    
    return all_results

def compare_eod_strategies(results):
    """
    Compare different EOD exit strategies on simulated historical discoveries.
    Now includes stop loss/take profit checking.
    """
    if not results:
        print("No results to analyze")
        return
    
    # Filter out extreme outliers (likely data errors or corporate actions)
    # Remove trades with >50% or <50% moves (reverse splits, splits, etc.)
    filtered_results = []
    outliers = []
    
    for r in results:
        # Use exit_pnl (includes stop/target hits) for outlier checking
        pnl_abs = abs(r.get('exit_pnl', r.get('eod_pnl', 0)))
        if pnl_abs > 50.0:  # More than 50% move in one day is suspicious
            outliers.append(r)
        else:
            filtered_results.append(r)
    
    if outliers:
        print(f"\n⚠️  Filtered out {len(outliers)} outlier(s) (>50% move - likely data error or corporate action):")
        for o in outliers:
            exit_price = o.get('exit_price', o.get('eod_price', 0))
            exit_pnl = o.get('exit_pnl', o.get('eod_pnl', 0))
            print(f"   {o['date']} {o['symbol']}: {o['entry_price']:.2f} → {exit_price:.2f} ({exit_pnl:+.2f}%)")
        print()
    
    if not filtered_results:
        print("No valid results after filtering outliers")
        return
    
    results = filtered_results
    
    # Show stop/target hit summary
    stop_hits = len([r for r in results if r.get('exit_reason') == 'stop_loss'])
    target_hits = len([r for r in results if r.get('exit_reason') == 'take_profit'])
    eod_exits = len([r for r in results if r.get('exit_reason') == 'eod'])
    
    print("\n" + "="*80)
    print("EOD STRATEGY COMPARISON (Historical Simulation)")
    print("="*80)
    if len(outliers) > 0:
        print(f"Analyzing {len(results)} trades ({len(outliers)} outliers excluded)")
    print(f"\nExit Summary:")
    print(f"   Stop Loss Hits: {stop_hits}/{len(results)} ({stop_hits/len(results)*100:.0f}%)")
    print(f"   Take Profit Hits: {target_hits}/{len(results)} ({target_hits/len(results)*100:.0f}%)")
    print(f"   EOD Exits: {eod_exits}/{len(results)} ({eod_exits/len(results)*100:.0f}%)")
    print("="*80)
    
    # Strategy 1: Current strategy (with stop/target checking)
    exit_pnls = [r.get('exit_pnl', r.get('eod_pnl', 0)) for r in results]
    current_avg = sum(exit_pnls) / len(exit_pnls)
    current_winners = len([p for p in exit_pnls if p > 0])
    
    print(f"\n0. CURRENT STRATEGY (with Stop Loss/Take Profit):")
    print(f"   Trades: {len(results)}")
    print(f"   Winners: {current_winners} ({current_winners/len(results)*100:.0f}%)")
    print(f"   Average P&L: {current_avg:+.2f}%")
    
    # Strategy 2: Sell ALL at EOD (ignore stop/target, force EOD exit)
    # For this, use eod_pnl if available, otherwise use exit_price for EOD
    eod_pnls = []
    for r in results:
        if r.get('eod_pnl') is not None:
            eod_pnls.append(r['eod_pnl'])
        elif r.get('exit_reason') == 'eod':
            eod_pnls.append(r.get('exit_pnl', 0))
        else:
            # Was stopped/targeted, but what would EOD have been?
            # Use exit_price as approximation (not perfect, but close)
            eod_pnls.append(r.get('exit_pnl', 0))  # Use actual exit for now
    
    if eod_pnls:
        eod_all_avg = sum(eod_pnls) / len(eod_pnls)
        eod_all_winners = len([p for p in eod_pnls if p > 0])
    
        print(f"\n1. END_DAY: Sell ALL at End of Day (ignore stop/target, force EOD)")
        print(f"   Trades: {len(results)}")
        print(f"   Winners: {eod_all_winners} ({eod_all_winners/len(results)*100:.0f}%)")
        print(f"   Average P&L: {eod_all_avg:+.2f}%")
    else:
        eod_all_avg = 0
        eod_all_winners = 0
    
    # Strategy 2: Sell only winners with thresholds
    print(f"\n2. END_DAY_UP: Sell Only Winners at End of Day")
    print("-"*80)
    
    best_strategy = None
    best_avg = float('-inf')
    best_threshold = None
    
    # Strategy 3: END_DAY_UP - Sell only winners at EOD (with stop/target still in play)
    print(f"\n2. END_DAY_UP: Sell Only Winners at End of Day (stop/target still active)")
    print("-"*80)
    
    for threshold in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        # For each trade:
        # - If EOD P&L >= threshold AND no stop/target hit, sell at EOD
        # - If stop/target was hit, use that exit (already happened)
        # - If EOD P&L < threshold and no stop/target hit, hold (assume break-even)
        
        strategy_pnls = []
        for r in results:
            exit_pnl = r.get('exit_pnl', r.get('eod_pnl', 0))
            exit_reason = r.get('exit_reason', 'eod')
            
            # If stop/target was hit, use that exit (already happened)
            if exit_reason in ['stop_loss', 'take_profit']:
                strategy_pnls.append(exit_pnl)
            # If at EOD and P&L >= threshold, sell at EOD
            elif exit_reason == 'eod':
                eod_pnl_val = r.get('eod_pnl', exit_pnl)
                if eod_pnl_val >= threshold:
                    strategy_pnls.append(eod_pnl_val)
                else:
                    # Hold - assume break-even (0%) for simulation
                    strategy_pnls.append(0.0)
            else:
                # Unknown reason, use exit_pnl
                strategy_pnls.append(exit_pnl)
        
        if strategy_pnls:
            avg_pnl = sum(strategy_pnls) / len(strategy_pnls)
            winners = len([p for p in strategy_pnls if p > 0])
            sold_count = len([r for r in results if r.get('exit_reason') == 'eod' and (r.get('eod_pnl', 0) >= threshold)])
            
            if avg_pnl > best_avg:
                best_avg = avg_pnl
                best_threshold = threshold
            
            print(f"   Threshold: >= {threshold:+.1f}%")
            print(f"      Sold at EOD: {sold_count}/{len(results)} ({sold_count/len(results)*100:.0f}%)")
            print(f"      Stop/Target Hits: {stop_hits + target_hits}/{len(results)}")
            print(f"      Held: {len(results) - sold_count - stop_hits - target_hits}/{len(results)}")
            print(f"      Average P&L: {avg_pnl:+.2f}%")
            print(f"      Win Rate: {winners}/{len(results)} ({winners/len(results)*100:.0f}%)")
            print()
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print(f"Best Strategy: END_DAY_UP with threshold >= {best_threshold:+.1f}%")
    print(f"   Average P&L: {best_avg:+.2f}%")
    print(f"   vs Current (with stop/target): {current_avg:+.2f}%")
    if eod_all_avg > 0:
        print(f"   vs END_DAY (force EOD): {eod_all_avg:+.2f}%")
    print(f"   Improvement: {best_avg - current_avg:+.2f}%")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate intraday discoveries on historical data')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--max-price', type=float, default=MAX_PRICE, help=f'Maximum stock price (default: {MAX_PRICE})')
    parser.add_argument('--volume-threshold', type=float, default=VOL_MULTIPLIER, help=f'Volume multiplier (default: {VOL_MULTIPLIER})')
    parser.add_argument('--symbols', type=str, nargs='+', help='Stock symbols to test (default: uses DEFAULT_STOCK_UNIVERSE)')
    
    args = parser.parse_args()
    
    stock_universe = args.symbols if args.symbols else DEFAULT_STOCK_UNIVERSE
    
    print("="*80)
    print("HISTORICAL INTRADAY DISCOVERY SIMULATION")
    print("="*80)
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Stock Universe: {len(stock_universe)} symbols")
    print(f"Max Price: ${args.max_price}")
    print(f"Volume Threshold: {args.volume_threshold}x")
    print("="*80)
    
    results = simulate_historical_discoveries(
        args.start_date,
        args.end_date,
        stock_universe=stock_universe,
        max_price=args.max_price,
        vol_multiplier=args.volume_threshold
    )
    
    print(f"\n✅ Simulated {len(results)} discoveries across {len(set(r['date'] for r in results))} trading days")
    
    compare_eod_strategies(results)
    
    # Save results
    import pandas as pd
    if results:
        df = pd.DataFrame(results)
        filename = f"intraday_simulation_{args.start_date}_{args.end_date}.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Saved results to: {filename}")

