#!/usr/bin/env python
"""
Test script to analyze high-volume stocks for cyclical up/down patterns.

This script:
1. Fetches recent high-volume stocks
2. Analyzes 3-6 months of historical data
3. Detects cyclical patterns (support/resistance, cycle periods)
4. Identifies stocks currently near support (potential buy signals)
5. Outputs results for review

Usage:
    python test_cyclical_patterns.py
    python test_cyclical_patterns.py --limit 50 --lookback 180
"""

import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from yfinance.screener import EquityQuery as YfEquityQuery
from collections import defaultdict


def get_high_volume_stocks(limit=100, min_price=1.0, historical_date=None):
    """
    Get most active (high volume) stocks from Yahoo Finance.
    If historical_date is provided, gets stocks based on volume at that date (for backtesting).
    Otherwise, gets current high-volume stocks.
    
    Args:
        limit: Maximum number of stocks to return
        min_price: Minimum stock price filter
        historical_date: Date to check historical volume (datetime). If None, uses current volume.
    
    Returns:
        List of stock symbols sorted by volume (highest first)
    """
    if historical_date:
        print(f"📊 Fetching {limit} high-volume stocks based on volume at {historical_date.strftime('%Y-%m-%d')}...")
        return get_historical_high_volume_stocks(limit, min_price, historical_date)
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
        return []  # Return empty instead of fallback list


def get_historical_high_volume_stocks(limit=100, min_price=1.0, historical_date=None):
    """
    Get high-volume stocks based on historical volume at a specific date.
    This is for proper backtesting without look-ahead bias.
    
    Args:
        limit: Maximum number of stocks to return
        min_price: Minimum stock price filter
        historical_date: Date to check historical volume (datetime)
    
    Returns:
        List of stock symbols sorted by historical volume (highest first)
    """
    # Get a broad list of candidate stocks (use current screener as starting point)
    # We'll check their historical volume
    print(f"   Getting candidate stocks...")
    try:
        most_active_query = YfEquityQuery(
            "and",
            [
                YfEquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
                YfEquityQuery("gt", ["intradayprice", min_price]),
            ]
        )
        
        # Get more candidates than we need (we'll filter by historical volume)
        max_size = min(limit * 5, 500)
        response = yf.screen(
            most_active_query,
            offset=0,
            size=max_size,
            sortField="intradayprice",
            sortAsc=True,
        )
        
        quotes = response.get("quotes", [])
        candidate_symbols = []
        for quote in quotes:
            symbol = quote.get('symbol')
            if symbol:
                candidate_symbols.append(symbol)
    except Exception as e:
        print(f"   ⚠️  Screener failed: {e}")
        print("   ❌ Cannot retrieve candidate stocks - screener unavailable")
        return []  # Return empty instead of fallback list
    
    print(f"   Checking historical volume for {len(candidate_symbols)} candidates at {historical_date.strftime('%Y-%m-%d')}...")
    
    # Check historical volume for each candidate
    stocks_with_volume = []
    date_str = historical_date.strftime('%Y-%m-%d')
    
    # Get data for a range around the date (in case market was closed)
    start_date = historical_date - timedelta(days=5)
    end_date = historical_date + timedelta(days=1)
    
    for i, symbol in enumerate(candidate_symbols, 1):
        if i % 20 == 0:
            print(f"   Progress: {i}/{len(candidate_symbols)}...", end='\r', flush=True)
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                              end=end_date.strftime('%Y-%m-%d'),
                              interval="1d")
            
            if df.empty:
                continue
            
            # Find the closest trading day to historical_date
            # Get the row closest to our target date
            target_idx = None
            min_diff = None
            for idx in df.index:
                if hasattr(idx, 'date'):
                    idx_date = idx.date()
                elif hasattr(idx, 'to_pydatetime'):
                    idx_date = idx.to_pydatetime().date()
                else:
                    continue
                
                diff = abs((idx_date - historical_date.date()).days)
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    target_idx = idx
            
            if target_idx is None:
                continue
            
            # Get volume and price at that date
            volume = df.loc[target_idx, 'Volume']
            price = df.loc[target_idx, 'Close']
            
            if price >= min_price and volume > 0:
                stocks_with_volume.append({
                    'symbol': symbol,
                    'volume': float(volume),
                    'price': float(price),
                })
        except Exception:
            # Skip stocks that fail
            continue
    
    print(f"\n   ✓ Checked {len(candidate_symbols)} candidates, found {len(stocks_with_volume)} with valid data")
    
    # Sort by historical volume (highest first)
    stocks_with_volume.sort(key=lambda x: x['volume'], reverse=True)
    symbols = [s['symbol'] for s in stocks_with_volume[:limit]]
    
    print(f"✓ Retrieved {len(symbols)} high-volume stocks (based on {historical_date.strftime('%Y-%m-%d')} volume)")
    return symbols


def find_local_extrema(df, window=20):
    """
    Find local peaks (maxima) and troughs (minima) in price data.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size for finding extrema
    
    Returns:
        peaks: List of indices where local peaks occur
        troughs: List of indices where local troughs occur
    """
    peaks = []
    troughs = []
    
    for i in range(window, len(df) - window):
        # Check for local peak (maximum in window)
        if df['Close'].iloc[i] == df['Close'].iloc[i-window:i+window].max():
            peaks.append(i)
        
        # Check for local trough (minimum in window)
        if df['Close'].iloc[i] == df['Close'].iloc[i-window:i+window].min():
            troughs.append(i)
    
    return peaks, troughs


def calculate_support_resistance(df, peaks, troughs):
    """
    Calculate support and resistance levels from peaks and troughs.
    
    Args:
        df: DataFrame with price data
        peaks: List of peak indices
        troughs: List of trough indices
    
    Returns:
        support_price: Average of trough prices
        resistance_price: Average of peak prices
    """
    if len(troughs) < 2:
        return None, None
    
    if len(peaks) < 2:
        return None, None
    
    # Use recent peaks/troughs (last 6 months)
    recent_troughs = troughs[-min(5, len(troughs)):]
    recent_peaks = peaks[-min(5, len(peaks)):]
    
    support = df['Low'].iloc[recent_troughs].mean()
    resistance = df['High'].iloc[recent_peaks].mean()
    
    return support, resistance


def detect_cycle_period(peaks, troughs):
    """
    Detect the average cycle period (time between peaks or troughs).
    
    Args:
        peaks: List of peak indices
        troughs: List of trough indices
    
    Returns:
        avg_cycle_days: Average days between cycles
        cycle_consistency: Score 0-1 (higher = more consistent)
        num_cycles: Number of complete cycles detected
    """
    if len(peaks) < 3:
        return None, 0.0, 0
    
    # Calculate time between peaks
    peak_periods = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
    avg_peak_period = np.mean(peak_periods)
    
    # Calculate consistency (inverse of coefficient of variation)
    if avg_peak_period > 0:
        std_dev = np.std(peak_periods)
        consistency = 1.0 / (1.0 + (std_dev / avg_peak_period))
    else:
        consistency = 0.0
    
    num_cycles = len(peak_periods)
    
    return avg_peak_period, consistency, num_cycles


def analyze_volume_pattern(df, peaks, troughs):
    """
    Analyze volume patterns during peaks vs troughs.
    
    Args:
        df: DataFrame with OHLCV data
        peaks: List of peak indices
        troughs: List of trough indices
    
    Returns:
        volume_ratio: Average volume at peaks / average volume at troughs
    """
    if len(peaks) < 2 or len(troughs) < 2:
        return 1.0
    
    peak_volumes = df['Volume'].iloc[peaks].mean()
    trough_volumes = df['Volume'].iloc[troughs].mean()
    
    if trough_volumes > 0:
        return peak_volumes / trough_volumes
    return 1.0


def analyze_cyclical_pattern(symbol, lookback_days=180, window=20, end_date=None, min_cycles=3, buy_threshold=0.2):
    """
    Analyze a stock for cyclical up/down patterns.
    
    Args:
        symbol: Stock symbol
        lookback_days: Days of history to analyze
        window: Window size for finding extrema
        end_date: Historical end date for analysis (datetime). If None, uses today.
    
    Returns:
        dict with pattern analysis results
    """
    try:
        ticker = yf.Ticker(symbol)
        
        if end_date:
            # Calculate start date from end_date
            start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer
            # Use start/end dates for historical analysis
            df = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                              end=end_date.strftime('%Y-%m-%d'), 
                              interval="1d")
        else:
            # Use period for current analysis
            df = ticker.history(period=f"{lookback_days}d", interval="1d")
        
        if df.empty or len(df) < window * 2:
            return {'symbol': symbol, 'error': 'Insufficient data'}
        
        # Get price at analysis date and volume info
        analysis_price = df['Close'].iloc[-1]  # Price at end_date (or today)
        avg_volume = df['Volume'].tail(30).mean()
        
        # Store the analysis date for reference
        if end_date:
            analysis_date = end_date
        else:
            # Get date from last index
            last_idx = df.index[-1]
            if hasattr(last_idx, 'date'):
                analysis_date = last_idx.date() if hasattr(last_idx, 'date') else last_idx
            elif hasattr(last_idx, 'to_pydatetime'):
                analysis_date = last_idx.to_pydatetime().date()
            else:
                analysis_date = datetime.now().date()
        
        # Find local extrema
        peaks, troughs = find_local_extrema(df, window=window)
        
        if len(peaks) < min_cycles or len(troughs) < min_cycles:
            return {
                'symbol': symbol,
                'has_pattern': False,
                'reason': f'Insufficient cycles (need at least {min_cycles} peaks/troughs)',
                'analysis_price': analysis_price,
                'peaks': len(peaks),
                'troughs': len(troughs),
            }
        
        # Calculate support/resistance
        support, resistance = calculate_support_resistance(df, peaks, troughs)
        
        if support is None or resistance is None or support >= resistance:
            return {
                'symbol': symbol,
                'has_pattern': False,
                'reason': 'Invalid support/resistance levels',
                'analysis_price': analysis_price,
            }
        
        # Calculate cycle period
        avg_cycle_days, consistency, num_cycles = detect_cycle_period(peaks, troughs)
        
        # Analyze volume pattern
        volume_ratio = analyze_volume_pattern(df, peaks, troughs)
        
        # Determine position in cycle at analysis date
        price_range = resistance - support
        if price_range > 0:
            position_pct = (analysis_price - support) / price_range
        else:
            position_pct = 0.5
        
        if position_pct < buy_threshold:
            position = 'near_support'
            signal = 'BUY'
        elif position_pct > 0.8:
            position = 'near_resistance'
            signal = 'SELL'
        else:
            position = 'middle'
            signal = 'HOLD'
        
        # Calculate pattern score (0-1)
        # Factors: cycle consistency, number of cycles, clear support/resistance
        pattern_score = (
            consistency * 0.4 +  # Cycle consistency
            min(num_cycles / 10.0, 1.0) * 0.3 +  # Number of cycles
            min((resistance - support) / analysis_price, 0.3) * 0.3  # Price range size
        )
        
        # Calculate potential upside/downside
        upside_pct = ((resistance - analysis_price) / analysis_price) * 100
        downside_pct = ((analysis_price - support) / analysis_price) * 100
        
        result = {
            'symbol': symbol,
            'has_pattern': True,
            'analysis_price': round(analysis_price, 2),
            'analysis_date': analysis_date,
            'support_price': round(support, 2),
            'resistance_price': round(resistance, 2),
            'price_range_pct': round((price_range / analysis_price) * 100, 1),
            'position': position,
            'position_pct': round(position_pct * 100, 1),
            'signal': signal,
            'cycle_days': round(avg_cycle_days, 1) if avg_cycle_days is not None else None,
            'cycle_consistency': round(consistency, 2),
            'num_cycles': num_cycles,
            'volume_ratio': round(volume_ratio, 2),
            'pattern_score': round(pattern_score, 2),
            'upside_pct': round(upside_pct, 1),
            'downside_pct': round(downside_pct, 1),
            'avg_volume': int(avg_volume) if avg_volume is not None else 0,
        }
        
        # For backward compatibility, also include 'current_price'
        result['current_price'] = result['analysis_price']
        
        return result
        
    except Exception as e:
        return {
            'symbol': symbol,
            'error': str(e),
            'has_pattern': False
        }


def simulate_sell_instructions(symbol, buy_price, buy_date, support_price, resistance_price, current_price=None, current_date=None, descending_trend_threshold=None, descending_trend_lookback=None, after_days=None, peak_passed_threshold=None, stop_loss=None):
    """
    Simulate sell instructions and check if any would have triggered.
    If current_price is None, fetches it from yfinance.
    
    Args:
        symbol: Stock symbol
        buy_price: Price when bought
        buy_date: Date when bought
        support_price: Support level (for STOP_PRICE)
        resistance_price: Resistance level (unused, kept for compatibility)
        current_price: Current price (None to fetch)
        current_date: Current date (None uses today)
        descending_trend_threshold: DESCENDING_TREND threshold - sell if price drops this %% from average (e.g., -5.0 = 5%%, None = disabled)
        descending_trend_lookback: DESCENDING_TREND lookback days (e.g., 7 = 7-day average, None = disabled)
        after_days: Force sell after this many days (None = no AFTER_DAYS sell instruction)
        peak_passed_threshold: PEAK_PASSED threshold - sell if price drops this %% from peak (None = disabled)
        stop_loss: Stop loss multiplier relative to buy price (e.g., 0.95 = 5%% below, 0.78 = 22%% below, None = disabled)
    
    Returns:
        dict with sell simulation results
    """
    if current_date is None:
        current_date = datetime.now()
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get daily prices from buy_date to current_date
        days_diff = (current_date - buy_date).days
        if days_diff <= 0:
            days_diff = 0
        
        # Fetch historical data
        df = ticker.history(start=buy_date.strftime('%Y-%m-%d'), 
                          end=(current_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                          interval="1d")
        
        if df.empty or len(df) < 2:
            # No historical data, get current price
            if current_price is None:
                current_info = ticker.fast_info
                current_price = current_info.get('lastPrice') or current_info.get('regularMarketPrice')
                if current_price:
                    current_price = float(current_price)
                else:
                    current_price = buy_price  # Fallback
            
            return {
                'sold': False,
                'sell_reason': None,
                'sell_price': current_price,
                'sell_date': current_date.date() if hasattr(current_date, 'date') else current_date,
                'days_held': days_diff,
                'actual_return_pct': ((current_price - buy_price) / buy_price) * 100,
            }
        
        # Calculate sell instruction levels
        # Stop loss: Configurable multiplier relative to buy price (e.g., 0.95 = 5% below, 0.78 = 22% below)
        stop_price = None
        if stop_loss is not None:
            stop_price = buy_price * stop_loss  # Stop price relative to buy price
        
        # Check each day to see if sell instruction triggers
        for i, (date, row) in enumerate(df.iterrows()):
            daily_low = row['Low']
            daily_close = row['Close']
            
            # Priority 1: STOP_PRICE (check low of day)
            # STOP_PRICE triggers when price falls below stop level (configurable % below buy price)
            if stop_price is not None and daily_low <= stop_price:
                # When stop triggers, sell at stop price (or worse if close is lower)
                # We should never sell at a gain when stop loss triggers
                sell_price = min(stop_price, daily_close)
                date_naive = date.to_pydatetime().replace(tzinfo=None) if hasattr(date, 'to_pydatetime') else date
                if hasattr(date_naive, 'tzinfo') and date_naive.tzinfo:
                    date_naive = date_naive.replace(tzinfo=None)
                return {
                    'sold': True,
                    'sell_reason': 'STOP_PRICE',
                    'sell_price': sell_price,
                    'sell_date': date_naive.date() if hasattr(date_naive, 'date') else date_naive,
                    'days_held': (date_naive - buy_date).days if hasattr(date_naive, '__sub__') else i,
                    'actual_return_pct': ((sell_price - buy_price) / buy_price) * 100,
                }
            
            # Priority 2: PEAK_PASSED (check if price has dropped from peak)
            # Track the highest price (peak) since purchase, trigger if current price drops X% from peak
            if peak_passed_threshold is not None:
                # Calculate peak price up to current day
                peak_price = max(buy_price, df['High'].iloc[:i+1].max())
                
                if peak_price > 0:
                    drop_from_peak_pct = ((daily_close - peak_price) / peak_price) * 100
                    # Trigger if price has dropped by threshold percentage from peak
                    if drop_from_peak_pct <= -peak_passed_threshold:
                        sell_price = daily_close
                        date_naive = date.to_pydatetime().replace(tzinfo=None) if hasattr(date, 'to_pydatetime') else date
                        if hasattr(date_naive, 'tzinfo') and date_naive.tzinfo:
                            date_naive = date_naive.replace(tzinfo=None)
                        return {
                            'sold': True,
                            'sell_reason': 'PEAK_PASSED',
                            'sell_price': sell_price,
                            'sell_date': date_naive.date() if hasattr(date_naive, 'date') else date_naive,
                            'days_held': (date_naive - buy_date).days if hasattr(date_naive, '__sub__') else i,
                            'actual_return_pct': ((sell_price - buy_price) / buy_price) * 100,
                        }
            
            # Priority 3: DESCENDING_TREND (check if price drops significantly from recent average)
            # Only active if both threshold and lookback parameters are provided
            if descending_trend_threshold is not None and descending_trend_lookback is not None:
                lookback_days = descending_trend_lookback
                # Need at least (lookback_days - 1) days of data to calculate average
                if i >= (lookback_days - 1):
                    # Calculate average of last N days (where N = lookback_days)
                    start_idx = max(0, i - lookback_days + 1)
                    recent_avg = df['Close'].iloc[start_idx:i+1].mean()
                    if recent_avg > 0:
                        trend_pct = ((daily_close - recent_avg) / recent_avg) * 100
                        # Trigger if price drops by threshold percentage from recent average
                        if trend_pct <= descending_trend_threshold:
                            sell_price = daily_close
                            date_naive = date.to_pydatetime().replace(tzinfo=None) if hasattr(date, 'to_pydatetime') else date
                            if hasattr(date_naive, 'tzinfo') and date_naive.tzinfo:
                                date_naive = date_naive.replace(tzinfo=None)
                            return {
                                'sold': True,
                                'sell_reason': 'DESCENDING_TREND',
                                'sell_price': sell_price,
                                'sell_date': date_naive.date() if hasattr(date_naive, 'date') else date_naive,
                                'days_held': (date_naive - buy_date).days if hasattr(date_naive, '__sub__') else i,
                                'actual_return_pct': ((sell_price - buy_price) / buy_price) * 100,
                            }
            
            # Priority 4: NOT_TRENDING (check if stock is still high-volume/trending)
            # Stock is "not trending" if volume drops significantly
            try:
                # Get volume for current day
                daily_volume = row['Volume']
                
                # Need at least 3 days of data to compare
                if i >= 2:
                    # Calculate average volume over last 5 days (or available days)
                    lookback_days = min(5, i + 1)
                    avg_volume = df['Volume'].iloc[max(0, i - lookback_days + 1):i + 1].mean()
                    
                    # Stock is "not trending" if current volume is very low
                    # Threshold: less than 20% of recent average, or absolute minimum
                    volume_threshold = avg_volume * 0.2  # 20% of average
                    min_volume = 100000  # Absolute minimum (100k shares)
                    
                    if daily_volume < max(volume_threshold, min_volume) and avg_volume > 0:
                        sell_price = daily_close
                        date_naive = date.to_pydatetime().replace(tzinfo=None) if hasattr(date, 'to_pydatetime') else date
                        if hasattr(date_naive, 'tzinfo') and date_naive.tzinfo:
                            date_naive = date_naive.replace(tzinfo=None)
                        return {
                            'sold': True,
                            'sell_reason': 'NOT_TRENDING',
                            'sell_price': sell_price,
                            'sell_date': date_naive.date() if hasattr(date_naive, 'date') else date_naive,
                            'days_held': (date_naive - buy_date).days if hasattr(date_naive, '__sub__') else i,
                            'actual_return_pct': ((sell_price - buy_price) / buy_price) * 100,
                        }
            except Exception:
                # If we can't check volume, don't trigger this sell
                pass
            
            # Priority 5: AFTER_DAYS (force sell after specified days, if parameter provided)
            if after_days is not None:
                date_naive = date.to_pydatetime().replace(tzinfo=None) if hasattr(date, 'to_pydatetime') else date
                if hasattr(date_naive, 'tzinfo') and date_naive.tzinfo:
                    date_naive = date_naive.replace(tzinfo=None)
                days_held = (date_naive - buy_date).days if hasattr(date_naive, '__sub__') else i
                
                if days_held >= after_days:
                    sell_price = daily_close
                    return {
                        'sold': True,
                        'sell_reason': 'AFTER_DAYS',
                        'sell_price': sell_price,
                        'sell_date': date_naive.date() if hasattr(date_naive, 'date') else date_naive,
                        'days_held': days_held,
                        'actual_return_pct': ((sell_price - buy_price) / buy_price) * 100,
                    }
        
        # No sell instruction triggered - still holding, use current price
        if current_price is None:
            current_info = ticker.fast_info
            current_price = current_info.get('lastPrice') or current_info.get('regularMarketPrice')
            if current_price:
                current_price = float(current_price)
            else:
                current_price = float(df['Close'].iloc[-1])  # Use last available price
        
        return {
            'sold': False,
            'sell_reason': None,
            'sell_price': current_price,
            'sell_date': current_date.date() if hasattr(current_date, 'date') else current_date,
            'days_held': days_diff,
            'actual_return_pct': ((current_price - buy_price) / buy_price) * 100,
        }
        
    except Exception as e:
        # Error, use current price if available
        if current_price is None:
            try:
                ticker = yf.Ticker(symbol)
                current_info = ticker.fast_info
                current_price = current_info.get('lastPrice') or current_info.get('regularMarketPrice')
                if current_price:
                    current_price = float(current_price)
                else:
                    current_price = buy_price
            except:
                current_price = buy_price
        
        return {
            'sold': False,
            'sell_reason': f'Error: {str(e)[:30]}',
            'sell_price': current_price,
            'sell_date': current_date.date() if hasattr(current_date, 'date') else current_date,
            'days_held': days_diff if 'days_diff' in locals() else 0,
            'actual_return_pct': ((current_price - buy_price) / buy_price) * 100,
        }


def check_current_prices(buy_signals, after_days=None, peak_passed_threshold=None, descending_trend_threshold=None, descending_trend_lookback=None, stop_loss=None):
    """
    Simulate sell instructions for stocks that had BUY signals at historical date.
    Each stock exits when a sell instruction triggers (STOP_PRICE, PEAK_PASSED, DESCENDING_TREND, NOT_TRENDING, AFTER_DAYS).
    No fixed holding period - stocks exit based on sell instructions or current date.
    
    Args:
        buy_signals: List of dicts with BUY signals from historical analysis
        after_days: Force sell after this many days (None = no AFTER_DAYS sell instruction)
        peak_passed_threshold: PEAK_PASSED threshold - sell if price drops this %% from peak (None = disabled)
        descending_trend_threshold: DESCENDING_TREND threshold - sell if price drops this %% from average (e.g., -5.0 = 5%%, None = disabled)
        descending_trend_lookback: DESCENDING_TREND lookback days - number of days to average (e.g., 7 = 7-day average, None = disabled)
        stop_loss: Stop loss multiplier relative to buy price (e.g., 0.95 = 5%% below, 0.78 = 22%% below, None = disabled)
    
    Returns:
        List of dicts with prices and returns added
    """
    print(f"\n🔍 Simulating sell instructions for {len(buy_signals)} BUY signals...")
    
    results = []
    for i, signal in enumerate(buy_signals, 1):
        symbol = signal['symbol']
        historical_price = signal['analysis_price']
        historical_date = signal.get('analysis_date')
        support = signal.get('support_price', historical_price * 0.9)
        resistance = signal.get('resistance_price', historical_price * 1.5)
        
        print(f"  [{i}/{len(buy_signals)}] {symbol}...", end=' ', flush=True)
        
        try:
            # Convert analysis_date to datetime
            if historical_date:
                if isinstance(historical_date, datetime):
                    buy_date = historical_date.replace(tzinfo=None) if historical_date.tzinfo else historical_date
                elif hasattr(historical_date, 'date'):
                    buy_date = datetime.combine(historical_date.date(), datetime.min.time())
                else:
                    buy_date = datetime.now() - timedelta(weeks=2)
            else:
                buy_date = datetime.now() - timedelta(weeks=2)
            
            # Use current date (or today) - no fixed holding period
            # Stocks will exit based on sell instructions (STOP_PRICE, PEAK_PASSED, DESCENDING_TREND, NOT_TRENDING, AFTER_DAYS)
            sell_date = datetime.now()
            
            # Simulate sell instructions
            sell_sim = simulate_sell_instructions(
                symbol=symbol,
                buy_price=historical_price,
                buy_date=buy_date,
                support_price=support,
                resistance_price=resistance,
                current_price=None,  # Will be fetched in function
                current_date=sell_date,  # Check until current date
                descending_trend_threshold=descending_trend_threshold,
                descending_trend_lookback=descending_trend_lookback,
                after_days=after_days,
                peak_passed_threshold=peak_passed_threshold,
                stop_loss=stop_loss
            )
            
            # Store results
            signal['current_price'] = round(sell_sim['sell_price'], 2)
            signal['return_pct'] = round(sell_sim['actual_return_pct'], 1)
            signal['return_abs'] = round(sell_sim['sell_price'] - historical_price, 2)
            signal['sold'] = sell_sim['sold']
            signal['sell_reason'] = sell_sim['sell_reason']
            signal['sell_date'] = sell_sim['sell_date']
            signal['days_held'] = sell_sim['days_held']
            
            status = f"sold ({sell_sim['sell_reason']})" if sell_sim['sold'] else "holding"
            print(f"✓ ${sell_sim['sell_price']:.2f} ({sell_sim['actual_return_pct']:+.1f}%, {sell_sim['days_held']}d - {status})")
            
        except Exception as e:
            signal['current_price'] = None
            signal['return_pct'] = None
            signal['return_abs'] = None
            signal['sold'] = False
            signal['sell_reason'] = f'Error: {str(e)[:30]}'
            signal['days_held'] = None
            print(f"✗ Error: {str(e)[:30]}")
        
        results.append(signal)
    
    return results


def print_backtest_results(backtest_results):
    """Print backtest results showing historical BUY signals vs current prices."""
    
    # Filter out stocks without current price data
    valid_results = [r for r in backtest_results if r.get('current_price') is not None]
    
    if not valid_results:
        print("\n❌ No valid current price data found")
        return
    
    # Sort by return percentage (best performers first)
    valid_results.sort(key=lambda x: x.get('return_pct', -999), reverse=True)
    
    print("\n" + "="*120)
    print("BACKTEST RESULTS: Historical BUY Signals vs Current Prices")
    print("="*120)
    
    # Summary statistics
    returns = [r['return_pct'] for r in valid_results if r.get('return_pct') is not None]
    winners = [r for r in valid_results if r.get('return_pct', 0) > 0]
    losers = [r for r in valid_results if r.get('return_pct', 0) <= 0]
    sold_stocks = [r for r in valid_results if r.get('sold', False)]
    
    # Average days held
    days_held_list = [r.get('days_held', 0) for r in valid_results if r.get('days_held') is not None]
    avg_days_held = sum(days_held_list) / len(days_held_list) if days_held_list else 0
    
    # Count by exit reason
    stop_price_exits = [r for r in sold_stocks if r.get('sell_reason') == 'STOP_PRICE']
    peak_passed_exits = [r for r in sold_stocks if r.get('sell_reason') == 'PEAK_PASSED']
    descending_trend_exits = [r for r in sold_stocks if r.get('sell_reason') == 'DESCENDING_TREND']
    not_trending_exits = [r for r in sold_stocks if r.get('sell_reason') == 'NOT_TRENDING']
    after_days_exits = [r for r in sold_stocks if r.get('sell_reason') == 'AFTER_DAYS']
    
    avg_return = sum(returns) / len(returns) if returns else 0
    win_rate = (len(winners) / len(valid_results) * 100) if valid_results else 0
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total BUY signals tested: {len(backtest_results)}")
    print(f"  Valid price data: {len(valid_results)}")
    print(f"  Stocks sold: {len(sold_stocks)} ({len(sold_stocks)/len(valid_results)*100:.1f}%)")
    print(f"    - STOP_PRICE: {len(stop_price_exits)}")
    if peak_passed_exits:
        print(f"    - PEAK_PASSED: {len(peak_passed_exits)}")
    print(f"    - DESCENDING_TREND: {len(descending_trend_exits)}")
    print(f"    - NOT_TRENDING: {len(not_trending_exits)}")
    if after_days_exits:
        print(f"    - AFTER_DAYS: {len(after_days_exits)}")
    print(f"  Still holding: {len(valid_results) - len(sold_stocks)}")
    print(f"  Winners: {len(winners)} ({len(winners)/len(valid_results)*100:.1f}%)")
    print(f"  Losers: {len(losers)} ({len(losers)/len(valid_results)*100:.1f}%)")
    print(f"  Average return: {avg_return:+.1f}%")
    print(f"  Average days held: {avg_days_held:.1f} days")
    print(f"  Win rate: {win_rate:.1f}%")
    
    if winners:
        best_return = max(returns)
        print(f"  Best performer: {best_return:+.1f}%")
    if losers:
        worst_return = min(returns)
        print(f"  Worst performer: {worst_return:+.1f}%")
    
    # Detailed results table
    print("\n" + "="*120)
    print("DETAILED RESULTS")
    print("="*120)
    print(f"\n{'Symbol':<8} {'Buy Price':<12} {'Exit Price':<12} {'Return%':<10} {'Return$':<10} "
          f"{'Exit Reason':<18} {'Days Held':<10}")
    print("-"*120)
    
    for r in valid_results:
        buy_price = r.get('analysis_price', 0)
        exit_price = r.get('current_price', 0)
        return_pct = r.get('return_pct', 0)
        return_abs = r.get('return_abs', 0)
        days_held = r.get('days_held', 0)
        sold = r.get('sold', False)
        sell_reason = r.get('sell_reason', 'Holding')
        
        return_str = f"{return_pct:+.1f}%" if return_pct else "N/A"
        exit_reason = sell_reason if sold else "Holding"
        
        print(f"{r['symbol']:<8} ${buy_price:>10.2f} ${exit_price:>10.2f} "
              f"{return_str:>9} ${return_abs:>9.2f} {exit_reason:<18} {days_held:>9}d")
    
    print("\n" + "="*120)


def print_results(results):
    """Print analysis results in a readable format."""
    
    # Separate stocks with patterns from those without
    with_patterns = [r for r in results if r.get('has_pattern', False)]
    without_patterns = [r for r in results if not r.get('has_pattern', False) and 'error' not in r]
    errors = [r for r in results if 'error' in r]
    
    print("\n" + "="*120)
    print("CYCLICAL PATTERN ANALYSIS RESULTS")
    print("="*120)
    print(f"\nTotal stocks analyzed: {len(results)}")
    print(f"  ✓ Stocks with patterns: {len(with_patterns)}")
    print(f"  ✗ Stocks without patterns: {len(without_patterns)}")
    print(f"  ⚠️  Errors: {len(errors)}")
    
    if with_patterns:
        # Sort by pattern score (highest first)
        with_patterns.sort(key=lambda x: x.get('pattern_score', 0), reverse=True)
        
        print("\n" + "="*120)
        print("STOCKS WITH CYCLICAL PATTERNS (Sorted by Pattern Score)")
        print("="*120)
        print(f"\n{'Symbol':<8} {'Price':<8} {'Support':<8} {'Resist':<8} {'Pos%':<6} {'Signal':<6} "
              f"{'Cycle':<7} {'Consist':<7} {'Cycles':<7} {'Score':<6} {'Upside%':<9} {'Vol':<12}")
        print("-"*120)
        
        for r in with_patterns:
            cycle_days_str = f"{r['cycle_days']:>6.1f}d" if r.get('cycle_days') is not None else "   N/A"
            print(f"{r['symbol']:<8} ${r['current_price']:>7.2f} ${r['support_price']:>7.2f} "
                  f"${r['resistance_price']:>7.2f} {r['position_pct']:>5.1f}% {r['signal']:<6} "
                  f"{cycle_days_str} {r['cycle_consistency']:>6.2f} {r['num_cycles']:>6} "
                  f"{r['pattern_score']:>5.2f} {r['upside_pct']:>+8.1f}% {r['avg_volume']:>11,}")
        
        # Show buy signals separately
        buy_signals = [r for r in with_patterns if r.get('signal') == 'BUY']
        if buy_signals:
            print("\n" + "="*120)
            print(f"BUY SIGNALS ({len(buy_signals)} stocks near support)")
            print("="*120)
            print(f"\n{'Symbol':<8} {'Price':<8} {'Support':<8} {'Resist':<8} {'Upside%':<9} "
                  f"{'Cycle':<7} {'Score':<6} {'Reason':<30}")
            print("-"*120)
            
            for r in sorted(buy_signals, key=lambda x: x.get('pattern_score', 0), reverse=True):
                reason = f"Near support ({r['position_pct']:.1f}%), {r['num_cycles']} cycles"
                cycle_days_str = f"{r['cycle_days']:>6.1f}d" if r.get('cycle_days') is not None else "   N/A"
                print(f"{r['symbol']:<8} ${r['current_price']:>7.2f} ${r['support_price']:>7.2f} "
                      f"${r['resistance_price']:>7.2f} {r['upside_pct']:>+8.1f}% "
                      f"{cycle_days_str} {r['pattern_score']:>5.2f} {reason:<30}")
    
    if without_patterns:
        print("\n" + "="*120)
        print(f"STOCKS WITHOUT CLEAR PATTERNS ({len(without_patterns)} stocks)")
        print("="*120)
        for r in without_patterns[:10]:  # Show first 10
            reason = r.get('reason', 'Unknown')
            print(f"  {r['symbol']:<8} - {reason}")
        if len(without_patterns) > 10:
            print(f"  ... and {len(without_patterns) - 10} more")
    
    if errors:
        print("\n" + "="*120)
        print(f"ERRORS ({len(errors)} stocks)")
        print("="*120)
        for r in errors[:5]:  # Show first 5
            symbol = r.get('symbol', 'UNKNOWN')
            error_msg = r.get('error', 'Unknown error')
            print(f"  {symbol:<8} - {error_msg}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


def save_results_csv(results, filename=None):
    """Save results to CSV file."""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"cyclical_patterns_{timestamp}.csv"
    
    # Flatten results for CSV
    rows = []
    for r in results:
        analysis_date = r.get('analysis_date', '')
        if analysis_date and hasattr(analysis_date, 'strftime'):
            analysis_date = analysis_date.strftime('%Y-%m-%d')
        elif analysis_date and hasattr(analysis_date, '__str__'):
            analysis_date = str(analysis_date)
        
        row = {
            'symbol': r.get('symbol', ''),
            'has_pattern': r.get('has_pattern', False),
            'analysis_date': analysis_date,
            'analysis_price': r.get('analysis_price', r.get('current_price', '')),
            'current_price': r.get('current_price', ''),
            'return_pct': r.get('return_pct', ''),
            'return_abs': r.get('return_abs', ''),
            'support_price': r.get('support_price', ''),
            'resistance_price': r.get('resistance_price', ''),
            'position': r.get('position', ''),
            'position_pct': r.get('position_pct', ''),
            'signal': r.get('signal', ''),
            'cycle_days': r.get('cycle_days', ''),
            'cycle_consistency': r.get('cycle_consistency', ''),
            'num_cycles': r.get('num_cycles', ''),
            'pattern_score': r.get('pattern_score', ''),
            'upside_pct': r.get('upside_pct', ''),
            'downside_pct': r.get('downside_pct', ''),
            'volume_ratio': r.get('volume_ratio', ''),
            'avg_volume': r.get('avg_volume', ''),
            'error': r.get('error', ''),
            'reason': r.get('reason', ''),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to: {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(
        description='Analyze high-volume stocks for cyclical patterns'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=50,
        help='Number of high-volume stocks to analyze (default: 50)'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=180,
        help='Days of history to analyze (default: 180 = ~6 months)'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=20,
        help='Window size for finding peaks/troughs (default: 20 days)'
    )
    parser.add_argument(
        '--min-price',
        type=float,
        default=1.0,
        help='Minimum stock price filter (default: 1.0)'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save results to CSV file (default: auto-generate filename)'
    )
    parser.add_argument(
        '--historical-date',
        type=str,
        default=None,
        help='Historical end date for analysis (YYYY-MM-DD). If not provided, uses 2 weeks ago. Use "today" for current analysis.'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Enable backtest mode: analyze historical data and simulate sell instructions'
    )
    parser.add_argument(
        '--weeks-ago',
        type=int,
        default=None,
        help='Weeks ago to analyze data from (default: 2 if --backtest, ignored otherwise)'
    )
    parser.add_argument(
        '--min-cycles',
        type=int,
        default=3,
        help='Minimum number of cycles required (default: 3, use 2 for more candidates)'
    )
    parser.add_argument(
        '--buy-threshold',
        type=float,
        default=0.2,
        help='BUY signal threshold - position must be < this value (default: 0.2 = 20%%, use 0.3 for more candidates)'
    )
    parser.add_argument(
        '--after-days',
        type=int,
        default=None,
        help='Force sell after this many days (for testing purposes, default: None = no AFTER_DAYS sell instruction)'
    )
    parser.add_argument(
        '--peak-passed-threshold',
        type=float,
        default=None,
        help='PEAK_PASSED sell instruction threshold - sell if price drops this %% from peak (e.g., 15.0 = 15%%, default: None = disabled)'
    )
    parser.add_argument(
        '--descending-trend-threshold',
        type=float,
        default=None,
        help='DESCENDING_TREND sell instruction threshold - sell if price drops this %% from average (e.g., -5.0 = 5%%, default: None = disabled)'
    )
    parser.add_argument(
        '--descending-trend-lookback',
        type=int,
        default=None,
        help='DESCENDING_TREND lookback days - number of days to average for trend calculation (e.g., 7 = 7-day average, default: None = disabled)'
    )
    parser.add_argument(
        '--stop-loss',
        type=float,
        default=None,
        help='Stop loss multiplier relative to buy price (e.g., 0.95 = 5%% below buy price, 0.78 = 22%% below, default: None = disabled)'
    )
    
    args = parser.parse_args()
    
    # Determine analysis end date
    if args.historical_date:
        if args.historical_date.lower() == 'today':
            end_date = None  # Use current date
            is_backtest = False
        else:
            try:
                end_date = datetime.strptime(args.historical_date, '%Y-%m-%d')
                is_backtest = True
            except ValueError:
                print(f"❌ Invalid date format. Use YYYY-MM-DD")
                sys.exit(1)
    elif args.backtest:
        # Use weeks-ago parameter
        weeks_ago = args.weeks_ago if args.weeks_ago else 2
        # Find Monday of that week, then add 1 day (Tuesday) for more consistent testing
        target_date = datetime.now() - timedelta(weeks=weeks_ago)
        days_since_monday = target_date.weekday()  # 0=Monday, 1=Tuesday, etc.
        monday_of_week = target_date - timedelta(days=days_since_monday)
        end_date = monday_of_week # + timedelta(days=1)  # Tuesday of that week
        is_backtest = True
    else:
        end_date = None
        is_backtest = False
    
    print("="*120)
    if is_backtest:
        print("CYCLICAL PATTERN DETECTION - BACKTEST MODE")
        print(f"Analyzing data ending: {end_date.strftime('%Y-%m-%d')}")
        print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
    else:
        print("CYCLICAL PATTERN DETECTION TEST")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*120)
    print(f"Settings:")
    print(f"  Stock limit: {args.limit}")
    print(f"  Lookback period: {args.lookback} days (~{args.lookback//30} months)")
    print(f"  Window size: {args.window} days")
    print(f"  Min price: ${args.min_price:.2f}")
    print(f"  Min cycles required: {args.min_cycles}")
    print(f"  BUY threshold: {args.buy_threshold*100:.0f}% (stocks within {args.buy_threshold*100:.0f}% of support)")
    if is_backtest:
        print(f"  Analysis date: {end_date.strftime('%Y-%m-%d')}")
        print(f"  Sell date: {datetime.now().strftime('%Y-%m-%d')} (current date - no fixed holding period)")
        sell_instructions_str = ""
        if args.stop_loss is not None:
            stop_loss_pct = (1 - args.stop_loss) * 100
            sell_instructions_str = f"STOP_PRICE ({stop_loss_pct:.1f}% below buy price)"
        if args.peak_passed_threshold:
            if sell_instructions_str:
                sell_instructions_str += ", "
            sell_instructions_str += f"PEAK_PASSED ({args.peak_passed_threshold}% from peak)"
        if args.descending_trend_threshold is not None and args.descending_trend_lookback is not None:
            if sell_instructions_str:
                sell_instructions_str += ", "
            sell_instructions_str += f"DESCENDING_TREND ({abs(args.descending_trend_threshold)}% below {args.descending_trend_lookback}-day avg)"
        if args.after_days:
            if sell_instructions_str:
                sell_instructions_str += ", "
            sell_instructions_str += f"AFTER_DAYS ({args.after_days} days)"
        if not sell_instructions_str:
            sell_instructions_str = "NOT_TRENDING (low volume)"
        else:
            sell_instructions_str += ", NOT_TRENDING (low volume)"
        print(f"  Sell instructions: {sell_instructions_str}")
    print()
    
    # Step 1: Get high-volume stocks
    # For backtests, use historical volume at the backtest date (no look-ahead bias)
    historical_date_for_selection = end_date if is_backtest else None
    symbols = get_high_volume_stocks(limit=args.limit, min_price=args.min_price, historical_date=historical_date_for_selection)
    
    if not symbols:
        print("❌ No stocks retrieved")
        sys.exit(1)
    
    # Step 2: Analyze each stock
    print(f"\n🔍 Analyzing {len(symbols)} stocks for cyclical patterns...")
    if is_backtest:
        print(f"   (Using historical data ending {end_date.strftime('%Y-%m-%d')})")
    print("   (This may take a few minutes)")
    print()
    
    results = []
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] Analyzing {symbol}...", end=' ', flush=True)
        result = analyze_cyclical_pattern(
            symbol, 
            lookback_days=args.lookback, 
            window=args.window, 
            end_date=end_date,
            min_cycles=args.min_cycles,
            buy_threshold=args.buy_threshold
        )
        results.append(result)
        
        if result.get('has_pattern'):
            print(f"✓ Pattern found (score: {result.get('pattern_score', 0):.2f}, signal: {result.get('signal', 'N/A')})")
        elif 'error' in result:
            print(f"✗ Error: {result['error']}")
        else:
            print(f"✗ No clear pattern")
    
    # Step 3: Print results
    print_results(results)
    
    # Step 4: If backtest mode, check prices at check_date for BUY signals
    if is_backtest:
        buy_signals = [r for r in results if r.get('has_pattern', False) and r.get('signal') == 'BUY']
        
        if buy_signals:
            print(f"\n{'='*120}")
            print(f"BACKTEST: Found {len(buy_signals)} BUY signals from {end_date.strftime('%Y-%m-%d')}")
            print(f"Simulating sell instructions until current date ({datetime.now().strftime('%Y-%m-%d')})")
            sell_instructions = ""
            if args.stop_loss is not None:
                stop_loss_pct = (1 - args.stop_loss) * 100
                sell_instructions = f"STOP_PRICE ({stop_loss_pct:.1f}% below buy price)"
            if args.peak_passed_threshold:
                if sell_instructions:
                    sell_instructions += ", "
                sell_instructions += f"PEAK_PASSED ({args.peak_passed_threshold}% from peak)"
            if args.descending_trend_threshold is not None and args.descending_trend_lookback is not None:
                if sell_instructions:
                    sell_instructions += ", "
                sell_instructions += f"DESCENDING_TREND ({abs(args.descending_trend_threshold)}% below {args.descending_trend_lookback}-day avg)"
            if args.after_days:
                if sell_instructions:
                    sell_instructions += ", "
                sell_instructions += f"AFTER_DAYS ({args.after_days} days)"
            if not sell_instructions:
                sell_instructions = "NOT_TRENDING (low volume)"
            else:
                sell_instructions += ", NOT_TRENDING (low volume)"
            print(f"Sell instructions: {sell_instructions}")
            print(f"{'='*120}")
            
            # Simulate sell instructions for each BUY signal
            backtest_results = check_current_prices(
                buy_signals, 
                after_days=args.after_days, 
                peak_passed_threshold=args.peak_passed_threshold,
                descending_trend_threshold=args.descending_trend_threshold,
                descending_trend_lookback=args.descending_trend_lookback,
                stop_loss=args.stop_loss
            )
            
            # Print backtest results
            print_backtest_results(backtest_results)
            
            # Save backtest results
            if args.save or True:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backtest_filename = f"backtest_{timestamp}.csv"
                save_results_csv(backtest_results, backtest_filename)
        else:
            print(f"\n⚠️  No BUY signals found for backtesting")
    
    # Step 5: Save to CSV
    if args.save or True:  # Always save by default
        save_results_csv(results, args.save)
    
    print("\n" + "="*120)
    print("ANALYSIS COMPLETE")
    print("="*120)
    print("\n💡 INTERPRETATION:")
    print("   - Pattern Score: 0-1 (higher = more consistent/reliable pattern)")
    print("   - Cycle Days: Average days between peaks")
    print("   - Cycle Consistency: 0-1 (higher = more regular cycles)")
    print("   - Position %: Where current price is in the range (0% = at support, 100% = at resistance)")
    print("   - BUY signals: Stocks near support (bottom of range) - potential bounce")
    print("   - SELL signals: Stocks near resistance (top of range) - potential reversal")
    if is_backtest:
        print("\n   📊 BACKTEST: Results show actual returns from historical BUY signals")
    print("="*120)


if __name__ == "__main__":
    main()

