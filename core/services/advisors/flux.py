"""
Flux Advisor - Cyclical Pattern Detection

Discovers stocks with cyclical up/down patterns that are currently near support levels.
Uses pattern analysis to identify buy opportunities based on historical price cycles.
"""
import logging
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from yfinance.screener import EquityQuery as YfEquityQuery

from core.services.advisors.advisor import AdvisorBase, register

logger = logging.getLogger(__name__)

# Configuration
MIN_PRICE = 1.0
STOCK_LIMIT = 50
LOOKBACK_DAYS = 180
WINDOW_SIZE = 20
MIN_CYCLES = 3
BUY_THRESHOLD = 0.2  # 20% from support
STOP_LOSS_MULTIPLIER = 0.9  # 10% below buy price


def find_local_extrema(df, window=20):
    """Find local peaks and troughs in price data."""
    peaks = []
    troughs = []
    
    for i in range(window, len(df) - window):
        if df['Close'].iloc[i] == df['Close'].iloc[i-window:i+window].max():
            peaks.append(i)
        if df['Close'].iloc[i] == df['Close'].iloc[i-window:i+window].min():
            troughs.append(i)
    
    return peaks, troughs


def calculate_support_resistance(df, peaks, troughs):
    """Calculate support and resistance levels from peaks and troughs."""
    if len(troughs) < 2 or len(peaks) < 2:
        return None, None
    
    recent_troughs = troughs[-min(5, len(troughs)):]
    recent_peaks = peaks[-min(5, len(peaks)):]
    
    support = df['Low'].iloc[recent_troughs].mean()
    resistance = df['High'].iloc[recent_peaks].mean()
    
    return support, resistance


def detect_cycle_period(peaks, troughs):
    """Detect average cycle period and consistency."""
    if len(peaks) < 3:
        return None, 0.0, 0
    
    peak_periods = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
    avg_peak_period = np.mean(peak_periods)
    
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
    Matches backtest logic exactly.
    
    Args:
        symbol: Stock symbol
        lookback_days: Days of history to analyze
        window: Window size for finding extrema
        end_date: Historical end date for analysis (datetime). If None, uses today.
        min_cycles: Minimum number of cycles required
        buy_threshold: BUY signal threshold (position must be < this value)
    
    Returns:
        dict with pattern analysis results
    """
    try:
        ticker = yf.Ticker(symbol)
        
        if end_date:
            # Calculate start date from end_date (same as backtest)
            start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer
            # Use start/end dates for historical analysis
            df = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                              end=end_date.strftime('%Y-%m-%d'), 
                              interval="1d")
        else:
            # Use period for current analysis
            df = ticker.history(period=f"{lookback_days}d", interval="1d")
        
        if df.empty or len(df) < window * 2:
            return {'symbol': symbol, 'has_pattern': False, 'error': 'Insufficient data'}
        
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
        
        return {
            'symbol': symbol,
            'has_pattern': True,
            'analysis_price': round(analysis_price, 2),
            'support_price': round(support, 2),
            'resistance_price': round(resistance, 2),
            'position': position,
            'position_pct': round(position_pct * 100, 1),
            'signal': signal,
            'cycle_days': round(avg_cycle_days, 1) if avg_cycle_days is not None else None,
            'cycle_consistency': round(consistency, 2),
            'num_cycles': num_cycles,
            'pattern_score': round(pattern_score, 2),
            'upside_pct': round(upside_pct, 1),
            'downside_pct': round(downside_pct, 1),
            'volume_ratio': round(volume_ratio, 2),
        }
        
    except Exception as e:
        logger.debug(f"Error analyzing {symbol}: {e}")
        return {'symbol': symbol, 'has_pattern': False, 'error': str(e)}


def get_high_volume_stocks(limit=50, min_price=1.0):
    """
    Get high-volume stocks based on historical volume at today's date.
    Uses the same methodology as backtests for consistency.
    """
    # Use today's date as the historical date (for consistency with backtest methodology)
    historical_date = datetime.now()
    
    logger.info(f"Getting candidate stocks from screener...")
    try:
        most_active_query = YfEquityQuery(
            "and",
            [
                YfEquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
                YfEquityQuery("gt", ["intradayprice", min_price]),
            ]
        )
        
        # Get more candidates than we need (we'll filter by historical volume)
        # Same as backtest: limit * 5, max 500
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
        logger.warning(f"Screener failed: {e}")
        logger.warning("Cannot retrieve candidate stocks - screener unavailable")
        return []
    
    logger.info(f"Checking historical volume for {len(candidate_symbols)} candidates at {historical_date.strftime('%Y-%m-%d')}...")
    
    # Check historical volume for each candidate
    stocks_with_volume = []
    
    # Get data for a range around the date (in case market was closed)
    start_date = historical_date - timedelta(days=5)
    end_date = historical_date + timedelta(days=1)
    
    for i, symbol in enumerate(candidate_symbols, 1):
        if i % 20 == 0:
            logger.info(f"Progress: {i}/{len(candidate_symbols)}...")
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                              end=end_date.strftime('%Y-%m-%d'),
                              interval="1d")
            
            if df.empty:
                continue
            
            # Find the closest trading day to historical_date
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
        except Exception as e:
            # Skip stocks that fail
            logger.debug(f"Error checking {symbol}: {e}")
            continue
    
    logger.info(f"Checked {len(candidate_symbols)} candidates, found {len(stocks_with_volume)} with valid data")
    
    # Sort by historical volume (highest first)
    stocks_with_volume.sort(key=lambda x: x['volume'], reverse=True)
    symbols = [s['symbol'] for s in stocks_with_volume[:limit]]
    
    logger.info(f"Retrieved {len(symbols)} high-volume stocks (based on {historical_date.strftime('%Y-%m-%d')} volume)")
    return symbols


class Flux(AdvisorBase):
    """
    Flux Advisor - Discovers stocks with cyclical patterns near support levels.
    """
    
    def __init__(self, advisor):
        if isinstance(advisor, str):
            super().__init__(advisor)
            self.advisor = None
        else:
            super().__init__(advisor.name)
            self.advisor = advisor
    
    def discover(self, sa):
        """
        Discover stocks with cyclical patterns that are near support (BUY signals).
        """
        logger.info(f"{self.advisor.name if self.advisor else 'Flux'} starting discovery...")
        
        # Check if market is open - Flux buys are price sensitive, so don't discover when market is closed
        market_status = self.market_open()
        if market_status is None:
            logger.info("Market is closed (weekend or after hours) - skipping Flux discovery")
            return
        if market_status < 0:
            logger.info(f"Market not open yet (opens in {-market_status} minutes) - skipping Flux discovery")
            return
        
        # Get high-volume stocks
        symbols = get_high_volume_stocks(limit=STOCK_LIMIT, min_price=MIN_PRICE)
        
        if not symbols:
            logger.warning("No high-volume stocks found")
            return
        
        buy_signals = []
        
        # Analyze each stock for cyclical patterns
        for i, symbol in enumerate(symbols, 1):
            if i % 10 == 0:
                logger.info(f"Analyzing {i}/{len(symbols)} stocks...")
            
            result = analyze_cyclical_pattern(
                symbol,
                lookback_days=LOOKBACK_DAYS,
                window=WINDOW_SIZE,
                end_date=datetime.now(),  # Use today's date with 30-day buffer (matches backtest)
                min_cycles=MIN_CYCLES,
                buy_threshold=BUY_THRESHOLD
            )
            
            # Explicitly filter for BUY signals only (matching backtest filter)
            if result and result.get('has_pattern', False) and result.get('signal') == 'BUY':
                buy_signals.append(result)
        
        logger.info(f"Found {len(buy_signals)} BUY signals with cyclical patterns")
        
        # Sort by pattern score (best first)
        buy_signals.sort(key=lambda x: x.get('pattern_score', 0), reverse=True)
        
        # Create discoveries for top signals
        for signal in buy_signals:
            symbol = signal['symbol']

            # Check if already discovered - rediscover if >5 days ago OR price dropped to 90%
            if not self.allow_discovery(symbol, period=5 * 24, price_decline=0.9):
                continue

            price = signal['analysis_price']
            support = signal['support_price']
            resistance = signal['resistance_price']
            pattern_score = signal.get('pattern_score', 0)
            num_cycles = signal.get('num_cycles', 0)
            upside_pct = signal.get('upside_pct', 0)
            position_pct = signal.get('position_pct', 0)  # Already a percentage (0-100)
            
            explanation = (
                f"Cyclical pattern detected (score: {pattern_score:.2f}, {num_cycles} cycles). "
                f"Price ${price:.2f} near support ${support:.2f} ({position_pct:.1f}% position). "
                f"Resistance: ${resistance:.2f} (+{upside_pct:.1f}% upside potential)."
            )
            
            # Create sell instructions:
            # 1. STOP_PERCENTAGE: 10% below buy price (loss protection)
            # 2. TARGET_PRICE: resistance price (profit taking)
            # 3. NOT_TRENDING: exit if volume drops significantly (no value needed)
            # 4. END_WEEK: if up 5% after 7 days the following friday
            sell_instructions = [
                ("STOP_PERCENTAGE", Decimal(str(STOP_LOSS_MULTIPLIER)), None),
                ("TARGET_PRICE", Decimal(str(resistance)), None),
                ("NOT_TRENDING", None, None),
                ("END_WEEK", 1.05, None)
            ]
            
            # Weight based on pattern score (higher score = higher weight)
            weight = 1.0 # = float(pattern_score) * 2.0  # Scale to 0-2.0 range
            
            self.discovered(
                sa,
                symbol,
                explanation,
                sell_instructions,
                weight
            )


# Register the advisor
register("Flux", "Flux")

