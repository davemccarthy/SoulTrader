"""
Intraday Momentum Trading Advisor

Discovers stocks with intraday momentum signals during optimal trading hours.
Only available for EXPERIMENTAL risk users.

Discovery Window: 15:30-16:30 GMT (10:30-11:30 ET) - once per day
Entry Signals: Price > VWAP, EMA momentum, RSI < 75, Volume surge
Exit Instructions: STOP_LOSS and TARGET_PRICE based on ATR
"""

import logging
import time
from decimal import Decimal
from datetime import datetime, timedelta
from pytz import timezone as tz

import yfinance as yf
import pandas as pd
import numpy as np
from yfinance.screener import EquityQuery as YfEquityQuery

from core.services.advisors.advisor import AdvisorBase, register
from core.models import Discovery, SmartAnalysis

logger = logging.getLogger(__name__)

# Trading parameters
EMA_SHORT = 8
EMA_LONG = 21
RSI_PERIOD = 14
RSI_OVERBOUGHT = 75
VOL_MULTIPLIER = 1.5
SL_ATR_MULT = 2.0
TP_ATR_MULT = 3.0
MIN_MARKET_CAP = 100_000_000
MAX_STOCKS_TO_CHECK = 100  # Scan top 100 most active stocks for better coverage

# Discovery window: 15:30-16:30 GMT = 10:30-11:30 ET
DISCOVERY_START_HOUR = 15
DISCOVERY_START_MINUTE = 30
DISCOVERY_END_HOUR = 16
DISCOVERY_END_MINUTE = 30


def get_current_et_time():
    """Get current time in Eastern Time (ET)."""
    et = tz('US/Eastern')
    return datetime.now(et)


def is_discovery_window():
    """
    Check if current time is within discovery window.
    GMT: 15:30-16:30 (10:30-11:30 ET) - hours 1-2 of market opening.
    Returns: (is_valid: bool, message: str)
    """
    now_et = get_current_et_time()
    current_time = now_et.time()
    weekday = now_et.weekday()  # 0=Monday, 6=Sunday
    
    # Market is closed on weekends
    if weekday >= 5:
        return False, f"Market closed (weekend). Current ET: {now_et.strftime('%H:%M:%S %Z')}"
    
    # Check if within discovery window (10:30-11:30 ET)
    discovery_start = datetime.strptime("10:30", "%H:%M").time()
    discovery_end = datetime.strptime("11:30", "%H:%M").time()
    
    if discovery_start <= current_time < discovery_end:
        return True, f"Within discovery window. ET: {now_et.strftime('%H:%M:%S %Z')}"
    
    # Before market open
    market_open = datetime.strptime("09:30", "%H:%M").time()
    if current_time < market_open:
        return False, f"Before market open. ET: {now_et.strftime('%H:%M:%S %Z')}"
    
    # After discovery window but before market close
    market_close = datetime.strptime("16:00", "%H:%M").time()
    if current_time < market_close:
        return False, f"Discovery window closed (10:30-11:30 ET only). Current ET: {now_et.strftime('%H:%M:%S %Z')}"
    
    # After market close
    return False, f"Market closed. Current ET: {now_et.strftime('%H:%M:%S %Z')}"


def has_discovered_today(sa):
    """
    Check if Intraday advisor has already discovered stocks in this SA session.
    Returns True if discoveries exist for this session.
    """
    # Check if any Intraday discoveries exist for this SA session
    from core.models import Advisor
    try:
        intraday_advisor = Advisor.objects.get(name="Intraday")
        discoveries_exist = Discovery.objects.filter(
            sa=sa,
            advisor=intraday_advisor
        ).exists()
        return discoveries_exist
    except Advisor.DoesNotExist:
        return False


def normalize_dataframe(df):
    """
    Normalize DataFrame from yfinance download/history.
    Handles MultiIndex columns and ensures proper column names.
    """
    df = df.copy()
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels) > 1:
            df.columns = df.columns.droplevel(1)
        else:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Ensure columns are accessible
    df.columns = [str(col).split('.')[0] if '.' in str(col) else str(col) for col in df.columns]
    
    return df


def ema(series, span):
    """Calculate Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def sma(series, window):
    """Calculate Simple Moving Average."""
    return series.rolling(window).mean()


def rsi(series, period=14):
    """Calculate Relative Strength Index."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    """Calculate Average True Range."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr_df = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3})
    tr = tr_df.max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def vwap(df):
    """Calculate Volume Weighted Average Price."""
    tpv = (df['Close'] * df['Volume']).cumsum()
    vol_cum = df['Volume'].cumsum()
    return tpv / vol_cum


def generate_signals(df, ema_short=EMA_SHORT, ema_long=EMA_LONG,
                     rsi_period=RSI_PERIOD, rsi_overbought=RSI_OVERBOUGHT,
                     vol_multiplier=VOL_MULTIPLIER):
    """Generate technical indicators and entry signals."""
    df = df.copy()
    df['EMA_S'] = ema(df['Close'], ema_short)
    df['EMA_L'] = ema(df['Close'], ema_long)
    df['RSI'] = rsi(df['Close'], rsi_period)
    df['ATR'] = atr(df, period=14)
    df['VWAP'] = vwap(df)
    df['Vol_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()

    # Entry long conditions: price > VWAP, EMA_S > EMA_L, RSI not overbought, volume surge
    df['enter_long'] = (
        (df['Close'] > df['VWAP']) &
        (df['EMA_S'] > df['EMA_L']) &
        (df['RSI'] < rsi_overbought) &
        (df['Volume'] > df['Vol_MA'] * vol_multiplier)
    )

    return df


def check_entry_signal(symbol, period="7d", interval="1h"):
    """
    Check if a stock currently meets ALL entry conditions.
    Returns discovery dict if signals present, None otherwise.
    """
    try:
        ticker = yf.Ticker(symbol)
        
        df = ticker.history(period=period, interval=interval)
        df = normalize_dataframe(df)
        
        # Need at least 40 bars for reliable indicators
        if df.empty or len(df) < 40:
            return None
        
        # Calculate indicators
        df = generate_signals(df)
        
        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        atr_val = float(latest['ATR']) if not np.isnan(latest['ATR']) else 0.0
        
        # Check ALL entry conditions (must all be true)
        if not latest.get('enter_long', False):
            return None
        
        # Calculate stop/target based on ATR
        stop_loss_price = current_price - (SL_ATR_MULT * atr_val)
        take_profit_price = current_price + (TP_ATR_MULT * atr_val)
        
        # Get company name and market cap
        try:
            info = ticker.info
            company_name = info.get('longName') or info.get('shortName') or symbol
            market_cap = info.get('marketCap', 0)
        except:
            company_name = symbol
            market_cap = 0
        
        # Filter by market cap
        if market_cap > 0 and market_cap < MIN_MARKET_CAP:
            return None
        
        # Build explanation
        explanation_parts = [
            f"Intraday momentum: Price ${current_price:.2f} > VWAP ${latest['VWAP']:.2f}",
            f"EMA {latest['EMA_S']:.2f} > {latest['EMA_L']:.2f}",
            f"RSI {latest['RSI']:.1f}",
            f"Volume {latest['Volume']/latest['Vol_MA']:.1f}x avg"
        ]
        
        return {
            'symbol': symbol,
            'company': company_name,
            'entry_price': current_price,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'explanation': " | ".join(explanation_parts),
            'rsi': float(latest['RSI']),
            'volume_ratio': float(latest['Volume'] / latest['Vol_MA']),
            'atr': atr_val,
            'vwap': float(latest['VWAP']),
            'ema_short': float(latest['EMA_S']),
            'ema_long': float(latest['EMA_L']),
            'market_cap': market_cap,
        }
    except Exception as e:
        logger.debug(f"Error checking {symbol}: {e}")
        return None


def get_active_stocks(limit=MAX_STOCKS_TO_CHECK):
    """Get most active stocks from Yahoo Finance."""
    try:
        most_active_query = YfEquityQuery(
            "and",
            [
                YfEquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
                YfEquityQuery("gt", ["intradayprice", 1.0]),
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
            
            if symbol:
                stocks.append({
                    'symbol': symbol,
                    'volume': float(volume) if volume else 0.0,
                })
        
        # Sort by volume AFTER getting results
        stocks.sort(key=lambda x: x['volume'], reverse=True)
        return [s['symbol'] for s in stocks[:limit]]
        
    except Exception as e:
        logger.warning(f"Screener failed, using fallback list: {e}")
        # Fallback: Top liquid + volatile stocks
        return [
            "TSLA", "AMD", "PLTR",
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
            "SPY", "QQQ", "IWM",
            "NIO", "RIVN", "LCID",
            "SOFI", "SNAP", "HOOD", "RBLX"
        ]


class Intraday(AdvisorBase):
    """
    Intraday Momentum Trading Advisor
    
    Discovers stocks with intraday momentum signals during optimal trading hours.
    Only runs during 10:30-11:30 ET (15:30-16:30 GMT) - once per day.
    """

    def discover(self, sa):
        """
        Discover stocks with intraday momentum signals.
        Only runs during discovery window (10:30-11:30 ET) and once per day.
        """
        # Check if within discovery window
        is_valid, message = is_discovery_window()
        if not is_valid:
            logger.info(f"Intraday discovery skipped: {message}")
            return
        
        # Check if already discovered today
        if has_discovered_today(sa):
            logger.info("Intraday advisor has already discovered stocks today. Skipping.")
            return
        
        logger.info("Starting Intraday discovery scan...")
        
        # Get active stocks to check
        symbols = get_active_stocks(limit=MAX_STOCKS_TO_CHECK)
        logger.info(f"Evaluating {len(symbols)} stocks for entry signals...")
        
        discoveries = []
        
        for i, symbol in enumerate(symbols, 1):
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(symbols)} stocks...")
            
            discovery_data = check_entry_signal(symbol)
            
            if discovery_data is None:
                continue
            
            # Create discovery with sell instructions
            # Store actual dollar prices (analysis.py compares price directly to instruction.value)
            entry_price = discovery_data['entry_price']
            stop_loss_price = discovery_data['stop_loss_price']
            take_profit_price = discovery_data['take_profit_price']
            
            sell_instructions = [
                ("STOP_LOSS", stop_loss_price),  # Actual dollar price
                ("TARGET_PRICE", take_profit_price),  # Actual dollar price
            ]
            
            # Create discovery
            self.discovered(
                sa=sa,
                symbol=discovery_data['symbol'],
                company=discovery_data['company'],
                explanation=discovery_data['explanation'],
                sell_instructions=sell_instructions,
                weight=1.0
            )
            
            discoveries.append(discovery_data['symbol'])
            
            time.sleep(0.1)  # Rate limiting
        
        logger.info(f"Intraday discovery complete: {len(discoveries)} stocks discovered")
    
    def analyze(self, sa, stock):
        """
        Intraday advisor doesn't provide recommendations/analysis.
        It only discovers stocks based on momentum signals.
        """
        # This advisor only discovers, doesn't analyze
        pass


register(name="Intraday", python_class="Intraday")

