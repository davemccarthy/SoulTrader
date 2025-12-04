"""
Diagnostic script to understand why no discoveries were found.
Tests a few sample stocks and shows which conditions pass/fail.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pytz import timezone as tz

# Import functions from test script
import sys
sys.path.insert(0, '.')

# Copy the helper functions we need
def normalize_dataframe(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels) > 1:
            df.columns = df.columns.droplevel(1)
        else:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = [str(col).split('.')[0] if '.' in str(col) else str(col) for col in df.columns]
    return df

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

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
    tr_df = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3})
    tr = tr_df.max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def vwap(df):
    tpv = (df['Close'] * df['Volume']).cumsum()
    vol_cum = df['Volume'].cumsum()
    return tpv / vol_cum

def check_conditions(symbol, period="7d", interval="1h"):
    """Check each condition individually and report results."""
    print(f"\n{'='*80}")
    print(f"🔍 DIAGNOSING: {symbol}")
    print(f"{'='*80}")
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get data
        print("\n📊 Fetching data...")
        df = ticker.history(period=period, interval=interval)
        df = normalize_dataframe(df)
        
        print(f"   Data bars: {len(df)}")
        if df.empty:
            print("   ❌ NO DATA")
            return
        if len(df) < 40:
            print(f"   ⚠️  Only {len(df)} bars (need 40+ for reliable indicators)")
        
        # Calculate indicators
        print("\n📈 Calculating indicators...")
        df['EMA_S'] = ema(df['Close'], 8)
        df['EMA_L'] = ema(df['Close'], 21)
        df['RSI'] = rsi(df['Close'], 14)
        df['ATR'] = atr(df, period=14)
        df['VWAP'] = vwap(df)
        df['Vol_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        
        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   EMA Short: ${latest['EMA_S']:.2f}")
        print(f"   EMA Long: ${latest['EMA_L']:.2f}")
        print(f"   RSI: {latest['RSI']:.1f}")
        print(f"   VWAP: ${latest['VWAP']:.2f}")
        print(f"   Volume: {latest['Volume']:,.0f}")
        print(f"   Volume MA: {latest['Vol_MA']:,.0f}")
        
        # Check each condition
        print("\n✅ CONDITIONS CHECK:")
        print("-" * 80)
        
        # 1. Price > VWAP
        price_vwap = current_price > latest['VWAP']
        print(f"1. Price > VWAP: {'✅ PASS' if price_vwap else '❌ FAIL'}")
        print(f"   Price ${current_price:.2f} vs VWAP ${latest['VWAP']:.2f} (diff: ${current_price - latest['VWAP']:.2f})")
        
        # 2. EMA Short > EMA Long
        ema_momentum = latest['EMA_S'] > latest['EMA_L']
        print(f"2. EMA Short > EMA Long: {'✅ PASS' if ema_momentum else '❌ FAIL'}")
        print(f"   EMA_S ${latest['EMA_S']:.2f} vs EMA_L ${latest['EMA_L']:.2f} (diff: ${latest['EMA_S'] - latest['EMA_L']:.2f})")
        
        # 3. RSI < 75
        rsi_ok = latest['RSI'] < 75
        print(f"3. RSI < 75: {'✅ PASS' if rsi_ok else '❌ FAIL'}")
        print(f"   RSI: {latest['RSI']:.1f}")
        
        # 4. Volume > 1.5x average
        vol_ratio = latest['Volume'] / latest['Vol_MA'] if latest['Vol_MA'] > 0 else 0
        vol_surge = vol_ratio > 1.5
        print(f"4. Volume > 1.5x avg: {'✅ PASS' if vol_surge else '❌ FAIL'}")
        print(f"   Volume ratio: {vol_ratio:.2f}x (current: {latest['Volume']:,.0f} vs avg: {latest['Vol_MA']:,.0f})")
        
        # Overall
        all_pass = price_vwap and ema_momentum and rsi_ok and vol_surge
        print(f"\n{'='*80}")
        print(f"🎯 OVERALL: {'✅ ALL CONDITIONS MET' if all_pass else '❌ FAILED'}")
        print(f"{'='*80}")
        
        if not all_pass:
            failed = []
            if not price_vwap:
                failed.append("Price > VWAP")
            if not ema_momentum:
                failed.append("EMA momentum")
            if not rsi_ok:
                failed.append("RSI < 75")
            if not vol_surge:
                failed.append("Volume surge")
            print(f"Failed conditions: {', '.join(failed)}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


# Test a few active stocks
test_symbols = [
    "TSLA", "AMD", "AAPL", "NVDA", "SPY", "QQQ",  # High volume stocks
    "PLTR", "SOFI", "NIO",  # Volatile favorites
]

print("🔍 INTRADAY DISCOVERY DIAGNOSTIC")
print("="*80)
print("Testing sample stocks to see why no discoveries were found...")
print("\nThis will check each entry condition individually for a few stocks.")
print("="*80)

for symbol in test_symbols:
    check_conditions(symbol)
    print("\n")

print("\n" + "="*80)
print("💡 TIPS:")
print("   - If all stocks fail the same condition, that filter might be too strict")
print("   - If volume ratio is consistently < 1.5x, market might be quiet today")
print("   - If RSI is high (>75), stocks might be overbought")
print("   - If Price < VWAP, market might be in a down trend")
print("="*80)










