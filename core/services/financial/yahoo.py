import pandas as pd
import yfinance as yf
from typing import Any, Dict
from datetime import datetime, timedelta


def get_6m_history(ticker: str) -> pd.DataFrame:
    """
    Fetch ~6 months of daily history for a ticker from yfinance.
    Returns DataFrame with columns: date, open, close, high, low, volume.
    """
    try:
        ticker_client = yf.Ticker(ticker)
        hist = ticker_client.history(period="6mo", interval="1d", raise_errors=False)
        if hist is None or hist.empty:
            return pd.DataFrame()
        df = pd.DataFrame(
            {
                "date": hist.index,
                "open": hist["Open"].values,
                "close": hist["Close"].values,
                "high": hist["High"].values,
                "low": hist["Low"].values,
                "volume": hist["Volume"].values,
            }
        )
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_ticker_info(ticker: str) -> Dict[str, Any]:
    """Fetch ticker fundamentals/info map from yfinance."""
    try:
        ticker_client = yf.Ticker(ticker)
        return ticker_client.info or {}
    except Exception:
        return {}


def get_historical_data_yfinance(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch date-ranged historical bars and update final row with current session fields.

    Returns DataFrame with columns: date, close, high, low, volume.
    """
    try:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        end_date_exclusive = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        ticker_client = yf.Ticker(ticker)
        hist = ticker_client.history(start=start_date, end=end_date_exclusive, raise_errors=False)
        if hist is None or hist.empty:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "date": hist.index,
                "close": hist["Close"].values,
                "high": hist["High"].values,
                "low": hist["Low"].values,
                "volume": hist["Volume"].values,
            }
        )

        info = get_ticker_info(ticker)
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if len(df) > 0 and current_price and current_price > 0:
            df.iloc[-1, df.columns.get_loc("close")] = current_price
            df.iloc[-1, df.columns.get_loc("low")] = info.get("dayLow") or current_price
            df.iloc[-1, df.columns.get_loc("high")] = info.get("dayHigh") or current_price
            current_volume = info.get("volume")
            if current_volume:
                df.iloc[-1, df.columns.get_loc("volume")] = current_volume

        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

