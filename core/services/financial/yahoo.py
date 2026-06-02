import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


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


def _news_item_title(item: Any) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    content = item.get("content")
    if isinstance(content, dict):
        title = content.get("title")
        if title and str(title).strip():
            return str(title).strip()
    title = item.get("title")
    if title and str(title).strip():
        return str(title).strip()
    return None


def _news_item_published_at(item: Any) -> Optional[datetime]:
    """Parse publish time from yfinance news payload (UTC-aware when possible)."""
    if not isinstance(item, dict):
        return None

    candidates = []
    content = item.get("content")
    if isinstance(content, dict):
        candidates.extend([content.get("pubDate"), content.get("providerPublishTime")])
    candidates.extend([item.get("pubDate"), item.get("providerPublishTime")])

    for raw in candidates:
        if raw is None:
            continue
        try:
            if isinstance(raw, (int, float)):
                return datetime.fromtimestamp(float(raw), tz=timezone.utc)
            text = str(raw).strip()
            if not text:
                continue
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except (TypeError, ValueError, OSError):
            continue
    return None


def latest_headlines(ticker: str, limit: int = 3, max_age_days: int = 7) -> List[str]:
    """
    Recent Yahoo Finance news titles for a ticker (yfinance Ticker.news).

    Only items with a parseable publish time within ``max_age_days`` are included
    (default 7). Set ``max_age_days=0`` to disable the age filter. Returns up to
    ``limit`` titles, newest first.
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return ["No ticker provided."]

    try:
        news_items = yf.Ticker(symbol).news or []
        if not news_items:
            return ["No recent public headlines found."]

        age_days = int(max_age_days)
        cutoff = None
        if age_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=age_days)

        dated: List[Tuple[datetime, str]] = []
        for item in news_items:
            title = _news_item_title(item)
            if not title:
                continue
            published = _news_item_published_at(item)
            if cutoff is not None:
                if published is None or published < cutoff:
                    continue
            elif published is None:
                published = datetime.min.replace(tzinfo=timezone.utc)
            dated.append((published, title))

        dated.sort(key=lambda pair: pair[0], reverse=True)
        headlines = [title for _, title in dated[:limit]]

        if not headlines:
            if age_days > 0:
                return [f"No headlines in the last {age_days} days."]
            return ["No recent public headlines found."]

        return headlines
    except Exception as exc:
        logger.warning("latest_headlines failed for %s: %s", symbol, exc)
        return ["Error pulling live news stream."]


def _safe_float(value: Any) -> Any:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def get_consensus_snapshot(ticker: str) -> Dict[str, Any]:
    """
    Fetch analyst consensus/target snapshot from yfinance info.

    Returns a normalized dict with optional numeric fields and a basic usability flag.
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return {"source": "yfinance", "symbol": "", "is_usable": False}

    info = get_ticker_info(symbol)
    current_price = _safe_float(
        info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    )
    target_mean = _safe_float(info.get("targetMeanPrice"))
    target_median = _safe_float(info.get("targetMedianPrice"))
    target_high = _safe_float(info.get("targetHighPrice"))
    target_low = _safe_float(info.get("targetLowPrice"))
    recommendation_mean = _safe_float(info.get("recommendationMean"))
    recommendation_key = (info.get("recommendationKey") or "").strip().lower() or None

    analyst_count_raw = info.get("numberOfAnalystOpinions")
    try:
        analyst_count = int(analyst_count_raw) if analyst_count_raw is not None else None
    except Exception:
        analyst_count = None

    upside_to_mean_pct = None
    if current_price and target_mean and current_price > 0:
        try:
            upside_to_mean_pct = (target_mean - current_price) / current_price * 100.0
        except Exception:
            upside_to_mean_pct = None

    upside_to_low_pct = None
    if current_price and target_low and current_price > 0:
        try:
            upside_to_low_pct = (target_low - current_price) / current_price * 100.0
        except Exception:
            upside_to_low_pct = None

    return {
        "source": "yfinance",
        "symbol": symbol,
        "current_price": current_price,
        "target_mean": target_mean,
        "target_median": target_median,
        "target_high": target_high,
        "target_low": target_low,
        "upside_to_mean_pct": upside_to_mean_pct,
        "upside_to_low_pct": upside_to_low_pct,
        "recommendation_mean": recommendation_mean,
        "recommendation_key": recommendation_key,
        "analyst_count": analyst_count,
        "is_usable": bool(target_mean is not None or recommendation_mean is not None),
    }


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

