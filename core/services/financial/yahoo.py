import logging
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytz

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


def headline_lookback_hours(now_et: Optional[datetime] = None) -> float:
    """
    Session-aware headline lookback window (US/Eastern).

    Mon pre-market 72h | Mon regular 48h | Tue–Thu 24h | Fri session 24h
    | Fri after close 12h | weekend / Mon after close 72h
    """
    et = pytz.timezone("US/Eastern")
    now = now_et or datetime.now(et)
    if now.tzinfo is None:
        now = et.localize(now)
    else:
        now = now.astimezone(et)

    wd = now.weekday()
    t = now.time()
    open_t = dt_time(9, 30)
    close_t = dt_time(16, 0)

    if wd == 0 and t < open_t:
        return 72.0
    if wd == 0 and open_t <= t < close_t:
        return 48.0
    if wd in (1, 2, 3):
        return 24.0
    if wd == 4 and t >= close_t:
        return 12.0
    if wd == 4:
        return 24.0
    return 72.0


def headline_lookback_slot(now_et: Optional[datetime] = None) -> Tuple[float, str]:
    """Return (hours, human slot label) for the active lookback window."""
    et = pytz.timezone("US/Eastern")
    now = now_et or datetime.now(et)
    if now.tzinfo is None:
        now = et.localize(now)
    else:
        now = now.astimezone(et)

    wd = now.weekday()
    t = now.time()
    open_t = dt_time(9, 30)
    close_t = dt_time(16, 0)

    if wd == 0 and t < open_t:
        return 72.0, "Mon pre-market"
    if wd == 0 and open_t <= t < close_t:
        return 48.0, "Mon regular session"
    if wd in (1, 2, 3):
        return 24.0, "Tue–Thu"
    if wd == 4 and t >= close_t:
        return 12.0, "Fri after close"
    if wd == 4:
        return 24.0, "Fri session"
    if wd == 0 and t >= close_t:
        return 72.0, "Mon after close"
    return 72.0, "weekend"


def latest_headlines(
    ticker: str,
    limit: int = 3,
    max_age_days: int = 7,
    *,
    max_age_hours: Optional[float] = None,
) -> List[str]:
    """
    Recent Yahoo Finance news titles for a ticker (yfinance Ticker.news).

    Age filter: ``max_age_hours`` when set, else ``max_age_days`` (default 7).
    Set days or hours to 0 to disable the age filter. Returns up to ``limit`` titles.
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return ["No ticker provided."]

    try:
        news_items = yf.Ticker(symbol).news or []
        if not news_items:
            return ["No recent public headlines found."]

        cutoff = None
        age_label = ""
        if max_age_hours is not None and float(max_age_hours) > 0:
            hours = float(max_age_hours)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            age_label = f"{hours:g} hours"
        else:
            age_days = int(max_age_days)
            if age_days > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=age_days)
                age_label = f"{age_days} days"

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
            if age_label:
                return [f"No headlines in the last {age_label}."]
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

