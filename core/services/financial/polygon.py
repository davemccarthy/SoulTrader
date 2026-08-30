import logging
import os
from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)

_polygon_stocks_cache: Optional[pd.DataFrame] = None
_POLYGON_STOCK_COLUMNS = ["ticker", "price", "today_volume"]


def get_last_trading_day(test_date: Optional[str] = None) -> Optional[str]:
    """
    Get the most recent US equity session date for Polygon grouped daily aggs.

    Uses calendar rollback (Sat/Sun → Friday; Mon → prior Friday). Returns None
    only when ``test_date`` is invalid or when today is Saturday/Sunday (no
    weekday anchor). Does not detect exchange holidays.
    """
    if test_date:
        try:
            datetime.strptime(test_date, "%Y-%m-%d")
            return test_date
        except ValueError:
            logger.warning("Invalid test_date format: %s", test_date)
            return None

    today = datetime.now().date()
    weekday = today.weekday()  # Monday=0, Sunday=6

    if weekday >= 5:
        logger.info("Skipping discovery on weekend")
        return None

    previous_day = today - timedelta(days=1)
    if previous_day.weekday() == 6:
        previous_day = previous_day - timedelta(days=2)
    elif previous_day.weekday() == 5:
        previous_day = previous_day - timedelta(days=1)

    return previous_day.strftime("%Y-%m-%d")


def _polygon_client():
    polygon_api_key = getattr(settings, "POLYGON_API_KEY", None) or os.getenv("POLYGON_API_KEY")
    if not polygon_api_key:
        raise RuntimeError("POLYGON_API_KEY not set in Django settings or environment")
    from polygon import RESTClient

    return RESTClient(polygon_api_key)


def polygon_eod_data_unavailable(exc: BaseException) -> bool:
    """True when Polygon rejects grouped daily because EOD bars are not published yet."""
    text = str(exc).lower()
    return (
        "not_authorized" in text
        or "before end of day" in text
        or ("end of day" in text and "upgrade" in text)
    )


def _prior_weekday(day: date) -> date:
    current = day - timedelta(days=1)
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def _fetch_grouped_daily_aggs(reference_date: str, *, adjusted: bool) -> list[Any]:
    client = _polygon_client()
    logger.info("Fetching Polygon grouped daily for %s (adjusted=%s)...", reference_date, adjusted)
    return list(
        client.get_grouped_daily_aggs(
            locale="us",
            date=reference_date,
            adjusted=adjusted,
        )
    )


def fetch_grouped_daily_map(
    session_date: date | str,
    *,
    adjusted: bool = True,
    max_lookback: int = 5,
) -> tuple[dict[str, dict[str, Any]], date]:
    """
    Grouped daily OHLCV map for one US session.

    Lower-tier Polygon plans reject same-calendar-day requests until EOD bars are
    published; on NOT_AUTHORIZED / "before end of day" we step back to prior sessions.
    """
    from core.services.market import prior_trading_day

    current = date.fromisoformat(session_date) if isinstance(session_date, str) else session_date
    last_error: Optional[BaseException] = None

    for _ in range(max_lookback):
        reference = current.isoformat()
        try:
            aggs = _fetch_grouped_daily_aggs(reference, adjusted=adjusted)
        except Exception as exc:
            last_error = exc
            if polygon_eod_data_unavailable(exc):
                logger.warning(
                    "Polygon grouped daily unavailable for %s (%s); trying prior session",
                    reference,
                    exc,
                )
                current = prior_trading_day(current)
                continue
            raise

        out: dict[str, dict[str, Any]] = {}
        for agg in aggs:
            symbol = str(getattr(agg, "ticker", "") or "").strip().upper()
            if not symbol:
                continue
            close = float(agg.close)
            if close <= 0:
                continue
            out[symbol] = {
                "open": float(getattr(agg, "open", None) or close),
                "close": close,
                "volume": int(getattr(agg, "volume", None) or 0),
            }

        if out:
            requested = session_date.isoformat() if isinstance(session_date, date) else str(session_date)
            if reference != requested:
                logger.info(
                    "Polygon grouped daily resolved %s -> %s (%s symbols)",
                    session_date,
                    reference,
                    len(out),
                )
            return out, current

        logger.warning("No Polygon grouped daily rows for %s (may be holiday)", reference)
        current = prior_trading_day(current)

    if last_error is not None:
        raise RuntimeError(
            f"No Polygon grouped daily data within {max_lookback} sessions of {session_date}"
        ) from last_error
    raise RuntimeError(
        f"No Polygon grouped daily data within {max_lookback} sessions of {session_date}"
    )


def _fetch_polygon_stocks_for_date(reference_date: str) -> pd.DataFrame:
    """
    Fetch stocks using Polygon's get_grouped_daily_aggs (1 API call for all stocks on a date).

    Returns a DataFrame with columns: ticker, price, today_volume.
    Returns empty DataFrame on errors.
    """
    try:
        session_map, _resolved = fetch_grouped_daily_map(reference_date, adjusted=False, max_lookback=1)
    except Exception as exc:
        logger.error("Error fetching stocks from Polygon for %s: %s", reference_date, exc, exc_info=True)
        return pd.DataFrame()

    rows = [
        {
            "ticker": symbol,
            "price": values["close"],
            "today_volume": values["volume"],
        }
        for symbol, values in session_map.items()
    ]
    df = pd.DataFrame(rows, columns=_POLYGON_STOCK_COLUMNS)
    if not df.empty:
        logger.info("Fetched %s stocks from Polygon for %s", len(df), reference_date)
    return df


def get_filtered_stocks(
    min_price=None,
    max_price=None,
    min_volume=None,
    test_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get filtered stocks from Polygon (last trading day).
    Fetches once per session, caches, then applies filters.
    """
    global _polygon_stocks_cache

    if _polygon_stocks_cache is None:
        reference_date = get_last_trading_day(test_date=test_date)
        if not reference_date:
            logger.warning("No valid trading date available (Mon/weekend/holiday)")
            return pd.DataFrame()

        attempts = 5
        for _ in range(attempts):
            _polygon_stocks_cache = _fetch_polygon_stocks_for_date(reference_date)
            if _polygon_stocks_cache is not None and not _polygon_stocks_cache.empty:
                break

            previous_day = _prior_weekday(datetime.strptime(reference_date, "%Y-%m-%d").date())
            reference_date = previous_day.strftime("%Y-%m-%d")
        else:
            logger.warning("No stocks fetched from Polygon after %s attempts", attempts)
            return pd.DataFrame()

    df = _polygon_stocks_cache.copy()
    missing_columns = [col for col in _POLYGON_STOCK_COLUMNS if col not in df.columns]
    if missing_columns:
        logger.warning("Polygon stocks missing columns %s; skipping filters", missing_columns)
        return pd.DataFrame(columns=_POLYGON_STOCK_COLUMNS)

    if min_price is not None:
        df = df[df["price"] >= min_price]
    if max_price is not None:
        df = df[df["price"] <= max_price]
    if min_volume is not None:
        df = df[df["today_volume"] >= min_volume]
    return df


def clear_polygon_cache() -> None:
    """Clear the Polygon stocks cache (useful for testing or between runs)."""
    global _polygon_stocks_cache
    _polygon_stocks_cache = None
    logger.info("Polygon stocks cache cleared")

