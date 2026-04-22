import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)

_polygon_stocks_cache: Optional[pd.DataFrame] = None


def get_last_trading_day(test_date: Optional[str] = None) -> Optional[str]:
    """
    Get the previous working day (Mon-Fri) for Polygon API.

    Only works Tue-Fri (skips Mon/Sat/Sun discoveries).
    Returns None if today is Mon/Sat/Sun or if last day was a holiday.
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

    if weekday == 0:
        logger.info("Skipping discovery on Monday")
        return None
    if weekday >= 5:
        logger.info("Skipping discovery on weekend")
        return None

    previous_day = today - timedelta(days=1)
    if previous_day.weekday() == 6:
        previous_day = previous_day - timedelta(days=2)
    elif previous_day.weekday() == 5:
        previous_day = previous_day - timedelta(days=1)

    return previous_day.strftime("%Y-%m-%d")


def _fetch_polygon_stocks_for_date(reference_date: str) -> pd.DataFrame:
    """
    Fetch stocks using Polygon's get_grouped_daily_aggs (1 API call for all stocks on a date).

    Returns a DataFrame with columns: ticker, price, today_volume.
    Returns empty DataFrame on errors.
    """
    polygon_api_key = getattr(settings, "POLYGON_API_KEY", None)
    if not polygon_api_key:
        polygon_api_key = os.getenv("POLYGON_API_KEY")

    if not polygon_api_key:
        logger.warning("POLYGON_API_KEY not set in Django settings or environment")
        return pd.DataFrame()

    try:
        from polygon import RESTClient

        client = RESTClient(polygon_api_key)
        logger.info("Fetching all stocks for %s using Polygon (1 API call)...", reference_date)
        aggs = client.get_grouped_daily_aggs(
            locale="us",
            date=reference_date,
            adjusted=False,
        )

        rows = []
        for agg in aggs:
            rows.append(
                {
                    "ticker": agg.ticker,
                    "price": float(agg.close),
                    "today_volume": int(agg.volume),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            logger.info("Fetched %s stocks from Polygon for %s", len(df), reference_date)
        else:
            logger.warning("No stocks returned from Polygon for %s (may be holiday)", reference_date)
        return df
    except Exception as exc:
        logger.error("Error fetching stocks from Polygon for %s: %s", reference_date, exc, exc_info=True)
        return pd.DataFrame()


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
        last_trading_date = get_last_trading_day(test_date=test_date)
        if not last_trading_date:
            logger.warning("No valid trading date available (Mon/weekend/holiday)")
            return pd.DataFrame()

        _polygon_stocks_cache = _fetch_polygon_stocks_for_date(last_trading_date)
        if _polygon_stocks_cache is None or _polygon_stocks_cache.empty:
            logger.warning("No stocks fetched for %s (may be holiday)", last_trading_date)
            return pd.DataFrame()

    df = _polygon_stocks_cache.copy()
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

