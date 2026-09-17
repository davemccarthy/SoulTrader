"""
Intraday price stabilization: current quote vs ~N minutes ago (15m bar closes).
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

STABILIZE_MINUTES_DEFAULT = 30

# 15m bars reused within one SA so many names in one sector do not re-hit Yahoo.
_hist_15m_cache: dict[str, pd.DataFrame] = {}


def _intraday_15m(symbol: str) -> Optional[pd.DataFrame]:
    import yfinance as yf

    key = (symbol or "").strip().upper()
    if not key:
        return None
    cached = _hist_15m_cache.get(key)
    if cached is not None:
        return cached if not cached.empty else None
    hist = yf.Ticker(key).history(period="1d", interval="15m")
    _hist_15m_cache[key] = hist if hist is not None else pd.DataFrame()
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    return hist


def is_down_vs_minutes_ago(symbol: str, minutes: int = STABILIZE_MINUTES_DEFAULT) -> Optional[bool]:
    """
    True when the latest 15m close is strictly below the close ~minutes ago.
    False when flat or up. None when there is no usable reference.
    """
    try:
        hist = _intraday_15m(symbol)
        if hist is None:
            return None
        closes = hist["Close"].astype(float).copy()
        closes.index = pd.to_datetime(hist.index, utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes)
        eligible = closes.index <= cutoff
        if not eligible.any():
            return None
        px_ago = float(closes.loc[eligible].iloc[-1])
        px_now = float(closes.iloc[-1])
        if px_ago <= 0 or px_now <= 0:
            return None
        return px_now < px_ago
    except Exception as exc:
        logger.warning("Downer check failed for %s: %s", symbol, exc)
        return None


def sector_etf_for(sector: str) -> Optional[str]:
    """Map yfinance sector string to a sector ETF. No SPY fallback."""
    from core.services.market.midway_candidates import SECTOR_ETF

    key = (sector or "").strip().lower()
    if not key:
        return None
    for needle, etf in SECTOR_ETF.items():
        if needle in key:
            return etf
    return None


def price_above_minutes_ago(stock, minutes: int = STABILIZE_MINUTES_DEFAULT) -> Optional[bool]:
    """
    True when stock.price is above the last 15m close at or before (now - minutes).
    False when still falling or flat. None when no intraday reference.
    """
    import yfinance as yf

    try:
        px_now = float(stock.price)
        if px_now <= 0:
            return None

        hist = yf.Ticker(stock.symbol).history(period="1d", interval="15m")
        if hist.empty or "Close" not in hist.columns:
            return None

        idx = pd.to_datetime(hist.index, utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes)
        eligible = idx <= cutoff
        if not eligible.any():
            return None

        px_ago = float(hist.loc[eligible, "Close"].astype(float).iloc[-1])
        if px_ago <= 0:
            return None
        return px_now > px_ago
    except Exception as exc:
        logger.warning("Intraday stabilization check failed for %s: %s", stock.symbol, exc)
        return None
