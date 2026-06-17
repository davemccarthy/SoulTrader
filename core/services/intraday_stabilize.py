"""
Intraday price stabilization: current quote vs ~N minutes ago (15m bar closes).
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

STABILIZE_MINUTES_DEFAULT = 30


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
