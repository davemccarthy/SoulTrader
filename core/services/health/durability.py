"""
Business durability / scale signal for stability axis.

Penalises micro-caps with thin coverage vs megacaps with proven scale.
Uses yfinance info only (market cap, revenue, analysts, headcount).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import yfinance as yf

from core.services.health._util import linear_map

# Sub-factor weights (sum = 1.0)
_DURABILITY_WEIGHTS = {
    "market_cap": 0.40,
    "revenue": 0.35,
    "analyst_coverage": 0.15,
    "headcount": 0.10,
}


def _score_market_cap(mcap: Optional[float]) -> Optional[float]:
    if mcap is None or mcap <= 0:
        return None
    b = mcap / 1_000_000_000.0
    if b >= 100:
        return 98.0
    if b >= 10:
        return linear_map(b, in_lo=10, in_hi=100, out_lo=82, out_hi=98)
    if b >= 2:
        return linear_map(b, in_lo=2, in_hi=10, out_lo=68, out_hi=82)
    if b >= 0.3:
        return linear_map(b, in_lo=0.3, in_hi=2, out_lo=52, out_hi=68)
    if b >= 0.05:
        return linear_map(b, in_lo=0.05, in_hi=0.3, out_lo=35, out_hi=52)
    return 25.0


def _score_revenue(revenue: Optional[float]) -> Optional[float]:
    if revenue is None or revenue <= 0:
        return None
    b = revenue / 1_000_000_000.0
    if b >= 50:
        return 95.0
    if b >= 5:
        return linear_map(b, in_lo=5, in_hi=50, out_lo=78, out_hi=95)
    if b >= 0.5:
        return linear_map(b, in_lo=0.5, in_hi=5, out_lo=62, out_hi=78)
    if b >= 0.1:
        return linear_map(b, in_lo=0.1, in_hi=0.5, out_lo=45, out_hi=62)
    return 32.0


def _score_analyst_coverage(count: Optional[int]) -> Optional[float]:
    if count is None:
        return 30.0
    if count >= 30:
        return 90.0
    if count >= 15:
        return linear_map(float(count), in_lo=15, in_hi=30, out_lo=72, out_hi=90)
    if count >= 5:
        return linear_map(float(count), in_lo=5, in_hi=15, out_lo=55, out_hi=72)
    if count >= 1:
        return linear_map(float(count), in_lo=1, in_hi=5, out_lo=38, out_hi=55)
    return 30.0


def _score_headcount(employees: Optional[int]) -> Optional[float]:
    if employees is None or employees <= 0:
        return None
    if employees >= 50_000:
        return 92.0
    if employees >= 5_000:
        return linear_map(float(employees), in_lo=5_000, in_hi=50_000, out_lo=72, out_hi=92)
    if employees >= 500:
        return linear_map(float(employees), in_lo=500, in_hi=5_000, out_lo=50, out_hi=72)
    return linear_map(float(employees), in_lo=1, in_hi=500, out_lo=30, out_hi=50)


def fetch_durability_inputs(symbol: str) -> Dict[str, Any]:
    sym = (symbol or "").strip().upper()
    info = yf.Ticker(sym).info or {}
    analysts = info.get("numberOfAnalystOpinions")
    return {
        "symbol": sym,
        "market_cap": info.get("marketCap"),
        "revenue": info.get("totalRevenue"),
        "analyst_coverage": int(analysts) if analysts is not None else None,
        "headcount": info.get("fullTimeEmployees"),
    }


def score_business_durability(symbol: str) -> Optional[float]:
    """
    Durability score 0–100 from scale and institutional coverage.
    Returns None only when no inputs are available.
    """
    try:
        raw = fetch_durability_inputs(symbol)
    except Exception:
        return None

    parts = {
        "market_cap": _score_market_cap(raw.get("market_cap")),
        "revenue": _score_revenue(raw.get("revenue")),
        "analyst_coverage": _score_analyst_coverage(raw.get("analyst_coverage")),
        "headcount": _score_headcount(raw.get("headcount")),
    }

    num = 0.0
    den = 0.0
    for key, w in _DURABILITY_WEIGHTS.items():
        sc = parts.get(key)
        if sc is None:
            continue
        num += float(sc) * w
        den += w
    if den <= 0:
        return None
    return round(num / den, 1)
