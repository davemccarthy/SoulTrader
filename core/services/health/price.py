"""
Price position component (20% of final buy score in v2 model).

Where the current price sits in the 52-week range, vs the 2-week high, and
same-session move (anti-chase for headline-driven buys).
Lower in range / smaller same-day rip → higher score for buy timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yfinance as yf

from core.services.health._util import linear_map, pct_change, safe_div, score_range_percentile

COMPONENT_WEIGHT = 0.20

# Minimum composite price score for news_flash discoveries (after LLM BUY).
NEWS_FLASH_MIN_PRICE_SCORE = 60.0

METRIC_WEIGHTS = {
    "range_52w": 0.70,
    "near_2w_high": 0.15,
    "change_1d": 0.15,
}


@dataclass
class MetricResult:
    key: str
    label: str
    weight: float
    raw: Any = None
    raw_display: str = "N/A"
    score: Optional[float] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "weight": self.weight,
            "raw": self.raw,
            "raw_display": self.raw_display,
            "score": self.score,
            "note": self.note,
        }


@dataclass
class PriceHealthResult:
    symbol: str
    score: Optional[float]
    component_weight: float = COMPONENT_WEIGHT
    price: Optional[float] = None
    week52_low: Optional[float] = None
    week52_high: Optional[float] = None
    range_percentile: Optional[float] = None
    week2_high: Optional[float] = None
    change_1d: Optional[float] = None
    metrics: List[MetricResult] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "component": "price",
            "component_weight": self.component_weight,
            "score": self.score,
            "price": self.price,
            "week52_low": self.week52_low,
            "week52_high": self.week52_high,
            "range_percentile": self.range_percentile,
            "week2_high": self.week2_high,
            "change_1d": self.change_1d,
            "metrics": [m.to_dict() for m in self.metrics],
            "missing": self.missing,
            "error": self.error,
        }


def _f(info: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        v = info.get(k)
        if v is not None:
            try:
                fv = float(v)
                if fv == fv and fv not in (float("inf"), float("-inf")):
                    return fv
            except (TypeError, ValueError):
                continue
    return None


def _score_near_2w_high(price: float, high_2w: float) -> float:
    """Penalty for chasing: at/above 95% of 2-week high scores low."""
    if high_2w <= 0:
        return 50.0
    ratio = price / high_2w
    if ratio >= 0.98:
        return 15.0
    if ratio >= 0.95:
        return 30.0
    if ratio >= 0.90:
        return 50.0
    if ratio >= 0.80:
        return 70.0
    return 85.0


def _score_change_1d(change_frac: Optional[float]) -> Optional[float]:
    """
    Penalize chasing a same-day rip; mild moves and down days score higher.
    change_frac is fractional (+0.08 = +8%).
    """
    if change_frac is None:
        return None
    c = float(change_frac)
    if c > 0.15:
        return 15.0
    if c > 0.08:
        return linear_map(c, in_lo=0.08, in_hi=0.15, out_lo=25, out_hi=15)
    if c > 0.03:
        return linear_map(c, in_lo=0.03, in_hi=0.08, out_lo=70, out_hi=25)
    if c >= -0.05:
        return linear_map(c, in_lo=-0.05, in_hi=0.03, out_lo=75, out_hi=90)
    return 72.0


def fetch_price_inputs(symbol: str) -> Dict[str, Any]:
    sym = (symbol or "").strip().upper()
    t = yf.Ticker(sym)
    info = t.info or {}

    price = _f(info, "currentPrice", "regularMarketPrice", "previousClose")
    low_52 = _f(info, "fiftyTwoWeekLow")
    high_52 = _f(info, "fiftyTwoWeekHigh")

    percentile = None
    if price is not None and low_52 is not None and high_52 is not None and high_52 > low_52:
        percentile = (price - low_52) / (high_52 - low_52)

    high_2w = None
    try:
        hist = t.history(period="2wk")
        if hist is not None and not hist.empty:
            high_2w = float(hist["High"].max())
    except Exception:
        pass

    prev_close = _f(info, "previousClose", "regularMarketPreviousClose")
    change_1d = pct_change(price, prev_close)
    if change_1d is None:
        ch_pct = _f(info, "regularMarketChangePercent")
        if ch_pct is not None:
            change_1d = ch_pct / 100.0

    return {
        "symbol": sym,
        "price": price,
        "week52_low": low_52,
        "week52_high": high_52,
        "range_percentile": percentile,
        "week2_high": high_2w,
        "change_1d": change_1d,
    }


def score_price_health(symbol: str) -> PriceHealthResult:
    sym = (symbol or "").strip().upper()
    if not sym:
        return PriceHealthResult(symbol="", score=None, error="empty symbol")

    try:
        raw = fetch_price_inputs(sym)
    except Exception as e:
        return PriceHealthResult(symbol=sym, score=None, error=str(e))

    price = raw["price"]
    low_52 = raw["week52_low"]
    high_52 = raw["week52_high"]
    percentile = raw["range_percentile"]
    high_2w = raw["week2_high"]
    change_1d = raw["change_1d"]

    metrics: List[MetricResult] = []
    missing: List[str] = []
    weighted_sum = 0.0
    weight_total = 0.0

    # 52-week range percentile
    pct_score = score_range_percentile(percentile)
    if percentile is not None and low_52 is not None and high_52 is not None and price is not None:
        raw_display = f"{percentile * 100:.0f}% of 52w (${low_52:.2f}–${high_52:.2f})"
    else:
        raw_display = "N/A"

    m52 = MetricResult(
        key="range_52w",
        label="52-week range position",
        weight=METRIC_WEIGHTS["range_52w"],
        raw=percentile,
        raw_display=raw_display,
    )
    if pct_score is None:
        m52.note = "insufficient 52-week range data"
        missing.append("range_52w")
    else:
        m52.score = round(pct_score, 1)
        weighted_sum += pct_score * m52.weight
        weight_total += m52.weight
    metrics.append(m52)

    # 2-week high proximity (legacy overlay analogue)
    m2w = MetricResult(
        key="near_2w_high",
        label="vs 2-week high",
        weight=METRIC_WEIGHTS["near_2w_high"],
    )
    if price is not None and high_2w is not None and high_2w > 0:
        m2w.raw = safe_div(price, high_2w)
        m2w.raw_display = f"${price:.2f} vs 2w high ${high_2w:.2f} ({m2w.raw:.2f}x)"
        m2w.score = round(_score_near_2w_high(price, high_2w), 1)
        weighted_sum += m2w.score * m2w.weight
        weight_total += m2w.weight
    else:
        m2w.note = "2-week history unavailable"
        missing.append("near_2w_high")
    metrics.append(m2w)

    m1d = MetricResult(
        key="change_1d",
        label="1-day change",
        weight=METRIC_WEIGHTS["change_1d"],
    )
    ch_score = _score_change_1d(change_1d)
    if change_1d is not None:
        m1d.raw = change_1d
        m1d.raw_display = f"{change_1d * 100:+.1f}%"
    else:
        m1d.raw_display = "N/A"
    if ch_score is None:
        m1d.note = "1-day change unavailable"
        missing.append("change_1d")
    else:
        m1d.score = round(ch_score, 1)
        weighted_sum += ch_score * m1d.weight
        weight_total += m1d.weight
    metrics.append(m1d)

    if weight_total <= 0:
        return PriceHealthResult(
            symbol=sym,
            score=None,
            price=price,
            week52_low=low_52,
            week52_high=high_52,
            range_percentile=percentile,
            week2_high=high_2w,
            change_1d=change_1d,
            metrics=metrics,
            missing=missing,
            error="no scorable price metrics",
        )

    composite = weighted_sum / weight_total

    return PriceHealthResult(
        symbol=sym,
        score=round(composite, 1),
        price=price,
        week52_low=low_52,
        week52_high=high_52,
        range_percentile=percentile,
        week2_high=high_2w,
        change_1d=change_1d,
        metrics=metrics,
        missing=missing,
    )
