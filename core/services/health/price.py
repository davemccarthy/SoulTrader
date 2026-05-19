"""
Price position component (10% of final buy score in v2 model).

Where the current price sits in the 52-week range (and optional 2-week high check).
Lower in range → higher score for buy timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yfinance as yf

from core.services.health._util import safe_div, score_range_percentile

COMPONENT_WEIGHT = 0.10

METRIC_WEIGHTS = {
    "range_52w": 0.85,
    "near_2w_high": 0.15,
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

    return {
        "symbol": sym,
        "price": price,
        "week52_low": low_52,
        "week52_high": high_52,
        "range_percentile": percentile,
        "week2_high": high_2w,
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

    if weight_total <= 0:
        return PriceHealthResult(
            symbol=sym,
            score=None,
            price=price,
            week52_low=low_52,
            week52_high=high_52,
            range_percentile=percentile,
            week2_high=high_2w,
            metrics=metrics,
            missing=missing,
            error="no scorable price metrics",
        )

    # If 52w missing but 2w present, renormalize happened via weight_total
    composite = weighted_sum / weight_total

    return PriceHealthResult(
        symbol=sym,
        score=round(composite, 1),
        price=price,
        week52_low=low_52,
        week52_high=high_52,
        range_percentile=percentile,
        week2_high=high_2w,
        metrics=metrics,
        missing=missing,
    )
