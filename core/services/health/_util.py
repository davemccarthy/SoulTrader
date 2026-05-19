"""Shared helpers for health scoring components."""

from __future__ import annotations

from typing import Any, Optional


def safe_div(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        if numerator is None or denominator in (0, None):
            return None
        result = float(numerator) / float(denominator)
        if result != result or result in (float("inf"), float("-inf")):  # NaN / inf
            return None
        return result
    except (TypeError, ValueError):
        return None


def pct_change(current: Any, prior: Any) -> Optional[float]:
    """Return fractional change (0.12 = +12%)."""
    if current is None or prior is None:
        return None
    try:
        c, p = float(current), float(prior)
    except (TypeError, ValueError):
        return None
    if p == 0:
        return None if c == 0 else 1.0
    return (c - p) / abs(p)


def clip_score(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def linear_map(
    value: float,
    *,
    in_lo: float,
    in_hi: float,
    out_lo: float,
    out_hi: float,
) -> float:
    """Map value linearly from [in_lo, in_hi] to [out_lo, out_hi], then clip to 0–100."""
    if in_hi == in_lo:
        return clip_score(out_hi)
    t = (value - in_lo) / (in_hi - in_lo)
    return clip_score(out_lo + t * (out_hi - out_lo))


def score_relative_multiple(ratio: Optional[float]) -> Optional[float]:
    """
    Map price/fair-value or stock/sector ratio to 0–100.
    1.0 ≈ fairly valued (~68); lower ratio → higher score.
    """
    if ratio is None or ratio <= 0:
        return None
    if ratio <= 0.65:
        return 98.0
    if ratio <= 0.85:
        return linear_map(ratio, in_lo=0.65, in_hi=0.85, out_lo=90, out_hi=78)
    if ratio <= 1.05:
        return linear_map(ratio, in_lo=0.85, in_hi=1.05, out_lo=78, out_hi=68)
    if ratio <= 1.35:
        return linear_map(ratio, in_lo=1.05, in_hi=1.35, out_lo=68, out_hi=45)
    if ratio <= 1.75:
        return linear_map(ratio, in_lo=1.35, in_hi=1.75, out_lo=45, out_hi=25)
    if ratio <= 2.5:
        return linear_map(ratio, in_lo=1.75, in_hi=2.5, out_lo=25, out_hi=12)
    return 10.0


def score_range_percentile(percentile: Optional[float]) -> Optional[float]:
    """
  Map 52-week range percentile (0 = at low, 1 = at high) to 0–100.
  Favors buying nearer lows unless fundamentals justify highs.
    """
    if percentile is None:
        return None
    p = max(0.0, min(1.0, float(percentile)))
    if p <= 0.20:
        return linear_map(p, in_lo=0.0, in_hi=0.20, out_lo=98, out_hi=90)
    if p <= 0.40:
        return linear_map(p, in_lo=0.20, in_hi=0.40, out_lo=90, out_hi=75)
    if p <= 0.60:
        return linear_map(p, in_lo=0.40, in_hi=0.60, out_lo=75, out_hi=55)
    if p <= 0.80:
        return linear_map(p, in_lo=0.60, in_hi=0.80, out_lo=55, out_hi=35)
    return linear_map(p, in_lo=0.80, in_hi=1.0, out_lo=35, out_hi=15)
