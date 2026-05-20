"""
Intrinsic valuation component (ROE fair-value model; AdvisorBase.evaluate_stock).

Scores price vs model fair value (15% of final model when combined).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.services.health._util import score_relative_multiple
from core.services.health.roe_fair_value import compute_roe_fair_value

COMPONENT_WEIGHT = 0.15
NEUTRAL_FALLBACK_SCORE = 50.0


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
class IntrinsicHealthResult:
    symbol: str
    score: Optional[float]
    component_weight: float = COMPONENT_WEIGHT
    price: Optional[float] = None
    fair_value: Optional[float] = None
    price_to_fair_ratio: Optional[float] = None
    justified_pe: Optional[float] = None
    neutral_fallback: bool = False
    fallback_reason: str = ""
    metrics: List[MetricResult] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "component": "intrinsic",
            "component_weight": COMPONENT_WEIGHT,
            "score": self.score,
            "price": self.price,
            "fair_value": self.fair_value,
            "price_to_fair_ratio": self.price_to_fair_ratio,
            "justified_pe": self.justified_pe,
            "neutral_fallback": self.neutral_fallback,
            "fallback_reason": self.fallback_reason,
            "metrics": [m.to_dict() for m in self.metrics],
            "missing": self.missing,
            "error": self.error,
        }


def score_intrinsic_health(symbol: str) -> IntrinsicHealthResult:
    sym = (symbol or "").strip().upper()
    if not sym:
        return IntrinsicHealthResult(symbol="", score=None, error="empty symbol")

    try:
        fv = compute_roe_fair_value(sym)
    except Exception as e:
        return IntrinsicHealthResult(symbol=sym, score=None, error=str(e))

    if fv.neutral_fallback:
        sub_score = NEUTRAL_FALLBACK_SCORE
        note = fv.reason or "neutral fallback (ratio=1.0)"
    else:
        sub_score = score_relative_multiple(fv.ratio)
        note = ""
        if sub_score is None:
            return IntrinsicHealthResult(
                symbol=sym,
                score=None,
                price=fv.price,
                fair_value=fv.fair_value,
                price_to_fair_ratio=fv.ratio,
                justified_pe=fv.justified_pe,
                error="could not score price/fair ratio",
            )

    ratio = fv.ratio
    price = fv.price
    fair = fv.fair_value
    if price is not None and fair is not None and fair > 0:
        raw_display = f"${price:.2f} / ${fair:.2f} ({ratio:.2f}x)"
    else:
        raw_display = f"{ratio:.2f}x" if ratio is not None else "N/A"

    metric = MetricResult(
        key="price_to_fair_value",
        label="Price / fair value (ROE)",
        weight=1.0,
        raw=ratio,
        raw_display=raw_display,
        score=round(sub_score, 1),
        note=note,
    )

    return IntrinsicHealthResult(
        symbol=sym,
        score=round(sub_score, 1),
        price=price,
        fair_value=fair,
        price_to_fair_ratio=ratio,
        justified_pe=fv.justified_pe,
        neutral_fallback=fv.neutral_fallback,
        fallback_reason=fv.reason,
        metrics=[metric],
        missing=[] if sub_score is not None else ["price_to_fair_value"],
    )
