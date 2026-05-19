"""
Analyst consensus component (10% of final buy score in v2 model).

Street rating + target upside from yfinance (no LLM). Uses financial.yahoo.get_consensus_snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.services.financial.yahoo import get_consensus_snapshot
from core.services.health._util import linear_map

COMPONENT_WEIGHT = 0.10
NEUTRAL_FALLBACK_SCORE = 50.0
MIN_ANALYST_COUNT = 3

METRIC_WEIGHTS = {
    "analyst_rating": 0.55,
    "upside_to_mean": 0.35,
    "vs_target_low": 0.10,
}

# yfinance recommendationKey → base score
RECOMMENDATION_KEY_SCORES = {
    "strong_buy": 95.0,
    "buy": 80.0,
    "outperform": 80.0,
    "overweight": 78.0,
    "hold": 55.0,
    "neutral": 55.0,
    "market_perform": 50.0,
    "equal_weight": 50.0,
    "underperform": 30.0,
    "underweight": 30.0,
    "sell": 10.0,
    "strong_sell": 10.0,
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
class ConsensusHealthResult:
    symbol: str
    score: Optional[float]
    component_weight: float = COMPONENT_WEIGHT
    recommendation_key: Optional[str] = None
    recommendation_mean: Optional[float] = None
    analyst_count: Optional[int] = None
    target_mean: Optional[float] = None
    current_price: Optional[float] = None
    upside_to_mean_pct: Optional[float] = None
    upside_to_low_pct: Optional[float] = None
    thin_coverage: bool = False
    neutral_fallback: bool = False
    metrics: List[MetricResult] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "component": "consensus",
            "component_weight": self.component_weight,
            "score": self.score,
            "recommendation_key": self.recommendation_key,
            "recommendation_mean": self.recommendation_mean,
            "analyst_count": self.analyst_count,
            "target_mean": self.target_mean,
            "current_price": self.current_price,
            "upside_to_mean_pct": self.upside_to_mean_pct,
            "upside_to_low_pct": self.upside_to_low_pct,
            "thin_coverage": self.thin_coverage,
            "neutral_fallback": self.neutral_fallback,
            "metrics": [m.to_dict() for m in self.metrics],
            "missing": self.missing,
            "error": self.error,
        }


def _score_recommendation_key(key: Optional[str]) -> Optional[float]:
    if not key:
        return None
    normalized = key.strip().lower().replace(" ", "_")
    if normalized in RECOMMENDATION_KEY_SCORES:
        return RECOMMENDATION_KEY_SCORES[normalized]
    if "strong" in normalized and "buy" in normalized:
        return 95.0
    if "strong" in normalized and "sell" in normalized:
        return 10.0
    if "buy" in normalized:
        return 80.0
    if "sell" in normalized:
        return 10.0
    if "hold" in normalized or "neutral" in normalized:
        return 55.0
    return None


def _score_recommendation_mean(mean: Optional[float]) -> Optional[float]:
    """yfinance: lower mean = more bullish (typically 1.0–5.0)."""
    if mean is None:
        return None
    m = float(mean)
    if m <= 1.5:
        return 95.0
    if m <= 2.0:
        return linear_map(m, in_lo=1.5, in_hi=2.0, out_lo=90, out_hi=80)
    if m <= 2.5:
        return linear_map(m, in_lo=2.0, in_hi=2.5, out_lo=80, out_hi=65)
    if m <= 3.0:
        return linear_map(m, in_lo=2.5, in_hi=3.0, out_lo=65, out_hi=55)
    if m <= 3.5:
        return linear_map(m, in_lo=3.0, in_hi=3.5, out_lo=55, out_hi=40)
    if m <= 4.0:
        return linear_map(m, in_lo=3.5, in_hi=4.0, out_lo=40, out_hi=25)
    return 15.0


def _score_analyst_rating(key: Optional[str], mean: Optional[float]) -> Optional[float]:
    from_key = _score_recommendation_key(key)
    if from_key is not None:
        return from_key
    return _score_recommendation_mean(mean)


def _score_upside_to_mean(pct: Optional[float]) -> Optional[float]:
    if pct is None:
        return None
    if pct < -15:
        return 20.0
    if pct < -5:
        return linear_map(pct, in_lo=-15, in_hi=-5, out_lo=20, out_hi=40)
    if pct < 0:
        return linear_map(pct, in_lo=-5, in_hi=0, out_lo=40, out_hi=55)
    if pct < 10:
        return linear_map(pct, in_lo=0, in_hi=10, out_lo=55, out_hi=70)
    if pct < 25:
        return linear_map(pct, in_lo=10, in_hi=25, out_lo=70, out_hi=90)
    return 95.0


def _score_vs_target_low(pct: Optional[float]) -> Optional[float]:
    """Upside to bearish (low) target — room even vs pessimistic street."""
    if pct is None:
        return None
    if pct < -10:
        return 25.0
    if pct < 0:
        return linear_map(pct, in_lo=-10, in_hi=0, out_lo=25, out_hi=50)
    if pct < 15:
        return linear_map(pct, in_lo=0, in_hi=15, out_lo=50, out_hi=75)
    return 85.0


def score_consensus_health(symbol: str) -> ConsensusHealthResult:
    sym = (symbol or "").strip().upper()
    if not sym:
        return ConsensusHealthResult(symbol="", score=None, error="empty symbol")

    try:
        snap = get_consensus_snapshot(sym)
    except Exception as e:
        return ConsensusHealthResult(symbol=sym, score=None, error=str(e))

    if not snap.get("is_usable"):
        return ConsensusHealthResult(
            symbol=sym,
            score=NEUTRAL_FALLBACK_SCORE,
            neutral_fallback=True,
            error="no analyst consensus data",
        )

    rec_key = snap.get("recommendation_key")
    rec_mean = snap.get("recommendation_mean")
    analyst_count = snap.get("analyst_count")
    target_mean = snap.get("target_mean")
    price = snap.get("current_price")
    upside_mean = snap.get("upside_to_mean_pct")
    upside_low = snap.get("upside_to_low_pct")
    thin = analyst_count is not None and analyst_count < MIN_ANALYST_COUNT

    metrics: List[MetricResult] = []
    missing: List[str] = []
    weighted_sum = 0.0
    weight_total = 0.0

    # Analyst rating
    rating_score = _score_analyst_rating(rec_key, rec_mean)
    rec_display = rec_key or (f"mean {rec_mean:.2f}" if rec_mean is not None else "N/A")
    if analyst_count is not None:
        rec_display = f"{rec_display} ({analyst_count} analysts)"

    m_rating = MetricResult(
        key="analyst_rating",
        label="Analyst rating",
        weight=METRIC_WEIGHTS["analyst_rating"],
        raw=rec_key or rec_mean,
        raw_display=rec_display,
    )
    if rating_score is None:
        m_rating.note = "no recommendation field"
        missing.append("analyst_rating")
    else:
        m_rating.score = round(rating_score, 1)
        if thin:
            m_rating.note = f"thin coverage (<{MIN_ANALYST_COUNT} analysts)"
            m_rating.score = round((rating_score + NEUTRAL_FALLBACK_SCORE) / 2, 1)
        weighted_sum += m_rating.score * m_rating.weight
        weight_total += m_rating.weight
    metrics.append(m_rating)

    # Upside to mean target
    m_up = MetricResult(
        key="upside_to_mean",
        label="Upside to mean target",
        weight=METRIC_WEIGHTS["upside_to_mean"],
        raw=upside_mean,
    )
    if upside_mean is not None and target_mean is not None and price is not None:
        m_up.raw_display = f"{upside_mean:+.1f}% (${price:.2f} → ${target_mean:.2f})"
    elif upside_mean is not None:
        m_up.raw_display = f"{upside_mean:+.1f}%"
    up_score = _score_upside_to_mean(upside_mean)
    if up_score is None:
        m_up.note = "target mean or price missing"
        missing.append("upside_to_mean")
    else:
        m_up.score = round(up_score, 1)
        weighted_sum += m_up.score * m_up.weight
        weight_total += m_up.weight
    metrics.append(m_up)

    # vs consensus low target
    m_low = MetricResult(
        key="vs_target_low",
        label="vs consensus low target",
        weight=METRIC_WEIGHTS["vs_target_low"],
        raw=upside_low,
    )
    if upside_low is not None:
        m_low.raw_display = f"{upside_low:+.1f}% to low target"
    low_score = _score_vs_target_low(upside_low)
    if low_score is None:
        m_low.note = "target low or price missing"
        missing.append("vs_target_low")
    else:
        m_low.score = round(low_score, 1)
        weighted_sum += m_low.score * m_low.weight
        weight_total += m_low.weight
    metrics.append(m_low)

    if weight_total <= 0:
        return ConsensusHealthResult(
            symbol=sym,
            score=NEUTRAL_FALLBACK_SCORE,
            recommendation_key=rec_key,
            recommendation_mean=rec_mean,
            analyst_count=analyst_count,
            target_mean=target_mean,
            current_price=price,
            upside_to_mean_pct=upside_mean,
            upside_to_low_pct=upside_low,
            thin_coverage=thin,
            neutral_fallback=True,
            metrics=metrics,
            missing=missing,
            error="no scorable consensus metrics",
        )

    return ConsensusHealthResult(
        symbol=sym,
        score=round(weighted_sum / weight_total, 1),
        recommendation_key=rec_key,
        recommendation_mean=rec_mean,
        analyst_count=analyst_count,
        target_mean=target_mean,
        current_price=price,
        upside_to_mean_pct=upside_mean,
        upside_to_low_pct=upside_low,
        thin_coverage=thin,
        metrics=metrics,
        missing=missing,
    )
