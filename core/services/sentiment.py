"""
Sentiment equity purge scoring.

Rank holdings when over the equity cap; higher score = more willing to sell.
"""

from __future__ import annotations

from datetime import timezone as dt_timezone
from decimal import Decimal
from typing import Any, Dict, Tuple

from django.utils import timezone

from core.models import Holding, Profile

# v2 weights (slots / age / gain / flat).
PURGE_WEIGHT_SLOTS = 0.25
PURGE_WEIGHT_AGE = 0.20
PURGE_WEIGHT_GAIN = 0.40
PURGE_WEIGHT_FLAT = 0.15

SLOTS_FULL_AT = 5
AGE_FULL_AT_DAYS = 30
GAIN_NORM_OF_MIN_RECYCLE = 2
FLAT_PNL_PCT_BAND = Decimal("0.03")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _holding_days(holding: Holding) -> int:
    created = None
    if holding.discovery and holding.discovery.created:
        created = holding.discovery.created
    elif holding.created:
        created = holding.created
    if not created:
        return 0
    anchor = created
    if timezone.is_naive(anchor):
        anchor = timezone.make_aware(anchor, dt_timezone.utc)
    return max(0, (timezone.now() - anchor).days)


def holding_purge_metrics(holding: Holding, fund: Profile) -> Dict[str, Any]:
    """Raw inputs for purge_score (dry-run / logging)."""
    shares = Decimal(str(holding.shares or 0))
    avg = Decimal(str(holding.average_price or 0))
    px = Decimal(str(holding.stock.price or 0))
    cost = shares * avg
    mv = shares * px
    pnl = mv - cost
    pnl_pct = (pnl / cost) if cost > 0 else Decimal("0")
    avg_spend = fund.average_spend()
    slots = (cost / avg_spend) if avg_spend > 0 else Decimal("0")
    return {
        "cost": cost,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "days": _holding_days(holding),
        "tranches": holding.tranches or 0,
        "slots": slots,
    }


def purge_score(holding: Holding, fund: Profile) -> Tuple[float, Dict[str, float]]:
    """
    Purge willingness score 0–100. Higher = sell sooner when over equity cap.

    Returns (score, component_norms) for logging / dry-run.
    """
    m = holding_purge_metrics(holding, fund)
    cost = m["cost"]
    pnl = m["pnl"]
    pnl_pct = m["pnl_pct"]
    days = m["days"]
    slots = float(m["slots"])

    slots_norm = _clamp(slots / SLOTS_FULL_AT)
    age_norm = _clamp(days / AGE_FULL_AT_DAYS)

    min_rec = fund.min_recycle_profit()
    if min_rec > 0:
        gain_norm = _clamp(float(pnl / (min_rec * GAIN_NORM_OF_MIN_RECYCLE)))
    else:
        gain_norm = 0.0

    flat_pct = abs(float(pnl_pct))
    flat_norm = age_norm * (1.0 - _clamp(flat_pct / float(FLAT_PNL_PCT_BAND)))

    score = 100.0 * (
        PURGE_WEIGHT_SLOTS * slots_norm
        + PURGE_WEIGHT_AGE * age_norm
        + PURGE_WEIGHT_GAIN * gain_norm
        + PURGE_WEIGHT_FLAT * flat_norm
    )
    components = {
        "slots": slots_norm,
        "age": age_norm,
        "gain": gain_norm,
        "flat": flat_norm,
    }
    return score, components


def purge_eligible(score: float, fund: Profile) -> bool:
    return score >= Profile.PURGE_MIN_SCORE
