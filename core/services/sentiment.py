"""
Sentiment equity purge — overshoot bands.

When equity_ratio >= sentiment target, sell holdings that clear a min P&L
threshold. The further over target, the looser the threshold (tranche-% of
average_spend). One ladder for all sentiments.
"""

from __future__ import annotations

from datetime import timezone as dt_timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from django.utils import timezone

from core.models import Holding, Profile

# (min_overshoot, g) — g × average_spend = min pnl_$. Highest overshoot first.
# overshoot = equity_ratio - recycle_at.
PURGE_BANDS: List[Tuple[Decimal, Decimal]] = [
    (Decimal("0.15"), Decimal("-0.01")),  # desperate: tiny red OK
    (Decimal("0.10"), Decimal("0")),  # stressed: flat+
    (Decimal("0.05"), Decimal("0.025")),  # clearly over: soft winner
    (Decimal("0.00"), Decimal("0.05")),  # just over: solid winner
]

# Skip dust — frees too little cash (WBS-class 1-lots).
PURGE_MIN_SLOTS = Decimal("0.75")


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
    """Raw inputs for purge dry-run / logging."""
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


def equity_overshoot(fund: Profile) -> Optional[Decimal]:
    """
    equity_ratio - recycle_at. None when sentiment disabled / no target.
    Negative when under target.
    """
    target = fund.equity_target()
    if target is None:
        return None
    return fund.equity_ratio() - target


def purge_band_for_overshoot(overshoot: Decimal) -> Tuple[Decimal, Decimal]:
    """
    Return (min_overshoot, g) for this overshoot.
    If under target (overshoot < 0), still returns the pickiest band for display.
    """
    if overshoot < 0:
        return PURGE_BANDS[-1]
    for min_os, g in PURGE_BANDS:
        if overshoot >= min_os:
            return min_os, g
    return PURGE_BANDS[-1]


def min_purge_pnl(fund: Profile) -> Optional[Decimal]:
    """
    Dollar floor for purge eligibility: g × average_spend for current overshoot.
    None when sentiment disabled.
    """
    overshoot = equity_overshoot(fund)
    if overshoot is None:
        return None
    _min_os, g = purge_band_for_overshoot(overshoot)
    return (g * fund.average_spend()).quantize(Decimal("0.01"))


def purge_band_label(overshoot: Optional[Decimal]) -> str:
    if overshoot is None:
        return "n/a"
    if overshoot < 0:
        return "under"
    _min_os, g = purge_band_for_overshoot(overshoot)
    return f"g={g:+.3f} (os>={_min_os})"


def holding_meets_purge_pnl(
    metrics: Dict[str, Any],
    min_pnl: Decimal,
) -> bool:
    """True if slots clear dust floor and pnl_$ >= band minimum."""
    if metrics["slots"] < PURGE_MIN_SLOTS:
        return False
    return metrics["pnl"] >= min_pnl
