"""
Distress-aware opportunity adjustments for low-durability names.

Cheap multiples and “near 52-week low” can mean distress, not value. Caps
valuation/price sub-scores when business durability is weak (micro-cap, thin
coverage) so VCIG-style names do not inherit megacap opportunity grades.
"""

from __future__ import annotations

from typing import Dict, Optional

from core.services.health._util import linear_map
from core.services.health.durability import score_business_durability

# Durability at/above ceiling → no adjustment; at/below floor → full caps.
_DURABILITY_CEILING = 50.0
_DURABILITY_FLOOR = 30.0


def _subscore_cap(durability: float, *, floor_cap: float, ceiling_cap: float) -> float:
    if durability >= _DURABILITY_CEILING:
        return ceiling_cap
    if durability <= _DURABILITY_FLOOR:
        return floor_cap
    return linear_map(
        durability,
        in_lo=_DURABILITY_FLOOR,
        in_hi=_DURABILITY_CEILING,
        out_lo=floor_cap,
        out_hi=ceiling_cap,
    )


def adjust_opportunity_parts(
    symbol: str,
    parts: Dict[str, Optional[float]],
    *,
    durability: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """
    Cap valuation and price opportunity inputs for low-durability symbols.
    Returns a shallow copy; other opportunity legs unchanged.
    """
    if durability is None:
        durability = score_business_durability(symbol)
    if durability is None:
        return parts
    if durability >= _DURABILITY_CEILING:
        return parts

    out = dict(parts)
    val_cap = _subscore_cap(durability, floor_cap=42.0, ceiling_cap=100.0)
    if out.get("valuation") is not None:
        out["valuation"] = round(min(float(out["valuation"]), val_cap), 1)

    # Price component rewards 52w lows for entry timing — cap when durability is weak.
    price_cap = _subscore_cap(durability, floor_cap=58.0, ceiling_cap=100.0)
    price = out.get("price")
    if price is not None and float(price) >= 70.0:
        out["price"] = round(min(float(price), price_cap), 1)

    return out
