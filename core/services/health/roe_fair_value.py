"""
ROE-based notional fair value (AdvisorBase.evaluate_stock methodology).

Shared by health.intrinsic and AdvisorBase.evaluate_stock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import yfinance as yf

# Match AdvisorBase.VALUATION_RATIO_TRUST_MAX
RATIO_TRUST_MAX = 5.0


@dataclass
class RoeFairValueResult:
    symbol: str
    ratio: float  # price / fair_value; 1.0 when neutral fallback
    fair_value: Optional[float] = None
    price: Optional[float] = None
    justified_pe: Optional[float] = None
    eps: Optional[float] = None
    roe: Optional[float] = None
    neutral_fallback: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ratio": self.ratio,
            "fair_value": self.fair_value,
            "price": self.price,
            "justified_pe": self.justified_pe,
            "eps": self.eps,
            "roe": self.roe,
            "neutral_fallback": self.neutral_fallback,
            "reason": self.reason,
        }


def compute_roe_fair_value(
    symbol: str,
    info: Optional[Dict[str, Any]] = None,
    *,
    required_return: float = 0.10,
    max_growth: float = 0.20,
    max_roe: float = 0.30,
) -> RoeFairValueResult:
    """
    ROE + payout → justified P/E → fair value → price/fair_value ratio.

    Returns ratio=1.0 with neutral_fallback=True when inputs are unusable
  or ratio is outside the trust band (same as evaluate_stock).
    """
    sym = (symbol or "").strip().upper()
    if info is None:
        info = yf.Ticker(sym).info or {}

    eps = info.get("trailingEps")
    roe = info.get("returnOnEquity")
    payout = info.get("payoutRatio")
    price = info.get("currentPrice") or info.get("regularMarketPrice")

    def _neutral(reason: str) -> RoeFairValueResult:
        return RoeFairValueResult(
            symbol=sym,
            ratio=1.0,
            price=float(price) if price is not None else None,
            eps=float(eps) if eps is not None else None,
            roe=float(roe) if roe is not None else None,
            neutral_fallback=True,
            reason=reason,
        )

    if eps is None or eps <= 0 or roe is None or roe <= 0:
        return _neutral("missing or non-positive EPS/ROE")
    if price is None or price <= 0:
        return _neutral("missing or non-positive price")

    payout_adj = 0 if payout is None or payout < 0 else min(max(float(payout), 0), 0.9)
    adjusted_roe = min(float(roe), max_roe)
    g = adjusted_roe * (1 - payout_adj)
    g = min(g, max_growth)
    denominator = max(required_return - g, 0.01)
    justified_pe = (adjusted_roe * (1 - payout_adj)) / denominator
    fair_value = float(eps) * justified_pe

    if fair_value <= 0:
        return _neutral("invalid fair value")

    ratio = float(price) / fair_value
    trust_min = 1.0 / RATIO_TRUST_MAX
    if ratio > RATIO_TRUST_MAX or ratio < trust_min:
        return RoeFairValueResult(
            symbol=sym,
            ratio=1.0,
            fair_value=fair_value,
            price=float(price),
            justified_pe=justified_pe,
            eps=float(eps),
            roe=float(roe),
            neutral_fallback=True,
            reason=f"extreme ratio {ratio:.2f} outside trust band",
        )

    return RoeFairValueResult(
        symbol=sym,
        ratio=ratio,
        fair_value=fair_value,
        price=float(price),
        justified_pe=justified_pe,
        eps=float(eps),
        roe=float(roe),
        neutral_fallback=False,
        reason="",
    )
