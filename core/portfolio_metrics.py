"""
Portfolio summary numbers for a single fund (Profile).

Used by the header portfolio widget and the Funds dashboard cards.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

from core.models import Holding, Profile, Trade

EMPTY_PORTFOLIO_DASHBOARD: Dict[str, Any] = {
    "total_value": 0.0,
    "cash": 0.0,
    "return_percent": 0.0,
    "return_amount": 0.0,
    "invested": 0.0,
    "holdings_count": 0,
    "trades_count": 0,
    "trade_pnl": 0.0,
    "holdings_pnl": 0.0,
}


def get_portfolio_dashboard_data(fund: Optional[Profile]) -> Dict[str, Any]:
    """
    Compute the same metrics as the header portfolio widget for one fund.

    Returns EMPTY_PORTFOLIO_DASHBOARD when ``fund`` is None.
    """
    if fund is None:
        return dict(EMPTY_PORTFOLIO_DASHBOARD)

    holdings = Holding.objects.filter(fund=fund).select_related("stock")
    holdings_count = holdings.count()

    holdings_value = sum(
        Decimal(str(h.shares)) * h.stock.price for h in holdings
    )
    invested = holdings_value
    total_value = holdings_value + fund.cash

    if fund.investment > 0:
        return_amount = total_value - fund.investment
        return_percent = ((total_value - fund.investment) / fund.investment) * 100
    else:
        return_amount = Decimal("0.0")
        return_percent = Decimal("0.0")

    trades_count = Trade.objects.filter(fund=fund).count()

    sell_trades = Trade.objects.filter(fund=fund, action="SELL", cost__isnull=False)
    trade_pnl = sum(
        (Decimal(str(trade.price)) - Decimal(str(trade.cost))) * Decimal(trade.shares)
        for trade in sell_trades
        if trade.cost
    )

    trade_cost_basis = sum(
        Decimal(str(trade.cost)) * Decimal(trade.shares)
        for trade in sell_trades
        if trade.cost
    )

    if trade_cost_basis > 0:
        trade_pnl_percent = (trade_pnl / trade_cost_basis) * 100
    else:
        trade_pnl_percent = Decimal("0.0")

    holdings_cost_basis = sum(
        (h.average_price or Decimal("0")) * Decimal(str(h.shares)) for h in holdings
    )

    holdings_pnl_raw = holdings_value - holdings_cost_basis

    if holdings_cost_basis > 0:
        holdings_pnl_pct = (holdings_pnl_raw / holdings_cost_basis) * 100
    else:
        holdings_pnl_pct = Decimal("0.0")

    return {
        "total_value": float(total_value),
        "cash": float(fund.cash),
        "return_amount": float(return_amount),
        "invested": float(invested),
        "holdings_count": holdings_count,
        "trades_count": trades_count,
        "trade_pnl": float(trade_pnl_percent),
        "holdings_pnl": holdings_pnl_pct,
        "return_percent": float(return_percent),
    }
