"""
Portfolio summary numbers for one fund (Profile) or aggregated across all enabled funds.

``get_portfolio_dashboard_data(0)`` sums every enabled fund (holdings, cash, trades).
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional, Union

from django.db.models import Sum
from django.utils import timezone

from core.models import Holding, Profile, Trade, Snapshot

FundArg = Union[Optional[Profile], int]

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
    # "Estimated Annualized Basis" (dummy naming for now):
    # ABV is an annualized version of current return, based on fund age.
    "estab_days": 0,
    "est_abv_percent": 0.0,
    "today_percent": 0.0,
}


def _dashboard_all_enabled_funds() -> Dict[str, Any]:
    """Aggregate Wealth / Return / … across every ``Profile`` with ``enabled=True``."""
    enabled = Profile.objects.filter(enabled=True)

    cash_row = enabled.aggregate(s=Sum("cash"))
    total_cash = cash_row["s"] or Decimal("0")

    inv_row = enabled.aggregate(s=Sum("investment"))
    total_investment = inv_row["s"] or Decimal("0")

    holdings = Holding.objects.filter(fund__enabled=True).select_related("stock")
    holdings_count = holdings.count()
    holdings_value = sum(
        Decimal(str(h.shares)) * h.stock.price for h in holdings
    )
    invested = holdings_value
    total_value = holdings_value + total_cash

    if total_investment > 0:
        return_amount = total_value - total_investment
        return_percent = (return_amount / total_investment) * Decimal("100")
    else:
        return_amount = Decimal("0.0")
        return_percent = Decimal("0.0")

    trades_count = Trade.objects.filter(fund__enabled=True).count()

    sell_trades = Trade.objects.filter(
        fund__enabled=True, action="SELL", cost__isnull=False
    )
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

    latest_dates = list(
        Snapshot.objects
        .filter(fund__enabled=True)
        .order_by("-date")
        .values_list("date", flat=True)
        .distinct()[:2]
    )
    today_percent = Decimal("0.0")
    if len(latest_dates) == 2:
        latest_agg = Snapshot.objects.filter(
            fund__enabled=True,
            date=latest_dates[0],
        ).aggregate(
            cash=Sum("cash_value"),
            holdings=Sum("holdings_value"),
        )
        previous_agg = Snapshot.objects.filter(
            fund__enabled=True,
            date=latest_dates[1],
        ).aggregate(
            cash=Sum("cash_value"),
            holdings=Sum("holdings_value"),
        )
        latest_total = (latest_agg["cash"] or Decimal("0")) + (latest_agg["holdings"] or Decimal("0"))
        previous_total = (previous_agg["cash"] or Decimal("0")) + (previous_agg["holdings"] or Decimal("0"))
        if previous_total > 0:
            today_percent = ((latest_total - previous_total) / previous_total) * Decimal("100")

    return {
        "total_value": float(total_value),
        "cash": float(total_cash),
        "return_amount": float(return_amount),
        "invested": float(invested),
        "holdings_count": holdings_count,
        "trades_count": trades_count,
        "trade_pnl": float(trade_pnl_percent),
        "holdings_pnl": float(holdings_pnl_pct),
        "return_percent": float(return_percent),
        "today_percent": float(today_percent),
    }


def _compute_today_percent_for_fund(fund: Profile) -> Decimal:
    """Snapshot-based day-over-day percent change for one fund."""
    latest_two = list(
        Snapshot.objects
        .filter(fund=fund)
        .order_by("-date", "-created")[:2]
    )
    if len(latest_two) < 2:
        return Decimal("0.0")

    latest = latest_two[0]
    previous = latest_two[1]

    latest_total = (latest.cash_value or Decimal("0")) + (latest.holdings_value or Decimal("0"))
    previous_total = (previous.cash_value or Decimal("0")) + (previous.holdings_value or Decimal("0"))

    if previous_total <= 0:
        return Decimal("0.0")

    return ((latest_total - previous_total) / previous_total) * Decimal("100")


def get_portfolio_dashboard_data(fund: FundArg) -> Dict[str, Any]:
    """
    Header-style metrics for one fund, or all enabled funds.

    * ``fund`` is a ``Profile`` → that fund only.
    * ``fund is None`` → empty dashboard.
    * ``fund == 0`` → aggregate every enabled ``Profile`` (holdings, cash, trades).
    """
    if fund is None:
        return dict(EMPTY_PORTFOLIO_DASHBOARD)
    if fund == 0:
        return _dashboard_all_enabled_funds()

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

    # Fund age drives EST ABV annualization.
    # `Profile.created` is nullable, so we defensively handle missing values.
    if getattr(fund, "created", None):
        days_active = (timezone.now() - fund.created).days
    else:
        days_active = 0
    days_active = max(days_active, 1)

    # Annualize the current return% over the observed time window.
    # Linear annualization: return_percent * (365 / days_active).
    est_abv_percent = (return_percent * (Decimal(365) / Decimal(days_active)))

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

    today_percent = _compute_today_percent_for_fund(fund)

    return {
        "total_value": float(total_value),
        "cash": float(fund.cash),
        "return_amount": float(return_amount),
        "invested": float(invested),
        "holdings_count": holdings_count,
        "trades_count": trades_count,
        "trade_pnl": float(trade_pnl_percent),
        "holdings_pnl": float(holdings_pnl_pct),
        "return_percent": float(return_percent),
        "estab_days": days_active,
        "est_abv_percent": float(est_abv_percent),
        "today_percent": float(today_percent),
    }
