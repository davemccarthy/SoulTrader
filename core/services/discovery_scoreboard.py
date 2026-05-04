"""
Discovery trade outcomes: FIFO lots, advisor attribution, scoreboard aggregates.

Shared by `analyse_discovery` management command and API (fund-scoped scoreboard).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime as dt, timedelta
from decimal import Decimal
from typing import Any

from django.utils import timezone

from core.models import Advisor, Discovery, Stock, Trade

# Scoreboard column keys (CLI table)
COL_TRADES = "Trades"
COL_WINNERS = "Winners"
COL_LOSERS = "Losers"
COL_WIN_RATE_PCT = "Win rate %"
COL_TOTAL_PCT = "gain/loss %"
COL_PEAK_INVEST = "peak invest"
COL_PNL = "P&L"
COL_RETURN_PEAK = "return"

COLUMNS = (
    COL_TRADES,
    COL_WINNERS,
    COL_LOSERS,
    COL_WIN_RATE_PCT,
    COL_TOTAL_PCT,
    COL_PEAK_INVEST,
    COL_PNL,
    COL_RETURN_PEAK,
)


@dataclass
class LotOutcome:
    """Result of FIFO for one BUY lot: cost, exit value, and advisor name."""

    advisor: str
    cost: Decimal
    exit_value: Decimal
    pct: float
    winner: bool


def cutoff_lookback(days: int):
    """Start of lookback: start of calendar day N days ago."""
    now = timezone.now()
    period_date = now.date() - timedelta(days=days)
    midnight = dt.combine(period_date, dt.min.time())
    if timezone.get_current_timezone():
        return timezone.make_aware(midnight, timezone.get_current_timezone())
    return midnight


def discovery_advisor_map(sa_ids: list[int], stock_ids: list[int]) -> dict[tuple[int, int], str]:
    """(sa_id, stock_id) -> advisor name. First discovery per (sa, stock) by created."""
    discoveries = (
        Discovery.objects.filter(sa_id__in=sa_ids, stock_id__in=stock_ids)
        .select_related("advisor")
        .order_by("sa_id", "stock_id", "created")
    )
    out: dict[tuple[int, int], str] = {}
    for d in discoveries:
        key = (d.sa_id, d.stock_id)
        if key not in out:
            out[key] = d.advisor.name
    return out


def fifo_outcomes_for_user_stock(
    user_id: int,
    stock_id: int,
    cutoff_dt,
    advisor_for_buy: dict[tuple[int, int], str],
) -> list[LotOutcome]:
    trades = (
        Trade.objects.filter(user_id=user_id, stock_id=stock_id, action__in=("BUY", "SELL"))
        .select_related("sa", "stock")
        .order_by("created")
    )
    queue: list[tuple[int, int, Decimal]] = []
    sold_from: dict[int, dict[str, Any]] = defaultdict(lambda: {"shares": 0, "proceeds": Decimal("0")})

    for t in trades:
        if t.action == "BUY":
            queue.append((t.id, t.shares, t.price))
        else:
            remaining = t.shares
            sell_price = t.price
            while remaining > 0 and queue:
                buy_id, lot_shares, cost_per_share = queue[0]
                take = min(remaining, lot_shares)
                sold_from[buy_id]["shares"] += take
                sold_from[buy_id]["proceeds"] += Decimal(take) * sell_price
                remaining -= take
                if lot_shares == take:
                    queue.pop(0)
                else:
                    queue[0] = (buy_id, lot_shares - take, cost_per_share)

    remaining_by_buy_id = {buy_id: sh for (buy_id, sh, _) in queue}

    outcomes: list[LotOutcome] = []
    buy_trades = (
        Trade.objects.filter(user_id=user_id, stock_id=stock_id, action="BUY")
        .select_related("sa", "stock", "stock__advisor")
        .order_by("created")
    )
    for t in buy_trades:
        if t.sa.started < cutoff_dt:
            continue
        advisor = advisor_for_buy.get((t.sa_id, t.stock_id))
        if not advisor and t.stock and getattr(t.stock, "advisor_id", None):
            advisor = t.stock.advisor.name
        if not advisor:
            continue
        cost = Decimal(t.shares) * t.price
        sold = sold_from.get(t.id, {"shares": 0, "proceeds": Decimal("0")})
        proceeds = sold["proceeds"]
        shares_remaining = remaining_by_buy_id.get(t.id, 0)
        current_price = t.stock.price if t.stock else Decimal("0")
        exit_value = proceeds + (Decimal(shares_remaining) * current_price)
        if cost and cost > 0:
            pct = float((exit_value - cost) / cost * 100)
        else:
            pct = 0.0
        outcomes.append(
            LotOutcome(
                advisor=advisor,
                cost=cost,
                exit_value=exit_value,
                pct=pct,
                winner=(pct >= 0),
            )
        )
    return outcomes


def run_fifo_and_collect_lots(
    buy_trades_in_period: list[Trade],
    advisor_for_buy: dict[tuple[int, int], str],
    cutoff_dt,
) -> tuple[list[LotOutcome], set[int]]:
    by_user_stock: dict[tuple[int, int], list[Trade]] = defaultdict(list)
    for t in buy_trades_in_period:
        by_user_stock[(t.user_id, t.stock_id)].append(t)

    all_outcomes: list[LotOutcome] = []
    stocks_to_refresh: set[int] = set()

    for (user_id, stock_id), _buys in by_user_stock.items():
        outcomes = fifo_outcomes_for_user_stock(user_id, stock_id, cutoff_dt, advisor_for_buy)
        for o in outcomes:
            all_outcomes.append(o)
        stocks_to_refresh.add(stock_id)

    return all_outcomes, stocks_to_refresh


def refresh_stock_prices(stock_ids: set[int]) -> None:
    for stock in Stock.objects.filter(id__in=stock_ids):
        stock.refresh()


def peak_holdings_cost(user_ids: list[int], cutoff) -> float:
    trades = (
        Trade.objects.filter(
            user_id__in=user_ids,
            action__in=("BUY", "SELL"),
            created__gte=cutoff,
        )
        .order_by("created")
    )
    positions: dict[tuple[int, int], list[tuple[int, Decimal]]] = defaultdict(list)
    peak = Decimal("0")

    for t in trades:
        key = (t.user_id, t.stock_id)
        if t.action == "BUY":
            positions[key].append((t.shares, t.price))
        else:
            remaining = t.shares
            while remaining > 0 and positions[key]:
                lot_shares, cost_per_share = positions[key][0]
                take = min(remaining, lot_shares)
                remaining -= take
                if lot_shares == take:
                    positions[key].pop(0)
                else:
                    positions[key][0] = (lot_shares - take, cost_per_share)

        total_cost = sum(
            Decimal(str(sh)) * Decimal(str(cost))
            for (uid, sid), lots in positions.items()
            for sh, cost in lots
        )
        if total_cost > peak:
            peak = total_cost

    return float(peak)


def build_scoreboard(
    lot_outcomes: list[LotOutcome],
    return_peak_pct: float | None = None,
    peak_cost: float | None = None,
) -> dict[tuple[str, str], Any]:
    """Sparse (row_label, column) -> value for CLI printing."""
    by_advisor: dict[str, list[LotOutcome]] = defaultdict(list)
    for o in lot_outcomes:
        by_advisor[o.advisor].append(o)

    scoreboard: dict[tuple[str, str], Any] = {}

    for advisor, outcomes in sorted(by_advisor.items()):
        n = len(outcomes)
        winners = sum(1 for o in outcomes if o.winner)
        losers = n - winners
        win_rate = (winners / n * 100) if n else 0.0
        total_cost = sum(o.cost for o in outcomes)
        total_exit = sum(o.exit_value for o in outcomes)
        total_pct = float((total_exit - total_cost) / total_cost * 100) if total_cost else 0.0
        pnl = float(total_exit - total_cost)

        scoreboard[(advisor, COL_TRADES)] = n
        scoreboard[(advisor, COL_WINNERS)] = winners
        scoreboard[(advisor, COL_LOSERS)] = losers
        scoreboard[(advisor, COL_WIN_RATE_PCT)] = win_rate
        scoreboard[(advisor, COL_TOTAL_PCT)] = total_pct
        scoreboard[(advisor, COL_PNL)] = pnl

    if by_advisor:
        n_total = len(lot_outcomes)
        winners_total = sum(1 for o in lot_outcomes if o.winner)
        scoreboard[("Total", COL_TRADES)] = n_total
        scoreboard[("Total", COL_WINNERS)] = winners_total
        scoreboard[("Total", COL_LOSERS)] = n_total - winners_total
        scoreboard[("Total", COL_WIN_RATE_PCT)] = (winners_total / n_total * 100) if n_total else 0.0
        total_cost_all = sum(o.cost for o in lot_outcomes)
        total_exit_all = sum(o.exit_value for o in lot_outcomes)
        scoreboard[("Total", COL_TOTAL_PCT)] = (
            float((total_exit_all - total_cost_all) / total_cost_all * 100) if total_cost_all else 0.0
        )
        if peak_cost is not None:
            scoreboard[("Total", COL_PEAK_INVEST)] = peak_cost
        scoreboard[("Total", COL_PNL)] = float(total_exit_all - total_cost_all)
        if return_peak_pct is not None:
            scoreboard[("Total", COL_RETURN_PEAK)] = return_peak_pct

    return scoreboard


def print_scoreboard(scoreboard: dict[tuple[str, str], Any], stdout, cols: tuple = COLUMNS) -> None:
    rows = sorted({r for (r, c) in scoreboard}, key=lambda x: (x != "Total", x))
    if not rows:
        return

    width = {c: max(len(c), 8) for c in cols}
    for (r, c), v in scoreboard.items():
        if c in (COL_TRADES, COL_WINNERS, COL_LOSERS):
            width[c] = max(width[c], len(str(int(v))))
        elif c in (COL_PNL, COL_PEAK_INVEST):
            width[c] = max(width[c], len(f"{v:,.2f}"))
        elif c == COL_RETURN_PEAK and v != "" and v is not None:
            width[c] = max(width[c], len(f"{v:.1f}"))
        else:
            width[c] = max(width[c], len(f"{v:.1f}"))
    row_label_width = max(len(r) for r in rows)

    sep = "  "
    header = sep.join(c.ljust(width[c]) for c in cols)
    stdout.write((" " * row_label_width) + sep + header + "\n")
    stdout.write(
        "-" * (row_label_width + len(sep) + sum(width[c] for c in cols) + len(sep) * (len(cols) - 1)) + "\n"
    )

    for r in rows:
        cells = []
        for c in cols:
            v = scoreboard.get((r, c), "")
            if c in (COL_TRADES, COL_WINNERS, COL_LOSERS):
                cells.append(str(int(v)).rjust(width[c]))
            elif c in (COL_PNL, COL_PEAK_INVEST):
                cells.append((f"{v:,.2f}" if v != "" and v is not None else "-").rjust(width[c]))
            elif c == COL_RETURN_PEAK:
                cells.append((f"{v:.1f}" if v != "" and v is not None else "-").rjust(width[c]))
            else:
                cells.append(f"{v:.1f}".rjust(width[c]))
        stdout.write(r.ljust(row_label_width) + sep + sep.join(cells) + "\n")


def fund_period_buy_trades(fund_id: int, days: int) -> list[Trade]:
    """BUY trades for a fund in the lookback window (by SA started)."""
    cutoff = cutoff_lookback(days)
    return list(
        Trade.objects.filter(
            fund_id=fund_id,
            action="BUY",
            sa__started__gte=cutoff,
        )
        .select_related("sa", "user", "stock")
        .order_by("created")
    )


def fund_advisor_scoreboard_rows(
    fund_id: int,
    days: int,
    *,
    refresh_prices: bool = False,
) -> list[dict[str, Any]]:
    """
    Per-advisor stats for one fund (BUY lots in lookback attributed via Discovery).

    API-friendly rows: advisor_id, trades, winners, losers, win_rate, gain_loss_pct.
    Omits advisors with zero attributed lots in the window.
    """
    buy_list = fund_period_buy_trades(fund_id, days)
    if not buy_list:
        return []

    cutoff = cutoff_lookback(days)
    sa_ids = [t.sa_id for t in buy_list]
    stock_ids = [t.stock_id for t in buy_list]
    advisor_for_buy = discovery_advisor_map(sa_ids, stock_ids)

    stocks_to_refresh = set(stock_ids)
    if refresh_prices:
        refresh_stock_prices(stocks_to_refresh)

    lot_outcomes, _ = run_fifo_and_collect_lots(buy_list, advisor_for_buy, cutoff)
    if not lot_outcomes:
        return []

    by_name: dict[str, list[LotOutcome]] = defaultdict(list)
    for o in lot_outcomes:
        by_name[o.advisor].append(o)

    names = list(by_name.keys())
    id_by_name = {
        a.name: a.id
        for a in Advisor.objects.filter(name__in=names).only("id", "name")
    }

    rows: list[dict[str, Any]] = []
    for name in sorted(by_name.keys()):
        outcomes = by_name[name]
        advisor_id = id_by_name.get(name)
        if advisor_id is None:
            continue
        n = len(outcomes)
        winners = sum(1 for o in outcomes if o.winner)
        losers = n - winners
        win_rate = (winners / n * 100) if n else 0.0
        total_cost = sum(o.cost for o in outcomes)
        total_exit = sum(o.exit_value for o in outcomes)
        gain_loss_pct = float((total_exit - total_cost) / total_cost * 100) if total_cost else 0.0

        rows.append(
            {
                "advisor_id": advisor_id,
                "trades": n,
                "winners": winners,
                "losers": losers,
                "win_rate": round(win_rate, 1),
                "gain_loss_pct": round(gain_loss_pct, 2),
            }
        )

    rows.sort(key=lambda r: r["advisor_id"])
    return rows
