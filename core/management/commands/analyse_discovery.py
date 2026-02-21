"""
Analyse Discovery Management Command

Trade-based discovery success: BUY trades in the last N days, attributed to
advisors via Discovery(sa, stock), with outcomes from FIFO-matched SELLs or
current holding. Outputs a scoreboard (winners vs losers, % gain/loss). No
weight updates in this version.

Usage:
    python manage.py analyse_discovery           # Last 14 days (lookback from now)
    python manage.py analyse_discovery --days 7 # Last 7 days ago
    python manage.py analyse_discovery --user joe
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime as dt, timedelta
from decimal import Decimal

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.models import Discovery, Holding, Stock, Trade


# Scoreboard column keys
COL_TRADES = "Trades"
COL_WINNERS = "Winners"
COL_LOSERS = "Losers"
COL_WIN_RATE_PCT = "Win rate %"
COL_AVG_PCT = "Avg % change"
COL_TOTAL_PCT = "Total % gain/loss"

COLUMNS = (COL_TRADES, COL_WINNERS, COL_LOSERS, COL_WIN_RATE_PCT, COL_AVG_PCT, COL_TOTAL_PCT)


@dataclass
class LotOutcome:
    """Result of FIFO for one BUY lot: cost, exit value, and advisor."""
    advisor: str
    cost: Decimal
    exit_value: Decimal
    pct: float  # (exit_value - cost) / cost * 100
    winner: bool  # pct >= 0


def _cutoff_lookback(days: int):
    """Start of lookback: start of calendar day N days ago (so --days 2 includes full Thursday if today is Saturday)."""
    now = timezone.now()
    period_date = now.date() - timedelta(days=days)
    midnight = dt.combine(period_date, dt.min.time())
    if timezone.get_current_timezone():
        return timezone.make_aware(midnight, timezone.get_current_timezone())
    return midnight


def _discovery_advisor_map(sa_ids, stock_ids):
    """(sa_id, stock_id) -> advisor name. Last discovery per (sa, stock) to match UI."""
    discoveries = (
        Discovery.objects.filter(sa_id__in=sa_ids, stock_id__in=stock_ids)
        .select_related("advisor")
        .order_by("sa_id", "stock_id", "id")
    )
    out = {}
    for d in discoveries:
        key = (d.sa_id, d.stock_id)
        out[key] = d.advisor.name  # last wins (same as UI trades view)
    return out


def _fifo_outcomes_for_user_stock(user_id, stock_id, cutoff_dt, advisor_for_buy):
    """
    For (user, stock) get all BUY and SELL trades, run FIFO. Return list of
    LotOutcome for BUYs that fall in period (sa.started >= cutoff_dt).
    """
    trades = (
        Trade.objects.filter(user_id=user_id, stock_id=stock_id, action__in=("BUY", "SELL"))
        .select_related("sa", "stock")
        .order_by("created")
    )
    # Queue: list of (trade_id, shares_remaining, cost_per_share)
    queue = []
    # buy_id -> (shares_sold, proceeds)
    sold_from = defaultdict(lambda: {"shares": 0, "proceeds": Decimal("0")})

    for t in trades:
        if t.action == "BUY":
            queue.append((t.id, t.shares, t.price))
        else:
            # SELL: consume from queue (FIFO)
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

    # Shares still open per buy_id (from queue)
    remaining_by_buy_id = {buy_id: sh for (buy_id, sh, _) in queue}

    # Build outcomes only for BUYs in period
    outcomes = []
    buy_trades = (
        Trade.objects.filter(user_id=user_id, stock_id=stock_id, action="BUY")
        .select_related("sa", "stock", "stock__advisor")
        .order_by("created")
    )
    for t in buy_trades:
        if t.sa.started < cutoff_dt:
            continue
        # Prefer Stock.advisor (matches trade table join to core_stock.advisor_id); else Discovery
        advisor = None
        if t.stock and getattr(t.stock, "advisor_id", None):
            advisor = t.stock.advisor.name
        if not advisor:
            advisor = advisor_for_buy.get((t.sa_id, t.stock_id))
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


def _run_fifo_and_collect_lots(buy_trades_in_period, advisor_for_buy, cutoff_dt):
    """
    For each (user, stock) that appears in buy_trades_in_period, run FIFO and
    collect LotOutcome for BUYs in period. Return list of LotOutcome. Current
    price for open lots must be set later (we'll refresh stocks).
    """
    # Group period BUYs by (user_id, stock_id)
    by_user_stock = defaultdict(list)
    for t in buy_trades_in_period:
        by_user_stock[(t.user_id, t.stock_id)].append(t)

    all_outcomes = []
    stocks_to_refresh = set()

    for (user_id, stock_id), buys in by_user_stock.items():
        outcomes = _fifo_outcomes_for_user_stock(user_id, stock_id, cutoff_dt, advisor_for_buy)
        for o in outcomes:
            all_outcomes.append(o)
        # We need current price for this stock for any open lot - will refresh below
        stocks_to_refresh.add(stock_id)

    return all_outcomes, stocks_to_refresh


def _refresh_stock_prices(stock_ids):
    """Update Stock.price for given ids (persisted to DB for later queries)."""
    for stock in Stock.objects.filter(id__in=stock_ids):
        stock.refresh()


def _build_scoreboard(lot_outcomes):
    """From list of LotOutcome, build (row, col) -> value. Row = advisor or 'Total'."""
    by_advisor = defaultdict(list)
    for o in lot_outcomes:
        by_advisor[o.advisor].append(o)

    scoreboard = {}
    all_pcts = []

    for advisor, outcomes in sorted(by_advisor.items()):
        n = len(outcomes)
        winners = sum(1 for o in outcomes if o.winner)
        losers = n - winners
        win_rate = (winners / n * 100) if n else 0.0
        avg_pct = sum(o.pct for o in outcomes) / n if n else 0.0
        total_cost = sum(o.cost for o in outcomes)
        total_exit = sum(o.exit_value for o in outcomes)
        total_pct = float((total_exit - total_cost) / total_cost * 100) if total_cost else 0.0

        scoreboard[(advisor, COL_TRADES)] = n
        scoreboard[(advisor, COL_WINNERS)] = winners
        scoreboard[(advisor, COL_LOSERS)] = losers
        scoreboard[(advisor, COL_WIN_RATE_PCT)] = win_rate
        scoreboard[(advisor, COL_AVG_PCT)] = avg_pct
        scoreboard[(advisor, COL_TOTAL_PCT)] = total_pct
        all_pcts.extend(o.pct for o in outcomes)

    # Total row
    if by_advisor:
        n_total = len(lot_outcomes)
        winners_total = sum(1 for o in lot_outcomes if o.winner)
        scoreboard[("Total", COL_TRADES)] = n_total
        scoreboard[("Total", COL_WINNERS)] = winners_total
        scoreboard[("Total", COL_LOSERS)] = n_total - winners_total
        scoreboard[("Total", COL_WIN_RATE_PCT)] = (winners_total / n_total * 100) if n_total else 0.0
        scoreboard[("Total", COL_AVG_PCT)] = sum(all_pcts) / len(all_pcts) if all_pcts else 0.0
        total_cost_all = sum(o.cost for o in lot_outcomes)
        total_exit_all = sum(o.exit_value for o in lot_outcomes)
        scoreboard[("Total", COL_TOTAL_PCT)] = (
            float((total_exit_all - total_cost_all) / total_cost_all * 100) if total_cost_all else 0.0
        )

    return scoreboard


def _print_scoreboard(scoreboard, stdout, cols=COLUMNS):
    """Print scoreboard as fixed-width table."""
    rows = sorted({r for (r, c) in scoreboard}, key=lambda x: (x != "Total", x))
    if not rows:
        return

    width = {c: max(len(c), 8) for c in cols}
    for (r, c), v in scoreboard.items():
        if c in (COL_TRADES, COL_WINNERS, COL_LOSERS):
            width[c] = max(width[c], len(str(int(v))))
        else:
            width[c] = max(width[c], len(f"{v:.1f}"))
    row_label_width = max(len(r) for r in rows)

    sep = "  "
    header = sep.join(c.ljust(width[c]) for c in cols)
    stdout.write((" " * row_label_width) + sep + header)
    stdout.write("-" * (row_label_width + len(sep) + sum(width[c] for c in cols) + len(sep) * (len(cols) - 1)))

    for r in rows:
        cells = []
        for c in cols:
            v = scoreboard.get((r, c), "")
            if c in (COL_TRADES, COL_WINNERS, COL_LOSERS):
                cells.append(str(int(v)).rjust(width[c]))
            else:
                cells.append(f"{v:.1f}".rjust(width[c]))
        stdout.write(r.ljust(row_label_width) + sep + sep.join(cells))


class Command(BaseCommand):
    help = "Discovery success from trades: BUYs in last N days, FIFO outcomes, scoreboard (winners/losers, %)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=14,
            help="Lookback days (days ago from now) for BUY trades (default: 14)",
        )
        parser.add_argument(
            "--user",
            type=str,
            default=None,
            metavar="USERNAME",
            help="Only include trades for this user (username).",
        )

    def handle(self, *args, **options):
        days = options["days"]
        username = options.get("user")
        cutoff = _cutoff_lookback(days)

        qs = Trade.objects.filter(
            action="BUY",
            sa__started__gte=cutoff,
        )
        if username:
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                raise CommandError(f"User '{username}' not found.")
            qs = qs.filter(user=user)
        buy_trades = qs.select_related("sa", "user", "stock").order_by("created")
        buy_list = list(buy_trades)
        if not buy_list:
            who = f" for user '{username}'" if username else ""
            self.stdout.write(self.style.WARNING(f"No BUY trades in the last {days} days{who}."))
            return

        sa_ids = [t.sa_id for t in buy_list]
        stock_ids = [t.stock_id for t in buy_list]
        advisor_for_buy = _discovery_advisor_map(sa_ids, stock_ids)

        # Refresh current prices for stocks that might have open positions
        stocks_to_refresh = set(stock_ids)
        _refresh_stock_prices(stocks_to_refresh)

        lot_outcomes, _ = _run_fifo_and_collect_lots(buy_list, advisor_for_buy, cutoff)

        if not lot_outcomes:
            self.stdout.write(self.style.WARNING("No BUY lots in period could be attributed to an advisor."))
            return

        scoreboard = _build_scoreboard(lot_outcomes)
        period_msg = f"last {days} days"
        user_note = f" (user: {username})" if username else ""
        self.stdout.write(f"\nDiscovery success (BUY trades, {period_msg}){user_note}\n")
        _print_scoreboard(scoreboard, self.stdout)
