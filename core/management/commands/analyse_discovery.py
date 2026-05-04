"""
Analyse Discovery Management Command

Trade-based discovery success: BUY trades in the last N days, attributed to
advisors via Discovery(sa, stock), with outcomes from FIFO-matched SELLs or
current holding. Core logic lives in core.services.discovery_scoreboard.

Usage:
    python manage.py analyse_discovery           # Last 14 days (lookback from now)
    python manage.py analyse_discovery --days 7 # Last 7 days ago
    python manage.py analyse_discovery --user joe
"""

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.models import Trade
from core.services.discovery_scoreboard import (
    COLUMNS,
    build_scoreboard,
    cutoff_lookback,
    discovery_advisor_map,
    peak_holdings_cost,
    print_scoreboard,
    refresh_stock_prices,
    run_fifo_and_collect_lots,
)


class Command(BaseCommand):
    help = "Discovery success from trades: BUYs in last N days, FIFO outcomes, scoreboard (winners/losers, %)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=None,
            metavar="N",
            help="Lookback days for BUY trades (default: 14, or user's account age when --user is set)",
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
        user = None

        if username:
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                raise CommandError(f"User '{username}' not found.")
            if days is None:
                days = (timezone.now() - user.date_joined).days
                if days <= 0:
                    days = 1
        if days is None:
            days = 14

        cutoff = cutoff_lookback(days)
        qs = Trade.objects.filter(
            action="BUY",
            sa__started__gte=cutoff,
        )
        if user is not None:
            qs = qs.filter(user=user)
        buy_trades = qs.select_related("sa", "user", "stock").order_by("created")
        buy_list = list(buy_trades)
        if not buy_list:
            who = f" for user '{username}'" if username else ""
            self.stdout.write(self.style.WARNING(f"No BUY trades in the last {days} days{who}."))
            return

        sa_ids = [t.sa_id for t in buy_list]
        stock_ids = [t.stock_id for t in buy_list]
        advisor_for_buy = discovery_advisor_map(sa_ids, stock_ids)

        refresh_stock_prices(set(stock_ids))

        lot_outcomes, _ = run_fifo_and_collect_lots(buy_list, advisor_for_buy, cutoff)

        if not lot_outcomes:
            self.stdout.write(self.style.WARNING("No BUY lots in period could be attributed to an advisor."))
            return

        user_ids = list({t.user_id for t in buy_list})
        peak_cost = peak_holdings_cost(user_ids, cutoff)
        total_pnl = sum(float(o.exit_value - o.cost) for o in lot_outcomes)
        return_peak_pct = (total_pnl / peak_cost * 100) if peak_cost and peak_cost > 0 else None

        scoreboard = build_scoreboard(
            lot_outcomes, return_peak_pct=return_peak_pct, peak_cost=peak_cost if peak_cost else None
        )
        period_msg = f"last {days} days"
        user_note = f" (user: {username})" if username else ""
        self.stdout.write(f"\nDiscovery success (BUY trades, {period_msg}){user_note}\n")
        print_scoreboard(scoreboard, self.stdout, cols=COLUMNS)
