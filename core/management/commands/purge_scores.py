"""
Dry-run purge_score ranking for a fund's holdings.

Usage:
    python manage.py purge_scores --fund AGR1
    python manage.py purge_scores --fund EXP1 --refresh
    python manage.py purge_scores --fund AGR1 --min-score 30
"""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from core.models import Holding, Profile
from core.services.sentiment import holding_purge_metrics, purge_score


class Command(BaseCommand):
    help = "Rank fund holdings by purge_score (dry-run; no sells)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--fund",
            type=str,
            required=True,
            help="Profile.name (exact match), e.g. AGR1",
        )
        parser.add_argument(
            "--refresh",
            action="store_true",
            help="Refresh stock prices before scoring.",
        )
        parser.add_argument(
            "--min-score",
            type=float,
            default=None,
            help=f"Highlight eligible rows (default: Profile.PURGE_MIN_SCORE).",
        )

    def handle(self, *args, **options):
        fund_name = (options["fund"] or "").strip()
        fund = Profile.objects.filter(name=fund_name).first()
        if not fund:
            raise CommandError(f'Fund "{fund_name}" not found')

        min_score = options["min_score"]
        if min_score is None:
            min_score = Profile.PURGE_MIN_SCORE

        ratio = float(fund.equity_ratio())
        recycle_at = fund.equity_target()
        buy_until = fund.equity_buy_threshold()
        over_recycle = fund.at_or_over_equity_cap()
        over_buy = fund.at_or_over_equity_buy_cap()

        self.stdout.write(
            f"{fund.name} sentiment={fund.sentiment} "
            f"ratio={ratio:.3f} recycle_at={recycle_at} buy_until={buy_until} "
            f"over_recycle={over_recycle} over_buy={over_buy}"
        )
        self.stdout.write(
            f"weights: slots=0.25 age=0.20 gain=0.40 flat=0.15  min_score={min_score}"
        )

        holdings = list(
            Holding.objects.filter(fund=fund, shares__gt=0)
            .select_related("stock", "discovery")
            .order_by("id")
        )
        if not holdings:
            self.stdout.write("No open holdings.")
            return

        rows = []
        for holding in holdings:
            if options["refresh"]:
                holding.stock.refresh()
            score, components = purge_score(holding, fund)
            metrics = holding_purge_metrics(holding, fund)
            rows.append(
                {
                    "symbol": holding.stock.symbol,
                    "score": score,
                    "days": metrics["days"],
                    "tranches": metrics["tranches"],
                    "slots": float(metrics["slots"]),
                    "cost": float(metrics["cost"]),
                    "pnl": float(metrics["pnl"]),
                    "pnl_pct": float(metrics["pnl_pct"]) * 100.0,
                    "eligible": score >= min_score,
                    "components": components,
                }
            )

        rows.sort(key=lambda r: (r["score"], r["cost"]), reverse=True)

        header = (
            f"{'#':>3} {'SYM':6} {'Score':>6} {'Elig':>4} "
            f"{'Days':>4} {'Tr':>3} {'Slots':>5} {'Cost':>8} {'P&L$':>8} {'P&L%':>7}"
        )
        self.stdout.write(header)
        for i, row in enumerate(rows, start=1):
            elig = "yes" if row["eligible"] else "—"
            self.stdout.write(
                f"{i:>3} {row['symbol']:6} {row['score']:6.1f} {elig:>4} "
                f"{row['days']:>4} {row['tranches']:>3} {row['slots']:>5.2f} "
                f"{row['cost']:>8.0f} {row['pnl']:>8.0f} {row['pnl_pct']:>6.1f}%"
            )

        eligible_count = sum(1 for r in rows if r["eligible"])
        self.stdout.write(
            f"\n{eligible_count}/{len(rows)} holdings score>={min_score}"
        )
        if over_recycle and eligible_count == 0 and rows:
            best = rows[0]
            self.stdout.write(
                self.style.WARNING(
                    f"Purge would stall: best={best['score']:.1f} {best['symbol']}"
                )
            )
