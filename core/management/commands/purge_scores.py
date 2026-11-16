"""
Dry-run equity purge ranking (overshoot bands; no sells).

Usage:
    python manage.py purge_scores --fund AGR1
    python manage.py purge_scores --fund BIZ --refresh
    python manage.py purge_scores --fund TRSH --assume-overshoot 0.12
"""

from __future__ import annotations

from decimal import Decimal

from django.core.management.base import BaseCommand, CommandError

from core.models import Holding, Profile
from core.services.sentiment import (
    PURGE_MIN_SLOTS,
    equity_overshoot,
    holding_meets_purge_pnl,
    holding_purge_metrics,
    min_purge_pnl,
    purge_band_for_overshoot,
    purge_band_label,
)


class Command(BaseCommand):
    help = "Rank fund holdings for equity purge (overshoot-band dry-run)."

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
            "--assume-overshoot",
            type=float,
            default=None,
            help=(
                "Pretend overshoot (equity_ratio - recycle_at) for band preview "
                "when under target or DISABLED (e.g. 0.12)."
            ),
        )

    def handle(self, *args, **options):
        fund_name = (options["fund"] or "").strip()
        fund = Profile.objects.filter(name=fund_name).first()
        if not fund:
            raise CommandError(f'Fund "{fund_name}" not found')

        ratio = float(fund.equity_ratio())
        recycle_at = fund.equity_target()
        buy_until = fund.equity_buy_threshold()
        over_recycle = fund.at_or_over_equity_cap()
        over_buy = fund.at_or_over_equity_buy_cap()
        overshoot = equity_overshoot(fund)

        assumed = options["assume_overshoot"]
        if assumed is not None:
            overshoot = Decimal(str(assumed))
            _min_os, g = purge_band_for_overshoot(overshoot)
            min_pnl = (g * fund.average_spend()).quantize(Decimal("0.01"))
            band = purge_band_label(overshoot)
        elif overshoot is None:
            min_pnl = None
            band = "n/a"
        else:
            min_pnl = min_purge_pnl(fund)
            # If under target, still show pickiest band for preview.
            if overshoot < 0 and assumed is None:
                _min_os, g = purge_band_for_overshoot(Decimal("0"))
                min_pnl = (g * fund.average_spend()).quantize(Decimal("0.01"))
                band = f"under (preview pickiest {purge_band_label(Decimal('0'))})"
            else:
                band = purge_band_label(overshoot)

        self.stdout.write(
            f"{fund.name} sentiment={fund.sentiment} "
            f"ratio={ratio:.3f} recycle_at={recycle_at} buy_until={buy_until} "
            f"over_recycle={over_recycle} over_buy={over_buy}"
        )
        os_disp = f"{float(overshoot):.3f}" if overshoot is not None else "n/a"
        self.stdout.write(
            f"overshoot={os_disp} band={band} min_pnl=${min_pnl} "
            f"min_slots={PURGE_MIN_SLOTS}"
        )

        holdings = list(
            Holding.objects.filter(fund=fund, shares__gt=0)
            .select_related("stock", "discovery")
            .order_by("id")
        )
        if not holdings:
            self.stdout.write("No open holdings.")
            return

        if min_pnl is None:
            self.stdout.write("Sentiment DISABLED — no purge band (use --assume-overshoot).")
            min_pnl = (Decimal("0.05") * fund.average_spend()).quantize(Decimal("0.01"))
            self.stdout.write(f"Showing preview with pickiest min_pnl=${min_pnl}")

        rows = []
        for holding in holdings:
            if options["refresh"]:
                holding.stock.refresh()
            metrics = holding_purge_metrics(holding, fund)
            dust = metrics["slots"] < PURGE_MIN_SLOTS
            eligible = holding_meets_purge_pnl(metrics, min_pnl)
            rows.append(
                {
                    "symbol": holding.stock.symbol,
                    "days": metrics["days"],
                    "tranches": metrics["tranches"],
                    "slots": float(metrics["slots"]),
                    "cost": float(metrics["cost"]),
                    "pnl": float(metrics["pnl"]),
                    "pnl_pct": float(metrics["pnl_pct"]) * 100.0,
                    "eligible": eligible,
                    "dust": dust,
                }
            )

        rows.sort(key=lambda r: (r["pnl"], r["slots"]), reverse=True)

        header = (
            f"{'#':>3} {'SYM':6} {'Elig':>4} {'Dust':>4} "
            f"{'Days':>4} {'Tr':>3} {'Slots':>5} {'Cost':>8} {'P&L$':>8} {'P&L%':>7}"
        )
        self.stdout.write(header)
        for i, row in enumerate(rows, start=1):
            elig = "yes" if row["eligible"] else "—"
            dust = "yes" if row["dust"] else "—"
            self.stdout.write(
                f"{i:>3} {row['symbol']:6} {elig:>4} {dust:>4} "
                f"{row['days']:>4} {row['tranches']:>3} {row['slots']:>5.2f} "
                f"{row['cost']:>8.0f} {row['pnl']:>8.0f} {row['pnl_pct']:>6.1f}%"
            )

        eligible_count = sum(1 for r in rows if r["eligible"])
        self.stdout.write(
            f"\n{eligible_count}/{len(rows)} holdings eligible "
            f"(pnl>=${min_pnl}, slots>={PURGE_MIN_SLOTS})"
        )
        if over_recycle and eligible_count == 0 and rows:
            best = rows[0]
            self.stdout.write(
                self.style.WARNING(
                    f"Purge would stall: best pnl=${best['pnl']:.0f} {best['symbol']}"
                )
            )
