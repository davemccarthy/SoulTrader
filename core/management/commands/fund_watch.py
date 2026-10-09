"""
Live fund holdings board: refresh quotes and show P&L (observe only).

Usage:
    python manage.py fund_watch PLSE-S
    python manage.py fund_watch PLSE-S --loop 3
    python manage.py fund_watch PLSE-S --loop 3 --gain-only

Ctrl+C stops --loop. No discovers or sells — use force_sell to exit.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

import pytz
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.models import Holding, Profile

ET = pytz.timezone("US/Eastern")


def _position_pnl(shares, price, average_price):
    """P&L if sold at price. Cost basis is average_price."""
    price = price or Decimal("0")
    avg = average_price or Decimal("0")
    shares_d = Decimal(shares)
    proceeds = shares_d * price
    cost = shares_d * avg
    pnl = proceeds - cost
    pnl_pct = (pnl / cost * Decimal("100")) if cost else None
    return proceeds, cost, pnl, pnl_pct


def _path_label(holding: Holding) -> str:
    """Short path/advisor tag for the board."""
    discovery = holding.discovery
    if discovery and discovery.explanation:
        lead = str(discovery.explanation).split("|", 1)[0].strip()
        # "RECOVERY: …" / "IMPULSE: …" / "TROUGH: …"
        if ":" in lead:
            kind = lead.split(":", 1)[0].strip()
            if kind:
                return kind[:12]
        if lead:
            return lead[:12]
    if discovery and discovery.advisor_id and discovery.advisor:
        return (discovery.advisor.name or discovery.advisor.python_class or "?")[:12]
    if holding.stock and holding.stock.advisor_id and holding.stock.advisor:
        return (holding.stock.advisor.name or "?")[:12]
    return "—"


def _age_label(holding: Holding) -> str:
    created = holding.created
    if not created:
        return "—"
    if timezone.is_naive(created):
        created = timezone.make_aware(created, timezone.get_current_timezone())
    delta = timezone.now() - created
    mins = int(delta.total_seconds() // 60)
    if mins < 60:
        return f"{mins}m"
    hours = mins // 60
    if hours < 48:
        return f"{hours}h"
    return f"{hours // 24}d"


class Command(BaseCommand):
    help = (
        "Watch a fund's open holdings with live quote refresh (no sells). "
        "Use --loop to redraw; Ctrl+C to stop."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "fund",
            type=str,
            help="Profile.name (exact match), e.g. PLSE-S",
        )
        parser.add_argument(
            "--loop",
            type=float,
            default=None,
            metavar="SECONDS",
            help="Redraw every SECONDS (Ctrl+C to stop). Omit for a single pass.",
        )
        parser.add_argument(
            "--gain-only",
            action="store_true",
            help="Only show holdings with P&L%% > 0 after refresh.",
        )

    def handle(self, *args, **options):
        fund_name = (options["fund"] or "").strip()
        fund = Profile.objects.filter(name=fund_name).first()
        if not fund:
            raise CommandError(f'Fund "{fund_name}" not found')

        loop_s: Optional[float] = options.get("loop")
        if loop_s is not None and loop_s <= 0:
            raise CommandError("--loop must be a positive number of seconds")

        gain_only = bool(options.get("gain_only"))

        try:
            while True:
                self._render_board(fund, gain_only=gain_only, looping=loop_s is not None)
                if loop_s is None:
                    break
                time.sleep(loop_s)
        except KeyboardInterrupt:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Stopped (Ctrl+C)."))

    def _clear_screen(self) -> None:
        # ANSI clear + home; fine on remote ssh / macOS Terminal.
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()

    def _render_board(self, fund: Profile, *, gain_only: bool, looping: bool) -> None:
        if looping:
            self._clear_screen()

        holdings = list(
            Holding.objects.filter(fund=fund, shares__gt=0)
            .select_related("stock", "stock__advisor", "discovery", "discovery__advisor")
            .order_by("stock__symbol", "id")
        )

        now_et = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S %Z")
        mode = "gain-only" if gain_only else "all"
        loop_note = " | Ctrl+C stop" if looping else ""
        self.stdout.write(
            f"{fund.name}  {now_et}  ({mode}{loop_note})  "
            f"holdings={len(holdings)}"
        )

        if not holdings:
            self.stdout.write(self.style.WARNING("No open holdings."))
            return

        rows: list[dict[str, Any]] = []
        for holding in holdings:
            holding.stock.refresh()
            price = holding.stock.price
            avg = holding.average_price
            proceeds, cost, pnl, pnl_pct = _position_pnl(
                holding.shares, price, avg
            )
            if gain_only and (pnl_pct is None or pnl_pct <= 0):
                continue
            rows.append(
                {
                    "symbol": holding.stock.symbol,
                    "path": _path_label(holding),
                    "shares": holding.shares,
                    "tranches": holding.tranches or 0,
                    "avg": avg or Decimal("0"),
                    "last": price or Decimal("0"),
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "value": proceeds,
                    "age": _age_label(holding),
                }
            )

        # Green first when chasing gains; else worst first for situational awareness.
        if gain_only:
            rows.sort(
                key=lambda r: float(r["pnl_pct"] or 0),
                reverse=True,
            )
        else:
            rows.sort(key=lambda r: float(r["pnl_pct"] or 0))

        if not rows:
            self.stdout.write(
                self.style.WARNING("No rows to show (nothing green under --gain-only).")
            )
            return

        header = (
            f"{'SYM':<6} {'PATH':<12} {'Sh':>4} {'Tr':>3} "
            f"{'Avg':>9} {'Last':>9} {'P&L%':>8} {'Value':>10} {'Age':>5}"
        )
        self.stdout.write(header)
        self.stdout.write("-" * len(header))

        total_value = Decimal("0")
        total_cost = Decimal("0")
        for row in rows:
            pct = row["pnl_pct"]
            pct_txt = "n/a" if pct is None else f"{pct:+.2f}%"
            if pct is not None and pct > 0:
                pct_styled = self.style.SUCCESS(f"{pct_txt:>8}")
            elif pct is not None and pct < 0:
                pct_styled = self.style.ERROR(f"{pct_txt:>8}")
            else:
                pct_styled = f"{pct_txt:>8}"

            line = (
                f"{row['symbol']:<6} {row['path']:<12} {row['shares']:>4} {row['tranches']:>3} "
                f"{row['avg']:>9.2f} {row['last']:>9.2f} "
            )
            self.stdout.write(line, ending="")
            self.stdout.write(pct_styled, ending="")
            self.stdout.write(f" {row['value']:>10,.2f} {row['age']:>5}")

            total_value += row["value"]
            # Reconstruct cost from value and pct when possible
            if pct is not None and pct != -100:
                # value = cost * (1 + pct/100) → cost = value / (1 + pct/100)
                total_cost += row["value"] / (Decimal("1") + pct / Decimal("100"))
            else:
                total_cost += row["value"]

        self.stdout.write("-" * len(header))
        book_pnl = total_value - total_cost
        book_pct = (
            (book_pnl / total_cost * Decimal("100")) if total_cost else None
        )
        book_pct_txt = "n/a" if book_pct is None else f"{book_pct:+.2f}%"
        if book_pct is not None and book_pct > 0:
            book_pct_styled = self.style.SUCCESS(book_pct_txt)
        elif book_pct is not None and book_pct < 0:
            book_pct_styled = self.style.ERROR(book_pct_txt)
        else:
            book_pct_styled = book_pct_txt
        self.stdout.write(
            f"shown n={len(rows)}  value ${total_value:,.2f}  "
            f"P&L ${book_pnl:+,.2f} (",
            ending="",
        )
        self.stdout.write(book_pct_styled, ending="")
        self.stdout.write(")")
