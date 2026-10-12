"""
Live fund holdings board: refresh quotes and show P&L (observe only).

Usage:
    python manage.py fund_watch PLSE-S
    python manage.py fund_watch PLSE-S --loop 6
    python manage.py fund_watch PLSE-S --loop 6 --gain-only
    python manage.py fund_watch PLSE-S --loop 6 --arm 0.8 --giveback 0.2

Trail (manual sell): arm at +arm% vs avg; track HWM after arm; SELL when
price falls below (peak P&L% − giveback) pp of entry. Status returns to ARMED
if price recovers above the exit. Ctrl+C stops --loop. No auto-sell.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

import pytz
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.models import Holding, Profile

ET = pytz.timezone("US/Eastern")


@dataclass
class TrailState:
    """Per-holding trail state for the watch session (in-memory)."""

    last_price: Optional[Decimal] = None
    armed: bool = False
    hwm: Optional[Decimal] = None  # highest price since arm


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


def _pnl_pct_at(price: Decimal, avg: Decimal) -> Optional[Decimal]:
    if not avg or avg <= 0:
        return None
    return (price / avg - Decimal("1")) * Decimal("100")


def _path_label(holding: Holding) -> str:
    """Short path/advisor tag for the board."""
    discovery = holding.discovery
    if discovery and discovery.explanation:
        lead = str(discovery.explanation).split("|", 1)[0].strip()
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


def _update_trail(
    state: TrailState,
    *,
    price: Decimal,
    avg: Decimal,
    pnl_pct: Optional[Decimal],
    arm_pct: Decimal,
    giveback_pp: Decimal,
) -> tuple[str, Optional[Decimal], Optional[Decimal]]:
    """
    Update trail state; return (status, high_pnl_pct, exit_px).

    Status: SELL | ARMED | UP | DOWN | —
    HIGH is peak P&L% since arm (None if not armed).
    """
    prev = state.last_price
    high_pct: Optional[Decimal] = None
    exit_px: Optional[Decimal] = None

    if pnl_pct is not None and avg and avg > 0 and price > 0:
        if pnl_pct >= arm_pct:
            if not state.armed:
                state.armed = True
                state.hwm = price
            elif state.hwm is None or price > state.hwm:
                state.hwm = price

        if state.armed and state.hwm is not None:
            high_pct = _pnl_pct_at(state.hwm, avg)
            if high_pct is not None:
                exit_pct = high_pct - giveback_pp
                exit_px = avg * (Decimal("1") + exit_pct / Decimal("100"))
                if price <= exit_px:
                    state.last_price = price
                    return "SELL", high_pct, exit_px
                state.last_price = price
                return "ARMED", high_pct, exit_px

    # Not armed (or cannot compute trail).
    state.last_price = price
    if prev is None:
        return "—", None, None
    if price > prev:
        return "UP", None, None
    if price < prev:
        return "DOWN", None, None
    return "—", None, None


class Command(BaseCommand):
    help = (
        "Watch a fund's open holdings with live quote refresh (no sells). "
        "Use --loop to redraw; Ctrl+C to stop. Shows HIGH/Status trail for manual exits."
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
        parser.add_argument(
            "--arm",
            type=float,
            default=0.8,
            metavar="PCT",
            help="Arm trail at this unrealised P&L%% vs avg (default: 0.8).",
        )
        parser.add_argument(
            "--giveback",
            type=float,
            default=0.2,
            metavar="PP",
            help="Exit when P&L%% falls giveback percentage-points below HIGH (default: 0.2).",
        )

    def handle(self, *args, **options):
        fund_name = (options["fund"] or "").strip()
        fund = Profile.objects.filter(name=fund_name).first()
        if not fund:
            raise CommandError(f'Fund "{fund_name}" not found')

        loop_s: Optional[float] = options.get("loop")
        if loop_s is not None and loop_s <= 0:
            raise CommandError("--loop must be a positive number of seconds")

        arm_pct = Decimal(str(options["arm"]))
        giveback_pp = Decimal(str(options["giveback"]))
        if arm_pct < 0:
            raise CommandError("--arm must be >= 0")
        if giveback_pp < 0:
            raise CommandError("--giveback must be >= 0")

        gain_only = bool(options.get("gain_only"))
        # holding.id -> TrailState for the life of this process
        trails: dict[int, TrailState] = {}

        try:
            while True:
                self._render_board(
                    fund,
                    gain_only=gain_only,
                    looping=loop_s is not None,
                    arm_pct=arm_pct,
                    giveback_pp=giveback_pp,
                    trails=trails,
                )
                if loop_s is None:
                    break
                time.sleep(loop_s)
        except KeyboardInterrupt:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Stopped (Ctrl+C)."))

    def _clear_screen(self) -> None:
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()

    def _style_status(self, status: str) -> str:
        label = f"{status:<5}"
        if status == "SELL":
            return self.style.ERROR(label)
        if status == "ARMED":
            # WARNING ≈ amber/yellow (Django Style has no orange / no_color flag)
            return self.style.WARNING(label)
        return label  # UP / DOWN / — default terminal colour

    def _render_board(
        self,
        fund: Profile,
        *,
        gain_only: bool,
        looping: bool,
        arm_pct: Decimal,
        giveback_pp: Decimal,
        trails: dict[int, TrailState],
    ) -> None:
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
            f"holdings={len(holdings)}  arm=+{arm_pct}%  giveback={giveback_pp}pp"
        )

        if not holdings:
            self.stdout.write(self.style.WARNING("No open holdings."))
            return

        rows: list[dict[str, Any]] = []
        live_ids: set[int] = set()

        for holding in holdings:
            live_ids.add(holding.id)
            holding.stock.refresh()
            price = holding.stock.price or Decimal("0")
            avg = holding.average_price or Decimal("0")
            proceeds, cost, pnl, pnl_pct = _position_pnl(
                holding.shares, price, avg
            )

            state = trails.setdefault(holding.id, TrailState())
            status, high_pct, exit_px = _update_trail(
                state,
                price=price,
                avg=avg,
                pnl_pct=pnl_pct,
                arm_pct=arm_pct,
                giveback_pp=giveback_pp,
            )

            if gain_only and (pnl_pct is None or pnl_pct <= 0):
                continue

            rows.append(
                {
                    "symbol": holding.stock.symbol,
                    "path": _path_label(holding),
                    "shares": holding.shares,
                    "tranches": holding.tranches or 0,
                    "avg": avg,
                    "last": price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "high_pct": high_pct,
                    "status": status,
                    "exit_px": exit_px,
                    "value": proceeds,
                    "age": _age_label(holding),
                }
            )

        # Drop trail state for holdings no longer open.
        for hid in list(trails.keys()):
            if hid not in live_ids:
                del trails[hid]

        if gain_only:
            rows.sort(key=lambda r: float(r["pnl_pct"] or 0), reverse=True)
        else:
            # SELL first, then ARMED, then by P&L
            order = {"SELL": 0, "ARMED": 1, "UP": 2, "DOWN": 3, "—": 4}

            def sort_key(r):
                return (order.get(r["status"], 9), float(r["pnl_pct"] or 0))

            rows.sort(key=sort_key)

        if not rows:
            self.stdout.write(
                self.style.WARNING("No rows to show (nothing green under --gain-only).")
            )
            return

        header = (
            f"{'SYM':<6} {'PATH':<12} {'Sh':>4} {'Tr':>3} "
            f"{'Avg':>9} {'Last':>9} {'P&L%':>8} {'HIGH':>8} {'Status':<6} "
            f"{'Value':>10} {'Age':>5}"
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

            high = row["high_pct"]
            high_txt = "—" if high is None else f"{high:+.2f}%"

            line = (
                f"{row['symbol']:<6} {row['path']:<12} {row['shares']:>4} {row['tranches']:>3} "
                f"{row['avg']:>9.2f} {row['last']:>9.2f} "
            )
            self.stdout.write(line, ending="")
            self.stdout.write(pct_styled, ending="")
            self.stdout.write(f" {high_txt:>8} ", ending="")
            self.stdout.write(self._style_status(row["status"]), ending="")
            self.stdout.write(f" {row['value']:>10,.2f} {row['age']:>5}")

            total_value += row["value"]
            if pct is not None and pct != -100:
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

        n_sell = sum(1 for r in rows if r["status"] == "SELL")
        n_armed = sum(1 for r in rows if r["status"] == "ARMED")
        self.stdout.write(
            f"shown n={len(rows)}  ARMED={n_armed}  SELL={n_sell}  "
            f"value ${total_value:,.2f}  P&L ${book_pnl:+,.2f} (",
            ending="",
        )
        self.stdout.write(book_pct_styled, ending="")
        self.stdout.write(")")
        if n_sell:
            self.stdout.write(
                self.style.ERROR(
                    "  → SELL: force_sell <SYM> --funds "
                    f"{fund.name}  (manual)"
                )
            )
