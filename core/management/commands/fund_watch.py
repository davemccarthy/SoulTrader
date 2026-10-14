"""
Live fund holdings board: refresh quotes, trail IPC, optional auto-sell.

Usage:
    python manage.py fund_watch PLSE-S
    python manage.py fund_watch PLSE-S --loop 6
    python manage.py fund_watch PLSE-S --loop 6 --arm 0.8 --giveback 0.2
    python manage.py fund_watch PLSE-S --cron --auto-sell --sell-funds PLSE-X PLSE-S
    python manage.py fund_watch PLSE-S --cron --auto-sell --diminish --sell-funds PLSE-X PLSE-S

Trail: arm at +arm% vs avg; track HWM after arm; SELL when price falls below
(peak P&L% − giveback) pp of entry. Status returns to ARMED if price recovers.

--cron: log to logs/soultrader.log (no TTY redraw); loop until flat book or RTH close.
--auto-sell: execute_sell on SELL signals (relayed to --sell-funds, else watch fund).
--diminish: unarmed ladder 0.8/0.2 → 0.6/0.1 → 0.4/0.0 by holding age (base from --arm/--giveback).
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

import pytz
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.models import Holding, Profile, SmartAnalysis, Stock
from core.services.execution import execute_sell
from core.services.market import market_open, rth_session_open

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")

# Unarmed diminish rungs: (min_age_minutes, arm_pct, giveback_pp)
# First rung uses CLI --arm/--giveback; later rungs are fixed.
DIMINISH_AFTER_M = (60, 120)
DIMINISH_MID = (Decimal("0.6"), Decimal("0.1"))
DIMINISH_LATE = (Decimal("0.4"), Decimal("0.0"))

# Dynamic sleep (seconds)
SLEEP_SELL = 3.0
SLEEP_ARMED = 6.0
SLEEP_NEAR_ARM = 10.0
SLEEP_CRON_DEFAULT = 30.0
NEAR_ARM_PP = Decimal("0.2")


@dataclass
class TrailState:
    """Per-holding trail state for the watch session (in-memory)."""

    last_price: Optional[Decimal] = None
    armed: bool = False
    hwm: Optional[Decimal] = None  # highest price since arm
    # Freeze IPC at arm so diminish does not cut a live trail.
    frozen_arm: Optional[Decimal] = None
    frozen_giveback: Optional[Decimal] = None


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


def _holding_age_minutes(holding: Holding) -> int:
    created = holding.created
    if not created:
        return 0
    if timezone.is_naive(created):
        created = timezone.make_aware(created, timezone.get_current_timezone())
    return max(0, int((timezone.now() - created).total_seconds() // 60))


def _effective_ipc(
    state: TrailState,
    *,
    age_min: int,
    base_arm: Decimal,
    base_giveback: Decimal,
    diminish: bool,
) -> tuple[Decimal, Decimal]:
    """Arm/giveback for this tick. Armed positions keep frozen IPC."""
    if state.armed and state.frozen_arm is not None and state.frozen_giveback is not None:
        return state.frozen_arm, state.frozen_giveback
    if not diminish:
        return base_arm, base_giveback
    if age_min >= DIMINISH_AFTER_M[1]:
        return DIMINISH_LATE
    if age_min >= DIMINISH_AFTER_M[0]:
        return DIMINISH_MID
    return base_arm, base_giveback


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
    mins = _holding_age_minutes(holding)
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
    hwm_raised = False

    if pnl_pct is not None and avg and avg > 0 and price > 0:
        if pnl_pct >= arm_pct:
            if not state.armed:
                state.armed = True
                state.hwm = price
                state.frozen_arm = arm_pct
                state.frozen_giveback = giveback_pp
                hwm_raised = True
            elif state.hwm is None or price > state.hwm:
                state.hwm = price
                hwm_raised = True

        if state.armed and state.hwm is not None:
            high_pct = _pnl_pct_at(state.hwm, avg)
            if high_pct is not None:
                exit_pct = high_pct - giveback_pp
                exit_px = avg * (Decimal("1") + exit_pct / Decimal("100"))
                # giveback 0: strict < HWM so we don't SELL on the arming tick.
                hit = (
                    (price < exit_px)
                    if giveback_pp == 0
                    else (price <= exit_px)
                )
                if hit and not hwm_raised:
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


def _dynamic_sleep_s(
    *,
    base_s: float,
    n_sell: int,
    n_armed: int,
    near_arm: bool,
) -> float:
    if n_sell > 0:
        return min(base_s, SLEEP_SELL)
    if n_armed > 0:
        return min(base_s, SLEEP_ARMED)
    if near_arm:
        return min(base_s, SLEEP_NEAR_ARM)
    return base_s


class Command(BaseCommand):
    help = (
        "Watch a fund's open holdings with live quote refresh and IPC trail. "
        "Optional --auto-sell / --cron / --diminish. Ctrl+C stops --loop."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "fund",
            type=str,
            help="Profile.name to watch (exact match), e.g. PLSE-S",
        )
        parser.add_argument(
            "--loop",
            type=float,
            default=None,
            metavar="SECONDS",
            help="Poll every SECONDS (quiet baseline; dynamic sleep may go faster). "
            "With --cron, defaults to 30 if omitted.",
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
            help="Arm trail at this unrealised P&L%% vs avg (default: 0.8). "
            "Also the first diminish rung when --diminish.",
        )
        parser.add_argument(
            "--giveback",
            type=float,
            default=0.2,
            metavar="PP",
            help="Exit when P&L%% falls giveback percentage-points below HIGH (default: 0.2).",
        )
        parser.add_argument(
            "--diminish",
            action="store_true",
            help="Unarmed only: after 60m use 0.6/0.1, after 120m use 0.4/0.0 "
            "(starting rung from --arm/--giveback). Armed trails freeze IPC.",
        )
        parser.add_argument(
            "--auto-sell",
            action="store_true",
            help="On SELL status, execute_sell (atomic cash). Uses --sell-funds.",
        )
        parser.add_argument(
            "--sell-funds",
            nargs="+",
            type=str,
            metavar="FUND",
            help="Funds to sell into on auto-sell (Profile.name). Default: watch fund only.",
        )
        parser.add_argument(
            "--cron",
            action="store_true",
            help="Headless: log to soultrader.log (no TTY board); loop until flat or RTH close.",
        )

    def handle(self, *args, **options):
        fund_name = (options["fund"] or "").strip()
        fund = Profile.objects.filter(name=fund_name).first()
        if not fund:
            raise CommandError(f'Fund "{fund_name}" not found')

        cron = bool(options.get("cron"))
        loop_s: Optional[float] = options.get("loop")
        if cron and loop_s is None:
            loop_s = SLEEP_CRON_DEFAULT
        if loop_s is not None and loop_s <= 0:
            raise CommandError("--loop must be a positive number of seconds")

        arm_pct = Decimal(str(options["arm"]))
        giveback_pp = Decimal(str(options["giveback"]))
        if arm_pct < 0:
            raise CommandError("--arm must be >= 0")
        if giveback_pp < 0:
            raise CommandError("--giveback must be >= 0")

        diminish = bool(options.get("diminish"))
        auto_sell = bool(options.get("auto_sell"))
        gain_only = bool(options.get("gain_only"))

        sell_fund_names = options.get("sell_funds") or [fund.name]
        sell_funds: list[Profile] = []
        for raw in sell_fund_names:
            name = (raw or "").strip()
            sf = Profile.objects.filter(name=name).first()
            if not sf:
                raise CommandError(f'Sell fund "{name}" not found')
            sell_funds.append(sf)
        sell_fund_ids = {p.id for p in sell_funds}

        trails: dict[int, TrailState] = {}
        sa: Optional[SmartAnalysis] = None

        self._emit(
            cron,
            (
                f"fund_watch start watch={fund.name} sell_funds="
                f"{','.join(p.name for p in sell_funds)} "
                f"arm=+{arm_pct}% giveback={giveback_pp}pp "
                f"diminish={diminish} auto_sell={auto_sell} cron={cron} "
                f"loop_base={loop_s}"
            ),
        )

        try:
            while True:
                if cron:
                    mo = market_open()
                    if not rth_session_open():
                        # Premarket (mo < 0): wait for the open. Else RTH over / holiday.
                        if mo is not None and mo < 0:
                            wait_s = min(float(loop_s or SLEEP_CRON_DEFAULT), 60.0)
                            self._emit(
                                cron,
                                f"fund_watch premarket ({mo}m to open); sleep {wait_s:.0f}s",
                            )
                            time.sleep(wait_s)
                            continue
                        self._emit(cron, "fund_watch stop: RTH closed", level="info")
                        break

                result = self._tick(
                    fund,
                    gain_only=gain_only,
                    looping=loop_s is not None,
                    cron=cron,
                    arm_pct=arm_pct,
                    giveback_pp=giveback_pp,
                    diminish=diminish,
                    auto_sell=auto_sell,
                    sell_funds=sell_funds,
                    sell_fund_ids=sell_fund_ids,
                    trails=trails,
                    sa=sa,
                )
                sa = result["sa"]

                if result["n_holdings"] == 0:
                    self._emit(cron, "fund_watch stop: book flat", level="info")
                    break

                if loop_s is None:
                    break

                sleep_s = _dynamic_sleep_s(
                    base_s=float(loop_s),
                    n_sell=result["n_sell_remaining"],
                    n_armed=result["n_armed"],
                    near_arm=result["near_arm"],
                )
                if cron:
                    logger.debug("fund_watch sleep=%.1fs", sleep_s)
                time.sleep(sleep_s)
        except KeyboardInterrupt:
            self._emit(cron, "fund_watch stopped (Ctrl+C)", level="warning")
            if not cron:
                self.stdout.write("")
                self.stdout.write(self.style.WARNING("Stopped (Ctrl+C)."))

    def _emit(self, cron: bool, msg: str, *, level: str = "info") -> None:
        if cron:
            getattr(logger, level, logger.info)(msg)
        else:
            if level == "warning":
                self.stdout.write(self.style.WARNING(msg))
            elif level == "error":
                self.stdout.write(self.style.ERROR(msg))
            else:
                self.stdout.write(msg)

    def _ensure_sa(self, sa: Optional[SmartAnalysis]) -> SmartAnalysis:
        if sa is not None:
            return sa
        return SmartAnalysis.objects.create(
            username="fund_watch",
            started=timezone.now(),
        )

    def _auto_sell_symbol(
        self,
        *,
        symbol: str,
        sell_funds: list[Profile],
        sell_fund_ids: set[int],
        sa: SmartAnalysis,
        cron: bool,
        explanation: str,
    ) -> int:
        """Sell symbol across sell_funds. Returns number of holdings sold."""
        try:
            stock = Stock.objects.get(symbol=symbol)
        except Stock.DoesNotExist:
            self._emit(cron, f"fund_watch auto-sell {symbol}: stock not in DB", level="warning")
            return 0

        holdings = list(
            Holding.objects.filter(
                stock=stock,
                shares__gt=0,
                fund_id__in=sell_fund_ids,
            ).select_related("stock", "fund")
        )
        sold = 0
        for holding in holdings:
            fund = holding.fund
            if fund is None:
                continue
            holding.stock.refresh()
            proceeds, cost, pnl, pnl_pct = _position_pnl(
                holding.shares, holding.stock.price, holding.average_price
            )
            pct_txt = "n/a" if pnl_pct is None else f"{pnl_pct:+.2f}%"
            msg = (
                f"fund_watch AUTO-SELL {symbol} fund={fund.name} "
                f"shares={holding.shares} @ ${holding.stock.price} "
                f"value=${proceeds:,.2f} P&L=${pnl:+,.2f} ({pct_txt}) "
                f"— {explanation}"
            )
            self._emit(cron, msg, level="info")
            if not cron:
                self.stdout.write(self.style.ERROR(msg))
            execute_sell(
                sa=sa,
                fund=fund,
                holding=holding,
                explanation=f"{explanation} ({symbol})",
            )
            sold += 1
        if sold == 0:
            self._emit(
                cron,
                f"fund_watch auto-sell {symbol}: no holdings in sell_funds",
                level="warning",
            )
        return sold

    def _clear_screen(self) -> None:
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()

    def _style_status(self, status: str) -> str:
        label = f"{status:<5}"
        if status == "SELL":
            return self.style.ERROR(label)
        if status == "ARMED":
            return self.style.WARNING(label)
        return label

    def _tick(
        self,
        fund: Profile,
        *,
        gain_only: bool,
        looping: bool,
        cron: bool,
        arm_pct: Decimal,
        giveback_pp: Decimal,
        diminish: bool,
        auto_sell: bool,
        sell_funds: list[Profile],
        sell_fund_ids: set[int],
        trails: dict[int, TrailState],
        sa: Optional[SmartAnalysis],
    ) -> dict[str, Any]:
        if looping and not cron:
            self._clear_screen()

        holdings = list(
            Holding.objects.filter(fund=fund, shares__gt=0)
            .select_related("stock", "stock__advisor", "discovery", "discovery__advisor")
            .order_by("stock__symbol", "id")
        )

        now_et = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S %Z")
        mode = "gain-only" if gain_only else "all"
        loop_note = " | Ctrl+C stop" if looping and not cron else ""
        header_line = (
            f"{fund.name}  {now_et}  ({mode}{loop_note})  "
            f"holdings={len(holdings)}  arm=+{arm_pct}%  giveback={giveback_pp}pp"
            f"{'  diminish' if diminish else ''}"
            f"{'  auto-sell' if auto_sell else ''}"
        )
        if not cron:
            self.stdout.write(header_line)
        else:
            logger.info("fund_watch tick %s", header_line)

        if not holdings:
            if not cron:
                self.stdout.write(self.style.WARNING("No open holdings."))
            return {
                "sa": sa,
                "n_holdings": 0,
                "n_sell_remaining": 0,
                "n_armed": 0,
                "near_arm": False,
            }

        rows: list[dict[str, Any]] = []
        live_ids: set[int] = set()
        sell_symbols: list[str] = []
        near_arm = False

        for holding in holdings:
            live_ids.add(holding.id)
            holding.stock.refresh()
            price = holding.stock.price or Decimal("0")
            avg = holding.average_price or Decimal("0")
            proceeds, cost, pnl, pnl_pct = _position_pnl(holding.shares, price, avg)

            state = trails.setdefault(holding.id, TrailState())
            age_min = _holding_age_minutes(holding)
            eff_arm, eff_gb = _effective_ipc(
                state,
                age_min=age_min,
                base_arm=arm_pct,
                base_giveback=giveback_pp,
                diminish=diminish,
            )
            status, high_pct, exit_px = _update_trail(
                state,
                price=price,
                avg=avg,
                pnl_pct=pnl_pct,
                arm_pct=eff_arm,
                giveback_pp=eff_gb,
            )

            if (
                not state.armed
                and pnl_pct is not None
                and pnl_pct >= eff_arm - NEAR_ARM_PP
                and pnl_pct < eff_arm
            ):
                near_arm = True

            if status == "SELL":
                sell_symbols.append(holding.stock.symbol)

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
                    "eff_arm": eff_arm,
                    "eff_gb": eff_gb,
                }
            )

        for hid in list(trails.keys()):
            if hid not in live_ids:
                del trails[hid]

        # Auto-sell before board summary so cron logs the action first.
        sold_syms: set[str] = set()
        if auto_sell and sell_symbols:
            sa = self._ensure_sa(sa)
            for symbol in sorted(set(sell_symbols)):
                n = self._auto_sell_symbol(
                    symbol=symbol,
                    sell_funds=sell_funds,
                    sell_fund_ids=sell_fund_ids,
                    sa=sa,
                    cron=cron,
                    explanation="fund_watch trail giveback",
                )
                if n:
                    sold_syms.add(symbol)
            # Drop trail state for sold watch-fund holdings
            for hid in list(trails.keys()):
                h = next((x for x in holdings if x.id == hid), None)
                if h and h.stock.symbol in sold_syms:
                    del trails[hid]
            # Refresh open count after sells
            n_holdings = Holding.objects.filter(fund=fund, shares__gt=0).count()
        else:
            n_holdings = len(holdings)

        if gain_only:
            rows.sort(key=lambda r: float(r["pnl_pct"] or 0), reverse=True)
        else:
            order = {"SELL": 0, "ARMED": 1, "UP": 2, "DOWN": 3, "—": 4}

            def sort_key(r):
                return (order.get(r["status"], 9), float(r["pnl_pct"] or 0))

            rows.sort(key=sort_key)

        # Filter sold symbols from display rows
        if sold_syms:
            rows = [r for r in rows if r["symbol"] not in sold_syms]

        n_sell = sum(1 for r in rows if r["status"] == "SELL")
        n_armed = sum(1 for r in rows if r["status"] == "ARMED")

        if cron:
            for r in rows:
                if r["status"] in ("SELL", "ARMED"):
                    pct = r["pnl_pct"]
                    pct_txt = "n/a" if pct is None else f"{pct:+.2f}%"
                    high = r["high_pct"]
                    high_txt = "—" if high is None else f"{high:+.2f}%"
                    logger.info(
                        "fund_watch %s %s pnl=%s high=%s arm=+%s gb=%spp age=%s",
                        r["status"],
                        r["symbol"],
                        pct_txt,
                        high_txt,
                        r["eff_arm"],
                        r["eff_gb"],
                        r["age"],
                    )
            logger.info(
                "fund_watch summary holdings=%s shown=%s ARMED=%s SELL=%s sold=%s",
                n_holdings,
                len(rows),
                n_armed,
                n_sell,
                ",".join(sorted(sold_syms)) or "—",
            )
            return {
                "sa": sa,
                "n_holdings": n_holdings,
                "n_sell_remaining": n_sell,
                "n_armed": n_armed,
                "near_arm": near_arm,
            }

        # Interactive board
        if not rows:
            self.stdout.write(
                self.style.WARNING("No rows to show (nothing green under --gain-only).")
            )
            return {
                "sa": sa,
                "n_holdings": n_holdings,
                "n_sell_remaining": n_sell,
                "n_armed": n_armed,
                "near_arm": near_arm,
            }

        board_header = (
            f"{'SYM':<6} {'PATH':<12} {'Sh':>4} {'Tr':>3} "
            f"{'Avg':>9} {'Last':>9} {'P&L%':>8} {'HIGH':>8} {'Status':<6} "
            f"{'Value':>10} {'Age':>5}"
        )
        self.stdout.write(board_header)
        self.stdout.write("-" * len(board_header))

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

        self.stdout.write("-" * len(board_header))
        book_pnl = total_value - total_cost
        book_pct = (book_pnl / total_cost * Decimal("100")) if total_cost else None
        book_pct_txt = "n/a" if book_pct is None else f"{book_pct:+.2f}%"
        if book_pct is not None and book_pct > 0:
            book_pct_styled = self.style.SUCCESS(book_pct_txt)
        elif book_pct is not None and book_pct < 0:
            book_pct_styled = self.style.ERROR(book_pct_txt)
        else:
            book_pct_styled = book_pct_txt

        self.stdout.write(
            f"shown n={len(rows)}  ARMED={n_armed}  SELL={n_sell}  "
            f"value ${total_value:,.2f}  P&L ${book_pnl:+,.2f} (",
            ending="",
        )
        self.stdout.write(book_pct_styled, ending="")
        self.stdout.write(")")
        if n_sell and not auto_sell:
            sell_names = " ".join(p.name for p in sell_funds)
            self.stdout.write(
                self.style.ERROR(
                    f"  → SELL: force_sell <SYM> --funds {sell_names}  (manual)"
                )
            )
        elif sold_syms:
            self.stdout.write(
                self.style.SUCCESS(f"  → auto-sold: {', '.join(sorted(sold_syms))}")
            )

        return {
            "sa": sa,
            "n_holdings": n_holdings,
            "n_sell_remaining": n_sell,
            "n_armed": n_armed,
            "near_arm": near_arm,
        }
