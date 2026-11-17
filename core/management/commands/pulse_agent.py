"""
Pulse agent: grade OPEN_VOL discoveries, buy on open→dip, IPC trail sells.

Pulse writes discoveries (SA discover); this agent alone buys into the trial fund
and owns the live trail (fund_watch can stay off for that book).

Usage:
    python manage.py pulse_agent PULSE --cron
    python manage.py pulse_agent PULSE --loop 30 --min-grade C --dip 0.8 --max-holds 10
    python manage.py pulse_agent PULSE --dry-run --once
    python manage.py pulse_agent --self-test
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, time as dt_time
from decimal import Decimal
from typing import Optional

import pytz
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Exists, OuterRef
from django.utils import timezone

from core.management.commands.fund_watch import (
    SLEEP_CRON_DEFAULT,
    TrailState,
    _dynamic_sleep_s,
    _effective_ipc,
    _holding_age_minutes,
    _position_pnl,
    _update_trail,
)
from core.models import Discovery, Holding, Profile, SmartAnalysis, Stock, Trade
from core.services.execution import execute_buy, execute_sell
from core.services.health.so_ratings import (
    opportunity_grade_at_least,
    score_to_opportunity_grade,
)
from core.services.market import market_open, rth_session_open

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")

# Defaults aligned with Pulse open×vol trial + fund_watch Meyka-style IPC.
DEFAULT_MIN_GRADE = "C"
DEFAULT_DIP_PCT = 0.8
DEFAULT_MAX_HOLDS = 10
DEFAULT_ARM = 0.8
DEFAULT_GIVEBACK = 0.2
# New buys stop mid-afternoon; trail/sells continue to RTH close.
BUY_END_TIME_ET = dt_time(15, 0)
BUY_START_TIME_ET = dt_time(9, 45)

_OPEN_PX_RE = re.compile(r"Open\s+\$([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def parse_open_px_from_explanation(explanation: str | None) -> Optional[Decimal]:
    if not explanation:
        return None
    match = _OPEN_PX_RE.search(str(explanation))
    if not match:
        return None
    try:
        value = Decimal(match.group(1))
    except Exception:
        return None
    return value if value > 0 else None


def dip_giveback_pct(open_px: Decimal, last: Decimal) -> Optional[Decimal]:
    if not open_px or open_px <= 0 or last is None or last <= 0:
        return None
    return (open_px - last) / open_px * Decimal("100")


def dip_ready(open_px: Decimal, last: Decimal, *, dip_pct: Decimal) -> bool:
    giveback = dip_giveback_pct(open_px, last)
    return giveback is not None and giveback >= dip_pct


def discovery_opportunity_letter(discovery: Discovery) -> Optional[str]:
    assessment = discovery.assessment
    if assessment is None or assessment.opportunity is None:
        return None
    grade = score_to_opportunity_grade(float(assessment.opportunity))
    return grade.letter if grade else None


def grade_passes(discovery: Discovery, *, min_grade: str) -> tuple[bool, str]:
    letter = discovery_opportunity_letter(discovery)
    if letter is None:
        return False, "no opportunity score"
    if not opportunity_grade_at_least(letter, min_grade):
        return False, f"opp={letter} < min={min_grade}"
    return True, f"opp={letter}"


def session_start_et() -> datetime:
    now_et = datetime.now(ET)
    start = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
    return start


def pending_open_vol_discoveries(fund: Profile, *, since: datetime):
    """Today's Pulse OPEN_VOL discoveries not yet bought into this fund."""
    bought = Trade.objects.filter(
        discovery_id=OuterRef("pk"),
        action="BUY",
        fund_id=fund.id,
    )
    held = Holding.objects.filter(
        stock_id=OuterRef("stock_id"),
        fund_id=fund.id,
        shares__gt=0,
    )
    return (
        Discovery.objects.filter(
            advisor__python_class="Pulse",
            explanation__startswith="OPEN_VOL",
            created__gte=since,
        )
        .annotate(has_buy=Exists(bought), has_hold=Exists(held))
        .filter(has_buy=False, has_hold=False)
        .select_related("stock", "assessment", "advisor")
        .order_by("id")
    )


def _self_test() -> None:
    assert parse_open_px_from_explanation(
        "OPEN_VOL: open gap +2.5% vs prior | Open $41.20 / prior $40.00 / live $40.50"
    ) == Decimal("41.20")
    assert parse_open_px_from_explanation("no open here") is None
    assert dip_ready(Decimal("100"), Decimal("99.1"), dip_pct=Decimal("0.8"))
    assert not dip_ready(Decimal("100"), Decimal("99.5"), dip_pct=Decimal("0.8"))
    assert dip_giveback_pct(Decimal("100"), Decimal("99")) == Decimal("1")
    print("pulse_agent self-test ok")


class Command(BaseCommand):
    help = (
        "Pulse agent: grade OPEN_VOL discoveries, buy on dip, IPC trail sells "
        "for one fund (Pulse mega trial)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "fund",
            nargs="?",
            type=str,
            default=None,
            help="Profile.name to trade (exact), e.g. PULSE",
        )
        parser.add_argument(
            "--loop",
            type=float,
            default=None,
            metavar="SECONDS",
            help="Poll interval. With --cron, defaults to 30 if omitted.",
        )
        parser.add_argument(
            "--cron",
            action="store_true",
            help="Headless: log to soultrader.log; loop until RTH close.",
        )
        parser.add_argument(
            "--once",
            action="store_true",
            help="Single tick then exit (implies no loop).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Log would-buy / would-sell; no execute_buy / execute_sell.",
        )
        parser.add_argument(
            "--min-grade",
            type=str,
            default=DEFAULT_MIN_GRADE,
            help=f"Minimum opportunity letter (default {DEFAULT_MIN_GRADE}).",
        )
        parser.add_argument(
            "--dip",
            type=float,
            default=DEFAULT_DIP_PCT,
            metavar="PCT",
            help=f"Min open→last giveback %% before buy (default {DEFAULT_DIP_PCT}).",
        )
        parser.add_argument(
            "--max-holds",
            type=int,
            default=DEFAULT_MAX_HOLDS,
            help=f"Max open holdings in the fund (default {DEFAULT_MAX_HOLDS}).",
        )
        parser.add_argument(
            "--arm",
            type=float,
            default=DEFAULT_ARM,
            help=f"IPC arm P&L%% (default {DEFAULT_ARM}).",
        )
        parser.add_argument(
            "--giveback",
            type=float,
            default=DEFAULT_GIVEBACK,
            help=f"IPC giveback pp below HIGH (default {DEFAULT_GIVEBACK}).",
        )
        parser.add_argument(
            "--diminish",
            action="store_true",
            help="Unarmed diminish ladder (same as fund_watch).",
        )
        parser.add_argument(
            "--no-buy",
            action="store_true",
            help="Trail/sell only; do not buy new discoveries.",
        )
        parser.add_argument(
            "--self-test",
            action="store_true",
            help="Run pure helper checks and exit.",
        )

    def handle(self, *args, **options):
        if options.get("self_test"):
            _self_test()
            return

        fund_name = (options.get("fund") or "").strip()
        if not fund_name:
            raise CommandError("Fund name required (e.g. PULSE), or use --self-test")

        fund = Profile.objects.filter(name=fund_name).first()
        if not fund:
            raise CommandError(f'Fund "{fund_name}" not found')

        cron = bool(options.get("cron"))
        once = bool(options.get("once"))
        dry_run = bool(options.get("dry_run"))
        no_buy = bool(options.get("no_buy"))
        diminish = bool(options.get("diminish"))

        loop_s: Optional[float] = options.get("loop")
        if once:
            loop_s = None
        elif cron and loop_s is None:
            loop_s = SLEEP_CRON_DEFAULT
        if loop_s is not None and loop_s <= 0:
            raise CommandError("--loop must be a positive number of seconds")

        min_grade = str(options["min_grade"] or DEFAULT_MIN_GRADE).strip().upper()
        if min_grade not in {"A", "B", "C", "D", "E", "F"}:
            raise CommandError("--min-grade must be A–F")

        dip_pct = Decimal(str(options["dip"]))
        max_holds = int(options["max_holds"])
        arm_pct = Decimal(str(options["arm"]))
        giveback_pp = Decimal(str(options["giveback"]))
        if dip_pct < 0 or arm_pct < 0 or giveback_pp < 0 or max_holds < 1:
            raise CommandError("Invalid --dip / --arm / --giveback / --max-holds")

        trails: dict[int, TrailState] = {}
        grade_rejects: set[int] = set()
        sa: Optional[SmartAnalysis] = None

        self._emit(
            cron,
            (
                f"pulse_agent start fund={fund.name} min_grade={min_grade} "
                f"dip>={dip_pct}% max_holds={max_holds} arm=+{arm_pct}% "
                f"giveback={giveback_pp}pp diminish={diminish} "
                f"dry_run={dry_run} no_buy={no_buy} cron={cron} loop_base={loop_s}"
            ),
        )

        try:
            while True:
                if cron:
                    mo = market_open()
                    if not rth_session_open():
                        if mo is not None and mo < 0:
                            wait_s = min(float(loop_s or SLEEP_CRON_DEFAULT), 60.0)
                            self._emit(
                                cron,
                                f"pulse_agent premarket ({mo}m to open); sleep {wait_s:.0f}s",
                            )
                            time.sleep(wait_s)
                            continue
                        self._emit(cron, "pulse_agent stop: RTH closed", level="info")
                        break

                sa = self._tick(
                    fund,
                    cron=cron,
                    dry_run=dry_run,
                    no_buy=no_buy,
                    min_grade=min_grade,
                    dip_pct=dip_pct,
                    max_holds=max_holds,
                    arm_pct=arm_pct,
                    giveback_pp=giveback_pp,
                    diminish=diminish,
                    trails=trails,
                    grade_rejects=grade_rejects,
                    sa=sa,
                )

                if loop_s is None:
                    break

                n_holdings = Holding.objects.filter(fund=fund, shares__gt=0).count()
                n_armed = sum(1 for st in trails.values() if st.armed)
                sleep_s = _dynamic_sleep_s(
                    base_s=float(loop_s),
                    n_sell=0,
                    n_armed=n_armed,
                    near_arm=False,
                )
                if cron:
                    logger.debug(
                        "pulse_agent sleep=%.1fs holdings=%s", sleep_s, n_holdings
                    )
                time.sleep(sleep_s)
        except KeyboardInterrupt:
            self._emit(cron, "pulse_agent stopped (Ctrl+C)", level="warning")

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
            username="pulse_agent",
            started=timezone.now(),
        )

    def _in_buy_window(self) -> bool:
        now_et = datetime.now(ET).time()
        return BUY_START_TIME_ET <= now_et < BUY_END_TIME_ET

    def _tick(
        self,
        fund: Profile,
        *,
        cron: bool,
        dry_run: bool,
        no_buy: bool,
        min_grade: str,
        dip_pct: Decimal,
        max_holds: int,
        arm_pct: Decimal,
        giveback_pp: Decimal,
        diminish: bool,
        trails: dict[int, TrailState],
        grade_rejects: set[int],
        sa: Optional[SmartAnalysis],
    ) -> Optional[SmartAnalysis]:
        now_et = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S %Z")
        holdings = list(
            Holding.objects.filter(fund=fund, shares__gt=0)
            .select_related("stock", "discovery")
            .order_by("stock__symbol", "id")
        )
        self._emit(
            cron,
            f"pulse_agent tick {fund.name} {now_et} holdings={len(holdings)}",
        )

        sa = self._trail_tick(
            fund,
            holdings=holdings,
            cron=cron,
            dry_run=dry_run,
            arm_pct=arm_pct,
            giveback_pp=giveback_pp,
            diminish=diminish,
            trails=trails,
            sa=sa,
        )

        if no_buy:
            return sa
        if not self._in_buy_window():
            self._emit(cron, "pulse_agent buy window closed (09:45–15:00 ET)")
            return sa

        open_count = Holding.objects.filter(fund=fund, shares__gt=0).count()
        slots = max_holds - open_count
        if slots <= 0:
            self._emit(
                cron,
                f"pulse_agent at max_holds={max_holds}; skip buys",
            )
            return sa

        if fund.sentiment_enabled() and fund.at_or_over_equity_buy_cap():
            self._emit(
                cron,
                f"pulse_agent over equity buy cap; skip buys",
                level="warning",
            )
            return sa

        since = session_start_et()
        # Discovery.created is typically UTC-aware; convert ET midnight.
        since_utc = since.astimezone(timezone.utc) if since.tzinfo else since
        pending = list(pending_open_vol_discoveries(fund, since=since_utc))
        self._emit(
            cron,
            f"pulse_agent pending OPEN_VOL={len(pending)} slots={slots}",
        )

        bought = 0
        for discovery in pending:
            if bought >= slots:
                break
            if discovery.id in grade_rejects:
                continue
            ok, detail = grade_passes(discovery, min_grade=min_grade)
            if not ok:
                grade_rejects.add(discovery.id)
                self._emit(
                    cron,
                    f"pulse_agent skip {discovery.stock.symbol}: grade ({detail})",
                )
                continue

            open_px = parse_open_px_from_explanation(discovery.explanation)
            if open_px is None:
                self._emit(
                    cron,
                    f"pulse_agent skip {discovery.stock.symbol}: no Open $ in explanation",
                    level="warning",
                )
                grade_rejects.add(discovery.id)
                continue

            stock = discovery.stock
            stock.refresh()
            last = stock.price or Decimal("0")
            if last <= 0:
                self._emit(
                    cron,
                    f"pulse_agent skip {stock.symbol}: no live price",
                    level="warning",
                )
                continue

            giveback = dip_giveback_pct(open_px, last)
            if not dip_ready(open_px, last, dip_pct=dip_pct):
                gb_txt = "n/a" if giveback is None else f"{giveback:.2f}%"
                self._emit(
                    cron,
                    f"pulse_agent wait dip {stock.symbol}: "
                    f"open=${open_px} last=${last} giveback={gb_txt} need>={dip_pct}% "
                    f"({detail})",
                )
                continue

            allowance = fund.average_spend()
            lead = (discovery.explanation or "").split("|", 1)[0].strip()
            explanation = f"pulse_agent dip-buy {lead}"
            gb_txt = "n/a" if giveback is None else f"{giveback:.2f}%"
            msg = (
                f"pulse_agent BUY {stock.symbol} fund={fund.name} "
                f"open=${open_px} last=${last} giveback={gb_txt} {detail} "
                f"allowance=${allowance}"
            )
            if dry_run:
                self._emit(cron, f"DRY-RUN {msg}")
                bought += 1  # reserve slot in dry-run so we don't spam all names
                continue

            sa = self._ensure_sa(sa)
            self._emit(cron, msg)
            execute_buy(
                sa,
                fund,
                stock,
                allowance,
                explanation,
                discovery=discovery,
            )
            # Confirm fill created a holding before counting slot.
            if Holding.objects.filter(fund=fund, stock=stock, shares__gt=0).exists():
                bought += 1
            fund.refresh_from_db()

        return sa

    def _trail_tick(
        self,
        fund: Profile,
        *,
        holdings: list[Holding],
        cron: bool,
        dry_run: bool,
        arm_pct: Decimal,
        giveback_pp: Decimal,
        diminish: bool,
        trails: dict[int, TrailState],
        sa: Optional[SmartAnalysis],
    ) -> Optional[SmartAnalysis]:
        if not holdings:
            trails.clear()
            return sa

        live_ids: set[int] = set()
        sell_symbols: list[str] = []

        for holding in holdings:
            live_ids.add(holding.id)
            holding.stock.refresh()
            price = holding.stock.price or Decimal("0")
            avg = holding.average_price or Decimal("0")
            _proceeds, _cost, _pnl, pnl_pct = _position_pnl(holding.shares, price, avg)
            state = trails.setdefault(holding.id, TrailState())
            age_min = _holding_age_minutes(holding)
            eff_arm, eff_gb = _effective_ipc(
                state,
                age_min=age_min,
                base_arm=arm_pct,
                base_giveback=giveback_pp,
                diminish=diminish,
            )
            status, high_pct, _exit_px = _update_trail(
                state,
                price=price,
                avg=avg,
                pnl_pct=pnl_pct,
                arm_pct=eff_arm,
                giveback_pp=eff_gb,
            )
            if status in ("SELL", "ARMED"):
                pct_txt = "n/a" if pnl_pct is None else f"{pnl_pct:+.2f}%"
                high_txt = "—" if high_pct is None else f"{high_pct:+.2f}%"
                self._emit(
                    cron,
                    f"pulse_agent {status} {holding.stock.symbol} "
                    f"pnl={pct_txt} high={high_txt} arm=+{eff_arm} gb={eff_gb}pp",
                )
            if status == "SELL":
                sell_symbols.append(holding.stock.symbol)

        for hid in list(trails.keys()):
            if hid not in live_ids:
                del trails[hid]

        if not sell_symbols:
            return sa

        for symbol in sorted(set(sell_symbols)):
            try:
                stock = Stock.objects.get(symbol=symbol)
            except Stock.DoesNotExist:
                self._emit(
                    cron,
                    f"pulse_agent auto-sell {symbol}: stock not in DB",
                    level="warning",
                )
                continue
            holding = (
                Holding.objects.filter(fund=fund, stock=stock, shares__gt=0)
                .select_related("stock", "fund")
                .first()
            )
            if holding is None:
                continue
            holding.stock.refresh()
            proceeds, _cost, pnl, pnl_pct = _position_pnl(
                holding.shares, holding.stock.price, holding.average_price
            )
            pct_txt = "n/a" if pnl_pct is None else f"{pnl_pct:+.2f}%"
            msg = (
                f"pulse_agent AUTO-SELL {symbol} fund={fund.name} "
                f"shares={holding.shares} @ ${holding.stock.price} "
                f"value=${proceeds:,.2f} P&L=${pnl:+,.2f} ({pct_txt})"
            )
            if dry_run:
                self._emit(cron, f"DRY-RUN {msg}")
                continue
            sa = self._ensure_sa(sa)
            self._emit(cron, msg)
            execute_sell(
                sa=sa,
                fund=fund,
                holding=holding,
                explanation=f"pulse_agent trail giveback ({symbol})",
            )
            trails.pop(holding.id, None)

        return sa
