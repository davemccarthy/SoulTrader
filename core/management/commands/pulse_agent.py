"""
Pulse agent: grade OPEN_VOL discoveries, arm on dip, buy on short-term trend, IPC sells.

Pulse writes discoveries (SA discover); this agent alone buys into the trial fund
and owns the live trail (fund_watch can stay off for that book).

Entry:
  1. Grade ≥ min (default C)
  2. Open→last giveback ≥ --dip (default 0.5%) → BUY-arm (start 10s trend tracking)
  3. Buy when ups > downs over 6 votes (3-3 hold) — fill may be lower than the arm print

BUY-armed or SELL-armed (IPC) poll every 10s; quiet stays at --loop (default 30s).

Usage:
    python manage.py pulse_agent PULSE --cron
    python manage.py pulse_agent PULSE --loop 30 --min-grade C --dip 0.5 --max-holds 10
    python manage.py pulse_agent PULSE --dry-run --once
    python manage.py pulse_agent --self-test
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, time as dt_time, timezone as dt_timezone
from decimal import Decimal
from typing import Optional

import pytz
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Exists, OuterRef
from django.utils import timezone

from core.management.commands.fund_watch import (
    SLEEP_CRON_DEFAULT,
    TrailState,
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
from core.services.short_term_trend import (
    DEFAULT_TREND_POLL_S,
    DEFAULT_TREND_VOTES,
    PriceTrendWindow,
    dynamic_arm_sleep_s,
    evaluate_trend,
)

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")

# Defaults aligned with Pulse open×vol trial + fund_watch Meyka-style IPC.
DEFAULT_MIN_GRADE = "C"
# Open→last giveback that starts BUY-arm / trend tracking (not the fill trigger).
DEFAULT_DIP_PCT = 0.5
DEFAULT_MAX_HOLDS = 10
DEFAULT_ARM = 0.8
DEFAULT_GIVEBACK = 0.2
ARM_POLL_S = DEFAULT_TREND_POLL_S  # 10s for BUY-armed and SELL-armed
TREND_VOTES = DEFAULT_TREND_VOTES  # 6 votes (~7×10s); ups > downs; 3-3 hold
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
    assert dip_ready(Decimal("100"), Decimal("99.4"), dip_pct=Decimal("0.5"))
    assert not dip_ready(Decimal("100"), Decimal("99.6"), dip_pct=Decimal("0.5"))
    assert dip_giveback_pct(Decimal("100"), Decimal("99")) == Decimal("1")

    # 7 prices → 6 votes; 3-3 hold; 4-2 buy
    tie = evaluate_trend([10, 9, 10, 9, 10, 9, 10], votes=6)
    assert tie.ready and tie.ups == 3 and tie.downs == 3 and not tie.bullish
    up = evaluate_trend([10, 11, 10, 11, 10, 11, 12], votes=6)
    assert up.ready and up.ups == 4 and up.downs == 2 and up.bullish
    win = PriceTrendWindow(votes=6)
    for px in (10, 9, 10, 9, 10, 9, 10):
        vote = win.push(px)
    assert vote.ready and not vote.bullish
    assert dynamic_arm_sleep_s(base_s=30, buy_armed=True, sell_armed=False) == 10.0
    assert dynamic_arm_sleep_s(base_s=30, buy_armed=False, sell_armed=True) == 10.0
    assert dynamic_arm_sleep_s(base_s=30, buy_armed=False, sell_armed=False) == 30.0
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
            help=(
                f"Open→last giveback %% to BUY-arm and start trend tracking "
                f"(default {DEFAULT_DIP_PCT}). Fill waits for ups>downs after arm."
            ),
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
        trend_windows: dict[str, PriceTrendWindow] = {}
        dip_armed_syms: set[str] = set()
        sa: Optional[SmartAnalysis] = None

        self._emit(
            cron,
            (
                f"pulse_agent start fund={fund.name} min_grade={min_grade} "
                f"buy_arm_dip>={dip_pct}% max_holds={max_holds} ipc_arm=+{arm_pct}% "
                f"giveback={giveback_pp}pp diminish={diminish} "
                f"trend_votes={TREND_VOTES} arm_poll={ARM_POLL_S}s "
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

                sa, buy_armed, sell_armed = self._tick(
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
                    trend_windows=trend_windows,
                    dip_armed_syms=dip_armed_syms,
                    sa=sa,
                )

                if loop_s is None:
                    break

                sleep_s = dynamic_arm_sleep_s(
                    base_s=float(loop_s),
                    buy_armed=buy_armed,
                    sell_armed=sell_armed,
                    arm_poll_s=ARM_POLL_S,
                )
                if cron:
                    logger.debug(
                        "pulse_agent sleep=%.1fs buy_armed=%s sell_armed=%s",
                        sleep_s,
                        buy_armed,
                        sell_armed,
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
        trend_windows: dict[str, PriceTrendWindow],
        dip_armed_syms: set[str],
        sa: Optional[SmartAnalysis],
    ) -> tuple[Optional[SmartAnalysis], bool, bool]:
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

        sa, sell_armed = self._trail_tick(
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
            return sa, False, sell_armed
        if not self._in_buy_window():
            self._emit(cron, "pulse_agent buy window closed (09:45–15:00 ET)")
            trend_windows.clear()
            return sa, False, sell_armed

        open_count = Holding.objects.filter(fund=fund, shares__gt=0).count()
        slots = max_holds - open_count
        if slots <= 0:
            self._emit(
                cron,
                f"pulse_agent at max_holds={max_holds}; skip buys",
            )
            trend_windows.clear()
            return sa, False, sell_armed

        if fund.sentiment_enabled() and fund.at_or_over_equity_buy_cap():
            self._emit(
                cron,
                f"pulse_agent over equity buy cap; skip buys",
                level="warning",
            )
            return sa, False, sell_armed

        since = session_start_et()
        # Discovery.created is typically UTC-aware; convert ET midnight.
        since_utc = since.astimezone(dt_timezone.utc) if since.tzinfo else since
        pending = list(pending_open_vol_discoveries(fund, since=since_utc))
        self._emit(
            cron,
            f"pulse_agent pending OPEN_VOL={len(pending)} slots={slots}",
        )

        bought = 0
        buy_armed = False
        live_syms: set[str] = set()

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
            symbol = stock.symbol
            stock.refresh()
            last = stock.price or Decimal("0")
            if last <= 0:
                self._emit(
                    cron,
                    f"pulse_agent skip {symbol}: no live price",
                    level="warning",
                )
                buy_armed = True
                continue

            live_syms.add(symbol)
            giveback = dip_giveback_pct(open_px, last)
            gb_txt = "n/a" if giveback is None else f"{giveback:.2f}%"

            # Not yet −dip% off open → watch only (faster poll); no trend samples yet.
            if not dip_ready(open_px, last, dip_pct=dip_pct):
                buy_armed = True
                if symbol in dip_armed_syms:
                    # Recovered above arm level — reset so we re-arm cleanly later.
                    dip_armed_syms.discard(symbol)
                    trend_windows.pop(symbol, None)
                self._emit(
                    cron,
                    f"pulse_agent wait arm {symbol}: "
                    f"open=${open_px} last=${last} giveback={gb_txt} need>={dip_pct}% "
                    f"({detail})",
                )
                continue

            # First time at/under arm dip → BUY-arm and start a fresh trend window.
            if symbol not in dip_armed_syms:
                dip_armed_syms.add(symbol)
                trend_windows[symbol] = PriceTrendWindow(votes=TREND_VOTES)
                self._emit(
                    cron,
                    f"pulse_agent BUY-arm {symbol}: "
                    f"open=${open_px} last=${last} giveback={gb_txt} "
                    f"tracking trend ({detail})",
                )

            buy_armed = True
            window = trend_windows.setdefault(
                symbol, PriceTrendWindow(votes=TREND_VOTES)
            )
            vote = window.push(last)

            if not vote.ready or not vote.bullish:
                self._emit(
                    cron,
                    f"pulse_agent wait trend {symbol}: "
                    f"open=${open_px} last=${last} giveback={gb_txt} "
                    f"trend[{vote.detail}] need ups>downs ({detail})",
                )
                continue

            # Still require below open at fill (no chase through the open).
            if last >= open_px:
                self._emit(
                    cron,
                    f"pulse_agent wait below-open {symbol}: "
                    f"last=${last} open=${open_px} trend[{vote.detail}] ({detail})",
                )
                continue

            allowance = fund.average_spend()
            lead = (discovery.explanation or "").split("|", 1)[0].strip()
            explanation = f"pulse_agent arm+trend {lead}"
            msg = (
                f"pulse_agent BUY {symbol} fund={fund.name} "
                f"open=${open_px} last=${last} giveback={gb_txt} "
                f"trend[{vote.detail}] {detail} allowance=${allowance}"
            )
            if dry_run:
                self._emit(cron, f"DRY-RUN {msg}")
                bought += 1
                trend_windows.pop(symbol, None)
                dip_armed_syms.discard(symbol)
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
            if Holding.objects.filter(fund=fund, stock=stock, shares__gt=0).exists():
                bought += 1
                trend_windows.pop(symbol, None)
                dip_armed_syms.discard(symbol)
            fund.refresh_from_db()

        for sym in list(trend_windows.keys()):
            if sym not in live_syms:
                del trend_windows[sym]
                dip_armed_syms.discard(sym)

        return sa, buy_armed, sell_armed

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
    ) -> tuple[Optional[SmartAnalysis], bool]:
        if not holdings:
            trails.clear()
            return sa, False

        live_ids: set[int] = set()
        sell_symbols: list[str] = []
        sell_armed = False

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
            if state.armed or status == "SELL":
                sell_armed = True
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
            return sa, sell_armed

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

        return sa, sell_armed
