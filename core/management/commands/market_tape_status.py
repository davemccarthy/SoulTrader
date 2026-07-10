"""
Print intraday benchmark tape and a suggested new-entry posture.

Usage:
    python manage.py market_tape_status
    python manage.py market_tape_status --symbols SPY,QQQ,IWM

Manual policy: on RED, remove discovery advisors from affected funds until tape
improves. Existing holdings continue IPC / rebuy / session exits via analyze_holdings.
"""
from __future__ import annotations

from django.core.management.base import BaseCommand
from django.utils import timezone

from core.services.market.tape import (
    DEFAULT_BENCHMARKS,
    TAPE_RED_VS_OPEN_PCT,
    TAPE_RED_VS_PRIOR_CLOSE_PCT,
    TAPE_YELLOW_VS_OPEN_PCT,
    TAPE_YELLOW_VS_PRIOR_CLOSE_PCT,
    evaluate_tape,
    fetch_tape,
)


class Command(BaseCommand):
    help = "Show intraday SPY/QQQ tape and suggested green/yellow/red for new entries."

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols",
            type=str,
            default=",".join(DEFAULT_BENCHMARKS),
            help="Comma-separated benchmark tickers (default SPY,QQQ)",
        )

    def handle(self, *args, **options):
        symbols = [s.strip().upper() for s in str(options["symbols"]).split(",") if s.strip()]
        readings = fetch_tape(symbols)
        verdict = evaluate_tape(readings)

        now = timezone.now()
        self.stdout.write(self.style.NOTICE("=== Market tape (intraday) ==="))
        self.stdout.write(f"As of (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}")
        self.stdout.write("")
        self.stdout.write(
            f"{'symbol':<8} {'price':>10} {'open':>10} {'prior':>10} "
            f"{'vs open':>10} {'vs prior':>10}"
        )
        for sym in symbols:
            r = readings.get(sym)
            if r is None:
                self.stdout.write(f"{sym:<8} (no data)")
                continue
            price = f"{r.price:.2f}" if r.price is not None else "n/a"
            open_px = f"{r.open_px:.2f}" if r.open_px is not None else "n/a"
            prior = f"{r.prior_close:.2f}" if r.prior_close is not None else "n/a"
            self.stdout.write(
                f"{sym:<8} {price:>10} {open_px:>10} {prior:>10} "
                f"{r.vs_open_display():>10} {r.vs_prior_close_display():>10}"
            )

        self.stdout.write("")
        self.stdout.write("Thresholds (new-entry suggestion):")
        self.stdout.write(
            f"  RED    vs open <= {TAPE_RED_VS_OPEN_PCT:+.1f}% "
            f"or vs prior close <= {TAPE_RED_VS_PRIOR_CLOSE_PCT:+.1f}%"
        )
        self.stdout.write(
            f"  YELLOW vs open <= {TAPE_YELLOW_VS_OPEN_PCT:+.1f}% "
            f"or vs prior close <= {TAPE_YELLOW_VS_PRIOR_CLOSE_PCT:+.1f}%"
        )
        self.stdout.write("")

        state = verdict.state.upper()
        style = self.style.SUCCESS
        if verdict.state == "yellow":
            style = self.style.WARNING
        elif verdict.state == "red":
            style = self.style.ERROR
        self.stdout.write(style(f"Verdict: {state}"))
        self.stdout.write(f"Reason: {verdict.reason}")
        self.stdout.write("")
        if verdict.state == "red":
            self.stdout.write(
                "Manual: remove discovery advisors from funds until tape improves. "
                "Holdings keep sell/rebuy handling."
            )
        elif verdict.state == "yellow":
            self.stdout.write(
                "Manual: consider pausing new discoveries if tape keeps weakening."
            )
        else:
            self.stdout.write("Manual: normal to run discovery advisors on funds.")
