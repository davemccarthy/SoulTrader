"""
Print intraday benchmark tape and suggested new-entry posture.

Usage:
    python manage.py market_tape_status
    python manage.py market_tape_status --symbols SPY,QQQ,IWM

Pulse: RED = no discover; AMBER/GREEN/WHITE = trade with tape-colored IPC
(amber 0.2/0.2, green 0.4/0.2, white 0.6/0.4). Push on color change.
"""
from __future__ import annotations

from django.core.management.base import BaseCommand
from django.utils import timezone

from core.services.market.tape import (
    DEFAULT_BENCHMARKS,
    TAPE_AMBER_VS_OPEN_PCT,
    TAPE_AMBER_VS_PRIOR_CLOSE_PCT,
    TAPE_RED_VS_OPEN_PCT,
    TAPE_RED_VS_PRIOR_CLOSE_PCT,
    TAPE_WHITE_VS_OPEN_PCT,
    TAPE_WHITE_VS_PRIOR_CLOSE_PCT,
    evaluate_tape,
    fetch_tape,
)


class Command(BaseCommand):
    help = "Show intraday SPY/QQQ tape and red/amber/green/white posture."

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
        self.stdout.write("Thresholds:")
        self.stdout.write(
            f"  RED    any vs open <= {TAPE_RED_VS_OPEN_PCT:+.2f}% "
            f"or vs prior <= {TAPE_RED_VS_PRIOR_CLOSE_PCT:+.2f}%"
        )
        self.stdout.write(
            f"  AMBER  any vs open <= {TAPE_AMBER_VS_OPEN_PCT:+.2f}% "
            f"or vs prior <= {TAPE_AMBER_VS_PRIOR_CLOSE_PCT:+.2f}% (not red)"
        )
        self.stdout.write(
            f"  WHITE  ALL vs open >= {TAPE_WHITE_VS_OPEN_PCT:+.2f}% "
            f"and vs prior >= {TAPE_WHITE_VS_PRIOR_CLOSE_PCT:+.2f}%"
        )
        self.stdout.write("  GREEN  otherwise")
        self.stdout.write("")
        self.stdout.write("Pulse IPC: amber 0.2/0.2 | green 0.4/0.2 | white 0.6/0.4")
        self.stdout.write("")

        state = verdict.state.upper()
        style = self.style.SUCCESS
        if verdict.state == "amber":
            style = self.style.WARNING
        elif verdict.state == "red":
            style = self.style.ERROR
        elif verdict.state == "white":
            style = self.style.SUCCESS
        self.stdout.write(style(f"Verdict: {state}"))
        self.stdout.write(f"Reason: {verdict.reason}")
        self.stdout.write("")
        if verdict.state == "red":
            self.stdout.write("Action: no new Pulse discovers.")
        elif verdict.state == "amber":
            self.stdout.write("Action: caution trade — IPC 0.2/0.2.")
        elif verdict.state == "white":
            self.stdout.write("Action: strong tape — IPC 0.6/0.4.")
        else:
            self.stdout.write("Action: normal trade — IPC 0.4/0.2.")
