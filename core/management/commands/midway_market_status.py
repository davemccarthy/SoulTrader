"""
Print MIDWAY multi-day market state: trend, mood, sector RS, stance.

Optional Stage 2: soft-rank a SO universe file vs SPY/sector.

Usage:
    python manage.py midway_market_status
    python manage.py midway_market_status --no-qqq
    python manage.py midway_market_status --universe .assessments/universe_midway_pre_llm_2026-08-14.json
    python manage.py midway_market_status --universe .assessments/universe_midway_pre_llm_2026-08-14.json --top 20
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.services.market.midway_candidates import (
    DEFAULT_UNIVERSE,
    format_soft_candidates,
    load_universe_rows,
    rank_soft_candidates,
    soft_bar_for_stance,
)
from core.services.market.midway_state import (
    evaluate_midway_market,
    format_midway_market_card,
)


class Command(BaseCommand):
    help = "Show MIDWAY market card; optionally soft-rank an SO universe."

    def add_arguments(self, parser):
        parser.add_argument(
            "--no-qqq",
            action="store_true",
            help="Skip QQQ benchmark (SPY + IWM only).",
        )
        parser.add_argument(
            "--universe",
            nargs="?",
            const=str(DEFAULT_UNIVERSE),
            default=None,
            help=(
                "Path to MIDWAY universe JSON (records with symbol + so_pair). "
                f"If flag alone: {DEFAULT_UNIVERSE}"
            ),
        )
        parser.add_argument(
            "--top",
            type=int,
            default=15,
            help="Max rows in soft-rank table (default 15).",
        )
        parser.add_argument(
            "--no-sector-tilt",
            action="store_true",
            help="Do not boost/penalize soft/hot sector sleeves.",
        )

    def handle(self, *args, **options):
        state = evaluate_midway_market(include_qqq=not options["no_qqq"])
        now = timezone.now()
        self.stdout.write(self.style.NOTICE(f"As of (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}"))
        self.stdout.write("")
        self.stdout.write(format_midway_market_card(state))
        self.stdout.write("")

        stance = state.stance
        if stance == "no_deployment":
            self.stdout.write(self.style.ERROR(f"Action: {state.permission}"))
        elif stance == "dont_chase":
            self.stdout.write(self.style.WARNING(f"Action: {state.permission}"))
        else:
            self.stdout.write(self.style.SUCCESS(f"Action: {state.permission}"))

        universe = options.get("universe")
        if not universe:
            return

        path = Path(universe)
        if not path.is_file():
            raise CommandError(f"Universe file not found: {path}")

        self.stdout.write("")
        rows = load_universe_rows(path)
        self.stdout.write(self.style.NOTICE(f"Universe: {path} ({len(rows)} names)"))
        candidates = rank_soft_candidates(
            state,
            rows,
            prefer_soft_sectors=not options["no_sector_tilt"],
        )
        self.stdout.write(
            format_soft_candidates(
                candidates,
                top=max(1, int(options["top"])),
                stance=state.stance,
                bar=soft_bar_for_stance(state.stance),
            )
        )
