"""
Temporary command for testing on-demand Form 4 intel.

Usage:
    python manage.py get_form4 TAT --days 30
    python manage.py get_form4 TAT --start-date 2026-06-01 --end-date 2026-06-30 --json
"""

from __future__ import annotations

import json
from datetime import datetime

from django.core.management.base import BaseCommand, CommandError

from core.services.sec.form4 import get_form4_intel


def _parse_date(value: str | None, option_name: str):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise CommandError(f"{option_name} must use YYYY-MM-DD") from exc


class Command(BaseCommand):
    help = "Temporary: fetch on-demand Form 4 intel for a stock"

    def add_arguments(self, parser):
        parser.add_argument("symbol", type=str, help="Stock ticker, e.g. TAT")
        parser.add_argument("--days", type=int, default=30, help="Lookback days when no start date is given")
        parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD")
        parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD")
        parser.add_argument("--limit", type=int, default=100, help="Maximum filings to fetch")
        parser.add_argument(
            "--min-abs-total",
            type=float,
            default=0.0,
            help="Only include scored filings with abs(total) at least this value",
        )
        parser.add_argument("--json", action="store_true", help="Print the full JSON payload")

    def handle(self, *args, **options):
        start_date = _parse_date(options["start_date"], "--start-date")
        end_date = _parse_date(options["end_date"], "--end-date")

        try:
            intel = get_form4_intel(
                options["symbol"],
                days=options["days"],
                start_date=start_date,
                end_date=end_date,
                limit=options["limit"],
                min_abs_total=options["min_abs_total"],
            )
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        if options["json"]:
            self.stdout.write(json.dumps(intel, indent=2, default=str))
            return

        self.stdout.write(
            f"{intel['symbol']} Form 4 intel "
            f"{intel['start_date']} to {intel['end_date']}"
        )
        self.stdout.write(
            f"Filings: {intel['filing_count']} | "
            f"Parsed: {intel['parsed_count']} | "
            f"Scoreable: {intel['scored_count']} | "
            f"Net: {intel['total']:+.1f} | "
            f"Buy: {intel['buy_total']:+.1f} | "
            f"Sell: {intel['sell_total']:+.1f}"
        )

        filings = intel.get("filings") or []
        if not filings:
            self.stdout.write(self.style.WARNING("No scoreable Form 4 buy/sell filings found."))
            return

        for filing in filings[:10]:
            self.stdout.write(
                f"- {filing['period']} {filing['accession']} "
                f"{filing['insider_name']} ({filing['role_label']}) "
                f"total={filing['total']:+.1f}"
            )
