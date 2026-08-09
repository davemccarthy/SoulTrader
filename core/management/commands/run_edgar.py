"""
Run Edgar advisor independently (no smartanalyse).

Usage:
    python manage.py run_edgar
    python manage.py run_edgar --accession 0001193125-26-084267
    python manage.py run_edgar --watch SHOP,WULF,KYMR
    python manage.py run_edgar --seed-watch SHOP,WULF --open-check-only
    python manage.py run_edgar --open-check-only
    python manage.py run_edgar --open-check-only --force-open-check --dry-run
"""

from django.core.management.base import BaseCommand

from core.services.advisors import edgar


class Command(BaseCommand):
    help = (
        "Run EDDIE-8 standalone: latest 8-Ks, Stage 1 watch, Stage 1b media, "
        "and/or Stage 2 open-check."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--accession",
            type=str,
            default="",
            help=(
                "SEC accession number(s) to analyze instead of latest filings. "
                "Comma-separated for multiple."
            ),
        )
        parser.add_argument(
            "--watch",
            type=str,
            default="",
            help=(
                "Comma-separated tickers: resolve latest 8-K and run full Stage 1 "
                "(filters + EX-99 LLM → watch on pass). Not a tape-only seed."
            ),
        )
        parser.add_argument(
            "--seed-watch",
            type=str,
            default="",
            help=(
                "Comma-separated tickers to seed as Pending edgar_earnings watches "
                "(tape-only; skips Stage 1). Prefer --watch for pass/fail."
            ),
        )
        parser.add_argument(
            "--open-check-only",
            action="store_true",
            help=(
                "Skip filing fetch; run Stage 1b media (due watches) then "
                "Stage 2 open-check."
            ),
        )
        parser.add_argument(
            "--force-open-check",
            action="store_true",
            help=(
                "Bypass +15m / RTH gate (local lab). Can discover outside the "
                "normal window — use with care."
            ),
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Media + open-check classify + log only (no watch writes / discover).",
        )

    def handle(self, *args, **options):
        raw = (options.get("accession") or "").strip()
        accessions = None
        if raw:
            accessions = [a.strip() for a in raw.split(",") if a.strip()]
            self.stdout.write(f"Running EDDIE-8 on {len(accessions)} accession(s)...")

        watch_raw = (options.get("watch") or "").strip()
        analyze_symbols = None
        if watch_raw:
            analyze_symbols = [
                s.strip().upper() for s in watch_raw.split(",") if s.strip()
            ]
            self.stdout.write(
                f"Stage-1 analyze for {len(analyze_symbols)} ticker(s)..."
            )

        seed_raw = (options.get("seed_watch") or "").strip()
        seed_watches = None
        if seed_raw:
            seed_watches = [s.strip().upper() for s in seed_raw.split(",") if s.strip()]
            self.stdout.write(
                f"Seeding {len(seed_watches)} tape-only watch(es) (no Stage 1)..."
            )

        open_check_only = bool(options.get("open_check_only"))
        force_open_check = bool(options.get("force_open_check"))
        dry_run = bool(options.get("dry_run"))
        # --watch already does Stage 1; don't also fetch latest unless accessions set.
        if analyze_symbols and not accessions:
            open_check_only = True
        if open_check_only and not analyze_symbols:
            self.stdout.write("Open-check only (no filing fetch)...")
        if force_open_check:
            self.stdout.write(self.style.WARNING("FORCE open-check (time gate bypassed)"))
        if dry_run:
            self.stdout.write("Dry-run open-check (no writes)")

        result, err = edgar.run_edgar_standalone(
            accessions=accessions,
            open_check_only=open_check_only,
            force_open_check=force_open_check,
            dry_run_open_check=dry_run,
            seed_watches=seed_watches,
            analyze_symbols=analyze_symbols,
        )
        if err:
            self.stdout.write(self.style.ERROR(err))
            return

        if result is not None:
            self.stdout.write(self.style.SUCCESS(str(result)))
        else:
            self.stdout.write(self.style.WARNING("Edgar discover() completed with no result"))
