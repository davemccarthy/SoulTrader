"""
Backfill SO snapshot fields on Assessment rows (stability, opportunity, sub-metrics).

Uses yfinance once per assessment — run offline, not on web requests.

    python manage.py backfill_assessment_so
    python manage.py backfill_assessment_so --advisor "Polygon.io" --days 30
    python manage.py backfill_assessment_so --limit 50 --dry-run
"""

from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models import Assessment, Discovery
from core.services.health.risk_matrix import persist_so_on_assessment


class Command(BaseCommand):
    help = "Backfill SO snapshot columns on Assessment rows missing stability."

    def add_arguments(self, parser):
        parser.add_argument(
            "--advisor",
            type=str,
            default="",
            help="Only assessments linked to discoveries from this advisor name.",
        )
        parser.add_argument(
            "--days",
            type=int,
            default=0,
            help="If set with --advisor, limit to discoveries within this many days.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Max assessments to process (0 = no limit).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List candidates only; do not call yfinance or save.",
        )

    def handle(self, *args, **options):
        qs = (
            Assessment.objects.filter(stability__isnull=True)
            .select_related("stock")
            .order_by("-id")
        )

        advisor_name = (options.get("advisor") or "").strip()
        days = int(options.get("days") or 0)
        if advisor_name:
            disc_qs = Discovery.objects.filter(advisor__name=advisor_name)
            if days > 0:
                disc_qs = disc_qs.filter(
                    created__gte=timezone.now() - timedelta(days=days)
                )
            assessment_ids = disc_qs.values_list("assessment_id", flat=True)
            qs = qs.filter(id__in=[i for i in assessment_ids if i])

        limit = int(options.get("limit") or 0)
        if limit > 0:
            qs = qs[:limit]

        total = qs.count()
        if options.get("dry_run"):
            for a in qs[:20]:
                self.stdout.write(f"  would backfill {a.id} {a.stock.symbol}")
            if total > 20:
                self.stdout.write(f"  ... and {total - 20} more")
            self.stdout.write(self.style.WARNING(f"Dry run: {total} assessment(s)"))
            return

        ok = 0
        fail = 0
        for assessment in qs.iterator():
            sym = assessment.stock.symbol if assessment.stock else "?"
            try:
                if persist_so_on_assessment(assessment):
                    ok += 1
                    self.stdout.write(f"  {sym} (assessment {assessment.id})")
                else:
                    fail += 1
                    self.stdout.write(self.style.WARNING(f"  skip {sym} ({assessment.id})"))
            except Exception as exc:
                fail += 1
                self.stdout.write(self.style.ERROR(f"  fail {sym} ({assessment.id}): {exc}"))

        self.stdout.write(self.style.SUCCESS(f"Done: {ok} updated, {fail} skipped/failed, {total} candidates"))
