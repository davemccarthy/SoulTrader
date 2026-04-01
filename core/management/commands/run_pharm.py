"""
Run PHARM advisor independently (no smartanalyse).

Usage:
    python manage.py run_pharm

No params to start with.
"""

from django.core.management.base import BaseCommand

from core.services.advisors import pharm


class Command(BaseCommand):
    help = "Run PHARM advisor standalone (pharma pipeline stub). No params yet."

    def handle(self, *args, **options):
        result, err = pharm.run_pharm_standalone()
        if err:
            self.stdout.write(self.style.ERROR(err))
            return

        if result is not None:
            self.stdout.write(self.style.SUCCESS(str(result)))
        else:
            self.stdout.write(self.style.WARNING("Pharm discover() completed with no result"))

