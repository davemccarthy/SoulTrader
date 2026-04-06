"""
Run BIZFEED advisor standalone (no smartanalyse).

Usage:
    python manage.py run_bizfeed
"""

from django.core.management.base import BaseCommand

from core.services.advisors import bizfeed


class Command(BaseCommand):
    help = "Run BIZFEED advisor standalone (corporate RSS stub)."

    def handle(self, *args, **options):
        result, err = bizfeed.run_bizfeed_standalone()
        if err:
            self.stdout.write(self.style.ERROR(err))
            return

        if result is not None:
            self.stdout.write(self.style.SUCCESS(str(result)))
        else:
            self.stdout.write(self.style.WARNING("BIZFEED run completed with no result message"))
