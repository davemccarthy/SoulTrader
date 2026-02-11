"""
Run Edgar advisor independently (no smartanalyse).

Usage:
    python manage.py run_edgar

No params to start with.
"""

from django.core.management.base import BaseCommand

from core.services.advisors import edgar


class Command(BaseCommand):
    help = "Run ED-8 advisor standalone (8-K pipeline stub). No params yet."

    def handle(self, *args, **options):
        result, err = edgar.run_edgar_standalone()
        if err:
            self.stdout.write(self.style.ERROR(err))
            return

        if result is not None:
            self.stdout.write(self.style.SUCCESS(str(result)))
        else:
            self.stdout.write(self.style.WARNING("Edgar discover() completed with no result"))

