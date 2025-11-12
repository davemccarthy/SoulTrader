#!/usr/bin/env python3
"""
One-off helper to align Discovery.created with the SmartAnalysis start timestamp.

Existing rows created before the `created` column was introduced were backfilled
with "now", which breaks any historical analysis. This script rewrites those
timestamps to match their associated SmartAnalysis session start.
"""

import os
import sys
from datetime import timedelta

import django
from django.db import transaction
from django.utils import timezone

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from core.models import Discovery  # noqa: E402


def backfill():
    tolerance = timedelta(hours=1)
    now = timezone.now()
    updated = 0
    total = 0

    with transaction.atomic():
        discoveries = Discovery.objects.select_related("sa").all()
        for discovery in discoveries:
            total += 1
            sa_started = discovery.sa.started
            if sa_started is None:
                continue

            created = discovery.created
            if created is None or created > now or abs(created - sa_started) > tolerance:
                discovery.created = sa_started
                discovery.save(update_fields=["created"])
                updated += 1

    print(f"Checked {total} discoveries; updated {updated} rows.")


if __name__ == "__main__":
    try:
        backfill()
    except KeyboardInterrupt:
        sys.exit(130)

