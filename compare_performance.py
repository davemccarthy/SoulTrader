#!/usr/bin/env python
"""Compare discovery performance pre vs post Thanksgiving 2025"""
import os
import sys
import django

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import SmartAnalysis, Discovery
from django.utils import timezone
from datetime import datetime

thanksgiving_2025 = timezone.make_aware(datetime(2025, 11, 27))
print(f'Thanksgiving 2025: Nov 27, 2025\n')

# Find SA sessions
pre_sessions = SmartAnalysis.objects.filter(
    started__gte=timezone.make_aware(datetime(2025, 10, 27)),
    started__lt=thanksgiving_2025
).order_by('started')

post_sessions = SmartAnalysis.objects.filter(
    started__gte=thanksgiving_2025
).order_by('started')

print(f'Pre-Thanksgiving (Oct 27 - Nov 26):')
print(f'  Sessions: {pre_sessions.count()}')
if pre_sessions.exists():
    print(f'  SA IDs: {pre_sessions.first().id} to {pre_sessions.last().id}')
    pre_discoveries = Discovery.objects.filter(sa_id__in=pre_sessions.values_list('id', flat=True))
    print(f'  Discoveries: {pre_discoveries.count()}')

print(f'\nPost-Thanksgiving (Nov 27+):')
print(f'  Sessions: {post_sessions.count()}')
if post_sessions.exists():
    print(f'  SA IDs: {post_sessions.first().id} to {post_sessions.last().id}')
    post_discoveries = Discovery.objects.filter(sa_id__in=post_sessions.values_list('id', flat=True))
    print(f'  Discoveries: {post_discoveries.count()}')

print(f'\nNow run:')
if pre_sessions.exists() and post_sessions.exists():
    print(f'  python manage.py analyse_discovery --start-sa {pre_sessions.first().id} --end-sa {pre_sessions.last().id}')
    print(f'  python manage.py analyse_discovery --start-sa {post_sessions.first().id} --end-sa {post_sessions.last().id}')


