#!/usr/bin/env python
"""Quick script to find SmartAnalysis sessions around Thanksgiving"""
import os
import sys
import django

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import SmartAnalysis
from django.utils import timezone
from datetime import timedelta

thanksgiving = timezone.make_aware(timezone.datetime(2024, 11, 28))
print(f'Thanksgiving 2024: Nov 28, 2024')
print(f'Today: {timezone.now().date()}\n')

# Pre-Thanksgiving (30 days before)
print('Pre-Thanksgiving (Oct 29 - Nov 27):')
pre = SmartAnalysis.objects.filter(
    started__gte=thanksgiving - timedelta(days=30),
    started__lt=thanksgiving
).order_by('started')

print(f'  Found {pre.count()} SA sessions')
if pre.exists():
    print(f'  First: SA#{pre.first().id} on {pre.first().started.date()}')
    print(f'  Last: SA#{pre.last().id} on {pre.last().started.date()}')
    print(f'  SA ID range: {pre.first().id} to {pre.last().id}')

# Post-Thanksgiving
print('\nPost-Thanksgiving (Nov 28 - Today):')
post = SmartAnalysis.objects.filter(
    started__gte=thanksgiving,
    started__lte=timezone.now()
).order_by('started')

print(f'  Found {post.count()} SA sessions')
if post.exists():
    print(f'  First: SA#{post.first().id} on {post.first().started.date()}')
    print(f'  Last: SA#{post.last().id} on {post.last().started.date()}')
    print(f'  SA ID range: {post.first().id} to {post.last().id}')


