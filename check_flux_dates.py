#!/usr/bin/env python
import os
import sys
import django

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import Discovery, Advisor
from django.utils import timezone
from datetime import datetime

flux_advisor = Advisor.objects.get(name='Flux')
discoveries = Discovery.objects.filter(advisor=flux_advisor).order_by('created')

print(f'Total Flux discoveries: {discoveries.count()}')

if discoveries.exists():
    print(f'\nDate range:')
    print(f'  Earliest: {discoveries.first().created}')
    print(f'  Latest: {discoveries.last().created}')
    
    print(f'\nDecember 2024 discoveries:')
    dec_discoveries = discoveries.filter(created__year=2024, created__month=12)
    print(f'  Count: {dec_discoveries.count()}')
    if dec_discoveries.exists():
        print(f'  Range: {dec_discoveries.first().created.date()} to {dec_discoveries.last().created.date()}')
        
        # Show breakdown by day
        from collections import Counter
        dates = [d.created.date() for d in dec_discoveries]
        date_counts = Counter(dates)
        print(f'\n  Breakdown by day:')
        for date, count in sorted(date_counts.items()):
            print(f'    {date}: {count} discoveries')







