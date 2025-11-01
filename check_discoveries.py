#!/usr/bin/env python
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import Discovery, Advisor, SmartAnalysis
from django.db.models import Count

# Get the latest SA session
latest_sa = SmartAnalysis.objects.order_by('-id').first()

if latest_sa:
    discoveries = Discovery.objects.filter(sa=latest_sa)
    print(f"\n=== Discovery Breakdown for SA #{latest_sa.id} ===\n")
    print(f"Total discoveries: {discoveries.count()}\n")
    
    print("By advisor:")
    for advisor in discoveries.values('advisor__name').annotate(count=Count('id')).order_by('-count'):
        print(f"  {advisor['advisor__name']}: {advisor['count']}")
    
    print("\nSample stocks discovered:")
    for d in discoveries[:10]:
        print(f"  {d.stock.symbol} - {d.advisor.name}: {d.explanation[:60]}")
else:
    print("No SmartAnalysis sessions found")

