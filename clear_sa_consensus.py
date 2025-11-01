#!/usr/bin/env python
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import Consensus, Recommendation, SmartAnalysis

# Get the latest SA session
latest_sa = SmartAnalysis.objects.order_by('-id').first()

if not latest_sa:
    print("No SmartAnalysis sessions found")
    exit(1)

print(f"\nClearing consensus and recommendations for SA #{latest_sa.id}...")

# Count before
rec_count = Recommendation.objects.filter(sa=latest_sa).count()
consensus_count = Consensus.objects.filter(sa=latest_sa).count()

print(f"Found {rec_count} recommendations and {consensus_count} consensus records")

# Delete
Recommendation.objects.filter(sa=latest_sa).delete()
Consensus.objects.filter(sa=latest_sa).delete()

print(f"✅ Cleared! You can now re-run with --reuse\n")

