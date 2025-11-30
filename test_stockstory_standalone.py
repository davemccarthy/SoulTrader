#!/usr/bin/env python3
"""
Standalone test script for StockStory discovery.
Tests the Story.discover() method with real or mock SA sessions.
"""

import os
import sys
import django
from datetime import datetime, timedelta
from django.utils import timezone

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import SmartAnalysis, Advisor
from core.services.advisors.story import Story

def test_stockstory_discovery(use_latest_sa=True, create_mock_sa=False):
    """
    Test StockStory discovery.
    
    Args:
        use_latest_sa: If True, use the latest SA from database
        create_mock_sa: If True, create a mock SA for testing
    """
    print("=" * 80)
    print("StockStory Discovery Test")
    print("=" * 80)
    
    # Get or create SA session
    if create_mock_sa:
        print("\nCreating mock SA session...")
        sa = SmartAnalysis()
        sa.username = "test_user"
        sa.save()
        print(f"Created SA session {sa.id} at {sa.started}")
    elif use_latest_sa:
        sa = SmartAnalysis.objects.order_by('-id').first()
        if not sa:
            print("No SA sessions found in database. Creating mock SA...")
            sa = SmartAnalysis()
            sa.username = "test_user"
            sa.save()
        print(f"\nUsing SA session {sa.id} (started: {sa.started})")
    else:
        print("Error: Must specify use_latest_sa=True or create_mock_sa=True")
        return
    
    # Check for previous SA
    prev_sa = SmartAnalysis.objects.filter(id__lt=sa.id).order_by('-id').first()
    if prev_sa:
        print(f"Previous SA session {prev_sa.id} (started: {prev_sa.started})")
        time_window = sa.started - prev_sa.started
        print(f"Time window: {time_window}")
    else:
        print("No previous SA session found (will use 24-hour fallback)")
    
    # Get StockStory advisor
    try:
        advisor = Advisor.objects.get(name="StockStory")
    except Advisor.DoesNotExist:
        print("\nError: StockStory advisor not found in database.")
        print("Make sure the advisor is registered in the database.")
        return
    
    print(f"\nUsing advisor: {advisor.name} (enabled: {advisor.enabled})")
    
    # Create Story instance
    story = Story(advisor)
    
    # Count discoveries before
    from core.models import Discovery
    discoveries_before = Discovery.objects.filter(sa=sa, advisor=advisor).count()
    print(f"\nDiscoveries before: {discoveries_before}")
    
    # Run discovery
    print("\n" + "-" * 80)
    print("Running discovery...")
    print("-" * 80)
    
    try:
        story.discover(sa)
        print("Discovery completed successfully!")
    except Exception as e:
        print(f"\nError during discovery: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Count discoveries after
    discoveries_after = Discovery.objects.filter(sa=sa, advisor=advisor).count()
    new_discoveries = discoveries_after - discoveries_before
    
    print("\n" + "-" * 80)
    print("Results:")
    print("-" * 80)
    print(f"New discoveries: {new_discoveries}")
    print(f"Total discoveries for this SA: {discoveries_after}")
    
    if new_discoveries > 0:
        print("\nNew discoveries:")
        for discovery in Discovery.objects.filter(sa=sa, advisor=advisor).order_by('-created')[:new_discoveries]:
            print(f"  - {discovery.stock.symbol}: {discovery.explanation[:100]}...")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test StockStory discovery")
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Create a mock SA session for testing'
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        default=True,
        help='Use the latest SA session from database (default)'
    )
    
    args = parser.parse_args()
    
    test_stockstory_discovery(
        use_latest_sa=args.latest and not args.mock,
        create_mock_sa=args.mock
    )



