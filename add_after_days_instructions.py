#!/usr/bin/env python3
"""
Add AFTER_DAYS sell instructions to holdings without sell instructions.

For all holdings with no sell instructions, adds an AFTER_DAYS(7.0) instruction
to their discovery. This will cause holdings to be sold after 7 days.

Usage:
    python add_after_days_instructions.py
    python add_after_days_instructions.py --dry-run
    python add_after_days_instructions.py --days 14
"""

import argparse
import os
import sys
from decimal import Decimal

import django

# Ensure project settings are available before importing ORM models
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from core.models import Holding, Discovery, SellInstruction
from django.contrib.auth.models import User


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Add AFTER_DAYS sell instructions to holdings without sell instructions"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be added without actually adding instructions'
    )
    parser.add_argument(
        '--days',
        type=float,
        default=7.0,
        help='Number of days for AFTER_DAYS instruction (default: 7.0)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output for each holding'
    )
    
    return parser.parse_args(argv)


def run(argv=None):
    args = parse_args(argv)
    days = Decimal(str(args.days))
    dry_run = args.dry_run
    verbose = args.verbose

    # Find all holdings with shares > 0 for enabled users only
    # Enabled users: is_active=True and not superuser
    enabled_users = User.objects.filter(is_active=True, is_superuser=False)
    holdings = Holding.objects.filter(
        shares__gt=0,
        user__in=enabled_users
    ).select_related('stock', 'user')
    
    if not holdings.exists():
        print("No holdings found with shares > 0")
        return 0

    print(f"Found {holdings.count()} holdings to check\n")

    added_count = 0
    skipped_count = 0
    no_discovery_count = 0

    for holding in holdings:
        # Get most recent discovery for this stock
        discovery = Discovery.objects.filter(
            stock=holding.stock
        ).order_by('-created').first()
        
        if not discovery:
            if verbose:
                print(f"No discovery found for {holding.stock.symbol} (user: {holding.user.username})")
            no_discovery_count += 1
            continue
        
        # Check if discovery already has any sell instructions
        existing_instructions = SellInstruction.objects.filter(discovery=discovery)
        
        if existing_instructions.exists():
            if verbose:
                instruction_types = [inst.instruction for inst in existing_instructions]
                print(
                    f"Skipping {holding.stock.symbol} (user: {holding.user.username}): "
                    f"Already has {existing_instructions.count()} sell instruction(s): {', '.join(instruction_types)}"
                )
            skipped_count += 1
            continue
        
        # Add AFTER_DAYS instruction
        if dry_run:
            print(
                f"Would add AFTER_DAYS({days}) to {holding.stock.symbol} "
                f"(user: {holding.user.username}, discovery: {discovery.id}, "
                f"shares: {holding.shares}, created: {discovery.created.date()})"
            )
        else:
            instruction = SellInstruction()
            instruction.discovery = discovery
            instruction.instruction = 'AFTER_DAYS'
            instruction.value = days
            instruction.save()
            
            if verbose:
                print(
                    f"Added AFTER_DAYS({days}) to {holding.stock.symbol} "
                    f"(user: {holding.user.username}, discovery: {discovery.id}, "
                    f"shares: {holding.shares}, created: {discovery.created.date()})"
                )
            else:
                print(f"Added AFTER_DAYS({days}) to {holding.stock.symbol} (user: {holding.user.username})")
        
        added_count += 1

    print(f"\n{'DRY RUN: ' if dry_run else ''}Summary:")
    print(f"  Added: {added_count} AFTER_DAYS({days}) instruction(s)")
    print(f"  Skipped: {skipped_count} (already have sell instructions)")
    print(f"  No discovery: {no_discovery_count} (no discovery found for stock)")

    if dry_run:
        print("\nDRY RUN - No changes saved. Run without --dry-run to apply changes.")

    return 0


if __name__ == '__main__':
    sys.exit(run())

