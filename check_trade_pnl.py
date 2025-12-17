#!/usr/bin/env python3
"""
Diagnostic script to check Trade P&L calculation for a user
"""

import os
import sys
import django
from decimal import Decimal

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.contrib.auth.models import User
from core.models import Trade

def check_trade_pnl(username):
    """Check Trade P&L calculation for a user."""
    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        print(f"User '{username}' not found")
        return
    
    # Get all SELL trades with cost
    sell_trades = Trade.objects.filter(
        user=user,
        action='SELL',
        cost__isnull=False
    ).order_by('created')
    
    print(f"\nTrade P&L Analysis for {username}")
    print("=" * 80)
    print(f"Total SELL trades with cost: {sell_trades.count()}")
    
    if sell_trades.count() == 0:
        print("No SELL trades with cost basis found")
        return
    
    # Calculate using same method as context_processors.py
    trade_pnl = sum(
        (Decimal(str(trade.price)) - Decimal(str(trade.cost))) * Decimal(trade.shares)
        for trade in sell_trades
        if trade.cost
    )
    
    trade_cost_basis = sum(
        Decimal(str(trade.cost)) * Decimal(trade.shares)
        for trade in sell_trades
        if trade.cost
    )
    
    if trade_cost_basis > 0:
        trade_pnl_percent = (trade_pnl / trade_cost_basis) * 100
    else:
        trade_pnl_percent = Decimal('0.0')
    
    print(f"\nTotal Cost Basis: ${trade_cost_basis:,.2f}")
    print(f"Total Proceeds: ${trade_cost_basis + trade_pnl:,.2f}")
    print(f"Total P&L: ${trade_pnl:+,.2f}")
    print(f"P&L Percentage: {trade_pnl_percent:+.2f}%")
    
    # Show individual trades
    print(f"\nIndividual Trades:")
    print("-" * 80)
    print(f"{'Date':<12} {'Symbol':<8} {'Shares':<8} {'Cost':<12} {'Price':<12} {'P&L':<12} {'P&L %':<10}")
    print("-" * 80)
    
    for trade in sell_trades:
        cost_total = Decimal(str(trade.cost)) * Decimal(trade.shares)
        proceeds = Decimal(str(trade.price)) * Decimal(trade.shares)
        trade_pnl_amt = proceeds - cost_total
        trade_pnl_pct = (trade_pnl_amt / cost_total * 100) if cost_total > 0 else Decimal('0')
        
        print(f"{trade.created.strftime('%Y-%m-%d'):<12} "
              f"{trade.stock.symbol:<8} "
              f"{trade.shares:<8} "
              f"${cost_total:>10,.2f} "
              f"${proceeds:>10,.2f} "
              f"${trade_pnl_amt:>+10,.2f} "
              f"{trade_pnl_pct:>+8.2f}%")
    
    # Check for SELL trades without cost
    sell_trades_no_cost = Trade.objects.filter(
        user=user,
        action='SELL',
        cost__isnull=True
    ).count()
    
    if sell_trades_no_cost > 0:
        print(f"\n⚠️  Warning: {sell_trades_no_cost} SELL trade(s) without cost basis (excluded from calculation)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_trade_pnl.py <username>")
        sys.exit(1)
    
    check_trade_pnl(sys.argv[1])














