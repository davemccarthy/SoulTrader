#!/usr/bin/env python
"""
Script to add TARGET_DIMINISHING sell instruction to a discovery.

Usage:
    python add_diminishing_target.py <stock_symbol> <sa_id> <target_price>
    
Example:
    python add_diminishing_target.py ODV 400 5.50
    
Note: max_days is hardcoded to 14 days in the system.
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from decimal import Decimal
from core.models import Discovery, SellInstruction, Stock, Holding

def add_diminishing_target(stock_symbol, sa_id, target_price):
    """Add TARGET_DIMINISHING instruction to a discovery."""
    try:
        # Find the stock
        try:
            stock = Stock.objects.get(symbol=stock_symbol.upper())
        except Stock.DoesNotExist:
            print(f"Error: Stock {stock_symbol} not found")
            return False
        
        # Find the discovery for this stock and SA
        try:
            discovery = Discovery.objects.get(stock=stock, sa_id=sa_id)
        except Discovery.DoesNotExist:
            print(f"Error: No discovery found for {stock_symbol} in SA #{sa_id}")
            return False
        except Discovery.MultipleObjectsReturned:
            # If multiple discoveries, use the most recent one
            discovery = Discovery.objects.filter(stock=stock, sa_id=sa_id).order_by('-created').first()
            print(f"Warning: Multiple discoveries found for {stock_symbol} in SA #{sa_id}, using most recent (ID: {discovery.id})")
        
        # Get buy_price from holding or discovery price
        holding = Holding.objects.filter(stock=stock).first()
        if holding and holding.average_price:
            buy_price = holding.average_price
            print(f"Using buy_price from holding: ${buy_price}")
        elif discovery.price:
            buy_price = discovery.price
            print(f"Using buy_price from discovery: ${buy_price}")
        else:
            print(f"Warning: No buy_price found (neither holding.average_price nor discovery.price). You may need to set this manually.")
            buy_price = None
        
        # Check if TARGET_DIMINISHING already exists
        existing = SellInstruction.objects.filter(
            discovery=discovery,
            instruction='TARGET_DIMINISHING'
        ).first()
        
        if existing:
            print(f"\nTARGET_DIMINISHING instruction already exists (ID: {existing.id})")
            print(f"  Current value1: ${existing.value1}")
            print(f"  Current value2 (max_days): {int(existing.value2) if existing.value2 else 'None (defaults to 14)'}")
            response = input("Update it? (y/n): ")
            if response.lower() != 'y':
                print("Cancelled")
                return False
            
            existing.value1 = Decimal(str(target_price))
            # value2 (max_days) can be set here if needed, otherwise defaults to 14
            existing.save()
            print(f"\n✓ Updated TARGET_DIMINISHING instruction (ID: {existing.id})")
            print(f"  Original target: ${existing.value1}")
            max_days = int(existing.value2) if existing.value2 else 14
            if buy_price:
                print(f"  Buy price: ${buy_price}")
                print(f"  Target will diminish from ${existing.value1} → ${buy_price} over {max_days} days")
        else:
            # Create new instruction
            instruction = SellInstruction.objects.create(
                discovery=discovery,
                instruction='TARGET_DIMINISHING',
                value1=Decimal(str(target_price)),
                value2=None  # Defaults to 14 days in analysis.py
            )
            print(f"\n✓ Created TARGET_DIMINISHING instruction (ID: {instruction.id})")
            print(f"  Stock: {stock_symbol}")
            print(f"  Discovery: SA #{sa_id}")
            print(f"  Original target: ${instruction.value1}")
            if buy_price:
                print(f"  Buy price: ${buy_price}")
                print(f"  Target will diminish from ${instruction.value1} → ${buy_price} over 14 days (default)")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    
    stock_symbol = sys.argv[1]
    sa_id = int(sys.argv[2])
    target_price = float(sys.argv[3])
    
    add_diminishing_target(stock_symbol, sa_id, target_price)

