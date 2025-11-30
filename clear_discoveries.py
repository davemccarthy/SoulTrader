#!/usr/bin/env python
"""Clear old discoveries for specific stocks"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import Discovery, Stock, Advisor

stocks = ['UAA', 'ATUS', 'TEF', 'GAU', 'CINT']
user_advisor = Advisor.objects.get(python_class='User')

deleted = 0
for symbol in stocks:
    try:
        stock = Stock.objects.get(symbol=symbol)
        count = Discovery.objects.filter(stock=stock, advisor=user_advisor).delete()[0]
        deleted += count
        print(f'Deleted {count} discovery(ies) for {symbol}')
    except Stock.DoesNotExist:
        print(f'Stock {symbol} not found')

print(f'\nTotal deleted: {deleted} discoveries')












