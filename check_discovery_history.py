#!/usr/bin/env python
"""Check discovery history for stocks"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import Discovery, Stock, Advisor, SmartAnalysis

stocks = ['UAA', 'ATUS', 'TEF', 'GAU', 'CINT']
user_advisor = Advisor.objects.get(python_class='User')

print('=== DISCOVERY HISTORY (User Advisor) ===\n')
for symbol in stocks:
    try:
        stock = Stock.objects.get(symbol=symbol)
        discoveries = Discovery.objects.filter(stock=stock, advisor=user_advisor).order_by('-created')
        
        if discoveries.exists():
            print(f'{symbol}:')
            for d in discoveries[:3]:  # Show last 3
                sa = d.sa
                print(f'  SA #{sa.id} on {d.created.strftime("%Y-%m-%d %H:%M")}')
                print(f'    Explanation: {d.explanation[:100]}...')
        else:
            print(f'{symbol}: No discoveries found')
        print()
    except Stock.DoesNotExist:
        print(f'{symbol}: Stock not found\n')












