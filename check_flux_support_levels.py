#!/usr/bin/env python
"""Check Flux discoveries - are prices actually near support?"""
import os
import sys
import django

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

import re
from django.utils import timezone
from datetime import timedelta
from core.models import Discovery, Advisor, Trade

flux_advisor = Advisor.objects.get(name='Flux')
cutoff = timezone.now() - timedelta(days=30)

discoveries = Discovery.objects.filter(
    advisor=flux_advisor,
    created__gte=cutoff
).select_related('stock', 'sa').order_by('-created')

print(f'Analyzing {discoveries.count()} recent Flux discoveries\n')
print('=' * 120)
print(f"{'Symbol':<8} {'Price':<8} {'Support':<10} {'Resistance':<12} {'Position%':<10} {'Upside%':<10} {'EOD Change%':<12}")
print('-' * 120)

import yfinance as yf
from datetime import date

for discovery in discoveries[:30]:
    symbol = discovery.stock.symbol
    discovery_price = float(discovery.price) if discovery.price else None
    explanation = discovery.explanation or ""
    
    # Parse support/resistance from explanation
    # Format: "Price $X.XX near support $Y.YY (Z.Z% position). Resistance: $A.AA (+B.B% upside potential)."
    support_match = re.search(r'support \$([\d.]+)', explanation)
    resistance_match = re.search(r'Resistance: \$([\d.]+)', explanation)
    position_match = re.search(r'\(([\d.]+)% position\)', explanation)
    upside_match = re.search(r'\+([\d.]+)% upside', explanation)
    
    support = float(support_match.group(1)) if support_match else None
    resistance = float(resistance_match.group(1)) if resistance_match else None
    position_pct = float(position_match.group(1)) if position_match else None
    upside_pct = float(upside_match.group(1)) if upside_match else None
    
    # Get EOD performance
    eod_change = None
    buy_trades = Trade.objects.filter(stock=discovery.stock, action='BUY', sa=discovery.sa).first()
    if buy_trades:
        try:
            ticker = yf.Ticker(symbol)
            trade_date = buy_trades.created.date()
            hist = ticker.history(start=trade_date.isoformat(), end=(trade_date + timedelta(days=1)).isoformat(), interval='1d')
            if not hist.empty:
                eod_price = float(hist['Close'].iloc[0])
                purchase_price = float(buy_trades.price)
                eod_change = ((eod_price - purchase_price) / purchase_price) * 100
        except:
            pass
    
    support_str = f"${support:.2f}" if support else "N/A"
    resistance_str = f"${resistance:.2f}" if resistance else "N/A"
    position_str = f"{position_pct:.1f}%" if position_pct else "N/A"
    upside_str = f"+{upside_pct:.1f}%" if upside_pct else "N/A"
    eod_str = f"{eod_change:+.2f}%" if eod_change is not None else "N/A"
    
    print(f"{symbol:<8} ${discovery_price:<7.2f} {support_str:<10} {resistance_str:<12} {position_str:<10} {upside_str:<10} {eod_str:<12}")

print('-' * 120)
print('\nNote: Position% should be < 20% for "near support" buys')
print('      Higher Position% = further from support = worse entry price')


