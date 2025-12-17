#!/usr/bin/env python
"""Check if Flux buy prices are too high compared to discovery prices"""
import os
import sys
import django

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.utils import timezone
from datetime import timedelta
from core.models import Discovery, Advisor, Trade

flux_advisor = Advisor.objects.get(name='Flux')
cutoff = timezone.now() - timedelta(days=60)

discoveries = Discovery.objects.filter(
    advisor=flux_advisor,
    created__gte=cutoff
).select_related('stock', 'sa').order_by('-created')

print(f'Analyzing {discoveries.count()} Flux discoveries\n')

price_gaps = []

for discovery in discoveries:
    symbol = discovery.stock.symbol
    discovery_price = float(discovery.price) if discovery.price else None
    discovery_time = discovery.created
    
    if not discovery_price:
        continue
    
    # Find BUY trades for this stock in the same session or shortly after
    buy_trades = Trade.objects.filter(
        stock=discovery.stock,
        action='BUY',
        sa=discovery.sa  # Same session
    ).order_by('created')
    
    if not buy_trades.exists():
        continue
    
    for trade in buy_trades:
        purchase_price = float(trade.price)
        trade_time = trade.created
        
        price_gap_pct = ((purchase_price - discovery_price) / discovery_price) * 100
        time_gap_minutes = (trade_time - discovery_time).total_seconds() / 60
        
        price_gaps.append({
            'symbol': symbol,
            'discovery_price': discovery_price,
            'purchase_price': purchase_price,
            'gap_pct': price_gap_pct,
            'discovery_time': discovery_time,
            'trade_time': trade_time,
            'time_gap_minutes': time_gap_minutes
        })

if not price_gaps:
    print('No matching trades found')
else:
    print('=' * 100)
    print(f'PRICE GAP ANALYSIS: Discovery Price vs Purchase Price')
    print('=' * 100)
    print(f"{'Symbol':<8} {'Discovery':<10} {'Purchase':<10} {'Gap %':<10} {'Time Gap':<12} {'Discovery Time':<20}")
    print('-' * 100)
    
    for gap in sorted(price_gaps, key=lambda x: abs(x['gap_pct']), reverse=True)[:30]:
        time_str = f"{gap['time_gap_minutes']:.1f}m"
        discovery_time_str = gap['discovery_time'].strftime('%Y-%m-%d %H:%M')
        gap_color = '+' if gap['gap_pct'] > 0 else ''
        print(f"{gap['symbol']:<8} ${gap['discovery_price']:<9.2f} ${gap['purchase_price']:<9.2f} "
              f"{gap_color}{gap['gap_pct']:>7.2f}%  {time_str:<12} {discovery_time_str}")
    
    avg_gap = sum(g['gap_pct'] for g in price_gaps) / len(price_gaps)
    avg_time_gap = sum(g['time_gap_minutes'] for g in price_gaps) / len(price_gaps)
    positive_gaps = len([g for g in price_gaps if g['gap_pct'] > 0])
    negative_gaps = len([g for g in price_gaps if g['gap_pct'] < 0])
    
    print('-' * 100)
    print(f'\nSummary:')
    print(f'  Total trades analyzed: {len(price_gaps)}')
    print(f'  Average price gap: {avg_gap:+.2f}%')
    print(f'  Trades purchased HIGHER than discovery: {positive_gaps} ({positive_gaps/len(price_gaps)*100:.1f}%)')
    print(f'  Trades purchased LOWER than discovery: {negative_gaps} ({negative_gaps/len(price_gaps)*100:.1f}%)')
    print(f'  Average time gap: {avg_time_gap:.1f} minutes')
    
    if avg_gap > 0:
        print(f'\n⚠️  WARNING: Average purchase price is {avg_gap:.2f}% HIGHER than discovery price!')
        print(f'   This means we\'re paying more than expected when buying.')
    elif avg_gap < 0:
        print(f'\n✅ Average purchase price is {abs(avg_gap):.2f}% LOWER than discovery price.')
        print(f'   This is actually beneficial - getting better prices than expected.')


