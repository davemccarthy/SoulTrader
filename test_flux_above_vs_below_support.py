#!/usr/bin/env python
"""
Test theory: Do Flux buys above support perform better than buys below support?
"""
import os
import sys
import django

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

import re
import yfinance as yf
from django.utils import timezone
from datetime import timedelta
from core.models import Discovery, Advisor, Trade

flux_advisor = Advisor.objects.get(name='Flux')
cutoff = timezone.now() - timedelta(days=60)

discoveries = Discovery.objects.filter(
    advisor=flux_advisor,
    created__gte=cutoff
).select_related('stock', 'sa').order_by('-created')

print(f'Analyzing {discoveries.count()} Flux discoveries to test theory\n')

above_support_trades = []
below_support_trades = []

for discovery in discoveries:
    symbol = discovery.stock.symbol
    discovery_price = float(discovery.price) if discovery.price else None
    explanation = discovery.explanation or ""
    
    if not discovery_price:
        continue
    
    # Parse support from explanation
    support_match = re.search(r'support \$([\d.]+)', explanation)
    if not support_match:
        continue
    
    support_price = float(support_match.group(1))
    
    # Find BUY trades for this discovery
    buy_trades = Trade.objects.filter(
        stock=discovery.stock,
        action='BUY',
        sa=discovery.sa
    )
    
    for trade in buy_trades:
        purchase_price = float(trade.price)
        trade_date = trade.created.date()
        
        # Determine if bought above or below support
        is_above_support = purchase_price >= support_price
        support_gap_pct = ((purchase_price - support_price) / support_price) * 100
        
        # Get EOD price on trade date
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=trade_date.isoformat(), end=(trade_date + timedelta(days=1)).isoformat(), interval='1d')
            if hist.empty:
                continue
            eod_price = float(hist['Close'].iloc[0])
            eod_change_pct = ((eod_price - purchase_price) / purchase_price) * 100
        except:
            continue
        
        trade_data = {
            'symbol': symbol,
            'purchase_price': purchase_price,
            'support_price': support_price,
            'support_gap_pct': support_gap_pct,
            'eod_price': eod_price,
            'eod_change_pct': eod_change_pct,
            'is_gainer': eod_change_pct > 0,
            'trade_date': trade_date
        }
        
        if is_above_support:
            above_support_trades.append(trade_data)
        else:
            below_support_trades.append(trade_data)

# Calculate statistics
def calc_stats(trades, label):
    if not trades:
        print(f'\n{label}: No trades found')
        return
    
    total = len(trades)
    gainers = len([t for t in trades if t['is_gainer']])
    losers = total - gainers
    win_rate = (gainers / total * 100) if total > 0 else 0
    avg_change = sum(t['eod_change_pct'] for t in trades) / total
    avg_gap = sum(t['support_gap_pct'] for t in trades) / total
    
    print(f'\n{label}:')
    print(f'  Total trades: {total}')
    print(f'  Win rate: {win_rate:.1f}% ({gainers} gainers, {losers} losers)')
    print(f'  Average EOD change: {avg_change:+.2f}%')
    print(f'  Average gap from support: {avg_gap:+.2f}%')
    
    if gainers > 0:
        avg_gain = sum(t['eod_change_pct'] for t in trades if t['is_gainer']) / gainers
        print(f'  Average gain (winners): +{avg_gain:.2f}%')
    
    if losers > 0:
        avg_loss = sum(t['eod_change_pct'] for t in trades if not t['is_gainer']) / losers
        print(f'  Average loss (losers): {avg_loss:.2f}%')

print('=' * 100)
print('FLUX PERFORMANCE: ABOVE SUPPORT vs BELOW SUPPORT')
print('=' * 100)

calc_stats(above_support_trades, 'BOUGHT ABOVE SUPPORT (price >= support)')
calc_stats(below_support_trades, 'BOUGHT BELOW SUPPORT (price < support)')

if above_support_trades and below_support_trades:
    print('\n' + '=' * 100)
    print('COMPARISON:')
    print('=' * 100)
    
    above_win_rate = len([t for t in above_support_trades if t['is_gainer']]) / len(above_support_trades) * 100
    below_win_rate = len([t for t in below_support_trades if t['is_gainer']]) / len(below_support_trades) * 100
    above_avg = sum(t['eod_change_pct'] for t in above_support_trades) / len(above_support_trades)
    below_avg = sum(t['eod_change_pct'] for t in below_support_trades) / len(below_support_trades)
    
    win_rate_diff = above_win_rate - below_win_rate
    avg_diff = above_avg - below_avg
    
    print(f'\nWin Rate Difference: {win_rate_diff:+.1f} percentage points')
    if win_rate_diff > 5:
        print(f'  ✅ Above support performs SIGNIFICANTLY better ({above_win_rate:.1f}% vs {below_win_rate:.1f}%)')
    elif win_rate_diff < -5:
        print(f'  ❌ Below support performs better ({below_win_rate:.1f}% vs {above_win_rate:.1f}%)')
    else:
        print(f'  ➡️  Similar performance ({above_win_rate:.1f}% vs {below_win_rate:.1f}%)')
    
    print(f'\nAverage Return Difference: {avg_diff:+.2f} percentage points')
    if avg_diff > 1.0:
        print(f'  ✅ Above support returns are SIGNIFICANTLY better ({above_avg:+.2f}% vs {below_avg:+.2f}%)')
    elif avg_diff < -1.0:
        print(f'  ❌ Below support returns are better ({below_avg:+.2f}% vs {above_avg:+.2f}%)')
    else:
        print(f'  ➡️  Similar returns ({above_avg:+.2f}% vs {below_avg:+.2f}%)')
    
    print(f'\n📊 Recommendation:')
    if win_rate_diff > 5 and avg_diff > 1.0:
        print('   Consider adding filter: require price >= support before buying')
    elif win_rate_diff < -5 or avg_diff < -1.0:
        print('   Buying below support may actually be beneficial (counterintuitive!)')
    else:
        print('   No clear advantage to either approach based on this data')


