#!/usr/bin/env python
"""
Analyze Flux advisor performance by session to verify correlation with:
1. Number of discoveries per session
2. High volume trading days (using SPY volume as proxy)

Usage:
    python analyze_flux_by_session.py
    python analyze_flux_by_session.py --days 60
"""
import os
import sys
import django

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

import yfinance as yf
from django.utils import timezone
from datetime import timedelta
from collections import defaultdict

from core.models import Discovery, Advisor, SmartAnalysis, Trade

def get_end_of_day_price(symbol, date_obj):
    """Get price at end of trading day (4:00 PM ET) for a given date."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=date_obj.isoformat(), end=(date_obj + timedelta(days=1)).isoformat(), interval='1d')
        if not hist.empty and 'Close' in hist.columns:
            return float(hist['Close'].iloc[0])
        hist = ticker.history(start=(date_obj + timedelta(days=1)).isoformat(), end=(date_obj + timedelta(days=2)).isoformat(), interval='1d')
        if not hist.empty and 'Close' in hist.columns:
            return float(hist['Close'].iloc[0])
        return None
    except:
        return None

def get_market_volume(date_obj):
    """Get SPY volume as proxy for market volume."""
    try:
        ticker = yf.Ticker('SPY')
        hist = ticker.history(start=date_obj.isoformat(), end=(date_obj + timedelta(days=1)).isoformat(), interval='1d')
        if not hist.empty and 'Volume' in hist.columns:
            return float(hist['Volume'].iloc[0])
        return None
    except:
        return None

def analyze_flux_by_session(days=60):
    """Analyze Flux discoveries grouped by session."""
    
    flux_advisor = Advisor.objects.get(name='Flux')
    cutoff = timezone.now() - timedelta(days=days)
    sessions = SmartAnalysis.objects.filter(started__gte=cutoff).order_by('started')
    
    discoveries = Discovery.objects.filter(
        advisor=flux_advisor,
        sa__in=sessions
    ).select_related('stock', 'sa').order_by('sa__started')
    
    print(f'\nAnalyzing Flux discoveries from last {days} days')
    print(f'Found {sessions.count()} sessions')
    print(f'Found {discoveries.count()} Flux discoveries\n')
    
    # Group by session
    session_data = defaultdict(lambda: {'discoveries': [], 'session': None, 'date': None})
    
    for discovery in discoveries:
        sa = discovery.sa
        session_data[sa.id]['session'] = sa
        session_data[sa.id]['discoveries'].append(discovery)
        session_data[sa.id]['date'] = sa.started.date()
    
    print('Calculating performance at end of trading day (4:00 PM ET)...\n')
    
    session_performance = []
    
    for session_id, data in session_data.items():
        session = data['session']
        session_date = data['date']
        discovery_count = len(data['discoveries'])
        
        market_volume = get_market_volume(session_date)
        
        gains = []
        losses = []
        total_evaluated = 0
        
        # Get BUY trades for stocks discovered in this session
        discovered_stocks = [d.stock for d in data['discoveries']]
        buy_trades = Trade.objects.filter(
            sa=session,
            stock__in=discovered_stocks,
            action='BUY'
        ).select_related('stock')
        
        for trade in buy_trades:
            symbol = trade.stock.symbol
            purchase_price = float(trade.price)
            trade_date = trade.created.date() if trade.created else session_date
            
            eod_price = get_end_of_day_price(symbol, trade_date)
            
            if eod_price:
                pct_change = ((eod_price - purchase_price) / purchase_price) * 100
                if pct_change > 0:
                    gains.append(pct_change)
                elif pct_change < 0:
                    losses.append(pct_change)
                total_evaluated += 1
        
        win_rate = len(gains) / total_evaluated * 100 if total_evaluated > 0 else 0
        avg_change = (sum(gains) + sum(losses)) / total_evaluated if total_evaluated > 0 else 0
        
        session_performance.append({
            'session_id': session.id,
            'date': session_date,
            'discovery_count': discovery_count,
            'gainers': len(gains),
            'losers': len(losses),
            'win_rate': win_rate,
            'avg_change': avg_change,
            'market_volume': market_volume
        })
    
    session_performance.sort(key=lambda x: x['discovery_count'], reverse=True)
    
    # Display results
    print('=' * 120)
    print('FLUX PERFORMANCE BY SESSION (End of Day Evaluation)')
    print('=' * 120)
    print(f"{'Session':<8} {'Date':<12} {'Count':<6} {'Win%':<7} {'G/L':<6} {'Avg%':<8} {'Mkt Vol':<12}")
    print('-' * 120)
    
    for perf in session_performance:
        market_vol_str = f"{perf['market_volume']/1e9:.2f}B" if perf['market_volume'] else "N/A"
        print(f"SA#{perf['session_id']:<7} {str(perf['date']):<12} "
              f"{perf['discovery_count']:<6} {perf['win_rate']:>6.1f}% "
              f"{perf['gainers']}/{perf['losers']:<5} {perf['avg_change']:>+7.2f}% {market_vol_str:<12}")
    
    sessions_with_data = [p for p in session_performance if p['gainers'] + p['losers'] > 0]
    
    print(f'\nNote: {len(session_performance) - len(sessions_with_data)} sessions have no price data (0/0), excluded from analysis')
    
    # Correlation analysis
    print('\n' + '=' * 120)
    print('CORRELATION ANALYSIS (Sessions with Valid Price Data Only)')
    print('=' * 120)
    
    # Group by discovery count
    count_ranges = {
        '1 discovery': [p for p in sessions_with_data if p['discovery_count'] == 1],
        '2-3 discoveries': [p for p in sessions_with_data if 2 <= p['discovery_count'] <= 3],
        '4-5 discoveries': [p for p in sessions_with_data if 4 <= p['discovery_count'] <= 5],
        '6+ discoveries': [p for p in sessions_with_data if p['discovery_count'] >= 6],
    }
    
    print('\nPerformance by Discovery Count:')
    for range_name, sessions in count_ranges.items():
        if not sessions:
            continue
        avg_win_rate = sum(s['win_rate'] for s in sessions) / len(sessions)
        avg_change = sum(s['avg_change'] for s in sessions) / len(sessions)
        total_gainers = sum(s['gainers'] for s in sessions)
        total_losers = sum(s['losers'] for s in sessions)
        print(f'\n{range_name} ({len(sessions)} sessions, {total_gainers + total_losers} total stocks):')
        print(f'  Average Win Rate: {avg_win_rate:.1f}%')
        print(f'  Average % Change: {avg_change:+.2f}%')
        print(f'  Total Gainers/Losers: {total_gainers}/{total_losers}')
    
    # Analyze by market volume
    sessions_with_volume = [p for p in sessions_with_data if p['market_volume']]
    if sessions_with_volume:
        median_volume = sorted(sessions_with_volume, key=lambda x: x['market_volume'])[len(sessions_with_volume)//2]['market_volume']
        high_volume = [p for p in sessions_with_volume if p['market_volume'] >= median_volume]
        low_volume = [p for p in sessions_with_volume if p['market_volume'] < median_volume]
        
        print(f'\nPerformance by Market Volume (SPY median: {median_volume/1e9:.2f}B):')
        
        if high_volume:
            total_gainers = sum(s['gainers'] for s in high_volume)
            total_losers = sum(s['losers'] for s in high_volume)
            print(f'\nHigh Volume ({len(high_volume)} sessions, {total_gainers + total_losers} stocks):')
            print(f'  Average Win Rate: {sum(s["win_rate"] for s in high_volume) / len(high_volume):.1f}%')
            print(f'  Average % Change: {sum(s["avg_change"] for s in high_volume) / len(high_volume):+.2f}%')
            print(f'  Total Gainers/Losers: {total_gainers}/{total_losers}')
        
        if low_volume:
            total_gainers = sum(s['gainers'] for s in low_volume)
            total_losers = sum(s['losers'] for s in low_volume)
            print(f'\nLow Volume ({len(low_volume)} sessions, {total_gainers + total_losers} stocks):')
            print(f'  Average Win Rate: {sum(s["win_rate"] for s in low_volume) / len(low_volume):.1f}%')
            print(f'  Average % Change: {sum(s["avg_change"] for s in low_volume) / len(low_volume):+.2f}%')
            print(f'  Total Gainers/Losers: {total_gainers}/{total_losers}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=60, help='Number of days to analyze')
    args = parser.parse_args()
    analyze_flux_by_session(args.days)


