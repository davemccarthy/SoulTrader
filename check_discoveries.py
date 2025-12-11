#!/usr/bin/env python
"""
Check discoveries from SA session and analyze unbought stocks.

Usage:
    python check_discoveries.py                          # Most recent SA session, all advisors
    python check_discoveries.py --advisor Intraday       # Filter by advisor
    python check_discoveries.py --sa-id 411              # Specific SA session
    python check_discoveries.py --user username          # Filter by user's risk profile
    python check_discoveries.py --advisor Intraday --show-eod  # Show EOD performance
"""
import os
import sys
import django
import argparse

sys.path.append('/Users/davidmccarthy/Development/CursorAI/Django/soultrader')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import (
    SmartAnalysis, Discovery, Health, Holding, Trade, Stock, 
    Advisor, Profile, User
)
from django.db.models import Q
from decimal import Decimal
import yfinance as yf
from datetime import datetime, timedelta


def get_eod_price(symbol, discovery_date):
    """Fetch EOD price for a stock on a given date."""
    try:
        # Fetch hourly data for that day to get EOD price (around 3:30-4:00 PM ET)
        ticker = yf.Ticker(symbol)
        hist = ticker.history(
            start=discovery_date, 
            end=discovery_date + timedelta(days=1), 
            interval='1h'
        )
        
        if not hist.empty:
            # Get prices around EOD (3:30 PM ET = 15:30)
            # Find the last hour before or at 4pm ET
            eod_price = None
            for idx in range(len(hist) - 1, -1, -1):
                row = hist.iloc[idx]
                hour = row.name.hour
                if hour >= 15:  # 3pm or later
                    eod_price = float(row['Close'])
                    break
            
            if eod_price:
                return eod_price
            
            # Fallback to last available price
            return float(hist.iloc[-1]['Close'])
    except Exception as e:
        pass
    
    return None


def get_min_health_for_user(user):
    """Get min_health threshold based on user's risk profile."""
    try:
        profile = Profile.objects.get(user=user)
        risk_settings = Profile.RISK.get(profile.risk, {})
        return Decimal(str(risk_settings.get("min_health", 30.0)))
    except Profile.DoesNotExist:
        return Decimal('30.0')  # Default


def main():
    parser = argparse.ArgumentParser(
        description='Analyze discoveries and unbought stocks from SA sessions'
    )
    parser.add_argument(
        '--sa-id', 
        type=int, 
        help='Specific SA session ID (default: most recent)'
    )
    parser.add_argument(
        '--advisor', 
        type=str, 
        help='Filter by advisor python_class (e.g., Intraday, Yahoo, Insider)'
    )
    parser.add_argument(
        '--user', 
        type=str, 
        help='Filter by username (uses user\'s risk profile for min_health)'
    )
    parser.add_argument(
        '--show-eod', 
        action='store_true',
        help='Show EOD performance for unbought stocks'
    )
    parser.add_argument(
        '--show-bought', 
        action='store_true',
        help='Show EOD performance for bought stocks (default: True)'
    )
    
    args = parser.parse_args()
    
    # Determine which SA session to analyze
    if args.sa_id:
        try:
            latest_sa = SmartAnalysis.objects.get(id=args.sa_id)
        except SmartAnalysis.DoesNotExist:
            print(f"❌ SA session {args.sa_id} not found")
            sys.exit(1)
    else:
        # Find most recent SA session, optionally filtered by advisor
        advisor_filter = None
        if args.advisor:
            advisor_filter = Advisor.objects.filter(python_class=args.advisor).first()
            if not advisor_filter:
                print(f"❌ Advisor '{args.advisor}' not found")
                sys.exit(1)
            # Find most recent SA session with discoveries from this advisor
            latest_discovery = Discovery.objects.filter(advisor=advisor_filter).order_by('-created').first()
            if latest_discovery:
                latest_sa = latest_discovery.sa
            else:
                print(f"❌ No discoveries found for advisor '{args.advisor}'")
                sys.exit(1)
        else:
            latest_sa = SmartAnalysis.objects.order_by('-started').first()
    
    if not latest_sa:
        print("❌ No SA sessions found")
        sys.exit(1)
    
    # Build discovery query
    discoveries_qs = Discovery.objects.filter(sa=latest_sa).select_related('stock', 'advisor').order_by('-created')
    
    if advisor_filter:
        discoveries_qs = discoveries_qs.filter(advisor=advisor_filter)
    
    discoveries = list(discoveries_qs)
    
    if not discoveries:
        print(f"❌ No discoveries found for SA session {latest_sa.id}")
        sys.exit(1)
    
    # Display header
    advisor_name = advisor_filter.name if advisor_filter else "All Advisors"
    print(f"\n=== SA Session {latest_sa.id} ({latest_sa.started}) ===")
    print(f"=== Advisor: {advisor_name} ===")
    print(f"=== Total Discoveries: {len(discoveries)} ===\n")
    
    # Get user context for min_health if provided
    min_health = None
    if args.user:
        try:
            user = User.objects.get(username=args.user)
            min_health = get_min_health_for_user(user)
            print(f"User: {user.username} (Risk Profile: {Profile.objects.get(user=user).risk})")
            print(f"Min Health Threshold: {min_health:.1f}\n")
        except User.DoesNotExist:
            print(f"⚠️  User '{args.user}' not found, using default min_health=30.0\n")
    
    if min_health is None:
        min_health = Decimal('30.0')  # Default
    
    # Check which ones were bought
    bought_symbols = set()
    all_trades = Trade.objects.filter(sa=latest_sa, action='BUY').values_list('stock__symbol', flat=True)
    bought_symbols.update(all_trades)
    
    # Also check current holdings
    for discovery in discoveries:
        holding = Holding.objects.filter(stock=discovery.stock).first()
        if holding:
            bought_symbols.add(discovery.stock.symbol)
    
    print("DISCOVERY ANALYSIS:")
    print("=" * 80)
    
    for discovery in discoveries:
        symbol = discovery.stock.symbol
        was_bought = symbol in bought_symbols
        
        # Get health check
        health = Health.objects.filter(stock=discovery.stock).order_by('-created').first()
        health_score = health.score if health else None
        
        status = "✅ BOUGHT" if was_bought else "❌ NOT BOUGHT"
        
        print(f"\n{status} | {symbol}")
        print(f"  Advisor: {discovery.advisor.name}")
        print(f"  Discovery price: ${discovery.price:.4f}" if discovery.price else "  Discovery price: N/A")
        print(f"  Health check: {health_score:.2f}" if health_score else "  Health check: ❌ MISSING")
        
        if health_score:
            passed = "✅ PASSED" if health_score >= min_health else f"❌ FAILED (need {min_health:.1f})"
            print(f"  Status: {passed}")
        else:
            print(f"  Status: ❌ NO HEALTH CHECK")
        
        # If not bought, show reason and optionally EOD performance
        if not was_bought:
            reason = 'Health check missing' if not health else f'Health score {health_score:.2f} < {min_health:.0f}'
            print(f"  Reason: {reason}")
            
            if args.show_eod and discovery.price:
                discovery_date = discovery.created.date()
                eod_price = get_eod_price(symbol, discovery_date)
                
                if eod_price:
                    pnl = ((eod_price - float(discovery.price)) / float(discovery.price)) * 100
                    print(f"  Would-be EOD price: ${eod_price:.4f}")
                    print(f"  Would-be P&L: {pnl:+.2f}%")
                else:
                    # Fallback to current price
                    try:
                        discovery.stock.refresh()
                        current_price = discovery.stock.price
                        if current_price > 0:
                            pnl = ((current_price - discovery.price) / discovery.price) * 100
                            print(f"  Current price: ${current_price:.4f}")
                            print(f"  Current P&L: {pnl:+.2f}%")
                    except:
                        print(f"  Could not fetch price data")
        
        # If bought, show actual EOD performance (default: always show)
        elif was_bought:
            # Check if there's a SELL trade from that day (EOD exit)
            eod_sell = Trade.objects.filter(
                stock=discovery.stock,
                action='SELL',
                created__date=discovery.created.date()
            ).first()
            
            if eod_sell and discovery.price:
                pnl = ((float(eod_sell.price) - float(discovery.price)) / float(discovery.price)) * 100
                print(f"  EOD sell price: ${eod_sell.price:.4f}")
                print(f"  Actual P&L: {pnl:+.2f}%")
    
    # Summary
    print("\n" + "=" * 80)
    print("\nSUMMARY:")
    bought_count = len([d for d in discoveries if d.stock.symbol in bought_symbols])
    not_bought_count = len(discoveries) - bought_count
    
    print(f"  Total discoveries: {len(discoveries)}")
    print(f"  Bought: {bought_count}")
    print(f"  Not bought: {not_bought_count}")
    
    if not_bought_count > 0:
        print("\n  Stocks not bought:")
        for discovery in discoveries:
            if discovery.stock.symbol not in bought_symbols:
                health = Health.objects.filter(stock=discovery.stock).order_by('-created').first()
                if health:
                    print(f"    - {discovery.stock.symbol}: health {health.score:.2f} (need {min_health:.1f})")
                else:
                    print(f"    - {discovery.stock.symbol}: no health check")
    
    print()


if __name__ == '__main__':
    main()
