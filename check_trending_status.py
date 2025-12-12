#!/usr/bin/env python3
"""
Check is_trending() status for all stocks in holdings.

Usage:
    python check_trending_status.py [username]
    python check_trending_status.py --all
"""

import os
import sys
import django
from decimal import Decimal

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.contrib.auth.models import User
from core.models import Stock, Holding

def check_trending_status(username=None):
    """Check is_trending() for stocks in holdings."""
    if username:
        try:
            user = User.objects.get(username=username)
            users = [user]
        except User.DoesNotExist:
            print(f"User '{username}' not found")
            return
    else:
        # Get all users with holdings
        users = User.objects.filter(holding__isnull=False).distinct()
    
    print("Checking is_trending() status for holdings...\n")
    print("=" * 100)
    
    total_holdings = 0
    trending_count = 0
    not_trending_count = 0
    unknown_count = 0
    
    for user in users:
        holdings = Holding.objects.filter(user=user)
        if not holdings.exists():
            continue
        
        print(f"\n{user.username}:")
        print("-" * 100)
        
        for holding in holdings:
            total_holdings += 1
            stock = holding.stock
            
            # Refresh price first
            stock.refresh()
            
            # Check trending status
            try:
                # First check if price is valid
                if not stock.price or stock.price == 0:
                    print(f"  {stock.symbol:6} ? UNKNOWN (no price)")
                    unknown_count += 1
                    continue
                
                # Get volume data to diagnose
                import yfinance as yf
                ticker = yf.Ticker(stock.symbol)
                hist = ticker.history(period="7d", interval="1d")
                
                if hist.empty:
                    print(f"  {stock.symbol:6} ? UNKNOWN (empty history)")
                    unknown_count += 1
                    continue
                
                if 'Volume' not in hist.columns:
                    print(f"  {stock.symbol:6} ? UNKNOWN (no Volume column)")
                    unknown_count += 1
                    continue
                
                if len(hist) < 2:
                    print(f"  {stock.symbol:6} ? UNKNOWN (insufficient data: {len(hist)} rows)")
                    unknown_count += 1
                    continue
                
                # Debug: Trace through is_trending logic manually
                lookback_days = 5
                volume_threshold_pct = 0.2
                min_volume = 100000
                
                # Check price
                if not stock.price or stock.price == 0:
                    print(f"  {stock.symbol:6} ? UNKNOWN (price={stock.price})")
                    unknown_count += 1
                    continue
                
                # Get history with same period as is_trending
                hist_check = ticker.history(period=f"{lookback_days + 2}d", interval="1d")
                
                if hist_check.empty or 'Volume' not in hist_check.columns or len(hist_check) < 2:
                    print(f"  {stock.symbol:6} ? UNKNOWN (hist_check: empty={hist_check.empty}, has_volume={'Volume' in hist_check.columns}, len={len(hist_check)})")
                    unknown_count += 1
                    continue
                
                current_volume = hist_check['Volume'].iloc[-1]
                available_days = min(lookback_days, len(hist_check) - 1)
                
                if available_days < 1:
                    print(f"  {stock.symbol:6} ? UNKNOWN (available_days={available_days}, len(hist)={len(hist_check)})")
                    unknown_count += 1
                    continue
                
                avg_volume = hist_check['Volume'].iloc[-(available_days + 1):-1].mean()
                
                if avg_volume <= 0:
                    print(f"  {stock.symbol:6} ? UNKNOWN (avg_volume={avg_volume})")
                    unknown_count += 1
                    continue
                
                # Now calculate trending status
                volume_threshold = avg_volume * volume_threshold_pct
                absolute_threshold = max(volume_threshold, min_volume)
                is_trending = bool(current_volume >= absolute_threshold)
                
                # Now check what the actual method returns
                try:
                    method_result = stock.is_trending()
                    if method_result != is_trending:
                        print(f"  {stock.symbol:6} ⚠ MISMATCH: manual={is_trending}, method={method_result}")
                    # Use method result if available, otherwise use manual calculation
                    if method_result is not None:
                        is_trending = method_result
                except Exception as e:
                    print(f"  {stock.symbol:6} ⚠ EXCEPTION in is_trending(): {e}")
                    import traceback
                    traceback.print_exc()
                
                # Determine status
                if is_trending == True:
                    status = "✓ TRENDING"
                    trending_count += 1
                elif is_trending == False:
                    status = "✗ NOT TRENDING"
                    not_trending_count += 1
                else:
                    status = "? UNKNOWN"
                    unknown_count += 1
                
                volume_pct = (current_volume / avg_volume * 100) if avg_volume > 0 else 0
                print(f"  {stock.symbol:6} {status:20} | "
                      f"Current: {current_volume:>12,.0f} | "
                      f"Avg: {avg_volume:>12,.0f} | "
                      f"{volume_pct:>6.1f}% of avg | "
                      f"Threshold: {absolute_threshold:>12,.0f}")
                    
            except Exception as e:
                print(f"  {stock.symbol:6} ERROR: {e}")
                import traceback
                traceback.print_exc()
                unknown_count += 1
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total holdings checked: {total_holdings}")
    print(f"  ✓ Trending:     {trending_count} ({trending_count/total_holdings*100:.1f}%)" if total_holdings > 0 else "  ✓ Trending:     0")
    print(f"  ✗ Not trending: {not_trending_count} ({not_trending_count/total_holdings*100:.1f}%)" if total_holdings > 0 else "  ✗ Not trending: 0")
    print(f"  ? Unknown:       {unknown_count} ({unknown_count/total_holdings*100:.1f}%)" if total_holdings > 0 else "  ? Unknown:      0")
    
    if not_trending_count > 0:
        print(f"\n⚠️  {not_trending_count} stock(s) would trigger NOT_TRENDING sell instruction")
    else:
        print("\n✓ All stocks are trending (or status unknown)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check is_trending() status for holdings')
    parser.add_argument(
        'username',
        nargs='?',
        type=str,
        help='Username to check (optional, checks all users if not provided)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Check all users (same as not providing username)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.all or not args.username:
            check_trending_status()
        else:
            check_trending_status(args.username)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

