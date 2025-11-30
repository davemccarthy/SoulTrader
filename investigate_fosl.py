#!/usr/bin/env python
"""
Investigate FOSL (Fossil) insider discovery and buy to understand what went wrong.

Usage:
    python investigate_fosl.py
"""

import os
import sys
import django

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import Stock, Discovery, Trade, Holding, Consensus, SmartAnalysis
from core.services.advisors.insider import check_net_selling, score_insider_purchase, scrape_openinsider_purchases, MAX_QTY, MAX_VALUE
from datetime import date, timedelta
import yfinance as yf

def investigate_fosl():
    """Investigate FOSL discovery, purchase, and performance."""
    
    print("=" * 80)
    print("FOSL (Fossil) Investigation")
    print("=" * 80)
    
    # 1. Check if FOSL exists in database
    try:
        stock = Stock.objects.get(symbol="FOSL")
        print(f"\n✓ FOSL found in database")
        print(f"  Company: {stock.company or 'N/A'}")
        print(f"  Current price: ${stock.price}")
        if hasattr(stock, 'refresh'):
            stock.refresh()
            print(f"  Refreshed price: ${stock.price}")
    except Stock.DoesNotExist:
        print("\n✗ FOSL not found in database")
        stock = None
    
    # 2. Find all discoveries of FOSL
    print("\n" + "=" * 80)
    print("DISCOVERIES")
    print("=" * 80)
    discoveries = Discovery.objects.filter(stock__symbol="FOSL").order_by('-sa__started')
    
    if not discoveries.exists():
        print("✗ No discoveries found for FOSL")
    else:
        print(f"Found {discoveries.count()} discovery(ies):\n")
        for d in discoveries:
            sa = d.sa
            print(f"Discovery ID: {d.id}")
            print(f"  SA Session: {sa.id} ({sa.started})")
            print(f"  Advisor: {d.advisor.name}")
            print(f"  Explanation: {d.explanation}")
            print(f"  Created: {d.created}")
            if hasattr(d, 'discovery_price'):
                print(f"  Discovery Price: ${d.discovery_price}")
            print()
    
    # 3. Find all trades (buys/sells) for FOSL
    print("=" * 80)
    print("TRADES")
    print("=" * 80)
    trades = Trade.objects.filter(stock__symbol="FOSL").order_by('-sa__started')
    
    if not trades.exists():
        print("✗ No trades found for FOSL")
    else:
        print(f"Found {trades.count()} trade(s):\n")
        for t in trades:
            print(f"Trade ID: {t.id}")
            print(f"  SA Session: {t.sa.id} ({t.sa.started})")
            print(f"  User: {t.user.username}")
            print(f"  Action: {t.action}")
            print(f"  Price: ${t.price}")
            print(f"  Shares: {t.shares}")
            print(f"  Explanation: {t.explanation}")
            if t.consensus:
                print(f"  Consensus Score: {t.consensus.avg_confidence}")
            print()
    
    # 4. Find current holdings
    print("=" * 80)
    print("CURRENT HOLDINGS")
    print("=" * 80)
    holdings = Holding.objects.filter(stock__symbol="FOSL")
    
    if not holdings.exists():
        print("✗ No current holdings of FOSL")
    else:
        print(f"Found {holdings.count()} holding(s):\n")
        for h in holdings:
            print(f"Holding ID: {h.id}")
            print(f"  User: {h.user.username}")
            print(f"  Shares: {h.shares}")
            print(f"  Avg Price: ${h.average_price}")
            if stock:
                current_value = h.shares * stock.price
                cost_basis = h.shares * h.average_price
                pnl = current_value - cost_basis
                pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
                print(f"  Cost Basis: ${cost_basis:.2f}")
                print(f"  Current Value: ${current_value:.2f}")
                print(f"  P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            print()
    
    # 5. Check OpenInsider for FOSL transactions around discovery time
    print("=" * 80)
    print("OPENINSIDER ANALYSIS")
    print("=" * 80)
    
    # Get the most recent discovery date
    if discoveries.exists():
        latest_discovery = discoveries.first()
        discovery_date = latest_discovery.sa.started.date()
        print(f"Latest discovery: {discovery_date}")
        
        # Check transactions from 2 weeks before to 1 week after discovery
        start_date = discovery_date - timedelta(days=14)
        end_date = discovery_date + timedelta(days=7)
        print(f"Checking transactions from {start_date} to {end_date}\n")
    else:
        # Default to last 30 days
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        print(f"No discovery found, checking last 30 days ({start_date} to {end_date})\n")
    
    # Check net selling
    has_net_selling, total_purchase_val, total_sale_val = check_net_selling("FOSL")
    print(f"Net Selling Check:")
    print(f"  Has Net Selling: {has_net_selling}")
    print(f"  Total Purchase Value: ${total_purchase_val:,.0f}")
    print(f"  Total Sale Value: ${total_sale_val:,.0f}")
    print()
    
    # Scrape recent purchases (this will show what the system would have seen)
    print("Recent purchases from OpenInsider main page:")
    purchases = scrape_openinsider_purchases(since_date=start_date)
    fosl_purchases = [p for p in purchases if p.get("ticker") == "FOSL"]
    
    if fosl_purchases:
        print(f"Found {len(fosl_purchases)} FOSL purchase(s) in recent data:\n")
        for i, p in enumerate(fosl_purchases, 1):
            print(f"Purchase {i}:")
            print(f"  Ticker: {p.get('ticker')}")
            print(f"  Company: {p.get('company', 'N/A')}")
            print(f"  Insider: {p.get('insider_name', 'N/A')}")
            print(f"  Title: {p.get('title', 'N/A')}")
            print(f"  Price: ${p.get('price', 0):.2f}")
            print(f"  Quantity: {p.get('quantity', 0):,}")
            print(f"  Value: ${p.get('value', 0):,.0f}")
            print(f"  Trade Date: {p.get('trade_date', 'N/A')}")
            print(f"  Filing Date: {p.get('filing_date', 'N/A')}")
            
            # Calculate score
            score = score_insider_purchase(p, MAX_QTY, MAX_VALUE, 1)
            print(f"  Calculated Score: {score:.3f}")
            print()
    else:
        print("✗ No FOSL purchases found in recent OpenInsider data")
        print("  (This could mean purchases were older or not on main page)")
    
    # 6. Price analysis
    print("=" * 80)
    print("PRICE ANALYSIS")
    print("=" * 80)
    
    if discoveries.exists() and trades.exists():
        latest_discovery = discoveries.first()
        latest_trade = trades.filter(action="BUY").first()
        
        if latest_trade:
            discovery_price = getattr(latest_discovery, 'discovery_price', None) or latest_trade.price
            buy_price = latest_trade.price
            current_price = stock.price if stock else None
            
            print(f"Discovery Price: ${discovery_price:.2f}")
            print(f"Buy Price: ${buy_price:.2f}")
            if current_price:
                print(f"Current Price: ${current_price:.2f}")
                
                discovery_to_buy = ((buy_price - discovery_price) / discovery_price * 100) if discovery_price > 0 else 0
                buy_to_current = ((current_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
                discovery_to_current = ((current_price - discovery_price) / discovery_price * 100) if discovery_price > 0 else 0
                
                print(f"\nPrice Changes:")
                print(f"  Discovery → Buy: {discovery_to_buy:+.2f}%")
                print(f"  Buy → Current: {buy_to_current:+.2f}%")
                print(f"  Discovery → Current: {discovery_to_current:+.2f}%")
    
    # 7. Consensus scores
    print("\n" + "=" * 80)
    print("CONSENSUS SCORES")
    print("=" * 80)
    if discoveries.exists():
        for d in discoveries:
            sa = d.sa
            consensus = Consensus.objects.filter(sa=sa, stock=d.stock).first()
            if consensus:
                print(f"SA {sa.id} ({sa.started}):")
                print(f"  Avg Confidence: {consensus.avg_confidence}")
                print(f"  Total Confidence: {consensus.tot_confidence}")
                print(f"  Recommendations: {consensus.recommendations}")
            else:
                print(f"SA {sa.id}: No consensus found")
    
    print("\n" + "=" * 80)
    print("Investigation Complete")
    print("=" * 80)

if __name__ == "__main__":
    investigate_fosl()

