#!/usr/bin/env python
"""
Review recommendation records for a specific SmartAnalysis session.
Shows what advisors recommended and why stocks didn't pass consensus.
"""

import os
import sys
import django

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import SmartAnalysis, Recommendation, Stock, Discovery, Consensus, Advisor
from decimal import Decimal

def review_sa_recommendations(sa_id):
    """Review all recommendations for a SmartAnalysis session."""
    
    try:
        sa = SmartAnalysis.objects.get(id=sa_id)
    except SmartAnalysis.DoesNotExist:
        print(f"SmartAnalysis session {sa_id} not found")
        return
    
    print("="*100)
    print(f"RECOMMENDATION REVIEW - SmartAnalysis Session #{sa_id}")
    print("="*100)
    print(f"Started: {sa.started}")
    print()
    
    # Get all discoveries for this session
    discoveries = Discovery.objects.filter(sa=sa).select_related('stock', 'advisor')
    print(f"📊 DISCOVERIES: {discoveries.count()} stocks discovered")
    
    # Get all recommendations for this session
    recommendations = Recommendation.objects.filter(sa=sa).select_related('stock', 'advisor').order_by('stock__symbol', 'advisor__name')
    
    if not recommendations.exists():
        print("\n❌ No recommendations found for this session")
        return
    
    print(f"📊 RECOMMENDATIONS: {recommendations.count()} total recommendations")
    print()
    
    # Group by stock
    stocks_with_recs = {}
    for rec in recommendations:
        symbol = rec.stock.symbol
        if symbol not in stocks_with_recs:
            stocks_with_recs[symbol] = {
                'stock': rec.stock,
                'recommendations': [],
                'consensus': None
            }
        stocks_with_recs[symbol]['recommendations'].append(rec)
    
    # Get consensus for each stock
    for symbol, data in stocks_with_recs.items():
        consensus = Consensus.objects.filter(sa=sa, stock=data['stock']).first()
        data['consensus'] = consensus
    
    # Display results
    print("="*100)
    print("STOCK-BY-STOCK BREAKDOWN")
    print("="*100)
    
    for symbol in sorted(stocks_with_recs.keys()):
        data = stocks_with_recs[symbol]
        stock = data['stock']
        recs = data['recommendations']
        consensus = data['consensus']
        
        print(f"\n{'='*100}")
        print(f"📈 {symbol} - {stock.company}")
        print(f"{'='*100}")
        print(f"Current Price: ${stock.price}")

        trend = stock.calc_trend()

        if trend is not None:
            print(f"Trend: {trend:+.2f}")
        
        # Show discovery info
        discovery = discoveries.filter(stock=stock).first()
        if discovery:
            print(f"Discovered by: {discovery.advisor.name}")
            print(f"Discovery reason: {discovery.explanation}")
        
        print(f"\n📊 ADVISOR RECOMMENDATIONS ({len(recs)} advisors):")
        print(f"{'Advisor':<25} {'Confidence':<12} {'Explanation'}")
        print("-" * 100)
        
        total_confidence = Decimal('0')
        for rec in sorted(recs, key=lambda x: x.confidence, reverse=True):
            advisor_name = rec.advisor.name[:23]
            confidence = rec.confidence
            explanation = rec.explanation[:65] + '...' if len(rec.explanation) > 68 else rec.explanation
            print(f"{advisor_name:<25} {confidence:>10.2f}  {explanation}")
            total_confidence += min(confidence, Decimal('1.0'))  # Cap at 1.0
        
        avg_confidence = total_confidence / len(recs) if recs else Decimal('0')
        print(f"\n  Average Confidence: {avg_confidence:.2f}")
        
        # Show consensus
        if consensus:
            print(f"\n✅ CONSENSUS:")
            print(f"  Recommendations: {consensus.recommendations}")
            print(f"  Total Confidence: {consensus.tot_confidence:.2f}")
            print(f"  Average Confidence: {consensus.avg_confidence:.2f}")
            
            # Check if it would pass different risk profiles
            print(f"\n  Risk Profile Thresholds:")
            from core.models import Profile
            for risk_level, thresholds in Profile.RISK.items():
                high_threshold = thresholds['confidence_high']
                passed = consensus.avg_confidence >= high_threshold
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"    {risk_level:<15}: {status} (needs {high_threshold:.2f}, got {consensus.avg_confidence:.2f})")
        else:
            print(f"\n❌ NO CONSENSUS BUILT")
            print(f"   (This usually means not enough advisors provided recommendations)")
    
    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print("="*100)
    
    total_stocks = len(stocks_with_recs)
    stocks_with_consensus = len([s for s in stocks_with_recs.values() if s['consensus']])
    stocks_without_consensus = total_stocks - stocks_with_consensus
    
    print(f"Total stocks analyzed: {total_stocks}")
    print(f"Stocks with consensus: {stocks_with_consensus}")
    print(f"Stocks without consensus: {stocks_without_consensus}")
    
    if stocks_with_consensus > 0:
        consensus_values = [s['consensus'].avg_confidence for s in stocks_with_recs.values() if s['consensus']]
        avg_consensus = sum(consensus_values) / len(consensus_values)
        print(f"Average consensus confidence: {avg_consensus:.2f}")
        print(f"Highest consensus: {max(consensus_values):.2f}")
        print(f"Lowest consensus: {min(consensus_values):.2f}")
    
    # Count advisors
    advisors_used = set(rec.advisor.name for rec in recommendations)
    print(f"\nAdvisors that provided recommendations: {len(advisors_used)}")
    for advisor_name in sorted(advisors_used):
        count = recommendations.filter(advisor__name=advisor_name).count()
        print(f"  - {advisor_name}: {count} recommendations")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python review_recommendations.py <sa_id>")
        print("Example: python review_recommendations.py 283")
        sys.exit(1)
    
    try:
        sa_id = int(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid SA ID (must be a number)")
        sys.exit(1)
    
    review_sa_recommendations(sa_id)






















