#!/usr/bin/env python
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import Recommendation, Consensus, SmartAnalysis, Stock
from django.db.models import Avg, Count, Min, Max

# Get the latest SA session
latest_sa = SmartAnalysis.objects.order_by('-id').first()

if not latest_sa:
    print("No SmartAnalysis sessions found")
    exit(1)

print(f"\n=== Recommendations Analysis for SA #{latest_sa.id} ===\n")

# Get all recommendations for this SA
recommendations = Recommendation.objects.filter(sa=latest_sa).order_by('stock__symbol', 'advisor__name')

# Group by stock
stocks = {}
for rec in recommendations:
    symbol = rec.stock.symbol
    if symbol not in stocks:
        stocks[symbol] = []
    stocks[symbol].append({
        'advisor': rec.advisor.name,
        'confidence': float(rec.confidence),
        'explanation': rec.explanation[:80]
    })

# Analyze each stock
print(f"Total stocks analyzed: {len(stocks)}\n")
print("=" * 100)

for symbol, recs in sorted(stocks.items()):
    confidences = [r['confidence'] for r in recs]
    avg_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)
    max_conf = max(confidences)
    spread = max_conf - min_conf
    
    # Get consensus
    consensus = Consensus.objects.filter(sa=latest_sa, stock__symbol=symbol).first()
    consensus_avg = float(consensus.avg_confidence) if consensus else 0
    
    print(f"\n{symbol}:")
    print(f"  Consensus: {consensus_avg:.2f} | Avg: {avg_conf:.2f} | Range: {min_conf:.2f} - {max_conf:.2f} | Spread: {spread:.2f}")
    
    # Show disagreement if spread is high
    if spread > 0.3:
        print(f"  ⚠️  HIGH DISAGREEMENT (spread > 0.30)")
    
    for rec in sorted(recs, key=lambda x: x['confidence'], reverse=True):
        confidence_str = f"{rec['confidence']:.2f}"
        if rec['confidence'] == max_conf:
            confidence_str = f"⬆️  {confidence_str}"
        elif rec['confidence'] == min_conf:
            confidence_str = f"⬇️  {confidence_str}"
        
        print(f"    {rec['advisor']:20s}: {confidence_str:8s} | {rec['explanation']}")

# Summary statistics
print("\n" + "=" * 100)
print("\nSUMMARY STATISTICS:\n")

all_confidences = [float(r.confidence) for r in recommendations]
advisors = set([r.advisor.name for r in recommendations])

print(f"Total recommendations: {len(recommendations)}")
print(f"Stocks analyzed: {len(stocks)}")
print(f"Advisors contributing: {len(advisors)} ({', '.join(sorted(advisors))})")
print(f"Average confidence: {sum(all_confidences)/len(all_confidences):.2f}")
print(f"Confidence range: {min(all_confidences):.2f} - {max(all_confidences):.2f}")

# Find stocks with high disagreement
high_spread_stocks = []
for symbol, recs in stocks.items():
    confidences = [r['confidence'] for r in recs]
    spread = max(confidences) - min(confidences)
    if spread > 0.3:
        high_spread_stocks.append((symbol, spread, recs))

if high_spread_stocks:
    print(f"\n⚠️  STOCKS WITH HIGH DISAGREEMENT (spread > 0.30):")
    for symbol, spread, recs in sorted(high_spread_stocks, key=lambda x: x[1], reverse=True):
        print(f"  {symbol}: spread = {spread:.2f}")
        for rec in sorted(recs, key=lambda x: x['confidence'], reverse=True):
            print(f"    {rec['advisor']}: {rec['confidence']:.2f}")
else:
    print("\n✅ No high disagreement found (all spreads < 0.30)")

