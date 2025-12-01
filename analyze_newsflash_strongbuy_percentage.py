"""
Analyze the percentage of STRONG_BUY recommendations from news_flash
calls made by Story (StockStory) and Polygon advisors.
"""

import os
import sys
import django

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import Discovery, Advisor

def analyze_newsflash_strongbuy():
    """Analyze STRONG_BUY percentage from news_flash recommendations"""
    
    # Get advisors that use news_flash (StockStory and Polygon)
    news_flash_advisors = Advisor.objects.filter(
        name__in=['StockStory', 'Polygon.io']
    )
    
    if not news_flash_advisors.exists():
        print("Warning: No news_flash advisors found (StockStory or Polygon.io)")
        return
    
    advisor_names = list(news_flash_advisors.values_list('name', flat=True))
    print(f"\nAnalyzing news_flash recommendations from: {', '.join(advisor_names)}")
    print("=" * 80)
    
    # Get all discoveries from these advisors
    discoveries = Discovery.objects.filter(
        advisor__in=news_flash_advisors
    ).select_related('stock', 'advisor', 'sa').order_by('-created')
    
    total_count = discoveries.count()
    print(f"\nTotal discoveries from news_flash advisors: {total_count}")
    
    # Breakdown by advisor
    print("\nBreakdown by advisor:")
    for advisor in news_flash_advisors:
        count = discoveries.filter(advisor=advisor).count()
        print(f"  - {advisor.name}: {count} discoveries")
    
    # Analyze recommendations from explanation field
    # News flash explanations contain: "recommended {RECOMMENDATION} from reading article"
    strong_buy_count = 0
    buy_count = 0
    dismiss_count = 0
    other_count = 0
    unknown_count = 0
    
    # Breakdown by advisor
    advisor_stats = {}
    for advisor in news_flash_advisors:
        advisor_stats[advisor.name] = {
            'total': 0,
            'STRONG_BUY': 0,
            'BUY': 0,
            'DISMISS': 0,
            'other': 0,
            'unknown': 0
        }
    
    print("\n" + "=" * 80)
    print("Analyzing recommendation types from explanation field...")
    
    for discovery in discoveries:
        advisor_name = discovery.advisor.name
        
        # Check explanation for recommendation type
        explanation = discovery.explanation.lower() if discovery.explanation else ""
        
        advisor_stats[advisor_name]['total'] += 1
        
        # Check for recommendation patterns in various formats:
        # - "recommended STRONG_BUY" 
        # - "Gemini: STRONG_BUY"
        # - "Newsflash: | STRONG_BUY"
        # - "STRONG_BUY |" (at start of explanation)
        if (explanation.startswith('strong_buy') or explanation.startswith('strong buy') or
            'recommended strong_buy' in explanation or 'recommended strong buy' in explanation or
            'gemini: strong_buy' in explanation or 'gemini: strong buy' in explanation or
            'newsflash: | strong_buy' in explanation or 'newsflash: | strong buy' in explanation):
            strong_buy_count += 1
            advisor_stats[advisor_name]['STRONG_BUY'] += 1
        elif (explanation.startswith('buy |') or
              'recommended buy' in explanation or 'gemini: buy' in explanation or
              'newsflash: | buy' in explanation):
            buy_count += 1
            advisor_stats[advisor_name]['BUY'] += 1
        elif ('recommended dismiss' in explanation or 'gemini: dismiss' in explanation or
              'newsflash: | dismiss' in explanation):
            dismiss_count += 1
            advisor_stats[advisor_name]['DISMISS'] += 1
        elif ('recommended' in explanation or 'gemini:' in explanation or 'newsflash:' in explanation):
            # Check for other recommendation types
            if ('recommended sell' in explanation or 'gemini: sell' in explanation or
                'newsflash: | sell' in explanation):
                other_count += 1
                advisor_stats[advisor_name]['other'] += 1
            elif ('recommended strong_sell' in explanation or 'recommended strong sell' in explanation or
                  'gemini: strong_sell' in explanation or 'gemini: strong sell' in explanation or
                  'newsflash: | strong_sell' in explanation or 'newsflash: | strong sell' in explanation):
                other_count += 1
                advisor_stats[advisor_name]['other'] += 1
            else:
                unknown_count += 1
                advisor_stats[advisor_name]['unknown'] += 1
        else:
            # No recommendation found in explanation
            unknown_count += 1
            advisor_stats[advisor_name]['unknown'] += 1
    
    # Calculate percentages
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    
    if total_count > 0:
        strong_buy_pct = (strong_buy_count / total_count) * 100
        buy_pct = (buy_count / total_count) * 100
        dismiss_pct = (dismiss_count / total_count) * 100
        other_pct = (other_count / total_count) * 100
        unknown_pct = (unknown_count / total_count) * 100
        
        print(f"\nTotal discoveries: {total_count}")
        print(f"\nSTRONG_BUY: {strong_buy_count} ({strong_buy_pct:.2f}%)")
        print(f"BUY:         {buy_count} ({buy_pct:.2f}%)")
        print(f"DISMISS:     {dismiss_count} ({dismiss_pct:.2f}%)")
        print(f"Other:       {other_count} ({other_pct:.2f}%)")
        print(f"Unknown:     {unknown_count} ({unknown_pct:.2f}%)")
        
        # Calculate BUY + STRONG_BUY percentage
        buy_or_strongbuy = buy_count + strong_buy_count
        buy_or_strongbuy_pct = (buy_or_strongbuy / total_count) * 100
        print(f"\nBUY + STRONG_BUY combined: {buy_or_strongbuy} ({buy_or_strongbuy_pct:.2f}%)")
        
        # Show percentage of STRONG_BUY out of all positive recommendations
        positive_recommendations = buy_count + strong_buy_count
        if positive_recommendations > 0:
            strongbuy_of_positive_pct = (strong_buy_count / positive_recommendations) * 100
            print(f"\nSTRONG_BUY as % of positive recommendations (BUY + STRONG_BUY): {strongbuy_of_positive_pct:.2f}%")
    
    # Per-advisor breakdown
    print("\n" + "=" * 80)
    print("PER-ADVISOR BREAKDOWN")
    print("=" * 80)
    
    for advisor_name, stats in advisor_stats.items():
        if stats['total'] > 0:
            print(f"\n{advisor_name}:")
            print(f"  Total: {stats['total']}")
            
            for rec_type in ['STRONG_BUY', 'BUY', 'DISMISS', 'other', 'unknown']:
                count = stats[rec_type]
                pct = (count / stats['total']) * 100
                print(f"  {rec_type:12} {count:4} ({pct:5.2f}%)")
            
            # STRONG_BUY percentage
            strongbuy_pct = (stats['STRONG_BUY'] / stats['total']) * 100
            print(f"  {'STRONG_BUY %':12} {strongbuy_pct:5.2f}%")
            
            # STRONG_BUY as % of positive recommendations
            positive = stats['BUY'] + stats['STRONG_BUY']
            if positive > 0:
                strongbuy_of_positive = (stats['STRONG_BUY'] / positive) * 100
                print(f"  {'STRONG_BUY of positive':12} {strongbuy_of_positive:5.2f}%")
    
    # Show sample unknown explanations to understand why they weren't categorized
    if unknown_count > 0:
        print("\n" + "=" * 80)
        print("SAMPLE UNKNOWN RECOMMENDATIONS (first 5):")
        print("=" * 80)
        unknown_samples = []
        for discovery in discoveries:
            explanation = discovery.explanation.lower() if discovery.explanation else ""
            # Skip if it has a recognizable pattern
            has_pattern = any([
                explanation.startswith('strong_buy') or explanation.startswith('strong buy'),
                explanation.startswith('buy |'),
                'recommended strong_buy' in explanation or 'recommended strong buy' in explanation,
                'gemini: strong_buy' in explanation or 'gemini: strong buy' in explanation,
                'newsflash: | strong_buy' in explanation or 'newsflash: | strong buy' in explanation,
                'recommended buy' in explanation or 'gemini: buy' in explanation,
                'newsflash: | buy' in explanation,
                'recommended dismiss' in explanation or 'gemini: dismiss' in explanation,
                'newsflash: | dismiss' in explanation,
            ])
            if not has_pattern and len(unknown_samples) < 5:
                full_explanation = discovery.explanation[:300] if discovery.explanation else "None"
                unknown_samples.append((discovery.advisor.name, full_explanation))
        for advisor_name, explanation in unknown_samples:
            print(f"\n{advisor_name}:")
            print(f"  {explanation}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    analyze_newsflash_strongbuy()

