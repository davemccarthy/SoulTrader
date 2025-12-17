#!/usr/bin/env python
"""
Analyze unsuccessful discoveries from news_flash method to identify common patterns
in article titles that can be used to filter articles before sending to Gemini.

This script:
1. Finds discoveries from news_flash (StockStory and Polygon advisors)
2. Identifies unsuccessful ones (discoveries that led to losses)
3. Extracts article titles from discovery explanations
4. Analyzes common keywords/patterns in unsuccessful article titles
"""

import os
import sys
import django
from collections import Counter
import re
from decimal import Decimal

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import Discovery, Trade, Stock, Advisor, Holding
from django.db.models import Q, F, Sum, Count
from django.utils import timezone

# Common stop words to exclude from keyword analysis
STOP_WORDS = {
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 
    'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 
    'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 
    'did', 'let', 'put', 'say', 'she', 'too', 'use', 'why', 'with', 'from'
}

def extract_article_title(explanation):
    """
    Extract article title from discovery explanation.
    Format: "... | Article: {title} | {url} | ..."
    """
    if not explanation:
        return None
    
    # Look for "Article: " pattern
    match = re.search(r'Article:\s*([^|]+)', explanation)
    if match:
        return match.group(1).strip()
    
    return None

def get_unsuccessful_discoveries():
    """
    Find discoveries that led to losses.
    A discovery is unsuccessful if:
    - It led to a BUY trade
    - The stock was later sold at a loss (sell price < buy price)
    """
    unsuccessful = []
    
    # Get advisors that use news_flash (StockStory and Polygon)
    news_flash_advisors = Advisor.objects.filter(
        name__in=['StockStory', 'Polygon.io']
    )
    
    if not news_flash_advisors.exists():
        print("Warning: No news_flash advisors found (StockStory or Polygon.io)")
        return []
    
    advisor_names = list(news_flash_advisors.values_list('name', flat=True))
    print(f"Analyzing discoveries from: {', '.join(advisor_names)}")
    
    # Get all discoveries from these advisors
    discoveries = Discovery.objects.filter(
        advisor__in=news_flash_advisors
    ).select_related('stock', 'advisor', 'sa').order_by('-created')
    
    print(f"Found {discoveries.count()} total discoveries from news_flash advisors")
    
    # Show breakdown by advisor
    for advisor in news_flash_advisors:
        count = discoveries.filter(advisor=advisor).count()
        print(f"  - {advisor.name}: {count} discoveries")
    
    for discovery in discoveries:
        # Find BUY trades for this stock that happened after discovery
        buy_trades = Trade.objects.filter(
            stock=discovery.stock,
            action='BUY',
            sa__started__gte=discovery.created
        ).order_by('created')
        
        if not buy_trades.exists():
            # Discovery didn't lead to a buy - could be considered unsuccessful
            # but we'll focus on ones that were bought and lost money
            continue
        
        # Check if there was a SELL trade at a loss
        for buy_trade in buy_trades:
            # Find corresponding SELL trades
            sell_trades = Trade.objects.filter(
                stock=discovery.stock,
                action='SELL',
                created__gt=buy_trade.created,
                user=buy_trade.user
            ).order_by('created')
            
            for sell_trade in sell_trades:
                # Calculate profit/loss
                buy_price = buy_trade.price
                sell_price = sell_trade.price
                
                if sell_price < buy_price:
                    # This is a loss
                    loss_pct = ((sell_price - buy_price) / buy_price) * 100
                    unsuccessful.append({
                        'discovery': discovery,
                        'buy_trade': buy_trade,
                        'sell_trade': sell_trade,
                        'loss_pct': loss_pct,
                        'article_title': extract_article_title(discovery.explanation)
                    })
                    break  # Only count first sell for each buy
    
    return unsuccessful

def analyze_keywords(unsuccessful_discoveries):
    """
    Analyze common keywords in unsuccessful article titles.
    """
    titles = [d['article_title'] for d in unsuccessful_discoveries if d['article_title']]
    
    if not titles:
        print("No article titles found in unsuccessful discoveries")
        return
    
    print(f"\nAnalyzing {len(titles)} unsuccessful discoveries with article titles")
    
    # Extract all words (case-insensitive)
    all_words = []
    for title in titles:
        # Split on common delimiters and filter out short words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        all_words.extend(words)
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Also look for common phrases (2-3 word combinations)
    phrases = []
    for title in titles:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        # 2-word phrases
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
        # 3-word phrases
        for i in range(len(words) - 2):
            phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    
    phrase_counts = Counter(phrases)
    
    print("\n" + "="*80)
    print("TOP KEYWORDS IN UNSUCCESSFUL ARTICLE TITLES")
    print("="*80)
    print("\nMost common single words (excluding common stop words):")
    
    for word, count in word_counts.most_common(30):
        if word not in STOP_WORDS:
            print(f"  {word:20s} : {count:3d} occurrences")
    
    print("\nMost common 2-word phrases:")
    for phrase, count in phrase_counts.most_common(20):
        if count >= 2:  # Only show phrases that appear at least twice
            print(f"  {phrase:40s} : {count:3d} occurrences")
    
    return word_counts, phrase_counts

def show_examples(unsuccessful_discoveries, top_keywords=None):
    """
    Show example unsuccessful discoveries with their titles and loss percentages.
    """
    print("\n" + "="*80)
    print("EXAMPLE UNSUCCESSFUL DISCOVERIES")
    print("="*80)
    
    # Sort by loss percentage (worst first)
    sorted_discoveries = sorted(unsuccessful_discoveries, key=lambda x: x['loss_pct'])
    
    print(f"\nShowing top 20 worst losses:\n")
    for i, disc in enumerate(sorted_discoveries[:20], 1):
        title = disc['article_title'] or "No title found"
        print(f"{i:2d}. Loss: {disc['loss_pct']:6.2f}% | {disc['discovery'].stock.symbol:6s} | {title[:70]}")
        print(f"    Discovery: {disc['discovery'].advisor.name} | {disc['discovery'].created.strftime('%Y-%m-%d %H:%M')}")
        print(f"    Buy: ${disc['buy_trade'].price:.2f} | Sell: ${disc['sell_trade'].price:.2f}")
        print()

def main():
    print("Analyzing unsuccessful discoveries from news_flash method...")
    print("="*80)
    
    # Get unsuccessful discoveries
    unsuccessful = get_unsuccessful_discoveries()
    
    if not unsuccessful:
        print("\nNo unsuccessful discoveries found (discoveries that led to losses)")
        print("This could mean:")
        print("  - All discoveries were profitable")
        print("  - No trades have been executed yet")
        print("  - Discoveries haven't been sold yet")
        return
    
    print(f"\nFound {len(unsuccessful)} unsuccessful discoveries (led to losses)")
    
    # Analyze keywords
    word_counts, phrase_counts = analyze_keywords(unsuccessful)
    
    # Show examples
    show_examples(unsuccessful, word_counts)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    avg_loss = sum(d['loss_pct'] for d in unsuccessful) / len(unsuccessful)
    worst_loss = min(d['loss_pct'] for d in unsuccessful)
    best_loss = max(d['loss_pct'] for d in unsuccessful)  # Still a loss, but smallest
    
    print(f"Total unsuccessful discoveries: {len(unsuccessful)}")
    print(f"Average loss: {avg_loss:.2f}%")
    print(f"Worst loss: {worst_loss:.2f}%")
    print(f"Best (smallest) loss: {best_loss:.2f}%")
    
    # Count by advisor (with article title examples)
    advisor_counts = Counter(d['discovery'].advisor.name for d in unsuccessful)
    print(f"\nUnsuccessful discoveries by advisor:")
    for advisor, count in advisor_counts.items():
        print(f"  {advisor:20s}: {count:3d} unsuccessful discoveries")
        
        # Show a few example titles from this advisor
        advisor_examples = [d for d in unsuccessful if d['discovery'].advisor.name == advisor and d['article_title']]
        if advisor_examples:
            print(f"    Example titles:")
            for ex in advisor_examples[:3]:
                title = ex['article_title'][:70] if ex['article_title'] else "No title"
                print(f"      - {title}")
    
    # Recommendations for filtering
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR FILTERING")
    print("="*80)
    print("\nBased on the analysis above, consider filtering articles with these keywords:")
    print("(Add these to the existing filters in story.py and polygon.py)")
    
    if word_counts:
        top_filter_words = [word for word, count in word_counts.most_common(15) 
                           if word not in STOP_WORDS and count >= 3]
        print(f"\nTop words to consider filtering: {', '.join(top_filter_words[:10])}")
    
    if phrase_counts:
        top_phrases = [phrase for phrase, count in phrase_counts.most_common(10) if count >= 2]
        print(f"\nTop phrases to consider filtering: {', '.join(top_phrases[:5])}")

if __name__ == '__main__':
    main()
















































