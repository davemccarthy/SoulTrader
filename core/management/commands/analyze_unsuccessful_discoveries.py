"""
Django management command to analyze unsuccessful discoveries from news_flash method.

Usage:
    python manage.py analyze_unsuccessful_discoveries
    python manage.py analyze_unsuccessful_discoveries --include-no-buy
"""

from django.core.management.base import BaseCommand
from django.db.models import Q, F
from collections import Counter
import re
from decimal import Decimal

from core.models import Discovery, Trade, Stock, Advisor, Holding


class Command(BaseCommand):
    help = 'Analyze unsuccessful discoveries to find common patterns in article titles'

    def add_arguments(self, parser):
        parser.add_argument(
            '--include-no-buy',
            action='store_true',
            help='Include discoveries that never led to a BUY trade (wasted Gemini calls)',
        )
        parser.add_argument(
            '--min-loss-pct',
            type=float,
            default=0.0,
            help='Minimum loss percentage to consider (default: 0.0)',
        )

    def extract_article_title(self, explanation):
        """Extract article title from discovery explanation."""
        if not explanation:
            return None
        
        # Look for "Article: " pattern
        match = re.search(r'Article:\s*([^|]+)', explanation)
        if match:
            return match.group(1).strip()
        
        return None

    def get_unsuccessful_discoveries(self, include_no_buy=False, min_loss_pct=0.0):
        """
        Find discoveries that led to losses or never led to buys.
        """
        unsuccessful = []
        
        # Get advisors that use news_flash (StockStory and Polygon)
        news_flash_advisors = Advisor.objects.filter(
            name__in=['StockStory', 'Polygon.io']
        )
        
        if not news_flash_advisors.exists():
            self.stdout.write(self.style.WARNING("No news_flash advisors found"))
            return []
        
        # Get all discoveries from these advisors
        discoveries = Discovery.objects.filter(
            advisor__in=news_flash_advisors
        ).select_related('stock', 'advisor', 'sa').order_by('-created')
        
        self.stdout.write(f"Found {discoveries.count()} total discoveries from news_flash advisors")
        
        for discovery in discoveries:
            # Find BUY trades for this stock that happened after discovery
            buy_trades = Trade.objects.filter(
                stock=discovery.stock,
                action='BUY',
                sa__started__gte=discovery.created
            ).order_by('created')
            
            if not buy_trades.exists():
                if include_no_buy:
                    # Discovery didn't lead to a buy - wasted Gemini call
                    unsuccessful.append({
                        'discovery': discovery,
                        'buy_trade': None,
                        'sell_trade': None,
                        'loss_pct': None,
                        'article_title': self.extract_article_title(discovery.explanation),
                        'reason': 'no_buy'
                    })
                continue
            
            # Check if there was a SELL trade at a loss
            found_loss = False
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
                        if loss_pct <= min_loss_pct:  # Note: loss_pct is negative, so <= means more negative
                            unsuccessful.append({
                                'discovery': discovery,
                                'buy_trade': buy_trade,
                                'sell_trade': sell_trade,
                                'loss_pct': loss_pct,
                                'article_title': self.extract_article_title(discovery.explanation),
                                'reason': 'loss'
                            })
                            found_loss = True
                            break  # Only count first sell for each buy
            
            # If no loss found but we want to track discoveries that led to buys but haven't been sold yet
            # (we skip these for now as we can't determine success/failure)
        
        return unsuccessful

    def analyze_keywords(self, unsuccessful_discoveries):
        """Analyze common keywords in unsuccessful article titles."""
        titles = [d['article_title'] for d in unsuccessful_discoveries if d['article_title']]
        
        if not titles:
            self.stdout.write(self.style.WARNING("No article titles found in unsuccessful discoveries"))
            return None, None
        
        self.stdout.write(f"\nAnalyzing {len(titles)} unsuccessful discoveries with article titles")
        
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
        
        return word_counts, phrase_counts

    def show_examples(self, unsuccessful_discoveries, limit=20):
        """Show example unsuccessful discoveries."""
        self.stdout.write("\n" + "="*80)
        self.stdout.write("EXAMPLE UNSUCCESSFUL DISCOVERIES")
        self.stdout.write("="*80)
        
        # Separate by reason
        losses = [d for d in unsuccessful_discoveries if d['reason'] == 'loss']
        no_buys = [d for d in unsuccessful_discoveries if d['reason'] == 'no_buy']
        
        if losses:
            # Sort by loss percentage (worst first)
            sorted_losses = sorted(losses, key=lambda x: x['loss_pct'])
            
            self.stdout.write(f"\nTop {min(limit, len(sorted_losses))} worst losses:\n")
            for i, disc in enumerate(sorted_losses[:limit], 1):
                title = disc['article_title'] or "No title found"
                self.stdout.write(f"{i:2d}. Loss: {disc['loss_pct']:6.2f}% | {disc['discovery'].stock.symbol:6s} | {title[:70]}")
                self.stdout.write(f"    Discovery: {disc['discovery'].advisor.name} | {disc['discovery'].created.strftime('%Y-%m-%d %H:%M')}")
                self.stdout.write(f"    Buy: ${disc['buy_trade'].price:.2f} | Sell: ${disc['sell_trade'].price:.2f}")
                self.stdout.write("")
        
        if no_buys:
            self.stdout.write(f"\nDiscoveries that never led to BUY trades ({len(no_buys)} total):\n")
            for i, disc in enumerate(no_buys[:limit], 1):
                title = disc['article_title'] or "No title found"
                self.stdout.write(f"{i:2d}. {disc['discovery'].stock.symbol:6s} | {title[:70]}")
                self.stdout.write(f"    Discovery: {disc['discovery'].advisor.name} | {disc['discovery'].created.strftime('%Y-%m-%d %H:%M')}")
                self.stdout.write("")

    def handle(self, *args, **options):
        include_no_buy = options['include_no_buy']
        min_loss_pct = options['min_loss_pct']
        
        self.stdout.write("Analyzing unsuccessful discoveries from news_flash method...")
        self.stdout.write("="*80)
        
        # Get unsuccessful discoveries
        unsuccessful = self.get_unsuccessful_discoveries(include_no_buy=include_no_buy, min_loss_pct=min_loss_pct)
        
        if not unsuccessful:
            self.stdout.write(self.style.WARNING(
                "\nNo unsuccessful discoveries found (discoveries that led to losses)"
            ))
            self.stdout.write("This could mean:")
            self.stdout.write("  - All discoveries were profitable")
            self.stdout.write("  - No trades have been executed yet")
            self.stdout.write("  - Discoveries haven't been sold yet")
            return
        
        self.stdout.write(f"\nFound {len(unsuccessful)} unsuccessful discoveries")
        
        # Separate by type
        losses = [d for d in unsuccessful if d['reason'] == 'loss']
        no_buys = [d for d in unsuccessful if d['reason'] == 'no_buy']
        
        if losses:
            self.stdout.write(f"  - {len(losses)} discoveries that led to losses")
        if no_buys:
            self.stdout.write(f"  - {len(no_buys)} discoveries that never led to BUY trades")
        
        # Analyze keywords
        word_counts, phrase_counts = self.analyze_keywords(unsuccessful)
        
        if word_counts:
            # Show top keywords
            self.stdout.write("\n" + "="*80)
            self.stdout.write("TOP KEYWORDS IN UNSUCCESSFUL ARTICLE TITLES")
            self.stdout.write("="*80)
            
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 
                'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 
                'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 
                'did', 'let', 'put', 'say', 'she', 'too', 'use', 'why', 'with', 'from'
            }
            
            self.stdout.write("\nMost common single words (excluding stop words):")
            for word, count in word_counts.most_common(30):
                if word not in stop_words:
                    self.stdout.write(f"  {word:20s} : {count:3d} occurrences")
            
            if phrase_counts:
                self.stdout.write("\nMost common 2-3 word phrases:")
                for phrase, count in phrase_counts.most_common(20):
                    if count >= 2:  # Only show phrases that appear at least twice
                        self.stdout.write(f"  {phrase:40s} : {count:3d} occurrences")
        
        # Show examples
        self.show_examples(unsuccessful)
        
        # Summary statistics
        self.stdout.write("\n" + "="*80)
        self.stdout.write("SUMMARY STATISTICS")
        self.stdout.write("="*80)
        
        if losses:
            avg_loss = sum(d['loss_pct'] for d in losses) / len(losses)
            worst_loss = min(d['loss_pct'] for d in losses)
            best_loss = max(d['loss_pct'] for d in losses)  # Still a loss, but smallest
            
            self.stdout.write(f"\nLosses:")
            self.stdout.write(f"  Total: {len(losses)}")
            self.stdout.write(f"  Average loss: {avg_loss:.2f}%")
            self.stdout.write(f"  Worst loss: {worst_loss:.2f}%")
            self.stdout.write(f"  Best (smallest) loss: {best_loss:.2f}%")
        
        # Count by advisor
        advisor_counts = Counter(d['discovery'].advisor.name for d in unsuccessful)
        self.stdout.write(f"\nBy advisor:")
        for advisor, count in advisor_counts.items():
            self.stdout.write(f"  {advisor:20s}: {count:3d} unsuccessful discoveries")
        
        # Recommendations for filtering
        self.stdout.write("\n" + "="*80)
        self.stdout.write("RECOMMENDATIONS FOR FILTERING")
        self.stdout.write("="*80)
        self.stdout.write("\nBased on the analysis above, consider filtering articles with these keywords:")
        self.stdout.write("(Add these to the existing filters in story.py and polygon.py)")
        
        if word_counts:
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 
                'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 
                'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 
                'did', 'let', 'put', 'say', 'she', 'too', 'use', 'why', 'with', 'from'
            }
            top_filter_words = [word for word, count in word_counts.most_common(20) 
                               if word not in stop_words and count >= 3]
            if top_filter_words:
                self.stdout.write(f"\nTop words to consider filtering: {', '.join(top_filter_words[:10])}")
        
        if phrase_counts:
            top_phrases = [phrase for phrase, count in phrase_counts.most_common(10) if count >= 2]
            if top_phrases:
                self.stdout.write(f"\nTop phrases to consider filtering: {', '.join(top_phrases[:5])}")




































