"""
Analyze price performance of STRONG_BUY vs BUY recommendations from news_flash.
Compares price changes from discovery to current price.

Uses yfinance to fetch historical prices (same method as test_discovery.py)
since discovery.price field is relatively new and many older discoveries don't have it.
"""

import os
import sys
import django
from decimal import Decimal
import statistics
from datetime import timedelta
from typing import Dict, Optional, Tuple
from django.utils import timezone

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import Discovery, Advisor, Stock

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("This script requires yfinance and pandas. Install with: pip install yfinance pandas", file=sys.stderr)
    sys.exit(1)

class YfPriceFetcher:
    """Fetch historical and current prices from yfinance (based on test_discovery.py)"""
    
    def __init__(self):
        self._history_cache: Dict[Tuple[str, str], Optional[float]] = {}
        self._latest_cache: Dict[Tuple[str, str], Optional[float]] = {}
    
    def price_from(self, symbol: str, ref_datetime) -> Optional[float]:
        """Fetch the first available close on/after ref_datetime"""
        key = (symbol.upper(), ref_datetime.strftime("%Y-%m-%d"))
        if key not in self._history_cache:
            self._history_cache[key] = self._lookup_price(symbol, ref_datetime)
        return self._history_cache[key]
    
    def latest_price(self, symbol: str, as_of) -> Optional[float]:
        """Fetch latest price as of the given datetime"""
        use_realtime = as_of.date() >= timezone.now().date()
        cache_key = (symbol.upper(), "realtime" if use_realtime else as_of.strftime("%Y-%m-%d"))
        
        if cache_key not in self._latest_cache:
            if use_realtime:
                price = self._lookup_realtime_price(symbol)
                if price is None:
                    price = self._lookup_price(symbol, as_of, forward_only=True)
            else:
                price = self._lookup_price(symbol, as_of, forward_only=True)
            self._latest_cache[cache_key] = price
        
        return self._latest_cache[cache_key]
    
    @staticmethod
    def _lookup_realtime_price(symbol: str) -> Optional[float]:
        """Attempt to fetch the latest traded price"""
        try:
            ticker = yf.Ticker(symbol)
            fast_info = getattr(ticker, "fast_info", None)
            if fast_info:
                for field in ("lastPrice", "regularMarketPrice", "postMarketPrice"):
                    value = getattr(fast_info, field, None)
                    if value is not None:
                        return float(value)
            
            price = ticker.info.get("currentPrice") if hasattr(ticker, "info") else None
            if price is not None:
                return float(price)
        except Exception:
            return None
        return None
    
    @staticmethod
    def _lookup_price(symbol: str, ref_datetime, forward_only: bool = False) -> Optional[float]:
        """Fetch the first available close on/after ref_datetime"""
        start_date = ref_datetime.date() - timedelta(days=2 if not forward_only else 0)
        end_date = ref_datetime.date() + timedelta(days=5)
        try:
            hist = yf.download(
                symbol,
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                progress=False,
            )
        except Exception:
            return None
        
        if hist.empty:
            return None
        
        # Normalize index to naive datetime for comparison
        index_dates = []
        for ts in hist.index.to_pydatetime():
            if ts.tzinfo:
                index_dates.append(ts.replace(tzinfo=None))
            else:
                index_dates.append(ts)
        
        close_data = hist.get("Close")
        if close_data is None:
            return None
        if isinstance(close_data, pd.DataFrame):
            close_series = close_data.iloc[:, 0]
        else:
            close_series = close_data
        
        for idx, close in zip(index_dates, close_series.tolist()):
            if idx.date() >= ref_datetime.date():
                if isinstance(close, pd.Series):
                    close = close.iloc[0]
                return float(close)
        
        if forward_only:
            return None
        
        # Fall back to last available close before the reference date
        last_close = close_series.iloc[-1]
        if isinstance(last_close, pd.Series):
            last_close = last_close.iloc[0]
        return float(last_close)


def _discovery_timestamp(discovery) -> timezone.datetime:
    """Return best guess of discovery timestamp (from test_discovery.py)"""
    tolerance = timedelta(hours=1)
    now = timezone.now()
    sa_started = discovery.sa.started if discovery.sa_id else None
    
    if discovery.created:
        candidate = discovery.created
        if candidate > now and sa_started:
            return sa_started
        if sa_started and abs(candidate - sa_started) > tolerance:
            return sa_started
        return candidate
    
    if sa_started:
        return sa_started
    
    return now


def categorize_recommendation(explanation):
    """Categorize discovery by recommendation type"""
    if not explanation:
        return None
    
    explanation_lower = explanation.lower()
    
    # Check for STRONG_BUY patterns (case-insensitive)
    if (explanation_lower.startswith('strong_buy') or explanation_lower.startswith('strong buy') or
        'strong_buy |' in explanation_lower or 'strong buy |' in explanation_lower or
        'recommended strong_buy' in explanation_lower or 'recommended strong buy' in explanation_lower or
        'gemini: strong_buy' in explanation_lower or 'gemini: strong buy' in explanation_lower or
        'newsflash: | strong_buy' in explanation_lower or 'newsflash: | strong buy' in explanation_lower):
        return 'STRONG_BUY'
    
    # Check for BUY patterns (case-insensitive)
    elif (explanation_lower.startswith('buy |') or
          'recommended buy' in explanation_lower or 
          'gemini: buy' in explanation_lower or
          'newsflash: | buy' in explanation_lower):
        return 'BUY'
    
    return None

def analyze_performance():
    """Analyze price performance of STRONG_BUY vs BUY"""
    
    # Get advisors that use news_flash
    news_flash_advisors = Advisor.objects.filter(
        name__in=['StockStory', 'Polygon.io']
    )
    
    if not news_flash_advisors.exists():
        print("Warning: No news_flash advisors found")
        return
    
    print("\n" + "=" * 80)
    print("STRONG_BUY vs BUY PRICE PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Get all discoveries
    discoveries = Discovery.objects.filter(
        advisor__in=news_flash_advisors
    ).select_related('stock', 'advisor').order_by('-created')
    
    total_discoveries = discoveries.count()
    print(f"\nTotal discoveries: {total_discoveries}")
    
    # First pass: categorize and count
    categorized = {'STRONG_BUY': 0, 'BUY': 0, 'OTHER': 0}
    
    for discovery in discoveries:
        recommendation_type = categorize_recommendation(discovery.explanation)
        if recommendation_type in ['STRONG_BUY', 'BUY']:
            categorized[recommendation_type] += 1
        else:
            categorized['OTHER'] += 1
    
    print(f"\nCategorized discoveries:")
    print(f"  STRONG_BUY: {categorized['STRONG_BUY']}")
    print(f"  BUY:        {categorized['BUY']}")
    print(f"  Other:      {categorized['OTHER']}")
    
    print("\nAnalyzing price performance using yfinance historical data...")
    print("(This may take a while as we fetch historical prices)")
    
    fetcher = YfPriceFetcher()
    as_of = timezone.now()
    
    strong_buy_performances = []
    buy_performances = []
    
    processed = 0
    skipped_no_rec_type = 0
    missing_entry_price = 0
    missing_current_price = 0
    errors = 0
    
    for discovery in discoveries:
        recommendation_type = categorize_recommendation(discovery.explanation)
        
        if recommendation_type not in ['STRONG_BUY', 'BUY']:
            skipped_no_rec_type += 1
            continue
        
        try:
            symbol = discovery.stock.symbol
            discovery_dt = _discovery_timestamp(discovery)
            
            # Fetch entry price from yfinance (historical price at discovery time)
            entry_price = fetcher.price_from(symbol, discovery_dt)
            if entry_price is None:
                missing_entry_price += 1
                if missing_entry_price <= 3:
                    print(f"  [warn] Missing entry price for {symbol} (discovered {discovery_dt.date()})")
                continue
            
            # Fetch current price
            current_price = fetcher.latest_price(symbol, as_of)
            if current_price is None:
                missing_current_price += 1
                if missing_current_price <= 3:
                    print(f"  [warn] Missing current price for {symbol}")
                continue
            
            # Calculate price change percentage
            price_change_pct = ((current_price - entry_price) / entry_price) * 100
            
            if recommendation_type == 'STRONG_BUY':
                strong_buy_performances.append(float(price_change_pct))
            else:
                buy_performances.append(float(price_change_pct))
            
            processed += 1
            
            if processed % 10 == 0:
                print(f"  Processed {processed} discoveries...")
                
        except Exception as e:
            errors += 1
            if errors <= 5:  # Show first few errors
                print(f"  Error processing {discovery.stock.symbol}: {e}")
            continue
    
    print(f"\nProcessed {processed} discoveries with complete price data")
    print(f"  Skipped (no rec type): {skipped_no_rec_type}")
    print(f"  Missing entry price: {missing_entry_price}")
    print(f"  Missing current price: {missing_current_price}")
    print(f"  Errors: {errors}")
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("PERFORMANCE STATISTICS")
    print("=" * 80)
    
    def print_stats(label, values):
        if not values:
            print(f"\n{label}: No data")
            return
        
        winners = [v for v in values if v > 0]
        losers = [v for v in values if v < 0]
        
        print(f"\n{label} (n={len(values)}):")
        print(f"  Average return:     {statistics.mean(values):+7.2f}%")
        if len(values) > 1:
            print(f"  Median return:      {statistics.median(values):+7.2f}%")
            print(f"  Std deviation:      {statistics.stdev(values):7.2f}%")
        
        print(f"  Win rate:           {len(winners)/len(values)*100:6.2f}% ({len(winners)}/{len(values)})")
        print(f"  Average gain:       {statistics.mean(winners):+7.2f}%" if winners else "  Average gain:       N/A")
        print(f"  Average loss:       {statistics.mean(losers):+7.2f}%" if losers else "  Average loss:       N/A")
        print(f"  Best performer:     {max(values):+7.2f}%")
        print(f"  Worst performer:    {min(values):+7.2f}%")
    
    print_stats("STRONG_BUY", strong_buy_performances)
    print_stats("BUY", buy_performances)
    
    # Comparison
    if strong_buy_performances and buy_performances:
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        
        sb_avg = statistics.mean(strong_buy_performances)
        buy_avg = statistics.mean(buy_performances)
        difference = sb_avg - buy_avg
        
        print(f"\nAverage return difference: {difference:+.2f}%")
        print(f"  STRONG_BUY outperforms by {abs(difference):.2f}%" if difference > 0 else 
              f"  BUY outperforms by {abs(difference):.2f}%")
        
        if len(strong_buy_performances) > 1 and len(buy_performances) > 1:
            sb_median = statistics.median(strong_buy_performances)
            buy_median = statistics.median(buy_performances)
            median_diff = sb_median - buy_median
            print(f"\nMedian return difference: {median_diff:+.2f}%")
        
        sb_wr = len([v for v in strong_buy_performances if v > 0]) / len(strong_buy_performances) * 100
        buy_wr = len([v for v in buy_performances if v > 0]) / len(buy_performances) * 100
        wr_diff = sb_wr - buy_wr
        print(f"\nWin rate difference: {wr_diff:+.2f}%")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    analyze_performance()
