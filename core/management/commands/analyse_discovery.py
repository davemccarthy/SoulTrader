"""
Analyse Discovery Management Command

Calculates advisor weights based on historical discovery performance (win rate)
and updates the Advisor.weight field.

Usage:
    python manage.py analyse_discovery                    # Show results without updating
    python manage.py analyse_discovery --set              # Calculate and update weights
    python manage.py analyse_discovery --days 14 --set     # Update weights for last 14 days
    python manage.py analyse_discovery --reset             # Show what would be reset to 1.0
    python manage.py analyse_discovery --reset --set       # Reset all weights to 1.0
"""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone as dt_timezone
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models import Advisor, Discovery

try:
    import yfinance as yf
except ImportError:
    yf = None


def _discovery_timestamp(discovery) -> datetime:
    """Return best guess of discovery timestamp."""
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


def _dedupe_discoveries(discoveries: Iterable[Discovery]) -> List[Discovery]:
    """Remove duplicate discoveries (same stock)."""
    seen = set()
    unique: List[Discovery] = []
    for discovery in discoveries:
        if discovery.stock_id in seen:
            continue
        seen.add(discovery.stock_id)
        unique.append(discovery)
    return unique


class YfPriceFetcher:
    """Basic yfinance helper with simple caching."""

    def __init__(self):
        self._history_cache: Dict[tuple, Optional[float]] = {}
        self._latest_cache: Dict[tuple, Optional[float]] = {}

    def price_from(self, symbol: str, ref_datetime: datetime) -> Optional[float]:
        """Get price at a specific datetime."""
        key = (symbol.upper(), ref_datetime.strftime("%Y-%m-%d"))
        if key not in self._history_cache:
            self._history_cache[key] = self._lookup_price(symbol, ref_datetime)
        return self._history_cache[key]

    def latest_price(self, symbol: str, as_of: datetime) -> Optional[float]:
        """Get latest price as of a specific datetime."""
        use_realtime = self._should_use_realtime(as_of)
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
    def _should_use_realtime(as_of: datetime) -> bool:
        """Use real-time quotes when the reference point is today (or in the future)."""
        now = timezone.now()
        return as_of.date() >= now.date()

    @staticmethod
    def _lookup_realtime_price(symbol: str) -> Optional[float]:
        """Attempt to fetch the latest traded price."""
        if yf is None:
            return None
        try:
            ticker = yf.Ticker(symbol)
            fast_info = getattr(ticker, "fast_info", None)
            if fast_info:
                for field in ("last_price", "regular_market_price", "post_market_price"):
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
    def _lookup_price(symbol: str, ref_datetime: datetime, forward_only: bool = False) -> Optional[float]:
        """Fetch the first available close on/after ref_datetime."""
        if yf is None:
            return None
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

        # Normalise index to naive datetime for comparison
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


def _summarise(results: Sequence[Dict]) -> Dict[str, float]:
    """Calculate summary statistics from discovery results."""
    summary = defaultdict(float)
    summary["count"] = len(results)
    total_entry = 0.0
    total_current = 0.0

    for row in results:
        change_pct = row.get("pct_change")
        if change_pct is None:
            summary["missing"] += 1
            continue
        if change_pct > 0:
            summary["gainers"] += 1
        elif change_pct < 0:
            summary["losers"] += 1
        else:
            summary["flat"] += 1
        summary["avg_change"] += change_pct
        entry = row.get("entry_price")
        current = row.get("current_price")
        if entry is not None and current is not None:
            total_entry += entry
            total_current += current

    if summary["count"] - summary.get("missing", 0) > 0:
        summary["avg_change"] /= summary["count"] - summary.get("missing", 0)
    else:
        summary["avg_change"] = 0.0

    summary["total_entry"] = total_entry
    summary["total_current"] = total_current
    summary["net_change"] = total_current - total_entry
    if total_entry > 0:
        summary["total_pct"] = (summary["net_change"] / total_entry) * 100
    else:
        summary["total_pct"] = 0.0

    return summary


class Command(BaseCommand):
    help = 'Calculate and update advisor weights based on discovery win rates'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            help='Number of trailing days to include (default: 14)',
            default=14
        )
        parser.add_argument(
            '--start-sa',
            type=int,
            help='First SmartAnalysis id (inclusive)'
        )
        parser.add_argument(
            '--end-sa',
            type=int,
            help='Last SmartAnalysis id (inclusive)'
        )
        parser.add_argument(
            '--set',
            action='store_true',
            help='Actually update advisor weights (default: show results without updating)'
        )
        parser.add_argument(
            '--reset',
            action='store_true',
            help='Reset all advisor weights to 1.0 (skips discovery analysis)'
        )
        parser.add_argument(
            '--as-of',
            type=str,
            help='Reference timestamp for current price (ISO format, default: now)'
        )
        parser.add_argument(
            '--after-days',
            type=int,
            help='Evaluate performance N days after discovery (0 = end of trading day, 7 = 1 week later). Overrides --as-of.'
        )

    def handle(self, *args, **options):
        if yf is None:
            self.stdout.write(self.style.ERROR('yfinance is required. Install it with: pip install yfinance'))
            return

        days = options.get('days')
        start_sa = options.get('start_sa')
        end_sa = options.get('end_sa')
        set_weights = options.get('set', False)
        reset_weights = options.get('reset', False)
        as_of_str = options.get('as_of')
        after_days = options.get('after_days')
        
        # Handle --reset: set all advisors to weight 1.0 and exit
        if reset_weights:
            advisors = Advisor.objects.all()
            count = 0
            for advisor in advisors:
                old_weight = advisor.weight
                if set_weights:
                    advisor.weight = Decimal('1.0')
                    advisor.save(update_fields=['weight'])
                self.stdout.write(f"{advisor.name:<20} Weight: {old_weight:.3f} → 1.000")
                count += 1
            
            if not set_weights:
                self.stdout.write(self.style.WARNING(f'\nDRY RUN - Would reset {count} advisor weights to 1.0'))
                self.stdout.write(self.style.WARNING('Use --set to actually update weights'))
            else:
                self.stdout.write(self.style.SUCCESS(f'\nReset {count} advisor weights to 1.0'))
            return

        # Parse as_of if provided (only if --after-days not specified)
        if after_days is None:
            if as_of_str:
                try:
                    dt = datetime.fromisoformat(as_of_str)
                    if dt.tzinfo is None:
                        as_of = timezone.make_aware(dt)
                    else:
                        as_of = dt.astimezone(timezone.get_current_timezone())
                except ValueError:
                    self.stdout.write(self.style.ERROR(f'Invalid --as-of format: {as_of_str}'))
                    return
            else:
                as_of = timezone.now()
        else:
            as_of = None  # Will be calculated per discovery

        # Build discovery queryset
        qs = Discovery.objects.select_related("advisor", "stock", "sa").order_by("created")

        if start_sa is not None:
            qs = qs.filter(sa_id__gte=start_sa)
            if end_sa is not None:
                qs = qs.filter(sa_id__lte=end_sa)
        else:
            cutoff = timezone.now() - timedelta(days=days)
            qs = qs.filter(created__gte=cutoff)

        # Filter to only include discoveries from enabled advisors
        qs = qs.filter(advisor__enabled=True)

        discoveries = list(qs)
        if not discoveries:
            self.stdout.write(self.style.WARNING('No discoveries found for provided criteria.'))
            return

        unique_discoveries = _dedupe_discoveries(discoveries)
        fetcher = YfPriceFetcher()
        rows = []

        # Show evaluation period
        if after_days is not None:
            if after_days == 0:
                self.stdout.write(f"Evaluating discoveries at end of trading day (4:00 PM ET) on discovery date")
            else:
                self.stdout.write(f"Evaluating discoveries {after_days} day(s) after discovery")
        else:
            self.stdout.write(f"Evaluating discoveries as of: {as_of.strftime('%Y-%m-%d %H:%M:%S')}")

        # Calculate performance for each discovery
        for discovery in unique_discoveries:
            symbol = discovery.stock.symbol
            discovery_dt = _discovery_timestamp(discovery)

            # Use stored price if available, otherwise lookup
            if discovery.price and discovery.price > 0:
                entry_price = float(discovery.price)
            else:
                entry_price = fetcher.price_from(symbol, discovery_dt)

            # Determine evaluation date
            if after_days is not None:
                # Calculate date N days after discovery
                evaluation_dt = discovery_dt + timedelta(days=after_days)
                
                # For --after-days 0, use end of trading day (4:00 PM ET)
                if after_days == 0:
                    import pytz
                    et = pytz.timezone('US/Eastern')
                    # Ensure discovery_dt is timezone-aware
                    if discovery_dt.tzinfo is None:
                        discovery_dt_aware = timezone.make_aware(discovery_dt)
                    else:
                        discovery_dt_aware = discovery_dt
                    
                    # Convert to ET and set to 4:00 PM on discovery date
                    evaluation_dt_et = discovery_dt_aware.astimezone(et)
                    evaluation_dt_et = evaluation_dt_et.replace(hour=16, minute=0, second=0, microsecond=0)
                    
                    # Convert back to discovery_dt's timezone
                    evaluation_dt = evaluation_dt_et.astimezone(discovery_dt_aware.tzinfo)
                # For other values, evaluation_dt is already set to discovery_dt + after_days
                
                current_price = fetcher.latest_price(symbol, evaluation_dt)
            else:
                # Use as_of (default: now)
                current_price = fetcher.latest_price(symbol, as_of)

            pct_change = None
            abs_change = None
            if entry_price and current_price:
                abs_change = current_price - entry_price
                pct_change = (abs_change / entry_price) * 100

            rows.append({
                "symbol": symbol,
                "advisor": discovery.advisor.name,
                "created": discovery_dt,
                "entry_price": entry_price,
                "current_price": current_price,
                "abs_change": abs_change,
                "pct_change": pct_change,
            })

        # Group by advisor and calculate win rates (scores)
        per_advisor = defaultdict(list)
        for row in rows:
            per_advisor[row["advisor"]].append(row)

        # First pass: Calculate scores (win rates) for all advisors
        advisor_scores = {}
        advisor_stats = {}
        
        for advisor_name, advisor_rows in sorted(per_advisor.items()):
            advisor_summary = _summarise(advisor_rows)
            complete = int(advisor_summary["count"] - advisor_summary.get("missing", 0))
            gainers = int(advisor_summary.get("gainers", 0))
            losers = int(advisor_summary.get("losers", 0))

            # Calculate win rate (score)
            if complete > 0:
                win_rate = gainers / complete
                score = Decimal(str(win_rate))
            else:
                score = Decimal('1.0')  # Default if no data
            
            advisor_scores[advisor_name] = score
            advisor_stats[advisor_name] = {
                'complete': complete,
                'gainers': gainers,
                'losers': losers,
                'win_rate': win_rate * 100 if complete > 0 else 0.0
            }

        # Calculate average score for normalization
        if advisor_scores:
            avg_score = sum(advisor_scores.values()) / len(advisor_scores)
        else:
            avg_score = Decimal('1.0')
        
        self.stdout.write(f"\nAverage advisor score: {avg_score:.3f}\n")

        # Second pass: Normalize scores to weights and update
        updates = []
        for advisor_name in sorted(advisor_scores.keys()):
            score = advisor_scores[advisor_name]
            stats = advisor_stats[advisor_name]
            
            # Normalize: weight = score / avg_score
            # If score equals average, weight = 1.0
            # If score above average, weight > 1.0
            # If score below average, weight < 1.0
            if avg_score > 0:
                weight = score / avg_score
            else:
                weight = Decimal('1.0')

            try:
                advisor = Advisor.objects.get(name=advisor_name)
                old_weight = advisor.weight

                if set_weights:
                    advisor.weight = weight
                    advisor.save(update_fields=['weight'])

                updates.append({
                    'advisor': advisor_name,
                    'old_weight': old_weight,
                    'score': score,
                    'new_weight': weight,
                    'complete': stats['complete'],
                    'gainers': stats['gainers'],
                    'win_rate': stats['win_rate']
                })

                # Format ratio as "Gainers/Losers" or just "Gainers" if no losers
                if stats['losers'] > 0:
                    ratio_str = f"{stats['gainers']}/{stats['losers']}"
                else:
                    ratio_str = f"{stats['gainers']}/0"
                
                self.stdout.write(
                    f"{advisor_name:<20} "
                    f"Total: {stats['complete']:>3} | "
                    f"Gainers: {stats['gainers']:>3} | "
                    f"Losers: {stats['losers']:>3} | "
                    f"Ratio: {ratio_str:>6} | "
                    f"Win Rate: {stats['win_rate']:>5.1f}% | "
                    f"Score: {score:.3f} | "
                    f"Weight: {old_weight:.3f} → {weight:.3f}"
                )
            except Advisor.DoesNotExist:
                self.stdout.write(self.style.WARNING(f'Advisor "{advisor_name}" not found in database'))

        if not set_weights:
            self.stdout.write(self.style.WARNING('\nDRY RUN - No changes saved'))
            self.stdout.write(self.style.WARNING('Use --set to actually update advisor weights'))
        else:
            self.stdout.write(self.style.SUCCESS(f'\nUpdated {len(updates)} advisor weights'))

