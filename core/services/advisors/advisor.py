
import logging
import os
import time
import json
import re
import numpy as np
import yfinance as yf
from google import genai
from google.genai import types
import pandas as pd
from datetime import datetime, timedelta
from django.db.models import F, ExpressionWrapper, DateTimeField, Func
from django.db.models.functions import Cast
from core.models import Stock, Discovery, Recommendation, Advisor
from django.conf import settings
from decimal import Decimal
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_gemini_keys() -> List[Optional[str]]:
    """Build list of Gemini API keys from settings (GEMINI_KEY, GEMINI_KEY_2, ...) and env."""
    keys: List[Optional[str]] = []
    primary = getattr(settings, "GEMINI_KEY", None) or os.environ.get("GEMINI_KEY")
    if primary:
        keys.append(primary)
    for i in range(2, 10):
        k = getattr(settings, f"GEMINI_KEY_{i}", None) or os.environ.get(f"GEMINI_KEY_{i}")
        if k:
            keys.append(k)
    return keys

"""
 Note: Gemini models
 https://ai.google.dev/gemini-api/docs/rate-limits 
"""

# Gemini models
models = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

class AdvisorBase:
    # Class-level cache for Polygon stock list (shared across all advisor instances)
    _polygon_stocks_cache = None

    def __init__(self, advisor):
        self.advisor = advisor
        self.gemini_model = 0
        self._gemini_key_index = 0

    def market_open(self):
        """
        Check market open status.
        
        Returns:
            int: Minutes until market opens (negative = not open yet, positive = already open)
                 Examples:
                 - -30 = market opens in 30 minutes
                 - 0 = market just opened
                 - +30 = market has been open for 30 minutes
                 - None = market is closed (weekend or after hours)
        """
        import pytz
        from datetime import datetime
        
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if now_et.weekday() >= 5:  # Saturday or Sunday
            return None
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # If after market close, return None (market closed)
        if now_et.time() >= market_close_time.time():
            return None
        
        # Calculate minutes until/from market open
        minutes_diff = (now_et - market_open_time).total_seconds() / 60
        
        return int(minutes_diff)

    def evaluate_stock(
        self,
        ticker_symbol: str,
        required_return: float = 0.10,
        max_growth: float = 0.20,
        max_roe: float = 0.30,
        info: Optional[Dict] = None,
    ) -> float:
        """
        ROE-based notional valuation. Returns current_price / fair_value (valuation ratio).
        If no usable EPS/ROE or on error, logs a warning and returns 1.0 (neutral).
        If info is provided, uses it instead of fetching yf.Ticker(ticker_symbol).info.
        """
        try:
            if info is None:
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info or {}
            eps = info.get("trailingEps")
            roe = info.get("returnOnEquity")
            payout = info.get("payoutRatio")
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            if eps is None or eps <= 0 or roe is None or roe <= 0:
                logger.warning("evaluate_stock %s: no usable EPS/ROE", ticker_symbol)
                return 1.0
            if current_price is None or current_price <= 0:
                logger.warning("evaluate_stock %s: no usable price", ticker_symbol)
                return 1.0
            payout = 0 if payout is None or payout < 0 else min(max(payout, 0), 0.9)
            adjusted_roe = min(roe, max_roe)
            g = adjusted_roe * (1 - payout)
            g = min(g, max_growth)
            denominator = max(required_return - g, 0.01)
            justified_pe = (adjusted_roe * (1 - payout)) / denominator
            fair_value = eps * justified_pe
            if fair_value <= 0:
                logger.warning("evaluate_stock %s: invalid fair value", ticker_symbol)
                return 1.0
            return float(current_price / fair_value)
        except Exception as e:
            logger.warning("evaluate_stock %s: %s", ticker_symbol, e)
            return 1.0

    @staticmethod
    def get_last_trading_day(test_date=None):
        """
        Get the previous working day (Mon-Fri) for Polygon API.
        
        Only works Tue-Fri (skips Mon/Sat/Sun discoveries).
        Returns None if today is Mon/Sat/Sun or if last day was a holiday.
        
        Args:
            test_date: Optional date string (YYYY-MM-DD) for testing
            
        Returns:
            date string (YYYY-MM-DD) or None
        """
        if test_date:
            # For testing - return the date as-is
            try:
                datetime.strptime(test_date, "%Y-%m-%d")
                return test_date
            except ValueError:
                logger.warning(f"Invalid test_date format: {test_date}")
                return None
        
        today = datetime.now().date()
        weekday = today.weekday()  # Monday=0, Sunday=6
        
        # Only run Tue-Fri (1-4)
        if weekday == 0:  # Monday
            logger.info("Skipping discovery on Monday")
            return None
        elif weekday >= 5:  # Saturday (5) or Sunday (6)
            logger.info("Skipping discovery on weekend")
            return None
        
        # Tue-Fri: previous working day is just yesterday
        # (if today is Tue, yesterday is Mon - both weekdays)
        previous_day = today - timedelta(days=1)
        
        # If yesterday was Sunday (previous_day.weekday() == 6), go back to Friday
        if previous_day.weekday() == 6:  # Yesterday was Sunday
            previous_day = previous_day - timedelta(days=2)  # Go to Friday
        # If yesterday was Saturday (previous_day.weekday() == 5), go back to Friday
        elif previous_day.weekday() == 5:  # Yesterday was Saturday
            previous_day = previous_day - timedelta(days=1)  # Go to Friday
        
        return previous_day.strftime("%Y-%m-%d")

    @classmethod
    def _fetch_polygon_stocks_for_date(cls, reference_date):
        """
        Fetch stocks using Polygon's get_grouped_daily_aggs (1 API call for all stocks on a date).
        
        Args:
            reference_date: Date string (YYYY-MM-DD)
        
        Returns:
            pandas DataFrame with columns: ticker, price, today_volume
            Returns empty DataFrame on error or if no data available
        """
        # Try to get Polygon API key from settings or environment
        polygon_api_key = getattr(settings, 'POLYGON_API_KEY', None)
        if not polygon_api_key:
            # Fallback to environment variable (for compatibility with test scripts)
            import os
            polygon_api_key = os.getenv('POLYGON_API_KEY')
        
        if not polygon_api_key:
            logger.warning("POLYGON_API_KEY not set in Django settings or environment")
            return pd.DataFrame()
        
        try:
            from polygon import RESTClient
            client = RESTClient(polygon_api_key)
            
            logger.info(f"Fetching all stocks for {reference_date} using Polygon (1 API call)...")
            aggs = client.get_grouped_daily_aggs(
                locale="us",
                date=reference_date,
                adjusted=False
            )
            
            rows = []
            for agg in aggs:
                rows.append({
                    "ticker": agg.ticker,
                    "price": float(agg.close),
                    "today_volume": int(agg.volume)
                })
            
            df = pd.DataFrame(rows)
            
            if not df.empty:
                logger.info(f"Fetched {len(df)} stocks from Polygon for {reference_date}")
            else:
                logger.warning(f"No stocks returned from Polygon for {reference_date} (may be holiday)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching stocks from Polygon for {reference_date}: {e}", exc_info=True)
            return pd.DataFrame()

    @classmethod
    def get_filtered_stocks(cls, sa=None, min_price=None, max_price=None, min_volume=None, test_date=None):
        """
        Get filtered stocks from Polygon (last trading day).
        Fetches once per session, caches, then applies advisor-specific filters.
        
        Args:
            sa: SmartAnalysis session (optional, for logging)
            min_price: Minimum stock price filter
            max_price: Maximum stock price filter  
            min_volume: Minimum volume filter
            test_date: Optional date string (YYYY-MM-DD) for testing
            
        Returns:
            pandas DataFrame with filtered stocks (columns: ticker, price, today_volume)
            Returns empty DataFrame if no valid trading date or fetch fails
        """
        # Fetch and cache if needed
        if cls._polygon_stocks_cache is None:
            last_trading_date = cls.get_last_trading_day(test_date=test_date)
            
            if not last_trading_date:
                logger.warning("No valid trading date available (Mon/weekend/holiday)")
                return pd.DataFrame()
            
            # Fetch from Polygon (will handle holiday failures gracefully)
            cls._polygon_stocks_cache = cls._fetch_polygon_stocks_for_date(last_trading_date)
            
            if cls._polygon_stocks_cache is None or cls._polygon_stocks_cache.empty:
                logger.warning(f"No stocks fetched for {last_trading_date} (may be holiday)")
                return pd.DataFrame()
        
        # Apply advisor's filters
        df = cls._polygon_stocks_cache.copy()
        
        if min_price is not None:
            df = df[df['price'] >= min_price]
        if max_price is not None:
            df = df[df['price'] <= max_price]
        if min_volume is not None:
            df = df[df['today_volume'] >= min_volume]
        
        return df

    @classmethod
    def clear_polygon_cache(cls):
        """Clear the Polygon stocks cache (useful for testing or between runs)."""
        cls._polygon_stocks_cache = None
        logger.info("Polygon stocks cache cleared")

    def allow_discovery(self, symbol, period=None, price_decline=None):
        """
        Allow discovery based on previous discoveries.
        
        Args:

            symbol: Stock symbol to check
            period: Optional hours to look back (e.g., 24 for 1 day, 168 for 7 days)
                    If None, no time-based filtering
            price_decline: Optional price decline ratio (e.g., 0.8 = 80% of discovery price)
                   If None, no price-based filtering
        
        Returns:
            True if discovery should be allowed, False if not
        """
        from core.models import Discovery, Stock
        from django.utils import timezone
        
        try:
            stock = Stock.objects.get(symbol=symbol)
        except Stock.DoesNotExist:
            return True  # Stock doesn't exist, so proceed with discovery
        
        # Get most recent discovery by this advisor for this stock
        discovery = Discovery.objects.filter(
            advisor=self.advisor,
            stock=stock
        ).order_by('-created').first()
        
        # If not discovered before, proceed
        if not discovery:
            return True
        
        discovery_price = discovery.price
        time_diff = timezone.now() - discovery.created
        hours_ago = time_diff.total_seconds() / 3600
        
        # Time-based filtering: skip if discovered within period (hours)
        if period is not None:
            if hours_ago < period:
                logger.info(f"{self.advisor.name}: Skipping {symbol} - discovered {hours_ago:.1f} hours ago (within {period}h period)")
                return False  # Filtered out - too recent
        
        # Price-based filtering: skip if price hasn't dropped below discounted price
        if price_decline is not None:
            stock.refresh()
            from decimal import Decimal
            
            # Convert price_decline to Decimal for proper multiplication
            price_decline_decimal = Decimal(str(price_decline))
            threshold_price = discovery_price * price_decline_decimal

            if stock.price > threshold_price:
                logger.info(f"{self.advisor.name}: Skipping {symbol} - price ${discovery_price:.2f} hasn't dropped to {price_decline} threshold")
                return False  # Disallow discovery - hasn't dropped enough
        
        # Passed all filters
        return True

    def discover(self, sa):
        return

    def get_previous_sa_timestamp(self, sa, username=None):
        """
        Return the started datetime of the most recent SmartAnalysis before sa,
        for time-window / dedupe. Returns None if there is no previous SA
        or if sa has no id (e.g. unsaved stub).
        username: if set, scope to that user; else global previous SA.
        """
        if sa is None or getattr(sa, "id", None) is None:
            return None
        from core.models import SmartAnalysis
        qs = SmartAnalysis.objects.filter(id__lt=sa.id)
        if username is not None:
            qs = qs.filter(username=username)
        prev = qs.order_by('-id').first()
        return prev.started if prev else None

    def analyze(self, sa, stock):
        return

    def discovered(self, sa, symbol, explanation, sell_instructions = None, weight = 1.0):
        #-- find stock or create stock
        try:
            stock = Stock.objects.get(symbol=symbol)
        except Stock.DoesNotExist:
            stock = Stock.create(symbol, self.advisor)
            if stock is None:
                return None
            logger.info(f"{self.advisor.name} created stock {stock.symbol}")

        # Latest price
        stock.refresh()

        # Always attempt health check (graceful failure - don't block discovery)
        logger.info(f"Attempting health check for {stock.symbol}...")
        try:
            health = self.health_check(stock, sa)
            if health:
                logger.info(f"Successfully created health check for {stock.symbol}")
            else:
                logger.error(f"Health check returned None for {stock.symbol}")
                return None

        except Exception as e:
            logger.error(f"Health check failed for {stock.symbol} during discovery: {e}", exc_info=True)
            return None

        # REMOVED: 7-day duplicate discovery check - no longer needed since SA sessions are faster without Polygon
        # This allows stocks to be re-discovered and bought again if they show strong movement

        # Create new Discovery record
        discovery = Discovery()
        discovery.sa = sa
        discovery.stock = stock
        discovery.price = stock.price  # NEW: Capture price at discovery time
        discovery.advisor = self.advisor
        discovery.explanation = explanation[:1000]
        discovery.weight = weight
        discovery.save()

        # Create sell instructions if provided
        if sell_instructions:
            from core.models import SellInstruction
            from decimal import Decimal

            for instruction_type, instruction_value, value2 in sell_instructions:
                instruction = SellInstruction()
                instruction.discovery = discovery
                instruction.instruction = instruction_type

                if instruction_type in ['STOP_PERCENTAGE', 'TARGET_PERCENTAGE', 'END_DAY', 'PERCENTAGE_DIMINISHING', 'PERCENTAGE_AUGMENTING']:
                    # instruction_value is a percentage multiplier
                    instruction.value1 = Decimal(str(stock.price)) * Decimal(str(instruction_value)) if instruction_value is not None else None
                elif instruction_type in ['TARGET_DIMINISHING', 'STOP_AUGMENTING']:
                    # instruction_value is already a price (not percentage)
                    instruction.value1 = Decimal(str(instruction_value)) if instruction_value is not None else None
                elif instruction_type == 'NOT_TRENDING':
                    instruction.value1 = None
                else:
                    # For all other types, instruction_value is the direct value
                    instruction.value1 = Decimal(str(instruction_value)) if instruction_value is not None else None
                
                # Set value2 (max_days for diminishing/augmenting, None for others)
                instruction.value2 = Decimal(str(value2)) if value2 is not None else None

                instruction.save()

        logger.info(f"{self.advisor.name} discovers {stock.symbol}")
        return stock

    def recommend(self, sa, stock, confidence, explanation=""):

        recommendation = Recommendation()
        recommendation.sa = sa
        recommendation.stock = stock
        recommendation.advisor = self.advisor
        recommendation.confidence = confidence
        recommendation.explanation = explanation
        recommendation.save()

        logger.info(f"{self.advisor.name} scores {stock.symbol} a confidences of {confidence:.2f}")

    def watch_sells(self, instructions, explanation, days=14):
        """
        Query database for sell instructions triggered today and add to watchlist.

        Only watches stocks discovered by this advisor.
        Filters by instruction types listed in the advisor's 'watch' class attribute.
        Called after market close (last 30 min) during discovery phase.

        Args:
            instructions: Sell instruction types
            explanation: description
            days: Number of days to watch (default 14)

        Returns:
            int: Number of stocks added/updated in watchlist
        """
        from core.models import SellInstruction, Discovery
        from datetime import date

        # Find sells from today for this advisor's discoveries
        today = date.today()
        sells = SellInstruction.objects.filter(
            discovery__advisor=self.advisor,
            discovery__created__date=today,  # Sells from discoveries created today
            instruction__in=instructions  # Filter by watch instruction types
        ).select_related('discovery', 'discovery__stock').distinct('discovery__stock')

        # Add each unique stock to watchlist - watch() handles duplicates
        watched = 0
        for sell in sells:
            discovery = sell.discovery
            self.watch(discovery.stock.symbol, explanation, days=days)
            watched += 1

        return watched

    def watch(self, symbol, explanation, days=14, meta=None, status=None):
        """
        Add a stock to the watchlist for this advisor.

        Args:
            symbol: Stock symbol (str) - will auto-create Stock if it doesn't exist
            explanation: Explanation for why this stock is being watched
            days: Number of days until expiration (defaults to 14)
            meta: Optional dict (e.g. filter state, accession) stored on Watchlist.meta
            status: Optional str ('Pending', 'Excluded', etc.). Default 'Pending'.

        Returns:
            Watchlist object or None if creation fails
        """
        from core.models import Watchlist

        # Get or create Stock object (auto-create like discovered() does)
        try:
            stock = Stock.objects.get(symbol=symbol)
        except Stock.DoesNotExist:
            stock = Stock.create(symbol, self.advisor)
            if stock is None:
                return None
            logger.info(f"{self.advisor.name} created stock {stock.symbol} for watchlist")

        # Refresh stock price to ensure we capture current price at watchlist entry time
        stock.refresh()

        # Create watchlist entry
        watchlist_entry = Watchlist.objects.create(
            advisor=self.advisor,
            stock=stock,  # Model uses Stock object
            price=stock.price,  # Current price at time of adding to watchlist
            explanation=explanation[:500],  # Truncate to max length
            days=days,
            meta=meta if meta is not None else {},
            status=(status or "Pending"),
        )

        logger.info(f"{self.advisor.name} added {stock.symbol} to watchlist at ${stock.price:.2f}: {explanation[:100]}")
        return watchlist_entry


    def watchlist(self):
        """
        Get list of non-expired pending watchlist entries for this advisor.

        Returns:
            QuerySet of Watchlist entries that are pending and not expired
        """
        from core.models import Watchlist
        from django.utils import timezone
        from django.db.models import F, ExpressionWrapper, DateTimeField

        # Get all pending watchlist entries for this advisor
        # Filter out expired entries: created + days >= now (not expired)
        # Calculate expiration_date: created + (days as interval)
        now = timezone.now()
        
        # Use database-level date arithmetic (PostgreSQL supports interval arithmetic)
        # For PostgreSQL: created + (days || ' days')::interval
        # For Django ORM, we'll use ExpressionWrapper with proper casting
        return Watchlist.objects.filter(
            advisor=self.advisor,
            status='Pending'
        ).extra(
            where=["created + (days || ' days')::interval >= %s"],
            params=[now]
        ).select_related('stock')

    def watched(self, symbol):
        """
        True if this advisor has any non-expired watchlist entry for this symbol
        (any status: Pending or Excluded). Use to skip re-processing in discover.
        """
        from core.models import Watchlist
        from django.utils import timezone

        now = timezone.now()
        return Watchlist.objects.filter(
            advisor=self.advisor,
            stock__symbol=(symbol or "").strip().upper(),
        ).extra(
            where=["created + (days || ' days')::interval >= %s"],
            params=[now],
        ).exists()

    def health_check(self, stock, sa):
        """
        Calculate health check for a stock.
        Only creates a new health check if one doesn't exist or if the latest one is more than a week old.

        Args:
            stock: Stock model instance (with symbol, company, price populated)
            sa: SmartAnalysis session instance

        Returns:
            Health object or None if calculation fails or skipped
        """
        logger.info(f"Starting health_check for {stock.symbol}")
        try:
            from core.models import Health
            from django.utils import timezone
            from datetime import timedelta

            # Check if a recent health check exists (within last week)
            one_week_ago = timezone.now() - timedelta(days=7)
            recent_health = Health.objects.filter(
                stock=stock
            ).order_by('-created').first()

            if recent_health and recent_health.created > one_week_ago:
                logger.info(
                    f"Skipping health check for {stock.symbol}: existing check from {recent_health.created} is less than a week old"
                )
                return recent_health  # Return existing health check

            # 1. Run algorithm to get confidence score
            logger.info(f"Calculating confidence score for {stock.symbol}...")
            score_data = self._get_buy_score(stock.symbol)
            if not score_data:
                logger.warning(f"Failed to calculate confidence score for {stock.symbol}")
                return None
            confidence_score = Decimal(str(score_data['confidence_score']))
            logger.info(f"Confidence score calculated for {stock.symbol}: {confidence_score}")

            # 2. Call Gemini for opinion (using V5 prompt) — skip if score invalid to avoid wasted LLM call
            if confidence_score.is_nan():
                logger.warning(f"Skipping health check for {stock.symbol}: confidence score is NaN (bad data)")
                return None
            gemini_result = self._get_gemini_opinion(stock)
            gemini_weight = Decimal('1.0')  # Default to neutral if Gemini fails
            gemini_recommendation = None
            gemini_explanation = None

            if gemini_result:
                gemini_weight = Decimal(str(gemini_result['gemini_weight']))
                gemini_recommendation = gemini_result['recommendation']
                gemini_explanation = gemini_result['explanation']
                logger.info(
                    f"Gemini opinion for {stock.symbol}: {gemini_recommendation} (weight: {gemini_weight})")
            else:
                logger.warning(f"Gemini opinion failed for {stock.symbol}, using default weight 1.0")

            # 3. Calculate final score
            final_score = confidence_score * gemini_weight

            # 4. Create Health record (only if none exists or existing one is old)
            health = Health.objects.create(
                stock=stock,
                sa=sa,
                score=final_score,
                meta={
                    'confidence_score': float(confidence_score),
                    'gemini_weight': float(gemini_weight),
                    'gemini_recommendation': gemini_recommendation,
                    'gemini_explanation': gemini_explanation,
                    'health_score': score_data['health_score'],
                    'valuation_score': score_data['valuation_score'],
                    'piotroski': score_data['piotroski'],
                    'altman_z': float(score_data['altman_z']) if score_data['altman_z'] is not None else None,
                    'ratios': {
                        'current_ratio': float(score_data['ratios']['current_ratio']) if
                        score_data['ratios']['current_ratio'] is not None else None,
                        'debt_to_equity': float(score_data['ratios']['debt_to_equity']) if
                        score_data['ratios']['debt_to_equity'] is not None else None,
                        'net_margin': float(score_data['ratios']['net_margin']) if score_data['ratios'][
                                                                                       'net_margin'] is not None else None,
                        'roa': float(score_data['ratios']['roa']) if score_data['ratios'][
                                                                         'roa'] is not None else None,
                    },
                    'valuation': {
                        'pe': float(score_data['valuation']['pe']) if score_data['valuation'][
                                                                          'pe'] is not None else None,
                        'pb': float(score_data['valuation']['pb']) if score_data['valuation'][
                                                                          'pb'] is not None else None,
                        'ev_ebitda': float(score_data['valuation']['ev_ebitda']) if score_data['valuation'][
                                                                                        'ev_ebitda'] is not None else None,
                        'fcf_yield': float(score_data['valuation']['fcf_yield']) if score_data['valuation'][
                                                                                        'fcf_yield'] is not None else None,
                    },
                },
            )

            logger.info(f"Health check created for {stock.symbol}: score={health.score} (confidence={confidence_score} * weight={gemini_weight})")
            return health

        except ImportError:
            logger.warning("Health model not found - model needs to be created first")
            return None
        except Exception as e:
            logger.error(f"Error calculating health check for {stock.symbol}: {e}", exc_info=True)
            return None

    def _safe_div(self, a, b):
        """Safely divide two numbers, returning None for invalid operations or NaN/Inf."""
        try:
            if a is None or b in (0, None):
                return None
            result = a / b
            if np.isnan(result) or np.isinf(result):
                return None
            return result
        except Exception:
            return None

    def _get_buy_score(self, symbol: str) -> Optional[Dict]:
        """
        Calculate buy score for a stock using financial ratios and scoring metrics.
        Ported from test_health_check.py
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
        
        Returns:
            Dictionary with buy_score (0-100), component scores, and financial metrics
            None if calculation fails
        """
        try:
            t = yf.Ticker(symbol)

            info = t.info
            bs = t.balance_sheet.transpose()
            is_ = t.financials.transpose()
            cf = t.cashflow.transpose()

            # --- Extract Key Fields ---
            try:
                latest_bs = bs.iloc[0]
                latest_is = is_.iloc[0]
                latest_cf = cf.iloc[0]
            except:
                raise ValueError("Missing financial statement data.")

            # Financial statement items
            total_assets = latest_bs.get("Total Assets")
            total_liab = latest_bs.get("Total Liab")
            current_assets = latest_bs.get("Total Current Assets")
            current_liab = latest_bs.get("Total Current Liabilities")
            inventory = latest_bs.get("Inventory", 0)
            long_term_debt = latest_bs.get("Long Term Debt") or 0
            short_term_debt = latest_bs.get("Short Long Term Debt") or 0

            revenue = latest_is.get("Total Revenue")
            gross_profit = latest_is.get("Gross Profit")
            net_income = latest_is.get("Net Income")
            ebit = latest_is.get("EBIT") or latest_is.get("Operating Income")

            op_cf = latest_cf.get("Total Cash From Operating Activities")

            # Market (price-based)
            price = info.get("currentPrice")
            eps = info.get("trailingEps")
            book_value = info.get("bookValue")
            market_cap = info.get("marketCap")
            enterprise_value = info.get("enterpriseValue")
            ebitda = info.get("ebitda")

            # --- Core Ratios ---
            ratios = {
                "current_ratio": self._safe_div(current_assets, current_liab),
                "quick_ratio": self._safe_div((current_assets or 0) - (inventory or 0), current_liab),
                "debt_to_equity": self._safe_div(
                    long_term_debt + short_term_debt,
                    (total_assets - total_liab) if total_assets and total_liab else None
                ),
                "net_margin": self._safe_div(net_income, revenue),
                "roa": self._safe_div(net_income, total_assets),
                "asset_turnover": self._safe_div(revenue, total_assets),
            }

            # --- Valuation Ratios (share-price dependent) ---
            valuation = {
                "pe": self._safe_div(price, eps),
                "pb": self._safe_div(price, book_value),
                "ev_ebitda": self._safe_div(enterprise_value, ebitda),
                "price_sales": self._safe_div(market_cap, revenue),
                "fcf_yield": self._safe_div(op_cf, market_cap)
            }

            # --- Simple Piotroski F-score (reduced version) ---
            try:
                prev_is = is_.iloc[1]
                prev_bs = bs.iloc[1]

                roa_cur = self._safe_div(net_income, total_assets)
                roa_prev = self._safe_div(prev_is.get("Net Income"), prev_bs.get("Total Assets"))
                ocf_pos = op_cf is not None and op_cf > 0
                roa_improve = roa_prev is not None and roa_cur is not None and roa_cur > roa_prev
                gross_cur = self._safe_div(gross_profit, revenue)
                gross_prev = self._safe_div(prev_is.get("Gross Profit"), prev_is.get("Total Revenue"))
                gross_improve = gross_cur is not None and gross_prev is not None and gross_cur > gross_prev

                piotroski = sum([
                    1 if roa_cur and roa_cur > 0 else 0,
                    1 if ocf_pos else 0,
                    1 if roa_improve else 0,
                    1 if gross_improve else 0
                ])
            except:
                piotroski = None

            # --- Altman Z Simplified ---
            try:
                # Working Capital / Total Assets
                working_capital = (current_assets or 0) - (current_liab or 0)
                A = self._safe_div(working_capital, total_assets)
                
                # Retained Earnings / Total Assets (try to get from balance sheet)
                retained_earnings = latest_bs.get("Retained Earnings") or latest_bs.get("Retained Earnings All Equity") or None
                B = self._safe_div(retained_earnings, total_assets)
                
                # EBIT / Total Assets
                C = self._safe_div(ebit, total_assets)
                
                # Market Cap / Total Liabilities (or Book Value of Equity if market cap not available)
                total_equity = (total_assets or 0) - (total_liab or 0)
                if market_cap and total_liab:
                    D = self._safe_div(market_cap, total_liab)
                elif total_equity and total_liab:
                    D = self._safe_div(total_equity, total_liab)
                else:
                    D = None
                
                # Sales / Total Assets (Asset Turnover)
                E = self._safe_div(revenue, total_assets)
                
                # Calculate Altman Z with proper handling of None values
                if A is None and C is None and E is None:
                    altman = None
                else:
                    altman = (
                        1.2 * (A or 0) + 
                        1.4 * (B or 0) + 
                        3.3 * (C or 0) + 
                        0.6 * (D or 0) + 
                        1.0 * (E or 0)
                    )
                    # Handle NaN values
                    if altman is not None and (np.isnan(altman) or np.isinf(altman)):
                        altman = None
            except Exception as e:
                altman = None

            # --- Convert each component into a 0-100 score ---
            # Business health
            health_score = np.mean([
                np.clip((ratios["current_ratio"] or 1) * 20, 0, 100),
                np.clip(100 - (ratios["debt_to_equity"] or 0) * 20, 0, 100),
                np.clip((ratios["net_margin"] or 0) * 500, 0, 100),
                np.clip((ratios["roa"] or 0) * 2000, 0, 100)
            ])

            # Valuation attractiveness
            valuation_score = np.mean([
                np.clip(50 - (valuation["pe"] or 50), 0, 100),
                np.clip(50 - (valuation["pb"] or 50), 0, 100),
                np.clip(50 - (valuation["ev_ebitda"] or 50), 0, 100),
                np.clip((valuation["fcf_yield"] or 0) * 500, 0, 100)
            ])

            # Piotroski normalized
            pscore = (piotroski / 4 * 100) if piotroski is not None else 50

            # Altman mapped to 0-100
            if altman is None:
                zscore = 50
            elif altman > 3:
                zscore = 90
            elif altman < 1.8:
                zscore = 20
            else:
                zscore = 50

            # --- Combined Buy Score (now confidence_score) ---
            # Weighted toward valuation for buying decisions
            buy_score = (
                health_score * 0.35 +      # 35% - Safety/fundamentals
                valuation_score * 0.40 +   # 40% - Value focus
                pscore * 0.15 +            # 15% - Trend indicators
                zscore * 0.10              # 10% - Bankruptcy risk
            )

            return {
                "ticker": symbol,
                "confidence_score": round(buy_score, 1),
                "health_score": round(health_score, 1),
                "valuation_score": round(valuation_score, 1),
                "piotroski": piotroski,
                "altman_z": altman,
                "ratios": ratios,
                "valuation": valuation
            }
        except Exception as e:
            logger.error(f"Error calculating buy score for {symbol}: {e}")
            return None

    def _get_gemini_opinion(self, stock):
        """
        Get Gemini's independent opinion on a stock using V5 prompt.
        
        Args:
            stock: Stock model instance (with symbol, company, price populated)
        
        Returns:
            Dictionary with 'recommendation', 'explanation', and 'gemini_weight' (multiplier)
            or None if Gemini call fails
        """
        prompt = f"""
Starting afresh,

Would you recommend buying {stock.symbol} ({stock.company}) @ ${float(stock.price):.2f}?

Considering:
- Recent news and media coverage
- Market sentiment
- Any significant developments

Respond in JSON format:
{{
    "recommendation": "STRONG_BUY|BUY|NEUTRAL|AVOID|STRONG_AVOID",
    "explanation": "Your reasoning based on recent news and sentiment"
}}

Thank you
"""
        
        model, results = self.ask_gemini(prompt, use_search=True)
        
        if not results:
            return None
        
        recommendation = results.get("recommendation", "").upper()
        explanation = results.get("explanation", "")
        
        # Map recommendation to gemini_weight multiplier
        weight_mapping = {
            "STRONG_BUY": 1.5,
            "BUY": 1.25,
            "NEUTRAL": 1.0,
            "AVOID": 0.75,
            "STRONG_AVOID": 0.5,
        }
        gemini_weight = weight_mapping.get(recommendation, 1.0)  # Default to 1.0 (neutral) if unknown
        
        return {
            "recommendation": recommendation,
            "explanation": explanation,
            "gemini_weight": gemini_weight,
            "model": model,
        }


    def ask_gemini(self, prompt, timeout=120.0, use_search=False):
        """
        Call Gemini API (google.genai) with retry over models and round-robin API keys.
        Returns tuple (model, results) or (None, None) on failure.
        use_search: if True, enable Google Search grounding (e.g. for media reaction).
        https://ai.google.dev/gemini-api/docs/rate-limits
        """
        keys = _get_gemini_keys()
        if not keys:
            logger.warning("No GEMINI_KEY (or GEMINI_KEY_2, ...) configured for ask_gemini")
            return None, None

        for attempt in range(len(models)):
            try:
                key = keys[self._gemini_key_index % len(keys)]
                self._gemini_key_index += 1
                model = models[self.gemini_model]
                logger.info(f"{self.advisor.name} using {model}")

                client = genai.Client(
                    api_key=key,
                    http_options=types.HttpOptions(timeout=int(timeout * 1000)),
                )
                config = None
                if use_search:
                    config = types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())]
                    )
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )

                response_text = getattr(response, "text", None) if response else None
                if not response_text:
                    logger.warning(f"No text in Gemini response for {self.advisor.name}")
                    return None, None

                results = self._extract_json(response_text)
                if not results:
                    logger.warning(f"Cannot parse response for {self.advisor.name}")
                    return None, None

                time.sleep(1)
                return model, results

            except Exception as e:
                err_str = str(e)
                if "429" in err_str and "RESOURCE_EXHAUSTED" in err_str:
                    short_error = "429 RESOURCE_EXHAUSTED"
                elif "403" in err_str and "PERMISSION_DENIED" in err_str:
                    short_error = "403 PERMISSION_DENIED"
                elif "429" in err_str:
                    short_error = "429 (quota exceeded)"
                elif "403" in err_str:
                    short_error = "403 (permission denied)"
                else:
                    short_error = err_str[:80] + ("..." if len(err_str) > 80 else "")
                logger.warning(
                    f"Attempt {attempt + 1}: {model} {short_error} for {self.advisor.name}. Trying next model."
                )
                self.gemini_model += 1
                self.gemini_model %= len(models)

        logger.error(f"All Gemini models exhausted for {self.advisor.name}")
        return None, None

    def news_flash(self, sa, title, url):

        # Check if market is open
        market_status = self.market_open()
        is_market_open = market_status is not None and market_status >= 0

        # Filter by title: reject articles containing common low-value phrases
        title_lower = title.lower() if title else ""
        filter_phrases = [
            "stock market",
            "market today",
            "today nasdaq",
            "nasdaq futures",
            "futures slip",
            "analyst questions",
            "stocks"
        ]
        
        for phrase in filter_phrases:
            if phrase in title_lower:
                logger.info(f"{self.advisor.name} filtering article (skipping Gemini): {title[:80]}")
                return

        # Check if this article has already been processed by this advisor
        from core.models import Discovery
        existing = Discovery.objects.filter(
            advisor=self.advisor,
            explanation__contains=url
        ).exists()

        if existing:
            logger.info(f"{self.advisor.name} skipping article (already processed): {title[:80]} | {url}")
            return

        # Carefully worded script for the robot
        prompt = f"""
            Starting afresh
            You are an expert analyzing business articles
            How do you interpret this article by way of speculation of rising share prices? Supply a recommendation.
            
            If in doubt, lean toward skepticism 

            Please respond in JSON only choosing one of the below recommendations and supply relevant company symbol / ticker 

            RETURN JSON:
            {{
                "recommendation": "DISMISS|BUY|SELL|STRONG_BUY|STRONG_SELL",
                "tickers": ["SYM1", "SYM2"],
                "explanation": "A brief reason you came to your descion"
            }}

            url: {url}"""

        # Pass sell instructions to discovery
        sell_instructions = [
            ("PERCENTAGE_DIMINISHING", 1.50, 7),
            ("PERCENTAGE_AUGMENTING", 0.95, 21),
            ('DESCENDING_TREND', -0.20, None),
            ('NOT_TRENDING', None, None)
        ]

        # Ask AI for opinion
        model, results = self.ask_gemini(prompt)

        if not results:
            return

        explanation = results.get("explanation", "")
        recommendation = results.get("recommendation", "")
        tickers = results.get("tickers", [])

        # Log it
        logger.info(f"{recommendation}: {tickers} | {title}")

        # Anything better than DISMISS is put forward for consensus
        if recommendation != "BUY" and recommendation != "STRONG_BUY":
            return

        # If market not open yet, add to watchlist (will be processed when market opens via news_watch())
        if not is_market_open:
            for ticker in tickers:
                self.watch(
                    symbol=ticker,
                    explanation=f"{model} recommended {recommendation} from reading article. | Article: {title} | {url} | {explanation} ",
                    days=1
                )
                logger.info(f"{self.advisor.name} added {ticker} to watchlist (market closed, {recommendation}): {title[:80]}")
            return

        # Market open - new discoveries (process immediately)
        for ticker in tickers:
            self.discovered(sa, ticker, f"{model} recommended {recommendation} from reading article. | Article: {title} | {url} | {explanation} ",
                sell_instructions, 1.5 if recommendation == "STRONG_BUY" else 1.0)


    def news_watch(self, sa):
        # Check if market is open
        market_status = self.market_open()
        if market_status is None or market_status < 0:
            return

        # Pass sell instructions to discovery
        sell_instructions = [
            ("PERCENTAGE_DIMINISHING", 1.50, 7),
            ("PERCENTAGE_AUGMENTING", 0.95, 21),
            ('DESCENDING_TREND', -0.20, None),
            ('NOT_TRENDING', None, None)
        ]

        # Market open - process pending watchlist entries
        pending_entries = self.watchlist()

        for entry in pending_entries:
            self.discovered(sa, entry.stock.symbol, entry.explanation, sell_instructions, 1.0)

            entry.status = "Executed"
            entry.save()

            logger.info(f"{self.advisor.name} processed pending watchlist entry: {entry.stock.symbol}")

    def _extract_json(self, text):
        """Extract JSON from Gemini response, handling markdown code blocks."""
        if not text:
            return None

        # Remove markdown code blocks
        cleaned = re.sub(r'```json\s*', '', text)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object with regex
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None


def register(name, python_class):
    try:
        Advisor.objects.get(python_class=python_class)

    except Advisor.DoesNotExist:
        logger.info(f"Created new advisor: {name}")

        advisor = Advisor(name=name, python_class=python_class)
        advisor.save()
