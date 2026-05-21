"""
Core Models for SoulTrader

Based on analysis.py design - 8 simple models, SQL-first approach

TODO: Implement these models based on your analysis.py classes:
    - Stock
    - Advisor  
    - SmartAnalysis
    - Discovery
    - Recommendation
    - Consensus
    - Holding
    - Trade

Reference your /Users/davidmccarthy/Development/scratch/analysis.py for the structure.
"""

from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField
from django.utils import timezone
from datetime import timedelta, datetime as dt_class, date as date_class, time as time_class
from decimal import Decimal
from typing import Optional

import yfinance as yf
import logging

# Your models go here
# Keep them simple and aligned with your SQL queries in analysis.py

# Profile / user settings
class Profile(models.Model):

    # Depicts asperational number of stock in portfolio
    SPREAD = {
        "MEGA": 100,
        "LARGE": 60,
        "MEDIUM": 40,
        "SMALL": 20,
        "MICRO": 10
    }

    # Minimum v2 grade letter per profile; min_score() maps via ratings.RATING_BANDS.
    RISK = {
        "CONSERVATIVE": "B",
        "MODERATE": "C",
        "AGGRESSIVE": "D",
        "RECKLESS": "F",
    }

    # But / sell sentiment
    SENTIMENT = {
        "STRONG_BULL": 1.2,
        "BULL": 1.1,
        "STAG": 1.0,
        "AUTO": 1.0,
        "BEAR": 0.9,
        "STRONG_BEAR": 0.8
    }

    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    name = models.CharField(max_length=120, blank=True, default="")
    created = models.DateTimeField(null=True, blank=True, auto_now_add=True)
    description = models.TextField(blank=True, default="")
    enabled = models.BooleanField(default=True)
    advisors = ArrayField(models.CharField(max_length=100), default=list, blank=True)
    risk = models.CharField(max_length=20, choices=[(key, key) for key in RISK.keys()], default='MODERATE')
    spread = models.CharField(max_length=10, choices=[(key, key) for key in SPREAD.keys()], null=True, blank=True)
    sentiment = models.CharField(max_length=16, choices=[(key, key) for key in SENTIMENT.keys()], null=False, default='AUTO')
    investment = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('100000.00'))
    cash = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('100000.00'))

    def average_spend(self):
        num_stocks = Decimal(Profile.SPREAD[self.spread])
        return self.investment / num_stocks

    def min_grade(self) -> str:
        """Minimum allowed v2 letter grade for this profile."""
        return Profile.RISK[self.risk]

    def min_score(self):
        """Minimum v2 composite (0–100) for profile min grade."""
        from core.services.health.ratings import min_composite_for_letter

        return Decimal(str(min_composite_for_letter(self.min_grade())))


# External advisor services (pairs with python class)
class Advisor(models.Model):
    name = models.CharField(max_length=100, unique=True)
    python_class = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, default="")
    enabled = models.BooleanField(default=True)
    endpoint = models.CharField(max_length=500, default="")
    key = models.CharField(max_length=255, default="")
    blob = models.TextField(blank=True, default="")
    weight = models.DecimalField(max_digits=5, decimal_places=2, default=1.0)  # Win rate

    def is_enabled(self):
        self.refresh_from_db(fields=['enabled'])
        return self.enabled


# East-coast national exchanges only (exclude OTC, regional, etc.)
# NMS=NASDAQ Global Select, NYQ=NYSE, NAS=NASDAQ, NYS=NYSE MKT, NGM=NASDAQ Global Market, NCM=NASDAQ Capital Market
NATIONAL_EXCHANGES = ('NMS', 'NYQ', 'NAS', 'NYS', 'NGM', 'NCM')

# Basic stock at the core of everything
class Stock(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    company = models.CharField(max_length=200, blank=True, default="")
    exchange = models.CharField(max_length=32, blank=True, default="")
    sector = models.CharField(max_length=100, blank=True, default="")
    industry = models.CharField(max_length=200, blank=True, default="")
    website = models.URLField(blank=True, default="")
    beta = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    advisor = models.ForeignKey(Advisor, on_delete=models.DO_NOTHING, null=True, blank=True, default=None)
    updated = models.DateTimeField(auto_now=True)

    @classmethod
    def create(cls, symbol, advisor):

        logger = logging.getLogger(__name__)

        # Create base class
        stock = cls(symbol=symbol, advisor=advisor)

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Company
            company = info.get('longName') or info.get('shortName') or info.get('name')
            if company:
                stock.company = company[:200]

            # Exchange
            exchange = info.get('fullExchangeName') or info.get('exchange')
            if exchange:
                stock.exchange = exchange[:32]

            # Exclude non–east-coast national (e.g. OTC)
            exchange_code = info.get('exchange')
            if exchange_code not in NATIONAL_EXCHANGES:
                logger.warning(
                    "Stock.create: excluded %s (exchange=%s); only national exchanges allowed: %s",
                    symbol,
                    exchange_code or "(unknown)",
                    ", ".join(NATIONAL_EXCHANGES),
                )
                return None

            # Sector
            sector = info.get('sector')
            if sector:
                stock.sector = sector[:100]

            # Industry
            industry = info.get('industry')
            if industry:
                stock.industry = industry[:200]

            # Website
            website = info.get('website')
            if website:
                stock.website = website[:200]  # URLField will validate URL format

            # Beta
            beta = info.get('beta')
            if beta is not None:
                try:
                    stock.beta = Decimal(str(beta))
                except (ValueError, TypeError):
                    pass

            stock.save()

        except Exception as e:
            logger.warning(f"Could not fetch info for {symbol}: {e}")

        return stock

    def calc_trend(self, period="1d", interval="15m", hours=12):
        """
        Calculate trend using linear regression slope on price history.
        
        Args:
            period: Time period for history (default: "1d" for 1 day)
            interval: Data interval (default: "15m" for 15 minutes)
            hours: Optional - limit to last N hours of data (default: 12 hours)
        
        Returns:
            Decimal: normalized slope value (positive = uptrend, negative = downtrend, ~0 = sideways) or None if calculation fails or markets closed
        """
        import numpy as np
        from sklearn.linear_model import LinearRegression
        import pytz
        from datetime import datetime

        logger = logging.getLogger(__name__)

        try:
            if not self.price or self.price == 0:
                return None

            # Only check market hours for intraday periods (short periods with minute intervals)
            # For longer periods (e.g., 5d, 1mo, 1y), use historical data regardless of market status
            is_intraday = False
            if period in ["1d", "5d"] and interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
                is_intraday = True
            
            if is_intraday:
                # Check if markets are open or recently opened
                et = pytz.timezone('US/Eastern')
                now_et = datetime.now(et)
                
                # Check if it's a weekday (Monday=0, Sunday=6)
                if now_et.weekday() >= 5:  # Saturday or Sunday
                    logger.debug(f"Markets closed: weekend for {self.symbol}")
                    return None
                
                # Market hours: 9:30 AM - 4:00 PM ET
                market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
                
                # If before market open or after market close, return None
                if now_et.time() < market_open.time() or now_et.time() >= market_close.time():
                    logger.debug(f"Markets closed: outside trading hours for {self.symbol}")
                    return None
                
                # Check if market just opened (less than 2 hours of trading)
                hours_since_open = (now_et - market_open).total_seconds() / 3600
                if hours_since_open < 2:
                    logger.debug(f"Market recently opened (< 2 hours) for {self.symbol}")
                    return None

            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period=period, interval=interval)

            if hist.empty or 'Close' not in hist.columns:
                logger.warning(f"No history data for {self.symbol}")
                return None

            # For intraday, check if the last data point is recent (within last 30 minutes)
            if is_intraday and not hist.index.empty:
                et = pytz.timezone('US/Eastern')
                now_et = datetime.now(et)
                last_timestamp = hist.index[-1]
                # Convert to ET if needed
                if last_timestamp.tzinfo is None:
                    last_timestamp = pytz.UTC.localize(last_timestamp)
                last_timestamp_et = last_timestamp.astimezone(et)
                minutes_since_last = (now_et - last_timestamp_et).total_seconds() / 60
                
                if minutes_since_last > 30:
                    logger.debug(f"Data too stale ({minutes_since_last:.1f} minutes old) for {self.symbol}")
                    return None

            # If hours specified, slice to last N hours
            if hours is not None:
                # Calculate number of intervals for N hours
                # Parse interval (e.g., "15m", "30m", "60m", "1h")
                interval_str = interval.lower()
                if 'h' in interval_str:
                    interval_minutes = int(interval_str.replace('h', '')) * 60
                elif 'm' in interval_str:
                    interval_minutes = int(interval_str.replace('m', ''))
                else:
                    interval_minutes = 60  # Default to 60 minutes if unclear
                
                intervals_per_hour = 60 / interval_minutes
                num_intervals = int(hours * intervals_per_hour)
                
                if len(hist) > num_intervals:
                    hist = hist.tail(num_intervals)

            # Get price array
            prices = hist['Close'].values

            if len(prices) < 2:
                logger.warning(f"Not enough data points for {self.symbol}")
                return None

            # Create x values (time indices: 0, 1, 2, ...)
            y = np.array(prices)
            x = np.arange(len(prices)).reshape(-1, 1)

            # Fit linear regression
            model = LinearRegression().fit(x, y)
            slope = model.coef_[0]

            # Normalize by average price
            avg_price = hist['Close'].mean()
            normalized_slope = (slope / avg_price) * 100

            return Decimal(str(normalized_slope))

        except Exception as e:
            logger.warning(f"Could not calculate trend for {self.symbol}: {e}")
            return None

    def is_trending(self, lookback_days=5, volume_threshold_pct=0.2, min_volume=100000):
        """
        Check if stock is still trending based on trading volume.
        
        Args:
            lookback_days: Number of days to look back for average volume (default: 5)
            volume_threshold_pct: Current volume must be at least this % of average (default: 0.2 = 20%)
            min_volume: Absolute minimum volume threshold in shares (default: 100000)
        
        Returns:
            bool: True if stock is trending (high volume), False if not trending (low volume), None if can't determine
        """
        import yfinance as yf
        
        logger = logging.getLogger(__name__)
        
        try:
            if not self.price or self.price == 0:
                return None
            
            ticker = yf.Ticker(self.symbol)
            # Get last (lookback_days + 2) days to have enough data
            hist = ticker.history(period=f"{lookback_days + 2}d", interval="1d")
            
            if hist.empty or 'Volume' not in hist.columns or len(hist) < 2:
                logger.debug(f"Not enough volume data for {self.symbol}")
                return None
            
            # Get current day's volume (last row)
            current_volume = hist['Volume'].iloc[-1]
            
            # Calculate average volume over last N days (excluding current day)
            # Use last lookback_days rows, or all available if fewer
            available_days = min(lookback_days, len(hist) - 1)
            if available_days < 1:
                return None
            
            avg_volume = hist['Volume'].iloc[-(available_days + 1):-1].mean()
            
            if avg_volume <= 0:
                return None
            
            # Stock is "not trending" if current volume is very low
            # Threshold: less than volume_threshold_pct of recent average, or below absolute minimum
            volume_threshold = avg_volume * volume_threshold_pct
            absolute_threshold = max(volume_threshold, min_volume)
            
            is_trending = current_volume >= absolute_threshold
            
            return is_trending
            
        except Exception as e:
            logger.warning(f"Could not check trending status for {self.symbol}: {e}")
            return None

    def downturned(self, since_date, buy_price, giveback_pct=33.0, min_peak_gain_pct=5.0):
        """
        True if current gain has given back giveback_pct of peak gain since since_date.

        Example:
            buy=100, peak=112 (+12%), current=108 (+8%) -> giveback = 33.3%.
            Triggers when giveback_pct <= 33.3 and peak_gain >= min_peak_gain_pct.

        Args:
            since_date: datetime or date - start of window (e.g. discovery.created).
            buy_price: Decimal/float - cost basis anchor.
            giveback_pct: float - percent of peak gain allowed to be given back.
            min_peak_gain_pct: float - activate trailing only after this peak gain.
        """
        logger = logging.getLogger(__name__)

        try:
            buy = float(buy_price) if buy_price is not None else 0.0
            if buy <= 0:
                return False

            peak = self.peak_since(since_date)
            if peak is None or peak <= 0:
                return False

            current = float(self.price) if self.price is not None and self.price > 0 else None
            if current is None:
                return False

            peak_gain_pct = ((peak - buy) / buy) * 100.0
            if peak_gain_pct < float(min_peak_gain_pct):
                return False
            if peak_gain_pct <= 0:
                return False

            current_gain_pct = ((current - buy) / buy) * 100.0
            # PEAKED is intended to protect gains; do not sell at/below break-even.
            if current_gain_pct <= 0:
                return False
            giveback_ratio = (peak_gain_pct - current_gain_pct) / peak_gain_pct

            return giveback_ratio >= (float(giveback_pct) / 100.0)

        except Exception as e:
            logger.warning(f"Error checking profit downturn for {self.symbol}: {e}")
            return False

    def peak_since(self, since_date):
        """
        Maximum high since since_date (anchor), aligned with downturned().

        - Full trading days strictly after the anchor calendar day use daily highs.
        - On the anchor calendar day, only hourly bars whose start time is >= anchor
          (US/Eastern) contribute — so pre-discovery / pre-anchor session spikes do not
          inflate the peak when since_date is a datetime.
        - For a date-only anchor (no time), anchor is start of that calendar day in ET,
          matching prior daily-bar coverage for that day.
        - When the anchor day is before today, today's segment uses the full session so far.

        Returns:
            float or None: Peak price since since_date, or None if unavailable.
        """
        import yfinance as yf
        import pytz

        logger = logging.getLogger(__name__)

        try:
            et = pytz.timezone("US/Eastern")
            today_et = timezone.now().astimezone(et).date()

            if isinstance(since_date, dt_class):
                dt = since_date
                if timezone.is_naive(dt):
                    dt = timezone.make_aware(dt, timezone.utc)
                anchor_et = dt.astimezone(et)
            elif isinstance(since_date, date_class):
                anchor_et = et.localize(dt_class.combine(since_date, time_class.min))
            else:
                return None

            anchor_date = anchor_et.date()
            ticker = yf.Ticker(self.symbol)

            days_span = max((today_et - anchor_date).days + 5, 5)
            period = f"{days_span}d"

            # Full sessions strictly between anchor day and today (exclude anchor and today)
            hist_daily = ticker.history(period=period, interval="1d")
            peak_middle = None
            if not hist_daily.empty and "High" in hist_daily.columns:
                mask = (hist_daily.index.date > anchor_date) & (hist_daily.index.date < today_et)
                mid = hist_daily[mask]
                if not mid.empty:
                    peak_middle = float(mid["High"].max())

            hist_intraday = ticker.history(period=period, interval="1h")
            if hist_intraday.empty or "High" not in hist_intraday.columns:
                return peak_middle

            def bar_start_et(ts):
                if getattr(ts, "tzinfo", None):
                    return ts.astimezone(et)
                py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                return et.localize(py)

            # Anchor calendar day: highs only from bars starting at or after anchor_et
            highs_anchor = []
            for ts in hist_intraday.index:
                t_et = bar_start_et(ts)
                if t_et.date() != anchor_date:
                    continue
                if t_et >= anchor_et:
                    highs_anchor.append(float(hist_intraday.loc[ts, "High"]))
            anchor_peak = max(highs_anchor) if highs_anchor else None

            # Today (full session): only when anchor is before today's date
            today_full = None
            if anchor_date < today_et:
                highs_today = []
                for ts in hist_intraday.index:
                    t_et = bar_start_et(ts)
                    if t_et.date() == today_et:
                        highs_today.append(float(hist_intraday.loc[ts, "High"]))
                if highs_today:
                    today_full = max(highs_today)

            candidates = [p for p in (peak_middle, anchor_peak, today_full) if p is not None and p > 0]
            return max(candidates) if candidates else None

        except Exception as e:
            logger.warning(f"Error getting peak for {self.symbol}: {e}")
            return None

    def upturned(self):
        """
        Check if stock is showing upturn signals (higher close AND higher low vs previous day).

        Returns:
            bool: True if upturn detected (higher close AND higher low), False if not, None if can't determine
        """
        import yfinance as yf

        logger = logging.getLogger(__name__)

        try:
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period="5d", interval="1d")

            if hist.empty or len(hist) < 2:
                return False

            # Get previous day's close and low (second to last row)
            prev_close = float(hist['Close'].iloc[-2])
            prev_low = float(hist['Low'].iloc[-2])

            # Get current day's close and low (last row)
            today_close = float(hist['Close'].iloc[-1])
            today_low = float(hist['Low'].iloc[-1])

            # Upturn: higher close AND higher low (buyer momentum)
            return (today_close > prev_close) and (today_low > prev_low)

        except Exception as e:
            logger.warning(f"Error checking upturn for {self.symbol}: {e}")
            return False

    def refresh(self):

        logger = logging.getLogger(__name__)

        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.fast_info
            price = info.get('lastPrice') or info.get('regularMarketPrice')

            # Update price
            if price:
                self.price =  Decimal(str(price))

            self.save()

        except Exception as e:
            logger.warning(f"Could not auto-update {self.symbol}: {e}")

        return self


# Users stock holdingß
class Holding(models.Model):
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    fund = models.ForeignKey(Profile, null=True, blank=True, on_delete=models.DO_NOTHING)
    stock = models.ForeignKey(Stock, on_delete=models.DO_NOTHING)
    discovery = models.ForeignKey('Discovery', null=True, blank=True, on_delete=models.DO_NOTHING)
    shares = models.IntegerField(default=0)
    average_price = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    created = models.DateTimeField(auto_now_add=True, null=True, blank=True)  # When holding was first created (backfilled from first BUY trade)


class Watchlist(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Executed', 'Executed'),
        ('Excluded', 'Excluded'),
    ]

    created = models.DateTimeField(auto_now_add=True)
    advisor = models.ForeignKey(Advisor, on_delete=models.DO_NOTHING)
    stock = models.ForeignKey(Stock, on_delete=models.DO_NOTHING)
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)  # NEW
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')
    meta = models.JSONField(default=dict, null=True, blank=True)
    explanation = models.CharField(max_length=500)
    days = models.IntegerField()


# Stock health check
class Health(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='health')
    sa = models.ForeignKey('SmartAnalysis', on_delete=models.DO_NOTHING)
    created = models.DateTimeField(auto_now_add=True)
    score = models.DecimalField(max_digits=5, decimal_places=1)
    meta = models.JSONField(default=dict)

    class Meta:
        verbose_name_plural = "Health"
        indexes = [
            models.Index(fields=['stock', '-created']),
            models.Index(fields=['-created']),
        ]

    def __str__(self):
        return f"{self.stock.symbol} - Health: {self.score}"


class Assessment(models.Model):
    """Health v2 snapshot: six component scores and composite (0–100)."""

    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='assessments')
    created = models.DateTimeField(auto_now_add=True)

    financial = models.DecimalField(max_digits=5, decimal_places=1, null=True, blank=True)
    valuation = models.DecimalField(max_digits=5, decimal_places=1, null=True, blank=True)
    intrinsic = models.DecimalField(max_digits=5, decimal_places=1, null=True, blank=True)
    price = models.DecimalField(max_digits=5, decimal_places=1, null=True, blank=True)
    consensus = models.DecimalField(max_digits=5, decimal_places=1, null=True, blank=True)
    sector = models.DecimalField(max_digits=5, decimal_places=1, null=True, blank=True)

    score = models.DecimalField(
        max_digits=5,
        decimal_places=1,
        null=True,
        blank=True,
        help_text="Weighted v2 composite (0–100) from component scores.",
    )

    class Meta:
        indexes = [
            models.Index(fields=['stock', '-created']),
        ]

    def __str__(self):
        return f"{self.stock.symbol} — assessment @ {self.created}"


# Smart analysis session
class SmartAnalysis(models.Model):
    started = models.DateTimeField(auto_now_add=True)
    username = models.CharField(max_length=150)
    duration = models.DurationField(default=timedelta)


#   Advisor suggested stock
class Discovery(models.Model):
    sa = models.ForeignKey(SmartAnalysis, on_delete=models.DO_NOTHING)
    stock = models.ForeignKey(Stock, on_delete=models.DO_NOTHING)
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    advisor = models.ForeignKey(Advisor, on_delete=models.DO_NOTHING)
    health = models.ForeignKey(Health, null=True, blank=True, on_delete=models.SET_NULL)
    assessment = models.ForeignKey(Assessment, null=True, blank=True, on_delete=models.SET_NULL)
    created = models.DateTimeField(auto_now_add=True)
    weight = models.DecimalField(max_digits=5, decimal_places=2, default=1.0)
    explanation = models.TextField()

class SellInstruction(models.Model):
    choices = [
        ("STOP_PRICE", "Stop Loss (Price)"),
        ("TARGET_PRICE", "Target Price (Price)"),
        ("STOP_PERCENTAGE", "Stop Loss (Percentage)"),
        ("TARGET_PERCENTAGE", "Target Price (Percentage)"),
        ("AFTER_DAYS", "After Days"),
        ("DESCENDING_TREND", 'Descending trend'),
        ("END_WEEK", "End of current week"),
        ("END_DAY", "End of current day"),
        ("NOT_TRENDING", "No longer trending (low volume)"),
        ("TARGET_DIMINISHING", "Target Price (Diminishing)"),
        ("STOP_AUGMENTING", "Stop Loss (Augmenting)"),
        ("PERCENTAGE_DIMINISHING", "Target Price (Percentage diminishing)"),
        ("PERCENTAGE_AUGMENTING", "Stop Loss (Percentage augmenting)"),
        ("PROFIT_TARGET", "Target Profit (Fixed Dollar Amount"),
        ("PERCENTAGE_REBUY", "Loss - will gamble a Rebuy"),
        ("PROFIT_FLAT", "Price flatlined"),
        ("PEAKED", "Sell when down X% from peak since purchase"),
    ]

    discovery = models.ForeignKey(Discovery, on_delete=models.DO_NOTHING)
    instruction = models.CharField(max_length=25, choices=choices)
    value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)  # Keep temporarily, can be removed later
    value1 = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True, help_text='Primary value (price, percentage, days, threshold, etc.)')
    value2 = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True, help_text='Secondary value (e.g., max_days for diminishing/augmenting instructions)')


# Recommendation for advisors
class Recommendation(models.Model):

    sa = models.ForeignKey(SmartAnalysis, on_delete=models.DO_NOTHING)
    stock = models.ForeignKey(Stock, on_delete=models.DO_NOTHING)
    advisor = models.ForeignKey(Advisor, on_delete=models.DO_NOTHING)
    confidence = models.DecimalField(max_digits=3, decimal_places=2)
    created = models.DateTimeField(default=timezone.now)
    explanation = models.CharField(max_length=500)


# Stock consensus
class Consensus(models.Model):
    sa = models.ForeignKey(SmartAnalysis, on_delete=models.DO_NOTHING)
    stock = models.ForeignKey(Stock, on_delete=models.DO_NOTHING)
    recommendations = models.IntegerField()
    avg_confidence = models.DecimalField(max_digits=4, decimal_places=2)
    tot_confidence = models.DecimalField(max_digits=5, decimal_places=2)


class Trade(models.Model):
    ACTION = [
        ('BUY', 'Suggest buy'),
        ('SELL', 'Suggest sell')
    ]

    sa = models.ForeignKey(SmartAnalysis, on_delete=models.DO_NOTHING)
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    fund = models.ForeignKey(Profile, null=True, blank=True, on_delete=models.DO_NOTHING)
    stock = models.ForeignKey(Stock, on_delete=models.DO_NOTHING)
    discovery = models.ForeignKey('Discovery', null=True, blank=True, on_delete=models.DO_NOTHING)
    created = models.DateTimeField(auto_now_add=True)
    action = models.CharField(max_length=20, choices=ACTION)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    shares = models.IntegerField()
    cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True,
                                help_text="Average cost basis at time of SELL (for P&L calculation)")
    explanation = models.CharField(max_length=256, null=True, blank=True)


class PushDevice(models.Model):
    """FCM / provider registration tokens; multiple devices per user."""

    class Platform(models.TextChoices):
        IOS = 'ios', 'iOS'
        ANDROID = 'android', 'Android'

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='push_devices')
    token = models.CharField(
        max_length=512,
        unique=True,
        db_index=True,
        help_text='Provider registration token; globally unique.',
    )
    platform = models.CharField(max_length=16, choices=Platform.choices)
    environment = models.CharField(
        max_length=16,
        blank=True,
        default='',
        help_text='Optional: sandbox vs production for native APNS routing.',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['user', 'updated_at']),
        ]

    def __str__(self) -> str:
        suffix = self.token[-12:] if len(self.token) > 12 else self.token
        return f'{self.user_id} {self.platform} …{suffix}'


class Snapshot(models.Model):
    """Daily snapshot of user's portfolio (cash + holdings value)"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    fund = models.ForeignKey(Profile, null=True, blank=True, on_delete=models.DO_NOTHING)
    created = models.DateTimeField(auto_now_add=True)
    date = models.DateField()  # For easy daily filtering
    cash_value = models.DecimalField(max_digits=10, decimal_places=2)
    holdings_value = models.DecimalField(max_digits=10, decimal_places=2)
    trade_cumulative = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0'))
    trade_daily = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0'))
    
    class Meta:
        # One snapshot per fund per day (funds are the portfolio scope; user is denormalized).
        unique_together = [['fund', 'date']]
        indexes = [
            models.Index(fields=['user', '-date']),
        ]

