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
from datetime import timedelta
from decimal import Decimal

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

    # Risky business
    RISK = {
        "CONSERVATIVE": 40,
        "MODERATE": 30,
        "AGGRESSIVE": 20,
        "RECKLESS": 10
    }

    # But / sell sentiment
    SENTIMENT = {
        "STRONG_BULL": 1.4,
        "BULL": 1.2,
        "STAG": 1.0,
        "AUTO": 1.0,
        "BEAR": 0.8,
        "STRONG_BEAR": 0.6
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

    def min_score(self):
        min_score = Decimal(Profile.RISK[self.risk])
        return min_score


# External advisor services (pairs with python class)
class Advisor(models.Model):
    name = models.CharField(max_length=100, unique=True)
    python_class = models.CharField(max_length=100, unique=True)
    enabled = models.BooleanField(default=True)
    endpoint = models.CharField(max_length=500, default="")
    key = models.CharField(max_length=255, default="")
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

    def downturned(self, since_date, pullback_pct=5.0):
        """
        True if current price is down pullback_pct from the peak since since_date.
        Used for take-profit: sell when price has given back X% from the high since purchase.
        Intraday for current day (today's high from 1h bars), daily for history.

        Args:
            since_date: datetime or date - start of window (e.g. discovery.created).
            pullback_pct: float - downturn threshold, e.g. 5.0 = 5% down from peak.

        Returns:
            bool: True if current price <= peak * (1 - pullback_pct/100), False otherwise.
        """
        import yfinance as yf
        from datetime import datetime
        import pytz

        logger = logging.getLogger(__name__)

        try:
            et = pytz.timezone("US/Eastern")
            today_et = datetime.now(et).date()
            start_date = since_date.date() if isinstance(since_date, datetime) else since_date

            ticker = yf.Ticker(self.symbol)

            # Daily: peak from daily bars (since_date up to and including yesterday)
            days_back = (today_et - start_date).days + 5
            period = f"{days_back}d" if days_back > 1 else "5d"
            hist_daily = ticker.history(period=period, interval="1d")
            peak_daily = None
            if not hist_daily.empty and "High" in hist_daily.columns:
                # Filter to start_date <= date < today_et (historical only; today from intraday)
                hist_daily = hist_daily[hist_daily.index.date >= start_date]
                hist_daily = hist_daily[hist_daily.index.date < today_et]
                if not hist_daily.empty:
                    peak_daily = float(hist_daily["High"].max())

            # Intraday: today's high so far (1h bars)
            today_high = None
            hist_intraday = ticker.history(period="5d", interval="1h")
            if not hist_intraday.empty and "High" in hist_intraday.columns:
                # Filter to rows where date (in ET) is today
                today_mask = []
                for ts in hist_intraday.index:
                    if getattr(ts, "tzinfo", None):
                        d = ts.astimezone(et).date()
                    else:
                        d = ts.date() if hasattr(ts, "date") else ts
                    today_mask.append(d == today_et)
                if any(today_mask):
                    hist_today = hist_intraday[today_mask]
                    today_high = float(hist_today["High"].max())

            peak = None
            if peak_daily is not None and today_high is not None:
                peak = max(peak_daily, today_high)
            elif peak_daily is not None:
                peak = peak_daily
            elif today_high is not None:
                peak = today_high

            if peak is None or peak <= 0:
                return False

            current = (
                float(self.price)
                if self.price is not None and self.price > 0
                else (float(hist_daily["Close"].iloc[-1]) if not hist_daily.empty else None)
            )
            if current is None:
                return False

            threshold = peak * (1 - pullback_pct / 100)
            return current <= threshold

        except Exception as e:
            logger.warning(f"Error checking downturn for {self.symbol}: {e}")
            return False

    def peak_since(self, since_date):
        """
        Get the peak (max high) price since a given date.
        Uses daily bars plus intraday today's high to match downturned() logic.

        Returns:
            float or None: Peak price since since_date, or None if unavailable.
        """
        import yfinance as yf
        from datetime import datetime
        import pytz

        try:
            et = pytz.timezone("US/Eastern")
            today_et = datetime.now(et).date()
            start_date = since_date.date() if isinstance(since_date, datetime) else since_date

            ticker = yf.Ticker(self.symbol)

            # Daily: peak from daily bars
            days_back = (today_et - start_date).days + 5
            period = f"{days_back}d" if days_back > 1 else "5d"
            hist_daily = ticker.history(period=period, interval="1d")
            peak_daily = None
            if not hist_daily.empty and "High" in hist_daily.columns:
                hist_daily = hist_daily[hist_daily.index.date >= start_date]
                hist_daily = hist_daily[hist_daily.index.date < today_et]
                if not hist_daily.empty:
                    peak_daily = float(hist_daily["High"].max())

            # Intraday: today's high so far
            today_high = None
            hist_intraday = ticker.history(period="5d", interval="1h")
            if not hist_intraday.empty and "High" in hist_intraday.columns:
                today_mask = []
                for ts in hist_intraday.index:
                    if getattr(ts, "tzinfo", None):
                        d = ts.astimezone(et).date()
                    else:
                        d = ts.date() if hasattr(ts, "date") else ts
                    today_mask.append(d == today_et)
                if any(today_mask):
                    hist_today = hist_intraday[today_mask]
                    today_high = float(hist_today["High"].max())

            if peak_daily is not None and today_high is not None:
                return max(peak_daily, today_high)
            if peak_daily is not None:
                return peak_daily
            if today_high is not None:
                return today_high
            return None

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

    def peaked(self, since_date, target_price):
        """
        Check if stock reached a target price on or after a given date.

        Args:
            since_date: datetime or date object - check from this date onwards
            target_price: Decimal or float - price to check if stock reached

        Returns:
            bool: True if stock's high reached target_price on or after since_date, False otherwise, None if can't determine
        """
        import yfinance as yf
        from datetime import datetime, timedelta

        logger = logging.getLogger(__name__)

        try:
            # Convert since_date to datetime if it's a date
            if isinstance(since_date, datetime):
                start_date = since_date.date()
            else:
                start_date = since_date

            # Get enough history to cover since_date (add buffer for weekends/holidays)
            days_back = (datetime.now().date() - start_date).days + 5
            period = f"{days_back}d" if days_back > 1 else "5d"

            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period=period, interval="1d")

            if hist.empty:
                return None

            # Convert target_price to float for comparison
            target = float(target_price)

            # Filter to dates on or after since_date
            hist_dates = hist.index.date if hasattr(hist.index[0], 'date') else [d.date() for d in hist.index]

            # Get highs for dates >= since_date
            for i, date in enumerate(hist_dates):
                if date >= start_date:
                    high = float(hist['High'].iloc[i])
                    if high >= target:
                        return True

            return False

        except Exception as e:
            logger.warning(f"Error checking peak for {self.symbol} (since {since_date}, target ${target_price}): {e}")
            return None


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
    created = models.DateTimeField(auto_now_add=True)
    consensus = models.ForeignKey(Consensus, null=True, blank=True, on_delete=models.DO_NOTHING)
    action = models.CharField(max_length=20, choices=ACTION)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    shares = models.IntegerField()
    cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True,
                                help_text="Average cost basis at time of SELL (for P&L calculation)")
    explanation = models.CharField(max_length=256, null=True, blank=True)


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


# Signal to auto-create Profile when User is created
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Automatically create Profile when User is created"""
    if created:
        Profile.objects.get_or_create(
            user=instance,
            defaults={
                'risk': 'MODERATE',
                'cash': Decimal('100000.00'),
                'investment': Decimal('100000.00')
            }
        )