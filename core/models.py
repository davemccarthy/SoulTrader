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
from django.utils import timezone
from datetime import timedelta
from decimal import Decimal

import yfinance as yf
import logging

# Your models go here
# Keep them simple and aligned with your SQL queries in analysis.py

# Profile / user settings
class Profile(models.Model):

    # Risky business
    RISK = {
        "CONSERVATIVE": {
            "min_health": 50.0,  # Only top ~20% of your scores
            "advisors": ['StockStory', 'Polygon', 'FDA', 'Insider'],
            "weight": 1.0,
            "stocks": 50
        },
        "MODERATE": {
            "min_health": 40.0,  # Above average
            "advisors": ['StockStory', 'Polygon', 'FDA', 'Insider'],
            "weight": 1.00,
            "stocks": 40
        },
        "AGGRESSIVE": {
            "min_health": 30.0,  # Below average but not bottom
            "advisors": ['User', 'FDA', 'Insider','StockStory', 'Polygon'],
            "weight": 1.25,
            "stocks": 30
        },
        "EXPERIMENTAL": {
            "min_health": 0.0,  # No filter
            "advisors": ['Intraday'],  # Intraday momentum advisor for experimental users
            "weight": 1.0,
            "stocks": 40
        },
    }

    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    risk = models.CharField(max_length=20, choices=[(key, key.replace('_', ' ').title()) for key in RISK.keys()], default='MODERATE')
    investment = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('100000.00'))
    cash = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('100000.00'))


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


# Basic stock at the core of everything
class Stock(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    company = models.CharField(max_length=200)
    exchange = models.CharField(max_length=32)
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    advisor = models.ForeignKey(Advisor, on_delete=models.DO_NOTHING, null=True, blank=True, default=None)
    updated = models.DateTimeField(auto_now=True)

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

    def refresh(self):

        logger = logging.getLogger(__name__)

        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.fast_info
            price = info.get('lastPrice') or info.get('regularMarketPrice')

            # Update price
            if price:
                self.price =  Decimal(str(price))

            # Update company (and exchange) if missing
            if not self.company:
                full_info = ticker.info

                company = full_info.get('longName') or full_info.get('shortName')
                exchange = full_info.get('fullExchangeName') or full_info.get('exchange')

                if company:
                    self.company = company

                if exchange:
                    self.exchange = exchange

            # Calculate trend using last 12 hours of 15-minute history
            #self.trend = self.calc_trend()
            self.save()
        except Exception as e:
            logger.warning(f"Could not auto-update {self.symbol}: {e}")

        return self


# Users stock holdingß
class Holding(models.Model):
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    stock = models.ForeignKey(Stock, on_delete=models.DO_NOTHING)
    shares = models.IntegerField(default=0)
    average_price = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    consensus = models.DecimalField(max_digits=4, decimal_places=2, default=5.0)
    volatile = models.BooleanField(default=False)  # Your flag!

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
    created = models.DateTimeField(auto_now_add=True)
    weight = models.DecimalField(max_digits=5, decimal_places=2, default=1.0)
    explanation = models.CharField(max_length=1000)

class SellInstruction(models.Model):
    choices = [
        ("STOP_PRICE", "Stop Loss (Price)"),
        ("TARGET_PRICE", "Target Price (Price)"),
        ("STOP_PERCENTAGE", "Stop Loss (Percentage)"),
        ("TARGET_PERCENTAGE", "Target Price (Percentage)"),
        ("CS_FLOOR", "CS Floor"),
        ("AFTER_DAYS", "After Days"),
        ("DESCENDING_TREND", 'Descending trend'),
        ("END_WEEK", "End of current week"),
        ("END_DAY", "End of current day")
    ]

    discovery = models.ForeignKey(Discovery, on_delete=models.DO_NOTHING)
    instruction = models.CharField(max_length=20, choices=choices)
    value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)


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
    created = models.DateTimeField(auto_now_add=True)
    date = models.DateField()  # For easy daily filtering
    cash_value = models.DecimalField(max_digits=10, decimal_places=2)
    holdings_value = models.DecimalField(max_digits=10, decimal_places=2)
    trade_cumulative = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0'))
    trade_daily = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0'))
    
    class Meta:
        unique_together = [['user', 'date']]  # One snapshot per user per day
        indexes = [
            models.Index(fields=['user', '-date']),  # For efficient querying
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