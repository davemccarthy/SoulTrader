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
            "confidence_high": 0.85,
            "confidence_low": 0.6,
            "stocks": 50
        },
        "MODERATE": {
            "confidence_high": 0.7,
            "confidence_low": 0.55,
            "stocks": 40
        },
        "AGGRESSIVE": {
            "confidence_high": 0.55,
            "confidence_low": 0.55,
            "stocks": 30
        },
        "EXPERIMENTAL": {
            "confidence_high": 0.0,
            "confidence_low": 0.0,
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

    def is_enabled(self):
        self.refresh_from_db(fields=['enabled'])
        return self.enabled

# Basic stock at the core of everything
class Stock(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    company = models.CharField(max_length=200)
    exchange = models.CharField(max_length=32)
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    trend = models.DecimalField(max_digits=5, decimal_places=4, null=True, blank=True)
    advisor = models.ForeignKey(Advisor, on_delete=models.DO_NOTHING, null=True, blank=True, default=None)
    updated = models.DateTimeField(auto_now=True)

    def calc_trend(self, period="1d", interval="15m"):
        """
        Calculate trend using linear regression slope on price history.

        Args:
            period: Time period for history (default: "2d" for 2 days)
            interval: Data interval (default: "60m" for hourly)

        Returns:
            Decimal: slope value (positive = uptrend, negative = downtrend, ~0 = sideways) or None if calculation fails
        """
        import numpy as np
        from sklearn.linear_model import LinearRegression

        logger = logging.getLogger(__name__)

        try:
            if not self.price or self.price == 0:
                return None

            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period=period, interval=interval)

            if hist.empty or 'Close' not in hist.columns:
                logger.warning(f"No history data for {self.symbol}")
                return None

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

            # Calculate weighted trend (combines trend and velocity)  1-day 15-minute history
            self.trend = self.calc_trend()
            self.save()

            logger.info(f"Updated {self.symbol} {self.price} (trend: {self.trend})") # TMP
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


# Smart analysis session
class SmartAnalysis(models.Model):
    started = models.DateTimeField(auto_now_add=True)
    username = models.CharField(max_length=150)
    duration = models.DurationField(default=timedelta)
    # Other stats
    # allowance (this sessions spend)
    # spent


#   Advisor suggested stock
class Discovery(models.Model):
    sa = models.ForeignKey(SmartAnalysis, on_delete=models.DO_NOTHING)
    stock = models.ForeignKey(Stock, on_delete=models.DO_NOTHING)
    advisor = models.ForeignKey(Advisor, on_delete=models.DO_NOTHING)
    created = models.DateTimeField(auto_now_add=True)
    explanation = models.CharField(max_length=1000)

class SellInstruction(models.Model):
    choices = [
        ("STOP_LOSS", "Stop Loss"),
        ("TARGET_PRICE", "Target Price"),
        ("CS_FLOOR", "CS Floor"),
        ("AFTER_DAYS", "After Days"),
        ("DESCENDING_TREND", 'Dedcending trend')
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
    consensus = models.ForeignKey(Consensus, null=True, blank=True, on_delete=models.DO_NOTHING)
    action = models.CharField(max_length=20, choices=ACTION)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    shares = models.IntegerField()
    explanation = models.CharField(max_length=256, null=True, blank=True)
    # TODO NO DATE STAMP


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