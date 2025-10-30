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

# Your models go here
# Keep them simple and aligned with your SQL queries in analysis.py

# Profile / user settings
class Profile(models.Model):
    SENTIMENT = [
        ('BEAR' ,'Market decline'),
        ('STAG', 'Mixed market'),
        ('BULL', 'Market optimism'),
    ]

    RISK = [
        ('CONSERVATIVE', 'Conservative'),
        ('MODERATE', 'Moderate'),
        ('AGGRESSIVE', 'Aggressive'),
    ]

    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    sentiment = models.CharField(max_length=20, choices=SENTIMENT, default='STAG')
    risk = models.CharField(max_length=20, choices=RISK, default='MODERATE')

    investment = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('100000.00'))
    cash = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('100000.00'))
    #allowance = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0.00'))


# External advisor services (pairs with python class)
class Advisor(models.Model):
    name = models.CharField(max_length=100, unique=True)
    python_class = models.CharField(max_length=100, unique=True)
    enabled = models.BooleanField(default=True)
    endpoint = models.CharField(max_length=500, default="")
    key = models.CharField(max_length=255, default="")


# Basic stock at the core of everything
class Stock(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    company = models.CharField(max_length=200)
    exchange = models.CharField(max_length=32, default='Unknown')
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    advisor = models.ForeignKey(Advisor, on_delete=models.DO_NOTHING, null=True, blank=True, default=None)
    updated = models.DateTimeField(auto_now=True)


# Users stock holdingß
class Holding(models.Model):
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    stock = models.ForeignKey(Stock, on_delete=models.DO_NOTHING)
    shares = models.IntegerField(default=0)
    average_price = models.DecimalField(max_digits=10, decimal_places=2,default=0.0)
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
    explanation = models.CharField(max_length=500)


# Recommendation for advisors
class Recommendation(models.Model):

    sa = models.ForeignKey(SmartAnalysis, on_delete=models.DO_NOTHING)
    stock = models.ForeignKey(Stock, on_delete=models.DO_NOTHING)
    advisor = models.ForeignKey(Advisor, on_delete=models.DO_NOTHING)
    confidence = models.DecimalField(max_digits=3, decimal_places=2)
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
    consensus = models.ForeignKey(Consensus, on_delete=models.DO_NOTHING)
    action = models.CharField(max_length=20, choices=ACTION)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    shares = models.IntegerField()