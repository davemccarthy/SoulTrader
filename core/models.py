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

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    sentiment = models.CharField(max_length=20, choices=SENTIMENT, default='STAG')
    risk = models.CharField(max_length=20, choices=RISK, default='MODERATE')

    investment = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('100000.00'))
    cash = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('100000.00'))
    #allowance = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0.00'))


# Basic stock at the core of everything
class Stock(models.Model):
    symbol = models.CharField(unique=True)
    company = models.CharField()
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    updated = models.DateTimeField(auto_now=True)


# Users stock holding
class Holding(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    shares = models.IntegerField(default=0)
    average_price = models.DecimalField(max_digits=10, decimal_places=2,default=0.0)
    volatile = models.BooleanField(default=False)  # Your flag!


# Smart analysis session
class SmartAnalysis(models.Model):
    started = models.DateTimeField(auto_now_add=True)
    username = models.CharField()
    duration = models.DurationField(default=timedelta)
    # Other stats
    # allowance (this sessions spend)
    # spent


# External advisor services (pairs with python class)
class Advisor(models.Model):
    name = models.CharField(unique=True)
    python_class = models.CharField(unique=True)
    enabled = models.BooleanField(default=True)
    endpoint = models.CharField(default="")
    key = models.CharField(default="")

#   Advisor suggested stock
class Discovery(models.Model):
    sa = models.ForeignKey(SmartAnalysis, on_delete=models.CASCADE)
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    advisor = models.ForeignKey(Advisor, on_delete=models.CASCADE)
    explanation = models.CharField()


# Recommendation for advisors
class Recommendation(models.Model):
    ACTION = [
        ('BUY', 'Suggest buy'),
        ('HOLD', 'Do not sell or buy'),
        ('SELL', 'Suggest sell'),
        ('STRONG_BUY', 'Definitely buy'),
        ('STRONG_SELL', 'Definitely sell')
    ]

    sa = models.ForeignKey(SmartAnalysis, on_delete=models.CASCADE)
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    advisor = models.ForeignKey(Advisor, on_delete=models.CASCADE)
    action = models.CharField(max_length=20, choices=ACTION, default='HOLD')
    confidence = models.DecimalField(max_digits=3, decimal_places=2)
    explanation = models.CharField()


# Stock consensus
class Consensus(models.Model):
    sa = models.ForeignKey(SmartAnalysis, on_delete=models.CASCADE)
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    recommendations = models.IntegerField()
    avg_confidence = models.DecimalField(max_digits=2, decimal_places=2)
    tot_confidence = models.DecimalField(max_digits=5, decimal_places=2)


class Trade(models.Model):
    ACTION = [
        ('BUY', 'Suggest buy'),
        ('SELL', 'Suggest sell')
    ]

    sa = models.ForeignKey(SmartAnalysis, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    consensus = models.ForeignKey(Consensus, on_delete=models.CASCADE)
    action = models.CharField(max_length=20, choices=ACTION, default='HOLD')
    price = models.DecimalField(max_digits=5, decimal_places=2)
    shares = models.IntegerField()