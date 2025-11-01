"""
Stock Analysis Service

Analyzes stocks using AI advisors and builds consensus
Maps to build_consensus() from your analysis.py
"""

import logging
import random
from decimal import Decimal

from django.db import connection
from django.db.models import Sum
from core.models import Stock, Holding, Discovery, Recommendation, Consensus, Trade, Profile
from core.services.execution import execute_buy, execute_sell
"""""
# Risky business
risk_settings = {
    "CONSERVATIVE" : {
        "allowance" : 0.1,
        "confidence_high": 0.8,
        "confidence_low": 0.6
    },
    "MODERATE" : {
        "allowance" : 0.2,
        "confidence_high": 0.7,
        "confidence_low": 0.55
    },
    "AGGRESSIVE" : {
        "allowance" : 0.4,
        "confidence_high": 0.55,
        "confidence_low": 0.4
    },
}
"""
logger = logging.getLogger(__name__)

#  Build consensus on given stock
def build_consensus(sa, advisors, stock):

    # Only build consensus once for stock in a session (saves on API calls)
    consensus = Consensus.objects.filter(sa_id=sa.id, stock_id=stock.id).first()
    if consensus is not None:
        return consensus
    # TODO I'm sure there's nicer way to do the above

    logger.info(f"Building consensus for stock {stock.symbol}")

    # Gather advice from external financial services
    for a in advisors:
        if a.advisor.enabled is True:
            a.analyze(sa, stock)

    # Collate advice and form a consensus
    with connection.cursor() as cursor:
        cursor.execute('''insert into core_consensus(sa_id, stock_id, recommendations, tot_confidence, avg_confidence) 
            select %s, %s, count(*) as recommendations, sum(confidence) as tot_confidence, 
            avg(confidence) as avg_confidence from core_recommendation where sa_id = %s and stock_id = %s
        ''', [sa.id, stock.id, sa.id, stock.id])
    # TODO research alternative to raw SQL

    return Consensus.objects.filter(sa=sa,stock=stock).first()

# Review current holding stocks
def analyze_holdings(sa, users, advisors):
    logger.info(f"Analyzing holdings for SA session {sa.id}")

    # 1. Filter stocks to sell on a per user basis
    for u in users:
        profile = Profile.objects.get(user=u)

        sell_below = Profile.RISK[profile.risk]["confidence_low"]

        for h in Holding.objects.filter(user=u):
            consensus = build_consensus(sa, advisors, h.stock)

            if consensus.avg_confidence < sell_below:
                execute_sell(sa, u, consensus, h)


# Discovery new stock
def analyze_discovery(sa, users, advisors):
    logger.info(f"Analyzing discovery for SA session {sa.id}")

    # 1. Look for new stock (only once per sa to save API calls)
    if not Discovery.objects.filter(sa=sa).exists():
        for a in advisors:
            a.discover(sa)

    # 2. Build consensus on discovered stocl
    for d in Discovery.objects.filter(sa=sa):
        build_consensus(sa, advisors, d.stock)

    # 3. Filter stocks to buy on a per user basis
    for u in users:
        profile = Profile.objects.get(user=u)
        allowance = profile.cash * Decimal(Profile.RISK[profile.risk]["allowance"])
        buy_from = Profile.RISK[profile.risk]["confidence_high"]

        # Get shares to buy based on high confidence
        consensus = Consensus.objects.filter(sa=sa.id, avg_confidence__gte=buy_from)

        # Get tot_consensus so buy allowance can be decided by each stocks confidence score
        tot_confidence = consensus.aggregate(Sum('avg_confidence'))['avg_confidence__sum']

        # Do the buy process
        for c in consensus:
            execute_buy(sa, u, c, allowance, tot_confidence, c.avg_confidence)


