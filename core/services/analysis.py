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

logger = logging.getLogger(__name__)

# Calculate allowance
def calc_allowance(user, profile):
    allowance = profile.investment * Decimal(str(Profile.RISK[profile.risk]["allowance"]))
    return allowance

#  Build consensus on given stock
def build_consensus(sa, advisors, stock):

    # Only build consensus once for stock in a session (saves on API calls)
    consensus = Consensus.objects.filter(sa_id=sa.id, stock_id=stock.id).first()
    if consensus is not None:
        return consensus

    logger.info(f"Building consensus for stock {stock.symbol}")

    # Gather advice from external financial services
    for a in advisors:
        a.analyze(sa, stock)

    # Verify recommendations for this stock
    if Recommendation.objects.filter(sa=sa.id,stock_id=stock.id).first() is None:
        logger.info(f"No recommendations for stock {stock.symbol}")
        return None

    # Collate advice and form a consensus
    with connection.cursor() as cursor:
        cursor.execute('''insert into core_consensus(sa_id, stock_id, recommendations, tot_confidence, avg_confidence) 
            select %s, %s, count(*) as recommendations,
            sum(case when confidence > 1 then 1 else confidence end) as tot_confidence,
            avg(case when confidence > 1 then 1 else confidence end) as avg_confidence
            from core_recommendation
            where sa_id = %s and stock_id = %s
        ''', [sa.id, stock.id, sa.id, stock.id])
    # TODO research alternative to raw SQL

    return Consensus.objects.filter(sa=sa,stock=stock).first()

def analyze_holdings(sa, users, advisors):
    logger.info(f"Analyzing holdings for SA session {sa.id}")

    for u in users:
        profile = Profile.objects.get(user=u)

        allowance = calc_allowance(u, profile)
        sell_below = Decimal(str(Profile.RISK[profile.risk]["confidence_low"]))
        buy_from = Decimal(str(Profile.RISK[profile.risk]["confidence_high"]))

        holdings = list(Holding.objects.filter(user=u).select_related("stock"))
        if not holdings:
            continue

        candidates = []
        for holding in holdings:
            if holding.shares <= 0:
                continue

            consensus = build_consensus(sa, advisors, holding.stock)
            if consensus is None or consensus.avg_confidence is None:
                continue

            if consensus.avg_confidence < buy_from:
                candidates.append((holding, consensus))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[1].avg_confidence)

        for holding, consensus in candidates:
            profile.refresh_from_db(fields=["cash"])

            sell_for_confidence = consensus.avg_confidence < sell_below
            sell_for_cash = profile.cash < allowance

            if not (sell_for_confidence or sell_for_cash):
                if profile.cash >= allowance:
                    break
                continue

            execute_sell(sa, u, profile, consensus, holding)
            profile.refresh_from_db(fields=["cash"])

            if profile.cash >= allowance:
                break


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
        allowance = min(calc_allowance(u, profile), profile.cash)
        buy_from = Profile.RISK[profile.risk]["confidence_high"]

        # Get shares to buy based on high confidence
        consensus = Consensus.objects.filter(sa=sa.id, avg_confidence__gte=buy_from)

        # Get tot_consensus so buy allowance can be decided by each stocks confidence score
        tot_confidence = consensus.aggregate(Sum('avg_confidence'))['avg_confidence__sum']

        # Do the buy process
        for c in consensus:
            execute_buy(sa, u, c, allowance, tot_confidence, c.avg_confidence)


