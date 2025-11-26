"""
Stock Analysis Service

Analyzes stocks using AI advisors and builds consensus
Maps to build_consensus() from your analysis.py
"""

import logging
import random
from decimal import Decimal

import h11
from django.db import connection
from django.db.models import Sum
from django.utils import timezone
from core.models import Stock, Holding, Discovery, Recommendation, Consensus, Trade, Profile
from core.services.execution import execute_buy, execute_sell

logger = logging.getLogger(__name__)


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

    for user in users:
        profile = Profile.objects.get(user=user)
        sell_below = Decimal(str(Profile.RISK[profile.risk]["confidence_low"]))

        for holding in Holding.objects.filter(user=user):

            # Latest prices
            holding.stock.refresh()

            # Get most recent discovery for this stock
            discovery = Discovery.objects.filter(stock=holding.stock).order_by('-created').first()
            if discovery:
                # Get all sell instructions for this discovery
                from core.models import SellInstruction
                instructions = SellInstruction.objects.filter(discovery=discovery)

                # Check sell conditions in priority order
                try:
                    for instruction in instructions:
                        if instruction.instruction == 'STOP_LOSS':
                            if holding.stock.price < instruction.value:
                                execute_sell(sa, user, profile, consensus, holding, f"{holding.stock.symbol} fell to stop-loss of ${instruction.value:.2f}")
                                break

                        elif instruction.instruction == 'TARGET_PRICE' and instruction.value != 0.0: # TMP CHECK
                            if holding.stock.price >= instruction.value:
                                execute_sell(sa, user, profile, consensus, holding, f"{holding.stock.symbol} reached target price of ${instruction.value:.2f}")
                                break

                        elif instruction.instruction == 'AFTER_DAYS':
                            days_held = (timezone.now() - discovery.created).days
                            if days_held >= instruction.value:
                                execute_sell(sa, user, profile, consensus, holding, f"{holding.stock.symbol} after holding for {days_held} days (target: {instruction.value} days)")
                                break

                        elif instruction.instruction == 'DESCENDING_TREND':
                            if holding.stock.trend < instruction.value:
                                execute_sell(sa, user, profile, consensus, holding, f"{holding.stock.symbol} descending detection of ${instruction.value:.2f}")
                                print(f"*************** HAVE SOLD {holding.stock.symbol} descending detection of ${holding.stock.trend:.2f}")
                                break

                        elif instruction.instruction == 'CS_FLOOR':

                            consensus = build_consensus(sa, advisors, holding.stock)
                            if consensus is None or consensus.avg_confidence is None:
                                continue

                            holding.consensus = consensus.avg_confidence
                            holding.save(update_fields=["consensus"])

                            if consensus.avg_confidence < sell_below:
                                execute_sell(sa, user, profile, consensus, holding, f"Consensus confidence ({consensus.avg_confidence:.2f}) fell below threshold ({sell_below:.2f})")
                                break
                except Exception as e:
                    logger.error(
                        f"Error checking sell instructions for {holding.stock.symbol} (holding {holding.id}): {e}",
                        exc_info=True
                    )
                    # Continue processing other holdings
                    continue


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
        buy_from = Profile.RISK[profile.risk]["confidence_high"]

        # Limit to stocks discovered in this SA and above confidence threshold
        discovered_stock_ids = Discovery.objects.filter(sa=sa).values_list('stock_id', flat=True)
        consensus = Consensus.objects.filter(
            sa=sa.id,
            stock_id__in=discovered_stock_ids,
            avg_confidence__gte=buy_from
        )

        # Calculate allowance based on target slot count
        allowance = len(consensus) * (profile.investment / Decimal(str(Profile.RISK[profile.risk]["stocks"])))

        # Get tot_consensus so buy allowance can be decided by each stocks confidence score
        tot_confidence = consensus.aggregate(Sum('avg_confidence'))['avg_confidence__sum']
        if not tot_confidence:
            continue

        # Do the buy process
        for c in consensus:
            # Buy stock
            execute_buy(sa, u, c, allowance, tot_confidence, c.avg_confidence)


