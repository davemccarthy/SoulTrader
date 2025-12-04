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
from core.models import Stock, Holding, Discovery, Recommendation, Consensus, Trade, Profile, Advisor
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
                    consensus = None  # Initialize consensus variable

                    for instruction in instructions:
                        if instruction.instruction in ['STOP_PRICE', 'STOP_PERCENTAGE']:
                            if holding.stock.price < instruction.value:
                                execute_sell(sa, user, profile, consensus, holding, f"{holding.stock.symbol} fell to stop-loss of ${instruction.value:.2f}")
                                break

                        elif instruction.instruction in ['TARGET_PRICE', 'TARGET_PERCENTAGE']:
                            if holding.stock.price >= instruction.value:
                                execute_sell(sa, user, profile, consensus, holding, f"{holding.stock.symbol} reached target price of ${instruction.value:.2f}")
                                break

                        elif instruction.instruction == 'AFTER_DAYS':
                            days_held = (timezone.now() - discovery.created).days
                            if days_held >= instruction.value:
                                execute_sell(sa, user, profile, consensus, holding, f"{holding.stock.symbol} after holding for {days_held} days (target: {instruction.value} days)")
                                break

                        elif instruction.instruction == 'DESCENDING_TREND':

                            trend = holding.stock.calc_trend(hours=2)

                            if trend and trend < instruction.value:
                                execute_sell(sa, user, profile, consensus, holding, f"{holding.stock.symbol} descending detection of ${instruction.value:.2f}")
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

    # 1. Look for new stock
    for a in advisors:
        a.discover(sa)

    # 2. Build consensus on discovered stocl
    for d in Discovery.objects.filter(sa=sa):
        build_consensus(sa, advisors, d.stock)

    # 3. Filter stocks to buy on a per user basis
    for u in users:
        profile = Profile.objects.get(user=u)
        risk_settings = Profile.RISK[profile.risk]
        buy_from = risk_settings["confidence_high"]
        allowed_advisors = risk_settings.get("advisors", [])

        # 1. Filter discoveries by allowed advisors (if specified)
        discoveries_qs = Discovery.objects.filter(sa=sa)
        
        if allowed_advisors:
            # Get Advisor objects for the allowed advisor python_class values
            allowed_advisor_objects = Advisor.objects.filter(python_class__in=allowed_advisors)

            if allowed_advisor_objects.exists():
                discoveries_qs = discoveries_qs.filter(advisor__in=allowed_advisor_objects)
                logger.info(f"User {u.username} ({profile.risk}): Filtering to advisors: {allowed_advisors}")
            else:
                logger.warning(f"User {u.username} ({profile.risk}): No advisors found matching {allowed_advisors}")
                continue

        # 2. Get filtered discoveries (will check consensus for each)
        filtered_discoveries = list(discoveries_qs.select_related('advisor', 'stock'))
        
        if not filtered_discoveries:
            logger.info(f"User {u.username} ({profile.risk}): No discoveries from allowed advisors")
            continue

        # Deduplicate by stock - only process each stock once per user
        seen_stocks = set()
        unique_discoveries = []
        for discovery in filtered_discoveries:
            if discovery.stock_id not in seen_stocks:
                seen_stocks.add(discovery.stock_id)
                unique_discoveries.append(discovery)

        if not unique_discoveries:
            continue

        # 3. Base allowance calculation (to be enhanced with different risk methods)
        base_allowance = profile.investment / Decimal(str(risk_settings["stocks"]))
        risk_weight = Decimal(str(risk_settings["weight"]))

        # 4. Iterate through unique discoveries and calculate allowance per discovery
        for discovery in unique_discoveries:
            # Check if this discovery has consensus that passes threshold
            consensus = Consensus.objects.filter(
                sa=sa.id,
                stock_id=discovery.stock_id,
                avg_confidence__gte=buy_from
            ).first()

            if not consensus:
                logger.debug(f"Discovery {discovery.stock.symbol} by {discovery.advisor.name} did not pass consensus threshold ({buy_from})")
                continue

            # Get advisor weight (normalized)
            advisor_weight = discovery.advisor.weight

            # Calculate allowance with weighting
            allowance = base_allowance

            # Apply advisor weight and risk weight
            # Exaggerate: if advisor weight > 1.0, multiply by risk.weight; if < 1.0, divide by risk.weight
            if advisor_weight > Decimal('1.0'):
                allowance = allowance * (advisor_weight * risk_weight)
            else:
                allowance = allowance * (advisor_weight / risk_weight)

            # TODO: Will update execute_buy function later
            # For now, using existing signature - may need to adapt
            logger.info(
                f"User {u.username}: Discovery {discovery.stock.symbol} by {discovery.advisor.name} "
                f"(advisor_weight={advisor_weight:.3f}, risk_weight={risk_weight:.3f}, allowance=${allowance:.2f})"
            )
            
            # Call execute_buy (signature may need updating)
            execute_buy(sa, u, consensus, allowance)
