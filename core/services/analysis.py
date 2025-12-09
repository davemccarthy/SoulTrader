"""
Stock Analysis Service
"""

import logging
from decimal import Decimal
from django.utils import timezone
from core.models import Stock, Holding, Discovery, Profile, Advisor
from core.services.execution import execute_buy, execute_sell

logger = logging.getLogger(__name__)

def analyze_holdings(sa, users, advisors):
    logger.info(f"Analyzing holdings for SA session {sa.id}")

    for user in users:
        profile = Profile.objects.get(user=user)

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
                        if instruction.instruction in ['STOP_PRICE', 'STOP_PERCENTAGE']:
                            if holding.stock.price < instruction.value:
                                execute_sell(sa, user, profile, holding, f"{holding.stock.symbol} fell to stop-loss of ${instruction.value:.2f}")
                                break

                        elif instruction.instruction in ['TARGET_PRICE', 'TARGET_PERCENTAGE']:
                            if holding.stock.price >= instruction.value:
                                execute_sell(sa, user, profile, holding, f"{holding.stock.symbol} reached target price of ${instruction.value:.2f}")
                                break

                        elif instruction.instruction == 'AFTER_DAYS':
                            days_held = (timezone.now() - discovery.created).days
                            if days_held >= instruction.value:
                                execute_sell(sa, user, profile, holding, f"{holding.stock.symbol} after holding for {days_held} days (target: {instruction.value} days)")
                                break

                        elif instruction.instruction == 'DESCENDING_TREND':
                            trend = holding.stock.calc_trend(hours=2)

                            if trend and trend < instruction.value:
                                execute_sell(sa, user, profile, holding, f"{holding.stock.symbol} descending detection of ${instruction.value:.2f}")
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

    # 2. Filter stocks to buy on a per user basis
    for u in users:
        profile = Profile.objects.get(user=u)
        risk_settings = Profile.RISK[profile.risk]
        allowed_advisors = risk_settings.get("advisors", [])

        # 1. Filter discoveries by allowed advisors (if specified)
        discoveries_qs = Discovery.objects.filter(sa=sa)
        
        if allowed_advisors:
            # Get Advisor objects for the allowed advisor python_class values
            allowed_advisor_objects = Advisor.objects.filter(python_class__in=allowed_advisors)

            if allowed_advisor_objects.exists():
                discoveries_qs = discoveries_qs.filter(advisor__in=allowed_advisor_objects)
            else:
                logger.warning(f"User {u.username} ({profile.risk}): No advisors found matching {allowed_advisors}")
                continue

        # 3. Get filtered discoveries
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
            # Get most recent health check for this stock
            from core.models import Health
            health_check = Health.objects.filter(
                stock=discovery.stock
            ).order_by('-created').first()

            # Skip stocks without health check or with score below threshold
            if not health_check:
                logger.warning(f"Discovery {discovery.stock.symbol} by {discovery.advisor.name} has no health check")
                continue

            min_health = Decimal(str(risk_settings["min_health"]))

            if health_check.score < min_health:
                logger.info(
                    f"User {u.username}: Discovery {discovery.stock.symbol} by {discovery.advisor.name} "
                    f"health check score ({health_check.score}) below threshold ({min_health})"
                )
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

            logger.info(
                f"User {u.username}: Discovery {discovery.stock.symbol} by {discovery.advisor.name} "
                f"(advisor_weight={advisor_weight:.3f}, risk_weight={risk_weight:.3f}, allowance=${allowance:.2f})"
            )
            
            # Call execute_buy (signature may need updating)
            execute_buy(sa, u, discovery.stock, allowance)
