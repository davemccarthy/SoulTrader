"""
Stock Analysis Service
"""

import logging
from decimal import Decimal
from datetime import datetime
from pytz import timezone as tz
from django.utils import timezone
from core.models import Stock, Holding, Discovery, Profile, Advisor
from core.services.execution import execute_buy, execute_sell

logger = logging.getLogger(__name__)


def analyse_target(discovery, holding, target):
    stock = holding.stock
    current = holding.stock.price
    buy_price = holding.average_price if holding.average_price else discovery.price

    # Case 1: Targets should only trigger sells at a profit, not at a loss
    if current < buy_price:
        return False

    # Case 2: Price >= target and downturn detected (down pullback_pct from peak since purchase; intraday today, daily history)
    if current >= target and stock.downturned(discovery.created, pullback_pct=5.0):
        return True

    # Case 3: Price < target but previously peaked at/above target (protect gains)
    if current < target and stock.peaked(discovery.created, target):
        return True

    return False


def analyze_holdings(sa, users, advisors):
    logger.info(f"Analyzing holdings for SA session {sa.id}")

    # Check if we're in the last 30 minutes of trading day (3:30 PM ET onwards)
    et = tz('US/Eastern')
    now_et = timezone.now().astimezone(et)
    current_time = now_et.time()
    weekday = now_et.weekday()
    end_day = False
    end_week = False

    # Only check on weekdays (0=Monday, 4=Friday)
    if weekday < 5:
        # Check if after 3:30 PM ET (last 30 minutes of trading)
        end_day_check_time = datetime.strptime("15:30", "%H:%M").time()

        if current_time >= end_day_check_time:
            end_day = True

    # Check if it's Friday and market is open (anytime during trading hours)
    if weekday == 4:  # Friday
        market_open_time = datetime.strptime("09:30", "%H:%M").time()
        market_close_time = datetime.strptime("16:00", "%H:%M").time()

        # Check if within market hours (9:30 AM - 4:00 PM ET)
        if market_open_time <= current_time < market_close_time:
            end_week = True

    # Iterate thru users
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
                    # Calculate days_held and buy_price for dynamic instructions
                    days_held = (timezone.now() - discovery.created).days if discovery.created else 0
                    buy_price = holding.average_price if holding.average_price else discovery.price

                    for instruction in instructions:
                        if instruction.instruction in ['STOP_PRICE', 'STOP_PERCENTAGE']:
                            if instruction.value1 and holding.stock.price < instruction.value1:
                                execute_sell(sa, user, profile, holding, f"{holding.stock.symbol} fell to stop-loss of ${instruction.value1:.2f}")
                                break

                        elif instruction.instruction in ['TARGET_PRICE', 'TARGET_PERCENTAGE']:
                            if analyse_target(discovery, holding, instruction.value1):
                                execute_sell(sa, user, profile, holding, f"{holding.stock.symbol} reached target price of ${instruction.value1:.2f}")
                                break

                        elif instruction.instruction in ['TARGET_DIMINISHING', 'PERCENTAGE_DIMINISHING']:
                            # Calculate diminishing target: original_target → buy_price over max_days
                            # value1 = original target price, value2 = max_days
                            if instruction.value1 and buy_price:
                                max_days = int(instruction.value2) if instruction.value2 is not None else 14
                                if days_held <= max_days:
                                    progress = float(days_held) / float(max_days) if max_days > 0 else 1.0
                                    original_target = float(instruction.value1)
                                    current_target = original_target - progress * (original_target - float(buy_price))
                                else:
                                    current_target = float(buy_price)  # After max_days, target = buy_price (break-even)

                                if analyse_target(discovery, holding, Decimal(str(current_target))):
                                    execute_sell(sa, user, profile, holding, f"{holding.stock.symbol} downturn detected at diminishing target ${current_target:.2f} (day {days_held}/{max_days})")
                                    break
                            else:
                                logger.warning(f"TARGET_DIMINISHING instruction {instruction.id} missing required fields (value1 or buy_price)")

                        elif instruction.instruction == 'PROFIT_TARGET':
                            # Calculate target profit based on average spend and ratio
                            # value1 = ratio (e.g., 0.10 for 10% profit)
                            if instruction.value1 and holding.shares > 0:
                                ratio = Decimal(str(instruction.value1))
                                base_allowance = profile.average_spend()
                                target_value = base_allowance * (Decimal('1.0') + ratio)
                                target_price = target_value / Decimal(str(holding.shares))
                                
                                if analyse_target(discovery, holding, target_price):
                                    execute_sell(sa, user, profile, holding, f"{holding.stock.symbol} reached profit target (target price: ${target_price:.2f})")
                                    break
                            else:
                                logger.warning(f"PROFIT_TARGET instruction {instruction.id} missing required fields (value1 or shares)")


                        elif instruction.instruction in ['STOP_AUGMENTING', 'PERCENTAGE_AUGMENTING']:
                            # Calculate augmenting stop: original_stop → buy_price over max_days
                            # value1 = original stop price, value2 = max_days
                            if instruction.value1 and buy_price:
                                max_days = int(instruction.value2) if instruction.value2 is not None else 28
                                if days_held <= max_days:
                                    progress = float(days_held) / float(max_days) if max_days > 0 else 1.0
                                    original_stop = float(instruction.value1)
                                    current_stop = original_stop + progress * (float(buy_price) - original_stop)
                                else:
                                    current_stop = float(buy_price)  # After max_days, stop = buy_price (break-even)
                                
                                if holding.stock.price < Decimal(str(current_stop)):
                                    execute_sell(sa, user, profile, holding, 
                                                f"{holding.stock.symbol} hit augmenting stop-loss of ${current_stop:.2f} (day {days_held}/{max_days})")
                                    break
                            else:
                                logger.warning(f"STOP_AUGMENTING instruction {instruction.id} missing required fields (value1 or buy_price)")

                        elif instruction.instruction == 'PERCENTAGE_REBUY':
                            # Rebuy on price drop (alternative to stop-loss)
                            # value1 = drop percentage (e.g., 0.10 for 10% drop)
                            # value2 = rebuy percentage of cost basis (e.g., 0.10 for 10% of cost basis)
                            if instruction.value1 and instruction.value2 and holding.shares > 0:
                                drop_pct = Decimal(str(instruction.value1))
                                rebuy_pct = Decimal(str(instruction.value2))
                                cost_basis = Decimal(str(holding.average_price)) * Decimal(str(holding.shares))
                                drop_threshold_price = Decimal(str(buy_price)) * (Decimal('1.0') - drop_pct)
                                
                                # Check if price has dropped by value1% from average_price
                                if holding.stock.price <= drop_threshold_price:
                                    rebuy_amount = cost_basis * rebuy_pct

                                    health1 = holding.stock.latest_health()
                                    health2 = advisors[0].health_check(holding.stock,sa)

                                    if health1 is None or health2 is None:
                                        logger.warning(f"PERCENTAGE_REBUY missing health")
                                        continue

                                    if health2.score < health1.score:
                                        logger.warning(f"PERCENTAGE_REBUY failed health")
                                        continue

                                    # Force a REBUY
                                    execute_buy(sa, user, holding.stock, rebuy_amount, f"Rebuying {rebuy_pct*100:.0f}% after drop", True)
                            else:
                                logger.warning(f"PERCENTAGE_REBUY instruction {instruction.id} missing required fields (value1, value2, or shares)")

                        elif instruction.instruction == 'PROFIT_FLAT':
                            # Sell if price is flat (low volatility) and in profit
                            # value1 = X (percentage threshold for price range, e.g., 0.05 for 5%)
                            # value2 = Y (evaluation period in days, e.g., 30)
                            if instruction.value1 and instruction.value2:
                                range_threshold_pct = Decimal(str(instruction.value1))
                                evaluation_days = int(instruction.value2)
                                current_price = holding.stock.price
                                
                                # Only check if stock has been held for at least the evaluation period
                                if days_held < evaluation_days:
                                    continue  # Skip check - not enough time has passed
                                
                                # Only check if in profit
                                if current_price >= buy_price:
                                    try:
                                        import yfinance as yf
                                        ticker = yf.Ticker(holding.stock.symbol)
                                        # Get enough days (add buffer for weekends/holidays)
                                        period_days = evaluation_days + 10
                                        hist = ticker.history(period=f"{period_days}d", interval="1d")
                                        
                                        if not hist.empty and len(hist) >= evaluation_days:
                                            # Get last Y days of close prices
                                            prices = hist['Close'].tail(evaluation_days).values
                                            
                                            max_price = Decimal(str(float(prices.max())))
                                            min_price = Decimal(str(float(prices.min())))
                                            avg_price = Decimal(str(float(prices.mean())))
                                            
                                            # Calculate price range
                                            price_range = max_price - min_price
                                            
                                            # Check if range is within X% of average (flat)
                                            if avg_price > 0:
                                                range_threshold = avg_price * range_threshold_pct
                                                
                                                if price_range <= range_threshold:
                                                    execute_sell(sa, user, profile, holding, f"Flat near {range_threshold_pct:.0f}% over {evaluation_days} days")
                                                    continue
                                    except Exception as e:
                                        logger.warning(f"Error checking PROFIT_FLAT for {holding.stock.symbol}: {e}")
                                        continue
                            else:
                                logger.warning(f"PROFIT_FLAT instruction {instruction.id} missing required fields (value1 or value2)")

                        elif instruction.instruction == 'AFTER_DAYS':
                            days_held = (timezone.now() - discovery.created).days
                            if instruction.value1 and days_held >= int(instruction.value1):
                                execute_sell(sa, user, profile, holding, f"{holding.stock.symbol} after holding for {days_held} days (target: {int(instruction.value1)} days)")
                                break

                        elif instruction.instruction == 'DESCENDING_TREND':
                            trend = holding.stock.calc_trend(hours=2)

                            if instruction.value1 and trend and trend < instruction.value1:
                                execute_sell(sa, user, profile, holding, f"{holding.stock.symbol} descending detected of ${instruction.value1:.2f}")
                                break

                        elif instruction.instruction == 'NOT_TRENDING':
                            trending = holding.stock.is_trending()
                            if trending is False:  # Explicitly False (not None)
                                execute_sell(sa, user, profile, holding, f"{holding.stock.symbol} no longer trending (low volume)")
                                break

                        elif instruction.instruction == 'END_DAY' and end_day:
                            if instruction.value1 and holding.stock.price >= instruction.value1:
                                execute_sell(sa, user, profile, holding, f"{holding.stock.symbol} end of day sell target: ${instruction.value1:.2f}")
                                break

                        elif instruction.instruction == 'END_WEEK' and end_week:
                            # Only apply END_WEEK if holding has been held for at least 7 days
                            if holding.created:
                                days_held = (timezone.now() - holding.created).days
                                if days_held >= 7:
                                    if instruction.value1 and holding.stock.price >= instruction.value1:
                                        execute_sell(
                                            sa, user, profile, holding,
                                            f"{holding.stock.symbol} end of week sell target: ${instruction.value1:.2f} (held {days_held} days)"
                                        )
                                        break
                            # If holding.created is None, skip END_WEEK (edge case - no BUY trade found)

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

    # Clear Polygon cache at start of discovery run to ensure fresh data
    from core.services.advisors.advisor import AdvisorBase
    AdvisorBase.clear_polygon_cache()

    # 1. Look for new stock
    for a in advisors:
        logger.info(f"Discovery ------------- {a.advisor.name}")
        a.discover(sa)

    # 2. Filter stocks to buy on a per user basis
    for u in users:
        logger.info(f"Buying ------------- {u.username}")

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
        base_allowance = profile.average_spend()
        risk_weight = Decimal(str(risk_settings["weight"]))

        # 4. Iterate through unique discoveries and calculate allowance per discovery
        for discovery in unique_discoveries:
            # Get most recent health check for this stock
            health_check = discovery.stock.latest_health()

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

            # Extract first clause from discovery explanation as summary
            explanation = discovery.explanation.split(" | ")[0].strip() if discovery.explanation else f"Stock health score {health_check.score} exceeded minimum score {min_health}"

            # Call execute_buy
            execute_buy(sa, u, discovery.stock, allowance, explanation)
