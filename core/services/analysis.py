"""
Stock Analysis Service
"""

import logging
from decimal import Decimal
from datetime import datetime
from pytz import timezone as tz
from django.utils import timezone
from core.models import Holding, Discovery, Advisor, Profile
from core.services.execution import execute_buy, execute_sell
from core.services.llm.gemini import ask_gemini as llm_ask_gemini

logger = logging.getLogger(__name__)
DT_EXIT_CONFIDENCE_MIN = 0.70

def factor_sentiment(fund: Profile) -> Decimal:
    """
    Resolve fund sentiment to a scalar.
    - Manual presets: map via Profile.SENTIMENT (float values).
    - AUTO: derive from cash / wealth thresholds.
    Always returns a Decimal.
    """

    sentiment_key = fund.sentiment or "AUTO"
    if sentiment_key != "AUTO":
        return Decimal(Profile.SENTIMENT.get(sentiment_key, 1.0))

    cash_value = fund.cash or Decimal("0")
    holdings_value = Decimal("0")
    for holding in Holding.objects.filter(fund=fund).select_related("stock"):
        if holding.stock and holding.stock.price and holding.shares:
            holdings_value += Decimal(str(holding.stock.price)) * Decimal(str(holding.shares))

    wealth = cash_value + holdings_value
    if wealth <= 0:
        logger.warning(
            f"{fund.name}: AUTO sentiment fallback to STAG (wealth <= 0). "
            f"cash={cash_value}, holdings={holdings_value}"
        )
        return Decimal(1.0)

    cash_ratio = float(cash_value / wealth)
    if cash_ratio < 0.25:
        band = "STRONG_BEAR"
    elif cash_ratio < 0.50:
        band = "BEAR"
    else:
        band = "STAG"

    sentiment = Profile.SENTIMENT[band]

    if sentiment < 1.0:
        logger.info(
            f"{fund.name}: AUTO sentiment {band} ({sentiment:.2f}) "
            f"cash_ratio={cash_ratio:.3f}, cash={cash_value}, holdings={holdings_value}, wealth={wealth}"
        )

    return Decimal(str(sentiment))

def analyse_target(discovery, holding, target, sentiment):
    stock = holding.stock
    current = holding.stock.price
    buy_price = holding.average_price if holding.average_price else discovery.price

    target = buy_price + (Decimal(str(target)) - buy_price) * sentiment

    # Case 1: Targets should only trigger sells at a profit, not at a loss
    if current < buy_price:
        return False

    # Case 2: Price >= target and downturn detected (down pullback_pct from peak since purchase; intraday today, daily history)
    if current >= target and stock.downturned(discovery.created, pullback_pct=5.0 * float(sentiment)):
        return True

    # Case 3: Price < target but previously peaked at/above target (protect gains)
    if current < target and stock.peaked(discovery.created, target):
        return True

    return False


def _build_drop_prompt(context_block: str) -> str:
    return f"""
You are a live risk triage assistant for equity positions.

A descending-trend alert has triggered for the tickers below after a recent price drop.
Decide whether each position should be exited NOW due to a materially negative, very recent catalyst.

Source quality policy:
- Primary sources (highest trust): Reuters, Bloomberg, Dow Jones Newswires
- Secondary source: Benzinga
- Give most weight to primary sources.
- Use Benzinga alone only if the catalyst is clear and material.
- If credible recent evidence is missing or ambiguous, choose HOLD with lower confidence.

What qualifies for EXIT:
- A recent catalyst likely to impair near-term value materially, such as:
  earnings/guidance miss, major downgrade cluster, regulatory/legal action,
  financing/liquidity stress, or thesis-breaking company-specific news.
- The catalyst should plausibly explain the drop and suggest continued downside asymmetry.

What qualifies for HOLD:
- No credible material catalyst found in trusted sources, or
- Evidence appears stale, minor, speculative, or already absorbed.

Output requirements:
- For each ticker, return:
  - action: "EXIT" or "HOLD" only
  - confidence: number 0.0–1.0
  - reason: one concise sentence (max 25 words) naming the key catalyst/risk or lack of credible evidence
  - sources_used: array of source names you relied on (from Reuters/Bloomberg/Dow Jones Newswires/Benzinga; empty if none)

Rules:
- No prose outside JSON.
- If uncertain, default to HOLD with lower confidence.

Context:
{context_block}

Return ONLY a single JSON object in this exact shape:
{{
  "TICKER": {{
    "action": "EXIT|HOLD",
    "confidence": 0.00,
    "reason": "short reason",
    "sources_used": ["Reuters", "Bloomberg"]
  }}
}}
"""


def analyse_drop(sa, dropped_stocks):
    """
    Evaluate DT-triggered holdings via LLM and execute sells for EXIT decisions.
    """
    if not dropped_stocks:
        return

    # Deduplicate symbols for one batched LLM call, while preserving first-seen context.
    contexts_by_symbol = {}
    for item in dropped_stocks:
        holding = item["holding"]
        symbol = holding.stock.symbol.upper()
        if symbol in contexts_by_symbol:
            continue

        current_price = holding.stock.price or Decimal("0")
        buy_price = item.get("buy_price") or Decimal("0")
        pnl_pct = None
        if buy_price and buy_price > 0:
            pnl_pct = (Decimal(str(current_price)) - Decimal(str(buy_price))) / Decimal(str(buy_price)) * Decimal("100")

        trend = item.get("trend")
        threshold = item.get("threshold")
        company = (holding.stock.company or "").strip() or "unknown"
        fund = item["fund"]
        line = (
            f"  {symbol}: {company!r}, ${float(current_price):.2f}; "
            f"fund={fund.name!r}; trend={float(trend):.4f} < threshold={float(threshold):.4f}"
        )
        if pnl_pct is not None:
            line += (
                f"; buy_price=${float(buy_price):.2f}; "
                f"pnl_pct={float(pnl_pct):.2f}%"
            )
        contexts_by_symbol[symbol] = line

    context_block = "\n".join(contexts_by_symbol[s] for s in sorted(contexts_by_symbol.keys()))
    prompt = _build_drop_prompt(context_block)

    model, results, _next_model_idx, _next_key_idx = llm_ask_gemini(
        prompt=prompt,
        advisor_name="analyse_drop",
        gemini_model_index=0,
        gemini_key_index=0,
        timeout=120.0,
        use_search=True,
    )

    if not results or not isinstance(results, dict):
        logger.warning("analyse_drop: no usable JSON response from Gemini")
        return

    exit_by_symbol = {}
    for symbol in contexts_by_symbol.keys():
        data = results.get(symbol) or results.get(symbol.upper()) or results.get(symbol.lower())
        if not isinstance(data, dict):
            logger.info("analyse_drop: %s missing/invalid decision payload", symbol)
            continue

        action = (data.get("action") or "").strip().upper()
        raw_conf = data.get("confidence")
        try:
            confidence = float(raw_conf) if raw_conf is not None else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        reason = (data.get("reason") or "").strip()
        sources = data.get("sources_used") if isinstance(data.get("sources_used"), list) else []

        logger.info(
            "analyse_drop decision %s: action=%s confidence=%.2f sources=%s reason=%s",
            symbol,
            action or "N/A",
            confidence,
            ",".join(sources) if sources else "-",
            reason or "-",
        )

        if action == "EXIT" and confidence >= DT_EXIT_CONFIDENCE_MIN:
            exit_by_symbol[symbol] = {
                "confidence": confidence,
                "reason": reason,
                "sources_used": sources,
            }
            logger.info(
                "analyse_drop qualified EXIT %s (confidence %.2f >= %.2f)",
                symbol,
                confidence,
                DT_EXIT_CONFIDENCE_MIN,
            )
        else:
            logger.info(
                "analyse_drop HOLD/skip %s (action=%s, confidence=%.2f, threshold=%.2f)",
                symbol,
                action or "N/A",
                confidence,
                DT_EXIT_CONFIDENCE_MIN,
            )

    if not exit_by_symbol:
        logger.info("analyse_drop: no EXIT actions above confidence threshold (model=%s)", model)
        return

    logger.info(
        "analyse_drop: exiting %s symbols from %s DT candidates (model=%s)",
        len(exit_by_symbol),
        len(dropped_stocks),
        model,
    )

    for item in dropped_stocks:
        holding = item["holding"]
        fund = item["fund"]
        symbol = holding.stock.symbol.upper()
        decision = exit_by_symbol.get(symbol)
        if not decision:
            continue
        if not holding.pk or holding.shares <= 0:
            continue

        reason = decision["reason"] or "Recent material downside catalyst after descending trend."
        conf = decision["confidence"]
        logger.info(
            "analyse_drop executing SELL %s fund=%s holding_id=%s conf=%.2f",
            symbol,
            fund.name,
            holding.id,
            conf,
        )
        execute_sell(
            sa,
            fund,
            holding,
            reason[:240],
        )


def analyze_holdings(sa, funds):
    logger.info(f"Analyzing holdings for SA session {sa.id}")
    dropped_stocks = []

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

    # Iterate thru funds
    for fund in funds:

        sentiment = factor_sentiment(fund)

        for holding in Holding.objects.filter(fund=fund):

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
                                execute_sell(sa, fund, holding, f"{holding.stock.symbol} fell to stop-loss of ${instruction.value1:.2f}")
                                break

                        elif instruction.instruction in ['TARGET_PRICE', 'TARGET_PERCENTAGE']:
                            if analyse_target(discovery, holding, instruction.value1, sentiment):
                                execute_sell(sa, fund, holding, f"{holding.stock.symbol} reached target price of ${instruction.value1:.2f}")
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

                                if analyse_target(discovery, holding, Decimal(str(current_target)), sentiment):
                                    execute_sell(sa, fund, holding, f"{holding.stock.symbol} downturn detected at diminishing target ${current_target:.2f} (day {days_held}/{max_days})")
                                    break
                            else:
                                logger.warning(f"TARGET_DIMINISHING instruction {instruction.id} missing required fields (value1 or buy_price)")

                        elif instruction.instruction == 'PROFIT_TARGET':
                            # Calculate target profit based on average spend and ratio
                            # value1 = ratio (e.g., 0.10 for 10% profit)
                            if instruction.value1 and holding.shares > 0:
                                ratio = Decimal(str(instruction.value1))
                                # Base allowance is derived from the fund's aspirational spread.
                                base_allowance = fund.average_spend()
                                target_value = base_allowance * (Decimal('1.0') + ratio)
                                target_price = target_value / Decimal(str(holding.shares))
                                
                                if analyse_target(discovery, holding, target_price, sentiment):
                                    execute_sell(sa, fund, holding, f"{holding.stock.symbol} reached profit target (target price: ${target_price:.2f})")
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
                                    execute_sell(sa, fund, holding,
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

                                    # Force a REBUY
                                    execute_buy(sa, fund, holding.stock, rebuy_amount, f"Rebuying {rebuy_pct*100:.0f}% after drop", True)
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
                                                    execute_sell(sa, fund, holding, f"Flat near {range_threshold_pct:.0f}% over {evaluation_days} days")
                                                    break
                                    except Exception as e:
                                        logger.warning(f"Error checking PROFIT_FLAT for {holding.stock.symbol}: {e}")
                                        continue
                            else:
                                logger.warning(f"PROFIT_FLAT instruction {instruction.id} missing required fields (value1 or value2)")

                        elif instruction.instruction == 'AFTER_DAYS':
                            days_held = (timezone.now() - discovery.created).days
                            if instruction.value1 and days_held >= int(instruction.value1):
                                execute_sell(sa, fund, holding, f"{holding.stock.symbol} after holding for {days_held} days (target: {int(instruction.value1)} days)")
                                break

                        elif instruction.instruction == 'DESCENDING_TREND':

                            trend = holding.stock.calc_trend(hours=2)

                            if instruction.value1 is not None and trend is not None and trend < instruction.value1:

                                dropped_stocks.append(
                                    {
                                        "fund": fund,
                                        "holding": holding,
                                        "discovery": discovery,
                                        "buy_price": buy_price,
                                        "current_price": holding.stock.price,
                                        "trend": trend,
                                        "threshold": instruction.value1,
                                    }
                                )
                                break

                        elif instruction.instruction == 'NOT_TRENDING':
                            trending = holding.stock.is_trending()
                            if trending is False:  # Explicitly False (not None)
                                execute_sell(sa, fund, holding, f"{holding.stock.symbol} no longer trending (low volume)")
                                break

                        elif instruction.instruction == 'END_DAY' and end_day:
                            if instruction.value1 and holding.stock.price >= instruction.value1:
                                execute_sell(sa, fund, holding, f"{holding.stock.symbol} end of day sell target: ${instruction.value1:.2f}")
                                break

                        elif instruction.instruction == 'END_WEEK' and end_week:
                            # Only apply END_WEEK if holding has been held for at least 7 days
                            if holding.created:
                                days_held = (timezone.now() - holding.created).days
                                if days_held >= 7:
                                    if instruction.value1 and holding.stock.price >= instruction.value1:
                                        execute_sell(sa, fund, holding,
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

    if dropped_stocks:
        logger.info("DT candidates collected: %s", len(dropped_stocks))
        analyse_drop(sa, dropped_stocks)


# Discovery new stock
def analyze_discovery(sa, funds, advisors):
    logger.info(f"Analyzing discovery for SA session {sa.id}")

    # Clear Polygon cache at start of discovery run to ensure fresh data
    from core.services.advisors.advisor import AdvisorBase  # TODO: Review
    AdvisorBase.clear_polygon_cache()

    # 1. Look for new stock
    for a in advisors:
        logger.info(f"Discovery ------------- {a.advisor.name}")
        a.discover(sa)

    # 2. Filter stocks to buy on a per user basis
    for fund in funds:
        logger.info(f"Buying ------------- {fund.name}")

        sentiment = factor_sentiment(fund)
        allowed_advisors = list(fund.advisors or [])

        # 1. Filter discoveries by allowed advisors (if specified)
        discoveries_qs = Discovery.objects.filter(sa=sa)
        
        if allowed_advisors:
            # Get Advisor objects for the allowed advisor python_class values
            allowed_advisor_objects = Advisor.objects.filter(python_class__in=allowed_advisors)

            if allowed_advisor_objects.exists():
                discoveries_qs = discoveries_qs.filter(advisor__in=allowed_advisor_objects)
            else:
                logger.warning(f"{fund.name}: no matching advisor")
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

        # 3. Base allowance
        # Base allowance is derived from the fund's aspirational spread.
        base_allowance = fund.average_spend()

        # 4. Iterate through unique discoveries and calculate allowance per discovery
        for discovery in unique_discoveries:
            # Get health for this discovery
            health = discovery.health
            min_score = fund.min_score()
            health_score = health.score

            # Factor in sentiment
            health_score *= Decimal(str(sentiment))

            if health and health_score < min_score:
                logger.info(
                    f"User {fund.name}: Discovery {discovery.stock.symbol} by {discovery.advisor.name} "
                    f"health check score ({health_score}) below threshold ({min_score})"
                )
                continue

            # Get advisor weight (normalized)
            advisor_weight = discovery.advisor.weight
            discovery_weight = discovery.weight
            combined_weight = advisor_weight * discovery_weight

            # Calculate allowance with weighting
            if combined_weight > Decimal('1.0'):
                allowance = base_allowance * combined_weight
            else:
                allowance = base_allowance * combined_weight

            # Factor in sentiment
            allowance *= sentiment

            logger.info(
                f"User {fund.name}: Discovery {discovery.stock.symbol} by {discovery.advisor.name} "
                f"(Weight={combined_weight:.3f} allowance=${allowance:.2f})"
            )

            # Extract first clause from discovery explanation as summary
            explanation = discovery.explanation.split(" | ")[0].strip()

            # Call execute_buy
            execute_buy(sa, fund, discovery.stock, allowance, explanation)
