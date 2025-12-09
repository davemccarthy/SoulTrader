
import logging
import time
import json
import re
import numpy as np
import yfinance as yf
import google.generativeai as genai
from core.models import Stock, Discovery, Recommendation, Advisor
from google.api_core import exceptions
from django.conf import settings
from decimal import Decimal
from typing import Dict, Optional

logger = logging.getLogger(__name__)

genai.configure(api_key=getattr(settings, 'GEMINI_KEY', None))

"""
 Note: Gemini models
 https://ai.google.dev/gemini-api/docs/rate-limits 
"""

# Gemini models
models = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

class AdvisorBase:

    def __init__(self, advisor):
        self.advisor = advisor
        self.gemini_model = 0

    def discover(self, sa):
        return

    def analyze(self, sa, stock):
        return

    def discovered(self, sa, symbol, company, explanation, sell_instructions = None, weight = 1.0):
        #-- find stock or create stock
        try:
            stock = Stock.objects.get(symbol=symbol)
        except Stock.DoesNotExist:
            stock = Stock()
            stock.symbol = symbol
            logger.info(f"{self.advisor.name} created stock {stock.symbol}")

        # Latest price
        stock.advisor = self.advisor
        stock.refresh()

        # Always attempt health check (graceful failure - don't block discovery)
        logger.info(f"Attempting health check for {stock.symbol}...")
        try:
            health = self.health_check(stock, sa)
            if health:
                logger.info(f"Successfully created health check for {stock.symbol}")
            else:
                logger.error(f"Health check returned None for {stock.symbol}")
                return None

        except Exception as e:
            logger.error(f"Health check failed for {stock.symbol} during discovery: {e}", exc_info=True)
            return None

        # REMOVED: 7-day duplicate discovery check - no longer needed since SA sessions are faster without Polygon
        # This allows stocks to be re-discovered and bought again if they show strong movement

        # Create new Discovery record
        discovery = Discovery()
        discovery.sa = sa
        discovery.stock = stock
        discovery.price = stock.price  # NEW: Capture price at discovery time
        discovery.advisor = self.advisor
        discovery.explanation = explanation[:1000]
        discovery.weight = weight
        discovery.save()

        # Create sell instructions if provided
        if sell_instructions:
            from core.models import SellInstruction
            from decimal import Decimal

            for instruction_type, instruction_value in sell_instructions:
                instruction = SellInstruction()
                instruction.discovery = discovery
                instruction.instruction = instruction_type

                # Calculate value based on instruction type
                if instruction_type == 'STOP_PERCENTAGE':
                    instruction.value = Decimal(str(stock.price)) * Decimal(str(instruction_value))

                elif instruction_type == 'TARGET_PERCENTAGE' :
                    instruction.value = Decimal(str(stock.price)) * Decimal(str(instruction_value))

                elif instruction_type == 'STOP_PRICE':
                    instruction.value = Decimal(str(instruction_value))

                elif instruction_type == 'TARGET_PRICE':
                    instruction.value = Decimal(str(instruction_value))

                elif instruction_type == 'AFTER_DAYS':
                    instruction.value = Decimal(str(instruction_value))

                elif instruction_type == 'DESCENDING_TREND':
                    instruction.value = Decimal(str(instruction_value))

                else:
                    logger.warning(f"Unknown instruction_type: {instruction_type}")
                    continue

                instruction.save()
                logger.info(f"Created {instruction_type} sell instruction for {stock.symbol}: {instruction.value}")

        logger.info(f"{self.advisor.name} discovers {stock.symbol}")
        return stock

    def recommend(self, sa, stock, confidence, explanation=""):

        recommendation = Recommendation()
        recommendation.sa = sa
        recommendation.stock = stock
        recommendation.advisor = self.advisor
        recommendation.confidence = confidence
        recommendation.explanation = explanation
        recommendation.save()

        logger.info(f"{self.advisor.name} scores {stock.symbol} a confidences of {confidence:.2f}")

    def health_check(self, stock, sa):
        """
        Calculate health check for a stock.
        Only creates a new health check if one doesn't exist or if the latest one is more than a week old.

        Args:
            stock: Stock model instance (with symbol, company, price populated)
            sa: SmartAnalysis session instance

        Returns:
            Health object or None if calculation fails or skipped
        """
        logger.info(f"Starting health_check for {stock.symbol}")
        try:
            from core.models import Health
            from django.utils import timezone
            from datetime import timedelta

            # Check if a recent health check exists (within last week)
            one_week_ago = timezone.now() - timedelta(days=7)
            recent_health = Health.objects.filter(
                stock=stock
            ).order_by('-created').first()

            if recent_health and recent_health.created > one_week_ago:
                logger.info(
                    f"Skipping health check for {stock.symbol}: existing check from {recent_health.created} is less than a week old"
                )
                return recent_health  # Return existing health check

            # 1. Run algorithm to get confidence score
            logger.info(f"Calculating confidence score for {stock.symbol}...")
            score_data = self._get_buy_score(stock.symbol)
            if not score_data:
                logger.warning(f"Failed to calculate confidence score for {stock.symbol}")
                return None
            confidence_score = Decimal(str(score_data['confidence_score']))
            logger.info(f"Confidence score calculated for {stock.symbol}: {confidence_score}")

            # 2. Call Gemini for opinion (using V5 prompt)
            gemini_result = self._get_gemini_opinion(stock)
            gemini_weight = Decimal('1.0')  # Default to neutral if Gemini fails
            gemini_recommendation = None
            gemini_explanation = None

            if gemini_result:
                gemini_weight = Decimal(str(gemini_result['gemini_weight']))
                gemini_recommendation = gemini_result['recommendation']
                gemini_explanation = gemini_result['explanation']
                logger.info(
                    f"Gemini opinion for {stock.symbol}: {gemini_recommendation} (weight: {gemini_weight})")
            else:
                logger.warning(f"Gemini opinion failed for {stock.symbol}, using default weight 1.0")

            # 3. Calculate final score
            final_score = confidence_score * gemini_weight

            # 4. Create Health record (only if none exists or existing one is old)
            health = Health.objects.create(
                stock=stock,
                sa=sa,
                score=final_score,
                meta={
                    'confidence_score': float(confidence_score),
                    'gemini_weight': float(gemini_weight),
                    'gemini_recommendation': gemini_recommendation,
                    'gemini_explanation': gemini_explanation,
                    'health_score': score_data['health_score'],
                    'valuation_score': score_data['valuation_score'],
                    'piotroski': score_data['piotroski'],
                    'altman_z': float(score_data['altman_z']) if score_data['altman_z'] is not None else None,
                    'ratios': {
                        'current_ratio': float(score_data['ratios']['current_ratio']) if
                        score_data['ratios']['current_ratio'] is not None else None,
                        'debt_to_equity': float(score_data['ratios']['debt_to_equity']) if
                        score_data['ratios']['debt_to_equity'] is not None else None,
                        'net_margin': float(score_data['ratios']['net_margin']) if score_data['ratios'][
                                                                                       'net_margin'] is not None else None,
                        'roa': float(score_data['ratios']['roa']) if score_data['ratios'][
                                                                         'roa'] is not None else None,
                    },
                    'valuation': {
                        'pe': float(score_data['valuation']['pe']) if score_data['valuation'][
                                                                          'pe'] is not None else None,
                        'pb': float(score_data['valuation']['pb']) if score_data['valuation'][
                                                                          'pb'] is not None else None,
                        'ev_ebitda': float(score_data['valuation']['ev_ebitda']) if score_data['valuation'][
                                                                                        'ev_ebitda'] is not None else None,
                        'fcf_yield': float(score_data['valuation']['fcf_yield']) if score_data['valuation'][
                                                                                        'fcf_yield'] is not None else None,
                    },
                },
            )

            logger.info(f"Health check created for {stock.symbol}: score={health.score} (confidence={confidence_score} * weight={gemini_weight})")
            return health

        except ImportError:
            logger.warning("Health model not found - model needs to be created first")
            return None
        except Exception as e:
            logger.error(f"Error calculating health check for {stock.symbol}: {e}", exc_info=True)
            return None

    def _safe_div(self, a, b):
        """Safely divide two numbers, returning None for invalid operations."""
        try:
            if a is None or b in (0, None):
                return None
            return a / b
        except:
            return None

    def _get_buy_score(self, symbol: str) -> Optional[Dict]:
        """
        Calculate buy score for a stock using financial ratios and scoring metrics.
        Ported from test_health_check.py
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
        
        Returns:
            Dictionary with buy_score (0-100), component scores, and financial metrics
            None if calculation fails
        """
        try:
            t = yf.Ticker(symbol)

            info = t.info
            bs = t.balance_sheet.transpose()
            is_ = t.financials.transpose()
            cf = t.cashflow.transpose()

            # --- Extract Key Fields ---
            try:
                latest_bs = bs.iloc[0]
                latest_is = is_.iloc[0]
                latest_cf = cf.iloc[0]
            except:
                raise ValueError("Missing financial statement data.")

            # Financial statement items
            total_assets = latest_bs.get("Total Assets")
            total_liab = latest_bs.get("Total Liab")
            current_assets = latest_bs.get("Total Current Assets")
            current_liab = latest_bs.get("Total Current Liabilities")
            inventory = latest_bs.get("Inventory", 0)
            long_term_debt = latest_bs.get("Long Term Debt") or 0
            short_term_debt = latest_bs.get("Short Long Term Debt") or 0

            revenue = latest_is.get("Total Revenue")
            gross_profit = latest_is.get("Gross Profit")
            net_income = latest_is.get("Net Income")
            ebit = latest_is.get("EBIT") or latest_is.get("Operating Income")

            op_cf = latest_cf.get("Total Cash From Operating Activities")

            # Market (price-based)
            price = info.get("currentPrice")
            eps = info.get("trailingEps")
            book_value = info.get("bookValue")
            market_cap = info.get("marketCap")
            enterprise_value = info.get("enterpriseValue")
            ebitda = info.get("ebitda")

            # --- Core Ratios ---
            ratios = {
                "current_ratio": self._safe_div(current_assets, current_liab),
                "quick_ratio": self._safe_div((current_assets or 0) - (inventory or 0), current_liab),
                "debt_to_equity": self._safe_div(
                    long_term_debt + short_term_debt,
                    (total_assets - total_liab) if total_assets and total_liab else None
                ),
                "net_margin": self._safe_div(net_income, revenue),
                "roa": self._safe_div(net_income, total_assets),
                "asset_turnover": self._safe_div(revenue, total_assets),
            }

            # --- Valuation Ratios (share-price dependent) ---
            valuation = {
                "pe": self._safe_div(price, eps),
                "pb": self._safe_div(price, book_value),
                "ev_ebitda": self._safe_div(enterprise_value, ebitda),
                "price_sales": self._safe_div(market_cap, revenue),
                "fcf_yield": self._safe_div(op_cf, market_cap)
            }

            # --- Simple Piotroski F-score (reduced version) ---
            try:
                prev_is = is_.iloc[1]
                prev_bs = bs.iloc[1]

                roa_cur = self._safe_div(net_income, total_assets)
                roa_prev = self._safe_div(prev_is.get("Net Income"), prev_bs.get("Total Assets"))
                ocf_pos = op_cf is not None and op_cf > 0
                roa_improve = roa_prev is not None and roa_cur is not None and roa_cur > roa_prev
                gross_cur = self._safe_div(gross_profit, revenue)
                gross_prev = self._safe_div(prev_is.get("Gross Profit"), prev_is.get("Total Revenue"))
                gross_improve = gross_cur is not None and gross_prev is not None and gross_cur > gross_prev

                piotroski = sum([
                    1 if roa_cur and roa_cur > 0 else 0,
                    1 if ocf_pos else 0,
                    1 if roa_improve else 0,
                    1 if gross_improve else 0
                ])
            except:
                piotroski = None

            # --- Altman Z Simplified ---
            try:
                # Working Capital / Total Assets
                working_capital = (current_assets or 0) - (current_liab or 0)
                A = self._safe_div(working_capital, total_assets)
                
                # Retained Earnings / Total Assets (try to get from balance sheet)
                retained_earnings = latest_bs.get("Retained Earnings") or latest_bs.get("Retained Earnings All Equity") or None
                B = self._safe_div(retained_earnings, total_assets)
                
                # EBIT / Total Assets
                C = self._safe_div(ebit, total_assets)
                
                # Market Cap / Total Liabilities (or Book Value of Equity if market cap not available)
                total_equity = (total_assets or 0) - (total_liab or 0)
                if market_cap and total_liab:
                    D = self._safe_div(market_cap, total_liab)
                elif total_equity and total_liab:
                    D = self._safe_div(total_equity, total_liab)
                else:
                    D = None
                
                # Sales / Total Assets (Asset Turnover)
                E = self._safe_div(revenue, total_assets)
                
                # Calculate Altman Z with proper handling of None values
                if A is None and C is None and E is None:
                    altman = None
                else:
                    altman = (
                        1.2 * (A or 0) + 
                        1.4 * (B or 0) + 
                        3.3 * (C or 0) + 
                        0.6 * (D or 0) + 
                        1.0 * (E or 0)
                    )
                    # Handle NaN values
                    if altman is not None and (np.isnan(altman) or np.isinf(altman)):
                        altman = None
            except Exception as e:
                altman = None

            # --- Convert each component into a 0-100 score ---
            # Business health
            health_score = np.mean([
                np.clip((ratios["current_ratio"] or 1) * 20, 0, 100),
                np.clip(100 - (ratios["debt_to_equity"] or 0) * 20, 0, 100),
                np.clip((ratios["net_margin"] or 0) * 500, 0, 100),
                np.clip((ratios["roa"] or 0) * 2000, 0, 100)
            ])

            # Valuation attractiveness
            valuation_score = np.mean([
                np.clip(50 - (valuation["pe"] or 50), 0, 100),
                np.clip(50 - (valuation["pb"] or 50), 0, 100),
                np.clip(50 - (valuation["ev_ebitda"] or 50), 0, 100),
                np.clip((valuation["fcf_yield"] or 0) * 500, 0, 100)
            ])

            # Piotroski normalized
            pscore = (piotroski / 4 * 100) if piotroski is not None else 50

            # Altman mapped to 0-100
            if altman is None:
                zscore = 50
            elif altman > 3:
                zscore = 90
            elif altman < 1.8:
                zscore = 20
            else:
                zscore = 50

            # --- Combined Buy Score (now confidence_score) ---
            # Weighted toward valuation for buying decisions
            buy_score = (
                health_score * 0.35 +      # 35% - Safety/fundamentals
                valuation_score * 0.40 +   # 40% - Value focus
                pscore * 0.15 +            # 15% - Trend indicators
                zscore * 0.10              # 10% - Bankruptcy risk
            )

            return {
                "ticker": symbol,
                "confidence_score": round(buy_score, 1),
                "health_score": round(health_score, 1),
                "valuation_score": round(valuation_score, 1),
                "piotroski": piotroski,
                "altman_z": altman,
                "ratios": ratios,
                "valuation": valuation
            }
        except Exception as e:
            logger.error(f"Error calculating buy score for {symbol}: {e}")
            return None

    def _get_gemini_opinion(self, stock):
        """
        Get Gemini's independent opinion on a stock using V5 prompt (nervous investor perspective).
        
        Args:
            stock: Stock model instance (with symbol, company, price populated)
        
        Returns:
            Dictionary with 'recommendation', 'explanation', and 'gemini_weight' (multiplier)
            or None if Gemini call fails
        """
        prompt = f"""
Starting afresh,

From the perspective of a nervous investor, would you recommend buying {stock.symbol} ({stock.company}) @ ${float(stock.price):.2f}?

Considering:
- Recent news and media coverage
- Market sentiment
- Any significant developments

Respond in JSON format:
{{
    "recommendation": "STRONG_BUY|BUY|NEUTRAL|AVOID|STRONG_AVOID",
    "explanation": "Your reasoning based on recent news and sentiment"
}}

Thank you
"""
        
        model, results = self.ask_gemini(prompt)
        
        if not results:
            return None
        
        recommendation = results.get("recommendation", "").upper()
        explanation = results.get("explanation", "")
        
        # Map recommendation to gemini_weight multiplier
        weight_mapping = {
            "STRONG_BUY": 1.5,
            "BUY": 1.25,
            "NEUTRAL": 1.0,
            "AVOID": 0.75,
            "STRONG_AVOID": 0.5,
        }
        gemini_weight = weight_mapping.get(recommendation, 1.0)  # Default to 1.0 (neutral) if unknown
        
        return {
            "recommendation": recommendation,
            "explanation": explanation,
            "gemini_weight": gemini_weight,
            "model": model,
        }


    def ask_gemini(self, prompt, timeout=120.0):
        """
        Call Gemini API with retry logic.
        Returns tuple (model, results) or (None, None) on failure.
        
        Note: Gemini models
        https://ai.google.dev/gemini-api/docs/rate-limits
        """
        # Call on 3rd party gemini AI - exhaust all models if unavailable
        for attempt in range(len(models)):
            try:
                retry_exceptions = (
                    exceptions.ServiceUnavailable,  # 503
                    exceptions.ResourceExhausted,   # 429
                    exceptions.DeadlineExceeded,    # 504
                    exceptions.InternalServerError, # 500
                )

                model = models[self.gemini_model]
                logger.info(f"{self.advisor.name} using {model}")

                # In the Gods hand's now
                response = genai.GenerativeModel(model).generate_content(
                    prompt, 
                    request_options={"timeout": timeout}
                )

                # Extract text from nested structure
                if not response.candidates or len(response.candidates) == 0:
                    logger.warning(f"No candidates in Gemini response for {self.advisor.name}")
                    return None, None

                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    logger.warning(f"No content/parts in Gemini response for {self.advisor.name}")
                    return None, None

                response_text = candidate.content.parts[0].text

                if not response_text:
                    logger.warning(f"Empty text in Gemini response for {self.advisor.name}")
                    return None, None

                # Verify feedback
                results = self._extract_json(response_text)

                if not results:
                    logger.warning(f"Cannot parse response for {self.advisor.name}")
                    return None, None

                # Give Gemini a rest
                time.sleep(1)
                return model, results

            except retry_exceptions as e:
                logger.warning(f"Attempt {attempt + 1}: Service {model} unavailable. Retrying...")
                # Try another model
                self.gemini_model += 1
                self.gemini_model %= len(models)

            except Exception as e:
                logger.error(f"Unexpected error in ask_gemini for {self.advisor.name}: {e}")
                return None, None
        
        # All retries exhausted
        logger.error(f"All Gemini models exhausted for {self.advisor.name}")
        return None, None

    def news_flash(self, sa, title, url):

        # Filter by title: reject articles containing common low-value phrases
        title_lower = title.lower() if title else ""
        filter_phrases = [
            "stock market",
            "market today",
            "today nasdaq",
            "nasdaq futures",
            "futures slip",
            "analyst questions",
            "stocks"
        ]
        
        for phrase in filter_phrases:
            if phrase in title_lower:
                logger.info(f"{self.advisor.name} filtering article (skipping Gemini): {title[:80]}")
                return

        # Carefully worded script for the robot
        prompt = f"""
            Starting afresh
            You are an expert analyzing business articles
            How do you interpret this article by way of speculation of rising share prices? Supply a recommendation.
            
            If in doubt, lean toward skepticism 

            Please respond in JSON only choosing one of the below recommendations and supply relevant company symbol / ticker 

            RETURN JSON:
            {{
                "recommendation": "DISMISS|BUY|SELL|STRONG_BUY|STRONG_SELL",
                "tickers": ["SYM1", "SYM2"],
                "explanation": "A brief reason you came to your descion"
            }}

            url: {url}"""

        # Pass sell instructions to discovery
        sell_instructions = [
            ("TARGET_PERCENTAGE", 1.50),
            ("STOP_PERCENTAGE", 0.98),
            ('DESCENDING_TREND', -0.20),
            ('CS_FLOOR', 0.00)
        ]

        # Ask AI for opinion
        model, results = self.ask_gemini(prompt)

        if not results:
            return

        explanation = results.get("explanation", "")
        recommendation = results.get("recommendation", "")
        tickers = results.get("tickers", [])

        # tmp
        #print(url)
        if explanation:
            print(explanation)

        # Log it
        logger.info(f"{recommendation}: {tickers} | {title}")

        # Anything better than DISMISS is put forward for consensus
        if recommendation != "BUY" and recommendation != "STRONG_BUY":
            return

        for ticker in tickers:
            self.discovered(sa, ticker, '',
                f"{model} recommended {recommendation} from reading article. | Article: {title} | {url} | {explanation} ",
                sell_instructions, 1.5 if recommendation == "STRONG_BUY" else 1.0)


    def _extract_json(self, text):
        """Extract JSON from Gemini response, handling markdown code blocks."""
        if not text:
            return None

        # Remove markdown code blocks
        cleaned = re.sub(r'```json\s*', '', text)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object with regex
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None


def register(name, python_class):
    try:
        Advisor.objects.get(python_class=python_class)

    except Advisor.DoesNotExist:
        logger.info(f"Created new advisor: {name}")

        advisor = Advisor(name=name, python_class=python_class)
        advisor.save()
