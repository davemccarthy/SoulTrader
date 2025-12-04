
import logging
import time
import json
import re
import google.generativeai as genai
from core.models import Stock, Discovery, Recommendation, Advisor
from google.api_core import exceptions
from django.conf import settings

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
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.0-flash",
]

class AdvisorBase:

    def __init__(self, advisor):
        self.advisor = advisor
        #self.gemini = genai.Client(api_key=getattr(settings, 'GEMINI_KEY', None))
        self.gemini_model = 0

    # Are these two needed?
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
                if instruction_type == 'STOP_LOSS':
                    instruction.value = Decimal(str(stock.price)) * Decimal(str(instruction_value))

                elif instruction_type == 'TARGET_PRICE':
                    instruction.value = Decimal(str(stock.price)) * Decimal(str(instruction_value))

                elif instruction_type == 'AFTER_DAYS':
                    instruction.value = Decimal(str(instruction_value))

                elif instruction_type == 'DESCENDING_TREND':
                    instruction.value = Decimal(str(instruction_value))

                elif instruction_type == 'CS_FLOOR':
                    instruction.value = None
                else:
                    logger.warning(f"Unknown instruction_type: {instruction_type}")
                    continue

                instruction.save()
                logger.info(f"Created {instruction_type} sell instruction for {stock.symbol}: {instruction.value}")
        else:
            # Default: create CS_FLOOR instruction so existing behavior continues
            from core.models import SellInstruction
            instruction = SellInstruction()
            instruction.discovery = discovery
            instruction.instruction = 'CS_FLOOR'
            instruction.value = None  # CS_FLOOR value is determined at sell time from profile risk
            instruction.save()
            logger.info(f"Created default CS_FLOOR sell instruction for {stock.symbol}")

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
