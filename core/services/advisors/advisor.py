
import logging
import time
import json
import re
from core.models import Stock, Discovery, Recommendation, Advisor
from google import genai
from django.conf import settings

logger = logging.getLogger(__name__)

class AdvisorBase:

    def __init__(self, advisor):
        self.advisor = advisor
        self.gemini = genai.Client(api_key=getattr(settings, 'GEMINI_KEY', None))

    # Are these two needed?
    def discover(self, sa):
        return

    def analyze(self, sa, stock):
        return

    def discovered(self, sa, symbol, company, explanation):
        #-- find stock or create stock
        try:
            stock = Stock.objects.get(symbol=symbol)

        except Stock.DoesNotExist:
            stock = Stock()
            stock.symbol = symbol
            # Let yahoo fix this later
            # stock.company = company
            stock.advisor = self.advisor
            stock.save()

            logger.info(f"{self.advisor.name} created stock {stock.symbol}")

        # Avoid duplicating discovery within this SmartAnalysis run
        if Discovery.objects.filter(advisor=self.advisor, stock=stock, sa=sa).exists():
            logger.info(f"{self.advisor.name} already recorded discovery for {stock.symbol} in SA {sa.id}")
            return stock

        # Check for repeating discovery in previous run
        if Discovery.objects.filter(advisor=self.advisor, stock=stock, sa_id=sa.id - 1).exists():
            logger.info(f"{self.advisor.name} previously discovered stock {stock.symbol}")
            return stock

        # Create new Discovery record
        discovery = Discovery()
        discovery.sa = sa
        discovery.stock = stock
        discovery.advisor = self.advisor
        discovery.explanation = explanation[:1000]
        discovery.save()

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

    def news_flash(self, sa, title, url):
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

        # Call on 3rd part gemini AI
        response = self.gemini.models.generate_content(model="gemini-2.5-flash", contents=prompt)

        # Extract text from nested structure
        if not response.candidates or len(response.candidates) == 0:
            logger.warning(f"No candidates in Gemini response for {self.advisor.name} article")
            return False

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            logger.warning(f"No content/parts in Gemini response for {self.advisor.name} article")
            return False

        response_text = candidate.content.parts[0].text

        if not response_text:
            logger.warning(f"Empty text in Gemini response for {self.advisor.name} article")
            return False

        # Verify feedback
        results = self._extract_json(response.text)

        if not results:
            logger.warning(f"Cannot parse response for {self.advisor.name} article")
            return False

        # tmp
        print(url)
        print(results["explanation"])

        explanation = results["explanation"]
        recommendation = results["recommendation"]
        tickers = results["tickers"]

        # Log it
        logger.info(f"{recommendation}: {tickers} | {title}")

        # Anything better than DISMISS is put forward for consensus
        if recommendation == "BUY" or recommendation == "STRONG_BUY":
            for ticker in tickers:
                stock = self.discovered(sa, ticker, '', f"Gemini: {recommendation} | {title} | {explanation} | {url} ")

                # A strong buy skews consensus in favour of BUY
                if recommendation == "STRONG_BUY":
                    self.recommend(sa, stock, 0.85, f"Submittied a high score based on above article | A strong buy skews consensus in favour of BUY")

        # Give Gemini a rest
        time.sleep(2)

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
