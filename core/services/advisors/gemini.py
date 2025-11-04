from core.services.advisors.advisor import AdvisorBase, register
from core.models import SmartAnalysis
import logging
import requests
import json
import re
import time
from datetime import timedelta
from django.conf import settings
from django.utils import timezone
from google import genai

logger = logging.getLogger(__name__)

class Gemini(AdvisorBase):

    # Discover only
    def discover(self, sa):
        try:
            # 1. Get polygon news
            polygon_key = getattr(settings, 'POLYGON_API_KEY', None)
            if not polygon_key:
                logger.warning("POLYGON_API_KEY not configured in settings")
                return

            # Calculate time window: since last SA session for this username
            prev_sa = SmartAnalysis.objects.filter(
                username=sa.username,
                id__lt=sa.id
            ).order_by('-id').first()

            # Set bounds: prev SA started -> current SA started
            sa_end_utc = timezone.make_aware(sa.started) if timezone.is_naive(sa.started) else sa.started
            if prev_sa:
                sa_start_utc = timezone.make_aware(prev_sa.started) if timezone.is_naive(
                    prev_sa.started) else prev_sa.started
            else:
                sa_start_utc = sa_end_utc - timedelta(hours=24)

            # Fetch from Polygon API
            url = "https://api.polygon.io/v2/reference/news"
            params = {
                "published_utc.gte": sa_start_utc.isoformat(timespec="seconds"),
                "published_utc.lte": sa_end_utc.isoformat(timespec="seconds"),
                "sort": "published_utc",
                "order": "desc",
                "limit": 50,
                "apiKey": polygon_key,
            }

            try:
                resp = requests.get(url, params=params, timeout=20)
                resp.raise_for_status()
                articles = resp.json().get("results", [])

            except requests.RequestException as e:
                logger.error(f"Polygon API error: {e}")
                return

            if not articles:
                logger.info("No Polygon news to process")
                return

            client = genai.Client(api_key=self.advisor.key)

            # 2. Process said articles
            for idx, article in enumerate(articles, start=1):
                title = article.get("title", "")
                url = article.get("article_url", "")

                prompt = f"""
                    Starting afresh
                    You are an expert analyzing business articles
                    How do you interpret this article by way of speculation of rising share prices and supply a recommendation?

                    Please respond in JSON only choosing one of the below recommendations and supply relevant company symbol / ticker 

                    RETURN JSON:
                    {{
                    "recommendation": "DISMISS|BUY|SELL|STRONG_BUY|STRONG_SELL",
                    "tickers": ["SYM1", "SYM2"]
                    }}

                    url: {url}"""

                # Call on 3rd part gemini AI
                response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

                # Extract text from nested structure
                if not response.candidates or len(response.candidates) == 0:
                    logger.warning(f"No candidates in Gemini response for article {idx}")
                    continue

                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    logger.warning(f"No content/parts in Gemini response for article {idx}")
                    continue

                response_text = candidate.content.parts[0].text

                if not response_text:
                    logger.warning(f"Empty text in Gemini response for article {idx}")
                    continue

                # Verify feedback
                results = self._extract_json(response.text)
                if not results: continue

                recommendation  = results["recommendation"]
                tickers = results["tickers"]

                # Log it
                logger.info(f"{recommendation}: {tickers} | {title}")

                # TMP
                print(url)

                # Anything better than DISMISS is put forward for consensus
                if recommendation == "BUY" or recommendation == "STRONG_BUY":
                    for ticker in tickers:
                        stock = self.discovered(sa, ticker, '',f"{recommendation} | Polygon News: {title} | Article: {url} ")

                        # A strong buy skews consensus in favour of BUY
                        if recommendation == "STRONG_BUY":
                            self.recommend(sa, stock, 0.85, f"STRONG_BUY | Polygon News: {title}")

                # Give Gemini a rest
                time.sleep(2)

        # Problems
        except Exception as e:
            logger.error(f"Discovery error: {e}", exc_info=True)

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


register(name="Google Gemini", python_class="Gemini")

article = 'https://stockstory.org/us/stocks/nyse/br/news/earnings/broadridge-nysebr-delivers-impressive-q3'

prompt = f"""
Starting afresh
You are an expert analyzing business articles
How do you interpret this article by way of speculation of rising share prices and supply a recommendation?
Please respond in JSON only choosing one of the below recommendations and supply relevant company symbol / ticker 
RETURN JSON:
{{
  "recommendation": "DISMISS|BUY|SELL|STRONG_BUY|STRONG_SELL",
  "tickers": ["SYM1", "SYM2"]
}}
url: {article}"""