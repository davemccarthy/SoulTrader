import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from core.models import SmartAnalysis
from core.services.advisors.advisor import AdvisorBase, register
import logging

from polygon import RESTClient
import requests
import pandas as pd
import pandas_ta as ta
from django.utils import timezone
from django.conf import settings
import json
import re

logger = logging.getLogger(__name__)

class Polygon(AdvisorBase):

    # Class-level rate limiting
    _last_request_time = 0
    _min_request_interval = 12  # seconds (60 seconds / 5 requests)

    def discover(self, sa):
        try:
            # Get polygon news
            polygon_key = self.advisor.key
            if not polygon_key:
                logger.warning("No POLYGON_API_KEY")
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
                sa_start_utc = sa_end_utc - timedelta(days=7)

            # Fetch from Polygon API
            url = "https://api.polygon.io/v2/reference/news"
            params = {
                "published_utc.gte": sa_start_utc.isoformat(timespec="seconds"),
                "published_utc.lte": sa_end_utc.isoformat(timespec="seconds"),
                "sort": "published_utc",
                "order": "desc",
                "limit": 20,
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

            # 2. Process said articles
            for idx, article in enumerate(articles, start=1):
                title = article.get("title", "")
                url = article.get("article_url", "")

                self.news_flash(sa, title, url)

        # Problems
        except Exception as e:
            logger.error(f"Discovery error: {e}", exc_info=True)

    def analyze(self, sa, stock):
        """Analyze stock using Polygon technical indicators + fundamentals"""
        try:
            # Real-time check (see advisor model) to minimize delay mid-analysis
            if not self.advisor.is_enabled():
                return

            self._rate_limit()
            client = RESTClient(api_key=self.advisor.key)

            # --- FETCH DAILY PRICE DATA ---
            # Fetch 2 years of data to ensure SMA200 (200 days) calculation
            today = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")  # ~2 years
            try:
                aggs = client.get_aggs(stock.symbol, multiplier=1, timespan="day", from_=from_date, to=today)
                if not aggs:
                    logger.warning(f"Polygon: No price data available for {stock.symbol}")
                    return None
            except Exception as e:
                logger.error(f"Polygon: Error fetching price data for {stock.symbol}: {e}")
                return None

            data = pd.DataFrame(aggs)

            if data.empty:
                logger.warning(f"Polygon: Empty DataFrame for {stock.symbol}")
                return None

            # Detect timestamp key
            if 't' in data.columns:
                data['timestamp'] = pd.to_datetime(data['t'], unit='ms')
                data = data.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            elif 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            else:
                logger.error(f"Polygon: No timestamp found in response for {stock.symbol}")
                return None

            data.set_index('timestamp', inplace=True)
            data = data.sort_index()

            # Check if we have enough data (need at least 200 days for SMA200)
            if len(data) < 200:
                logger.debug(
                    f"Polygon: Insufficient data for {stock.symbol} ({len(data)} days, need 200 for SMA200) - using SMA50 only")
                # Continue anyway, but SMA200 will be None (SMA50 will be used instead)

            # --- TECHNICAL INDICATORS ---
            data['SMA50'] = ta.sma(data['close'], length=50)
            data['SMA200'] = ta.sma(data['close'], length=200)
            data['RSI'] = ta.rsi(data['close'], length=14)
            macd = ta.macd(data['close'])
            data = pd.concat([data, macd], axis=1)

            latest = data.iloc[-1]

            # --- FETCH FUNDAMENTAL DATA (2025 SDK, attributes) ---
            self._rate_limit()  # Rate limit before second API call
            try:
                fundamentals = client.get_ticker_details(stock.symbol)
            except Exception as e:
                logger.warning(f"Polygon: Could not fetch fundamental data for {stock.symbol}: {e}")
                fundamentals = None

            # Safe attribute access with defaults
            pe_ratio = getattr(fundamentals, 'peRatio', 30) if fundamentals else 30
            eps = getattr(fundamentals, 'eps', 0) if fundamentals else 0
            dividend_yield = getattr(fundamentals, 'dividendYield', 0) if fundamentals else 0
            market_cap = getattr(fundamentals, 'marketcap', 0) if fundamentals else 0

            # --- COMPUTE CONFIDENCE SCORE ---
            score = 0.5  # neutral base

            # Technical contributions - handle None values
            sma50 = latest.get('SMA50')
            sma200 = latest.get('SMA200')
            rsi = latest.get('RSI')

            if sma50 is not None and sma200 is not None:
                if latest['close'] > sma50 > sma200:
                    score += 0.2
                elif latest['close'] < sma50 < sma200:
                    score -= 0.2
                elif latest['close'] > sma50:
                    score += 0.05  # Partial bullish
                else:
                    score -= 0.05  # Partial bearish
            elif sma50 is not None:  # Only SMA50 available
                if latest['close'] > sma50:
                    score += 0.05
                else:
                    score -= 0.05

            # RSI contribution - handle None
            if rsi is not None and not pd.isna(rsi):
                rsi_norm = max(0, min(1, (50 - abs(rsi - 50)) / 50))
                score += 0.1 * rsi_norm

            # MACD contribution - handle None
            macd_hist = latest.get('MACDh_12_26_9', 0)
            if macd_hist is not None and not pd.isna(macd_hist):
                score += 0.1 if macd_hist > 0 else -0.1

            # Fundamental contributions
            if pe_ratio and pe_ratio < 25:  # cheaper valuation increases score
                score += 0.1
            elif pe_ratio and pe_ratio >= 25:
                score -= 0.05

            if eps and eps > 0:  # positive EPS
                score += 0.05

            if dividend_yield and dividend_yield > 1:  # yield >1%
                score += 0.05

            # Clip score to [0,1]
            score = max(0, min(1, score))

            # Build explanation - safely handle None values
            sma50_str = f"{sma50:.2f}" if sma50 is not None and not pd.isna(sma50) else "N/A"
            sma200_str = f"{sma200:.2f}" if sma200 is not None and not pd.isna(sma200) else "N/A"
            rsi_str = f"{rsi:.2f}" if rsi is not None and not pd.isna(rsi) else "N/A"
            macd_hist_str = f"{macd_hist:.2f}" if macd_hist is not None and not pd.isna(macd_hist) else "N/A"

            explanation_parts = [
                f"Close: ${latest['close']:.2f}, SMA50: {sma50_str}, SMA200: {sma200_str}, RSI: {rsi_str}",
                f"MACD Hist: {macd_hist_str}",
                f"P/E: {pe_ratio}, EPS: {eps}, Dividend Yield: {dividend_yield:.2%}, Market Cap: {market_cap:,}" if market_cap else f"P/E: {pe_ratio}, EPS: {eps}, Dividend Yield: {dividend_yield:.2%}",
            ]

            explanation = " | ".join(explanation_parts)
            super().recommend(sa, stock, Decimal(str(score)), explanation)

            return score

        except Exception as e:
            logger.error(f"Polygon analysis error for {stock.symbol}: {e}")
            return None

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

    def _rate_limit(self):
        """Ensure we don't exceed 5 requests/minute (1 per 12 seconds)"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            logger.debug(f"{self.advisor.name} rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        self._last_request_time = time.time()


register(name="Polygon.io", python_class="Polygon")
