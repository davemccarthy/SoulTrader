import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from core.services.advisors.advisor import AdvisorBase, register
import logging

from polygon import RESTClient
import requests
import pandas as pd
import pandas_ta as ta
from django.utils import timezone
import json
import re

logger = logging.getLogger(__name__)

# Advisor.blob: newest article published_utc from last fetch (next run uses published_utc.gt).
POLYGON_NEWS_WATERMARK_KEY = "polygon_news_watermark"


class Polygon(AdvisorBase):

    # Class-level rate limiting
    _last_request_time = 0
    _min_request_interval = 12  # seconds (60 seconds / 5 requests)

    def discover(self, sa):
        skip = self.news_discover_skip_reason()
        if skip:
            logger.info("Polygon skip: %s", skip)
            return

        try:
            polygon_key = self.advisor.key
            if not polygon_key:
                logger.warning("No POLYGON_API_KEY")
                return

            now_utc = timezone.now()
            state = self._advisor_blob_state()
            watermark = (state.get(POLYGON_NEWS_WATERMARK_KEY) or "").strip()

            # Incremental: published_utc.gt last watermark. Bootstrap: last 1 hour through now.
            params = {
                "sort": "published_utc",
                "order": "desc",
                "limit": 20,
                "apiKey": polygon_key,
                "published_utc.lte": now_utc.isoformat(timespec="seconds"),
            }
            if watermark:
                params["published_utc.gt"] = watermark
            else:
                params["published_utc.gte"] = (
                    now_utc - timedelta(hours=1)
                ).isoformat(timespec="seconds")

            api_url = "https://api.polygon.io/v2/reference/news"

            try:
                resp = requests.get(api_url, params=params, timeout=20)
                resp.raise_for_status()
                articles = resp.json().get("results", [])

            except requests.RequestException as e:
                logger.error(f"Polygon API error: {e}")
                return

            if not articles:
                logger.info("No Polygon news to process")
                return

            for article in articles:
                title = article.get("title", "")
                article_url = article.get("article_url", "")
                self.news_flash(sa, title, article_url)

            newest_published = (articles[0] or {}).get("published_utc")
            if newest_published:
                state[POLYGON_NEWS_WATERMARK_KEY] = newest_published
                self._save_advisor_blob_state(state)

        except Exception as e:
            logger.error(f"Discovery error: {e}", exc_info=True)

    def analyze_defunct(self, sa, stock):
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
