import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from core.services.advisors.advisor import AdvisorBase, register
import logging

from polygon import RESTClient
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

class Polygon(AdvisorBase):
    """
    Polygon.io advisor - Free tier: 5 requests/minute
    Rate limiting: 1 request per 12 seconds (60/5 = 12)
    """

    # Class-level rate limiting
    _last_request_time = 0
    _min_request_interval = 12  # seconds (60 seconds / 5 requests)
    
    def _rate_limit(self):
        """Ensure we don't exceed 5 requests/minute (1 per 12 seconds)"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            logger.debug(f"Polygon rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()

    def discover(self, sa):
        return
        """Discover stocks from recent Polygon news articles"""
        try:
            # Get API key from advisor config
            api_key = getattr(self.advisor, "key", "") if self.advisor else ""
            if not api_key:
                logger.warning("Polygon advisor missing API key; skipping discovery")
                return

            # Fetch recent news (last 48 hours)
            news_items = self._fetch_recent_news(api_key)
            
            if not news_items:
                logger.info("No recent news items found")
                return

            # Process each news item
            for item in news_items:
                self._process_news_item(sa, item)

        except Exception as e:
            logger.error(f"Polygon discovery error: {e}")

    def _fetch_recent_news(self, api_key):
        """Fetch recent news from Polygon API"""
        try:
            import requests
            
            url = "https://api.polygon.io/v2/reference/news"
            
            # Last 24 hours - timezone-aware ISO timestamps
            now_utc = datetime.now(timezone.utc)
            since_utc = now_utc - timedelta(hours=24)
            since = since_utc.isoformat(timespec="seconds")
            now_iso = now_utc.isoformat(timespec="seconds")
            
            params = {
                "published_utc.gte": since,
                "published_utc.lte": now_iso,
                "sort": "published_utc",
                "order": "desc",
                "limit": 50,  # Get more items to filter
                "apiKey": api_key
            }
            
            logger.info(f"Fetching Polygon news from {since} to {now_iso}")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                logger.info(f"Found {len(results)} news items from Polygon")
                return results
            else:
                logger.warning(f"Polygon news API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching Polygon news: {e}")
            return []

    def _process_news_item(self, sa, item):
        """Process a single news item and discover stocks"""
        try:
            # Extract tickers from news item
            tickers = item.get("tickers", [])
            if not tickers:
                return

            title = item.get("title", "")
            url = item.get("article_url", "")
            published = item.get("published_utc", "")
            
            # Create explanation with news context
            explanation = f"News: {title} ({url})"
            
            # Discover each ticker mentioned in the news
            for ticker in tickers:
                if ticker and len(ticker) <= 10:  # Basic validation
                    # Use ticker as both symbol and company name for now
                    # The Stock model will be created if it doesn't exist
                    self.discovered(sa, ticker, ticker, explanation)
                    logger.info(f"Polygon discovered {ticker} from news: {title[:50]}...")

        except Exception as e:
            logger.error(f"Error processing news item: {e}")

    def analyze(self, sa, stock):
        """Analyze stock using Polygon technical indicators + fundamentals"""
        try:
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
                logger.debug(f"Polygon: Insufficient data for {stock.symbol} ({len(data)} days, need 200 for SMA200) - using SMA50 only")
                # Continue anyway, but SMA200 will be None (SMA50 will be used instead)

            # --- TECHNICAL INDICATORS ---
            data['SMA50'] = ta.sma(data['close'], length=50)
            data['SMA200'] = ta.sma(data['close'], length=200)
            data['RSI'] = ta.rsi(data['close'], length=14)
            macd = ta.macd(data['close'])
            data = pd.concat([data, macd], axis=1)

            latest = data.iloc[-1]
            
            # Update stock price with latest close
            if 'close' in latest and latest['close'] is not None:
                stock.price = Decimal(str(latest['close']))
                stock.save()

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


register(name="Polygon.io", python_class="Polygon")
