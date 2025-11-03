from core.services.advisors.advisor import AdvisorBase, register
from core.models import SmartAnalysis
from decimal import Decimal
import logging
import requests
import json
import time
from datetime import timedelta
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


class Gemini(AdvisorBase):

    def analyze(self, sa, stock):
        """Analyze stock using Google Gemini AI - DISABLED: Minimal data input makes analysis unreliable.
        
        Discovery is kept active (AI news filtering is valuable).
        Analysis is disabled - rely on quantitative advisors for consensus.
        """
        return
        # DISABLED: Gemini analysis removed - insufficient input data (only symbol/company/price/exchange)
        # makes AI analysis unreliable. Discovery remains active for news filtering.
        try:
            # Ensure advisor config is present
            endpoint = getattr(self.advisor, "endpoint", "") if self.advisor else ""
            api_key = getattr(self.advisor, "key", "") if self.advisor else ""

            if not endpoint or not api_key:
                logger.warning("Gemini advisor missing endpoint or API key; skipping analysis")
                return None

            # Get stock data for analysis
            stock_data = self._get_stock_data(stock)
            
            # Build analysis prompt
            prompt = self._build_analysis_prompt(stock, stock_data)
            
            # Make Gemini API call
            response = self._call_gemini_api(endpoint, api_key, prompt)
            
            if not response:
                return None
                
            # Parse Gemini response - extract confidence score and reasoning only
            confidence_score, reasoning = self._parse_gemini_response(response)
            
            if confidence_score is None:
                return None
            
            # If Gemini says NO_RECOMMENDATION, don't create a recommendation
            if "NO_RECOMMENDATION" in response.upper():
                logger.info(f"Gemini declined to recommend {stock.symbol}: insufficient information")
                return None
            
            # Create recommendation with reasoning as explanation
            explanation = f"Gemini AI: {reasoning[:400]}"
            
            return super().recommend(
                sa, 
                stock, 
                confidence=Decimal(str(confidence_score)), 
                explanation=explanation
            )

        except Exception as e:
            logger.error(f"Gemini analyze error for {stock.symbol}: {e}")
            return None

    def _get_stock_data(self, stock):
        """Get basic stock data for analysis."""
        return {
            'symbol': stock.symbol,
            'company': stock.company,
            'current_price': float(stock.price) if stock.price else 0,
            'exchange': stock.exchange or 'Unknown'
        }

    def _build_analysis_prompt(self, stock, stock_data):
        """Build analysis prompt for Gemini."""
        return f"""You are an expert financial analyst. Analyze {stock.symbol} and provide a clear investment recommendation.

Stock Information:
- Symbol: {stock_data['symbol']}
- Company: {stock_data['company']}
- Exchange: {stock_data['exchange']}

Please analyze this stock and respond in this exact format:

RECOMMENDATION: [STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL]
CONFIDENCE_SCORE: [0.0-1.0 where 1.0 = very confident this is a good BUY opportunity]
TARGET_PRICE: $[predicted price or N/A]
REASONING: [Brief explanation of your analysis and decision]

IMPORTANT CONFIDENCE SCORING:
- For BUY/STRONG_BUY: Use high confidence (0.7-1.0) if you're very sure it's a good investment
- For HOLD: Use medium confidence (0.4-0.6) 
- For SELL/STRONG_SELL: Use low confidence (0.0-0.3) since you're recommending against buying

CRITICAL: If you don't have sufficient information about the company (name, sector, fundamentals), respond with:
RECOMMENDATION: NO_RECOMMENDATION
CONFIDENCE_SCORE: 0.0
TARGET_PRICE: N/A
REASONING: Insufficient information to make a meaningful recommendation

Focus on:
1. Current valuation and fundamentals
2. Growth prospects and competitive position
3. Market conditions and sector trends
4. Risk factors and potential downside

Keep your reasoning concise but informative."""

    def _call_gemini_api(self, endpoint, api_key, prompt, max_retries=3):
        """Call Gemini API with better error handling, logging, and retry logic."""
        import urllib.parse
        
        headers = {"Content-Type": "application/json"}
        encoded_key = urllib.parse.quote(api_key, safe='')

        base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        model_name = 'gemma-3-12b-it'
        url = f"{base_url}/{model_name}:generateContent?key={encoded_key}"

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    if 'candidates' in data and data['candidates']:
                        return data['candidates'][0]['content']['parts'][0]['text']
                    logger.warning(f"Gemini API: Unexpected response structure: {data}")
                    return None
                elif response.status_code == 429:
                    # Rate limit exceeded
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Gemini API: Rate limit exceeded (429). Retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 503:
                    # Model overloaded - retry with exponential backoff
                    wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s
                    logger.warning(f"Gemini API: Model overloaded (503). Retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 403:
                    logger.error(f"Gemini API: Forbidden (403) - Check API key or quota")
                    return None
                else:
                    error_msg = response.text[:200] if response.text else "No error message"
                    logger.error(f"Gemini API error {response.status_code}: {error_msg}")
                    return None

            except requests.exceptions.Timeout:
                logger.warning(f"Gemini API: Request timeout after 30s (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None
            except requests.exceptions.RequestException as e:
                logger.error(f"Gemini API: Request error: {e}")
                return None
            except Exception as e:
                logger.error(f"Gemini API: Unexpected error: {e}")
                return None
        
        # All retries exhausted
        logger.error(f"Gemini API: Max retries ({max_retries}) exhausted")
        return None

    def discover(self, sa):
        """Fetch Polygon news since last SA session; ask Gemini for Dismiss/Interesting; create Discoveries for Interesting items."""
        try:
            polygon_key = getattr(settings, 'POLYGON_API_KEY', None)
            if not polygon_key:
                print("[Gemini.discover] POLYGON_API_KEY not configured in settings")
                return

            # Calculate time window: since last SA session for this username
            # Find previous SA session by sequential ID (not chronological order)
            # This allows manual adjustment of previous SA start times for testing
            prev_sa = SmartAnalysis.objects.filter(
                username=sa.username,
                id__lt=sa.id
            ).order_by('-id').first()

            # Set bounds: prev SA started -> current SA started (not now(), so we can replay/duplicate sessions)
            sa_end_utc = timezone.make_aware(sa.started) if timezone.is_naive(sa.started) else sa.started
            if prev_sa:
                sa_start_utc = timezone.make_aware(prev_sa.started) if timezone.is_naive(prev_sa.started) else prev_sa.started
                print(f"[Gemini.discover] Using window: {sa_start_utc} to {sa_end_utc} (since last SA #{prev_sa.id})")
            else:
                sa_start_utc = sa_end_utc - timedelta(hours=24)
                print(f"[Gemini.discover] No previous SA found, using 24h window: {sa_start_utc} to {sa_end_utc}")

            url = "https://api.polygon.io/v2/reference/news"
            params = {
                "published_utc.gte": sa_start_utc.isoformat(timespec="seconds"),
                "published_utc.lte": sa_end_utc.isoformat(timespec="seconds"),
                "sort": "published_utc",
                "order": "desc",
                "limit": 50,  # Increased to get more articles within time window
                "apiKey": polygon_key,
            }
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code != 200:
                print(f"[Gemini.discover] Polygon news error {resp.status_code}: {resp.text[:200]}")
                return
            results = resp.json().get("results", [])
            if not results:
                print("[Gemini.discover] No Polygon news returned")
                return

            endpoint = getattr(self.advisor, "endpoint", "") if self.advisor else ""
            api_key = getattr(self.advisor, "key", "") if self.advisor else ""
            if not api_key:
                print("[Gemini.discover] Gemini key not configured; skipping AI judgment")
                for idx, item in enumerate(results, start=1):
                    title = item.get("title", "")
                    article_url = item.get("article_url", "")
                    published = item.get("published_utc", "")
                    print(f"[Gemini.discover] Article {idx}:\n- Title: {title}\n- URL: {article_url}\n- Published: {published}")
                return

            for idx, item in enumerate(results, start=1):
                title = item.get("title", "")
                article_url = item.get("article_url", "")
                published = item.get("published_utc", "")
                # Get Polygon's tickers for this article (for validation)
                polygon_tickers = [str(t).upper() for t in item.get("tickers", []) if t]
                print(f"[Gemini.discover] Article {idx}:\n- Title: {title}\n- URL: {article_url}\n- Published: {published}")
                if polygon_tickers:
                    print(f"[Gemini.discover]  - Polygon tickers: {polygon_tickers}")

                prompt = self._build_keep_prompt(title, article_url, polygon_tickers)
                response_text = self._call_gemini_api(endpoint, api_key, prompt)
                if not response_text:
                    print("[Gemini.discover]  → No response from Gemini")
                    continue

                parsed = self._parse_json_loose(response_text)
                rank = str(parsed.get('rank', 'DISMISS')).upper()
                category = str(parsed.get('category', 'OTHER')).upper()
                reason = str(parsed.get('reason', '')).strip()
                tickers = [str(t).upper() for t in (parsed.get('tickers') or []) if t]

                # Validate rank - must be one of the allowed values
                valid_ranks = {'DISMISS', 'INTERESTING', 'OPPORTUNITY', 'NOBRAINER'}
                if rank not in valid_ranks:
                    logger.warning(f"Gemini returned invalid rank '{rank}', defaulting to DISMISS. Response: {response_text[:200]}")
                    rank = 'DISMISS'

                if rank == 'DISMISS':
                    print(f"[Gemini.discover]  → Dismiss: {reason}")
                    continue

                # INTERESTING, OPPORTUNITY, or NOBRAINER
                rank_label = rank.replace('_', ' ')
                print(f"[Gemini.discover]  → {rank_label}: {category} — {reason}")

                if tickers:
                    print(f"[Gemini.discover]  → Tickers: {tickers}")
                    # Validate tickers against Polygon's list (if available)
                    validated_tickers = []
                    if polygon_tickers:
                        validated_tickers = [t for t in tickers if t in polygon_tickers]
                        if validated_tickers != tickers:
                            print(f"[Gemini.discover]  → Validated tickers (filtered): {validated_tickers}")
                    else:
                        validated_tickers = tickers  # Use Gemini's tickers if Polygon has none

                    if validated_tickers:
                        # Build explanation with rank prefix
                        explanation = f"{rank_label}: {reason}. {title} ({article_url})"
                        # Truncate to 500 chars for database field
                        if len(explanation) > 500:
                            explanation = explanation[:497] + "..."

                        for ticker in validated_tickers:
                            if ticker and len(ticker) <= 10:  # Basic validation
                                self.discovered(sa, ticker, ticker, explanation)
                                logger.info(f"Gemini discovered {ticker} from news: {title[:50]}... (rank: {rank})")

                                # Auto-create Recommendation for OPPORTUNITY and NOBRAINER
                                if rank in ['OPPORTUNITY', 'NOBRAINER']:
                                    # Get the stock object
                                    from core.models import Stock, Recommendation
                                    from decimal import Decimal

                                    try:
                                        stock_obj = Stock.objects.get(symbol=ticker)

                                        # Create Recommendation with boosted confidence
                                        confidence = Decimal('0.85') if rank == 'OPPORTUNITY' else Decimal('1.0')
                                        rec_explanation = f"{rank_label}: {reason}"
                                        if len(rec_explanation) > 500:
                                            rec_explanation = rec_explanation[:497] + "..."

                                        recommendation = Recommendation()
                                        recommendation.sa = sa
                                        recommendation.stock = stock_obj
                                        recommendation.advisor = self.advisor
                                        recommendation.confidence = confidence
                                        recommendation.explanation = rec_explanation
                                        recommendation.save()

                                        logger.info(
                                            f"Gemini auto-created {rank_label} recommendation for {ticker} with confidence {confidence}")
                                    except Stock.DoesNotExist:
                                        logger.warning(
                                            f"Stock {ticker} not found when creating {rank_label} recommendation")
                    else:
                        print(f"[Gemini.discover]  → No validated tickers after filtering")

        except Exception as e:
            logger.error(f"Gemini discovery error (Polygon+AI print): {e}")
            return

    def _build_keep_prompt(self, title: str, url: str, polygon_tickers: list = None) -> str:
        allowed_list = polygon_tickers if polygon_tickers else []
        allowed_str = ", ".join(allowed_list) if allowed_list else "[]"
        
        return ("""
You are a financial news analyst. Read the news using headline and URL below and return STRICT JSON ONLY.

Headline: {title}
URL: {url}

Return JSON with fields:
{{
  "rank": "DISMISS|INTERESTING|OPPORTUNITY|NOBRAINER",
  "category": "FDA_APPROVAL|EARNINGS|MA|GUIDANCE|MACRO|OTHER",
  "reason": "<=240 chars",
  "tickers": ["SYM1","SYM2","SYM3","SYM4"]
}}

Rank definitions:
- **DISMISS**: Routine announcements, scheduling, generic commentary, market movements, forecasts, ETF/articles/recommendations
- **INTERESTING**: Solid clinical/regulatory milestones OR small but disclosed deals/contracts worth tracking
- **OPPORTUNITY**: Strong buy signals (FDA approvals, earnings beats with results, M&A $100M+)
- **NOBRAINER**: Exceptional opportunities (blockbuster deals $1B+, transformative approvals)

Rules for each rank:

**DISMISS (use if ANY apply - be STRICT):**
- **CRITICAL: Earnings SCHEDULING - ALWAYS dismiss if article says "to announce", "will announce", "schedules earnings", "to report", "plans to announce" with only a date**
- **Stock price movements or reactions ("plunges", "surges", "climbs") without news event**
- **Investment/financing rounds without disclosed deal terms (e.g., "closes $2M investment")**
- **ETF articles, stock picks, "best stocks to buy" lists**
- **General macro commentary, sector-wide discussions, AI trends**
- **Market forecasts, industry projections, trend analyses**
- **Partnership/team/collaboration announcements without disclosed financial terms or contract values**
- **Supply/delivery/provision announcements without disclosed contract values or revenue impact**
- **Board appointments, governance changes, AGM notices**
- **Marketing campaigns, branding initiatives, PR announcements**

**INTERESTING (rarely used - must have BOTH concrete event AND disclosed terms):**
- Phase 3 clinical trial START with specific enrollment numbers AND company name disclosed
- Guidance changes with actual numbers (not forecasts, not "plans to")
- Signed partnership/contract with disclosed dollar value under $100M
- Minor regulatory approvals (facility expansion with capacity numbers, export licenses with volume)

**OPPORTUNITY (use when):**
- FDA drug approval with specific indication
- Earnings release with actual results (beat/miss percentages) AND forward guidance
- Signed M&A with disclosed deal value ($100M or more)
- Major guidance changes with transformative numbers
- Breakthrough clinical trial results published with data

**NOBRAINER (use when):**
- Blockbuster M&A ($1B+ deal value)
- Transformative FDA approvals (first-in-class, breakthrough therapy designation)
- Massive earnings beats (>50% surprise)
- Acquisition of strategic/high-value assets

Examples:
- "Company to announce earnings on Nov 5" → DISMISS (scheduling only)
- "Stock plunges 16% on earnings miss" → DISMISS (price movement only)
- "Company closes $2M strategic investment" → DISMISS (no deal terms)
- "3 Best Energy Stocks to Buy Now" → DISMISS (stock picks/ETF)
- "Company partners with Industry Leader" → DISMISS (no financial terms)
- "Company reports Q3 earnings beat with revenue of $2.5B" → OPPORTUNITY (actual results)
- "FDA approves Company's drug for lung cancer" → OPPORTUNITY or NOBRAINER
- "Company starts Phase 3 trial enrolling 500 patients" → DISMISS (no outcome data)
- "Phase 3 trial completes successfully for 500 patients" → INTERESTING (concrete milestone)

Tickers:
- CRITICAL: Only use tickers from this allowed list: {allowed_tickers}
- If the allowed list is empty or none of these tickers are relevant, return empty tickers array []
- Maximum 4 tickers per article
- Use primary US common stock tickers only (NASDAQ, NYSE, OTC)
- If unsure of valid ticker, omit tickers entirely
"""
        ).format(title=title[:300], url=url, allowed_tickers=allowed_str)

    def _parse_json_loose(self, text: str) -> dict:
        try:
            cleaned = text.strip().replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            return {}


register(name="Google Gemini", python_class="Gemini")