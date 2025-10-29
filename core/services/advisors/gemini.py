from core.services.advisors.advisor import AdvisorBase, register
from decimal import Decimal
import logging
import requests
import json
import time

logger = logging.getLogger(__name__)


class Gemini(AdvisorBase):

    def analyze(self, sa, stock):
        """Analyze stock using Google Gemini AI - simplified to confidence score only."""
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

    def _call_gemini_api(self, endpoint, api_key, prompt):
        """Make API call to Gemini using correct API format."""
        try:
            import urllib.parse
            
            headers = {
                "Content-Type": "application/json",
            }
            
            # URL encode the API key to handle special characters
            encoded_key = urllib.parse.quote(api_key, safe='')

            base_url = "https://generativelanguage.googleapis.com/v1beta/models"
            model_name = 'gemma-3-12b-it'  # Reverting to working model
            endpoint = f"{base_url}/{model_name}:generateContent"

            # Add API key to URL as query parameter
            url = f"{endpoint}?key={encoded_key}"
            
            # Gemini native API payload format
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                # Extract text from Gemini response structure
                if 'candidates' in data and len(data['candidates']) > 0:
                    return data['candidates'][0]['content']['parts'][0]['text']
                return str(data)
            else:
                return None
                
        except Exception as e:
            return None

    def _parse_gemini_response(self, response_text):
        """Parse Gemini response to extract confidence score and reasoning only.

        Robust to markdown/bold, alternate keys, and word-based confidence.
        """
        try:
            import re

            text = response_text.strip()
            # Normalize simple markdown artifacts
            text = text.replace('**', '')

            confidence_score = None
            reasoning = text

            # 1) Try explicit numeric confidence score
            m = re.search(r"CONFIDENCE[_ ]?SCORE\s*:\s*([01](?:\.\d+)?)", text, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1))
                    if 0.0 <= val <= 1.0:
                        confidence_score = val
                except Exception:
                    pass

            # 2) Try word-based confidence mapping
            if confidence_score is None:
                m2 = re.search(r"CONFIDENCE\s*:\s*(LOW|MEDIUM|HIGH|VERY_HIGH)", text, re.IGNORECASE)
                if m2:
                    word = m2.group(1).upper()
                    mapping = {
                        'LOW': 0.3,
                        'MEDIUM': 0.5,
                        'HIGH': 0.7,
                        'VERY_HIGH': 0.9,
                    }
                    confidence_score = mapping.get(word)

            # 3) As a fallback, find the first 0-1 float near the word CONFIDENCE
            if confidence_score is None:
                m3 = re.search(r"CONFIDENCE[^\d]{0,20}([01](?:\.\d+)?)", text, re.IGNORECASE)
                if m3:
                    try:
                        val = float(m3.group(1))
                        if 0.0 <= val <= 1.0:
                            confidence_score = val
                    except Exception:
                        pass

            # Extract reasoning: everything after the first REASONING: line, else full text
            r_idx = text.upper().find('REASONING:')
            if r_idx != -1:
                reasoning = text[r_idx + len('REASONING:'):].strip()
                if not reasoning:
                    reasoning = text

            return confidence_score, reasoning

        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            return None, response_text


register(name="Google Gemini", python_class="Gemini")