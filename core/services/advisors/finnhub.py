import requests
import json
from datetime import datetime, timedelta
from decimal import Decimal
from core.services.advisors.advisor import AdvisorBase, register
import logging

logger = logging.getLogger(__name__)

class Finnhub(AdvisorBase):
    
    def analyze(self, sa, consensus):
        """Analyze stock using Finnhub free tier data"""
        try:
            # Get free tier data from Finnhub
            stock = consensus

            quote_data = self._get_quote(stock.symbol)
            company_profile = self._get_company_profile(stock.symbol)
            
            # Try premium endpoints (will fail gracefully with 403)
            recommendation_trends = self._get_recommendation_trends(stock.symbol)
            price_target = self._get_price_target(stock.symbol)
            
            # Calculate confidence based on available data
            confidence = self._calculate_confidence_free_tier(
                stock, quote_data, company_profile, recommendation_trends, price_target
            )
            
            # Build detailed analysis explanation
            explanation_parts = []
            #explanation_parts.append(f"Confidence Score: {confidence:.2f}")
            
            # Add data from available sources
            if quote_data and 'c' in quote_data:
                price = quote_data['c']
                change_pct = quote_data.get('dp')
                if change_pct is not None:
                    if change_pct > 5:
                        explanation_parts.append(f"🚀 STRONG: {change_pct:.1f}% gain")
                    elif change_pct > 2:
                        explanation_parts.append(f"📈 UP: {change_pct:.1f}% gain")
                    elif change_pct < -5:
                        explanation_parts.append(f"📉 DOWN: {change_pct:.1f}% loss")
            
            if company_profile:
                market_cap = company_profile.get('marketCapitalization')
                if market_cap and market_cap > 1_000_000_000_000:
                    explanation_parts.append(f"🏢 MEGA-CAP: ${market_cap/1e12:.1f}T")
                elif market_cap and market_cap < 100_000_000:
                    explanation_parts.append(f"⚠️ MICRO-CAP")
            
            if recommendation_trends and len(recommendation_trends) > 0:
                latest_rec = recommendation_trends[0]
                buy_ratings = (latest_rec.get('strongBuy') or 0) + (latest_rec.get('buy') or 0)
                total = sum([latest_rec.get(k) or 0 for k in ['strongBuy', 'buy', 'hold', 'sell', 'strongSell']])
                if total and total > 0:
                    buy_pct = buy_ratings / total
                    if buy_pct > 0.6:
                        explanation_parts.append(f"⭐ ANALYST CONSENSUS: {buy_pct:.0%} buy")
            
            explanation = " | ".join(explanation_parts)
            super().recommend(sa, consensus, confidence, explanation)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error analyzing {stock.symbol} with Finnhub: {e}")
            return 0.5  # Neutral on error

    def _get_quote(self, symbol):
        """Get real-time quote (FREE TIER)"""
        try:
            url = f"{self.advisor.endpoint}/quote"
            params = {'symbol': symbol, 'token': self.advisor.key}
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                logger.debug("Finnhub Quote: Premium endpoint required")
                return None
            else:
                logger.warning(f"Finnhub Quote Error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Quote error for {symbol}: {e}")
            return None

    def _get_company_profile(self, symbol):
        """Get company profile (FREE TIER)"""
        try:
            url = f"{self.advisor.endpoint}/stock/profile2"
            params = {'symbol': symbol, 'token': self.advisor.key}
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                logger.debug("Finnhub Profile: Premium endpoint required")
                return None
            else:
                logger.warning(f"Finnhub Profile Error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Company profile error for {symbol}: {e}")
            return None

    def _get_recommendation_trends(self, symbol):
        """Get analyst recommendation trends (PREMIUM ONLY)"""
        try:
            url = f"{self.advisor.endpoint}/stock/recommendation"
            params = {'symbol': symbol, 'token': self.advisor.key}
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                logger.debug("Finnhub Recommendations: Premium endpoint (403)")
                return None
            elif response.status_code == 429:
                logger.warning("Finnhub Rate limit hit (429)")
                return None
            else:
                logger.warning(f"Finnhub Recommendations Error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Recommendation trends error for {symbol}: {e}")
            return None

    def _get_price_target(self, symbol):
        """Get analyst price targets (PREMIUM ONLY)"""
        try:
            url = f"{self.advisor.endpoint}/stock/price-target"
            params = {'symbol': symbol, 'token': self.advisor.key}
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                logger.debug("Finnhub Price Targets: Premium endpoint (403)")
                return None
            elif response.status_code == 429:
                logger.warning("Finnhub Rate limit hit (429)")
                return None
            else:
                logger.warning(f"Finnhub Price Target Error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Price target error for {symbol}: {e}")
            return None

    def _calculate_confidence_free_tier(self, stock, quote_data, company_profile, rec_trends, price_target):
        """Calculate confidence using free tier data primarily"""
        score = 0.5  # Start neutral
        
        # QUOTE DATA ANALYSIS (FREE)
        if quote_data and 'c' in quote_data:
            current_price = quote_data.get('c')
            change_pct = quote_data.get('dp')  # Day percent change (handle None)
            
            # Update stock price (handle None price)
            if current_price is not None:
                stock.price = Decimal(str(current_price))
                stock.save()
            
            if change_pct is not None:
                if change_pct > 5:
                    score += 0.15
                elif change_pct > 2:
                    score += 0.1
                elif change_pct < -5:
                    score -= 0.15
                elif change_pct < -2:
                    score -= 0.1
        
        # COMPANY PROFILE ANALYSIS (FREE)
        if company_profile:
            market_cap = company_profile.get('marketCapitalization')
            if market_cap:
                if market_cap > 1_000_000_000_000:
                    score += 0.1
                elif market_cap > 100_000_000_000:
                    score += 0.05
                elif market_cap < 100_000_000:
                    score -= 0.1
        
        # ANALYST RECOMMENDATIONS (PREMIUM - Graceful degradation)
        if rec_trends and len(rec_trends) > 0:
            latest_rec = rec_trends[0]
            buy = (latest_rec.get('strongBuy') or 0) + (latest_rec.get('buy') or 0)
            total = sum([latest_rec.get(k) or 0 for k in ['strongBuy', 'buy', 'hold', 'sell', 'strongSell']])
            
            if total and total > 0:
                buy_pct = buy / total
                if buy_pct > 0.8:
                    score += 0.2
                elif buy_pct > 0.6:
                    score += 0.1
                elif buy_pct < 0.3:
                    score -= 0.1
        
        # PRICE TARGETS (PREMIUM - Graceful degradation)
        if price_target:
            target_mean = price_target.get('targetMean')
            if target_mean is not None and stock.price is not None and float(stock.price) > 0:
                upside = ((target_mean - float(stock.price)) / float(stock.price))
                if upside > 0.3:
                    score += 0.15
                elif upside > 0.1:
                    score += 0.05
                elif upside < -0.2:
                    score -= 0.1
        
        # Ensure score is between 0 and 1 and round to 2 decimals
        final_score = round(max(0.0, min(1.0, score)), 2)
        
        return final_score


register(name="Finnhub", python_class="Finnhub")
