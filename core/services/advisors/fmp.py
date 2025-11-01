"""
Financial Modeling Prep (FMP) Advisor
Provides comprehensive financial data and analysis using FMP API
"""

import requests
import logging
from decimal import Decimal
from typing import Dict, Optional
from .advisor import AdvisorBase, register

logger = logging.getLogger(__name__)


class FMP(AdvisorBase):
    """Financial Modeling Prep advisor implementation"""
    
    def analyze(self, sa, stock):

        """Analyze stock using FMP API"""
        try:
            # Get company profile (required)
            profile_data = self._get_company_profile(stock.symbol)

            if not profile_data:
                logger.warning(f"FMP: No profile data available for {stock.symbol}, skipping recommendation")
                return None  # Don't create a recommendation if we can't get data

            # Update image url
            image_url = profile_data.get('image')

            if image_url and stock.image is None:
                stock.image = image_url
                stock.save()

                logger.info(f"Updated {stock.symbol} image from profile: {image_url}")
            else:
                logger.warning(f"{stock.symbol} - No image in profile")

            # Get financial ratios (optional)
            ratios_data = self._get_financial_ratios(stock.symbol)
            
            # Get analyst grades/consensus (optional)
            grades_data = self._get_analyst_grades(stock.symbol)
            
            # Double-check we still have valid profile data
            if not profile_data or not any(profile_data.get(k) for k in ['marketCap', 'pe', 'profitMargins', 'symbol']):
                logger.warning(f"FMP: Profile data for {stock.symbol} is empty/invalid, skipping recommendation")
                return None
            
            # Calculate confidence score
            confidence = self._calculate_confidence(profile_data, ratios_data, grades_data)
            
            # Only create recommendation if confidence is meaningful (not the default 0.5)
            if confidence == 0.5 and not ratios_data and not grades_data:
                logger.warning(f"FMP: Only default confidence score for {stock.symbol}, skipping recommendation")
                return None
            
            # Build explanation
            explanation = self._build_explanation(profile_data, ratios_data, grades_data, confidence)
            
            # Only create recommendation if explanation is meaningful
            if explanation == "FMP analysis:" or explanation == "FMP analysis: Insufficient data":
                logger.warning(f"FMP: Empty explanation for {stock.symbol}, skipping recommendation")
                return None
            
            # Only create recommendation if we have valid data
            self.recommend(sa, stock, confidence, explanation)
            
        except Exception as e:
            logger.error(f"FMP analysis failed for {stock.symbol}: {e}")
            return None
    
    def _get_company_profile(self, symbol):
        """Get company profile from FMP"""
        try:
            params = {'symbol': symbol, 'apikey': self.advisor.key}
            response = requests.get(f"{self.advisor.endpoint}/profile", params=params, timeout=30)
            
            # Check for rate limit or payment required
            if response.status_code == 402:
                logger.warning(f"FMP API rate limit exceeded for {symbol}")
                return None

            # Check for 403 Forbidden - log response body to understand the issue
            if response.status_code == 403:
                try:
                    error_body = response.text[:200]  # First 200 chars of error response
                    logger.warning(f"FMP API 403 Forbidden for {symbol}: {error_body}")
                except:
                    logger.warning(f"FMP API 403 Forbidden for {symbol} - could not read error response")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Check if we got actual data (not an error message)
            if isinstance(data, dict) and 'Error Message' in data:
                logger.warning(f"FMP API error for {symbol}: {data['Error Message']}")
                return None
            
            if isinstance(data, list) and data:
                profile = data[0]
                # Check if profile has actual data (not all None/empty)
                if not any(profile.get(k) for k in ['marketCap', 'pe', 'profitMargins', 'symbol']):
                    logger.warning(f"FMP profile for {symbol} has no usable data")
                    return None
                return profile
            return data
            
        except Exception as e:
            logger.error(f"FMP profile request failed for {symbol}: {e}")
            return None
    
    def _get_financial_ratios(self, symbol):
        """Get financial ratios from FMP"""
        try:
            params = {'symbol': symbol, 'apikey': self.advisor.key}
            response = requests.get(f"{self.advisor.endpoint}/ratios", params=params, timeout=30)
            
            # Check for rate limit or payment required
            if response.status_code == 402:
                logger.warning(f"FMP API rate limit exceeded for {symbol}")
                return None
                
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and data:
                return data[0]
            return data
            
        except Exception as e:
            logger.error(f"FMP ratios request failed for {symbol}: {e}")
            return None
    
    def _get_analyst_grades(self, symbol):
        """Get analyst grades/consensus from FMP"""
        try:
            params = {'symbol': symbol, 'apikey': self.advisor.key}
            response = requests.get(f"{self.advisor.endpoint}/grades-consensus", params=params, timeout=30)
            
            # Check for rate limit or payment required
            if response.status_code == 402:
                logger.warning(f"FMP API rate limit exceeded for {symbol}")
                return None
                
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and data:
                return data[0]
            return data
            
        except Exception as e:
            logger.error(f"FMP grades request failed for {symbol}: {e}")
            return None
    
    def _calculate_confidence(self, profile_data, ratios_data, grades_data):
        """Calculate confidence score based on FMP data"""
        score = 0.5  # Start neutral
        
        if not profile_data:
            return score
        
        # Market cap analysis
        market_cap = profile_data.get('marketCap', 0)
        if market_cap > 1_000_000_000_000:  # >$1T
            score += 0.1
        elif market_cap > 100_000_000_000:  # >$100B
            score += 0.05
        elif market_cap < 100_000_000:  # <$100M micro-cap
            score -= 0.1
        
        # P/E ratio analysis
        pe_ratio = profile_data.get('pe', 0)
        if pe_ratio > 0:
            if pe_ratio < 1.0:  # Distressed company
                score -= 0.3
            elif pe_ratio < 15:  # Good value
                score += 0.2
            elif pe_ratio > 30:  # Overvalued
                score -= 0.1
        
        # Profit margin analysis
        profit_margin = profile_data.get('profitMargins', 0)
        if profit_margin < 0:  # Losing money
            score -= 0.2
        elif profit_margin > 0.20:  # >20% margin
            score += 0.2
        elif profit_margin > 0.10:  # >10% margin
            score += 0.1
        elif profit_margin < 0.05:  # <5% margin
            score -= 0.1
        
        # Debt to equity analysis
        debt_to_equity = profile_data.get('debtToEquity', 0)
        if debt_to_equity < 0.5:  # Low debt
            score += 0.1
        elif debt_to_equity > 1.0:  # High debt
            score -= 0.1
        
        # ROE analysis
        roe = profile_data.get('returnOnEquity', 0)
        if roe > 0.20:  # >20% ROE
            score += 0.15
        elif roe > 0.10:  # >10% ROE
            score += 0.1
        elif roe < 0:  # Negative ROE
            score -= 0.15
        
        # Analyst grades/consensus (if available)
        if grades_data:
            strong_buy = grades_data.get('strongBuy', 0)
            buy = grades_data.get('buy', 0)
            hold = grades_data.get('hold', 0)
            sell = grades_data.get('sell', 0)
            
            total_ratings = strong_buy + buy + hold + sell
            if total_ratings > 0:
                buy_ratio = (strong_buy + buy) / total_ratings
                if buy_ratio > 0.8:  # >80% buy ratings
                    score += 0.15
                elif buy_ratio > 0.6:  # >60% buy ratings
                    score += 0.1
                elif buy_ratio < 0.3:  # <30% buy ratings
                    score -= 0.1
        
        # Ensure score stays within bounds
        return max(0.0, min(1.0, score))
    
    def _build_explanation(self, profile_data, ratios_data, grades_data, confidence):
        """Build explanation for the recommendation"""
        if not profile_data:
            return "FMP analysis: Insufficient data"
        
        explanation_parts = ["FMP analysis:"]
        
        # Market cap
        market_cap = profile_data.get('marketCap', 0)
        if market_cap > 1_000_000_000_000:
            explanation_parts.append("Large cap ($1T+)")
        elif market_cap < 100_000_000:
            explanation_parts.append("Micro cap risk")
        
        # P/E ratio
        pe_ratio = profile_data.get('pe', 0)
        if pe_ratio > 0:
            if pe_ratio < 1.0:
                explanation_parts.append("Distressed valuation")
            elif pe_ratio < 15:
                explanation_parts.append("Attractive valuation")
            elif pe_ratio > 30:
                explanation_parts.append("High valuation")
        
        # Profitability
        profit_margin = profile_data.get('profitMargins', 0)
        if profit_margin > 0.20:
            explanation_parts.append("Strong profitability")
        elif profit_margin < 0:
            explanation_parts.append("Unprofitable")
        
        # ROE
        roe = profile_data.get('returnOnEquity', 0)
        if roe > 0.20:
            explanation_parts.append("High ROE")
        elif roe < 0:
            explanation_parts.append("Negative ROE")
        
        return " ".join(explanation_parts)


register(name="Financial Modeling", python_class="FMP")