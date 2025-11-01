import requests
import time
from decimal import Decimal
from core.services.advisors.advisor import AdvisorBase, register
import logging

logger = logging.getLogger(__name__)

class Alpha(AdvisorBase):

    def discover(self, sa):
        """Discover stocks using Alpha Vantage market movers - DISABLED for now"""
        return
        # DISABLED: Alpha discovery disabled - can be re-enabled for high-risk/high-reward (bouncing stocks) later
        try:
            # Get market movers data
            params = {
                'function': 'TOP_GAINERS_LOSERS',
                'apikey': self.advisor.key
            }
            
            response = requests.get(self.advisor.endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract top gainers, losers, and most active
            top_gainers = data.get('top_gainers', [])[:5]  # Top 5
            most_active = data.get('most_actively_traded', [])[:5]  # Top 5
            
            discoveries = []
            
            # Discover top gainers with strong momentum
            for stock_data in top_gainers:
                symbol = stock_data.get('ticker')
                change_percent = float(stock_data.get('change_percentage', '0').rstrip('%'))
                
                # Only discover stocks with significant gains (>5%)
                if change_percent > 5.0:
                    company = stock_data.get('ticker')  # Fallback to ticker if name not available
                    explanation = f"Top gainer: +{change_percent:.1f}%"
                    
                    self.discovered(sa, symbol, company, explanation)
                    discoveries.append(symbol)
                    logger.info(f"Alpha discovered {symbol}: {explanation}")
            
            # Discover most actively traded (high volume = interest)
            for stock_data in most_active:
                symbol = stock_data.get('ticker')
                
                # Skip if already discovered as top gainer
                if symbol in discoveries:
                    continue
                
                # Only discover high volume if price is rising (buying pressure, not selling)
                change_percent_str = stock_data.get('change_percentage', '0%')
                change_percent = float(change_percent_str.rstrip('%'))
                
                # Only discover if price is rising (positive change)
                if change_percent > 0:
                    volume = stock_data.get('volume')
                    company = stock_data.get('ticker')

                    # Format volume with commas if it's a number
                    volume_str = f"{volume:,}" if isinstance(volume, int) else str(volume)
                    explanation = f"High volume +{change_percent:.1f}%: {volume_str} shares"
                    
                    self.discovered(sa, symbol, company, explanation)
                    discoveries.append(symbol)
                    logger.info(f"Alpha discovered {symbol}: {explanation}")
            
            logger.info(f"Alpha Vantage discovery complete: {len(discoveries)} stocks found")
            
        except Exception as e:
            logger.error(f"Alpha Vantage discovery failed: {e}")

    def analyze(self, sa, stock):
        """Analyze stock using Alpha Vantage data"""
        try:
            # Get comprehensive data from Alpha Vantage
            quote_data = self._get_quote(stock.symbol)
            overview_data = self._get_overview(stock.symbol)
            
            if not quote_data:
                # No data available (likely rate-limited) - don't submit recommendation
                # Alpha Vantage will simply not participate in consensus for this stock
                logger.debug(f"Alpha Vantage: Skipping {stock.symbol} - no data available (rate-limited)")
                return None
            
            # Update stock price
            current_price = quote_data.get('price')
            if current_price:
                stock.price = Decimal(str(current_price))
                stock.save()
            
            # Calculate confidence based on Alpha Vantage metrics
            confidence = self._calculate_confidence(quote_data, overview_data)
            
            # Build detailed analysis explanation
            explanation_parts = []
            explanation_parts.append(f"Confidence Score: {confidence:.2f}")
            
            # Add key factors that influenced the score
            change_percent = quote_data.get('change_percent', 0)
            if change_percent > 5.0:
                explanation_parts.append(f"🚀 STRONG DAILY: {change_percent:.1f}% gain")
            elif change_percent > 2.0:
                explanation_parts.append(f"📈 POSITIVE DAILY: {change_percent:.1f}% gain")
            elif change_percent < -5.0:
                explanation_parts.append(f"📉 CATASTROPHIC DAILY: {change_percent:.1f}% loss")
            elif change_percent < -2.0:
                explanation_parts.append(f"📉 MAJOR DAILY: {change_percent:.1f}% loss")
            
            pe_ratio = overview_data.get('PERatio')
            if pe_ratio and pe_ratio != 'None':
                pe_ratio = float(pe_ratio)
                if pe_ratio < 1.0:
                    explanation_parts.append(f"⚠️ DISTRESSED: P/E={pe_ratio:.2f}")
                elif pe_ratio > 30:
                    explanation_parts.append(f"🔴 OVERVALUED: P/E={pe_ratio:.2f}")
                elif pe_ratio < 15:
                    explanation_parts.append(f"🟢 UNDERVALUED: P/E={pe_ratio:.2f}")
            
            profit_margin = overview_data.get('ProfitMargin')
            if profit_margin and profit_margin != 'None':
                profit_margin = float(profit_margin)
                if profit_margin < 0:
                    explanation_parts.append(f"💸 LOSING MONEY: {profit_margin:.1%} margin")
                elif profit_margin > 0.20:
                    explanation_parts.append(f"💪 STRONG PROFITS: {profit_margin:.1%} margin")
                elif profit_margin < 0.05:
                    explanation_parts.append(f"📉 WEAK PROFITS: {profit_margin:.1%} margin")
            
            market_cap = overview_data.get('MarketCapitalization')
            if market_cap and market_cap != 'None':
                market_cap = float(market_cap)
                if market_cap < 100_000_000:
                    explanation_parts.append(f"⚠️ MICRO-CAP RISK: ${market_cap:,.0f}")
                elif market_cap > 1_000_000_000_000:
                    explanation_parts.append(f"🏢 MEGA-CAP STABILITY: ${market_cap:,.0f}")
            
            volume = quote_data.get('volume', 0)
            if volume > 1_000_000:
                explanation_parts.append(f"📊 HIGH VOLUME: {volume:,} shares")
            elif volume < 100_000:
                explanation_parts.append(f"📉 LOW VOLUME: {volume:,} shares")
            
            debt_to_equity = overview_data.get('DebtToEquity')
            if debt_to_equity and debt_to_equity != 'None':
                debt_to_equity = float(debt_to_equity)
                if debt_to_equity > 2.0:
                    explanation_parts.append(f"⚠️ HIGH DEBT: {debt_to_equity:.1f} D/E")
                elif debt_to_equity < 0.5:
                    explanation_parts.append(f"💪 LOW DEBT: {debt_to_equity:.1f} D/E")
            
            explanation = " | ".join(explanation_parts)
            super().recommend(sa, stock, confidence, explanation)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error analyzing {stock.symbol} with Alpha Vantage: {e}")
            return 0.5  # Neutral on error

    def _get_quote(self, symbol):
        """Get real-time quote from Alpha Vantage"""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.advisor.key
            }
            
            response = requests.get(self.advisor.endpoint, params=params, timeout=30)
            
            # Check HTTP status first - if not 200, log once and return
            if response.status_code != 200:
                logger.warning(f"Alpha Vantage API HTTP {response.status_code} for {symbol}: {response.text[:200]}")
                return None
            
            data = response.json()
            
            # Log response for debugging
            logger.debug(f"Alpha Vantage quote response for {symbol}: {list(data.keys())}")
            
            # Check for API error messages
            if 'Note' in data:
                logger.warning(f"Alpha Vantage API note for {symbol}: {data['Note']}")
                return None
            
            if 'Error Message' in data:
                logger.warning(f"Alpha Vantage API error for {symbol}: {data['Error Message']}")
                return None
            
            # Parse Alpha Vantage response
            if 'Global Quote' in data:
                quote = data['Global Quote']
                # Check if quote is empty or None
                if not quote or (isinstance(quote, dict) and len(quote) == 0):
                    logger.warning(f"Alpha Vantage quote for {symbol}: Empty Global Quote")
                    return None
                return {
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': float(quote.get('10. change percent', 0).rstrip('%')),
                    'volume': int(quote.get('06. volume', 0)),
                    'high': float(quote.get('03. high', 0)),
                    'low': float(quote.get('04. low', 0)),
                    'open': float(quote.get('02. open', 0))
                }
            
            # If response only has 'Information' key, it's likely rate-limited
            if 'Information' in data and len(data) == 1:
                logger.warning(f"Alpha Vantage quote for {symbol}: Rate-limited - {data.get('Information', 'No data available')[:100]}")
                return None
            
            # Other cases - log warning
            logger.warning(f"Alpha Vantage quote for {symbol}: No 'Global Quote' in response. Keys: {list(data.keys())}")
            return None
            
        except Exception as e:
            logger.error(f"Alpha Vantage quote error for {symbol}: {e}")
            return None

    def _get_overview(self, symbol):
        """Get company overview from Alpha Vantage"""
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.advisor.key
            }
            
            response = requests.get(self.advisor.endpoint, params=params, timeout=30)
            
            # Check HTTP status first - if not 200, return silently (already logged in _get_quote)
            if response.status_code != 200:
                logger.warning(f"{self.advisor.name} returns {response}")
                return {}
            
            data = response.json()
            
            # Check for API error messages
            if 'Note' in data:
                # Don't log again if quote already logged
                return {}
            
            if 'Error Message' in data:
                # Don't log again if quote already logged
                return {}
            
            # Check if we got actual data (overview should have multiple fields)
            if not data or len(data) < 5:
                # Don't log - insufficient data is expected when rate limited
                return {}
            
            return data
            
        except Exception as e:
            logger.error(f"Alpha Vantage overview error for {symbol}: {e}")
            return {}

    def _calculate_confidence(self, quote_data, overview_data):
        """Calculate confidence score based on Alpha Vantage metrics"""
        score = 0.5  # Start neutral
        
        # PRICE MOMENTUM ANALYSIS
        change_percent = quote_data.get('change_percent', 0)
        if change_percent > 5.0:  # >5% daily gain
            score += 0.2
        elif change_percent > 2.0:  # >2% daily gain
            score += 0.1
        elif change_percent < -5.0:  # >5% daily loss
            score -= 0.2
        elif change_percent < -2.0:  # >2% daily loss
            score -= 0.1
        
        # VALUATION ANALYSIS (from overview)
        pe_ratio = overview_data.get('PERatio')
        if pe_ratio and pe_ratio != 'None':
            pe_ratio = float(pe_ratio)
            if pe_ratio < 1.0:  # Very low P/E = distressed
                score -= 0.3
            elif pe_ratio < 15:
                score += 0.2
            elif pe_ratio > 30:
                score -= 0.3
            elif pe_ratio > 25:
                score -= 0.1
        
        # PROFITABILITY ANALYSIS
        profit_margin = overview_data.get('ProfitMargin')
        if profit_margin and profit_margin != 'None':
            profit_margin = float(profit_margin)
            if profit_margin < 0:  # Losing money
                score -= 0.2
            elif profit_margin > 0.20:  # >20%
                score += 0.2
            elif profit_margin > 0.10:  # >10%
                score += 0.1
            elif profit_margin < 0.05:  # <5%
                score -= 0.1
        
        # MARKET CAP ANALYSIS
        market_cap = overview_data.get('MarketCapitalization')
        if market_cap and market_cap != 'None':
            market_cap = float(market_cap)
            if market_cap > 1_000_000_000_000:  # >$1T
                score += 0.1
            elif market_cap > 100_000_000_000:  # >$100B
                score += 0.05
            elif market_cap < 100_000_000:  # <$100M
                score -= 0.1
        
        # VOLUME ANALYSIS
        volume = quote_data.get('volume', 0)
        if volume > 1_000_000:  # High volume
            score += 0.05
        elif volume < 100_000:  # Low volume
            score -= 0.05
        
        # DEBT ANALYSIS
        debt_to_equity = overview_data.get('DebtToEquity')
        if debt_to_equity and debt_to_equity != 'None':
            debt_to_equity = float(debt_to_equity)
            if debt_to_equity > 2.0:  # Very high debt
                score -= 0.2
            elif debt_to_equity > 1.0:  # High debt
                score -= 0.1
            elif debt_to_equity < 0.5:  # Low debt
                score += 0.1
        
        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, score))
        
        return final_score


register(name="Alpha Vantage", python_class="Alpha")