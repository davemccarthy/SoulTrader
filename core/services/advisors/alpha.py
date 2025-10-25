import requests
import time
from decimal import Decimal
from core.services.advisors.advisor import AdvisorBase, register

class Alpha(AdvisorBase):

    def __init__(self, name):
        super().__init__(name)
        # Get API credentials from database
        from core.models import Advisor
        advisor = Advisor.objects.get(name=name)
        self.api_key = advisor.key
        self.endpoint = advisor.endpoint
        self.base_url = self.endpoint if self.endpoint else "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # seconds between API calls TODO might add to dbase

    def discover(self, sa):
        """Skip discovery - we analyze existing holdings"""
        return  # No auto-discovery

    def analyze(self, sa, stock):
        """Analyze stock using Alpha Vantage data"""
        try:
            # Get comprehensive data from Alpha Vantage
            quote_data = self._get_quote(stock.symbol)
            overview_data = self._get_overview(stock.symbol)
            
            if not quote_data:
                return 0.5  # Neutral on error
            
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
            print(f"Error analyzing {stock.symbol}: {e}")
            return 0.5  # Neutral on error

    def _get_quote(self, symbol):
        """Get real-time quote from Alpha Vantage"""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parse Alpha Vantage response
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return {
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': float(quote.get('10. change percent', 0).rstrip('%')),
                    'volume': int(quote.get('06. volume', 0)),
                    'high': float(quote.get('03. high', 0)),
                    'low': float(quote.get('04. low', 0)),
                    'open': float(quote.get('02. open', 0))
                }
            return None
            
        except Exception as e:
            print(f"Alpha Vantage quote error for {symbol}: {e}")
            return None

    def _get_overview(self, symbol):
        """Get company overview from Alpha Vantage"""
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return data
            
        except Exception as e:
            print(f"Alpha Vantage overview error for {symbol}: {e}")
            return {}

    def _calculate_confidence(self, quote_data, overview_data):
        """Calculate confidence score based on Alpha Vantage metrics"""
        score = 0.5  # Start neutral
        print(f"\n=== Alpha Vantage Confidence Calculation ===")
        print(f"Starting score: {score}")
        
        # PRICE MOMENTUM ANALYSIS
        change_percent = quote_data.get('change_percent', 0)
        if change_percent > 5.0:  # >5% daily gain
            score += 0.2
            print(f"+ Daily Momentum ({change_percent:.1f}% > 5%): +0.2 → {score}")
        elif change_percent > 2.0:  # >2% daily gain
            score += 0.1
            print(f"+ Daily Momentum ({change_percent:.1f}% > 2%): +0.1 → {score}")
        elif change_percent < -5.0:  # >5% daily loss
            score -= 0.2
            print(f"- Daily Momentum ({change_percent:.1f}% < -5%): -0.2 → {score}")
        elif change_percent < -2.0:  # >2% daily loss
            score -= 0.1
            print(f"- Daily Momentum ({change_percent:.1f}% < -2%): -0.1 → {score}")
        else:
            print(f"  Daily Momentum ({change_percent:.1f}%): neutral")
        
        # VALUATION ANALYSIS (from overview)
        pe_ratio = overview_data.get('PERatio')
        if pe_ratio and pe_ratio != 'None':
            pe_ratio = float(pe_ratio)
            if pe_ratio < 1.0:  # Very low P/E = distressed
                score -= 0.3
                print(f"- Valuation (P/E={pe_ratio:.2f} < 1.0 DISTRESSED): -0.3 → {score}")
            elif pe_ratio < 15:
                score += 0.2
                print(f"+ Valuation (P/E={pe_ratio:.2f} < 15): +0.2 → {score}")
            elif pe_ratio > 30:
                score -= 0.3
                print(f"- Valuation (P/E={pe_ratio:.2f} > 30): -0.3 → {score}")
            elif pe_ratio > 25:
                score -= 0.1
                print(f"- Valuation (P/E={pe_ratio:.2f} > 25): -0.1 → {score}")
            else:
                print(f"  Valuation (P/E={pe_ratio:.2f}): neutral")
        else:
            print(f"  Valuation: P/E not available")
        
        # PROFITABILITY ANALYSIS
        profit_margin = overview_data.get('ProfitMargin')
        if profit_margin and profit_margin != 'None':
            profit_margin = float(profit_margin)
            if profit_margin < 0:  # Losing money
                score -= 0.2
                print(f"- Profitability (margin={profit_margin:.1%} NEGATIVE): -0.2 → {score}")
            elif profit_margin > 0.20:  # >20%
                score += 0.2
                print(f"+ Profitability (margin={profit_margin:.1%} > 20%): +0.2 → {score}")
            elif profit_margin > 0.10:  # >10%
                score += 0.1
                print(f"+ Profitability (margin={profit_margin:.1%} > 10%): +0.1 → {score}")
            elif profit_margin < 0.05:  # <5%
                score -= 0.1
                print(f"- Profitability (margin={profit_margin:.1%} < 5%): -0.1 → {score}")
            else:
                print(f"  Profitability (margin={profit_margin:.1%}): neutral")
        else:
            print(f"  Profitability: margin not available")
        
        # MARKET CAP ANALYSIS
        market_cap = overview_data.get('MarketCapitalization')
        if market_cap and market_cap != 'None':
            market_cap = float(market_cap)
            if market_cap > 1_000_000_000_000:  # >$1T
                score += 0.1
                print(f"+ Market Cap (${market_cap:,.0f} > $1T): +0.1 → {score}")
            elif market_cap > 100_000_000_000:  # >$100B
                score += 0.05
                print(f"+ Market Cap (${market_cap:,.0f} > $100B): +0.05 → {score}")
            elif market_cap < 100_000_000:  # <$100M
                score -= 0.1
                print(f"- Market Cap (${market_cap:,.0f} < $100M MICRO-CAP): -0.1 → {score}")
            else:
                print(f"  Market Cap (${market_cap:,.0f}): neutral")
        else:
            print(f"  Market Cap: not available")
        
        # VOLUME ANALYSIS
        volume = quote_data.get('volume', 0)
        if volume > 1_000_000:  # High volume
            score += 0.05
            print(f"+ Volume ({volume:,} > 1M): +0.05 → {score}")
        elif volume < 100_000:  # Low volume
            score -= 0.05
            print(f"- Volume ({volume:,} < 100K): -0.05 → {score}")
        else:
            print(f"  Volume ({volume:,}): neutral")
        
        # DEBT ANALYSIS
        debt_to_equity = overview_data.get('DebtToEquity')
        if debt_to_equity and debt_to_equity != 'None':
            debt_to_equity = float(debt_to_equity)
            if debt_to_equity > 2.0:  # Very high debt
                score -= 0.2
                print(f"- Debt (D/E={debt_to_equity:.2f} > 2.0): -0.2 → {score}")
            elif debt_to_equity > 1.0:  # High debt
                score -= 0.1
                print(f"- Debt (D/E={debt_to_equity:.2f} > 1.0): -0.1 → {score}")
            elif debt_to_equity < 0.5:  # Low debt
                score += 0.1
                print(f"+ Debt (D/E={debt_to_equity:.2f} < 0.5): +0.1 → {score}")
            else:
                print(f"  Debt (D/E={debt_to_equity:.2f}): neutral")
        else:
            print(f"  Debt: D/E not available")
        
        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, score))
        print(f"\nFinal confidence score: {final_score}")
        print(f"=== End Alpha Vantage Calculation ===\n")
        
        return final_score


register(name="Alpha Vantage", python_class="Alpha")