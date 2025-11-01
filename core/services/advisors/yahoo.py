import yfinance as yf
import random

from core.services.advisors.advisor import AdvisorBase, register
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

class Yahoo(AdvisorBase):

    def __init__(self, advisor):
        if isinstance(advisor, str):
            # If passed a string, use it as the name
            super().__init__(advisor)
            self.advisor = None
        else:
            # If passed an advisor object, use its name
            super().__init__(advisor.name)
            self.advisor = advisor

    def discover(self, sa):

        """Discover stocks using Yahoo Finance screeners"""
        try:
            discovered_symbols = []
            
            # Method 1: Undervalued Growth Stocks (best quality)
            try:
                query = yf.PREDEFINED_SCREENER_QUERIES['undervalued_growth_stocks']['query']
                data = yf.screen(query)
                
                if isinstance(data, dict) and 'quotes' in data:
                    for quote in data['quotes'][:8]:  # Top 8
                        if 'symbol' in quote:
                            symbol = quote['symbol']
                            company = quote.get('longName', symbol)
                            explanation = "Yahoo: Undervalued growth stock"
                            
                            self.discovered(sa, symbol, company, explanation)
                            discovered_symbols.append(symbol)
                            # logger.info(f"Yahoo discovered {symbol}: {explanation}")
            except Exception as e:
                logger.warning(f"Yahoo growth stocks discovery failed: {e}")
            
            # Method 2: Undervalued Large Caps (stability)
            try:
                query = yf.PREDEFINED_SCREENER_QUERIES['undervalued_large_caps']['query']
                data = yf.screen(query)
                
                if isinstance(data, dict) and 'quotes' in data:
                    for quote in data['quotes'][:6]:  # Top 6
                        if 'symbol' in quote:
                            symbol = quote['symbol']
                            # Skip if already discovered
                            if symbol in discovered_symbols:
                                continue
                            
                            company = quote.get('longName', symbol)
                            explanation = "Yahoo: Undervalued large cap"
                            
                            self.discovered(sa, symbol, company, explanation)
                            discovered_symbols.append(symbol)
                            #logger.info(f"Yahoo discovered {symbol}: {explanation}")
            except Exception as e:
                logger.warning(f"Yahoo large caps discovery failed: {e}")
            
            # Method 3: High-Risk/High-Reward (bouncing stocks)
            # Day losers - oversold stocks with bounce potential
            try:
                query = yf.PREDEFINED_SCREENER_QUERIES['day_losers']['query']
                data = yf.screen(query)
                
                if isinstance(data, dict) and 'quotes' in data:
                    for quote in data['quotes'][:5]:  # Top 5 losers (oversold)
                        if 'symbol' in quote:
                            symbol = quote['symbol']
                            # Skip if already discovered
                            if symbol in discovered_symbols:
                                continue
                            
                            company = quote.get('longName', symbol)
                            explanation = "Yahoo: High-risk oversold (bounce potential)"
                            
                            self.discovered(sa, symbol, company, explanation)
                            discovered_symbols.append(symbol)
                            logger.info(f"Yahoo discovered {symbol}: {explanation}")
            except Exception as e:
                logger.warning(f"Yahoo day losers discovery failed: {e}")
            
            # Small cap gainers - volatile momentum plays
            try:
                query = yf.PREDEFINED_SCREENER_QUERIES['small_cap_gainers']['query']
                data = yf.screen(query)
                
                if isinstance(data, dict) and 'quotes' in data:
                    for quote in data['quotes'][:5]:  # Top 5 small cap gainers
                        if 'symbol' in quote:
                            symbol = quote['symbol']
                            # Skip if already discovered
                            if symbol in discovered_symbols:
                                continue
                            
                            company = quote.get('longName', symbol)
                            explanation = "Yahoo: High-risk small cap momentum"
                            
                            self.discovered(sa, symbol, company, explanation)
                            discovered_symbols.append(symbol)
                            logger.info(f"Yahoo discovered {symbol}: {explanation}")
            except Exception as e:
                logger.warning(f"Yahoo small cap gainers discovery failed: {e}")
            
            logger.info(f"Yahoo Finance discovery complete: {len(discovered_symbols)} stocks found")
            
        except Exception as e:
            logger.error(f"Yahoo Finance discovery failed: {e}")

    def analyze(self, sa, stock):
        """Analyze stock using Yahoo Finance data"""
        try:
            # Get real-time data
            ticker = yf.Ticker(stock.symbol)
            info = ticker.info

            # Update stock price
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if current_price:
                stock.price = Decimal(str(current_price))

                # Update name if user discovered
                if stock.company == "":
                    stock.company = info.get('longName') or info.get('shortName')

                # Update exchange if unknown
                if stock.exchange == "":
                    stock.exchange = info.get('fullExchangeName')

                # 1ogo_url
                stock.save()

            # Calculate confidence based on real metrics
            confidence = self._calculate_confidence(info, stock.symbol)
            
            # Build detailed analysis explanation
            explanation_parts = []
            explanation_parts.append(f"Confidence Score: {confidence:.2f}")
            
            # Add key factors that influenced the score
            pe_ratio = info.get('trailingPE', 0)
            if pe_ratio > 0:
                if pe_ratio < 1.0:
                    explanation_parts.append(f"⚠️ DISTRESSED: P/E={pe_ratio:.2f}")
                elif pe_ratio > 30:
                    explanation_parts.append(f"🔴 OVERVALUED: P/E={pe_ratio:.2f}")
                elif pe_ratio < 15:
                    explanation_parts.append(f"🟢 UNDERVALUED: P/E={pe_ratio:.2f}")
            
            profit_margin = info.get('profitMargins', 0)
            if profit_margin > 0.20:
                explanation_parts.append(f"💪 STRONG PROFITS: {profit_margin:.1%} margin")
            elif profit_margin < 0:
                explanation_parts.append(f"💸 LOSING MONEY: {profit_margin:.1%} margin")
            elif profit_margin < 0.05:
                explanation_parts.append(f"📉 WEAK PROFITS: {profit_margin:.1%} margin")
            
            change_52w = info.get('fiftyTwoWeekChangePercent', 0)
            if change_52w < -0.50:
                explanation_parts.append(f"📉 CATASTROPHIC: {change_52w:.1%} yearly loss")
            elif change_52w < -0.20:
                explanation_parts.append(f"📉 MAJOR DECLINE: {change_52w:.1%} yearly loss")
            elif change_52w > 0.20:
                explanation_parts.append(f"🚀 STRONG MOMENTUM: {change_52w:.1%} yearly gain")
            elif change_52w > 0.10:
                explanation_parts.append(f"📈 POSITIVE MOMENTUM: {change_52w:.1%} yearly gain")
            
            market_cap = info.get('marketCap', 0)
            if market_cap < 100_000_000:
                explanation_parts.append(f"⚠️ MICRO-CAP RISK: ${market_cap:,.0f}")
            elif market_cap > 1_000_000_000_000:
                explanation_parts.append(f"🏢 MEGA-CAP STABILITY: ${market_cap:,.0f}")
            
            
            debt_to_equity = info.get('debtToEquity', 0)
            if debt_to_equity > 2.0:
                explanation_parts.append(f"⚠️ HIGH DEBT: {debt_to_equity:.1f} D/E")
            elif debt_to_equity < 0.5:
                explanation_parts.append(f"💪 LOW DEBT: {debt_to_equity:.1f} D/E")
            
            # Add target price information if available
            target_mean_price = info.get('targetMeanPrice')
            if target_mean_price:
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if current_price:
                    upside_percent = ((target_mean_price - current_price) / current_price) * 100
                    if upside_percent > 20:
                        explanation_parts.append(f"🎯 STRONG TARGET: ${target_mean_price:.2f} (+{upside_percent:.1f}% upside)")
                    elif upside_percent > 10:
                        explanation_parts.append(f"🎯 TARGET: ${target_mean_price:.2f} (+{upside_percent:.1f}% upside)")
                    elif upside_percent > 0:
                        explanation_parts.append(f"🎯 MODEST TARGET: ${target_mean_price:.2f} (+{upside_percent:.1f}% upside)")
                    else:
                        explanation_parts.append(f"🎯 BELOW TARGET: ${target_mean_price:.2f} ({upside_percent:.1f}% downside)")
            
            explanation = " | ".join(explanation_parts)
            return super().recommend(sa, stock, confidence, explanation)

        except Exception as e:
            logger.error(f"Error analyzing {stock.symbol}: {e}")

    def _calculate_confidence(self, info, symbol):
        """Calculate confidence score based on Yahoo Finance metrics including target price"""
        score = 0.5  # Start neutral

        # VALUATION ANALYSIS
        pe_ratio = info.get('trailingPE', 0)
        if pe_ratio > 0:
            if pe_ratio < 1.0:  # Very low P/E = distressed company
                score -= 0.3
            elif pe_ratio < 15:
                score += 0.2
            elif pe_ratio > 30:
                score -= 0.3
            elif pe_ratio > 25:
                score -= 0.1

        # PROFITABILITY ANALYSIS
        profit_margin = info.get('profitMargins', 0)
        if profit_margin < 0:  # Losing money
            score -= 0.2
        elif profit_margin > 0.20:  # >20%
            score += 0.2
        elif profit_margin > 0.10:  # >10%
            score += 0.1
        elif profit_margin < 0.05:  # <5%
            score -= 0.1

        # MOMENTUM ANALYSIS
        change_52w = info.get('fiftyTwoWeekChangePercent', 0)
        if change_52w < -0.50:  # >50% loss = catastrophic
            score -= 0.3
        elif change_52w < -0.20:  # >20% loss = major decline
            score -= 0.2
        elif change_52w < -0.10:  # >10% loss
            score -= 0.1
        elif change_52w > 0.20:
            score += 0.15
        elif change_52w > 0.10:
            score += 0.1

        # MARKET POSITION ANALYSIS
        market_cap = info.get('marketCap', 0)
        if market_cap > 1_000_000_000_000:  # >$1T
            score += 0.1
        elif market_cap > 100_000_000_000:  # >$100B
            score += 0.05
        elif market_cap < 100_000_000:  # <$100M = micro-cap risk
            score -= 0.1

        # ANALYST CONSENSUS
        analyst_rating = info.get('recommendationMean', 3.0)
        if analyst_rating < 2.0:
            score += 0.15
        elif analyst_rating < 2.5:
            score += 0.1
        elif analyst_rating > 3.5:
            score -= 0.1

        # DEBT ANALYSIS
        debt_to_equity = info.get('debtToEquity', 0)
        if debt_to_equity > 2.0:
            score -= 0.2
        elif debt_to_equity > 1.0:
            score -= 0.1
        elif debt_to_equity < 0.5:
            score += 0.1

        # VOLUME ANALYSIS
        volume = info.get('volume', 0)
        avg_volume = info.get('averageVolume', 0)
        if avg_volume > 0:
            volume_ratio = volume / avg_volume
            if volume > avg_volume * 1.5:
                score += 0.05
            elif volume < avg_volume * 0.5:
                score -= 0.05

        # TARGET PRICE ANALYSIS
        target_mean_price = info.get('targetMeanPrice')
        if target_mean_price:
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if current_price and current_price > 0:
                upside_percent = ((target_mean_price - current_price) / current_price) * 100
                # Conservative scaling: 50% weight of upside percentage
                target_score_adjustment = (upside_percent / 100) * 0.5
                score += target_score_adjustment

        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, score))

        return final_score


register(name="Yahoo Finances", python_class="Yahoo")