import yfinance as yf
import random

from core.services.advisors.advisor import AdvisorBase, register
from decimal import Decimal

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
        pass

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
            
            explanation = " | ".join(explanation_parts)
            return super().recommend(sa, stock, confidence, explanation)

        except Exception as e:
            print(f"Error analyzing {stock.symbol}: {e}")

    def _calculate_confidence(self, info, symbol):
        """Calculate confidence score based on Yahoo Finance metrics"""
        score = 0.5  # Start neutral
        print(f"\n=== Confidence Calculation {symbol} ===")
        print(f"Starting score: {score}")

        # VALUATION ANALYSIS
        pe_ratio = info.get('trailingPE', 0)
        if pe_ratio > 0:
            if pe_ratio < 1.0:  # Very low P/E = distressed company
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
        profit_margin = info.get('profitMargins', 0)
        if profit_margin < 0:  # Losing money
            score -= 0.2
            print(f"- Profitability (margin={profit_margin:.1%} NEGATIVE): -0.2 → {score}")
        elif profit_margin > 0.20:
            score += 0.2
            print(f"+ Profitability (margin={profit_margin:.1%} > 20%): +0.2 → {score}")
        elif profit_margin > 0.10:
            score += 0.1
            print(f"+ Profitability (margin={profit_margin:.1%} > 10%): +0.1 → {score}")
        elif profit_margin < 0.05:
            score -= 0.1
            print(f"- Profitability (margin={profit_margin:.1%} < 5%): -0.1 → {score}")
        else:
            print(f"  Profitability (margin={profit_margin:.1%}): neutral")

        # MOMENTUM ANALYSIS
        change_52w = info.get('fiftyTwoWeekChangePercent', 0)
        if change_52w < -0.50:  # >50% loss = catastrophic
            score -= 0.3
            print(f"- Momentum (52w change={change_52w:.1%} CATASTROPHIC): -0.3 → {score}")
        elif change_52w < -0.20:  # >20% loss = major decline
            score -= 0.2
            print(f"- Momentum (52w change={change_52w:.1%} MAJOR DECLINE): -0.2 → {score}")
        elif change_52w < -0.10:  # >10% loss
            score -= 0.1
            print(f"- Momentum (52w change={change_52w:.1%} < -10%): -0.1 → {score}")
        elif change_52w > 0.20:
            score += 0.15
            print(f"+ Momentum (52w change={change_52w:.1%} > 20%): +0.15 → {score}")
        elif change_52w > 0.10:
            score += 0.1
            print(f"+ Momentum (52w change={change_52w:.1%} > 10%): +0.1 → {score}")
        else:
            print(f"  Momentum (52w change={change_52w:.1%}): neutral")

        # MARKET POSITION ANALYSIS
        market_cap = info.get('marketCap', 0)
        if market_cap > 1_000_000_000_000:  # >$1T
            score += 0.1
            print(f"+ Market Cap (${market_cap:,.0f} > $1T): +0.1 → {score}")
        elif market_cap > 100_000_000_000:  # >$100B
            score += 0.05
            print(f"+ Market Cap (${market_cap:,.0f} > $100B): +0.05 → {score}")
        elif market_cap < 100_000_000:  # <$100M = micro-cap risk
            score -= 0.1
            print(f"- Market Cap (${market_cap:,.0f} < $100M MICRO-CAP): -0.1 → {score}")
        else:
            print(f"  Market Cap (${market_cap:,.0f}): neutral")

            # ANALYST CONSENSUS
            analyst_rating = info.get('recommendationMean', 3.0)
            if analyst_rating < 2.0:
                score += 0.15
                print(f"+ Analyst Rating ({analyst_rating:.2f} < 2.0 Strong Buy): +0.15 → {score}")
            elif analyst_rating < 2.5:
                score += 0.1
                print(f"+ Analyst Rating ({analyst_rating:.2f} < 2.5 Buy): +0.1 → {score}")
            elif analyst_rating > 3.5:
                score -= 0.1
                print(f"- Analyst Rating ({analyst_rating:.2f} > 3.5 Sell): -0.1 → {score}")
            else:
                print(f"  Analyst Rating ({analyst_rating:.2f}): neutral")

            # DEBT ANALYSIS
            debt_to_equity = info.get('debtToEquity', 0)
            if debt_to_equity > 2.0:
                score -= 0.2
                print(f"- Debt (D/E={debt_to_equity:.2f} > 2.0): -0.2 → {score}")
            elif debt_to_equity > 1.0:
                score -= 0.1
                print(f"- Debt (D/E={debt_to_equity:.2f} > 1.0): -0.1 → {score}")
            elif debt_to_equity < 0.5:
                score += 0.1
                print(f"+ Debt (D/E={debt_to_equity:.2f} < 0.5): +0.1 → {score}")
            else:
                print(f"  Debt (D/E={debt_to_equity:.2f}): neutral")

            # VOLUME ANALYSIS
            volume = info.get('volume', 0)
            avg_volume = info.get('averageVolume', 0)
            if avg_volume > 0:
                volume_ratio = volume / avg_volume
                if volume > avg_volume * 1.5:
                    score += 0.05
                    print(f"+ Volume ({volume_ratio:.2f}x avg): +0.05 → {score}")
                elif volume < avg_volume * 0.5:
                    score -= 0.05
                    print(f"- Volume ({volume_ratio:.2f}x avg): -0.05 → {score}")
                else:
                    print(f"  Volume ({volume_ratio:.2f}x avg): neutral")

        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, score))
        print(f"\nFinal confidence score: {final_score}")
        print(f"=== End Calculation ===\n")

        return final_score


register(name="Yahoo Finances", python_class="Yahoo")