import requests
import json
from datetime import datetime, timedelta
from decimal import Decimal
from core.services.advisors.advisor import AdvisorBase, register

class Finnhub(AdvisorBase):

    def __init__(self, name):
        super().__init__(name)
        # Get API credentials from database
        from core.models import Advisor
        advisor = Advisor.objects.get(name=name)
        self.api_key = advisor.key
        self.base_url = advisor.endpoint

    def discover(self, sa):
        """Skip discovery - we analyze existing holdings"""
        return  # No auto-discovery

    def analyze(self, sa, stock):
        """Analyze stock using Finnhub data"""
        try:
            print(f"\n=== Finnhub API Calls for {stock.symbol} ===")
            
            # Get comprehensive data from Finnhub
            recommendation_trends = self._get_recommendation_trends(stock.symbol)
            price_target = self._get_price_target(stock.symbol)
            basic_financials = self._get_basic_financials(stock.symbol)
            company_news = self._get_company_news(stock.symbol)
            
            # Calculate confidence based on Finnhub data
            confidence = self._calculate_confidence(
                stock, recommendation_trends, price_target, basic_financials, company_news
            )
            
            # Build detailed analysis explanation
            explanation_parts = []
            explanation_parts.append(f"Confidence Score: {confidence:.2f}")
            
            # Add key factors from Finnhub analysis
            if recommendation_trends and len(recommendation_trends) > 0:
                latest_rec = recommendation_trends[0]
                strong_buy = latest_rec.get('strongBuy', 0)
                buy = latest_rec.get('buy', 0)
                hold = latest_rec.get('hold', 0)
                sell = latest_rec.get('sell', 0)
                strong_sell = latest_rec.get('strongSell', 0)
                total = strong_buy + buy + hold + sell + strong_sell
                
                if total > 0:
                    buy_pct = (strong_buy + buy) / total
                    if buy_pct > 0.6:
                        explanation_parts.append(f"⭐ ANALYST CONSENSUS: {buy_pct:.0%} buy ratings")
                    elif buy_pct < 0.3:
                        explanation_parts.append(f"⚠️ ANALYST CONCERN: {(sell + strong_sell)/total:.0%} sell ratings")
            
            if price_target:
                target_mean = price_target.get('targetMean')
                if target_mean and stock.price:
                    upside = ((target_mean - float(stock.price)) / float(stock.price)) * 100
                    if upside > 20:
                        explanation_parts.append(f"🎯 HIGH UPSIDE: {upside:.0f}% to target")
                    elif upside < -20:
                        explanation_parts.append(f"📉 OVERVALUED: {upside:.0f}% from target")
            
            if company_news:
                explanation_parts.append(f"📰 NEWS: {len(company_news)} recent articles")
            
            if basic_financials and 'metric' in basic_financials:
                metrics = basic_financials['metric']
                pe_ratio = metrics.get('peNormalizedAnnual')
                if pe_ratio:
                    if pe_ratio < 15:
                        explanation_parts.append(f"🟢 UNDERVALUED: P/E={pe_ratio:.1f}")
                    elif pe_ratio > 30:
                        explanation_parts.append(f"🔴 OVERVALUED: P/E={pe_ratio:.1f}")
            
            explanation = " | ".join(explanation_parts)
            print(f"\n🔍 DEBUG: About to call super().recommend with confidence: {confidence}")
            super().recommend(sa, stock, confidence, explanation)
            print(f"🔍 DEBUG: After super().recommend call, returning: {confidence}")
            
            return confidence
            
        except Exception as e:
            print(f"Error analyzing {stock.symbol} with Finnhub: {e}")
            return 0.5  # Neutral on error

    def _get_recommendation_trends(self, symbol):
        """Get analyst recommendation trends"""
        try:
            url = f"{self.base_url}/stock/recommendation"
            params = {'symbol': symbol, 'token': self.api_key}
            
            response = requests.get(url, params=params, timeout=30)
            
            # Log response status for debugging
            print(f"Recommendation Trends API: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                return data
            elif response.status_code == 429:
                print("⚠️ RATE LIMIT HIT (429) - Too Many Requests")
                return None
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Recommendation trends error for {symbol}: {e}")
            return None

    def _get_price_target(self, symbol):
        """Get analyst price targets"""
        try:
            url = f"{self.base_url}/stock/price-target"
            params = {'symbol': symbol, 'token': self.api_key}
            
            response = requests.get(url, params=params, timeout=30)
            
            # Log response status for debugging
            print(f"Price Target API: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                return data
            elif response.status_code == 429:
                print("⚠️ RATE LIMIT HIT (429) - Too Many Requests")
                return None
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Price target error for {symbol}: {e}")
            return None

    def _get_basic_financials(self, symbol):
        """Get basic financial metrics"""
        try:
            url = f"{self.base_url}/stock/metric"
            params = {'symbol': symbol, 'metric': 'all', 'token': self.api_key}
            
            response = requests.get(url, params=params, timeout=30)
            
            # Log response status for debugging
            print(f"Basic Financials API: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                return data
            elif response.status_code == 429:
                print("⚠️ RATE LIMIT HIT (429) - Too Many Requests")
                return None
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Basic financials error for {symbol}: {e}")
            return None

    def _get_company_news(self, symbol):
        """Get recent company news for sentiment analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            url = f"{self.base_url}/company-news"
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            # Log response status for debugging
            print(f"Company News API: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Total articles: {len(data)}")
                return data
            elif response.status_code == 429:
                print("⚠️ RATE LIMIT HIT (429) - Too Many Requests")
                return None
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Company news error for {symbol}: {e}")
            return None

    def _calculate_confidence(self, stock, rec_trends, price_target, financials, news):
        """Calculate confidence score based on Finnhub data with granular scoring"""
        score = 0.5  # Start neutral
        print(f"\n=== Finnhub Confidence Calculation for {stock.symbol} ===")
        print(f"Starting score: {score}")
        
        # ANALYST RECOMMENDATION ANALYSIS (More Granular)
        if rec_trends and len(rec_trends) > 0:
            latest_rec = rec_trends[0]
            
            strong_buy = latest_rec.get('strongBuy', 0)
            buy = latest_rec.get('buy', 0)
            hold = latest_rec.get('hold', 0)
            sell = latest_rec.get('sell', 0)
            strong_sell = latest_rec.get('strongSell', 0)
            
            total_recs = strong_buy + buy + hold + sell + strong_sell
            
            if total_recs > 0:
                buy_percentage = (strong_buy + buy) / total_recs
                sell_percentage = (sell + strong_sell) / total_recs
                strong_buy_percentage = strong_buy / total_recs
                
                # More granular analyst scoring
                if buy_percentage >= 0.90:  # >90% buy (exceptional)
                    score += 0.4
                    print(f"+ Exceptional Consensus ({buy_percentage:.0%} buy): +0.4 → {score}")
                elif buy_percentage >= 0.80:  # >80% buy (very strong)
                    score += 0.3
                    print(f"+ Strong Consensus ({buy_percentage:.0%} buy): +0.3 → {score}")
                elif buy_percentage >= 0.70:  # >70% buy (strong)
                    score += 0.2
                    print(f"+ Good Consensus ({buy_percentage:.0%} buy): +0.2 → {score}")
                elif buy_percentage >= 0.60:  # >60% buy (moderate)
                    score += 0.1
                    print(f"+ Moderate Consensus ({buy_percentage:.0%} buy): +0.1 → {score}")
                elif buy_percentage >= 0.50:  # >50% buy (weak)
                    score += 0.05
                    print(f"+ Weak Consensus ({buy_percentage:.0%} buy): +0.05 → {score}")
                elif sell_percentage >= 0.60:  # >60% sell (very concerning)
                    score -= 0.3
                    print(f"- Major Concern ({sell_percentage:.0%} sell): -0.3 → {score}")
                elif sell_percentage >= 0.40:  # >40% sell (concerning)
                    score -= 0.2
                    print(f"- Analyst Concern ({sell_percentage:.0%} sell): -0.2 → {score}")
                elif sell_percentage >= 0.30:  # >30% sell (slight concern)
                    score -= 0.1
                    print(f"- Slight Concern ({sell_percentage:.0%} sell): -0.1 → {score}")
                else:
                    print(f"  Analyst Consensus ({buy_percentage:.0%} buy, {sell_percentage:.0%} sell): neutral")
        else:
            print(f"  Analyst Consensus: not available")
        
        # PRICE TARGET ANALYSIS (More Granular)
        if price_target:
            target_mean = price_target.get('targetMean')
            if target_mean and stock.price:
                current_price = float(stock.price)
                upside_percent = ((target_mean - current_price) / current_price)
                
                if upside_percent > 0.50:  # >50% upside (exceptional)
                    score += 0.3
                    print(f"+ Exceptional Upside ({upside_percent:.0%} to target): +0.3 → {score}")
                elif upside_percent > 0.30:  # >30% upside (high)
                    score += 0.2
                    print(f"+ High Upside ({upside_percent:.0%} to target): +0.2 → {score}")
                elif upside_percent > 0.20:  # >20% upside (good)
                    score += 0.15
                    print(f"+ Good Upside ({upside_percent:.0%} to target): +0.15 → {score}")
                elif upside_percent > 0.10:  # >10% upside (moderate)
                    score += 0.1
                    print(f"+ Moderate Upside ({upside_percent:.0%} to target): +0.1 → {score}")
                elif upside_percent > 0.05:  # >5% upside (slight)
                    score += 0.05
                    print(f"+ Slight Upside ({upside_percent:.0%} to target): +0.05 → {score}")
                elif upside_percent < -0.30:  # >30% overvalued (major concern)
                    score -= 0.25
                    print(f"- Major Overvaluation ({upside_percent:.0%} from target): -0.25 → {score}")
                elif upside_percent < -0.20:  # >20% overvalued (concerning)
                    score -= 0.15
                    print(f"- Overvalued ({upside_percent:.0%} from target): -0.15 → {score}")
                elif upside_percent < -0.10:  # >10% overvalued (slight concern)
                    score -= 0.05
                    print(f"- Slightly Overvalued ({upside_percent:.0%} from target): -0.05 → {score}")
                else:
                    print(f"  Price Target ({upside_percent:.0%} from target): neutral")
        else:
            print(f"  Price Target: not available")
        
        # FINANCIAL METRICS ANALYSIS (More Granular)
        if financials and 'metric' in financials:
            metrics = financials['metric']
            
            # P/E Ratio (More Granular)
            pe_ratio = metrics.get('peNormalizedAnnual')
            if pe_ratio:
                if pe_ratio < 0.5:  # Extremely distressed
                    score -= 0.4
                    print(f"- Extremely Distressed (P/E={pe_ratio:.1f}): -0.4 → {score}")
                elif pe_ratio < 1.0:  # Distressed
                    score -= 0.3
                    print(f"- Distressed (P/E={pe_ratio:.1f}): -0.3 → {score}")
                elif pe_ratio < 5.0:  # Very undervalued
                    score += 0.25
                    print(f"+ Very Undervalued (P/E={pe_ratio:.1f}): +0.25 → {score}")
                elif pe_ratio < 10.0:  # Undervalued
                    score += 0.2
                    print(f"+ Undervalued (P/E={pe_ratio:.1f}): +0.2 → {score}")
                elif pe_ratio < 15.0:  # Fair value
                    score += 0.1
                    print(f"+ Fair Value (P/E={pe_ratio:.1f}): +0.1 → {score}")
                elif pe_ratio < 20.0:  # Slightly expensive
                    score -= 0.05
                    print(f"- Slightly Expensive (P/E={pe_ratio:.1f}): -0.05 → {score}")
                elif pe_ratio < 30.0:  # Expensive
                    score -= 0.15
                    print(f"- Expensive (P/E={pe_ratio:.1f}): -0.15 → {score}")
                elif pe_ratio < 50.0:  # Very expensive
                    score -= 0.25
                    print(f"- Very Expensive (P/E={pe_ratio:.1f}): -0.25 → {score}")
                else:  # Extremely expensive
                    score -= 0.35
                    print(f"- Extremely Expensive (P/E={pe_ratio:.1f}): -0.35 → {score}")
            
            # ROE (Return on Equity) - More Granular
            roe = metrics.get('roeTTM')
            if roe:
                if roe > 0.50:  # >50% ROE (exceptional)
                    score += 0.2
                    print(f"+ Exceptional ROE ({roe:.1%}): +0.2 → {score}")
                elif roe > 0.30:  # >30% ROE (excellent)
                    score += 0.15
                    print(f"+ Excellent ROE ({roe:.1%}): +0.15 → {score}")
                elif roe > 0.20:  # >20% ROE (very good)
                    score += 0.1
                    print(f"+ Very Good ROE ({roe:.1%}): +0.1 → {score}")
                elif roe > 0.15:  # >15% ROE (good)
                    score += 0.05
                    print(f"+ Good ROE ({roe:.1%}): +0.05 → {score}")
                elif roe > 0.10:  # >10% ROE (acceptable)
                    print(f"  Acceptable ROE ({roe:.1%}): neutral")
                elif roe > 0.05:  # >5% ROE (poor)
                    score -= 0.05
                    print(f"- Poor ROE ({roe:.1%}): -0.05 → {score}")
                elif roe > 0:  # Positive but very low
                    score -= 0.1
                    print(f"- Very Poor ROE ({roe:.1%}): -0.1 → {score}")
                else:  # Negative ROE
                    score -= 0.2
                    print(f"- Negative ROE ({roe:.1%}): -0.2 → {score}")
            
            # Profit Margin (New Metric)
            profit_margin = metrics.get('netMargin', 0)
            if profit_margin:
                if profit_margin > 0.20:  # >20% margin (excellent)
                    score += 0.15
                    print(f"+ Excellent Profitability ({profit_margin:.1%}): +0.15 → {score}")
                elif profit_margin > 0.15:  # >15% margin (very good)
                    score += 0.1
                    print(f"+ Very Good Profitability ({profit_margin:.1%}): +0.1 → {score}")
                elif profit_margin > 0.10:  # >10% margin (good)
                    score += 0.05
                    print(f"+ Good Profitability ({profit_margin:.1%}): +0.05 → {score}")
                elif profit_margin > 0.05:  # >5% margin (acceptable)
                    print(f"  Acceptable Profitability ({profit_margin:.1%}): neutral")
                elif profit_margin > 0:  # Positive but low
                    score -= 0.05
                    print(f"- Low Profitability ({profit_margin:.1%}): -0.05 → {score}")
                else:  # Negative margin
                    score -= 0.15
                    print(f"- Negative Profitability ({profit_margin:.1%}): -0.15 → {score}")
            
            # Debt-to-Equity (New Metric)
            debt_to_equity = metrics.get('totalDebtToEquity', 0)
            if debt_to_equity:
                if debt_to_equity < 0.1:  # <10% D/E (excellent)
                    score += 0.1
                    print(f"+ Excellent Debt Management (D/E={debt_to_equity:.1%}): +0.1 → {score}")
                elif debt_to_equity < 0.3:  # <30% D/E (good)
                    score += 0.05
                    print(f"+ Good Debt Management (D/E={debt_to_equity:.1%}): +0.05 → {score}")
                elif debt_to_equity < 0.5:  # <50% D/E (acceptable)
                    print(f"  Acceptable Debt (D/E={debt_to_equity:.1%}): neutral")
                elif debt_to_equity < 1.0:  # <100% D/E (concerning)
                    score -= 0.05
                    print(f"- High Debt (D/E={debt_to_equity:.1%}): -0.05 → {score}")
                else:  # >100% D/E (very concerning)
                    score -= 0.15
                    print(f"- Very High Debt (D/E={debt_to_equity:.1%}): -0.15 → {score}")
        else:
            print(f"  Financial Metrics: not available")
        
        # NEWS SENTIMENT ANALYSIS (More Granular)
        if news and len(news) > 0:
            news_count = len(news)
            if news_count > 50:  # Very high news activity
                score += 0.1
                print(f"+ Very High News Activity ({news_count} articles): +0.1 → {score}")
            elif news_count > 20:  # High news activity
                score += 0.05
                print(f"+ High News Activity ({news_count} articles): +0.05 → {score}")
            elif news_count > 10:  # Moderate news activity
                print(f"  Moderate News Activity ({news_count} articles): neutral")
            elif news_count > 5:  # Low news activity
                score -= 0.05
                print(f"- Low News Activity ({news_count} articles): -0.05 → {score}")
            else:  # Very low news activity
                score -= 0.1
                print(f"- Very Low News Activity ({news_count} articles): -0.1 → {score}")
        else:
            print(f"  News Activity: not available")
        
        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, score))
        print(f"\nFinal confidence score: {final_score}")
        print(f"=== End Finnhub Calculation ===\n")
        
        return final_score


register(name="Finnhub", python_class="Finnhub")
