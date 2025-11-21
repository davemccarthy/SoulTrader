import logging
from decimal import Decimal

import yfinance as yf
from yfinance.screener import EquityQuery as YfEquityQuery

from core.services.advisors.advisor import AdvisorBase, register

logger = logging.getLogger(__name__)

MAX_PRICE = 5.0


CUSTOM_SCREEN_QUERY = YfEquityQuery(
    "and",
    [
        YfEquityQuery("lt", ["pegratio_5y", 1]),  # PEG below 1
        YfEquityQuery(
            "or",
            [
                YfEquityQuery("btwn", ["epsgrowth.lasttwelvemonths", 25, 50]),
                YfEquityQuery("gt", ["epsgrowth.lasttwelvemonths", 100]),
            ],
        ),
        YfEquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
        YfEquityQuery("lt", ["intradayprice", MAX_PRICE]),  # Price filter: prefer lower-priced stocks
    ],
)

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

    def discover_yfinance(self, sa):
        """Discover stocks using Yahoo Finance screener filters."""
        try:
            quotes = self._fetch_filtered_quotes()
        except Exception as exc:  # pragma: no cover - network failure fallback
            logger.warning("Yahoo custom screener failed (%s)", exc)
            return

        discovered = 0
        for quote in quotes:
            symbol = quote.get("symbol")
            if not symbol:
                continue

            last_price = self._safe_float(
                quote.get("regularMarketPrice") or quote.get("intradayprice")
            )
            if last_price is None:
                # Skip if no price data available
                continue

            company = quote.get("longName") or quote.get("shortName") or symbol
            peg_ratio = self._safe_float(quote.get("pegRatio"))
            eps_growth = self._safe_float(quote.get("epsGrowthQuarterly"))

            explanation_parts = ["Undervalued growth screener match"]
            if peg_ratio is not None:
                explanation_parts.append(f"PEG {peg_ratio:.2f}")
            if eps_growth is not None:
                explanation_parts.append(f"EPS growth {eps_growth:.0%}")
            explanation_parts.append(f"Price ${last_price:.2f}")

            explanation = " | ".join(explanation_parts)
            self.discovered(sa, symbol, company, explanation)
            discovered += 1

        logger.info("Yahoo Finance discovery complete: %s stocks found", discovered)


    def _fetch_filtered_quotes(self, limit: int = 10):
        """Run custom Yahoo screener query using yfinance."""
        response = yf.screen(
            CUSTOM_SCREEN_QUERY,
            offset=0,
            size=limit * 2,  # Fetch more to ensure we get enough after sorting
            sortField="pegratio_5y",
            sortAsc=True,
        )
        quotes = response.get("quotes", [])
        
        # Sort by PEG ratio first, then by price (lower is better)
        # This ensures we get the most undervalued stocks, with preference for lower-priced ones
        def get_sort_key(quote):
            peg_value = quote.get("pegRatio") or quote.get("pegratio_5y")
            peg = self._safe_float(peg_value)
            # Put None/missing PEG at the end
            peg_sort = peg if peg is not None and peg > 0 else float('inf')
            
            # Get price for secondary sort (lower price = better)
            price = self._safe_float(
                quote.get("regularMarketPrice") or quote.get("intradayprice")
            )
            price_sort = price if price is not None and price > 0 else float('inf')
            
            # Return tuple: (PEG, price) - both sorted ascending (lower is better)
            return (peg_sort, price_sort)
        
        quotes.sort(key=get_sort_key)
        
        # Return top N most undervalued (lowest PEG, then lowest price)
        return quotes[:limit]

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def analyze(self, sa, stock):
        """Analyze stock using Yahoo Finance data"""
        try:
            # Get real-time data
            ticker = yf.Ticker(stock.symbol)
            info = ticker.info

            # Calculate confidence based on real metrics
            confidence = self._calculate_confidence(info, stock.symbol)
            
            # Build detailed analysis explanation
            explanation_parts = []
            #explanation_parts.append(f"Confidence Score: {confidence:.2f}")
            
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
        negative_flag = False

        # VALUATION ANALYSIS
        pe_ratio = info.get('trailingPE', 0)
        if pe_ratio > 0:
            if pe_ratio < 1.0:  # Very low P/E = distressed company
                score -= 0.3
                negative_flag = True
            elif pe_ratio < 15:
                score += 0.2
            elif pe_ratio > 30:
                score -= 0.3
                negative_flag = True
            elif pe_ratio > 25:
                score -= 0.1
                negative_flag = True

        # PROFITABILITY ANALYSIS
        profit_margin = info.get('profitMargins', 0)
        if profit_margin < 0:  # Losing money
            score -= 0.2
            negative_flag = True
        elif profit_margin > 0.20:  # >20%
            score += 0.2
        elif profit_margin > 0.10:  # >10%
            score += 0.1
        elif profit_margin < 0.05:  # <5%
            score -= 0.1
            negative_flag = True

        # MOMENTUM ANALYSIS
        change_52w = info.get('fiftyTwoWeekChangePercent', 0)
        if change_52w < -0.50:  # >50% loss = catastrophic
            score -= 0.3
            negative_flag = True
        elif change_52w < -0.20:  # >20% loss = major decline
            score -= 0.2
            negative_flag = True
        elif change_52w < -0.10:  # >10% loss
            score -= 0.1
            negative_flag = True
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
            negative_flag = True

        # ANALYST CONSENSUS
        analyst_rating = info.get('recommendationMean', 3.0)
        if analyst_rating < 2.0:
            score += 0.15
        elif analyst_rating < 2.5:
            score += 0.1
        elif analyst_rating > 3.5:
            score -= 0.1
            negative_flag = True

        # DEBT ANALYSIS
        debt_to_equity = info.get('debtToEquity', 0)
        if debt_to_equity > 2.0:
            score -= 0.2
            negative_flag = True
        elif debt_to_equity > 1.0:
            score -= 0.1
            negative_flag = True
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
                # Conservative scaling: 35% weight and clamp extreme moves
                target_score_adjustment = (upside_percent / 100) * 0.35
                target_score_adjustment = max(-0.2, min(0.25, target_score_adjustment))
                if target_score_adjustment < 0:
                    negative_flag = True
                score += target_score_adjustment

        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, score))
        if negative_flag:
            final_score = min(final_score, 0.95)

        return final_score


register(name="Yahoo Finances", python_class="Yahoo")