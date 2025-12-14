import logging
from decimal import Decimal
import pandas as pd
import pandas_ta as ta

import yfinance as yf
from yfinance.screener import EquityQuery as YfEquityQuery

from core.services.advisors.advisor import AdvisorBase, register

logger = logging.getLogger(__name__)

MAX_PRICE = 5.0

# Notional price discovery settings
UNDERVALUED_RATIO_THRESHOLD = 0.70  # Changed from 0.66 to capture more opportunities (e.g., WDH at 31.2% discount)
MIN_PROFIT_MARGIN = 0.0
MAX_YEARLY_LOSS = -50.0
MIN_MARKET_CAP = 25_000_000  # Lowered from 100M to 25M to be less strict
MAX_RECENT_TREND_PCT = 5.0  # Filter out stocks with >5% gain over last 5 days (avoid stocks that already ran up)

# Additional quality filters (Phase 1)
MIN_REVENUE_GROWTH = -5.0  # Reject if revenue decline >5% (catches declining companies)
MAX_30_DAY_DECLINE = -15.0  # Reject if down >15% in last 30 days (catches sustained declines)

# Average low filter (TEMPORARY - experimental)
# Set to None to disable, or a value like 1.10 to allow price up to 10% above 50-day SMA
MAX_PRICE_VS_SMA50 = 1.10  # TEMPORARY: Filter stocks trading >10% above 50-day SMA (None = disabled)


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

    # ============================================================================
    # NOTIONAL PRICE CALCULATION METHODS
    # ============================================================================

    def _calculate_notional_price_dcf(self, ticker_info):
        """Calculate notional price using DCF method (simplified)."""
        try:
            fcf = ticker_info.get('freeCashflow') or ticker_info.get('operatingCashflow')
            shares = ticker_info.get('sharesOutstanding')
            
            if not fcf or not shares or fcf <= 0 or shares <= 0:
                return None
            
            # Simplified DCF: assume 5% growth, 10% discount rate, 10x terminal multiple
            growth_rate = 0.05
            discount_rate = 0.10
            terminal_multiple = 10.0
            years = 5
            
            # Project FCF for 5 years
            pv_fcf = 0
            for year in range(1, years + 1):
                future_fcf = fcf * ((1 + growth_rate) ** year)
                pv_fcf += future_fcf / ((1 + discount_rate) ** year)
            
            # Terminal value
            terminal_fcf = fcf * ((1 + growth_rate) ** years)
            terminal_value = terminal_fcf * terminal_multiple
            pv_terminal = terminal_value / ((1 + discount_rate) ** years)
            
            equity_value = pv_fcf + pv_terminal
            notional_price = equity_value / shares
            
            return notional_price
        except:
            return None

    def _calculate_notional_price_pe(self, ticker_info):
        """Calculate notional price using P/E multiple method."""
        try:
            eps = ticker_info.get('trailingEps') or ticker_info.get('forwardEps')
            shares = ticker_info.get('sharesOutstanding')
            
            if not eps or not shares or eps <= 0 or shares <= 0:
                return None
            
            # Use sector average P/E, not company's own P/E
            sector_pe_map = {
                'Technology': 28.0,
                'Healthcare': 22.0,
                'Financial Services': 13.0,
                'Consumer Cyclical': 19.0,
                'Consumer Defensive': 21.0,
                'Energy': 11.0,
                'Industrials': 19.0,
                'Basic Materials': 16.0,
                'Real Estate': 18.0,
                'Utilities': 16.0,
                'Communication Services': 16.0,
            }
            
            sector = ticker_info.get('sector', '')
            pe_ratio = sector_pe_map.get(sector, 18.0)  # Default to market average
            
            notional_price = eps * pe_ratio
            return notional_price
        except:
            return None

    def _calculate_notional_price_ev_ebitda(self, ticker_info):
        """Calculate notional price using EV/EBITDA multiple method."""
        try:
            ebitda = ticker_info.get('ebitda')
            shares = ticker_info.get('sharesOutstanding')
            market_cap = ticker_info.get('marketCap')
            total_debt = ticker_info.get('totalDebt') or 0
            cash = ticker_info.get('totalCash') or 0
            
            if not ebitda or not shares or ebitda <= 0 or shares <= 0:
                return None
            
            # Use company's own EV/EBITDA if available, otherwise use sector average
            ev_ebitda = ticker_info.get('enterpriseToEbitda') or 10.0
            
            # Calculate Enterprise Value
            enterprise_value = ebitda * ev_ebitda
            
            # Convert to Equity Value
            net_debt = total_debt - cash
            equity_value = enterprise_value - net_debt
            
            # If we have market cap, use it to validate/adjust
            if market_cap and market_cap > 0:
                # Use a blend: 70% calculated, 30% market-based
                equity_value = equity_value * 0.7 + market_cap * 0.3
            
            notional_price = equity_value / shares
            return notional_price
        except:
            return None

    def _calculate_notional_price_revenue(self, ticker_info):
        """Calculate notional price using revenue multiple method."""
        try:
            revenue_per_share = ticker_info.get('revenuePerShare')
            shares = ticker_info.get('sharesOutstanding')
            total_revenue = ticker_info.get('totalRevenue')
            
            if not revenue_per_share and total_revenue and shares and shares > 0:
                revenue_per_share = total_revenue / shares
            
            if not revenue_per_share or revenue_per_share <= 0:
                return None
            
            # Use price-to-sales ratio (default 2.0 for growth companies)
            ps_ratio = ticker_info.get('priceToSalesTrailing12Months')
            if not ps_ratio or ps_ratio < 0.5 or ps_ratio > 20:
                ps_ratio = 2.0
            
            notional_price = revenue_per_share * ps_ratio
            
            # Sanity check
            if notional_price > 10000:
                return None
            
            return notional_price
        except:
            return None

    def _calculate_notional_price_book(self, ticker_info):
        """Calculate notional price using price-to-book method."""
        try:
            book_value = ticker_info.get('bookValue')
            shares = ticker_info.get('sharesOutstanding')
            
            if not book_value or not shares or book_value <= 0 or shares <= 0:
                return None
            
            book_per_share = book_value / shares
            
            # Use company's P/B if available, otherwise use sector average (default 1.5)
            pb_ratio = ticker_info.get('priceToBook') or 1.5
            
            notional_price = book_per_share * pb_ratio
            return notional_price
        except:
            return None

    def _calculate_best_notional_price(self, ticker_info):
        """Calculate notional price using the best available method."""
        methods = []
        actual_price = ticker_info.get('currentPrice') or ticker_info.get('regularMarketPrice') or 0
        
        # Try EV/EBITDA first (most reliable for most companies)
        ev_ebitda_price = self._calculate_notional_price_ev_ebitda(ticker_info)
        if ev_ebitda_price and ev_ebitda_price > 0 and ev_ebitda_price < actual_price * 10:
            methods.append(('EV/EBITDA', ev_ebitda_price))
        
        # Try P/E (good for profitable companies)
        pe_price = self._calculate_notional_price_pe(ticker_info)
        if pe_price and pe_price > 0 and pe_price < actual_price * 10:
            methods.append(('P/E', pe_price))
        
        # Try DCF (most rigorous but requires assumptions)
        dcf_price = self._calculate_notional_price_dcf(ticker_info)
        if dcf_price and dcf_price > 0 and dcf_price < actual_price * 10:
            methods.append(('DCF', dcf_price))
        
        # Try P/B (for financial companies)
        pb_price = self._calculate_notional_price_book(ticker_info)
        if pb_price and pb_price > 0 and pb_price < actual_price * 10:
            methods.append(('P/B', pb_price))
        
        # Try Revenue multiple last (for growth companies, less reliable)
        revenue_price = self._calculate_notional_price_revenue(ticker_info)
        if revenue_price and revenue_price > 0 and revenue_price < actual_price * 10:
            methods.append(('Revenue', revenue_price))
        
        if not methods:
            return None, None
        
        # Use the method with the most reasonable price
        if actual_price and actual_price > 0:
            valid_methods = [m for m in methods if m[1] > actual_price * 0.5 and m[1] < actual_price * 5]
            if valid_methods:
                best_method = min(valid_methods, key=lambda x: abs(x[1] - actual_price))
            else:
                best_method = min(methods, key=lambda x: abs(x[1] - actual_price))
        else:
            best_method = methods[0]
        
        return best_method[0], best_method[1]

    # ============================================================================
    # DISCOVERY HELPER METHODS
    # ============================================================================

    def _get_active_stocks(self, limit=200, min_volume=None, max_volume=None, sort_volume_asc=False):
        """
        Get stocks from Yahoo Finance with configurable volume filtering.
        
        Args:
            limit: Maximum number of stocks to return
            min_volume: Optional minimum daily volume (filters out illiquid stocks)
            max_volume: Optional maximum daily volume (filters out over-traded stocks)
            sort_volume_asc: If True, sort by volume ascending (lowest first, for less discovered stocks).
                           If False, sort descending (highest first, current default behavior)
        """
        try:
            most_active_query = YfEquityQuery(
                "and",
                [
                    YfEquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
                    YfEquityQuery("gt", ["intradayprice", 1.0]),
                ],
            )
            
            max_size = min(limit * 2, 250)
            response = yf.screen(
                most_active_query,
                offset=0,
                size=max_size,
                sortField="intradayprice",
                sortAsc=True,
            )
            
            quotes = response.get("quotes", [])
            stocks = []
            
            for quote in quotes:
                symbol = quote.get('symbol')
                name = quote.get('shortName') or quote.get('longName', 'N/A')
                price = quote.get('regularMarketPrice') or quote.get('intradayprice')
                volume = quote.get('volume') or quote.get('regularMarketVolume') or 0
                
                if symbol and price:
                    stocks.append({
                        'symbol': symbol,
                        'name': name,
                        'price': float(price),
                        'volume': float(volume) if volume else 0.0,
                    })
            
            # Filter by volume range if specified
            if min_volume is not None or max_volume is not None:
                stocks = [
                    s for s in stocks 
                    if (min_volume is None or s['volume'] >= min_volume) and
                       (max_volume is None or s['volume'] <= max_volume)
                ]
            
            # Sort by volume (ascending or descending based on param)
            stocks.sort(key=lambda x: x['volume'], reverse=not sort_volume_asc)
            return stocks[:limit]
            
        except Exception as e:
            logger.warning(f"Error fetching active stocks: {e}")
            return []

    def _calculate_recent_trend(self, symbol, days=5):
        """Calculate price trend over the last N trading days."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d", interval="1d")
            
            if hist.empty or len(hist) < 2:
                return None
            
            # Get first and last close prices
            first_close = hist['Close'].iloc[0]
            last_close = hist['Close'].iloc[-1]
            
            # Calculate percentage change
            trend_pct = ((last_close - first_close) / first_close) * 100
            return trend_pct
        except Exception as e:
            logger.debug(f"Could not calculate recent trend for {symbol}: {e}")
            return None

    def _filter_fundamentals(self, results):
        """Filter stocks by fundamental metrics."""
        filtered = []
        
        for stock in results:
            profit_margin = stock.get('profit_margin')
            yearly_change = stock.get('yearly_change_pct', 0) or 0
            market_cap = stock.get('market_cap', 0) or 0
            revenue_growth = stock.get('revenue_growth')
            
            # Check profit margin - None means missing data (unprofitable companies often have None)
            if profit_margin is None or profit_margin < MIN_PROFIT_MARGIN:
                continue
            
            # Check yearly loss
            if yearly_change < MAX_YEARLY_LOSS:
                continue
            
            # Check market cap
            if market_cap < MIN_MARKET_CAP:
                continue
            
            # Check revenue growth - reject if revenue declining >5% (Phase 1 filter)
            if revenue_growth is not None and revenue_growth < MIN_REVENUE_GROWTH:
                logger.debug(f"Filtered out {stock.get('symbol', 'unknown')} - revenue declining: {revenue_growth:.2f}%")
                continue
            
            filtered.append(stock)
        
        return filtered

    def _filter_recent_trend(self, results):
        """Filter out stocks with strong upward trends over recent days."""
        filtered = []
        
        for stock in results:
            symbol = stock.get('symbol')
            if not symbol:
                continue
            
            # Calculate trend over last 5 trading days
            trend_pct = self._calculate_recent_trend(symbol, days=5)
            
            # If we can't calculate trend, include the stock (don't exclude due to data issues)
            if trend_pct is None:
                filtered.append(stock)
                continue
            
            # Filter out stocks with strong upward trends (> MAX_RECENT_TREND_PCT)
            # Keep stocks with flat (0-2%) or slightly positive (2-5%) trends
            if trend_pct > MAX_RECENT_TREND_PCT:
                logger.debug(f"Filtered out {symbol} - recent trend too strong: {trend_pct:.2f}%")
                continue
            
            # Store trend for reference
            stock['recent_trend_pct'] = trend_pct
            filtered.append(stock)
        
        return filtered

    def _filter_longer_term_trend(self, results):
        """Filter out stocks with sustained declines over longer periods (Phase 1 filter)."""
        filtered = []
        
        for stock in results:
            symbol = stock.get('symbol')
            if not symbol:
                continue
            
            # Calculate trend over last 30 trading days
            trend_pct = self._calculate_recent_trend(symbol, days=30)
            
            # If we can't calculate trend, include the stock (don't exclude due to data issues)
            if trend_pct is None:
                filtered.append(stock)
                continue
            
            # Filter out stocks with sustained declines (> MAX_30_DAY_DECLINE)
            if trend_pct < MAX_30_DAY_DECLINE:
                logger.debug(f"Filtered out {symbol} - 30-day decline too severe: {trend_pct:.2f}%")
                continue
            
            # Store 30-day trend for reference
            stock['thirty_day_trend_pct'] = trend_pct
            filtered.append(stock)
        
        return filtered

    # ============================================================================
    # DISCOVERY METHODS
    # ============================================================================

    def discover(self, sa):
        """Discover undervalued stocks using notional price method with fundamental filters."""
        try:
            # Get most active stocks
            stocks = self._get_active_stocks(limit=200)
            if not stocks:
                logger.warning("No active stocks retrieved")
                return
            
            # Calculate notional prices
            results = []
            batch_size = 20
            
            for i in range(0, len(stocks), batch_size):
                batch = stocks[i:i+batch_size]
                
                for stock in batch:
                    try:
                        ticker = yf.Ticker(stock['symbol'])
                        info = ticker.info
                        
                        method, notional_price = self._calculate_best_notional_price(info)
                        
                        if notional_price and notional_price > 0:
                            actual_price = stock['price']
                            discount_ratio = actual_price / notional_price
                            
                            # Get fundamental metrics
                            # Store None if profitMargins is missing (unprofitable companies often have None)
                            profit_margin = info.get('profitMargins')
                            # fiftyTwoWeekChangePercent is already a percentage (e.g., -29.47 = -29.47%), don't multiply by 100
                            yearly_change = (info.get('fiftyTwoWeekChangePercent', 0) or 0)
                            market_cap = info.get('marketCap', 0) or 0
                            # Revenue growth as percentage (e.g., -0.069 = -6.9%), multiply by 100 to get percentage
                            revenue_growth = (info.get('revenueGrowth') * 100) if info.get('revenueGrowth') is not None else None
                            
                            results.append({
                                'symbol': stock['symbol'],
                                'name': stock['name'],
                                'actual_price': actual_price,
                                'notional_price': notional_price,
                                'discount_ratio': discount_ratio,
                                'method': method,
                                'profit_margin': profit_margin,  # Can be None for unprofitable companies
                                'yearly_change_pct': yearly_change,
                                'market_cap': market_cap,
                                'revenue_growth': revenue_growth,  # Phase 1: revenue growth filter
                            })
                    except Exception as e:
                        logger.debug(f"Error processing {stock.get('symbol', 'unknown')}: {e}")
                        continue
            
            if not results:
                logger.info("No stocks with notional prices calculated")
                return
            
            # Filter by fundamentals
            fundamental_filtered = self._filter_fundamentals(results)
            
            if not fundamental_filtered:
                logger.info("No stocks passed fundamental filters")
                return
            
            # Filter out stocks with strong recent upward trends
            trend_filtered = self._filter_recent_trend(fundamental_filtered)
            
            if not trend_filtered:
                logger.info("No stocks passed recent trend filter")
                return
            
            # Filter out stocks with sustained longer-term declines (Phase 1 filter)
            longer_term_filtered = self._filter_longer_term_trend(trend_filtered)
            
            if not longer_term_filtered:
                logger.info("No stocks passed longer-term trend filter")
                return
            
            # Filter for stocks trading near/below average (TEMPORARY - experimental)
            average_low_filtered = self._filter_average_low(longer_term_filtered)
            
            if not average_low_filtered:
                logger.info("No stocks passed average low filter")
                return
            
            # Filter for undervalued (actual/notional <= threshold)
            undervalued = [
                r for r in average_low_filtered 
                if r['discount_ratio'] <= UNDERVALUED_RATIO_THRESHOLD
            ]
            
            if not undervalued:
                logger.info("No undervalued stocks found (ratio <= %.2f)", UNDERVALUED_RATIO_THRESHOLD)
                return

            # Pass sell instructions 
            sell_instructions = [
                ("TARGET_PERCENTAGE", 1.20),
                ("STOP_PERCENTAGE", 0.98),
                ('DESCENDING_TREND', -0.20),
                ('CS FLOOR', 0.00)
            ]
            
            # Discover each undervalued stock
            discovered = 0
            for stock in undervalued:
                discount_pct = (1 - stock['discount_ratio']) * 100
                upside = stock['notional_price'] - stock['actual_price']
                upside_pct = (upside / stock['actual_price']) * 100 if stock['actual_price'] > 0 else 0
                
                recent_trend = stock.get('recent_trend_pct')
                trend_str = f", Recent trend: {recent_trend:+.1f}%" if recent_trend is not None else ""
                
                explanation_parts = [
                    f"Actual: ${stock['actual_price']:.2f}",
                    f"Notional: ${stock['notional_price']:.2f}",
                    f"Discount: {discount_pct:.1f}%",
                    f"Upside: {upside_pct:.1f}%",
                    f"Method: {stock['method']}{trend_str}",
                ]
                
                explanation = " | ".join(explanation_parts)
                self.discovered(sa, stock['symbol'], explanation)
                discovered += 1
            
            logger.info("Yahoo notional price discovery complete: %s stocks found", discovered)
            
        except Exception as e:
            logger.error(f"Error in notional price discovery: {e}", exc_info=True)

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
            self.discovered(sa, symbol, explanation)
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
        """Analyze stock using Yahoo Finance data + technical indicators"""
        try:
            # Get real-time data
            ticker = yf.Ticker(stock.symbol)
            info = ticker.info

            # --- TECHNICAL ANALYSIS (from Polygon) ---
            technical_score = None
            technical_explanation = []
            
            try:
                # Fetch 2 years of daily historical data
                hist = ticker.history(period="2y", interval="1d")
                
                if not hist.empty and len(hist) >= 50:  # Need at least 50 days for SMA50
                    # Calculate technical indicators (same as Polygon)
                    hist['SMA50'] = ta.sma(hist['Close'], length=50)
                    hist['SMA200'] = ta.sma(hist['Close'], length=200) if len(hist) >= 200 else None
                    hist['RSI'] = ta.rsi(hist['Close'], length=14)
                    macd = ta.macd(hist['Close'])
                    hist = pd.concat([hist, macd], axis=1)
                    
                    latest = hist.iloc[-1]
                    current_price = float(latest['Close'])
                    
                    # Calculate technical score (same logic as Polygon)
                    tech_score = 0.5  # neutral base
                    
                    sma50 = latest.get('SMA50')
                    sma200 = latest.get('SMA200')
                    rsi = latest.get('RSI')
                    
                    # SMA trend analysis
                    if sma50 is not None and sma200 is not None:
                        if current_price > sma50 > sma200:
                            tech_score += 0.2
                        elif current_price < sma50 < sma200:
                            tech_score -= 0.2
                        elif current_price > sma50:
                            tech_score += 0.05
                        else:
                            tech_score -= 0.05
                    elif sma50 is not None:
                        if current_price > sma50:
                            tech_score += 0.05
                        else:
                            tech_score -= 0.05
                    
                    # RSI contribution
                    if rsi is not None and not pd.isna(rsi):
                        rsi_norm = max(0, min(1, (50 - abs(rsi - 50)) / 50))
                        tech_score += 0.1 * rsi_norm
                    
                    # MACD contribution
                    macd_hist = latest.get('MACDh_12_26_9', 0)
                    if macd_hist is not None and not pd.isna(macd_hist):
                        tech_score += 0.1 if macd_hist > 0 else -0.1
                    
                    technical_score = max(0.0, min(1.0, tech_score))
                    
                    # Build technical explanation
                    sma50_str = f"{sma50:.2f}" if sma50 is not None and not pd.isna(sma50) else "N/A"
                    sma200_str = f"{sma200:.2f}" if sma200 is not None and not pd.isna(sma200) else "N/A"
                    rsi_str = f"{rsi:.2f}" if rsi is not None and not pd.isna(rsi) else "N/A"
                    macd_hist_str = f"{macd_hist:.2f}" if macd_hist is not None and not pd.isna(macd_hist) else "N/A"
                    technical_explanation.append(f"SMA50: {sma50_str}, SMA200: {sma200_str}, RSI: {rsi_str}, MACD: {macd_hist_str}")
                    
            except Exception as e:
                logger.debug(f"Yahoo technical analysis failed for {stock.symbol}: {e}")
                # Continue with fundamental-only analysis
            
            # --- FUNDAMENTAL ANALYSIS (EXISTING) ---
            fundamental_score = self._calculate_confidence(info, stock.symbol)
            
            # --- COMBINE SCORES ---
            if technical_score is not None:
                # Weighted combination: 60% technical, 40% fundamental
                confidence = (technical_score * 0.6) + (fundamental_score * 0.4)
            else:
                # Fallback to fundamental-only if technical fails
                confidence = fundamental_score
            
            # Build detailed analysis explanation
            explanation_parts = []
            
            # Add technical indicators if available
            if technical_explanation:
                explanation_parts.extend(technical_explanation)
            
            # Add key factors that influenced the score
            pe_ratio = self._safe_float(info.get('trailingPE', 0)) or 0
            if pe_ratio > 0:
                if pe_ratio < 1.0:
                    explanation_parts.append(f"⚠️ DISTRESSED: P/E={pe_ratio:.2f}")
                elif pe_ratio > 30:
                    explanation_parts.append(f"🔴 OVERVALUED: P/E={pe_ratio:.2f}")
                elif pe_ratio < 15:
                    explanation_parts.append(f"🟢 UNDERVALUED: P/E={pe_ratio:.2f}")
            
            profit_margin = self._safe_float(info.get('profitMargins', 0)) or 0
            if profit_margin > 0.20:
                explanation_parts.append(f"💪 STRONG PROFITS: {profit_margin:.1%} margin")
            elif profit_margin < 0:
                explanation_parts.append(f"💸 LOSING MONEY: {profit_margin:.1%} margin")
            elif profit_margin < 0.05:
                explanation_parts.append(f"📉 WEAK PROFITS: {profit_margin:.1%} margin")
            
            change_52w = self._safe_float(info.get('fiftyTwoWeekChangePercent', 0)) or 0
            if change_52w < -0.50:
                explanation_parts.append(f"📉 CATASTROPHIC: {change_52w:.1%} yearly loss")
            elif change_52w < -0.20:
                explanation_parts.append(f"📉 MAJOR DECLINE: {change_52w:.1%} yearly loss")
            elif change_52w > 0.20:
                explanation_parts.append(f"🚀 STRONG MOMENTUM: {change_52w:.1%} yearly gain")
            elif change_52w > 0.10:
                explanation_parts.append(f"📈 POSITIVE MOMENTUM: {change_52w:.1%} yearly gain")
            
            market_cap = self._safe_float(info.get('marketCap', 0)) or 0
            if market_cap < 100_000_000:
                explanation_parts.append(f"⚠️ MICRO-CAP RISK: ${market_cap:,.0f}")
            elif market_cap > 1_000_000_000_000:
                explanation_parts.append(f"🏢 MEGA-CAP STABILITY: ${market_cap:,.0f}")
            
            
            debt_to_equity = self._safe_float(info.get('debtToEquity', 0)) or 0
            if debt_to_equity > 2.0:
                explanation_parts.append(f"⚠️ HIGH DEBT: {debt_to_equity:.1f} D/E")
            elif debt_to_equity < 0.5:
                explanation_parts.append(f"💪 LOW DEBT: {debt_to_equity:.1f} D/E")
            
            # Add target price information if available
            target_mean_price = self._safe_float(info.get('targetMeanPrice'))
            if target_mean_price:
                current_price = self._safe_float(info.get('currentPrice') or info.get('regularMarketPrice'))
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
        pe_ratio = self._safe_float(info.get('trailingPE', 0)) or 0
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
        profit_margin = self._safe_float(info.get('profitMargins', 0)) or 0
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
        change_52w = self._safe_float(info.get('fiftyTwoWeekChangePercent', 0)) or 0
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
        market_cap = self._safe_float(info.get('marketCap', 0)) or 0
        if market_cap > 1_000_000_000_000:  # >$1T
            score += 0.1
        elif market_cap > 100_000_000_000:  # >$100B
            score += 0.05
        elif market_cap < 100_000_000:  # <$100M = micro-cap risk
            score -= 0.1
            negative_flag = True

        # ANALYST CONSENSUS
        analyst_rating = self._safe_float(info.get('recommendationMean', 3.0)) or 3.0
        if analyst_rating < 2.0:
            score += 0.15
        elif analyst_rating < 2.5:
            score += 0.1
        elif analyst_rating > 3.5:
            score -= 0.1
            negative_flag = True

        # DEBT ANALYSIS
        debt_to_equity = self._safe_float(info.get('debtToEquity', 0)) or 0
        if debt_to_equity > 2.0:
            score -= 0.2
            negative_flag = True
        elif debt_to_equity > 1.0:
            score -= 0.1
            negative_flag = True
        elif debt_to_equity < 0.5:
            score += 0.1

        # VOLUME ANALYSIS
        volume = self._safe_float(info.get('volume', 0)) or 0
        avg_volume = self._safe_float(info.get('averageVolume', 0)) or 0
        if avg_volume > 0:
            volume_ratio = volume / avg_volume
            if volume > avg_volume * 1.5:
                score += 0.05
            elif volume < avg_volume * 0.5:
                score -= 0.05

        # TARGET PRICE ANALYSIS
        target_mean_price = self._safe_float(info.get('targetMeanPrice'))
        if target_mean_price:
            current_price = self._safe_float(info.get('currentPrice') or info.get('regularMarketPrice'))
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