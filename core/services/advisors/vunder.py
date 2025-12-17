"""
Vunder Advisor - The search for undervalued stocks

Based on notional price calculation with comprehensive filtering:
- Fundamental filters (profit margin, revenue growth, market cap)
- Trend filters (5-day and 30-day)
- Average low filter (50-day SMA)
- Undervalued selection (discount ratio threshold)
"""
import logging
from decimal import Decimal
import pandas as pd

import yfinance as yf
from yfinance.screener import EquityQuery as YfEquityQuery

from core.services.advisors.advisor import AdvisorBase, register

logger = logging.getLogger(__name__)

# Discovery settings
UNDERVALUED_RATIO_THRESHOLD = 0.70  # Discount threshold (actual/notional <= 0.70)
MIN_PROFIT_MARGIN = 0.0
MAX_YEARLY_LOSS = -50.0
MIN_MARKET_CAP = 25_000_000
MAX_RECENT_TREND_PCT = 5.0  # Filter out stocks with >5% gain over last 5 days

# Quality filters (Phase 1)
MIN_REVENUE_GROWTH = -5.0  # Reject if revenue decline >5%
MAX_30_DAY_DECLINE = -15.0  # Reject if down >15% in last 30 days

# Average low filter
MAX_PRICE_VS_SMA50 = 1.10  # Allow price up to 10% above 50-day SMA (None = disabled)

# Volume strategy (can be configured)
DEFAULT_VOLUME_STRATEGY = "high"  # "high" or "low" volume


class Vunder(AdvisorBase):
    """
    Vunder Advisor - Discovers undervalued stocks using notional price calculation
    with comprehensive fundamental and technical filtering.
    """

    def __init__(self, advisor):
        if isinstance(advisor, str):
            super().__init__(advisor)
            self.advisor = None
        else:
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
            
            # Use sector average P/E
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
    # STOCK FETCHING
    # ============================================================================

    def _get_active_stocks(self, limit=200, min_volume=None, max_volume=None, sort_volume_asc=False):
        """
        Get stocks from Yahoo Finance with configurable volume filtering.
        
        Args:
            limit: Maximum number of stocks to return
            min_volume: Optional minimum daily volume
            max_volume: Optional maximum daily volume
            sort_volume_asc: If True, sort by volume ascending (lowest first, for less discovered stocks)
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

    # ============================================================================
    # TREND CALCULATION
    # ============================================================================

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

    # ============================================================================
    # FILTERING METHODS
    # ============================================================================

    def _filter_fundamentals(self, results):
        """Filter stocks by fundamental metrics."""
        filtered = []
        
        for stock in results:
            profit_margin = stock.get('profit_margin')
            yearly_change = stock.get('yearly_change_pct', 0) or 0
            market_cap = stock.get('market_cap', 0) or 0
            revenue_growth = stock.get('revenue_growth')
            
            # Check profit margin
            if profit_margin is None or profit_margin < MIN_PROFIT_MARGIN:
                continue
            
            # Check yearly loss
            if yearly_change < MAX_YEARLY_LOSS:
                continue
            
            # Check market cap
            if market_cap < MIN_MARKET_CAP:
                continue
            
            # Check revenue growth
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
            
            # If we can't calculate trend, include the stock
            if trend_pct is None:
                filtered.append(stock)
                continue
            
            # Filter out stocks with strong upward trends
            if trend_pct > MAX_RECENT_TREND_PCT:
                logger.debug(f"Filtered out {symbol} - recent trend too strong: {trend_pct:.2f}%")
                continue
            
            # Store trend for reference
            stock['recent_trend_pct'] = trend_pct
            filtered.append(stock)
        
        return filtered

    def _filter_longer_term_trend(self, results):
        """Filter out stocks with sustained declines over longer periods."""
        filtered = []
        
        for stock in results:
            symbol = stock.get('symbol')
            if not symbol:
                continue
            
            # Calculate trend over last 30 trading days
            trend_pct = self._calculate_recent_trend(symbol, days=30)
            
            # If we can't calculate trend, include the stock
            if trend_pct is None:
                filtered.append(stock)
                continue
            
            # Filter out stocks with sustained declines
            if trend_pct < MAX_30_DAY_DECLINE:
                logger.debug(f"Filtered out {symbol} - 30-day decline too severe: {trend_pct:.2f}%")
                continue
            
            # Store 30-day trend for reference
            stock['thirty_day_trend_pct'] = trend_pct
            filtered.append(stock)
        
        return filtered

    def _check_average_low(self, symbol, current_price):
        """Check if stock is trading near/below its 50-day SMA."""
        # If filter is disabled, always return True
        if MAX_PRICE_VS_SMA50 is None:
            return True
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='60d', interval='1d')
            
            if hist.empty or len(hist) < 50:
                return True
            
            # Calculate 50-day SMA
            hist['SMA50'] = hist['Close'].rolling(window=50).mean()
            sma50 = hist['SMA50'].iloc[-1]
            
            if pd.isna(sma50) or sma50 <= 0:
                return True
            
            # Check if current price is within threshold of SMA50
            price_ratio = current_price / sma50
            return price_ratio <= MAX_PRICE_VS_SMA50
            
        except Exception as e:
            logger.debug(f"Could not check average low for {symbol}: {e}")
            return True

    def _filter_average_low(self, results):
        """Filter stocks to only include those trading near/below their 50-day average."""
        # If filter is disabled, return all results unchanged
        if MAX_PRICE_VS_SMA50 is None:
            return results
        
        filtered = []
        
        for stock in results:
            symbol = stock.get('symbol')
            current_price = stock.get('actual_price', 0)
            
            if not symbol or not current_price:
                continue
            
            if self._check_average_low(symbol, current_price):
                filtered.append(stock)
            else:
                logger.debug(f"Filtered out {symbol} - trading too far above 50-day SMA")
        
        return filtered

    # ============================================================================
    # DISCOVERY METHOD
    # ============================================================================

    def discover(self, sa):
        """Discover undervalued stocks using notional price method with comprehensive filters."""
        try:
            # Get stocks (can be configured for high or low volume strategy)
            # For now, use high volume (can be made configurable later)
            stocks = self._get_active_stocks(limit=200, sort_volume_asc=False)
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
                            profit_margin = info.get('profitMargins')
                            yearly_change = (info.get('fiftyTwoWeekChangePercent', 0) or 0)
                            market_cap = info.get('marketCap', 0) or 0
                            revenue_growth = (info.get('revenueGrowth') * 100) if info.get('revenueGrowth') is not None else None
                            
                            results.append({
                                'symbol': stock['symbol'],
                                'name': stock['name'],
                                'actual_price': actual_price,
                                'notional_price': notional_price,
                                'discount_ratio': discount_ratio,
                                'method': method,
                                'profit_margin': profit_margin,
                                'yearly_change_pct': yearly_change,
                                'market_cap': market_cap,
                                'revenue_growth': revenue_growth,
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
            
            # Filter out stocks with sustained longer-term declines
            longer_term_filtered = self._filter_longer_term_trend(trend_filtered)
            
            if not longer_term_filtered:
                logger.info("No stocks passed longer-term trend filter")
                return
            
            # Filter for stocks trading near/below average
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

            # Pass sell instructions - Balanced approach for value discovery
            sell_instructions = [
                ("TARGET_PERCENTAGE", 1.25),       # 25% gain target
                ("STOP_PERCENTAGE", 0.95),         # 5% stop loss (wider tolerance)
                ("AFTER_DAYS", 45.0),              # Exit after 45 days if no progress
                ('DESCENDING_TREND', -0.25),       # Exit on significant downtrend (-25%)
            ]
            
            # Discover each undervalued stock
            discovered = 0
            for stock in undervalued:
                symbol = stock['symbol']
                current_price = stock['actual_price']

                # Check if already discovered - rediscover if >30 days ago OR price dropped to 80%
                if not self.allow_discovery(symbol, period=30 * 24, price_decline = 0.75):
                    continue

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
                self.discovered(sa, symbol, explanation, sell_instructions=sell_instructions)
                discovered += 1
            
            logger.info("Vunder discovery complete: %s stocks found", discovered)
            
        except Exception as e:
            logger.error(f"Error in Vunder discovery: {e}", exc_info=True)


register(name="Vunder", python_class="Vunder")

