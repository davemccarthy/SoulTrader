"""
Vunder Advisor - The search for undervalued stocks

Based on notional price calculation with comprehensive filtering:
- Fundamental filters (profit margin, revenue growth, market cap)
- Enhanced trend filters (5-day positive momentum, 30-day decline protection)
- Price structure filters (20-day SMA hybrid, volatility, dollar volume)
- Undervalued selection (discount ratio threshold)
"""
import logging
from decimal import Decimal
import pandas as pd

import yfinance as yf

from core.services.advisors.advisor import AdvisorBase, register

logger = logging.getLogger(__name__)

# Discovery settings
UNDERVALUED_RATIO_THRESHOLD = 0.75  # Discount threshold (actual/notional <= 0.75)
MIN_PROFIT_MARGIN = 0.0
MAX_YEARLY_LOSS = -50.0
MIN_MARKET_CAP = 25_000_000

# Quality filters (Phase 1)
MIN_REVENUE_GROWTH = -5.0  # Reject if revenue decline >5%
MAX_30_DAY_DECLINE = -10.0  # Reject if down >10% in last 30 days (stricter)

# Enhanced price structure filters (ChatGPT recommendations - relaxed)
MIN_PRICE_VS_SMA20 = 0.90  # Price must be >= 90% of 20-day SMA (relaxed from 95%)
MAX_PRICE_VS_SMA20 = 1.10  # Price must be <= 110% of 20-day SMA (relaxed from 105%)
MIN_5_DAY_TREND = -2.0     # Allow slightly negative 5-day trend (relaxed from 0.0)
MAX_5_DAY_TREND = 5.0      # But not > 5% (avoid extended moves)

# Volume filter (enhanced)
MIN_20_DAY_DOLLAR_VOLUME = 3_000_000  # $3M average dollar volume (20-day)

# Volatility filter (relaxed)
MAX_ATR_PCT = 0.07  # ATR(20) / Price <= 7% (relaxed from 5%)

# Daily return spike filter
MAX_DAILY_RETURN_PCT = 8.0  # |today's return| <= 8% (avoid distorted valuations)

# Old average low filter (50-day SMA) - kept for compatibility, disabled by default
MAX_PRICE_VS_SMA50 = None  # Set to None to disable, or value like 1.10 to enable


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

    def _get_active_stocks(self, limit=200, min_volume=None, max_volume=None, sort_volume_asc=False, sa=None):
        """
        Get stocks from Polygon using base advisor method with price and volume filtering.
        
        Args:
            limit: Maximum number of stocks to return
            min_volume: Optional minimum daily volume (not used, average volume is calculated)
            max_volume: Optional maximum daily volume (not used)
            sort_volume_asc: If True, sort by volume ascending (lowest first, for less discovered stocks)
            sa: SmartAnalysis session (optional, for logging)
        """
        try:
            # Use base advisor method to fetch stocks from Polygon
            # Filter by price range: $2 - $18
            df = self.get_filtered_stocks(
                sa=sa,
                min_price=2.0,
                max_price=18.0,
                min_volume=None  # We'll filter by average volume below
            )
            
            if df.empty:
                logger.warning("No stocks returned from Polygon")
                return []
            
            # Calculate average volume and filter to stocks with at least average volume
            avg_volume = df['today_volume'].mean()
            initial_count = len(df)
            df = df[df['today_volume'] >= avg_volume]
            
            if df.empty:
                logger.warning("No stocks with average or above average volume")
                return []
            
            logger.info(f"Filtered {initial_count} stocks (price $2-$18) to {len(df)} stocks with >= average volume ({avg_volume:,.0f})")
            
            # Convert DataFrame to list of dicts (matching original format)
            stocks = []
            for _, row in df.iterrows():
                # Get company name from yfinance (for display purposes)
                try:
                    ticker = yf.Ticker(row['ticker'])
                    info = ticker.info
                    name = info.get('shortName') or info.get('longName', 'N/A')
                except:
                    name = row['ticker']  # Fallback to ticker if name fetch fails
                
                stocks.append({
                    'symbol': row['ticker'],
                    'name': name,
                    'price': float(row['price']),
                    'volume': float(row['today_volume']),
                })
            
            # Sort by volume (ascending or descending based on param)
            stocks.sort(key=lambda x: x['volume'], reverse=not sort_volume_asc)
            return stocks[:limit]
            
        except Exception as e:
            logger.warning(f"Error fetching stocks from Polygon: {e}", exc_info=True)
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

    def _calculate_sma20(self, symbol):
        """Calculate 20-day Simple Moving Average."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='30d', interval='1d')
            
            if hist.empty or len(hist) < 20:
                return None
            
            sma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            return float(sma20) if not pd.isna(sma20) else None
        except Exception as e:
            logger.debug(f"Could not calculate SMA20 for {symbol}: {e}")
            return None

    def _calculate_atr20(self, symbol):
        """Calculate 20-day Average True Range."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='30d', interval='1d')
            
            if hist.empty or len(hist) < 20:
                return None
            
            # Calculate True Range
            hist['High-Low'] = hist['High'] - hist['Low']
            hist['High-PrevClose'] = abs(hist['High'] - hist['Close'].shift(1))
            hist['Low-PrevClose'] = abs(hist['Low'] - hist['Close'].shift(1))
            hist['TR'] = hist[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
            
            # Calculate 20-day ATR
            atr20 = hist['TR'].rolling(window=20).mean().iloc[-1]
            return float(atr20) if not pd.isna(atr20) else None
        except Exception as e:
            logger.debug(f"Could not calculate ATR20 for {symbol}: {e}")
            return None

    def _calculate_20day_dollar_volume(self, symbol):
        """Calculate 20-day average dollar volume."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='30d', interval='1d')
            
            if hist.empty or len(hist) < 20:
                return None
            
            # Calculate dollar volume (price * volume) for each day
            hist['DollarVolume'] = hist['Close'] * hist['Volume']
            
            # Calculate 20-day average
            avg_dollar_volume = hist['DollarVolume'].tail(20).mean()
            return float(avg_dollar_volume) if not pd.isna(avg_dollar_volume) else None
        except Exception as e:
            logger.debug(f"Could not calculate 20-day dollar volume for {symbol}: {e}")
            return None

    def _calculate_daily_return(self, symbol):
        """Calculate today's return percentage."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='2d', interval='1d')
            
            if hist.empty or len(hist) < 2:
                return None
            
            prev_close = hist['Close'].iloc[-2]
            current_close = hist['Close'].iloc[-1]
            
            daily_return = ((current_close - prev_close) / prev_close) * 100
            return float(daily_return)
        except Exception as e:
            logger.debug(f"Could not calculate daily return for {symbol}: {e}")
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
        """Filter for stocks with positive but not extended 5-day momentum (enhanced)."""
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
            
            # Require positive momentum (early turn confirmation) - relaxed to allow slightly negative
            if trend_pct < MIN_5_DAY_TREND:
                logger.debug(f"Filtered out {symbol} - negative 5-day trend: {trend_pct:.2f}%")
                continue
            
            # But not too extended
            if trend_pct > MAX_5_DAY_TREND:
                logger.debug(f"Filtered out {symbol} - trend too extended: {trend_pct:.2f}%")
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
            
            # If we can't calculate trend, include the stock (don't exclude due to data issues)
            if trend_pct is None:
                filtered.append(stock)
                continue
            
            # Filter out stocks with sustained declines (stricter: -10% instead of -15%)
            if trend_pct < MAX_30_DAY_DECLINE:
                logger.debug(f"Filtered out {symbol} - 30-day decline too severe: {trend_pct:.2f}%")
                continue
            
            # Store 30-day trend for reference
            stock['thirty_day_trend_pct'] = trend_pct
            filtered.append(stock)
        
        return filtered

    def _filter_price_vs_sma20(self, results):
        """Hybrid SMA filter: Price >= 90% of 20-day SMA AND <= 110% of 20-day SMA."""
        filtered = []
        ratios_below = []
        ratios_above = []
        ratios_valid = []
        no_sma_count = 0
        
        for stock in results:
            symbol = stock.get('symbol')
            current_price = stock.get('actual_price', 0)
            
            if not symbol or not current_price:
                continue
            
            sma20 = self._calculate_sma20(symbol)
            
            # If we can't calculate SMA20, include the stock (don't exclude due to data issues)
            if sma20 is None or sma20 <= 0:
                no_sma_count += 1
                filtered.append(stock)
                continue
            
            price_ratio = current_price / sma20
            
            # Hybrid: slightly below OK (>= 90%), but not far below, and not extended (<= 110%)
            if price_ratio < MIN_PRICE_VS_SMA20:
                ratios_below.append((symbol, price_ratio, current_price, sma20))
                logger.debug(f"Filtered out {symbol} - too far below 20-day SMA: price=${current_price:.2f}, SMA20=${sma20:.2f}, ratio={price_ratio:.2%}")
                continue
            
            if price_ratio > MAX_PRICE_VS_SMA20:
                ratios_above.append((symbol, price_ratio, current_price, sma20))
                logger.debug(f"Filtered out {symbol} - too far above 20-day SMA: price=${current_price:.2f}, SMA20=${sma20:.2f}, ratio={price_ratio:.2%}")
                continue
            
            ratios_valid.append((symbol, price_ratio))
            filtered.append(stock)
        
        # Log summary statistics
        logger.info(f"SMA20 filter: {len(results)} stocks -> {len(filtered)} passed")
        logger.info(f"  {no_sma_count} stocks included (no SMA20 data)")
        logger.info(f"  {len(ratios_below)} stocks filtered (too far below: < {MIN_PRICE_VS_SMA20:.0%})")
        logger.info(f"  {len(ratios_above)} stocks filtered (too far above: > {MAX_PRICE_VS_SMA20:.0%})")
        logger.info(f"  {len(ratios_valid)} stocks passed (ratio between {MIN_PRICE_VS_SMA20:.0%} and {MAX_PRICE_VS_SMA20:.0%})")
        
        # Log sample of ratios below threshold (first 5)
        if ratios_below:
            logger.info(f"Sample stocks below threshold (first 5):")
            for symbol, ratio, price, sma in ratios_below[:5]:
                logger.info(f"  {symbol}: price=${price:.2f}, SMA20=${sma:.2f}, ratio={ratio:.2%}")
        
        # Log sample of ratios above threshold (first 5)
        if ratios_above:
            logger.info(f"Sample stocks above threshold (first 5):")
            for symbol, ratio, price, sma in ratios_above[:5]:
                logger.info(f"  {symbol}: price=${price:.2f}, SMA20=${sma:.2f}, ratio={ratio:.2%}")
        
        return filtered

    def _filter_dollar_volume(self, results):
        """Filter by 20-day average dollar volume."""
        filtered = []
        
        for stock in results:
            symbol = stock.get('symbol')
            if not symbol:
                continue
            
            avg_dollar_volume = self._calculate_20day_dollar_volume(symbol)
            
            # If we can't calculate, include the stock (don't exclude due to data issues)
            if avg_dollar_volume is None:
                filtered.append(stock)
                continue
            
            if avg_dollar_volume < MIN_20_DAY_DOLLAR_VOLUME:
                logger.debug(f"Filtered out {symbol} - dollar volume too low: ${avg_dollar_volume:,.0f}")
                continue
            
            filtered.append(stock)
        
        return filtered

    def _filter_volatility(self, results):
        """Filter by ATR/Price ratio to avoid excessive volatility."""
        filtered = []
        
        for stock in results:
            symbol = stock.get('symbol')
            current_price = stock.get('actual_price', 0)
            
            if not symbol or not current_price or current_price <= 0:
                continue
            
            atr20 = self._calculate_atr20(symbol)
            
            # If we can't calculate ATR, include the stock (don't exclude due to data issues)
            if atr20 is None or atr20 <= 0:
                filtered.append(stock)
                continue
            
            atr_pct = atr20 / current_price
            
            if atr_pct > MAX_ATR_PCT:
                logger.debug(f"Filtered out {symbol} - volatility too high: {atr_pct:.2%}")
                continue
            
            filtered.append(stock)
        
        return filtered

    def _filter_daily_spikes(self, results):
        """Filter out stocks with extreme daily return spikes."""
        filtered = []
        
        for stock in results:
            symbol = stock.get('symbol')
            if not symbol:
                continue
            
            daily_return = self._calculate_daily_return(symbol)
            
            # If we can't calculate, include the stock (don't exclude due to data issues)
            if daily_return is None:
                filtered.append(stock)
                continue
            
            abs_return = abs(daily_return)
            
            if abs_return > MAX_DAILY_RETURN_PCT:
                logger.debug(f"Filtered out {symbol} - daily return spike: {daily_return:.2f}%")
                continue
            
            filtered.append(stock)
        
        return filtered

    def _check_average_low(self, symbol, current_price):
        """Check if stock is trading near/below its 50-day SMA (legacy filter, disabled by default)."""
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
        """Filter stocks to only include those trading near/below their 50-day average (legacy, disabled by default)."""
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
        """Discover undervalued stocks using notional price method with comprehensive filters.
        
        Only runs during the first hour of market open (9:30-10:30 AM ET).
        """
        # Check if within first hour of market open
        market_status = self.market_open()
        if market_status is None or market_status < 0 or market_status >= 60:
            logger.info(f"Vunder discovery skipped: outside first hour window (market_status={market_status})")
            return
        
        try:
            # Get stocks from Polygon with price filter ($2-$18) and average volume filter
            stocks = self._get_active_stocks(limit=200, sort_volume_asc=False, sa=sa)
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
            
            # STEP 1: Filter for undervalued (discount ratio <= threshold) - EARLY FILTER FOR PERFORMANCE
            undervalued_filtered = [
                r for r in results 
                if r['discount_ratio'] <= UNDERVALUED_RATIO_THRESHOLD
            ]
            
            if not undervalued_filtered:
                logger.info("No undervalued stocks found (ratio <= %.2f)", UNDERVALUED_RATIO_THRESHOLD)
                return
            
            logger.info(f"Early filter: {len(results)} stocks with notional prices -> {len(undervalued_filtered)} undervalued stocks")
            
            # STEP 2: Filter by fundamentals
            fundamental_filtered = self._filter_fundamentals(undervalued_filtered)
            
            if not fundamental_filtered:
                logger.info("No stocks passed fundamental filters")
                return
            
            # STEP 3: Filter for stocks with positive but not extended 5-day momentum (enhanced)
            trend_filtered = self._filter_recent_trend(fundamental_filtered)
            
            if not trend_filtered:
                logger.info("No stocks passed recent trend filter")
                return
            
            # STEP 4: Filter out stocks with sustained longer-term declines
            longer_term_filtered = self._filter_longer_term_trend(trend_filtered)
            
            if not longer_term_filtered:
                logger.info("No stocks passed longer-term trend filter")
                return
            
            # STEP 5: Filter for stocks trading near/below average (legacy, disabled by default)
            average_low_filtered = self._filter_average_low(longer_term_filtered)
            
            if not average_low_filtered:
                logger.info("No stocks passed average low filter")
                return
            
            # STEP 6: Filter by price vs 20-day SMA (hybrid)
            sma20_filtered = self._filter_price_vs_sma20(average_low_filtered)
            
            if not sma20_filtered:
                logger.info("No stocks passed 20-day SMA filter")
                return
            
            # STEP 7: Filter by dollar volume
            dollar_volume_filtered = self._filter_dollar_volume(sma20_filtered)
            
            if not dollar_volume_filtered:
                logger.info("No stocks passed dollar volume filter")
                return
            
            # STEP 8: Filter by volatility (ATR)
            volatility_filtered = self._filter_volatility(dollar_volume_filtered)
            
            if not volatility_filtered:
                logger.info("No stocks passed volatility filter")
                return
            
            # STEP 9: Filter by daily return spikes
            final_stocks = self._filter_daily_spikes(volatility_filtered)
            
            if not final_stocks:
                logger.info("No stocks passed daily spike filter")
                return

            # Pass sell instructions - Balanced approach for value discovery
            sell_instructions = [
                ("PERCENTAGE_DIMINISHING", 1.30, None),
                ("PERCENTAGE_AUGMENTING", 0.90, None),
            ]
            
            # Discover each stock that passed all filters
            discovered = 0
            for stock in final_stocks:
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
