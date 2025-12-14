import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime, date
from decimal import Decimal

from core.services.advisors.advisor import AdvisorBase, register
from core.models import SmartAnalysis

logger = logging.getLogger(__name__)

BASE_URL = "http://openinsider.com"

# Discovery threshold - only discover stocks with score above this
DISCOVERY_THRESHOLD = 0.3

# Scoring normalization constants
MAX_QTY = 1000000  # 1M shares
MAX_VALUE = 5000000  # $5M


def parse_date(date_str: str) -> Optional[date]:
    """Parse date string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS) to date object."""
    if not date_str:
        return None
    # Try exact format first (YYYY-MM-DD)
    if len(date_str) == 10:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except:
            pass
    # Try with timestamp (YYYY-MM-DD HH:MM:SS)
    if len(date_str) >= 10:
        try:
            # Extract just the date part (first 10 characters)
            date_part = date_str[:10]
            return datetime.strptime(date_part, "%Y-%m-%d").date()
        except:
            pass
    return None


def days_ago(date_obj: Optional[date]) -> Optional[int]:
    """Calculate days ago from today."""
    if not date_obj:
        return None
    return (date.today() - date_obj).days


def recency_score(days_ago_val: Optional[int], max_days: int = 7) -> float:
    """
    Calculate recency score (0-1) based on days ago.
    More recent = higher score.
    
    Args:
        days_ago_val: Number of days ago (None = unknown = 0.5)
        max_days: Maximum days to consider (beyond this = 0.0)
    
    Returns:
        Score from 0.0 (old) to 1.0 (very recent)
    """
    if days_ago_val is None:
        return 0.5  # Unknown = neutral
    
    if days_ago_val < 0:
        return 1.0  # Future date = max score
    
    if days_ago_val > max_days:
        return 0.0  # Too old
    
    # Linear decay: 0 days = 1.0, max_days = 0.0
    return 1.0 - (days_ago_val / max_days)


def score_insider_purchase(purchase: Dict, max_qty: float = MAX_QTY, max_value: float = MAX_VALUE, cluster_size: int = 1) -> float:
    """
    Score an insider purchase (0-1) based on multiple factors.
    
    Factors:
    - Quantity: Larger purchases = higher conviction (normalized to max_qty)
    - Value: Larger dollar amounts = higher conviction (normalized to max_value)
    - Recency: More recent = better (prefer trade_date over filing_date, 7-day window)
    - Price: Lower price = better (penny stocks tend to outperform)
    - Title Bonus: +0.1 for CEO, +0.05 for CFO
    - Cluster Bonus: +0.05 per additional insider buying same stock (max +0.15)
    
    Args:
        purchase: Purchase dictionary with price, quantity, value, trade_date, filing_date, title
        max_qty: Maximum quantity for normalization (default 1M shares)
        max_value: Maximum value for normalization (default $5M)
        cluster_size: Number of insiders buying this stock (default 1)
    
    Returns:
        Composite score from 0.0 to 1.0 (can exceed 1.0 with bonuses, but capped)
    """
    # Quantity score (0-1)
    qty = purchase.get("quantity", 0) or 0
    qty_score = min(1.0, qty / max_qty) if max_qty > 0 else 0.0
    
    # Value score (0-1)
    value = purchase.get("value", 0) or 0
    if value == 0 and qty > 0 and purchase.get("price"):
        value = qty * purchase.get("price", 0)
    value_score = min(1.0, value / max_value) if max_value > 0 else 0.0
    
    # Recency score (0-1) - prefer trade_date over filing_date
    trade_date_str = purchase.get("trade_date", "")
    filing_date_str = purchase.get("filing_date", "")
    
    days_ago_val = None
    if trade_date_str:
        trade_date_obj = parse_date(trade_date_str)
        if trade_date_obj:
            days_ago_val = days_ago(trade_date_obj)
    elif filing_date_str:
        filing_date_obj = parse_date(filing_date_str)
        if filing_date_obj:
            days_ago_val = days_ago(filing_date_obj)
    
    recency = recency_score(days_ago_val, max_days=7)
    
    # Price score (0-1) - lower price = better (penny stocks < $5)
    price = purchase.get("price", 0) or 0
    if price <= 0:
        price_score = 0.0
    elif price <= 5.0:  # Penny stock range
        price_score = 1.0 - (price / 5.0) * 0.3  # 1.0 to 0.7 for $0-$5
    elif price <= 20.0:
        price_score = 0.7 - ((price - 5.0) / 15.0) * 0.4  # 0.7 to 0.3 for $5-$20
    else:
        price_score = max(0.0, 0.3 - ((price - 20.0) / 50.0) * 0.3)  # 0.3 to 0.0 for $20+
    
    # Insider title bonus
    title = str(purchase.get("title", "")).upper()
    title_bonus = 0.0
    if "CEO" in title:
        title_bonus = 0.1
    elif "CFO" in title:
        title_bonus = 0.05
    
    # Cluster buy bonus (multiple insiders buying same stock)
    cluster_bonus = 0.0
    if cluster_size >= 2:
        cluster_bonus = min(0.15, (cluster_size - 1) * 0.05)  # +0.05 per additional insider, max +0.15
    
    # Weighted combination
    # Quantity: 30%, Value: 30%, Recency: 25%, Price: 15%
    composite_score = (
        qty_score * 0.30 +
        value_score * 0.30 +
        recency * 0.25 +
        price_score * 0.15
    )
    
    # Add bonuses (can push score above 1.0, so cap at 1.0)
    composite_score += title_bonus + cluster_bonus
    
    return min(1.0, max(0.0, composite_score))


def scrape_openinsider_purchases(since_date: Optional[date] = None) -> List[Dict]:
    """
    Scrape OpenInsider main page for purchase transactions (DISCOVERY).
    
    Args:
        since_date: Only return purchases with trade_date or filing_date after this date
    
    Returns:
        List of purchase dictionaries with details
    """
    purchases = []
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        logger.info(f"Fetching {BASE_URL}...")
        if since_date:
            logger.debug(f"Date filter: Only including purchases from {since_date} onwards")
        else:
            logger.debug("No date filter: Including all purchases")
        resp = requests.get(BASE_URL, headers=headers, timeout=15)
        
        if not resp.ok:
            logger.warning(f"OpenInsider request failed: {resp.status_code}")
            return purchases
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find all tables on the page
        tables = soup.find_all("table")
        logger.debug(f"Found {len(tables)} table(s)")
        
        # Look for specific sections: "Latest Penny Stock Buys" and "Latest Insider Purchases"
        penny_stock_section = None
        insider_purchases_section = None
        
        # Method 1: Find headings/links and their associated tables
        for element in soup.find_all(["a", "h2", "h3", "h4", "strong", "b"]):
            element_text = element.get_text().strip().lower()
            
            # Find "Latest Penny Stock Buys"
            if "penny stock" in element_text and "buy" in element_text:
                next_table = element.find_next("table")
                if next_table and penny_stock_section is None:
                    penny_stock_section = next_table
                    logger.debug("Found 'Latest Penny Stock Buys' table")
            
            # Find "Latest Insider Purchases" or "Latest Insider Buys"
            if ("insider" in element_text and "purchase" in element_text) or \
               ("insider" in element_text and "buy" in element_text and "latest" in element_text):
                next_table = element.find_next("table")
                if next_table and insider_purchases_section is None:
                    insider_purchases_section = next_table
                    logger.debug("Found 'Latest Insider Purchases' table")
        
        # Method 2: Look for tables with section names in nearby text
        for table in tables:
            # Check previous elements for section identifiers
            for prev in table.find_all_previous(limit=10):
                if prev and hasattr(prev, 'get_text'):
                    prev_text = prev.get_text().lower()
                    
                    if "penny stock" in prev_text and "buy" in prev_text and penny_stock_section is None:
                        penny_stock_section = table
                        logger.debug("Found 'Latest Penny Stock Buys' table (by nearby text)")
                    
                    if (("insider" in prev_text and "purchase" in prev_text) or \
                        ("insider" in prev_text and "buy" in prev_text and "latest" in prev_text)) and \
                       insider_purchases_section is None:
                        insider_purchases_section = table
                        logger.debug("Found 'Latest Insider Purchases' table (by nearby text)")
        
        # Collect all purchase tables to process
        tables_to_process = []
        if penny_stock_section:
            tables_to_process.append(penny_stock_section)
        if insider_purchases_section and insider_purchases_section not in tables_to_process:
            tables_to_process.append(insider_purchases_section)
        
        # Also add any other purchase tables we might have missed
        for table in tables:
            table_text = table.get_text()
            # Only process tables that mention purchases
            if "P - Purchase" in table_text or "Purchase" in table_text:
                if table not in tables_to_process:
                    tables_to_process.append(table)
        
        for table in tables_to_process:
            # Find all rows in this table
            rows = table.find_all("tr")
            
            # Look for header row (contains "Ticker" or "Trade Type")
            header_idx = -1
            for i, row in enumerate(rows):
                row_text = row.get_text().lower()
                if "ticker" in row_text and ("trade type" in row_text or "price" in row_text):
                    header_idx = i
                    break
            
            # If no header found, look for rows with actual data (have ticker links)
            if header_idx < 0:
                # Try to find first row with a ticker link (format /TICKER)
                for i, row in enumerate(rows):
                    links = row.find_all("a", href=True)
                    for link in links:
                        href = link.get("href", "")
                        if href.startswith("/") and len(href) > 1:
                            potential_ticker = href[1:].split("/")[0].upper()
                            if 2 <= len(potential_ticker) <= 5 and potential_ticker.isalpha():
                                header_idx = i - 1  # Assume header is one row before
                                break
                    if header_idx >= 0:
                        break
            
            if header_idx < 0:
                # Last resort: start from row 10 (skip complex headers)
                header_idx = 9
            
            # Process data rows
            data_rows = rows[header_idx + 1:]
            logger.debug(f"Processing {len(data_rows)} data rows from table")
            
            rows_processed = 0
            rows_skipped_no_purchase = 0
            rows_skipped_no_ticker = 0
            rows_skipped_essential_data = 0
            rows_skipped_date_filter = 0
            
            for row in data_rows:
                rows_processed += 1
                cells = row.find_all(["td", "th"])
                
                if len(cells) < 5:  # Need at least ticker, company, type, price, qty
                    continue
                
                try:
                    # Extract data - look for cells with purchase info
                    row_text = " ".join([c.text.strip() for c in cells])
                    
                    # Check if this is a purchase row
                    if "P - Purchase" not in row_text and "Purchase" not in row_text:
                        rows_skipped_no_purchase += 1
                        continue
                    
                    # Find ticker (usually in a link or bold text)
                    ticker = ""
                    for cell in cells:
                        # Look for links with ticker in href (format: /TICKER)
                        links = cell.find_all("a", href=True)
                        for link in links:
                            href = link.get("href", "")
                            # Ticker is often in the href like /TICKER or /TICKER/...
                            if href.startswith("/") and len(href) > 1:
                                # Extract first part after /
                                parts = href[1:].split("/")
                                potential_ticker = parts[0].upper()
                                # Ticker should be 2-5 uppercase letters (skip single letters like "M")
                                if 2 <= len(potential_ticker) <= 5 and potential_ticker.isalpha():
                                    ticker = potential_ticker
                                    break
                        if ticker:
                            break
                        
                        # Or in bold text within the cell
                        bold = cell.find("b")
                        if bold:
                            text = bold.text.strip()
                            # Remove any trailing colons or punctuation
                            text = text.rstrip(":.,;")
                            if 2 <= len(text) <= 5 and text.isupper() and text.isalpha():
                                ticker = text
                                break
                        
                        # Or just in cell text (but be more careful)
                        text = cell.text.strip()
                        # Remove trailing colons/punctuation
                        text = text.rstrip(":.,;").split()[0] if text.split() else ""
                        if 2 <= len(text) <= 5 and text.isupper() and text.isalpha():
                            ticker = text
                            break
                    
                    if not ticker:
                        rows_skipped_no_ticker += 1
                        continue
                    
                    # Find company name (usually near ticker)
                    company = ""
                    for cell in cells:
                        text = cell.text.strip()
                        # Company names are longer and often have links
                        if len(text) > 5 and len(text) < 100 and ticker not in text:
                            link = cell.find("a", href=True)
                            if link:
                                company = text
                                break
                    
                    # Find trade type, price, quantity, value, insider info
                    trade_type = ""
                    price = None
                    qty = None
                    value = None
                    insider_name = ""
                    title = ""
                    filing_date = ""
                    trade_date = ""
                    
                    for i, cell in enumerate(cells):
                        cell_text = cell.text.strip()
                        
                        # Trade type
                        if "P - Purchase" in cell_text:
                            trade_type = "P - Purchase"
                        
                        # Price (starts with $)
                        if cell_text.startswith("$") and not cell_text.startswith("$0"):
                            try:
                                price = float(cell_text.replace("$", "").replace(",", ""))
                            except ValueError:
                                pass
                        
                        # Quantity (starts with + and has numbers)
                        if cell_text.startswith("+") and any(c.isdigit() for c in cell_text):
                            try:
                                qty = int(cell_text.replace("+", "").replace(",", ""))
                            except ValueError:
                                pass
                        
                        # Value (starts with +$)
                        if cell_text.startswith("+$"):
                            try:
                                value = float(cell_text.replace("+$", "").replace(",", ""))
                            except ValueError:
                                pass
                        
                        # Insider name and title (usually in cells with links to /insider/)
                        links = cell.find_all("a", href=True)
                        for link in links:
                            href = link.get("href", "")
                            if "/insider/" in href:
                                insider_name = link.text.strip()
                                break
                        
                        # Look for common titles (check if cell contains title keywords)
                        cell_text_upper = cell_text.upper()
                        if not title or len(cell_text) < 50:  # Only set if empty or if cell looks like a title
                            if "CEO" in cell_text_upper:
                                title = "CEO"
                            elif "CFO" in cell_text_upper:
                                title = "CFO"
                            elif "COO" in cell_text_upper:
                                title = "COO"
                            elif "PRESIDENT" in cell_text_upper:
                                title = "President"
                            elif "DIRECTOR" in cell_text_upper:
                                title = "Director"
                            elif "VP" in cell_text_upper or "VICE PRESIDENT" in cell_text_upper:
                                title = "VP"
                            elif "OFFICER" in cell_text_upper:
                                title = "Officer"
                        
                        # Dates (format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
                        # Check if it looks like a date (has YYYY-MM-DD pattern)
                        if cell_text.count("-") >= 2 and len(cell_text) >= 10:
                            try:
                                # Try to parse as date (handles both formats)
                                parsed_date = parse_date(cell_text)
                                if parsed_date:
                                    # If it's a link, it's likely a filing date (timestamp format)
                                    # Otherwise it might be a trade date
                                    if cell.find("a") and not filing_date:
                                        filing_date = cell_text[:10] if len(cell_text) > 10 else cell_text
                                    elif not trade_date:
                                        trade_date = cell_text[:10] if len(cell_text) > 10 else cell_text
                            except:
                                pass
                    
                    # Skip if we don't have essential data
                    if not ticker or len(ticker) < 2 or ":" in ticker or not trade_type or price is None:
                        rows_skipped_essential_data += 1
                        logger.debug(f"Skipping {ticker or 'unknown'}: ticker={bool(ticker)}, trade_type={bool(trade_type)}, price={price}")
                        continue
                    
                    # Filter by date if since_date is provided
                    # Use filing_date for filtering (when info became public), not trade_date
                    # This ensures we catch new filings even if the trade happened earlier
                    if since_date:
                        # Prefer filing_date for filtering since that's when the info became public
                        purchase_date = None
                        if filing_date:
                            purchase_date = parse_date(filing_date)
                        elif trade_date:
                            purchase_date = parse_date(trade_date)
                        
                        # Skip if purchase date is before since_date
                        # Use >= to include purchases from the same day as last SA
                        if purchase_date and purchase_date < since_date:
                            rows_skipped_date_filter += 1
                            logger.debug(f"Skipping {ticker}: purchase_date={purchase_date} < since_date={since_date} (trade_date={trade_date}, filing_date={filing_date})")
                            continue
                        elif purchase_date is None:
                            logger.debug(f"{ticker}: No date found (trade_date={trade_date}, filing_date={filing_date}), including anyway")
                    
                    # Calculate value if not found
                    if value is None and price is not None and qty is not None:
                        value = price * qty
                    
                    purchase = {
                        "ticker": ticker,
                        "company": company,
                        "insider_name": insider_name,
                        "title": title,
                        "trade_type": trade_type,
                        "price": price,
                        "quantity": qty,
                        "value": value,
                        "filing_date": filing_date,
                        "trade_date": trade_date,
                    }
                    
                    purchases.append(purchase)
                    logger.debug(f"Added purchase: {ticker} @ ${price} on {trade_date or filing_date or 'no date'}")
                    
                except Exception as e:
                    logger.debug(f"Error parsing purchase row: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue
            
            logger.debug(f"Table processing summary: {rows_processed} rows processed, "
                        f"{rows_skipped_no_purchase} skipped (not purchase), "
                        f"{rows_skipped_no_ticker} skipped (no ticker), "
                        f"{rows_skipped_essential_data} skipped (missing data), "
                        f"{rows_skipped_date_filter} skipped (date filter), "
                        f"{len([p for p in purchases if any(t.get('ticker') == p.get('ticker') for t in purchases)])} purchases found")
        
        logger.info(f"Scraped {len(purchases)} purchase transactions from OpenInsider")
        if purchases:
            logger.debug(f"Sample purchases: {[(p['ticker'], p.get('trade_date') or p.get('filing_date', 'no date')) for p in purchases[:5]]}")
        
    except Exception as e:
        logger.error(f"Error scraping OpenInsider: {e}")
        import traceback
        traceback.print_exc()
    
    return purchases


def check_net_selling(ticker: str) -> tuple:
    """
    Check if a ticker has net selling (more sales than purchases) by scraping the ticker page.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Tuple of (has_net_selling: bool, total_purchase_value: float, total_sale_value: float)
        Returns (False, 0, 0) if unable to check (e.g., page not found)
    """
    url = f"{BASE_URL}/{ticker}"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=15)
        
        if not resp.ok:
            logger.debug(f"Could not fetch {url} for net selling check: HTTP {resp.status_code}")
            return (False, 0, 0)
        
        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")
        
        total_purchase_value = 0.0
        total_sale_value = 0.0
        
        for table in tables:
            rows = table.find_all("tr")
            
            # Look for header row
            header_idx = -1
            for i, row in enumerate(rows):
                row_text = row.get_text().lower()
                if any(keyword in row_text for keyword in ["filing date", "trade date", "transaction", "insider", "price"]):
                    header_idx = i
                    break
            
            if header_idx < 0:
                continue
            
            # Process data rows
            for row in rows[header_idx + 1:]:
                cells = row.find_all(["td", "th"])
                if len(cells) < 5:
                    continue
                
                try:
                    row_text = " ".join([c.text.strip() for c in cells])
                    row_text_lower = row_text.lower()
                    
                    # Determine transaction type
                    is_purchase = any(indicator in row_text_lower for indicator in ["p - purchase", "purchase", "p -", "buy"])
                    is_sale = any(indicator in row_text_lower for indicator in ["s - sale", "sale", "s -", "sell"])
                    
                    if not is_purchase and not is_sale:
                        # Check individual cells for transaction type
                        for cell in cells:
                            cell_text = cell.text.strip().upper()
                            if "P" == cell_text or "PURCHASE" in cell_text:
                                is_purchase = True
                                break
                            elif "S" == cell_text or "SALE" in cell_text:
                                is_sale = True
                                break
                    
                    if not is_purchase and not is_sale:
                        continue
                    
                    # Extract price, quantity, and value
                    price = None
                    qty = None
                    transaction_value = 0.0
                    
                    for cell in cells:
                        cell_text = cell.text.strip()
                        cell_text_clean = cell_text.replace("$", "").replace(",", "").replace("+", "").replace("-", "")
                        
                        # Price (starts with $)
                        if cell_text.startswith("$") and not cell_text.startswith("$0") and price is None:
                            try:
                                price = float(cell_text.replace("$", "").replace(",", "").split()[0])
                            except:
                                pass
                        
                        # Quantity (starts with + or -)
                        if (cell_text.startswith("+") or cell_text.startswith("-")) and any(c.isdigit() for c in cell_text) and qty is None:
                            try:
                                qty_str = cell_text.replace("+", "").replace("-", "").replace(",", "")
                                qty = int(qty_str)
                            except:
                                pass
                        
                        # Value (starts with +$ or -$)
                        if cell_text.startswith("+$") or cell_text.startswith("-$"):
                            try:
                                value_str = cell_text.replace("+$", "").replace("-$", "").replace(",", "")
                                transaction_value = float(value_str)
                                break  # Found explicit value, use it
                            except:
                                pass
                    
                    # Calculate value if not found explicitly
                    if transaction_value == 0.0 and price is not None and qty is not None:
                        transaction_value = price * abs(qty)
                    
                    # Add to appropriate total (use absolute value for sales)
                    if is_purchase and transaction_value > 0:
                        total_purchase_value += transaction_value
                    elif is_sale and transaction_value > 0:
                        total_sale_value += abs(transaction_value)  # Sales are always positive in our calculation
                
                except Exception as e:
                    logger.debug(f"Error parsing transaction row for {ticker}: {e}")
                    continue
        
        has_net_selling = total_sale_value > total_purchase_value
        
        if has_net_selling:
            logger.info(f"{ticker}: Net selling detected - ${total_sale_value:,.0f} sold vs ${total_purchase_value:,.0f} bought")
        
        return has_net_selling, total_purchase_value, total_sale_value
    
    except Exception as e:
        logger.warning(f"Error checking net selling for {ticker}: {e}")
        return False, 0, 0


class Insider(AdvisorBase):

    def discover(self, sa):
        """Discover stocks with insider purchases above threshold score."""
        try:
            # Get last SA timestamp to filter purchases
            # We want purchases that happened after the last SA ran
            # Since we only have dates (not times) from OpenInsider, we include purchases
            # from the same day as last SA or later (>=)

            # Filter by today's date - we want to see all new filings from today
            # regardless of when the last SA ran (since filings can come in throughout the day)
            from datetime import date
            since_date = date.today()
            logger.info(f"Filtering purchases from today ({since_date}) onwards")
            # Scrape purchases
            purchases = scrape_openinsider_purchases(since_date=since_date)
            
            if not purchases:
                logger.info("No insider purchases found")
                return

            # Pass sell instructions to siacovery
            sell_instructions = [
                ("STOP_PERCENTAGE", 0.99),
                ("TARGET_PERCENTAGE", 1.50),
                ("AFTER_DAYS", 7.0),
                ('DESCENDING_TREND', -0.20),
                ('NOT_TRENDING', None)
            ]
            
            # Group purchases by ticker
            by_ticker = {}
            for p in purchases:
                ticker = p["ticker"]
                if ticker not in by_ticker:
                    by_ticker[ticker] = []
                by_ticker[ticker].append(p)
            
            # Calculate cluster sizes (purchases per ticker)
            ticker_counts = {ticker: len(purchases_list) for ticker, purchases_list in by_ticker.items()}
            
            # Calculate scores and discover stocks above threshold
            discovered_count = 0
            skipped_net_selling = 0
            for ticker, purchases_list in by_ticker.items():
                # Check for net selling - filter out stocks where insiders are net sellers
                has_net_selling, total_purchase_val, total_sale_val = check_net_selling(ticker)
                if has_net_selling:
                    logger.info(f"{ticker}: Skipping discovery due to net selling (${total_sale_val:,.0f} sold vs ${total_purchase_val:,.0f} bought)")
                    skipped_net_selling += 1
                    continue
                
                # Calculate score for each purchase
                # Sum scores - multiple purchases should strengthen the signal
                # The cluster bonus already rewards multiple insiders buying, and summing
                # further rewards having multiple insiders buying the same stock
                scores = [
                    score_insider_purchase(p, MAX_QTY, MAX_VALUE, ticker_counts.get(ticker, 1))
                    for p in purchases_list
                ]
                # Sum all purchase scores (capped at 1.0) - multiple purchases = stronger signal
                total_score = sum(scores)
                avg_score = min(1.0, total_score)  # Cap at 1.0 to keep in valid range
                
                # Only discover if score is above threshold
                if avg_score < DISCOVERY_THRESHOLD:
                    continue
                
                # Build explanation with purchase details
                total_value = sum(p.get("value", 0) or 0 for p in purchases_list)
                total_qty = sum(p.get("quantity", 0) or 0 for p in purchases_list)
                avg_price = sum(p.get("price", 0) for p in purchases_list) / len(purchases_list)
                
                # Get insider titles
                titles = [p.get("title", "") for p in purchases_list if p.get("title")]
                unique_titles = list(set(titles))
                title_info = ", ".join(unique_titles) if unique_titles else "Various"
                
                # Get dates
                dates = []
                for p in purchases_list:
                    if p.get("trade_date"):
                        dates.append(f"trade:{p['trade_date']}")
                    elif p.get("filing_date"):
                        dates.append(f"filing:{p['filing_date']}")
                date_info = ", ".join(dates[:3])  # Show first 3 dates
                if len(dates) > 3:
                    date_info += f" (+{len(dates)-3} more)"
                
                explanation = (
                    f"Insider purchase: {len(purchases_list)} purchase(s), total score {avg_score:.2f} | "
                    f"{total_qty:,} shares @ ${avg_price:.2f} avg = ${total_value:,.0f} | "
                    f"Insiders: {title_info} | Dates: {date_info}"
                )

                # Check if already discovered - rediscover if >1 days ago
                if not self.allow_discovery(ticker, period=24):
                    continue
                
                # Discover the stock
                stock = self.discovered(sa, ticker, explanation, sell_instructions)
                
                discovered_count += 1
            
            logger.info(f"Insider advisor discovered {discovered_count} stocks (from {len(by_ticker)} with purchases, {skipped_net_selling} skipped due to net selling)")
            
        except Exception as e:
            logger.error(f"Error in Insider.discover: {e}")
            import traceback
            traceback.print_exc()


register(name="Insider Tr", python_class="Insider")
