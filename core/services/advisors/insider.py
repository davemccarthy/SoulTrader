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
    Scrape OpenInsider screener page for purchase transactions (DISCOVERY).
    Uses /screener?daysago=0 to get today's transactions (more reliable than main page).
    
    Args:
        since_date: Only return purchases with trade_date or filing_date after this date
    
    Returns:
        List of purchase dictionaries with details
    """
    purchases = []
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        # Use screener URL for more reliable, structured data (like test script)
        screener_url = f"{BASE_URL}/screener?daysago=0"
        logger.info(f"Fetching {screener_url}...")
        if since_date:
            logger.debug(f"Date filter: Only including purchases from {since_date} onwards")
        else:
            logger.debug("Using screener page (today's transactions)")
        resp = requests.get(screener_url, headers=headers, timeout=15)
        
        if not resp.ok:
            logger.warning(f"OpenInsider request failed: {resp.status_code}")
            return purchases
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # The screener page has a single main table with all transactions
        # Find the main data table (usually the largest one with transaction data)
        tables = soup.find_all("table")
        logger.debug(f"Found {len(tables)} table(s)")
        
        # Look for the main screener results table
        # It typically has headers like "Filing Date", "Trade Date", "Ticker", etc.
        main_table = None
        for table in tables:
            table_text = table.get_text()
            # The screener table should have these key columns
            if all(keyword in table_text for keyword in ["Filing Date", "Trade Date", "Ticker", "Trade Type"]):
                main_table = table
                logger.debug("Found main screener table")
                break
        
        if not main_table:
            # Fallback: use the largest table
            main_table = max(tables, key=lambda t: len(t.find_all("tr"))) if tables else None
            if main_table:
                logger.debug("Using largest table as fallback")
        
        if not main_table:
            logger.warning("Could not find main data table")
            return purchases
        
        tables_to_process = [main_table]
        
        for table in tables_to_process:
            # Find all rows in this table
            rows = table.find_all("tr")
            
            # Find header row - try multiple strategies (same as test script)
            header_idx = -1
            
            # Strategy 1: Look for row with all key headers together
            for i, row in enumerate(rows):
                row_text = row.get_text().lower()
                if "filing date" in row_text and "trade date" in row_text and "ticker" in row_text:
                    header_idx = i
                    logger.debug(f"Found header at row {i+1} (strategy 1: all headers together)")
                    break
            
            # Strategy 2: Look for row with at least "ticker" and "trade type" or "price"
            if header_idx < 0:
                for i, row in enumerate(rows):
                    row_text = row.get_text().lower()
                    if "ticker" in row_text and ("trade type" in row_text or "price" in row_text):
                        header_idx = i
                        logger.debug(f"Found header at row {i+1} (strategy 2: ticker + trade type/price)")
                        break
            
            # Strategy 3: Look for row with filing date or trade date plus ticker
            if header_idx < 0:
                for i, row in enumerate(rows):
                    row_text = row.get_text().lower()
                    if ("filing date" in row_text or "trade date" in row_text) and "ticker" in row_text:
                        header_idx = i
                        logger.debug(f"Found header at row {i+1} (strategy 3: date + ticker)")
                        break
            
            # Strategy 4: Look for row that looks like a header (has multiple column headers)
            if header_idx < 0:
                for i, row in enumerate(rows[:20]):  # Check first 20 rows
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 8:  # Screener table has many columns
                        cell_texts = [c.get_text().strip().lower() for c in cells]
                        # Count how many cells look like headers
                        header_like_count = 0
                        header_keywords = ["date", "ticker", "company", "insider", "title", "type", "price", "qty", "value", "owned"]
                        for cell_text in cell_texts:
                            if len(cell_text) < 30 and any(keyword in cell_text for keyword in header_keywords):
                                header_like_count += 1
                        if header_like_count >= 4:  # At least 4 cells look like headers
                            header_idx = i
                            logger.debug(f"Found header at row {i+1} (strategy 4: multiple header-like cells)")
                            break
            
            if header_idx < 0:
                logger.warning("Could not find header row, using row 10 as fallback")
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
                
                if len(cells) < 8:  # Screener table has many columns
                    continue
                
                try:
                    # Extract data from cells - screener page format
                    row_text = " ".join([c.text.strip() for c in cells])
                    
                    # Check if this is a purchase row
                    if "P - Purchase" not in row_text and "Purchase" not in row_text:
                        rows_skipped_no_purchase += 1
                        continue
                    
                    # Screener table column order is typically:
                    # [0]: X/indicators, [1]: Filing Date, [2]: Trade Date, [3]: Ticker, 
                    # [4]: Company, [5]: Insider, [6]: Title, [7]: Trade Type, [8]: Price, etc.
                    
                    ticker = ""
                    company = ""
                    insider_name = ""
                    title = ""
                    trade_type = ""
                    price = None
                    qty = None
                    value = None
                    filing_date = ""
                    trade_date = ""
                    
                    # Extract ticker (look for links with format /TICKER)
                    for i, cell in enumerate(cells):
                        cell_text = cell.text.strip()
                        
                        # Ticker is usually in a link
                        links = cell.find_all("a", href=True)
                        for link in links:
                            href = link.get("href", "")
                            if href.startswith("/") and len(href) > 1:
                                # Extract first part after / (skip /insider/, etc.)
                                parts = href[1:].split("/")
                                potential_ticker = parts[0].upper()
                                # Ticker should be 2-5 uppercase letters, not "insider", "screener", etc.
                                if (2 <= len(potential_ticker) <= 5 and 
                                    potential_ticker.isalpha() and 
                                    potential_ticker not in ["INSIDER", "SCREENER", "LATEST"]):
                                    ticker = potential_ticker
                                    break
                        if ticker:
                            break
                    
                    if not ticker:
                        rows_skipped_no_ticker += 1
                        continue
                    
                    # Extract other fields from cells
                    for i, cell in enumerate(cells):
                        cell_text = cell.text.strip()
                        
                        # Filing Date (usually has link with date in YYYY-MM-DD format)
                        if not filing_date:
                            link = cell.find("a", href=True)
                            if link and link.get("href", "").startswith("/"):
                                # Extract date from link text or href
                                date_text = link.text.strip()
                                if date_text.count("-") >= 2 and len(date_text) >= 10:
                                    filing_date = date_text[:10]
                        
                        # Trade Date (similar to filing date)
                        if not trade_date and filing_date:
                            # Trade date is usually in same cell or nearby
                            if cell_text.count("-") >= 2 and len(cell_text) >= 10:
                                try:
                                    # Try to parse as date
                                    parsed_date = parse_date(cell_text[:10])
                                    if parsed_date:
                                        trade_date = cell_text[:10]
                                except:
                                    pass
                        
                        # Company (usually after ticker, has link to company page)
                        if not company and len(cell_text) > 5 and len(cell_text) < 100:
                            link = cell.find("a", href=True)
                            if link and ticker not in cell_text:
                                company = cell_text
                        
                        # Trade type
                        if "P - Purchase" in cell_text:
                            trade_type = "P - Purchase"
                        
                        # Price (starts with $)
                        if cell_text.startswith("$") and not cell_text.startswith("$0"):
                            try:
                                price = float(cell_text.replace("$", "").replace(",", ""))
                            except ValueError:
                                pass
                        
                        # Quantity (starts with + and has numbers, or just numbers)
                        if cell_text.startswith("+") and any(c.isdigit() for c in cell_text):
                            try:
                                qty = int(cell_text.replace("+", "").replace(",", ""))
                            except ValueError:
                                pass
                        elif not qty and cell_text.replace(",", "").isdigit() and int(cell_text.replace(",", "")) > 0:
                            try:
                                qty = int(cell_text.replace(",", ""))
                            except ValueError:
                                pass
                        
                        # Value (starts with +$ or just $)
                        if cell_text.startswith("+$") or (cell_text.startswith("$") and "," in cell_text):
                            try:
                                value = float(cell_text.replace("+$", "").replace("$", "").replace(",", ""))
                            except ValueError:
                                pass
                        
                        # Insider name (link to /insider/)
                        links = cell.find_all("a", href=True)
                        for link in links:
                            href = link.get("href", "")
                            if "/insider/" in href:
                                insider_name = link.text.strip()
                                break
                        
                        # Title (contains CEO, CFO, etc.)
                        cell_text_upper = cell_text.upper()
                        if len(cell_text) < 50:  # Titles are usually short
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
                        
                        # Dates - extract from cells
                        # Filing Date usually has a link with timestamp
                        if not filing_date and cell.find("a"):
                            link = cell.find("a", href=True)
                            if link:
                                link_text = link.text.strip()
                                if link_text.count("-") >= 2 and len(link_text) >= 10:
                                    filing_date = link_text[:10]  # Extract YYYY-MM-DD part
                        
                        # Trade Date (usually plain text, no link)
                        if not trade_date and not cell.find("a") and cell_text.count("-") >= 2 and len(cell_text) >= 10:
                            trade_date = cell_text[:10]  # Extract YYYY-MM-DD part
                    
                    # Skip if we don't have essential data
                    if not ticker or len(ticker) < 2 or ":" in ticker or not trade_type or price is None:
                        rows_skipped_essential_data += 1
                        logger.debug(f"Skipping {ticker or 'unknown'}: ticker={bool(ticker)}, trade_type={bool(trade_type)}, price={price}")
                        continue
                    
                    # Filter by date if since_date is provided
                    if since_date:
                        purchase_date = None
                        if filing_date:
                            purchase_date = parse_date(filing_date)
                        elif trade_date:
                            purchase_date = parse_date(trade_date)
                        
                        if purchase_date and purchase_date < since_date:
                            rows_skipped_date_filter += 1
                            continue
                        elif purchase_date is None:
                            logger.debug(f"{ticker}: No date found, including anyway")
                    
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


def check_net_selling(ticker: str, days_lookback: int = 90) -> tuple:
    """
    Check if a ticker has net selling (more sales than purchases) by scraping the ticker page.
    Only looks at recent transactions (within days_lookback days) to avoid counting old historical sales.
    
    Args:
        ticker: Stock ticker symbol
        days_lookback: Only consider transactions within this many days (default 90)
        
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
        total_purchase_shares = 0
        total_sale_shares = 0
        
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
                    
                    # Extract date first to check if transaction is recent (within lookback window)
                    transaction_date = None
                    transaction_is_recent = False
                    
                    for cell in cells:
                        cell_text = cell.text.strip()
                        # Dates (YYYY-MM-DD format or YYYY-MM-DD HH:MM:SS)
                        if len(cell_text) >= 10 and cell_text.count("-") >= 2:
                            try:
                                date_part = cell_text[:10]
                                transaction_date = parse_date(date_part)
                                if transaction_date:
                                    # Check if transaction is within lookback window
                                    days_ago = (date.today() - transaction_date).days
                                    if days_ago <= days_lookback:
                                        transaction_is_recent = True
                                    break
                            except:
                                pass
                    
                    # Skip if transaction is too old (outside lookback window)
                    # If no date found, include it (to avoid missing valid transactions)
                    if transaction_date is not None and not transaction_is_recent:
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
                    
                    # Add to appropriate totals (track both shares and values)
                    if is_purchase and transaction_value > 0:
                        total_purchase_value += transaction_value
                        if qty is not None and qty > 0:
                            total_purchase_shares += qty
                    elif is_sale and transaction_value > 0:
                        total_sale_value += abs(transaction_value)  # Sales are always positive in our calculation
                        if qty is not None and qty > 0:
                            total_sale_shares += abs(qty)  # Use absolute value for shares
                
                except Exception as e:
                    logger.debug(f"Error parsing transaction row for {ticker}: {e}")
                    continue
        
        # Check net selling by share quantity (more meaningful for insider sentiment)
        # Compare shares first, then dollar values as tiebreaker
        has_net_selling = False
        if total_sale_shares > 0 and total_purchase_shares > 0:
            # If more shares sold than bought, that's net selling
            has_net_selling = total_sale_shares > total_purchase_shares
        elif total_sale_shares > 0 and total_purchase_shares == 0:
            # If only sales and no purchases, that's net selling
            has_net_selling = True
        elif total_sale_shares == 0:
            # No sales at all, so no net selling
            has_net_selling = False
        else:
            # Fallback to dollar value comparison if share counts aren't available
            has_net_selling = total_sale_value > total_purchase_value
        
        if has_net_selling:
            if total_purchase_shares > 0 or total_sale_shares > 0:
                logger.info(f"{ticker}: Net selling detected - {total_sale_shares:,} shares (${total_sale_value:,.0f}) sold vs {total_purchase_shares:,} shares (${total_purchase_value:,.0f}) bought")
            else:
                logger.info(f"{ticker}: Net selling detected - ${total_sale_value:,.0f} sold vs ${total_purchase_value:,.0f} bought (share counts unavailable)")
        
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

            # The screener URL (/screener?daysago=0) already filters to today's transactions,
            # so no need for additional date filtering (which could exclude purchases due to date parsing issues)
            logger.info("Scraping purchases from screener (already filtered to today by URL)")
            # Scrape purchases without date filter - screener URL handles filtering
            purchases = scrape_openinsider_purchases(since_date=None)
            
            if not purchases:
                logger.info("No insider purchases found")
                return

            # Pass sell instructions to siacovery
            sell_instructions = [
                ("PERCENTAGE_DIMINISHING", 1.30, 7),
                ("PERCENTAGE_AUGMENTING", 0.95, 14),
                ('DESCENDING_TREND', -0.20, None),
                ('NOT_TRENDING', None, None)
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
