"""
Simple OpenInsider scraper for discovering stocks with insider purchases.

Scrapes the OpenInsider screener page (http://openinsider.com/screener?daysago=0) for purchase transactions.
Uses the screener URL to get today's transactions in a more structured format.
Focuses on discovery - finds stocks with insider buying activity.
Shows current prices and gains to explore insider data performance.
Includes recency scoring - more recent purchases are weighted higher.
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import yfinance as yf
from datetime import datetime, date

BASE_URL = "http://openinsider.com"

# Discovery threshold - only discover stocks with score above this (matches insider.py)
DISCOVERY_THRESHOLD = 0.3

# Scoring normalization constants (matches insider.py)
MAX_QTY = 1000000  # 1M shares
MAX_VALUE = 5000000  # $5M


def scrape_openinsider_purchases() -> List[Dict]:
    """
    Scrape OpenInsider screener page for purchase transactions (DISCOVERY).
    Uses /screener?daysago=0 to get today's transactions.
    
    Returns:
        List of purchase dictionaries with details
    """
    purchases = []
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        # Use screener URL with daysago=0 to get today's transactions
        screener_url = f"{BASE_URL}/screener?daysago=0"
        print(f"Fetching {screener_url}...")
        resp = requests.get(screener_url, headers=headers, timeout=15)
        
        if not resp.ok:
            print(f"❌ Request failed: {resp.status_code}")
            return purchases
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # The screener page has a single main table with all transactions
        # Find the main data table (usually the largest one with transaction data)
        tables = soup.find_all("table")
        print(f"Found {len(tables)} table(s)")
        
        # Look for the main screener results table
        # It typically has headers like "Filing Date", "Trade Date", "Ticker", etc.
        main_table = None
        for table in tables:
            table_text = table.get_text()
            # The screener table should have these key columns
            if all(keyword in table_text for keyword in ["Filing Date", "Trade Date", "Ticker", "Trade Type"]):
                main_table = table
                print(f"  Found main screener table")
                break
        
        if not main_table:
            # Fallback: use the largest table
            main_table = max(tables, key=lambda t: len(t.find_all("tr"))) if tables else None
            if main_table:
                print(f"  Using largest table as fallback")
        
        if not main_table:
            print("❌ Could not find main data table")
            return purchases
        
        # Find header row - try multiple strategies
        rows = main_table.find_all("tr")
        header_idx = -1
        
        # Strategy 1: Look for row with all key headers together
        for i, row in enumerate(rows):
            row_text = row.get_text().lower()
            if "filing date" in row_text and "trade date" in row_text and "ticker" in row_text:
                header_idx = i
                print(f"    Found header at row {i+1} (strategy 1: all headers together)")
                break
        
        # Strategy 2: Look for row with at least "ticker" and "trade type" or "price"
        if header_idx < 0:
            for i, row in enumerate(rows):
                row_text = row.get_text().lower()
                if "ticker" in row_text and ("trade type" in row_text or "price" in row_text):
                    header_idx = i
                    print(f"    Found header at row {i+1} (strategy 2: ticker + trade type/price)")
                    break
        
        # Strategy 3: Look for row with filing date or trade date plus ticker
        if header_idx < 0:
            for i, row in enumerate(rows):
                row_text = row.get_text().lower()
                if ("filing date" in row_text or "trade date" in row_text) and "ticker" in row_text:
                    header_idx = i
                    print(f"    Found header at row {i+1} (strategy 3: date + ticker)")
                    break
        
        # Strategy 4: Look for row that looks like a header (has multiple column headers)
        if header_idx < 0:
            for i, row in enumerate(rows[:20]):  # Check first 20 rows
                cells = row.find_all(["td", "th"])
                if len(cells) >= 8:  # Screener table has many columns
                    cell_texts = [c.get_text().strip().lower() for c in cells]
                    # Count how many cells look like headers (short text, no numbers, common header terms)
                    header_like_count = 0
                    header_keywords = ["date", "ticker", "company", "insider", "title", "type", "price", "qty", "value", "owned"]
                    for cell_text in cell_texts:
                        if len(cell_text) < 30 and any(keyword in cell_text for keyword in header_keywords):
                            header_like_count += 1
                    if header_like_count >= 4:  # At least 4 cells look like headers
                        header_idx = i
                        print(f"    Found header at row {i+1} (strategy 4: multiple header-like cells)")
                        break
        
        if header_idx < 0:
            print("    Could not find header row, using row 10 as fallback")
            header_idx = 9
        
        # Process data rows
        data_rows = rows[header_idx + 1:]
        print(f"    Found {len(data_rows)} data rows")
        
        rows_processed = 0
        rows_skipped_no_purchase = 0
        rows_skipped_no_ticker = 0
        rows_skipped_essential_data = 0
        
        for row in data_rows:
            rows_processed += 1
            cells = row.find_all(["td", "th"])
            
            if len(cells) < 8:  # Screener table has many columns
                continue
            
            try:
                row_text = " ".join([c.text.strip() for c in cells])
                
                # Check if this is a purchase row
                if "P - Purchase" not in row_text and "Purchase" not in row_text:
                    rows_skipped_no_purchase += 1
                    continue
                
                # Extract data from cells
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
                
                # Find ticker (look for links with format /TICKER)
                for i, cell in enumerate(cells):
                    cell_text = cell.text.strip()
                    
                    # Ticker is usually in a link
                    links = cell.find_all("a", href=True)
                    for link in links:
                        href = link.get("href", "")
                        if href.startswith("/") and len(href) > 1:
                            potential_ticker = href[1:].split("/")[0].upper()
                            if 2 <= len(potential_ticker) <= 5 and potential_ticker.isalpha():
                                ticker = potential_ticker
                                # Company name might be in the link text
                                company_text = link.text.strip()
                                if len(company_text) > 5:
                                    company = company_text
                                break
                    
                    if ticker:
                        break
                
                if not ticker:
                    rows_skipped_no_ticker += 1
                    continue
                
                # Extract other fields
                for i, cell in enumerate(cells):
                    cell_text = cell.text.strip()
                    
                    # Dates (YYYY-MM-DD format or YYYY-MM-DD HH:MM:SS)
                    if len(cell_text) >= 10 and cell_text.count("-") >= 2:
                        try:
                            # Extract date part (first 10 characters for YYYY-MM-DD)
                            date_part = cell_text[:10]
                            # Also check if it might be a timestamp format
                            if len(cell_text) > 10 and " " in cell_text:
                                # Could be YYYY-MM-DD HH:MM:SS format
                                parts = cell_text.split()
                                if len(parts) > 0:
                                    date_part = parts[0]
                            
                            parsed_date = parse_date(date_part)
                            if parsed_date:
                                # Usually filing date comes before trade date
                                # Check if it's a link (filing date) or not (trade date)
                                if cell.find("a") and not filing_date:
                                    filing_date = date_part
                                elif not trade_date:
                                    trade_date = date_part
                        except:
                            pass
                    
                    # Trade type
                    if "P - Purchase" in cell_text:
                        trade_type = "P - Purchase"
                    
                    # Price (starts with $)
                    if cell_text.startswith("$") and not cell_text.startswith("$0") and price is None:
                        try:
                            price = float(cell_text.replace("$", "").replace(",", "").split()[0])
                        except ValueError:
                            pass
                    
                    # Quantity (starts with +)
                    if cell_text.startswith("+") and any(c.isdigit() for c in cell_text) and qty is None:
                        try:
                            qty = int(cell_text.replace("+", "").replace(",", "").split()[0])
                        except ValueError:
                            pass
                    
                    # Value (starts with +$)
                    if cell_text.startswith("+$") and value is None:
                        try:
                            value = float(cell_text.replace("+$", "").replace(",", "").split()[0])
                        except ValueError:
                            pass
                    
                    # Insider name (links to /insider/)
                    links = cell.find_all("a", href=True)
                    for link in links:
                        href = link.get("href", "")
                        if "/insider/" in href:
                            insider_name = link.text.strip()
                            break
                    
                    # Title (check cell text for common titles)
                    cell_text_upper = cell_text.upper()
                    if not title or len(cell_text) < 50:
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
                
                # Skip if essential data missing
                if not ticker or not trade_type or price is None:
                    rows_skipped_essential_data += 1
                    continue
                
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
                    "is_penny_stock": price <= 5.0 if price else False,
                    "source": "screener",
                    "filing_date": filing_date,
                    "trade_date": trade_date,
                }
                
                purchases.append(purchase)
                
            except Exception as e:
                print(f"  Error parsing row: {e}")
                continue
        
        print(f"✅ Found {len(purchases)} purchase transactions from screener")
        print(f"  Processing summary: {rows_processed} rows processed, "
              f"{rows_skipped_no_purchase} skipped (not purchase), "
              f"{rows_skipped_no_ticker} skipped (no ticker), "
              f"{rows_skipped_essential_data} skipped (missing data)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    return purchases


def get_current_price(ticker: str) -> Optional[float]:
    """Get current stock price using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if current_price:
            return float(current_price)
        
        # Fallback to history
        hist = stock.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        
        return None
    except Exception as e:
        return None


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
            print(f"  ⚠️  Could not check net selling for {ticker}: HTTP {resp.status_code}")
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
                                    days_ago_val = days_ago(transaction_date)
                                    if days_ago_val is not None and days_ago_val <= days_lookback:
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
                print(f"  ⚠️  {ticker}: Net selling detected - {total_sale_shares:,} shares (${total_sale_value:,.0f}) sold vs {total_purchase_shares:,} shares (${total_purchase_value:,.0f}) bought")
            else:
                print(f"  ⚠️  {ticker}: Net selling detected - ${total_sale_value:,.0f} sold vs ${total_purchase_value:,.0f} bought (share counts unavailable)")
        
        return has_net_selling, total_purchase_value, total_sale_value
    
    except Exception as e:
        print(f"  ⚠️  Error checking net selling for {ticker}: {e}")
        return False, 0, 0


def score_insider_purchase(purchase: Dict, max_qty: float = 1000000, max_value: float = 5000000, cluster_size: int = 1) -> float:
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


def discover_stocks_with_insider_purchases(show_prices: bool = True, filter_ticker: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    Discover stocks with insider purchases from main page.
    
    Args:
        show_prices: If True, fetch and display current prices and gains
        filter_ticker: If provided, only show this ticker
    
    Returns:
        Dictionary mapping ticker to list of purchases
    """
    print(f"\n{'='*60}")
    print(f"DISCOVERING stocks with insider purchases")
    if filter_ticker:
        print(f"Filtering for: {filter_ticker}")
    print(f"{'='*60}\n")
    
    purchases = scrape_openinsider_purchases()
    
    if not purchases:
        print("❌ No purchases found")
        return {}
    
    # Group by ticker
    by_ticker = {}
    for p in purchases:
        ticker = p["ticker"]
        if filter_ticker and ticker.upper() != filter_ticker.upper():
            continue
        if ticker not in by_ticker:
            by_ticker[ticker] = []
        by_ticker[ticker].append(p)
    
    if not by_ticker:
        print(f"❌ No purchases found for filter: {filter_ticker}")
        return {}
    
    print(f"\n✅ Found {len(by_ticker)} unique stocks with purchases")
    
    # Calculate scores and filter by threshold to find "potential buys"
    potential_buys = {}
    ticker_counts = {ticker: len(purchases_list) for ticker, purchases_list in by_ticker.items()}
    
    skipped_net_selling = 0
    for ticker, purchases_list in by_ticker.items():
        # Check for net selling - filter out stocks where insiders are net sellers (matches advisor logic)
        has_net_selling, total_purchase_val, total_sale_val = check_net_selling(ticker)
        if has_net_selling:
            print(f"  ❌ {ticker}: Skipping due to net selling (${total_sale_val:,.0f} sold vs ${total_purchase_val:,.0f} bought)")
            skipped_net_selling += 1
            continue
        
        # Calculate score for each purchase
        scores = [
            score_insider_purchase(p, MAX_QTY, MAX_VALUE, ticker_counts.get(ticker, 1))
            for p in purchases_list
        ]
        # Sum all purchase scores (capped at 1.0) - multiple purchases = stronger signal
        total_score = sum(scores)
        avg_score = min(1.0, total_score)  # Cap at 1.0 to keep in valid range
        
        # Only include if score is above threshold (potential buy)
        if avg_score >= DISCOVERY_THRESHOLD:
            potential_buys[ticker] = {
                'purchases': purchases_list,
                'score': avg_score,
                'total_score': total_score
            }
    
    print(f"   → {len(potential_buys)} stocks meet discovery threshold (score >= {DISCOVERY_THRESHOLD})")
    if skipped_net_selling > 0:
        print(f"   → {skipped_net_selling} stocks skipped due to net selling\n")
    else:
        print()
    
    if not potential_buys:
        print("❌ No stocks meet the discovery threshold. All purchases scored too low.")
        return {}
    
    # Sort by score (best opportunities first)
    sorted_tickers = sorted(
        potential_buys.items(),
        key=lambda x: x[1]['score'],
        reverse=True,
    )
    
    # Get current prices if requested
    price_data = {}
    if show_prices:
        print("Fetching current prices...")
        for ticker, _ in sorted_tickers:
            current_price = get_current_price(ticker)
            price_data[ticker] = current_price
            if current_price:
                print(f"  {ticker}: ${current_price:.2f}")
            else:
                print(f"  {ticker}: ❌ Price not available")
        print()
    
    # Display results (only potential buys)
    for ticker, data in sorted_tickers:
        purchases_list = data['purchases']
        score = data['score']
        total_value = sum(p.get("value", 0) or 0 for p in purchases_list)
        avg_purchase_price = sum(p.get("price", 0) for p in purchases_list) / len(purchases_list)
        
        print(f"{'='*60}")
        print(f"{ticker}: Score {score:.2f} | {len(purchases_list)} purchase(s), ${total_value:,.0f} total")
        print(f"{'='*60}")
        
        # Show purchase details
        for i, p in enumerate(purchases_list, 1):
            print(f"  Purchase {i}: ${p.get('price', 0):.2f} x {p.get('quantity', 0):,} = ${p.get('value', 0):,.0f}")
        
        # Show current price and gains if available
        if show_prices and ticker in price_data:
            current_price = price_data[ticker]
            if current_price:
                gain_pct = ((current_price - avg_purchase_price) / avg_purchase_price) * 100
                gain_sign = "📈" if gain_pct > 0 else "📉" if gain_pct < 0 else "➡️"
                print(f"\n  Avg Purchase Price: ${avg_purchase_price:.2f}")
                print(f"  Current Price:       ${current_price:.2f}")
                print(f"  Gain/Loss:           {gain_sign} {gain_pct:+.2f}%")
                
                # Calculate total gain if we had bought
                total_shares = sum(p.get("quantity", 0) for p in purchases_list)
                total_cost = sum(p.get("value", 0) or 0 for p in purchases_list)
                current_value = total_shares * current_price
                total_gain = current_value - total_cost
                total_gain_pct = (total_gain / total_cost * 100) if total_cost > 0 else 0
                print(f"  Total Value:         ${current_value:,.2f} (${total_gain:+,.2f}, {total_gain_pct:+.2f}%)")
            else:
                print(f"\n  ❌ Current price not available")
        
        print()
    
    # Return in same format as before for compatibility, but only potential buys
    return {ticker: data['purchases'] for ticker, data in potential_buys.items()}


def check_ticker_manual(ticker: str, purchase_price: float):
    """Manually check a ticker's performance given a purchase price."""
    print(f"\n{'='*60}")
    print(f"MANUAL CHECK: {ticker}")
    print(f"{'='*60}\n")
    
    current_price = get_current_price(ticker)
    if not current_price:
        print(f"❌ Could not get current price for {ticker}")
        return
    
    gain_pct = ((current_price - purchase_price) / purchase_price) * 100
    gain_sign = "📈" if gain_pct > 0 else "📉" if gain_pct < 0 else "➡️"
    
    print(f"Purchase Price: ${purchase_price:.2f}")
    print(f"Current Price:  ${current_price:.2f}")
    print(f"Gain/Loss:      {gain_sign} {gain_pct:+.2f}%")
    print()


if __name__ == "__main__":
    import sys
    
    # Check for manual ticker check (ticker + price)
    if len(sys.argv) >= 3:
        try:
            ticker = sys.argv[1].upper()
            purchase_price = float(sys.argv[2])
            check_ticker_manual(ticker, purchase_price)
        except ValueError:
            print("Usage: python test_openinsider.py [TICKER] [PURCHASE_PRICE]")
            print("   or: python test_openinsider.py [TICKER]")
            print("   or: python test_openinsider.py")
    # Check for filter argument (just ticker)
    elif len(sys.argv) > 1:
        filter_ticker = sys.argv[1].upper()
        discover_stocks_with_insider_purchases(show_prices=True, filter_ticker=filter_ticker)
    else:
        # Show all with summary
        results = discover_stocks_with_insider_purchases(show_prices=True, filter_ticker=None)
        
        # Calculate scores for all stocks and display sorted by score
        print(f"\n{'='*60}")
        print("STOCKS RANKED BY INSIDER PURCHASE SCORE")
        print(f"{'='*60}\n")
        
        # Calculate scores for all purchases
        all_purchases = []
        for ticker, purchases_list in results.items():
            all_purchases.extend(purchases_list)
        
        if all_purchases:
            # Find max values for normalization
            max_qty = max((p.get("quantity", 0) or 0 for p in all_purchases), default=1000000)
            max_value = max((p.get("value", 0) or 0 for p in all_purchases), default=5000000)
            
            # Calculate cluster sizes (purchases per ticker)
            ticker_counts = {}
            for p in all_purchases:
                ticker = p.get("ticker")
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
            
            # Calculate average score per stock
            stock_scores = {}
            for ticker, purchases_list in results.items():
                scores = [score_insider_purchase(p, max_qty, max_value, ticker_counts.get(ticker, 1)) for p in purchases_list]
                stock_scores[ticker] = sum(scores) / len(scores)  # Average score for the stock
            
            # Get current prices and calculate gains for each stock
            stock_performance = {}
            print("Fetching current prices for performance validation...")
            for ticker, purchases_list in results.items():
                avg_purchase_price = sum(p.get("price", 0) for p in purchases_list) / len(purchases_list)
                current_price = get_current_price(ticker)
                if current_price:
                    gain_pct = ((current_price - avg_purchase_price) / avg_purchase_price) * 100
                    stock_performance[ticker] = {
                        'gain_pct': gain_pct,
                        'buy_price': avg_purchase_price,
                        'current_price': current_price
                    }
            
            # Sort by score descending
            sorted_by_score = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Display in format: STK - 0.XX (gain %)
            print("\nScore | Price Performance")
            print("-" * 60)
            for ticker, score in sorted_by_score:
                perf = stock_performance.get(ticker)
                if perf:
                    gain_sign = "📈" if perf['gain_pct'] > 0 else "📉" if perf['gain_pct'] < 0 else "➡️"
                    print(f"{ticker:6s} - {score:.2f}  {gain_sign} {perf['gain_pct']:+6.2f}%  (${perf['buy_price']:.2f} → ${perf['current_price']:.2f})")
                else:
                    print(f"{ticker:6s} - {score:.2f}  (price data unavailable)")
            print()
        
        # Calculate summary stats
        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*60}\n")
        
        gains = []
        for ticker, purchases_list in results.items():
            avg_price = sum(p.get("price", 0) for p in purchases_list) / len(purchases_list)
            current_price = get_current_price(ticker)
            if current_price:
                gain_pct = ((current_price - avg_price) / avg_price) * 100
                gains.append((ticker, gain_pct, avg_price, current_price))
        
        if gains:
            gains.sort(key=lambda x: x[1], reverse=True)
            
            top_performers = gains[:5]
            worst_performers = gains[-5:]
            
            print("📈 TOP PERFORMERS:")
            for ticker, gain_pct, buy_price, current_price in top_performers:
                print(f"  {ticker:6s} {gain_pct:+7.2f}%  (${buy_price:.2f} → ${current_price:.2f})")
            
            print("\n📉 WORST PERFORMERS:")
            for ticker, gain_pct, buy_price, current_price in worst_performers:
                print(f"  {ticker:6s} {gain_pct:+7.2f}%  (${buy_price:.2f} → ${current_price:.2f})")
            
            avg_gain = sum(g[1] for g in gains) / len(gains)
            winners = sum(1 for g in gains if g[1] > 0)
            print(f"\n📊 STATS:")
            print(f"  Average Gain: {avg_gain:+.2f}%")
            print(f"  Winners: {winners}/{len(gains)} ({winners/len(gains)*100:.1f}%)")
            
            # Show all CEO purchases across all stocks
            all_ceo_purchases = []
            for ticker, purchases_list in results.items():
                for p in purchases_list:
                    if "CEO" in str(p.get("title", "")).upper():
                        all_ceo_purchases.append((ticker, p))
            
            if all_ceo_purchases:
                print(f"\n👔 ALL CEO PURCHASES (across all stocks):")
                for ticker, p in all_ceo_purchases:
                    price = p.get("price", 0) or 0
                    qty = p.get("quantity", 0) or 0
                    value = p.get("value", 0) or 0
                    if value == 0 and qty > 0:
                        value = price * qty
                    print(f"  {ticker}: ${price:.2f} x {qty:,} = ${value:,.0f} ({p.get('insider_name', 'Unknown')})")
            
            # Analyze differences between top and worst performers
            print(f"\n{'='*60}")
            print("COMPARATIVE ANALYSIS: TOP vs WORST PERFORMERS")
            print(f"{'='*60}\n")
            
            # Get purchase data for top and worst
            top_tickers = [g[0] for g in top_performers]
            worst_tickers = [g[0] for g in worst_performers]
            
            top_purchases = []
            worst_purchases = []
            
            for ticker, purchases_list in results.items():
                if ticker in top_tickers:
                    top_purchases.extend(purchases_list)
                elif ticker in worst_tickers:
                    worst_purchases.extend(purchases_list)
            
            # Compare metrics
            def analyze_purchases(purchases, label):
                if not purchases:
                    return {}
                
                # Calculate date metrics
                trade_dates = []
                filing_dates = []
                days_ago_list = []
                recency_scores = []
                
                for p in purchases:
                    trade_date_str = p.get("trade_date", "")
                    filing_date_str = p.get("filing_date", "")
                    
                    trade_date_obj = parse_date(trade_date_str)
                    filing_date_obj = parse_date(filing_date_str)
                    
                    if trade_date_obj:
                        trade_dates.append(trade_date_obj)
                        days = days_ago(trade_date_obj)
                        if days is not None:
                            days_ago_list.append(days)
                            recency_scores.append(recency_score(days))
                    elif filing_date_obj:
                        filing_dates.append(filing_date_obj)
                        days = days_ago(filing_date_obj)
                        if days is not None:
                            days_ago_list.append(days)
                            recency_scores.append(recency_score(days))
                
                avg_days_ago = sum(days_ago_list) / len(days_ago_list) if days_ago_list else None
                avg_recency = sum(recency_scores) / len(recency_scores) if recency_scores else None
                
                metrics = {
                    "count": len(purchases),
                    "avg_price": sum(p.get("price", 0) or 0 for p in purchases) / len(purchases),
                    "total_qty": sum(p.get("quantity", 0) or 0 for p in purchases),
                    "avg_qty": sum(p.get("quantity", 0) or 0 for p in purchases) / len(purchases),
                    "total_value": sum(p.get("value", 0) or 0 for p in purchases),
                    "avg_value": sum(p.get("value", 0) or 0 for p in purchases) / len(purchases),
                    "penny_stocks": sum(1 for p in purchases if p.get("is_penny_stock", False)),
                    "has_ceo": sum(1 for p in purchases if "CEO" in str(p.get("title", ""))),
                    "has_cfo": sum(1 for p in purchases if "CFO" in str(p.get("title", ""))),
                    "has_director": sum(1 for p in purchases if "Director" in str(p.get("title", ""))),
                    "has_officer": sum(1 for p in purchases if "Officer" in str(p.get("title", ""))),
                    "avg_days_ago": avg_days_ago,
                    "avg_recency_score": avg_recency,
                    "with_trade_date": len(trade_dates),
                    "with_filing_date": len(filing_dates),
                }
                
                print(f"{label}:")
                print(f"  Purchase Count: {metrics['count']}")
                print(f"  Avg Price: ${metrics['avg_price']:.2f}")
                print(f"  Total Quantity: {metrics['total_qty']:,} shares")
                print(f"  Avg Quantity per Purchase: {metrics['avg_qty']:,.0f} shares")
                print(f"  Total Value: ${metrics['total_value']:,.0f}")
                print(f"  Avg Value per Purchase: ${metrics['avg_value']:,.0f}")
                print(f"  Penny Stocks: {metrics['penny_stocks']}/{metrics['count']} ({metrics['penny_stocks']/metrics['count']*100:.1f}%)")
                if metrics['avg_days_ago'] is not None:
                    print(f"  Avg Days Ago: {metrics['avg_days_ago']:.1f} days")
                    print(f"  Avg Recency Score: {metrics['avg_recency_score']:.3f} (1.0 = very recent, 0.0 = old, 7-day window)")
                print(f"  With Trade Date: {metrics['with_trade_date']}/{metrics['count']}")
                print(f"  With Filing Date: {metrics['with_filing_date']}/{metrics['count']}")
                print(f"  CEO Purchases: {metrics['has_ceo']}/{metrics['count']}")
                if metrics['has_ceo'] > 0:
                    ceo_tickers = [p.get("ticker") for p in purchases if "CEO" in str(p.get("title", "")).upper()]
                    print(f"    CEO Tickers: {', '.join(set(ceo_tickers))}")
                print(f"  CFO Purchases: {metrics['has_cfo']}/{metrics['count']}")
                print(f"  Director Purchases: {metrics['has_director']}/{metrics['count']}")
                print(f"  Officer Purchases: {metrics['has_officer']}/{metrics['count']}")
                print()
                
                return metrics
            
            top_metrics = analyze_purchases(top_purchases, "📈 TOP PERFORMERS")
            worst_metrics = analyze_purchases(worst_purchases, "📉 WORST PERFORMERS")
            
            # Show differences
            if top_metrics and worst_metrics:
                print("🔍 KEY DIFFERENCES:")
                print(f"  Price: Top avg ${top_metrics['avg_price']:.2f} vs Worst avg ${worst_metrics['avg_price']:.2f}")
                print(f"  Quantity: Top avg {top_metrics['avg_qty']:,.0f} vs Worst avg {worst_metrics['avg_qty']:,.0f} ({top_metrics['avg_qty']/worst_metrics['avg_qty']:.2f}x)")
                print(f"  Value: Top avg ${top_metrics['avg_value']:,.0f} vs Worst avg ${worst_metrics['avg_value']:,.0f} ({top_metrics['avg_value']/worst_metrics['avg_value']:.2f}x)")
                print(f"  Penny Stocks: Top {top_metrics['penny_stocks']/top_metrics['count']*100:.1f}% vs Worst {worst_metrics['penny_stocks']/worst_metrics['count']*100:.1f}%")
                if top_metrics['avg_days_ago'] and worst_metrics['avg_days_ago']:
                    print(f"  Recency: Top avg {top_metrics['avg_days_ago']:.1f} days ago (score: {top_metrics['avg_recency_score']:.3f}) vs Worst avg {worst_metrics['avg_days_ago']:.1f} days ago (score: {worst_metrics['avg_recency_score']:.3f})")
                    print(f"    → Top performers are {worst_metrics['avg_days_ago']/top_metrics['avg_days_ago']:.2f}x more recent!")
                print(f"  CEO Involvement: Top {top_metrics['has_ceo']}/{top_metrics['count']} vs Worst {worst_metrics['has_ceo']}/{worst_metrics['count']}")
                
                # Explain trade date vs filing date
                print(f"\n📅 DATE EXPLANATION:")
                print(f"  Trade Date: When the insider actually bought the stock (more accurate)")
                print(f"  Filing Date: When the SEC filing was submitted (can be delayed)")
                print(f"  → Trade date is preferred for recency scoring")
                
                # Show scoring example
                print(f"\n{'='*60}")
                print("SCORING FACTORS FOR INSIDER PURCHASES")
                print(f"{'='*60}\n")
                print("Composite Score = Weighted combination of:")
                print("  • Quantity Score (30%): Larger purchases = higher conviction")
                print("  • Value Score (30%): Larger dollar amounts = higher conviction")
                print("  • Recency Score (25%): More recent = better (7-day window, prefer trade_date)")
                print("  • Price Score (15%): Lower price = better (penny stocks < $5)")
                print("  • Title Bonus: +0.1 for CEO, +0.05 for CFO")
                print("  • Cluster Bonus: +0.05 per additional insider (max +0.15)")
                print()
                
                # Calculate and show scores for top/worst
                if top_purchases and worst_purchases:
                    # Find max values for normalization
                    all_purchases = top_purchases + worst_purchases
                    max_qty = max((p.get("quantity", 0) or 0 for p in all_purchases), default=1000000)
                    max_value = max((p.get("value", 0) or 0 for p in all_purchases), default=5000000)
                    
                    # Calculate cluster sizes for each purchase (count purchases per ticker)
                    ticker_counts = {}
                    for p in all_purchases:
                        ticker = p.get("ticker")
                        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
                    
                    top_scores = [score_insider_purchase(p, max_qty, max_value, ticker_counts.get(p.get("ticker"), 1)) for p in top_purchases]
                    worst_scores = [score_insider_purchase(p, max_qty, max_value, ticker_counts.get(p.get("ticker"), 1)) for p in worst_purchases]
                    
                    print(f"📊 SCORING RESULTS:")
                    print(f"  Top Performers Avg Score: {sum(top_scores)/len(top_scores):.3f}")
                    print(f"  Worst Performers Avg Score: {sum(worst_scores)/len(worst_scores):.3f}")
                    print(f"  → Top performers score {sum(top_scores)/len(top_scores) / (sum(worst_scores)/len(worst_scores)):.2f}x higher!")
