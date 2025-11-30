"""
Simple OpenInsider scraper for discovering stocks with insider purchases.

Scrapes the main OpenInsider page (http://openinsider.com) for purchase transactions.
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


def scrape_openinsider_purchases() -> List[Dict]:
    """
    Scrape OpenInsider main page for purchase transactions (DISCOVERY).
    
    Returns:
        List of purchase dictionaries with details
    """
    purchases = []
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        print(f"Fetching {BASE_URL}...")
        resp = requests.get(BASE_URL, headers=headers, timeout=15)
        
        if not resp.ok:
            print(f"❌ Request failed: {resp.status_code}")
            return purchases
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find all tables on the page
        tables = soup.find_all("table")
        print(f"Found {len(tables)} table(s)")
        
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
                    print(f"  Found 'Latest Penny Stock Buys' table (by heading)")
            
            # Find "Latest Insider Purchases" or "Latest Insider Buys"
            if ("insider" in element_text and "purchase" in element_text) or \
               ("insider" in element_text and "buy" in element_text and "latest" in element_text):
                next_table = element.find_next("table")
                if next_table and insider_purchases_section is None:
                    insider_purchases_section = next_table
                    print(f"  Found 'Latest Insider Purchases' table (by heading)")
        
        # Method 2: Look for tables with section names in nearby text
        for table in tables:
            # Check previous elements for section identifiers
            for prev in table.find_all_previous(limit=10):
                if prev and hasattr(prev, 'get_text'):
                    prev_text = prev.get_text().lower()
                    
                    if "penny stock" in prev_text and "buy" in prev_text and penny_stock_section is None:
                        penny_stock_section = table
                        print(f"  Found 'Latest Penny Stock Buys' table (by nearby text)")
                    
                    if (("insider" in prev_text and "purchase" in prev_text) or \
                        ("insider" in prev_text and "buy" in prev_text and "latest" in prev_text)) and \
                       insider_purchases_section is None:
                        insider_purchases_section = table
                        print(f"  Found 'Latest Insider Purchases' table (by nearby text)")
        
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
            table_text = table.get_text()
            
            # Identify table type - check if this is one of our identified sections
            is_penny_stock = False
            is_insider_purchases = False
            
            # Check by comparing table objects or by checking nearby text
            if penny_stock_section and (table == penny_stock_section or id(table) == id(penny_stock_section)):
                is_penny_stock = True
            elif insider_purchases_section and (table == insider_purchases_section or id(table) == id(insider_purchases_section)):
                is_insider_purchases = True
            else:
                # Fallback: check nearby text
                prev = table.find_previous()
                for _ in range(5):
                    if prev:
                        prev_text = prev.get_text().lower() if hasattr(prev, 'get_text') else str(prev).lower()
                        if "penny stock" in prev_text and "buy" in prev_text:
                            is_penny_stock = True
                            break
                        if ("insider" in prev_text and "purchase" in prev_text) or \
                           ("insider" in prev_text and "buy" in prev_text and "latest" in prev_text):
                            is_insider_purchases = True
                            break
                        prev = prev.find_previous() if hasattr(prev, 'find_previous') else None
                    else:
                        break
            
            if is_penny_stock:
                print(f"  Processing PENNY STOCK BUYS table...")
            elif is_insider_purchases:
                print(f"  Processing LATEST INSIDER PURCHASES table...")
            else:
                print(f"  Processing table with purchase data...")
            
            # Find all rows in this table
            rows = table.find_all("tr")
            
            # Look for header row (contains "Ticker" or "Trade Type")
            header_idx = -1
            for i, row in enumerate(rows):
                row_text = row.get_text().lower()
                if "ticker" in row_text and ("trade type" in row_text or "price" in row_text):
                    header_idx = i
                    print(f"    Found header at row {i+1}")
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
                                print(f"    Found data row at {i+1}, assuming header at {header_idx+1}")
                                break
                    if header_idx >= 0:
                        break
            
            if header_idx < 0:
                # Last resort: start from row 10 (skip complex headers)
                header_idx = 9
                print(f"    Using default header position: row {header_idx+1}")
            
            # Process data rows
            data_rows = rows[header_idx + 1:]
            print(f"    Found {len(data_rows)} data rows")
            
            for row in data_rows:
                cells = row.find_all(["td", "th"])
                
                if len(cells) < 5:  # Need at least ticker, company, type, price, qty
                    continue
                
                try:
                    # Extract data - look for cells with purchase info
                    row_text = " ".join([c.text.strip() for c in cells])
                    
                    # Check if this is a purchase row
                    if "P - Purchase" not in row_text and "Purchase" not in row_text:
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
                                # Title might be in next cell or same cell
                                break
                        
                        # Look for common titles (check if cell contains title keywords)
                        # Don't overwrite if we already found a title, but prefer more specific ones
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
                        
                        # Dates (format: YYYY-MM-DD)
                        if len(cell_text) == 10 and cell_text.count("-") == 2:
                            try:
                                # Try to parse as date
                                parts = cell_text.split("-")
                                if len(parts) == 3 and all(p.isdigit() for p in parts):
                                    if not filing_date:
                                        filing_date = cell_text
                                    else:
                                        trade_date = cell_text
                            except:
                                pass
                    
                    # Skip if we don't have essential data
                    # Also skip if ticker looks invalid (single letter or contains colon)
                    if not ticker or len(ticker) < 2 or ":" in ticker or not trade_type or price is None:
                        continue
                    
                    # Calculate value if not found
                    if value is None and price is not None and qty is not None:
                        value = price * qty
                    
                    # Determine source section
                    source = "other"
                    if is_penny_stock:
                        source = "penny_stock_buys"
                    elif is_insider_purchases:
                        source = "insider_purchases"
                    
                    purchase = {
                        "ticker": ticker,
                        "company": company,
                        "insider_name": insider_name,
                        "title": title,
                        "trade_type": trade_type,
                        "price": price,
                        "quantity": qty,
                        "value": value,
                        "is_penny_stock": is_penny_stock,
                        "source": source,
                        "filing_date": filing_date,
                        "trade_date": trade_date,
                    }
                    
                    purchases.append(purchase)
                    
                except Exception as e:
                    continue  # Skip problematic rows
        
        print(f"✅ Found {len(purchases)} purchase transactions")
        
        # Show breakdown by source
        penny_stock_count = sum(1 for p in purchases if p.get("source") == "penny_stock_buys")
        insider_purchases_count = sum(1 for p in purchases if p.get("source") == "insider_purchases")
        other_count = len(purchases) - penny_stock_count - insider_purchases_count
        print(f"  Breakdown: {penny_stock_count} from Penny Stock Buys, {insider_purchases_count} from Insider Purchases, {other_count} from other tables")
        
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
    """Parse date string (YYYY-MM-DD) to date object."""
    if not date_str or len(date_str) != 10:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
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
    
    print(f"\n✅ Found {len(by_ticker)} unique stocks with purchases:\n")
    
    # Sort by total purchase value
    sorted_tickers = sorted(
        by_ticker.items(),
        key=lambda x: sum(p.get("value", 0) or 0 for p in x[1]),
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
    
    # Display results
    for ticker, purchases_list in sorted_tickers:
        total_value = sum(p.get("value", 0) or 0 for p in purchases_list)
        avg_purchase_price = sum(p.get("price", 0) for p in purchases_list) / len(purchases_list)
        
        print(f"{'='*60}")
        print(f"{ticker}: {len(purchases_list)} purchase(s), ${total_value:,.0f} total")
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
    
    return by_ticker


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
