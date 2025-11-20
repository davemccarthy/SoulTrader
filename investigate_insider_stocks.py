#!/usr/bin/env python
"""
Investigate specific stocks on openinsider.com to find clues about why they fell.

Usage:
    python investigate_insider_stocks.py STRZ RCG SRTS AISP
"""

import sys
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime, date
from collections import defaultdict

BASE_URL = "http://openinsider.com"


def parse_date(date_str: str) -> Optional[date]:
    """Parse date string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS) to date object."""
    if not date_str:
        return None
    if len(date_str) >= 10:
        try:
            date_part = date_str[:10]
            return datetime.strptime(date_part, "%Y-%m-%d").date()
        except:
            pass
    return None


def scrape_ticker_transactions(ticker: str) -> Dict:
    """
    Scrape all insider transactions for a specific ticker from openinsider.com.
    
    Returns:
        Dictionary with 'purchases', 'sales', 'red_flags', and 'summary'
    """
    url = f"{BASE_URL}/{ticker}"
    result = {
        'ticker': ticker,
        'url': url,
        'purchases': [],
        'sales': [],
        'red_flags': [],
        'summary': {}
    }
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        print(f"\nFetching {url}...")
        resp = requests.get(url, headers=headers, timeout=15)
        
        if not resp.ok:
            result['red_flags'].append(f"Failed to fetch page: HTTP {resp.status_code}")
            return result
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find all tables with transaction data
        tables = soup.find_all("table")
        
        # Debug: print table count
        print(f"  Found {len(tables)} table(s)")
        
        for table in tables:
            rows = table.find_all("tr")
            
            # Look for header row - try multiple patterns
            header_idx = -1
            for i, row in enumerate(rows):
                row_text = row.get_text().lower()
                if any(keyword in row_text for keyword in ["filing date", "trade date", "transaction", "insider", "price", "quantity"]):
                    header_idx = i
                    break
            
            if header_idx < 0:
                # Try to find first row with data (has links or numbers)
                for i, row in enumerate(rows):
                    row_text = row.get_text()
                    # Check if row has ticker link or transaction data
                    links = row.find_all("a", href=True)
                    has_ticker_link = any(href.get("href", "").startswith(f"/{ticker}") for href in links)
                    has_numbers = any(c.isdigit() for c in row_text)
                    if has_ticker_link or (has_numbers and "$" in row_text):
                        header_idx = i - 1 if i > 0 else 0
                        break
                
                if header_idx < 0:
                    continue
            
            # Process data rows
            for row in rows[header_idx + 1:]:
                cells = row.find_all(["td", "th"])
                if len(cells) < 5:
                    continue
                
                try:
                    # Extract transaction data
                    transaction = {}
                    
                    # Find ticker (should match)
                    for cell in cells:
                        links = cell.find_all("a", href=True)
                        for link in links:
                            href = link.get("href", "")
                            if href.startswith("/") and len(href) > 1:
                                potential_ticker = href[1:].split("/")[0].upper()
                                if potential_ticker == ticker:
                                    transaction['ticker'] = ticker
                                    break
                    
                    # Find trade type (P - Purchase, S - Sale, etc.)
                    row_text = " ".join([c.text.strip() for c in cells])
                    row_text_lower = row_text.lower()
                    
                    # Check for purchase indicators
                    if any(indicator in row_text_lower for indicator in ["p - purchase", "purchase", "p -", "buy"]):
                        transaction['type'] = 'Purchase'
                    # Check for sale indicators
                    elif any(indicator in row_text_lower for indicator in ["s - sale", "sale", "s -", "sell"]):
                        transaction['type'] = 'Sale'
                    else:
                        # If we can't determine type, skip unless we have other strong indicators
                        # Sometimes the type is in a specific cell
                        for cell in cells:
                            cell_text = cell.text.strip().upper()
                            if "P" == cell_text or "PURCHASE" in cell_text:
                                transaction['type'] = 'Purchase'
                                break
                            elif "S" == cell_text or "SALE" in cell_text:
                                transaction['type'] = 'Sale'
                                break
                        
                        if 'type' not in transaction:
                            continue
                    
                    # Extract price, quantity, value
                    for cell in cells:
                        cell_text = cell.text.strip()
                        cell_text_clean = cell_text.replace("$", "").replace(",", "").replace("+", "").replace("-", "")
                        
                        # Price - look for dollar amounts
                        if '$' in cell_text and 'price' not in transaction:
                            try:
                                # Extract number after $
                                price_match = cell_text.replace(",", "")
                                if price_match.startswith("$"):
                                    price_val = float(price_match[1:].split()[0] if " " in price_match[1:] else price_match[1:])
                                    if price_val > 0:
                                        transaction['price'] = price_val
                            except:
                                pass
                        
                        # Quantity - look for numbers with + or - prefix, or just large numbers
                        if 'quantity' not in transaction:
                            if (cell_text.startswith("+") or cell_text.startswith("-")) and any(c.isdigit() for c in cell_text):
                                try:
                                    qty_str = cell_text.replace("+", "").replace("-", "").replace(",", "")
                                    transaction['quantity'] = int(qty_str)
                                    if cell_text.startswith("-"):
                                        transaction['quantity'] = -transaction['quantity']
                                except:
                                    pass
                            # Also check for large numbers that might be quantities (no $ sign)
                            elif not cell_text.startswith("$") and cell_text_clean.replace(".", "").isdigit():
                                try:
                                    potential_qty = int(float(cell_text_clean))
                                    # If it's a large number (likely shares, not price), it might be quantity
                                    if potential_qty > 100 and potential_qty < 100000000:  # Reasonable share range
                                        transaction['quantity'] = potential_qty
                                except:
                                    pass
                        
                        # Value - look for +$ or -$ patterns
                        if cell_text.startswith("+$") or cell_text.startswith("-$"):
                            try:
                                value_str = cell_text.replace("+$", "").replace("-$", "").replace(",", "")
                                transaction['value'] = float(value_str)
                                if cell_text.startswith("-$"):
                                    transaction['value'] = -transaction['value']
                            except:
                                pass
                        
                        # Insider name and title
                        links = cell.find_all("a", href=True)
                        for link in links:
                            href = link.get("href", "")
                            if "/insider/" in href:
                                transaction['insider_name'] = link.text.strip()
                                break
                        
                        # Title
                        cell_text_upper = cell_text.upper()
                        if not transaction.get('title') or len(cell_text) < 50:
                            if "CEO" in cell_text_upper:
                                transaction['title'] = "CEO"
                            elif "CFO" in cell_text_upper:
                                transaction['title'] = "CFO"
                            elif "COO" in cell_text_upper:
                                transaction['title'] = "COO"
                            elif "PRESIDENT" in cell_text_upper:
                                transaction['title'] = "President"
                            elif "DIRECTOR" in cell_text_upper:
                                transaction['title'] = "Director"
                            elif "VP" in cell_text_upper or "VICE PRESIDENT" in cell_text_upper:
                                transaction['title'] = "VP"
                            elif "OFFICER" in cell_text_upper:
                                transaction['title'] = "Officer"
                        
                        # Dates
                        if cell_text.count("-") >= 2 and len(cell_text) >= 10:
                            try:
                                parsed_date = parse_date(cell_text)
                                if parsed_date:
                                    if cell.find("a") and not transaction.get('filing_date'):
                                        transaction['filing_date'] = cell_text[:10] if len(cell_text) > 10 else cell_text
                                        transaction['filing_date_obj'] = parsed_date
                                    elif not transaction.get('trade_date'):
                                        transaction['trade_date'] = cell_text[:10] if len(cell_text) > 10 else cell_text
                                        transaction['trade_date_obj'] = parsed_date
                            except:
                                pass
                    
                    # Calculate value if missing
                    if 'value' not in transaction and 'price' in transaction and 'quantity' in transaction:
                        transaction['value'] = transaction['price'] * abs(transaction['quantity'])
                    
                    # Add to appropriate list
                    if transaction['type'] == 'Purchase':
                        result['purchases'].append(transaction)
                    elif transaction['type'] == 'Sale':
                        result['sales'].append(transaction)
                
                except Exception as e:
                    continue
        
        # Analyze for red flags
        analyze_red_flags(result)
        
    except Exception as e:
        result['red_flags'].append(f"Error scraping: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def analyze_red_flags(result: Dict):
    """Analyze transactions for red flags."""
    purchases = result['purchases']
    sales = result['sales']
    
    # Sort by date (most recent first)
    purchases.sort(key=lambda x: x.get('filing_date_obj') or x.get('trade_date_obj') or date.min, reverse=True)
    sales.sort(key=lambda x: x.get('filing_date_obj') or x.get('trade_date_obj') or date.min, reverse=True)
    
    # Red flag 1: Sales after recent purchases
    if purchases and sales:
        latest_purchase_date = purchases[0].get('filing_date_obj') or purchases[0].get('trade_date_obj')
        if latest_purchase_date:
            recent_sales = [s for s in sales 
                           if (s.get('filing_date_obj') or s.get('trade_date_obj')) and 
                           (s.get('filing_date_obj') or s.get('trade_date_obj')) >= latest_purchase_date]
            if recent_sales:
                result['red_flags'].append(
                    f"⚠️  {len(recent_sales)} sale(s) occurred after latest purchase on {latest_purchase_date}"
                )
    
    # Red flag 2: Small purchase amounts
    if purchases:
        small_purchases = [p for p in purchases if p.get('value', 0) < 10000]
        if small_purchases:
            result['red_flags'].append(
                f"⚠️  {len(small_purchases)} purchase(s) under $10k (low conviction)"
            )
    
    # Red flag 3: Low-level insiders
    if purchases:
        low_level = [p for p in purchases if p.get('title') not in ['CEO', 'CFO', 'COO', 'President']]
        if low_level:
            result['red_flags'].append(
                f"⚠️  {len(low_level)} purchase(s) by non-C-level insiders"
            )
    
    # Red flag 4: More sales than purchases
    total_purchase_value = sum(p.get('value', 0) for p in purchases)
    total_sale_value = sum(abs(s.get('value', 0)) for s in sales)
    if total_sale_value > total_purchase_value and purchases:
        result['red_flags'].append(
            f"⚠️  Net selling: ${total_sale_value:,.0f} sold vs ${total_purchase_value:,.0f} bought"
        )
    
    # Red flag 5: Old purchases
    if purchases:
        today = date.today()
        old_purchases = [p for p in purchases 
                        if (p.get('filing_date_obj') or p.get('trade_date_obj')) and
                        (today - (p.get('filing_date_obj') or p.get('trade_date_obj'))).days > 30]
        if old_purchases and len(old_purchases) == len(purchases):
            result['red_flags'].append(
                f"⚠️  All purchases are over 30 days old (stale signal)"
            )
    
    # Summary
    result['summary'] = {
        'total_purchases': len(purchases),
        'total_sales': len(sales),
        'total_purchase_value': total_purchase_value,
        'total_sale_value': total_sale_value,
        'latest_purchase_date': purchases[0].get('filing_date_obj') or purchases[0].get('trade_date_obj') if purchases else None,
        'latest_sale_date': sales[0].get('filing_date_obj') or sales[0].get('trade_date_obj') if sales else None,
    }


def print_investigation_report(result: Dict):
    """Print a formatted investigation report."""
    ticker = result['ticker']
    print(f"\n{'='*80}")
    print(f"INVESTIGATION: {ticker}")
    print(f"{'='*80}")
    print(f"URL: {result['url']}")
    
    summary = result['summary']
    print(f"\nSUMMARY:")
    print(f"  Total Purchases: {summary.get('total_purchases', 0)}")
    print(f"  Total Sales: {summary.get('total_sales', 0)}")
    print(f"  Total Purchase Value: ${summary.get('total_purchase_value', 0):,.0f}")
    print(f"  Total Sale Value: ${summary.get('total_sale_value', 0):,.0f}")
    if summary.get('latest_purchase_date'):
        print(f"  Latest Purchase: {summary['latest_purchase_date']}")
    if summary.get('latest_sale_date'):
        print(f"  Latest Sale: {summary['latest_sale_date']}")
    
    # Red flags
    if result['red_flags']:
        print(f"\n🚨 RED FLAGS:")
        for flag in result['red_flags']:
            print(f"  {flag}")
    else:
        print(f"\n✓ No obvious red flags detected")
    
    # Recent purchases
    if result['purchases']:
        print(f"\nRECENT PURCHASES (last 5):")
        print(f"  {'Date':<12} {'Insider':<25} {'Title':<12} {'Price':>8} {'Qty':>12} {'Value':>12}")
        print(f"  {'-'*12} {'-'*25} {'-'*12} {'-'*8} {'-'*12} {'-'*12}")
        for p in result['purchases'][:5]:
            date_str = (p.get('filing_date') or p.get('trade_date') or 'N/A')[:10]
            insider = (p.get('insider_name') or 'Unknown')[:24]
            title = (p.get('title') or 'N/A')[:11]
            price = p.get('price', 0)
            qty = p.get('quantity', 0)
            value = p.get('value', 0)
            print(f"  {date_str:<12} {insider:<25} {title:<12} ${price:>7.2f} {qty:>12,} ${value:>11,.0f}")
    
    # Recent sales
    if result['sales']:
        print(f"\nRECENT SALES (last 5):")
        print(f"  {'Date':<12} {'Insider':<25} {'Title':<12} {'Price':>8} {'Qty':>12} {'Value':>12}")
        print(f"  {'-'*12} {'-'*25} {'-'*12} {'-'*8} {'-'*12} {'-'*12}")
        for s in result['sales'][:5]:
            date_str = (s.get('filing_date') or s.get('trade_date') or 'N/A')[:10]
            insider = (s.get('insider_name') or 'Unknown')[:24]
            title = (s.get('title') or 'N/A')[:11]
            price = s.get('price', 0)
            qty = abs(s.get('quantity', 0))
            value = abs(s.get('value', 0))
            print(f"  {date_str:<12} {insider:<25} {title:<12} ${price:>7.2f} {qty:>12,} ${value:>11,.0f}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python investigate_insider_stocks.py TICKER1 TICKER2 ...")
        print("Example: python investigate_insider_stocks.py STRZ RCG SRTS AISP")
        sys.exit(1)
    
    tickers = [t.upper() for t in sys.argv[1:]]
    
    print("="*80)
    print("INSIDER TRANSACTION INVESTIGATION")
    print("="*80)
    print(f"Investigating {len(tickers)} ticker(s): {', '.join(tickers)}")
    
    results = []
    for ticker in tickers:
        result = scrape_ticker_transactions(ticker)
        results.append(result)
        print_investigation_report(result)
    
    # Summary across all tickers
    print(f"\n{'='*80}")
    print("CROSS-TICKER SUMMARY")
    print(f"{'='*80}")
    
    all_red_flags = []
    for result in results:
        if result['red_flags']:
            all_red_flags.append(f"{result['ticker']}: {len(result['red_flags'])} red flag(s)")
    
    if all_red_flags:
        print("\nTickers with red flags:")
        for flag in all_red_flags:
            print(f"  {flag}")
    else:
        print("\n✓ No red flags across all tickers")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

