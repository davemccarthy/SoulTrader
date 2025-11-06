import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import re
import yfinance as yf
import json


def scrape_fda_approvals(month=None, year=None, months_back=1):
    """
    Scrape FDA drug approvals from the FDA website.
    
    Args:
        month: Month number (1-12), defaults to current month
        year: Year (e.g., 2025), defaults to current year
        months_back: Number of months to scrape going back (default: 1)
    
    Returns:
        List of dictionaries with approval data
    """
    base_url = "https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm"
    
    # Default to current month/year if not specified
    if month is None:
        month = datetime.now().month
    if year is None:
        year = datetime.now().year
    
    approvals = []
    
    # Scrape multiple months if requested
    for i in range(months_back):
        current_month = month - i
        current_year = year
        
        # Handle year rollover
        while current_month < 1:
            current_month += 12
            current_year -= 1
        
        print(f"Scraping FDA approvals for {current_month}/{current_year}...")
        
        # Build POST request data (the form uses POST, not GET)
        form_data = {
            'rptName': '1',
            'reportSelectMonth': str(current_month),
            'reportSelectYear': str(current_year),
            'nav': '#navigation'
        }
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            # Use POST method (the form submits via POST)
            response = requests.post(
                f"{base_url}?event=reportsSearch.process",
                data=form_data,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract approvals from the table
            month_approvals = _parse_approvals_table(soup, current_month, current_year)
            approvals.extend(month_approvals)
            
            # Update month/year for next iteration
            month = current_month
            year = current_year
            
            print(f"  Found {len(month_approvals)} approvals")
            
            # Small delay between requests to be respectful
            if i < months_back - 1:
                time.sleep(1)
            
        except requests.RequestException as e:
            print(f"Error fetching FDA data for {current_month}/{current_year}: {e}")
            continue
    
    return approvals


def _parse_approvals_table(soup, month, year):
    """
    Parse the approvals table from the FDA page HTML.
    
    The table has id="example_1" with columns:
    - Approval Date
    - Drug Name (with link and application number)
    - Submission
    - Active Ingredients
    - Company
    - Submission Classification
    - Submission Status
    """
    approvals = []
    
    # Find the table with id="example_1"
    table = soup.find('table', id='example_1')
    
    if not table:
        print(f"  Warning: Table not found for {month}/{year}")
        return approvals
    
    # Find all data rows (skip header row)
    tbody = table.find('tbody')
    if not tbody:
        return approvals
    
    rows = tbody.find_all('tr')
    
    for row in rows:
        cells = row.find_all('td')
        
        if len(cells) < 7:  # Need all 7 columns
            continue
        
        # Extract data from each cell
        approval_date = cells[0].get_text(strip=True)
        
        # Drug name and application number are in the second cell
        drug_cell = cells[1]
        drug_link = drug_cell.find('a', href=True)
        drug_name = drug_link.get_text(strip=True) if drug_link else drug_cell.get_text(strip=True)
        
        # Extract application number from the link text (e.g., "NIPENT\nNDA   #020122")
        app_number_full = drug_cell.get_text(strip=True)
        app_type = None
        app_number = None
        
        # Parse application type and number (e.g., "NDA   #020122", "ANDA  #201851", "BLA   #125261")
        if 'NDA' in app_number_full:
            app_type = 'NDA'
            parts = app_number_full.split('NDA')
        elif 'ANDA' in app_number_full:
            app_type = 'ANDA'
            parts = app_number_full.split('ANDA')
        elif 'BLA' in app_number_full:
            app_type = 'BLA'
            parts = app_number_full.split('BLA')
        
        if app_type and len(parts) > 1:
            # Extract number after #
            number_part = parts[1].strip()
            if '#' in number_part:
                app_number = number_part.split('#')[1].strip()
        
        # Get the link URL
        link_url = None
        if drug_link:
            href = drug_link.get('href', '')
            if href and not href.startswith('http'):
                link_url = 'https://www.accessdata.fda.gov' + href
            else:
                link_url = href
        
        submission = cells[2].get_text(strip=True)
        active_ingredients = cells[3].get_text(strip=True)
        company = cells[4].get_text(strip=True)
        submission_classification = cells[5].get_text(strip=True)
        submission_status = cells[6].get_text(strip=True)  # "Approval" or "Tentative Approval"
        
        # Try to match company to stock symbol
        stock_symbol, match_confidence = _match_company_to_symbol(company)
        
        approval_data = {
            'approval_date': approval_date,
            'drug_name': drug_name,
            'application_type': app_type,
            'application_number': app_number,
            'submission': submission,
            'active_ingredients': active_ingredients,
            'company': company,
            'stock_symbol': stock_symbol,
            'match_confidence': match_confidence,
            'submission_classification': submission_classification,
            'status': submission_status,  # "Approval" or "Tentative Approval"
            'link': link_url,
            'month': month,
            'year': year
        }
        
        approvals.append(approval_data)
    
    return approvals


def _get_company_to_symbol_map():
    """
    Map of common pharmaceutical company names to stock symbols.
    This is a basic mapping - can be expanded as needed.
    """
    return {
        # Major pharma companies
        'PFIZER': 'PFE',
        'JOHNSON & JOHNSON': 'JNJ',
        'JANSSEN': 'JNJ',  # Janssen is J&J subsidiary
        'JANSSEN BIOTECH': 'JNJ',
        'MERCK': 'MRK',
        'MERCK & CO': 'MRK',
        'ABBV': 'ABBV',
        'ABBVIE': 'ABBV',
        'GILEAD': 'GILD',
        'GILEAD SCIENCES': 'GILD',
        'BRISTOL MYERS SQUIBB': 'BMY',
        'BMS': 'BMY',
        'AMGEN': 'AMGN',
        'ELI LILLY': 'LLY',
        'LILLY': 'LLY',
        'NOVARTIS': 'NVS',
        'ROCHE': 'RHHBY',
        'GLAXOSMITHKLINE': 'GSK',
        'GSK': 'GSK',
        'SANOFI': 'SNY',
        'ASTRAZENECA': 'AZN',
        'TEVA': 'TEVA',
        'TEVA PHARMACEUTICAL': 'TEVA',
        'TEVA PHARMS USA': 'TEVA',
        'TEVA PHARMACEUTICALS': 'TEVA',
        'MYLAN': 'MYL',  # Now part of Viatris
        'VIATRIS': 'VTRS',
        'SUN PHARMA': 'SUNPHARMA',
        'SUN PHARMACEUTICAL': 'SUNPHARMA',
        'DR REDDYS': 'RDY',
        'DR. REDDYS': 'RDY',
        'DR REDDYS LABORATORIES': 'RDY',
        'AUROBINDO': 'AUPH',  # Note: Aurobindo Pharma Ltd is private, but AUPH exists
        'AUROBINDO PHARMA': 'AUPH',
        'AUROBINDO PHARMA LTD': 'AUPH',
        'HOSPIRA': 'HSP',  # Acquired by Pfizer
        'HOSPIRA INC': 'HSP',  # Now part of Pfizer
        'CENTOCOR': 'JNJ',  # Now part of J&J
        'CENTOCOR ORTHO BIOTECH': 'JNJ',
        'PHARMACIA': 'PFE',  # Now part of Pfizer
        'PHARMACIA AND UPJOHN': 'PFE',
        'UPJOHN': 'PFE',
        'SCYNEXIS': 'SCYX',
        'UCB': 'UCB',
        'UCB INC': 'UCB',
        'AMNEAL': 'AMRX',
        'AMNEAL PHARMACEUTICALS': 'AMRX',
        'AMNEAL PHARMS': 'AMRX',
        'SAGENT': 'SGNT',
        'SAGENT PHARMS': 'SGNT',
        'SAGENT PHARMACEUTICALS': 'SGNT',
        'BRECKENRIDGE': 'BRX',  # Breckenridge Pharmaceutical
        'MSN': 'MSN',  # MSN Laboratories
        'MSN LABORATORIES': 'MSN',
        'MSN LABORATORIES PRIVATE LTD': 'MSN',
        # Add more as needed
    }


def _normalize_company_name(company_name):
    """
    Normalize company name for matching (remove common suffixes, uppercase, etc.)
    """
    if not company_name:
        return ""
    
    # Convert to uppercase
    normalized = company_name.upper().strip()
    
    # Remove common suffixes
    suffixes = [' INC', ' INC.', ' CORPORATION', ' CORP', ' CORP.', ' LTD', ' LTD.', 
                ' LIMITED', ' LLC', ' LLC.', ' PHARMACEUTICALS', ' PHARMA', 
                ' PHARMACEUTICAL', ' PHARMS', ' LABORATORIES', ' LABS', ' COMPANY', ' CO']
    
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].strip()
    
    return normalized


def _match_company_to_symbol(company_name):
    """
    Match a company name to a stock symbol.
    
    Returns:
        Tuple of (symbol, confidence) where confidence is 'exact', 'normalized', or None
    """
    if not company_name:
        return None, None
    
    company_map = _get_company_to_symbol_map()
    
    # Try exact match first
    company_upper = company_name.upper().strip()
    if company_upper in company_map:
        return company_map[company_upper], 'exact'
    
    # Try normalized match
    normalized = _normalize_company_name(company_name)
    if normalized in company_map:
        return company_map[normalized], 'normalized'
    
    # Try partial matching (contains)
    for key, symbol in company_map.items():
        if key in company_upper or company_upper in key:
            return symbol, 'partial'
    
    return None, None


def analyze_historical_prices(approvals, days_later=7):
    """
    Analyze stock price changes after tentative approvals.
    
    Args:
        approvals: List of approval dictionaries
        days_later: Number of days after approval to check price (default: 7)
    
    Returns:
        List of dictionaries with price analysis data
    """
    # Filter for tentative approvals with stock symbols
    tentative_with_symbols = [
        a for a in approvals 
        if a['status'] == 'Tentative Approval' and a.get('stock_symbol')
    ]
    
    if not tentative_with_symbols:
        print("No tentative approvals with stock symbols found.")
        return []
    
    print(f"\nAnalyzing {len(tentative_with_symbols)} tentative approvals...")
    print("Fetching stock prices (this may take a moment)...\n")
    
    results = []
    
    for approval in tentative_with_symbols:
        symbol = approval['stock_symbol']
        approval_date_str = approval['approval_date']
        
        # Parse approval date (format: MM/DD/YYYY)
        try:
            approval_date = datetime.strptime(approval_date_str, '%m/%d/%Y').date()
        except ValueError:
            print(f"  Warning: Could not parse date '{approval_date_str}' for {symbol}")
            continue
        
        # Calculate target date (1 week later)
        target_date = approval_date + timedelta(days=days_later)
        
        # Get stock data
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data around the approval date
            # Fetch a bit more data to ensure we have both dates
            start_date = approval_date - timedelta(days=5)
            end_date = target_date + timedelta(days=5)
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                print(f"  {symbol}: No price data available")
                continue
            
            # Get price on approval date (use closest available date)
            approval_price = None
            approval_actual_date = None
            for date in hist.index:
                if date.date() >= approval_date:
                    approval_price = hist.loc[date, 'Close']
                    approval_actual_date = date.date()
                    break
            
            # Get price 1 week later (use closest available date)
            target_price = None
            target_actual_date = None
            for date in hist.index:
                if date.date() >= target_date:
                    target_price = hist.loc[date, 'Close']
                    target_actual_date = date.date()
                    break
            
            if approval_price is None or target_price is None:
                print(f"  {symbol}: Missing price data")
                continue
            
            # Calculate change
            price_change = target_price - approval_price
            price_change_pct = (price_change / approval_price) * 100
            
            analysis = {
                'symbol': symbol,
                'company': approval['company'],
                'drug_name': approval['drug_name'],
                'approval_date': approval_date_str,
                'approval_actual_date': approval_actual_date.strftime('%Y-%m-%d'),
                'approval_price': round(approval_price, 2),
                'target_date': target_date.strftime('%Y-%m-%d'),
                'target_actual_date': target_actual_date.strftime('%Y-%m-%d'),
                'target_price': round(target_price, 2),
                'price_change': round(price_change, 2),
                'price_change_pct': round(price_change_pct, 2),
                'application_number': approval['application_number'],
                'application_type': approval['application_type'],
                'submission': approval.get('submission', ''),  # Include submission type (ORIG-1, SUPPL-XX, etc.)
                'submission_classification': approval.get('submission_classification', '')
            }
            
            results.append(analysis)
            
            # Show progress
            change_indicator = "↑" if price_change_pct > 0 else "↓"
            print(f"  {symbol}: ${approval_price:.2f} → ${target_price:.2f} ({change_indicator}{abs(price_change_pct):.2f}%)")
            
            # Small delay to be respectful to API
            time.sleep(0.2)
            
        except Exception as e:
            print(f"  {symbol}: Error - {e}")
            continue
    
    return results


def save_analysis_to_file(results, filename=None):
    """
    Save analysis results to a JSON file for later review.
    
    Args:
        results: List of analysis dictionaries
        filename: Optional filename (defaults to timestamped filename)
    """
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'fda_analysis_{timestamp}.json'
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis saved to: {filename}")
        return filename
    except Exception as e:
        print(f"\nError saving analysis: {e}")
        return None


def print_price_analysis_report(results):
    """
    Print a formatted analysis report of stock price changes.
    """
    if not results:
        print("\nNo price data available for analysis.")
        return
    
    print("\n" + "=" * 80)
    print("STOCK PRICE ANALYSIS - 1 WEEK AFTER TENTATIVE APPROVAL")
    print("=" * 80)
    
    # Sort by price change percentage (highest gains first)
    sorted_results = sorted(results, key=lambda x: x['price_change_pct'], reverse=True)
    
    # Calculate statistics
    gains = [r for r in results if r['price_change_pct'] > 0]
    losses = [r for r in results if r['price_change_pct'] < 0]
    flat = [r for r in results if r['price_change_pct'] == 0]
    
    avg_change = sum(r['price_change_pct'] for r in results) / len(results)
    avg_gain = sum(r['price_change_pct'] for r in gains) / len(gains) if gains else 0
    avg_loss = sum(r['price_change_pct'] for r in losses) / len(losses) if losses else 0
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"  Total analyzed: {len(results)}")
    print(f"  Gains: {len(gains)} ({len(gains)/len(results)*100:.1f}%)")
    print(f"  Losses: {len(losses)} ({len(losses)/len(results)*100:.1f}%)")
    print(f"  Flat: {len(flat)}")
    print(f"  Average change: {avg_change:+.2f}%")
    if gains:
        print(f"  Average gain: +{avg_gain:.2f}%")
    if losses:
        print(f"  Average loss: {avg_loss:.2f}%")
    
    print(f"\nTOP GAINERS (1 week after tentative approval):")
    print("-" * 80)
    for i, result in enumerate(sorted_results[:10], 1):
        if result['price_change_pct'] > 0:
            print(f"{i}. {result['symbol']} ({result['company']})")
            print(f"   Drug: {result['drug_name']}")
            print(f"   Approval: {result['approval_date']} @ ${result['approval_price']:.2f}")
            print(f"   Week later: {result['target_actual_date']} @ ${result['target_price']:.2f}")
            print(f"   Change: +${result['price_change']:.2f} (+{result['price_change_pct']:.2f}%)")
            print()
    
    print(f"\nTOP DECLINERS (1 week after tentative approval):")
    print("-" * 80)
    for i, result in enumerate(sorted_results[-10:], 1):
        if result['price_change_pct'] < 0:
            print(f"{i}. {result['symbol']} ({result['company']})")
            print(f"   Drug: {result['drug_name']}")
            print(f"   Approval: {result['approval_date']} @ ${result['approval_price']:.2f}")
            print(f"   Week later: {result['target_actual_date']} @ ${result['target_price']:.2f}")
            print(f"   Change: ${result['price_change']:.2f} ({result['price_change_pct']:.2f}%)")
            print()
    
    # Show TEVA specifically if present
    teva_results = [r for r in results if r['symbol'] == 'TEVA']
    if teva_results:
        print("\nTEVA SPECIFIC ANALYSIS:")
        print("-" * 80)
        for result in teva_results:
            print(f"  {result['drug_name']} ({result['application_type']} #{result['application_number']})")
            print(f"  Approval: {result['approval_date']} @ ${result['approval_price']:.2f}")
            print(f"  Week later: {result['target_actual_date']} @ ${result['target_price']:.2f}")
            print(f"  Change: {result['price_change']:+.2f} ({result['price_change_pct']:+.2f}%)")
            print()


def find_previous_month_link(soup):
    """
    Find the link to go to the previous month.
    
    Returns:
        Dictionary with month and year, or None if not found
    """
    # Look for the "Previous Month" link
    # Format: /scripts/cder/daf/index.cfm?event=reportsSearch.process&rptName=1&reportSelectMonth=10&reportSelectYear=2025&nav#navigation
    links = soup.find_all('a', href=True)
    
    for link in links:
        href = link.get('href', '')
        text = link.get_text(strip=True).lower()
        
        if 'previous month' in text and 'reportSelectMonth' in href:
            # Parse month and year from href
            try:
                if 'reportSelectMonth=' in href and 'reportSelectYear=' in href:
                    month_part = href.split('reportSelectMonth=')[1].split('&')[0]
                    year_part = href.split('reportSelectYear=')[1].split('&')[0]
                    return {'month': int(month_part), 'year': int(year_part)}
            except (ValueError, IndexError):
                pass
    
    return None


if __name__ == "__main__":
    import sys
    
    # Test scraping current month
    print("Testing FDA approval scraper...")
    print("=" * 70)
    
    # Allow command line argument for months_back
    months_back = 1
    if len(sys.argv) > 1:
        try:
            months_back = int(sys.argv[1])
            print(f"Scraping {months_back} month(s) back...\n")
        except ValueError:
            print(f"Invalid argument '{sys.argv[1]}'. Using default: 1 month back.\n")
    
    approvals = scrape_fda_approvals(months_back=months_back)
    
    if approvals:
        print(f"\nFound {len(approvals)} total approvals:\n")
        
        # Show tentative approvals first (they're interesting!)
        tentative = [a for a in approvals if a['status'] == 'Tentative Approval']
        final = [a for a in approvals if a['status'] == 'Approval']
        
        if tentative:
            print(f"TENTATIVE APPROVALS ({len(tentative)}):")
            print("-" * 70)
            for i, approval in enumerate(tentative, 1):
                print(f"{i}. {approval['drug_name']}")
                print(f"   Company: {approval['company']}", end="")
                if approval.get('stock_symbol'):
                    print(f" [{approval['stock_symbol']}]", end="")
                    if approval.get('match_confidence'):
                        print(f" ({approval['match_confidence']} match)", end="")
                print()
                print(f"   Date: {approval['approval_date']}")
                print(f"   Type: {approval['application_type']} #{approval['application_number']}")
                print(f"   Status: {approval['status']}")
                if approval.get('link'):
                    print(f"   Link: {approval['link']}")
                print()
        
        if final:
            print(f"\nFINAL APPROVALS ({len(final)}):")
            print("-" * 70)
            for i, approval in enumerate(final[:10], 1):  # Show first 10
                print(f"{i}. {approval['drug_name']}")
                print(f"   Company: {approval['company']}", end="")
                if approval.get('stock_symbol'):
                    print(f" [{approval['stock_symbol']}]", end="")
                    if approval.get('match_confidence'):
                        print(f" ({approval['match_confidence']} match)", end="")
                print()
                print(f"   Date: {approval['approval_date']}")
                print(f"   Type: {approval['application_type']} #{approval['application_number']}")
                if approval.get('submission_classification'):
                    print(f"   Classification: {approval['submission_classification']}")
                print()
            
            if len(final) > 10:
                print(f"  ... and {len(final) - 10} more final approvals")
    else:
        print("No approvals found.")
    
    # Show summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  Total approvals: {len(approvals)}")
    tentative_count = len([a for a in approvals if a['status'] == 'Tentative Approval'])
    final_count = len([a for a in approvals if a['status'] == 'Approval'])
    with_symbols = len([a for a in approvals if a.get('stock_symbol')])
    print(f"  - Tentative: {tentative_count}")
    print(f"  - Final: {final_count}")
    print(f"  - With stock symbols: {with_symbols}")
    
    # Run historical price analysis if requested
    if len(sys.argv) > 2 and sys.argv[2] == '--analyze':
        tentative_with_symbols = [
            a for a in approvals 
            if a['status'] == 'Tentative Approval' and a.get('stock_symbol')
        ]
        
        if tentative_with_symbols:
            results = analyze_historical_prices(approvals, days_later=7)
            if results:
                print_price_analysis_report(results)
                # Save results to file for later analysis
                save_analysis_to_file(results)
        else:
            print("\nNo tentative approvals with stock symbols to analyze.")
    
    print(f"\nUsage: python test_fda.py [months_back] [--analyze]")
    print(f"  Example: python test_fda.py 6  (scrapes last 6 months)")
    print(f"  Example: python test_fda.py 6 --analyze  (scrapes + analyzes prices)")
