import json
import os
import re
import time
from datetime import datetime, timedelta

import requests
import yfinance as yf
from bs4 import BeautifulSoup


def scrape_fda_approvals(days=7):
    """
    Scrape recent FDA drug approvals from the Drugs@FDA report page.

    Args:
        days: Either 7 (default) or 14, corresponding to the report tabs
              available on the public page.

    Returns:
        List of dictionaries with approval data
    """
    if days not in (7, 14):
        raise ValueError("days must be either 7 or 14 to match available FDA reports")

    report_url = "https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=report.page"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    }

    try:
        response = requests.get(report_url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error fetching FDA report page: {exc}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    tab_id = 'example2-tab1' if days <= 7 else 'example2-tab2'
    tab_content = soup.find('div', id=tab_id)

    if not tab_content:
        print(f"  Warning: Could not locate the {days}-day report tab on the FDA page.")
        return []

    approvals = []
    headings = tab_content.find_all('h4')

    if not headings:
        print(f"  Warning: No report sections found in the {days}-day tab.")
        return []

    for heading in headings:
        report_date_str = heading.get_text(strip=True)
        table = heading.find_next('table')
        if not table:
            continue

        approvals.extend(_parse_report_table(table, report_date_str))

    return approvals


def _parse_report_table(table, report_date_str):
    """
    Parse a single report table (7-day or 14-day section) into approval dicts.
    """
    approvals = []

    try:
        report_date = datetime.strptime(report_date_str, '%B %d, %Y').date()
    except ValueError:
        report_date = None

    tbody = table.find('tbody')
    if not tbody:
        return approvals

    rows = tbody.find_all('tr')

    for row in rows:
        cells = row.find_all('td')

        if len(cells) < 7:
            continue

        drug_name, app_type, app_number, link_url = _extract_drug_info(cells[0])
        active_ingredients = cells[1].get_text(" ", strip=True)
        dosage_form = cells[2].get_text(" ", strip=True)
        submission = cells[3].get_text(strip=True)
        company = cells[4].get_text(strip=True)
        submission_classification = cells[5].get_text(strip=True)
        submission_status = cells[6].get_text(strip=True)
        normalized_status = (
            'Approval' if submission_status.lower() == 'approved' else submission_status
        )

        classification_upper = submission_classification.upper() if submission_classification else ''
        if classification_upper and 'LABELING' in classification_upper:
            has_other_signals = any(
                keyword in classification_upper
                for keyword in (
                    'EFFICACY',
                    'PRIOR APPROVAL',
                    'MANUFACTURING',
                    'CMC',
                    'BIOEQUIVALENCE',
                    'PEDIATRIC',
                    'SAFETY',
                    'REMS',
                )
            )
            if not has_other_signals:
                continue

        stock_symbol, match_confidence = _match_company_to_symbol(company)

        market_snapshot = _get_market_snapshot(stock_symbol) if stock_symbol else None
        market_cap = market_snapshot.get("market_cap") if market_snapshot else None
        market_price = market_snapshot.get("price") if market_snapshot else None

        approval_date_str = (
            report_date.strftime('%m/%d/%Y') if report_date else report_date_str
        )

        approval_data = {
            'approval_date': approval_date_str,
            'report_date': report_date_str,
            'drug_name': drug_name,
            'application_type': app_type,
            'application_number': app_number,
            'dosage_form': dosage_form,
            'submission': submission,
            'active_ingredients': active_ingredients,
            'company': company,
            'stock_symbol': stock_symbol,
            'match_confidence': match_confidence,
            'market_cap': market_cap,
            'market_price': market_price,
            'submission_classification': submission_classification,
            'status': normalized_status,
            'raw_status': submission_status,
            'link': link_url,
            'month': report_date.month if report_date else None,
            'year': report_date.year if report_date else None,
        }

        approval_data['confidence_score'] = _calculate_confidence_score(approval_data)

        approvals.append(approval_data)

    return approvals


def _extract_drug_info(drug_cell):
    """
    Extract the drug name, application type/number, and link from the cell.
    """
    text = drug_cell.get_text(separator='\n', strip=True)
    parts = [part.strip() for part in text.split('\n') if part.strip()]

    drug_name = parts[0] if parts else ''
    app_info = ' '.join(parts[1:]) if len(parts) > 1 else ''

    app_type = None
    app_number = None

    match = re.search(r'(NDA|ANDA|BLA)\s*#?\s*([0-9]+)', app_info, re.IGNORECASE)
    if match:
        app_type = match.group(1).upper()
        app_number = match.group(2).strip()

    link_url = None
    drug_link = drug_cell.find('a', href=True)
    if drug_link:
        href = drug_link.get('href', '')
        if href and not href.startswith('http'):
            link_url = 'https://www.accessdata.fda.gov' + href
        else:
            link_url = href

    return drug_name, app_type, app_number, link_url


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
        Tuple of (symbol, confidence) where confidence is one of:
        'exact', 'normalized', 'partial', 'openfigi', or None.
    """
    if not company_name:
        return None, None
    
    company_map = _get_company_to_symbol_map()
    
    # Try exact match first
    company_upper = company_name.upper().strip()
    
    if 'PRIVATE LTD' in company_upper or 'PRIVATE LIMITED' in company_upper:
        return None, None
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
    
    # Fallback to OpenFIGI lookup if an API key is available
    openfigi_symbol = _lookup_symbol_via_openfigi(company_name)
    if openfigi_symbol:
        return openfigi_symbol, 'openfigi'
    
    return None, None


# --- OpenFIGI integration --------------------------------------------------

_OPENFIGI_ENDPOINT = "https://api.openfigi.com/v3/mapping"
_OPENFIGI_CACHE = {}
_MARKET_DATA_CACHE = {}


def _lookup_symbol_via_openfigi(company_name, api_key=None):
    """
    Use the OpenFIGI mapping API to resolve a company name to a ticker.
    Results are cached per process to respect rate limits.
    """
    if not company_name:
        return None

    cache_key = company_name.strip().upper()
    if cache_key in _OPENFIGI_CACHE:
        return _OPENFIGI_CACHE[cache_key]

    api_key = api_key or os.environ.get("OPENFIGI_API_KEY")
    if not api_key:
        _OPENFIGI_CACHE[cache_key] = None
        return None

    payload = [{
        "idType": "NAME",
        "idValue": company_name,
        "securityType2": "Common Stock",
    }]

    headers = {
        "Content-Type": "application/json",
        "X-OPENFIGI-APIKEY": api_key,
    }

    try:
        response = requests.post(_OPENFIGI_ENDPOINT, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  OpenFIGI lookup failed for '{company_name}': {exc}")
        _OPENFIGI_CACHE[cache_key] = None
        return None

    try:
        results = response.json()
    except ValueError:
        print(f"  OpenFIGI returned invalid JSON for '{company_name}'")
        _OPENFIGI_CACHE[cache_key] = None
        return None

    if not isinstance(results, list) or not results:
        _OPENFIGI_CACHE[cache_key] = None
        return None

    mapping_result = results[0] or {}
    data_entries = mapping_result.get("data") or []

    ticker = _select_best_openfigi_ticker(data_entries)
    _OPENFIGI_CACHE[cache_key] = ticker
    return ticker


def _select_best_openfigi_ticker(entries):
    """
    Return the most appropriate ticker from an OpenFIGI response.
    Prefers US-listed common stock tickers with an explicit exchange code.
    """
    if not entries:
        return None

    preferred_exchanges = {"XNYS", "XNAS", "ARCX", "BATS"}
    for entry in entries:
        ticker = entry.get("ticker")
        if not ticker:
            continue
        security_type = (entry.get("securityType2") or "").lower()
        if entry.get("exchCode") in preferred_exchanges and security_type.startswith("common stock"):
            return ticker

    for entry in entries:
        ticker = entry.get("ticker")
        if not ticker:
            continue
        security_type = (entry.get("securityType2") or "").lower()
        if "stock" in security_type or "equity" in security_type:
            return ticker

    for entry in entries:
        ticker = entry.get("ticker")
        if ticker:
            return ticker

    return None


def _get_market_snapshot(symbol):
    """
    Fetch market cap and current price for a symbol using yfinance.
    Results cached per run to limit network calls.
    """
    if not symbol:
        return None

    cache_key = symbol.upper()
    if cache_key in _MARKET_DATA_CACHE:
        return _MARKET_DATA_CACHE[cache_key]

    try:
        ticker = yf.Ticker(symbol)
        market_cap = None
        price = None

        # fast_info is lightweight if available
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            market_cap = getattr(fast_info, "market_cap", None)
            price = getattr(fast_info, "last_price", None)

        if market_cap is None or price is None:
            info = ticker.info
            if market_cap is None:
                market_cap = info.get("marketCap")
            if price is None:
                price = info.get("regularMarketPrice") or info.get("currentPrice")

        snapshot = {
            "market_cap": market_cap,
            "price": price,
        }
        _MARKET_DATA_CACHE[cache_key] = snapshot
        return snapshot
    except Exception as exc:
        print(f"  Warning: unable to fetch market data for {symbol}: {exc}")
        _MARKET_DATA_CACHE[cache_key] = None
        return None


STATUS_WEIGHTS = {
    'approval': 0.30,
    'tentative approval': 0.15,
}

APPLICATION_TYPE_WEIGHTS = {
    'BLA': 0.25,
    'NDA': 0.20,
    'ANDA': 0.12,
}

SUBMISSION_PREFIX_WEIGHTS = {
    'ORIG': 0.20,
    'SUPPL': 0.08,
    'RESUB': 0.06,
    'EFFICACY': 0.10,
}

CLASSIFICATION_WEIGHTS = {
    'EFFICACY': 0.20,
    'PRIOR APPROVAL': 0.12,
    'MANUFACTURING': 0.10,
    'CMC': 0.08,
    'BIOEQUIVALENCE': 0.12,
    'PEDIATRIC': 0.08,
    'SAFETY': 0.08,
    'LABELING': 0.05,
    'REMS': 0.05,
    'BREAKTHROUGH THERAPY': 0.25,
    'FAST TRACK': 0.18,
    'PRIORITY REVIEW': 0.18,
    'ACCELERATED APPROVAL': 0.18,
}

DOSAGE_FORM_WEIGHTS = {
    'injectable': 0.08,
    'injection': 0.08,
    'tablet': 0.05,
    'capsule': 0.05,
    'oral solution': 0.04,
    'solution': 0.04,
    'suspension': 0.04,
    'topical': 0.03,
}

MATCH_CONFIDENCE_WEIGHTS = {
    'exact': 0.05,
    'normalized': 0.04,
    'openfigi': 0.04,
    'partial': 0.03,
}

MAX_CONFIDENCE_SCORE = 1.5

MARKET_CAP_THRESHOLDS_ASC = [
    (500_000_000, 0.12),
    (2_000_000_000, 0.08),
    (10_000_000_000, 0.04),
]

MARKET_CAP_PENALTIES_DESC = [
    (200_000_000_000, -0.08),
    (100_000_000_000, -0.05),
    (50_000_000_000, -0.03),
]

PRICE_THRESHOLDS_ASC = [
    (5, 0.08),
    (20, 0.05),
    (50, 0.02),
]

PRICE_PENALTIES_DESC = [
    (300, -0.05),
    (150, -0.02),
]


def _humanize_number(value):
    if value is None:
        return None
    abs_value = abs(value)
    for divisor, suffix in ((1_000_000_000_000, 'T'), (1_000_000_000, 'B'), (1_000_000, 'M')):
        if abs_value >= divisor:
            return f"{value / divisor:.2f}{suffix}"
    if abs_value >= 1_000:
        return f"{value:,.0f}"
    return f"{value:.2f}"


def _calculate_confidence_score(approval):
    """
    Estimate a confidence score for an approval based on its metadata.
    Allows scores > 1.0 for highly compelling signals.
    """
    score = 0.35  # Baseline expectation of value

    status = (approval.get('status') or '').lower()
    score += STATUS_WEIGHTS.get(status, 0.05)

    application_type = (approval.get('application_type') or '').upper()
    score += APPLICATION_TYPE_WEIGHTS.get(application_type, 0.08)

    submission = (approval.get('submission') or '').upper()
    prefix_weight = 0.0
    if submission:
        prefix = submission.split('-')[0]
        for key, weight in SUBMISSION_PREFIX_WEIGHTS.items():
            if prefix.startswith(key):
                prefix_weight = weight
                break
        if prefix_weight == 0.0:
            prefix_weight = 0.05
    score += prefix_weight

    classification = (approval.get('submission_classification') or '').upper()
    if classification:
        matched_any = False
        for key, weight in CLASSIFICATION_WEIGHTS.items():
            if key in classification:
                score += weight
                matched_any = True
        if not matched_any:
            score += 0.05

    dosage_form = (approval.get('dosage_form') or '').lower()
    for key, weight in DOSAGE_FORM_WEIGHTS.items():
        if key in dosage_form:
            score += weight
            break

    match_confidence = (approval.get('match_confidence') or '').lower()
    if match_confidence:
        score += MATCH_CONFIDENCE_WEIGHTS.get(match_confidence, 0.02)
    else:
        score -= 0.02

    market_cap = approval.get('market_cap')
    if market_cap:
        added = False
        for threshold, weight in MARKET_CAP_THRESHOLDS_ASC:
            if market_cap <= threshold:
                score += weight
                added = True
                break
        if not added:
            for threshold, penalty in MARKET_CAP_PENALTIES_DESC:
                if market_cap >= threshold:
                    score += penalty
                    break

    market_price = approval.get('market_price')
    if market_price:
        added = False
        for threshold, weight in PRICE_THRESHOLDS_ASC:
            if market_price <= threshold:
                score += weight
                added = True
                break
        if not added:
            for threshold, penalty in PRICE_PENALTIES_DESC:
                if market_price >= threshold:
                    score += penalty
                    break

    if not approval.get('stock_symbol'):
        score *= 0.4

    if application_type == 'BLA' and 'EFFICACY' in classification:
        score *= 1.15
    elif application_type == 'ANDA' and 'LABELING' in classification:
        score *= 0.9
    if 'BREAKTHROUGH THERAPY' in classification:
        score *= 1.2
        if market_cap and market_cap <= 2_000_000_000:
            score += 0.08

    return max(0.0, min(score, MAX_CONFIDENCE_SCORE))


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
                'submission_classification': approval.get('submission_classification', ''),
                'confidence_score': approval.get('confidence_score'),
                'market_cap': approval.get('market_cap'),
                'market_price_at_approval': approval.get('market_price'),
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
            score = result.get('confidence_score')
            if score is not None:
                print(f"   Confidence Score at approval: {score:.2f}")
            market_cap = result.get('market_cap')
            if market_cap:
                print(f"   Market Cap at approval: {_humanize_number(market_cap)}")
            market_price = result.get('market_price_at_approval')
            if market_price:
                print(f"   Share Price at approval: ${market_price:.2f}")
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
            score = result.get('confidence_score')
            if score is not None:
                print(f"   Confidence Score at approval: {score:.2f}")
            market_cap = result.get('market_cap')
            if market_cap:
                print(f"   Market Cap at approval: {_humanize_number(market_cap)}")
            market_price = result.get('market_price_at_approval')
            if market_price:
                print(f"   Share Price at approval: ${market_price:.2f}")
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


if __name__ == "__main__":
    import sys
    
    # Test scraping current month
    print("Testing FDA approval scraper...")
    print("=" * 70)
    
    analyze_flag = '--analyze' in sys.argv[1:]
    cli_args = [arg for arg in sys.argv[1:] if arg != '--analyze']

    if not os.environ.get("OPENFIGI_API_KEY"):
        print("Tip: set OPENFIGI_API_KEY to enable OpenFIGI symbol lookups.")
        print()

    days = 7
    if cli_args:
        try:
            candidate_days = int(cli_args[0])
            if candidate_days not in (7, 14):
                raise ValueError
            days = candidate_days
        except ValueError:
            print(f"Invalid days argument '{cli_args[0]}'. Using default 7-day report.\n")
    
    print(f"Scraping {days}-day report...\n")
    approvals = scrape_fda_approvals(days=days)
    
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
                if approval.get('submission_classification'):
                    print(f"   Classification: {approval['submission_classification']}")
                score = approval.get('confidence_score')
                if score is not None:
                    print(f"   Confidence Score: {score:.2f}")
                market_cap = approval.get('market_cap')
                if market_cap:
                    print(f"   Market Cap: {_humanize_number(market_cap)}")
                market_price = approval.get('market_price')
                if market_price:
                    print(f"   Share Price: ${market_price:.2f}")
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
                score = approval.get('confidence_score')
                if score is not None:
                    print(f"   Confidence Score: {score:.2f}")
                market_cap = approval.get('market_cap')
                if market_cap:
                    print(f"   Market Cap: {_humanize_number(market_cap)}")
                market_price = approval.get('market_price')
                if market_price:
                    print(f"   Share Price: ${market_price:.2f}")
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
    if analyze_flag:
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
    
    print(f"\nUsage: python test_fda.py [days] [--analyze]")
    print(f"  Example: python test_fda.py            (scrapes last 7 days)")
    print(f"  Example: python test_fda.py 14 --analyze  (scrapes 14 days, then analyzes)")
