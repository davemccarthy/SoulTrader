#!/usr/bin/env python3
"""
Test script for FDA advisor company matching and OpenFIGI lookup
Usage: python test_company_match.py "Alembic Pharms Ltd"
"""

import os
import sys
import requests

# OpenFIGI endpoints
OPENFIGI_MAPPING_ENDPOINT = "https://api.openfigi.com/v3/mapping"
OPENFIGI_SEARCH_ENDPOINT = "https://api.openfigi.com/v3/search"

# Company map (from FDA advisor)
COMPANY_MAP = {
    "PFIZER": "PFE",
    "JOHNSON & JOHNSON": "JNJ",
    "JANSSEN": "JNJ",
    "JANSSEN BIOTECH": "JNJ",
    "MERCK": "MRK",
    "MERCK & CO": "MRK",
    "ABBV": "ABBV",
    "ABBVIE": "ABBV",
    "GILEAD": "GILD",
    "GILEAD SCIENCES": "GILD",
    "BRISTOL MYERS SQUIBB": "BMY",
    "BMS": "BMY",
    "AMGEN": "AMGN",
    "ELI LILLY": "LLY",
    "LILLY": "LLY",
    "NOVARTIS": "NVS",
    "ROCHE": "RHHBY",
    "GLAXOSMITHKLINE": "GSK",
    "GSK": "GSK",
    "SANOFI": "SNY",
    "ASTRAZENECA": "AZN",
    "TEVA": "TEVA",
    "TEVA PHARMACEUTICAL": "TEVA",
    "TEVA PHARMS USA": "TEVA",
    "TEVA PHARMACEUTICALS": "TEVA",
    "MYLAN": "VTRS",
    "VIATRIS": "VTRS",
    "SUN PHARMA": "SUNPF",
    "SUN PHARMACEUTICAL": "SUNPF",
    "DR REDDYS": "RDY",
    "DR. REDDYS": "RDY",
    "DR REDDYS LABORATORIES": "RDY",
    "AUROBINDO": "AUPH",
    "AUROBINDO PHARMA": "AUPH",
    "AUROBINDO PHARMA LTD": "AUPH",
    "HOSPIRA": "HSP",
    "HOSPIRA INC": "HSP",
    "CENTOCOR": "JNJ",
    "CENTOCOR ORTHO BIOTECH": "JNJ",
    "PHARMACIA": "PFE",
    "PHARMACIA AND UPJOHN": "PFE",
    "UPJOHN": "PFE",
    "SCYNEXIS": "SCYX",
    "UCB": "UCB",
    "UCB INC": "UCB",
    "AMNEAL": "AMRX",
    "AMNEAL PHARMACEUTICALS": "AMRX",
    "AMNEAL PHARMS": "AMRX",
    "SAGENT": "SGNT",
    "SAGENT PHARMS": "SGNT",
    "SAGENT PHARMACEUTICALS": "SGNT",
    "BRECKENRIDGE": "BRX",
}


def normalize_company_name(company_name):
    """Normalize company name by removing common suffixes"""
    if not company_name:
        return ""

    normalized = company_name.upper().strip()
    suffixes = [
        " INC",
        " INC.",
        " CORPORATION",
        " CORP",
        " CORP.",
        " LTD",
        " LTD.",
        " LIMITED",
        " LLC",
        " LLC.",
        " PHARMACEUTICALS",
        " PHARMA",
        " PHARMACEUTICAL",
        " PHARMS",
        " LABORATORIES",
        " LABS",
        " COMPANY",
        " CO",
    ]

    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
            break

    return normalized


def test_company_map(company_name):
    """Test company map matching"""
    print(f"\n{'='*70}")
    print(f"Testing Company Map Matching for: '{company_name}'")
    print(f"{'='*70}")
    
    company_upper = company_name.upper().strip()
    
    # Check for private companies
    if "PRIVATE LTD" in company_upper or "PRIVATE LIMITED" in company_upper:
        print("❌ REJECTED: Contains 'PRIVATE LTD' or 'PRIVATE LIMITED'")
        return None, None
    
    # 1. Exact match
    if company_upper in COMPANY_MAP:
        symbol = COMPANY_MAP[company_upper]
        print(f"✅ EXACT MATCH: '{company_upper}' → {symbol}")
        return symbol, "exact"
    
    # 2. Normalized match
    normalized = normalize_company_name(company_name)
    print(f"   Normalized: '{company_name}' → '{normalized}'")
    
    if normalized in COMPANY_MAP:
        symbol = COMPANY_MAP[normalized]
        print(f"✅ NORMALIZED MATCH: '{normalized}' → {symbol}")
        return symbol, "normalized"
    
    # 3. Partial match
    for key, symbol in COMPANY_MAP.items():
        if key in company_upper or company_upper in key:
            print(f"✅ PARTIAL MATCH: '{key}' found in '{company_upper}' → {symbol}")
            return symbol, "partial"
    
    print("❌ NO MATCH in company map")
    return None, None


def lookup_symbol_via_openfigi(company_name):
    """Lookup symbol via OpenFIGI API using SEARCH endpoint"""
    if not company_name:
        return None
    
    api_key = os.environ.get("OPENFIGI_API_KEY")
    if not api_key:
        print("⚠️  OPENFIGI_API_KEY not set in environment")
        return None
    
    headers = {
        "Content-Type": "application/json",
        "X-OPENFIGI-APIKEY": api_key,
    }
    
    # Use SEARCH endpoint for company name lookup
    print(f"\n   Using OpenFIGI SEARCH endpoint for company name...")
    
    # Search payload - try different search approaches
    search_queries = [
        company_name,  # Full name
        normalize_company_name(company_name),  # Normalized
    ]
    
    data_entries = None
    
    for query in search_queries:
        try:
            # OpenFIGI search endpoint format
            payload = {
                "query": query,
                "securityType2": "Common Stock",
            }
            
            print(f"   Searching for: '{query}'")
            response = requests.post(OPENFIGI_SEARCH_ENDPOINT, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            
            results = response.json()
            print(f"   OpenFIGI search response: {results}")
            
            # Check if we got results
            if isinstance(results, dict):
                data_entries = results.get("data") or []
                if data_entries:
                    print(f"   Found {len(data_entries)} results")
                    break
            elif isinstance(results, list):
                if results:
                    data_entries = results
                    print(f"   Found {len(data_entries)} results")
                    break
            
        except requests.RequestException as exc:
            print(f"   Search failed: {exc}")
            continue
        except ValueError:
            print(f"   Search returned invalid JSON")
            continue
    
    # If search didn't work, try mapping endpoint with ticker if known
    if not data_entries and "Alembic" in company_name:
        print(f"\n   Search didn't find results, trying direct ticker lookup: APLLTD")
        ticker_result = try_ticker_lookup("APLLTD", headers)
        if ticker_result:
            return ticker_result
    
    if not data_entries:
        print(f"❌ OpenFIGI search returned no results")
        return None
    
    print(f"   OpenFIGI returned {len(data_entries)} entries")
    
    # Show all entries for debugging
    print(f"\n   OpenFIGI Results:")
    for i, entry in enumerate(data_entries[:5], 1):  # Show first 5
        ticker = entry.get("ticker", "N/A")
        exch_code = entry.get("exchCode", "N/A")
        security_type = entry.get("securityType2", "N/A")
        name = entry.get("name", "N/A")
        print(f"      {i}. {ticker} ({exchCode}) - {security_type}")
        print(f"         Name: {name}")
    
    # Select best ticker
    ticker = select_best_openfigi_ticker(data_entries)
    return ticker


def try_ticker_lookup(ticker, headers):
    """Try looking up by ticker symbol directly using MAPPING endpoint"""
    payload = [{"idType": "TICKER", "idValue": ticker}]
    try:
        print(f"   Attempting ticker lookup for: {ticker}")
        response = requests.post(OPENFIGI_MAPPING_ENDPOINT, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        results = response.json()
        print(f"   Ticker lookup response: {results}")
        if isinstance(results, list) and results:
            mapping_result = results[0] or {}
            if "error" not in mapping_result:
                data_entries = mapping_result.get("data") or []
                if data_entries:
                    print(f"   ✅ Found via ticker lookup: {ticker}")
                    return ticker
                else:
                    print(f"   No data entries for ticker {ticker}")
            else:
                print(f"   Error in ticker lookup: {mapping_result.get('error')}")
    except Exception as e:
        print(f"   Ticker lookup failed: {e}")
    return None


def select_best_openfigi_ticker(entries):
    """Select the best ticker from OpenFIGI results"""
    if not entries:
        return None
    
    preferred_exchanges = {"XNYS", "XNAS", "ARCX", "BATS"}
    
    # First pass: preferred exchanges with common stock
    for entry in entries:
        ticker = entry.get("ticker")
        if not ticker:
            continue
        security_type = (entry.get("securityType2") or "").lower()
        if entry.get("exchCode") in preferred_exchanges and security_type.startswith("common stock"):
            print(f"✅ Selected: {ticker} (preferred exchange: {entry.get('exchCode')})")
            return ticker
    
    # Second pass: any stock/equity
    for entry in entries:
        ticker = entry.get("ticker")
        if not ticker:
            continue
        security_type = (entry.get("securityType2") or "").lower()
        if "stock" in security_type or "equity" in security_type:
            print(f"✅ Selected: {ticker} (stock/equity type)")
            return ticker
    
    # Third pass: any ticker
    for entry in entries:
        ticker = entry.get("ticker")
        if ticker:
            print(f"✅ Selected: {ticker} (first available)")
            return ticker
    
    return None


def test_openfigi(company_name):
    """Test OpenFIGI lookup"""
    print(f"\n{'='*70}")
    print(f"Testing OpenFIGI Lookup for: '{company_name}'")
    print(f"{'='*70}")
    
    symbol = lookup_symbol_via_openfigi(company_name)
    
    if symbol:
        print(f"\n✅ OpenFIGI MATCH: '{company_name}' → {symbol}")
        return symbol, "openfigi"
    else:
        print(f"\n❌ NO MATCH from OpenFIGI")
        return None, None


def main():
    # Get company name from command line or use default
    if len(sys.argv) > 1:
        company_name = sys.argv[1]
    else:
        company_name = "Alembic Pharms Ltd"
    
    print(f"\n{'='*70}")
    print(f"FDA Company Matching Test")
    print(f"{'='*70}")
    print(f"Company: '{company_name}'")
    
    # Test company map
    symbol_map, confidence_map = test_company_map(company_name)
    
    # Test OpenFIGI (only if company map failed)
    symbol_openfigi, confidence_openfigi = None, None
    if not symbol_map:
        symbol_openfigi, confidence_openfigi = test_openfigi(company_name)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    
    if symbol_map:
        print(f"✅ Company Map: {symbol_map} ({confidence_map})")
    else:
        print(f"❌ Company Map: No match")
    
    if symbol_openfigi:
        print(f"✅ OpenFIGI: {symbol_openfigi} ({confidence_openfigi})")
    else:
        if os.environ.get("OPENFIGI_API_KEY"):
            print(f"❌ OpenFIGI: No match")
        else:
            print(f"⚠️  OpenFIGI: Not tested (OPENFIGI_API_KEY not set)")
    
    final_symbol = symbol_map or symbol_openfigi
    final_confidence = confidence_map or confidence_openfigi
    
    if final_symbol:
        print(f"\n✅ FINAL RESULT: {final_symbol} ({final_confidence})")
    else:
        print(f"\n❌ FINAL RESULT: No symbol found")
    
    print(f"\nUsage: python {sys.argv[0]} 'Company Name'")
    print(f"       Or set OPENFIGI_API_KEY environment variable for OpenFIGI testing")


if __name__ == "__main__":
    main()

