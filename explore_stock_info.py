#!/usr/bin/env python3
"""
Explore yfinance data for existing stocks to see what fields are available.
Can also backfill Stock records with sector, industry, website, and beta.

Usage:
    python explore_stock_info.py                    # Explore only (no changes)
    python explore_stock_info.py --output stock_info.csv  # Export to CSV
    python explore_stock_info.py --backfill          # Update Stock records
"""

import os
import sys
import csv
import argparse
from collections import defaultdict

import django
import yfinance as yf

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from core.models import Stock, Holding, Discovery  # noqa: E402


def format_market_cap(market_cap):
    """Format market cap in human-readable format."""
    if market_cap is None:
        return "N/A"
    if market_cap >= 1_000_000_000_000:
        return f"${market_cap/1_000_000_000_000:.2f}T"
    elif market_cap >= 1_000_000_000:
        return f"${market_cap/1_000_000_000:.2f}B"
    elif market_cap >= 1_000_000:
        return f"${market_cap/1_000_000:.2f}M"
    else:
        return f"${market_cap:,.0f}"


def fetch_stock_info(stock, update_stock=False):
    """Fetch info from yfinance for a stock. Optionally update the stock record."""
    from decimal import Decimal
    
    try:
        ticker = yf.Ticker(stock.symbol)
        info = ticker.info
        
        # Extract data
        company = info.get('longName') or info.get('shortName') or info.get('name', 'N/A')
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        exchange = info.get('fullExchangeName') or info.get('exchange', 'N/A')
        website = info.get('website', 'N/A')
        beta = info.get('beta')
        
        # Update stock if requested
        if update_stock:
            updated = False
            
            # Company
            if company != 'N/A' and not stock.company:
                stock.company = company[:200]
                updated = True
            
            # Exchange
            if exchange != 'N/A' and not stock.exchange:
                stock.exchange = exchange[:32]
                updated = True
            
            # Sector
            if sector != 'N/A' and not stock.sector:
                stock.sector = sector[:100]
                updated = True
            
            # Industry
            if industry != 'N/A' and not stock.industry:
                stock.industry = industry[:200]
                updated = True
            
            # Website
            if website != 'N/A' and not stock.website:
                stock.website = website[:200]
                updated = True
            
            # Beta
            if beta is not None and stock.beta is None:
                try:
                    stock.beta = Decimal(str(beta))
                    updated = True
                except (ValueError, TypeError):
                    pass
            
            if updated:
                stock.save()
        
        return {
            'symbol': stock.symbol,
            'company': company,
            'sector': sector,
            'industry': industry,
            'exchange': exchange,
            'country': info.get('country', 'N/A'),
            'marketCap': info.get('marketCap'),
            'marketCapFormatted': format_market_cap(info.get('marketCap')),
            'beta': beta,
            'dividendYield': info.get('dividendYield'),
            'fullTimeEmployees': info.get('fullTimeEmployees'),
            'website': website,
            'businessSummary': (info.get('longBusinessSummary') or 'N/A')[:200] + '...' if info.get('longBusinessSummary') else 'N/A',
            'businessSummaryFull': info.get('longBusinessSummary', ''),
            'error': None
        }
    except Exception as e:
        return {
            'symbol': stock.symbol,
            'company': stock.company or 'N/A',
            'sector': 'ERROR',
            'industry': 'ERROR',
            'exchange': stock.exchange or 'N/A',
            'country': 'ERROR',
            'marketCap': None,
            'marketCapFormatted': 'ERROR',
            'beta': None,
            'dividendYield': None,
            'fullTimeEmployees': None,
            'website': 'ERROR',
            'businessSummary': 'ERROR',
            'businessSummaryFull': '',
            'error': str(e)
        }


def main(output_file=None, backfill=False):
    """Main function to explore stock info."""
    # Get unique stocks from holdings only
    holding_stocks = Holding.objects.values_list('stock', flat=True).distinct()
    stocks = Stock.objects.filter(id__in=holding_stocks).order_by('symbol')
    total = stocks.count()
    
    if backfill:
        print(f"Backfilling yfinance data for {total} unique stocks from holdings...\n")
    else:
        print(f"Exploring yfinance data for {total} unique stocks from holdings...\n")
    
    results = []
    sector_counts = defaultdict(int)
    industry_counts = defaultdict(int)
    country_counts = defaultdict(int)
    errors = 0
    updated = 0
    
    for i, stock in enumerate(stocks, 1):
        action = "Updating" if backfill else "Fetching"
        print(f"[{i}/{total}] {action} info for {stock.symbol}...", end=' ', flush=True)
        
        info = fetch_stock_info(stock, update_stock=backfill)
        results.append(info)
        
        if backfill and not info['error']:
            # Check if any fields were updated
            if info['sector'] != 'N/A' or info['industry'] != 'N/A' or info['website'] != 'N/A' or info['beta'] is not None:
                updated += 1
        
        if info['error']:
            print(f"ERROR: {info['error']}")
            errors += 1
        else:
            print(f"✓ {info['sector']} / {info['industry']}")
            sector_counts[info['sector']] += 1
            industry_counts[info['industry']] += 1
            country_counts[info['country']] += 1
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total stocks: {total}")
    print(f"Errors: {errors}")
    print(f"Success: {total - errors}")
    if backfill:
        print(f"Updated: {updated}")
    
    print("\nSectors (top 10):")
    for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        if sector != 'N/A':
            print(f"  {sector}: {count}")
    
    print("\nIndustries (top 10):")
    for industry, count in sorted(industry_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        if industry != 'N/A':
            print(f"  {industry}: {count}")
    
    print("\nCountries (top 10):")
    for country, count in sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        if country != 'N/A':
            print(f"  {country}: {count}")
    
    # Market cap stats
    market_caps = [r['marketCap'] for r in results if r['marketCap'] is not None]
    if market_caps:
        print(f"\nMarket Cap Stats:")
        print(f"  Min: {format_market_cap(min(market_caps))}")
        print(f"  Max: {format_market_cap(max(market_caps))}")
        print(f"  Avg: {format_market_cap(sum(market_caps) / len(market_caps))}")
        print(f"  Stocks with market cap data: {len(market_caps)}/{total}")
    
    # Beta stats
    betas = [r['beta'] for r in results if r['beta'] is not None]
    if betas:
        print(f"\nBeta Stats:")
        print(f"  Min: {min(betas):.2f}")
        print(f"  Max: {max(betas):.2f}")
        print(f"  Avg: {sum(betas) / len(betas):.2f}")
        print(f"  Stocks with beta data: {len(betas)}/{total}")
    
    # Output to CSV if requested
    if output_file:
        print(f"\nWriting results to {output_file}...")
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'symbol', 'company', 'sector', 'industry', 'exchange', 'country',
                'marketCap', 'marketCapFormatted', 'beta', 'dividendYield',
                'fullTimeEmployees', 'website', 'businessSummary', 'error'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Don't write the full business summary to CSV (too long)
                row = {k: v for k, v in result.items() if k != 'businessSummaryFull'}
                writer.writerow(row)
        
        print(f"✓ Results written to {output_file}")
        print(f"\nNote: Full business summaries are in the data but truncated in CSV.")
        print(f"      Check individual stock info for full text if needed.")
    
    # Show sample of data
    print("\n" + "="*80)
    print("SAMPLE DATA (first 5 stocks):")
    print("="*80)
    for result in results[:5]:
        print(f"\n{result['symbol']}: {result['company']}")
        print(f"  Sector: {result['sector']}")
        print(f"  Industry: {result['industry']}")
        print(f"  Country: {result['country']}")
        print(f"  Market Cap: {result['marketCapFormatted']}")
        if result['beta']:
            print(f"  Beta: {result['beta']:.2f}")
        if result['error']:
            print(f"  ERROR: {result['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Explore yfinance data for existing stocks')
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output CSV file path (optional)'
    )
    parser.add_argument(
        '--backfill',
        action='store_true',
        help='Update Stock records with fetched data (sector, industry, website, beta)'
    )
    
    args = parser.parse_args()
    
    try:
        main(output_file=args.output, backfill=args.backfill)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)



