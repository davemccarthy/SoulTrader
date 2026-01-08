#!/usr/bin/env python
"""
Compare Yahoo Finance current stocks vs Polygon historical stocks for a specific date
"""
import os
import sys
from datetime import datetime
import pandas as pd
from polygon import RESTClient
import yfinance as yf
from yfinance.screener import EquityQuery as YfEquityQuery

def get_yahoo_current_stocks(limit=50, min_price=1.0):
    """Get current high-volume stocks from Yahoo Finance"""
    print(f"\n{'='*80}")
    print("YAHOO FINANCE (CURRENT)")
    print(f"{'='*80}")
    
    try:
        most_active_query = YfEquityQuery(
            "and",
            [
                YfEquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
                YfEquityQuery("gt", ["intradayprice", min_price]),
            ]
        )
        
        max_size = min(limit * 2, 250)
        response = yf.screen(
            most_active_query,
            offset=0,
            size=max_size,
            sortField="intradayprice",
            sortAsc=True,
        )
        
        quotes = response.get("quotes", [])
        stocks = []
        
        for quote in quotes:
            symbol = quote.get('symbol')
            volume = quote.get('volume') or quote.get('regularMarketVolume') or 0
            price = quote.get('regularMarketPrice') or quote.get('intradayprice', 0)
            
            if symbol and price and price >= min_price:
                stocks.append({
                    'symbol': symbol,
                    'volume': float(volume) if volume else 0.0,
                    'price': float(price),
                })
        
        # Sort by volume (highest first)
        stocks.sort(key=lambda x: x['volume'], reverse=True)
        
        print(f"Found {len(stocks)} stocks")
        print(f"\nTop {limit} by volume:")
        print(f"{'Rank':<6} {'Symbol':<8} {'Volume':<15} {'Price':<10}")
        print("-" * 80)
        for i, stock in enumerate(stocks[:limit], 1):
            print(f"{i:<6} {stock['symbol']:<8} {stock['volume']:>14,.0f} ${stock['price']:>8.2f}")
        
        return [s['symbol'] for s in stocks[:limit]], {s['symbol']: s['volume'] for s in stocks[:limit]}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return [], {}


def get_polygon_historical_stocks_aggs(historical_date, limit=50, min_price=1.0):
    """Get historical high-volume stocks from Polygon using get_grouped_daily_aggs"""
    print(f"\n{'='*80}")
    print(f"POLYGON.IO - AGGS METHOD (HISTORICAL: {historical_date.strftime('%Y-%m-%d')})")
    print(f"{'='*80}")
    
    polygon_api_key = os.getenv('POLYGON_API_KEY')
    if not polygon_api_key:
        print("❌ POLYGON_API_KEY not set")
        return [], {}
    
    try:
        import time
        time.sleep(1)  # Rate limiting
        
        client = RESTClient(polygon_api_key)
        trade_date = historical_date.strftime('%Y-%m-%d')
        
        print(f"Fetching data using get_grouped_daily_aggs for {trade_date}...")
        aggs = client.get_grouped_daily_aggs(
            locale="us",
            date=trade_date,
            adjusted=False
        )
        
        # Convert to list first to see what we got
        aggs_list = list(aggs)
        print(f"  Polygon returned {len(aggs_list)} aggregates")
        
        if len(aggs_list) == 0:
            print(f"❌ No data found for {trade_date}")
            return [], {}
        
        df = pd.DataFrame([{
            "ticker": a.ticker,
            "volume": a.volume,
            "close": a.close,
            "vwap": a.vwap
        } for a in aggs_list])
        
        print(f"Found {len(df)} stocks with data")
        
        # Show sample of what we got
        print(f"\n  Sample of first 5 stocks from Polygon:")
        for i, row in df.head(5).iterrows():
            print(f"    {row['ticker']}: Vol={row['volume']:,.0f}, Close=${row['close']:.2f}")
        
        # Filter by minimum price
        df = df[df['close'] >= min_price]
        print(f"After filtering by min_price >= ${min_price:.2f}: {len(df)} stocks")
        
        # Sort by volume (descending)
        df = df.sort_values('volume', ascending=False)
        
        # Get top N
        top_df = df.head(limit)
        
        print(f"\nTop {limit} by volume:")
        print(f"{'Rank':<6} {'Symbol':<8} {'Volume':<15} {'Close':<10}")
        print("-" * 80)
        for i, (idx, row) in enumerate(top_df.iterrows(), 1):
            print(f"{i:<6} {row['ticker']:<8} {row['volume']:>14,.0f} ${row['close']:>8.2f}")
        
        symbols = top_df['ticker'].tolist()
        volumes = {row['ticker']: row['volume'] for idx, row in top_df.iterrows()}
        
        return symbols, volumes
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return [], {}


def compare_lists(yahoo_symbols, yahoo_volumes, polygon_symbols, polygon_volumes, limit=50):
    """Compare the two lists"""
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    yahoo_set = set(yahoo_symbols)
    polygon_set = set(polygon_symbols)
    
    # Common stocks
    common = yahoo_set & polygon_set
    print(f"\n✓ Common stocks (in both lists): {len(common)}/{limit}")
    
    # Yahoo only
    yahoo_only = yahoo_set - polygon_set
    print(f"📊 Yahoo only: {len(yahoo_only)}")
    
    # Polygon only
    polygon_only = polygon_set - yahoo_set
    print(f"📈 Polygon only: {len(polygon_only)}")
    
    if common:
        print(f"\nCommon stocks (showing overlap):")
        print(f"{'Symbol':<8} {'Yahoo Vol':<15} {'Polygon Vol':<15} {'Yahoo Rank':<12} {'Polygon Rank':<12}")
        print("-" * 80)
        for symbol in sorted(common):
            yahoo_vol = yahoo_volumes.get(symbol, 0)
            polygon_vol = polygon_volumes.get(symbol, 0)
            yahoo_rank = yahoo_symbols.index(symbol) + 1
            polygon_rank = polygon_symbols.index(symbol) + 1
            print(f"{symbol:<8} {yahoo_vol:>14,.0f} {polygon_vol:>14,.0f} {yahoo_rank:>11} {polygon_rank:>11}")
    
    if yahoo_only:
        print(f"\n📊 Stocks in Yahoo but NOT in Polygon (top 10):")
        for symbol in list(yahoo_only)[:10]:
            rank = yahoo_symbols.index(symbol) + 1
            vol = yahoo_volumes.get(symbol, 0)
            print(f"  {symbol} (rank {rank}, vol {vol:,.0f})")
    
    if polygon_only:
        print(f"\n📈 Stocks in Polygon but NOT in Yahoo (top 20):")
        for symbol in list(polygon_only)[:20]:
            rank = polygon_symbols.index(symbol) + 1
            vol = polygon_volumes.get(symbol, 0)
            print(f"  {symbol} (rank {rank}, vol {vol:,.0f})")
    
    # Show volume comparison for top stocks
    print(f"\n{'='*80}")
    print("VOLUME COMPARISON (Top 10 from each)")
    print(f"{'='*80}")
    print(f"{'Yahoo Top 10':<50} {'Polygon Top 10':<50}")
    print("-" * 100)
    for i in range(10):
        yahoo_info = ""
        polygon_info = ""
        if i < len(yahoo_symbols):
            sym = yahoo_symbols[i]
            vol = yahoo_volumes.get(sym, 0)
            yahoo_info = f"{i+1}. {sym} ({vol:,.0f})"
        if i < len(polygon_symbols):
            sym = polygon_symbols[i]
            vol = polygon_volumes.get(sym, 0)
            polygon_info = f"{i+1}. {sym} ({vol:,.0f})"
        print(f"{yahoo_info:<50} {polygon_info:<50}")


if __name__ == "__main__":
    import sys
    
    # Date to check (Friday, two days ago)
    # Today is 2025-12-21, so Friday two days ago is 2025-12-19
    if len(sys.argv) > 1:
        # Allow date to be passed as argument: python compare_yahoo_polygon.py 2025-12-19
        try:
            test_date = datetime.strptime(sys.argv[1], '%Y-%m-%d')
        except ValueError:
            print(f"❌ Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        test_date = datetime(2025, 12, 19)  # Default: Friday two days ago
    
    limit = 50
    
    print(f"Today's date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Comparing with historical date: {test_date.strftime('%Y-%m-%d')} ({test_date.strftime('%A')})")
    
    print("="*80)
    print("STOCK COMPARISON: Yahoo Finance (Current) vs Polygon (Historical)")
    print("="*80)
    
    # Get Yahoo stocks (current)
    yahoo_symbols, yahoo_volumes = get_yahoo_current_stocks(limit=limit, min_price=1.0)
    
    # Get Polygon stocks using AGGS method (historical)
    polygon_symbols, polygon_volumes = get_polygon_historical_stocks_aggs(
        historical_date=test_date, 
        limit=limit, 
        min_price=1.0
    )
    
    # Compare Yahoo vs Polygon AGGS
    print(f"\n{'='*80}")
    print("COMPARISON: Yahoo (Current) vs Polygon AGGS (Historical)")
    print(f"{'='*80}")
    if yahoo_symbols and polygon_symbols:
        compare_lists(yahoo_symbols, yahoo_volumes, polygon_symbols, polygon_volumes, limit=limit)
    else:
        print("\n❌ Cannot compare - one or both lists are empty")

