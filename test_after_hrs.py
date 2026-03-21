#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
from typing import Dict, Any

import requests


def get_finnhub_price(symbol: str, api_key: str) -> Dict[str, Any]:
    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": symbol.upper(), "token": api_key}

    res = requests.get(url, params=params, timeout=10)
    res.raise_for_status()
    data = res.json()

    # Finnhub quote fields:
    # c=current, d=change, dp=percent change, h=high, l=low, o=open, pc=prev close, t=timestamp
    if not data or data.get("c") in (None, 0) or not data.get("t"):
        raise ValueError(f"No valid quote returned for symbol '{symbol.upper()}': {data}")

    return {
        "symbol": symbol.upper(),
        "price": data["c"],
        "timestamp": datetime.fromtimestamp(data["t"]),
        "raw": data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch latest quote from Finnhub for a ticker symbol.")
    parser.add_argument("ticker", help="Ticker symbol, e.g. IONQ or AAPL")
    args = parser.parse_args()

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: FINNHUB_API_KEY environment variable is not set.")

    try:
        quote = get_finnhub_price(args.ticker, api_key)
        print(f"Symbol: {quote['symbol']}")
        print(f"Price: ${quote['price']}")
        print(f"Time : {quote['timestamp']}")
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        body = e.response.text[:300] if e.response is not None else ""
        raise SystemExit(f"HTTP error ({status}) from Finnhub: {body}")
    except Exception as e:
        raise SystemExit(f"Failed to fetch quote: {e}")


if __name__ == "__main__":
    main()
