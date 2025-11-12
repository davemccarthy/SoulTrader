#!/usr/bin/env python
"""
Quick harness for exercising the Yahoo Finance screener logic that powers the
`Yahoo` advisor discovery flow.  Run this script while iterating on filters to
see exactly which symbols come back before wiring changes into the project.

Examples:
    python test_yf.py
    python test_yf.py --limit 20 --max-price 3.5 --runs 3
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import yfinance as yf
from yfinance.screener import EquityQuery as YfEquityQuery

logger = logging.getLogger("test_yf")


def _build_query(max_price: float) -> YfEquityQuery:
    """Replicates the `Yahoo` advisor query with a configurable price cap."""
    return YfEquityQuery(
        "and",
        [
            YfEquityQuery("lt", ["pegratio_5y", 1]),  # PEG below 1
            YfEquityQuery(
                "or",
                [
                    YfEquityQuery("btwn", ["epsgrowth.lasttwelvemonths", 25, 50]),
                    YfEquityQuery("gt", ["epsgrowth.lasttwelvemonths", 100]),
                ],
            ),
            YfEquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
            YfEquityQuery("lt", ["intradayprice", max_price]),
        ],
    )


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class ScreenerResult:
    symbol: str
    company: str
    peg_ratio: Optional[float]
    eps_growth: Optional[float]
    price: Optional[float]
    raw: dict
    details_fetched: bool = field(default=False, repr=False)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "company": self.company,
            "peg_ratio": self.peg_ratio,
            "eps_growth": self.eps_growth,
            "price": self.price,
        }


def fetch_filtered_quotes(
    limit: int,
    max_price: float,
) -> Iterable[ScreenerResult]:
    """
    Fetch screener results, randomising the starting offset (with a fallback to
    the first page). Mirrors the production discovery logic.
    """
    query = _build_query(max_price=max_price)

    response = yf.screen(
        query,
        offset=0,
        size=limit,
        sortField="pegratio_5y",
        sortAsc=True,
    )
    quotes = response.get("quotes") or []

    logger.info("Fetched %s quotes (offset=0)", len(quotes))
    for quote in quotes:
        symbol = quote.get("symbol")
        if not symbol:
            continue

        price = _safe_float(
            quote.get("regularMarketPrice") or quote.get("intradayprice")
        )
        if price is None or price >= max_price:
            # Safety check in case Yahoo returns stale data outside our filter.
            continue

        company = quote.get("longName") or quote.get("shortName") or symbol
        yield ScreenerResult(
            symbol=symbol,
            company=company,
            peg_ratio=_safe_float(quote.get("pegRatio")),
            eps_growth=_safe_float(quote.get("epsGrowthQuarterly")),
            price=price,
            raw=quote,
        )


def enrich_with_ticker_info(results: List[ScreenerResult]) -> None:
    """Populate missing PEG/EPS values via individual ticker lookups."""
    symbols_to_fetch = [
        res.symbol
        for res in results
        if res.peg_ratio is None or res.eps_growth is None
    ]
    if not symbols_to_fetch:
        return

    tickers = yf.Tickers(" ".join(symbols_to_fetch))
    peg_keys = (
        "pegRatio",
        "pegRatio5Y",
        "pegRatioFiveYear",
        "pegRatioFiveYears",
        "pegRatios",
    )
    eps_keys = (
        "earningsQuarterlyGrowth",
        "revenueQuarterlyGrowth",
        "earningsGrowth",
        "epsGrowth",
    )

    for res in results:
        if res.symbol not in tickers.tickers:
            continue

        info = {}
        try:
            info = tickers.tickers[res.symbol].info or {}
        except Exception as exc:  # pragma: no cover - network errors, etc.
            logger.debug("Ticker info lookup failed for %s (%s)", res.symbol, exc)
            continue

        if res.peg_ratio is None:
            for key in peg_keys:
                if key in info:
                    res.peg_ratio = _safe_float(info.get(key))
                    if res.peg_ratio is not None:
                        break

        if res.eps_growth is None:
            for key in eps_keys:
                if key in info:
                    res.eps_growth = _safe_float(info.get(key))
                    if res.eps_growth is not None:
                        break

        res.details_fetched = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yahoo screener sandbox")
    parser.add_argument("--limit", type=int, default=10, help="Max quotes to fetch")
    parser.add_argument(
        "--max-price",
        type=float,
        default=5.0,
        help="Upper bound for share price filter",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent screener pulls to perform",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Fetch ticker fundamentals when PEG/EPS data is missing",
    )
    parser.add_argument(
        "--dump-raw",
        action="store_true",
        help="Print the raw quote payload for the first result (debugging)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    for run_idx in range(1, args.runs + 1):
        logger.info("Run %s/%s", run_idx, args.runs)
        results = list(
            fetch_filtered_quotes(
                limit=args.limit,
                max_price=args.max_price,
            )
        )

        if args.enrich and results:
            enrich_with_ticker_info(results)

        if not results:
            print(f"[Run {run_idx}] No quotes returned")
            continue

        print(f"[Run {run_idx}] {len(results)} symbols:")
        for res in results:
            eps_str = (
                f"{res.eps_growth:.0%}" if res.eps_growth is not None else "n/a"
            )
            peg_str = f"{res.peg_ratio:.2f}" if res.peg_ratio is not None else "n/a"
            price_str = f"${res.price:.2f}" if res.price is not None else "n/a"
            print(
                f"  {res.symbol:<7} {price_str:>10}  PEG={peg_str:>6}  EPS={eps_str:>6}  {res.company}"
            )

        if args.dump_raw:
            print("\nFirst raw record:")
            print(results[0].raw)


if __name__ == "__main__":
    main()


