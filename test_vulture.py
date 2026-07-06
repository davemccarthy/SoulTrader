#!/usr/bin/env python3
"""
Single-stock recovery diagnostic lab CLI.

Core logic lives in core.services.health.diagnostic.

Examples:
  source ~/Development/scratch/python/tutorial-env/bin/activate
  python test_vulture.py COIN
  python test_vulture.py COIN AMD DIS TSLA
  python test_vulture.py COIN --json
  python test_vulture.py COIN --headlines 5
  python test_vulture.py INSP EEFT COIN --buy-ready-only --summary
  python test_vulture.py INSP EEFT --buy-ready-only --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.services.health.diagnostic import (
    analyze_symbol,
    diagnostic_to_dict,
    filter_buy_ready,
    print_report,
    print_summary_table,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain damaged quality/recovery diagnostic score for one or more tickers."
    )
    parser.add_argument("symbols", nargs="+", help="Ticker symbols, e.g. COIN AMD DIS")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    parser.add_argument(
        "--headlines",
        type=int,
        default=3,
        help="Number of recent Yahoo Finance headlines to inspect per symbol.",
    )
    parser.add_argument(
        "--buy-ready-only",
        action="store_true",
        help="Only output symbols with decision BUY READY.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print compact summary table instead of full per-symbol reports.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    results = [analyze_symbol(symbol, headline_limit=args.headlines) for symbol in args.symbols]
    if args.buy_ready_only:
        results = filter_buy_ready(results)

    if args.json:
        print(json.dumps([diagnostic_to_dict(r) for r in results], indent=2, sort_keys=True))
        return

    if args.summary:
        label = "BUY READY" if args.buy_ready_only else "Summary"
        print_summary_table(results, title=label)
        return

    if not results and args.buy_ready_only:
        print("No BUY READY symbols in the requested set.")
        return

    for idx, result in enumerate(results):
        if idx:
            print()
        print_report(result)

    if len(results) > 1:
        print_summary_table(results)


if __name__ == "__main__":
    main()
