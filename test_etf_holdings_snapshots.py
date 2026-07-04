#!/usr/bin/env python3
"""
CLI wrapper for ETF holdings snapshots (uses core.services.financial.etf_holdings).

Usage:
  source ~/Development/scratch/python/tutorial-env/bin/activate
  python test_etf_holdings_snapshots.py
  python test_etf_holdings_snapshots.py --etfs QTUM,BOTZ,AIQ,ARKQ
  python test_etf_holdings_snapshots.py --out-dir .etf_holdings --refresh
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Django setup for service default paths
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

from core.services.financial import etf_holdings  # noqa: E402


def _parse_etfs(value: str) -> List[str]:
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def _print_summary(result: etf_holdings.RefreshResult, preview: int) -> None:
    if not result.ok:
        print(f"{result.etf}: error {result.error}", file=sys.stderr)
        return
    diff = result.diff
    added = len(diff.added) if diff else 0
    removed = len(diff.removed) if diff else 0
    print(
        f"{result.etf}: ok date={result.holdings_date} saved={result.snapshot_path} "
        f"added={added} removed={removed}"
    )
    if diff and diff.added:
        print(f"  added preview: {', '.join(h.ticker for h in diff.added[:preview])}")
    if diff and diff.removed:
        print(f"  removed preview: {', '.join(diff.removed[:preview])}")
    if preview and result.snapshot_path:
        snap = etf_holdings.load_snapshot(result.snapshot_path)
        for holding in snap.holdings[:preview]:
            weight = "" if holding.weight_pct is None else f" {holding.weight_pct:.2f}%"
            print(f"  {holding.ticker:12s} {weight:>8s} {holding.name}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch issuer ETF holdings snapshots.")
    parser.add_argument(
        "--etfs",
        default=",".join(etf_holdings.DEFAULT_ETF_LIST),
        help="Comma-separated ETF tickers",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--refresh", action="store_true", help="Overwrite an existing same-date snapshot")
    parser.add_argument("--preview", type=int, default=5, help="Rows to print per ETF")
    args = parser.parse_args(argv)

    out_dir = args.out_dir or etf_holdings.default_holdings_dir()
    failures = 0
    for result in etf_holdings.refresh_snapshots(
        etfs=_parse_etfs(args.etfs),
        out_dir=out_dir,
        refresh=args.refresh,
    ):
        if not result.ok:
            failures += 1
        _print_summary(result, args.preview)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
