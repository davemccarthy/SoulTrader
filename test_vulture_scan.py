#!/usr/bin/env python3
"""
Vulture price-damage universe scan (lab CLI).

Core logic: core.services.advisors.vulture

Examples:
  source ~/Development/scratch/python/tutorial-env/bin/activate
  python test_vulture_scan.py
  python test_vulture_scan.py --top 50 --seed-universe 1500
  python test_vulture_scan.py --min-price 5 --min-dollar-volume 50000000
  python test_vulture_scan.py --date 2026-06-26 --refresh
  python test_vulture_scan.py --diagnose 5
  python test_vulture_scan.py --screen-buy-ready
  python test_vulture_scan.py --screen-buy-ready --screen-limit 100
  python test_vulture_scan.py --screen-buy-ready --screen-report
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _setup_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()


_setup_django()

from core.services.advisors.vulture import (  # noqa: E402
    DEFAULT_MIN_DOLLAR_VOLUME,
    DEFAULT_MIN_PRICE,
    DEFAULT_MIN_VOLUME,
    SCAN_BATCH_SIZE,
    SCAN_MAX_52W_DAMAGE_PCT,
    SCAN_MIN_3M_DAMAGE_PCT,
    SCAN_MIN_52W_DAMAGE_PCT,
    SCAN_SEED_UNIVERSE,
    SCAN_TOP,
    VultureScanCandidate,
    build_scan_candidates,
    build_seed_rows,
    build_weekly_scan_candidates,
    fetch_grouped_daily_rows,
)
from core.services.market import last_completed_trading_day  # noqa: E402

DEFAULT_CACHE_DIR = Path(".vulture_scan")


def _safe_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_pct(value: Optional[float]) -> str:
    return "" if value is None else f"{value:+.1f}"


def _fmt_money(value: Optional[float]) -> str:
    if value is None:
        return ""
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    return f"{value:.0f}"


def _cache_path(cache_dir: Path, scan_date: date) -> Path:
    return cache_dir / f"vulture_price_scan_{scan_date.isoformat()}.csv"


def write_csv(path: Path, rows: Sequence[VultureScanCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(VultureScanCandidate.__dataclass_fields__.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def print_preview(rows: Sequence[VultureScanCandidate | dict[str, str]], *, limit: int) -> None:
    print(
        f"{'rank':>4}  {'symbol':<7} {'price':>9} {'dmg3m%':>8} {'dmg52%':>8} "
        f"{'dv':>8} {'trigger':>7}"
    )
    for row in rows[:limit]:
        if isinstance(row, VultureScanCandidate):
            rank = row.rank
            symbol = row.symbol
            price = row.price
            damage_3m = row.damage_3m_pct
            damage_52w = row.damage_52w_pct
            dollar_volume = row.dollar_volume
            trigger = row.trigger
        else:
            rank = row.get("rank", "")
            symbol = row.get("symbol", "")
            price = _safe_float(row.get("price")) or 0.0
            damage_3m = _safe_float(row.get("damage_3m_pct"))
            damage_52w = _safe_float(row.get("damage_52w_pct"))
            dollar_volume = _safe_float(row.get("dollar_volume")) or 0.0
            trigger = row.get("trigger", "")

        print(
            f"{rank:>4}  {symbol:<7} {price:>9.2f} {_fmt_pct(damage_3m):>8} "
            f"{_fmt_pct(damage_52w):>8} {_fmt_money(dollar_volume):>8} {trigger:>7}"
        )


def run_diagnostics(rows: Sequence[VultureScanCandidate | dict[str, str]], *, count: int, headlines: int) -> None:
    from core.services.health.diagnostic import analyze_symbol, print_report

    for row in rows[:count]:
        symbol = row.symbol if isinstance(row, VultureScanCandidate) else str(row.get("symbol") or "")
        if not symbol:
            continue
        print()
        print_report(analyze_symbol(symbol, headline_limit=headlines))


def _row_symbol(row: VultureScanCandidate | dict[str, str]) -> str:
    if isinstance(row, VultureScanCandidate):
        return row.symbol.strip().upper()
    return str(row.get("symbol") or "").strip().upper()


def screen_buy_ready(
    rows: Sequence[VultureScanCandidate | dict[str, str]],
    *,
    limit: Optional[int],
    headlines: int,
    report: bool,
) -> int:
    from core.services.health.diagnostic import (
        analyze_symbol,
        filter_buy_ready,
        print_report,
        print_summary_table,
    )

    symbols = [_row_symbol(row) for row in rows]
    if limit is not None:
        symbols = symbols[:limit]
    symbols = [s for s in symbols if s]

    results = [analyze_symbol(symbol, headline_limit=headlines) for symbol in symbols]
    ready = filter_buy_ready(results)
    print_summary_table(ready, title="BUY READY")
    print(f"\n{len(ready)} BUY READY of {len(symbols)} screened")

    if report:
        for result in ready:
            print()
            print_report(result)

    return len(ready)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/reuse a Vulture price-damage universe scan.")
    parser.add_argument("--top", type=int, default=SCAN_TOP)
    parser.add_argument("--preview", type=int, default=25)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--date", type=str, help="Polygon grouped-aggs date YYYY-MM-DD")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--include-etfs", action="store_true")
    parser.add_argument("--min-price", type=float, default=DEFAULT_MIN_PRICE)
    parser.add_argument("--min-volume", type=int, default=DEFAULT_MIN_VOLUME)
    parser.add_argument("--min-dollar-volume", type=float, default=DEFAULT_MIN_DOLLAR_VOLUME)
    parser.add_argument("--seed-universe", type=int, default=SCAN_SEED_UNIVERSE)
    parser.add_argument("--batch-size", type=int, default=SCAN_BATCH_SIZE)
    parser.add_argument("--min-3m-damage-pct", type=float, default=SCAN_MIN_3M_DAMAGE_PCT)
    parser.add_argument("--min-52w-damage-pct", type=float, default=SCAN_MIN_52W_DAMAGE_PCT)
    parser.add_argument("--max-52w-damage-pct", type=float, default=SCAN_MAX_52W_DAMAGE_PCT)
    parser.add_argument("--diagnose", type=int, default=0)
    parser.add_argument("--headlines", type=int, default=3)
    parser.add_argument("--screen-buy-ready", action="store_true")
    parser.add_argument("--screen-limit", type=int, default=None)
    parser.add_argument("--screen-report", action="store_true")
    return parser.parse_args(argv)


def _run_post_scan_actions(
    rows: Sequence[VultureScanCandidate | dict[str, str]],
    args: argparse.Namespace,
) -> None:
    if args.screen_buy_ready:
        screen_buy_ready(
            rows,
            limit=args.screen_limit,
            headlines=args.headlines,
            report=args.screen_report,
        )
    elif args.diagnose:
        run_diagnostics(rows, count=args.diagnose, headlines=args.headlines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    scan_date = date.fromisoformat(args.date) if args.date else last_completed_trading_day()
    path = _cache_path(args.cache_dir, scan_date)

    if path.exists() and not args.refresh:
        rows = read_csv(path)
        print(f"Using cached Vulture price scan: {path} ({len(rows)} rows)")
        print_preview(rows, limit=args.preview)
        _run_post_scan_actions(rows, args)
        return 0

    candidates, stats = build_weekly_scan_candidates(
        scan_date,
        min_price=args.min_price,
        min_volume=args.min_volume,
        min_dollar_volume=args.min_dollar_volume,
        seed_universe=args.seed_universe,
        include_etfs=args.include_etfs,
        min_3m_damage_pct=args.min_3m_damage_pct,
        min_52w_damage_pct=args.min_52w_damage_pct,
        max_52w_damage_pct=args.max_52w_damage_pct,
        top=args.top,
        batch_size=args.batch_size,
    )
    write_csv(path, candidates)

    print(
        f"Built Vulture price scan for {scan_date}: "
        f"polygon_rows={stats['polygon_rows']} seeds={stats['seeds']} candidates={len(candidates)}"
    )
    print(f"Saved: {path}")
    print_preview(candidates, limit=args.preview)
    _run_post_scan_actions(candidates, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
