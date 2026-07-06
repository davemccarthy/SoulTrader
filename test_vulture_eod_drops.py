#!/usr/bin/env python3
"""
Vulture end-of-day drop monitor (lab CLI).

Core logic: core.services.advisors.vulture

Examples:
  source ~/Development/scratch/python/tutorial-env/bin/activate
  python test_vulture_eod_drops.py
  python test_vulture_eod_drops.py --date 2026-07-02
  python test_vulture_eod_drops.py --min-day-drop-pct 8 --top 25
  python test_vulture_eod_drops.py --prompt-only
  python test_vulture_eod_drops.py --llm
  python test_vulture_eod_drops.py --llm --llm-backend router --llm-rank-from 11 --llm-rank-to 30
  python test_vulture_eod_drops.py --llm --diagnose 5
"""

from __future__ import annotations

import argparse
import csv
import json
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
    EOD_LLM_BATCH_SIZE,
    EOD_MIN_DAY_DROP_PCT,
    EOD_TOP,
    DEFAULT_MIN_DOLLAR_VOLUME,
    DEFAULT_MIN_PRICE,
    DEFAULT_MIN_VOLUME,
    EodDropCandidate,
    build_drop_candidates,
    build_llm_context_block,
    build_vulture_drop_prompt,
    filter_monitor_rows,
    run_llm_triage,
)
from core.services.market import last_completed_trading_day, prior_trading_day  # noqa: E402

DEFAULT_CACHE_DIR = Path(".vulture_eod")


def _safe_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


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


def _cache_path(cache_dir: Path, session_date: date) -> Path:
    return cache_dir / f"eod_drops_{session_date.isoformat()}.csv"


def write_csv(path: Path, rows: Sequence[EodDropCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(EodDropCandidate.__dataclass_fields__.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_csv(path: Path) -> list[EodDropCandidate]:
    rows: list[EodDropCandidate] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            conf = raw.get("llm_confidence")
            rows.append(
                EodDropCandidate(
                    rank=_safe_int(raw.get("rank")),
                    symbol=str(raw.get("symbol") or "").upper(),
                    session_date=str(raw.get("session_date") or ""),
                    close=_safe_float(raw.get("close")) or 0.0,
                    prior_close=_safe_float(raw.get("prior_close")) or 0.0,
                    open=_safe_float(raw.get("open")) or 0.0,
                    volume=_safe_int(raw.get("volume")),
                    dollar_volume=_safe_float(raw.get("dollar_volume")) or 0.0,
                    day_change_pct=_safe_float(raw.get("day_change_pct")) or 0.0,
                    session_change_pct=_safe_float(raw.get("session_change_pct")) or 0.0,
                    llm_verdict=str(raw.get("llm_verdict") or ""),
                    llm_damage_type=str(raw.get("llm_damage_type") or ""),
                    llm_reason=str(raw.get("llm_reason") or ""),
                    llm_confidence=_safe_float(conf),
                )
            )
    return rows


def print_preview(rows: Sequence[EodDropCandidate], *, limit: int) -> None:
    print(
        f"{'rank':>4}  {'symbol':<7} {'close':>9} {'day%':>8} {'sess%':>8} "
        f"{'dv':>8} {'verdict':>8} {'reason'}"
    )
    for row in rows[:limit]:
        verdict = row.llm_verdict or "—"
        reason = (row.llm_reason or "")[:60]
        print(
            f"{row.rank:>4}  {row.symbol:<7} {row.close:>9.2f} "
            f"{_fmt_pct(row.day_change_pct):>8} {_fmt_pct(row.session_change_pct):>8} "
            f"{_fmt_money(row.dollar_volume):>8} {verdict:>8}  {reason}"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vulture EOD large-drop monitor (Polygon filter + optional LLM triage)."
    )
    parser.add_argument("--date", type=str, help="Session date YYYY-MM-DD (default: last completed trading day)")
    parser.add_argument("--top", type=int, default=EOD_TOP, help="Max drop candidates to keep")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh", action="store_true", help="Rebuild even if cached CSV exists")
    parser.add_argument("--include-etfs", action="store_true")
    parser.add_argument(
        "--include-non-equity",
        action="store_true",
        help="Keep ETFs, funds, and leveraged products (default: common equity only)",
    )
    parser.add_argument("--min-price", type=float, default=DEFAULT_MIN_PRICE)
    parser.add_argument("--min-volume", type=int, default=DEFAULT_MIN_VOLUME)
    parser.add_argument("--min-dollar-volume", type=float, default=DEFAULT_MIN_DOLLAR_VOLUME)
    parser.add_argument("--min-day-drop-pct", type=float, default=EOD_MIN_DAY_DROP_PCT)
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--llm-backend", choices=("router", "gemini", "deepseek"), default="router")
    parser.add_argument("--llm-rank-from", type=int, default=None)
    parser.add_argument("--llm-rank-to", type=int, default=None)
    parser.add_argument("--llm-batch-size", type=int, default=EOD_LLM_BATCH_SIZE)
    parser.add_argument("--no-llm-search", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--monitor-only", action="store_true")
    parser.add_argument("--diagnose", type=int, default=0)
    parser.add_argument("--headlines", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    session_date = date.fromisoformat(args.date) if args.date else last_completed_trading_day()
    prior = prior_trading_day(session_date)
    path = _cache_path(args.cache_dir, session_date)

    if path.exists() and not args.refresh and not args.prompt_only:
        rows = read_csv(path)
        print(f"Using cached EOD drop scan: {path} ({len(rows)} rows)")
    else:
        rows, raw_count = build_drop_candidates(
            session_date,
            min_price=args.min_price,
            min_volume=args.min_volume,
            min_dollar_volume=args.min_dollar_volume,
            min_day_drop_pct=args.min_day_drop_pct,
            include_etfs=args.include_etfs,
            equities_only=not args.include_non_equity,
            top=args.top,
        )
        if not args.prompt_only:
            write_csv(path, rows)
            equity_note = ""
            if not args.include_non_equity and not args.include_etfs:
                equity_note = f" raw_drops={raw_count} equities_kept={len(rows)}"
            print(
                f"EOD drop scan {session_date} (prior {prior}): "
                f"candidates={len(rows)} (min drop {args.min_day_drop_pct:.1f}%){equity_note}"
            )
            print(f"Saved: {path}")

    if args.prompt_only:
        if not rows:
            rows, _raw_count = build_drop_candidates(
                session_date,
                min_price=args.min_price,
                min_volume=args.min_volume,
                min_dollar_volume=args.min_dollar_volume,
                min_day_drop_pct=args.min_day_drop_pct,
                include_etfs=args.include_etfs,
                equities_only=not args.include_non_equity,
                top=args.top,
            )
        print(build_vulture_drop_prompt(build_llm_context_block(rows)))
        return 0

    if args.llm:
        rows = run_llm_triage(
            rows,
            advisor_name="vulture_eod_drops",
            backend=args.llm_backend,
            rank_from=args.llm_rank_from,
            rank_to=args.llm_rank_to,
            batch_size=args.llm_batch_size,
            use_search=not args.no_llm_search,
        )
        write_csv(path, rows)

    display_rows = filter_monitor_rows(rows) if args.monitor_only else rows
    if args.json:
        print(json.dumps([asdict(r) for r in display_rows], indent=2))
    else:
        print_preview(display_rows, limit=len(display_rows) or args.top)
        if args.llm and args.monitor_only:
            print(f"\n{len(display_rows)} monitor of {len(rows)} triaged")

    if args.diagnose:
        from core.services.health.diagnostic import analyze_symbol, print_report

        targets = filter_monitor_rows(rows) if rows and rows[0].llm_verdict else rows
        if not targets:
            targets = rows
        for row in targets[: args.diagnose]:
            print()
            print_report(analyze_symbol(row.symbol, headline_limit=args.headlines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
