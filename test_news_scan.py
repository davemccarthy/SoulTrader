#!/usr/bin/env python3
"""
Standalone headline red-flag scan (no Django).

Two-stage safety net prototype:
  1. Yahoo headlines (recent titles)
  2. Regex keyword screen — ESCALATE on hit (run --llm for ALLOW/BLOCK)
  3. Optional Gemini gatekeeper (--llm) on keyword hits, or all symbols with --llm-all

Usage:
    source ~/Development/scratch/python/tutorial-env/bin/activate
    python test_news_scan.py LLY
    python test_news_scan.py PFE TSLA --days auto --limit 5
    python test_news_scan.py PVH --days 3          # fixed 3-day window
    python test_news_scan.py LLY --llm --trigger "Flux below_ma_up entry"
    python test_news_scan.py LLY --json

Requires GEMINI_KEY in env or .env for --llm.

Benchmark calibration set (see --benchmark):
  PVH  — outlook cut + plunge → expect HIGH regex + LLM risk ~80-95
  GTM  — ZoomInfo (use instead of ZI on Yahoo); downgrade/guidance → ~60-80
  META — product/monetization noise → false-positive test, LLM ~30-50 not 90
  AVGO — earnings miss + AI beat → nuance test, LLM reads context
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from core.services.financial.yahoo import headline_lookback_slot
from core.services.risk.headline_screen import (
    RED_FLAG_PATTERNS,
    HeadlineScreenResult,
    apply_llm_verdict,
    fetch_headlines,
    run_llm_gatekeeper,
    scan_keywords,
)

# Calibration notes for --benchmark (human review targets, not assertions).
BENCHMARK_CASES: Dict[str, str] = {
    "PVH": "HIGH RISK — outlook cut + plunge; regex should hit; LLM risk ~80-95",
    "GTM": "MEDIUM — downgrade/guidance (ZoomInfo); LLM risk ~60-80 (ZI alias)",
    "ZI": "Same as GTM — Yahoo uses GTM for ZoomInfo headlines",
    "META": "FALSE-POSITIVE test — monetization/delay noise; LLM ~30-50 not 90",
    "AVGO": "NUANCE — ‘miss’ vs AI strength; LLM must read full headline context",
}


def resolve_lookback(days_arg: str, *, benchmark: bool) -> Tuple[Optional[float], Optional[int], str]:
    if benchmark:
        return None, 7, "7d (benchmark calibration)"
    if str(days_arg).strip().lower() == "auto":
        hours, slot = headline_lookback_slot()
        return hours, None, f"{hours:g}h auto ({slot})"
    days = int(days_arg)
    return None, days, f"{days}d"


def scan_symbol(
    symbol: str,
    *,
    limit: int = 5,
    max_age_days: Optional[int] = None,
    max_age_hours: Optional[float] = None,
    lookback: str = "",
    use_llm: bool = False,
    llm_all: bool = False,
    trigger: str = "",
) -> HeadlineScreenResult:
    requested = (symbol or "").strip().upper()

    if max_age_hours is not None:
        resolved, headlines, lb = fetch_headlines(
            requested, limit=limit, max_age_hours=max_age_hours
        )
        lookback = lookback or lb
    elif max_age_days is not None:
        resolved, headlines, lb = fetch_headlines(
            requested, limit=limit, max_age_days=max_age_days
        )
        lookback = lookback or lb
    else:
        resolved, headlines, lookback = fetch_headlines(requested, limit=limit)

    hits = scan_keywords(headlines)
    result = HeadlineScreenResult(
        symbol=requested,
        headlines=headlines,
        lookback=lookback,
        resolved_symbol=resolved,
    )
    if resolved != requested:
        result.reason = f"Headlines from alias {resolved}"

    if not hits:
        result.stage = "clear"
        result.allowed = True
        if not result.reason:
            result.reason = "No red-flag keywords in recent headlines"
        if not (use_llm and llm_all):
            return result
    else:
        result.keyword_hits = hits
        result.stage = "keyword_hit"
        result.allowed = True
        labels = sorted({h.label for h in hits})
        prefix = f"Headlines from alias {resolved}; " if resolved != requested else ""
        result.reason = f"{prefix}Keyword hit: {', '.join(labels)}"

    if not use_llm:
        return result

    llm = run_llm_gatekeeper(
        resolved,
        headlines,
        trigger=trigger,
        keyword_hits=hits,
        requested_symbol=requested,
    )
    if llm is None:
        result.stage = "llm_error"
        result.reason = "LLM unavailable or unparseable; keyword hits remain"
        return result

    apply_llm_verdict(result, llm)
    return result


def _display_verdict(result: HeadlineScreenResult) -> str:
    if result.stage == "keyword_hit":
        return "ESCALATE"
    if result.stage == "llm_block":
        return "BLOCK"
    if result.stage == "llm_allow":
        return "ALLOW"
    if result.stage == "llm_error" and result.keyword_hits:
        return "ESCALATE"
    if result.stage == "clear":
        return "ALLOW"
    return "BLOCK" if not result.allowed else "ALLOW"


def _print_result(result: HeadlineScreenResult, *, benchmark_note: str = "") -> None:
    print(f"\n{'=' * 60}")
    print(f"News scan — {result.symbol}")
    print(f"{'=' * 60}")
    if benchmark_note:
        print(f"  Benchmark: {benchmark_note}")
    if result.lookback:
        print(f"  Lookback:  {result.lookback}")

    print(f"  Headlines ({len(result.headlines)}):")
    for i, h in enumerate(result.headlines, 1):
        print(f"    [{i}] {h}")

    if result.keyword_hits:
        print("\n  Keyword hits:")
        for hit in result.keyword_hits:
            snippet = hit.headline if len(hit.headline) <= 90 else f"{hit.headline[:90]}…"
            print(f"    • {hit.label}: “{hit.match}” in “{snippet}”")
    else:
        print("\n  Keyword screen: CLEAR")

    verdict = _display_verdict(result)
    print(f"\n  Stage:   {result.stage}")
    print(f"  Verdict: {verdict}")
    if verdict == "ESCALATE":
        print("  Hint:    run with --llm to resolve to ALLOW / BLOCK")
    print(f"  Reason:  {result.reason}")

    if result.llm:
        print("\n  LLM:")
        print(f"    action:      {result.llm.get('action')}")
        print(f"    risk_score:  {result.llm.get('risk_score')}")
        print(f"    severity:    {result.llm.get('severity')}")
        print(f"    risk_type:   {result.llm.get('risk_type')}")
        print(f"    reasoning:   {result.llm.get('reasoning')}")
        drivers = result.llm.get("main_drivers") or []
        if drivers:
            print("    drivers:")
            for d in drivers:
                print(f"      - {d}")
        counters = result.llm.get("counterarguments") or []
        if counters:
            print("    counterarguments:")
            for c in counters:
                print(f"      - {c}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headline red-flag scan: regex screen + optional LLM gatekeeper"
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Ticker symbol(s), e.g. LLY PFE (omit with --benchmark)",
    )
    parser.add_argument(
        "--days",
        default="auto",
        metavar="DAYS",
        help="Headline window: 'auto' (session-aware) or integer days (default auto)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max headlines per symbol (default 5)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Run Gemini gatekeeper when keywords hit (requires GEMINI_KEY)",
    )
    parser.add_argument(
        "--llm-all",
        action="store_true",
        help="With --llm, score every symbol even when regex is clear (nuance tests)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run calibration set: PVH ZI META AVGO (ZI uses GTM headlines)",
    )
    parser.add_argument(
        "--trigger",
        default="",
        help="Optional entry context for LLM, e.g. 'Flux below_ma_up'",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON results only",
    )
    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="Print configured regex labels and exit",
    )
    args = parser.parse_args()

    if args.list_patterns:
        print("Red-flag patterns (label → regex):")
        for label, pat in RED_FLAG_PATTERNS:
            print(f"  {label:20} {pat}")
        return

    symbols = list(args.symbols or [])
    benchmark = args.benchmark
    if benchmark:
        symbols = ["PVH", "ZI", "META", "AVGO"]
    if not symbols:
        parser.error("Provide ticker symbol(s) or use --benchmark")

    try:
        max_age_hours, max_age_days, lookback_label = resolve_lookback(
            args.days, benchmark=benchmark
        )
    except ValueError:
        parser.error(f"--days must be 'auto' or an integer, got {args.days!r}")

    if not args.json:
        print(f"Headline window: {lookback_label}")

    results = [
        scan_symbol(
            sym,
            limit=args.limit,
            max_age_days=max_age_days,
            max_age_hours=max_age_hours,
            lookback=lookback_label,
            use_llm=args.llm,
            llm_all=args.llm_all,
            trigger=args.trigger,
        )
        for sym in symbols
    ]

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    for result in results:
        note = BENCHMARK_CASES.get(result.symbol, "") if benchmark else ""
        _print_result(result, benchmark_note=note)
    print()


if __name__ == "__main__":
    main()
