#!/usr/bin/env python3
"""
Standalone health score lab (v2). Does not call advisor.health_check() or Django.

Usage:
    source ~/Development/scratch/python/tutorial-env/bin/activate
    python health_score.py AAPL
    python health_score.py AAPL MSFT --detail
    python health_score.py AAPL -c valuation
    python health_score.py AAPL --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.services.health.consensus import score_consensus_health
from core.services.health.ratings import score_to_rating
from core.services.health.financial import score_financial_health
from core.services.health.intrinsic import score_intrinsic_health
from core.services.health.price import score_price_health
from core.services.health.sector import score_sector_health
from core.services.health.valuation import score_valuation_health

COMPONENTS = ("financial", "valuation", "intrinsic", "price", "consensus", "sector")

# Order and planned final-model weights (LLM qualitative 20% not implemented yet).
_COMPONENT_SPECS: List[Tuple[str, str, float, Callable[[str], Any]]] = [
    ("financial", "Financial health", 0.25, score_financial_health),
    ("valuation", "Valuation", 0.25, score_valuation_health),
    ("intrinsic", "Intrinsic valuation", 0.10, score_intrinsic_health),
    ("price", "Price position", 0.10, score_price_health),
    ("consensus", "Analyst consensus", 0.10, score_consensus_health),
    ("sector", "Sector / industry", 0.10, score_sector_health),
]

_SCORERS: Dict[str, tuple[str, Callable[[str], Any]]] = {
    key: (title, fn) for key, title, _wt, fn in _COMPONENT_SPECS
}


def _run_all_components(symbol: str) -> List[Tuple[str, str, float, Any]]:
    """Return (key, title, weight, result) for each component."""
    out: List[Tuple[str, str, float, Any]] = []
    for key, title, weight, scorer in _COMPONENT_SPECS:
        out.append((key, title, weight, scorer(symbol)))
    return out


def _weighted_preview(rows: List[Tuple[str, str, float, Any]]) -> Optional[float]:
    """Renormalized weighted mean over components that returned a score."""
    num = 0.0
    den = 0.0
    for _key, _title, weight, result in rows:
        if result.score is not None:
            num += float(result.score) * weight
            den += weight
    if den <= 0:
        return None
    return round(num / den, 1)


def _print_summary(symbol: str, rows: List[Tuple[str, str, float, Any]]) -> None:
    preview = _weighted_preview(rows)

    print(f"\n{'=' * 60}")
    print(f"Health score v2 — {symbol}")
    print(f"{'=' * 60}")
    rating = score_to_rating(preview)
    if rating:
        print(f"  {rating.display_line}")
        print()
    print(f"  {'Component':<26} {'Wt':>5}  {'Score':>8}")
    print(f"  {'-' * 26} {'-' * 5}  {'-' * 8}")
    for _key, title, weight, result in rows:
        sc = f"{result.score:.1f}" if result.score is not None else "—"
        err = f"  ({result.error})" if result.score is None and result.error else ""
        print(f"  {title:<26} {weight * 100:>4.0f}%  {sc:>8}{err}")

    weight_sum = sum(w for _, _, w, r in rows if r.score is not None)
    print(f"  {'-' * 26} {'-' * 5}  {'-' * 8}")
    if preview is not None:
        note = ""
        if weight_sum < 0.99:
            note = f"  (preview over {weight_sum * 100:.0f}% of model; LLM 20% not included)"
        print(f"  {'Weighted preview':<26} {'':>5}  {preview:>8}{note}")
    else:
        print("  Weighted preview: unavailable (all components failed)")
    print()


def _print_component(title: str, result) -> None:
    if result.error and result.score is None:
        print(f"ERROR {result.symbol}: {result.error}")
        return

    share = getattr(result, "component_weight", None)
    share_pct = f"{share * 100:.0f}%" if share is not None else "—"

    print(f"\n{'-' * 60}")
    print(f"{title} — {result.symbol}")
    print(f"{'-' * 60}")
    print(f"  Score (0–100):     {result.score}")
    print(f"  Final model share: {share_pct} (when combined)")

    sector = getattr(result, "sector", None)
    industry = getattr(result, "industry", None)
    bkey = getattr(result, "sector_benchmark_key", None) or getattr(result, "sector_key", None)
    if sector or industry:
        if industry and sector:
            print(f"  Sector / industry: {sector} / {industry}")
        else:
            print(f"  Sector / industry: {sector or industry}")
        if getattr(result, "match_source", None):
            print(f"  Match:             {result.match_source} — {getattr(result, 'match_label', '')}")
        elif bkey and bkey != "default":
            print(f"  Benchmark set:     {bkey}")

    fair = getattr(result, "fair_value", None)
    price = getattr(result, "price", None)
    low_52 = getattr(result, "week52_low", None)
    high_52 = getattr(result, "week52_high", None)
    pct = getattr(result, "range_percentile", None)
    if low_52 is not None and high_52 is not None and price is not None:
        pct_s = f"{pct * 100:.0f}%" if pct is not None else "—"
        print(f"  52-week range:     ${low_52:.2f} – ${high_52:.2f}  (now ${price:.2f}, {pct_s})")
    if fair is not None and price is not None:
        print(f"  Price / fair:      ${price:.2f} / ${fair:.2f}")
    if getattr(result, "neutral_fallback", False):
        reason = getattr(result, "fallback_reason", "") or "neutral"
        print(f"  Neutral fallback:  {reason} (scored {result.score})")

    rec = getattr(result, "recommendation_key", None)
    upside = getattr(result, "upside_to_mean_pct", None)
    analysts = getattr(result, "analyst_count", None)
    if rec or upside is not None:
        parts = []
        if rec:
            parts.append(f"rating={rec}")
        if upside is not None:
            parts.append(f"upside to mean={upside:+.1f}%")
        if analysts is not None:
            parts.append(f"n={analysts}")
        print(f"  Street snapshot:   {', '.join(parts)}")
    if getattr(result, "thin_coverage", False):
        print(f"  Thin coverage:     <3 analysts (rating blended toward neutral)")

    if result.missing:
        print(f"  Missing metrics:   {', '.join(result.missing)}")
    print()
    print(f"  {'Metric':<28} {'Raw':>22} {'Score':>8}  Wt")
    print(f"  {'-' * 28} {'-' * 22} {'-' * 8}  ----")
    for m in result.metrics:
        sc = f"{m.score:.1f}" if m.score is not None else "—"
        print(f"  {m.label:<28} {m.raw_display:>22} {sc:>8}  {m.weight * 100:>4.0f}%")
        if m.note:
            print(f"    ({m.note})")
    print()


def main() -> None:
    all_choices = COMPONENTS + ("all",)
    parser = argparse.ArgumentParser(
        description="Health score v2 lab (standalone; no Django health_check)."
    )
    parser.add_argument(
        "symbols",
        nargs="+",
        help="Ticker symbol(s), e.g. AAPL MSFT",
    )
    parser.add_argument(
        "-c",
        "--component",
        choices=all_choices,
        default="all",
        help="Component to run, or 'all' for every component (default: all)",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="With --component all, print full metric breakdown per component",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON only",
    )
    parser.add_argument(
        "--ratings-table",
        action="store_true",
        help="Print rating band reference table and exit",
    )
    args = parser.parse_args()

    if args.ratings_table:
        print("Health v2 rating bands:")
        print(f"  {'Grade':<7} {'Score':<12} Label")
        print(f"  {'-' * 7} {'-' * 12} {'-' * 20}")
        for row in (
            ("A", "80+", "Exceptional"),
            ("B", "75–79", "Strong buy"),
            ("C", "65–74", "Buy"),
            ("D", "50–64", "Watch / mixed"),
            ("E", "35–49", "Avoid"),
            ("F", "<35", "Strong avoid"),
        ):
            print(f"  {row[0]:<7} {row[1]:<12} {row[2]}")
        return

    any_failed = False
    json_payload: List[Dict[str, Any]] = []

    for sym in args.symbols:
        symbol = sym.strip().upper()
        if not symbol:
            continue

        if args.component == "all":
            rows = _run_all_components(symbol)
            if args.json:
                preview = _weighted_preview(rows)
                rating = score_to_rating(preview)
                json_payload.append(
                    {
                        "symbol": symbol,
                        "weighted_preview": preview,
                        "rating": rating.to_dict() if rating else None,
                        "components": {
                            key: result.to_dict() for key, _t, _w, result in rows
                        },
                    }
                )
            else:
                _print_summary(symbol, rows)
                if args.detail:
                    for _key, title, _w, result in rows:
                        _print_component(title, result)
            if any(r.score is None for _, _, _, r in rows):
                any_failed = True
        else:
            title, scorer = _SCORERS[args.component]
            result = scorer(symbol)
            if args.json:
                json_payload.append(result.to_dict())
            else:
                _print_component(title, result)
            if result.score is None:
                any_failed = True

    if args.json:
        print(json.dumps(json_payload, indent=2))

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
