#!/usr/bin/env python3
"""
Trial-filter MIDWAY opportunity universe from an existing assess_universe JSONL.

Gates (v1 opportunity book):
  - not excluded
  - stability grade >= B, opportunity grade >= C  (same as Aug 14 midway)
  - beta in [BETA_MIN, BETA_MAX] (Yahoo)
  - optional price band

No hard mid-cap wall. marketCap is recorded for size-bucket stats only.

Usage:
  source ~/Development/scratch/python/tutorial-env/bin/activate
  python filter_midway_opportunity_trial.py
  python filter_midway_opportunity_trial.py --source .assessments/universe_2026-08-14.jsonl \\
      --output .assessments/universe_midway_opportunity_trial_2026-09-10.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
sys.path.insert(0, str(BASE_DIR))

from core.services.health.so_ratings import (  # noqa: E402
    opportunity_grade_at_least,
    stability_grade_at_least,
)

BETA_MIN = 0.9
BETA_MAX = 1.6
DEFAULT_PAUSE = 0.15

# Cap buckets (USD marketCap)
CAP_MICRO = 300_000_000
CAP_SMALL = 2_000_000_000
CAP_MID = 10_000_000_000
CAP_LARGE = 200_000_000_000


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _so_pass(row: Dict[str, Any], min_stab: str, min_opp: str) -> bool:
    if row.get("excluded"):
        return False
    if row.get("error"):
        return False
    stab = row.get("stability_grade") or ""
    opp = row.get("opportunity_grade") or ""
    if not stability_grade_at_least(stab, min_stab):
        return False
    if not opportunity_grade_at_least(opp, min_opp):
        return False
    return True


def _cap_bucket(market_cap: Optional[float]) -> str:
    if market_cap is None or market_cap <= 0:
        return "unknown"
    if market_cap < CAP_MICRO:
        return "micro"
    if market_cap < CAP_SMALL:
        return "small"
    if market_cap < CAP_MID:
        return "mid"
    if market_cap < CAP_LARGE:
        return "large"
    return "mega"


def _fetch_beta_cap(symbol: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Return (beta, market_cap, sector). Prefer local Stock.beta; Yahoo for mcap/sector."""
    beta_f: Optional[float] = None
    mcap_f: Optional[float] = None
    sector: Optional[str] = None

    try:
        import django

        if not django.apps.apps.ready:
            import os

            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
            django.setup()
        from core.models import Stock

        st = Stock.objects.filter(symbol=symbol.upper()).only(
            "beta", "sector"
        ).first()
        if st is not None:
            if st.beta is not None:
                beta_f = float(st.beta)
            if st.sector:
                sector = str(st.sector)
    except Exception:
        pass

    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        info = {}
    if beta_f is None and info.get("beta") is not None:
        try:
            beta_f = float(info["beta"])
        except (TypeError, ValueError):
            beta_f = None
    if info.get("marketCap") is not None:
        try:
            mcap_f = float(info["marketCap"])
        except (TypeError, ValueError):
            mcap_f = None
    if sector is None and isinstance(info.get("sector"), str):
        sector = info["sector"]
    return beta_f, mcap_f, sector


def main() -> int:
    p = argparse.ArgumentParser(description="Trial MIDWAY opportunity filter (SO + beta)")
    p.add_argument(
        "--source",
        type=Path,
        default=BASE_DIR / ".assessments" / "universe_2026-08-14.jsonl",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR
        / ".assessments"
        / "universe_midway_opportunity_trial_2026-09-10.json",
    )
    p.add_argument("--min-stability-grade", default="B")
    p.add_argument("--min-opportunity-grade", default="C")
    p.add_argument("--beta-min", type=float, default=BETA_MIN)
    p.add_argument("--beta-max", type=float, default=BETA_MAX)
    p.add_argument("--min-price", type=float, default=8.0)
    p.add_argument("--max-price", type=float, default=150.0)
    p.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max SO-passers to enrich (0 = all); for smoke tests",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=BASE_DIR
        / ".assessments"
        / "universe_midway_opportunity_beta_cache_2026-09-10.json",
        help="Read/write per-symbol beta/marketCap cache to avoid re-fetch",
    )
    p.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore existing cache entries and re-fetch",
    )
    args = p.parse_args()

    if not args.source.exists():
        print(f"Source not found: {args.source}", file=sys.stderr)
        return 1

    rows = _read_jsonl(args.source)
    so_pool = [
        r
        for r in rows
        if _so_pass(r, args.min_stability_grade, args.min_opportunity_grade)
    ]
    # Prefer higher composite when duplicates (shouldn't happen)
    by_sym: Dict[str, Dict[str, Any]] = {}
    for r in so_pool:
        sym = (r.get("symbol") or "").strip().upper()
        if not sym:
            continue
        px = r.get("polygon_price")
        if px is not None:
            try:
                px_f = float(px)
            except (TypeError, ValueError):
                px_f = None
            if px_f is not None and (
                px_f < args.min_price or px_f > args.max_price
            ):
                continue
        prev = by_sym.get(sym)
        if prev is None or float(r.get("composite") or 0) > float(
            prev.get("composite") or 0
        ):
            by_sym[sym] = r

    so_list = sorted(by_sym.values(), key=lambda r: r.get("symbol") or "")
    if args.limit and args.limit > 0:
        so_list = so_list[: args.limit]

    print(
        f"SO pool (stab>={args.min_stability_grade} opp>={args.min_opportunity_grade}, "
        f"price {args.min_price}-{args.max_price}): {len(so_list)} "
        f"(from {len(rows)} jsonl rows)"
    )

    cache: Dict[str, Any] = {}
    if args.cache.exists() and not args.refresh_cache:
        try:
            cache = json.loads(args.cache.read_text(encoding="utf-8"))
            if not isinstance(cache, dict):
                cache = {}
        except json.JSONDecodeError:
            cache = {}
        print(f"Loaded beta cache: {len(cache)} symbols from {args.cache}")

    print(f"Fetching beta/marketCap (pause={args.pause}s)…")

    kept: List[Dict[str, Any]] = []
    drop_beta: List[Dict[str, Any]] = []
    drop_missing: List[str] = []
    cap_counts: Counter = Counter()
    beta_stats: List[float] = []
    cache_hits = 0
    fetches = 0

    for i, row in enumerate(so_list, start=1):
        sym = row["symbol"].strip().upper()
        cached = cache.get(sym) if not args.refresh_cache else None
        if isinstance(cached, dict) and (
            cached.get("beta") is not None or cached.get("fetched")
        ):
            beta = cached.get("beta")
            mcap = cached.get("market_cap")
            sector = cached.get("sector")
            try:
                beta = float(beta) if beta is not None else None
            except (TypeError, ValueError):
                beta = None
            try:
                mcap = float(mcap) if mcap is not None else None
            except (TypeError, ValueError):
                mcap = None
            cache_hits += 1
        else:
            beta, mcap, sector = _fetch_beta_cap(sym)
            fetches += 1
            cache[sym] = {
                "beta": beta,
                "market_cap": mcap,
                "sector": sector,
                "fetched": True,
            }
            time.sleep(max(0.0, args.pause))

        rec = {
            "symbol": sym,
            "so_pair": row.get("so_pair"),
            "stability": row.get("stability"),
            "opportunity": row.get("opportunity"),
            "stability_grade": row.get("stability_grade"),
            "opportunity_grade": row.get("opportunity_grade"),
            "composite": row.get("composite"),
            "polygon_price": row.get("polygon_price"),
            "polygon_volume": row.get("polygon_volume"),
            "exchange": row.get("exchange"),
            "components": row.get("components"),
            "beta": beta,
            "market_cap": mcap,
            "cap_bucket": _cap_bucket(mcap),
            "sector": sector,
        }

        if beta is None:
            drop_missing.append(sym)
            if i % 25 == 0 or i == len(so_list):
                print(
                    f"  {i}/{len(so_list)} … kept={len(kept)} "
                    f"missing_beta={len(drop_missing)} "
                    f"cache_hits={cache_hits} fetches={fetches}"
                )
            continue

        beta_stats.append(beta)
        if args.beta_min <= beta <= args.beta_max:
            kept.append(rec)
            cap_counts[rec["cap_bucket"]] += 1
        else:
            drop_beta.append(
                {"symbol": sym, "beta": beta, "cap_bucket": rec["cap_bucket"]}
            )

        if i % 25 == 0 or i == len(so_list):
            print(
                f"  {i}/{len(so_list)} … kept={len(kept)} "
                f"missing_beta={len(drop_missing)} "
                f"cache_hits={cache_hits} fetches={fetches}"
            )

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.cache.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote beta cache ({len(cache)} symbols) → {args.cache}")

    kept.sort(key=lambda r: (-(r.get("composite") or 0), r["symbol"]))

    # Also report how many SO-passers sit outside beta band by side
    low_beta = [d for d in drop_beta if d["beta"] < args.beta_min]
    high_beta = [d for d in drop_beta if d["beta"] > args.beta_max]

    meta = {
        "created_at": _utc_now_iso(),
        "source": str(args.source),
        "rules": {
            "min_stability_grade": args.min_stability_grade,
            "min_opportunity_grade": args.min_opportunity_grade,
            "beta_min": args.beta_min,
            "beta_max": args.beta_max,
            "min_price": args.min_price,
            "max_price": args.max_price,
            "mid_cap_hard_wall": False,
            "note": "Trial opportunity book: SO + beta; cap buckets for stats only",
        },
        "counts": {
            "jsonl_rows": len(rows),
            "so_price_pool": len(so_list),
            "kept_beta": len(kept),
            "dropped_beta_low": len(low_beta),
            "dropped_beta_high": len(high_beta),
            "missing_beta": len(drop_missing),
            "cap_buckets": dict(cap_counts),
        },
        "beta_band_examples": {
            "too_low": sorted(low_beta, key=lambda x: x["beta"])[:8],
            "too_high": sorted(high_beta, key=lambda x: -x["beta"])[:8],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"meta": meta, "records": kept}, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== Trial filter result ===")
    print(f"Kept (β {args.beta_min}–{args.beta_max}): {len(kept)}")
    print(f"Dropped β low (<{args.beta_min}): {len(low_beta)}")
    print(f"Dropped β high (>{args.beta_max}): {len(high_beta)}")
    print(f"Missing β: {len(drop_missing)}")
    print(f"Cap buckets: {dict(cap_counts)}")
    if beta_stats:
        print(
            f"SO-pool β stats: min={min(beta_stats):.2f} "
            f"median={sorted(beta_stats)[len(beta_stats)//2]:.2f} "
            f"max={max(beta_stats):.2f}"
        )
    print(f"Wrote {args.output}")
    target_lo, target_hi = 100, 200
    n = len(kept)
    if n < target_lo:
        print(
            f"BELOW target {target_lo}–{target_hi} → consider looser SO, "
            f"wider β, or full re-assess including more small/large caps"
        )
    elif n > target_hi:
        print(f"ABOVE target {target_lo}–{target_hi} → tighten β or SO")
    else:
        print(f"IN target {target_lo}–{target_hi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
