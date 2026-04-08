#!/usr/bin/env python3
"""
Standalone probe for OpenFDA drugsfda.json — no Django required.

Loads OPENFDA_API_KEY from the project root .env (same as Django) if present.

Usage:
  python test_openfda.py
  python test_openfda.py --preset orig-ap
  python test_openfda.py --preset orig-ap --limit 10 --resort-orig
  python test_openfda.py --preset orig-ap --limit 15 --score
  python test_openfda.py --preset orig-ap --score --update-ingredient-cache
  python test_openfda.py --preset orig-ap --limit 100 --pages 5   # up to 500 rows
  python test_openfda.py --recent-months 6 --preset orig-ap --score

Docs: https://open.fda.gov/apis/drug/drugsfda/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Project root (sibling to core/)
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")

from core.services.openfda.client import (
    MAX_DRUGSFDA_LIMIT,
    fetch_drugsfda_paginated,
    get_openfda_api_key,
)
from core.services.openfda.significance import (
    collect_active_ingredient_names,
    ingredient_cache_load,
    ingredient_cache_merge,
    latest_orig_ap_event_date,
    normalize_ingredient_key,
    recent_orig_ap_cutoff_months,
    score_drugsfda_record,
)

from core.services.market.yahoo_event_window import (
    price_stats_for_event,
    symbol_name_plausible,
    yfinance_search_us_equity,
)
from core.services.market.company_symbol_map import match_company_to_symbol

OPENFDA_DRUGSFDA = "https://api.fda.gov/drug/drugsfda.json"

# Presets: (search, sort, note)
PRESETS = {
    "prescription": (
        "products.marketing_status:Prescription",
        "submissions.submission_status_date:desc",
        "Broad: Rx products; API sort may reflect any submission date on the app.",
    ),
    "orig-ap": (
        "submissions.submission_status:AP AND submissions.submission_type:ORIG",
        "submissions.submission_status_date:desc",
        "Apps with an approved ORIG submission; API order ≠ newest ORIG date — use --resort-orig.",
    ),
}


def _orig_ap_dates(item):
    """YYYYMMDD strings for submissions that are both ORIG and AP."""
    out = []
    for s in item.get("submissions") or []:
        if (s.get("submission_type") or "").upper() != "ORIG":
            continue
        if (s.get("submission_status") or "").upper() != "AP":
            continue
        d = s.get("submission_status_date")
        if d:
            out.append(d)
    return out


def _max_orig_ap_date(item):
    dates = _orig_ap_dates(item)
    return max(dates) if dates else ""


def _fmt_orig_ap_date(s_out: dict) -> str:
    """Format latest ORIG+AP submission_status_date (YYYYMMDD) for display."""
    sub = s_out.get("latest_orig_ap_submission") or {}
    raw = sub.get("submission_status_date")
    if not raw or not isinstance(raw, str) or len(raw) != 8 or not raw.isdigit():
        return "—"
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def main():
    parser = argparse.ArgumentParser(description="Probe OpenFDA drugsfda.json")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="prescription",
        help="Query preset (default: prescription)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help=f"Page size per API request, max {MAX_DRUGSFDA_LIMIT} (default: 5)",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Number of pages to fetch (skip = page_index × limit). Default: 1",
    )
    parser.add_argument(
        "--resort-orig",
        action="store_true",
        help="After fetch, sort by newest ORIG+AP submission_status_date (useful with orig-ap preset).",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Rule-based significance score (see core/services/openfda/significance.py).",
    )
    parser.add_argument(
        "--ingredient-cache",
        type=Path,
        default=BASE_DIR / ".cache" / "openfda_ingredients.json",
        help="JSON file of seen active-ingredient keys for first_seen bonus (default: .cache/openfda_ingredients.json).",
    )
    parser.add_argument(
        "--update-ingredient-cache",
        action="store_true",
        help="With --score, append all ingredient keys from this batch to the cache file.",
    )
    parser.add_argument(
        "--sequential-ingredients",
        action="store_true",
        help="With --score, update seen-ingredients after each record (order-dependent first_seen).",
    )
    parser.add_argument(
        "--verbose-json",
        action="store_true",
        help="With --score, still print truncated full JSON per record.",
    )
    parser.add_argument(
        "--recent-months",
        type=int,
        default=0,
        metavar="N",
        help="Only latest ORIG+AP dates on or after (today − N months); 0 = no filter (default: 0).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="",
        metavar="YYYY-MM-DD",
        help=(
            "Filter to exact latest ORIG+AP event date (YYYY-MM-DD). "
            "Overrides --recent-months."
        ),
    )
    args = parser.parse_args()
    if args.pages < 1:
        parser.error("--pages must be >= 1")

    target_date = None
    if args.date.strip():
        try:
            target_date = datetime.strptime(args.date.strip(), "%Y-%m-%d").date()
        except ValueError:
            parser.error("--date must be in YYYY-MM-DD format")

    search, sort, preset_note = PRESETS[args.preset]
    api_key = get_openfda_api_key()

    # When caller provides --date, constrain the OpenFDA query itself.
    # This avoids the "fetch newest N then post-filter" failure mode.
    if target_date is not None:
        yyyymmdd = target_date.strftime("%Y%m%d")
        # Constrain to ORIG+AP approvals on the exact submission_status_date.
        search = f"{search} AND submissions.submission_status_date:{yyyymmdd}"

    print("GET", OPENFDA_DRUGSFDA)
    print("preset:", args.preset, "—", preset_note)
    eff_limit = min(args.limit, MAX_DRUGSFDA_LIMIT)
    if args.limit > MAX_DRUGSFDA_LIMIT:
        print(f"(note: --limit {args.limit} clipped to {MAX_DRUGSFDA_LIMIT} per request)")
    print(
        "params:",
        {
            "search": search,
            "sort": sort,
            "limit": eff_limit,
            "pages": args.pages,
            "recent_months": args.recent_months,
            "date": target_date.isoformat() if target_date else "",
        },
    )
    if api_key:
        print("(api_key: set)")
    else:
        print("(no OPENFDA_API_KEY — lower rate limit)\n")

    try:
        results, meta = fetch_drugsfda_paginated(
            search=search,
            sort=sort,
            limit=args.limit,
            pages=args.pages,
            api_key=api_key,
        )
    except Exception as e:
        print("Request failed:", e, file=sys.stderr)
        sys.exit(1)

    if args.resort_orig:
        results.sort(key=_max_orig_ap_date, reverse=True)

    if target_date is not None:
        results = [
            x
            for x in results
            if (ev0 := latest_orig_ap_event_date(x)) is not None and ev0 == target_date
        ]
        print(f"(filtered to latest ORIG+AP date == {target_date.isoformat()})")
    else:
        cutoff = recent_orig_ap_cutoff_months(args.recent_months)
        if cutoff is not None:
            results = [
                x
                for x in results
                if (ev0 := latest_orig_ap_event_date(x)) is not None and ev0 >= cutoff
            ]
            print(f"(filtered to latest ORIG+AP date >= {cutoff.isoformat()})")

    print("\n--- meta (partial) ---")
    print(
        json.dumps(
            {
                k: meta.get(k)
                for k in ("disclaimer", "terms", "license", "last_updated")
                if k in meta
            },
            indent=2,
        )
    )

    print(f"\n--- results: {len(results)} record(s) ---\n")

    seen: set[str] | None = None
    if args.score:
        seen = set(ingredient_cache_load(args.ingredient_cache))
        if not args.sequential_ingredients:
            pass  # same seen for all rows in batch
        else:
            seen = set(seen)  # copy for mutation

    scored_rows: list[tuple[float, dict]] = []

    for i, item in enumerate(results):
        if args.score:
            batch_seen = seen if seen is not None else None
            s_out = score_drugsfda_record(item, seen_ingredient_keys=batch_seen)
            scored_rows.append((s_out["score"], s_out))
            if args.sequential_ingredients and batch_seen is not None:
                for k in s_out.get("ingredient_keys") or []:
                    batch_seen.add(k)

            print(f"========== record {i + 1} (scored) ==========")
            print(
                f"score={s_out['score']} tier={s_out['tier']} | "
                f"ORIG+AP={_fmt_orig_ap_date(s_out)} | "
                f"{s_out.get('application_number')} | {s_out.get('sponsor_name')} | "
                f"brand={s_out.get('brand_name')!r}"
            )
            print(f"  reasons: {', '.join(s_out['reasons'])}")
            if s_out.get("latest_orig_ap_submission"):
                sub = s_out["latest_orig_ap_submission"]
                print(
                    "  latest ORIG+AP:",
                    sub.get("submission_status_date"),
                    sub.get("submission_class_code"),
                    sub.get("submission_class_code_description"),
                    "| review:",
                    sub.get("review_priority"),
                )
            if args.verbose_json:
                blob = json.dumps(item, indent=2)
                print("\n--- full JSON (truncated) ---")
                print(blob[:8000])
                if len(blob) > 8000:
                    print("\n... (truncated) ...")
            print()
            continue

        print(f"========== record {i + 1} ==========")
        print("top-level keys:", sorted(item.keys()))
        sponsor = item.get("sponsor_name")
        app_no = item.get("application_number")
        products = item.get("products") or []
        subs = item.get("submissions") or []

        print(f"application_number: {app_no!r}")
        print(f"sponsor_name: {sponsor!r}")
        if args.preset == "orig-ap" or args.resort_orig:
            print(f"ORIG+AP submission_status_date(s): {_orig_ap_dates(item)}")

        if products:
            p0 = products[0]
            print("products[0] keys:", sorted(p0.keys()))
            print(
                "  brand_name:",
                p0.get("brand_name"),
                "| generic_name:",
                p0.get("generic_name"),
                "| marketing_status:",
                p0.get("marketing_status"),
            )

        if subs:
            s0 = subs[0]
            print("submissions[0] keys:", sorted(s0.keys()))
            print(
                "  submission_status:",
                s0.get("submission_status"),
                "| submission_status_date:",
                s0.get("submission_status_date"),
                "| submission_type:",
                s0.get("submission_type"),
            )

        print("\n--- full JSON for this record (pretty) ---")
        blob = json.dumps(item, indent=2)
        print(blob[:8000])
        if len(blob) > 8000:
            print("\n... (truncated; full record longer than 8000 chars) ...")
        print()

    if args.score and scored_rows:
        scored_rows.sort(key=lambda x: x[0], reverse=True)
        print("--- ranked by significance score ---")
        print(
            "  "
            f"{'score':>6} {'tier':12} {'ORIG+AP':10} "
            f"{'application_number':18} {'brand':22} {'sym':5} {'6m→pre%':>8}"
        )
        ticker_cache: dict[str, str | None] = {}
        price_cache: dict[tuple[str, str], float | None] = {}

        for score_val, s_out in scored_rows:
            sponsor = (s_out.get("sponsor_name") or "").strip()
            # Derive the event date for ORIG+AP (YYYYMMDD -> date) from the scoring output.
            sub = s_out.get("latest_orig_ap_submission") or {}
            raw = sub.get("submission_status_date")
            event_date = None
            if isinstance(raw, str) and len(raw) == 8 and raw.isdigit():
                try:
                    event_date = datetime.strptime(raw, "%Y%m%d").date()
                except ValueError:
                    event_date = None

            sym = None
            if sponsor:
                sym = ticker_cache.get(sponsor)
                if sponsor not in ticker_cache:
                    sym_guess = None
                    sym0, how0 = match_company_to_symbol(sponsor)
                    if sym0:
                        # For exact/normalized mappings, trust the static map.
                        sym_guess = sym0
                    sym2, _how2 = yfinance_search_us_equity(sponsor)
                    if sym_guess is None and sym2 and symbol_name_plausible(sym2, sponsor):
                        sym_guess = sym2
                    sym = sym_guess
                    ticker_cache[sponsor] = sym

            pct_6m_to_pre = None
            if sym and event_date:
                cache_key = (sym, event_date.isoformat())
                if cache_key in price_cache:
                    pct_6m_to_pre = price_cache[cache_key]
                else:
                    px = price_stats_for_event(sym, event_date, min_year=None)
                    pct_6m_to_pre = px.get("pct_6m_to_pre")
                    price_cache[cache_key] = pct_6m_to_pre

            sym_disp = sym or "—"
            pct_disp = "—" if pct_6m_to_pre is None else f"{pct_6m_to_pre:+.1f}%"
            app_no_disp = (s_out.get("application_number") or "")[:18]
            brand_disp = (s_out.get("brand_name") or "")[:22]
            print(
                f"  {score_val:6.1f} {s_out['tier']:12} "
                f"{_fmt_orig_ap_date(s_out):10} "
                f"{app_no_disp:18} {brand_disp:22} {sym_disp:5} {pct_disp:>8}"
            )

    if args.update_ingredient_cache and results:
        keys_to_add: list[str] = []
        for item in results:
            for n in collect_active_ingredient_names(item.get("products")):
                keys_to_add.append(normalize_ingredient_key(n))
        ingredient_cache_merge(args.ingredient_cache, keys_to_add)
        print(f"Updated ingredient cache: {args.ingredient_cache}")


if __name__ == "__main__":
    main()
