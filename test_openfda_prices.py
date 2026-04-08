#!/usr/bin/env python3
"""
Fetch OpenFDA approvals, score them, map sponsor/manufacturer → ticker, then print
yfinance closes around the ORIG+AP date.

Scores match test_openfda.py --score when using the same preset/limit/resort and the
same ingredient cache file (default .cache/openfda_ingredients.json). Use
--skip-ingredient-cache to omit the +5 first_seen_ingredient term.

Resolution order (per company string):
  1) FDA advisor match_company_to_symbol (map → OpenFIGI → yfinance heuristics)
  2) If symbol fails a name plausibility check vs Yahoo metadata → Yahoo Search (US equity)
  3) If still none → small heuristics (e.g. mirati → MRTX)

Columns:
  6m→pre%   ~6 calendar months of trading days before last pre-event close
  pre→1st%  first close on/after ORIG date vs last close strictly before event
  pre→5td%  +5 trading sessions after that first post-event close vs pre

Requires Django (config.settings) for advisor maps / OpenFIGI. .env for API keys.

Usage:
  python test_openfda_prices.py
  python test_openfda_prices.py --min-tier moderate --limit 25 --resort-orig
  python test_openfda_prices.py --min-year 1990   # widen yfinance (noisy pre-2000)
  python test_openfda_prices.py --limit 100 --pages 3 --min-tier moderate
  python test_openfda_prices.py --recent-months 0   # all dates (no recency filter)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yfinance as yf
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

from core.services.openfda.client import (
    MAX_DRUGSFDA_LIMIT,
    fetch_drugsfda_paginated,
    get_openfda_api_key,
)
from core.services.market.yahoo_event_window import (
    default_min_event_year,
    price_stats_for_event,
    symbol_name_plausible,
    yfinance_search_us_equity,
)
from core.services.openfda.significance import (
    ingredient_cache_load,
    latest_orig_ap_event_date,
    recent_orig_ap_cutoff_months,
    score_drugsfda_record,
)


def setup_django():
    import django
    from django.conf import settings

    if settings.configured:
        return
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    django.setup()


def _openfda_text_blob(record: dict) -> str:
    """Flatten openfda + sponsor strings for keyword heuristics."""
    parts: list[str] = []
    sp = record.get("sponsor_name")
    if isinstance(sp, str):
        parts.append(sp)
    o = record.get("openfda")
    if isinstance(o, dict):
        for key in (
            "manufacturer_name",
            "brand_name",
            "generic_name",
            "substance_name",
        ):
            v = o.get(key)
            if isinstance(v, list):
                parts.extend(str(x) for x in v if x)
            elif isinstance(v, str):
                parts.append(v)
    return " ".join(parts)


def _is_bristol_stub_sponsor(sponsor_upper: str) -> bool:
    """Sponsor like 'BRISTOL' (not full Bristol Myers Squibb) — use with Mirati mfr."""
    s = sponsor_upper.strip()
    if not s:
        return False
    if s == "BRISTOL":
        return True
    return s.startswith("BRISTOL ") and "MYERS" not in s


def company_name_candidates(record: dict) -> list[str]:
    """
    Order for price attribution: sponsor / marketing holder first (fixes CMO listed
    before Janssen → wrong THE ticker), then manufacturers.

    If sponsor is a short 'BRISTOL' stub and a manufacturer is Mirati, try Mirati
    first so pre-acquisition tape maps to MRTX.
    """
    seen: set[str] = set()
    openfda = record.get("openfda") if isinstance(record.get("openfda"), dict) else {}
    mfrs: list[str] = []
    for m in openfda.get("manufacturer_name") or []:
        if isinstance(m, str) and m.strip():
            mfrs.append(m.strip())

    sp = record.get("sponsor_name")
    sponsor = sp.strip() if isinstance(sp, str) and sp.strip() else ""
    sponsor_u = sponsor.upper()

    mirati_mfrs = [m for m in mfrs if "MIRATI" in m.upper()]
    ordered: list[str] = []

    if mirati_mfrs and _is_bristol_stub_sponsor(sponsor_u):
        for m in mirati_mfrs:
            u = m.upper()
            if u not in seen:
                seen.add(u)
                ordered.append(m)

    if sponsor:
        u = sponsor.upper()
        if u not in seen:
            seen.add(u)
            ordered.append(sponsor)

    for m in mfrs:
        u = m.upper()
        if u not in seen:
            seen.add(u)
            ordered.append(m)

    return ordered


def tier_rank(tier: str) -> int:
    return {"low": 0, "moderate": 1, "significant": 2}.get(tier, 0)


def main():
    parser = argparse.ArgumentParser(
        description="OpenFDA significant approvals + ticker + price path around ORIG+AP date"
    )
    parser.add_argument(
        "--min-tier",
        choices=("significant", "moderate", "low"),
        default="significant",
        help="Minimum significance tier to include (default: significant)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help=f"Page size per OpenFDA request (max {MAX_DRUGSFDA_LIMIT})",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="OpenFDA pages to merge (skip = page_index × limit). Default: 1",
    )
    parser.add_argument(
        "--resort-orig",
        action="store_true",
        help="Sort results by newest ORIG+AP date before scoring",
    )
    parser.add_argument(
        "--preset",
        choices=("orig-ap", "prescription"),
        default="orig-ap",
        help="OpenFDA search preset (default: orig-ap)",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        metavar="YEAR",
        help=(
            "Skip yfinance windows for event dates before this calendar year "
            "(default: current year − 30)"
        ),
    )
    parser.add_argument(
        "--ingredient-cache",
        type=Path,
        default=BASE_DIR / ".cache" / "openfda_ingredients.json",
        help="Same JSON cache as test_openfda.py for first_seen scoring (default: .cache/...)",
    )
    parser.add_argument(
        "--skip-ingredient-cache",
        action="store_true",
        help="Do not load cache — omits first_seen_ingredient score (+5); tiers may shift",
    )
    parser.add_argument(
        "--recent-months",
        type=int,
        default=6,
        metavar="N",
        help="Only latest ORIG+AP dates on or after (today − N calendar months); 0 = no filter (default: 6).",
    )
    args = parser.parse_args()
    if args.pages < 1:
        parser.error("--pages must be >= 1")

    min_year = args.min_year if args.min_year is not None else default_min_event_year()

    setup_django()

    from core.services.advisors.fda import match_company_to_symbol

    PRESETS = {
        "prescription": (
            "products.marketing_status:Prescription",
            "submissions.submission_status_date:desc",
        ),
        "orig-ap": (
            "submissions.submission_status:AP AND submissions.submission_type:ORIG",
            "submissions.submission_status_date:desc",
        ),
    }

    search, sort = PRESETS[args.preset]

    def max_orig_ap_date(item):
        best = ""
        for s in item.get("submissions") or []:
            if (s.get("submission_type") or "").upper() != "ORIG":
                continue
            if (s.get("submission_status") or "").upper() != "AP":
                continue
            d = s.get("submission_status_date") or ""
            if d > best:
                best = d
        return best

    api_key = get_openfda_api_key()
    if args.limit > MAX_DRUGSFDA_LIMIT:
        print(
            f"(note: --limit {args.limit} clipped to {MAX_DRUGSFDA_LIMIT} per request)",
            file=sys.stderr,
        )
    try:
        results, _meta = fetch_drugsfda_paginated(
            search=search,
            sort=sort,
            limit=args.limit,
            pages=args.pages,
            api_key=api_key,
        )
    except Exception as e:
        print("OpenFDA request failed:", e, file=sys.stderr)
        sys.exit(1)
    fetched_count = len(results)
    if args.resort_orig:
        results.sort(key=max_orig_ap_date, reverse=True)

    cutoff = recent_orig_ap_cutoff_months(args.recent_months)
    if cutoff is not None:
        results = [
            x
            for x in results
            if (ev0 := latest_orig_ap_event_date(x)) is not None and ev0 >= cutoff
        ]

    min_rank = tier_rank(args.min_tier)
    seen_ingredients: set[str] | None = None
    if not args.skip_ingredient_cache:
        seen_ingredients = ingredient_cache_load(args.ingredient_cache)

    rows_out: list[dict] = []

    for item in results:
        s_out = score_drugsfda_record(item, seen_ingredient_keys=seen_ingredients)
        if tier_rank(s_out["tier"]) < min_rank:
            continue
        ev = latest_orig_ap_event_date(item)
        if not ev:
            continue

        symbol = None
        match_how = None
        for cand in company_name_candidates(item):
            sym, how = match_company_to_symbol(cand)
            trusted = how in ("exact", "normalized")
            if sym and (trusted or symbol_name_plausible(sym, cand)):
                symbol, match_how = sym, f"{how} ({cand[:40]})"
                break
            if sym:
                # Bad match (e.g. wrong yfinance heuristic) — try Yahoo search
                sym2, how2 = yfinance_search_us_equity(cand)
                if sym2 and symbol_name_plausible(sym2, cand):
                    symbol, match_how = sym2, how2 or "yfinance_search"
                    break

        if not symbol:
            for cand in company_name_candidates(item):
                sym, how = yfinance_search_us_equity(cand)
                if sym and symbol_name_plausible(sym, cand):
                    symbol, match_how = sym, how or "yfinance_search"
                    break

        # Post-M&A / weak Yahoo metadata: extend as you discover bad auto-matches
        if not symbol or (
            symbol == "BMY"
            and _is_bristol_stub_sponsor(
                (item.get("sponsor_name") or "").upper().strip()
            )
        ):
            blob = _openfda_text_blob(item).lower()
            if "mirati" in blob:
                symbol, match_how = "MRTX", "heuristic (mirati→MRTX)"

        if not symbol:
            rows_out.append(
                {
                    "s_out": s_out,
                    "event": ev,
                    "symbol": None,
                    "match_how": None,
                    "px": None,
                }
            )
            continue

        px = price_stats_for_event(symbol, ev, min_year=min_year)
        if px.get("error") and symbol == "MRTX":
            px_bmy = price_stats_for_event("BMY", ev, min_year=min_year)
            if not px_bmy.get("error"):
                symbol, match_how = (
                    "BMY",
                    f"{match_how} + BMY history (MRTX inactive on Yahoo)",
                )
                px = px_bmy
        rows_out.append(
            {
                "s_out": s_out,
                "event": ev,
                "symbol": symbol,
                "match_how": match_how,
                "px": px,
            }
        )

    # --- print ---
    cut_note = f" cutoff>={cutoff}" if cutoff else ""
    print(
        f"min_tier={args.min_tier} preset={args.preset} "
        f"limit={min(args.limit, MAX_DRUGSFDA_LIMIT)} pages={args.pages} "
        f"recent_months={args.recent_months}{cut_note} "
        f"fetched={fetched_count} after_recent_filter={len(results)} table_rows={len(rows_out)}\n"
    )
    hdr = (
        f"{'score':>6} {'tier':12} {'event':10} {'sym':6} "
        f"{'6m→pre%':>8} {'pre→1st%':>9} {'pre→5td%':>9} brand / sponsor"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in sorted(rows_out, key=lambda x: x["s_out"]["score"], reverse=True):
        s_out = r["s_out"]
        evs = r["event"].isoformat()
        sym = r["symbol"] or "—"
        brand = s_out.get("brand_name") or "?"
        sponsor = s_out.get("sponsor_name") or ""
        px = r["px"]

        if not r["symbol"]:
            print(
                f"{s_out['score']:6.1f} {s_out['tier']:12} {evs} {sym:6} "
                f"{'—':>8} {'—':>9} {'—':>9} {brand} ({sponsor}) [no ticker]"
            )
            continue

        if px and px.get("error"):
            print(
                f"{s_out['score']:6.1f} {s_out['tier']:12} {evs} {sym:6} "
                f"{'—':>8} {'—':>9} {'—':>9} {brand} — {px['error']}"
            )
            continue

        def fmt_pct(v):
            if v is None:
                return "—"
            return f"{v:+.1f}%"

        print(
            f"{s_out['score']:6.1f} {s_out['tier']:12} {evs} {sym:6} "
            f"{fmt_pct(px['pct_6m_to_pre']):>8} {fmt_pct(px['pct_pre_to_post']):>9} "
            f"{fmt_pct(px['pct_pre_to_5td']):>9} {brand}"
        )


if __name__ == "__main__":
    main()
