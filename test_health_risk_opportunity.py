#!/usr/bin/env python3
"""
Risk vs opportunity lab. Reuses v2 component scorers unchanged; adds stability /
opportunity blends for experimentation before core changes.

Mode A — single ticker (no Django):
    source ~/Development/scratch/python/tutorial-env/bin/activate
    python test_health_risk_opportunity.py AAPL
    python test_health_risk_opportunity.py VRAX APPS --detail
    python test_health_risk_opportunity.py MSFT --json
    python test_health_risk_opportunity.py LCID MCD MSFT

Mode B — discovery autopsy (Django + yfinance forward returns):
    python test_health_risk_opportunity.py --from-date 2026-05-15 --days 30 --advisor Polygon.io
    python test_health_risk_opportunity.py --from-date 2026-05-25 --days 10 --horizon 5 --table
    python test_health_risk_opportunity.py --from-date 2026-05-15 --days 30 --csv risk_autopsy.csv

Fund matrix uses opportunity × discovery.weight (weight applies to opportunity only).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_health_v2_autopsy import (
    DEFAULT_HORIZON_TRADING_DAYS,
    CohortWindow,
    _cohort_window,
    _entry_index,
    _fetch_close_series,
    _filter_discoveries_in_window,
    _forward_return,
    _setup_django,
)

from core.services.health.assess import COMPONENT_MODEL_WEIGHTS, composite_from_scores
from core.services.health.durability import score_business_durability
from core.services.health.risk_matrix import (
    OPPORTUNITY_WEIGHTS,
    RISK_LEVELS,
    RISK_MATRIX,
    STABILITY_WEIGHTS,
    interpret_axes as interpret,
    opportunity_adjusted,
    risk_fit_all as fund_fit_all,
    risk_fit_for as fund_fit,
    risk_floors_for,
)
from core.services.health.so_ratings import (
    score_to_opportunity_grade,
    score_to_stability_grade,
    so_grade_pair,
)
from core.services.health.consensus import score_consensus_health
from core.services.health.financial import score_financial_health
from core.services.health.intrinsic import score_intrinsic_health
from core.services.health.price import score_price_health
from core.services.health.ratings import score_to_rating
from core.services.health.sector import score_sector_health
from core.services.health.valuation import score_valuation_health

# Layer-2 weights: imported from risk_matrix (single source of truth).

BLEND_STABILITY_WEIGHT = 0.40
BLEND_OPPORTUNITY_WEIGHT = 0.60

# Alias lab names to core risk matrix (single source of truth).
FUND_MATRIX = RISK_MATRIX
FUND_PROFILE_ORDER: Tuple[str, ...] = RISK_LEVELS

COMPONENT_SCORERS = {
    "financial": score_financial_health,
    "valuation": score_valuation_health,
    "intrinsic": score_intrinsic_health,
    "price": score_price_health,
    "consensus": score_consensus_health,
    "sector": score_sector_health,
}

COMPONENT_TITLES = {
    "financial": "Financial health",
    "valuation": "Valuation",
    "intrinsic": "Intrinsic valuation",
    "price": "Price position",
    "consensus": "Analyst consensus",
    "sector": "Sector / industry",
}

STORED_COMPONENT_KEYS: Tuple[str, ...] = (
    "financial",
    "valuation",
    "intrinsic",
    "price",
    "consensus",
    "sector",
)


@dataclass
class StoredComponent:
    score: Optional[float]
    metrics: List[Any] = field(default_factory=list)


def _metric_score(result: Any, key: str) -> Optional[float]:
    for m in getattr(result, "metrics", []) or []:
        if m.key == key and m.score is not None:
            return float(m.score)
    return None


def _weighted_blend(parts: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    num = 0.0
    den = 0.0
    for key, w in weights.items():
        sc = parts.get(key)
        if sc is None:
            continue
        num += float(sc) * w
        den += w
    if den <= 0:
        return None
    return round(num / den, 1)


def _fin_growth_opportunity(financial: Any) -> Optional[float]:
    rev = _metric_score(financial, "revenue_growth")
    eps = _metric_score(financial, "eps_growth")
    roe = _metric_score(financial, "return_on_equity")
    vals = [v for v in (rev, eps, roe) if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 1)


def stability_parts(results: Dict[str, Any]) -> Dict[str, Optional[float]]:
    fin = results["financial"]
    sym = getattr(fin, "symbol", "") or ""
    return {
        "sector": results["sector"].score,
        "fin_debt_to_equity": _metric_score(fin, "debt_to_equity"),
        "fin_fcf_margin": _metric_score(fin, "fcf_margin"),
        "fin_operating_margin": _metric_score(fin, "operating_margin"),
        "durability": score_business_durability(sym) if sym else None,
    }


def opportunity_parts(results: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "price": results["price"].score,
        "valuation": results["valuation"].score,
        "intrinsic": results["intrinsic"].score,
        "consensus": results["consensus"].score,
        "fin_growth": _fin_growth_opportunity(results["financial"]),
    }


def compute_axes(results: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    stability = _weighted_blend(stability_parts(results), STABILITY_WEIGHTS)
    opportunity = _weighted_blend(opportunity_parts(results), OPPORTUNITY_WEIGHTS)
    return stability, opportunity


def legacy_composite(results: Dict[str, Any]) -> Optional[float]:
    scores = {key: r.score for key, r in results.items()}
    comp = composite_from_scores(scores)
    return float(comp) if comp is not None else None


def blended_score(stability: Optional[float], opportunity: Optional[float]) -> Optional[float]:
    if stability is None and opportunity is None:
        return None
    s = stability if stability is not None else 50.0
    o = opportunity if opportunity is not None else 50.0
    w_s = BLEND_STABILITY_WEIGHT
    w_o = BLEND_OPPORTUNITY_WEIGHT
    if stability is None:
        w_s, w_o = 0.0, 1.0
    elif opportunity is None:
        w_s, w_o = 1.0, 0.0
    return round((s * w_s + o * w_o) / (w_s + w_o), 1)


def qualitative_band(score: Optional[float]) -> str:
    if score is None:
        return "—"
    if score >= 80:
        return "Strong"
    if score >= 65:
        return "Medium"
    if score >= 50:
        return "Low–Medium"
    if score >= 35:
        return "Weak–Medium"
    return "Weak"


def run_symbol(symbol: str) -> Dict[str, Any]:
    sym = symbol.strip().upper()
    results = {key: scorer(sym) for key, scorer in COMPONENT_SCORERS.items()}
    stability, opportunity = compute_axes(results)
    legacy = legacy_composite(results)
    blend = blended_score(stability, opportunity)
    rating = score_to_rating(legacy)
    return {
        "symbol": sym,
        "legacy_composite": legacy,
        "legacy_grade": rating.letter if rating else None,
        "stability": stability,
        "stability_band": qualitative_band(stability),
        "opportunity": opportunity,
        "opportunity_band": qualitative_band(opportunity),
        "blended_40_60": blend,
        "interpretation": interpret(stability, opportunity),
        "fund_fit": fund_fit_all(stability, opportunity),
        "fund_matrix": FUND_MATRIX,
        "stability_parts": stability_parts(results),
        "opportunity_parts": opportunity_parts(results),
        "components": {k: r.to_dict() for k, r in results.items()},
    }


def _stored_score(assessment: Any, key: str) -> Optional[float]:
    raw = getattr(assessment, key, None)
    return float(raw) if raw is not None else None


def component_results_from_assessment(
    assessment: Any,
    symbol: str,
    fin_cache: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Cohort autopsy: stored Assessment columns for six components; re-run financial
    scorer for sub-metrics (debt, FCF, growth) not persisted on Assessment.
    """
    sym = symbol.strip().upper()
    if sym not in fin_cache:
        fin_cache[sym] = score_financial_health(sym)
    fin = fin_cache[sym]
    results: Dict[str, Any] = {"financial": fin}
    for key in STORED_COMPONENT_KEYS:
        if key == "financial":
            stored_fin = _stored_score(assessment, "financial")
            if stored_fin is not None:
                fin.score = stored_fin
            continue
        results[key] = StoredComponent(score=_stored_score(assessment, key))
    return results


@dataclass
class AutopsyRow:
    discovery_id: int
    created: date
    symbol: str
    advisor: str
    weight: float
    legacy_composite: Optional[float]
    legacy_grade: str
    stability: Optional[float]
    opportunity: Optional[float]
    opportunity_adjusted: Optional[float]
    blended_40_60: Optional[float]
    entry_price: float


def _load_autopsy_rows(
    window: CohortWindow,
    advisor: Optional[str],
) -> Tuple[List[AutopsyRow], int]:
    from core.discovery_scoring import discovery_scoring_context
    from core.models import Discovery

    qs = _filter_discoveries_in_window(
        Discovery.objects.filter(assessment__isnull=False),
        window,
    ).select_related("stock", "advisor", "assessment").order_by("created")
    if advisor:
        qs = qs.filter(advisor__name=advisor)

    fin_cache: Dict[str, Any] = {}
    rows: List[AutopsyRow] = []
    skipped = 0
    for d in qs:
        symbol = (d.stock.symbol or "").strip().upper()
        if not symbol:
            skipped += 1
            continue
        ctx = discovery_scoring_context(d)
        if ctx.get("source") != "v2":
            skipped += 1
            continue
        entry = d.price
        if entry is None or float(entry) <= 0:
            skipped += 1
            continue

        assessment = d.assessment
        results = component_results_from_assessment(assessment, symbol, fin_cache)
        stability, opportunity = compute_axes(results)
        if stability is None and opportunity is None:
            skipped += 1
            continue

        legacy = _stored_score(assessment, "score")
        if legacy is None:
            legacy = legacy_composite(results)
        w = float(d.weight) if d.weight is not None else 1.0
        opp_adj = opportunity_adjusted(opportunity, w)
        rows.append(
            AutopsyRow(
                discovery_id=d.id,
                created=d.created.date(),
                symbol=symbol,
                advisor=d.advisor.name,
                weight=w,
                legacy_composite=legacy,
                legacy_grade=score_to_rating(legacy).letter if legacy is not None else "—",
                stability=stability,
                opportunity=opportunity,
                opportunity_adjusted=opp_adj,
                blended_40_60=blended_score(stability, opportunity),
                entry_price=float(entry),
            )
        )
    return rows, skipped


def _ret_column(horizon: int) -> str:
    return f"ret_{horizon}d"


def _summarize_score_buckets(
    df: pd.DataFrame,
    score_col: str,
    ret_col: str,
    title: str,
    *,
    abs_return: bool = False,
) -> None:
    if df.empty or score_col not in df.columns:
        return
    valid = df.dropna(subset=[score_col, ret_col])
    if valid.empty:
        print(f"\n=== {title} ===")
        print("  No rows with score and return.")
        return

    n = len(valid)
    try:
        valid = valid.copy()
        valid["_bucket"] = pd.qcut(valid[score_col], q=4, duplicates="drop")
    except ValueError:
        valid["_bucket"] = pd.cut(valid[score_col], bins=3, duplicates="drop")

    print(f"\n=== {title} ===")
    for bucket, g in valid.groupby("_bucket", observed=True):
        vals = g[ret_col]
        if abs_return:
            vals = vals.abs()
        pos = int((g[ret_col] > 0).sum())
        print(
            f"  {bucket}: n={len(g):>3}  mean={vals.mean():+6.2f}%  "
            f"median={vals.median():+6.2f}%  positive={pos}/{len(g)} ({100 * pos / len(g):.0f}%)"
        )
    print(f"  (total with returns: {n})")


def _summarize_fund_fit(df: pd.DataFrame, ret_col: str, horizon: int) -> None:
    if df.empty:
        return
    print(f"\n=== Fund matrix (opp × weight) — {horizon}d forward return ===")
    for profile in FUND_PROFILE_ORDER:
        col = f"fit_{profile.lower()}"
        floors = risk_floors_for(profile)
        for decision in ("BUY", "AVOID"):
            g = df[df[col] == decision]
            vals = g[ret_col].dropna()
            if len(vals) == 0:
                print(f"  {profile:<14} {decision:<5} n=0")
                continue
            pos = int((vals > 0).sum())
            print(
                f"  {profile:<14} {decision:<5} n={len(vals):>3}  mean={vals.mean():+6.2f}%  "
                f"median={vals.median():+6.2f}%  positive={pos}/{len(vals)} ({100 * pos / len(vals):.0f}%)"
            )
        print(f"    floor SO grade: >={floors['so_floor_display']}")


def _summarize_by_grade(df: pd.DataFrame, ret_col: str, horizon: int) -> None:
    if df.empty:
        return
    print(f"\n=== Legacy grade — {horizon}d forward return ===")
    for grade, g in df.groupby("legacy_grade", dropna=False):
        vals = g[ret_col].dropna()
        if len(vals) == 0:
            print(f"  {grade}: n=0")
            continue
        pos = int((vals > 0).sum())
        print(
            f"  {grade}: n={len(vals):>3}  mean={vals.mean():+6.2f}%  "
            f"median={vals.median():+6.2f}%  positive={pos}/{len(vals)} ({100 * pos / len(vals):.0f}%)"
        )


def _print_autopsy_table(df: pd.DataFrame, limit: int, ret_col: str) -> None:
    if df.empty:
        print("\nNo rows to display.")
        return
    cols = [
        "created",
        "symbol",
        "advisor",
        "legacy_grade",
        "stability",
        "opportunity",
        "opp_adjusted",
        "weight",
        "fit_conservative",
        "fit_moderate",
        "fit_aggressive",
        "entry_price",
        ret_col,
    ]
    table = df.copy()
    for c in cols:
        if c not in table.columns:
            table[c] = None
    table = table.sort_values(by="created", ascending=False)
    if limit > 0:
        table = table.head(limit)
    print(f"\n=== Per-discovery ({len(table)} rows) ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(table[cols].to_string(index=False))


def run_autopsy(args: argparse.Namespace) -> None:
    from_date = date.fromisoformat(args.from_date)
    if args.days <= 0:
        raise SystemExit("--days must be positive")
    if args.horizon <= 0:
        raise SystemExit("--horizon must be a positive integer")

    horizon = args.horizon
    ret_col = _ret_column(horizon)

    _setup_django()
    window = _cohort_window(from_date, args.days, args.window_tz)
    rows_in, skip_load = _load_autopsy_rows(window, args.advisor)

    print(
        f"Risk / opportunity autopsy — discoveries {from_date} .. {window.to_date} "
        f"({window.label}) (assessment required, horizon={horizon}td)"
    )
    if args.advisor:
        print(f"  advisor filter: {args.advisor}")
    print(f"  loaded: {len(rows_in)} discoveries")
    if skip_load:
        print(f"  skipped at load: {skip_load} (no symbol / not v2 / no price / no axes)")
    if not rows_in:
        print("  No rows to autopsy — widen --from-date/--days or check advisor filter.")
        return

    start = min(r.created for r in rows_in) - timedelta(days=5)
    end = max(r.created for r in rows_in) + timedelta(days=horizon + 20)
    cache: Dict[Tuple[str, date, date], pd.Series] = {}

    records: List[dict] = []
    skip_no_hist = 0
    skip_no_entry = 0
    skip_incomplete = 0
    for r in rows_in:
        close = _fetch_close_series(r.symbol, start, end, cache)
        if close is None:
            skip_no_hist += 1
            continue
        eidx = _entry_index(close, r.created)
        if eidx is None:
            skip_no_entry += 1
            continue
        ret = _forward_return(close, eidx, r.entry_price, horizon)
        if ret is None:
            skip_incomplete += 1
            continue

        fit = fund_fit_all(r.stability, r.opportunity, weight=r.weight)
        records.append(
            {
                "discovery_id": r.discovery_id,
                "created": r.created.isoformat(),
                "symbol": r.symbol,
                "advisor": r.advisor,
                "legacy_composite": r.legacy_composite,
                "legacy_grade": r.legacy_grade,
                "stability": r.stability,
                "opportunity": r.opportunity,
                "opp_adjusted": r.opportunity_adjusted,
                "blended_40_60": r.blended_40_60,
                "weight": r.weight,
                "fit_conservative": fit["CONSERVATIVE"],
                "fit_moderate": fit["MODERATE"],
                "fit_aggressive": fit["AGGRESSIVE"],
                "entry_price": round(r.entry_price, 4),
                ret_col: round(ret, 2),
            }
        )

    df = pd.DataFrame(records)
    skipped_total = skip_no_hist + skip_no_entry + skip_incomplete
    if skipped_total:
        print(f"  skipped (returns): {skipped_total}")
        if args.verbose:
            print(f"    no yfinance history: {skip_no_hist}")
            print(f"    no entry bar on/after created date: {skip_no_entry}")
            print(f"    incomplete {horizon} trading-day horizon: {skip_incomplete}")

    if df.empty:
        print(f"  No complete {horizon}d returns in cohort.")
        return

    print("\n=== Overall ===")
    vals = df[ret_col].dropna()
    pos = int((vals > 0).sum())
    print(
        f"  rows={len(df)}  with_ret={len(vals)}  mean={vals.mean():+6.2f}%  "
        f"median={vals.median():+6.2f}%  positive={pos}/{len(vals)} ({100 * pos / len(vals):.0f}%)"
    )

    _summarize_by_grade(df, ret_col, horizon)
    _summarize_score_buckets(
        df, "stability", ret_col, f"By stability quartile — |{horizon}d return|", abs_return=True
    )
    _summarize_score_buckets(
        df, "stability", ret_col, f"By stability quartile — signed {horizon}d return"
    )
    _summarize_score_buckets(
        df, "opportunity", ret_col, f"By opportunity quartile — signed {horizon}d return"
    )
    _summarize_score_buckets(
        df, "opp_adjusted", ret_col, f"By opp×weight quartile — signed {horizon}d return"
    )
    _summarize_fund_fit(df, ret_col, horizon)

    if args.table:
        _print_autopsy_table(df, args.table_limit, ret_col)

    if args.csv:
        out_path = Path(args.csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved rows to: {out_path}")


def _print_report(payload: Dict[str, Any], detail: bool) -> None:
    sym = payload["symbol"]
    print(f"\n{'=' * 60}")
    print(f"Risk / opportunity lab — {sym}")
    print(f"{'=' * 60}")
    leg = payload["legacy_composite"]
    leg_s = f"{leg:.1f}" if leg is not None else "—"
    grade = payload["legacy_grade"] or "—"
    print(f"  Legacy assessment:  {leg_s}  (grade {grade})")
    stab = payload["stability"]
    opp = payload["opportunity"]
    stab_s = f"{stab:.1f}" if stab is not None else "—"
    opp_s = f"{opp:.1f}" if opp is not None else "—"
    blend = payload["blended_40_60"]
    blend_s = f"{blend:.1f}" if blend is not None else "—"
    so_grade = so_grade_pair(
        score_to_stability_grade(stab), score_to_opportunity_grade(opp)
    ) or "—"
    print(f"  SO grade:           {so_grade}")
    print(f"  Stability:          {stab_s}  ({payload['stability_band']})")
    print(f"  Opportunity:        {opp_s}  ({payload['opportunity_band']})")
    print(f"  Blended (40/60):    {blend_s}")
    print(f"\n  Interpretation:")
    print(f"    {payload['interpretation']}")
    print()
    print("  Fund fit (matrix; opp × weight=1.0):")
    for profile in FUND_PROFILE_ORDER:
        floors = risk_floors_for(profile)
        decision = payload["fund_fit"][profile]
        print(
            f"    {profile:<14} {decision:<5}  "
            f"(floor SO>={floors['so_floor_display']})"
        )
    print()

    print(f"  {'Axis':<12} {'Input':<28} {'Score':>8}  {'Wt':>5}")
    print(f"  {'-' * 12} {'-' * 28} {'-' * 8}  {'-' * 5}")
    for key, w in STABILITY_WEIGHTS.items():
        sc = payload["stability_parts"].get(key)
        sc_s = f"{sc:.1f}" if sc is not None else "—"
        print(f"  {'stability':<12} {key:<28} {sc_s:>8}  {w * 100:>4.0f}%")
    for key, w in OPPORTUNITY_WEIGHTS.items():
        sc = payload["opportunity_parts"].get(key)
        sc_s = f"{sc:.1f}" if sc is not None else "—"
        print(f"  {'opportunity':<12} {key:<28} {sc_s:>8}  {w * 100:>4.0f}%")
    print()

    print(f"  {'Component':<26} {'Score':>8}  {'Legacy wt':>9}")
    print(f"  {'-' * 26} {'-' * 8}  {'-' * 9}")
    for key in COMPONENT_SCORERS:
        comp = payload["components"][key]
        sc = comp.get("score")
        sc_s = f"{sc:.1f}" if sc is not None else "—"
        lw = float(COMPONENT_MODEL_WEIGHTS[key]) * 100
        print(f"  {COMPONENT_TITLES[key]:<26} {sc_s:>8}  {lw:>8.0f}%")
    print()

    if detail:
        for key in COMPONENT_SCORERS:
            comp = payload["components"][key]
            title = COMPONENT_TITLES[key]
            print(f"{'-' * 60}")
            print(f"{title} — {sym}")
            if comp.get("error") and comp.get("score") is None:
                print(f"  ERROR: {comp['error']}")
                continue
            print(f"  Score: {comp.get('score')}")
            for m in comp.get("metrics") or []:
                sc = m.get("score")
                sc_s = f"{sc:.1f}" if sc is not None else "—"
                print(f"    {m.get('label', m.get('key')):<26} {m.get('raw_display', ''):>22}  {sc_s:>6}")
            print()


def _run_symbols(args: argparse.Namespace) -> None:
    if not args.symbols:
        raise SystemExit("Mode A requires at least one ticker symbol.")

    payloads: List[Dict[str, Any]] = []
    any_failed = False

    for sym in args.symbols:
        try:
            payload = run_symbol(sym)
            payloads.append(payload)
            if payload["legacy_composite"] is None and payload["stability"] is None:
                any_failed = True
            if not args.json:
                _print_report(payload, args.detail)
        except Exception as exc:
            any_failed = True
            if args.json:
                payloads.append({"symbol": sym.strip().upper(), "error": str(exc)})
            else:
                print(f"ERROR {sym}: {exc}", file=sys.stderr)

    if args.json:
        print(json.dumps(payloads if len(payloads) != 1 else payloads[0], indent=2))

    if any_failed:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stability vs opportunity lab (single ticker or discovery autopsy)."
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Mode A: ticker symbol(s). Omit when using --from-date (Mode B).",
    )
    parser.add_argument("--detail", action="store_true", help="Mode A: full component metric breakdown")
    parser.add_argument("--json", action="store_true", help="Mode A: JSON output")

    parser.add_argument(
        "--from-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Mode B: cohort start (inclusive)",
    )
    parser.add_argument("--days", type=int, default=30, help="Mode B: cohort length in calendar days")
    parser.add_argument("--advisor", default=None, help="Mode B: filter by advisor name")
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON_TRADING_DAYS,
        help=f"Mode B: forward return in trading days (default {DEFAULT_HORIZON_TRADING_DAYS})",
    )
    parser.add_argument("--table", action="store_true", help="Mode B: per-discovery table")
    parser.add_argument(
        "--table-limit",
        type=int,
        default=50,
        help="Mode B: max table rows (0 = all; default 50)",
    )
    parser.add_argument("--csv", default=None, help="Mode B: optional CSV output path")
    parser.add_argument("--verbose", action="store_true", help="Mode B: print return skip breakdown")
    parser.add_argument(
        "--window-tz",
        default=None,
        metavar="ZONE",
        help="Mode B: cohort by calendar date in timezone (e.g. Europe/Dublin)",
    )

    args = parser.parse_args()

    if args.from_date:
        run_autopsy(args)
    else:
        _run_symbols(args)


if __name__ == "__main__":
    main()
