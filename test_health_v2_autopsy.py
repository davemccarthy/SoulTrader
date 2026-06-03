#!/usr/bin/env python3
"""
Health v2 assessment autopsy: discovery cohort vs 7-trading-day forward return.

Only discoveries with a linked Assessment (v2). Rows without a full
7 trading-day forward window are skipped when returns are computed (see --verbose).

Grade/score source (default: stored full-model composite):
  --score-component price     # 100% Price position (assessment.price)
  --weights price=1.0         # custom mix from stored component columns
  --weights price=0.5,financial=0.5

Examples:
  python test_health_v2_autopsy.py --from-date 2026-05-15 --days 30 --advisor Polygon.io
  python test_health_v2_autopsy.py --from-date 2026-05-15 --days 30 --score-component price --table
  python test_health_v2_autopsy.py --from-date 2026-05-15 --days 30 --weights price=1.0
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

HORIZON_TRADING_DAYS = 7

COMPONENT_KEYS: Tuple[str, ...] = (
    "financial",
    "valuation",
    "intrinsic",
    "price",
    "consensus",
    "sector",
)

COMPONENT_LABELS: Dict[str, str] = {
    "financial": "Financial health",
    "valuation": "Valuation",
    "intrinsic": "Intrinsic valuation",
    "price": "Price position",
    "consensus": "Analyst consensus",
    "sector": "Sector / industry",
}


def _setup_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()


def _cohort_window(from_date: date, days: int) -> Tuple[datetime, datetime, date]:
    """
    Inclusive start, exclusive end as aware datetimes (UTC with default settings).

    Matches SQL ``created >= from_date`` and ``created < from_date + days`` at midnight,
    instead of ``created__date`` which can drop rows when DB timestamps are UTC+1.
    """
    from django.utils import timezone as dj_tz

    to_date = from_date + timedelta(days=days)
    start = dj_tz.make_aware(datetime.combine(from_date, time.min))
    end = dj_tz.make_aware(datetime.combine(to_date, time.min))
    return start, end, to_date


def _fetch_close_series(
    symbol: str,
    start_date: date,
    end_date: date,
    cache: Dict[Tuple[str, date, date], pd.Series],
) -> Optional[pd.Series]:
    key = (symbol, start_date, end_date)
    if key in cache:
        return cache[key]

    try:
        hist = yf.Ticker(symbol).history(
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=True,
        )
    except Exception:
        cache[key] = None
        return None

    if hist is None or hist.empty or "Close" not in hist.columns:
        cache[key] = None
        return None

    close = hist["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")]
    cache[key] = close
    return close


def _entry_index(close: pd.Series, anchor: date) -> Optional[int]:
    if close is None or close.empty:
        return None
    pos = close.index.searchsorted(pd.Timestamp(anchor), side="left")
    if pos >= len(close):
        return None
    return int(pos)


def _forward_return(
    close: pd.Series,
    entry_idx: int,
    entry_price: float,
    trading_days: int,
) -> Optional[float]:
    target_idx = entry_idx + trading_days
    if target_idx >= len(close):
        return None
    if entry_price <= 0:
        return None
    exit_price = float(close.iloc[target_idx])
    return (exit_price - entry_price) / entry_price * 100.0


def _parse_weights(value: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid weight segment {part!r}; use key=0.2")
        key, raw_w = part.split("=", 1)
        key = key.strip().lower()
        if key not in COMPONENT_KEYS:
            raise ValueError(
                f"Unknown component {key!r}; choose from: {', '.join(COMPONENT_KEYS)}"
            )
        w = float(raw_w.strip())
        if w < 0:
            raise ValueError(f"Weight for {key} must be non-negative")
        weights[key] = w
    if not weights:
        raise ValueError("No weights provided")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    if abs(total - 1.0) > 1e-6:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def _assessment_component_scores(assessment) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key in COMPONENT_KEYS:
        raw = getattr(assessment, key, None)
        out[key] = float(raw) if raw is not None else None
    return out


def _composite_from_stored(
    scores: Dict[str, Optional[float]],
    weights: Dict[str, float],
) -> Optional[float]:
    """
    Weighted mean from persisted Assessment columns.

    If valuation is in weights and missing, use 0 (matches assess.composite_from_scores).
    Other missing components are omitted and weights renormalized.
    """
    num = Decimal("0")
    den = Decimal("0")
    for key, w in weights.items():
        raw = scores.get(key)
        if raw is None:
            if key == "valuation":
                raw = 0.0
            else:
                continue
        num += Decimal(str(float(raw))) * Decimal(str(w))
        den += Decimal(str(w))
    if den <= 0:
        return None
    return float((num / den).quantize(Decimal("0.1")))


def _rating_letter(score: Optional[float]) -> str:
    from core.services.health.ratings import score_to_rating

    rating = score_to_rating(score)
    return rating.letter if rating else "—"


@dataclass
class ScoringMode:
    label: str
    component: Optional[str] = None
    weights: Optional[Dict[str, float]] = None

    def test_score(self, assessment, model_composite: Optional[float]) -> Optional[float]:
        if self.component:
            return _assessment_component_scores(assessment).get(self.component)
        if self.weights:
            return _composite_from_stored(_assessment_component_scores(assessment), self.weights)
        return model_composite


@dataclass
class DiscoveryRow:
    discovery_id: int
    created: date
    symbol: str
    advisor: str
    model_composite: Optional[float]
    adjusted: Optional[float]
    weight: float
    test_score: Optional[float]
    grade: str
    entry_price: float


def _diagnose_empty_load(
    window_start: datetime,
    window_end: datetime,
    advisor: Optional[str],
    scoring: ScoringMode,
) -> None:
    """Help when loaded=0: counts in DB without component/price filters."""
    from core.models import Discovery

    base = Discovery.objects.filter(
        created__gte=window_start,
        created__lt=window_end,
    )
    if advisor:
        base = base.filter(advisor__name=advisor)
        alt = Discovery.objects.filter(
            created__gte=window_start,
            created__lt=window_end,
            advisor__name__icontains="polygon",
        ).count()
        print(f"  diagnose: Polygon-like advisor rows in window: {alt}")

    print(f"  diagnose: discoveries in window: {base.count()}")
    print(f"  diagnose: with assessment: {base.filter(assessment__isnull=False).count()}")

    need_components = bool(scoring.component or scoring.weights)
    if need_components:
        kept = 0
        for d in base.filter(assessment__isnull=False).select_related("assessment")[:500]:
            if scoring.test_score(d.assessment, None) is not None:
                kept += 1
        print(
            f"  diagnose: with scores for {scoring.label} (sample<=500): {kept}"
        )

    if advisor and base.count() == 0:
        names = (
            Discovery.objects.filter(
                created__gte=window_start,
                created__lt=window_end,
            )
            .values_list("advisor__name", flat=True)
            .distinct()
        )
        print(f"  diagnose: advisor names in window: {', '.join(sorted(set(names))) or '(none)'}")


def _load_discoveries(
    from_date: date,
    days: int,
    advisor: Optional[str],
    scoring: ScoringMode,
) -> List[DiscoveryRow]:
    from core.discovery_scoring import discovery_scoring_context
    from core.models import Discovery

    window_start, window_end, _to_date = _cohort_window(from_date, days)

    qs = (
        Discovery.objects.filter(
            assessment__isnull=False,
            created__gte=window_start,
            created__lt=window_end,
        )
        .select_related("stock", "advisor", "assessment")
        .order_by("created")
    )
    if advisor:
        qs = qs.filter(advisor__name=advisor)

    out: List[DiscoveryRow] = []
    for d in qs:
        symbol = (d.stock.symbol or "").strip().upper()
        if not symbol:
            continue

        ctx = discovery_scoring_context(d)
        if ctx.get("source") != "v2":
            continue

        assessment = d.assessment
        model_composite = ctx.get("composite_score")
        if model_composite is None and assessment.score is not None:
            model_composite = float(assessment.score)

        test_score = scoring.test_score(assessment, model_composite)
        if scoring.component or scoring.weights:
            if test_score is None:
                continue

        entry = d.price
        if entry is None or float(entry) <= 0:
            continue

        w = float(d.weight) if d.weight is not None else 1.0
        out.append(
            DiscoveryRow(
                discovery_id=d.id,
                created=d.created.date(),
                symbol=symbol,
                advisor=d.advisor.name,
                model_composite=model_composite,
                adjusted=ctx.get("adjusted_score"),
                weight=w,
                test_score=test_score,
                grade=_rating_letter(test_score),
                entry_price=float(entry),
            )
        )
    return out


def _summarize_by_grade(df: pd.DataFrame, score_label: str) -> None:
    if df.empty:
        return

    print(f"\n=== By grade — {score_label} (7d forward return) ===")
    for grade, g in df.groupby("grade", dropna=False):
        vals = g["ret_7d"].dropna()
        if len(vals) == 0:
            print(f"  {grade}: n=0")
            continue
        pos = int((vals > 0).sum())
        print(
            f"  {grade}: n={len(vals):>3}  mean={vals.mean():+6.2f}%  "
            f"median={vals.median():+6.2f}%  positive={pos}/{len(vals)} ({100 * pos / len(vals):.0f}%)"
        )

    print("\n=== Overall ===")
    vals = df["ret_7d"].dropna()
    if len(vals) == 0:
        print("  No complete 7d returns.")
        return
    pos = int((vals > 0).sum())
    print(
        f"  rows={len(df)}  with_ret={len(vals)}  mean={vals.mean():+6.2f}%  "
        f"median={vals.median():+6.2f}%  positive={pos}/{len(vals)} ({100 * pos / len(vals):.0f}%)"
    )


def _print_table(df: pd.DataFrame, limit: int, show_model_composite: bool) -> None:
    if df.empty:
        print("\nNo rows to display.")
        return

    cols = [
        "created",
        "symbol",
        "advisor",
        "grade",
        "test_score",
    ]
    if show_model_composite:
        cols.append("model_composite")
    cols.extend(["weight", "adjusted", "entry_price", "ret_7d"])
    table = df.copy()
    for c in cols:
        if c not in table.columns:
            table[c] = None
    table = table.sort_values(by="created", ascending=False)
    if limit > 0:
        table = table.head(limit)

    print(f"\n=== Per-discovery ({len(table)} rows) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table[cols].to_string(index=False))


def _build_scoring_mode(args: argparse.Namespace) -> ScoringMode:
    if args.score_component and args.weights:
        raise SystemExit("Use only one of --score-component or --weights")

    if args.score_component:
        key = args.score_component.strip().lower()
        if key not in COMPONENT_KEYS:
            raise SystemExit(
                f"Unknown --score-component {key!r}; "
                f"choose from: {', '.join(COMPONENT_KEYS)}"
            )
        label = COMPONENT_LABELS[key]
        return ScoringMode(label=label, component=key)

    if args.weights:
        try:
            weights = _parse_weights(args.weights)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        parts = ", ".join(f"{k}={weights[k]:.2g}" for k in COMPONENT_KEYS if k in weights)
        return ScoringMode(label=f"custom ({parts})", weights=weights)

    return ScoringMode(label="full model composite")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Health v2 autopsy: discovery assessment grade vs 7-trading-day return.",
    )
    parser.add_argument(
        "--from-date",
        required=True,
        help="Cohort start (inclusive), YYYY-MM-DD",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Cohort length in calendar days from --from-date (default 30)",
    )
    parser.add_argument("--advisor", default=None, help="Filter by advisor name")
    parser.add_argument(
        "--score-component",
        dest="score_component",
        default=None,
        metavar="KEY",
        help=(
            "Grade on one stored component only "
            f"({', '.join(COMPONENT_KEYS)})"
        ),
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Custom component weights from stored scores, e.g. price=1.0 or price=0.5,financial=0.5",
    )
    parser.add_argument("--table", action="store_true", help="Print per-discovery table")
    parser.add_argument(
        "--table-limit",
        type=int,
        default=50,
        help="Max table rows (0 = all; default 50)",
    )
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skip reason breakdown (yfinance vs entry vs incomplete horizon)",
    )
    args = parser.parse_args()

    from_date = date.fromisoformat(args.from_date)
    if args.days <= 0:
        raise SystemExit("--days must be positive")

    scoring = _build_scoring_mode(args)
    show_model = scoring.component is not None or scoring.weights is not None

    _setup_django()
    window_start, window_end, to_date = _cohort_window(from_date, args.days)
    rows_in = _load_discoveries(from_date, args.days, args.advisor, scoring)

    print(
        f"Health v2 autopsy — discoveries {from_date} .. {to_date} (UTC window) "
        f"(assessment required, horizon={HORIZON_TRADING_DAYS}td)"
    )
    print(f"  score for grade: {scoring.label}")
    if args.advisor:
        print(f"  advisor filter: {args.advisor}")
    print(f"  loaded: {len(rows_in)} discoveries")
    if rows_in:
        print(
            f"  created range: {min(r.created for r in rows_in)} .. "
            f"{max(r.created for r in rows_in)}"
        )

    if not rows_in:
        if args.verbose:
            _diagnose_empty_load(window_start, window_end, args.advisor, scoring)
        return

    start = min(r.created for r in rows_in) - timedelta(days=5)
    end = max(r.created for r in rows_in) + timedelta(days=HORIZON_TRADING_DAYS + 20)
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
        ret = _forward_return(close, eidx, r.entry_price, HORIZON_TRADING_DAYS)
        if ret is None:
            skip_incomplete += 1
            continue

        records.append(
            {
                "discovery_id": r.discovery_id,
                "created": r.created.isoformat(),
                "symbol": r.symbol,
                "advisor": r.advisor,
                "score_mode": scoring.label,
                "grade": r.grade,
                "test_score": r.test_score,
                "model_composite": r.model_composite,
                "adjusted": r.adjusted,
                "weight": r.weight,
                "entry_price": round(r.entry_price, 4),
                "ret_7d": round(ret, 2),
            }
        )

    df = pd.DataFrame(records)
    skipped_total = skip_no_hist + skip_no_entry + skip_incomplete
    if skipped_total:
        print(f"  skipped: {skipped_total} total")
        if args.verbose or not len(records):
            print(f"    no yfinance history: {skip_no_hist}")
            print(f"    no entry bar on/after created date: {skip_no_entry}")
            print(
                f"    incomplete {HORIZON_TRADING_DAYS} trading-day horizon: {skip_incomplete}"
            )
        if skip_no_hist == skipped_total and skipped_total:
            print(
                "  hint: all skips are missing yfinance data — check outbound network on this host."
            )
        elif skip_incomplete == skipped_total and skipped_total and rows_in:
            latest = max(r.created for r in rows_in)
            print(
                f"  hint: cohort too recent for {HORIZON_TRADING_DAYS}td forward returns "
                f"(newest discovery {latest}; today {date.today()}). "
                f"Use an earlier --from-date (e.g. {(date.today() - timedelta(days=21)).isoformat()}) "
                f"and wider --days."
            )

    _summarize_by_grade(df, scoring.label)
    if args.table:
        _print_table(df, args.table_limit, show_model)

    if args.csv:
        out_path = Path(args.csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved rows to: {out_path}")


if __name__ == "__main__":
    main()
