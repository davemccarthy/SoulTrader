#!/usr/bin/env python3
"""
Health v2 assessment autopsy: discovery cohort vs 7-trading-day forward return.

Only discoveries with a linked Assessment (v2). Discoveries must be at least
7 calendar days old so the forward window can complete.

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
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

HORIZON_TRADING_DAYS = 7
MIN_AGE_CALENDAR_DAYS = 7

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


def _load_discoveries(
    from_date: date,
    days: int,
    advisor: Optional[str],
    scoring: ScoringMode,
) -> List[DiscoveryRow]:
    from core.discovery_scoring import discovery_scoring_context
    from core.models import Discovery

    to_date = from_date + timedelta(days=days)
    max_created = date.today() - timedelta(days=MIN_AGE_CALENDAR_DAYS)

    qs = (
        Discovery.objects.filter(
            assessment__isnull=False,
            created__date__gte=from_date,
            created__date__lt=to_date,
            created__date__lte=max_created,
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
    args = parser.parse_args()

    from_date = date.fromisoformat(args.from_date)
    if args.days <= 0:
        raise SystemExit("--days must be positive")

    scoring = _build_scoring_mode(args)
    show_model = scoring.component is not None or scoring.weights is not None

    _setup_django()
    rows_in = _load_discoveries(from_date, args.days, args.advisor, scoring)
    to_date = from_date + timedelta(days=args.days)
    max_created = date.today() - timedelta(days=MIN_AGE_CALENDAR_DAYS)

    print(
        f"Health v2 autopsy — discoveries {from_date} .. {to_date} "
        f"(created <= {max_created}), assessment required, horizon={HORIZON_TRADING_DAYS}td"
    )
    print(f"  score for grade: {scoring.label}")
    if args.advisor:
        print(f"  advisor filter: {args.advisor}")
    print(f"  loaded: {len(rows_in)} discoveries")

    if not rows_in:
        return

    start = min(r.created for r in rows_in) - timedelta(days=5)
    end = max(r.created for r in rows_in) + timedelta(days=HORIZON_TRADING_DAYS + 15)
    cache: Dict[Tuple[str, date, date], pd.Series] = {}

    records: List[dict] = []
    skipped_price = 0
    for r in rows_in:
        close = _fetch_close_series(r.symbol, start, end, cache)
        if close is None:
            skipped_price += 1
            continue
        eidx = _entry_index(close, r.created)
        if eidx is None:
            skipped_price += 1
            continue
        ret = _forward_return(close, eidx, r.entry_price, HORIZON_TRADING_DAYS)
        if ret is None:
            skipped_price += 1
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
    if skipped_price:
        print(f"  skipped (no price data / incomplete {HORIZON_TRADING_DAYS}td): {skipped_price}")

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
