#!/usr/bin/env python3
"""
Quick P&L snapshot for advisor discoveries.

Filters Discovery rows by SmartAnalysis id range or trailing days and
computes price performance from discovery date to a reference date (default: now).
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import django
import pandas as pd
from django.utils import timezone

# Ensure project settings are available before importing ORM models
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from core.models import Advisor, Discovery  # noqa: E402

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - helper message for first-time runs
    print("This tool requires yfinance. Install it with `pip install yfinance`.", file=sys.stderr)
    sys.exit(1)


def _parse_as_of(value: str) -> datetime:
    """Parse --as-of input into an aware datetime."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return timezone.make_aware(dt)
    return dt.astimezone(timezone.get_current_timezone())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate discovery P&L over a time/span period.")
    span = parser.add_mutually_exclusive_group()
    span.add_argument("--days", type=int, help="Number of trailing days to include (mutually exclusive with SA range).")
    span.add_argument("--start-sa", type=int, help="First SmartAnalysis id (inclusive).")
    parser.add_argument("--end-sa", type=int, help="Last SmartAnalysis id (inclusive).")
    parser.add_argument("--advisor", help="Filter discoveries by advisor name (case insensitive).")
    parser.add_argument("--as-of", type=_parse_as_of, help="Reference timestamp for current price (default: now).")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of discoveries to evaluate (after dedupe).")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show per-discovery rows. By default only per-advisor aggregates are displayed.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print extra information such as skipped symbols and data gaps."
    )

    args = parser.parse_args(argv)

    if args.days is None and args.start_sa is None:
        parser.error("You must specify either --days or --start-sa/--end-sa.")

    if args.end_sa is not None and args.start_sa is None:
        parser.error("--end-sa requires --start-sa.")

    return args


def _normalize_as_of(arg_as_of: Optional[datetime]) -> datetime:
    if arg_as_of:
        return arg_as_of
    return timezone.now()


def _resolve_advisor(advisor_name: Optional[str]) -> Optional[Advisor]:
    if not advisor_name:
        return None
    return Advisor.objects.filter(name__iexact=advisor_name).first()


def _build_discovery_queryset(args: argparse.Namespace, advisor: Optional[Advisor]):
    qs = Discovery.objects.select_related("advisor", "stock", "sa").order_by("created")

    if args.days is not None:
        cutoff = timezone.now() - timedelta(days=args.days)
        qs = qs.filter(created__gte=cutoff)
    else:
        qs = qs.filter(sa_id__gte=args.start_sa)
        if args.end_sa is not None:
            qs = qs.filter(sa_id__lte=args.end_sa)

    if advisor:
        qs = qs.filter(advisor=advisor)

    return qs


def _discovery_timestamp(discovery) -> datetime:
    """Return best guess of discovery timestamp."""
    tolerance = timedelta(hours=1)
    now = timezone.now()
    sa_started = discovery.sa.started if discovery.sa_id else None

    if discovery.created:
        candidate = discovery.created
        if candidate > now and sa_started:
            return sa_started
        if sa_started and abs(candidate - sa_started) > tolerance:
            return sa_started
        return candidate

    if sa_started:
        return sa_started

    return now


def _dedupe_discoveries(discoveries: Iterable[Discovery]) -> List[Discovery]:
    seen = set()
    unique: List[Discovery] = []
    for discovery in discoveries:
        if discovery.stock_id in seen:
            continue
        seen.add(discovery.stock_id)
        unique.append(discovery)
    return unique


class YfPriceFetcher:
    """Basic yfinance helper with simple caching."""

    def __init__(self):
        self._history_cache: Dict[Tuple[str, str], Optional[float]] = {}
        self._latest_cache: Dict[Tuple[str, str], Optional[float]] = {}

    def price_from(self, symbol: str, ref_datetime: datetime) -> Optional[float]:
        key = (symbol.upper(), ref_datetime.strftime("%Y-%m-%d"))
        if key not in self._history_cache:
            self._history_cache[key] = self._lookup_price(symbol, ref_datetime)
        return self._history_cache[key]

    def latest_price(self, symbol: str, as_of: datetime) -> Optional[float]:
        use_realtime = self._should_use_realtime(as_of)
        cache_key = (symbol.upper(), "realtime" if use_realtime else as_of.strftime("%Y-%m-%d"))

        if cache_key not in self._latest_cache:
            if use_realtime:
                price = self._lookup_realtime_price(symbol)
                if price is None:
                    price = self._lookup_price(symbol, as_of, forward_only=True)
            else:
                price = self._lookup_price(symbol, as_of, forward_only=True)
            self._latest_cache[cache_key] = price

        return self._latest_cache[cache_key]

    @staticmethod
    def _should_use_realtime(as_of: datetime) -> bool:
        now = timezone.now()
        # Use real-time quotes when the reference point is today (or in the future).
        return as_of.date() >= now.date()

    @staticmethod
    def _lookup_realtime_price(symbol: str) -> Optional[float]:
        """Attempt to fetch the latest traded price (regular or post market)."""
        try:
            ticker = yf.Ticker(symbol)
            fast_info = getattr(ticker, "fast_info", None)
            if fast_info:
                for field in ("last_price", "regular_market_price", "post_market_price"):
                    value = getattr(fast_info, field, None)
                    if value is not None:
                        return float(value)

            price = ticker.info.get("currentPrice") if hasattr(ticker, "info") else None
            if price is not None:
                return float(price)
        except Exception:
            return None
        return None

    @staticmethod
    def _lookup_price(symbol: str, ref_datetime: datetime, forward_only: bool = False) -> Optional[float]:
        """Fetch the first available close on/after ref_datetime (or before if forward_only)."""
        start_date = ref_datetime.date() - timedelta(days=2 if not forward_only else 0)
        end_date = ref_datetime.date() + timedelta(days=5)
        try:
            hist = yf.download(
                symbol,
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                progress=False,
            )
        except Exception:  # pragma: no cover - passthrough network/library errors
            return None

        if hist.empty:
            return None

        # Normalise index to naive datetime for comparison
        index_dates = []
        for ts in hist.index.to_pydatetime():
            if ts.tzinfo:
                index_dates.append(ts.replace(tzinfo=None))
            else:
                index_dates.append(ts)

        close_data = hist.get("Close")
        if close_data is None:
            return None
        if isinstance(close_data, pd.DataFrame):
            close_series = close_data.iloc[:, 0]
        else:
            close_series = close_data

        for idx, close in zip(index_dates, close_series.tolist()):
            if idx.date() >= ref_datetime.date():
                if isinstance(close, pd.Series):
                    close = close.iloc[0]
                return float(close)

        if forward_only:
            return None

        # Fall back to last available close before the reference date
        last_close = close_series.iloc[-1]
        if isinstance(last_close, pd.Series):
            last_close = last_close.iloc[0]
        return float(last_close)


def _summarise(results: Sequence[Dict]) -> Dict[str, float]:
    summary = defaultdict(float)
    summary["count"] = len(results)
    total_entry = 0.0
    total_current = 0.0

    for row in results:
        change_pct = row.get("pct_change")
        if change_pct is None:
            summary["missing"] += 1
            continue
        if change_pct > 0:
            summary["gainers"] += 1
        elif change_pct < 0:
            summary["losers"] += 1
        else:
            summary["flat"] += 1
        summary["avg_change"] += change_pct
        entry = row.get("entry_price")
        current = row.get("current_price")
        if entry is not None and current is not None:
            total_entry += entry
            total_current += current

    if summary["count"] - summary.get("missing", 0) > 0:
        summary["avg_change"] /= summary["count"] - summary.get("missing", 0)
    else:
        summary["avg_change"] = 0.0

    summary["total_entry"] = total_entry
    summary["total_current"] = total_current
    summary["net_change"] = total_current - total_entry
    if total_entry > 0:
        summary["total_pct"] = (summary["net_change"] / total_entry) * 100
    else:
        summary["total_pct"] = 0.0

    return summary


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    as_of = _normalize_as_of(args.as_of)
    advisor = _resolve_advisor(args.advisor)
    if args.advisor and not advisor:
        print(f"No advisor found matching '{args.advisor}'.", file=sys.stderr)
        return 2

    queryset = _build_discovery_queryset(args, advisor)
    discoveries = list(queryset)
    if not discoveries:
        print("No discoveries found for provided criteria.")
        return 0

    unique_discoveries = _dedupe_discoveries(discoveries)
    if args.limit:
        unique_discoveries = unique_discoveries[: args.limit]

    fetcher = YfPriceFetcher()
    rows: List[Dict] = []

    for discovery in unique_discoveries:
        symbol = discovery.stock.symbol
        discovery_dt = _discovery_timestamp(discovery)
        entry_price = fetcher.price_from(symbol, discovery_dt)
        current_price = fetcher.latest_price(symbol, as_of)

        if args.verbose and (entry_price is None or current_price is None):
            print(f"[warn] Missing price data for {symbol} (entry={entry_price} current={current_price}).")

        pct_change = None
        abs_change = None
        if entry_price and current_price:
            abs_change = current_price - entry_price
            pct_change = (abs_change / entry_price) * 100

        rows.append(
            {
                "symbol": symbol,
                "advisor": discovery.advisor.name,
                "created": discovery_dt,
                "entry_price": entry_price,
                "current_price": current_price,
                "abs_change": abs_change,
                "pct_change": pct_change,
            }
        )

    print(f"Evaluated {len(rows)} unique discoveries (from {len(discoveries)} raw rows).")
    print(f"As of: {as_of.isoformat(timespec='seconds')}")
    if advisor:
        print(f"Advisor filter: {advisor.name}")

    # Per-advisor aggregates
    per_advisor: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        per_advisor[row["advisor"]].append(row)

    advisor_header = f"{'Advisor':<20} {'Discoveries':>12} {'Complete':>10} {'Gainers':>10} {'Win %':>8} {'Avg Δ%':>10} {'Net Δ$':>12} {'Net %':>10}"
    print("\nPer-advisor summary:")
    print(advisor_header)
    print("-" * len(advisor_header))
    for advisor_name, advisor_rows in sorted(per_advisor.items()):
        advisor_summary = _summarise(advisor_rows)
        complete = int(advisor_summary["count"] - advisor_summary.get("missing", 0))
        gainers = int(advisor_summary.get("gainers", 0))
        win_pct = (gainers / complete * 100) if complete > 0 else 0.0
        print(
            f"{advisor_name:<20} "
            f"{int(advisor_summary['count']):>12} "
            f"{complete:>10} "
            f"{gainers:>10} "
            f"{win_pct:>7.1f}% "
            f"{advisor_summary['avg_change']:+10.2f} "
            f"{advisor_summary['net_change']:+12.2f} "
            f"{advisor_summary['total_pct']:+10.2f}"
        )

    if args.details:
        header = f"{'Symbol':<8} {'Advisor':<20} {'Discovered':<20} {'Entry':>10} {'Current':>10} {'Δ$':>10} {'Δ%':>8}"
        print("\nDetails:")
        print(header)
        print("-" * len(header))
        for row in rows:
            entry_str = f"{row['entry_price']:.2f}" if row["entry_price"] is not None else "n/a"
            current_str = f"{row['current_price']:.2f}" if row["current_price"] is not None else "n/a"
            abs_str = f"{row['abs_change']:+.2f}" if row["abs_change"] is not None else "n/a"
            pct_str = f"{row['pct_change']:+.2f}" if row["pct_change"] is not None else "n/a"
            created_str = row["created"].astimezone(dt_timezone.utc).strftime("%Y-%m-%d %H:%M")
            print(
                f"{row['symbol']:<8} {row['advisor']:<20} {created_str:<20} "
                f"{entry_str:>10} {current_str:>10} {abs_str:>10} {pct_str:>8}"
            )

    summary = _summarise(rows)
    print("\nSummary:")
    evaluated = summary["count"] - summary.get("missing", 0)
    print(f"- Evaluated: {int(summary['count'])} discoveries ({int(summary.get('missing', 0))} missing price data).")
    print(f"- Positive: {int(summary.get('gainers', 0))}, Negative: {int(summary.get('losers', 0))}, Flat: {int(summary.get('flat', 0))}.")
    print(f"- Average change: {summary['avg_change']:+.2f}% over evaluated samples ({int(evaluated)} with complete data).")
    print(
        f"- Net return: {summary['net_change']:+.2f} ({summary['total_pct']:+.2f}%) "
        f"on {summary['total_entry']:.2f} cumulative entry cost."
    )

    return 0


if __name__ == "__main__":
    sys.exit(run())

