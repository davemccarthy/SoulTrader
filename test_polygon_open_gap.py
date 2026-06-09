#!/usr/bin/env python3
"""
Fund BUY open-gap report for a single calendar date.

Uses actual BUY trades for --fund on --date (not all discoveries). By default
only buys outside regular US market hours (pre-market / after-hours). Optional
--advisors filters via linked discovery.advisor.

For each buy, fetch OHLC for the next relevant trading session (auto by default):
  pre      -> same weekday session (open that morning)
  after    -> next trading day
  weekend  -> next trading day (e.g. Sat/Sun -> Monday)

  PREV %       = (open - prev_close) / prev_close * 100
  CLOSE %      = (close - open) / open * 100
  BUYvsPRV %   = (trade.price - prev_close) / prev_close * 100  (vs last regular close)
  BUYvsCL %    = (trade.price - close) / close * 100

Examples:
  python test_polygon_open_gap.py 2026-06-06 --fund mix1
  python test_polygon_open_gap.py --date 2026-06-06 --fund mix1 --advisors Polygon.io
  python test_polygon_open_gap.py --date 2026-06-06 --fund mix1 --advisors Polygon.io,StockStory --csv out.csv
  python test_polygon_open_gap.py --date 2026-06-08 --fund ZOMB --advisors Polygon.io
  python test_polygon_open_gap.py --date 2026-06-08 --fund ZOMB --all-hours
  python test_polygon_open_gap.py --date 2026-06-06 --fund mix1 --session next-open --window-tz US/Eastern
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import zoneinfo
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf

ET = zoneinfo.ZoneInfo("US/Eastern")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
DISC_CLOSE_MATCH_TOL_PCT = 0.5


def _setup_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()


@dataclass
class Row:
    ticker: str
    advisor: str
    buy_session: str
    bought_at: datetime
    session_date: date
    discovery_price: Optional[float]
    buy_price: float
    prev_close: float
    open_px: float
    close_px: float
    prev_pct: float
    close_pct: float
    disc_vs_close_pct: Optional[float]
    disc_vs_prev_pct: Optional[float]
    buy_vs_close_pct: float
    buy_vs_prev_pct: float


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _parse_advisors(value: str) -> List[str]:
    names = [part.strip() for part in (value or "").split(",") if part.strip()]
    if not names:
        raise ValueError("at least one advisor name required")
    return names


def _date_bounds(target: date, window_tz: Optional[str]) -> Tuple[datetime, datetime]:
    from django.utils import timezone as dj_tz

    if window_tz:
        tz = zoneinfo.ZoneInfo(window_tz)
        start = datetime.combine(target, time.min, tzinfo=tz)
        end = start + timedelta(days=1)
        return start, end

    start = dj_tz.make_aware(datetime.combine(target, time.min))
    end = start + timedelta(days=1)
    return start, end


def _resolve_advisors(advisor_names: Sequence[str]):
    from core.models import Advisor

    advisors = list(Advisor.objects.filter(name__in=advisor_names))
    found = {a.name for a in advisors}
    missing = [n for n in advisor_names if n not in found]
    if missing:
        raise SystemExit(f"Advisor(s) not found: {', '.join(missing)}")
    return advisors


def _resolve_fund(fund_name: str):
    from core.models import Profile

    try:
        return Profile.objects.get(name=fund_name)
    except Profile.DoesNotExist:
        raise SystemExit(f'Fund "{fund_name}" not found')


def _buys_for_date(
    fund_name: str,
    target: date,
    advisor_names: Sequence[str],
    window_tz: Optional[str],
) -> list:
    from core.models import Trade

    fund = _resolve_fund(fund_name)
    advisors = _resolve_advisors(advisor_names)
    start, end = _date_bounds(target, window_tz)

    qs = (
        Trade.objects.filter(
            fund=fund,
            action="BUY",
            created__gte=start,
            created__lt=end,
            discovery__advisor__in=advisors,
        )
        .select_related("stock", "discovery", "discovery__advisor", "fund")
        .order_by("created")
    )
    return list(qs)


def _buy_session_label(bought_at: datetime) -> str:
    """pre | rth | after | weekend"""
    local = bought_at.astimezone(ET)
    if local.weekday() >= 5:
        return "weekend"
    t = local.time()
    if t < MARKET_OPEN:
        return "pre"
    if t >= MARKET_CLOSE:
        return "after"
    return "rth"


def _is_outside_market_hours(bought_at: datetime) -> bool:
    return _buy_session_label(bought_at) != "rth"


def _filter_trades_by_hours(trades: list, outside_only: bool) -> Tuple[list, int]:
    if not outside_only:
        return trades, 0
    kept = []
    for trade in trades:
        bought_at = trade.created
        if bought_at.tzinfo is None:
            from django.utils import timezone as dj_tz

            bought_at = dj_tz.make_aware(bought_at)
        if _is_outside_market_hours(bought_at):
            kept.append(trade)
    return kept, len(trades) - len(kept)


def _next_trading_day(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _ohlc_session_date(bought_at: datetime, mode: str) -> date:
    """
    Trading session whose open / prev_close / close we measure.

    auto (default): pre -> same weekday; after/weekend -> next trading session.
    """
    local = bought_at.astimezone(ET)
    d = local.date()
    label = _buy_session_label(bought_at)

    if mode == "same-day":
        return _next_trading_day(d)

    if mode == "next-open":
        return _next_trading_day(d + timedelta(days=1))

    # auto
    if label == "pre":
        return _next_trading_day(d)
    if label in ("after", "weekend"):
        return _next_trading_day(d + timedelta(days=1))
    return d


def _fetch_ohlc(
    symbol: str,
    session_date: date,
    cache: Dict[str, Optional[pd.DataFrame]],
) -> Optional[Tuple[float, float, float]]:
    """Return (prev_close, open, close) for session_date."""
    if symbol not in cache:
        start = session_date - timedelta(days=14)
        end = session_date + timedelta(days=5)
        try:
            hist = yf.Ticker(symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=True,
            )
        except Exception:
            cache[symbol] = None
        else:
            if hist is None or hist.empty:
                cache[symbol] = None
            else:
                hist = hist.copy()
                hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
                hist = hist[~hist.index.duplicated(keep="last")]
                cache[symbol] = hist

    df = cache[symbol]
    if df is None or df.empty:
        return None

    idx = df.index
    ts = pd.Timestamp(session_date)
    pos = int(idx.searchsorted(ts, side="left"))
    if pos >= len(idx):
        return None

    if pos == 0:
        return None

    row = df.iloc[pos]
    prev_close = float(df.iloc[pos - 1]["Close"])
    open_px = float(row["Open"])
    close_px = float(row["Close"])
    if prev_close <= 0 or open_px <= 0 or close_px <= 0:
        return None
    return prev_close, open_px, close_px


def _pct(numer: float, denom: float) -> float:
    return (numer - denom) / denom * 100.0


def _to_float(value: Optional[Decimal]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _build_rows(trades, session_mode: str) -> Tuple[List[Row], List[str]]:
    cache: Dict[str, Optional[pd.DataFrame]] = {}
    rows: List[Row] = []
    skipped: List[str] = []

    for trade in trades:
        sym = trade.stock.symbol.upper()
        bought_at = trade.created
        if bought_at.tzinfo is None:
            from django.utils import timezone as dj_tz

            bought_at = dj_tz.make_aware(bought_at)

        discovery = trade.discovery
        advisor = discovery.advisor.name if discovery and discovery.advisor else "?"
        discovery_price = _to_float(discovery.price) if discovery else None
        buy_price = float(trade.price)

        session_date = _ohlc_session_date(bought_at, session_mode)
        ohlc = _fetch_ohlc(sym, session_date, cache)
        if ohlc is None:
            skipped.append(sym)
            continue

        prev_close, open_px, close_px = ohlc
        disc_vs_close = _pct(discovery_price, close_px) if discovery_price is not None else None
        disc_vs_prev = _pct(discovery_price, prev_close) if discovery_price is not None else None
        rows.append(
            Row(
                ticker=sym,
                advisor=advisor,
                buy_session=_buy_session_label(bought_at),
                bought_at=bought_at,
                session_date=session_date,
                discovery_price=discovery_price,
                buy_price=buy_price,
                prev_close=prev_close,
                open_px=open_px,
                close_px=close_px,
                prev_pct=_pct(open_px, prev_close),
                close_pct=_pct(close_px, open_px),
                disc_vs_close_pct=disc_vs_close,
                disc_vs_prev_pct=disc_vs_prev,
                buy_vs_close_pct=_pct(buy_price, close_px),
                buy_vs_prev_pct=_pct(buy_price, prev_close),
            )
        )
    return rows, skipped


def _fmt_price(value: Optional[float]) -> str:
    if value is None:
        return "     —"
    return f"{value:7.2f}"


def _fmt_pct(value: Optional[float], width: int = 8) -> str:
    if value is None:
        return f"{'—':>{width}}"
    return f"{value:+{width - 1}.2f}"


def _print_table(rows: List[Row], fund_name: str) -> None:
    if not rows:
        print("No rows.")
        return

    print(f"Fund: {fund_name}")
    print(
        f"{'TICKER':<7} {'WHEN':<7} {'ADVISOR':<12} {'BUY$':>7} {'PRVCL$':>7} {'OPEN$':>7} "
        f"{'CLOSE$':>7} {'BUYvsPRV':>9} {'BUYvsCL':>8} {'PREV %':>8} {'CLOSE %':>8}  "
        f"ohlc_sess  bought (ET)"
    )
    print("-" * 118)
    for r in sorted(rows, key=lambda x: (x.advisor, x.ticker)):
        bought_et = r.bought_at.astimezone(ET).strftime("%Y-%m-%d %H:%M")
        print(
            f"{r.ticker:<7} {r.buy_session:<7} {r.advisor:<12} {r.buy_price:7.2f} "
            f"{r.prev_close:7.2f} {r.open_px:7.2f} {r.close_px:7.2f} "
            f"{_fmt_pct(r.buy_vs_prev_pct, 9)} {_fmt_pct(r.buy_vs_close_pct, 8)} "
            f"{r.prev_pct:+8.2f} {r.close_pct:+8.2f}  "
            f"{r.session_date}  {bought_et}"
        )

    prev_vals = [r.prev_pct for r in rows]
    close_vals = [r.close_pct for r in rows]
    buy_vs_close = [r.buy_vs_close_pct for r in rows]
    buy_vs_prev = [r.buy_vs_prev_pct for r in rows]

    print("-" * 118)
    print(
        f"{'MEAN':<7} {'':7} {'':12} {'':>7} {'':>7} {'':>7} {'':>7} "
        f"{_fmt_pct(statistics.mean(buy_vs_prev), 9)} {_fmt_pct(statistics.mean(buy_vs_close), 8)} "
        f"{statistics.mean(prev_vals):+8.2f} {statistics.mean(close_vals):+8.2f}  "
        f"n={len(rows)}"
    )
    if len(rows) >= 2:
        print(
            f"{'MEDIAN':<7} {'':7} {'':12} {'':>7} {'':>7} {'':>7} {'':>7} "
            f"{_fmt_pct(statistics.median(buy_vs_prev), 9)} "
            f"{_fmt_pct(statistics.median(buy_vs_close), 8)} "
            f"{statistics.median(prev_vals):+8.2f} {statistics.median(close_vals):+8.2f}"
        )

    neg_prev = sum(1 for v in prev_vals if v < 0)
    print(f"\nPREV % < 0 (gapped down at open): {neg_prev}/{len(rows)} ({100 * neg_prev / len(rows):.0f}%)")

    buy_near_prev = sum(1 for v in buy_vs_prev if abs(v) <= DISC_CLOSE_MATCH_TOL_PCT)
    print(
        f"Buy price within {DISC_CLOSE_MATCH_TOL_PCT:.1f}% of prior close (PRVCL): "
        f"{buy_near_prev}/{len(rows)} ({100 * buy_near_prev / len(rows):.0f}%)"
    )
    buy_near_close = sum(1 for v in buy_vs_close if abs(v) <= DISC_CLOSE_MATCH_TOL_PCT)
    print(
        f"Buy price within {DISC_CLOSE_MATCH_TOL_PCT:.1f}% of session close: "
        f"{buy_near_close}/{len(rows)} ({100 * buy_near_close / len(rows):.0f}%)"
    )


def _write_csv(path: str, rows: List[Row]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "ticker",
                "buy_session",
                "advisor",
                "bought_at_utc",
                "session_date",
                "discovery_price",
                "buy_price",
                "prev_close",
                "open",
                "close",
                "disc_vs_close_pct",
                "disc_vs_prev_pct",
                "buy_vs_close_pct",
                "buy_vs_prev_pct",
                "prev_pct",
                "close_pct",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.ticker,
                    r.buy_session,
                    r.advisor,
                    r.bought_at.isoformat(),
                    r.session_date.isoformat(),
                    "" if r.discovery_price is None else f"{r.discovery_price:.4f}",
                    f"{r.buy_price:.4f}",
                    f"{r.prev_close:.4f}",
                    f"{r.open_px:.4f}",
                    f"{r.close_px:.4f}",
                    "" if r.disc_vs_close_pct is None else f"{r.disc_vs_close_pct:.4f}",
                    "" if r.disc_vs_prev_pct is None else f"{r.disc_vs_prev_pct:.4f}",
                    f"{r.buy_vs_close_pct:.4f}",
                    f"{r.buy_vs_prev_pct:.4f}",
                    f"{r.prev_pct:.4f}",
                    f"{r.close_pct:.4f}",
                ]
            )
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fund BUY open-gap report for one date (open as baseline)."
    )
    parser.add_argument("date", nargs="?", help="Calendar date YYYY-MM-DD")
    parser.add_argument("--date", dest="date_opt", help="Same as positional date")
    parser.add_argument("--fund", required=True, help="Fund name (Profile.name), e.g. mix1")
    parser.add_argument(
        "--advisors",
        default="Polygon.io",
        help="Comma-separated advisor names on linked discovery (default: Polygon.io)",
    )
    parser.add_argument(
        "--session",
        choices=("auto", "same-day", "next-open"),
        default="auto",
        help="OHLC session: auto=pre same day, after/weekend next trading day (default)",
    )
    parser.add_argument(
        "--window-tz",
        default=None,
        help="Filter buys by local calendar date (e.g. US/Eastern)",
    )
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    parser.add_argument(
        "--all-hours",
        action="store_true",
        help="Include regular-hours buys (default: outside market hours only)",
    )
    args = parser.parse_args()

    raw = args.date_opt or args.date
    if not raw:
        parser.error("Provide a date: YYYY-MM-DD")

    target = _parse_date(raw)
    advisor_names = _parse_advisors(args.advisors)
    _setup_django()

    all_trades = _buys_for_date(args.fund, target, advisor_names, args.window_tz)
    outside_only = not args.all_hours
    trades, excluded_rth = _filter_trades_by_hours(all_trades, outside_only)

    scope = "outside market hours" if outside_only else "all hours"
    print(
        f"BUY trades on {target} for fund {args.fund!r} "
        f"({', '.join(advisor_names)}): {len(all_trades)} total, "
        f"{len(trades)} {scope}"
    )
    if outside_only and excluded_rth:
        print(f"Excluded {excluded_rth} regular-hours (9:30–16:00 ET) buy(s)")

    rows, skipped = _build_rows(trades, args.session)
    if skipped:
        print(f"Skipped (no OHLC): {', '.join(sorted(set(skipped)))}")

    _print_table(rows, args.fund)
    if args.csv:
        _write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
