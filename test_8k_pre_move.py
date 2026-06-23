#!/usr/bin/env python3
"""
Pre-filing price run for earnings 8-K candidates on a given date (or season window).

Measures close-to-close move over N trading days ending at the last close before the
8-K is public (close T-1 for pre-market / regular filings; same-day close for after-hours).

Usage:
  python test_8k_pre_move.py --date 2025-10-28
  python test_8k_pre_move.py --date 2025-10-28 --min-pct 5
  python test_8k_pre_move.py --from 2025-10-14 --to 2025-11-01 --min-pct 5 --csv pre_move.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from test_8k_inspector import analyze_8k, get_8ks_for_date, set_identity

# SEC identity (inspector module sets this on import; keep explicit for standalone runs)
set_identity("David McCarthy david@example.com")

ET = ZoneInfo("America/New_York")
DEFAULT_LOOKBACK = 20
DEFAULT_MIN_PCT = 5.0


@dataclass
class PreMoveRow:
    filing_date: str
    time_et: str
    ticker: str
    cik: str
    accession: str
    filing_bucket: str
    close_tmN: Optional[float]
    anchor_close: Optional[float]
    pre_move_pct: Optional[float]
    ret_p1_pct: Optional[float]
    ret_p7_pct: Optional[float]
    late_share_pct: Optional[float]
    max_1d_jump_pct: Optional[float]
    up_days_lookback: Optional[int]
    creep_label: str
    flag_ge_min: bool


def _to_et(filing_dt: datetime) -> datetime:
    if filing_dt.tzinfo is None:
        return filing_dt.replace(tzinfo=ET)
    return filing_dt.astimezone(ET)


def filing_bucket(filing_dt: Optional[datetime]) -> str:
    if filing_dt is None:
        return "unknown"
    dt_et = _to_et(filing_dt)
    t = dt_et.time()
    if t < dt_time(9, 30):
        return "pre_market"
    if t < dt_time(16, 0):
        return "regular"
    return "after_hours"


def anchor_close_index(
    close: pd.Series,
    filing_dt: Optional[datetime],
) -> Tuple[Optional[int], str]:
    """
    Index of the last regular-session close before the 8-K is public.

    Pre-market / regular on day F → close on the previous trading day.
    After-hours on day F → close on day F.
    """
    bucket = filing_bucket(filing_dt)
    if filing_dt is None or close is None or close.empty:
        return None, bucket

    filing_day = _to_et(filing_dt).date()
    idx_dates = [ts.date() for ts in close.index]

    if bucket == "after_hours":
        matches = [i for i, d in enumerate(idx_dates) if d == filing_day]
        if not matches:
            return None, bucket
        return matches[-1], bucket

    anchor_idx: Optional[int] = None
    for i, d in enumerate(idx_dates):
        if d < filing_day:
            anchor_idx = i
        elif d >= filing_day:
            break
    return anchor_idx, bucket


def _fetch_close_series(
    symbol: str,
    start_date: date,
    end_date: date,
    cache: Dict[Tuple[str, date, date], Optional[pd.Series]],
) -> Optional[pd.Series]:
    key = (symbol.upper(), start_date, end_date)
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


def _pct(a: float, b: float) -> Optional[float]:
    if a <= 0:
        return None
    return (b - a) / a * 100.0


def _forward_return(close: pd.Series, entry_idx: int, days: int) -> Optional[float]:
    target_idx = entry_idx + days
    if target_idx >= len(close):
        return None
    try:
        p0 = float(close.iloc[entry_idx])
        p1 = float(close.iloc[target_idx])
    except (TypeError, ValueError, IndexError):
        return None
    return _pct(p0, p1)


def _daily_returns_pct(close_window: pd.Series) -> List[float]:
    if close_window is None or len(close_window) < 2:
        return []
    daily = close_window.pct_change().dropna() * 100.0
    out: List[float] = []
    for v in daily.tolist():
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def classify_creep_bang(
    close: pd.Series,
    *,
    start_idx: int,
    anchor_idx: int,
    pre_move_pct: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[int], str]:
    """
    Returns (late_share_pct, max_1d_jump_pct, up_days_lookback, label)
    label in {"creep", "bang", "mixed", "flat"}.
    """
    if pre_move_pct is None or anchor_idx <= start_idx:
        return None, None, None, "flat"

    lookback_len = anchor_idx - start_idx
    late_days = min(5, lookback_len)
    late_start_idx = anchor_idx - late_days
    if late_start_idx < start_idx:
        late_start_idx = start_idx

    try:
        p_late0 = float(close.iloc[late_start_idx])
        p_anchor = float(close.iloc[anchor_idx])
    except (TypeError, ValueError, IndexError):
        return None, None, None, "flat"

    late_move_pct = _pct(p_late0, p_anchor)
    late_share_pct = None
    if late_move_pct is not None and abs(pre_move_pct) > 1e-9:
        late_share_pct = (late_move_pct / pre_move_pct) * 100.0

    window = close.iloc[start_idx : anchor_idx + 1]
    daily = _daily_returns_pct(window)
    if not daily:
        return late_share_pct, None, None, "flat"

    max_1d_jump_pct = max(daily)
    up_days = sum(1 for d in daily if d > 0)

    if pre_move_pct <= 0:
        label = "flat"
    elif (late_share_pct is not None and late_share_pct >= 65.0) or max_1d_jump_pct >= 6.0:
        label = "bang"
    elif (late_share_pct is not None and late_share_pct <= 45.0) and up_days >= max(8, int(len(daily) * 0.55)):
        label = "creep"
    else:
        label = "mixed"

    return late_share_pct, max_1d_jump_pct, up_days, label


def compute_pre_move_pct(
    ticker: str,
    filing_dt: Optional[datetime],
    lookback: int,
    cache: Dict[Tuple[str, date, date], Optional[pd.Series]],
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[int],
    str,
]:
    """
    Return (close_tmN, anchor_close, pre_move_pct) using trading-day offsets.
    """
    sym = (ticker or "").strip().upper()
    if not sym:
        return None, None, None, None, None, None, None, None, "flat"

    if filing_dt is None:
        end = date.today()
    else:
        end = _to_et(filing_dt).date()

    start = end - timedelta(days=lookback * 3 + 30)
    close = _fetch_close_series(sym, start, end, cache)
    if close is None or close.empty:
        return None, None, None, None, None, None, None, None, "flat"

    anchor_idx, _ = anchor_close_index(close, filing_dt)
    if anchor_idx is None:
        return None, None, None, None, None, None, None, None, "flat"

    start_idx = anchor_idx - lookback
    if start_idx < 0:
        return None, None, None, None, None, None, None, None, "flat"

    try:
        p0 = float(close.iloc[start_idx])
        p1 = float(close.iloc[anchor_idx])
    except (TypeError, ValueError, IndexError):
        return None, None, None

    if p0 <= 0:
        return None, None, None, None, None, None, None, None, "flat"

    pct = (p1 - p0) / p0 * 100.0
    ret_p1 = _forward_return(close, anchor_idx, 1)
    ret_p7 = _forward_return(close, anchor_idx, 7)
    late_share, max_jump, up_days, label = classify_creep_bang(
        close,
        start_idx=start_idx,
        anchor_idx=anchor_idx,
        pre_move_pct=pct,
    )
    return p0, p1, pct, ret_p1, ret_p7, late_share, max_jump, up_days, label


def _format_time_et(filing_dt: Optional[datetime]) -> str:
    if filing_dt is None:
        return "N/A"
    return _to_et(filing_dt).strftime("%H:%M")


def _format_filing_date(filing_dt: Optional[datetime], fallback: date) -> str:
    if filing_dt is not None:
        return _to_et(filing_dt).date().isoformat()
    return fallback.isoformat()


def process_date(
    target_date: date,
    *,
    lookback: int,
    min_pct: float,
    only_ge_min: bool,
    cache: Dict[Tuple[str, date, date], Optional[pd.Series]],
    quiet: bool,
) -> List[PreMoveRow]:
    filings = list(get_8ks_for_date(target_date) or [])
    rows: List[PreMoveRow] = []

    if not quiet:
        print(f"\n{target_date.isoformat()}: {len(filings)} 8-K filings")

    for filing in filings:
        try:
            candidate = analyze_8k(filing, verbose=False)
        except Exception as exc:
            if not quiet:
                print(f"  skip: {exc}")
            continue

        if candidate is None:
            continue

        ticker, cik, accession, _eps, filing_dt = candidate
        (
            close_tmN,
            anchor_close,
            pre_move_pct,
            ret_p1_pct,
            ret_p7_pct,
            late_share_pct,
            max_1d_jump_pct,
            up_days_lookback,
            creep_label,
        ) = compute_pre_move_pct(
            ticker, filing_dt, lookback, cache
        )
        flag = pre_move_pct is not None and pre_move_pct >= min_pct
        if only_ge_min and not flag:
            continue

        rows.append(
            PreMoveRow(
                filing_date=_format_filing_date(filing_dt, target_date),
                time_et=_format_time_et(filing_dt),
                ticker=(ticker or "").upper(),
                cik=str(cik or ""),
                accession=str(accession or ""),
                filing_bucket=filing_bucket(filing_dt),
                close_tmN=close_tmN,
                anchor_close=anchor_close,
                pre_move_pct=pre_move_pct,
                ret_p1_pct=ret_p1_pct,
                ret_p7_pct=ret_p7_pct,
                late_share_pct=late_share_pct,
                max_1d_jump_pct=max_1d_jump_pct,
                up_days_lookback=up_days_lookback,
                creep_label=creep_label,
                flag_ge_min=flag,
            )
        )
        time.sleep(0.05)

    return rows


def _iter_dates(from_date: date, to_date: date) -> Sequence[date]:
    out: List[date] = []
    d = from_date
    while d <= to_date:
        out.append(d)
        d += timedelta(days=1)
    return out


def _print_table(rows: List[PreMoveRow], lookback: int, min_pct: float) -> None:
    if not rows:
        print("\nNo rows to display.")
        return

    sorted_rows = sorted(
        rows,
        key=lambda r: (r.pre_move_pct is None, -(r.pre_move_pct or 0.0)),
    )

    print(f"\n{'=' * 152}")
    print(
        f"Pre-move close(T-{lookback}) → anchor close | min_pct={min_pct:g}% | "
        f"rows={len(sorted_rows)}"
    )
    print(f"{'=' * 152}")
    header = (
        f"{'Date':<12} {'Time':<6} {'Ticker':<8} {'Bucket':<12} "
        f"{f'Close-{lookback}d':>10} {'Anchor':>10} {'Move%':>8} "
        f"{'+1d%':>7} {'+7d%':>7} {'Late%':>7} {'Max1d%':>8} {'UpD':>4} {'Shape':<6} "
        f"{'>=min':>6} {'Accession':<22}"
    )
    print(header)
    print("-" * 152)

    for r in sorted_rows:
        tmN = f"${r.close_tmN:.2f}" if r.close_tmN is not None else "N/A"
        anc = f"${r.anchor_close:.2f}" if r.anchor_close is not None else "N/A"
        mv = f"{r.pre_move_pct:+.1f}%" if r.pre_move_pct is not None else "N/A"
        r1 = f"{r.ret_p1_pct:+.1f}%" if r.ret_p1_pct is not None else "N/A"
        r7 = f"{r.ret_p7_pct:+.1f}%" if r.ret_p7_pct is not None else "N/A"
        late = f"{r.late_share_pct:.0f}%" if r.late_share_pct is not None else "N/A"
        j1 = f"{r.max_1d_jump_pct:+.1f}%" if r.max_1d_jump_pct is not None else "N/A"
        upd = str(r.up_days_lookback) if r.up_days_lookback is not None else "N/A"
        flag = "Y" if r.flag_ge_min else "N"
        print(
            f"{r.filing_date:<12} {r.time_et:<6} {r.ticker:<8} {r.filing_bucket:<12} "
            f"{tmN:>10} {anc:>10} {mv:>8} {r1:>7} {r7:>7} {late:>7} {j1:>8} {upd:>4} {r.creep_label:<6} "
            f"{flag:>6} {(r.accession or '')[:22]:<22}"
        )
    print(f"{'=' * 152}\n")


def _write_csv(path: Path, rows: List[PreMoveRow], lookback: int) -> None:
    fieldnames = [
        "filing_date",
        "time_et",
        "ticker",
        "cik",
        "accession",
        "filing_bucket",
        f"close_t{lookback}d",
        "anchor_close",
        "pre_move_pct",
        "ret_p1_pct",
        "ret_p7_pct",
        "late_share_pct",
        "max_1d_jump_pct",
        "up_days_lookback",
        "creep_label",
        "flag_ge_min",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "filing_date": r.filing_date,
                    "time_et": r.time_et,
                    "ticker": r.ticker,
                    "cik": r.cik,
                    "accession": r.accession,
                    "filing_bucket": r.filing_bucket,
                    f"close_t{lookback}d": r.close_tmN,
                    "anchor_close": r.anchor_close,
                    "pre_move_pct": (
                        round(r.pre_move_pct, 2) if r.pre_move_pct is not None else None
                    ),
                    "ret_p1_pct": (round(r.ret_p1_pct, 2) if r.ret_p1_pct is not None else None),
                    "ret_p7_pct": (round(r.ret_p7_pct, 2) if r.ret_p7_pct is not None else None),
                    "late_share_pct": (
                        round(r.late_share_pct, 2) if r.late_share_pct is not None else None
                    ),
                    "max_1d_jump_pct": (
                        round(r.max_1d_jump_pct, 2) if r.max_1d_jump_pct is not None else None
                    ),
                    "up_days_lookback": r.up_days_lookback,
                    "creep_label": r.creep_label,
                    "flag_ge_min": r.flag_ge_min,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Earnings 8-K pre-filing price run (close T-N → anchor close before news)."
    )
    parser.add_argument("--date", help="Single calendar date YYYY-MM-DD")
    parser.add_argument("--from", dest="from_date", help="Season start YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", help="Season end YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help=f"Trading days back from anchor close (default {DEFAULT_LOOKBACK})",
    )
    parser.add_argument(
        "--min-pct",
        type=float,
        default=DEFAULT_MIN_PCT,
        help=f"Highlight threshold %% (default {DEFAULT_MIN_PCT:g})",
    )
    parser.add_argument(
        "--only-ge-min",
        action="store_true",
        help="Print/save only rows with pre_move_pct >= --min-pct",
    )
    parser.add_argument("--csv", type=Path, help="Write results to CSV")
    parser.add_argument("--quiet", action="store_true", help="Less per-day logging")
    args = parser.parse_args()

    if args.lookback <= 0:
        print("ERROR: --lookback must be positive", file=sys.stderr)
        raise SystemExit(2)

    if args.date:
        try:
            dates = [date.fromisoformat(args.date)]
        except ValueError:
            print(f"ERROR: invalid --date {args.date}", file=sys.stderr)
            raise SystemExit(2)
    elif args.from_date and args.to_date:
        try:
            start = date.fromisoformat(args.from_date)
            end = date.fromisoformat(args.to_date)
        except ValueError as exc:
            print(f"ERROR: invalid --from/--to: {exc}", file=sys.stderr)
            raise SystemExit(2)
        if end < start:
            print("ERROR: --to must be on or after --from", file=sys.stderr)
            raise SystemExit(2)
        dates = list(_iter_dates(start, end))
    else:
        parser.print_help()
        print("\nERROR: provide --date or both --from and --to", file=sys.stderr)
        raise SystemExit(2)

    cache: Dict[Tuple[str, date, date], Optional[pd.Series]] = {}
    all_rows: List[PreMoveRow] = []

    for d in dates:
        all_rows.extend(
            process_date(
                d,
                lookback=args.lookback,
                min_pct=args.min_pct,
                only_ge_min=args.only_ge_min,
                cache=cache,
                quiet=args.quiet,
            )
        )

    ge_min = [r for r in all_rows if r.flag_ge_min]
    if not args.quiet:
        print(
            f"\nSummary: {len(all_rows)} candidate row(s), "
            f"{len(ge_min)} with pre_move >= {args.min_pct:g}%"
        )

    _print_table(all_rows, args.lookback, args.min_pct)

    if args.csv:
        _write_csv(args.csv, all_rows, args.lookback)
        print(f"Wrote {len(all_rows)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()
