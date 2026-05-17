#!/usr/bin/env python3
"""
Standalone Flux entry-rule study (no Django).

For each signal day, measures forward close-to-close returns and whether
+tp% was touched within N sessions. Use to pick an entry filter before wiring Flux advisor.

Examples:
  python test_flux_entry.py --from 2020-01-01
  python test_flux_entry.py --entry-mode all
  python test_flux_entry.py --entry-mode below_ma_up --ma 20
  python test_flux_entry.py --symbols NVDA,AMD,TSLA --entry-mode pullback --pullback 0.03
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Flux production watchlist (duplicated here; not imported from core).
ENTRY_UNIVERSE: Tuple[str, ...] = (
    "AAPL", "AMD", "AMZN", "ARM", "AVGO", "CAT", "COIN", "COST", "CRWD",
    "GOOGL", "JPM", "KO", "LLY", "MA", "META", "MSFT", "MSTR", "NET", "NVDA",
    "PG", "PLTR", "TSM", "TSLA", "UNH", "V",
)

ENTRY_MODES = (
    "open",
    "pullback",
    "below_ma",
    "below_ma_up",
    "below_ma_up_trend",
    "rsi",
)

FORWARD_DAYS_DEFAULT = (1, 3, 5, 10)


@dataclass
class StudyConfig:
    symbols: List[str]
    start: pd.Timestamp
    end: pd.Timestamp
    ma_period: int
    pullback_pct: float
    lookback: int
    rsi_period: int
    rsi_max: float
    trend_lookback: int
    trend_min_pct: float
    min_price: float
    tp_mult: float
    tp_within_days: int
    mae_days: int


@dataclass
class SignalRecord:
    symbol: str
    date: pd.Timestamp
    entry_price: float
    fwd: Dict[int, float]
    hit_tp_within: bool
    mae_pct: float


def download_ohlc(
    symbols: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    if not symbols:
        raise ValueError("empty symbol list")

    start_buf = start - pd.Timedelta(days=400)
    data = yf.download(
        list(symbols),
        start=start_buf.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data.empty:
        raise RuntimeError("yfinance returned no data")

    out: Dict[str, pd.DataFrame] = {}
    fields = ("Open", "High", "Low", "Close")

    if isinstance(data.columns, pd.MultiIndex):
        for field in fields:
            frame = data[field].copy()
            frame.index = pd.to_datetime(frame.index).tz_localize(None)
            out[field.lower()] = frame.sort_index().loc[lambda df: df.index <= end]
    else:
        sym = symbols[0]
        for field in fields:
            col = field if field in data.columns else "Close"
            series = data[col].copy()
            frame = series.to_frame(name=sym)
            frame.index = pd.to_datetime(frame.index).tz_localize(None)
            out[field.lower()] = frame.sort_index().loc[lambda df: df.index <= end]

    return out


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _entry_price(
    cfg: StudyConfig,
    mode: str,
    close: float,
    low: float,
    limit: Optional[float],
) -> float:
    if mode == "pullback" and limit is not None and low <= limit:
        return limit
    return close


def entry_signal(
    cfg: StudyConfig,
    mode: str,
    dt: pd.Timestamp,
    close_s: pd.Series,
    low_s: pd.Series,
    sma: pd.Series,
    rolling_high: pd.Series,
    rsi: pd.Series,
) -> Optional[float]:
    if dt not in close_s.index:
        return None
    close = float(close_s.loc[dt])
    if close < cfg.min_price or np.isnan(close):
        return None
    low = float(low_s.loc[dt]) if dt in low_s.index else close
    if np.isnan(low):
        low = close

    if mode == "open":
        return _entry_price(cfg, mode, close, low, None)

    if mode == "pullback":
        rh = rolling_high.loc[dt] if dt in rolling_high.index else np.nan
        if pd.isna(rh) or rh <= 0:
            return None
        limit = float(rh) * (1.0 - cfg.pullback_pct)
        if low > limit:
            return None
        return _entry_price(cfg, mode, close, low, limit)

    if mode == "below_ma":
        ma = sma.loc[dt] if dt in sma.index else np.nan
        if pd.isna(ma) or close >= ma:
            return None
        return close

    if mode == "below_ma_up":
        ma = sma.loc[dt] if dt in sma.index else np.nan
        if pd.isna(ma) or close >= ma:
            return None
        idx = close_s.index.get_loc(dt)
        if idx < 1:
            return None
        prev_close = float(close_s.iloc[idx - 1])
        if close <= prev_close:
            return None
        return close

    if mode == "below_ma_up_trend":
        ma = sma.loc[dt] if dt in sma.index else np.nan
        if pd.isna(ma) or close >= ma:
            return None
        idx = close_s.index.get_loc(dt)
        if idx < 1:
            return None
        prev_close = float(close_s.iloc[idx - 1])
        if close <= prev_close:
            return None
        if idx < cfg.trend_lookback:
            return None
        ref = float(close_s.iloc[idx - cfg.trend_lookback])
        if ref <= 0:
            return None
        ret_lb = (close / ref - 1.0) * 100.0
        if ret_lb < cfg.trend_min_pct:
            return None
        return close

    if mode == "rsi":
        r = rsi.loc[dt] if dt in rsi.index else np.nan
        if pd.isna(r) or r > cfg.rsi_max:
            return None
        return close

    raise ValueError(f"unknown entry mode: {mode}")


def _forward_metrics(
    entry_dt: pd.Timestamp,
    entry_px: float,
    dates: pd.DatetimeIndex,
    close_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    fwd_days: Sequence[int],
    tp_mult: float,
    tp_within_days: int,
    mae_days: int,
) -> Optional[Tuple[Dict[int, float], bool, float]]:
    if entry_px <= 0:
        return None
    loc = dates.get_loc(entry_dt)
    if isinstance(loc, slice):
        return None
    max_fwd = max(max(fwd_days), tp_within_days, mae_days)
    if loc + max_fwd >= len(dates):
        return None

    fwd: Dict[int, float] = {}
    for d in fwd_days:
        px = float(close_s.iloc[loc + d])
        fwd[d] = (px / entry_px - 1.0) * 100.0

    tp_level = entry_px * tp_mult
    window_high = high_s.iloc[loc + 1 : loc + tp_within_days + 1]
    hit_tp = bool((window_high >= tp_level).any())

    window_low = low_s.iloc[loc + 1 : loc + mae_days + 1]
    min_low = float(window_low.min())
    mae_pct = (min_low / entry_px - 1.0) * 100.0

    return fwd, hit_tp, mae_pct


def collect_signals(
    cfg: StudyConfig,
    mode: str,
    ohlc: Dict[str, pd.DataFrame],
    fwd_days: Sequence[int],
) -> List[SignalRecord]:
    closes = ohlc["close"]
    highs = ohlc["high"]
    lows = ohlc["low"]
    symbols = [s for s in cfg.symbols if s in closes.columns]
    trade_dates = closes.index[(closes.index >= cfg.start) & (closes.index <= cfg.end)]

    records: List[SignalRecord] = []

    for sym in symbols:
        close_s = closes[sym].dropna()
        high_s = highs[sym].reindex(close_s.index)
        low_s = lows[sym].reindex(close_s.index)
        sma = close_s.rolling(cfg.ma_period, min_periods=cfg.ma_period).mean()
        rolling_high = close_s.rolling(cfg.lookback, min_periods=cfg.lookback).max()
        rsi = _rsi(close_s, cfg.rsi_period)

        for dt in trade_dates:
            if dt not in close_s.index:
                continue
            px = entry_signal(cfg, mode, dt, close_s, low_s, sma, rolling_high, rsi)
            if px is None:
                continue
            metrics = _forward_metrics(
                dt,
                px,
                close_s.index,
                close_s,
                high_s,
                low_s,
                fwd_days,
                cfg.tp_mult,
                cfg.tp_within_days,
                cfg.mae_days,
            )
            if metrics is None:
                continue
            fwd, hit_tp, mae = metrics
            records.append(
                SignalRecord(
                    symbol=sym,
                    date=dt,
                    entry_price=px,
                    fwd=fwd,
                    hit_tp_within=hit_tp,
                    mae_pct=mae,
                )
            )

    return records


def summarize(
    mode: str,
    records: List[SignalRecord],
    fwd_days: Sequence[int],
    sessions: int,
    n_symbols: int,
    mae_days: int,
) -> None:
    print("=" * 72)
    print(f"ENTRY MODE: {mode}")
    print("=" * 72)
    if not records:
        print("No signals in range.")
        print()
        return

    years = max(sessions / 252.0, 1e-6)
    per_sym_year = len(records) / max(n_symbols, 1) / years

    print(f"Signals:          {len(records)}  ({per_sym_year:.1f} per symbol per year)")
    for d in fwd_days:
        rets = [r.fwd[d] for r in records if d in r.fwd]
        if not rets:
            continue
        wins = sum(1 for x in rets if x > 0)
        print(
            f"  +{d}d return:     mean {np.mean(rets):+.2f}%  "
            f"median {np.median(rets):+.2f}%  win {wins / len(rets) * 100:.1f}%"
        )

    hit = sum(1 for r in records if r.hit_tp_within)
    print(f"  Hit TP within window: {hit / len(records) * 100:.1f}%  ({hit}/{len(records)})")
    maes = [r.mae_pct for r in records]
    print(f"  MAE ({mae_days}d):       mean {np.mean(maes):.2f}%  median {np.median(maes):.2f}%  worst {min(maes):.2f}%")
    print()


def print_comparison(rows: List[Tuple[str, int, float, float, float, float]]) -> None:
    print("=" * 72)
    print("COMPARISON (primary: 5d mean return, hit TP %, median MAE)")
    print("=" * 72)
    print(f"{'mode':<22} {'n':>6} {'5d mean':>9} {'5d med':>9} {'hitTP%':>8} {'MAE med':>9}")
    for mode, n, m5, med5, hit, mae in sorted(rows, key=lambda r: r[2], reverse=True):
        print(f"{mode:<22} {n:>6} {m5:>+8.2f}% {med5:>+8.2f}% {hit:>7.1f}% {mae:>8.2f}%")
    print()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flux entry-rule forward-return study.")
    p.add_argument("--from", dest="date_from", default="2020-01-01", metavar="DATE")
    p.add_argument("--to", dest="date_to", default=None, metavar="DATE")
    p.add_argument("--symbols", default=",".join(ENTRY_UNIVERSE))
    p.add_argument(
        "--entry-mode",
        choices=(*ENTRY_MODES, "all"),
        default="all",
        help="Entry rule to test, or 'all' for comparison table",
    )
    p.add_argument("--ma", type=int, default=20, help="SMA period for below_ma* modes")
    p.add_argument("--pullback", type=float, default=0.03, help="Pullback from N-day high (fraction)")
    p.add_argument("--lookback", type=int, default=20, help="High lookback for pullback")
    p.add_argument("--rsi-period", type=int, default=14)
    p.add_argument("--rsi-max", type=float, default=30.0, help="Enter when RSI <= this")
    p.add_argument("--trend-lookback", type=int, default=5, help="Days for min return filter")
    p.add_argument("--trend-min-pct", type=float, default=-2.0, help="Min %% return over trend-lookback")
    p.add_argument("--min-price", type=float, default=1.0)
    p.add_argument("--tp-mult", type=float, default=1.01, help="TP touch level = entry * mult")
    p.add_argument("--tp-within-days", type=int, default=5, help="Sessions to check TP touch")
    p.add_argument("--mae-days", type=int, default=5, help="Sessions for max adverse excursion")
    p.add_argument("--csv", default="", help="Write signals for single mode to CSV")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    end = pd.Timestamp(args.date_to) if args.date_to else pd.Timestamp.now().normalize()
    start = pd.Timestamp(args.date_from)
    if start >= end:
        print("error: --from must be before --to", file=sys.stderr)
        return 1

    symbols = [s.strip().upper().replace(".", "-") for s in args.symbols.split(",") if s.strip()]
    fwd_days = FORWARD_DAYS_DEFAULT

    cfg = StudyConfig(
        symbols=symbols,
        start=start,
        end=end,
        ma_period=args.ma,
        pullback_pct=args.pullback,
        lookback=args.lookback,
        rsi_period=args.rsi_period,
        rsi_max=args.rsi_max,
        trend_lookback=args.trend_lookback,
        trend_min_pct=args.trend_min_pct,
        min_price=args.min_price,
        tp_mult=args.tp_mult,
        tp_within_days=args.tp_within_days,
        mae_days=args.mae_days,
    )

    print(f"Downloading OHLC for {len(symbols)} symbols…")
    try:
        ohlc = download_ohlc(symbols, start, end)
    except Exception as exc:
        print(f"error: download failed: {exc}", file=sys.stderr)
        return 1

    closes = ohlc["close"]
    trade_dates = closes.index[(closes.index >= start) & (closes.index <= end)]
    n_symbols = len([s for s in symbols if s in closes.columns])
    sessions = len(trade_dates)

    print(f"Range: {start.date()} → {end.date()} ({sessions} sessions, {n_symbols} symbols)")
    print(f"TP touch: high >= entry×{args.tp_mult} within {args.tp_within_days} sessions\n")

    modes = list(ENTRY_MODES) if args.entry_mode == "all" else [args.entry_mode]
    comparison: List[Tuple[str, int, float, float, float, float]] = []

    for mode in modes:
        records = collect_signals(cfg, mode, ohlc, fwd_days)
        summarize(mode, records, fwd_days, sessions, n_symbols, cfg.mae_days)

        if records:
            rets5 = [r.fwd[5] for r in records if 5 in r.fwd]
            hit_pct = sum(1 for r in records if r.hit_tp_within) / len(records) * 100.0
            mae_med = float(np.median([r.mae_pct for r in records]))
            comparison.append(
                (
                    mode,
                    len(records),
                    float(np.mean(rets5)) if rets5 else 0.0,
                    float(np.median(rets5)) if rets5 else 0.0,
                    hit_pct,
                    mae_med,
                )
            )

        if args.csv and mode == args.entry_mode and args.entry_mode != "all":
            rows = [
                {
                    "symbol": r.symbol,
                    "date": r.date.strftime("%Y-%m-%d"),
                    "entry_price": round(r.entry_price, 4),
                    **{f"ret_{d}d": round(r.fwd[d], 3) for d in fwd_days if d in r.fwd},
                    "hit_tp_within": r.hit_tp_within,
                    "mae_pct": round(r.mae_pct, 3),
                }
                for r in records
            ]
            pd.DataFrame(rows).to_csv(args.csv, index=False)
            print(f"Wrote {len(rows)} signals to {args.csv}\n")

    if len(modes) > 1:
        print_comparison(comparison)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
