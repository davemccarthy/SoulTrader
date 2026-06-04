#!/usr/bin/env python3
"""
Flux Entry v2 filter study (standalone, no Django).

Layers (cumulative):
  1. Universe — symbol in approved FLUX list
  2. Pullback — close <= (1 - pullback_pct) × N-day high (default 4% off 20d high)
  3. Relative weakness — 20d return minus benchmark <= -rel_weak_pct (default -2%)
     Benchmark: QQQ (growth/tech) or SPY (other); see SYMBOL_BENCHMARK

Also compares combined v2 stack vs production-style below_ma_up.

Examples:
  python test_flux_entry_v2.py --no-history
  python test_flux_entry_v2.py --no-history --to 2025-10-15   # snapshot on that session
  python test_flux_entry_v2.py --matrix --from 2022-01-01    # filter variant comparison
  python test_flux_entry_v2.py --from 2022-01-01
  python test_flux_entry_v2.py --pullback 0.04 --rel-weak 2.0 --lookback 20
  python test_flux_entry_v2.py --symbols NFLX,GOOGL,MA --from 2024-01-01
  python test_flux_entry_v2.py --csv v2_signals.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Match core/services/advisors/flux.py FLUX_UNIVERSE
FLUX_UNIVERSE: Tuple[str, ...] = (
    "AAPL",
    "ADBE",
    "AMD",
    "AMZN",
    "ARM",
    "AVGO",
    "CAT",
    "COST",
    "CRM",
    "CRWD",
    "GOOGL",
    "JPM",
    "LLY",
    "MA",
    "META",
    "MSFT",
    "MU",
    "NET",
    "NFLX",
    "NOW",
    "NVDA",
    "PLTR",
    "TSM",
    "TSLA",
    "UBER",
    "UNH",
    "V",
)

# Growth / tech vs rest → QQQ vs SPY (tweak as needed)
SYMBOL_BENCHMARK: Dict[str, str] = {
    "AAPL": "QQQ",
    "ADBE": "QQQ",
    "AMD": "QQQ",
    "AMZN": "QQQ",
    "ARM": "QQQ",
    "AVGO": "QQQ",
    "CRM": "QQQ",
    "CRWD": "QQQ",
    "GOOGL": "QQQ",
    "META": "QQQ",
    "MSFT": "QQQ",
    "MU": "QQQ",
    "NET": "QQQ",
    "NFLX": "QQQ",
    "NOW": "QQQ",
    "NVDA": "QQQ",
    "PLTR": "QQQ",
    "TSLA": "QQQ",
    "TSM": "QQQ",
}
DEFAULT_BENCHMARK = "SPY"

FORWARD_DAYS_DEFAULT = (1, 3, 5, 10)


@dataclass
class StudyConfig:
    symbols: List[str]
    start: pd.Timestamp
    end: pd.Timestamp
    lookback: int
    pullback_pct: float
    rel_weak_pct: float
    ret_lookback: int
    ma_period: int
    min_price: float
    tp_mult: float
    tp_within_days: int
    mae_days: int


@dataclass
class SignalRecord:
    symbol: str
    date: pd.Timestamp
    entry_price: float
    benchmark: str
    pullback_pct: float
    rel_weak: float
    stock_ret_20d: float
    bench_ret_20d: float
    pass_pullback: bool
    pass_rel_weak: bool
    pass_below_ma_up: bool
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


def ret_over_sessions(close: pd.Series, dt: pd.Timestamp, sessions: int) -> Optional[float]:
    if dt not in close.index:
        return None
    loc = close.index.get_loc(dt)
    if isinstance(loc, slice) or loc < sessions:
        return None
    cur = float(close.iloc[loc])
    ref = float(close.iloc[loc - sessions])
    if ref <= 0 or cur <= 0 or np.isnan(ref) or np.isnan(cur):
        return None
    return (cur / ref - 1.0) * 100.0


def benchmark_for(symbol: str) -> str:
    return SYMBOL_BENCHMARK.get(symbol.upper(), DEFAULT_BENCHMARK)


def pass_pullback(
    close_s: pd.Series,
    dt: pd.Timestamp,
    lookback: int,
    pullback_pct: float,
) -> Tuple[bool, float, float]:
    """Close <= (1 - pullback_pct) × N-day high. Returns (pass, close, pct_below_high)."""
    if dt not in close_s.index:
        return False, 0.0, 0.0
    window = close_s.loc[:dt].tail(lookback)
    if len(window) < lookback:
        return False, 0.0, 0.0
    rh = float(window.max())
    close = float(close_s.loc[dt])
    if rh <= 0 or close <= 0:
        return False, close, 0.0
    threshold = rh * (1.0 - pullback_pct)
    pct_below = (1.0 - close / rh) * 100.0
    return close <= threshold, close, pct_below


def pass_below_ma_up(
    close_s: pd.Series,
    dt: pd.Timestamp,
    ma_period: int,
) -> bool:
    if dt not in close_s.index:
        return False
    window = close_s.loc[:dt]
    if len(window) < ma_period + 1:
        return False
    ma = float(window.tail(ma_period).mean())
    close = float(close_s.loc[dt])
    loc = close_s.index.get_loc(dt)
    if isinstance(loc, slice) or loc < 1:
        return False
    prev = float(close_s.iloc[loc - 1])
    if ma <= 0:
        return False
    return close < ma and close > prev


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


def collect_layer_stats(
    cfg: StudyConfig,
    ohlc: Dict[str, pd.DataFrame],
    bench_closes: Dict[str, pd.Series],
    fwd_days: Sequence[int],
) -> Tuple[List[SignalRecord], Dict[str, int]]:
    closes = ohlc["close"]
    highs = ohlc["high"]
    lows = ohlc["low"]
    symbols = [s for s in cfg.symbols if s in closes.columns]
    trade_dates = closes.index[(closes.index >= cfg.start) & (closes.index <= cfg.end)]

    records: List[SignalRecord] = []
    funnel = {
        "universe_days": 0,
        "pullback": 0,
        "rel_weak": 0,
        "pullback_and_rel": 0,
        "v2_stack": 0,
        "below_ma_up": 0,
        "below_ma_up_not_v2": 0,
        "v2_not_below_ma_up": 0,
    }

    for sym in symbols:
        bench_sym = benchmark_for(sym)
        bench_s = bench_closes.get(bench_sym)
        if bench_s is None:
            continue

        close_s = closes[sym].dropna()
        high_s = highs[sym].reindex(close_s.index)
        low_s = lows[sym].reindex(close_s.index)

        for dt in trade_dates:
            if dt not in close_s.index:
                continue
            funnel["universe_days"] += 1

            ok_pb, px, _pct_below = pass_pullback(
                close_s, dt, cfg.lookback, cfg.pullback_pct
            )
            if ok_pb:
                funnel["pullback"] += 1

            stock_ret = ret_over_sessions(close_s, dt, cfg.ret_lookback)
            bench_ret = ret_over_sessions(bench_s, dt, cfg.ret_lookback)
            rel_weak = None
            ok_rw = False
            if stock_ret is not None and bench_ret is not None:
                rel_weak = stock_ret - bench_ret
                ok_rw = rel_weak <= -cfg.rel_weak_pct
                if ok_rw:
                    funnel["rel_weak"] += 1

            if ok_pb and ok_rw:
                funnel["pullback_and_rel"] += 1

            ok_ma_up = pass_below_ma_up(close_s, dt, cfg.ma_period)
            if ok_ma_up:
                funnel["below_ma_up"] += 1

            v2 = ok_pb and ok_rw
            if v2:
                funnel["v2_stack"] += 1
                if not ok_ma_up:
                    funnel["v2_not_below_ma_up"] += 1
            if ok_ma_up and not v2:
                funnel["below_ma_up_not_v2"] += 1

            if not v2:
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
                    benchmark=bench_sym,
                    pullback_pct=_pct_below,
                    rel_weak=float(rel_weak) if rel_weak is not None else 0.0,
                    stock_ret_20d=float(stock_ret) if stock_ret is not None else 0.0,
                    bench_ret_20d=float(bench_ret) if bench_ret is not None else 0.0,
                    pass_pullback=ok_pb,
                    pass_rel_weak=ok_rw,
                    pass_below_ma_up=ok_ma_up,
                    fwd=fwd,
                    hit_tp_within=hit_tp,
                    mae_pct=mae,
                )
            )

    return records, funnel


def print_funnel(
    funnel: Dict[str, int],
    n_symbols: int,
    sessions: int,
    cfg: StudyConfig,
) -> None:
    years = max(sessions / 252.0, 1e-6)
    uni = funnel["universe_days"]
    print("=" * 72)
    print("ENTRY V2 LAYER FUNNEL (symbol × session cells in universe)")
    print("=" * 72)
    print(f"Universe:     {len(cfg.symbols)} symbols ({', '.join(cfg.symbols[:5])}…)")
    print(f"Pullback:     close <= {100 * cfg.pullback_pct:.1f}% below {cfg.lookback}d high")
    print(f"Rel weakness: 20d stock − 20d bench <= -{cfg.rel_weak_pct:.1f}%  (QQQ/SPY map)")
    print(f"Sessions:     {sessions}  (~{years:.1f} years, {n_symbols} symbols with data)\n")

    def rate(n: int, base: int) -> str:
        if base <= 0:
            return "—"
        return f"{n / base * 100:.1f}%"

    print(f"  L1 Universe cells:        {uni:>7}  (100%)")
    print(f"  L2 Pullback:              {funnel['pullback']:>7}  ({rate(funnel['pullback'], uni)} of L1)")
    print(f"  L3 Relative weakness:     {funnel['rel_weak']:>7}  ({rate(funnel['rel_weak'], uni)} of L1)")
    print(f"  L2 + L3 (v2 stack):       {funnel['v2_stack']:>7}  ({rate(funnel['v2_stack'], uni)} of L1)")
    print(f"  Production below_ma_up:   {funnel['below_ma_up']:>7}  ({rate(funnel['below_ma_up'], uni)} of L1)")
    print(f"  v2 only (not below_ma_up): {funnel['v2_not_below_ma_up']:>7}")
    print(f"  below_ma_up not v2:       {funnel['below_ma_up_not_v2']:>7}")
    if funnel["v2_stack"] > 0:
        per_year = funnel["v2_stack"] / max(n_symbols, 1) / years
        print(f"\n  v2 signals per symbol-year: {per_year:.2f}")
    print()


def per_symbol_counts(records: List[SignalRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = []
    for sym in sorted({r.symbol for r in records}):
        sub = [r for r in records if r.symbol == sym]
        rows.append({"symbol": sym, "v2_signals": len(sub), "benchmark": benchmark_for(sym)})
    return pd.DataFrame(rows).sort_values("v2_signals", ascending=False)


def summarize_signals(
    label: str,
    records: List[SignalRecord],
    fwd_days: Sequence[int],
    sessions: int,
    n_symbols: int,
    mae_days: int,
) -> None:
    print("=" * 72)
    print(label)
    print("=" * 72)
    if not records:
        print("No signals.")
        print()
        return

    years = max(sessions / 252.0, 1e-6)
    print(f"Signals: {len(records)}  ({len(records) / max(n_symbols, 1) / years:.2f} per symbol-year)")
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
    print(f"  Hit TP within window: {hit / len(records) * 100:.1f}%")
    maes = [r.mae_pct for r in records]
    print(f"  MAE ({mae_days}d):       mean {np.mean(maes):.2f}%  median {np.median(maes):.2f}%  worst {min(maes):.2f}%")
    print()


def summarize_v2_by_ma_up(
    records: List[SignalRecord],
    fwd_days: Sequence[int],
    mae_days: int,
) -> None:
    """Among v2 stack signals, compare forward stats when below_ma_up also passes."""
    if not records:
        return
    yes = [r for r in records if r.pass_below_ma_up]
    no = [r for r in records if not r.pass_below_ma_up]
    print("=" * 72)
    print("V2 STACK SPLIT BY below_ma_up (production entry overlay)")
    print("=" * 72)
    print(
        f"  v2 signals: {len(records)}   ma_up=Y: {len(yes)} ({len(yes) / len(records) * 100:.1f}%)   "
        f"ma_up=n: {len(no)}"
    )
    print("  (Y = close < SMA and close > prior close, same as Flux discover today)\n")

    for label, sub in (("v2 + ma_up=Y", yes), ("v2 + ma_up=n", no)):
        print(f"  {label}  (n={len(sub)})")
        if not sub:
            print("    No signals.\n")
            continue
        for d in fwd_days:
            rets = [r.fwd[d] for r in sub if d in r.fwd]
            if not rets:
                continue
            wins = sum(1 for x in rets if x > 0)
            print(
                f"    +{d}d return:     mean {np.mean(rets):+.2f}%  "
                f"median {np.median(rets):+.2f}%  win {wins / len(rets) * 100:.1f}%"
            )
        hit = sum(1 for r in sub if r.hit_tp_within)
        print(f"    Hit TP within window: {hit / len(sub) * 100:.1f}%")
        maes = [r.mae_pct for r in sub]
        print(
            f"    MAE ({mae_days}d):       mean {np.mean(maes):.2f}%  "
            f"median {np.median(maes):.2f}%  worst {min(maes):.2f}%"
        )
        print()


def filter_top_n_per_day(records: List[SignalRecord], n: int) -> List[SignalRecord]:
    """Keep up to n signals per session, most negative rel first (cadence cap)."""
    if n <= 0:
        return []
    by_date: Dict[pd.Timestamp, List[SignalRecord]] = {}
    for r in records:
        by_date.setdefault(r.date, []).append(r)
    out: List[SignalRecord] = []
    for group in by_date.values():
        group.sort(key=lambda r: r.rel_weak)
        out.extend(group[:n])
    return out


@dataclass
class MatrixRow:
    label: str
    n: int
    per_sym_year: float
    ret_5d: float
    ret_10d: float
    tp_pct: float
    mae_mean: float
    mae_worst: float


def _matrix_stats(
    records: List[SignalRecord],
    fwd_days: Sequence[int],
    sessions: int,
    n_symbols: int,
) -> Optional[MatrixRow]:
    if not records:
        return None
    years = max(sessions / 252.0, 1e-6)
    r5 = [r.fwd[5] for r in records if 5 in r.fwd]
    r10 = [r.fwd[10] for r in records if 10 in r.fwd]
    if not r5:
        return None
    tp = sum(1 for r in records if r.hit_tp_within) / len(records) * 100.0
    maes = [r.mae_pct for r in records]
    return MatrixRow(
        label="",
        n=len(records),
        per_sym_year=len(records) / max(n_symbols, 1) / years,
        ret_5d=float(np.mean(r5)),
        ret_10d=float(np.mean(r10)) if r10 else float("nan"),
        tp_pct=tp,
        mae_mean=float(np.mean(maes)),
        mae_worst=float(min(maes)),
    )


def build_matrix_variants(top_n: int) -> List[Tuple[str, Callable[[List[SignalRecord]], List[SignalRecord]]]]:
    """Label + filter applied on top of baseline v2 signals."""

    def band(lo: float, hi: float) -> Callable[[List[SignalRecord]], List[SignalRecord]]:
        return lambda rs: [r for r in rs if lo <= r.rel_weak <= hi]

    return [
        ("v2 baseline", lambda rs: rs),
        ("s20 >= 0 (MSFT-style)", lambda rs: [r for r in rs if r.stock_ret_20d >= 0]),
        ("rel [-12,-4]", band(-12.0, -4.0)),
        ("SPY bench only", lambda rs: [r for r in rs if r.benchmark == "SPY"]),
        (f"top {top_n}/day by rel", lambda rs: filter_top_n_per_day(rs, top_n)),
        ("rel [-8,-2) mild", band(-8.0, -2.0)),
        ("rel [-15,-8) mid", band(-15.0, -8.0)),
        ("rel < -15 deep", lambda rs: [r for r in rs if r.rel_weak < -15]),
        (
            "COST-ish SPY pb>=8 rel[-15,-5]",
            lambda rs: [
                r
                for r in rs
                if r.benchmark == "SPY"
                and r.pullback_pct >= 8.0
                and -15.0 <= r.rel_weak <= -5.0
            ],
        ),
        (
            "knife s20<-5 rel<-10",
            lambda rs: [r for r in rs if r.stock_ret_20d < -5.0 and r.rel_weak < -10.0],
        ),
    ]


def print_entry_matrix(
    records: List[SignalRecord],
    fwd_days: Sequence[int],
    sessions: int,
    n_symbols: int,
    mae_days: int,
    cfg: StudyConfig,
    top_n: int,
) -> None:
    """One-off comparison of v2 sub-filters (s20 / rel bands / SPY / daily cap)."""
    variants = build_matrix_variants(top_n)
    rows: List[MatrixRow] = []
    for label, fn in variants:
        sub = fn(records)
        stats = _matrix_stats(sub, fwd_days, sessions, n_symbols)
        if stats is None:
            rows.append(
                MatrixRow(label=label, n=0, per_sym_year=0.0, ret_5d=0.0, ret_10d=0.0, tp_pct=0.0, mae_mean=0.0, mae_worst=0.0)
            )
        else:
            stats.label = label
            rows.append(stats)

    years = max(sessions / 252.0, 1e-6)
    print("=" * 100)
    print("ENTRY V2 FILTER MATRIX (one-off comparison on same v2 signal pool)")
    print("=" * 100)
    print(
        f"Base: pullback >= {100 * cfg.pullback_pct:.1f}% below {cfg.lookback}d high, "
        f"rel <= -{cfg.rel_weak_pct:.1f}% vs QQQ/SPY"
    )
    print(
        f"Pool: {len(records)} v2 signals | {sessions} sessions | {n_symbols} symbols | "
        f"~{years:.1f}y | TP {cfg.tp_mult} within {cfg.tp_within_days}d | MAE {mae_days}d\n"
    )
    hdr = (
        f"{'variant':<32} {'n':>6} {'/sym-yr':>7} {'+5d':>7} {'+10d':>7} "
        f"{'TP%':>6} {'MAE':>7} {'worst':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        if row.n == 0:
            print(f"{row.label:<32} {'—':>6} {'—':>7} {'—':>7} {'—':>7} {'—':>6} {'—':>7} {'—':>7}")
            continue
        r10 = f"{row.ret_10d:+.2f}" if not np.isnan(row.ret_10d) else "  n/a"
        print(
            f"{row.label:<32} {row.n:>6} {row.per_sym_year:>7.2f} {row.ret_5d:>+6.2f}% {r10:>6}% "
            f"{row.tp_pct:>5.1f}% {row.mae_mean:>+6.2f}% {row.mae_worst:>+6.2f}%"
        )
    print()
    print("Notes: /sym-yr = signals per symbol per year. top N/day keeps most negative rel per session.")
    print()


def print_recent(records: List[SignalRecord]) -> None:
    if not records:
        return
    print("=" * 72)
    print(f"ALL V2 SIGNALS ({len(records)} rows, chronological)")
    print("=" * 72)
    recent = sorted(records, key=lambda r: r.date)
    print(f"{'date':<12} {'sym':<6} {'bench':<4} {'px':>8} {'pb%':>6} {'rel':>6} {'s20':>7} {'b20':>7} {'ma_up':>5}")
    for r in recent:
        print(
            f"{r.date.strftime('%Y-%m-%d'):<12} {r.symbol:<6} {r.benchmark:<4} "
            f"{r.entry_price:>8.2f} {r.pullback_pct:>5.1f}% {r.rel_weak:>+5.1f}% "
            f"{r.stock_ret_20d:>+6.1f}% {r.bench_ret_20d:>+6.1f}% "
            f"{'Y' if r.pass_below_ma_up else 'n':>5}"
        )
    print()


def collect_below_ma_up_only(
    cfg: StudyConfig,
    ohlc: Dict[str, pd.DataFrame],
    bench_closes: Dict[str, pd.Series],
    fwd_days: Sequence[int],
) -> List[SignalRecord]:
    """Signals that pass below_ma_up but NOT v2 stack (for comparison metrics)."""
    closes = ohlc["close"]
    highs = ohlc["high"]
    lows = ohlc["low"]
    symbols = [s for s in cfg.symbols if s in closes.columns]
    trade_dates = closes.index[(closes.index >= cfg.start) & (closes.index <= cfg.end)]
    records: List[SignalRecord] = []

    for sym in symbols:
        bench_sym = benchmark_for(sym)
        if bench_sym not in bench_closes:
            continue
        bench_s = bench_closes[bench_sym]
        close_s = closes[sym].dropna()
        high_s = highs[sym].reindex(close_s.index)
        low_s = lows[sym].reindex(close_s.index)

        for dt in trade_dates:
            if dt not in close_s.index:
                continue
            ok_pb, px, pct_below = pass_pullback(close_s, dt, cfg.lookback, cfg.pullback_pct)
            stock_ret = ret_over_sessions(close_s, dt, cfg.ret_lookback)
            bench_ret = ret_over_sessions(bench_s, dt, cfg.ret_lookback)
            rel_weak = (stock_ret - bench_ret) if stock_ret is not None and bench_ret is not None else None
            ok_rw = rel_weak is not None and rel_weak <= -cfg.rel_weak_pct
            ok_ma_up = pass_below_ma_up(close_s, dt, cfg.ma_period)
            if not ok_ma_up or (ok_pb and ok_rw):
                continue
            metrics = _forward_metrics(
                dt, px if px > 0 else float(close_s.loc[dt]),
                close_s.index, close_s, high_s, low_s,
                fwd_days, cfg.tp_mult, cfg.tp_within_days, cfg.mae_days,
            )
            if metrics is None:
                continue
            fwd, hit_tp, mae = metrics
            entry_px = float(close_s.loc[dt])
            records.append(
                SignalRecord(
                    symbol=sym,
                    date=dt,
                    entry_price=entry_px,
                    benchmark=bench_sym,
                    pullback_pct=pct_below,
                    rel_weak=float(rel_weak) if rel_weak is not None else 0.0,
                    stock_ret_20d=float(stock_ret) if stock_ret is not None else 0.0,
                    bench_ret_20d=float(bench_ret) if bench_ret is not None else 0.0,
                    pass_pullback=ok_pb,
                    pass_rel_weak=ok_rw,
                    pass_below_ma_up=True,
                    fwd=fwd,
                    hit_tp_within=hit_tp,
                    mae_pct=mae,
                )
            )
    return records


@dataclass
class SnapshotRow:
    symbol: str
    benchmark: str
    entry_price: float
    pullback_pct: float
    rel_weak: float
    stock_ret_20d: float
    bench_ret_20d: float
    pass_pullback: bool
    pass_rel_weak: bool
    pass_v2: bool
    pass_below_ma_up: bool


def collect_snapshot(
    cfg: StudyConfig,
    ohlc: Dict[str, pd.DataFrame],
    bench_closes: Dict[str, pd.Series],
    as_of: pd.Timestamp,
) -> Tuple[pd.Timestamp, List[SnapshotRow], Dict[str, int]]:
    """Evaluate entry layers on a single session (latest bar <= as_of)."""
    closes = ohlc["close"]
    symbols = [s for s in cfg.symbols if s in closes.columns]
    valid = closes.index[closes.index <= as_of]
    if len(valid) == 0:
        raise ValueError(f"no sessions on or before {as_of.date()}")

    dt = valid[-1]
    rows: List[SnapshotRow] = []
    counts = {
        "symbols": len(symbols),
        "pullback": 0,
        "rel_weak": 0,
        "v2": 0,
        "below_ma_up": 0,
    }

    for sym in symbols:
        bench_sym = benchmark_for(sym)
        bench_s = bench_closes.get(bench_sym)
        if bench_s is None or bench_s.empty:
            continue

        close_s = closes[sym].dropna()
        if dt not in close_s.index:
            continue

        ok_pb, px, pct_below = pass_pullback(
            close_s, dt, cfg.lookback, cfg.pullback_pct
        )
        if ok_pb:
            counts["pullback"] += 1

        stock_ret = ret_over_sessions(close_s, dt, cfg.ret_lookback)
        bench_ret = ret_over_sessions(bench_s, dt, cfg.ret_lookback)
        rel_weak = None
        ok_rw = False
        if stock_ret is not None and bench_ret is not None:
            rel_weak = stock_ret - bench_ret
            ok_rw = rel_weak <= -cfg.rel_weak_pct
            if ok_rw:
                counts["rel_weak"] += 1

        ok_ma_up = pass_below_ma_up(close_s, dt, cfg.ma_period)
        if ok_ma_up:
            counts["below_ma_up"] += 1

        v2 = ok_pb and ok_rw
        if v2:
            counts["v2"] += 1

        rows.append(
            SnapshotRow(
                symbol=sym,
                benchmark=bench_sym,
                entry_price=px if px > 0 else float(close_s.loc[dt]),
                pullback_pct=pct_below,
                rel_weak=float(rel_weak) if rel_weak is not None else float("nan"),
                stock_ret_20d=float(stock_ret) if stock_ret is not None else float("nan"),
                bench_ret_20d=float(bench_ret) if bench_ret is not None else float("nan"),
                pass_pullback=ok_pb,
                pass_rel_weak=ok_rw,
                pass_v2=v2,
                pass_below_ma_up=ok_ma_up,
            )
        )

    return dt, rows, counts


def print_snapshot(
    dt: pd.Timestamp,
    rows: List[SnapshotRow],
    counts: Dict[str, int],
    cfg: StudyConfig,
) -> None:
    v2_rows = [r for r in rows if r.pass_v2]
    v2_rows.sort(key=lambda r: (r.rel_weak, -r.pullback_pct))

    print("=" * 72)
    print("ENTRY SNAPSHOT (latest session only)")
    print("=" * 72)
    print(f"As of:        {dt.strftime('%Y-%m-%d')}")
    print(f"Universe:     {counts['symbols']} symbols")
    print(f"Pullback:     close <= {100 * cfg.pullback_pct:.1f}% below {cfg.lookback}d high")
    print(f"Rel weakness: {cfg.ret_lookback}d stock − bench <= -{cfg.rel_weak_pct:.1f}%  (QQQ/SPY map)")
    print()
    print(
        f"  L2 pullback:        {counts['pullback']:>3} / {counts['symbols']}"
    )
    print(
        f"  L3 rel weakness:    {counts['rel_weak']:>3} / {counts['symbols']}"
    )
    print(
        f"  L2+L3 (v2 entry):   {counts['v2']:>3} / {counts['symbols']}"
    )
    print(
        f"  below_ma_up:        {counts['below_ma_up']:>3} / {counts['symbols']}"
    )
    print()

    if not v2_rows:
        print("No v2 entry candidates on this session.")
        print()
        partial = [r for r in rows if r.pass_pullback or r.pass_rel_weak]
        if partial:
            print("Partial passes (pullback and/or rel weak, not both):")
            partial.sort(key=lambda r: (-int(r.pass_pullback), r.rel_weak if not np.isnan(r.rel_weak) else 0))
            print(f"{'sym':<6} {'bench':<4} {'px':>8} {'pb':>4} {'rel':>4} {'pull':>6} {'relW':>6} {'ma_up':>5}")
            for r in partial:
                print(
                    f"{r.symbol:<6} {r.benchmark:<4} {r.entry_price:>8.2f} "
                    f"{'Y' if r.pass_pullback else 'n':>4} {'Y' if r.pass_rel_weak else 'n':>4} "
                    f"{r.pullback_pct:>5.1f}% {r.rel_weak:>+5.1f}% "
                    f"{'Y' if r.pass_below_ma_up else 'n':>5}"
                )
            print()
        return

    print(f"V2 ENTRY CANDIDATES ({len(v2_rows)})")
    print(f"{'sym':<6} {'bench':<4} {'px':>8} {'pb%':>6} {'rel':>7} {'s20':>7} {'b20':>7} {'ma_up':>5}")
    for r in v2_rows:
        print(
            f"{r.symbol:<6} {r.benchmark:<4} {r.entry_price:>8.2f} "
            f"{r.pullback_pct:>5.1f}% {r.rel_weak:>+6.1f}% "
            f"{r.stock_ret_20d:>+6.1f}% {r.bench_ret_20d:>+6.1f}% "
            f"{'Y' if r.pass_below_ma_up else 'n':>5}"
        )
    print()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flux Entry v2 layer study (universe + pullback + rel weakness).")
    p.add_argument("--from", dest="date_from", default="2022-01-01", metavar="DATE")
    p.add_argument(
        "--to",
        dest="date_to",
        default=None,
        metavar="DATE",
        help="End date. With --no-history: snapshot date (last session on or before DATE; default today). "
        "Otherwise: last session in backtest range.",
    )
    p.add_argument("--symbols", default=",".join(FLUX_UNIVERSE), help="Comma list or FLUX subset")
    p.add_argument("--pullback", type=float, default=0.04, help="Min %% below N-day high (0.04 = 4%%)")
    p.add_argument("--rel-weak", type=float, default=2.0, help="Min underperformance vs bench (%% points)")
    p.add_argument("--lookback", type=int, default=20, help="High lookback for pullback")
    p.add_argument("--ret-lookback", type=int, default=20, help="Sessions for 20d return")
    p.add_argument("--ma", type=int, default=20, help="SMA period for below_ma_up compare")
    p.add_argument("--min-price", type=float, default=1.0)
    p.add_argument("--tp-mult", type=float, default=1.01)
    p.add_argument("--tp-within-days", type=int, default=5)
    p.add_argument("--mae-days", type=int, default=5)
    p.add_argument("--csv", default="", help="Write v2 stack signals to CSV")
    p.add_argument("--no-compare", action="store_true", help="Skip below_ma_up-only forward stats")
    p.add_argument(
        "--no-history",
        action="store_true",
        help="Snapshot only: one session (--to, or today). Ignores --from. No funnel/backtest.",
    )
    p.add_argument(
        "--matrix",
        action="store_true",
        help="One-off matrix: compare v2 filter variants (s20, rel bands, SPY, top-N/day). Needs --from/--to.",
    )
    p.add_argument(
        "--matrix-top-n",
        type=int,
        default=5,
        metavar="N",
        help="Daily cap variant: keep N weakest-rel signals per session (default 5)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.matrix and args.no_history:
        print("error: --matrix cannot be used with --no-history", file=sys.stderr)
        return 1

    end = pd.Timestamp(args.date_to) if args.date_to else pd.Timestamp.now().normalize()
    if args.no_history:
        # Enough history for lookback / 20d returns; no full backtest range needed.
        start = end - pd.Timedelta(days=120)
    else:
        start = pd.Timestamp(args.date_from)
        if start >= end:
            print("error: --from must be before --to", file=sys.stderr)
            return 1

    symbols = [s.strip().upper().replace(".", "-") for s in args.symbols.split(",") if s.strip()]
    # L1: restrict to universe membership
    universe_set = set(FLUX_UNIVERSE)
    symbols = [s for s in symbols if s in universe_set]
    if not symbols:
        print("error: no symbols left after universe filter", file=sys.stderr)
        return 1

    fwd_days = FORWARD_DAYS_DEFAULT
    cfg = StudyConfig(
        symbols=symbols,
        start=start,
        end=end,
        lookback=args.lookback,
        pullback_pct=args.pullback,
        rel_weak_pct=args.rel_weak,
        ret_lookback=args.ret_lookback,
        ma_period=args.ma,
        min_price=args.min_price,
        tp_mult=args.tp_mult,
        tp_within_days=args.tp_within_days,
        mae_days=args.mae_days,
    )

    download_syms = list(dict.fromkeys(symbols + ["QQQ", "SPY"]))
    print(f"Downloading OHLC for {len(download_syms)} tickers (universe + QQQ + SPY)…")
    try:
        ohlc = download_ohlc(download_syms, start, end)
    except Exception as exc:
        print(f"error: download failed: {exc}", file=sys.stderr)
        return 1

    closes = ohlc["close"]
    bench_closes = {
        "QQQ": closes["QQQ"].dropna() if "QQQ" in closes.columns else pd.Series(dtype=float),
        "SPY": closes["SPY"].dropna() if "SPY" in closes.columns else pd.Series(dtype=float),
    }
    if bench_closes["QQQ"].empty or bench_closes["SPY"].empty:
        print("error: missing QQQ or SPY benchmark data", file=sys.stderr)
        return 1

    if args.no_history:
        snap_label = args.date_to if args.date_to else "today"
        print(f"Snapshot date (--to): {snap_label}\n")
        try:
            dt, snap_rows, snap_counts = collect_snapshot(cfg, ohlc, bench_closes, end)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print_snapshot(dt, snap_rows, snap_counts, cfg)
        return 0

    trade_dates = closes.index[(closes.index >= start) & (closes.index <= end)]
    n_symbols = len([s for s in symbols if s in closes.columns])
    sessions = len(trade_dates)

    print(f"Range: {start.date()} → {end.date()} ({sessions} sessions, {n_symbols} symbols)\n")

    records, funnel = collect_layer_stats(cfg, ohlc, bench_closes, fwd_days)

    if args.matrix:
        print_entry_matrix(
            records,
            fwd_days,
            sessions,
            n_symbols,
            cfg.mae_days,
            cfg,
            args.matrix_top_n,
        )
        return 0
    print_funnel(funnel, n_symbols, sessions, cfg)

    sym_df = per_symbol_counts(records)
    if not sym_df.empty:
        print("=" * 72)
        print("V2 SIGNALS BY SYMBOL")
        print("=" * 72)
        print(sym_df.to_string(index=False))
        print()

    summarize_signals("V2 STACK (L1+L2+L3) — forward returns", records, fwd_days, sessions, n_symbols, cfg.mae_days)
    summarize_v2_by_ma_up(records, fwd_days, cfg.mae_days)

    if not args.no_compare:
        ma_only = collect_below_ma_up_only(cfg, ohlc, bench_closes, fwd_days)
        summarize_signals(
            "BELOW_MA_UP ONLY (production-style, NOT v2)",
            ma_only,
            fwd_days,
            sessions,
            n_symbols,
            cfg.mae_days,
        )

    print_recent(records)

    if args.csv and records:
        rows = [
            {
                "symbol": r.symbol,
                "date": r.date.strftime("%Y-%m-%d"),
                "benchmark": r.benchmark,
                "entry_price": round(r.entry_price, 4),
                "pullback_pct": round(r.pullback_pct, 3),
                "rel_weak": round(r.rel_weak, 3),
                "stock_ret_20d": round(r.stock_ret_20d, 3),
                "bench_ret_20d": round(r.bench_ret_20d, 3),
                "below_ma_up": r.pass_below_ma_up,
                **{f"ret_{d}d": round(r.fwd[d], 3) for d in fwd_days if d in r.fwd},
                "hit_tp_within": r.hit_tp_within,
                "mae_pct": round(r.mae_pct, 3),
            }
            for r in records
        ]
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"Wrote {len(rows)} v2 signals to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
