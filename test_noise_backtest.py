#!/usr/bin/env python3
"""
Standalone Noise advisor backtest (no Django) — entry v2 + Flux-style exits.

Entry v2 (default):
  - Pullback: close <= (1 - pullback) × N-day high (default 4% / 20d)
  - Relative weakness: rel = stock 20d return − benchmark 20d return in [rel-min, rel-max]
    (default −12% … −4% vs QQQ/SPY per symbol map)
  - Cadence: at most max-per-day symbols per session (most negative rel first)

Exits/adds: same as test_flux_backtest (TP, optional stop, 2% add from avg).

Examples:
  python test_noise_backtest.py --from 2022-01-01 --tp-mult 1.01 --stop 0.06
  python test_noise_backtest.py --matrix-stop --from 2022-01-01
  python test_noise_backtest.py --entry-mode below_ma_up --tp-mult 1.01 --stop 0.06
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# SoulTrader Profile.SPREAD / SENTIMENT (duplicated to avoid Django)
SPREAD = {
    "MEGA": 100,
    "LARGE": 60,
    "MEDIUM": 40,
    "SMALL": 25,
    "MICRO": 15,
    "NANO": 10,
}

SENTIMENT = {
    "STRONG_BULL": 1.2,
    "BULL": 1.1,
    "STAG": 1.0,
    "BEAR": 0.9,
    "STRONG_BEAR": 0.8,
}

# Noise watchlist (not shared with flux.py — duplicate for standalone backtest).
NOISE_UNIVERSE = [
    "AAPL", "ADBE", "AMD", "AMZN", "ARM", "AVGO", "CAT", "COST", "CRM", "CRWD",
    "GOOGL", "JPM", "LLY", "MA", "META", "MSFT", "MU", "NET", "NFLX", "NOW",
    "NVDA", "PLTR", "TSM", "TSLA", "UBER", "UNH", "V",
]

SYMBOL_BENCHMARK = {
    "AAPL": "QQQ", "ADBE": "QQQ", "AMD": "QQQ", "AMZN": "QQQ", "ARM": "QQQ",
    "AVGO": "QQQ", "CRM": "QQQ", "CRWD": "QQQ", "GOOGL": "QQQ", "META": "QQQ",
    "MSFT": "QQQ", "MU": "QQQ", "NET": "QQQ", "NFLX": "QQQ", "NOW": "QQQ",
    "NVDA": "QQQ", "PLTR": "QQQ", "TSLA": "QQQ", "TSM": "QQQ",
}
DEFAULT_BENCHMARK = "SPY"

OhlcField = Literal["open", "high", "low", "close"]
BuyFillMode = Literal["limit", "close"]
ExitFillMode = Literal["limit", "close", "high"]
StopFillMode = Literal["limit", "close", "low"]


@dataclass
class Position:
    shares: int = 0
    avg_cost: float = 0.0
    tranches: int = 0
    cost_basis: float = 0.0
    entry_date: Optional[pd.Timestamp] = None


@dataclass
class ClosedTrade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    tranches: int
    cost_basis: float
    proceeds: float
    return_pct: float
    days_held: int
    exit_reason: str  # tp | stop


@dataclass
class BacktestConfig:
    symbols: List[str]
    start: pd.Timestamp
    end: pd.Timestamp
    investment: float
    spread: str
    sentiment: float
    add_pct: float
    tp_pct: float
    stop_pct: Optional[float]  # None = disabled; exit when probe <= avg * (1 - stop_pct)
    stop_on: OhlcField
    stop_fill: StopFillMode
    entry_mode: str  # v2 | open | pullback | below_ma_up
    entry_pullback: float
    entry_lookback: int
    entry_ma_period: int
    rel_weak_min: float  # e.g. -12 (most negative rel allowed)
    rel_weak_max: float  # e.g. -4 (least negative rel required)
    ret_lookback: int
    max_entries_per_day: int
    min_price: float
    add_on: OhlcField
    exit_on: OhlcField
    entry_on: OhlcField
    buy_fill: BuyFillMode
    exit_fill: ExitFillMode
    no_rebuy: bool  # after a full exit, do not open a new position in that symbol


def tranche_size(cfg: BacktestConfig) -> float:
    n = SPREAD[cfg.spread]
    return (cfg.investment / n) * cfg.sentiment


def download_ohlc(
    symbols: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """Daily adjusted OHLC; each key maps to a DataFrame (dates × symbols)."""
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
            if field not in data.columns.get_level_values(0):
                raise RuntimeError(f"missing {field} in download")
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


def _bar(ohlc: Dict[str, pd.DataFrame], field: OhlcField, dt: pd.Timestamp, sym: str) -> Optional[float]:
    frame = ohlc[field]
    if dt not in frame.index or sym not in frame.columns:
        return None
    val = frame.at[dt, sym]
    if pd.isna(val):
        return None
    return float(val)


def _resolve_buy_fill(threshold: float, bar_close: float, fill: BuyFillMode) -> float:
    if fill == "close":
        return bar_close
    return threshold


def _resolve_exit_fill(
    tp_limit: float,
    bar_close: float,
    bar_high: float,
    fill: ExitFillMode,
) -> float:
    if fill == "close":
        return bar_close
    if fill == "high":
        return bar_high
    return tp_limit


def _resolve_stop_fill(
    stop_limit: float,
    bar_close: float,
    bar_low: float,
    fill: StopFillMode,
) -> float:
    if fill == "close":
        return bar_close
    if fill == "low":
        return bar_low
    return stop_limit


def _buy(
    cash: float,
    pos: Position,
    price: float,
    amount: float,
    dt: pd.Timestamp,
) -> Tuple[float, bool]:
    if price <= 0 or amount <= 0 or cash < price:
        return cash, False
    spend = min(amount, cash)
    shares = int(spend // price)
    if shares <= 0:
        return cash, False
    cost = shares * price
    if pos.shares == 0:
        pos.avg_cost = price
        pos.entry_date = dt
    else:
        pos.avg_cost = (pos.cost_basis + cost) / (pos.shares + shares)
    pos.shares += shares
    pos.cost_basis += cost
    pos.tranches += 1
    return cash - cost, True


def _sell_all(pos: Position, price: float) -> Tuple[float, float, int, Optional[pd.Timestamp]]:
    if pos.shares <= 0:
        return 0.0, 0.0, 0, None
    proceeds = pos.shares * price
    cost = pos.cost_basis
    tr = pos.tranches
    entry = pos.entry_date
    pos.shares = 0
    pos.avg_cost = 0.0
    pos.cost_basis = 0.0
    pos.tranches = 0
    pos.entry_date = None
    return proceeds, cost, tr, entry


def benchmark_for(symbol: str) -> str:
    return SYMBOL_BENCHMARK.get(symbol.upper(), DEFAULT_BENCHMARK)


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


def pass_pullback_v2(
    close_s: pd.Series,
    dt: pd.Timestamp,
    lookback: int,
    pullback_pct: float,
) -> Tuple[bool, float]:
    if dt not in close_s.index:
        return False, 0.0
    window = close_s.loc[:dt].tail(lookback)
    if len(window) < lookback:
        return False, 0.0
    rh = float(window.max())
    close = float(close_s.loc[dt])
    if rh <= 0 or close <= 0:
        return False, close
    return close <= rh * (1.0 - pullback_pct), close


def build_v2_entry_allowlist(
    cfg: BacktestConfig,
    ohlc: Dict[str, pd.DataFrame],
    trade_dates: pd.DatetimeIndex,
) -> Dict[pd.Timestamp, set]:
    closes = ohlc["close"]
    symbols = [s for s in cfg.symbols if s in closes.columns and s not in ("QQQ", "SPY")]
    qqq = closes["QQQ"].dropna() if "QQQ" in closes.columns else pd.Series(dtype=float)
    spy = closes["SPY"].dropna() if "SPY" in closes.columns else pd.Series(dtype=float)
    allow: Dict[pd.Timestamp, set] = {}

    for dt in trade_dates:
        candidates: List[Tuple[float, str]] = []
        for sym in symbols:
            close_s = closes[sym].dropna()
            ok_pb, _px = pass_pullback_v2(close_s, dt, cfg.entry_lookback, cfg.entry_pullback)
            if not ok_pb:
                continue
            bs = benchmark_for(sym)
            bench_s = qqq if bs == "QQQ" else spy
            if bench_s.empty:
                continue
            sr = ret_over_sessions(close_s, dt, cfg.ret_lookback)
            br = ret_over_sessions(bench_s, dt, cfg.ret_lookback)
            if sr is None or br is None:
                continue
            rel = sr - br
            if rel < cfg.rel_weak_min or rel > cfg.rel_weak_max:
                continue
            candidates.append((rel, sym))
        candidates.sort(key=lambda x: x[0])
        allow[dt] = {sym for _, sym in candidates[: cfg.max_entries_per_day]}
    return allow


def _entry_signal(
    cfg: BacktestConfig,
    sym: str,
    dt: pd.Timestamp,
    ohlc: Dict[str, pd.DataFrame],
    rolling_high: pd.DataFrame,
    closes: pd.DataFrame,
    sma: pd.DataFrame,
) -> Tuple[bool, Optional[float]]:
    """Return (should_enter, limit_price for fill)."""
    close_px = _bar(ohlc, "close", dt, sym)
    if close_px is None or close_px < cfg.min_price:
        return False, None

    if cfg.entry_mode == "open":
        return True, close_px

    if cfg.entry_mode == "below_ma_up":
        ma = sma.at[dt, sym] if dt in sma.index else np.nan
        if pd.isna(ma) or close_px >= float(ma):
            return False, None
        if dt not in closes.index:
            return False, None
        idx = closes.index.get_loc(dt)
        if isinstance(idx, slice) or idx < 1:
            return False, None
        prev_close = float(closes.iloc[idx - 1][sym])
        if np.isnan(prev_close) or close_px <= prev_close:
            return False, None
        return True, close_px

    if cfg.entry_mode == "pullback":
        rh = rolling_high.at[dt, sym] if dt in rolling_high.index else np.nan
        if pd.isna(rh) or rh <= 0:
            return False, None
        limit = float(rh) * (1.0 - cfg.entry_pullback)
        probe = _bar(ohlc, cfg.entry_on, dt, sym)
        if probe is None or probe > limit:
            return False, None
        fill_px = _resolve_buy_fill(limit, close_px, cfg.buy_fill)
        return True, fill_px

    raise ValueError(f"unknown entry_mode: {cfg.entry_mode}")


def run_simulation(cfg: BacktestConfig, ohlc: Dict[str, pd.DataFrame]) -> Dict:
    closes = ohlc["close"]
    symbols = [s for s in cfg.symbols if s in closes.columns]
    missing = set(cfg.symbols) - set(symbols)
    if missing:
        print(f"warning: no price data for: {', '.join(sorted(missing))}", file=sys.stderr)

    if not symbols:
        raise ValueError("no symbols with price data")

    trade_dates = closes.index[(closes.index >= cfg.start) & (closes.index <= cfg.end)]
    v2_allow = (
        build_v2_entry_allowlist(cfg, ohlc, trade_dates)
        if cfg.entry_mode == "v2"
        else None
    )

    bench_cols = [c for c in ("QQQ", "SPY") if c in closes.columns]
    keep_cols = list(dict.fromkeys(symbols + bench_cols))
    for key in ohlc:
        ohlc[key] = ohlc[key][keep_cols].copy()

    closes = ohlc["close"][symbols].copy()
    rolling_high = closes.rolling(cfg.entry_lookback, min_periods=cfg.entry_lookback).max()
    sma = closes.rolling(cfg.entry_ma_period, min_periods=cfg.entry_ma_period).mean()
    if len(trade_dates) == 0:
        raise ValueError("no trading dates in range")

    cash = float(cfg.investment)
    positions: Dict[str, Position] = {s: Position() for s in symbols}
    closed: List[ClosedTrade] = []
    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    retired: set[str] = set()

    tranche = tranche_size(cfg)
    tp_mult = 1.0 + cfg.tp_pct
    add_mult = 1.0 - cfg.add_pct

    for dt in trade_dates:
        # 1) Exits — stop-loss first (if enabled), then take-profit
        for sym in symbols:
            pos = positions[sym]
            if pos.shares <= 0:
                continue
            close_px = _bar(ohlc, "close", dt, sym)
            high_px = _bar(ohlc, "high", dt, sym)
            low_px = _bar(ohlc, "low", dt, sym)
            if close_px is None or high_px is None or low_px is None or close_px < cfg.min_price:
                continue

            exit_reason: Optional[str] = None
            fill_px: Optional[float] = None

            if cfg.stop_pct is not None and cfg.stop_pct > 0:
                stop_limit = pos.avg_cost * (1.0 - cfg.stop_pct)
                stop_probe = _bar(ohlc, cfg.stop_on, dt, sym)
                if stop_probe is not None and stop_probe <= stop_limit:
                    fill_px = _resolve_stop_fill(stop_limit, close_px, low_px, cfg.stop_fill)
                    exit_reason = "stop"

            if exit_reason is None:
                tp_limit = pos.avg_cost * tp_mult
                probe = _bar(ohlc, cfg.exit_on, dt, sym)
                if probe is not None and probe >= tp_limit:
                    fill_px = _resolve_exit_fill(tp_limit, close_px, high_px, cfg.exit_fill)
                    exit_reason = "tp"

            if exit_reason is None or fill_px is None:
                continue

            proceeds, cost, tranches, entry_dt = _sell_all(pos, fill_px)
            cash += proceeds
            ret = (proceeds - cost) / cost * 100.0 if cost > 0 else 0.0
            days = (dt - entry_dt).days if entry_dt is not None else 0
            closed.append(
                ClosedTrade(
                    symbol=sym,
                    entry_date=entry_dt or dt,
                    exit_date=dt,
                    tranches=tranches,
                    cost_basis=cost,
                    proceeds=proceeds,
                    return_pct=ret,
                    days_held=days,
                    exit_reason=exit_reason,
                )
            )
            if cfg.no_rebuy:
                retired.add(sym)

        # 2) Adds — default: daily LOW touches add level (limit buy at threshold)
        for sym in symbols:
            pos = positions[sym]
            if pos.shares <= 0:
                continue
            close_px = _bar(ohlc, "close", dt, sym)
            if close_px is None or close_px < cfg.min_price:
                continue
            add_limit = pos.avg_cost * add_mult
            probe = _bar(ohlc, cfg.add_on, dt, sym)
            if probe is None or probe > add_limit:
                continue
            fill_px = _resolve_buy_fill(add_limit, close_px, cfg.buy_fill)
            cash, _ = _buy(cash, pos, fill_px, tranche, dt)

        # 3) Initial entries when flat
        for sym in symbols:
            pos = positions[sym]
            if pos.shares > 0:
                continue
            if cfg.no_rebuy and sym in retired:
                continue
            if cfg.entry_mode == "v2":
                if v2_allow is None or sym not in v2_allow.get(dt, set()):
                    continue
                close_px = _bar(ohlc, "close", dt, sym)
                if close_px is None or close_px < cfg.min_price:
                    continue
                fill_px = close_px
            else:
                ok, fill_px = _entry_signal(cfg, sym, dt, ohlc, rolling_high, closes, sma)
                if not ok or fill_px is None:
                    continue
            cash, _ = _buy(cash, pos, fill_px, tranche, dt)

        # Mark-to-market at close
        mtm = cash
        for sym in symbols:
            pos = positions[sym]
            if pos.shares <= 0:
                continue
            px = _bar(ohlc, "close", dt, sym)
            if px is not None:
                mtm += pos.shares * px
        equity_curve.append((dt, mtm))

    open_positions = []
    last_dt = trade_dates[-1]
    for sym in symbols:
        pos = positions[sym]
        if pos.shares <= 0:
            continue
        px = _bar(ohlc, "close", last_dt, sym)
        if px is None:
            continue
        mtm = pos.shares * px
        unreal = (mtm - pos.cost_basis) / pos.cost_basis * 100.0 if pos.cost_basis else 0.0
        open_positions.append(
            {
                "symbol": sym,
                "tranches": pos.tranches,
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "cost_basis": pos.cost_basis,
                "last_price": px,
                "mtm": mtm,
                "unrealized_pct": unreal,
                "entry_date": pos.entry_date,
            }
        )

    return {
        "closed": closed,
        "equity_curve": equity_curve,
        "open_positions": open_positions,
        "final_cash": cash,
        "tranche": tranche,
        "symbols": symbols,
        "trade_dates": trade_dates,
        "retired": retired,
    }


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min() * 100.0)


def _cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    start_v = float(equity.iloc[0])
    end_v = float(equity.iloc[-1])
    if start_v <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (pow(end_v / start_v, 1.0 / years) - 1.0) * 100.0


def _sharpe(daily_returns: pd.Series) -> float:
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))


def print_summary(cfg: BacktestConfig, result: Dict) -> None:
    closed: List[ClosedTrade] = result["closed"]
    tranche = result["tranche"]
    eq = pd.Series({d: v for d, v in result["equity_curve"]}, dtype=float).sort_index()

    print("=" * 72)
    print("NOISE AVERAGING-DOWN BACKTEST")
    print("=" * 72)
    print(f"Period:        {cfg.start.date()} → {cfg.end.date()} ({len(result['trade_dates'])} sessions)")
    print(f"Universe:      {len(result['symbols'])} symbols")
    print(f"Investment:    ${cfg.investment:,.2f}")
    print(f"Spread:        {cfg.spread} ({SPREAD[cfg.spread]} names) → tranche ${tranche:,.2f}")
    print(f"Sentiment:     {cfg.sentiment:.2f}x")
    print(f"Add trigger:   {cfg.add_on.upper()} <= avg×{1 - cfg.add_pct:.4f}")
    print(f"Exit trigger:  {cfg.exit_on.upper()} >= avg×{1 + cfg.tp_pct:.4f}")
    if cfg.stop_pct is not None and cfg.stop_pct > 0:
        print(f"Stop trigger:  {cfg.stop_on.upper()} <= avg×{1 - cfg.stop_pct:.4f}")
        print(f"Stop fill:     {cfg.stop_fill}")
    else:
        print("Stop loss:     off")
    print(f"Buy fill:      {cfg.buy_fill}")
    print(f"Exit fill:     {cfg.exit_fill} (high = sell at session high when TP touched)")
    print(f"Entry mode:    {cfg.entry_mode}", end="")
    if cfg.entry_mode == "v2":
        print(
            f" (pb>={cfg.entry_pullback * 100:.1f}% off {cfg.entry_lookback}d high, "
            f"rel [{cfg.rel_weak_min:.0f},{cfg.rel_weak_max:.0f}]%, "
            f"top {cfg.max_entries_per_day}/day)"
        )
    elif cfg.entry_mode == "pullback":
        print(f" (probe {cfg.entry_on.upper()}, {cfg.entry_pullback * 100:.1f}% below {cfg.entry_lookback}d high)")
    elif cfg.entry_mode == "below_ma_up":
        print(f" (close < SMA{cfg.entry_ma_period} and close > prior close)")
    else:
        print(f" (fill at close)")
    print(f"Rebuy:         {'off (one round-trip per symbol)' if cfg.no_rebuy else 'on (re-enter when flat)'}")
    if cfg.no_rebuy and result.get("retired"):
        print(f"Retired:       {len(result['retired'])} symbols completed a round-trip")
    print()

    if len(eq):
        total_ret = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0
        daily_ret = eq.pct_change().dropna()
        print("=== Portfolio ===")
        print(f"Final equity:     ${eq.iloc[-1]:,.2f}")
        print(f"Total return:     {total_ret:+.2f}%")
        print(f"CAGR:             {_cagr(eq):+.2f}%")
        print(f"Max drawdown:     {_max_drawdown(eq):.2f}%")
        print(f"Sharpe (daily):   {_sharpe(daily_ret):.2f}")
        print(f"Cash (unsettled): ${result['final_cash']:,.2f}")
        print()

    if closed:
        rets = [t.return_pct for t in closed]
        tranches = [t.tranches for t in closed]
        days = [t.days_held for t in closed]
        winners = sum(1 for r in rets if r > 0)
        print("=== Closed trades ===")
        print(f"Count:            {len(closed)}")
        print(f"Win rate:         {winners / len(closed) * 100:.1f}%")
        print(f"Avg return:       {np.mean(rets):+.2f}%")
        print(f"Median return:    {np.median(rets):+.2f}%")
        print(f"Avg tranches:     {np.mean(tranches):.2f}")
        print(f"Avg days held:    {np.mean(days):.1f}")
        print(f"Best / worst:     {max(rets):+.2f}% / {min(rets):+.2f}%")
        n_stop = sum(1 for t in closed if t.exit_reason == "stop")
        n_tp = sum(1 for t in closed if t.exit_reason == "tp")
        if n_stop or n_tp:
            print(f"Exits:            {n_tp} take-profit, {n_stop} stop-loss")
        print()
        if cfg.no_rebuy:
            _print_exit_dates_by_symbol(closed)
    else:
        print("=== Closed trades ===")
        print("No completed round-trips in range.")
        print()

    open_pos = result["open_positions"]
    if open_pos:
        print(f"=== Open at end ({len(open_pos)}) ===")
        for p in sorted(open_pos, key=lambda x: x["unrealized_pct"]):
            entry = p["entry_date"].strftime("%Y-%m-%d") if p["entry_date"] is not None else "?"
            print(
                f"  {p['symbol']:<6} tr={p['tranches']} avg=${p['avg_cost']:.2f} "
                f"px=${p['last_price']:.2f} unreal={p['unrealized_pct']:+.1f}% since {entry}"
            )
        print()


def _print_exit_dates_by_symbol(closed: List[ClosedTrade]) -> None:
    """One line per closed round-trip: symbol, exit date, tranches, return %."""
    if not closed:
        return
    rows = sorted(closed, key=lambda t: (t.symbol, t.exit_date))
    print("=== Sell dates by symbol ===")
    print(f"{'Symbol':<8} {'Sell date':<12} {'Entry':<12} {'Tr':>3} {'Return':>8} {'Days':>5}")
    print("-" * 52)
    for t in rows:
        print(
            f"{t.symbol:<8} {t.exit_date.strftime('%Y-%m-%d'):<12} "
            f"{t.entry_date.strftime('%Y-%m-%d'):<12} {t.tranches:>3} "
            f"{t.return_pct:>+7.2f}% {t.days_held:>5}"
        )
    print()


def trades_to_dataframe(closed: List[ClosedTrade]) -> pd.DataFrame:
    if not closed:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "symbol": t.symbol,
                "entry_date": t.entry_date.strftime("%Y-%m-%d"),
                "exit_date": t.exit_date.strftime("%Y-%m-%d"),
                "tranches": t.tranches,
                "cost_basis": round(t.cost_basis, 2),
                "proceeds": round(t.proceeds, 2),
                "return_pct": round(t.return_pct, 3),
                "days_held": t.days_held,
                "exit_reason": t.exit_reason,
            }
            for t in closed
        ]
    )


def print_stop_matrix(
    cfg_base: BacktestConfig,
    ohlc: Dict[str, pd.DataFrame],
    stops: Sequence[Optional[float]],
) -> None:
    print("=" * 72)
    print("STOP-LOSS MATRIX (same entry, varying --stop)")
    print("=" * 72)
    print(f"{'stop':<8} {'return':>10} {'mdd':>10} {'sharpe':>8} {'trades':>7} {'tp':>5} {'sl':>5}")
    print("-" * 56)
    for stop in stops:
        cfg = BacktestConfig(
            **{**cfg_base.__dict__, "stop_pct": stop}
        )
        result = run_simulation(cfg, ohlc)
        eq = pd.Series({d: v for d, v in result["equity_curve"]}, dtype=float).sort_index()
        closed = result["closed"]
        if len(eq) < 2:
            continue
        total_ret = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0
        mdd = _max_drawdown(eq)
        sharpe = _sharpe(eq.pct_change().dropna())
        n_tp = sum(1 for t in closed if t.exit_reason == "tp")
        n_sl = sum(1 for t in closed if t.exit_reason == "stop")
        label = "off" if stop is None else f"{stop * 100:.0f}%"
        print(
            f"{label:<8} {total_ret:>+9.2f}% {mdd:>9.2f}% {sharpe:>8.2f} "
            f"{len(closed):>7} {n_tp:>5} {n_sl:>5}"
        )
    print()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Noise entry-v2 averaging-down backtest.")
    p.add_argument("--from", dest="date_from", default="2022-01-01", metavar="DATE")
    p.add_argument("--to", dest="date_to", default=None, metavar="DATE")
    p.add_argument("--symbols", default=",".join(NOISE_UNIVERSE), help="Comma-separated tickers")
    p.add_argument("--investment", type=float, default=100_000.0)
    p.add_argument("--spread", choices=list(SPREAD.keys()), default="MEDIUM")
    p.add_argument(
        "--sentiment",
        default="1.0",
        help="Multiplier (e.g. 1.1) or preset: STRONG_BULL, BULL, STAG, BEAR, STRONG_BEAR",
    )
    p.add_argument("--add-pct", type=float, default=0.02, help="Add when probe <= avg * (1 - pct)")
    p.add_argument(
        "--tp-pct",
        "--tp",
        type=float,
        default=0.02,
        dest="tp_pct",
        metavar="PCT",
        help="Take-profit: exit when probe >= avg * (1 + PCT); default 0.02 (+2%%)",
    )
    p.add_argument(
        "--stop-pct",
        "--stop",
        type=float,
        default=None,
        dest="stop_pct",
        metavar="PCT",
        help="Stop-loss: exit when probe <= avg * (1 - PCT); omit to disable",
    )
    p.add_argument(
        "--stop-on",
        choices=("low", "close"),
        default="low",
        help="Bar field that must touch stop level (default: low)",
    )
    p.add_argument(
        "--stop-fill",
        choices=("limit", "close", "low"),
        default="limit",
        help="Stop fill: limit=stop price, close=close, low=session low (default: limit)",
    )
    p.add_argument(
        "--add-on",
        choices=("low", "close"),
        default="low",
        help="Bar field that must touch add level (default: low)",
    )
    p.add_argument(
        "--exit-on",
        choices=("high", "close"),
        default="high",
        help="Bar field that must touch take-profit (default: high)",
    )
    p.add_argument(
        "--entry-on",
        choices=("low", "close"),
        default="low",
        help="Pullback entry probe field (default: low)",
    )
    p.add_argument(
        "--buy-fill",
        choices=("limit", "close"),
        default="limit",
        help="Add/entry fill: limit at threshold or close (default: limit)",
    )
    p.add_argument(
        "--exit-fill",
        choices=("limit", "close", "high"),
        default="high",
        help="Exit fill: high=session high when TP touched (default), limit=avg×(1+tp), close",
    )
    p.add_argument(
        "--fill",
        choices=("limit", "close", "high"),
        default=None,
        help="Set both buy-fill and exit-fill (overrides defaults if used alone)",
    )
    p.add_argument(
        "--entry-mode",
        choices=("v2", "open", "pullback", "below_ma_up"),
        default="v2",
        help="v2 (default) | open | pullback | below_ma_up",
    )
    p.add_argument("--entry-pullback", type=float, default=0.04, help="Min %% below N-day high (v2/pullback)")
    p.add_argument("--entry-lookback", type=int, default=20, help="High lookback for v2/pullback")
    p.add_argument("--entry-ma", type=int, default=20, help="SMA period for below_ma_up (default 20)")
    p.add_argument("--rel-min", type=float, default=-12.0, help="v2: min rel weak (%% points, e.g. -12)")
    p.add_argument("--rel-max", type=float, default=-4.0, help="v2: max rel weak (%% points, e.g. -4)")
    p.add_argument("--ret-lookback", type=int, default=20, help="v2: sessions for stock/bench return")
    p.add_argument("--max-per-day", type=int, default=5, help="v2: max new entries per session")
    p.add_argument(
        "--matrix-stop",
        action="store_true",
        help="Sweep stop-loss (off, 4%%, 6%%, 8%%, 10%%, 12%%) then exit",
    )
    p.add_argument("--tp-mult", type=float, default=1.01, help="Default TP mult 1.01 (Flux)")
    p.add_argument("--min-price", type=float, default=1.0)
    p.add_argument(
        "--no-rebuy",
        action="store_true",
        help="After a full exit, never open that symbol again (one round-trip per name)",
    )
    p.add_argument("--csv", default="", help="Write closed trades to CSV path")
    return p.parse_args(argv)


def resolve_sentiment(value: str) -> float:
    key = value.strip().upper()
    if key in SENTIMENT:
        return float(SENTIMENT[key])
    return float(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    end = pd.Timestamp(args.date_to) if args.date_to else pd.Timestamp.now().normalize()
    start = pd.Timestamp(args.date_from)
    if start >= end:
        print("error: --from must be before --to", file=sys.stderr)
        return 1

    symbols = [s.strip().upper().replace(".", "-") for s in args.symbols.split(",") if s.strip()]

    try:
        sentiment = resolve_sentiment(args.sentiment)
    except ValueError:
        print(f"error: invalid --sentiment {args.sentiment!r}", file=sys.stderr)
        return 1

    if args.tp_mult <= 0:
        print("error: --tp-mult must be > 0", file=sys.stderr)
        return 1
    tp_pct = float(args.tp_mult) - 1.0
    if tp_pct <= 0:
        print("error: take-profit must be > 0 (use --tp or --tp-mult > 1)", file=sys.stderr)
        return 1

    stop_pct = args.stop_pct
    if stop_pct is not None and stop_pct <= 0:
        print("error: --stop must be > 0 when set", file=sys.stderr)
        return 1

    cfg = BacktestConfig(
        symbols=symbols,
        start=start,
        end=end,
        investment=args.investment,
        spread=args.spread,
        sentiment=sentiment,
        add_pct=args.add_pct,
        tp_pct=tp_pct,
        stop_pct=stop_pct,
        stop_on=args.stop_on,
        stop_fill=args.stop_fill,
        entry_mode=args.entry_mode,
        entry_pullback=args.entry_pullback,
        entry_lookback=args.entry_lookback,
        entry_ma_period=args.entry_ma,
        rel_weak_min=args.rel_min,
        rel_weak_max=args.rel_max,
        ret_lookback=args.ret_lookback,
        max_entries_per_day=args.max_per_day,
        min_price=args.min_price,
        add_on=args.add_on,
        exit_on=args.exit_on,
        entry_on=args.entry_on,
        buy_fill=args.buy_fill if args.fill is None else (
            "limit" if args.fill == "high" else args.fill
        ),
        exit_fill=args.exit_fill if args.fill is None else args.fill,
        no_rebuy=args.no_rebuy,
    )

    download_syms = list(dict.fromkeys(symbols + ["QQQ", "SPY"]))
    print(f"Downloading OHLC for {len(download_syms)} symbols (universe + QQQ + SPY)…")
    try:
        ohlc = download_ohlc(download_syms, start, end)
    except Exception as exc:
        print(f"error: download failed: {exc}", file=sys.stderr)
        return 1

    if args.matrix_stop:
        print_stop_matrix(cfg, ohlc, [None, 0.04, 0.06, 0.08, 0.10, 0.12])
        return 0

    try:
        result = run_simulation(cfg, ohlc)
    except Exception as exc:
        print(f"error: simulation failed: {exc}", file=sys.stderr)
        return 1

    print_summary(cfg, result)

    if args.csv:
        df = trades_to_dataframe(result["closed"])
        df.to_csv(args.csv, index=False)
        print(f"Wrote {len(df)} closed trades to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
