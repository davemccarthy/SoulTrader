#!/usr/bin/env python3
"""
Standalone intraday take-profit sampler.

Goal:
  - Mimic remote-style smartanalyse cadence by checking prices every N minutes.
  - Start with fixed take-profit exits, then use the same scaffolding for richer
    profit-harvesting policies later.

Examples:
  source ~/Development/scratch/python/tutorial-env/bin/activate
  python test_intraday_tp.py --date 2026-06-25 --tickers CIEN GLW KLAC --entry-time 12:00
  python test_intraday_tp.py --date 2026-06-25 --tickers CIEN,GLW,KLAC --entry-time 12:00 --tp-pct 0.002
  python test_intraday_tp.py --date 2026-06-25 --tickers CIEN GLW --entry-times CIEN=13:00 GLW=13:00 --entry-prices CIEN=482.73 GLW=225.81
  python test_intraday_tp.py --date 2026-06-25 --tickers CIEN GLW KLAC --entry-time 13:00 --exit-type IPC --activate-pct 0.005 --giveback-pct 0.002
  python test_intraday_tp.py --date 2026-06-25 --tickers CIEN GLW KLAC --entry-time 13:00 --exit-type PEAK_AGE --max-peak-age-minutes 30
  python test_intraday_tp.py --date 2026-06-25 --tickers CIEN GLW KLAC --entry-time 13:00 --exit-type VELOCITY
  python test_intraday_tp.py --date 2026-05-20 --tickers CIEN GLW --entry-time 12:00 --bar-interval 15m

Times are US/Eastern.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


ET = ZoneInfo("US/Eastern")


@dataclass(frozen=True)
class SimulationResult:
    symbol: str
    entry_time: Optional[str]
    entry_price: Optional[float]
    target_price: Optional[float]
    exit_time: Optional[str]
    exit_price: Optional[float]
    exit_reason: str
    return_pct: Optional[float]
    mfe_pct: Optional[float]
    mae_pct: Optional[float]
    missed_tp: bool
    first_high_tp_time: Optional[str]
    checks: int


@dataclass(frozen=True)
class EntryContext:
    entry_bar: pd.Timestamp
    entry_price: float
    target: float
    mfe_pct: Optional[float]
    mae_pct: Optional[float]
    first_high_tp_time: Optional[pd.Timestamp]
    checks: list[pd.Timestamp]


def _parse_time(value: str) -> time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("time must be HH:MM, e.g. 12:00") from exc


def _parse_tickers(values: list[str]) -> list[str]:
    tickers: list[str] = []
    for value in values:
        for part in value.split(","):
            symbol = part.strip().upper()
            if symbol:
                tickers.append(symbol)
    return list(dict.fromkeys(tickers))


def _parse_time_map(values: Optional[list[str]]) -> dict[str, time]:
    if not values:
        return {}
    out: dict[str, time] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("--entry-times values must look like SYMBOL=HH:MM")
        symbol, raw_time = value.split("=", 1)
        out[symbol.strip().upper()] = _parse_time(raw_time.strip())
    return out


def _parse_price_map(values: Optional[list[str]]) -> dict[str, float]:
    if not values:
        return {}
    out: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("--entry-prices values must look like SYMBOL=123.45")
        symbol, raw_price = value.split("=", 1)
        price = _safe_float(raw_price.strip())
        if price is None:
            raise argparse.ArgumentTypeError(f"invalid entry price: {value}")
        out[symbol.strip().upper()] = price
    return out


def _safe_float(value) -> Optional[float]:
    try:
        out = float(value)
        if out <= 0:
            return None
        return out
    except (TypeError, ValueError):
        return None


def _as_et_timestamp(scan_date: date, wall_time: time) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(scan_date, wall_time, tzinfo=ET))


def _download_intraday(symbols: list[str], scan_date: date, interval: str) -> pd.DataFrame:
    end = scan_date + timedelta(days=1)
    return yf.download(
        symbols,
        start=scan_date.isoformat(),
        end=end.isoformat(),
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )


def _symbol_hist(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        level1 = data.columns.get_level_values(1)
        if symbol in level0:
            return data[symbol].dropna(how="all")
        if symbol in level1:
            return data.xs(symbol, axis=1, level=1).dropna(how="all")
        return pd.DataFrame()
    return data.dropna(how="all")


def _hist_for_date(hist: pd.DataFrame, scan_date: date) -> pd.DataFrame:
    if hist.empty:
        return hist
    idx = pd.to_datetime(hist.index)
    if idx.tz is None:
        idx = idx.tz_localize(ET)
    else:
        idx = idx.tz_convert(ET)
    out = hist.copy()
    out.index = idx
    return out[out.index.date == scan_date]


def _first_bar_at_or_after(hist: pd.DataFrame, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    if hist.empty:
        return None
    eligible = hist.index >= ts
    if not eligible.any():
        return None
    return hist.index[eligible][0]


def _price_at_or_before(hist: pd.DataFrame, ts: pd.Timestamp) -> Optional[float]:
    if hist.empty or "Close" not in hist.columns:
        return None
    eligible = hist.index <= ts
    if not eligible.any():
        return None
    return _safe_float(hist.loc[eligible, "Close"].iloc[-1])


def _check_times(
    scan_date: date,
    *,
    start_ts: pd.Timestamp,
    end_time: time,
    cadence_minutes: int,
    offset_minutes: int,
) -> list[pd.Timestamp]:
    # Remote-style cadence: :01, :16, :31, :46 by default.
    session_end = _as_et_timestamp(scan_date, end_time)
    current = _as_et_timestamp(scan_date, time(start_ts.hour, 0)) + pd.Timedelta(minutes=offset_minutes)
    while current < start_ts:
        current += pd.Timedelta(minutes=cadence_minutes)

    out: list[pd.Timestamp] = []
    while current <= session_end:
        out.append(current)
        current += pd.Timedelta(minutes=cadence_minutes)
    return out


def _first_high_tp_time(hist: pd.DataFrame, start_ts: pd.Timestamp, target: float) -> Optional[pd.Timestamp]:
    if hist.empty or "High" not in hist.columns:
        return None
    sub = hist[hist.index >= start_ts]
    if sub.empty:
        return None
    hits = sub["High"].astype(float) >= target
    if not hits.any():
        return None
    return sub.index[hits][0]


def _slope_pct_per_hour(hist: pd.DataFrame, end_ts: pd.Timestamp, lookback_minutes: int) -> Optional[float]:
    if hist.empty or "Close" not in hist.columns:
        return None
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)
    window = hist[(hist.index >= start_ts) & (hist.index <= end_ts)]
    if len(window) < 2:
        return None

    closes = window["Close"].astype(float).tolist()
    first = closes[0]
    if first <= 0:
        return None
    # Use endpoint slope first: simple, robust, and easy to reason about.
    elapsed_hours = max((window.index[-1] - window.index[0]).total_seconds() / 3600.0, 1 / 60)
    return ((closes[-1] / first) - 1.0) * 100.0 / elapsed_hours


def _high_water_through(hist: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> tuple[float, pd.Timestamp] | None:
    if hist.empty or "High" not in hist.columns:
        return None
    window = hist[(hist.index >= start_ts) & (hist.index <= end_ts)]
    if window.empty:
        return None
    highs = window["High"].astype(float)
    high_ts = highs.idxmax()
    high = _safe_float(highs.loc[high_ts])
    if high is None:
        return None
    return high, high_ts


def _sell_result(
    *,
    symbol: str,
    ctx: EntryContext,
    check_ts: pd.Timestamp,
    price: float,
    reason: str,
    target_price: float,
) -> SimulationResult:
    return SimulationResult(
        symbol=symbol,
        entry_time=ctx.entry_bar.strftime("%H:%M"),
        entry_price=ctx.entry_price,
        target_price=target_price,
        exit_time=check_ts.strftime("%H:%M"),
        exit_price=price,
        exit_reason=reason,
        return_pct=(price / ctx.entry_price - 1.0) * 100.0,
        mfe_pct=ctx.mfe_pct,
        mae_pct=ctx.mae_pct,
        missed_tp=False,
        first_high_tp_time=None if ctx.first_high_tp_time is None else ctx.first_high_tp_time.strftime("%H:%M"),
        checks=len(ctx.checks),
    )


def _eod_result(
    *,
    symbol: str,
    ctx: EntryContext,
    last_price: float,
    target_price: float,
) -> SimulationResult:
    return SimulationResult(
        symbol=symbol,
        entry_time=ctx.entry_bar.strftime("%H:%M"),
        entry_price=ctx.entry_price,
        target_price=target_price,
        exit_time=ctx.checks[-1].strftime("%H:%M") if ctx.checks else ctx.entry_bar.strftime("%H:%M"),
        exit_price=last_price,
        exit_reason="EOD",
        return_pct=(last_price / ctx.entry_price - 1.0) * 100.0,
        mfe_pct=ctx.mfe_pct,
        mae_pct=ctx.mae_pct,
        missed_tp=ctx.first_high_tp_time is not None,
        first_high_tp_time=None if ctx.first_high_tp_time is None else ctx.first_high_tp_time.strftime("%H:%M"),
        checks=len(ctx.checks),
    )


def _build_entry_context(
    symbol: str,
    hist: pd.DataFrame,
    *,
    scan_date: date,
    entry_time: time,
    end_time: time,
    cadence_minutes: int,
    offset_minutes: int,
    tp_pct: float,
    entry_price_override: Optional[float],
) -> EntryContext | SimulationResult:
    entry_ts = _as_et_timestamp(scan_date, entry_time)
    entry_bar = _first_bar_at_or_after(hist, entry_ts)
    if entry_bar is None:
        return SimulationResult(symbol, None, None, None, None, None, "NO_ENTRY", None, None, None, False, None, 0)

    entry_price = entry_price_override or _safe_float(hist.loc[entry_bar, "Close"])
    if entry_price is None:
        return SimulationResult(symbol, None, None, None, None, None, "NO_ENTRY_PRICE", None, None, None, False, None, 0)

    target = entry_price * (1.0 + tp_pct)
    after_entry = hist[hist.index >= entry_bar]
    hi = _safe_float(after_entry["High"].max()) if "High" in after_entry.columns and not after_entry.empty else None
    lo = _safe_float(after_entry["Low"].min()) if "Low" in after_entry.columns and not after_entry.empty else None
    mfe = None if hi is None else (hi / entry_price - 1.0) * 100.0
    mae = None if lo is None else (lo / entry_price - 1.0) * 100.0
    first_high_hit = _first_high_tp_time(hist, entry_bar, target)

    checks = _check_times(
        scan_date,
        start_ts=entry_bar,
        end_time=end_time,
        cadence_minutes=cadence_minutes,
        offset_minutes=offset_minutes,
    )
    return EntryContext(
        entry_bar=entry_bar,
        entry_price=entry_price,
        target=target,
        mfe_pct=mfe,
        mae_pct=mae,
        first_high_tp_time=first_high_hit,
        checks=checks,
    )


def simulate_fixed_tp(
    symbol: str,
    hist: pd.DataFrame,
    *,
    scan_date: date,
    entry_time: time,
    end_time: time,
    cadence_minutes: int,
    offset_minutes: int,
    tp_pct: float,
    entry_price_override: Optional[float],
) -> SimulationResult:
    ctx = _build_entry_context(
        symbol,
        hist,
        scan_date=scan_date,
        entry_time=entry_time,
        end_time=end_time,
        cadence_minutes=cadence_minutes,
        offset_minutes=offset_minutes,
        tp_pct=tp_pct,
        entry_price_override=entry_price_override,
    )
    if isinstance(ctx, SimulationResult):
        return ctx

    last_price = ctx.entry_price
    for check_ts in ctx.checks:
        price = _price_at_or_before(hist, check_ts)
        if price is None:
            continue
        last_price = price
        if price >= ctx.target:
            return SimulationResult(
                symbol=symbol,
                entry_time=ctx.entry_bar.strftime("%H:%M"),
                entry_price=ctx.entry_price,
                target_price=ctx.target,
                exit_time=check_ts.strftime("%H:%M"),
                exit_price=price,
                exit_reason="TP",
                return_pct=(price / ctx.entry_price - 1.0) * 100.0,
                mfe_pct=ctx.mfe_pct,
                mae_pct=ctx.mae_pct,
                missed_tp=False,
                first_high_tp_time=None if ctx.first_high_tp_time is None else ctx.first_high_tp_time.strftime("%H:%M"),
                checks=len(ctx.checks),
            )

    missed_tp = ctx.first_high_tp_time is not None
    return SimulationResult(
        symbol=symbol,
        entry_time=ctx.entry_bar.strftime("%H:%M"),
        entry_price=ctx.entry_price,
        target_price=ctx.target,
        exit_time=ctx.checks[-1].strftime("%H:%M") if ctx.checks else ctx.entry_bar.strftime("%H:%M"),
        exit_price=last_price,
        exit_reason="EOD",
        return_pct=(last_price / ctx.entry_price - 1.0) * 100.0,
        mfe_pct=ctx.mfe_pct,
        mae_pct=ctx.mae_pct,
        missed_tp=missed_tp,
        first_high_tp_time=None if ctx.first_high_tp_time is None else ctx.first_high_tp_time.strftime("%H:%M"),
        checks=len(ctx.checks),
    )


def simulate_ipc(
    symbol: str,
    hist: pd.DataFrame,
    *,
    scan_date: date,
    entry_time: time,
    end_time: time,
    cadence_minutes: int,
    offset_minutes: int,
    tp_pct: float,
    entry_price_override: Optional[float],
    activate_pct: float,
    giveback_pct: float,
) -> SimulationResult:
    ctx = _build_entry_context(
        symbol,
        hist,
        scan_date=scan_date,
        entry_time=entry_time,
        end_time=end_time,
        cadence_minutes=cadence_minutes,
        offset_minutes=offset_minutes,
        tp_pct=tp_pct,
        entry_price_override=entry_price_override,
    )
    if isinstance(ctx, SimulationResult):
        return ctx

    activate_price = ctx.entry_price * (1.0 + activate_pct)
    high_water = ctx.entry_price
    last_price = ctx.entry_price
    activated = False

    for check_ts in ctx.checks:
        high_state = _high_water_through(hist, ctx.entry_bar, check_ts)
        if high_state is not None:
            high_water = max(high_water, high_state[0])
        if high_water >= activate_price:
            activated = True

        price = _price_at_or_before(hist, check_ts)
        if price is None:
            continue
        last_price = price

        # IPC never sells red/flat; it protects profit after a meaningful peak.
        if price <= ctx.entry_price:
            continue
        if activated and price <= high_water * (1.0 - giveback_pct):
            return _sell_result(
                symbol=symbol,
                ctx=ctx,
                check_ts=check_ts,
                price=price,
                reason="IPC_GIVEBACK",
                target_price=activate_price,
            )

    return _eod_result(
        symbol=symbol,
        ctx=ctx,
        last_price=last_price,
        target_price=activate_price,
    )


def simulate_wave(
    symbol: str,
    hist: pd.DataFrame,
    *,
    scan_date: date,
    entry_time: time,
    end_time: time,
    cadence_minutes: int,
    offset_minutes: int,
    tp_pct: float,
    entry_price_override: Optional[float],
    activate_pct: float,
    giveback_pct: float,
    slope_lookback_minutes: int,
    min_slope_pct_per_hour: float,
) -> SimulationResult:
    ctx = _build_entry_context(
        symbol,
        hist,
        scan_date=scan_date,
        entry_time=entry_time,
        end_time=end_time,
        cadence_minutes=cadence_minutes,
        offset_minutes=offset_minutes,
        tp_pct=tp_pct,
        entry_price_override=entry_price_override,
    )
    if isinstance(ctx, SimulationResult):
        return ctx

    activate_price = ctx.entry_price * (1.0 + activate_pct)
    high_water = ctx.entry_price
    last_price = ctx.entry_price
    activated = False

    for check_ts in ctx.checks:
        high_state = _high_water_through(hist, ctx.entry_bar, check_ts)
        if high_state is not None:
            high_water = max(high_water, high_state[0])
        if high_water >= activate_price:
            activated = True

        price = _price_at_or_before(hist, check_ts)
        if price is None:
            continue
        last_price = price
        if price <= ctx.entry_price:
            continue
        if not activated:
            continue

        slope = _slope_pct_per_hour(hist, check_ts, slope_lookback_minutes)
        if slope is not None and slope <= min_slope_pct_per_hour:
            return _sell_result(
                symbol=symbol,
                ctx=ctx,
                check_ts=check_ts,
                price=price,
                reason="WAVE_SLOPE",
                target_price=activate_price,
            )
        if price <= high_water * (1.0 - giveback_pct):
            return _sell_result(
                symbol=symbol,
                ctx=ctx,
                check_ts=check_ts,
                price=price,
                reason="WAVE_GIVEBACK",
                target_price=activate_price,
            )

    return _eod_result(
        symbol=symbol,
        ctx=ctx,
        last_price=last_price,
        target_price=activate_price,
    )


def simulate_peak_age(
    symbol: str,
    hist: pd.DataFrame,
    *,
    scan_date: date,
    entry_time: time,
    end_time: time,
    cadence_minutes: int,
    offset_minutes: int,
    tp_pct: float,
    entry_price_override: Optional[float],
    activate_pct: float,
    giveback_pct: float,
    max_peak_age_minutes: int,
) -> SimulationResult:
    ctx = _build_entry_context(
        symbol,
        hist,
        scan_date=scan_date,
        entry_time=entry_time,
        end_time=end_time,
        cadence_minutes=cadence_minutes,
        offset_minutes=offset_minutes,
        tp_pct=tp_pct,
        entry_price_override=entry_price_override,
    )
    if isinstance(ctx, SimulationResult):
        return ctx

    activate_price = ctx.entry_price * (1.0 + activate_pct)
    high_water = ctx.entry_price
    high_water_ts = ctx.entry_bar
    last_price = ctx.entry_price
    activated = False

    for check_ts in ctx.checks:
        high_state = _high_water_through(hist, ctx.entry_bar, check_ts)
        if high_state is not None and high_state[0] > high_water:
            high_water, high_water_ts = high_state
        if high_water >= activate_price:
            activated = True

        price = _price_at_or_before(hist, check_ts)
        if price is None:
            continue
        last_price = price
        if price <= ctx.entry_price or not activated:
            continue

        peak_age = (check_ts - high_water_ts).total_seconds() / 60.0
        if peak_age >= max_peak_age_minutes:
            return _sell_result(
                symbol=symbol,
                ctx=ctx,
                check_ts=check_ts,
                price=price,
                reason="PEAK_AGE",
                target_price=activate_price,
            )
        if price <= high_water * (1.0 - giveback_pct):
            return _sell_result(
                symbol=symbol,
                ctx=ctx,
                check_ts=check_ts,
                price=price,
                reason="PEAK_GIVEBACK",
                target_price=activate_price,
            )

    return _eod_result(symbol=symbol, ctx=ctx, last_price=last_price, target_price=activate_price)


def simulate_velocity(
    symbol: str,
    hist: pd.DataFrame,
    *,
    scan_date: date,
    entry_time: time,
    end_time: time,
    cadence_minutes: int,
    offset_minutes: int,
    tp_pct: float,
    entry_price_override: Optional[float],
    activate_pct: float,
    giveback_pct: float,
    min_velocity_pct: float,
) -> SimulationResult:
    ctx = _build_entry_context(
        symbol,
        hist,
        scan_date=scan_date,
        entry_time=entry_time,
        end_time=end_time,
        cadence_minutes=cadence_minutes,
        offset_minutes=offset_minutes,
        tp_pct=tp_pct,
        entry_price_override=entry_price_override,
    )
    if isinstance(ctx, SimulationResult):
        return ctx

    activate_price = ctx.entry_price * (1.0 + activate_pct)
    high_water = ctx.entry_price
    last_price = ctx.entry_price
    previous_profit_pct: Optional[float] = None
    activated = False

    for check_ts in ctx.checks:
        high_state = _high_water_through(hist, ctx.entry_bar, check_ts)
        if high_state is not None:
            high_water = max(high_water, high_state[0])
        if high_water >= activate_price:
            activated = True

        price = _price_at_or_before(hist, check_ts)
        if price is None:
            continue
        last_price = price
        profit_pct = (price / ctx.entry_price - 1.0) * 100.0
        velocity = None if previous_profit_pct is None else profit_pct - previous_profit_pct
        previous_profit_pct = profit_pct

        if price <= ctx.entry_price or not activated:
            continue
        if velocity is not None and velocity <= min_velocity_pct:
            return _sell_result(
                symbol=symbol,
                ctx=ctx,
                check_ts=check_ts,
                price=price,
                reason="VELOCITY",
                target_price=activate_price,
            )
        if price <= high_water * (1.0 - giveback_pct):
            return _sell_result(
                symbol=symbol,
                ctx=ctx,
                check_ts=check_ts,
                price=price,
                reason="VELOCITY_GIVEBACK",
                target_price=activate_price,
            )

    return _eod_result(symbol=symbol, ctx=ctx, last_price=last_price, target_price=activate_price)


def _fmt_price(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.2f}"


def _fmt_pct(value: Optional[float]) -> str:
    return "" if value is None else f"{value:+.2f}%"


def print_results(results: list[SimulationResult]) -> None:
    print(
        f"{'sym':<6} {'entry':>5} {'entry_px':>9} {'target':>9} "
        f"{'exit':>5} {'exit_px':>9} {'ret':>8} {'MFE':>8} {'MAE':>8} "
        f"{'miss':>5} {'hi_hit':>6} {'reason':>12}"
    )
    for r in results:
        print(
            f"{r.symbol:<6} {r.entry_time or '':>5} {_fmt_price(r.entry_price):>9} "
            f"{_fmt_price(r.target_price):>9} {r.exit_time or '':>5} {_fmt_price(r.exit_price):>9} "
            f"{_fmt_pct(r.return_pct):>8} {_fmt_pct(r.mfe_pct):>8} {_fmt_pct(r.mae_pct):>8} "
            f"{'Y' if r.missed_tp else '':>5} {r.first_high_tp_time or '':>6} {r.exit_reason:>12}"
        )

    ok = [r for r in results if r.entry_price is not None and r.return_pct is not None]
    if not ok:
        return
    tp = [r for r in ok if r.exit_reason == "TP"]
    sold = [r for r in ok if r.exit_reason != "EOD"]
    missed = [r for r in ok if r.missed_tp]
    total_ret = sum(float(r.return_pct or 0) for r in ok)
    avg_ret = total_ret / len(ok)
    avg_mfe = sum(float(r.mfe_pct or 0) for r in ok if r.mfe_pct is not None) / len(ok)
    sold_total_ret = sum(float(r.return_pct or 0) for r in sold)
    sold_avg_ret = sold_total_ret / len(sold) if sold else 0.0
    print(
        f"\nSummary: ok={len(ok)} TP={len(tp)} missed_tp={len(missed)} "
        f"sold={len(sold)} sold_total={sold_total_ret:+.2f}% sold_avg={sold_avg_ret:+.2f}% "
        f"basket_total={total_ret:+.2f}% basket_avg={avg_ret:+.2f}% avg_mfe={avg_mfe:+.2f}%"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample intraday exit policies.")
    parser.add_argument("--date", required=True, type=date.fromisoformat, help="Trading date YYYY-MM-DD")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols, space or comma separated")
    parser.add_argument("--entry-time", required=True, type=_parse_time, help="ET entry time HH:MM")
    parser.add_argument("--entry-times", nargs="*", help="Optional per-ticker entry times, e.g. CIEN=13:00")
    parser.add_argument("--entry-prices", nargs="*", help="Optional per-ticker entry prices, e.g. CIEN=482.73")
    parser.add_argument("--end-time", default=time(16, 0), type=_parse_time, help="ET end time HH:MM")
    parser.add_argument(
        "--exit-type",
        choices=["FIXED_TP", "IPC", "WAVE", "PEAK_AGE", "VELOCITY"],
        default="FIXED_TP",
    )
    parser.add_argument("--tp-pct", type=float, default=0.002, help="Take-profit decimal, e.g. 0.002 = 0.2%%")
    parser.add_argument("--activate-pct", type=float, default=0.005, help="IPC activation profit, e.g. 0.005 = 0.5%%")
    parser.add_argument("--giveback-pct", type=float, default=0.002, help="IPC giveback from high-water mark")
    parser.add_argument("--slope-lookback-minutes", type=int, default=30, help="WAVE slope lookback window")
    parser.add_argument(
        "--min-slope-pct-per-hour",
        type=float,
        default=0.0,
        help="WAVE exits green when recent slope is at/below this percent per hour",
    )
    parser.add_argument("--max-peak-age-minutes", type=int, default=30, help="PEAK_AGE max minutes since high")
    parser.add_argument(
        "--min-velocity-pct",
        type=float,
        default=0.0,
        help="VELOCITY exits green when profit change vs prior check is at/below this percentage",
    )
    parser.add_argument("--cadence-minutes", type=int, default=15, help="Check cadence in minutes")
    parser.add_argument(
        "--check-offset-minutes",
        type=int,
        default=1,
        help="Minute offset after the hour for checks; default gives :01/:16/:31/:46",
    )
    parser.add_argument(
        "--bar-interval",
        choices=["1m", "2m", "5m", "15m"],
        default="1m",
        help="yfinance intraday bar interval (1m is recent-only; use 15m for older dates)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tickers = _parse_tickers(args.tickers)
    entry_times = _parse_time_map(args.entry_times)
    entry_prices = _parse_price_map(args.entry_prices)
    data = _download_intraday(tickers, args.date, args.bar_interval)

    results: list[SimulationResult] = []
    for symbol in tickers:
        hist = _hist_for_date(_symbol_hist(data, symbol), args.date)
        common_args = {
            "scan_date": args.date,
            "entry_time": entry_times.get(symbol, args.entry_time),
            "end_time": args.end_time,
            "cadence_minutes": args.cadence_minutes,
            "offset_minutes": args.check_offset_minutes,
            "tp_pct": args.tp_pct,
            "entry_price_override": entry_prices.get(symbol),
        }
        if args.exit_type == "IPC":
            results.append(
                simulate_ipc(
                    symbol,
                    hist,
                    **common_args,
                    activate_pct=args.activate_pct,
                    giveback_pct=args.giveback_pct,
                )
            )
        elif args.exit_type == "WAVE":
            results.append(
                simulate_wave(
                    symbol,
                    hist,
                    **common_args,
                    activate_pct=args.activate_pct,
                    giveback_pct=args.giveback_pct,
                    slope_lookback_minutes=args.slope_lookback_minutes,
                    min_slope_pct_per_hour=args.min_slope_pct_per_hour,
                )
            )
        elif args.exit_type == "PEAK_AGE":
            results.append(
                simulate_peak_age(
                    symbol,
                    hist,
                    **common_args,
                    activate_pct=args.activate_pct,
                    giveback_pct=args.giveback_pct,
                    max_peak_age_minutes=args.max_peak_age_minutes,
                )
            )
        elif args.exit_type == "VELOCITY":
            results.append(
                simulate_velocity(
                    symbol,
                    hist,
                    **common_args,
                    activate_pct=args.activate_pct,
                    giveback_pct=args.giveback_pct,
                    min_velocity_pct=args.min_velocity_pct,
                )
            )
        else:
            results.append(simulate_fixed_tp(symbol, hist, **common_args))

    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
