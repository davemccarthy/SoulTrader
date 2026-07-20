#!/usr/bin/env python3
"""
Pulse v0 daily attention scan.

Baby-step goal:
  - Run once per trading day after 11:00 ET.
  - Fetch a broad Polygon grouped daily aggregate seed (same endpoint family as Vunder).
  - Rank liquid names by yfinance intraday volume so far on the scan date.
  - Save a date-named CSV and reuse it on later runs that day.

This builds a same-day "attention universe" from intraday volume so later
Pulse tests can run only against this shortlist. Optional time-adjusted RVOL
can compare today's cumulative volume with prior sessions at the same time.

Usage:
  source ~/Development/scratch/python/tutorial-env/bin/activate
  python test_pulse_scan.py
  python test_pulse_scan.py --top 200 --min-price 5 --min-dollar-volume 50000000
  python test_pulse_scan.py --refresh
  python test_pulse_scan.py --force   # allow before 11:00 ET
  python test_pulse_scan.py --stable --date 2026-07-08 --entry-time 11:00
  python test_pulse_scan.py --impulse-backtest --from 2026-07-06 --to 2026-07-10
  python test_pulse_scan.py --impulse-backtest --from 2026-06-22 --to 2026-07-10 \\
    --exit-type IPC --activate-pct 0.002 --giveback-pct 0.002 \\
    --impulse-activate-pct 0.006 --impulse-giveback-pct 0.004 \\
    --bandwagon-min-vol-ratio 1.5 --bandwagon-min-ret-15m 0.3 \\
    --bandwagon-lookback-minutes 5
  python test_pulse_scan.py --recovery-gate-backtest --from 2026-06-22 --to 2026-07-17 \\
    --exit-type IPC --activate-pct 0.002 --giveback-pct 0.002
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


ET = ZoneInfo("US/Eastern")
PULSE_SCAN_TIME_ET = time(11, 0)
DEFAULT_RANGE_LOOKBACK_MINUTES = 60
DEFAULT_CACHE_DIR = Path(".pulse_scan")
DEFAULT_SEED_UNIVERSE = 500
DEFAULT_MAX_PRICE_DRIFT_FROM_SEED = 0.50
DEFAULT_RVOL_LOOKBACK_DAYS = 20
PULSE_MIN_RANGE_PCT = 1.25
# Discover impulse (no profit signal; mirrors analysis.py runner minus profit).
IMPULSE_LOOKBACK_MINUTES = 30
IMPULSE_MIN_RET_30M_PCT = 1.0
IMPULSE_MIN_VOL_RATIO = 2.0
IMPULSE_MIN_CLOSE_POSITION = 0.6
IMPULSE_MIN_SIGNALS_DEFAULT = 3
# Bandwagon partner (vol spike + sharp price up); independent of 1m impulse scoring.
BANDWAGON_LOOKBACK_MINUTES_DEFAULT = 15
BANDWAGON_MIN_VOL_RATIO_DEFAULT = 1.5
BANDWAGON_MIN_RET_PCT_DEFAULT = 0.3
# Recovery-gate paper variants (keep 30m pullback fixed; change short confirm / trend).
RECOVERY_GATE_VARIANTS = (
    "A",  # production-like: down 30m, up 5m on 15m bars + stable_60/open
    "B",  # down 30m, up 1m on 1m bars + stable_60/open
    "C",  # B + calc_trend-style 2h slope > threshold
)
RECOVERY_GATE_TREND_HOURS = 2
RECOVERY_GATE_MIN_TREND_DEFAULT = -0.05
DISCOVER_BUCKETS = (
    "recovery",
    "impulse_only",
    "impulse_any",
    "both",
    "bandwagon",
    "bandwagon_only",
)
# Back-compat alias used by reports and aggregation loops.
IMPULSE_BUCKETS = DISCOVER_BUCKETS
# Partner buckets use --impulse-activate-pct / --impulse-giveback-pct when set.
PARTNER_IPC_BUCKETS = frozenset(
    {"impulse_only", "impulse_any", "both", "bandwagon", "bandwagon_only"}
)
IMPULSE_IPC_BUCKETS = PARTNER_IPC_BUCKETS
ETF_EXCLUDE_TICKERS = frozenset(
    {
        "DIA",
        "EEM",
        "EFA",
        "EWY",
        "GDX",
        "GLD",
        "HYG",
        "IEFA",
        "IWD",
        "IWF",
        "IWM",
        "IVV",
        "LQD",
        "NVD",
        "NVDL",
        "NVDS",
        "NVDU",
        "QQQ",
        "RSP",
        "SDS",
        "SGOV",
        "SH",
        "SMH",
        "SNXX",
        "SOXL",
        "SOXS",
        "SOXX",
        "SPXL",
        "SPXS",
        "SPY",
        "SQQQ",
        "TLT",
        "TSLL",
        "TSLQ",
        "TSLR",
        "TSLS",
        "TSLT",
        "TQQQ",
        "UDOW",
        "UPRO",
        "VCIT",
        "VCSH",
        "VEA",
        "VOO",
        "VTI",
        "VXUS",
        "XBI",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLV",
        "VYM",
    }
)


@dataclass(frozen=True)
class PulseCandidate:
    rank: int
    symbol: str
    price: float
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    volume: int
    dollar_volume: float
    prior_dollar_volume: Optional[float]
    rvol: Optional[float]
    change_pct: Optional[float]
    prev_close: Optional[float]


def _setup_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()


def _today_et() -> date:
    return datetime.now(ET).date()


def _cache_path(cache_dir: Path, scan_date: date) -> Path:
    return cache_dir / f"pulse_candidates_{scan_date.isoformat()}.csv"


def _stable_cache_path(cache_dir: Path, scan_date: date) -> Path:
    return cache_dir / f"pulse_candidates_{scan_date.isoformat()}_stable.csv"


def _simulation_cache_path(cache_dir: Path, scan_date: date) -> Path:
    return cache_dir / f"pulse_candidates_{scan_date.isoformat()}_sim.csv"


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if out <= 0:
            return None
        return out
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def fetch_grouped_daily_rows(
    *,
    min_price: float,
    scan_date: date,
) -> list[dict[str, Any]]:
    """Fetch Polygon grouped daily aggs via the existing Vunder-compatible helper."""
    _setup_django()
    from core.services.financial import polygon as financial_polygon

    df = financial_polygon.get_filtered_stocks(
        min_price=min_price,
        test_date=scan_date.isoformat(),
    )
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


def build_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    scan_date: date,
    as_of_time: time,
    min_price: float,
    min_volume: int,
    min_dollar_volume: float,
    min_rvol: Optional[float],
    rvol_lookback_days: int,
    top: int,
    seed_universe: int,
    max_price_drift_from_seed: float,
    exclude_etfs: bool,
) -> list[PulseCandidate]:
    seed_rows: list[dict[str, Any]] = []

    for row in rows:
        symbol = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
        if not symbol or "." in symbol:
            continue
        if exclude_etfs and symbol in ETF_EXCLUDE_TICKERS:
            continue

        price = _safe_float(row.get("price"))
        prior_volume = _safe_int(row.get("today_volume") or row.get("volume"))
        if price is None or price < min_price or prior_volume <= 0:
            continue

        prior_dollar_volume = price * prior_volume
        seed_rows.append(
            {
                "symbol": symbol,
                "seed_price": price,
                "prior_volume": prior_volume,
                "prior_dollar_volume": prior_dollar_volume,
            }
        )

    seed_rows.sort(key=lambda r: float(r["prior_dollar_volume"]), reverse=True)
    shortlist = seed_rows[:seed_universe]
    symbols = [str(r["symbol"]) for r in shortlist]
    data = _download_intraday(symbols, scan_date=scan_date)
    rvol_data = (
        _download_intraday_lookback(symbols, scan_date=scan_date, lookback_days=rvol_lookback_days)
        if min_rvol is not None
        else pd.DataFrame()
    )
    as_of_ts = _as_of_timestamp(scan_date, as_of_time)

    ranked_rows: list[dict[str, Any]] = []
    for row in shortlist:
        symbol = str(row["symbol"])
        hist = _hist_for_date(_symbol_hist(data, symbol), scan_date)
        if hist.empty or "Close" not in hist.columns or "Open" not in hist.columns:
            continue

        price_now = _bar_close_at_as_of(hist, as_of_ts)
        seed_price = _safe_float(row.get("seed_price"))
        if price_now is None or seed_price is None:
            continue
        if abs(price_now / seed_price - 1.0) > max_price_drift_from_seed:
            continue

        volume_so_far = _volume_at_or_before(hist, as_of_ts)
        if volume_so_far < min_volume:
            continue
        rvol = None
        if min_rvol is not None:
            rvol_hist = _symbol_hist(rvol_data, symbol)
            rvol = _time_adjusted_rvol(
                rvol_hist,
                scan_date=scan_date,
                as_of_time=as_of_time,
                today_volume_so_far=volume_so_far,
            )
            if rvol is None or rvol < min_rvol:
                continue
        dollar_volume = price_now * volume_so_far
        if dollar_volume < min_dollar_volume:
            continue

        open_px = _safe_float(hist["Open"].iloc[0])
        day_high = _safe_float(hist.loc[pd.to_datetime(hist.index, utc=True) <= as_of_ts, "High"].max()) if "High" in hist.columns else None
        day_low = _safe_float(hist.loc[pd.to_datetime(hist.index, utc=True) <= as_of_ts, "Low"].min()) if "Low" in hist.columns else None
        ranked_rows.append(
            {
                **row,
                "price": price_now,
                "open": open_px,
                "high": day_high,
                "low": day_low,
                "volume": volume_so_far,
                "dollar_volume": dollar_volume,
                "rvol": rvol,
            }
        )

    ranked_rows.sort(key=lambda r: float(r["dollar_volume"]), reverse=True)
    candidates: list[PulseCandidate] = []
    for rank, row in enumerate(ranked_rows[:top], start=1):
        open_px = _safe_float(row.get("open"))
        price = float(row["price"])
        change_pct = _pct(price, open_px)

        candidates.append(
            PulseCandidate(
                rank=rank,
                symbol=str(row["symbol"]),
                price=price,
                open=open_px,
                high=_safe_float(row.get("high")),
                low=_safe_float(row.get("low")),
                volume=int(row["volume"]),
                dollar_volume=float(row["dollar_volume"]),
                prior_dollar_volume=float(row["prior_dollar_volume"]),
                rvol=_safe_float(row.get("rvol")),
                change_pct=change_pct,
                prev_close=None,
            )
        )
    return candidates


def write_candidates(path: Path, candidates: list[PulseCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "symbol",
                "price",
                "open",
                "high",
                "low",
                "volume",
                "dollar_volume",
                "prior_dollar_volume",
                "rvol",
                "change_pct",
                "prev_close",
            ],
        )
        writer.writeheader()
        for c in candidates:
            writer.writerow(
                {
                    "rank": c.rank,
                    "symbol": c.symbol,
                    "price": f"{c.price:.4f}",
                    "open": "" if c.open is None else f"{c.open:.4f}",
                    "high": "" if c.high is None else f"{c.high:.4f}",
                    "low": "" if c.low is None else f"{c.low:.4f}",
                    "volume": c.volume,
                    "dollar_volume": f"{c.dollar_volume:.2f}",
                    "prior_dollar_volume": "" if c.prior_dollar_volume is None else f"{c.prior_dollar_volume:.2f}",
                    "rvol": "" if c.rvol is None else f"{c.rvol:.4f}",
                    "change_pct": "" if c.change_pct is None else f"{c.change_pct:.4f}",
                    "prev_close": "" if c.prev_close is None else f"{c.prev_close:.4f}",
                }
            )


def read_cached(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _pct(current: Optional[float], base: Optional[float]) -> Optional[float]:
    if current is None or base is None or base <= 0:
        return None
    return (current / base - 1.0) * 100.0


def _as_of_timestamp(scan_date: date, as_of_time: time) -> pd.Timestamp:
    return pd.Timestamp(
        datetime.combine(scan_date, as_of_time, tzinfo=ET)
    ).tz_convert("UTC")


def _bar_close_at_or_before(
    hist: pd.DataFrame,
    minutes_ago: int,
    *,
    as_of_ts: Optional[pd.Timestamp] = None,
) -> Optional[float]:
    if hist.empty or "Close" not in hist.columns:
        return None
    idx = pd.to_datetime(hist.index, utc=True)
    anchor = as_of_ts or pd.Timestamp.now(tz="UTC")
    cutoff = anchor - pd.Timedelta(minutes=minutes_ago)
    eligible = idx <= cutoff
    if not eligible.any():
        return None
    return _safe_float(hist.loc[eligible, "Close"].astype(float).iloc[-1])


def _bar_close_at_as_of(hist: pd.DataFrame, as_of_ts: pd.Timestamp) -> Optional[float]:
    if hist.empty or "Close" not in hist.columns:
        return None
    idx = pd.to_datetime(hist.index, utc=True)
    eligible = idx <= as_of_ts
    if not eligible.any():
        return None
    return _safe_float(hist.loc[eligible, "Close"].astype(float).iloc[-1])


def _range_pct_at_or_before(hist: pd.DataFrame, as_of_ts: pd.Timestamp) -> Optional[float]:
    if hist.empty or "High" not in hist.columns or "Low" not in hist.columns:
        return None
    idx = pd.to_datetime(hist.index, utc=True)
    eligible = idx <= as_of_ts
    if not eligible.any():
        return None
    sub = hist.loc[eligible]
    hi = _safe_float(sub["High"].max())
    lo = _safe_float(sub["Low"].min())
    if hi is None or lo is None or lo <= 0:
        return None
    return (hi / lo - 1.0) * 100.0


def _range_pct_between(
    hist: pd.DataFrame,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Optional[float]:
    if hist.empty or "High" not in hist.columns or "Low" not in hist.columns:
        return None
    idx = pd.to_datetime(hist.index, utc=True)
    eligible = (idx >= start_ts) & (idx <= end_ts)
    if not eligible.any():
        return None
    sub = hist.loc[eligible]
    hi = _safe_float(sub["High"].max())
    lo = _safe_float(sub["Low"].min())
    if hi is None or lo is None or lo <= 0:
        return None
    return (hi / lo - 1.0) * 100.0


def _stable_bool(value: Optional[bool]) -> str:
    if value is True:
        return "Y"
    if value is False:
        return "n"
    return ""


def _bandwagon_bar_interval(lookback_minutes: int) -> str:
    """Finer bars when lookback is short (5m data for <=5m windows)."""
    if lookback_minutes <= 5:
        return "5m"
    return "15m"


def _download_intraday_bars(
    symbols: list[str],
    *,
    scan_date: date,
    interval: str,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    start = scan_date.isoformat()
    end = (scan_date + timedelta(days=1)).isoformat()
    return yf.download(
        symbols,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )


def _download_intraday_1m(symbols: list[str], *, scan_date: date) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    start = scan_date.isoformat()
    end = (scan_date + timedelta(days=1)).isoformat()
    return yf.download(
        symbols,
        start=start,
        end=end,
        interval="1m",
        auto_adjust=True,
        progress=False,
        threads=True,
    )


def _download_intraday(symbols: list[str], *, scan_date: Optional[date] = None) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    if scan_date is not None:
        start = scan_date.isoformat()
        end = (scan_date + timedelta(days=1)).isoformat()
        return yf.download(
            symbols,
            start=start,
            end=end,
            interval="15m",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    return yf.download(
        symbols,
        period="1d",
        interval="15m",
        auto_adjust=True,
        progress=False,
        threads=True,
    )


def _download_intraday_lookback(
    symbols: list[str],
    *,
    scan_date: date,
    lookback_days: int,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    # Include extra calendar days so weekends/holidays still leave enough sessions.
    calendar_days = max(lookback_days * 2, lookback_days + 10)
    start = (scan_date - timedelta(days=calendar_days)).isoformat()
    end = (scan_date + timedelta(days=1)).isoformat()
    return yf.download(
        symbols,
        start=start,
        end=end,
        interval="15m",
        auto_adjust=True,
        progress=False,
        threads=True,
    )


def _hist_for_date(hist: pd.DataFrame, scan_date: date) -> pd.DataFrame:
    if hist.empty:
        return hist
    idx = pd.to_datetime(hist.index, utc=True)
    local_dates = idx.tz_convert(ET).date
    return hist.loc[local_dates == scan_date].dropna(how="all")


def _volume_at_or_before(hist: pd.DataFrame, as_of_ts: pd.Timestamp) -> int:
    if hist.empty or "Volume" not in hist.columns:
        return 0
    idx = pd.to_datetime(hist.index, utc=True)
    eligible = idx <= as_of_ts
    if not eligible.any():
        return 0
    return _safe_int(hist.loc[eligible, "Volume"].fillna(0).sum())


def _time_adjusted_rvol(
    hist: pd.DataFrame,
    *,
    scan_date: date,
    as_of_time: time,
    today_volume_so_far: int,
) -> Optional[float]:
    if hist.empty or "Volume" not in hist.columns or today_volume_so_far <= 0:
        return None

    idx = pd.to_datetime(hist.index, utc=True)
    local_idx = idx.tz_convert(ET)
    df = hist.copy()
    df["_local_date"] = local_idx.date
    df["_local_time"] = local_idx.time

    prior_volumes: list[int] = []
    for session_date, session in df.groupby("_local_date"):
        if session_date >= scan_date:
            continue
        comparable = session[session["_local_time"] <= as_of_time]
        if comparable.empty:
            continue
        volume = _safe_int(comparable["Volume"].fillna(0).sum())
        if volume > 0:
            prior_volumes.append(volume)

    if not prior_volumes:
        return None
    avg_prior = sum(prior_volumes) / len(prior_volumes)
    if avg_prior <= 0:
        return None
    return float(today_volume_so_far / avg_prior)


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


def enrich_stability(
    rows: list[dict[str, str]],
    *,
    scan_date: date,
    as_of_time: time,
    range_lookback_minutes: Optional[int],
) -> list[dict[str, str]]:
    symbols = [str(r.get("symbol") or "").strip().upper() for r in rows if r.get("symbol")]
    data = _download_intraday(symbols, scan_date=scan_date)
    as_of_ts = _as_of_timestamp(scan_date, as_of_time)
    enriched: list[dict[str, str]] = []

    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        hist = _hist_for_date(_symbol_hist(data, symbol), scan_date)
        out = dict(row)

        px_now = _bar_close_at_as_of(hist, as_of_ts)
        open_px = _safe_float(hist["Open"].iloc[0]) if not hist.empty and "Open" in hist.columns else None
        px_30 = _bar_close_at_or_before(hist, 30, as_of_ts=as_of_ts)
        px_5 = _bar_close_at_or_before(hist, 5, as_of_ts=as_of_ts)
        px_60 = _bar_close_at_or_before(hist, 60, as_of_ts=as_of_ts)

        pct_open = _pct(px_now, open_px)
        pct_5 = _pct(px_now, px_5)
        pct_30 = _pct(px_now, px_30)
        pct_60 = _pct(px_now, px_60)
        range_pct_to_asof = _range_pct_at_or_before(hist, as_of_ts)
        recent_range_pct = None
        if range_lookback_minutes is not None and range_lookback_minutes > 0:
            recent_range_pct = _range_pct_between(
                hist,
                start_ts=as_of_ts - pd.Timedelta(minutes=range_lookback_minutes),
                end_ts=as_of_ts,
            )
        full_day_range_pct = None
        if not hist.empty and "High" in hist.columns and "Low" in hist.columns:
            hi = _safe_float(hist["High"].max())
            lo = _safe_float(hist["Low"].min())
            if hi is not None and lo is not None and lo > 0:
                full_day_range_pct = (hi / lo - 1.0) * 100.0

        recovering = (
            px_now is not None
            and px_30 is not None
            and px_5 is not None
            and px_now < px_30
            and px_now > px_5
        )
        stable_60 = px_now is not None and px_60 is not None and px_now >= px_60 * 0.995
        stable_open = px_now is not None and open_px is not None and px_now >= open_px * 0.99
        normal_stable = recovering and stable_60 and stable_open

        out.update(
            {
                "price_now": "" if px_now is None else f"{px_now:.4f}",
                "open_now": "" if open_px is None else f"{open_px:.4f}",
                "price_30m_ago": "" if px_30 is None else f"{px_30:.4f}",
                "price_5m_ago": "" if px_5 is None else f"{px_5:.4f}",
                "price_60m_ago": "" if px_60 is None else f"{px_60:.4f}",
                "pct_from_open": "" if pct_open is None else f"{pct_open:.4f}",
                "pct_5m": "" if pct_5 is None else f"{pct_5:.4f}",
                "pct_30m": "" if pct_30 is None else f"{pct_30:.4f}",
                "pct_60m": "" if pct_60 is None else f"{pct_60:.4f}",
                "range_pct_to_asof": "" if range_pct_to_asof is None else f"{range_pct_to_asof:.4f}",
                "recent_range_pct": "" if recent_range_pct is None else f"{recent_range_pct:.4f}",
                "recent_range_minutes": "" if range_lookback_minutes is None else str(range_lookback_minutes),
                "day_range_pct": "" if full_day_range_pct is None else f"{full_day_range_pct:.4f}",
                "recovering": _stable_bool(recovering if px_30 is not None and px_5 is not None else None),
                "stable_60m": _stable_bool(stable_60 if px_60 is not None else None),
                "stable_open": _stable_bool(stable_open if open_px is not None else None),
                "normal_stable": _stable_bool(normal_stable),
                "stable_as_of_et": as_of_time.strftime("%H:%M"),
            }
        )
        enriched.append(out)

    return enriched


def _normalized_slope_closes(closes: list[float]) -> Optional[float]:
    """Mirror Stock.calc_trend normalize: (OLS slope / mean price) * 100."""
    n = len(closes)
    if n < 2:
        return None
    x_mean = (n - 1) / 2.0
    y_mean = sum(closes) / n
    if y_mean == 0:
        return None
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(closes))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return None
    return (num / den / y_mean) * 100.0


def _trend_at_as_of(
    hist: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    *,
    hours: float = RECOVERY_GATE_TREND_HOURS,
) -> Optional[float]:
    """2h (default) normalized slope on bars at or before as_of."""
    if hist.empty or "Close" not in hist.columns:
        return None
    idx = pd.to_datetime(hist.index, utc=True)
    eligible = hist.loc[idx <= as_of_ts]
    if eligible.empty:
        return None
    start = as_of_ts - pd.Timedelta(hours=hours)
    window = eligible.loc[pd.to_datetime(eligible.index, utc=True) >= start]
    closes = [float(x) for x in window["Close"].astype(float).tolist() if float(x) > 0]
    return _normalized_slope_closes(closes)


def _recovering_vs_lookbacks(
    px_now: Optional[float],
    px_medium: Optional[float],
    px_short: Optional[float],
) -> Optional[bool]:
    if px_now is None or px_medium is None or px_short is None:
        return None
    return px_now < px_medium and px_now > px_short


def enrich_recovery_gate_variants(
    rows: list[dict[str, str]],
    *,
    scan_date: date,
    as_of_time: time,
    min_trend: float = RECOVERY_GATE_MIN_TREND_DEFAULT,
) -> list[dict[str, str]]:
    """
    Paper flags for recovery-gate A/B/C (does not rewrite normal_stable).

    A: down 30m / up 5m on 15m bars + stable_60/open (production-like)
    B: down 30m / up 1m on 1m bars + same stable_60/open from 15m
    C: B and 2h trend slope > min_trend
    """
    symbols = [str(r.get("symbol") or "").strip().upper() for r in rows if r.get("symbol")]
    data_15m = _download_intraday(symbols, scan_date=scan_date)
    data_1m = _download_intraday_1m(symbols, scan_date=scan_date)
    as_of_ts = _as_of_timestamp(scan_date, as_of_time)
    enriched: list[dict[str, str]] = []

    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        hist_15 = _hist_for_date(_symbol_hist(data_15m, symbol), scan_date)
        hist_1m = _hist_for_date(_symbol_hist(data_1m, symbol), scan_date)
        out = dict(row)

        px_now_15 = _bar_close_at_as_of(hist_15, as_of_ts)
        open_px = (
            _safe_float(hist_15["Open"].iloc[0])
            if not hist_15.empty and "Open" in hist_15.columns
            else None
        )
        px_30_15 = _bar_close_at_or_before(hist_15, 30, as_of_ts=as_of_ts)
        px_5_15 = _bar_close_at_or_before(hist_15, 5, as_of_ts=as_of_ts)
        px_60_15 = _bar_close_at_or_before(hist_15, 60, as_of_ts=as_of_ts)

        # Prefer live-ish 1m close at as_of for B/C short window; fall back to 15m now.
        px_now_1m = _bar_close_at_as_of(hist_1m, as_of_ts)
        px_now = px_now_1m if px_now_1m is not None else px_now_15
        px_30_1m = _bar_close_at_or_before(hist_1m, 30, as_of_ts=as_of_ts)
        px_1_1m = _bar_close_at_or_before(hist_1m, 1, as_of_ts=as_of_ts)

        stable_60 = px_now_15 is not None and px_60_15 is not None and px_now_15 >= px_60_15 * 0.995
        stable_open = px_now_15 is not None and open_px is not None and px_now_15 >= open_px * 0.99

        recovering_a = _recovering_vs_lookbacks(px_now_15, px_30_15, px_5_15)
        recovering_b = _recovering_vs_lookbacks(px_now, px_30_1m, px_1_1m)
        trend = _trend_at_as_of(hist_15, as_of_ts, hours=RECOVERY_GATE_TREND_HOURS)
        trend_ok = trend is not None and trend > min_trend

        gate_a = bool(recovering_a) and stable_60 and stable_open
        gate_b = bool(recovering_b) and stable_60 and stable_open
        gate_c = gate_b and trend_ok

        out.update(
            {
                "recovery_A": _stable_bool(gate_a if recovering_a is not None else None),
                "recovery_B": _stable_bool(gate_b if recovering_b is not None else None),
                "recovery_C": _stable_bool(gate_c if recovering_b is not None else None),
                "recovery_trend_2h": "" if trend is None else f"{trend:.4f}",
                "recovery_trend_ok": _stable_bool(trend_ok if trend is not None else None),
                "recovery_as_of_et": as_of_time.strftime("%H:%M"),
            }
        )
        enriched.append(out)

    return enriched


def _rows_for_recovery_variant(rows: list[dict[str, str]], variant: str) -> list[dict[str, str]]:
    """Map variant flag onto normal_stable so simulate_pulse(bucket=recovery) reuses existing path."""
    key = f"recovery_{variant}"
    out: list[dict[str, str]] = []
    for row in rows:
        cloned = dict(row)
        cloned["normal_stable"] = row.get(key) or ""
        cloned["recovery_variant"] = variant
        out.append(cloned)
    return out


def print_recovery_gate_report(
    summaries: dict[str, dict[str, Any]],
    *,
    scan_date: Optional[date],
) -> None:
    label = scan_date.isoformat() if scan_date else "TOTAL"
    print(f"\n=== Recovery gate variants @ {label} ===")
    print(
        f"{'var':>3}  {'n':>5}  {'ex%':>6}  {'win%':>7}  {'avg%':>8}  "
        f"{'worst%':>8}  {'best%':>8}  {'avg_tr':>7}"
    )
    for variant in RECOVERY_GATE_VARIANTS:
        s = summaries.get(variant) or {}
        n = int(s.get("n") or 0)
        if n == 0:
            print(f"{variant:>3}  {0:>5}")
            continue
        print(
            f"{variant:>3}  {n:>5}  {float(s['exit_pct']):>5.1f}%  "
            f"{float(s['win_pct']):>6.1f}%  {float(s['avg_ret']):>+7.3f}%  "
            f"{float(s['worst']):>+7.2f}%  {float(s['best']):>+7.2f}%  "
            f"{float(s['avg_tranches']):>7.2f}"
        )


def run_recovery_gate_backtest_for_date(
    *,
    cache_dir: Path,
    scan_date: date,
    entry_time: time,
    range_lookback_minutes: int,
    min_range_pct: float,
    exit_type: str,
    tp_pct: float,
    activate_pct: float,
    giveback_pct: float,
    rebuy_pct: float,
    max_tranches: int,
    min_trend: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """
    Compare recovery-gate A/B/C on one cached attention day.

    Requires candidate CSV; builds/reuses stable enrich for range filters, then
    scores A/B/C with fresh 15m+1m downloads (B/C need 1m).
    """
    path = _cache_path(cache_dir, scan_date)
    stable_path = _stable_cache_path(cache_dir, scan_date)

    if not path.exists():
        raise FileNotFoundError(f"Candidate CSV missing: {path}")
    if not stable_path.exists():
        rows = read_cached(path)
        enriched = enrich_stability(
            rows,
            scan_date=scan_date,
            as_of_time=entry_time,
            range_lookback_minutes=range_lookback_minutes,
        )
        write_dict_rows(stable_path, enriched)
    else:
        enriched = read_cached(stable_path)

    with_gates = enrich_recovery_gate_variants(
        enriched,
        scan_date=scan_date,
        as_of_time=entry_time,
        min_trend=min_trend,
    )

    summaries: dict[str, dict[str, Any]] = {}
    all_sim: list[dict[str, str]] = []
    for variant in RECOVERY_GATE_VARIANTS:
        variant_rows = _rows_for_recovery_variant(with_gates, variant)
        simulated = simulate_pulse(
            variant_rows,
            scan_date=scan_date,
            entry_time=entry_time,
            exit_type=exit_type,
            tp_pct=tp_pct,
            activate_pct=activate_pct,
            giveback_pct=giveback_pct,
            rebuy_pct=rebuy_pct,
            max_tranches=max_tranches,
            min_range_pct=min_range_pct,
            max_range_pct=None,
            min_recent_range_pct=None,
            max_recent_range_pct=None,
            bucket="recovery",
        )
        for row in simulated:
            row["sim_recovery_variant"] = variant
        summaries[variant] = _summarize_sim_rows(simulated)
        all_sim.extend(simulated)

    return summaries, all_sim


def _compute_bandwagon_at_as_of(
    hist: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    *,
    lookback_minutes: int,
    min_vol_ratio: float,
    min_ret_pct: float,
) -> dict[str, str]:
    empty = {
        "bandwagon_lookback_min": str(lookback_minutes),
        "bandwagon_ret_pct": "",
        "bandwagon_vol_last": "",
        "bandwagon_vol_prior": "",
        "bandwagon_vol_ratio": "",
        "bandwagon_pass": "",
        # legacy aliases (ret/vol over lookback window, not always 15m)
        "pct_15m": "",
        "vol_15m": "",
        "vol_prior_15m": "",
        "vol_ratio_15m": "",
    }
    if hist.empty or "Close" not in hist.columns:
        return empty

    px_now = _bar_close_at_as_of(hist, as_of_ts)
    px_ago = _bar_close_at_or_before(hist, lookback_minutes, as_of_ts=as_of_ts)
    ret_pct = _pct(px_now, px_ago)

    last_start = as_of_ts - pd.Timedelta(minutes=lookback_minutes)
    prior_start = as_of_ts - pd.Timedelta(minutes=lookback_minutes * 2)
    vol_last = _volume_sum_hist_range(hist, last_start, as_of_ts)
    vol_prior = _volume_sum_hist_range(hist, prior_start, last_start)

    vol_ratio = None
    if vol_last is not None and vol_prior is not None and vol_prior > 0:
        vol_ratio = vol_last / vol_prior

    passes = (
        vol_ratio is not None
        and vol_ratio >= min_vol_ratio
        and ret_pct is not None
        and ret_pct >= min_ret_pct
    )
    fields = {
        "bandwagon_lookback_min": str(lookback_minutes),
        "bandwagon_ret_pct": "" if ret_pct is None else f"{ret_pct:.4f}",
        "bandwagon_vol_last": "" if vol_last is None else f"{vol_last:.0f}",
        "bandwagon_vol_prior": "" if vol_prior is None else f"{vol_prior:.0f}",
        "bandwagon_vol_ratio": "" if vol_ratio is None else f"{vol_ratio:.4f}",
        "bandwagon_pass": _stable_bool(passes),
        "pct_15m": "" if ret_pct is None else f"{ret_pct:.4f}",
        "vol_15m": "" if vol_last is None else f"{vol_last:.0f}",
        "vol_prior_15m": "" if vol_prior is None else f"{vol_prior:.0f}",
        "vol_ratio_15m": "" if vol_ratio is None else f"{vol_ratio:.4f}",
    }
    return fields


def enrich_bandwagon(
    rows: list[dict[str, str]],
    *,
    scan_date: date,
    as_of_time: time,
    lookback_minutes: int,
    min_vol_ratio: float,
    min_ret_pct: float,
    min_range_pct: float,
) -> list[dict[str, str]]:
    """Bandwagon: recent volume spike + sharp price up over lookback (5m or 15m bars)."""
    symbols = [str(r.get("symbol") or "").strip().upper() for r in rows if r.get("symbol")]
    interval = _bandwagon_bar_interval(lookback_minutes)
    data = _download_intraday_bars(symbols, scan_date=scan_date, interval=interval)
    as_of_ts = _as_of_timestamp(scan_date, as_of_time)
    out: list[dict[str, str]] = []

    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        hist = _hist_for_date(_symbol_hist(data, symbol), scan_date)
        merged = dict(row)
        bandwagon_fields = _compute_bandwagon_at_as_of(
            hist,
            as_of_ts,
            lookback_minutes=lookback_minutes,
            min_vol_ratio=min_vol_ratio,
            min_ret_pct=min_ret_pct,
        )
        merged.update(bandwagon_fields)

        range_pct = _safe_float(row.get("range_pct_to_asof"))
        if merged.get("bandwagon_pass") == "Y":
            if range_pct is None or range_pct < min_range_pct:
                merged["bandwagon_pass"] = "n"

        out.append(merged)

    return out


def _high_in_hist_range(
    hist: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Optional[float]:
    if hist.empty or "High" not in hist.columns:
        return None
    idx = pd.to_datetime(hist.index, utc=True)
    window = hist[(idx >= start_ts) & (idx < end_ts)]
    if window.empty:
        return None
    return _safe_float(window["High"].astype(float).max())


def _volume_sum_hist_range(
    hist: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Optional[float]:
    if hist.empty or "Volume" not in hist.columns:
        return None
    idx = pd.to_datetime(hist.index, utc=True)
    window = hist[(idx >= start_ts) & (idx <= end_ts)]
    if window.empty:
        return None
    total = float(window["Volume"].astype(float).sum())
    return total if total > 0 else None


def _compute_impulse_at_as_of(
    hist: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    *,
    min_signals: int,
    min_ret_30m_pct: float,
) -> dict[str, str]:
    empty = {
        "impulse_score": "",
        "impulse_signals": "",
        "impulse_ret_30m": "",
        "impulse_vol_ratio": "",
        "impulse_close_pos": "",
        "impulse_pass": "",
    }
    if hist.empty or "Close" not in hist.columns:
        return empty

    idx = pd.to_datetime(hist.index, utc=True)
    eligible = idx <= as_of_ts
    if not eligible.any():
        return empty

    hist = hist.loc[eligible]
    idx = pd.to_datetime(hist.index, utc=True)
    price_now = _safe_float(hist["Close"].astype(float).iloc[-1])
    if price_now is None:
        return empty

    lookback = IMPULSE_LOOKBACK_MINUTES
    start_30 = as_of_ts - pd.Timedelta(minutes=lookback)
    prior_start = start_30 - pd.Timedelta(minutes=lookback)
    window = hist[(idx >= start_30) & (idx <= as_of_ts)]
    if window.empty or len(window) < 2:
        return empty

    prior_30 = hist.loc[idx <= start_30]
    if prior_30.empty:
        return empty
    price_30_ago = _safe_float(prior_30["Close"].astype(float).iloc[-1])
    if price_30_ago is None:
        return empty
    ret_30m_pct = (price_now / price_30_ago - 1.0) * 100.0

    highs = window["High"].astype(float) if "High" in window.columns else window["Close"].astype(float)
    lows = window["Low"].astype(float) if "Low" in window.columns else window["Close"].astype(float)
    high_max = float(highs.max())
    low_min = float(lows.min())
    if high_max > low_min:
        close_position = (price_now - low_min) / (high_max - low_min)
    else:
        close_position = 0.5

    last_10_start = as_of_ts - pd.Timedelta(minutes=10)
    prev_10_start = as_of_ts - pd.Timedelta(minutes=20)
    hi_last10 = _high_in_hist_range(hist, last_10_start, as_of_ts + pd.Timedelta(minutes=1))
    hi_prev10 = _high_in_hist_range(hist, prev_10_start, last_10_start)

    vol_last = _volume_sum_hist_range(hist, start_30, as_of_ts)
    vol_prior = _volume_sum_hist_range(hist, prior_start, start_30)
    vol_ratio = None
    if vol_last is not None and vol_prior is not None and vol_prior > 0:
        vol_ratio = vol_last / vol_prior

    signals: list[str] = []
    if ret_30m_pct is not None and ret_30m_pct >= min_ret_30m_pct:
        signals.append("ret30m")
    if hi_last10 is not None and hi_prev10 is not None and hi_last10 > hi_prev10:
        signals.append("hh")
    if vol_ratio is not None and vol_ratio >= IMPULSE_MIN_VOL_RATIO:
        signals.append("vol")
    if close_position >= IMPULSE_MIN_CLOSE_POSITION:
        signals.append("close")

    score = len(signals)
    return {
        "impulse_score": str(score),
        "impulse_signals": ",".join(signals),
        "impulse_ret_30m": "" if ret_30m_pct is None else f"{ret_30m_pct:.4f}",
        "impulse_vol_ratio": "" if vol_ratio is None else f"{vol_ratio:.4f}",
        "impulse_close_pos": f"{close_position:.4f}",
        "impulse_pass": _stable_bool(score >= min_signals),
    }


def enrich_impulse(
    rows: list[dict[str, str]],
    *,
    scan_date: date,
    as_of_time: time,
    min_range_pct: float,
    min_signals: int,
    min_ret_30m_pct: float,
) -> list[dict[str, str]]:
    """Add 1m impulse fields for rows passing range (cheap 15m gate already in stable CSV)."""
    eligible: list[dict[str, str]] = []
    for row in rows:
        range_pct = _safe_float(row.get("range_pct_to_asof"))
        if range_pct is None or range_pct < min_range_pct:
            continue
        eligible.append(row)

    symbols = [str(r.get("symbol") or "").strip().upper() for r in eligible if r.get("symbol")]
    data_1m = _download_intraday_1m(symbols, scan_date=scan_date)
    as_of_ts = _as_of_timestamp(scan_date, as_of_time)
    out: list[dict[str, str]] = []

    for row in rows:
        merged = dict(row)
        range_pct = _safe_float(row.get("range_pct_to_asof"))
        if range_pct is None or range_pct < min_range_pct:
            merged.update(
                {
                    "impulse_score": "",
                    "impulse_signals": "",
                    "impulse_ret_30m": "",
                    "impulse_vol_ratio": "",
                    "impulse_close_pos": "",
                    "impulse_pass": "",
                }
            )
            out.append(merged)
            continue

        symbol = str(row.get("symbol") or "").strip().upper()
        hist = _hist_for_date(_symbol_hist(data_1m, symbol), scan_date)
        merged.update(
            _compute_impulse_at_as_of(
                hist,
                as_of_ts,
                min_signals=min_signals,
                min_ret_30m_pct=min_ret_30m_pct,
            )
        )
        out.append(merged)

    return out


def _ipc_pcts_for_bucket(
    bucket: str,
    *,
    activate_pct: float,
    giveback_pct: float,
    impulse_activate_pct: float,
    impulse_giveback_pct: float,
) -> tuple[float, float]:
    if bucket in IMPULSE_IPC_BUCKETS:
        return impulse_activate_pct, impulse_giveback_pct
    return activate_pct, giveback_pct


def _row_in_bucket(row: dict[str, str], bucket: str) -> bool:
    stable = row.get("normal_stable") == "Y"
    impulse = row.get("impulse_pass") == "Y"
    bandwagon = row.get("bandwagon_pass") == "Y"
    if bucket == "recovery":
        return stable
    if bucket == "impulse_only":
        return impulse and not stable
    if bucket == "impulse_any":
        return impulse
    if bucket == "both":
        return stable and impulse
    if bucket == "bandwagon":
        return bandwagon
    if bucket == "bandwagon_only":
        return bandwagon and not stable
    raise ValueError(f"unknown bucket: {bucket}")


def write_dict_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_et_time(value: str) -> time:
    try:
        hh, mm = value.split(":", 1)
        return time(int(hh), int(mm))
    except Exception as exc:
        raise argparse.ArgumentTypeError("time must be HH:MM, e.g. 11:30") from exc


def _first_bar_at_or_after(hist: pd.DataFrame, entry_time: time) -> Optional[int]:
    if hist.empty:
        return None
    idx = pd.to_datetime(hist.index, utc=True)
    for i, ts in enumerate(idx):
        if ts.tz_convert(ET).time() >= entry_time:
            return i
    return None


def simulate_pulse(
    rows: list[dict[str, str]],
    *,
    scan_date: date,
    entry_time: time,
    exit_type: str,
    tp_pct: float,
    activate_pct: float,
    giveback_pct: float,
    rebuy_pct: float,
    max_tranches: int,
    min_range_pct: Optional[float],
    max_range_pct: Optional[float],
    min_recent_range_pct: Optional[float],
    max_recent_range_pct: Optional[float],
    bucket: str = "recovery",
) -> list[dict[str, str]]:
    entry_rows = []
    for row in rows:
        if not _row_in_bucket(row, bucket):
            continue
        range_pct = _safe_float(row.get("range_pct_to_asof"))
        if min_range_pct is not None and (range_pct is None or range_pct < min_range_pct):
            continue
        if max_range_pct is not None and (range_pct is None or range_pct > max_range_pct):
            continue
        recent_range_pct = _safe_float(row.get("recent_range_pct"))
        if min_recent_range_pct is not None and (
            recent_range_pct is None or recent_range_pct < min_recent_range_pct
        ):
            continue
        if max_recent_range_pct is not None and (
            recent_range_pct is None or recent_range_pct > max_recent_range_pct
        ):
            continue
        entry_rows.append(row)

    symbols = [str(r.get("symbol") or "").strip().upper() for r in entry_rows]
    data = _download_intraday(symbols, scan_date=scan_date)
    out: list[dict[str, str]] = []

    for row in entry_rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        hist = _hist_for_date(_symbol_hist(data, symbol), scan_date)
        result = dict(row)
        entry_i = _first_bar_at_or_after(hist, entry_time)

        if entry_i is None or hist.empty or "Close" not in hist.columns:
            result.update({"sim_status": "NO_DATA"})
            out.append(result)
            continue

        closes = hist["Close"].astype(float).reset_index(drop=True)
        highs = hist["High"].astype(float).reset_index(drop=True) if "High" in hist.columns else closes
        entry_px = float(closes.iloc[entry_i])
        avg = entry_px
        tranches = 1
        exit_px: Optional[float] = None
        exit_reason = "EOD"
        exit_i = len(closes) - 1
        rebuys = 0
        high_water = entry_px
        activated = False

        for i in range(entry_i + 1, len(closes)):
            close = float(closes.iloc[i])
            high_water = max(high_water, float(highs.iloc[i]))

            if exit_type == "IPC":
                activate_px = avg * (1.0 + activate_pct)
                if high_water >= activate_px:
                    activated = True
                if activated and close > avg and close <= high_water * (1.0 - giveback_pct):
                    exit_px = close
                    exit_reason = "IPC_GIVEBACK"
                    exit_i = i
                    break
            else:
                tp_px = avg * (1.0 + tp_pct)
                if float(highs.iloc[i]) >= tp_px:
                    exit_px = tp_px
                    exit_reason = "TP"
                    exit_i = i
                    break

            add_px = avg * (1.0 - rebuy_pct)
            if tranches < max_tranches and close <= add_px and i >= 2:
                close_30m_ago = float(closes.iloc[i - 2])
                if close > close_30m_ago:
                    avg = (avg * tranches + close) / (tranches + 1)
                    tranches += 1
                    rebuys += 1

        if exit_px is None:
            exit_px = float(closes.iloc[-1])

        ret_pct = (exit_px / avg - 1.0) * 100.0
        result.update(
            {
                "sim_bucket": bucket,
                "sim_ipc_activate_pct": f"{activate_pct:.6f}",
                "sim_ipc_giveback_pct": f"{giveback_pct:.6f}",
                "sim_entry_time_et": entry_time.strftime("%H:%M"),
                "sim_entry_price": f"{entry_px:.4f}",
                "sim_exit_price": f"{exit_px:.4f}",
                "sim_exit_type": exit_type,
                "sim_exit_reason": exit_reason,
                "sim_return_pct": f"{ret_pct:.4f}",
                "sim_tranches": str(tranches),
                "sim_rebuys": str(rebuys),
                "sim_bars_held": str(max(0, exit_i - entry_i)),
                "sim_status": "OK",
            }
        )
        out.append(result)

    return out


def _range_bucket(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if value < 1.0:
        return "<1%"
    if value < 2.0:
        return "1-2%"
    if value < 3.0:
        return "2-3%"
    if value <= 5.0:
        return "3-5%"
    return ">5%"


def print_range_report(rows: list[dict[str, str]]) -> None:
    buckets = ["<1%", "1-2%", "2-3%", "3-5%", ">5%", "n/a"]
    grouped: dict[str, list[dict[str, str]]] = {b: [] for b in buckets}

    for row in rows:
        if row.get("sim_status") != "OK":
            continue
        range_pct = _safe_float(row.get("range_pct_to_asof"))
        grouped[_range_bucket(range_pct)].append(row)

    print(f"{'range':>6}  {'n':>3} {'ex':>3} {'ex%':>6} {'avg_ret':>8} {'avg_tr':>7} {'worst':>8}")
    for bucket in buckets:
        sub = grouped[bucket]
        if not sub:
            continue
        tp = sum(1 for r in sub if r.get("sim_exit_reason") != "EOD")
        rets = [float(r.get("sim_return_pct") or 0) for r in sub]
        tranches = [int(r.get("sim_tranches") or 0) for r in sub]
        print(
            f"{bucket:>6}  {len(sub):>3} {tp:>3} {tp / len(sub) * 100:>5.1f}% "
            f"{sum(rets) / len(rets):>+7.2f}% {sum(tranches) / len(tranches):>7.2f} "
            f"{min(rets):>+7.2f}%"
        )


def _summarize_sim_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("sim_status") == "OK"]
    if not ok_rows:
        return {
            "n": 0,
            "exits": 0,
            "exit_pct": 0.0,
            "avg_ret": 0.0,
            "win_pct": 0.0,
            "avg_tranches": 0.0,
            "worst": 0.0,
            "best": 0.0,
        }
    rets = [float(r.get("sim_return_pct") or 0) for r in ok_rows]
    tranches = [int(r.get("sim_tranches") or 0) for r in ok_rows]
    exits = sum(1 for r in ok_rows if r.get("sim_exit_reason") != "EOD")
    wins = sum(1 for r in rets if r > 0)
    return {
        "n": len(ok_rows),
        "exits": exits,
        "exit_pct": exits / len(ok_rows) * 100.0,
        "avg_ret": sum(rets) / len(rets),
        "win_pct": wins / len(rets) * 100.0,
        "avg_tranches": sum(tranches) / len(tranches),
        "worst": min(rets),
        "best": max(rets),
    }


def print_impulse_bucket_report(
    summaries: dict[str, dict[str, Any]],
    *,
    scan_date: Optional[date] = None,
) -> None:
    if scan_date is not None:
        print(f"\n=== Impulse backtest {scan_date.isoformat()} ===")
    else:
        print("\n=== Impulse backtest (aggregated) ===")
    print(
        f"{'bucket':<16} {'n':>4} {'ex':>3} {'ex%':>6} {'avg_ret':>8} "
        f"{'win%':>6} {'avg_tr':>7} {'worst':>8} {'best':>8}"
    )
    for bucket in DISCOVER_BUCKETS:
        s = summaries.get(bucket) or {}
        if not s.get("n"):
            print(f"{bucket:<16}    0")
            continue
        print(
            f"{bucket:<16} {s['n']:>4} {s['exits']:>3} {s['exit_pct']:>5.1f}% "
            f"{s['avg_ret']:>+7.2f}% {s['win_pct']:>5.1f}% {s['avg_tranches']:>7.2f} "
            f"{s['worst']:>+7.2f}% {s['best']:>+7.2f}%"
        )


def _iter_dates(date_from: date, date_to: date) -> list[date]:
    out: list[date] = []
    current = date_from
    while current <= date_to:
        if current.weekday() < 5:
            out.append(current)
        current += timedelta(days=1)
    return out


def run_impulse_backtest_for_date(
    *,
    cache_dir: Path,
    scan_date: date,
    entry_time: time,
    range_lookback_minutes: int,
    min_range_pct: float,
    min_signals: int,
    min_ret_30m_pct: float,
    exit_type: str,
    tp_pct: float,
    activate_pct: float,
    giveback_pct: float,
    impulse_activate_pct: float,
    impulse_giveback_pct: float,
    bandwagon_min_vol_ratio: float,
    bandwagon_min_ret_pct: float,
    bandwagon_lookback_minutes: int,
    rebuy_pct: float,
    max_tranches: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    path = _cache_path(cache_dir, scan_date)
    stable_path = _stable_cache_path(cache_dir, scan_date)

    if not path.exists():
        raise FileNotFoundError(f"Candidate CSV missing: {path}")
    if not stable_path.exists():
        rows = read_cached(path)
        enriched = enrich_stability(
            rows,
            scan_date=scan_date,
            as_of_time=entry_time,
            range_lookback_minutes=range_lookback_minutes,
        )
        write_dict_rows(stable_path, enriched)
    else:
        enriched = read_cached(stable_path)

    with_bandwagon = enrich_bandwagon(
        enriched,
        scan_date=scan_date,
        as_of_time=entry_time,
        lookback_minutes=bandwagon_lookback_minutes,
        min_vol_ratio=bandwagon_min_vol_ratio,
        min_ret_pct=bandwagon_min_ret_pct,
        min_range_pct=min_range_pct,
    )
    with_impulse = enrich_impulse(
        with_bandwagon,
        scan_date=scan_date,
        as_of_time=entry_time,
        min_range_pct=min_range_pct,
        min_signals=min_signals,
        min_ret_30m_pct=min_ret_30m_pct,
    )

    summaries: dict[str, dict[str, Any]] = {}
    all_sim: list[dict[str, str]] = []
    for bucket in DISCOVER_BUCKETS:
        bucket_activate, bucket_giveback = _ipc_pcts_for_bucket(
            bucket,
            activate_pct=activate_pct,
            giveback_pct=giveback_pct,
            impulse_activate_pct=impulse_activate_pct,
            impulse_giveback_pct=impulse_giveback_pct,
        )
        simulated = simulate_pulse(
            with_impulse,
            scan_date=scan_date,
            entry_time=entry_time,
            exit_type=exit_type,
            tp_pct=tp_pct,
            activate_pct=bucket_activate,
            giveback_pct=bucket_giveback,
            rebuy_pct=rebuy_pct,
            max_tranches=max_tranches,
            min_range_pct=min_range_pct,
            max_range_pct=None,
            min_recent_range_pct=None,
            max_recent_range_pct=None,
            bucket=bucket,
        )
        summaries[bucket] = _summarize_sim_rows(simulated)
        all_sim.extend(simulated)

    return summaries, all_sim


def print_preview(rows: list[Any], *, limit: int, stable: bool = False, simulation: bool = False) -> None:
    if simulation:
        print(
            f"{'rank':>4}  {'symbol':<7} {'entry':>9} {'exit':>9} "
            f"{'rng%':>7} {'rrng%':>7} {'ret%':>8} {'tr':>3} {'rb':>3} {'reason':>6}"
        )
        for row in rows[:limit]:
            print(
                f"{row.get('rank', ''):>4}  {row.get('symbol', ''):<7} "
                f"{float(row.get('sim_entry_price') or 0):>9.2f} "
                f"{float(row.get('sim_exit_price') or 0):>9.2f} "
                f"{row.get('range_pct_to_asof', ''):>7} "
                f"{row.get('recent_range_pct', ''):>7} "
                f"{row.get('sim_return_pct', ''):>8} "
                f"{row.get('sim_tranches', ''):>3} {row.get('sim_rebuys', ''):>3} "
                f"{row.get('sim_exit_reason', row.get('sim_status', '')):>6}"
            )
        return

    if stable:
        print(
            f"{'rank':>4}  {'symbol':<7} {'price':>9} {'volume':>12} "
            f"{'dollar_vol':>14} {'30m':>7} {'60m':>7} {'open':>7} {'rng':>7} {'rrng':>7} {'stable':>6}"
        )
    else:
        print(f"{'rank':>4}  {'symbol':<7} {'price':>9} {'volume':>12} {'dollar_vol':>14} {'rvol':>6} {'chg%':>8}")
    for row in rows[:limit]:
        if isinstance(row, PulseCandidate):
            print(
                f"{row.rank:>4}  {row.symbol:<7} {row.price:>9.2f} "
                f"{row.volume:>12,d} {row.dollar_volume:>14,.0f} "
                f"{'' if row.rvol is None else f'{row.rvol:.2f}':>6} "
                f"{'' if row.change_pct is None else f'{row.change_pct:+.2f}':>8}"
            )
        else:
            volume = int(float(row.get("volume") or 0))
            dollar_volume = float(row.get("dollar_volume") or 0)
            rvol = row.get("rvol") or ""
            if stable:
                print(
                    f"{row.get('rank', ''):>4}  {row.get('symbol', ''):<7} "
                    f"{float(row.get('price_now') or row.get('price') or 0):>9.2f} "
                    f"{volume:>12,d} {dollar_volume:>14,.0f} "
                    f"{row.get('pct_30m', ''):>7} {row.get('pct_60m', ''):>7} "
                    f"{row.get('pct_from_open', ''):>7} {row.get('range_pct_to_asof', ''):>7} "
                    f"{row.get('recent_range_pct', ''):>7} "
                    f"{row.get('normal_stable', ''):>6}"
                )
                continue
            change = row.get("change_pct") or ""
            change_s = f"{float(change):+.2f}" if change else ""
            print(
                f"{row.get('rank', ''):>4}  {row.get('symbol', ''):<7} "
                f"{float(row.get('price') or 0):>9.2f} "
                f"{volume:>12,d} {dollar_volume:>14,.0f} "
                f"{f'{float(rvol):.2f}' if rvol else '':>6} {change_s:>8}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/reuse Pulse daily high-volume candidate CSV.")
    parser.add_argument("--top", type=int, default=100, help="Number of candidates to save (default 100)")
    parser.add_argument("--min-price", type=float, default=5.0, help="Minimum price (default 5)")
    parser.add_argument("--min-volume", type=int, default=500_000, help="Minimum day volume (default 500k)")
    parser.add_argument(
        "--min-dollar-volume",
        type=float,
        default=25_000_000,
        help="Minimum day dollar volume (default 25m)",
    )
    parser.add_argument("--min-rvol", type=float, help="Minimum time-adjusted intraday RVOL")
    parser.add_argument(
        "--rvol-lookback-days",
        type=int,
        default=DEFAULT_RVOL_LOOKBACK_DAYS,
        help=f"Prior sessions to use for RVOL average (default {DEFAULT_RVOL_LOOKBACK_DAYS})",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--date", type=str, help="Override scan date YYYY-MM-DD (default today ET)")
    parser.add_argument("--refresh", action="store_true", help="Rebuild even if today's CSV exists")
    parser.add_argument("--force", action="store_true", help="Allow scan before 11:00 ET")
    parser.add_argument("--preview", type=int, default=25, help="Rows to print (default 25)")
    parser.add_argument("--include-etfs", action="store_true", help="Do not exclude common ETF tickers")
    parser.add_argument(
        "--seed-universe",
        type=int,
        default=DEFAULT_SEED_UNIVERSE,
        help=f"Polygon seed size before intraday ranking (default {DEFAULT_SEED_UNIVERSE})",
    )
    parser.add_argument(
        "--max-price-drift-from-seed",
        type=float,
        default=DEFAULT_MAX_PRICE_DRIFT_FROM_SEED,
        help=f"Skip if intraday price differs from seed by more than this fraction (default {DEFAULT_MAX_PRICE_DRIFT_FROM_SEED})",
    )
    parser.add_argument("--stable", action="store_true", help="Enrich cached candidates with normal stability tier")
    parser.add_argument("--simulate", action="store_true", help="Simulate stable candidates through today's 15m bars")
    parser.add_argument("--range-report", action="store_true", help="Summarize simulation by entry-time range bucket")
    parser.add_argument("--entry-time", type=_parse_et_time, default=PULSE_SCAN_TIME_ET, help="ET entry time HH:MM")
    parser.add_argument(
        "--range-lookback-minutes",
        type=int,
        default=DEFAULT_RANGE_LOOKBACK_MINUTES,
        help=(
            "Calculate recent range over the N minutes before --entry-time "
            f"(default {DEFAULT_RANGE_LOOKBACK_MINUTES})"
        ),
    )
    parser.add_argument("--tp-pct", type=float, default=0.01, help="Take-profit percentage (default 0.01)")
    parser.add_argument(
        "--exit-type",
        choices=["FIXED_TP", "IPC"],
        default="FIXED_TP",
        help="Exit policy for simulation: fixed target or intraday peak/giveback",
    )
    parser.add_argument(
        "--activate-pct",
        type=float,
        default=0.002,
        help="IPC activation profit, e.g. 0.002 = 0.2%%",
    )
    parser.add_argument(
        "--giveback-pct",
        type=float,
        default=0.002,
        help="IPC giveback from high-water mark",
    )
    parser.add_argument("--rebuy-pct", type=float, default=0.02, help="Rebuy drop percentage (default 0.02)")
    parser.add_argument("--max-tranches", type=int, default=5, help="Max tranches including initial buy (default 5)")
    parser.add_argument("--min-range-pct", type=float, help="Only simulate names with range_pct_to_asof >= this")
    parser.add_argument("--max-range-pct", type=float, help="Only simulate names with range_pct_to_asof <= this")
    parser.add_argument("--min-recent-range-pct", type=float, help="Only simulate names with recent_range_pct >= this")
    parser.add_argument("--max-recent-range-pct", type=float, help="Only simulate names with recent_range_pct <= this")
    parser.add_argument(
        "--impulse-backtest",
        action="store_true",
        help="Compare recovery vs impulse discover buckets (requires stable/candidate CSV per date)",
    )
    parser.add_argument(
        "--from",
        dest="date_from",
        metavar="DATE",
        help="Start date YYYY-MM-DD for --impulse-backtest (inclusive)",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        metavar="DATE",
        help="End date YYYY-MM-DD for --impulse-backtest (inclusive)",
    )
    parser.add_argument(
        "--impulse-min-signals",
        type=int,
        default=IMPULSE_MIN_SIGNALS_DEFAULT,
        help=f"Min impulse signals (of 4) to pass gate (default {IMPULSE_MIN_SIGNALS_DEFAULT})",
    )
    parser.add_argument(
        "--impulse-min-range-pct",
        type=float,
        default=PULSE_MIN_RANGE_PCT,
        help=f"Min range_pct_to_asof before 1m impulse fetch (default {PULSE_MIN_RANGE_PCT})",
    )
    parser.add_argument(
        "--impulse-min-ret-30m-pct",
        type=float,
        default=IMPULSE_MIN_RET_30M_PCT,
        help=f"Impulse ret30m signal: min %% vs 30m ago (default {IMPULSE_MIN_RET_30M_PCT})",
    )
    parser.add_argument(
        "--impulse-activate-pct",
        type=float,
        default=None,
        help="IPC activation for impulse buckets (default: same as --activate-pct)",
    )
    parser.add_argument(
        "--impulse-giveback-pct",
        type=float,
        default=None,
        help="IPC giveback for partner buckets (default: same as --giveback-pct)",
    )
    parser.add_argument(
        "--bandwagon-min-vol-ratio",
        type=float,
        default=BANDWAGON_MIN_VOL_RATIO_DEFAULT,
        help=(
            f"Bandwagon: last lookback vol / prior lookback vol "
            f"(default {BANDWAGON_MIN_VOL_RATIO_DEFAULT})"
        ),
    )
    parser.add_argument(
        "--bandwagon-min-ret-15m",
        type=float,
        default=BANDWAGON_MIN_RET_PCT_DEFAULT,
        help=(
            f"Bandwagon: min %% price change over lookback window "
            f"(default {BANDWAGON_MIN_RET_PCT_DEFAULT})"
        ),
    )
    parser.add_argument(
        "--bandwagon-lookback-minutes",
        type=int,
        default=BANDWAGON_LOOKBACK_MINUTES_DEFAULT,
        help=(
            f"Bandwagon lookback for vol ratio and ret (default {BANDWAGON_LOOKBACK_MINUTES_DEFAULT}; "
            "uses 5m bars when <=5, else 15m)"
        ),
    )
    parser.add_argument(
        "--recovery-gate-backtest",
        action="store_true",
        help=(
            "Compare recovery-gate A (30m/5m 15m bars) vs B (30m/1m on 1m) vs "
            "C (B + 2h trend); requires candidate CSV per date"
        ),
    )
    parser.add_argument(
        "--recovery-min-trend",
        type=float,
        default=RECOVERY_GATE_MIN_TREND_DEFAULT,
        help=(
            f"Variant C: min normalized 2h slope (default {RECOVERY_GATE_MIN_TREND_DEFAULT}; "
            "mirrors Stock.calc_trend scale)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scan_date = date.fromisoformat(args.date) if args.date else _today_et()
    path = _cache_path(args.cache_dir, scan_date)
    stable_path = _stable_cache_path(args.cache_dir, scan_date)
    sim_path = _simulation_cache_path(args.cache_dir, scan_date)

    if args.impulse_backtest:
        date_from = date.fromisoformat(args.date_from) if args.date_from else scan_date
        date_to = date.fromisoformat(args.date_to) if args.date_to else date_from
        if date_to < date_from:
            print("--to must be on or after --from", file=sys.stderr)
            return 2
        if args.bandwagon_lookback_minutes <= 0:
            print("--bandwagon-lookback-minutes must be positive", file=sys.stderr)
            return 2

        aggregated: dict[str, list[dict[str, str]]] = {b: [] for b in DISCOVER_BUCKETS}
        dates = _iter_dates(date_from, date_to)
        if not dates:
            print("No weekdays in date range", file=sys.stderr)
            return 2

        impulse_activate_pct = (
            args.impulse_activate_pct
            if args.impulse_activate_pct is not None
            else args.activate_pct
        )
        impulse_giveback_pct = (
            args.impulse_giveback_pct
            if args.impulse_giveback_pct is not None
            else args.giveback_pct
        )

        for day in dates:
            try:
                summaries, sim_rows = run_impulse_backtest_for_date(
                    cache_dir=args.cache_dir,
                    scan_date=day,
                    entry_time=args.entry_time,
                    range_lookback_minutes=args.range_lookback_minutes,
                    min_range_pct=args.impulse_min_range_pct,
                    min_signals=args.impulse_min_signals,
                    min_ret_30m_pct=args.impulse_min_ret_30m_pct,
                    exit_type=args.exit_type,
                    tp_pct=args.tp_pct,
                    activate_pct=args.activate_pct,
                    giveback_pct=args.giveback_pct,
                    impulse_activate_pct=impulse_activate_pct,
                    impulse_giveback_pct=impulse_giveback_pct,
                    bandwagon_min_vol_ratio=args.bandwagon_min_vol_ratio,
                    bandwagon_min_ret_pct=args.bandwagon_min_ret_15m,
                    bandwagon_lookback_minutes=args.bandwagon_lookback_minutes,
                    rebuy_pct=args.rebuy_pct,
                    max_tranches=args.max_tranches,
                )
            except FileNotFoundError as exc:
                print(f"Skip {day.isoformat()}: {exc}", file=sys.stderr)
                continue
            print_impulse_bucket_report(summaries, scan_date=day)
            for row in sim_rows:
                if row.get("sim_status") == "OK":
                    bucket = row.get("sim_bucket") or ""
                    if bucket in aggregated:
                        aggregated[bucket].append(row)

        total_summaries = {b: _summarize_sim_rows(aggregated[b]) for b in DISCOVER_BUCKETS}
        print_impulse_bucket_report(total_summaries, scan_date=None)
        ipc_note = (
            f"recovery_ipc={args.activate_pct:g}/{args.giveback_pct:g} "
            f"partner_ipc={impulse_activate_pct:g}/{impulse_giveback_pct:g}"
        )
        bandwagon_note = (
            f"bandwagon {args.bandwagon_lookback_minutes}m vol>={args.bandwagon_min_vol_ratio:g}x "
            f"ret>={args.bandwagon_min_ret_15m:g}%"
        )
        print(
            f"Days attempted: {len(dates)} | impulse_min_signals={args.impulse_min_signals} "
            f"| impulse_ret30m>={args.impulse_min_ret_30m_pct:g}% "
            f"| min_range={args.impulse_min_range_pct:g}% | {bandwagon_note} "
            f"| exit={args.exit_type} | entry={args.entry_time.strftime('%H:%M')} ET | {ipc_note}"
        )
        return 0

    if args.recovery_gate_backtest:
        date_from = date.fromisoformat(args.date_from) if args.date_from else scan_date
        date_to = date.fromisoformat(args.date_to) if args.date_to else date_from
        if date_to < date_from:
            print("--to must be on or after --from", file=sys.stderr)
            return 2

        aggregated_rg: dict[str, list[dict[str, str]]] = {v: [] for v in RECOVERY_GATE_VARIANTS}
        dates = _iter_dates(date_from, date_to)
        if not dates:
            print("No weekdays in date range", file=sys.stderr)
            return 2

        for day in dates:
            try:
                summaries, sim_rows = run_recovery_gate_backtest_for_date(
                    cache_dir=args.cache_dir,
                    scan_date=day,
                    entry_time=args.entry_time,
                    range_lookback_minutes=args.range_lookback_minutes,
                    min_range_pct=args.impulse_min_range_pct,
                    exit_type=args.exit_type,
                    tp_pct=args.tp_pct,
                    activate_pct=args.activate_pct,
                    giveback_pct=args.giveback_pct,
                    rebuy_pct=args.rebuy_pct,
                    max_tranches=args.max_tranches,
                    min_trend=args.recovery_min_trend,
                )
            except FileNotFoundError as exc:
                print(f"Skip {day.isoformat()}: {exc}", file=sys.stderr)
                continue
            print_recovery_gate_report(summaries, scan_date=day)
            for row in sim_rows:
                if row.get("sim_status") != "OK":
                    continue
                variant = row.get("sim_recovery_variant") or ""
                if variant in aggregated_rg:
                    aggregated_rg[variant].append(row)

        total_rg = {v: _summarize_sim_rows(aggregated_rg[v]) for v in RECOVERY_GATE_VARIANTS}
        print_recovery_gate_report(total_rg, scan_date=None)
        print(
            "A=30m/5m@15m  B=30m/1m@1m  C=B+trend2h>"
            f"{args.recovery_min_trend:g} | days={len(dates)} | "
            f"exit={args.exit_type} | entry={args.entry_time.strftime('%H:%M')} ET | "
            f"ipc={args.activate_pct:g}/{args.giveback_pct:g} | "
            f"min_range={args.impulse_min_range_pct:g}%"
        )
        return 0

    if args.range_report:
        if not sim_path.exists():
            print(f"Simulation CSV missing: {sim_path}. Run --simulate first.", file=sys.stderr)
            return 2
        rows = read_cached(sim_path)
        print_range_report(rows)
        return 0

    if args.simulate:
        if not stable_path.exists():
            print(f"Stable CSV missing: {stable_path}. Run --stable first.", file=sys.stderr)
            return 2
        rows = read_cached(stable_path)
        simulated = simulate_pulse(
            rows,
            scan_date=scan_date,
            entry_time=args.entry_time,
            exit_type=args.exit_type,
            tp_pct=args.tp_pct,
            activate_pct=args.activate_pct,
            giveback_pct=args.giveback_pct,
            rebuy_pct=args.rebuy_pct,
            max_tranches=args.max_tranches,
            min_range_pct=args.min_range_pct,
            max_range_pct=args.max_range_pct,
            min_recent_range_pct=args.min_recent_range_pct,
            max_recent_range_pct=args.max_recent_range_pct,
            bucket="recovery",
        )
        write_dict_rows(sim_path, simulated)
        ok_rows = [r for r in simulated if r.get("sim_status") == "OK"]
        exit_rows = [r for r in ok_rows if r.get("sim_exit_reason") != "EOD"]
        avg_ret = (
            sum(float(r.get("sim_return_pct") or 0) for r in ok_rows) / len(ok_rows)
            if ok_rows
            else 0.0
        )
        avg_tranches = (
            sum(int(r.get("sim_tranches") or 0) for r in ok_rows) / len(ok_rows)
            if ok_rows
            else 0.0
        )
        range_filter = ""
        if args.min_range_pct is not None or args.max_range_pct is not None:
            lo = "-inf" if args.min_range_pct is None else f"{args.min_range_pct:g}"
            hi = "+inf" if args.max_range_pct is None else f"{args.max_range_pct:g}"
            range_filter = f", range={lo}..{hi}%"
        recent_range_filter = ""
        if args.min_recent_range_pct is not None or args.max_recent_range_pct is not None:
            lo = "-inf" if args.min_recent_range_pct is None else f"{args.min_recent_range_pct:g}"
            hi = "+inf" if args.max_recent_range_pct is None else f"{args.max_recent_range_pct:g}"
            recent_range_filter = f", recent_range={lo}..{hi}%"
        print(
            f"Wrote Pulse simulation: {sim_path} "
            f"(exit_type={args.exit_type}, ok={len(ok_rows)}, exits={len(exit_rows)}, "
            f"avg_ret={avg_ret:+.2f}%, avg_tranches={avg_tranches:.2f}"
            f"{range_filter}{recent_range_filter})"
        )
        print_preview(simulated, limit=args.preview, simulation=True)
        return 0

    if args.stable:
        if not path.exists():
            print(f"Candidate CSV missing: {path}. Run without --stable first.", file=sys.stderr)
            return 2
        rows = read_cached(path)
        enriched = enrich_stability(
            rows,
            scan_date=scan_date,
            as_of_time=args.entry_time,
            range_lookback_minutes=args.range_lookback_minutes,
        )
        write_dict_rows(stable_path, enriched)
        stable_count = sum(1 for r in enriched if r.get("normal_stable") == "Y")
        print(
            f"Wrote stable Pulse candidates: {stable_path} "
            f"({stable_count}/{len(enriched)} normal stable as of {args.entry_time.strftime('%H:%M')} ET)"
        )
        print_preview(enriched, limit=args.preview, stable=True)
        return 0

    if path.exists() and not args.refresh:
        rows = read_cached(path)
        print(f"Using cached Pulse candidates: {path} ({len(rows)} rows)")
        print_preview(rows, limit=args.preview)
        return 0

    now_et = datetime.now(ET)
    if not args.force and now_et.date() == scan_date and now_et.time() < PULSE_SCAN_TIME_ET:
        print(
            f"Pulse scan is intended for {PULSE_SCAN_TIME_ET.strftime('%H:%M')} ET or later "
            f"(now {now_et.strftime('%H:%M')} ET). Use --force to run anyway.",
            file=sys.stderr,
        )
        return 2

    print(f"Fetching Polygon grouped daily aggs for Pulse scan ({scan_date.isoformat()})...")
    rows = fetch_grouped_daily_rows(
        min_price=args.min_price,
        scan_date=scan_date,
    )
    print(f"Fetched {len(rows):,} grouped daily rows")

    candidates = build_candidates(
        rows,
        scan_date=scan_date,
        as_of_time=args.entry_time,
        min_price=args.min_price,
        min_volume=args.min_volume,
        min_dollar_volume=args.min_dollar_volume,
        min_rvol=args.min_rvol,
        rvol_lookback_days=args.rvol_lookback_days,
        top=args.top,
        seed_universe=args.seed_universe,
        max_price_drift_from_seed=args.max_price_drift_from_seed,
        exclude_etfs=not args.include_etfs,
    )
    write_candidates(path, candidates)
    print(f"Wrote {len(candidates)} Pulse candidates: {path}")
    print_preview(candidates, limit=args.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
