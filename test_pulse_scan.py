#!/usr/bin/env python3
"""
Pulse v0 daily attention scan.

Baby-step goal:
  - Run once per trading day after 11:30 ET.
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
  python test_pulse_scan.py --force   # allow before 11:30 ET
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
PULSE_SCAN_TIME_ET = time(11, 30)
DEFAULT_RANGE_LOOKBACK_MINUTES = 60
DEFAULT_CACHE_DIR = Path(".pulse_scan")
DEFAULT_SEED_UNIVERSE = 500
DEFAULT_MAX_PRICE_DRIFT_FROM_SEED = 0.50
DEFAULT_RVOL_LOOKBACK_DAYS = 20
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
        px_60 = _bar_close_at_or_before(hist, 60, as_of_ts=as_of_ts)

        pct_open = _pct(px_now, open_px)
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

        stable_30 = px_now is not None and px_30 is not None and px_now >= px_30
        stable_60 = px_now is not None and px_60 is not None and px_now >= px_60 * 0.995
        stable_open = px_now is not None and open_px is not None and px_now >= open_px * 0.99
        normal_stable = stable_30 and stable_60 and stable_open

        out.update(
            {
                "price_now": "" if px_now is None else f"{px_now:.4f}",
                "open_now": "" if open_px is None else f"{open_px:.4f}",
                "price_30m_ago": "" if px_30 is None else f"{px_30:.4f}",
                "price_60m_ago": "" if px_60 is None else f"{px_60:.4f}",
                "pct_from_open": "" if pct_open is None else f"{pct_open:.4f}",
                "pct_30m": "" if pct_30 is None else f"{pct_30:.4f}",
                "pct_60m": "" if pct_60 is None else f"{pct_60:.4f}",
                "range_pct_to_asof": "" if range_pct_to_asof is None else f"{range_pct_to_asof:.4f}",
                "recent_range_pct": "" if recent_range_pct is None else f"{recent_range_pct:.4f}",
                "recent_range_minutes": "" if range_lookback_minutes is None else str(range_lookback_minutes),
                "day_range_pct": "" if full_day_range_pct is None else f"{full_day_range_pct:.4f}",
                "stable_30m": _stable_bool(stable_30 if px_30 is not None else None),
                "stable_60m": _stable_bool(stable_60 if px_60 is not None else None),
                "stable_open": _stable_bool(stable_open if open_px is not None else None),
                "normal_stable": _stable_bool(normal_stable),
                "stable_as_of_et": as_of_time.strftime("%H:%M"),
            }
        )
        enriched.append(out)

    return enriched


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
    tp_pct: float,
    rebuy_pct: float,
    max_tranches: int,
    min_range_pct: Optional[float],
    max_range_pct: Optional[float],
    min_recent_range_pct: Optional[float],
    max_recent_range_pct: Optional[float],
) -> list[dict[str, str]]:
    stable_rows = []
    for row in rows:
        if row.get("normal_stable") != "Y":
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
        stable_rows.append(row)

    symbols = [str(r.get("symbol") or "").strip().upper() for r in stable_rows]
    data = _download_intraday(symbols, scan_date=scan_date)
    out: list[dict[str, str]] = []

    for row in stable_rows:
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

        for i in range(entry_i + 1, len(closes)):
            tp_px = avg * (1.0 + tp_pct)
            if float(highs.iloc[i]) >= tp_px:
                exit_px = tp_px
                exit_reason = "TP"
                exit_i = i
                break

            close = float(closes.iloc[i])
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
                "sim_entry_time_et": entry_time.strftime("%H:%M"),
                "sim_entry_price": f"{entry_px:.4f}",
                "sim_exit_price": f"{exit_px:.4f}",
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

    print(f"{'range':>6}  {'n':>3} {'TP':>3} {'tp%':>6} {'avg_ret':>8} {'avg_tr':>7} {'worst':>8}")
    for bucket in buckets:
        sub = grouped[bucket]
        if not sub:
            continue
        tp = sum(1 for r in sub if r.get("sim_exit_reason") == "TP")
        rets = [float(r.get("sim_return_pct") or 0) for r in sub]
        tranches = [int(r.get("sim_tranches") or 0) for r in sub]
        print(
            f"{bucket:>6}  {len(sub):>3} {tp:>3} {tp / len(sub) * 100:>5.1f}% "
            f"{sum(rets) / len(rets):>+7.2f}% {sum(tranches) / len(tranches):>7.2f} "
            f"{min(rets):>+7.2f}%"
        )


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
    parser.add_argument("--force", action="store_true", help="Allow scan before 11:30 ET")
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
    parser.add_argument("--rebuy-pct", type=float, default=0.02, help="Rebuy drop percentage (default 0.02)")
    parser.add_argument("--max-tranches", type=int, default=4, help="Max tranches including initial buy (default 4)")
    parser.add_argument("--min-range-pct", type=float, help="Only simulate names with range_pct_to_asof >= this")
    parser.add_argument("--max-range-pct", type=float, help="Only simulate names with range_pct_to_asof <= this")
    parser.add_argument("--min-recent-range-pct", type=float, help="Only simulate names with recent_range_pct >= this")
    parser.add_argument("--max-recent-range-pct", type=float, help="Only simulate names with recent_range_pct <= this")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scan_date = date.fromisoformat(args.date) if args.date else _today_et()
    path = _cache_path(args.cache_dir, scan_date)
    stable_path = _stable_cache_path(args.cache_dir, scan_date)
    sim_path = _simulation_cache_path(args.cache_dir, scan_date)

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
            tp_pct=args.tp_pct,
            rebuy_pct=args.rebuy_pct,
            max_tranches=args.max_tranches,
            min_range_pct=args.min_range_pct,
            max_range_pct=args.max_range_pct,
            min_recent_range_pct=args.min_recent_range_pct,
            max_recent_range_pct=args.max_recent_range_pct,
        )
        write_dict_rows(sim_path, simulated)
        ok_rows = [r for r in simulated if r.get("sim_status") == "OK"]
        tp_rows = [r for r in ok_rows if r.get("sim_exit_reason") == "TP"]
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
            f"(ok={len(ok_rows)}, TP={len(tp_rows)}, avg_ret={avg_ret:+.2f}%, "
            f"avg_tranches={avg_tranches:.2f}{range_filter}{recent_range_filter})"
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
