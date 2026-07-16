"""
Pulse advisor — daily attention universe + stable intraday entry.

Entry:
- Build a broad liquid-stock seed after 11:00 ET from Polygon grouped daily aggs.
- Rank/filter that seed by today's intraday 15m volume so far.
- Exclude common ETF/fund tickers.
- Keep names with recovery + stability at build time:
  executable quote below 30m ago and above 5m ago (short-term bounce after pullback),
  executable quote >= 60m ago * 0.995, executable quote >= open * 0.99.
- Require recent 60m intraday range >= 1.25%.
- Discover qualifying names between 11:00 and 13:30 ET (MEGA spread is expected for initial test funds).
- Live IMPULSE/COMBO paths (optional): 1m momentum + COMBO (impulse + normal_stable).
- Market tape (SPY/QQQ): refresh only during Pulse buying hours (11:00–13:30 ET);
  persist green/yellow/red on Advisor.blob (default red); skip discovers on red or yellow;
  push superusers on status change.

Exit/add: TARGET_INTRADAY (+0.2% / 0.2% giveback), -2% rebuy (default max tranches; 2h trend + 5m/30m recovery),
END_DAY flat at 3:30 ET (1.00× avg). No END_WEEK, DT, or SL.

Shadow: when enabled, logs IMPULSE/COMBO hits once per cache bucket (same gates as live).
"""
from __future__ import annotations

import logging
from datetime import datetime, time
from decimal import Decimal
from typing import Any, Dict, Final, List, Optional

import pandas as pd
import pytz
import yfinance as yf

from core.services.advisors.advisor import AdvisorBase, register
from core.services.financial import polygon as financial_polygon
from core.services.market.tape import evaluate_tape, fetch_tape
from core.services.push import push_super

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")
PULSE_CANDIDATE_VERSION = 10
PULSE_BUILD_TIME_ET = time(11, 0)
PULSE_DISCOVERY_END_TIME_ET = time(13, 30)
PULSE_SEED_UNIVERSE = 500
PULSE_TOP_DAILY_VOLUME = 100
PULSE_MIN_PRICE = 5.0
PULSE_MIN_SESSION_VOLUME = 500_000
PULSE_MIN_DOLLAR_VOLUME = 25_000_000.0
PULSE_MIN_RANGE_PCT = 1.25
PULSE_RANGE_LOOKBACK_MINUTES = 60
PULSE_RECOVERY_MEDIUM_MINUTES = 30
PULSE_RECOVERY_SHORT_MINUTES = 5
PULSE_CANDIDATE_CACHE_MINUTES = 15
PULSE_MAX_PRICE_DRIFT_FROM_SEED = 0.50
PULSE_MAX_QUOTE_DRIFT_FROM_BAR = 0.02
PULSE_DISCOVERY_COOLDOWN_HOURS = 6

PULSE_TP_MULT = Decimal("1.002")
PULSE_INTRADAY_GIVEBACK = Decimal("0.002")
PULSE_REBUY_DROP = Decimal("0.02")
PULSE_ENDDAY_TAKE = Decimal("1.00")

# Opportunity floor at discover (Oracle uses C at pre-discover gate).
PULSE_MIN_OPPORTUNITY_GRADE = "C"

# Impulse/combo gates (shadow + live discover when enabled below).
PULSE_IMPULSE_SHADOW = True
PULSE_IMPULSE_LIVE = True
PULSE_COMBO_LIVE = True
PULSE_IMPULSE_LOOKBACK_MINUTES = 30
PULSE_IMPULSE_MIN_RET_30M_PCT = 1.0
PULSE_IMPULSE_MIN_VOL_RATIO = 2.0
PULSE_IMPULSE_MIN_CLOSE_POSITION = 0.6
PULSE_IMPULSE_MIN_SIGNALS = 3

# Market tape on Advisor.blob (missing → red). Push titles on status change.
PULSE_TAPE_DEFAULT = "red"
PULSE_TAPE_PUSH_TITLES: Final[Dict[str, str]] = {
    "red": "RED MARKET ALERT",
    "yellow": "YELLOW MARKET WARNING",
    "green": "GREEN MARKET STABLE",
}

ETF_EXCLUDE_TICKERS: Final[frozenset[str]] = frozenset(
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
        "USO",
        "VCIT",
        "VOO",
        "VTI",
        "XBI",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLV",
    }
)


def _today_et() -> str:
    return datetime.now(ET).date().isoformat()


def _cache_bucket_et() -> str:
    now_et = datetime.now(ET)
    bucket_minute = (now_et.minute // PULSE_CANDIDATE_CACHE_MINUTES) * PULSE_CANDIDATE_CACHE_MINUTES
    bucket = now_et.replace(minute=bucket_minute, second=0, microsecond=0)
    return bucket.isoformat(timespec="minutes")


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


def _pct(current: Optional[float], base: Optional[float]) -> Optional[float]:
    if current is None or base is None or base <= 0:
        return None
    return (current / base - 1.0) * 100.0


def _is_price_recovering(
    price_now: Optional[float],
    px_medium: Optional[float],
    px_short: Optional[float],
) -> bool:
    """Pullback vs medium lookback with short-term bounce (not a momentum chase)."""
    return (
        price_now is not None
        and px_medium is not None
        and px_short is not None
        and price_now < px_medium
        and price_now > px_short
    )


def _bar_close_at_or_before(hist: pd.DataFrame, minutes_ago: int) -> Optional[float]:
    if hist.empty or "Close" not in hist.columns:
        return None
    idx = pd.to_datetime(hist.index, utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes_ago)
    eligible = idx <= cutoff
    if not eligible.any():
        return None
    return _safe_float(hist.loc[eligible, "Close"].astype(float).iloc[-1])


def _range_pct_last_minutes(hist: pd.DataFrame, minutes: int) -> Optional[float]:
    if hist.empty or "High" not in hist.columns or "Low" not in hist.columns:
        return None
    idx = pd.to_datetime(hist.index, utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes)
    eligible = idx >= cutoff
    if not eligible.any():
        return None
    recent = hist.loc[eligible]
    hi = _safe_float(recent["High"].max())
    lo = _safe_float(recent["Low"].min())
    if hi is None or lo is None or lo <= 0:
        return None
    return (hi / lo - 1.0) * 100.0


def _volume_so_far(hist: pd.DataFrame) -> int:
    if hist.empty or "Volume" not in hist.columns:
        return 0
    return _safe_int(hist["Volume"].fillna(0).sum())


def _fast_info_price(symbol: str) -> Optional[float]:
    try:
        info = yf.Ticker(symbol).fast_info
        return _safe_float(info.get("lastPrice") or info.get("regularMarketPrice"))
    except Exception as exc:
        logger.debug("Pulse: fast_info unavailable for %s: %s", symbol, exc)
        return None


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


def _download_intraday(symbols: List[str]) -> pd.DataFrame:
    return yf.download(
        symbols,
        period="1d",
        interval="15m",
        auto_adjust=True,
        progress=False,
        threads=True,
    )


def _download_intraday_1m(symbols: List[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    return yf.download(
        symbols,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False,
        threads=True,
    )


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


def _compute_pulse_impulse(hist_1m: pd.DataFrame) -> Dict[str, Any]:
    """1m impulse score for shadow discover (mirrors test_pulse_scan; no profit signal)."""
    empty: Dict[str, Any] = {
        "impulse_score": 0,
        "impulse_signals": "",
        "impulse_ret_30m": None,
        "impulse_vol_ratio": None,
        "impulse_close_pos": None,
        "impulse_pass": False,
    }
    if hist_1m.empty or "Close" not in hist_1m.columns:
        return empty

    idx = pd.to_datetime(hist_1m.index, utc=True)
    check_ts = pd.Timestamp.now(tz="UTC")
    if hist_1m.index.tz is None:
        check_ts = pd.Timestamp(check_ts.replace(tzinfo=None))

    eligible = idx <= check_ts
    if not eligible.any():
        return empty

    hist = hist_1m.loc[eligible]
    idx = pd.to_datetime(hist.index, utc=True)
    price_now = _safe_float(hist["Close"].astype(float).iloc[-1])
    if price_now is None:
        return empty

    lookback = PULSE_IMPULSE_LOOKBACK_MINUTES
    start_30 = check_ts - pd.Timedelta(minutes=lookback)
    prior_start = start_30 - pd.Timedelta(minutes=lookback)
    window = hist[(idx >= start_30) & (idx <= check_ts)]
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

    last_10_start = check_ts - pd.Timedelta(minutes=10)
    prev_10_start = check_ts - pd.Timedelta(minutes=20)
    hi_last10 = _high_in_hist_range(hist, last_10_start, check_ts + pd.Timedelta(minutes=1))
    hi_prev10 = _high_in_hist_range(hist, prev_10_start, last_10_start)

    vol_last = _volume_sum_hist_range(hist, start_30, check_ts)
    vol_prior = _volume_sum_hist_range(hist, prior_start, start_30)
    vol_ratio = None
    if vol_last is not None and vol_prior is not None and vol_prior > 0:
        vol_ratio = vol_last / vol_prior

    signals: List[str] = []
    if ret_30m_pct >= PULSE_IMPULSE_MIN_RET_30M_PCT:
        signals.append("ret30m")
    if hi_last10 is not None and hi_prev10 is not None and hi_last10 > hi_prev10:
        signals.append("hh")
    if vol_ratio is not None and vol_ratio >= PULSE_IMPULSE_MIN_VOL_RATIO:
        signals.append("vol")
    if close_position >= PULSE_IMPULSE_MIN_CLOSE_POSITION:
        signals.append("close")

    score = len(signals)
    return {
        "impulse_score": score,
        "impulse_signals": ",".join(signals),
        "impulse_ret_30m": round(ret_30m_pct, 4),
        "impulse_vol_ratio": None if vol_ratio is None else round(vol_ratio, 4),
        "impulse_close_pos": round(close_position, 4),
        "impulse_pass": score >= PULSE_IMPULSE_MIN_SIGNALS,
    }


def _enrich_attention_impulse(attention_rows: List[Dict[str, Any]], *, log_shadow: bool) -> None:
    """Attach 1m impulse fields to attention rows; optionally log shadow hits."""
    eligible = [
        row
        for row in attention_rows
        if float(row.get("range_pct") or 0) >= PULSE_MIN_RANGE_PCT
    ]
    if not eligible:
        if log_shadow:
            logger.info("pulse_shadow: no attention rows with range>=%s%%", PULSE_MIN_RANGE_PCT)
        return

    symbols = [str(row["symbol"]) for row in eligible]
    data_1m = _download_intraday_1m(symbols)
    impulse_hits = 0
    combo_hits = 0

    for row in eligible:
        symbol = str(row["symbol"])
        hist_1m = _symbol_hist(data_1m, symbol)
        impulse = _compute_pulse_impulse(hist_1m)
        row.update(impulse)

        if not impulse.get("impulse_pass"):
            continue

        normal_stable = bool(row.get("normal_stable"))
        if log_shadow:
            impulse_hits += 1
            logger.info(
                "pulse_shadow path=IMPULSE symbol=%s rank=%s score=%s signals=%s ret_30m=%s "
                "vol_ratio=%s close_pos=%s range_pct=%s normal_stable=%s",
                symbol,
                row.get("rank"),
                impulse.get("impulse_score"),
                impulse.get("impulse_signals"),
                impulse.get("impulse_ret_30m"),
                impulse.get("impulse_vol_ratio"),
                impulse.get("impulse_close_pos"),
                row.get("range_pct"),
                normal_stable,
            )
            if normal_stable:
                combo_hits += 1
                logger.info(
                    "pulse_shadow path=COMBO symbol=%s rank=%s score=%s signals=%s ret_30m=%s "
                    "vol_ratio=%s close_pos=%s range_pct=%s",
                    symbol,
                    row.get("rank"),
                    impulse.get("impulse_score"),
                    impulse.get("impulse_signals"),
                    impulse.get("impulse_ret_30m"),
                    impulse.get("impulse_vol_ratio"),
                    impulse.get("impulse_close_pos"),
                    row.get("range_pct"),
                )

    if log_shadow:
        logger.info(
            "pulse_shadow summary eligible=%d impulse=%d combo=%d ret30m_min=%s min_signals=%s",
            len(eligible),
            impulse_hits,
            combo_hits,
            PULSE_IMPULSE_MIN_RET_30M_PCT,
            PULSE_IMPULSE_MIN_SIGNALS,
        )


def _attention_row_to_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symbol": row["symbol"],
        "rank": row.get("rank"),
        "price": row.get("price"),
        "bar_price": row.get("bar_price"),
        "volume": row.get("volume"),
        "dollar_volume": row.get("dollar_volume"),
        "prior_dollar_volume": row.get("prior_dollar_volume"),
        "range_pct": row.get("range_pct"),
        "pct_5m": row.get("pct_5m"),
        "pct_30m": row.get("pct_30m"),
        "pct_60m": row.get("pct_60m"),
        "pct_open": row.get("pct_open"),
    }


def _pulse_impulse_sentence(row: Dict[str, Any]) -> str:
    ret30m = row.get("impulse_ret_30m")
    ret_display = _format_signed_pct(float(ret30m) if ret30m is not None else None)
    return (
        f"Impulse {row.get('impulse_score')}/4 ({row.get('impulse_signals')}); "
        f"ret30m {ret_display}"
    )


def build_pulse_combo_discovery_explanation(candidate: Dict[str, Any], row: Dict[str, Any]) -> str:
    segments = [
        f"COMBO: {_pulse_impulse_sentence(row)}",
        _pulse_explanation_lead(candidate),
        _pulse_rank_sentence(candidate),
        _pulse_stability_sentence(candidate),
        _pulse_price_sentence(candidate),
        _pulse_volume_sentence(candidate),
    ]
    return " | ".join(segments)


def build_pulse_impulse_discovery_explanation(candidate: Dict[str, Any], row: Dict[str, Any]) -> str:
    range_pct = float(candidate.get("range_pct") or 0)
    segments = [
        f"IMPULSE: {_pulse_impulse_sentence(row)}",
        _pulse_rank_sentence(candidate),
        f"{range_pct:.1f}% intraday range",
        _pulse_price_sentence(candidate),
        _pulse_volume_sentence(candidate),
    ]
    return " | ".join(segments)


def _format_signed_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.1f}%"


def _pulse_explanation_lead(candidate: Dict[str, Any]) -> str:
    """Short holdings summary: recovery move + intraday range (not volume)."""
    range_pct = float(candidate.get("range_pct") or 0)
    pct_5 = candidate.get("pct_5m")
    pct_30 = candidate.get("pct_30m")

    if pct_5 is not None and pct_30 is not None:
        return (
            f"{_format_signed_pct(pct_5)} last 5m, "
            f"{_format_signed_pct(pct_30)} vs 30m, {range_pct:.1f}% range"
        )
    if pct_5 is not None:
        return f"{_format_signed_pct(pct_5)} last 5m, {range_pct:.1f}% range"
    if pct_30 is not None:
        return f"{_format_signed_pct(pct_30)} vs 30m, {range_pct:.1f}% range"
    return f"{range_pct:.1f}% intraday range"


def _pulse_stability_sentence(candidate: Dict[str, Any]) -> str:
    return "Pullback vs 30m, bouncing vs 5m; OK vs 60m and open"


def _pulse_rank_sentence(candidate: Dict[str, Any]) -> str:
    rank = candidate.get("rank")
    if rank is None:
        return "High session dollar volume"
    return f"Ranked #{int(rank)} for session dollar volume"


def _pulse_price_sentence(candidate: Dict[str, Any]) -> str:
    price = float(candidate.get("price") or 0)
    bar = float(candidate.get("bar_price") or 0)
    return f"Live ${price:.2f} (15m close ${bar:.2f})"


def _pulse_volume_sentence(candidate: Dict[str, Any]) -> str:
    vol = int(candidate.get("volume") or 0)
    dvol = float(candidate.get("dollar_volume") or 0)
    return f"~{vol:,} shares traded today (~${dvol:,.0f})"


def build_pulse_discovery_explanation(candidate: Dict[str, Any]) -> str:
    segments = [
        _pulse_explanation_lead(candidate),
        _pulse_rank_sentence(candidate),
        _pulse_stability_sentence(candidate),
        _pulse_price_sentence(candidate),
        _pulse_volume_sentence(candidate),
    ]
    return " | ".join(segments)


def _pulse_opportunity_gate(symbol: str) -> tuple[bool, str]:
    """
    Live SO check before discover (same scorers as Assessment; Oracle pattern).
    Returns (passes, detail) for logging.
    """
    from core.services.health.assess import run_component_results
    from core.services.health.risk_matrix import compute_so_snapshot
    from core.services.health.so_ratings import (
        opportunity_grade_at_least,
        score_to_opportunity_grade,
        score_to_stability_grade,
    )

    results = run_component_results(symbol)
    so = compute_so_snapshot(symbol, results)
    opp_score = so.get("opportunity")
    stab_score = so.get("stability")
    opp_grade = score_to_opportunity_grade(opp_score)
    stab_grade = score_to_stability_grade(stab_score)
    opp_letter = opp_grade.letter if opp_grade else "n/a"
    stab_letter = stab_grade.letter if stab_grade else "n/a"
    opp_display = round(float(opp_score), 1) if opp_score is not None else None

    detail = (
        f"opp={opp_letter} min={PULSE_MIN_OPPORTUNITY_GRADE} "
        f"stab={stab_letter} opportunity_score={opp_display}"
    )
    if opp_grade is None:
        return False, detail
    if not opportunity_grade_at_least(opp_grade.letter, PULSE_MIN_OPPORTUNITY_GRADE):
        return False, detail
    return True, detail


class Pulse(AdvisorBase):
    """Daily attention universe with stable intraday entry."""

    def _before_build_time(self) -> bool:
        now_et = datetime.now(ET)
        return now_et.time() < PULSE_BUILD_TIME_ET

    def _after_discovery_window(self) -> bool:
        now_et = datetime.now(ET)
        return now_et.time() >= PULSE_DISCOVERY_END_TIME_ET

    def _refresh_market_tape(self) -> str:
        """
        Fetch SPY/QQQ tape, persist status on Advisor.blob, push on change.

        Returns green | yellow | red. Missing/failed readings → red.
        """
        state = self._advisor_blob_state()
        prev = str(state.get("tape_status") or PULSE_TAPE_DEFAULT).strip().lower()
        if prev not in PULSE_TAPE_PUSH_TITLES:
            prev = PULSE_TAPE_DEFAULT

        try:
            readings = fetch_tape()
            verdict = evaluate_tape(readings)
            new_status = verdict.state
            reason = verdict.reason
            if not readings or new_status not in PULSE_TAPE_PUSH_TITLES:
                new_status = PULSE_TAPE_DEFAULT
                reason = reason or "no benchmark readings"
        except Exception as exc:
            logger.warning("Pulse tape fetch failed: %s", exc)
            new_status = PULSE_TAPE_DEFAULT
            reason = f"tape fetch failed: {exc.__class__.__name__}"

        now_et = datetime.now(ET)
        state["tape_status"] = new_status
        state["tape_reason"] = reason
        state["tape_updated_at_et"] = now_et.isoformat(timespec="seconds")
        state["tape_date"] = now_et.date().isoformat()
        self._save_advisor_blob_state(state)

        if new_status != prev:
            title = PULSE_TAPE_PUSH_TITLES[new_status]
            body = f"Was {prev.upper()}: {reason}"
            try:
                push_super(body, title=title)
                logger.info(
                    "pulse_tape state=%s prev=%s push=sent reason=%s",
                    new_status.upper(),
                    prev.upper(),
                    reason,
                )
            except Exception:
                logger.exception(
                    "pulse_tape state=%s prev=%s push=failed reason=%s",
                    new_status.upper(),
                    prev.upper(),
                    reason,
                )
        else:
            logger.info(
                "pulse_tape state=%s unchanged reason=%s",
                new_status.upper(),
                reason,
            )
        return new_status

    def _build_daily_candidates(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        df = financial_polygon.get_filtered_stocks(
            min_price=PULSE_MIN_PRICE,
        )
        if df is None or df.empty:
            logger.warning("Pulse: Polygon grouped daily universe empty")
            return [], []

        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            symbol = str(row.get("ticker") or "").strip().upper()
            if not symbol or "." in symbol or symbol in ETF_EXCLUDE_TICKERS:
                continue
            price = _safe_float(row.get("price"))
            volume = _safe_int(row.get("today_volume"))
            if price is None or volume <= 0:
                continue
            prior_dollar_volume = price * volume
            rows.append(
                {
                    "symbol": symbol,
                    "polygon_price": price,
                    "prior_volume": volume,
                    "prior_dollar_volume": prior_dollar_volume,
                }
            )

        rows.sort(key=lambda r: float(r["prior_dollar_volume"]), reverse=True)
        shortlist = rows[:PULSE_SEED_UNIVERSE]
        symbols = [str(r["symbol"]) for r in shortlist]
        intraday = _download_intraday(symbols)

        intraday_ranked: List[Dict[str, Any]] = []
        for row in shortlist:
            symbol = str(row["symbol"])
            hist = _symbol_hist(intraday, symbol)
            if hist.empty or "Close" not in hist.columns or "Open" not in hist.columns:
                continue

            bar_price = _safe_float(hist["Close"].iloc[-1])
            seed_price = _safe_float(row.get("polygon_price"))
            if (
                bar_price is not None
                and seed_price is not None
                and abs(bar_price / seed_price - 1.0) > PULSE_MAX_PRICE_DRIFT_FROM_SEED
            ):
                logger.debug(
                    "Pulse: skip %s - intraday price %.2f too far from seed %.2f",
                    symbol,
                    bar_price,
                    seed_price,
                )
                continue
            volume_so_far = _volume_so_far(hist)
            if bar_price is None or volume_so_far < PULSE_MIN_SESSION_VOLUME:
                continue
            dollar_volume_so_far = bar_price * volume_so_far
            if dollar_volume_so_far < PULSE_MIN_DOLLAR_VOLUME:
                continue

            intraday_ranked.append(
                {
                    **row,
                    "hist": hist,
                    "bar_price": bar_price,
                    "volume_so_far": volume_so_far,
                    "dollar_volume_so_far": dollar_volume_so_far,
                }
            )

        intraday_ranked.sort(key=lambda r: float(r["dollar_volume_so_far"]), reverse=True)

        attention: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []
        for rank, row in enumerate(intraday_ranked[:PULSE_TOP_DAILY_VOLUME], start=1):
            symbol = str(row["symbol"])
            hist = row["hist"]
            bar_price = float(row["bar_price"])
            quote_price = _fast_info_price(symbol)
            if quote_price is None:
                continue
            if abs(quote_price / bar_price - 1.0) > PULSE_MAX_QUOTE_DRIFT_FROM_BAR:
                logger.debug(
                    "Pulse: skip %s - quote %.2f too far from 15m bar %.2f",
                    symbol,
                    quote_price,
                    bar_price,
                )
                continue
            price_now = quote_price
            open_px = _safe_float(hist["Open"].iloc[0]) if "Open" in hist.columns else None
            px_30 = _bar_close_at_or_before(hist, PULSE_RECOVERY_MEDIUM_MINUTES)
            px_5 = _bar_close_at_or_before(hist, PULSE_RECOVERY_SHORT_MINUTES)
            px_60 = _bar_close_at_or_before(hist, 60)
            range_pct = _range_pct_last_minutes(hist, PULSE_RANGE_LOOKBACK_MINUTES)

            recovering = _is_price_recovering(price_now, px_30, px_5)
            stable_60 = price_now is not None and px_60 is not None and price_now >= px_60 * 0.995
            stable_open = price_now is not None and open_px is not None and price_now >= open_px * 0.99
            normal_stable = recovering and stable_60 and stable_open

            attention_row = {
                "symbol": symbol,
                "rank": rank,
                "price": round(float(price_now), 4),
                "bar_price": round(float(bar_price), 4),
                "volume": int(row["volume_so_far"]),
                "dollar_volume": round(float(row["dollar_volume_so_far"]), 2),
                "prior_dollar_volume": round(float(row["prior_dollar_volume"]), 2),
                "range_pct": None if range_pct is None else round(float(range_pct), 4),
                "pct_5m": None if px_5 is None else round(float(_pct(price_now, px_5) or 0), 4),
                "pct_30m": None if px_30 is None else round(float(_pct(price_now, px_30) or 0), 4),
                "pct_60m": None if px_60 is None else round(float(_pct(price_now, px_60) or 0), 4),
                "pct_open": None if open_px is None else round(float(_pct(price_now, open_px) or 0), 4),
                "recovering": recovering,
                "stable_60": stable_60,
                "stable_open": stable_open,
                "normal_stable": normal_stable,
            }
            attention.append(attention_row)

            if not normal_stable:
                continue
            if range_pct is None or range_pct < PULSE_MIN_RANGE_PCT:
                continue

            candidates.append(
                {
                    "symbol": symbol,
                    "rank": rank,
                    "price": attention_row["price"],
                    "bar_price": attention_row["bar_price"],
                    "volume": attention_row["volume"],
                    "dollar_volume": attention_row["dollar_volume"],
                    "prior_dollar_volume": attention_row["prior_dollar_volume"],
                    "range_pct": attention_row["range_pct"],
                    "pct_5m": attention_row["pct_5m"],
                    "pct_30m": attention_row["pct_30m"],
                    "pct_60m": attention_row["pct_60m"],
                    "pct_open": attention_row["pct_open"],
                }
            )

        if PULSE_IMPULSE_SHADOW or PULSE_IMPULSE_LIVE or PULSE_COMBO_LIVE:
            _enrich_attention_impulse(
                attention,
                log_shadow=PULSE_IMPULSE_SHADOW,
            )

        candidates.sort(key=lambda r: float(r["range_pct"]), reverse=True)
        return attention, candidates

    def _daily_candidates(self) -> List[Dict[str, Any]]:
        _attention, candidates = self._ensure_pulse_cache()
        return candidates

    def _ensure_pulse_cache(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        today = _today_et()
        bucket = _cache_bucket_et()
        state = self._advisor_blob_state()
        if (
            state.get("pulse_date") == today
            and state.get("pulse_version") == PULSE_CANDIDATE_VERSION
            and state.get("pulse_bucket_et") == bucket
            and isinstance(state.get("candidates"), list)
        ):
            return state.get("attention") or [], state["candidates"]

        attention, candidates = self._build_daily_candidates()
        state["pulse_date"] = today
        state["pulse_version"] = PULSE_CANDIDATE_VERSION
        state["pulse_bucket_et"] = bucket
        state["attention"] = attention
        state["candidates"] = candidates
        state["built_at_et"] = datetime.now(ET).isoformat(timespec="seconds")
        self._save_advisor_blob_state(state)
        return attention, candidates

    def _discover_pulse_candidate(
        self,
        sa,
        symbol: str,
        explanation: str,
        sell_instructions,
        discovered_symbols: set[str],
        *,
        path: str,
    ) -> str:
        """Returns discovered | cooldown | opp | dup | failed."""
        if symbol in discovered_symbols:
            return "dup"
        if not self.allow_discovery(symbol, period=PULSE_DISCOVERY_COOLDOWN_HOURS):
            return "cooldown"

        opp_ok, opp_detail = _pulse_opportunity_gate(symbol)
        if not opp_ok:
            logger.info("pulse_opp_gate block discover %s path=%s %s", symbol, path, opp_detail)
            return "opp"

        if self.discovered(
            sa,
            symbol,
            explanation,
            sell_instructions=sell_instructions,
            weight=1.0,
        ):
            discovered_symbols.add(symbol)
            logger.info("pulse_discover path=%s symbol=%s", path, symbol)
            return "discovered"
        return "failed"

    def discover(self, sa) -> None:
        market_status = self.market_open()
        if market_status is None:
            logger.info("Pulse skip: market closed")
            return
        if market_status < 0:
            logger.info("Pulse skip: market not open yet (%s min to open)", -market_status)
            return
        if self._before_build_time():
            logger.info("Pulse skip: before %s ET build time", PULSE_BUILD_TIME_ET.strftime("%H:%M"))
            return
        if self._after_discovery_window():
            logger.info(
                "Pulse skip: after %s ET discovery window",
                PULSE_DISCOVERY_END_TIME_ET.strftime("%H:%M"),
            )
            return

        tape_status = self._refresh_market_tape()
        if tape_status != "green":
            logger.info("Pulse skip: market tape not GREEN (discover paused)")
            return

        sell_instructions = [
            ("TARGET_INTRADAY", PULSE_TP_MULT, PULSE_INTRADAY_GIVEBACK),
            ("PERCENTAGE_REBUY", PULSE_REBUY_DROP, None),
            ("END_DAY", PULSE_ENDDAY_TAKE, None),
        ]

        attention, recovery_candidates = self._ensure_pulse_cache()
        discovered_symbols: set[str] = set()
        skipped_cooldown = 0
        skipped_opp = 0
        discoveries_combo = 0
        discoveries_recovery = 0
        discoveries_impulse = 0

        if PULSE_COMBO_LIVE:
            for row in attention:
                if not row.get("impulse_pass") or not row.get("normal_stable"):
                    continue
                symbol = str(row.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                candidate = _attention_row_to_candidate(row)
                explanation = build_pulse_combo_discovery_explanation(candidate, row)
                outcome = self._discover_pulse_candidate(
                    sa,
                    symbol,
                    explanation,
                    sell_instructions,
                    discovered_symbols,
                    path="COMBO",
                )
                if outcome == "discovered":
                    discoveries_combo += 1
                elif outcome == "cooldown":
                    skipped_cooldown += 1
                elif outcome == "opp":
                    skipped_opp += 1

        for candidate in recovery_candidates:
            symbol = str(candidate.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            explanation = build_pulse_discovery_explanation(candidate)
            outcome = self._discover_pulse_candidate(
                sa,
                symbol,
                explanation,
                sell_instructions,
                discovered_symbols,
                path="RECOVERY",
            )
            if outcome == "discovered":
                discoveries_recovery += 1
            elif outcome == "cooldown":
                skipped_cooldown += 1
            elif outcome == "opp":
                skipped_opp += 1

        if PULSE_IMPULSE_LIVE:
            for row in attention:
                if not row.get("impulse_pass") or row.get("normal_stable"):
                    continue
                symbol = str(row.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                candidate = _attention_row_to_candidate(row)
                explanation = build_pulse_impulse_discovery_explanation(candidate, row)
                outcome = self._discover_pulse_candidate(
                    sa,
                    symbol,
                    explanation,
                    sell_instructions,
                    discovered_symbols,
                    path="IMPULSE",
                )
                if outcome == "discovered":
                    discoveries_impulse += 1
                elif outcome == "cooldown":
                    skipped_cooldown += 1
                elif outcome == "opp":
                    skipped_opp += 1

        discoveries = discoveries_combo + discoveries_recovery + discoveries_impulse
        logger.info(
            "Pulse discover sa=%s: recovery_cands=%d attention=%d "
            "discoveries=%d (combo=%d recovery=%d impulse=%d) "
            "skipped_cooldown=%d skipped_opp=%d",
            sa.id,
            len(recovery_candidates),
            len(attention),
            discoveries,
            discoveries_combo,
            discoveries_recovery,
            discoveries_impulse,
            skipped_cooldown,
            skipped_opp,
        )

    def analyze(self, sa, stock) -> None:
        return


register(name="Pulse", python_class="Pulse")
