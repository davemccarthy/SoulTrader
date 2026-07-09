"""
Pulse advisor — daily attention universe + stable intraday entry.

Entry:
- Build a broad liquid-stock seed after 11:00 ET from Polygon grouped daily aggs.
- Rank/filter that seed by today's intraday 15m volume so far.
- Exclude common ETF/fund tickers.
- Keep names with recovery + stability at build time:
  executable quote below 30m ago and above 5m ago (short-term bounce after pullback),
  executable quote >= 60m ago * 0.995, executable quote >= open * 0.99.
- Require recent 60m intraday range >= 1.3%.
- Discover qualifying names between 11:00 and 13:30 ET (MEGA spread is expected for initial test funds).

Exit/add: TARGET_INTRADAY (+0.2% / 0.2% giveback), -2% stabilized rebuy (max 3 tranches),
END_DAY flat at 3:30 ET (1.00× avg). No END_WEEK, DT, or SL.
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

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")
PULSE_CANDIDATE_VERSION = 7
PULSE_BUILD_TIME_ET = time(11, 0)
PULSE_DISCOVERY_END_TIME_ET = time(13, 30)
PULSE_SEED_UNIVERSE = 500
PULSE_TOP_DAILY_VOLUME = 100
PULSE_MIN_PRICE = 5.0
PULSE_MIN_SESSION_VOLUME = 500_000
PULSE_MIN_DOLLAR_VOLUME = 25_000_000.0
PULSE_MIN_RANGE_PCT = 1.3
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
PULSE_REBUY_MAX_TRANCHES = Decimal("3")
PULSE_ENDDAY_TAKE = Decimal("1.00")

# Opportunity floor at discover (Oracle uses C at pre-discover gate).
PULSE_MIN_OPPORTUNITY_GRADE = "C"

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
            f"Recovering {_format_signed_pct(pct_5)} last 5m, "
            f"{_format_signed_pct(pct_30)} vs 30m, {range_pct:.1f}% range"
        )
    if pct_5 is not None:
        return f"Recovering {_format_signed_pct(pct_5)} last 5m, {range_pct:.1f}% range"
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

    def _build_daily_candidates(self) -> List[Dict[str, Any]]:
        df = financial_polygon.get_filtered_stocks(
            min_price=PULSE_MIN_PRICE,
        )
        if df is None or df.empty:
            logger.warning("Pulse: Polygon grouped daily universe empty")
            return []

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
            if not (recovering and stable_60 and stable_open):
                continue
            if range_pct is None or range_pct < PULSE_MIN_RANGE_PCT:
                continue

            candidates.append(
                {
                    "symbol": symbol,
                    "rank": rank,
                    "price": round(float(price_now), 4),
                    "bar_price": round(float(bar_price), 4),
                    "volume": int(row["volume_so_far"]),
                    "dollar_volume": round(float(row["dollar_volume_so_far"]), 2),
                    "prior_dollar_volume": round(float(row["prior_dollar_volume"]), 2),
                    "range_pct": round(float(range_pct), 4),
                    "pct_5m": None if px_5 is None else round(float(_pct(price_now, px_5) or 0), 4),
                    "pct_30m": None if px_30 is None else round(float(_pct(price_now, px_30) or 0), 4),
                    "pct_60m": None if px_60 is None else round(float(_pct(price_now, px_60) or 0), 4),
                    "pct_open": None if open_px is None else round(float(_pct(price_now, open_px) or 0), 4),
                }
            )

        candidates.sort(key=lambda r: float(r["range_pct"]), reverse=True)
        return candidates

    def _daily_candidates(self) -> List[Dict[str, Any]]:
        today = _today_et()
        bucket = _cache_bucket_et()
        state = self._advisor_blob_state()
        if (
            state.get("pulse_date") == today
            and state.get("pulse_version") == PULSE_CANDIDATE_VERSION
            and state.get("pulse_bucket_et") == bucket
            and isinstance(state.get("candidates"), list)
        ):
            return state["candidates"]

        candidates = self._build_daily_candidates()
        state["pulse_date"] = today
        state["pulse_version"] = PULSE_CANDIDATE_VERSION
        state["pulse_bucket_et"] = bucket
        state["candidates"] = candidates
        state["built_at_et"] = datetime.now(ET).isoformat(timespec="seconds")
        self._save_advisor_blob_state(state)
        return candidates

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

        sell_instructions = [
            ("TARGET_INTRADAY", PULSE_TP_MULT, PULSE_INTRADAY_GIVEBACK),
            ("PERCENTAGE_REBUY", PULSE_REBUY_DROP, PULSE_REBUY_MAX_TRANCHES),
            ("END_DAY", PULSE_ENDDAY_TAKE, None),
        ]

        candidates = self._daily_candidates()
        discoveries = 0
        skipped_cooldown = 0
        skipped_opp = 0

        for candidate in candidates:
            symbol = str(candidate.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            if not self.allow_discovery(symbol, period=PULSE_DISCOVERY_COOLDOWN_HOURS):
                skipped_cooldown += 1
                continue

            opp_ok, opp_detail = _pulse_opportunity_gate(symbol)
            if not opp_ok:
                logger.info("pulse_opp_gate block discover %s %s", symbol, opp_detail)
                skipped_opp += 1
                continue

            explanation = build_pulse_discovery_explanation(candidate)
            if self.discovered(
                sa,
                symbol,
                explanation,
                sell_instructions=sell_instructions,
                weight=1.0,
            ):
                discoveries += 1

        logger.info(
            "Pulse discover sa=%s: candidates=%d skipped_cooldown=%d skipped_opp=%d discoveries=%d",
            sa.id,
            len(candidates),
            skipped_cooldown,
            skipped_opp,
            discoveries,
        )

    def analyze(self, sa, stock) -> None:
        return


register(name="Pulse", python_class="Pulse")
