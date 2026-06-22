"""
Pulse advisor — daily attention universe + stable intraday entry.

Entry:
- Build a daily top-volume shortlist after 11:00 ET from Polygon grouped daily aggs.
- Exclude common ETF/fund tickers.
- Keep names with normal stability at build time:
  current >= 30m ago, current >= 60m ago * 0.995, current >= open * 0.99.
- Require intraday range so far >= 2%.
- Discover all qualifying names (MEGA spread is expected for initial test funds).

Exit/add: Noise-style +1% target, -2% stabilized rebuy, END_DAY, END_WEEK. No DT/SL.
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
PULSE_BUILD_TIME_ET = time(11, 0)
PULSE_DISCOVERY_END_TIME_ET = time(14, 30)
PULSE_TOP_DAILY_VOLUME = 100
PULSE_MIN_PRICE = 5.0
PULSE_MIN_SESSION_VOLUME = 500_000
PULSE_MIN_DOLLAR_VOLUME = 25_000_000.0
PULSE_MIN_RANGE_PCT = 2.0
PULSE_DISCOVERY_COOLDOWN_HOURS = 24

PULSE_TP_MULT = Decimal("1.01")
PULSE_REBUY_DROP = Decimal("0.02")
# value2 <= 0 means unlimited rebuys, capped only by fund cash.
PULSE_REBUY_MAX_TRANCHES = Decimal("0")
PULSE_ENDDAY_TAKE = Decimal("1.01")
PULSE_ENDWEEK_TAKE = Decimal("1.00")

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
        "QQQ",
        "RSP",
        "SDS",
        "SGOV",
        "SH",
        "SMH",
        "SOXL",
        "SOXS",
        "SOXX",
        "SPXL",
        "SPXS",
        "SPY",
        "SQQQ",
        "TLT",
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


def _bar_close_at_or_before(hist: pd.DataFrame, minutes_ago: int) -> Optional[float]:
    if hist.empty or "Close" not in hist.columns:
        return None
    idx = pd.to_datetime(hist.index, utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes_ago)
    eligible = idx <= cutoff
    if not eligible.any():
        return None
    return _safe_float(hist.loc[eligible, "Close"].astype(float).iloc[-1])


def _range_pct_so_far(hist: pd.DataFrame) -> Optional[float]:
    if hist.empty or "High" not in hist.columns or "Low" not in hist.columns:
        return None
    hi = _safe_float(hist["High"].max())
    lo = _safe_float(hist["Low"].min())
    if hi is None or lo is None or lo <= 0:
        return None
    return (hi / lo - 1.0) * 100.0


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
            min_volume=PULSE_MIN_SESSION_VOLUME,
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
            dollar_volume = price * volume
            if dollar_volume < PULSE_MIN_DOLLAR_VOLUME:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "polygon_price": price,
                    "volume": volume,
                    "dollar_volume": dollar_volume,
                }
            )

        rows.sort(key=lambda r: float(r["dollar_volume"]), reverse=True)
        shortlist = rows[:PULSE_TOP_DAILY_VOLUME]
        symbols = [str(r["symbol"]) for r in shortlist]
        intraday = _download_intraday(symbols)

        candidates: List[Dict[str, Any]] = []
        for rank, row in enumerate(shortlist, start=1):
            symbol = str(row["symbol"])
            hist = _symbol_hist(intraday, symbol)
            if hist.empty or "Close" not in hist.columns:
                continue

            price_now = _safe_float(hist["Close"].iloc[-1])
            open_px = _safe_float(hist["Open"].iloc[0]) if "Open" in hist.columns else None
            px_30 = _bar_close_at_or_before(hist, 30)
            px_60 = _bar_close_at_or_before(hist, 60)
            range_pct = _range_pct_so_far(hist)

            stable_30 = price_now is not None and px_30 is not None and price_now >= px_30
            stable_60 = price_now is not None and px_60 is not None and price_now >= px_60 * 0.995
            stable_open = price_now is not None and open_px is not None and price_now >= open_px * 0.99
            if not (stable_30 and stable_60 and stable_open):
                continue
            if range_pct is None or range_pct < PULSE_MIN_RANGE_PCT:
                continue

            candidates.append(
                {
                    "symbol": symbol,
                    "rank": rank,
                    "price": round(float(price_now), 4),
                    "dollar_volume": round(float(row["dollar_volume"]), 2),
                    "range_pct": round(float(range_pct), 4),
                    "pct_30m": None if px_30 is None else round(float(_pct(price_now, px_30) or 0), 4),
                    "pct_60m": None if px_60 is None else round(float(_pct(price_now, px_60) or 0), 4),
                    "pct_open": None if open_px is None else round(float(_pct(price_now, open_px) or 0), 4),
                }
            )

        candidates.sort(key=lambda r: float(r["range_pct"]), reverse=True)
        return candidates

    def _daily_candidates(self) -> List[Dict[str, Any]]:
        today = _today_et()
        state = self._advisor_blob_state()
        if state.get("pulse_date") == today and isinstance(state.get("candidates"), list):
            return state["candidates"]

        candidates = self._build_daily_candidates()
        state["pulse_date"] = today
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
            ("TARGET_PERCENTAGE", PULSE_TP_MULT, None),
            ("PERCENTAGE_REBUY", PULSE_REBUY_DROP, PULSE_REBUY_MAX_TRANCHES),
            ("END_DAY", PULSE_ENDDAY_TAKE, None),
            ("END_WEEK", PULSE_ENDWEEK_TAKE, None),
        ]

        candidates = self._daily_candidates()
        discoveries = 0
        skipped_cooldown = 0

        for candidate in candidates:
            symbol = str(candidate.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            if not self.allow_discovery(symbol, period=PULSE_DISCOVERY_COOLDOWN_HOURS):
                skipped_cooldown += 1
                continue

            explanation = (
                f"Pulse entry | rank {candidate.get('rank')} | "
                f"range {float(candidate.get('range_pct') or 0):.1f}% | "
                f"30m {float(candidate.get('pct_30m') or 0):+.1f}% | "
                f"60m {float(candidate.get('pct_60m') or 0):+.1f}% | "
                f"open {float(candidate.get('pct_open') or 0):+.1f}% | "
                f"dvol ${float(candidate.get('dollar_volume') or 0):,.0f}"
            )
            if self.discovered(
                sa,
                symbol,
                explanation,
                sell_instructions=sell_instructions,
                weight=1.0,
            ):
                discoveries += 1

        logger.info(
            "Pulse discover sa=%s: candidates=%d skipped_cooldown=%d discoveries=%d",
            sa.id,
            len(candidates),
            skipped_cooldown,
            discoveries,
        )

    def analyze(self, sa, stock) -> None:
        return


register(name="Pulse", python_class="Pulse")
