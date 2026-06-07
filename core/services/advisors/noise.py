"""
Noise advisor — v2 entry on a fixed watchlist (A/B vs Flux).

Entry: pullback off 20d high + relative weakness band vs QQQ/SPY; top N per session.
Runs only during regular session (AdvisorBase.market_open).
Exit/add: no stop-loss; unlimited rebuys (PERCENTAGE_REBUY value2=0); TARGET, DESCENDING_TREND, END_DAY, END_WEEK.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, Final, List, Optional, Tuple

import pandas as pd

from core.services.advisors.advisor import AdvisorBase, register
from core.services.financial import yahoo as financial_yahoo

logger = logging.getLogger(__name__)

# Watchlist (standalone copy — not imported from flux.py).
NOISE_UNIVERSE: Final[Tuple[str, ...]] = (
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

ENTRY_LOOKBACK = 20
ENTRY_PULLBACK = Decimal("0.04")
REL_WEAK_MIN = -12.0
REL_WEAK_MAX = -4.0
RET_LOOKBACK = 20
MAX_DISCOVERIES_PER_SESSION = 5

NOISE_TP_MULT = Decimal("1.01")
NOISE_REBUY_DROP = Decimal("0.02")
# PERCENTAGE_REBUY value2: <= 0 means unlimited tranches (capped only by fund cash).
NOISE_REBUY_MAX_TRANCHES = Decimal("0")
NOISE_DISCOVERY_COOLDOWN_HOURS = 48
# Technical entry bump for SO matrix (opp × weight); matches Flux.
NOISE_DISCOVERY_WEIGHT = Decimal("1.15")
NOISE_ENDDAY_TAKE = Decimal("1.01")
NOISE_ENDWEEK_TAKE = Decimal("1.00")
NOISE_DESCENDING_TREND = Decimal("-0.15")


def _benchmark_for(symbol: str) -> str:
    return SYMBOL_BENCHMARK.get(symbol.upper(), DEFAULT_BENCHMARK)


def _ret_over_sessions(close: pd.Series, sessions: int) -> Optional[float]:
    if close.empty or len(close) < sessions + 1:
        return None
    cur = float(close.iloc[-1])
    ref = float(close.iloc[-1 - sessions])
    if ref <= 0 or cur <= 0:
        return None
    return (cur / ref - 1.0) * 100.0


def _pass_pullback(closes: pd.Series, lookback: int, pullback_pct: float) -> Tuple[bool, float, float]:
    if len(closes) < lookback:
        return False, 0.0, 0.0
    window = closes.tail(lookback)
    rh = float(window.max())
    close = float(closes.iloc[-1])
    if rh <= 0 or close <= 0:
        return False, close, 0.0
    pct_below = (1.0 - close / rh) * 100.0
    return close <= rh * (1.0 - pullback_pct), close, pct_below


def _bench_closes(hist: pd.DataFrame) -> pd.Series:
    if hist.empty or "close" not in hist.columns:
        return pd.Series(dtype=float)
    return hist["close"].astype(float)


class Noise(AdvisorBase):
    """Fixed-universe Noise; discovers on entry v2 (pullback + rel weak band, capped per session)."""

    def discover(self, sa) -> None:
        market_status = self.market_open()
        if market_status is None:
            logger.info("Noise skip: market closed")
            return
        if market_status < 0:
            logger.info("Noise skip: market not open yet (%s min to open)", -market_status)
            return

        sell_instructions = [
            ("TARGET_PERCENTAGE", NOISE_TP_MULT, None),
            ("DESCENDING_TREND", NOISE_DESCENDING_TREND, None),
            ("PERCENTAGE_REBUY", NOISE_REBUY_DROP, NOISE_REBUY_MAX_TRANCHES),
            ("END_DAY", NOISE_ENDDAY_TAKE, None),
            ("END_WEEK", NOISE_ENDWEEK_TAKE, None),
        ]

        qqq_hist = financial_yahoo.get_6m_history("QQQ")
        spy_hist = financial_yahoo.get_6m_history("SPY")
        qqq_c = _bench_closes(qqq_hist)
        spy_c = _bench_closes(spy_hist)
        if qqq_c.empty or spy_c.empty:
            logger.warning("Noise skip: missing QQQ or SPY benchmark history")
            return

        pullback_pct = float(ENTRY_PULLBACK)
        candidates: List[Tuple[float, str, float, float, float, float, str]] = []

        for symbol in NOISE_UNIVERSE:
            if not self.allow_discovery(symbol, period=NOISE_DISCOVERY_COOLDOWN_HOURS):
                continue

            stock = self.get_stock(symbol)
            if stock is None:
                continue

            hist = financial_yahoo.get_6m_history(symbol)
            if hist.empty or "close" not in hist.columns:
                logger.debug("Noise: no history for %s", symbol)
                continue

            closes = hist["close"].astype(float)
            ok_pb, close_px, pct_below = _pass_pullback(closes, ENTRY_LOOKBACK, pullback_pct)
            if not ok_pb:
                continue

            bench_sym = _benchmark_for(symbol)
            bench_c = qqq_c if bench_sym == "QQQ" else spy_c
            stock_ret = _ret_over_sessions(closes, RET_LOOKBACK)
            bench_ret = _ret_over_sessions(bench_c, RET_LOOKBACK)
            if stock_ret is None or bench_ret is None:
                continue

            rel = stock_ret - bench_ret
            if rel < REL_WEAK_MIN or rel > REL_WEAK_MAX:
                continue

            candidates.append(
                (rel, symbol, close_px, pct_below, stock_ret, bench_ret, bench_sym)
            )

        candidates.sort(key=lambda x: x[0])
        picks = candidates[:MAX_DISCOVERIES_PER_SESSION]
        discoveries = 0

        for rel, symbol, close_px, pct_below, stock_ret, bench_ret, bench_sym in picks:
            explanation = (
                f"v2 entry | pullback {pct_below:.1f}% off {ENTRY_LOOKBACK}d high | "
                f"rel {rel:+.1f}% vs {bench_sym} (s20 {stock_ret:+.1f}%, b20 {bench_ret:+.1f}%) | "
                f"close ${close_px:.2f}"
            )
            if self.discovered(
                sa,
                symbol,
                explanation,
                sell_instructions=sell_instructions,
                weight=NOISE_DISCOVERY_WEIGHT,
            ):
                discoveries += 1

        logger.info(
            "Noise discover sa=%s: universe=%d candidates=%d picks=%d discoveries=%d",
            sa.id,
            len(NOISE_UNIVERSE),
            len(candidates),
            len(picks),
            discoveries,
        )

    def analyze(self, sa, stock) -> None:
        return


register(name="Noise", python_class="Noise")
