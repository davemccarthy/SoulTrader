"""
Intraday market tape: benchmark move vs today's open and prior close.

Generic service for manual ops / advisor discover gates (no session-history rules).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARKS: tuple[str, ...] = ("SPY", "QQQ")

# Suggested thresholds for new-entry caution (manual or future auto-gate).
TAPE_RED_VS_OPEN_PCT = -1.5
TAPE_YELLOW_VS_OPEN_PCT = -1.0
TAPE_RED_VS_PRIOR_CLOSE_PCT = -2.0
TAPE_YELLOW_VS_PRIOR_CLOSE_PCT = -1.0


@dataclass(frozen=True)
class TapeReading:
    symbol: str
    price: Optional[float]
    open_px: Optional[float]
    prior_close: Optional[float]
    vs_open_pct: Optional[float]
    vs_prior_close_pct: Optional[float]

    def vs_open_display(self) -> str:
        if self.vs_open_pct is None:
            return "n/a"
        return f"{self.vs_open_pct:+.2f}%"

    def vs_prior_close_display(self) -> str:
        if self.vs_prior_close_pct is None:
            return "n/a"
        return f"{self.vs_prior_close_pct:+.2f}%"


@dataclass(frozen=True)
class TapeVerdict:
    state: str  # green | yellow | red
    reason: str
    readings: Dict[str, TapeReading]


def _pct(current: Optional[float], base: Optional[float]) -> Optional[float]:
    if current is None or base is None or base <= 0:
        return None
    return round((current / base - 1.0) * 100.0, 3)


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if out <= 0:
            return None
        return out
    except (TypeError, ValueError):
        return None


def _reading_from_fast_info(symbol: str) -> TapeReading:
    import yfinance as yf

    sym = symbol.strip().upper()
    price = open_px = prior_close = None
    try:
        info = yf.Ticker(sym).fast_info
        price = _safe_float(info.get("lastPrice") or info.get("regularMarketPrice"))
        open_px = _safe_float(info.get("regularMarketOpen") or info.get("open"))
        prior_close = _safe_float(
            info.get("regularMarketPreviousClose") or info.get("previousClose")
        )
    except Exception as exc:
        logger.debug("tape fast_info failed for %s: %s", sym, exc)

    return TapeReading(
        symbol=sym,
        price=price,
        open_px=open_px,
        prior_close=prior_close,
        vs_open_pct=_pct(price, open_px),
        vs_prior_close_pct=_pct(price, prior_close),
    )


def fetch_tape(symbols: Sequence[str] = DEFAULT_BENCHMARKS) -> Dict[str, TapeReading]:
    """Live benchmark readings keyed by symbol."""
    readings: Dict[str, TapeReading] = {}
    for raw in symbols:
        sym = str(raw or "").strip().upper()
        if not sym or sym in readings:
            continue
        readings[sym] = _reading_from_fast_info(sym)
    return readings


def evaluate_tape(
    readings: Dict[str, TapeReading],
    *,
    red_vs_open: float = TAPE_RED_VS_OPEN_PCT,
    yellow_vs_open: float = TAPE_YELLOW_VS_OPEN_PCT,
    red_vs_prior_close: float = TAPE_RED_VS_PRIOR_CLOSE_PCT,
    yellow_vs_prior_close: float = TAPE_YELLOW_VS_PRIOR_CLOSE_PCT,
) -> TapeVerdict:
    """
    Aggregate tape into green / yellow / red for new-entry judgment.

    RED: any benchmark at or below red thresholds.
    YELLOW: not red, but any benchmark at or below yellow thresholds.
    GREEN: otherwise.
    """
    if not readings:
        return TapeVerdict("yellow", "no benchmark readings", {})

    red_reasons: list[str] = []
    yellow_reasons: list[str] = []

    for reading in readings.values():
        sym = reading.symbol
        vo = reading.vs_open_pct
        vpc = reading.vs_prior_close_pct
        if vo is not None and vo <= red_vs_open:
            red_reasons.append(f"{sym} vs open {vo:+.2f}%")
        if vpc is not None and vpc <= red_vs_prior_close:
            red_reasons.append(f"{sym} vs prior close {vpc:+.2f}%")
        if vo is not None and yellow_vs_open >= vo > red_vs_open:
            yellow_reasons.append(f"{sym} vs open {vo:+.2f}%")
        if vpc is not None and yellow_vs_prior_close >= vpc > red_vs_prior_close:
            yellow_reasons.append(f"{sym} vs prior close {vpc:+.2f}%")

    if red_reasons:
        return TapeVerdict("red", "; ".join(red_reasons), readings)
    if yellow_reasons:
        return TapeVerdict("yellow", "; ".join(yellow_reasons), readings)
    return TapeVerdict("green", "benchmarks within normal intraday band", readings)
