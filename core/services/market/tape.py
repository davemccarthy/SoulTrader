"""
Intraday market tape: benchmark move vs today's open and prior close.

Four-color posture for new-entry judgment (Pulse maps IPC by color):
  RED    — no new trades
  AMBER  — caution trade (tight IPC)
  GREEN  — normal trade
  WHITE  — strong tape (looser IPC)

Generic service for manual ops / advisor discover gates (no session-history rules).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARKS: tuple[str, ...] = ("SPY", "QQQ")

# Risk-off / caution (any benchmark trips → that color).
TAPE_RED_VS_OPEN_PCT = -1.5
TAPE_AMBER_VS_OPEN_PCT = -0.75
TAPE_RED_VS_PRIOR_CLOSE_PCT = -2.0
TAPE_AMBER_VS_PRIOR_CLOSE_PCT = -0.75

# Strong tape (ALL benchmarks must clear both floors → white).
# Loosened from 0.75/1.0 so white can fire on solid up days (never hit old bar).
TAPE_WHITE_VS_OPEN_PCT = 0.50
TAPE_WHITE_VS_PRIOR_CLOSE_PCT = 0.75

# Back-compat aliases (yellow → amber).
TAPE_YELLOW_VS_OPEN_PCT = TAPE_AMBER_VS_OPEN_PCT
TAPE_YELLOW_VS_PRIOR_CLOSE_PCT = TAPE_AMBER_VS_PRIOR_CLOSE_PCT

TAPE_STATES: frozenset[str] = frozenset({"red", "amber", "green", "white"})


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
    state: str  # red | amber | green | white
    reason: str
    readings: Dict[str, TapeReading]


def normalize_tape_state(state: Optional[str], default: str = "red") -> str:
    """Map legacy 'yellow' → 'amber'; unknown → default."""
    s = str(state or default).strip().lower()
    if s == "yellow":
        return "amber"
    if s in TAPE_STATES:
        return s
    return default


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
    amber_vs_open: float = TAPE_AMBER_VS_OPEN_PCT,
    red_vs_prior_close: float = TAPE_RED_VS_PRIOR_CLOSE_PCT,
    amber_vs_prior_close: float = TAPE_AMBER_VS_PRIOR_CLOSE_PCT,
    white_vs_open: float = TAPE_WHITE_VS_OPEN_PCT,
    white_vs_prior_close: float = TAPE_WHITE_VS_PRIOR_CLOSE_PCT,
    # Deprecated aliases
    yellow_vs_open: Optional[float] = None,
    yellow_vs_prior_close: Optional[float] = None,
) -> TapeVerdict:
    """
    Aggregate tape into red / amber / green / white.

    RED: any benchmark at or below red thresholds.
    AMBER: not red, but any benchmark at or below amber thresholds.
    WHITE: not red/amber, and ALL benchmarks at or above white floors (vs open + prior).
    GREEN: otherwise (normal trade band).
    """
    if yellow_vs_open is not None:
        amber_vs_open = yellow_vs_open
    if yellow_vs_prior_close is not None:
        amber_vs_prior_close = yellow_vs_prior_close

    if not readings:
        return TapeVerdict("amber", "no benchmark readings", {})

    red_reasons: list[str] = []
    amber_reasons: list[str] = []
    white_ok = True
    white_bits: list[str] = []

    for reading in readings.values():
        sym = reading.symbol
        vo = reading.vs_open_pct
        vpc = reading.vs_prior_close_pct
        if vo is not None and vo <= red_vs_open:
            red_reasons.append(f"{sym} vs open {vo:+.2f}%")
        if vpc is not None and vpc <= red_vs_prior_close:
            red_reasons.append(f"{sym} vs prior close {vpc:+.2f}%")
        if vo is not None and amber_vs_open >= vo > red_vs_open:
            amber_reasons.append(f"{sym} vs open {vo:+.2f}%")
        if vpc is not None and amber_vs_prior_close >= vpc > red_vs_prior_close:
            amber_reasons.append(f"{sym} vs prior close {vpc:+.2f}%")

        if vo is None or vpc is None or vo < white_vs_open or vpc < white_vs_prior_close:
            white_ok = False
        else:
            white_bits.append(
                f"{sym} vs open {vo:+.2f}% / vs prior {vpc:+.2f}%"
            )

    if red_reasons:
        return TapeVerdict("red", "; ".join(red_reasons), readings)
    if amber_reasons:
        return TapeVerdict("amber", "; ".join(amber_reasons), readings)
    if white_ok and white_bits:
        return TapeVerdict(
            "white",
            f"strong tape ({'; '.join(white_bits)})",
            readings,
        )
    return TapeVerdict("green", "benchmarks within normal intraday band", readings)
