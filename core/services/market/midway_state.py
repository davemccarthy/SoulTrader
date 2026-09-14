"""
MIDWAY market state — multi-day trend, mood, and sector relative strength.

Independent of advisors / discoveries. Price-based only (no RSS, 8-K, rates models).
Used by `manage.py midway_market_status`; later by a Midway advisor.

Clocks:
  TREND  — slow primary direction (SPY vs 50d, medium return)
  MOOD   — fast internals (5d, IWM vs SPY, distance from 20d high)
  SECTOR — sleeve tilt (sector ETF 20d vs SPY)
  STANCE — DON'T CHASE / ACTIVE soft / NO DEPLOYMENT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from core.services.financial import yahoo as financial_yahoo

logger = logging.getLogger(__name__)

# Benchmarks
BENCH_SPY = "SPY"
BENCH_IWM = "IWM"
BENCH_QQQ = "QQQ"
DEFAULT_BENCHMARKS: Tuple[str, ...] = (BENCH_SPY, BENCH_IWM, BENCH_QQQ)

# SPDR sector ETFs (same map spirit as ASS timing).
SECTOR_ETFS: Tuple[Tuple[str, str], ...] = (
    ("Technology", "XLK"),
    ("Healthcare", "XLV"),
    ("Financials", "XLF"),
    ("Energy", "XLE"),
    ("Consumer Disc.", "XLY"),
    ("Consumer Staples", "XLP"),
    ("Industrials", "XLI"),
    ("Utilities", "XLU"),
    ("Real Estate", "XLRE"),
    ("Materials", "XLB"),
    ("Comm. Services", "XLC"),
)

RET_5D = 5
RET_20D = 20
MA_50D = 50
HIGH_20D = 20

# Trend
TREND_BULL_MED_RET_PCT = 0.0  # SPY ~60-session proxy via 20d*3 not used; use vs 50d + 20d
TREND_BEAR_VS_50D_PCT = -2.0  # close meaningfully below 50d

# Mood
MOOD_SOFT_5D_PCT = -1.0  # SPY or IWM 5d
MOOD_IWM_LAG_VS_SPY_PCT = -1.0  # IWM 5d − SPY 5d
MOOD_OFF_HIGH_PCT = 3.0  # % below 20d high
MOOD_MELT_5D_PCT = 1.5  # strong short-term bid

# Sector soft/hot vs SPY over 20d
SECTOR_SOFT_VS_SPY_PCT = -1.5
SECTOR_HOT_VS_SPY_PCT = 1.5


@dataclass(frozen=True)
class SeriesMetrics:
    symbol: str
    price: Optional[float]
    ret_5d_pct: Optional[float]
    ret_20d_pct: Optional[float]
    vs_50d_pct: Optional[float]
    pct_below_20d_high: Optional[float]


@dataclass(frozen=True)
class SectorReading:
    name: str
    etf: str
    ret_20d_pct: Optional[float]
    vs_spy_20d_pct: Optional[float]  # sector − SPY over 20d


@dataclass
class MidwayMarketState:
    """Human-readable market card for MIDWAY permission."""

    trend: str  # bull | bull_mature | sideways | bear
    mood: str  # constructive | nervous | deteriorating | melt_up | risk_off
    stance: str  # active_soft | dont_chase | no_deployment
    permission: str  # short label for the card
    reason: str
    benchmarks: Dict[str, SeriesMetrics] = field(default_factory=dict)
    sectors: List[SectorReading] = field(default_factory=list)
    soft_sectors: List[str] = field(default_factory=list)  # ETF tickers
    hot_sectors: List[str] = field(default_factory=list)

    def soft_sector_names(self) -> List[str]:
        by_etf = {s.etf: s.name for s in self.sectors}
        return [f"{by_etf.get(t, t)} ({t})" for t in self.soft_sectors]

    def hot_sector_names(self) -> List[str]:
        by_etf = {s.etf: s.name for s in self.sectors}
        return [f"{by_etf.get(t, t)} ({t})" for t in self.hot_sectors]


def _closes(hist: pd.DataFrame) -> pd.Series:
    if hist is None or hist.empty or "close" not in hist.columns:
        return pd.Series(dtype=float)
    return hist["close"].astype(float)


def _ret_pct(closes: pd.Series, sessions: int) -> Optional[float]:
    if closes.empty or len(closes) < sessions + 1:
        return None
    cur = float(closes.iloc[-1])
    ref = float(closes.iloc[-1 - sessions])
    if ref <= 0 or cur <= 0:
        return None
    return round((cur / ref - 1.0) * 100.0, 3)


def _vs_ma_pct(closes: pd.Series, window: int) -> Optional[float]:
    if closes.empty or len(closes) < window:
        return None
    cur = float(closes.iloc[-1])
    ma = float(closes.tail(window).mean())
    if ma <= 0 or cur <= 0:
        return None
    return round((cur / ma - 1.0) * 100.0, 3)


def _pct_below_high(closes: pd.Series, window: int) -> Optional[float]:
    if closes.empty or len(closes) < window:
        return None
    cur = float(closes.iloc[-1])
    hi = float(closes.tail(window).max())
    if hi <= 0 or cur <= 0:
        return None
    return round((1.0 - cur / hi) * 100.0, 3)


def metrics_for_symbol(symbol: str, hist: Optional[pd.DataFrame] = None) -> SeriesMetrics:
    sym = symbol.strip().upper()
    if hist is None:
        hist = financial_yahoo.get_6m_history(sym)
    closes = _closes(hist)
    price = float(closes.iloc[-1]) if not closes.empty else None
    return SeriesMetrics(
        symbol=sym,
        price=price,
        ret_5d_pct=_ret_pct(closes, RET_5D),
        ret_20d_pct=_ret_pct(closes, RET_20D),
        vs_50d_pct=_vs_ma_pct(closes, MA_50D),
        pct_below_20d_high=_pct_below_high(closes, HIGH_20D),
    )


def _label_trend(spy: SeriesMetrics) -> str:
    vs50 = spy.vs_50d_pct
    r20 = spy.ret_20d_pct
    off = spy.pct_below_20d_high
    if vs50 is None:
        return "sideways"
    if vs50 <= TREND_BEAR_VS_50D_PCT and (r20 is None or r20 < 0):
        return "bear"
    if vs50 > 0 and (r20 is None or r20 >= TREND_BULL_MED_RET_PCT):
        # Extended if still above 50d but off local highs / soft 20d
        if off is not None and off >= MOOD_OFF_HIGH_PCT and (r20 is None or r20 < 2.0):
            return "bull_mature"
        if r20 is not None and r20 < 0:
            return "bull_mature"
        return "bull"
    if vs50 > 0:
        return "bull_mature"
    return "sideways"


def _label_mood(
    spy: SeriesMetrics,
    iwm: Optional[SeriesMetrics],
    soft_count: int,
    sector_count: int,
) -> str:
    spy5 = spy.ret_5d_pct
    spy_off = spy.pct_below_20d_high or 0.0
    iwm5 = iwm.ret_5d_pct if iwm else None
    lag = None
    if spy5 is not None and iwm5 is not None:
        lag = iwm5 - spy5

    # Melt-up: strong short-term bid, near highs
    if (
        spy5 is not None
        and spy5 >= MOOD_MELT_5D_PCT
        and spy_off < 1.5
        and (iwm5 is None or iwm5 >= 0)
    ):
        return "melt_up"

    # Risk-off: sharp short-term damage
    if spy5 is not None and spy5 <= -3.0:
        return "risk_off"
    if iwm5 is not None and iwm5 <= -3.5:
        return "risk_off"

    soft_internals = False
    if spy5 is not None and spy5 <= MOOD_SOFT_5D_PCT:
        soft_internals = True
    if iwm5 is not None and iwm5 <= MOOD_SOFT_5D_PCT:
        soft_internals = True
    if lag is not None and lag <= MOOD_IWM_LAG_VS_SPY_PCT:
        soft_internals = True
    if spy_off >= MOOD_OFF_HIGH_PCT and (spy5 is None or spy5 < 0.5):
        soft_internals = True
    if sector_count > 0 and soft_count >= max(3, sector_count // 3):
        soft_internals = True

    if soft_internals:
        # Broad sector soft → deteriorating; milder → nervous
        if sector_count > 0 and soft_count >= max(4, sector_count // 2):
            return "deteriorating"
        if lag is not None and lag <= MOOD_IWM_LAG_VS_SPY_PCT and (
            spy5 is not None and spy5 < 0
        ):
            return "deteriorating"
        return "nervous"

    return "constructive"


def _stance_and_permission(trend: str, mood: str) -> Tuple[str, str, str]:
    """
    Returns (stance, permission_label, reason).

    Stance keys:
      no_deployment — do not add new names
      dont_chase    — only extreme soft SO-passers (raised soft bar)
      active_soft   — hunt quality on weakness (best MIDWAY window)
    """
    if mood == "melt_up":
        return (
            "dont_chase",
            "DON'T CHASE — melt-up / near highs",
            "Short-term bid with price near local highs; raise bar or wait for soft sleeves.",
        )
    if mood == "risk_off":
        return (
            "no_deployment",
            "NO DEPLOYMENT — sharp risk-off",
            "Broad short-term damage; keep powder for held avg-downs, not new names.",
        )
    if trend == "bear" and mood in ("deteriorating", "nervous", "risk_off"):
        return (
            "no_deployment",
            "NO DEPLOYMENT — bear + soft mood",
            "Primary trend broken and internals soft; pause new discovers.",
        )
    if mood in ("deteriorating", "nervous"):
        return (
            "active_soft",
            "ACTIVE — don't chase; hunt soft quality",
            "Trend may still be intact but mood/internals soft — good stock + bad tape.",
        )
    if trend in ("bull", "bull_mature") and mood == "constructive":
        return (
            "dont_chase",
            "DON'T CHASE — constructive but selective",
            "Healthy tape; only extreme laggards vs SPY/sector, not indiscriminate refill.",
        )
    return (
        "dont_chase",
        "DON'T CHASE — mixed",
        "Mixed trend/mood; prefer soft SO-passers only.",
    )


def evaluate_midway_market(
    *,
    include_qqq: bool = True,
    sector_etfs: Sequence[Tuple[str, str]] = SECTOR_ETFS,
) -> MidwayMarketState:
    """
    Build MIDWAY market card from daily history (Yahoo 6mo).

    Does not consult fund equity band — that stays a fund-level gate.
    """
    symbols = [BENCH_SPY, BENCH_IWM]
    if include_qqq:
        symbols.append(BENCH_QQQ)

    hist_cache: Dict[str, pd.DataFrame] = {}
    benchmarks: Dict[str, SeriesMetrics] = {}
    for sym in symbols:
        hist = financial_yahoo.get_6m_history(sym)
        hist_cache[sym] = hist
        benchmarks[sym] = metrics_for_symbol(sym, hist)

    spy = benchmarks[BENCH_SPY]
    iwm = benchmarks.get(BENCH_IWM)
    spy_r20 = spy.ret_20d_pct

    sectors: List[SectorReading] = []
    for name, etf in sector_etfs:
        hist = financial_yahoo.get_6m_history(etf)
        m = metrics_for_symbol(etf, hist)
        vs_spy = None
        if m.ret_20d_pct is not None and spy_r20 is not None:
            vs_spy = round(m.ret_20d_pct - spy_r20, 3)
        sectors.append(
            SectorReading(
                name=name,
                etf=etf,
                ret_20d_pct=m.ret_20d_pct,
                vs_spy_20d_pct=vs_spy,
            )
        )

    soft = [
        s.etf
        for s in sectors
        if s.vs_spy_20d_pct is not None and s.vs_spy_20d_pct <= SECTOR_SOFT_VS_SPY_PCT
    ]
    hot = [
        s.etf
        for s in sectors
        if s.vs_spy_20d_pct is not None and s.vs_spy_20d_pct >= SECTOR_HOT_VS_SPY_PCT
    ]
    # Rank soft most negative first, hot most positive first
    soft.sort(
        key=lambda t: next(s.vs_spy_20d_pct for s in sectors if s.etf == t),
    )
    hot.sort(
        key=lambda t: next(s.vs_spy_20d_pct for s in sectors if s.etf == t),
        reverse=True,
    )

    trend = _label_trend(spy)
    mood = _label_mood(spy, iwm, soft_count=len(soft), sector_count=len(sectors))
    stance, permission, reason = _stance_and_permission(trend, mood)

    return MidwayMarketState(
        trend=trend,
        mood=mood,
        stance=stance,
        permission=permission,
        reason=reason,
        benchmarks=benchmarks,
        sectors=sectors,
        soft_sectors=soft,
        hot_sectors=hot,
    )


def format_midway_market_card(state: MidwayMarketState) -> str:
    """Plain-text card for CLI / logs."""
    lines: List[str] = []
    lines.append("=== MIDWAY market state ===")
    lines.append(f"TREND:      {state.trend}")
    lines.append(f"MOOD:       {state.mood}")
    lines.append(f"STANCE:     {state.stance}")
    lines.append(f"PERMISSION: {state.permission}")
    lines.append(f"Why:        {state.reason}")
    lines.append("")
    lines.append(
        f"{'sym':<6} {'px':>8} {'5d%':>7} {'20d%':>7} {'vs50d%':>8} {'off20h%':>8}"
    )
    for sym in (BENCH_SPY, BENCH_IWM, BENCH_QQQ):
        m = state.benchmarks.get(sym)
        if m is None:
            continue
        px = f"{m.price:.2f}" if m.price is not None else "n/a"
        def _f(v: Optional[float]) -> str:
            return f"{v:+.2f}" if v is not None else "n/a"

        lines.append(
            f"{sym:<6} {px:>8} {_f(m.ret_5d_pct):>7} {_f(m.ret_20d_pct):>7} "
            f"{_f(m.vs_50d_pct):>8} {_f(m.pct_below_20d_high):>8}"
        )

    lines.append("")
    lines.append("Sectors vs SPY (20d):")
    lines.append(f"{'sector':<18} {'etf':<6} {'20d%':>7} {'vs SPY':>8}")
    for s in sorted(
        state.sectors,
        key=lambda x: (x.vs_spy_20d_pct is None, x.vs_spy_20d_pct or 0.0),
    ):
        r20 = f"{s.ret_20d_pct:+.2f}" if s.ret_20d_pct is not None else "n/a"
        vs = f"{s.vs_spy_20d_pct:+.2f}" if s.vs_spy_20d_pct is not None else "n/a"
        tag = ""
        if s.etf in state.soft_sectors:
            tag = "  soft"
        elif s.etf in state.hot_sectors:
            tag = "  hot"
        lines.append(f"{s.name:<18} {s.etf:<6} {r20:>7} {vs:>8}{tag}")

    lines.append("")
    soft_n = ", ".join(state.soft_sector_names()) or "(none)"
    hot_n = ", ".join(state.hot_sector_names()) or "(none)"
    lines.append(f"Hunt (soft): {soft_n}")
    lines.append(f"Avoid chase (hot): {hot_n}")
    return "\n".join(lines)
