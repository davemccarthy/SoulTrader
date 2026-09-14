"""
MIDWAY Stage 2 — soft-score SO-universe names vs SPY / sector ETF.

Does not discover or buy. Ranks quality mid-caps by relative weakness under a
market permission card from midway_state.

Soft sector ≠ auto opportunity: sector sleeve is a preference boost only;
SO grades come from the assessment universe file (quality veto memory).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.services.financial import yahoo as financial_yahoo
from core.services.market.midway_state import (
    BENCH_SPY,
    MidwayMarketState,
    SeriesMetrics,
    metrics_for_symbol,
)

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = Path(".assessments/universe_midway_pre_llm_2026-08-14.json")

# Yahoo sector string → SPDR (same keys as ASS timing).
SECTOR_ETF: Dict[str, str] = {
    "technology": "XLK",
    "healthcare": "XLV",
    "financial services": "XLF",
    "financial": "XLF",
    "energy": "XLE",
    "consumer cyclical": "XLY",
    "consumer defensive": "XLP",
    "industrials": "XLI",
    "utilities": "XLU",
    "real estate": "XLRE",
    "basic materials": "XLB",
    "communication services": "XLC",
}

# Soft score: higher = more soft (better candidate under ACTIVE_SOFT).
# soft = -(w_spy * vs_spy_20d + w_sec * vs_sector_20d + w_5d * vs_spy_5d)
W_VS_SPY_20 = 0.40
W_VS_SECTOR_20 = 0.40
W_VS_SPY_5 = 0.20

# Soft sleeve preference (points added to soft_score when sector ETF is soft).
SOFT_SECTOR_BOOST = 1.0
HOT_SECTOR_PENALTY = 1.5

# Pass bars by stance (soft_score after boost/penalty).
SOFT_BAR_ACTIVE = 1.0  # ~1pp lag blend
SOFT_BAR_DONT_CHASE = 3.0  # raised — only clear laggards
SOFT_BAR_NO_DEPLOY = None  # nothing passes


@dataclass(frozen=True)
class UniverseRow:
    symbol: str
    so_pair: str
    stability: Optional[float]
    opportunity: Optional[float]
    composite: Optional[float]


@dataclass(frozen=True)
class SoftCandidate:
    symbol: str
    so_pair: str
    stability: Optional[float]
    opportunity: Optional[float]
    price: Optional[float]
    sector: str
    sector_etf: str
    ret_5d_pct: Optional[float]
    ret_20d_pct: Optional[float]
    vs_spy_5d_pct: Optional[float]
    vs_spy_20d_pct: Optional[float]
    vs_sector_20d_pct: Optional[float]
    soft_score: Optional[float]
    sleeve: str  # soft | hot | mid | unknown
    passes: bool
    skip_reason: str


def load_universe_rows(path: Path) -> List[UniverseRow]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("records") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError(f"No records list in {path}")
    rows: List[UniverseRow] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        sym = str(rec.get("symbol") or "").strip().upper()
        if not sym:
            continue
        rows.append(
            UniverseRow(
                symbol=sym,
                so_pair=str(rec.get("so_pair") or "").strip().upper() or "?",
                stability=_opt_float(rec.get("stability")),
                opportunity=_opt_float(rec.get("opportunity")),
                composite=_opt_float(rec.get("composite")),
            )
        )
    return rows


def _opt_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sector_etf_for(info: Dict[str, Any]) -> Tuple[str, str]:
    sector = (info.get("sector") or "").strip()
    key = sector.lower()
    for needle, etf in SECTOR_ETF.items():
        if needle in key:
            return sector or needle.title(), etf
    return sector or "Unknown", BENCH_SPY


def _soft_score(
    vs_spy_20: Optional[float],
    vs_sector_20: Optional[float],
    vs_spy_5: Optional[float],
) -> Optional[float]:
    parts: List[Tuple[float, float]] = []
    if vs_spy_20 is not None:
        parts.append((vs_spy_20, W_VS_SPY_20))
    if vs_sector_20 is not None:
        parts.append((vs_sector_20, W_VS_SECTOR_20))
    if vs_spy_5 is not None:
        parts.append((vs_spy_5, W_VS_SPY_5))
    if not parts:
        return None
    wsum = sum(w for _, w in parts)
    if wsum <= 0:
        return None
    blend = sum(v * w for v, w in parts) / wsum
    return round(-blend, 3)  # more lag → higher score


def soft_bar_for_stance(stance: str) -> Optional[float]:
    if stance == "no_deployment":
        return SOFT_BAR_NO_DEPLOY
    if stance == "dont_chase":
        return SOFT_BAR_DONT_CHASE
    return SOFT_BAR_ACTIVE


def rank_soft_candidates(
    state: MidwayMarketState,
    rows: Sequence[UniverseRow],
    *,
    prefer_soft_sectors: bool = True,
) -> List[SoftCandidate]:
    """
    Score each universe row for relative softness under market permission.
    """
    spy = state.benchmarks.get(BENCH_SPY)
    if spy is None:
        spy = metrics_for_symbol(BENCH_SPY)

    # Cache sector ETF metrics from market state when available.
    sector_ret_20: Dict[str, Optional[float]] = {
        s.etf: s.ret_20d_pct for s in state.sectors
    }
    soft_set = set(state.soft_sectors)
    hot_set = set(state.hot_sectors)
    bar = soft_bar_for_stance(state.stance)

    out: List[SoftCandidate] = []
    for row in rows:
        cand = _score_one(
            row,
            spy=spy,
            sector_ret_20=sector_ret_20,
            soft_set=soft_set,
            hot_set=hot_set,
            prefer_soft_sectors=prefer_soft_sectors,
            bar=bar,
            stance=state.stance,
        )
        out.append(cand)

    out.sort(
        key=lambda c: (
            not c.passes,
            -(c.soft_score if c.soft_score is not None else -999.0),
            c.symbol,
        )
    )
    return out


def _score_one(
    row: UniverseRow,
    *,
    spy: SeriesMetrics,
    sector_ret_20: Dict[str, Optional[float]],
    soft_set: set,
    hot_set: set,
    prefer_soft_sectors: bool,
    bar: Optional[float],
    stance: str,
) -> SoftCandidate:
    info = financial_yahoo.get_ticker_info(row.symbol)
    sector_name, sector_etf = _sector_etf_for(info)
    m = metrics_for_symbol(row.symbol)

    if sector_etf not in sector_ret_20:
        sec_m = metrics_for_symbol(sector_etf)
        sector_ret_20[sector_etf] = sec_m.ret_20d_pct

    sec_r20 = sector_ret_20.get(sector_etf)
    spy5 = spy.ret_5d_pct
    spy20 = spy.ret_20d_pct

    vs_spy_5 = None
    if m.ret_5d_pct is not None and spy5 is not None:
        vs_spy_5 = round(m.ret_5d_pct - spy5, 3)
    vs_spy_20 = None
    if m.ret_20d_pct is not None and spy20 is not None:
        vs_spy_20 = round(m.ret_20d_pct - spy20, 3)
    vs_sec_20 = None
    if m.ret_20d_pct is not None and sec_r20 is not None:
        vs_sec_20 = round(m.ret_20d_pct - sec_r20, 3)

    score = _soft_score(vs_spy_20, vs_sec_20, vs_spy_5)
    sleeve = "mid"
    if sector_etf in soft_set:
        sleeve = "soft"
        if prefer_soft_sectors and score is not None:
            score = round(score + SOFT_SECTOR_BOOST, 3)
    elif sector_etf in hot_set:
        sleeve = "hot"
        if prefer_soft_sectors and score is not None:
            score = round(score - HOT_SECTOR_PENALTY, 3)
    elif sector_etf == BENCH_SPY and sector_name == "Unknown":
        sleeve = "unknown"

    skip = ""
    passes = False
    if bar is None:
        skip = "stance=no_deployment"
    elif score is None:
        skip = "insufficient price history"
    elif score < bar:
        skip = f"soft_score {score:.2f} < bar {bar:.2f} ({stance})"
    else:
        passes = True

    return SoftCandidate(
        symbol=row.symbol,
        so_pair=row.so_pair,
        stability=row.stability,
        opportunity=row.opportunity,
        price=m.price,
        sector=sector_name,
        sector_etf=sector_etf,
        ret_5d_pct=m.ret_5d_pct,
        ret_20d_pct=m.ret_20d_pct,
        vs_spy_5d_pct=vs_spy_5,
        vs_spy_20d_pct=vs_spy_20,
        vs_sector_20d_pct=vs_sec_20,
        soft_score=score,
        sleeve=sleeve,
        passes=passes,
        skip_reason=skip,
    )


def format_soft_candidates(
    candidates: Sequence[SoftCandidate],
    *,
    top: int = 15,
    stance: str = "",
    bar: Optional[float] = None,
) -> str:
    lines: List[str] = []
    lines.append("=== MIDWAY soft rank (SO universe × relative weakness) ===")
    if stance:
        bar_s = "none" if bar is None else f"{bar:.1f}"
        lines.append(f"Stance: {stance}  |  soft bar: {bar_s}  |  higher soft = more lag")
    lines.append(
        "Note: soft sector is attention only — SO quality from universe file; "
        "not a fundamental 'why soft' check."
    )
    lines.append("")
    hdr = (
        f"{'#':>2} {'sym':<6} {'SO':<4} {'soft':>6} {'pass':>4} "
        f"{'5d%':>6} {'20d%':>6} {'vsSPY20':>8} {'vsSec20':>8} "
        f"{'etf':<5} {'sleeve':<6} sector"
    )
    lines.append(hdr)
    shown = 0
    for i, c in enumerate(candidates, start=1):
        if shown >= top:
            break
        shown += 1
        soft = f"{c.soft_score:.2f}" if c.soft_score is not None else "n/a"
        flag = "Y" if c.passes else "n"

        def _f(v: Optional[float]) -> str:
            return f"{v:+.1f}" if v is not None else "n/a"

        lines.append(
            f"{i:>2} {c.symbol:<6} {c.so_pair:<4} {soft:>6} {flag:>4} "
            f"{_f(c.ret_5d_pct):>6} {_f(c.ret_20d_pct):>6} "
            f"{_f(c.vs_spy_20d_pct):>8} {_f(c.vs_sector_20d_pct):>8} "
            f"{c.sector_etf:<5} {c.sleeve:<6} {c.sector[:22]}"
        )

    passed = [c for c in candidates if c.passes]
    lines.append("")
    if not passed:
        lines.append("Follow list: (none clear soft bar)")
    else:
        follow = " ".join(c.symbol for c in passed[: min(12, len(passed))])
        lines.append(f"Follow list (pass): {follow}")
    return "\n".join(lines)
