"""
Sector attractiveness component (10% of final buy score in v2 model).

Scores buy-regime favorability from static sector/industry tables in sectors.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yfinance as yf

from core.services.health.sectors import resolve_sector_score

COMPONENT_WEIGHT = 0.10


@dataclass
class MetricResult:
    key: str
    label: str
    weight: float
    raw: Any = None
    raw_display: str = "N/A"
    score: Optional[float] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "weight": self.weight,
            "raw": self.raw,
            "raw_display": self.raw_display,
            "score": self.score,
            "note": self.note,
        }


@dataclass
class SectorHealthResult:
    symbol: str
    score: Optional[float]
    component_weight: float = COMPONENT_WEIGHT
    sector: str = ""
    industry: str = ""
    sector_key: str = ""
    match_source: str = ""
    match_label: str = ""
    metrics: List[MetricResult] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "component": "sector",
            "component_weight": self.component_weight,
            "score": self.score,
            "sector": self.sector,
            "industry": self.industry,
            "sector_key": self.sector_key,
            "match_source": self.match_source,
            "match_label": self.match_label,
            "metrics": [m.to_dict() for m in self.metrics],
            "missing": self.missing,
            "error": self.error,
        }


def fetch_sector_inputs(symbol: str) -> Dict[str, Any]:
    sym = (symbol or "").strip().upper()
    info = yf.Ticker(sym).info or {}
    return {
        "symbol": sym,
        "sector": (info.get("sector") or "").strip(),
        "industry": (info.get("industry") or "").strip(),
    }


def score_sector_health(symbol: str) -> SectorHealthResult:
    sym = (symbol or "").strip().upper()
    if not sym:
        return SectorHealthResult(symbol="", score=None, error="empty symbol")

    try:
        raw = fetch_sector_inputs(sym)
    except Exception as e:
        return SectorHealthResult(symbol=sym, score=None, error=str(e))

    sector = raw["sector"]
    industry = raw["industry"]
    if not sector and not industry:
        return SectorHealthResult(
            symbol=sym,
            score=None,
            sector=sector,
            industry=industry,
            error="missing sector and industry from yfinance",
        )

    resolved = resolve_sector_score(sector, industry)
    score = float(resolved["score"])
    sector_key = resolved["sector_key"]
    source = resolved["source"]
    label = resolved.get("override_label") or ""

    raw_display = sector or "—"
    if industry:
        raw_display = f"{sector} / {industry}" if sector else industry

    metric = MetricResult(
        key="sector_attractiveness",
        label="Sector / industry",
        weight=1.0,
        raw=resolved.get("match_key"),
        raw_display=raw_display,
        score=round(score, 1),
        note=f"{source}: {label}" if label else source,
    )

    return SectorHealthResult(
        symbol=sym,
        score=round(score, 1),
        sector=sector,
        industry=industry,
        sector_key=sector_key,
        match_source=source,
        match_label=label,
        metrics=[metric],
    )
