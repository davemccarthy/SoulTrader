"""
Single source for sector/industry tables used by health v2 (and shared with valuation).

Buy-attractiveness sector bases are a one-time refresh from trending_radar
(.assessments/trending_radar.json, 2026-10-01): radar 50 → assessment 55,
scale 0.45. Industry overrides remain curated. Valuation benchmarks are
separate (typical multiples per sector). Keys are lowercase substrings
matched against yfinance `sector` / `industry`.

Review helper:
    python -c "from core.services.health.sectors import export_tables_json; print(export_tables_json())"
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# --- Buy-regime attractiveness (sector component, 0–100) -------------------

DEFAULT_SECTOR_SCORE = 55.0

# Sector bases: one-time refresh from .assessments/trending_radar.json (2026-10-01).
# Mapping: radar 50 → assessment 55, scale 0.45 → score = 55 + (trend - 50) * 0.45.
# Unlisted sectors → DEFAULT_SECTOR_SCORE (55). Longer keys first where needed.
SECTOR_BASE_SCORES: List[Tuple[str, float, str]] = [
    ("communication services", 49.0, "Communication Services base (Oct radar)"),
    ("communication", 49.0, "Communication Services base alias"),
    ("consumer defensive", 41.0, "Consumer Staples base (Oct radar)"),
    ("consumer staples", 41.0, "Consumer Staples base alias"),
    ("consumer cyclical", 39.0, "Consumer Discretionary base (Oct radar)"),
    ("consumer discretionary", 39.0, "Consumer Discretionary base alias"),
    ("basic materials", 42.0, "Materials base (Oct radar)"),
    ("healthcare", 58.0, "Healthcare base (Oct radar)"),
    ("technology", 67.0, "Technology base (Oct radar)"),
    ("industrials", 43.0, "Industrials base (Oct radar)"),
    ("financial services", 44.0, "Financials base (Oct radar)"),
    ("financial", 44.0, "Financials base alias"),
    ("real estate", 38.0, "Real Estate base (Oct radar)"),
    ("materials", 42.0, "Materials base alias"),
    ("utilities", 35.0, "Utilities base (Oct radar)"),
    ("energy", 55.0, "Energy base (Oct radar)"),
]

# Industry overrides. Checked before sector base; first match wins.
# industry None: sector_sub in sector OR industry. industry set: both must match.
INDUSTRY_SCORE_OVERRIDES: List[Tuple[str, Optional[str], float, str]] = [
    ("cannabis", None, 10.0, "Cannabis"),
    ("payment", None, 82.0, "Payments"),
    ("technology", "semiconductor", 85.0, "Semiconductors"),
    ("technology", "consumer electronic", 78.0, "Consumer electronics"),
    ("healthcare", "biotechnology", 78.0, "Biotechnology"),
    ("industrials", "aerospace", 76.0, "Aerospace & defense"),
    ("technology", "software", 70.0, "Software"),
    # META / GOOGL: sit above Communication Services base (49).
    ("communication", "internet content", 58.0, "Internet Content & Information"),
    ("industrials", "railroads", 68.0, "Railroads"),
    ("industrials", "airlines", 42.0, "Airlines"),
]

# --- Valuation multiples vs sector (valuation component) --------------------

DEFAULT_VALUATION_BENCHMARK: Dict[str, float] = {
    "forward_pe": 18.0,
    "ev_ebitda": 12.0,
    "ps": 2.5,
}

SECTOR_VALUATION_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "technology": {"forward_pe": 24.0, "ev_ebitda": 18.0, "ps": 6.0},
    "communication": {"forward_pe": 18.0, "ev_ebitda": 10.0, "ps": 3.5},
    "consumer cyclical": {"forward_pe": 18.0, "ev_ebitda": 12.0, "ps": 1.5},
    "consumer defensive": {"forward_pe": 20.0, "ev_ebitda": 14.0, "ps": 2.0},
    "energy": {"forward_pe": 12.0, "ev_ebitda": 6.0, "ps": 1.2},
    "financial": {"forward_pe": 13.0, "ev_ebitda": 10.0, "ps": 2.5},
    "financial services": {"forward_pe": 13.0, "ev_ebitda": 10.0, "ps": 2.5},
    "healthcare": {"forward_pe": 18.0, "ev_ebitda": 14.0, "ps": 3.5},
    "industrials": {"forward_pe": 18.0, "ev_ebitda": 12.0, "ps": 2.0},
    "basic materials": {"forward_pe": 14.0, "ev_ebitda": 9.0, "ps": 1.5},
    "real estate": {"forward_pe": 16.0, "ev_ebitda": 16.0, "ps": 5.0},
    "utilities": {"forward_pe": 16.0, "ev_ebitda": 10.0, "ps": 2.0},
}


def _norm(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def resolve_sector_key(sector: Optional[str]) -> str:
    """Map yfinance sector string to canonical benchmark/score key."""
    s = _norm(sector)
    if not s:
        return "default"
    for key, _, _ in SECTOR_BASE_SCORES:
        if key in s:
            return key
    for key in SECTOR_VALUATION_BENCHMARKS:
        if key in s:
            return key
    return s


def resolve_valuation_benchmark(sector: Optional[str]) -> Tuple[str, Dict[str, float]]:
    s = _norm(sector)
    if not s:
        return "default", dict(DEFAULT_VALUATION_BENCHMARK)
    for key, bench in SECTOR_VALUATION_BENCHMARKS.items():
        if key in s:
            return key, dict(bench)
    return s, dict(DEFAULT_VALUATION_BENCHMARK)


def _match_industry_override(
    sector: str,
    industry: str,
) -> Optional[Tuple[float, str, str]]:
    """First matching override in INDUSTRY_SCORE_OVERRIDES order."""
    for sector_sub, industry_sub, score, label in INDUSTRY_SCORE_OVERRIDES:
        if industry_sub is None:
            if sector_sub in sector or sector_sub in industry:
                return score, label, sector_sub
        else:
            if sector_sub in sector and industry_sub in industry:
                return score, label, f"{sector_sub}/{industry_sub}"
    return None


def resolve_sector_score(
    sector: Optional[str],
    industry: Optional[str],
) -> Dict[str, Any]:
    """
    Resolve sector attractiveness score and metadata.

    Returns dict with: score, sector_key, sector_label, industry_label,
    source ('industry_override' | 'sector_base' | 'default'), override_label.
    """
    sector_n = _norm(sector)
    industry_n = _norm(industry)
    sector_key = resolve_sector_key(sector)

    override = _match_industry_override(sector_n, industry_n)
    if override is not None:
        score, label, match_key = override
        return {
            "score": score,
            "sector_key": sector_key,
            "sector": sector or "",
            "industry": industry or "",
            "source": "industry_override",
            "override_label": label,
            "match_key": match_key,
        }

    for key, base_score, rationale in SECTOR_BASE_SCORES:
        if key in sector_n:
            return {
                "score": base_score,
                "sector_key": key,
                "sector": sector or "",
                "industry": industry or "",
                "source": "sector_base",
                "override_label": rationale,
                "match_key": key,
            }

    return {
        "score": DEFAULT_SECTOR_SCORE,
        "sector_key": "default",
        "sector": sector or "",
        "industry": industry or "",
        "source": "default",
        "override_label": "unknown sector — neutral",
        "match_key": "default",
    }


def export_tables_json(indent: int = 2) -> str:
    """Serialize tables for LLM / human review."""
    payload = {
        "default_sector_score": DEFAULT_SECTOR_SCORE,
        "sector_base_scores": [
            {"sector_key": k, "score": s, "rationale": r}
            for k, s, r in SECTOR_BASE_SCORES
        ],
        "industry_overrides": [
            {
                "sector_contains": sec,
                "industry_contains": ind,
                "score": sc,
                "label": lab,
            }
            for sec, ind, sc, lab in INDUSTRY_SCORE_OVERRIDES
        ],
        "valuation_benchmarks": SECTOR_VALUATION_BENCHMARKS,
        "default_valuation_benchmark": DEFAULT_VALUATION_BENCHMARK,
    }
    return json.dumps(payload, indent=indent)
