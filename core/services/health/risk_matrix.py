"""
Risk-band dual-floor buy matrix: stability + opportunity (× discovery.weight).

Replaces legacy min grade letter / single composite min_score on Profile.
Floors are SO letter grades; numeric mins are derived for logging.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from core.services.health.distress import adjust_opportunity_parts
from core.services.health.durability import score_business_durability
from core.services.health.so_ratings import (
    min_score_for_opportunity_letter,
    min_score_for_stability_letter,
    opportunity_grade_at_least,
    score_to_opportunity_grade,
    score_to_stability_grade,
    so_floor_display,
    stability_grade_at_least,
)

if TYPE_CHECKING:
    from core.models import Assessment, Discovery

# Risk band keys (profile.risk field choices).
RISK_LEVELS: Tuple[str, ...] = (
    "CONSERVATIVE",
    "MODERATE",
    "AGGRESSIVE",
    "RECKLESS",
)

# Dual-floor gates: min SO grade pair per fund (see so_ratings.py ladders).
RISK_MATRIX: Dict[str, Dict[str, str]] = {
    "CONSERVATIVE": {"min_stability": "B", "min_opportunity": "B"},  # BB — strong business + strong setup
    "MODERATE": {"min_stability": "C", "min_opportunity": "C"},      # CC — good on both axes
    "AGGRESSIVE": {"min_stability": "D", "min_opportunity": "C"},    # DC — weaker business, decent+ opp
    "RECKLESS": {"min_stability": "D", "min_opportunity": "D"},      # DD — broadest band, within reason
}

# Layer-2 weights (component scorers unchanged; tune here).
STABILITY_WEIGHTS: Dict[str, float] = {
    "fin_debt_to_equity": 0.25,
    "fin_fcf_margin": 0.20,
    "fin_operating_margin": 0.18,
    "durability": 0.32,
    "sector": 0.05,
}

OPPORTUNITY_WEIGHTS: Dict[str, float] = {
    "price": 0.25,
    "valuation": 0.28,
    "intrinsic": 0.22,
    "consensus": 0.15,
    "fin_growth": 0.10,
}


def risk_floors_for(risk: str) -> Dict[str, Any]:
    """Return letter floors, derived numeric mins, and concatenated SO floor display."""
    if risk not in RISK_MATRIX:
        raise KeyError(f"Unknown risk band: {risk!r}")
    floors = dict(RISK_MATRIX[risk])
    min_stab = floors["min_stability"]
    min_opp = floors["min_opportunity"]
    floors["min_stability_score"] = min_score_for_stability_letter(min_stab)
    floors["min_opportunity_score"] = min_score_for_opportunity_letter(min_opp)
    floors["so_floor_display"] = so_floor_display(min_stab, min_opp)
    return floors


def opportunity_adjusted(
    opportunity: Optional[float],
    weight: Optional[Decimal | float],
) -> Optional[float]:
    if opportunity is None:
        return None
    w = float(weight) if weight is not None else 1.0
    return round(float(opportunity) * w, 1)


def _metric_score(result: Any, key: str) -> Optional[float]:
    for m in getattr(result, "metrics", []) or []:
        if m.key == key and m.score is not None:
            return float(m.score)
    return None


def _weighted_blend(parts: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    num = 0.0
    den = 0.0
    for key, w in weights.items():
        sc = parts.get(key)
        if sc is None:
            continue
        num += float(sc) * w
        den += w
    if den <= 0:
        return None
    return round(num / den, 1)


def _fin_growth_opportunity(financial: Any) -> Optional[float]:
    rev = _metric_score(financial, "revenue_growth")
    eps = _metric_score(financial, "eps_growth")
    roe = _metric_score(financial, "return_on_equity")
    vals = [v for v in (rev, eps, roe) if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 1)


def _stored_score(assessment: "Assessment", key: str) -> Optional[float]:
    raw = getattr(assessment, key, None)
    return float(raw) if raw is not None else None


def _stab_parts_from_results(
    results: Dict[str, Any],
    *,
    durability: Optional[float],
) -> Dict[str, Optional[float]]:
    fin = results.get("financial")
    sector = results.get("sector")
    sector_score = getattr(sector, "score", None) if sector is not None else None
    return {
        "sector": float(sector_score) if sector_score is not None else None,
        "fin_debt_to_equity": _metric_score(fin, "debt_to_equity") if fin else None,
        "fin_fcf_margin": _metric_score(fin, "fcf_margin") if fin else None,
        "fin_operating_margin": _metric_score(fin, "operating_margin") if fin else None,
        "durability": durability,
    }


def _opp_parts_from_results(
    symbol: str,
    results: Dict[str, Any],
    *,
    durability: Optional[float],
) -> Dict[str, Optional[float]]:
    fin = results.get("financial")

    def _comp_score(key: str) -> Optional[float]:
        comp = results.get(key)
        sc = getattr(comp, "score", None) if comp is not None else None
        return float(sc) if sc is not None else None

    return adjust_opportunity_parts(
        symbol,
        {
            "price": _comp_score("price"),
            "valuation": _comp_score("valuation"),
            "intrinsic": _comp_score("intrinsic"),
            "consensus": _comp_score("consensus"),
            "fin_growth": _fin_growth_opportunity(fin) if fin else None,
        },
        durability=durability,
    )


def compute_so_snapshot(
    symbol: str,
    results: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """
    Compute SO axis scores and sub-metrics from v2 component scorer results.
    Called once at assessment creation; persisted on Assessment.
    """
    sym = (symbol or "").strip().upper()
    durability = score_business_durability(sym)
    stab_parts = _stab_parts_from_results(results, durability=durability)
    opp_parts = _opp_parts_from_results(sym, results, durability=durability)
    stability = _weighted_blend(stab_parts, STABILITY_WEIGHTS)
    opportunity = _weighted_blend(opp_parts, OPPORTUNITY_WEIGHTS)
    return {
        "stability": stability,
        "opportunity": opportunity,
        "stab_debt_to_equity": stab_parts.get("fin_debt_to_equity"),
        "stab_fcf_margin": stab_parts.get("fin_fcf_margin"),
        "stab_operating_margin": stab_parts.get("fin_operating_margin"),
        "stab_durability": stab_parts.get("durability"),
        "opp_fin_growth": opp_parts.get("fin_growth"),
        "opp_price_blend": opp_parts.get("price"),
        "opp_valuation_blend": opp_parts.get("valuation"),
    }


def _stab_parts_from_assessment(assessment: "Assessment") -> Dict[str, Optional[float]]:
    """Stability blend inputs from persisted Assessment columns only."""
    debt = _stored_score(assessment, "stab_debt_to_equity")
    fcf = _stored_score(assessment, "stab_fcf_margin")
    op_margin = _stored_score(assessment, "stab_operating_margin")
    durability = _stored_score(assessment, "stab_durability")
    sector = _stored_score(assessment, "sector")
    financial = _stored_score(assessment, "financial")

    if debt is None and fcf is None and op_margin is None and financial is not None:
        debt = fcf = op_margin = financial
    if durability is None:
        durability = 55.0

    return {
        "sector": sector,
        "fin_debt_to_equity": debt,
        "fin_fcf_margin": fcf,
        "fin_operating_margin": op_margin,
        "durability": durability,
    }


def _opp_parts_from_assessment(
    assessment: "Assessment",
    symbol: str,
) -> Dict[str, Optional[float]]:
    """Opportunity blend inputs from persisted Assessment columns only."""
    durability = _stored_score(assessment, "stab_durability")
    raw = {
        "price": _stored_score(assessment, "opp_price_blend") or _stored_score(assessment, "price"),
        "valuation": _stored_score(assessment, "opp_valuation_blend")
        or _stored_score(assessment, "valuation"),
        "intrinsic": _stored_score(assessment, "intrinsic"),
        "consensus": _stored_score(assessment, "consensus"),
        "fin_growth": _stored_score(assessment, "opp_fin_growth"),
    }
    if durability is not None:
        return adjust_opportunity_parts(symbol, raw, durability=durability)
    return raw


def axes_from_assessment(
    assessment: "Assessment",
    symbol: str,
    *,
    fin_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Read persisted SO snapshot; never calls yfinance on the UI path."""
    stored_stab = _stored_score(assessment, "stability")
    stored_opp = _stored_score(assessment, "opportunity")
    if stored_stab is not None and stored_opp is not None:
        return stored_stab, stored_opp

    sym = (symbol or "").strip().upper()
    stab_parts = _stab_parts_from_assessment(assessment)
    opp_parts = _opp_parts_from_assessment(assessment, sym)
    return (
        _weighted_blend(stab_parts, STABILITY_WEIGHTS),
        _weighted_blend(opp_parts, OPPORTUNITY_WEIGHTS),
    )


def persist_so_on_assessment(assessment: "Assessment") -> bool:
    """
    Recompute SO snapshot from live scorers and save on Assessment (backfill / repair).
    Uses yfinance once per call — not for request-time UI.
    """
    from core.services.health.assess import run_component_results

    stock = getattr(assessment, "stock", None)
    symbol = (getattr(stock, "symbol", None) or "").strip().upper()
    if not symbol:
        return False

    results = run_component_results(symbol)
    so = compute_so_snapshot(symbol, results)
    if so.get("stability") is None and so.get("opportunity") is None:
        return False

    def _dec(key: str):
        val = so.get(key)
        if val is None:
            return None
        return Decimal(str(round(float(val), 1)))

    assessment.stability = _dec("stability")
    assessment.opportunity = _dec("opportunity")
    assessment.stab_debt_to_equity = _dec("stab_debt_to_equity")
    assessment.stab_fcf_margin = _dec("stab_fcf_margin")
    assessment.stab_operating_margin = _dec("stab_operating_margin")
    assessment.stab_durability = _dec("stab_durability")
    assessment.opp_fin_growth = _dec("opp_fin_growth")
    assessment.opp_price_blend = _dec("opp_price_blend")
    assessment.opp_valuation_blend = _dec("opp_valuation_blend")
    assessment.save(
        update_fields=[
            "stability",
            "opportunity",
            "stab_debt_to_equity",
            "stab_fcf_margin",
            "stab_operating_margin",
            "stab_durability",
            "opp_fin_growth",
            "opp_price_blend",
            "opp_valuation_blend",
        ]
    )
    return True


def discovery_axes(
    discovery: Optional["Discovery"],
    *,
    fin_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    if discovery is None or discovery.assessment is None:
        return None, None
    symbol = discovery.stock.symbol if discovery.stock else ""
    return axes_from_assessment(discovery.assessment, symbol, fin_cache=fin_cache)


def passes_risk_gate(
    stability: Optional[float],
    opportunity: Optional[float],
    risk: str,
    *,
    weight: Optional[Decimal | float] = None,
) -> bool:
    """BUY if stability and weight-adjusted opportunity SO grades meet risk-band floors."""
    floors = risk_floors_for(risk)
    opp_adj = opportunity_adjusted(opportunity, weight)
    stab_grade = score_to_stability_grade(stability)
    opp_grade = score_to_opportunity_grade(opp_adj)
    if stab_grade is None or opp_grade is None:
        return False
    return (
        stability_grade_at_least(stab_grade.letter, floors["min_stability"])
        and opportunity_grade_at_least(opp_grade.letter, floors["min_opportunity"])
    )


def discovery_passes_risk_gate(discovery: Optional["Discovery"], risk: str) -> bool:
    stability, opportunity = discovery_axes(discovery)
    weight = discovery.weight if discovery is not None else None
    return passes_risk_gate(stability, opportunity, risk, weight=weight)


def risk_fit_for(
    stability: Optional[float],
    opportunity: Optional[float],
    risk: str,
    *,
    weight: Optional[Decimal | float] = None,
) -> str:
    return "BUY" if passes_risk_gate(stability, opportunity, risk, weight=weight) else "AVOID"


def risk_fit_all(
    stability: Optional[float],
    opportunity: Optional[float],
    *,
    weight: Optional[Decimal | float] = None,
) -> Dict[str, str]:
    return {
        risk: risk_fit_for(stability, opportunity, risk, weight=weight)
        for risk in RISK_LEVELS
    }


def interpret_axes(stability: Optional[float], opportunity: Optional[float]) -> str:
    from core.services.health.so_ratings import (
        LETTER_RANK,
        score_to_opportunity_grade,
        score_to_stability_grade,
    )

    if stability is None and opportunity is None:
        return "Insufficient data for stability or opportunity."
    stab_g = score_to_stability_grade(stability)
    opp_g = score_to_opportunity_grade(opportunity)
    s = stab_g.letter if stab_g else "D"
    o = opp_g.letter if opp_g else "D"
    sr = LETTER_RANK.get(s, 3)
    or_ = LETTER_RANK.get(o, 3)

    if s == "A" and o == "A":
        return "Exceptional stability and opportunity."
    if s == "A" and o == "B":
        return "Very strong stability with solid opportunity."
    if s == "A" and or_ <= LETTER_RANK["D"]:
        return "Strong stability profile; opportunity is the weaker leg."
    if sr >= LETTER_RANK["B"] and or_ >= LETTER_RANK["B"]:
        return "Solid setup with meaningful opportunity."
    if sr >= LETTER_RANK["C"] and or_ >= LETTER_RANK["C"]:
        return "Balanced profile — moderate stability and opportunity."
    if sr <= LETTER_RANK["E"] and or_ >= LETTER_RANK["C"]:
        return "Weaker business quality; opportunity-driven or turnaround setup."
    if sr <= LETTER_RANK["D"] and or_ <= LETTER_RANK["D"]:
        return "Weak on both stability and opportunity."
    if sr >= LETTER_RANK["C"]:
        return "Adequate stability; opportunity is the weaker leg."
    return "Mixed profile — review component details."
