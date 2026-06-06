"""
Risk-band dual-floor buy matrix: stability + opportunity (× discovery.weight).

Replaces legacy min grade letter / single composite min_score on Profile.
Floors are SO letter grades; numeric mins are derived for logging.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from core.services.health.durability import score_business_durability
from core.services.health.financial import score_financial_health
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
    "CONSERVATIVE": {"min_stability": "A", "min_opportunity": "B"},  # AB — fortress + attractive setup
    "MODERATE": {"min_stability": "B", "min_opportunity": "C"},      # BC — good business + good opportunity
    "AGGRESSIVE": {"min_stability": "D", "min_opportunity": "B"},    # DB — weaker business, compelling opp
    "RECKLESS": {"min_stability": "D", "min_opportunity": "D"},      # DD — broadest band, within reason
}

# Layer-2 weights (component scorers unchanged; tune here).
STABILITY_WEIGHTS: Dict[str, float] = {
    "fin_debt_to_equity": 0.30,
    "fin_fcf_margin": 0.25,
    "fin_operating_margin": 0.20,
    "durability": 0.15,
    "sector": 0.10,
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


def component_results_from_assessment(
    assessment: "Assessment",
    symbol: str,
    *,
    fin_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Stored Assessment columns + live financial sub-metrics (not persisted on Assessment).
    """
    sym = (symbol or "").strip().upper()
    cache = fin_cache if fin_cache is not None else {}
    if sym not in cache:
        cache[sym] = score_financial_health(sym)
    fin = cache[sym]
    stored_fin = _stored_score(assessment, "financial")
    if stored_fin is not None:
        fin.score = stored_fin

    class _Comp:
        def __init__(self, score: Optional[float]):
            self.score = score
            self.metrics: List[Any] = []

    results: Dict[str, Any] = {"financial": fin}
    for key in ("valuation", "intrinsic", "price", "consensus", "sector"):
        results[key] = _Comp(_stored_score(assessment, key))
    return results


def axes_from_assessment(
    assessment: "Assessment",
    symbol: str,
    *,
    fin_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    results = component_results_from_assessment(assessment, symbol, fin_cache=fin_cache)
    fin = results["financial"]
    stab_parts = {
        "sector": results["sector"].score,
        "fin_debt_to_equity": _metric_score(fin, "debt_to_equity"),
        "fin_fcf_margin": _metric_score(fin, "fcf_margin"),
        "fin_operating_margin": _metric_score(fin, "operating_margin"),
        "durability": score_business_durability(symbol),
    }
    opp_parts = {
        "price": results["price"].score,
        "valuation": results["valuation"].score,
        "intrinsic": results["intrinsic"].score,
        "consensus": results["consensus"].score,
        "fin_growth": _fin_growth_opportunity(fin),
    }
    return (
        _weighted_blend(stab_parts, STABILITY_WEIGHTS),
        _weighted_blend(opp_parts, OPPORTUNITY_WEIGHTS),
    )


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
