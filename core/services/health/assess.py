"""
Persist health v2 component scores as Assessment rows.
Used by AdvisorBase.assess_stock and the health_score lab.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

from core.services.health.consensus import score_consensus_health
from core.services.health.financial import score_financial_health
from core.services.health.intrinsic import score_intrinsic_health
from core.services.health.price import score_price_health
from core.services.health.sector import score_sector_health
from core.services.health.valuation import score_valuation_health

if TYPE_CHECKING:
    from core.models import Assessment, Discovery, Stock

# Final v2 model weights (sum to 1.0); keep in sync with health_score.py _COMPONENT_SPECS.
COMPONENT_MODEL_WEIGHTS: Dict[str, Decimal] = {
    "financial": Decimal("0.20"),
    "valuation": Decimal("0.20"),
    "intrinsic": Decimal("0.15"),
    "price": Decimal("0.20"),
    "consensus": Decimal("0.15"),
    "sector": Decimal("0.10"),
}

COMPONENT_SCORERS: List[Tuple[str, Callable[[str], Any]]] = [
    ("financial", score_financial_health),
    ("valuation", score_valuation_health),
    ("intrinsic", score_intrinsic_health),
    ("price", score_price_health),
    ("consensus", score_consensus_health),
    ("sector", score_sector_health),
]


def _to_decimal(score: Optional[float]) -> Optional[Decimal]:
    if score is None:
        return None
    return Decimal(str(round(float(score), 1)))


def composite_from_scores(scores: Dict[str, Optional[float]]) -> Optional[Decimal]:
    """
    Weighted mean over components that returned a score; renormalizes if some are missing.

    Missing valuation is treated as 0 (full 20% weight) so unvaluable names are not
    boosted by renormalization. Assessment.valuation stays NULL for UI display (—).
    """
    num = Decimal("0")
    den = Decimal("0")
    for key, w in COMPONENT_MODEL_WEIGHTS.items():
        raw = scores.get(key)
        if raw is None:
            if key == "valuation":
                raw = 0.0
            else:
                continue
        num += Decimal(str(float(raw))) * w
        den += w
    if den <= 0:
        return None
    return (num / den).quantize(Decimal("0.1"))


def discovery_adjusted_score(discovery: Optional["Discovery"]) -> Optional[Decimal]:
    """
    v2 composite × discovery.weight (catalyst multiplier on Discovery).
    Computed at analysis time, not stored on Assessment.
    """
    if discovery is None:
        return None
    assessment = discovery.assessment
    if assessment is None or assessment.score is None:
        return None
    w = discovery.weight if discovery.weight is not None else Decimal("1.0")
    return (assessment.score * w).quantize(Decimal("0.1"))


def run_component_scores(
    symbol: str,
    *,
    components: Optional[Iterable[str]] = None,
) -> Dict[str, Optional[float]]:
    """
    Run v2 component scorers; return {key: score or None}.

    components: if set, only run these keys (others omitted from dict).
    Default None runs all components (core / discovery behavior).
    """
    sym = (symbol or "").strip().upper()
    if components is None:
        active = {key for key, _ in COMPONENT_SCORERS}
    else:
        active = {c.strip().lower() for c in components if c}
        unknown = active - set(COMPONENT_MODEL_WEIGHTS.keys())
        if unknown:
            raise ValueError(f"Unknown assessment components: {sorted(unknown)}")

    out: Dict[str, Optional[float]] = {}
    for key, scorer in COMPONENT_SCORERS:
        if key not in active:
            continue
        try:
            result = scorer(sym)
            out[key] = result.score if result is not None else None
        except Exception:
            out[key] = None
    return out


def run_component_results(symbol: str) -> Dict[str, Any]:
    """Run all v2 component scorers; return {key: result object}."""
    sym = (symbol or "").strip().upper()
    results: Dict[str, Any] = {}
    for key, scorer in COMPONENT_SCORERS:
        try:
            results[key] = scorer(sym)
        except Exception:
            results[key] = None
    return results


def create_assessment_for_stock(stock: "Stock") -> Optional["Assessment"]:
    """
    Score stock via v2 components, compute SO snapshot, and persist an Assessment row.
    Returns None if every component failed to produce a score.
    """
    from core.models import Assessment
    from core.services.health.risk_matrix import compute_so_snapshot

    results = run_component_results(stock.symbol)
    scores = {
        key: (getattr(results.get(key), "score", None) if results.get(key) is not None else None)
        for key, _ in COMPONENT_SCORERS
    }
    if not any(v is not None for v in scores.values()):
        return None

    composite = composite_from_scores(scores)
    so = compute_so_snapshot(stock.symbol, results)

    return Assessment.objects.create(
        stock=stock,
        financial=_to_decimal(scores["financial"]),
        valuation=_to_decimal(scores["valuation"]),
        intrinsic=_to_decimal(scores["intrinsic"]),
        price=_to_decimal(scores["price"]),
        consensus=_to_decimal(scores["consensus"]),
        sector=_to_decimal(scores["sector"]),
        score=composite,
        stability=_to_decimal(so.get("stability")),
        opportunity=_to_decimal(so.get("opportunity")),
        stab_debt_to_equity=_to_decimal(so.get("stab_debt_to_equity")),
        stab_fcf_margin=_to_decimal(so.get("stab_fcf_margin")),
        stab_operating_margin=_to_decimal(so.get("stab_operating_margin")),
        stab_durability=_to_decimal(so.get("stab_durability")),
        opp_fin_growth=_to_decimal(so.get("opp_fin_growth")),
        opp_price_blend=_to_decimal(so.get("opp_price_blend")),
        opp_valuation_blend=_to_decimal(so.get("opp_valuation_blend")),
    )
