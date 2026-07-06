"""Multi-factor health scoring (v2)."""

from core.services.health.assess import (
    composite_from_scores,
    create_assessment_for_stock,
    discovery_adjusted_score,
    run_component_scores,
)
from core.services.health.risk_matrix import (
    RISK_MATRIX,
    RISK_LEVELS,
    RISK_COMPOSITE_FLOOR,
    composite_floor_score,
    compute_so_snapshot,
    discovery_axes,
    discovery_passes_risk_gate,
    discovery_so_pair,
    risk_fit_all,
    risk_floors_for,
    so_composite_from_scores,
    so_gate_fail_display,
    so_pair_from_scores,
)
from core.services.health.consensus import score_consensus_health
from core.services.health.ratings import (
    HealthRating,
    grade_at_least,
    min_composite_for_letter,
    score_to_rating,
)
from core.services.health.so_ratings import (
    OPPORTUNITY_BANDS,
    SOGrade,
    STABILITY_BANDS,
    opportunity_grade_at_least,
    score_to_opportunity_grade,
    score_to_stability_grade,
    so_grade_pair,
    so_composite_from_grades,
    composite_letter_from_score,
    composite_score_from_letters,
    letter_axis_score,
    stability_grade_at_least,
)
from core.services.health.distress import adjust_opportunity_parts
from core.services.health.diagnostic import (
    DiagnosticResult,
    analyze_symbol,
    diagnostic_to_dict,
    filter_buy_ready,
)
from core.services.health.durability import score_business_durability
from core.services.health.financial import score_financial_health
from core.services.health.intrinsic import score_intrinsic_health
from core.services.health.price import score_price_health
from core.services.health.sector import score_sector_health
from core.services.health.valuation import score_valuation_health

__all__ = [
    "HealthRating",
    "grade_at_least",
    "min_composite_for_letter",
    "RISK_MATRIX",
    "RISK_LEVELS",
    "RISK_COMPOSITE_FLOOR",
    "composite_from_scores",
    "discovery_adjusted_score",
    "compute_so_snapshot",
    "discovery_axes",
    "discovery_passes_risk_gate",
    "discovery_so_pair",
    "composite_floor_score",
    "so_composite_from_scores",
    "risk_fit_all",
    "risk_floors_for",
    "so_gate_fail_display",
    "so_pair_from_scores",
    "create_assessment_for_stock",
    "run_component_scores",
    "score_consensus_health",
    "score_to_rating",
    "SOGrade",
    "STABILITY_BANDS",
    "OPPORTUNITY_BANDS",
    "score_to_stability_grade",
    "score_to_opportunity_grade",
    "stability_grade_at_least",
    "opportunity_grade_at_least",
    "so_grade_pair",
    "so_composite_from_grades",
    "composite_letter_from_score",
    "composite_score_from_letters",
    "letter_axis_score",
    "adjust_opportunity_parts",
    "DiagnosticResult",
    "analyze_symbol",
    "diagnostic_to_dict",
    "filter_buy_ready",
    "score_business_durability",
    "score_financial_health",
    "score_intrinsic_health",
    "score_price_health",
    "score_sector_health",
    "score_valuation_health",
]
