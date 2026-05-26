"""
Unified discovery scoring metadata for web templates, API, and analysis.

v2: Assessment component scores + composite + discovery.weight-adjusted score + letter ratings.
v1 fallback: nested health_record_* payload when no assessment is linked.

Web UI: render v2 with {% include 'core/includes/assessment_block.html' with scoring=... %}.
Do not embed discovery provenance or v1 health in that partial.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.health_display import health_record_payload, health_record_template_context
from core.services.health.assess import COMPONENT_MODEL_WEIGHTS
from core.services.health.ratings import score_to_rating

if TYPE_CHECKING:
    from core.models import Assessment, Discovery

# Display labels; weights from assess.COMPONENT_MODEL_WEIGHTS (single source of truth).
COMPONENT_LABELS: Dict[str, str] = {
    "financial": "Financial health",
    "valuation": "Valuation",
    "intrinsic": "Intrinsic valuation",
    "price": "Price position",
    "consensus": "Analyst consensus",
    "sector": "Sector / industry",
}

COMPONENT_ORDER: tuple[str, ...] = tuple(COMPONENT_LABELS.keys())


def _float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _rating_dict(score: Optional[float]) -> Optional[Dict[str, Any]]:
    rating = score_to_rating(score)
    if rating is None:
        return None
    return rating.to_dict()


def _grade_display(rating: Optional[Dict[str, Any]]) -> Optional[str]:
    if not rating:
        return None
    letter = rating.get("letter")
    label = rating.get("label")
    if letter and label:
        return f"{letter} — {label}"
    return rating.get("display_short")


def _weight_display(weight: Optional[float]) -> str:
    if weight is None:
        return "—"
    return f"×{weight:.2f}"


def _build_summary(
    *,
    assessment_score: Optional[float],
    discovery_weight: Optional[float],
    discovery_score: Optional[float],
    grade: Optional[Dict[str, Any]],
    grade_adjusted: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """UI/API summary: assessment score, discovery weight (always shown), grade on composite."""
    adjusted_grade = None
    if (
        grade
        and grade_adjusted
        and discovery_weight is not None
        and abs(discovery_weight - 1.0) > 1e-9
        and grade_adjusted.get("letter") != grade.get("letter")
    ):
        adjusted_grade = grade_adjusted

    return {
        "assessment_score": assessment_score,
        "discovery_weight": discovery_weight,
        "discovery_weight_display": _weight_display(discovery_weight),
        "discovery_score": discovery_score,
        "grade": grade,
        "grade_display": _grade_display(grade),
        "adjusted_grade": adjusted_grade,
    }


def assessment_payload(
    assessment: "Assessment",
    discovery_weight: Optional[Decimal] = None,
) -> Dict[str, Any]:
    """v2 scoring from a persisted Assessment row (no Discovery required)."""
    composite = _float_or_none(assessment.score)
    weight = float(discovery_weight) if discovery_weight is not None else None
    adjusted = None
    if composite is not None and weight is not None:
        adjusted = float(
            (Decimal(str(composite)) * Decimal(str(weight))).quantize(Decimal("0.1"))
        )

    components: List[Dict[str, Any]] = []
    for key in COMPONENT_ORDER:
        raw = getattr(assessment, key, None)
        score = _float_or_none(raw)
        w = float(COMPONENT_MODEL_WEIGHTS[key])
        components.append({
            "key": key,
            "label": COMPONENT_LABELS[key],
            "weight": w,
            "weight_percent": int(round(w * 100)),
            "score": score,
        })

    rating = _rating_dict(composite)
    rating_adjusted = _rating_dict(adjusted)

    payload = {
        "source": "v2",
        "assessment_id": assessment.id,
        "composite_score": composite,
        "adjusted_score": adjusted,
        "discovery_weight": weight,
        "rating": rating,
        "rating_adjusted": rating_adjusted,
        "components": components,
    }
    payload["summary"] = _build_summary(
        assessment_score=composite,
        discovery_weight=weight,
        discovery_score=adjusted,
        grade=rating,
        grade_adjusted=rating_adjusted,
    )
    return payload


def discovery_scoring_context(
    discovery: Optional["Discovery"],
    *,
    for_api: bool = False,
) -> Dict[str, Any]:
    """
    Single entry point for discovery scoring metadata.

    Prefer v2 when discovery.assessment is set; otherwise expose v1 health only.
    """
    empty: Dict[str, Any] = {
        "source": None,
        "composite_score": None,
        "adjusted_score": None,
        "discovery_weight": None,
        "rating": None,
        "rating_adjusted": None,
        "components": [],
        "summary": None,
        "headline_display": "—",
        "health_v1": None,
    }
    if discovery is None:
        return empty

    weight = discovery.weight if discovery.weight is not None else Decimal("1.0")
    assessment = discovery.assessment

    if assessment is not None:
        ctx = assessment_payload(assessment, discovery_weight=weight)
        ctx["headline_display"] = _headline_from_scoring(ctx)
        if discovery.health:
            ctx["health_v1"] = (
                health_record_payload(discovery.health)
                if for_api
                else health_record_template_context(discovery.health)
            )
        else:
            ctx["health_v1"] = None
        return ctx

    # v1-only path
    health = discovery.health
    composite = _float_or_none(health.score) if health else None
    rating = _rating_dict(composite)
    health_v1 = None
    if health:
        health_v1 = (
            health_record_payload(health)
            if for_api
            else health_record_template_context(health)
        )

    w = float(weight)
    ctx = {
        "source": "v1" if health else None,
        "assessment_id": None,
        "composite_score": composite,
        "adjusted_score": None,
        "discovery_weight": w,
        "rating": rating,
        "rating_adjusted": None,
        "components": [],
        "headline_display": _headline_v1(composite),
        "health_v1": health_v1,
    }
    if health:
        ctx["summary"] = _build_summary(
            assessment_score=composite,
            discovery_weight=w,
            discovery_score=composite,
            grade=rating,
        )
    else:
        ctx["summary"] = None
    return ctx


def _headline_from_scoring(ctx: Dict[str, Any]) -> str:
    """List/summary display: discovery-adjusted numeric score (matches Assessment header)."""
    adjusted = ctx.get("adjusted_score")
    if adjusted is not None:
        return f"{adjusted:.1f}"
    composite = ctx.get("composite_score")
    if composite is not None:
        return f"{composite:.1f}"
    return "—"


def _headline_v1(score: Optional[float]) -> str:
    if score is None:
        return "—"
    if abs(score) < 1e-9:
        return "AVOID"
    return f"{score:.1f}"


def discovery_list_score_column(scoring: Dict[str, Any]) -> Dict[str, str]:
    """Kicker + value for list score column (v2 → Grade letter when available)."""
    if scoring.get("source") == "v2":
        summary = scoring.get("summary") or {}
        grade = summary.get("adjusted_grade") or summary.get("grade")
        if isinstance(grade, dict) and grade.get("letter"):
            return {"kicker": "Grade", "value": grade["letter"]}
        headline = (scoring.get("headline_display") or "").strip()
        if headline and headline != "—":
            return {"kicker": "Score", "value": headline}
        return {"kicker": "Score", "value": "—"}

    from core.health_display import format_health_score

    comp = scoring.get("composite_score")
    if comp is not None:
        return {"kicker": "Score", "value": format_health_score(float(comp))}
    return {"kicker": "Score", "value": "—"}


def discovery_outcome_score(scoring: Dict[str, Any]) -> Optional[float]:
    """Numeric score for outcome heuristics (prefer discovery-adjusted when present)."""
    if not scoring:
        return None
    adjusted = scoring.get("adjusted_score")
    if adjusted is not None:
        return float(adjusted)
    composite = scoring.get("composite_score")
    if composite is not None:
        return float(composite)
    return None


def render_assessment_block_html(discovery: Optional["Discovery"]) -> str:
    """Server-render assessment_block.html for AJAX detail panels (e.g. holdings history)."""
    from django.template.loader import render_to_string

    scoring = discovery_scoring_context(discovery)
    if scoring.get("source") != "v2":
        return ""
    return render_to_string(
        "core/includes/assessment_block.html",
        {"scoring": scoring},
    )


def discovery_scoring_log_fields(ctx: Dict[str, Any]) -> str:
    """Compact string for analyze_discovery log lines."""
    parts = [
        f"source={ctx.get('source')}",
        f"composite={ctx.get('composite_score')}",
        f"adjusted={ctx.get('adjusted_score')}",
    ]
    rating = ctx.get("rating")
    if rating:
        parts.append(f"rating={rating.get('letter')}")
    return " ".join(parts)
