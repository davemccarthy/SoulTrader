"""
Stability / Opportunity (SO) letter grades — separate ladders per axis.

A is always better than B within each ladder; stability A and opportunity A
use different numeric cutoffs. Display pair as concatenated letters, e.g. AB.

Separate from legacy composite RATING_BANDS in ratings.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Stability — A is rare (90+ fortress); cutoffs align with matrix floors.
STABILITY_BANDS: List[Tuple[float, str, str]] = [
    (90.0, "A", "Fortress"),
    (75.0, "B", "Strong"),
    (70.0, "C", "Good"),
    (50.0, "D", "Adequate"),
    (25.0, "E", "Fragile"),
    (15.0, "F", "Distressed"),
]

# Opportunity — D/E split lowered so ~45–55 names land D not cliff-edge E.
# A 80+, B 65+, C 55+, D 45+, E 35+, F <35
OPPORTUNITY_BANDS: List[Tuple[float, str, str]] = [
    (80.0, "A", "Exceptional"),
    (65.0, "B", "Strong"),
    (55.0, "C", "Good"),
    (45.0, "D", "Limited but respectable"),
    (35.0, "E", "Weak"),
    (0.0, "F", "Very little reason to buy"),
]

LETTER_RANK: Dict[str, int] = {
    "A": 6,
    "B": 5,
    "C": 4,
    "D": 3,
    "E": 2,
    "F": 1,
}

# Axis letters (S/O pair legs) → numeric score for composite: round((S + O) / 2).
AXIS_LETTER_SCORE: Dict[str, int] = {
    "A": 10,
    "B": 8,
    "C": 6,
    "D": 4,
    "E": 2,
    "F": 0,
}

# Composite SO letter from rounded mean of axis scores.
COMPOSITE_SCORE_TO_LETTER: Dict[int, str] = {
    10: "A",
    9: "B+",
    8: "B",
    7: "C+",
    6: "C",
    5: "D+",
    4: "D",
    3: "E+",
    2: "E",
    1: "F+",
    0: "F",
}


@dataclass(frozen=True)
class SOGrade:
    letter: str
    label: str
    min_score: float
    axis: str  # "stability" | "opportunity"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "letter": self.letter,
            "label": self.label,
            "min_score": self.min_score,
            "axis": self.axis,
        }


def _min_score_for_letter(letter: str, bands: List[Tuple[float, str, str]]) -> float:
    key = (letter or "").strip().upper()
    for min_score, band_letter, _label in bands:
        if band_letter == key:
            return min_score
    raise ValueError(f"Unknown grade letter: {letter!r}")


def min_score_for_stability_letter(letter: str) -> float:
    return _min_score_for_letter(letter, STABILITY_BANDS)


def min_score_for_opportunity_letter(letter: str) -> float:
    return _min_score_for_letter(letter, OPPORTUNITY_BANDS)


def _grade_at_least(letter: str, min_letter: str) -> bool:
    a = LETTER_RANK.get((letter or "").strip().upper())
    b = LETTER_RANK.get((min_letter or "").strip().upper())
    if a is None or b is None:
        return False
    return a >= b


def stability_grade_at_least(letter: str, min_letter: str) -> bool:
    return _grade_at_least(letter, min_letter)


def opportunity_grade_at_least(letter: str, min_letter: str) -> bool:
    return _grade_at_least(letter, min_letter)


def _score_to_grade(
    score: Optional[float],
    bands: List[Tuple[float, str, str]],
    axis: str,
) -> Optional[SOGrade]:
    if score is None:
        return None
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    for min_score, letter, label in bands:
        if s >= min_score:
            return SOGrade(letter=letter, label=label, min_score=min_score, axis=axis)
    last = bands[-1]
    return SOGrade(letter=last[1], label=last[2], min_score=last[0], axis=axis)


def score_to_stability_grade(score: Optional[float]) -> Optional[SOGrade]:
    return _score_to_grade(score, STABILITY_BANDS, "stability")


def score_to_opportunity_grade(score: Optional[float]) -> Optional[SOGrade]:
    return _score_to_grade(score, OPPORTUNITY_BANDS, "opportunity")


def so_grade_pair(
    stability_grade: Optional[SOGrade],
    opportunity_grade: Optional[SOGrade],
) -> Optional[str]:
    """Concatenated SO grade, e.g. AB (stability A, opportunity B)."""
    if stability_grade is None or opportunity_grade is None:
        return None
    return f"{stability_grade.letter}{opportunity_grade.letter}"


def letter_axis_score(letter: Optional[str]) -> Optional[int]:
    """Numeric axis score for composite SO (A=10 … F=0)."""
    if not letter:
        return None
    return AXIS_LETTER_SCORE.get(letter.strip().upper())


def composite_score_from_letters(
    stability_letter: Optional[str],
    opportunity_letter: Optional[str],
) -> Optional[int]:
    """Rounded mean of axis letter scores, e.g. AB → 9."""
    s = letter_axis_score(stability_letter)
    o = letter_axis_score(opportunity_letter)
    if s is None or o is None:
        return None
    return round((s + o) / 2)


def composite_letter_from_score(score: int) -> str:
    return COMPOSITE_SCORE_TO_LETTER.get(int(score), "F")


def so_composite_from_grades(
    stability_grade: Optional[SOGrade],
    opportunity_grade: Optional[SOGrade],
) -> Optional[Dict[str, Any]]:
    """
    Single-letter SO grade from stability + opportunity pair.
    Uses weight-adjusted opportunity grade when supplied as second arg.
  """
    if stability_grade is None or opportunity_grade is None:
        return None
    score = composite_score_from_letters(
        stability_grade.letter,
        opportunity_grade.letter,
    )
    if score is None:
        return None
    letter = composite_letter_from_score(score)
    return {
        "letter": letter,
        "score": score,
        "pair": so_grade_pair(stability_grade, opportunity_grade),
    }


def so_floor_display(min_stability: str, min_opportunity: str) -> str:
    """Risk-band floor pair on respective ladders, e.g. CD or EC."""
    return f"{min_stability.strip().upper()}{min_opportunity.strip().upper()}"


def opportunity_grade_upgraded(
    base_grade: Optional[Dict[str, Any]],
    adjusted_grade: Optional[Dict[str, Any]],
) -> bool:
    """True when discovery weight lifts opportunity to a strictly better letter."""
    if not base_grade or not adjusted_grade:
        return False
    base_letter = (base_grade.get("letter") or "").strip().upper()
    adj_letter = (adjusted_grade.get("letter") or "").strip().upper()
    base_rank = LETTER_RANK.get(base_letter)
    adj_rank = LETTER_RANK.get(adj_letter)
    if base_rank is None or adj_rank is None:
        return False
    return adj_rank > base_rank


def opportunity_upgrade_display(
    opportunity_adjusted: Optional[float],
    weight_display: Optional[str],
) -> str:
    """Score cell for Upgrade row, e.g. '(×1.15) 81.1'."""
    if opportunity_adjusted is None:
        return "—"
    score_s = f"{float(opportunity_adjusted):.1f}"
    w = (weight_display or "").strip()
    if not w or w == "—":
        return score_s
    return f"({w}) {score_s}"
