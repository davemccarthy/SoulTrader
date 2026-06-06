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

# Opportunity — wider bands; megacap fair-setup names land D not cliff-edge E.
# A 80+, B 70+, C 60+, D 50+, E 40+, F <40
OPPORTUNITY_BANDS: List[Tuple[float, str, str]] = [
    (80.0, "A", "Exceptional"),
    (70.0, "B", "Strong"),
    (60.0, "C", "Good"),
    (50.0, "D", "Fair"),
    (40.0, "E", "Moderate"),
    (0.0, "F", "Weak"),
]

LETTER_RANK: Dict[str, int] = {
    "A": 6,
    "B": 5,
    "C": 4,
    "D": 3,
    "E": 2,
    "F": 1,
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


def so_floor_display(min_stability: str, min_opportunity: str) -> str:
    """Risk-band floor pair on respective ladders, e.g. CD or EC."""
    return f"{min_stability.strip().upper()}{min_opportunity.strip().upper()}"
