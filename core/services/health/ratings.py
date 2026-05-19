"""
Letter ratings for health v2 composite scores (0–100).

Bands aligned with LLM validation review (MSFT / NVDA / RIVN).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# (minimum score inclusive, letter, short label for display)
RATING_BANDS: List[Tuple[float, str, str]] = [
    (80.0, "A", "Exceptional"),
    (75.0, "B", "Strong buy"),
    (65.0, "C", "Buy"),
    (50.0, "D", "Watch / mixed"),
    (35.0, "E", "Avoid"),
    (0.0, "F", "Strong avoid"),
]


@dataclass(frozen=True)
class HealthRating:
    letter: str
    label: str
    min_score: float

    @property
    def display_short(self) -> str:
        """e.g. B (strong buy)"""
        return f"{self.letter} ({self.label.lower()})"

    @property
    def display_line(self) -> str:
        """e.g. RATING: B (strong buy)"""
        return f"RATING: {self.display_short}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "letter": self.letter,
            "label": self.label,
            "min_score": self.min_score,
            "display_short": self.display_short,
            "display_line": self.display_line,
        }


def score_to_rating(score: Optional[float]) -> Optional[HealthRating]:
    """Map composite score to letter rating; None if score is None."""
    if score is None:
        return None
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    for min_score, letter, label in RATING_BANDS:
        if s >= min_score:
            return HealthRating(letter=letter, label=label, min_score=min_score)
    return HealthRating(letter="F", label="Strong avoid", min_score=0.0)


def rating_table_rows() -> List[Dict[str, Any]]:
    """Human/LLM-readable band table."""
    rows = []
    prev_min = None
    for min_score, letter, label in RATING_BANDS:
        if prev_min is None:
            band = f"{min_score:.0f}+"
        else:
            band = f"{min_score:.0f}–{prev_min - 0.1:.0f}"
        rows.append({"letter": letter, "band": band, "label": label})
        prev_min = min_score
    return rows
