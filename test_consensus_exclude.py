#!/usr/bin/env python3
"""
Unit tests: no-coverage analyst consensus is excluded (score=None), not mid-neutral 50.

Usage:
    source ~/Development/scratch/python/tutorial-env/bin/activate
    python test_consensus_exclude.py
"""

from __future__ import annotations

import sys
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.services.health.assess import composite_from_scores
from core.services.health.consensus import score_consensus_health


class ConsensusExcludeTests(unittest.TestCase):
    @patch("core.services.health.consensus.get_consensus_snapshot")
    def test_unusable_snapshot_excludes(self, mock_snap) -> None:
        mock_snap.return_value = {"is_usable": False}
        result = score_consensus_health("ZSQR")
        self.assertIsNone(result.score)
        self.assertTrue(result.neutral_fallback)
        self.assertEqual(result.error, "no analyst consensus data")

    @patch("core.services.health.consensus.get_consensus_snapshot")
    def test_usable_but_no_scorable_metrics_excludes(self, mock_snap) -> None:
        mock_snap.return_value = {
            "is_usable": True,
            "recommendation_key": None,
            "recommendation_mean": None,
            "analyst_count": None,
            "target_mean": None,
            "current_price": 1.0,
            "upside_to_mean_pct": None,
            "upside_to_low_pct": None,
        }
        result = score_consensus_health("ZSQR")
        self.assertIsNone(result.score)
        self.assertTrue(result.neutral_fallback)
        self.assertEqual(result.error, "no scorable consensus metrics")

    def test_composite_renormalizes_without_consensus(self) -> None:
        with_50 = {
            "financial": 61.4,
            "valuation": None,
            "intrinsic": 50.0,
            "price": 81.3,
            "consensus": 50.0,
            "sector": 70.0,
        }
        without = dict(with_50)
        without["consensus"] = None

        old = composite_from_scores(with_50)
        new = composite_from_scores(without)
        self.assertIsNotNone(old)
        self.assertIsNotNone(new)
        self.assertNotEqual(old, new)

        # valuation missing → 0 at full 0.20 weight; consensus None → drop from den.
        num_old = (
            Decimal("61.4") * Decimal("0.20")
            + Decimal("0") * Decimal("0.20")
            + Decimal("50.0") * Decimal("0.15")
            + Decimal("81.3") * Decimal("0.20")
            + Decimal("50.0") * Decimal("0.15")
            + Decimal("70.0") * Decimal("0.10")
        )
        self.assertEqual(old, (num_old / Decimal("1.0")).quantize(Decimal("0.1")))

        num_new = (
            Decimal("61.4") * Decimal("0.20")
            + Decimal("0") * Decimal("0.20")
            + Decimal("50.0") * Decimal("0.15")
            + Decimal("81.3") * Decimal("0.20")
            + Decimal("70.0") * Decimal("0.10")
        )
        self.assertEqual(new, (num_new / Decimal("0.85")).quantize(Decimal("0.1")))


if __name__ == "__main__":
    unittest.main()
