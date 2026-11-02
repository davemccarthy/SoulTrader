#!/usr/bin/env python3
"""TOPUP gap: buy back to a fund-weight target, only through the band."""

from __future__ import annotations

import os
import unittest
from decimal import Decimal

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from core.services.analysis import TOPUP_BAND_DEFAULT, topup_gap


class TopupGapTests(unittest.TestCase):
    def test_inside_band_does_not_buy(self):
        # Target is $10,000. 1% under is still inside the 2% band.
        self.assertIsNone(topup_gap(100_000, "0.10", 9_900))

    def test_through_the_band_buys_the_full_gap(self):
        gap = topup_gap(100_000, Decimal("0.10"), Decimal("9700"))
        self.assertEqual(gap, Decimal("300"))

    def test_exact_band_edge_buys(self):
        gap = topup_gap(100_000, "0.10", 9_800)
        self.assertEqual(gap, Decimal("200"))

    def test_above_target_does_not_sell(self):
        self.assertIsNone(topup_gap(100_000, "0.10", 12_000))

    def test_blank_band_is_two_percent(self):
        self.assertEqual(TOPUP_BAND_DEFAULT, Decimal("0.02"))
        self.assertIsNone(topup_gap(100_000, "0.10", 9_900, None))

    def test_weight_above_one_is_rejected(self):
        self.assertIsNone(topup_gap(100_000, "10", 9_000))

    def test_target_tracks_wealth(self):
        # A larger book lifts the dollar target. 10% of $110,000 is $11,000.
        gap = topup_gap(110_000, "0.10", 10_000, "0.02")
        self.assertEqual(gap, Decimal("1000"))


if __name__ == "__main__":
    unittest.main()
