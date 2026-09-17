#!/usr/bin/env python3
"""
PERCENTAGE_REBUY headline gate: BUY / HOLD / SELL (no live Gemini).

Usage:
    source ~/Development/scratch/python/tutorial-env/bin/activate
    python test_rebuy_headline.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.services.risk.headline_screen import (
    HeadlineScreenResult,
    TRADE_EXPLANATION_MAX,
    apply_llm_rebuy_verdict,
    rebuy_flatten_explanation,
    rebuy_headline_decision,
    screen_headlines_for_rebuy,
)
from core.services.intraday_stabilize import is_down_vs_minutes_ago, sector_etf_for
import pandas as pd


OFFERING_HEADLINE = "Company announces $400 million secondary stock offering"


class RebuyDecisionTests(unittest.TestCase):
    def test_clear_is_buy(self):
        result = HeadlineScreenResult(symbol="ABC", stage="clear")
        self.assertEqual(rebuy_headline_decision(result), "buy")

    def test_llm_buy(self):
        result = HeadlineScreenResult(symbol="ABC")
        apply_llm_rebuy_verdict(result, {"action": "BUY", "reasoning": "Noise around an already-priced miss."})
        self.assertEqual(result.stage, "llm_buy")
        self.assertEqual(rebuy_headline_decision(result), "buy")

    def test_llm_hold(self):
        result = HeadlineScreenResult(symbol="ABC")
        apply_llm_rebuy_verdict(result, {"action": "HOLD", "reasoning": "Downgrade without a structural break."})
        self.assertEqual(rebuy_headline_decision(result), "hold")

    def test_llm_sell(self):
        result = HeadlineScreenResult(symbol="ABC")
        apply_llm_rebuy_verdict(
            result,
            {"action": "SELL", "reasoning": "FDA issued a CRL for the lead asset."},
        )
        self.assertEqual(result.stage, "llm_sell")
        self.assertEqual(rebuy_headline_decision(result), "sell")

    def test_execute_block_not_sell(self):
        result = HeadlineScreenResult(symbol="ABC")
        apply_llm_rebuy_verdict(result, {"action": "BLOCK", "reasoning": "discovery-shaped"})
        self.assertEqual(result.stage, "llm_error")
        self.assertEqual(rebuy_headline_decision(result), "hold")

    def test_llm_error_is_hold(self):
        result = HeadlineScreenResult(symbol="ABC", stage="llm_error")
        self.assertEqual(rebuy_headline_decision(result), "hold")

    def test_keyword_hit_without_llm_is_hold(self):
        result = HeadlineScreenResult(symbol="ABC", stage="keyword_hit")
        self.assertEqual(rebuy_headline_decision(result), "hold")


class FlattenExplanationTests(unittest.TestCase):
    def test_names_the_news(self):
        result = HeadlineScreenResult(symbol="ABC", stage="llm_sell")
        result.llm = {"reasoning": "FDA issued a CRL for the lead asset after the trial missed."}
        expl = rebuy_flatten_explanation(result)
        self.assertTrue(expl.startswith("REBUY flatten: "))
        self.assertIn("CRL", expl)
        self.assertLessEqual(len(expl), TRADE_EXPLANATION_MAX)

    def test_truncates_to_trade_field(self):
        result = HeadlineScreenResult(symbol="ABC")
        result.llm = {"reasoning": "X" * 400}
        expl = rebuy_flatten_explanation(result)
        self.assertEqual(len(expl), TRADE_EXPLANATION_MAX)
        self.assertTrue(expl.startswith("REBUY flatten: "))
        self.assertTrue(expl.endswith("..."))


class ScreenRebuyTests(unittest.TestCase):
    @patch("core.services.risk.headline_screen.run_llm_gatekeeper")
    @patch("core.services.risk.headline_screen.fetch_headlines")
    def test_clear_tape_does_not_call_llm(self, fetch, llm):
        fetch.return_value = ("ABC", ["Company beats estimates, raises full-year outlook"], "6h")
        result = screen_headlines_for_rebuy("ABC", advisor="Pharm")
        self.assertEqual(result.stage, "clear")
        self.assertEqual(rebuy_headline_decision(result), "buy")
        llm.assert_not_called()

    @patch("core.services.risk.headline_screen.run_llm_gatekeeper")
    @patch("core.services.risk.headline_screen.fetch_headlines")
    def test_keyword_llm_down_is_hold(self, fetch, llm):
        fetch.return_value = ("ABC", [OFFERING_HEADLINE], "6h")
        llm.return_value = None
        result = screen_headlines_for_rebuy("ABC", advisor="Pharm")
        self.assertEqual(result.stage, "llm_error")
        self.assertEqual(rebuy_headline_decision(result), "hold")
        llm.assert_called_once()
        self.assertIsNotNone(llm.call_args.kwargs.get("system"))

    @patch("core.services.risk.headline_screen.run_llm_gatekeeper")
    @patch("core.services.risk.headline_screen.fetch_headlines")
    def test_keyword_sell_passes_position_context(self, fetch, llm):
        fetch.return_value = ("ABC", [OFFERING_HEADLINE], "6h")
        llm.return_value = {
            "action": "SELL",
            "reasoning": "Secondary stock offering announced at a steep discount.",
            "risk_score": 88,
        }
        ctx = {"drop_vs_avg": 0.04, "entered_as": "OR trough"}
        result = screen_headlines_for_rebuy("ABC", position_context=ctx)
        self.assertEqual(rebuy_headline_decision(result), "sell")
        self.assertEqual(llm.call_args.kwargs.get("position_context"), ctx)
        expl = rebuy_flatten_explanation(result)
        self.assertIn("offering", expl.lower())


class SectorDownerTests(unittest.TestCase):
    def test_maps_yfinance_sector(self):
        self.assertEqual(sector_etf_for("Healthcare"), "XLV")
        self.assertEqual(sector_etf_for("Financial Services"), "XLF")
        self.assertEqual(sector_etf_for("Technology"), "XLK")
        self.assertIsNone(sector_etf_for(""))
        self.assertIsNone(sector_etf_for("Unknown Industry"))

    @patch("core.services.intraday_stabilize._intraday_15m")
    def test_strictly_down_is_downer(self, hist_fn):
        now = pd.Timestamp.now(tz="UTC")
        hist_fn.return_value = pd.DataFrame(
            {"Close": [100.0, 99.0]},
            index=[now - pd.Timedelta(minutes=30), now],
        )
        self.assertTrue(is_down_vs_minutes_ago("XLV", minutes=30))

    @patch("core.services.intraday_stabilize._intraday_15m")
    def test_flat_is_not_downer(self, hist_fn):
        now = pd.Timestamp.now(tz="UTC")
        hist_fn.return_value = pd.DataFrame(
            {"Close": [100.0, 100.0]},
            index=[now - pd.Timedelta(minutes=30), now],
        )
        self.assertFalse(is_down_vs_minutes_ago("XLV", minutes=30))

    @patch("core.services.intraday_stabilize._intraday_15m")
    def test_up_is_not_downer(self, hist_fn):
        now = pd.Timestamp.now(tz="UTC")
        hist_fn.return_value = pd.DataFrame(
            {"Close": [100.0, 100.4]},
            index=[now - pd.Timedelta(minutes=30), now],
        )
        self.assertFalse(is_down_vs_minutes_ago("XLV", minutes=30))

    @patch("core.services.intraday_stabilize._intraday_15m")
    def test_missing_bars_fail_open(self, hist_fn):
        hist_fn.return_value = None
        self.assertIsNone(is_down_vs_minutes_ago("XLV", minutes=30))


if __name__ == "__main__":
    unittest.main()
