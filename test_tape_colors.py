#!/usr/bin/env python3
"""Unit tests for four-color evaluate_tape + Pulse SI list (no network)."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django

django.setup()

from decimal import Decimal

from core.services.advisors.pulse import (
    PULSE_BUILD_TIME_ET,
    PULSE_ENDDAY_CLUTTER_MINUTES_BEFORE_CLOSE,
    PULSE_ENDDAY_CLUTTER_TAKE,
    PULSE_ENDDAY_MINUTES_BEFORE_CLOSE,
    PULSE_ENDDAY_TAKE,
    PULSE_MIN_GAP_PCT,
    PULSE_MIN_PRICE,
    PULSE_REBUY_DROP,
    PULSE_REBUY_MAX_TRANCHES,
    PULSE_TOP_EARLY_VOLUME,
    _extension_skip_reason,
    pulse_sell_instructions_for_tape,
    select_open_vol_candidates,
)
from core.services.market.tape import (
    TapeReading,
    evaluate_tape,
    normalize_tape_state,
)


def _r(sym: str, vs_open: float | None, vs_prior: float | None) -> TapeReading:
    return TapeReading(
        symbol=sym,
        price=100.0,
        open_px=100.0,
        prior_close=100.0,
        vs_open_pct=vs_open,
        vs_prior_close_pct=vs_prior,
    )


def test_red_beats_amber():
    readings = {
        "SPY": _r("SPY", -1.6, -0.5),
        "QQQ": _r("QQQ", 0.2, 0.2),
    }
    assert evaluate_tape(readings).state == "red"


def test_amber_at_soft_open():
    # Tue-style QQQ ~-0.84% vs open
    readings = {
        "SPY": _r("SPY", -0.57, -0.38),
        "QQQ": _r("QQQ", -0.84, -0.51),
    }
    assert evaluate_tape(readings).state == "amber"


def test_green_mid_band():
    readings = {
        "SPY": _r("SPY", 0.29, 0.61),
        "QQQ": _r("QQQ", 0.40, 1.17),
    }
    # Fri Aug 7 EOD-style — below white floors
    assert evaluate_tape(readings).state == "green"


def test_white_requires_all():
    readings = {
        "SPY": _r("SPY", 1.10, 1.42),
        "QQQ": _r("QQQ", 1.71, 1.76),
    }
    assert evaluate_tape(readings).state == "white"


def test_white_fails_if_one_weak():
    readings = {
        "SPY": _r("SPY", 0.80, 1.10),
        "QQQ": _r("QQQ", 0.40, 1.10),
    }
    assert evaluate_tape(readings).state == "green"


def test_normalize_yellow_to_amber():
    assert normalize_tape_state("yellow") == "amber"
    assert normalize_tape_state("white") == "white"
    assert normalize_tape_state("nope") == "red"


def test_pulse_sell_instructions_no_ipc():
    """TARGET_INTRADAY deferred to fund_watch; Pulse attaches rebuy + dual END_DAY."""
    sis = pulse_sell_instructions_for_tape("amber")
    sis_g = pulse_sell_instructions_for_tape("green")
    assert sis == sis_g  # tape-agnostic while IPC paused
    kinds = [row[0] for row in sis]
    assert "TARGET_INTRADAY" not in kinds
    assert (
        "PERCENTAGE_REBUY",
        PULSE_REBUY_DROP,
        PULSE_REBUY_MAX_TRANCHES,
    ) in sis
    assert (
        "END_DAY",
        PULSE_ENDDAY_TAKE,
        PULSE_ENDDAY_MINUTES_BEFORE_CLOSE,
    ) in sis
    assert (
        "END_DAY",
        PULSE_ENDDAY_CLUTTER_TAKE,
        PULSE_ENDDAY_CLUTTER_MINUTES_BEFORE_CLOSE,
    ) in sis


def test_extension_skip_year_change_and_near_52w():
    """Helper retained for agent/highs filter; not applied at Pulse discover."""
    assert _extension_skip_reason(price=100, year_high=200, year_change=0.5) is None
    assert "yearChange" in (_extension_skip_reason(price=100, year_high=200, year_change=0.80) or "")
    assert "yearChange" in (_extension_skip_reason(price=370, year_high=595, year_change=1.79) or "")
    assert "near52w" in (_extension_skip_reason(price=95, year_high=100, year_change=0.2) or "")
    assert _extension_skip_reason(price=80, year_high=100, year_change=0.2) is None
    assert _extension_skip_reason(price=None, year_high=None, year_change=None) is None


def test_open_vol_knobs():
    assert PULSE_BUILD_TIME_ET.hour == 9 and PULSE_BUILD_TIME_ET.minute == 45
    assert PULSE_MIN_PRICE >= 20.0
    assert PULSE_MIN_GAP_PCT == 2.0
    assert PULSE_TOP_EARLY_VOLUME == 30


def test_select_open_vol_top_n_then_gap():
    rows = []
    # 40 liquid names; ranks 1..40 by volume. Gaps alternate above/below 2%.
    for i in range(40):
        gap = 3.0 if i % 2 == 0 else 0.5
        prior = 100.0
        open_px = prior * (1.0 + gap / 100.0)
        rows.append(
            {
                "symbol": f"T{i:02d}",
                "open_px": open_px,
                "prior_close": prior,
                "prior_volume": 1_000_000,
                "volume": 5_000_000 - i * 10_000,
                "price": open_px,
            }
        )
    # Penny / ETF excluded
    rows.append(
        {
            "symbol": "CHEAP",
            "open_px": 10.0,
            "prior_close": 9.0,
            "prior_volume": 2_000_000,
            "volume": 9_000_000,
            "price": 10.0,
        }
    )
    rows.append(
        {
            "symbol": "SPY",
            "open_px": 500.0,
            "prior_close": 480.0,
            "prior_volume": 50_000_000,
            "volume": 80_000_000,
            "price": 500.0,
        }
    )
    attention, cands = select_open_vol_candidates(rows, top_n=30, min_gap_pct=2.0, min_price=20.0)
    assert len(attention) == 30
    assert all(r["symbol"] not in {"CHEAP", "SPY"} for r in attention)
    assert all(float(c["gap_pct"]) >= 2.0 for c in cands)
    assert 10 <= len(cands) <= 20
    assert attention[0]["rank"] == 1


if __name__ == "__main__":
    test_red_beats_amber()
    test_amber_at_soft_open()
    test_green_mid_band()
    test_white_requires_all()
    test_white_fails_if_one_weak()
    test_normalize_yellow_to_amber()
    test_pulse_sell_instructions_no_ipc()
    test_extension_skip_year_change_and_near_52w()
    test_open_vol_knobs()
    test_select_open_vol_top_n_then_gap()
    print("ok")
