#!/usr/bin/env python3
"""Unit tests for four-color evaluate_tape + Pulse IPC map (no network)."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django

django.setup()

from decimal import Decimal

from core.services.advisors.pulse import (
    PULSE_IPC_BY_TAPE,
    pulse_sell_instructions_for_tape,
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


def test_ipc_map_and_sell_instructions():
    assert PULSE_IPC_BY_TAPE["amber"] == (Decimal("1.002"), Decimal("0.002"))
    assert PULSE_IPC_BY_TAPE["green"] == (Decimal("1.004"), Decimal("0.002"))
    assert PULSE_IPC_BY_TAPE["white"] == (Decimal("1.006"), Decimal("0.004"))
    sis = pulse_sell_instructions_for_tape("amber")
    assert sis[0] == ("TARGET_INTRADAY", Decimal("1.002"), Decimal("0.002"))
    sis_g = pulse_sell_instructions_for_tape("green")
    assert sis_g[0][1:] == (Decimal("1.004"), Decimal("0.002"))


if __name__ == "__main__":
    test_red_beats_amber()
    test_amber_at_soft_open()
    test_green_mid_band()
    test_white_requires_all()
    test_white_fails_if_one_weak()
    test_normalize_yellow_to_amber()
    test_ipc_map_and_sell_instructions()
    print("ok")
