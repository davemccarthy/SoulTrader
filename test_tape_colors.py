#!/usr/bin/env python3
"""Unit tests for four-color evaluate_tape (no network)."""
from __future__ import annotations

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


def test_amber_not_red():
    readings = {
        "SPY": _r("SPY", -1.2, -0.5),
        "QQQ": _r("QQQ", 0.2, 0.2),
    }
    assert evaluate_tape(readings).state == "amber"


def test_green_mid_band():
    readings = {
        "SPY": _r("SPY", 0.29, 0.61),
        "QQQ": _r("QQQ", 0.40, 1.17),
    }
    # Fri Aug 7 EOD-style — strong but below white open floor on SPY
    assert evaluate_tape(readings).state == "green"


def test_white_requires_all():
    readings = {
        "SPY": _r("SPY", 0.60, 0.80),
        "QQQ": _r("QQQ", 0.55, 0.90),
    }
    assert evaluate_tape(readings).state == "white"


def test_white_fails_if_one_weak():
    readings = {
        "SPY": _r("SPY", 0.60, 0.80),
        "QQQ": _r("QQQ", 0.20, 0.90),
    }
    assert evaluate_tape(readings).state == "green"


def test_normalize_yellow_to_amber():
    assert normalize_tape_state("yellow") == "amber"
    assert normalize_tape_state("YELLOW") == "amber"
    assert normalize_tape_state("white") == "white"
    assert normalize_tape_state("nope") == "red"


def test_empty_readings_amber():
    assert evaluate_tape({}).state == "amber"


if __name__ == "__main__":
    test_red_beats_amber()
    test_amber_not_red()
    test_green_mid_band()
    test_white_requires_all()
    test_white_fails_if_one_weak()
    test_normalize_yellow_to_amber()
    test_empty_readings_amber()
    print("ok")
