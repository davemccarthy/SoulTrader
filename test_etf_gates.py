#!/usr/bin/env python3
"""Thin checks for ETF advisor open-check + trade-list defaults."""

from __future__ import annotations

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

from core.services.advisors.etf import (  # noqa: E402
    DEFAULT_TRADE_ETFS,
    ETF_SKIP_DEFAULT,
    classify_etf_open_bucket,
)
from core.services.financial.etf_holdings import DEFAULT_ETF_LIST  # noqa: E402


def test_classify_etf_open_bucket():
    assert classify_etf_open_bucket(None) == "unclear"
    assert classify_etf_open_bucket(-8.0) == "cliff"
    assert classify_etf_open_bucket(-8.01) == "cliff"
    assert classify_etf_open_bucket(-7.9) == "allow"
    assert classify_etf_open_bucket(0.0) == "allow"
    assert classify_etf_open_bucket(7.9) == "allow"
    assert classify_etf_open_bucket(8.0) == "rocket"
    assert classify_etf_open_bucket(12.0) == "rocket"


def test_default_trade_etfs_skip_xbi_xar():
    assert "XBI" in DEFAULT_ETF_LIST
    assert "XAR" in DEFAULT_ETF_LIST
    assert "XBI" not in DEFAULT_TRADE_ETFS
    assert "XAR" not in DEFAULT_TRADE_ETFS
    assert ETF_SKIP_DEFAULT == frozenset({"XBI", "XAR"})
    assert "ARKG" in DEFAULT_TRADE_ETFS
    assert "CIBR" in DEFAULT_TRADE_ETFS


if __name__ == "__main__":
    test_classify_etf_open_bucket()
    test_default_trade_etfs_skip_xbi_xar()
    print("ok")
