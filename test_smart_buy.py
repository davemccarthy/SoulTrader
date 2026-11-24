"""Unit tests for smart-buy range router and force_buy price band (no network)."""
from decimal import Decimal

from core.services.smart_buy import (
    SmartBuyWindow,
    dip_entry_ceiling,
    evaluate_smart_buy,
    improvement_pct,
    price_band_block_reason,
)


def test_improvement_positive_when_cheaper():
    assert improvement_pct(100, 98.5) == Decimal("1.5")


def test_dip_ceiling_urgency():
    low, high = Decimal("90"), Decimal("100")
    early = dip_entry_ceiling(low, high, elapsed_s=30)
    mid = dip_entry_ceiling(low, high, elapsed_s=180)
    late = dip_entry_ceiling(low, high, elapsed_s=400)
    assert early < mid < late
    assert early == low + (high - low) * Decimal("0.333333")


def test_cheap_waits_for_uptick():
    # Into lower third but still falling / not bullish.
    prices = [100, 99, 98, 97, 96, 95, 94, 93]
    d = evaluate_smart_buy(prices, elapsed_s=0)
    assert d.ready
    assert d.mode == "uptick"
    assert not d.buy
    assert "wait uptick" in d.reason


def test_cheap_buys_on_uptick():
    # Below early ceiling, last votes mostly up.
    prices = [100, 95, 90, 88, 87, 88, 89, 90, 91]
    d = evaluate_smart_buy(prices, elapsed_s=0)
    assert d.ready
    assert d.current <= d.entry_ceiling
    assert d.mode == "uptick"
    assert d.buy
    assert d.reason == "cheap+uptick"


def test_extended_waits_for_dip():
    # Grind up into upper half — should hunt a dip, not chase.
    prices = [90, 91, 92, 93, 94, 95, 96, 97, 98]
    d = evaluate_smart_buy(prices, elapsed_s=0)
    assert d.ready
    assert d.mode == "dip"
    assert not d.buy
    assert "wait dip" in d.reason


def test_dip_then_uptick_buys():
    # Spike, pull under mid urgency ceiling, then bounce → buy.
    prices = [90, 92, 94, 96, 98, 100, 97, 94, 92, 90, 91, 92, 93, 94]
    d = evaluate_smart_buy(prices, elapsed_s=200)
    assert d.ready
    assert d.current <= d.entry_ceiling
    assert d.mode == "uptick"
    assert d.buy
    assert d.reason == "cheap+uptick"


def test_urgency_relaxes_ceiling_into_mid():
    prices = [90, 91, 92, 93, 94, 95, 96, 97, 96.5]
    early = evaluate_smart_buy(prices, elapsed_s=30)
    later = evaluate_smart_buy(prices, elapsed_s=200)
    assert early.mode == "dip"
    assert early.entry_ceiling < later.entry_ceiling
    # Mid urgency can flip 96.5 from dip-hunt into cheap/uptick wait.
    assert later.mode in ("uptick", "dip")


def test_window_seed_and_push():
    w = SmartBuyWindow(max_samples=10, min_samples=4)
    seed = w.seed([10, 11, 12, 13, 14, 15])
    assert seed.ready
    assert len(w.prices) == 6
    d = w.push(14, elapsed_s=10)
    assert d.current == Decimal("14")
    assert len(w.prices) == 7


def test_price_band_blocks_above_max():
    assert price_band_block_reason(Decimal("156"), max_price=Decimal("155"))
    assert price_band_block_reason(Decimal("155"), max_price=Decimal("155")) is None
    assert price_band_block_reason(Decimal("149"), min_price=Decimal("150"))
    assert (
        price_band_block_reason(
            Decimal("152"), min_price=Decimal("150"), max_price=Decimal("155")
        )
        is None
    )


if __name__ == "__main__":
    test_improvement_positive_when_cheaper()
    test_dip_ceiling_urgency()
    test_cheap_waits_for_uptick()
    test_cheap_buys_on_uptick()
    test_extended_waits_for_dip()
    test_dip_then_uptick_buys()
    test_urgency_relaxes_ceiling_into_mid()
    test_window_seed_and_push()
    test_price_band_blocks_above_max()
    print("ok")
