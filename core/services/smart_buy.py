"""
Smart-buy entry from a rolling lookback range (force_buy --smart-buy).

B design: seed from recent bars (not only post-command noise), then:
  - already cheap in the window (≤ mid) → wait for short-term uptick
  - extended above mid → wait for a dip into the cheap side of the range
Rolling high/mid/low move with the noise. Start quote is scorecard only
(improvement % vs plan price).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Deque, Sequence

from core.services.short_term_trend import (
    DEFAULT_TREND_POLL_S,
    DEFAULT_TREND_VOTES,
    TrendVote,
    evaluate_trend,
)

# ~15 one-minute bars to seed context; keep ~18 samples rolling thereafter.
DEFAULT_LOOKBACK_MINUTES = 15
DEFAULT_RANGE_SAMPLES = 18
DEFAULT_SMART_POLL_S = DEFAULT_TREND_POLL_S
DEFAULT_SMART_BUY_TIMEOUT_S = 0.0  # 0 = run until buy or Ctrl+C

# Urgency widens the dip entry ceiling over time (seconds elapsed in wait).
URGENCY_LOWER_THIRD_S = 120.0
URGENCY_MID_S = 300.0


@dataclass(frozen=True)
class SmartBuyDecision:
    ready: bool
    buy: bool
    mode: str  # warming | uptick | dip
    reason: str
    low: Decimal | None
    high: Decimal | None
    mid: Decimal | None
    current: Decimal | None
    entry_ceiling: Decimal | None
    position: float | None  # 0=low … 1=high of window
    trend: TrendVote | None

    @property
    def detail(self) -> str:
        if not self.ready or self.low is None or self.high is None or self.mid is None:
            return f"warming mode={self.mode} {self.reason}"
        pos = f"{self.position:.2f}" if self.position is not None else "?"
        trend = self.trend.detail if self.trend else "n/a"
        ceil = f"{self.entry_ceiling:.4f}" if self.entry_ceiling is not None else "n/a"
        return (
            f"mode={self.mode} pos={pos} "
            f"range=[{self.low:.4f},{self.high:.4f}] mid={self.mid:.4f} "
            f"ceil={ceil} trend[{trend}] {self.reason}"
        )


def _as_decimal(value: Decimal | float | int | str) -> Decimal:
    return Decimal(str(value))


def range_stats(prices: Sequence[Decimal | float]) -> tuple[Decimal, Decimal, Decimal]:
    vals = [_as_decimal(p) for p in prices if _as_decimal(p) > 0]
    if not vals:
        zero = Decimal("0")
        return zero, zero, zero
    low = min(vals)
    high = max(vals)
    mid = (low + high) / Decimal("2")
    return low, high, mid


def range_position(current: Decimal | float, low: Decimal, high: Decimal) -> float:
    cur = _as_decimal(current)
    span = high - low
    if span <= 0:
        return 0.5
    return float((cur - low) / span)


def dip_entry_ceiling(
    low: Decimal,
    high: Decimal,
    *,
    elapsed_s: float,
) -> Decimal:
    """
    Cheap-side target that relaxes with urgency:
      <2m  → lower third
      <5m  → mid
      else → mid + 10% of span (still not the top)
    """
    span = high - low
    if span <= 0:
        return high
    lower_third = low + span * Decimal("0.333333")
    mid = low + span * Decimal("0.5")
    soft = mid + span * Decimal("0.10")
    if elapsed_s < URGENCY_LOWER_THIRD_S:
        return lower_third
    if elapsed_s < URGENCY_MID_S:
        return mid
    return soft


def evaluate_smart_buy(
    prices: Sequence[Decimal | float],
    *,
    current: Decimal | float | None = None,
    elapsed_s: float = 0.0,
    votes: int = DEFAULT_TREND_VOTES,
    min_samples: int = 6,
) -> SmartBuyDecision:
    """
    Decide whether to buy from the rolling lookback sample list.

    `current` defaults to the last price in `prices`.
    """
    clean = [_as_decimal(p) for p in prices if _as_decimal(p) > 0]
    if current is not None and _as_decimal(current) > 0:
        cur = _as_decimal(current)
    elif clean:
        cur = clean[-1]
    else:
        return SmartBuyDecision(
            ready=False,
            buy=False,
            mode="warming",
            reason="no prices",
            low=None,
            high=None,
            mid=None,
            current=None,
            entry_ceiling=None,
            position=None,
            trend=None,
        )

    need = max(int(min_samples), 3)
    if len(clean) < need:
        return SmartBuyDecision(
            ready=False,
            buy=False,
            mode="warming",
            reason=f"samples {len(clean)}/{need}",
            low=None,
            high=None,
            mid=None,
            current=cur,
            entry_ceiling=None,
            position=None,
            trend=None,
        )

    low, high, mid = range_stats(clean)
    pos = range_position(cur, low, high)
    ceiling = dip_entry_ceiling(low, high, elapsed_s=elapsed_s)
    trend = evaluate_trend(clean, votes=votes)

    # Flat tape: allow uptick path once trend is ready+bullish.
    if high <= low:
        buy = bool(trend.ready and trend.bullish)
        return SmartBuyDecision(
            ready=True,
            buy=buy,
            mode="uptick",
            reason="flat range; wait uptick" if not buy else "flat range uptick",
            low=low,
            high=high,
            mid=mid,
            current=cur,
            entry_ceiling=ceiling,
            position=pos,
            trend=trend,
        )

    # Still extended vs urgency ceiling → hunt a dip (no chase).
    if cur > ceiling:
        return SmartBuyDecision(
            ready=True,
            buy=False,
            mode="dip",
            reason="extended; wait dip",
            low=low,
            high=high,
            mid=mid,
            current=cur,
            entry_ceiling=ceiling,
            position=pos,
            trend=trend,
        )

    # At/under ceiling (cheap vs urgency) → wait for uptick / stabilize.
    mode = "uptick"
    if trend.ready and trend.bullish:
        return SmartBuyDecision(
            ready=True,
            buy=True,
            mode=mode,
            reason="cheap+uptick",
            low=low,
            high=high,
            mid=mid,
            current=cur,
            entry_ceiling=ceiling,
            position=pos,
            trend=trend,
        )
    return SmartBuyDecision(
        ready=True,
        buy=False,
        mode=mode,
        reason="cheap; wait uptick",
        low=low,
        high=high,
        mid=mid,
        current=cur,
        entry_ceiling=ceiling,
        position=pos,
        trend=trend,
    )


class SmartBuyWindow:
    """Rolling lookback prices + evaluate_smart_buy each push."""

    def __init__(
        self,
        *,
        max_samples: int = DEFAULT_RANGE_SAMPLES,
        votes: int = DEFAULT_TREND_VOTES,
        min_samples: int = 6,
    ):
        self.max_samples = max(int(max_samples), 3)
        self.votes = max(int(votes), 2)
        self.min_samples = max(int(min_samples), 3)
        self._prices: Deque[Decimal] = deque(maxlen=self.max_samples)
        self.started_monotonic: float | None = None

    def seed(self, prices: Sequence[Decimal | float]) -> SmartBuyDecision:
        self._prices.clear()
        for p in prices:
            px = _as_decimal(p)
            if px > 0:
                self._prices.append(px)
        return self.decide(elapsed_s=0.0)

    def push(self, price: Decimal | float, *, elapsed_s: float) -> SmartBuyDecision:
        px = _as_decimal(price)
        if px > 0:
            self._prices.append(px)
        return self.decide(elapsed_s=elapsed_s)

    def decide(self, *, elapsed_s: float) -> SmartBuyDecision:
        return evaluate_smart_buy(
            list(self._prices),
            elapsed_s=elapsed_s,
            votes=self.votes,
            min_samples=self.min_samples,
        )

    @property
    def prices(self) -> list[Decimal]:
        return list(self._prices)


def improvement_pct(start: Decimal | float, execution: Decimal | float) -> Decimal:
    """(start - execution) / start * 100. Positive = cheaper than plan quote."""
    s = _as_decimal(start)
    e = _as_decimal(execution)
    if s <= 0:
        return Decimal("0")
    return (s - e) / s * Decimal("100")


def price_band_block_reason(
    price: Decimal | float | None,
    *,
    min_price: Decimal | float | None = None,
    max_price: Decimal | float | None = None,
) -> str | None:
    """
    Why a trigger price is outside an optional band, or None if allowed.

    Used by force_buy to veto smart-buy / wait-trend fills without stopping the chase.
    """
    if price is None or _as_decimal(price) <= 0:
        return "no price"
    cur = _as_decimal(price)
    if min_price is not None and cur < _as_decimal(min_price):
        return f"below min-price ${_as_decimal(min_price)}"
    if max_price is not None and cur > _as_decimal(max_price):
        return f"above max-price ${_as_decimal(max_price)}"
    return None
