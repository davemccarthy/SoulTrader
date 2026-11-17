"""
Short-term price trend from successive poll samples (not calendar bars).

Used by pulse_agent (BUY-armed) and force_buy --wait-trend:
  rolling window of votes; ups > downs → enter; ties (e.g. 3-3) hold.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Deque, Sequence

# 6 pairwise votes ⇒ 7 price samples; 3-3 is a hold (need ups > downs).
DEFAULT_TREND_VOTES = 6
DEFAULT_TREND_SAMPLES = DEFAULT_TREND_VOTES + 1
DEFAULT_TREND_POLL_S = 10.0
# Back-compat alias used by callers that say "window".
DEFAULT_TREND_WINDOW = DEFAULT_TREND_SAMPLES


@dataclass(frozen=True)
class TrendVote:
    ups: int
    downs: int
    flats: int
    samples: int
    need_samples: int
    ready: bool  # enough samples to vote
    bullish: bool  # ups > downs (ties are not bullish)

    @property
    def detail(self) -> str:
        if not self.ready:
            return f"warming {self.samples}/{self.need_samples}"
        return f"ups={self.ups} downs={self.downs} flats={self.flats}"


def count_votes(prices: Sequence[Decimal | float]) -> tuple[int, int, int]:
    """Consecutive sample votes: up / down / flat."""
    ups = downs = flats = 0
    for i in range(1, len(prices)):
        a = Decimal(str(prices[i - 1]))
        b = Decimal(str(prices[i]))
        if b > a:
            ups += 1
        elif b < a:
            downs += 1
        else:
            flats += 1
    return ups, downs, flats


def evaluate_trend(
    prices: Sequence[Decimal | float],
    *,
    votes: int = DEFAULT_TREND_VOTES,
) -> TrendVote:
    """
    Vote on the last (votes+1) prices → `votes` comparisons.

    Bullish only when ups > downs. Equal ups/downs (e.g. 3-3) is hold.
    """
    need_votes = max(int(votes), 2)
    need_samples = need_votes + 1
    tail = list(prices[-need_samples:]) if prices else []
    ready = len(tail) >= need_samples
    if not ready:
        ups, downs, flats = count_votes(tail) if len(tail) >= 2 else (0, 0, 0)
        return TrendVote(
            ups=ups,
            downs=downs,
            flats=flats,
            samples=len(tail),
            need_samples=need_samples,
            ready=False,
            bullish=False,
        )
    ups, downs, flats = count_votes(tail)
    return TrendVote(
        ups=ups,
        downs=downs,
        flats=flats,
        samples=len(tail),
        need_samples=need_samples,
        ready=True,
        bullish=ups > downs,
    )


class PriceTrendWindow:
    """Rolling price samples for one symbol (votes+1 length)."""

    def __init__(self, *, votes: int = DEFAULT_TREND_VOTES):
        self.votes = max(int(votes), 2)
        self.need_samples = self.votes + 1
        self._prices: Deque[Decimal] = deque(maxlen=self.need_samples)

    def push(self, price: Decimal | float) -> TrendVote:
        px = Decimal(str(price))
        if px <= 0:
            return self.vote()
        self._prices.append(px)
        return self.vote()

    def vote(self) -> TrendVote:
        return evaluate_trend(list(self._prices), votes=self.votes)

    def clear(self) -> None:
        self._prices.clear()

    @property
    def prices(self) -> list[Decimal]:
        return list(self._prices)


def dynamic_arm_sleep_s(
    *,
    base_s: float,
    buy_armed: bool,
    sell_armed: bool,
    arm_poll_s: float = DEFAULT_TREND_POLL_S,
) -> float:
    """Quiet → base; BUY-armed or SELL-armed → faster arm poll."""
    if buy_armed or sell_armed:
        return min(float(base_s), float(arm_poll_s))
    return float(base_s)
