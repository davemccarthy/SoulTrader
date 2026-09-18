"""
Yoyo advisor — fixed volatile pack, per-name mid-range revive entries.

Funnel:
  RTH (+15m) → universe (blob or AI-8 default) → compute revive
  → price <= revive → discover (48h cooldown; holding OK after that)

Revive (per name):
  mid = (lookback high + low) / 2
  X   = clip(0.25 * (high/low - 1), 4%, 10%)
  revive = mid * (1 - X)

Exit/add: PEAKED (arm +4%) + gated PERCENTAGE_REBUY 4%/5 + DESCENDING_TREND.
Pack correlation gate: deferred.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.services.advisors.advisor import AdvisorBase, register
from core.services.financial import yahoo as financial_yahoo

logger = logging.getLogger(__name__)

# Default pack (AI-CYC / AI-8 bounce sleeve).
DEFAULT_UNIVERSE: Tuple[str, ...] = (
    "QBTS",
    "IONQ",
    "POET",
    "CEVA",
    "QUIK",
    "MXL",
    "INOD",
    "VIAV",
)

YOYO_LOOKBACK_DAYS = 15
YOYO_X_FRAC_OF_SPAN = 0.25
YOYO_X_MIN = 0.04
YOYO_X_MAX = 0.10
YOYO_DISCOVERY_COOLDOWN_HOURS = 48
YOYO_MIN_MINUTES_AFTER_OPEN = 15

YOYO_PEAKED_GIVEBACK = 15.0
YOYO_PEAKED_MIN_PEAK = 4.0
YOYO_REBUY_DROP = Decimal("0.04")
YOYO_REBUY_MAX_TRANCHES = Decimal("5")


def compute_revive(
    highs: Sequence[float],
    lows: Sequence[float],
    *,
    lookback: int = YOYO_LOOKBACK_DAYS,
    x_frac: float = YOYO_X_FRAC_OF_SPAN,
    x_min: float = YOYO_X_MIN,
    x_max: float = YOYO_X_MAX,
) -> Optional[Dict[str, float]]:
    """
    Per-name revive from recent high/low mid-range.

    Returns dict with mid, high, low, span, x, revive — or None if insufficient data.
    """
    if lookback < 2:
        return None
    h = [float(v) for v in highs if v is not None and float(v) > 0]
    l = [float(v) for v in lows if v is not None and float(v) > 0]
    n = min(len(h), len(l), lookback)
    if n < 2:
        return None
    h_win = h[-n:]
    l_win = l[-n:]
    hi = max(h_win)
    lo = min(l_win)
    if lo <= 0 or hi < lo:
        return None
    mid = (hi + lo) / 2.0
    span = (hi / lo) - 1.0
    x = min(x_max, max(x_min, span * x_frac))
    revive = mid * (1.0 - x)
    return {
        "mid": mid,
        "high": hi,
        "low": lo,
        "span": span,
        "x": x,
        "revive": revive,
        "lookback": float(n),
    }


class Yoyo(AdvisorBase):
    """Volatile pack yo-yo: discover when live price is at/below per-name revive."""

    sell_instructions = [
        ("PEAKED", YOYO_PEAKED_GIVEBACK, YOYO_PEAKED_MIN_PEAK),
        ("PERCENTAGE_REBUY", YOYO_REBUY_DROP, YOYO_REBUY_MAX_TRANCHES),
        ("DESCENDING_TREND", -0.20, None),
    ]

    def discover(self, sa) -> None:
        market_status = self.market_open()
        if market_status is None:
            logger.info("Yoyo skip: market closed")
            return
        if market_status < 0:
            logger.info(
                "Yoyo skip: market not open yet (%s min to open)",
                -market_status,
            )
            return
        if market_status < YOYO_MIN_MINUTES_AFTER_OPEN:
            logger.info(
                "Yoyo skip: before discover window (%s min open; need %s)",
                market_status,
                YOYO_MIN_MINUTES_AFTER_OPEN,
            )
            return

        universe = self._universe()
        lookback = self._lookback_days()
        discoveries = 0
        revives_blob: Dict[str, Any] = {}

        for symbol in universe:
            levels = self._revive_for(symbol, lookback=lookback)
            if levels is None:
                logger.info("Yoyo skip %s: no revive (history)", symbol)
                continue

            revive = levels["revive"]
            revives_blob[symbol] = {
                "revive": round(revive, 4),
                "mid": round(levels["mid"], 4),
                "x": round(levels["x"], 4),
                "span": round(levels["span"], 4),
                "high": round(levels["high"], 4),
                "low": round(levels["low"], 4),
            }

            if not self.allow_discovery(symbol, period=YOYO_DISCOVERY_COOLDOWN_HOURS):
                logger.info(
                    "Yoyo skip %s: cooldown (%sh) revive=$%.2f",
                    symbol,
                    YOYO_DISCOVERY_COOLDOWN_HOURS,
                    revive,
                )
                continue

            stock = self.get_stock(symbol)
            if stock is None:
                logger.info("Yoyo skip %s: no stock", symbol)
                continue

            price = float(stock.price or 0)
            if price <= 0:
                logger.info("Yoyo skip %s: no price", symbol)
                continue

            if price > revive:
                logger.info(
                    "Yoyo skip %s: $%.2f > revive $%.2f (mid $%.2f X=%.1f%% span=%.1f%%)",
                    symbol,
                    price,
                    revive,
                    levels["mid"],
                    levels["x"] * 100,
                    levels["span"] * 100,
                )
                continue

            explanation = (
                f"Yoyo revive | ${price:.2f} <= revive ${revive:.2f} "
                f"(mid ${levels['mid']:.2f}, X={levels['x']*100:.1f}%, "
                f"{int(levels['lookback'])}d range ${levels['low']:.2f}-${levels['high']:.2f})"
            )
            meta = {
                "yoyo": {
                    "revive": revive,
                    "mid": levels["mid"],
                    "x": levels["x"],
                    "span": levels["span"],
                    "lookback": int(levels["lookback"]),
                    "high": levels["high"],
                    "low": levels["low"],
                    "price": price,
                }
            }
            if self.discovered(
                sa,
                symbol,
                explanation,
                sell_instructions=list(self.sell_instructions),
                weight=1.0,
                meta=meta,
            ):
                discoveries += 1
                logger.info(
                    "Yoyo discover %s @ $%.2f revive=$%.2f X=%.1f%%",
                    symbol,
                    price,
                    revive,
                    levels["x"] * 100,
                )

        state = self._advisor_blob_state()
        state["universe"] = list(universe)
        state["lookback_days"] = lookback
        state["revives"] = revives_blob
        if getattr(sa, "id", None) is not None:
            state["last_sa_id"] = sa.id
        self._save_advisor_blob_state(state)

        logger.info(
            "Yoyo discover sa=%s: universe=%d discoveries=%d",
            getattr(sa, "id", None),
            len(universe),
            discoveries,
        )

    def _universe(self) -> List[str]:
        state = self._advisor_blob_state()
        raw = state.get("universe")
        if isinstance(raw, list) and raw:
            out = []
            for item in raw:
                sym = str(item or "").strip().upper()
                if sym:
                    out.append(sym)
            if out:
                return out
        return list(DEFAULT_UNIVERSE)

    def _lookback_days(self) -> int:
        state = self._advisor_blob_state()
        try:
            n = int(state.get("lookback_days", YOYO_LOOKBACK_DAYS))
        except (TypeError, ValueError):
            n = YOYO_LOOKBACK_DAYS
        return max(5, min(n, 60))

    def _revive_for(self, symbol: str, lookback: int) -> Optional[Dict[str, float]]:
        hist = financial_yahoo.get_6m_history(symbol)
        if hist is None or hist.empty:
            return None
        if "high" not in hist.columns or "low" not in hist.columns:
            return None
        highs = hist["high"].astype(float).tolist()
        lows = hist["low"].astype(float).tolist()
        return compute_revive(highs, lows, lookback=lookback)

    def analyze(self, sa, stock) -> None:
        return


register(name="Yoyo", python_class="Yoyo")
