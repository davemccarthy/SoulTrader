"""
Intraday overlay advisor.

Phase-1 design:
- Universe is stocks already held across enabled funds.
- One pass per market day, only in first quarter of session.
- Trigger only when weakness is broad across spread.
- Reuse existing Health records (no new LLM health checks).
"""

import logging
from statistics import median
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from django.utils import timezone
from pytz import timezone as tz

from core.models import Health, Holding, Stock
from core.services.advisors.advisor import AdvisorBase, register

logger = logging.getLogger(__name__)

# Session windows (market_open() returns minutes since 9:30 ET)
TRIGGER_WINDOW_START_MINUTE = 0
TRIGGER_WINDOW_END_MINUTE = 98  # ~first quarter of regular session (390 / 4)

# Spread trigger thresholds
MIN_UNIVERSE_SIZE = 10
SPREAD_RED_RATIO_MIN = 0.70
SPREAD_MEDIAN_DROP_MAX = -0.60  # percent since open

# Candidate thresholds (percent since open)
ENTRY_DROP_MIN = -2.50
ENTRY_DROP_MAX = -1.00
EXTREME_BREAKDOWN_CUTOFF = -3.50
MIN_PRICE = 1.0

# Exit behavior (handled by core/services/analysis.py)
TARGET_PROFIT_MULTIPLIER = 1.015  # +1.5%
END_DAY_WINNER_MULTIPLIER = 1.0   # sell if >= break-even by EOD check
MAX_HOLD_DAYS = 3


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance history outputs, including MultiIndex columns."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels) > 1:
            df.columns = df.columns.droplevel(1)
        else:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = [str(col).split(".")[0] if "." in str(col) else str(col) for col in df.columns]
    return df


def _intraday_return_since_open(symbol: str) -> Optional[float]:
    """
    Return percentage change from today's first intraday open to latest close.
    Returns None when data is unavailable.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d", interval="5m")
        hist = normalize_dataframe(hist)
        if hist.empty:
            return None

        first_open = float(hist["Open"].iloc[0])
        latest_close = float(hist["Close"].iloc[-1])
        if first_open <= 0:
            return None
        if latest_close < MIN_PRICE:
            return None

        return ((latest_close - first_open) / first_open) * 100.0
    except Exception as exc:
        logger.debug("Intraday return failed for %s: %s", symbol, exc)
        return None


class Intraday(AdvisorBase):
    """Intraday spread-dip overlay for already-held stocks."""

    def _today_key(self) -> str:
        et_now = timezone.now().astimezone(tz("US/Eastern"))
        return et_now.strftime("%Y-%m-%d")

    def _held_universe(self) -> List[str]:
        symbols = (
            Holding.objects.filter(fund__enabled=True, shares__gt=0)
            .select_related("stock")
            .values_list("stock__symbol", flat=True)
            .distinct()
        )
        return [s.strip().upper() for s in symbols if s]

    def _latest_health(self, stock: Stock) -> Optional[Health]:
        return Health.objects.filter(stock=stock).order_by("-created").first()

    def discover(self, sa):
        market_status = self.market_open()
        if market_status is None or market_status < TRIGGER_WINDOW_START_MINUTE or market_status > TRIGGER_WINDOW_END_MINUTE:
            logger.info("Intraday skip: outside trigger window (%s min since open)", market_status)
            return

        today = self._today_key()
        state = self._advisor_blob_state()
        if (state.get("processed_date") or "").strip() == today:
            logger.info("Intraday skip: already processed for %s", today)
            return

        symbols = self._held_universe()
        if len(symbols) < MIN_UNIVERSE_SIZE:
            logger.info("Intraday skip: insufficient universe size (%s < %s)", len(symbols), MIN_UNIVERSE_SIZE)
            return

        returns: Dict[str, float] = {}
        for symbol in symbols:
            ret = _intraday_return_since_open(symbol)
            if ret is None:
                continue
            returns[symbol] = ret

        if len(returns) < MIN_UNIVERSE_SIZE:
            logger.info("Intraday skip: insufficient valid return samples (%s)", len(returns))
            return

        red_count = sum(1 for r in returns.values() if r < 0.0)
        red_ratio = red_count / float(len(returns))
        median_ret = median(returns.values())

        if red_ratio < SPREAD_RED_RATIO_MIN or median_ret > SPREAD_MEDIAN_DROP_MAX:
            logger.info(
                "Intraday skip: spread condition failed (red_ratio=%.2f median=%.2f%% size=%s)",
                red_ratio,
                median_ret,
                len(returns),
            )
            return

        candidates = []
        for symbol, ret in returns.items():
            if ret < EXTREME_BREAKDOWN_CUTOFF:
                continue
            if ENTRY_DROP_MIN <= ret <= ENTRY_DROP_MAX:
                candidates.append((symbol, ret))

        discoveries = 0
        for symbol, ret in sorted(candidates, key=lambda x: x[1]):
            if not self.allow_discovery(symbol, period=24):
                continue

            stock = self.get_stock(symbol)
            if stock is None:
                continue

            health = self._latest_health(stock)
            if health is None:
                logger.info("Intraday skip %s: no existing health object", symbol)
                continue

            explanation = (
                f"Intraday spread dip trigger: red_ratio={red_ratio:.2f}, median={median_ret:.2f}% | "
                f"{symbol} since-open return={ret:.2f}%"
            )
            sell_instructions = [
                ("TARGET_PERCENTAGE", TARGET_PROFIT_MULTIPLIER, None),
                ("END_DAY", END_DAY_WINNER_MULTIPLIER, None),
                ("AFTER_DAYS", MAX_HOLD_DAYS, None),
            ]

            self.discovered(
                sa=sa,
                symbol=symbol,
                explanation=explanation,
                sell_instructions=sell_instructions,
                weight=1.0,
                health=health,
            )
            discoveries += 1

        state.update(
            {
                "processed_date": today,
                "triggered_date": today,
                "triggered_at": timezone.now().isoformat(timespec="seconds"),
                "spread_red_ratio": round(red_ratio, 4),
                "spread_median_return": round(median_ret, 4),
                "universe_size": len(returns),
                "candidate_count": len(candidates),
                "discoveries_count": discoveries,
            }
        )
        self._save_advisor_blob_state(state)

        logger.info(
            "Intraday processed %s: discoveries=%s candidates=%s universe=%s red_ratio=%.2f median=%.2f%%",
            today,
            discoveries,
            len(candidates),
            len(returns),
            red_ratio,
            median_ret,
        )

    def analyze(self, sa, stock):
        # This advisor only discovers intraday overlays.
        return


register(name="Intraday", python_class="Intraday")

