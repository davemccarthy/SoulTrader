"""
Flux advisor — averaging-down on a fixed watchlist (WIP).

Entry: below_ma_up (close < SMA, close > prior close).
Runs only during regular session (AdvisorBase.market_open).
Exit/add: STOP_PERCENTAGE, TARGET_PERCENTAGE, PERCENTAGE_REBUY on discovery (holdings analysis).
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Final, Tuple

import pandas as pd

from core.services.advisors.advisor import AdvisorBase, register
from core.services.financial import yahoo as financial_yahoo

logger = logging.getLogger(__name__)

# Mega-cap / growth watchlist for below_ma_up mean-reversion (27 names).
# Dropped MSTR, KO, PG, COIN; added MU, NFLX, NOW, ADBE, CRM, UBER. Not shared with test_flux_backtest.py.
FLUX_UNIVERSE: Final[Tuple[str, ...]] = (
    "AAPL",
    "ADBE",
    "AMD",
    "AMZN",
    "ARM",
    "AVGO",
    "CAT",
    "COST",
    "CRM",
    "CRWD",
    "GOOGL",
    "JPM",
    "LLY",
    "MA",
    "META",
    "MSFT",
    "MU",
    "NET",
    "NFLX",
    "NOW",
    "NVDA",
    "PLTR",
    "TSM",
    "TSLA",
    "UBER",
    "UNH",
    "V",
)

ENTRY_MA_PERIOD = 20
FLUX_TP_MULT = Decimal("1.01")
FLUX_STOP_MULT = Decimal("0.96")
FLUX_REBUY_DROP = Decimal("0.02")
FLUX_MAX_TRANCHES = Decimal("4")
FLUX_DISCOVERY_COOLDOWN_HOURS = 24
FLUX_ENDDAY_TAKE = Decimal(1.01)
FLUX_ENDWEEK_TAKE = Decimal(1.00)

class Flux(AdvisorBase):
    """Fixed-universe Flux; discovers on below_ma_up entry."""

    def discover(self, sa) -> None:
        market_status = self.market_open()
        if market_status is None:
            logger.info("Flux skip: market closed")
            return
        if market_status < 0:
            logger.info("Flux skip: market not open yet (%s min to open)", -market_status)
            return

        discoveries = 0
        sell_instructions = [
            ("STOP_PERCENTAGE", FLUX_STOP_MULT, None),
            ("TARGET_PERCENTAGE", FLUX_TP_MULT, None),
            ("DESCENDING_TREND", Decimal("-0.15"), None),
            ("PERCENTAGE_REBUY", FLUX_REBUY_DROP, FLUX_MAX_TRANCHES),
            ("END_DAY", FLUX_ENDDAY_TAKE, None),
            ("END_WEEK",FLUX_ENDWEEK_TAKE, None),
        ]

        for symbol in FLUX_UNIVERSE:
            if not self.allow_discovery(symbol, period=FLUX_DISCOVERY_COOLDOWN_HOURS):
                continue

            stock = self.get_stock(symbol)
            if stock is None:
                continue

            stock.refresh()

            hist = financial_yahoo.get_6m_history(symbol)
            if hist.empty or "close" not in hist.columns:
                logger.debug("Flux: no history for %s", symbol)
                continue

            closes = hist["close"].astype(float)
            if not self.below_ma_up(closes, ENTRY_MA_PERIOD):
                continue

            ma = float(closes.rolling(ENTRY_MA_PERIOD).mean().iloc[-1])
            close = float(closes.iloc[-1])
            explanation = (
                f"Entering below the 20-day average on upturn | "
                f"close ${close:.2f}, SMA{ENTRY_MA_PERIOD} ${ma:.2f}"
            )

            if self.discovered(
                sa,
                symbol,
                explanation,
                sell_instructions=sell_instructions,
                weight=1.0,
            ):
                discoveries += 1

        logger.info(
            "Flux discover sa=%s: universe=%d discoveries=%d",
            sa.id,
            len(FLUX_UNIVERSE),
            discoveries,
        )

    def analyze(self, sa, stock) -> None:
        return


register(name="Flux", python_class="Flux")
