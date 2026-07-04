"""
ETF advisor — discovers stocks newly added to tracked thematic ETF holdings.

Uses core.services.financial.etf_holdings for snapshot/diff/lookup.
Exit: PERCENTAGE_REBUY + PROFIT_FLAT (no AFTER_DAYS on v1).
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from core.services.advisors.advisor import AdvisorBase, register
from core.services.financial import etf_holdings

logger = logging.getLogger(__name__)

PROCESS_CUTOFF_HOUR_UTC = 12
ETF_DISCOVERY_COOLDOWN_HOURS = 24 * 30

ETF_REBUY_DROP = Decimal("0.05")
ETF_REBUY_MAX_TRANCHES = Decimal("2")
ETF_PROFIT_FLAT_RANGE = Decimal("0.05")
ETF_PROFIT_FLAT_DAYS = Decimal("15")

DEFAULT_SELL_INSTRUCTIONS = [
    ("PERCENTAGE_REBUY", ETF_REBUY_DROP, ETF_REBUY_MAX_TRANCHES),
    ("PROFIT_FLAT", ETF_PROFIT_FLAT_RANGE, ETF_PROFIT_FLAT_DAYS),
]


def build_etf_discovery_explanation(event: etf_holdings.EntrantEvent) -> str:
    weight = (
        f"{event.weight_pct:.2f}% fund weight"
        if event.weight_pct is not None
        else "Fund weight unavailable"
    )
    segments = [
        f"Added to {event.etf} ETF",
        etf_holdings.fund_character_sentence(event.theme, event.management_style),
        weight,
        f"First seen in holdings {event.holdings_date}",
        etf_holdings.inclusion_label(event.inclusion_type),
    ]
    return " | ".join(segments)


class Etf(AdvisorBase):
    """Daily ETF holdings diff → discover new constituents."""

    sell_instructions = list(DEFAULT_SELL_INSTRUCTIONS)

    def _etf_settings(self) -> Dict[str, Any]:
        state = self._advisor_blob_state()
        return {
            "min_weight_pct": float(state.get("min_weight_pct", 0.0)),
            "min_price": float(state.get("min_price", 0.0)),
            "etfs": state.get("etfs"),
        }

    def _passes_filters(self, event: etf_holdings.EntrantEvent) -> bool:
        settings = self._etf_settings()
        if settings["min_weight_pct"] > 0:
            weight = event.weight_pct or 0.0
            if weight < settings["min_weight_pct"]:
                logger.info(
                    "ETF skip %s %s: weight %.4f < min %.4f",
                    event.symbol,
                    event.etf,
                    weight,
                    settings["min_weight_pct"],
                )
                return False

        if settings["min_price"] > 0:
            stock = self.get_stock(event.symbol)
            if stock is None:
                return False
            stock.refresh()
            if float(stock.price or 0) < settings["min_price"]:
                logger.info(
                    "ETF skip %s: price %.2f < min %.2f",
                    event.symbol,
                    float(stock.price or 0),
                    settings["min_price"],
                )
                return False
        return True

    def discover(self, sa) -> None:
        today = date.today().isoformat()
        if not self.should_process_market_date_once(
            target_date=today,
            cutoff_hour_utc=PROCESS_CUTOFF_HOUR_UTC,
        ):
            return

        settings = self._etf_settings()
        etf_list: Optional[List[str]] = None
        if settings.get("etfs"):
            etf_list = [str(e).strip().upper() for e in settings["etfs"] if str(e).strip()]

        logger.info("ETF sa=%s: refresh snapshots etfs=%s", sa.id, etf_list or "default")
        results = etf_holdings.refresh_snapshots(etfs=etf_list, refresh=False)
        events = etf_holdings.entrants_from_refresh(results)

        ok_count = sum(1 for r in results if r.ok)
        stale_count = sum(1 for r in results if r.stale)
        fail_count = len(results) - ok_count
        discoveries = 0
        skipped_cooldown = 0
        skipped_filter = 0

        for event in events:
            if not self.allow_discovery(event.symbol, period=ETF_DISCOVERY_COOLDOWN_HOURS):
                skipped_cooldown += 1
                continue
            if not self._passes_filters(event):
                skipped_filter += 1
                continue

            explanation = build_etf_discovery_explanation(event)
            if self.discovered(
                sa,
                event.symbol,
                explanation,
                sell_instructions=self.sell_instructions,
                weight=1.0,
            ):
                discoveries += 1

        state = self._advisor_blob_state()
        state["last_refresh"] = {
            "date": today,
            "ok": ok_count,
            "stale": stale_count,
            "failed": fail_count,
            "entrants": len(events),
            "discoveries": discoveries,
        }
        self._save_advisor_blob_state(state)
        self.mark_market_date_processed(today)

        logger.info(
            "ETF sa=%s: refresh ok=%d stale=%d failed=%d entrants=%d discoveries=%d "
            "cooldown_skip=%d filter_skip=%d",
            sa.id,
            ok_count,
            stale_count,
            fail_count,
            len(events),
            discoveries,
            skipped_cooldown,
            skipped_filter,
        )

    def analyze(self, sa, stock) -> None:
        return


register(name="ETF", python_class="Etf")
