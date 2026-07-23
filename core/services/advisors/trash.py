"""
Trash Advisor — bad-sell recovery.

Phase 1: collect SELL losses from the current SA into the watchlist.
Phase 2: on below_ma_up + consensus traffic-light, rediscover with rebuy / peaked / flat-profit exits.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from core.models import Trade
from core.services.advisors.advisor import AdvisorBase, register
from core.services.financial import yahoo as financial_yahoo

logger = logging.getLogger(__name__)

LOSS_THRESHOLD_PCT = Decimal("-5.0")
WATCH_DAYS = 21
ENTRY_MA_PERIOD = 20
TRASH_DISCOVERY_COOLDOWN_HOURS = 24

TRASH_REBUY_DROP = Decimal("0.05")
TRASH_REBUY_MAX_TRANCHES = Decimal("5")
# Min peak 20 → derived min exit 10% (÷2); arm on a real move, floor still green.
TRASH_PEAKED_GIVEBACK_PCT = Decimal("15.0")
TRASH_PEAKED_MIN_GAIN_PCT = Decimal("20.0")
TRASH_PROFIT_FLAT_RANGE = Decimal("0.05")
TRASH_PROFIT_FLAT_DAYS = Decimal("12")
TRASH_DESCENDING_TREND = Decimal("-0.20")
TRASH_MIN_ANALYSTS = 3

CONSENSUS_MEAN_BUY_MAX = 2.5
CONSENSUS_MEAN_SELL_MIN = 3.5
CONSENSUS_UPSIDE_GREEN_MIN = 10.0
CONSENSUS_WORSEN_DELTA = 0.05


def sell_loss_pct(trade: Trade) -> Optional[Decimal]:
    """Return P&L percent for a SELL trade, or None when not computable."""
    if trade.cost is None or trade.price is None:
        return None
    cost = Decimal(str(trade.cost))
    price = Decimal(str(trade.price))
    if cost <= 0 or price <= 0:
        return None
    return (price - cost) / cost * Decimal("100")


def _trade_snapshot(trade: Trade, loss_pct: Decimal) -> Dict[str, Any]:
    fund = trade.fund
    return {
        "trade_id": trade.id,
        "fund": fund.name if fund else None,
        "fund_id": fund.id if fund else None,
        "loss_pct": float(loss_pct),
        "sell_price": float(trade.price),
        "cost_basis": float(trade.cost),
        "shares": trade.shares,
        "sell_explanation": (trade.explanation or "")[:256],
    }


def _consensus_meta_slice(consensus: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "recommendation_mean": consensus.get("recommendation_mean"),
        "recommendation_key": consensus.get("recommendation_key"),
        "upside_to_mean_pct": consensus.get("upside_to_mean_pct"),
        "analyst_count": consensus.get("analyst_count"),
    }


class Trash(AdvisorBase):
    """Collect bad sells, then rediscover on technical upturn + consensus confirmation."""

    def discover(self, sa) -> None:
        self._collect_bad_sells(sa)
        self._rediscover_watchlist(sa)

    def _collect_bad_sells(self, sa) -> None:
        grouped = self._group_bad_sells(sa)
        if not grouped:
            logger.info("Trash phase-1 sa=%s: no bad sells at or below %.1f%%", sa.id, LOSS_THRESHOLD_PCT)
            return

        added = 0
        skipped_watched = 0

        for symbol, sells in sorted(grouped.items()):
            if self.watched(symbol):
                skipped_watched += 1
                continue

            worst = min(sells, key=lambda s: s["loss_pct"])
            explanation = self._watch_explanation(symbol, worst, len(sells))
            meta = self._watch_meta(sa, symbol, worst, sells)
            consensus = self.stock_consensus(symbol)
            meta["consensus_at_watch"] = _consensus_meta_slice(consensus)

            if self.watch(symbol, explanation, days=WATCH_DAYS, meta=meta):
                added += 1

        logger.info(
            "Trash phase-1 sa=%s: symbols=%d added=%d skipped_watched=%d threshold=%.1f%%",
            sa.id,
            len(grouped),
            added,
            skipped_watched,
            LOSS_THRESHOLD_PCT,
        )

    def _rediscover_watchlist(self, sa) -> None:
        market_status = self.market_open()
        if market_status is None:
            logger.info("Trash phase-2 sa=%s: skip (market closed)", sa.id)
            return
        if market_status < 0:
            logger.info("Trash phase-2 sa=%s: skip (pre-open)", sa.id)
            return

        entries = list(self.watchlist())
        if not entries:
            logger.info("Trash phase-2 sa=%s: empty watchlist", sa.id)
            return

        discoveries = 0
        skipped = defaultdict(int)

        for entry in entries:
            symbol = entry.stock.symbol.strip().upper()
            meta = entry.meta or {}

            if not self.allow_discovery(symbol, period=TRASH_DISCOVERY_COOLDOWN_HOURS):
                skipped["cooldown"] += 1
                continue

            stock = self.get_stock(symbol)
            if stock is None:
                skipped["no_stock"] += 1
                continue

            stock.refresh()

            hist = financial_yahoo.get_6m_history(symbol)
            if hist.empty or "close" not in hist.columns:
                skipped["no_history"] += 1
                continue

            closes = hist["close"].astype(float)
            if not self.below_ma_up(closes, ENTRY_MA_PERIOD):
                skipped["no_upturn"] += 1
                continue

            signal, consensus = self._consensus_signal(symbol, meta)
            if signal == "red":
                skipped["consensus_red"] += 1
                logger.debug("Trash phase-2 %s: consensus red", symbol)
                continue

            loss_pct = float(meta.get("loss_pct") or 0.0)
            sell_instructions = self._sell_instructions()
            weight = Decimal("1.25") if signal == "green" else Decimal("1.0")

            ma = float(closes.rolling(ENTRY_MA_PERIOD).mean().iloc[-1])
            close = float(closes.iloc[-1])
            explanation = self._discovery_explanation(
                symbol, loss_pct, meta, close, ma, signal, consensus
            )

            if self.discovered(
                sa,
                symbol,
                explanation,
                sell_instructions=sell_instructions,
                weight=float(weight),
            ):
                discoveries += 1

        logger.info(
            "Trash phase-2 sa=%s: watchlist=%d discoveries=%d skipped=%s",
            sa.id,
            len(entries),
            discoveries,
            dict(skipped),
        )

    def _consensus_signal(
        self, symbol: str, meta: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Traffic light from yfinance consensus. Returns ('green'|'yellow'|'red', snapshot).
        No analyst coverage -> yellow (do not block).
        """
        consensus = self.stock_consensus(symbol)
        if not consensus.get("is_usable"):
            return "yellow", consensus

        mean = consensus.get("recommendation_mean")
        key = (consensus.get("recommendation_key") or "").strip().lower()
        upside = consensus.get("upside_to_mean_pct")
        analyst_count = consensus.get("analyst_count")

        if analyst_count is not None and analyst_count < TRASH_MIN_ANALYSTS:
            return "yellow", consensus

        if key == "sell" or (mean is not None and mean > CONSENSUS_MEAN_SELL_MIN):
            return "red", consensus

        if upside is not None and upside < 0:
            return "red", consensus

        at_watch = meta.get("consensus_at_watch") or {}
        watch_mean = at_watch.get("recommendation_mean")
        if mean is not None and watch_mean is not None:
            if float(mean) > float(watch_mean) + CONSENSUS_WORSEN_DELTA:
                return "red", consensus

        if (
            mean is not None
            and mean <= CONSENSUS_MEAN_BUY_MAX
            and upside is not None
            and upside >= CONSENSUS_UPSIDE_GREEN_MIN
        ):
            return "green", consensus

        return "yellow", consensus

    def _sell_instructions(self) -> List[Tuple[str, Any, Any]]:
        """Scale-in on dips; exit on peak fade, stalled profit, or broken trend."""
        return [
            ("PERCENTAGE_REBUY", TRASH_REBUY_DROP, TRASH_REBUY_MAX_TRANCHES),
            ("PEAKED", TRASH_PEAKED_GIVEBACK_PCT, TRASH_PEAKED_MIN_GAIN_PCT),
            ("PROFIT_FLAT", TRASH_PROFIT_FLAT_RANGE, TRASH_PROFIT_FLAT_DAYS),
            ("DESCENDING_TREND", TRASH_DESCENDING_TREND, None),
        ]

    def _discovery_explanation(
        self,
        symbol: str,
        loss_pct: float,
        meta: Dict[str, Any],
        close: float,
        ma: float,
        signal: str,
        consensus: Dict[str, Any],
    ) -> str:
        sell_price = meta.get("sell_price")
        fund = meta.get("fund") or "?"
        mean = consensus.get("recommendation_mean")
        upside = consensus.get("upside_to_mean_pct")
        parts = [
            f"Trash rebound {symbol}",
            f"bad sell {loss_pct:.1f}% ({fund})",
            f"below_ma_up close ${close:.2f} < SMA{ENTRY_MA_PERIOD} ${ma:.2f}",
            f"consensus {signal}",
        ]
        if sell_price is not None:
            parts.append(f"prior sell @ ${float(sell_price):.2f}")
        if mean is not None:
            parts.append(f"rec mean {mean:.2f}")
        if upside is not None:
            parts.append(f"upside {upside:.1f}%")
        return " | ".join(parts)[:500]

    def _group_bad_sells(self, sa) -> Dict[str, List[Dict[str, Any]]]:
        sells_qs = (
            Trade.objects.filter(
                sa=sa,
                action="SELL",
                cost__isnull=False,
                fund__enabled=True,
            )
            .select_related("stock", "fund")
            .order_by("stock__symbol", "created")
        )

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for trade in sells_qs:
            loss_pct = sell_loss_pct(trade)
            if loss_pct is None or loss_pct > LOSS_THRESHOLD_PCT:
                continue
            symbol = (trade.stock.symbol if trade.stock else "").strip().upper()
            if not symbol:
                continue
            grouped[symbol].append(_trade_snapshot(trade, loss_pct))

        return dict(grouped)

    def _watch_explanation(self, symbol: str, worst: Dict[str, Any], sell_count: int) -> str:
        fund = worst.get("fund") or "unknown"
        loss = worst["loss_pct"]
        price = worst["sell_price"]
        base = f"Bad sell {symbol} {loss:.1f}% ({fund} @ ${price:.2f})"
        if sell_count > 1:
            base += f" +{sell_count - 1} more fund(s)"
        return base[:500]

    def _watch_meta(
        self,
        sa,
        symbol: str,
        worst: Dict[str, Any],
        sells: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "phase": 1,
            "source": "bad_sell",
            "sa_id": sa.id,
            "symbol": symbol,
            "loss_pct": worst["loss_pct"],
            "sell_price": worst["sell_price"],
            "cost_basis": worst["cost_basis"],
            "fund": worst.get("fund"),
            "fund_id": worst.get("fund_id"),
            "sell_explanation": worst.get("sell_explanation", ""),
            "trade_id": worst.get("trade_id"),
            "sell_count": len(sells),
            "sells": sells,
        }


register(name="Trash", python_class="Trash")
