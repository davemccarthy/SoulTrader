"""
Midway advisor — opportunity book: SO+β universe, market-regime soft entries.

Funnel:
  market card (stance) → soft-rank opportunity universe → stabilize → discover

Does not use RSS/8-K/Meyka. WHY_SOFT is still a human/LLM gate outside this path;
v1 relies on soft bar by stance (+ elevated bar when mood is soft) and skips
extremes that usually mark company events.

Exit/add: PEAKED (min exit +4%) + gated PERCENTAGE_REBUY + DESCENDING_TREND.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import Final, List, Optional, Set

from django.conf import settings

from core.models import Holding, Profile
from core.services.advisors.advisor import AdvisorBase, register
from core.services.intraday_stabilize import price_above_minutes_ago
from core.services.market.midway_candidates import (
    SoftCandidate,
    load_universe_rows,
    rank_soft_candidates,
    soft_bar_for_stance,
)
from core.services.market.midway_state import evaluate_midway_market

logger = logging.getLogger(__name__)

# Opportunity universe (SO + β 0.9–1.6); refresh via filter_midway_opportunity_trial.py
DEFAULT_UNIVERSE = (
    Path(settings.BASE_DIR) / ".assessments" / "universe_midway_opportunity_2026-09-10.json"
)

# Discover only after opening auction noise (open + 45m ≈ 10:15 ET).
MIDWAY_MIN_MINUTES_AFTER_OPEN = 45

# Soft bar: stance sets base; nervous/deteriorating raises the bar (bigger discount).
MIDWAY_SOFT_BAR_BEARISH = 2.0

# Soft scores this high are usually company/event soft (DYN/TBBK lesson).
MIDWAY_SOFT_SCORE_EXTREME = 20.0

MIDWAY_MAX_DISCOVERIES_PER_SESSION = 6
MIDWAY_DISCOVERY_COOLDOWN_HOURS = 48
MIDWAY_STABILIZE_MINUTES = 30

# PEAKED: giveback 15%, min peak 8% → min exit +4%. Rebuy −4%, max 5 tranches.
MIDWAY_PEAKED_GIVEBACK = 15.0
MIDWAY_PEAKED_MIN_PEAK = 8.0
MIDWAY_REBUY_DROP = Decimal("0.04")
MIDWAY_REBUY_MAX_TRANCHES = Decimal("5")

# Optional hard skips (process failures); empty by default — use soft extreme gate.
MIDWAY_HARD_SKIP: Final[frozenset[str]] = frozenset()


class Midway(AdvisorBase):
    """Regime-aware soft discovery on the Midway opportunity universe."""

    sell_instructions = [
        ("PEAKED", MIDWAY_PEAKED_GIVEBACK, MIDWAY_PEAKED_MIN_PEAK),
        ("PERCENTAGE_REBUY", MIDWAY_REBUY_DROP, MIDWAY_REBUY_MAX_TRANCHES),
        ("DESCENDING_TREND", -0.20, None),
    ]

    def discover(self, sa) -> None:
        market_status = self.market_open()
        if market_status is None:
            logger.info("Midway skip: market closed")
            return
        if market_status < 0:
            logger.info(
                "Midway skip: market not open yet (%s min to open)",
                -market_status,
            )
            return
        if market_status < MIDWAY_MIN_MINUTES_AFTER_OPEN:
            logger.info(
                "Midway skip: before discover window (%s min open; need %s)",
                market_status,
                MIDWAY_MIN_MINUTES_AFTER_OPEN,
            )
            return

        state = evaluate_midway_market()
        if state.stance == "no_deployment":
            logger.info(
                "Midway skip: stance=no_deployment (%s)",
                state.permission,
            )
            return

        universe_path = self._universe_path()
        if not universe_path.exists():
            logger.warning("Midway skip: universe missing %s", universe_path)
            return

        try:
            rows = load_universe_rows(universe_path)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Midway skip: universe load failed: %s", exc)
            return

        if not rows:
            logger.info("Midway skip: empty universe %s", universe_path)
            return

        candidates = rank_soft_candidates(state, rows, prefer_soft_sectors=True)
        bar = soft_bar_for_stance(state.stance)
        if bar is None:
            logger.info("Midway skip: no soft bar for stance=%s", state.stance)
            return
        if state.stance == "active_soft" and state.mood in ("deteriorating", "nervous"):
            bar = max(bar, MIDWAY_SOFT_BAR_BEARISH)

        held = self._held_symbols()
        discoveries = 0
        skipped_held = 0
        skipped_extreme = 0
        skipped_bar = 0
        skipped_stabilize = 0
        skipped_cooldown = 0
        skipped_hard = 0

        for cand in candidates:
            if discoveries >= MIDWAY_MAX_DISCOVERIES_PER_SESSION:
                break

            score = cand.soft_score
            if score is None or score < bar:
                skipped_bar += 1
                continue

            sym = cand.symbol
            if sym in MIDWAY_HARD_SKIP:
                skipped_hard += 1
                continue
            if sym in held:
                skipped_held += 1
                continue
            if score >= MIDWAY_SOFT_SCORE_EXTREME:
                skipped_extreme += 1
                logger.info(
                    "Midway skip %s: extreme soft_score=%.2f (likely company soft)",
                    sym,
                    score,
                )
                continue
            if not self.allow_discovery(sym, period=MIDWAY_DISCOVERY_COOLDOWN_HOURS):
                skipped_cooldown += 1
                continue

            stock = self.get_stock(sym)
            if stock is None:
                continue
            stock.refresh()
            stabilized = price_above_minutes_ago(stock, minutes=MIDWAY_STABILIZE_MINUTES)
            if stabilized is not True:
                skipped_stabilize += 1
                continue

            explanation = self._discovery_explanation(cand, state.stance, state.mood, bar)
            if self.discovered(
                sa,
                sym,
                explanation,
                sell_instructions=list(self.sell_instructions),
                weight=1.0,
                meta={
                    "midway": {
                        "stance": state.stance,
                        "mood": state.mood,
                        "soft_score": score,
                        "so_pair": cand.so_pair,
                        "sleeve": cand.sleeve,
                        "soft_bar": bar,
                    }
                },
            ):
                discoveries += 1
                held.add(sym)

        logger.info(
            "Midway sa=%s: stance=%s mood=%s bar=%s universe=%d "
            "discoveries=%d held_skip=%d extreme_skip=%d bar_skip=%d "
            "stabilize_skip=%d cooldown_skip=%d hard_skip=%d",
            sa.id,
            state.stance,
            state.mood,
            bar,
            len(rows),
            discoveries,
            skipped_held,
            skipped_extreme,
            skipped_bar,
            skipped_stabilize,
            skipped_cooldown,
            skipped_hard,
        )

    def analyze(self, sa, stock) -> None:
        return

    def _universe_path(self) -> Path:
        raw = self._advisor_blob_state().get("universe_path")
        if raw:
            p = Path(str(raw))
            if not p.is_absolute():
                p = Path(settings.BASE_DIR) / p
            return p
        return DEFAULT_UNIVERSE

    def _held_symbols(self) -> Set[str]:
        """Symbols already held by funds that subscribe to Midway."""
        fund_ids = [
            p.id
            for p in Profile.objects.filter(enabled=True).only("id", "advisors")
            if "Midway" in (p.advisors or [])
        ]
        qs = Holding.objects.filter(shares__gt=0)
        if fund_ids:
            qs = qs.filter(fund_id__in=fund_ids)
        else:
            qs = qs.none()
        return {
            str(s).strip().upper()
            for s in qs.values_list("stock__symbol", flat=True).distinct()
            if s
        }

    @staticmethod
    def _discovery_explanation(
        cand: SoftCandidate,
        stance: str,
        mood: str,
        bar: Optional[float],
    ) -> str:
        soft = f"{cand.soft_score:.2f}" if cand.soft_score is not None else "n/a"
        vs_spy = (
            f"{cand.vs_spy_20d_pct:+.1f}%"
            if cand.vs_spy_20d_pct is not None
            else "n/a"
        )
        vs_sec = (
            f"{cand.vs_sector_20d_pct:+.1f}%"
            if cand.vs_sector_20d_pct is not None
            else "n/a"
        )
        bar_s = f"{bar:.1f}" if bar is not None else "none"
        return (
            f"Midway soft entry | SO {cand.so_pair} soft={soft} "
            f"(bar {bar_s}) | vsSPY20 {vs_spy} vsSec20 {vs_sec} | "
            f"sleeve {cand.sleeve} {cand.sector_etf} | "
            f"stance {stance}/{mood} | {MIDWAY_STABILIZE_MINUTES}m stabilize"
        )


register(name="Midway", python_class="Midway")
