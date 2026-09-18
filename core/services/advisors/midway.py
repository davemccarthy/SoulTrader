"""
Midway advisor — opportunity book: SO+β universe, market-regime soft entries.

Funnel:
  market card (stance) → soft-rank → below session open → sector intraday OK
  → stabilize → discover

Does not use RSS/8-K/Meyka. WHY_SOFT is still a human/LLM gate outside this path;
v1 relies on soft bar by stance (active_soft ≥ 2.5), session discount vs open,
sector not freefalling on the day, and stabilize — skips extremes that usually
mark company events.

Exit/add: PEAKED (min exit +3%) + gated PERCENTAGE_REBUY (max 3) + DESCENDING_TREND.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import Dict, Final, Optional, Set, Tuple

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

# Soft bar: stance sets base; active_soft requires a slightly bigger discount than the
# shared 1.0 default (today's flood still cleared 2.0 — bump hunt bar).
MIDWAY_SOFT_BAR_ACTIVE = 2.5

# Soft scores this high are usually company/event soft (DYN/TBBK lesson).
MIDWAY_SOFT_SCORE_EXTREME = 20.0

MIDWAY_MAX_DISCOVERIES_PER_SESSION = 6
MIDWAY_DISCOVERY_COOLDOWN_HOURS = 48
MIDWAY_STABILIZE_MINUTES = 30

# PEAKED: giveback 15%, min peak 6% → min exit +3% (balance harvest vs multi-day hold).
# Rebuy −4%, max 3 tranches (less hole-digging than 5).
MIDWAY_PEAKED_GIVEBACK = 15.0
MIDWAY_PEAKED_MIN_PEAK = 6.0
MIDWAY_REBUY_DROP = Decimal("0.04")
MIDWAY_REBUY_MAX_TRANCHES = Decimal("3")

# Optional hard skips (process failures); empty by default — use soft extreme gate.
MIDWAY_HARD_SKIP: Final[frozenset[str]] = frozenset()


def _session_last_and_open(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (last, session open) from Yahoo fast_info; either may be None."""
    import yfinance as yf

    try:
        info = yf.Ticker(symbol).fast_info
        last_raw = info.get("lastPrice") or info.get("regularMarketPrice")
        open_raw = info.get("regularMarketOpen") or info.get("open")
        last = float(last_raw) if last_raw is not None else None
        open_px = float(open_raw) if open_raw is not None else None
        if last is not None and last <= 0:
            last = None
        if open_px is not None and open_px <= 0:
            open_px = None
        return last, open_px
    except Exception as exc:
        logger.debug("Midway session quote failed for %s: %s", symbol, exc)
        return None, None


def _session_open_px(symbol: str) -> Optional[float]:
    """Today's session open from Yahoo fast_info; None if unavailable."""
    _, open_px = _session_last_and_open(symbol)
    return open_px


def _price_below_session_open(stock) -> Optional[bool]:
    """
    True when last price is strictly below today's open (session discount).
    False when at/above open. None when open or price missing — skip discover.
    """
    try:
        px = float(stock.price) if stock.price is not None else 0.0
    except (TypeError, ValueError):
        return None
    if px <= 0:
        return None
    open_px = _session_open_px(stock.symbol)
    if open_px is None:
        return None
    return px < open_px


def _symbol_stabilized(symbol: str, last: float, minutes: int) -> Optional[bool]:
    """True when last is above the ~minutes-ago 15m close (same idea as stock stabilize)."""
    import pandas as pd
    import yfinance as yf

    try:
        if last <= 0:
            return None
        hist = yf.Ticker(symbol).history(period="1d", interval="15m")
        if hist.empty or "Close" not in hist.columns:
            return None
        idx = pd.to_datetime(hist.index, utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes)
        eligible = idx <= cutoff
        if not eligible.any():
            return None
        px_ago = float(hist.loc[eligible, "Close"].astype(float).iloc[-1])
        if px_ago <= 0:
            return None
        return last > px_ago
    except Exception as exc:
        logger.debug("Midway sector stabilize failed for %s: %s", symbol, exc)
        return None


def _sector_intraday_allows(sector_etf: str) -> Optional[bool]:
    """
    Intraday safety for Midway discover (not a multi-day soft-sleeve veto).

    True when sector ETF is flat/green vs open, or below open but stabilizing.
    False when below open and still falling/flat (hold off).
    None when quotes missing — skip discover.
    """
    etf = (sector_etf or "").strip().upper()
    if not etf:
        return None
    last, open_px = _session_last_and_open(etf)
    if last is None or open_px is None:
        return None
    if last >= open_px:
        return True
    return _symbol_stabilized(etf, last, MIDWAY_STABILIZE_MINUTES)


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
        if state.stance == "active_soft":
            bar = max(bar, MIDWAY_SOFT_BAR_ACTIVE)

        held = self._held_symbols()
        discoveries = 0
        skipped_held = 0
        skipped_extreme = 0
        skipped_bar = 0
        skipped_below_open = 0
        skipped_sector = 0
        skipped_stabilize = 0
        skipped_cooldown = 0
        skipped_hard = 0
        sector_cache: Dict[str, Optional[bool]] = {}

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

            below_open = _price_below_session_open(stock)
            if below_open is not True:
                skipped_below_open += 1
                continue

            etf = (cand.sector_etf or "").strip().upper()
            if etf not in sector_cache:
                sector_cache[etf] = _sector_intraday_allows(etf)
            sector_ok = sector_cache[etf]
            if sector_ok is not True:
                skipped_sector += 1
                logger.info(
                    "Midway skip %s: sector %s not OK intraday (hold off while sector soft/falling)",
                    sym,
                    etf or "?",
                )
                continue

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
                        "below_open": True,
                        "sector_etf": etf,
                        "sector_intraday_ok": True,
                    }
                },
            ):
                discoveries += 1
                held.add(sym)

        logger.info(
            "Midway sa=%s: stance=%s mood=%s bar=%s universe=%d "
            "discoveries=%d held_skip=%d extreme_skip=%d bar_skip=%d "
            "below_open_skip=%d sector_skip=%d stabilize_skip=%d "
            "cooldown_skip=%d hard_skip=%d",
            sa.id,
            state.stance,
            state.mood,
            bar,
            len(rows),
            discoveries,
            skipped_held,
            skipped_extreme,
            skipped_bar,
            skipped_below_open,
            skipped_sector,
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
            f"SO {cand.so_pair} soft={soft} (bar {bar_s}) | below open | sector OK | "
            f"vsSPY20 {vs_spy} vsSec20 {vs_sec} | "
            f"sleeve {cand.sleeve} {cand.sector_etf} | "
            f"stance {stance}/{mood} | {MIDWAY_STABILIZE_MINUTES}m stabilize"
        )


register(name="Midway", python_class="Midway")
