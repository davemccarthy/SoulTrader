"""
Midway advisor — opportunity book: SO+β universe, market-regime soft entries.

Funnel:
  market card (stance) → soft-rank → below session open → sector intraday OK
  → stabilize → WHY_SOFT (LLM) → discover

Does not use RSS/8-K/Meyka. Soft bar by stance (active_soft ≥ 2.5), session
discount vs open, sector not freefalling on the day, stabilize, then WHY_SOFT
A/B gate (market/sector discount — skip C/D company/structural traps).

Exit/add: PEAKED (arm +2% / min exit +1%), rebuy −2% (max 3),
EOD +2%, DESCENDING_TREND. (No hard TP — overlaps PEAKED arm.)
"""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set, Tuple

from django.conf import settings

from core.models import Holding, Profile
from core.services.advisors.advisor import (
    AdvisorBase,
    discovery_trade_explanation_lead,
    register,
)
from core.services.financial import yahoo as financial_yahoo
from core.services.intraday_stabilize import price_above_minutes_ago
from core.services.market.midway_candidates import (
    SoftCandidate,
    load_universe_rows,
    rank_soft_candidates,
    soft_bar_for_stance,
)
from core.services.market.midway_state import MidwayMarketState, evaluate_midway_market

logger = logging.getLogger(__name__)

# Opportunity universe (SO + β 0.9–1.6); refresh via filter_midway_opportunity_trial.py
DEFAULT_UNIVERSE = (
    Path(settings.BASE_DIR) / ".assessments" / "universe_midway_opportunity_2026-09-10.json"
)

# Discover only after first-hour noise (open + 60m ≈ 10:30 ET).
MIDWAY_MIN_MINUTES_AFTER_OPEN = 60

# Soft bar: stance sets base; active_soft requires a slightly bigger discount than the
# shared 1.0 default (today's flood still cleared 2.0 — bump hunt bar).
MIDWAY_SOFT_BAR_ACTIVE = 2.5

# Soft scores this high are usually company/event soft (DYN/TBBK lesson).
MIDWAY_SOFT_SCORE_EXTREME = 20.0

MIDWAY_MAX_DISCOVERIES_PER_SESSION = 6
MIDWAY_DISCOVERY_COOLDOWN_HOURS = 48
MIDWAY_STABILIZE_MINUTES = 30

# WHY_SOFT: live LLM class before discover. A/B = market/sector discount (pass);
# C/D/E/F blocked in phase 1 (company/structural/valuation/mixed — buy-gate later).
WHY_SOFT_PASS_CLASSES: Final[frozenset[str]] = frozenset({"A", "B"})
WHY_SOFT_LLM_TIMEOUT_S = 90.0
WHY_SOFT_HEADLINE_LIMIT = 5

# Exit pack: PEAKED arms at +2% (min exit +1%) — no hard TP (redundant with arm level);
# rebuy −2% / max 3; EOD +2% in last 60m; DT cuts freefalls.
MIDWAY_PEAKED_GIVEBACK = 15.0
MIDWAY_PEAKED_MIN_PEAK = 2.0
MIDWAY_REBUY_DROP = Decimal("0.02")
MIDWAY_REBUY_MAX_TRANCHES = Decimal("3")
MIDWAY_EOD_MULT = Decimal("1.02")
MIDWAY_EOD_MINUTES_BEFORE_CLOSE = Decimal("60")
MIDWAY_DT = -0.20

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


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def _why_soft_allows(parsed: Dict[str, Any]) -> bool:
    """Phase 1: discover only market/sector discounts (A/B); never SKIP."""
    why = str(parsed.get("why_soft") or "").strip().upper()
    action = str(parsed.get("action") or "").strip().upper()
    if action == "SKIP":
        return False
    return why in WHY_SOFT_PASS_CLASSES


def _normalize_why_soft_payload(raw: Any, symbol: str) -> Optional[Dict[str, Any]]:
    """Accept a single object or a one-element / matching-symbol array."""
    if isinstance(raw, list):
        if not raw:
            return None
        for item in raw:
            if not isinstance(item, dict):
                continue
            if str(item.get("symbol") or "").strip().upper() == symbol:
                return item
        return raw[0] if isinstance(raw[0], dict) else None
    if isinstance(raw, dict):
        return raw
    return None


def _build_why_soft_prompt(
    cand: SoftCandidate,
    state: MidwayMarketState,
    headlines: List[str],
) -> str:
    soft = (
        f"{cand.soft_score:.2f}" if cand.soft_score is not None else "n/a"
    )
    soft_secs = ",".join(state.soft_sectors) or "none"
    hot_secs = ",".join(state.hot_sectors) or "none"
    hl_lines = "\n".join(f"- {h}" for h in headlines) or "- (none)"
    return f"""You are classifying one soft mid-cap for SoulTrader MIDWAY.
MIDWAY wants quality names sold for market/sector reasons — not broken stories.
SO already passed. Soft-rank already said this name is relatively weak vs SPY/sector.
Your job is ONLY: WHY_SOFT class + discount quality. Do NOT invent a buy recommendation.
You may use search/recent news to judge the cause of softness.

Market context: TREND={state.trend} MOOD={state.mood} STANCE={state.stance} \
PERMISSION={state.permission} soft_sectors={soft_secs} hot_sectors={hot_secs}

WHY_SOFT classes (pick one letter):
A = Market selloff (broad pressure; peers also soft)
B = Sector selloff (sleeve/peers repricing)
C = Company-specific event (earnings/guidance/litigation/product/management)
D = Fundamental deterioration (economics correctly worsening)
E = Valuation reset (business ok; prior valuation unjustified)
F = Mixed (some real concern + some indiscriminate selling)

Discount quality: HIGH | MEDIUM | LOW
- HIGH: good business, soft mostly A/B, estimates/story intact
- MEDIUM: mixed or unresolved but SO intact
- LOW: C/D heavy, governance/controls, ADR-process miss, litigation, clear broken thesis

Action: FOLLOW | WATCH | SKIP
- FOLLOW = soft quality worth Midway discovery (prefer A/B, HIGH)
- WATCH = attractive tape but unresolved fundamental question
- SKIP = should not be discovered for MIDWAY

Desk row:
symbol={cand.symbol}
so_pair={cand.so_pair}
soft_score={soft}
sleeve={cand.sleeve}
sector={cand.sector}
sector_etf={cand.sector_etf}
ret_5d_pct={_fmt_pct(cand.ret_5d_pct)}
ret_20d_pct={_fmt_pct(cand.ret_20d_pct)}
vs_spy_20d_pct={_fmt_pct(cand.vs_spy_20d_pct)}
vs_sector_20d_pct={_fmt_pct(cand.vs_sector_20d_pct)}

Recent headlines:
{hl_lines}

Return ONLY a JSON object (no markdown):
{{
  "symbol": "{cand.symbol}",
  "why_soft": "A"|"B"|"C"|"D"|"E"|"F",
  "discount_quality": "HIGH"|"MEDIUM"|"LOW",
  "action": "FOLLOW"|"WATCH"|"SKIP",
  "primary": "short reason (one sentence)",
  "unresolved": "one question or null",
  "confidence": 1-5
}}
"""


class Midway(AdvisorBase):
    """Regime-aware soft discovery on the Midway opportunity universe."""

    sell_instructions = [
        ("PEAKED", MIDWAY_PEAKED_GIVEBACK, MIDWAY_PEAKED_MIN_PEAK),
        ("PERCENTAGE_REBUY", MIDWAY_REBUY_DROP, MIDWAY_REBUY_MAX_TRANCHES),
        ("END_DAY", MIDWAY_EOD_MULT, MIDWAY_EOD_MINUTES_BEFORE_CLOSE),
        ("DESCENDING_TREND", MIDWAY_DT, None),
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
        skipped_why_soft = 0
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

            why = self._classify_why_soft(cand, state)
            if why is None or not _why_soft_allows(why):
                skipped_why_soft += 1
                why_s = (why or {}).get("why_soft", "?")
                act_s = (why or {}).get("action", "fail")
                primary = str((why or {}).get("primary") or "")[:120]
                logger.info(
                    "Midway skip %s: WHY_SOFT why=%s action=%s primary=%s",
                    sym,
                    why_s,
                    act_s,
                    primary,
                )
                continue

            explanation = self._discovery_explanation(
                cand, state.stance, state.mood, bar, why
            )
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
                        "why_soft": why.get("why_soft"),
                        "discount_quality": why.get("discount_quality"),
                        "why_action": why.get("action"),
                        "why_primary": why.get("primary"),
                        "why_confidence": why.get("confidence"),
                    }
                },
            ):
                discoveries += 1
                held.add(sym)

        logger.info(
            "Midway sa=%s: stance=%s mood=%s bar=%s universe=%d "
            "discoveries=%d held_skip=%d extreme_skip=%d bar_skip=%d "
            "below_open_skip=%d sector_skip=%d stabilize_skip=%d "
            "why_soft_skip=%d cooldown_skip=%d hard_skip=%d",
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
            skipped_why_soft,
            skipped_cooldown,
            skipped_hard,
        )

    def analyze(self, sa, stock) -> None:
        return

    def _classify_why_soft(
        self,
        cand: SoftCandidate,
        state: MidwayMarketState,
    ) -> Optional[Dict[str, Any]]:
        """LLM WHY_SOFT class; None on failure (fail closed — no discover)."""
        try:
            headlines = financial_yahoo.latest_headlines(
                cand.symbol,
                limit=WHY_SOFT_HEADLINE_LIMIT,
                max_age_days=7,
            )
        except Exception as exc:
            logger.debug("Midway WHY_SOFT headlines failed for %s: %s", cand.symbol, exc)
            headlines = []
        # Drop placeholder-only lists
        headlines = [
            h
            for h in headlines
            if h
            and "No recent public headlines" not in h
            and "No ticker provided" not in h
        ]

        prompt = _build_why_soft_prompt(cand, state, headlines)
        try:
            model, raw = self.ask_llm(
                prompt,
                use_search=True,
                timeout=WHY_SOFT_LLM_TIMEOUT_S,
            )
        except Exception as exc:
            logger.warning("Midway WHY_SOFT LLM error %s: %s", cand.symbol, exc)
            return None

        if raw is None:
            logger.warning("Midway WHY_SOFT no response %s model=%s", cand.symbol, model)
            return None

        # Some paths return JSON text; prefer already-parsed.
        if isinstance(raw, str):
            raw = self._extract_json(raw)

        parsed = _normalize_why_soft_payload(raw, cand.symbol)
        if not parsed:
            logger.warning(
                "Midway WHY_SOFT bad payload %s model=%s type=%s",
                cand.symbol,
                model,
                type(raw).__name__,
            )
            return None

        why = str(parsed.get("why_soft") or "").strip().upper()
        action = str(parsed.get("action") or "").strip().upper()
        quality = str(parsed.get("discount_quality") or "").strip().upper()
        primary = str(parsed.get("primary") or "").strip()
        if why not in {"A", "B", "C", "D", "E", "F"}:
            logger.warning(
                "Midway WHY_SOFT invalid class %s why=%r model=%s",
                cand.symbol,
                parsed.get("why_soft"),
                model,
            )
            return None
        if action not in {"FOLLOW", "WATCH", "SKIP"}:
            logger.warning(
                "Midway WHY_SOFT invalid action %s action=%r model=%s",
                cand.symbol,
                parsed.get("action"),
                model,
            )
            return None

        conf = parsed.get("confidence")
        try:
            conf_i = int(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf_i = None

        out = {
            "symbol": cand.symbol,
            "why_soft": why,
            "discount_quality": quality or "MEDIUM",
            "action": action,
            "primary": primary,
            "unresolved": parsed.get("unresolved"),
            "confidence": conf_i,
            "model": model,
        }
        logger.info(
            "Midway WHY_SOFT %s: why=%s quality=%s action=%s conf=%s model=%s primary=%s",
            cand.symbol,
            why,
            out["discount_quality"],
            action,
            conf_i,
            model,
            primary[:120],
        )
        return out

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
        why: Dict[str, Any],
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
        why_letter = str(why.get("why_soft") or "?")
        primary = str(why.get("primary") or "").strip() or "no primary"
        # Trade.explanation uses first | segment — lead with class + LLM summary.
        lead = discovery_trade_explanation_lead(
            f"WHY_SOFT {why_letter} — {primary}"
        )
        quality = str(why.get("discount_quality") or "?")
        action = str(why.get("action") or "?")
        conf = why.get("confidence")
        conf_s = str(conf) if conf is not None else "?"
        return (
            f"{lead} | "
            f"SO {cand.so_pair} soft={soft} (bar {bar_s}) | "
            f"quality {quality} action {action} conf {conf_s} | "
            f"below open | sector OK | "
            f"vsSPY20 {vs_spy} vsSec20 {vs_sec} | "
            f"sleeve {cand.sleeve} {cand.sector_etf} | "
            f"stance {stance}/{mood} | {MIDWAY_STABILIZE_MINUTES}m stabilize"
        )


register(name="Midway", python_class="Midway")
