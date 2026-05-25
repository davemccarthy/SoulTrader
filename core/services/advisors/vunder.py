"""
Vunder advisor — undervalued watchlist + intraday trend investigation.

Phase 1 (any time): build liquid universe → v2 p/fv + quality → persist watchlist on Advisor.blob.
Phase 2 (regular session only): price up + volume vs average → LLM change investigation → optional Discovery.

LLM investigates what changed; it does not gate on Buy/Strong Buy consensus.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf

from core.services.advisors.advisor import AdvisorBase, register
from core.services.financial import polygon as financial_polygon
from core.services.financial import yahoo as financial_yahoo
from core.services.health.assess import composite_from_scores, run_component_scores
from core.services.health.ratings import score_to_rating

logger = logging.getLogger(__name__)

# --- Universe (Polygon band, aligned with test_vunder) ---
MIN_PRICE = 8.0
MAX_PRICE = 80.0
MIN_SESSION_VOLUME = 2_000_000
MAX_EVAL_SYMBOLS = 400
MAX_WATCHLIST = 40
MAX_P_FV = 0.90
MIN_QUALITY_SCORE = 70.0
UNIVERSE_REBUILD_DAYS = 7

# --- Monitor (LLM trigger, not a buy signal) ---
VOL_RATIO_MIN = 1.2
VOL_LOOKBACK_DAYS = 20
MAX_P_FV_RECHECK = 0.95

# --- LLM / discovery ---
LLM_COOLDOWN_DAYS = 5
DISCOVERY_COOLDOWN_HOURS = 48
MIN_NARRATIVE_SHIFT = 6
MAX_RISK_SCORE = 7
DISCOVERY_WEIGHT = 1.0

# --- Sell instructions (rerating / value; see _sell_instructions_for_entry) ---
# TARGET_PERCENTAGE: partial capture of p/fv gap to fair value (not full FV in one shot).
# STOP_PERCENTAGE: hard max loss vs average cost.
# PEAKED: exit after rally if price gives back from session peak (rerating failed to hold).
# DESCENDING_TREND: short-term trend breakdown → analyse_drop path.
# AFTER_DAYS: backstop if rerating never materializes.
VUNDER_STOP_MULT = Decimal("0.87")
VUNDER_TARGET_FV_CAPTURE = 0.70
VUNDER_TARGET_CAP = Decimal("1.50")
VUNDER_TARGET_FLOOR = Decimal("1.10")
VUNDER_PEAKED_GIVEBACK_PCT = Decimal("12.0")
VUNDER_PEAKED_MIN_GAIN_PCT = Decimal("6.0")
VUNDER_DESCENDING_TREND = Decimal("-0.20")
VUNDER_MAX_HOLD_DAYS = 120

RATIO_TRUST_MAX_V2 = 3.0
NON_EQUITY_QUOTE_TYPES = frozenset({"ETF", "MUTUALFUND", "TRUST"})
QUALITY_COMPONENTS = ("financial", "valuation", "consensus", "sector")

INVESTIGATION_PROMPT = """You are an equity research analyst assisting a quantitative investment system.

A stock on our undervalued + financially healthy watchlist has begun showing improving market behavior
(rising price with elevated volume). Your task is NOT to recommend buying or selling.

Investigate whether there are signs of improving fundamentals, sentiment, institutional interest, or
catalysts that may explain the improving price trend in the last 30-90 days.

Focus on: earnings/guidance, analyst revisions, insider activity, news/narratives, competitive shifts,
management tone, business momentum. Be skeptical; flag value-trap and hype risks.

Separate facts from interpretation:
- "evidence": dated facts with "fact", "source", optional "date"
- "interpretations": claims with "claim", "confidence" (low|moderate|high), "evidence_indices" (ints)

Also answer: what might a sophisticated institutional investor find more or less attractive today vs ~6 months ago?

Do not make price predictions or give financial advice.

CONTEXT:
{context_block}

Respond with ONLY a valid JSON object:
{{
  "executive_summary": ["..."],
  "evidence": [{{"fact": "...", "source": "...", "date": "YYYY-MM-DD or null"}}],
  "interpretations": [{{"claim": "...", "confidence": "moderate", "evidence_indices": [0]}}],
  "positive_developments": ["..."],
  "risks": ["..."],
  "narrative_change": "...",
  "catalysts": [{{"name": "...", "strength": "strong|moderate|weak|speculative"}}],
  "institutional_signals": "...",
  "scores": {{
    "bullish_evidence": 1-10,
    "risk": 1-10,
    "narrative_shift_confidence": 1-10
  }},
  "final_assessment": "...",
  "institutional_vs_6mo_ago": "..."
}}
"""


def _is_non_equity(info: Dict[str, Any]) -> Optional[str]:
    quote_type = (info.get("quoteType") or "").strip().upper()
    if quote_type in NON_EQUITY_QUOTE_TYPES:
        return quote_type
    return None


def _compute_roe_fair_value_v2(
    symbol: str,
    info: Optional[Dict[str, Any]] = None,
    *,
    required_return: float = 0.10,
    max_growth: float = 0.20,
    max_roe: float = 0.30,
) -> Dict[str, Any]:
    """v2 ROE fair value (trust band 3.0); mirrors test_vunder.compute_roe_fair_value_v2."""
    sym = (symbol or "").strip().upper()
    if info is None:
        try:
            info = yf.Ticker(sym).info or {}
        except Exception as exc:
            return {
                "symbol": sym,
                "ratio": 1.0,
                "fair_value": None,
                "price": None,
                "neutral_fallback": True,
                "reason": f"yfinance fetch failed: {exc}",
                "detail_note": "",
            }

    eps = info.get("trailingEps")
    roe = info.get("returnOnEquity")
    payout = info.get("payoutRatio")
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    detail_note = ""

    if eps is None or eps <= 0:
        forward_eps = info.get("forwardEps")
        if forward_eps is not None and forward_eps > 0:
            eps = float(forward_eps)
            detail_note = "eps from forwardEps"

    if eps is not None and eps > 0 and (roe is None or roe <= 0):
        roa = info.get("returnOnAssets")
        if roa is not None and roa > 0:
            roe = min(float(roa) * 2.0, max_roe)
            detail_note = "roe proxy from roa"
        else:
            try:
                operating_margin = float(info.get("operatingMargins") or 0.0)
            except (TypeError, ValueError):
                operating_margin = 0.0
            if operating_margin > 0:
                roe = 0.08
                detail_note = "roe baseline 8%"

    def _neutral(reason: str) -> Dict[str, Any]:
        return {
            "symbol": sym,
            "ratio": 1.0,
            "fair_value": None,
            "price": float(price) if price is not None else None,
            "neutral_fallback": True,
            "reason": reason,
            "detail_note": detail_note,
        }

    if eps is None or eps <= 0 or roe is None or roe <= 0:
        return _neutral("missing or non-positive EPS/ROE")
    if price is None or price <= 0:
        return _neutral("missing or non-positive price")
    if required_return <= 0:
        return _neutral("invalid required_return")

    raw_de = info.get("debtToEquity")
    debt_to_equity: Optional[float] = None
    try:
        if raw_de is not None:
            de_candidate = float(raw_de) / 100.0
            if math.isfinite(de_candidate) and de_candidate >= 0:
                debt_to_equity = de_candidate
    except (TypeError, ValueError):
        debt_to_equity = None

    adjusted_roe = min(float(roe), max_roe)
    if debt_to_equity is not None and debt_to_equity > 1.0:
        leverage_penalty = min(debt_to_equity, 2.5)
        adjusted_roe = adjusted_roe / math.sqrt(leverage_penalty)

    payout_adj = 0.0 if payout is None or payout < 0 else min(max(float(payout), 0.0), 0.9)
    g = adjusted_roe * (1.0 - payout_adj)

    if payout_adj > 0:
        spread_floor = 0.03
        if required_return <= spread_floor:
            return _neutral("required_return too low for dividend path")
        g = min(g, max_growth, required_return - spread_floor)
        g = max(g, 0.0)
        denominator = required_return - g
        if denominator <= 0:
            return _neutral("invalid denominator after growth cap")
        justified_pe = (payout_adj * (1.0 + g)) / denominator
    else:
        justified_pe = adjusted_roe / required_return
        justified_pe = min(
            justified_pe,
            max_growth / required_return if required_return > 0 else justified_pe,
        )

    justified_pe = max(justified_pe, 5.0)
    fair_value = float(eps) * justified_pe
    if fair_value <= 0:
        return _neutral("invalid fair value calculation")

    ratio = float(price) / fair_value
    trust_min = 1.0 / RATIO_TRUST_MAX_V2
    if ratio > RATIO_TRUST_MAX_V2 or ratio < trust_min:
        return {
            "symbol": sym,
            "ratio": 1.0,
            "fair_value": fair_value,
            "price": float(price),
            "neutral_fallback": True,
            "reason": f"extreme ratio {ratio:.2f} outside trust band",
            "detail_note": detail_note,
        }

    return {
        "symbol": sym,
        "ratio": ratio,
        "fair_value": fair_value,
        "price": float(price),
        "neutral_fallback": False,
        "reason": "",
        "detail_note": detail_note,
    }


def _valuation_snapshot(symbol: str) -> Dict[str, Any]:
    sym = symbol.strip().upper()
    info = financial_yahoo.get_ticker_info(sym)
    excluded = _is_non_equity(info)
    if excluded:
        return {"skip": True, "quote_type": excluded}

    result = _compute_roe_fair_value_v2(sym, info=info)
    raw_ratio: Optional[float] = None
    if result.get("fair_value") and result["fair_value"] > 0 and result.get("price"):
        raw_ratio = float(result["price"]) / float(result["fair_value"])
    elif not result.get("neutral_fallback"):
        raw_ratio = float(result["ratio"])

    return {
        "skip": False,
        "valuation_ratio": float(result["ratio"]),
        "raw_ratio": raw_ratio,
        "fair_value": result.get("fair_value"),
        "untrusted": bool(result.get("neutral_fallback")),
        "valuation_note": (result.get("reason") or result.get("detail_note") or "").strip(),
    }


class Vunder(AdvisorBase):
    """Weekly value universe; intraday monitor + investigation LLM during market hours."""

    PROCESS_CUTOFF_HOUR_UTC = 9

    # Fallback if discovered() is called without per-entry instructions.
    sell_instructions = [
        ("TARGET_PERCENTAGE", Decimal("1.25"), None),
        ("STOP_PERCENTAGE", VUNDER_STOP_MULT, None),
        ("PEAKED", VUNDER_PEAKED_GIVEBACK_PCT, VUNDER_PEAKED_MIN_GAIN_PCT),
        ("DESCENDING_TREND", VUNDER_DESCENDING_TREND, None),
        ("AFTER_DAYS", VUNDER_MAX_HOLD_DAYS, None),
    ]

    def discover(self, sa) -> None:
        target_date = financial_polygon.get_last_trading_day()
        if not target_date:
            logger.info("Vunder: no valid trading date")
            return

        state = self._advisor_blob_state()

        if self._needs_universe_rebuild(state, target_date):
            watchlist = self._build_watchlist()
            state["watchlist"] = watchlist
            state["universe_built"] = target_date
            self._save_advisor_blob_state(state)
            logger.info(
                "Vunder sa=%s: watchlist built (%d symbols) on %s",
                sa.id,
                len(watchlist),
                target_date,
            )

        market_status = self.market_open()
        if market_status is None:
            logger.info("Vunder sa=%s: market closed; universe ok, skip monitor/LLM", sa.id)
            return
        if market_status < 0:
            logger.info(
                "Vunder sa=%s: market not open yet (%s min); skip monitor/LLM",
                sa.id,
                -market_status,
            )
            return

        watchlist = state.get("watchlist") or {}
        if not watchlist:
            logger.info("Vunder sa=%s: empty watchlist; skip monitor", sa.id)
            return

        discoveries = 0
        investigated = 0

        for symbol, entry in sorted(watchlist.items()):
            sym = symbol.strip().upper()
            if self._on_llm_cooldown(state, sym, target_date):
                continue

            trigger = self._monitor_trigger(sym, entry)
            if not trigger:
                continue

            investigation = self._investigate(sym, entry, trigger)
            if not investigation:
                continue

            investigated += 1
            state.setdefault("last_investigation", {})[sym] = investigation
            state.setdefault("llm_cooldown", {})[sym] = target_date

            if not self._should_discover(investigation, entry):
                continue
            if not self.allow_discovery(sym, period=DISCOVERY_COOLDOWN_HOURS):
                continue

            explanation = self._discovery_explanation(sym, entry, trigger, investigation)
            sell_instructions = self._sell_instructions_for_entry(entry)
            if self.discovered(
                sa,
                sym,
                explanation,
                sell_instructions=sell_instructions,
                weight=DISCOVERY_WEIGHT,
            ):
                discoveries += 1

        self._save_advisor_blob_state(state)
        logger.info(
            "Vunder sa=%s: watchlist=%d investigated=%d discoveries=%d (session)",
            sa.id,
            len(watchlist),
            investigated,
            discoveries,
        )

    def _sell_instructions_for_entry(
        self, entry: Dict[str, Any]
    ) -> List[Tuple[Any, Any, Any]]:
        """
        Exits for undervalued rerating plays.

        Target scales with watchlist p/fv: cheaper vs fair value → higher TP multiplier
        (capped). Stop is a fixed % below cost. PEAKED / DESCENDING_TREND protect gains
        when momentum fades; AFTER_DAYS limits dead money.
        """
        try:
            p_fv = float(entry.get("p_fv") or 1.0)
        except (TypeError, ValueError):
            p_fv = 1.0
        p_fv = max(0.35, min(p_fv, MAX_P_FV_RECHECK))

        raw_target = (1.0 / p_fv) * VUNDER_TARGET_FV_CAPTURE
        target_mult = min(float(VUNDER_TARGET_CAP), max(float(VUNDER_TARGET_FLOOR), raw_target))
        target_mult = round(target_mult, 4)

        logger.debug(
            "Vunder sell instructions p/fv=%.3f -> TARGET_PERCENTAGE=%.4f",
            p_fv,
            target_mult,
        )

        return [
            ("TARGET_PERCENTAGE", Decimal(str(target_mult)), None),
            ("STOP_PERCENTAGE", VUNDER_STOP_MULT, None),
            ("PEAKED", VUNDER_PEAKED_GIVEBACK_PCT, VUNDER_PEAKED_MIN_GAIN_PCT),
            ("DESCENDING_TREND", VUNDER_DESCENDING_TREND, None),
            ("AFTER_DAYS", VUNDER_MAX_HOLD_DAYS, None),
        ]

    def _needs_universe_rebuild(self, state: Dict[str, Any], target_date: str) -> bool:
        watchlist = state.get("watchlist")
        if not watchlist:
            return True
        built = (state.get("universe_built") or "").strip()
        if not built:
            return True
        try:
            built_dt = datetime.strptime(built, "%Y-%m-%d").date()
            ref_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            return True
        return (ref_dt - built_dt).days >= UNIVERSE_REBUILD_DAYS

    def _build_watchlist(self) -> Dict[str, Dict[str, Any]]:
        df = financial_polygon.get_filtered_stocks(
            min_price=MIN_PRICE,
            max_price=MAX_PRICE,
            min_volume=MIN_SESSION_VOLUME,
        )
        if df is None or df.empty:
            logger.warning("Vunder: Polygon universe empty")
            return {}

        df = df.sort_values("today_volume", ascending=False).head(MAX_EVAL_SYMBOLS)
        candidates: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            sym = str(row["ticker"]).strip().upper()
            val = _valuation_snapshot(sym)
            if val.get("skip"):
                continue
            if val.get("untrusted"):
                continue
            ratio = val.get("raw_ratio") or val.get("valuation_ratio")
            if ratio is None or float(ratio) > MAX_P_FV:
                continue

            scores = run_component_scores(sym, components=QUALITY_COMPONENTS)
            composite = composite_from_scores(scores)
            if composite is None:
                continue
            q_score = float(composite)
            if q_score < MIN_QUALITY_SCORE:
                continue

            rating = score_to_rating(q_score)
            candidates.append(
                {
                    "ticker": sym,
                    "polygon_price": float(row["price"]),
                    "today_volume": int(row["today_volume"]),
                    "p_fv": round(float(ratio), 3),
                    "fair_value": val.get("fair_value"),
                    "quality_score": round(q_score, 1),
                    "quality_grade": rating.letter if rating else None,
                    "valuation_note": val.get("valuation_note") or "",
                }
            )

        candidates.sort(key=lambda c: (-c["quality_score"], c["p_fv"]))
        selected = candidates[:MAX_WATCHLIST]

        watchlist: Dict[str, Dict[str, Any]] = {}
        built_on = financial_polygon.get_last_trading_day() or datetime.now().strftime("%Y-%m-%d")
        for c in selected:
            sym = c["ticker"]
            watchlist[sym] = {
                "built": built_on,
                "p_fv": c["p_fv"],
                "fair_value": c["fair_value"],
                "quality_score": c["quality_score"],
                "quality_grade": c["quality_grade"],
                "polygon_price": c["polygon_price"],
                "valuation_note": c["valuation_note"][:120],
            }

        return watchlist

    def _monitor_trigger(
        self, symbol: str, entry: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Rising price vs prior close + volume above recent average.
        Returns trigger metrics dict or None.
        """
        info = financial_yahoo.get_ticker_info(symbol)
        current = info.get("regularMarketPrice") or info.get("currentPrice")
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

        stock = self.get_stock(symbol)
        if stock is not None:
            stock.refresh()
            if stock.price is not None:
                current = float(stock.price)

        hist = financial_yahoo.get_6m_history(symbol)
        if hist is None or hist.empty or len(hist) < VOL_LOOKBACK_DAYS + 1:
            return None

        if prev_close is None:
            prev_close = float(hist["close"].iloc[-1])
        if current is None:
            current = float(hist["close"].iloc[-1])

        try:
            current_f = float(current)
            prev_f = float(prev_close)
        except (TypeError, ValueError):
            return None

        if current_f <= prev_f:
            return None

        avg_vol = float(hist["volume"].tail(VOL_LOOKBACK_DAYS).mean())
        if avg_vol <= 0:
            return None

        today_vol = info.get("regularMarketVolume") or info.get("volume")
        if today_vol is None and len(hist) >= 1:
            today_vol = float(hist["volume"].iloc[-1])
        try:
            today_vol_f = float(today_vol) if today_vol is not None else 0.0
        except (TypeError, ValueError):
            today_vol_f = 0.0

        vol_ratio = today_vol_f / avg_vol if avg_vol > 0 else 0.0
        if vol_ratio < VOL_RATIO_MIN:
            return None

        val = _valuation_snapshot(symbol)
        if val.get("skip") or val.get("untrusted"):
            return None
        recheck_ratio = val.get("raw_ratio") or val.get("valuation_ratio")
        if recheck_ratio is not None and float(recheck_ratio) > MAX_P_FV_RECHECK:
            return None

        pct_vs_prior = (current_f / prev_f - 1.0) * 100.0
        closes = hist["close"].astype(float)
        ret_5d = None
        if len(closes) >= 6:
            ref = float(closes.iloc[-6])
            if ref > 0:
                ret_5d = (current_f / ref - 1.0) * 100.0

        return {
            "current_price": round(current_f, 2),
            "prev_close": round(prev_f, 2),
            "pct_vs_prior": round(pct_vs_prior, 2),
            "ret_5d_pct": round(ret_5d, 2) if ret_5d is not None else None,
            "volume_ratio": round(vol_ratio, 2),
            "today_volume": int(today_vol_f),
            "avg_volume_20d": int(avg_vol),
            "reason": "price_up_and_volume_up",
        }

    def _build_context_block(
        self,
        symbol: str,
        entry: Dict[str, Any],
        trigger: Dict[str, Any],
    ) -> str:
        info = financial_yahoo.get_ticker_info(symbol)
        company = (info.get("longName") or info.get("shortName") or symbol).strip()
        consensus = self.stock_consensus(symbol)
        as_of = datetime.now().strftime("%Y-%m-%d %H:%M ET")

        lines = [
            f"Ticker: {symbol}",
            f"Company: {company}",
            f"As-of: {as_of}",
            "",
            "Watchlist (weekly):",
            f"  built: {entry.get('built', 'n/a')}",
            f"  p/fv: {entry.get('p_fv')}",
            (
                f"  fair_value: ${float(entry['fair_value']):.2f}"
                if entry.get("fair_value") is not None
                else "  fair_value: n/a"
            ),
            f"  quality_score: {entry.get('quality_score')} ({entry.get('quality_grade')})",
            f"  valuation_note: {entry.get('valuation_note') or 'n/a'}",
            "",
            "Trigger (session — why LLM ran):",
            f"  reason: {trigger.get('reason')}",
            f"  price: ${trigger.get('current_price')} vs prior close ${trigger.get('prev_close')} ({trigger.get('pct_vs_prior'):+.2f}%)",
            f"  5d return vs price: {trigger.get('ret_5d_pct')}%"
            if trigger.get("ret_5d_pct") is not None
            else "  5d return: n/a",
            f"  volume_ratio vs {VOL_LOOKBACK_DAYS}d avg: {trigger.get('volume_ratio')}x",
            f"  today_volume: {trigger.get('today_volume'):,} | avg: {trigger.get('avg_volume_20d'):,}",
            "",
            "Pre-fetched consensus (yfinance):",
            f"  recommendation_mean: {consensus.get('recommendation_mean')}",
            f"  recommendation_key: {consensus.get('recommendation_key')}",
            f"  target_mean: {consensus.get('target_mean')}",
            f"  upside_to_mean_pct: {consensus.get('upside_to_mean_pct')}",
            f"  earningsGrowth: {info.get('earningsGrowth')}",
        ]
        return "\n".join(lines)

    def _investigate(
        self,
        symbol: str,
        entry: Dict[str, Any],
        trigger: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        context = self._build_context_block(symbol, entry, trigger)
        prompt = INVESTIGATION_PROMPT.format(context_block=context)
        model, results = self.ask_llm(prompt, use_search=True, timeout=180.0)
        if not results or not isinstance(results, dict):
            logger.info("Vunder: investigation failed for %s (Gemini+DeepSeek)", symbol)
            return None
        results["_model"] = model
        results["_symbol"] = symbol
        results["_trigger"] = trigger
        return results

    def _on_llm_cooldown(
        self, state: Dict[str, Any], symbol: str, target_date: str
    ) -> bool:
        raw = (state.get("llm_cooldown") or {}).get(symbol.upper())
        if not raw:
            return False
        try:
            last = datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
            ref = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            return False
        return (ref - last).days < LLM_COOLDOWN_DAYS

    def _should_discover(
        self, investigation: Dict[str, Any], entry: Dict[str, Any]
    ) -> bool:
        scores = investigation.get("scores") or {}
        try:
            shift = int(scores.get("narrative_shift_confidence", 0))
            risk = int(scores.get("risk", 10))
        except (TypeError, ValueError):
            return False
        p_fv = entry.get("p_fv")
        try:
            p_fv_f = float(p_fv) if p_fv is not None else 99.0
        except (TypeError, ValueError):
            p_fv_f = 99.0
        return (
            shift >= MIN_NARRATIVE_SHIFT
            and risk <= MAX_RISK_SCORE
            and p_fv_f <= MAX_P_FV_RECHECK
        )

    def _discovery_explanation(
        self,
        symbol: str,
        entry: Dict[str, Any],
        trigger: Dict[str, Any],
        investigation: Dict[str, Any],
    ) -> str:
        scores = investigation.get("scores") or {}
        shift = scores.get("narrative_shift_confidence", "?")
        risk = scores.get("risk", "?")
        summary = investigation.get("executive_summary") or []
        lead = ""
        if summary and isinstance(summary, list):
            lead = str(summary[0])[:120]
        elif investigation.get("final_assessment"):
            lead = str(investigation["final_assessment"])[:120]

        sells = self._sell_instructions_for_entry(entry)
        tp_mult = sells[0][1] if sells else "?"
        text = (
            f"Vunder investigate | {symbol} shift {shift}/10 risk {risk}/10 | "
            f"p/fv {entry.get('p_fv')} qual {entry.get('quality_score')} | "
            f"TP {tp_mult}x stop {VUNDER_STOP_MULT} | "
            f"vol {trigger.get('volume_ratio')}x +{trigger.get('pct_vs_prior')}% | "
            f"{lead}"
        )
        return text[:500]


register(name="Vunder", python_class="Vunder")
