"""
Meyka belief-tension scoring — shared by test_meyka.py and advisor late gates.

Does NOT emit buy/sell. Estimates market belief tension and maps it to a
discovery gate for short/medium-term (+4%-style) candidates.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Discovery gate defaults (env can override)
DEFAULT_IMPACT_MIN = 35.0
DEFAULT_CLARITY_MIN = 7.0
DEFAULT_REQUIRE_BIAS = "bullish_unpriced"
DEFAULT_GAP_UP_SOFT_PCT = 3.0
DEFAULT_GAP_UP_IMPACT_MIN = 40.0
DEFAULT_DISCOVER_MIN_MINUTES_AFTER_OPEN = 30
# Skip live discovery if price already ran this far vs prior close (move spent).
DEFAULT_MAX_PRIOR_CLOSE_RUN_PCT = 5.0
# Premarket score-pass → Watchlist until discover window (≥10:00 ET).
MEYKA_DEFER_KIND = "meyka_defer"
MEYKA_DEFER_WATCH_DAYS = 2

# Meyka-only sell pack: harvest pop, hard stop, EOD bank — no averaging down.
# PEAKED: value1=giveback %, value2=min peak %
# STOP_PERCENTAGE / END_DAY: multipliers vs avg; END_DAY value2 = minutes before close.
MEYKA_PEAKED_GIVEBACK_PCT = 12.0
MEYKA_PEAKED_MIN_PEAK_PCT = 4.0
MEYKA_STOP_MULT = 0.95
MEYKA_ENDDAY_TAKE = 1.00
MEYKA_ENDDAY_MINUTES_BEFORE_CLOSE = 120.0
MEYKA_ENDDAY_CLUTTER_TAKE = 0.99
MEYKA_ENDDAY_CLUTTER_MINUTES_BEFORE_CLOSE = 30.0
MEYKA_DESCENDING_TREND = -0.20

MEYKA_PROMPT = """You are Meyka, a consensus-vs-contradiction analyst for financial news.

Your job is NOT to predict price or recommend buy/sell.
Your job is to estimate MARKET BELIEF TENSION: what investors likely believe,
what in the text challenges that belief, and how strongly that matters.

TICKER: {ticker}
COMPANY: {company_name}

MARKET CONTEXT (use as belief anchors — do not invent figures beyond this):
{context_block}

ARTICLE / EVENT TEXT:
{text}

Step 1: What is the dominant market consensus belief about this company right now?
        (narrative + street positioning implied by context and text)
Step 2: What evidence in the text SUPPORTS that belief?
Step 3: What evidence CONTRADICTS or weakens that belief?
Step 4: If consensus is wrong, what is the alternative interpretation?
Step 5: If consensus is wrong, how CLEAR is the repricing path? (earnings revisions,
        product adoption data, positioning squeeze, etc. vs vague macro narrative)

SCORING (integers unless noted):
- consensus_strength: 0-10 how dominant/coherent the prevailing belief is
- contradiction_strength: 0-10 how strongly the text challenges that belief
- evidence_quality: 0-10 credibility/specificity of the contradicting evidence
- repricing_clarity: 0-10 how clear the mechanism is if belief shifts
- consensus_error_probability: 0-100 likelihood consensus is materially wrong
- repricing_potential: 0-100 magnitude of re-rating if belief shifts
- confidence: 0.0-1.0 your confidence in this assessment

directional_bias: one of bullish_unpriced | bearish_unpriced | neutral | unclear
  (which way mispricing likely runs IF consensus is wrong — not a trade signal)

Return ONLY valid JSON:
{{
  "consensus_view": "...",
  "alternative_view": "...",
  "supporting_evidence": ["..."],
  "contradicting_evidence": ["..."],
  "consensus_strength": 0,
  "contradiction_strength": 0,
  "evidence_quality": 0,
  "repricing_clarity": 0,
  "consensus_error_probability": 0,
  "repricing_potential": 0,
  "directional_bias": "bullish_unpriced|bearish_unpriced|neutral|unclear",
  "confidence": 0.0
}}
"""


def _env_truthy(name: str, default: str = "0") -> bool:
    raw = (os.getenv(name) or default).strip().lower()
    return raw in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def meyka_bizfeed_gate_enabled() -> bool:
    """When on, Bizfeed runs Meyka before creating a Discovery."""
    return _env_truthy("MEYKA_BIZFEED_GATE", "1")


def meyka_shadow_only() -> bool:
    """When on, score+log only; never create Discovery. Default off (live)."""
    return _env_truthy("MEYKA_SHADOW_ONLY", "0")


def meyka_impact_min() -> float:
    return _env_float("MEYKA_IMPACT_MIN", DEFAULT_IMPACT_MIN)


def meyka_clarity_min() -> float:
    return _env_float("MEYKA_CLARITY_MIN", DEFAULT_CLARITY_MIN)


def meyka_require_bias() -> str:
    return (os.getenv("MEYKA_REQUIRE_BIAS") or DEFAULT_REQUIRE_BIAS).strip().lower()


def meyka_discover_min_minutes_after_open() -> int:
    return _env_int(
        "MEYKA_DISCOVER_MIN_MINUTES_AFTER_OPEN",
        DEFAULT_DISCOVER_MIN_MINUTES_AFTER_OPEN,
    )


def meyka_gap_up_soft_pct() -> float:
    return _env_float("MEYKA_GAP_UP_SOFT_PCT", DEFAULT_GAP_UP_SOFT_PCT)


def meyka_gap_up_impact_min() -> float:
    return _env_float("MEYKA_GAP_UP_IMPACT_MIN", DEFAULT_GAP_UP_IMPACT_MIN)


def meyka_max_prior_close_run_pct() -> float:
    """Skip discovery when (price/prior_close - 1)*100 already exceeds this."""
    return _env_float("MEYKA_MAX_PRIOR_CLOSE_RUN_PCT", DEFAULT_MAX_PRIOR_CLOSE_RUN_PCT)


def meyka_sell_instructions() -> list:
    """
    Short-horizon SI pack for Meyka-gated Bizfeed discoveries.

    PEAKED harvest + hard stop + dual END_DAY (Pulse-style) + DT.
    No PERCENTAGE_REBUY — averaging down fights the spike/fade edge.
    """
    return [
        ("PEAKED", MEYKA_PEAKED_GIVEBACK_PCT, MEYKA_PEAKED_MIN_PEAK_PCT),
        ("STOP_PERCENTAGE", MEYKA_STOP_MULT, None),
        ("END_DAY", MEYKA_ENDDAY_TAKE, MEYKA_ENDDAY_MINUTES_BEFORE_CLOSE),
        ("END_DAY", MEYKA_ENDDAY_CLUTTER_TAKE, MEYKA_ENDDAY_CLUTTER_MINUTES_BEFORE_CLOSE),
        ("DESCENDING_TREND", MEYKA_DESCENDING_TREND, None),
    ]


def prior_close_run_pct(market_context: Dict[str, Any]) -> Optional[float]:
    """
    Percent move of current price vs prior close.
    Prefers vs_open + gap when both present; else current/prior_close.
    """
    ctx = market_context or {}
    current = _safe_float(ctx.get("current_price"))
    prior = _safe_float(ctx.get("prior_close"))
    if current is not None and prior is not None and prior > 0:
        return (current / prior - 1.0) * 100.0
    gap = _safe_float(ctx.get("gap_open_pct"))
    vs_open = _safe_float(ctx.get("vs_open_pct"))
    if gap is not None and vs_open is not None:
        # Approximate: open already gapped, then vs open.
        return gap + vs_open
    return gap


def _clamp(value: Any, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _price_position_label(
    current: Optional[float],
    low_52w: Optional[float],
    high_52w: Optional[float],
) -> str:
    if not current or not low_52w or not high_52w or high_52w <= low_52w:
        return "unknown"
    pct = (current - low_52w) / (high_52w - low_52w) * 100.0
    if pct <= 20:
        return f"near 52w low ({pct:.0f}% of range)"
    if pct >= 80:
        return f"near 52w high ({pct:.0f}% of range)"
    return f"mid-range ({pct:.0f}% of 52w range)"


def fetch_market_context(ticker: str) -> Dict[str, Any]:
    """Yahoo consensus + tape snapshot including open vs prior close gap."""
    from core.services.financial.yahoo import get_consensus_snapshot, get_ticker_info

    symbol = ticker.strip().upper()
    snap = get_consensus_snapshot(symbol)
    info = get_ticker_info(symbol)
    company_name = (info.get("shortName") or info.get("longName") or "").strip() or symbol

    current = snap.get("current_price")
    low_52w = _safe_float(info.get("fiftyTwoWeekLow"))
    high_52w = _safe_float(info.get("fiftyTwoWeekHigh"))
    open_px = _safe_float(info.get("regularMarketOpen") or info.get("open"))
    prior_close = _safe_float(
        info.get("regularMarketPreviousClose") or info.get("previousClose")
    )

    gap_open_pct: Optional[float] = None
    if open_px is not None and prior_close is not None and prior_close > 0:
        gap_open_pct = (open_px / prior_close - 1.0) * 100.0

    vs_open_pct: Optional[float] = None
    if current is not None and open_px is not None and open_px > 0:
        vs_open_pct = (float(current) / open_px - 1.0) * 100.0

    return {
        "symbol": symbol,
        "company_name": company_name,
        "current_price": current,
        "price_position": _price_position_label(current, low_52w, high_52w),
        "analyst_recommendation": snap.get("recommendation_key"),
        "analyst_count": snap.get("analyst_count"),
        "target_mean": snap.get("target_mean"),
        "upside_to_mean_pct": snap.get("upside_to_mean_pct"),
        "sector": (info.get("sector") or "").strip() or None,
        "industry": (info.get("industry") or "").strip() or None,
        "session_open": open_px,
        "prior_close": prior_close,
        "gap_open_pct": round(gap_open_pct, 3) if gap_open_pct is not None else None,
        "vs_open_pct": round(vs_open_pct, 3) if vs_open_pct is not None else None,
    }


def _format_context_block(ctx: Dict[str, Any]) -> str:
    gap = ctx.get("gap_open_pct")
    vs_open = ctx.get("vs_open_pct")
    gap_s = f"{gap:+.2f}%" if isinstance(gap, (int, float)) else "n/a"
    vs_s = f"{vs_open:+.2f}%" if isinstance(vs_open, (int, float)) else "n/a"
    lines = [
        f"- Price: {ctx.get('current_price')} ({ctx.get('price_position')})",
        f"- Session open: {ctx.get('session_open')} | prior close: {ctx.get('prior_close')} "
        f"| open gap: {gap_s} | vs open: {vs_s}",
        f"- Analyst recommendation: {ctx.get('analyst_recommendation') or 'unknown'} "
        f"({ctx.get('analyst_count') or '?'} analysts)",
        f"- Target mean: {ctx.get('target_mean')} "
        f"(upside to mean: {ctx.get('upside_to_mean_pct')})",
        f"- Sector / industry: {ctx.get('sector') or '?'} / {ctx.get('industry') or '?'}",
        "- Note: large positive open gaps often mean near-term upside is partly spent.",
    ]
    return "\n".join(lines)


def build_prompt(ticker: str, text: str, ctx: Dict[str, Any]) -> str:
    return MEYKA_PROMPT.format(
        ticker=ctx.get("symbol") or ticker.upper(),
        company_name=ctx.get("company_name") or ticker.upper(),
        context_block=_format_context_block(ctx),
        text=text.strip(),
    )


def compute_derived_scores(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Map LLM scores to Meyka impact and conviction multiplier."""
    contradiction = _clamp(parsed.get("contradiction_strength"), 0, 10)
    evidence = _clamp(parsed.get("evidence_quality"), 0, 10)
    clarity = _clamp(parsed.get("repricing_clarity"), 0, 10)
    confidence = _clamp(parsed.get("confidence"), 0, 1)

    impact = (contradiction / 10) * (evidence / 10) * (clarity / 10) * 100 * confidence
    impact = round(impact, 1)

    if impact < 30:
        position_adjustment = "ignore"
    elif impact < 60:
        position_adjustment = "monitor"
    elif impact < 80:
        position_adjustment = "increase"
    else:
        position_adjustment = "strong_increase"

    conviction_multiplier = round(1.0 + (impact / 100) * 0.8, 2)

    return {
        "meyka_impact": impact,
        "conviction_multiplier": conviction_multiplier,
        "position_adjustment": position_adjustment,
        "final_meyka_score": round(
            _clamp(parsed.get("consensus_error_probability"), 0, 100) * 0.5
            + _clamp(parsed.get("repricing_potential"), 0, 100) * 0.3
            + impact * 0.2,
            1,
        ),
    }


def trading_hours_ok(*, market_status: Optional[int] = None) -> Tuple[bool, str]:
    """
    True when regular session is open and past the post-open quiet window.
    market_status: minutes since 9:30 ET from market_open(); None = closed.
    """
    from core.services.market import market_open

    status = market_open() if market_status is None else market_status
    min_after = meyka_discover_min_minutes_after_open()
    if status is None:
        return False, "market closed"
    if status < 0:
        return False, f"premarket ({-status} min to open)"
    if status < min_after:
        return False, f"before discover window ({min_after - status} min until gate)"
    return True, f"ok ({status} min after open)"


def article_text_from_parts(
    *,
    title: str = "",
    summary: str = "",
    body: str = "",
    max_chars: int = 12000,
) -> str:
    """Build Meyka input text from RSS / fetched article parts."""
    chunks: List[str] = []
    t = (title or "").strip()
    s = (summary or "").strip()
    b = (body or "").strip()
    if t:
        chunks.append(t)
    if s and s != t:
        chunks.append(s)
    if b:
        chunks.append(b[: max(0, max_chars)])
    return "\n\n".join(chunks).strip()


@dataclass
class MeykaGateDecision:
    """Result of score + discovery gate for advisor wiring."""

    ticker: str
    model: Optional[str] = None
    scored: bool = False
    allow_discovery: bool = False
    reason: str = ""
    shadow_only: bool = True
    trading_hours_ok: bool = False
    trading_hours_detail: str = ""
    score_gate_ok: bool = False
    score_gate_detail: str = ""
    # Scores passed but session gate not open yet — watchlist until ≥10:00 ET.
    defer_for_open: bool = False
    market_context: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def meta_payload(self) -> Dict[str, Any]:
        """Compact dict for Discovery.meta['meyka']."""
        s = self.scores
        return {
            "model": self.model,
            "allow_discovery": self.allow_discovery,
            "defer_for_open": self.defer_for_open,
            "reason": self.reason,
            "shadow_only": self.shadow_only,
            "trading_hours_ok": self.trading_hours_ok,
            "trading_hours_detail": self.trading_hours_detail,
            "score_gate_ok": self.score_gate_ok,
            "meyka_impact": s.get("meyka_impact"),
            "conviction_multiplier": s.get("conviction_multiplier"),
            "position_adjustment": s.get("position_adjustment"),
            "final_meyka_score": s.get("final_meyka_score"),
            "directional_bias": s.get("directional_bias"),
            "repricing_clarity": s.get("repricing_clarity"),
            "consensus_error_probability": s.get("consensus_error_probability"),
            "repricing_potential": s.get("repricing_potential"),
            "gap_open_pct": (self.market_context or {}).get("gap_open_pct"),
            "vs_open_pct": (self.market_context or {}).get("vs_open_pct"),
            "session_open": (self.market_context or {}).get("session_open"),
            "prior_close": (self.market_context or {}).get("prior_close"),
        }


def passes_discovery_score_gate(
    output: Dict[str, Any],
    *,
    gap_open_pct: Optional[float] = None,
) -> Tuple[bool, str]:
    """Hard score thresholds for creating a Discovery."""
    bias = str(output.get("directional_bias") or "").strip().lower()
    require = meyka_require_bias()
    if bias != require:
        return False, f"bias={bias or 'missing'} (need {require})"

    impact = _clamp(output.get("meyka_impact"), 0, 100)
    clarity = _clamp(output.get("repricing_clarity"), 0, 10)
    impact_floor = meyka_impact_min()
    clarity_floor = meyka_clarity_min()

    if (
        gap_open_pct is not None
        and gap_open_pct >= meyka_gap_up_soft_pct()
    ):
        # Soft raise: move may already be spent at the open.
        impact_floor = max(impact_floor, meyka_gap_up_impact_min())

    if impact < impact_floor:
        return False, f"impact={impact:.1f} < {impact_floor:.1f}"
    if clarity < clarity_floor:
        return False, f"clarity={clarity:.1f} < {clarity_floor:.1f}"
    return True, f"pass impact={impact:.1f} clarity={clarity:.1f} bias={bias}"


def analyze_opportunity(
    text: str,
    ticker: str,
    *,
    timeout: float = 120.0,
    advisor_name: str = "meyka",
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Run Meyka LLM + derived scores. Returns (model, output_dict)."""
    from core.services.llm.router import ask_llm

    ctx = fetch_market_context(ticker)
    prompt = build_prompt(ticker, text, ctx)

    model, parsed, _, _ = ask_llm(
        prompt,
        advisor_name=advisor_name,
        timeout=timeout,
    )

    if not parsed:
        return model, {
            "error": "llm_failed",
            "ticker": ctx.get("symbol") or ticker.upper(),
            "market_context": ctx,
        }

    derived = compute_derived_scores(parsed)
    output = {
        "ticker": ctx.get("symbol") or ticker.upper(),
        "model": model,
        "market_context": ctx,
        **parsed,
        **derived,
    }
    return model, output


def evaluate_for_discovery(
    ticker: str,
    text: str,
    *,
    timeout: float = 120.0,
    advisor_name: str = "meyka",
    market_status: Optional[int] = None,
) -> MeykaGateDecision:
    """
    Score article and decide whether Discovery is allowed.

    Always attempts scoring when text is present (for shadow logs).
    Discovery requires: trading hours + score gate + not shadow_only.
    """
    symbol = (ticker or "").strip().upper()
    shadow = meyka_shadow_only()
    th_ok, th_detail = trading_hours_ok(market_status=market_status)

    decision = MeykaGateDecision(
        ticker=symbol,
        shadow_only=shadow,
        trading_hours_ok=th_ok,
        trading_hours_detail=th_detail,
    )

    if not (text or "").strip():
        decision.reason = "empty article text"
        return decision

    model, output = analyze_opportunity(
        text,
        symbol,
        timeout=timeout,
        advisor_name=advisor_name,
    )
    decision.model = model
    decision.raw = output
    decision.market_context = output.get("market_context") or {}

    if output.get("error"):
        decision.reason = str(output.get("error"))
        return decision

    decision.scored = True
    decision.scores = {
        "consensus_view": output.get("consensus_view"),
        "alternative_view": output.get("alternative_view"),
        "consensus_strength": output.get("consensus_strength"),
        "contradiction_strength": output.get("contradiction_strength"),
        "evidence_quality": output.get("evidence_quality"),
        "repricing_clarity": output.get("repricing_clarity"),
        "consensus_error_probability": output.get("consensus_error_probability"),
        "repricing_potential": output.get("repricing_potential"),
        "directional_bias": output.get("directional_bias"),
        "confidence": output.get("confidence"),
        "meyka_impact": output.get("meyka_impact"),
        "conviction_multiplier": output.get("conviction_multiplier"),
        "position_adjustment": output.get("position_adjustment"),
        "final_meyka_score": output.get("final_meyka_score"),
    }

    gap = decision.market_context.get("gap_open_pct")
    score_ok, score_detail = passes_discovery_score_gate(
        output,
        gap_open_pct=gap if isinstance(gap, (int, float)) else None,
    )
    decision.score_gate_ok = score_ok
    decision.score_gate_detail = score_detail

    if score_ok and not th_ok:
        decision.defer_for_open = True
        decision.reason = f"hours: {th_detail}; scores: {score_detail}"
        decision.allow_discovery = False
        return decision

    if not th_ok:
        decision.reason = f"hours: {th_detail}; scores: {score_detail}"
        decision.allow_discovery = False
        return decision

    if not score_ok:
        decision.reason = score_detail
        decision.allow_discovery = False
        return decision

    if shadow:
        decision.reason = f"shadow_only (would pass: {score_detail})"
        decision.allow_discovery = False
        return decision

    decision.reason = score_detail
    decision.allow_discovery = True
    return decision


SHADOW_CSV_FIELDS = [
    "timestamp",
    "ticker",
    "model",
    "source",
    "allow_discovery",
    "reason",
    "shadow_only",
    "trading_hours_ok",
    "trading_hours_detail",
    "gap_open_pct",
    "vs_open_pct",
    "consensus_view",
    "alternative_view",
    "consensus_strength",
    "contradiction_strength",
    "evidence_quality",
    "repricing_clarity",
    "consensus_error_probability",
    "repricing_potential",
    "directional_bias",
    "confidence",
    "meyka_impact",
    "conviction_multiplier",
    "position_adjustment",
    "final_meyka_score",
]


def default_shadow_log_path() -> Path:
    raw = (os.getenv("MEYKA_SHADOW_LOG") or "").strip()
    if raw:
        return Path(raw)
    # Separate from historical meyka_shadow.csv (manual case archive).
    return Path("meyka_bizfeed_shadow.csv")


def append_shadow_log(
    path: Optional[Path],
    row: Dict[str, Any],
    *,
    source: str = "",
    decision: Optional[MeykaGateDecision] = None,
) -> None:
    """Append a Meyka score / gate row to CSV (legacy + gate columns)."""
    dest = path or default_shadow_log_path()
    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        **{k: row.get(k) for k in SHADOW_CSV_FIELDS if k not in ("timestamp", "source")},
    }
    if decision is not None:
        payload.update(
            {
                "ticker": decision.ticker or payload.get("ticker"),
                "model": decision.model or payload.get("model"),
                "allow_discovery": decision.allow_discovery,
                "reason": decision.reason,
                "shadow_only": decision.shadow_only,
                "trading_hours_ok": decision.trading_hours_ok,
                "trading_hours_detail": decision.trading_hours_detail,
                "gap_open_pct": (decision.market_context or {}).get("gap_open_pct"),
                "vs_open_pct": (decision.market_context or {}).get("vs_open_pct"),
                **{k: decision.scores.get(k) for k in SHADOW_CSV_FIELDS if k in decision.scores},
            }
        )

    exists = dest.exists()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHADOW_CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(payload)


def decision_to_log_dict(decision: MeykaGateDecision) -> Dict[str, Any]:
    """Flat dict useful for logging / asdict-style dumps."""
    base = asdict(decision)
    base.pop("raw", None)
    return base
