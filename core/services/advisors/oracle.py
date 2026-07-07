"""
Oracle advisor — forward-looking pre-event scanner.

Pipeline:
  earnings calendar (+18 / +14), once per calendar day → price build → SO gate → consensus (buy+)
  → pre-Form4 shortlist → Form 4 confirm → LLM triage (all) → discover survivors

Production:
  python manage.py smartanalyse ORCL

LLM triage (default on): batch all gate passers; skip discover on verdict=veto or flag.
Disable: ORACLE_LLM_TRIAGE=0
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from core.services.advisors.advisor import AdvisorBase, register
from core.services.health.assess import run_component_results
from core.services.health.consensus import score_consensus_health
from core.services.health.risk_matrix import compute_so_snapshot
from core.services.health.so_ratings import (
    opportunity_grade_at_least,
    score_to_opportunity_grade,
    score_to_stability_grade,
)
from core.services.sec.form4 import get_form4_intel

logger = logging.getLogger(__name__)

EARNINGS_API_BASE = "https://api.earningsapi.com/v1/calendar/earnings"

# Earnings calendar lookaheads (calendar days); 2 API calls per daily refresh.
LOOKAHEAD_DAYS: Tuple[int, ...] = (18, 14)

MIN_OPPORTUNITY_GRADE = "C"
PRE_MOVE_LOOKBACK_DAYS = 20
MIN_PRICE = 5.0
DISCOVERY_COOLDOWN_HOURS = 48
YFINANCE_PAUSE_SEC = 0.05

# Only buy / strong_buy advance; hold, none, unknown, and sell tiers do not pass.
CONSENSUS_PASS_KEYS = frozenset({"strong_buy", "buy"})

_CONSENSUS_REC_DISPLAY = {
    "strong_buy": "Strong Buy",
    "buy": "Buy",
    "hold": "Hold",
    "sell": "Sell",
    "strong_sell": "Strong Sell",
}

# Form 4 confirm layer (on-demand per gate passer; aligned with Edgar thresholds).
FORM4_LOOKBACK_DAYS = 30
FORM4_VETO_TOTAL = -8.0
FORM4_BONUS_TOTAL = 8.0
FORM4_BONUS_WEIGHT = Decimal("1.15")
FORM4_RANK_BONUS_PCT = 3.0

# Run Form 4 only on the top N cheap-gate passers (SEC calls dominate large universes).
# Shortlist prefers creep/mixed over bang, then higher pre_move within each tier.
FORM4_PREFILTER_TOP_N = 25
_PRE_FORM4_SHAPE_TIER = {"creep": 0, "mixed": 1, "bang": 2, "flat": 3}

# LLM triage: batch attribution on gate passers; skip discover on verdict=veto or flag.
LLM_TRIAGE_TIMEOUT_SEC = 180.0

# Pre-earnings holds: wider than default PEAKED (see LNN lesson).
# STOP_PERCENTAGE first — hard cap on wrong picks (~6%); augmenting still ratchets winners.
# Hard stop is not evaluated in the first 60 min after open (same window as PEAKED).
ORACLE_STOP_MULT = Decimal("0.94")
ORACLE_SELL_INSTRUCTIONS = [
    ("STOP_PERCENTAGE", ORACLE_STOP_MULT, None),
    ("PERCENTAGE_AUGMENTING", Decimal("0.88"), 45),
    ("DESCENDING_TREND", Decimal("-0.20"), None),
    ("AFTER_DAYS", 45, None),
]

_CALENDAR_SESSIONS = ("pre", "after", "notSupplied")


def _earnings_api_key() -> Optional[str]:
    key = (os.environ.get("EARNINGS_API_KEY") or "").strip()
    return key or None


def _fetch_earnings_calendar(target: date, api_key: str) -> Dict[str, Any]:
    params = urllib.parse.urlencode({"date": target.isoformat(), "apikey": api_key})
    url = f"{EARNINGS_API_BASE}?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.load(resp)
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected earningsapi payload type: {type(payload)!r}")
    return payload


def _parse_calendar_row(row: Dict[str, Any], *, session: str) -> Optional[Dict[str, Any]]:
    symbol = (row.get("symbol") or row.get("ticker") or "").strip().upper()
    if not symbol:
        return None
    eps = (
        row.get("epsEstimate")
        or row.get("eps_estimate")
        or row.get("eps")
        or row.get("epsForecast")
    )
    rev = (
        row.get("revenueEstimate")
        or row.get("revenue_estimate")
        or row.get("revenue")
        or row.get("revenueForecast")
    )
    out: Dict[str, Any] = {
        "symbol": symbol,
        "name": (row.get("name") or "").strip(),
        "session": session,
    }
    if eps not in (None, ""):
        out["eps_estimate"] = eps
    if rev not in (None, ""):
        out["revenue_estimate"] = rev
    return out


def _calendar_rows_for_date(target: date, api_key: str) -> List[Dict[str, Any]]:
    payload = _fetch_earnings_calendar(target, api_key)
    rows: List[Dict[str, Any]] = []
    for session in _CALENDAR_SESSIONS:
        for row in payload.get(session) or []:
            if not isinstance(row, dict):
                continue
            parsed = _parse_calendar_row(row, session=session)
            if parsed:
                rows.append(parsed)
    return rows


def _fetch_close_series(
    symbol: str,
    start_date: date,
    end_date: date,
    cache: Dict[Tuple[str, date, date], Optional[pd.Series]],
) -> Optional[pd.Series]:
    key = (symbol.upper(), start_date, end_date)
    if key in cache:
        return cache[key]

    try:
        hist = yf.Ticker(symbol).history(
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=True,
        )
    except Exception:
        cache[key] = None
        return None

    if hist is None or hist.empty or "Close" not in hist.columns:
        cache[key] = None
        return None

    close = hist["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")]
    cache[key] = close
    return close


def _pct(a: float, b: float) -> Optional[float]:
    if a <= 0:
        return None
    return (b - a) / a * 100.0


def _daily_returns_pct(close_window: pd.Series) -> List[float]:
    if close_window is None or len(close_window) < 2:
        return []
    daily = close_window.pct_change().dropna() * 100.0
    out: List[float] = []
    for v in daily.tolist():
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def classify_creep_bang(
    close: pd.Series,
    *,
    start_idx: int,
    anchor_idx: int,
    pre_move_pct: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[int], str]:
    """Returns (late_share_pct, max_1d_jump_pct, up_days_lookback, label)."""
    if pre_move_pct is None or anchor_idx <= start_idx:
        return None, None, None, "flat"

    lookback_len = anchor_idx - start_idx
    late_days = min(5, lookback_len)
    late_start_idx = anchor_idx - late_days
    if late_start_idx < start_idx:
        late_start_idx = start_idx

    try:
        p_late0 = float(close.iloc[late_start_idx])
        p_anchor = float(close.iloc[anchor_idx])
    except (TypeError, ValueError, IndexError):
        return None, None, None, "flat"

    late_move_pct = _pct(p_late0, p_anchor)
    late_share_pct = None
    if late_move_pct is not None and abs(pre_move_pct) > 1e-9:
        late_share_pct = (late_move_pct / pre_move_pct) * 100.0

    window = close.iloc[start_idx : anchor_idx + 1]
    daily = _daily_returns_pct(window)
    if not daily:
        return late_share_pct, None, None, "flat"

    max_1d_jump_pct = max(daily)
    up_days = sum(1 for d in daily if d > 0)

    if pre_move_pct <= 0:
        label = "flat"
    elif (late_share_pct is not None and late_share_pct >= 65.0) or max_1d_jump_pct >= 6.0:
        label = "bang"
    elif (late_share_pct is not None and late_share_pct <= 45.0) and up_days >= max(
        8, int(len(daily) * 0.55)
    ):
        label = "creep"
    else:
        label = "mixed"

    return late_share_pct, max_1d_jump_pct, up_days, label


def compute_pre_move_build(
    symbol: str,
    lookback: int,
    cache: Dict[Tuple[str, date, date], Optional[pd.Series]],
    *,
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Forward pre-earnings build: close(T-lookback) → latest close (as-of today).
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return {"symbol": sym, "error": "empty symbol"}

    end = as_of or date.today()
    start = end - timedelta(days=lookback * 3 + 30)
    close = _fetch_close_series(sym, start, end, cache)
    if close is None or close.empty:
        return {"symbol": sym, "error": "no price history"}

    anchor_idx = len(close) - 1
    start_idx = anchor_idx - lookback
    if start_idx < 0:
        return {"symbol": sym, "error": "insufficient history"}

    try:
        p0 = float(close.iloc[start_idx])
        p1 = float(close.iloc[anchor_idx])
    except (TypeError, ValueError, IndexError):
        return {"symbol": sym, "error": "price parse failed"}

    if p0 <= 0 or p1 < MIN_PRICE:
        return {"symbol": sym, "error": "below min price or invalid anchor"}

    pre_move_pct = (p1 - p0) / p0 * 100.0
    late_share, max_jump, up_days, shape = classify_creep_bang(
        close,
        start_idx=start_idx,
        anchor_idx=anchor_idx,
        pre_move_pct=pre_move_pct,
    )
    return {
        "symbol": sym,
        "close_start": round(p0, 4),
        "close_anchor": round(p1, 4),
        "pre_move_pct": round(pre_move_pct, 2),
        "late_share_pct": round(late_share, 2) if late_share is not None else None,
        "max_1d_jump_pct": round(max_jump, 2) if max_jump is not None else None,
        "up_days_lookback": up_days,
        "shape": shape,
    }


def _consensus_gate(symbol: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Consensus allowlist gate.

    Returns (blocked, info). blocked=True when consensus is not buy/strong_buy
    (includes hold, none, unknown, sell).
    """
    result = score_consensus_health(symbol)
    rec = (result.recommendation_key or "").strip().lower().replace(" ", "_")
    info: Dict[str, Any] = {
        "consensus_rec": rec or None,
        "consensus_analyst_count": result.analyst_count,
    }
    return rec not in CONSENSUS_PASS_KEYS, info


def _so_gate(symbol: str) -> Tuple[bool, Dict[str, Any]]:
    """Opportunity gate (>= C) plus stability snapshot for explanation."""
    results = run_component_results(symbol)
    so = compute_so_snapshot(symbol, results)
    opp_score = so.get("opportunity")
    stab_score = so.get("stability")
    opp_grade = score_to_opportunity_grade(opp_score)
    stab_grade = score_to_stability_grade(stab_score)
    info: Dict[str, Any] = {
        "opp_grade": opp_grade.letter if opp_grade else None,
        "opp_label": opp_grade.label if opp_grade else None,
        "opp_score": round(float(opp_score), 1) if opp_score is not None else None,
        "stab_grade": stab_grade.letter if stab_grade else None,
        "stab_label": stab_grade.label if stab_grade else None,
        "stab_score": round(float(stab_score), 1) if stab_score is not None else None,
    }
    if opp_grade is None:
        return False, info
    if not opportunity_grade_at_least(opp_grade.letter, MIN_OPPORTUNITY_GRADE):
        return False, info
    return True, info


def _format_earnings_month_day(earnings_date: Optional[str]) -> str:
    if not earnings_date:
        return "earnings"
    try:
        parsed = date.fromisoformat(str(earnings_date))
    except ValueError:
        return str(earnings_date)
    return f"{parsed.strftime('%B')} {parsed.day}"


def _movement_sentence(shape: Optional[str]) -> str:
    key = (shape or "").strip().lower()
    if key == "creep":
        return "Price has shown a steady climb over the last 20 days"
    if key == "bang":
        return "Price has risen sharply, with much of the gain in the last few days"
    if key == "mixed":
        return "Price is up over 20 days with a mixed day-to-day pattern"
    return "Price is up over the last 20 days"


def _opportunity_sentence(candidate: Dict[str, Any]) -> str:
    grade = candidate.get("opp_grade") or "?"
    label = (candidate.get("opp_label") or "unknown").lower()
    score = candidate.get("opp_score")
    if score is not None:
        return (
            f"Opportunity grade {grade} ({label}, {score:.0f}) — passed minimum for discovery"
        )
    return f"Opportunity grade {grade} ({label}) — passed minimum for discovery"


def _consensus_sentence(candidate: Dict[str, Any]) -> str:
    rec = (candidate.get("consensus_rec") or "").strip().lower()
    if rec in ("", "none", "n/a", "null"):
        rec = ""
    count = candidate.get("consensus_analyst_count")
    if rec:
        display = _CONSENSUS_REC_DISPLAY.get(rec, rec.replace("_", " ").title())
        if count is not None:
            return f"Analyst consensus is {display} ({count} analysts)"
        return f"Analyst consensus is {display}"
    if count is not None:
        return f"Limited analyst coverage ({count} analysts)"
    return "Limited analyst coverage"


def _insider_sentence(candidate: Dict[str, Any]) -> str:
    days = FORM4_LOOKBACK_DAYS
    entries = candidate.get("form4_entries") or 0
    if not entries:
        return f"No insider buy or sell activity in the last {days} days"

    total = candidate.get("form4_total")
    name = (candidate.get("form4_insider_name") or "").strip()
    role = (candidate.get("form4_role_label") or "").strip()
    who = f"{role} {name}".strip() if role or name else ""

    try:
        total_f = float(total) if total is not None else 0.0
    except (TypeError, ValueError):
        total_f = 0.0

    if total_f >= FORM4_BONUS_TOTAL:
        if who:
            return f"{who} — net insider buying in the last {days} days (boosted rank)"
        return f"Net insider buying in the last {days} days (boosted rank)"
    if total_f < 0:
        if who:
            return f"{who} — insider selling in the last {days} days"
        return f"Insider selling in the last {days} days"
    if who:
        return f"{who} — insider activity in the last {days} days"
    return f"Insider activity in the last {days} days"


def _stability_sentence(candidate: Dict[str, Any]) -> str:
    grade = candidate.get("stab_grade")
    label = (candidate.get("stab_label") or "unknown").lower()
    if not grade:
        return "Stability grade unavailable"
    score = candidate.get("stab_score")
    if score is not None:
        return f"Stability grade {grade} ({label}, {score:.0f})"
    return f"Stability grade {grade} ({label})"


def _discovery_explanation_lead(candidate: Dict[str, Any]) -> str:
    pre_move = float(candidate.get("pre_move_pct") or 0.0)
    lookahead = candidate.get("lookahead_days")
    earnings = _format_earnings_month_day(candidate.get("earnings_date"))
    if pre_move < 0:
        return f"Down {abs(pre_move):.1f}% · {lookahead}d to {earnings}"
    return f"Up {pre_move:.1f}% · {lookahead}d to {earnings}"


def build_discovery_explanation(candidate: Dict[str, Any]) -> str:
    segments = [
        _discovery_explanation_lead(candidate),
        _movement_sentence(candidate.get("shape")),
        _opportunity_sentence(candidate),
        _consensus_sentence(candidate),
        _insider_sentence(candidate),
        _stability_sentence(candidate),
    ]
    return " | ".join(segments)


def _form4_summary(intel: Dict[str, Any]) -> Dict[str, Any]:
    total = float(intel.get("total") or 0.0)
    kind = intel.get("watch_kind") or ("form4_signal" if total >= 0 else "form4_sell")
    return {
        "form4_total": round(total, 2),
        "form4_kind": kind,
        "form4_buy_count": intel.get("buy_count"),
        "form4_sell_count": intel.get("sell_count"),
        "form4_entries": intel.get("entry_count"),
        "form4_insider_name": intel.get("insider_name"),
        "form4_role_label": intel.get("role_label"),
    }


def _form4_gate(symbol: str) -> Tuple[bool, str, Decimal, Dict[str, Any]]:
    """
    On-demand Form 4 check for one symbol.

    Returns (vetoed, note, discovery_weight, summary_fields).
  """
    sym = (symbol or "").strip().upper()
    try:
        intel = get_form4_intel(sym, days=FORM4_LOOKBACK_DAYS, limit=100)
    except Exception as exc:
        logger.warning("Oracle %s: Form4 lookup failed: %s", sym, exc)
        return False, "form4=unavailable", Decimal("1.0"), {}

    if not intel.get("entry_count"):
        return False, "form4=none", Decimal("1.0"), {
            "form4_total": None,
            "form4_kind": None,
            "form4_entries": 0,
            "form4_insider_name": None,
            "form4_role_label": None,
        }

    summary = _form4_summary(intel)
    total = float(summary["form4_total"] or 0.0)
    kind = summary.get("form4_kind") or "n/a"

    if kind == "form4_sell" and total <= FORM4_VETO_TOTAL:
        return True, f"form4={kind} total={total:+.1f}", Decimal("1.0"), summary

    weight = FORM4_BONUS_WEIGHT if total >= FORM4_BONUS_TOTAL else Decimal("1.0")
    note = f"form4={kind} total={total:+.1f}"
    if weight > Decimal("1.0"):
        note += " strong_buy"
    return False, note, weight, summary


def _pre_form4_shortlist_key(candidate: Dict[str, Any]) -> Tuple[int, float]:
    """Lower sorts first: creep/mixed before bang; then higher pre_move within tier."""
    shape = str(candidate.get("shape") or "flat").lower()
    tier = _PRE_FORM4_SHAPE_TIER.get(shape, 3)
    pre = float(candidate.get("pre_move_pct") or 0.0)
    return tier, -pre


def _select_form4_shortlist(
    pre_form4: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Return (symbols to run through Form 4, count skipped by prefilter).

    FORM4_PREFILTER_TOP_N <= 0 disables the cap (legacy: Form 4 on every passer).
    """
    if FORM4_PREFILTER_TOP_N <= 0 or len(pre_form4) <= FORM4_PREFILTER_TOP_N:
        return pre_form4, 0
    ranked = sorted(pre_form4, key=_pre_form4_shortlist_key)
    shortlist = ranked[:FORM4_PREFILTER_TOP_N]
    skipped = len(pre_form4) - len(shortlist)
    return shortlist, skipped


def _llm_triage_enabled() -> bool:
    raw = (os.environ.get("ORACLE_LLM_TRIAGE") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _fmt_triage_pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):+.1f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_triage_consensus(candidate: Dict[str, Any]) -> str:
    rec = (candidate.get("consensus_rec") or "").strip().lower() or "none"
    count = candidate.get("consensus_analyst_count")
    if count is not None:
        return f"{rec}({count})"
    return rec


def _fmt_triage_form4(candidate: Dict[str, Any]) -> str:
    entries = candidate.get("form4_entries") or 0
    if not entries:
        return "none"
    total = candidate.get("form4_total")
    kind = candidate.get("form4_kind") or "activity"
    if total is not None:
        try:
            return f"{kind}({float(total):+.0f})"
        except (TypeError, ValueError):
            pass
    return str(kind)


def build_pipe_row_from_candidate(candidate: Dict[str, Any]) -> str:
    """Compact pipe row: sym|earn|d|b20|shape|late%|jmp%|opp|f4|cons."""
    sym = str(candidate.get("symbol") or "").strip().upper()
    late = candidate.get("late_share_pct")
    jmp = candidate.get("max_1d_jump_pct")
    late_s = "—" if late is None else f"{float(late):.0f}"
    jmp_s = "—" if jmp is None else f"{float(jmp):.2f}"
    return "|".join(
        [
            sym,
            str(candidate.get("earnings_date") or ""),
            str(candidate.get("lookahead_days") or ""),
            _fmt_triage_pct(candidate.get("pre_move_pct")),
            str(candidate.get("shape") or "flat"),
            late_s,
            jmp_s,
            str(candidate.get("opp_grade") or "?"),
            _fmt_triage_form4(candidate),
            _fmt_triage_consensus(candidate),
        ]
    )


LLM_TRIAGE_PROMPT_HEADER = """Validate pre-earnings scanner picks. For each ticker: WHY moving? Pre-earnings positioning vs external catalyst? Skeptical; facts vs interpretation. No advice/targets.

One JSON object per input row, same order. Creeps usually allow unless discrete headline.
allow: fits pre-earnings drift, no obvious priced-in event
flag: macro/sector beta, late spike, or partial priced-in
veto: clear company event likely fully priced (guidance, contract, M&A)

Legend: sym|earn|d|b20|shape|late%|jmp%|opp|f4|cons

INPUT (as-of {as_of}):
{pipe_block}

JSON array only:
[{{"sym":"","driver_cat":"earnings_anticipation|sector_repricing|company_catalyst|speculative_surge|relief_reversal|mixed|unclear","driver_desc":"","thesis_fit":"supports|weakens|neutral","priced_in":"low|medium|high","verdict":"allow|flag|veto","reason":"","conf":1-10,"evidence":[{{"fact":"","src":"","date":"YYYY-MM-DD"}}],"peers":[],"risks":[]}}]
"""


def build_llm_triage_prompt(as_of: str, pipe_rows: List[str]) -> str:
    return LLM_TRIAGE_PROMPT_HEADER.format(as_of=as_of, pipe_block="\n".join(pipe_rows))


def _normalize_llm_triage_results(results: Any) -> List[Dict[str, Any]]:
    if results is None:
        return []
    if isinstance(results, list):
        return [r for r in results if isinstance(r, dict)]
    if isinstance(results, dict):
        for key in ("results", "tickers", "data"):
            inner = results.get(key)
            if isinstance(inner, list):
                return [r for r in inner if isinstance(r, dict)]
        return [results]
    return []


def _llm_triage_by_symbol(results: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in _normalize_llm_triage_results(results):
        sym = (item.get("sym") or item.get("symbol") or "").strip().upper()
        if sym:
            out[sym] = item
    return out


def _llm_triage_verdict(triage: Optional[Dict[str, Any]]) -> str:
    if not triage:
        return ""
    return (triage.get("verdict") or "").strip().lower()


def _llm_triage_is_veto(triage: Optional[Dict[str, Any]]) -> bool:
    return _llm_triage_verdict(triage) == "veto"


def _llm_triage_is_flag(triage: Optional[Dict[str, Any]]) -> bool:
    return _llm_triage_verdict(triage) == "flag"


def _llm_triage_skips_discover(triage: Optional[Dict[str, Any]]) -> bool:
    return _llm_triage_verdict(triage) in ("veto", "flag")


def _llm_triage_blob_summary(triage: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "verdict": triage.get("verdict"),
        "driver_cat": triage.get("driver_cat"),
        "thesis_fit": triage.get("thesis_fit"),
        "priced_in": triage.get("priced_in"),
        "conf": triage.get("conf", triage.get("confidence")),
        "reason": triage.get("reason"),
    }


def run_llm_triage_batch(
    candidates: List[Dict[str, Any]],
    *,
    as_of: str,
    timeout: float = LLM_TRIAGE_TIMEOUT_SEC,
) -> Tuple[Optional[str], Dict[str, Dict[str, Any]]]:
    """Batch LLM attribution. Returns (model, symbol -> triage dict). Fail open on error."""
    if not candidates:
        return None, {}

    from core.services.llm.router import ask_llm

    pipe_rows = [build_pipe_row_from_candidate(c) for c in candidates]
    prompt = build_llm_triage_prompt(as_of, pipe_rows)
    model, results, _, _ = ask_llm(
        prompt=prompt,
        advisor_name="Oracle",
        gemini_model_index=0,
        gemini_key_index=0,
        timeout=timeout,
        use_search=True,
    )
    if not results:
        logger.warning("Oracle LLM triage: no response (%d candidates)", len(candidates))
        return model, {}

    by_sym = _llm_triage_by_symbol(results)
    if len(candidates) == 1:
        logger.info(
            "Oracle LLM triage %s: model=%s verdict=%s",
            candidates[0].get("symbol"),
            model,
            (by_sym.get(str(candidates[0].get("symbol") or "").upper()) or {}).get("verdict"),
        )
    else:
        logger.info(
            "Oracle LLM triage: model=%s candidates=%d parsed=%d",
            model,
            len(candidates),
            len(by_sym),
        )
    return model, by_sym


class Oracle(AdvisorBase):
    """Forward pre-earnings scanner."""

    # First SA on/after this UTC hour may refresh the earnings calendar (once per calendar day).
    PROCESS_CUTOFF_HOUR_UTC = 0

    sell_instructions = ORACLE_SELL_INSTRUCTIONS

    def discover(self, sa) -> None:
        today = date.today().isoformat()

        if not self.should_process_market_date_once(
            target_date=today,
            cutoff_hour_utc=self.PROCESS_CUTOFF_HOUR_UTC,
        ):
            return

        logger.info(
            "Oracle sa=%s: discover start (lookaheads=%s)",
            sa.id,
            LOOKAHEAD_DAYS,
        )

        api_key = _earnings_api_key()
        if not api_key:
            logger.error("Oracle sa=%s: EARNINGS_API_KEY not set; skip calendar refresh", sa.id)
            return

        try:
            universe = self._build_earnings_universe(date.fromisoformat(today), api_key=api_key)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            logger.error("Oracle sa=%s: earningsapi fetch failed: %s", sa.id, exc)
            return

        state = self._advisor_blob_state()
        state["universe"] = universe["slots"]
        state["symbol_index"] = universe["symbol_index"]
        state["universe_count"] = universe["symbol_count"]
        state["calendar_refreshed"] = today
        self._save_advisor_blob_state(state)

        logger.info(
            "Oracle sa=%s: calendar refreshed slots=%d unique_symbols=%d",
            sa.id,
            len(universe["slots"]),
            universe["symbol_count"],
        )

        candidates, scan_stats = self._score_universe(universe["symbol_index"])

        discoveries, llm_vetoed, llm_flagged, triage_by_sym, triage_model = self._discover_top_candidates(
            sa,
            candidates,
            as_of=today,
        )

        state = self._advisor_blob_state()
        state["last_scan"] = {
            "date": today,
            "scored": scan_stats["scored"],
            "pre_form4_passers": scan_stats.get("pre_form4_passers", 0),
            "form4_prefilter_skipped": scan_stats.get("form4_prefilter_skipped", 0),
            "passed_gates": scan_stats["passed_gates"],
            "form4_vetoed": scan_stats.get("form4_vetoed", 0),
            "llm_vetoed": llm_vetoed,
            "llm_flagged": llm_flagged,
            "llm_triage_model": triage_model,
            "llm_triage": {
                sym: _llm_triage_blob_summary(t)
                for sym, t in triage_by_sym.items()
            },
            "discoveries": discoveries,
            "top_candidates": [
                {
                    "symbol": c["symbol"],
                    "pre_move_pct": c.get("pre_move_pct"),
                    "rank_score": c.get("rank_score"),
                    "shape": c.get("shape"),
                    "opp_grade": c.get("opp_grade"),
                    "form4_total": c.get("form4_total"),
                    "earnings_date": c.get("earnings_date"),
                    "llm_verdict": (
                        triage_by_sym.get(str(c.get("symbol") or "").upper()) or {}
                    ).get("verdict"),
                }
                for c in candidates[:10]
            ],
        }
        self._save_advisor_blob_state(state)
        self.mark_market_date_processed(today)

        logger.info(
            "Oracle sa=%s: scored=%d pre_form4=%d form4_skipped=%d passed_gates=%d "
            "form4_vetoed=%d llm_vetoed=%d llm_flagged=%d discoveries=%d",
            sa.id,
            scan_stats["scored"],
            scan_stats.get("pre_form4_passers", 0),
            scan_stats.get("form4_prefilter_skipped", 0),
            scan_stats["passed_gates"],
            scan_stats.get("form4_vetoed", 0),
            llm_vetoed,
            llm_flagged,
            discoveries,
        )

    def _build_earnings_universe(self, today: date, *, api_key: str) -> Dict[str, Any]:
        slots: List[Dict[str, Any]] = []
        symbol_index: Dict[str, Dict[str, Any]] = {}

        for days in LOOKAHEAD_DAYS:
            target = today + timedelta(days=days)
            try:
                symbols = _calendar_rows_for_date(target, api_key)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
                logger.warning(
                    "Oracle: earningsapi failed for +%dd (%s): %s",
                    days,
                    target.isoformat(),
                    exc,
                )
                symbols = []

            slots.append(
                {
                    "lookahead_days": days,
                    "earnings_date": target.isoformat(),
                    "count": len(symbols),
                    "symbols": symbols,
                }
            )

            for row in symbols:
                sym = row["symbol"]
                existing = symbol_index.get(sym)
                if existing is None or days < int(existing.get("lookahead_days") or 999):
                    symbol_index[sym] = {
                        **row,
                        "lookahead_days": days,
                        "earnings_date": target.isoformat(),
                    }

        return {
            "slots": slots,
            "symbol_index": symbol_index,
            "symbol_count": len(symbol_index),
        }

    def _score_universe(
        self,
        symbol_index: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        cache: Dict[Tuple[str, date, date], Optional[pd.Series]] = {}
        pre_form4: List[Dict[str, Any]] = []
        scored = 0

        for sym, meta in sorted(symbol_index.items()):
            scored += 1
            build = compute_pre_move_build(sym, PRE_MOVE_LOOKBACK_DAYS, cache)
            time.sleep(YFINANCE_PAUSE_SEC)

            if build.get("error"):
                logger.debug("Oracle %s: build skip — %s", sym, build["error"])
                continue

            pre_move = build.get("pre_move_pct")
            if pre_move is None or float(pre_move) <= 0:
                logger.debug("Oracle %s: no positive pre-move (%s)", sym, pre_move)
                continue

            opp_ok, so_info = _so_gate(sym)
            if not opp_ok:
                logger.debug(
                    "Oracle %s: opportunity below %s (grade=%s score=%s)",
                    sym,
                    MIN_OPPORTUNITY_GRADE,
                    so_info.get("opp_grade"),
                    so_info.get("opp_score"),
                )
                continue

            consensus_blocked, consensus_info = _consensus_gate(sym)
            if consensus_blocked:
                logger.debug(
                    "Oracle %s: consensus blocked (%s)",
                    sym,
                    consensus_info.get("consensus_rec") or "none",
                )
                continue

            pre_form4.append(
                {
                    **build,
                    **so_info,
                    **consensus_info,
                    "earnings_date": meta.get("earnings_date"),
                    "lookahead_days": meta.get("lookahead_days"),
                    "session": meta.get("session"),
                }
            )

        form4_shortlist, form4_prefilter_skipped = _select_form4_shortlist(pre_form4)
        if form4_prefilter_skipped:
            logger.info(
                "Oracle Form4 prefilter: %d cheap-gate passers → %d shortlisted (%d skipped)",
                len(pre_form4),
                len(form4_shortlist),
                form4_prefilter_skipped,
            )

        passed: List[Dict[str, Any]] = []
        passed_gates = 0
        form4_vetoed = 0

        for row in form4_shortlist:
            sym = str(row.get("symbol") or "").strip().upper()
            if not sym:
                continue

            f4_veto, f4_note, f4_weight, f4_summary = _form4_gate(sym)
            if f4_veto:
                form4_vetoed += 1
                logger.info("Oracle %s: Form4 veto (%s)", sym, f4_note)
                continue

            pre_move_f = float(row.get("pre_move_pct") or 0.0)
            f4_total = f4_summary.get("form4_total")
            rank_score = pre_move_f
            if f4_total is not None and float(f4_total) >= FORM4_BONUS_TOTAL:
                rank_score += FORM4_RANK_BONUS_PCT

            passed_gates += 1
            passed.append(
                {
                    **row,
                    **f4_summary,
                    "form4_note": f4_note,
                    "discovery_weight": f4_weight,
                    "rank_score": round(rank_score, 2),
                }
            )

        passed.sort(key=lambda c: float(c.get("rank_score") or c.get("pre_move_pct") or 0.0), reverse=True)
        return passed, {
            "scored": scored,
            "pre_form4_passers": len(pre_form4),
            "form4_prefilter_skipped": form4_prefilter_skipped,
            "passed_gates": passed_gates,
            "form4_vetoed": form4_vetoed,
        }

    def _discover_top_candidates(
        self,
        sa,
        candidates: List[Dict[str, Any]],
        *,
        as_of: str,
    ) -> Tuple[int, int, int, Dict[str, Dict[str, Any]], Optional[str]]:
        """
        Batch LLM triage on all gate passers; discover survivors (allow, not veto/flag).

        Returns (discoveries, llm_vetoed, llm_flagged, triage_by_sym, last_triage_model).
        """
        triage_by_sym: Dict[str, Dict[str, Any]] = {}
        triage_model: Optional[str] = None
        llm_vetoed = 0
        llm_flagged = 0
        discoveries = 0

        if _llm_triage_enabled() and candidates:
            triage_model, triage_by_sym = run_llm_triage_batch(candidates, as_of=as_of)

        for candidate in candidates:
            sym = str(candidate.get("symbol") or "").strip().upper()
            if not sym:
                continue

            if _llm_triage_enabled():
                triage = triage_by_sym.get(sym)
                if _llm_triage_skips_discover(triage):
                    reason = (triage.get("reason") or triage.get("driver_desc") or "LLM skip").strip()
                    if _llm_triage_is_veto(triage):
                        llm_vetoed += 1
                        label = "veto"
                    else:
                        llm_flagged += 1
                        label = "flag"
                    logger.info(
                        "Oracle %s: LLM triage %s (%s) — %s",
                        sym,
                        label,
                        triage.get("driver_cat") or "n/a",
                        reason[:200],
                    )
                    continue

            if not self.allow_discovery(sym, period=DISCOVERY_COOLDOWN_HOURS):
                logger.info("Oracle %s: allow_discovery false (cooldown)", sym)
                continue

            explanation = build_discovery_explanation(candidate)
            weight = candidate.get("discovery_weight") or Decimal("1.0")
            if self.discovered(
                sa,
                sym,
                explanation,
                sell_instructions=self.sell_instructions,
                weight=weight,
            ):
                discoveries += 1

        return discoveries, llm_vetoed, llm_flagged, triage_by_sym, triage_model


    def analyze(self, sa, stock) -> None:
        return


register(name="Oracle", python_class="Oracle")
