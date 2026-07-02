"""
Oracle advisor — forward-looking pre-event scanner.

Pipeline:
  earnings calendar (+21 / +14), once per calendar day → price build → SO gate → consensus veto
  → Form 4 confirm → discover

Production:
  python manage.py smartanalyse ORCL
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
)
from core.services.sec.form4 import get_form4_intel

logger = logging.getLogger(__name__)

EARNINGS_API_BASE = "https://api.earningsapi.com/v1/calendar/earnings"

# Earnings calendar lookaheads (calendar days); 2 API calls per daily refresh.
LOOKAHEAD_DAYS: Tuple[int, ...] = (21, 14)

MIN_OPPORTUNITY_GRADE = "C"
PRE_MOVE_LOOKBACK_DAYS = 20
MAX_DISCOVERIES_PER_RUN = 3
MIN_PRICE = 5.0
DISCOVERY_COOLDOWN_HOURS = 48
YFINANCE_PAUSE_SEC = 0.05

CONSENSUS_VETO_KEYS = frozenset({"sell", "strong_sell"})

# Form 4 confirm layer (on-demand per gate passer; aligned with Edgar thresholds).
FORM4_LOOKBACK_DAYS = 30
FORM4_VETO_TOTAL = -8.0
FORM4_BONUS_TOTAL = 8.0
FORM4_BONUS_WEIGHT = Decimal("1.15")
FORM4_RANK_BONUS_PCT = 3.0

# Pre-earnings holds: wider than default PEAKED (see LNN lesson).
ORACLE_SELL_INSTRUCTIONS = [
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


def _consensus_veto(symbol: str) -> Tuple[bool, str]:
    """Hard veto on major negative consensus."""
    result = score_consensus_health(symbol)
    rec = (result.recommendation_key or "").strip().lower()
    if rec in CONSENSUS_VETO_KEYS:
        return True, f"consensus={rec}"
    return False, rec or "n/a"


def _opportunity_passes(symbol: str) -> Tuple[bool, Optional[str], Optional[float]]:
    results = run_component_results(symbol)
    so = compute_so_snapshot(symbol, results)
    opp_score = so.get("opportunity")
    grade = score_to_opportunity_grade(opp_score)
    if grade is None:
        return False, None, opp_score
    if not opportunity_grade_at_least(grade.letter, MIN_OPPORTUNITY_GRADE):
        return False, grade.letter, opp_score
    return True, grade.letter, opp_score


def _form4_summary(intel: Dict[str, Any]) -> Dict[str, Any]:
    total = float(intel.get("total") or 0.0)
    kind = intel.get("watch_kind") or ("form4_signal" if total >= 0 else "form4_sell")
    return {
        "form4_total": round(total, 2),
        "form4_kind": kind,
        "form4_buy_count": intel.get("buy_count"),
        "form4_sell_count": intel.get("sell_count"),
        "form4_entries": intel.get("entry_count"),
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
        return False, "form4=none", Decimal("1.0"), {"form4_total": None, "form4_kind": None}

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
        discoveries = self._discover_top_candidates(sa, candidates)

        state = self._advisor_blob_state()
        state["last_scan"] = {
            "date": today,
            "scored": scan_stats["scored"],
            "passed_gates": scan_stats["passed_gates"],
            "form4_vetoed": scan_stats.get("form4_vetoed", 0),
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
                }
                for c in candidates[:10]
            ],
        }
        self._save_advisor_blob_state(state)
        self.mark_market_date_processed(today)

        logger.info(
            "Oracle sa=%s: scored=%d passed_gates=%d form4_vetoed=%d discoveries=%d",
            sa.id,
            scan_stats["scored"],
            scan_stats["passed_gates"],
            scan_stats.get("form4_vetoed", 0),
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
        passed: List[Dict[str, Any]] = []
        scored = 0
        passed_gates = 0
        form4_vetoed = 0

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

            opp_ok, opp_grade, opp_score = _opportunity_passes(sym)
            if not opp_ok:
                logger.debug(
                    "Oracle %s: opportunity below %s (grade=%s score=%s)",
                    sym,
                    MIN_OPPORTUNITY_GRADE,
                    opp_grade,
                    opp_score,
                )
                continue

            vetoed, consensus_note = _consensus_veto(sym)
            if vetoed:
                logger.debug("Oracle %s: consensus veto (%s)", sym, consensus_note)
                continue

            f4_veto, f4_note, f4_weight, f4_summary = _form4_gate(sym)
            if f4_veto:
                form4_vetoed += 1
                logger.info("Oracle %s: Form4 veto (%s)", sym, f4_note)
                continue

            pre_move_f = float(pre_move)
            f4_total = f4_summary.get("form4_total")
            rank_score = pre_move_f
            if f4_total is not None and float(f4_total) >= FORM4_BONUS_TOTAL:
                rank_score += FORM4_RANK_BONUS_PCT

            passed_gates += 1
            passed.append(
                {
                    **build,
                    **f4_summary,
                    "earnings_date": meta.get("earnings_date"),
                    "lookahead_days": meta.get("lookahead_days"),
                    "session": meta.get("session"),
                    "opp_grade": opp_grade,
                    "opp_score": round(float(opp_score), 1) if opp_score is not None else None,
                    "consensus": consensus_note,
                    "form4_note": f4_note,
                    "discovery_weight": f4_weight,
                    "rank_score": round(rank_score, 2),
                }
            )

        passed.sort(key=lambda c: float(c.get("rank_score") or c.get("pre_move_pct") or 0.0), reverse=True)
        return passed, {
            "scored": scored,
            "passed_gates": passed_gates,
            "form4_vetoed": form4_vetoed,
        }

    def _discover_top_candidates(
        self,
        sa,
        candidates: List[Dict[str, Any]],
    ) -> int:
        discoveries = 0
        for candidate in candidates:
            if discoveries >= MAX_DISCOVERIES_PER_RUN:
                break
            sym = str(candidate.get("symbol") or "").strip().upper()
            if not sym:
                continue
            if not self.allow_discovery(sym, period=DISCOVERY_COOLDOWN_HOURS):
                logger.info("Oracle %s: discovery cooldown active", sym)
                continue

            explanation = self._discovery_explanation(candidate)
            weight = candidate.get("discovery_weight") or Decimal("1.0")
            if self.discovered(
                sa,
                sym,
                explanation,
                sell_instructions=self.sell_instructions,
                weight=weight,
            ):
                discoveries += 1
        return discoveries

    @staticmethod
    def _discovery_explanation(candidate: Dict[str, Any]) -> str:
        sym = candidate.get("symbol") or ""
        earnings_date = candidate.get("earnings_date") or "?"
        lookahead = candidate.get("lookahead_days")
        pre_move = candidate.get("pre_move_pct")
        shape = candidate.get("shape") or "?"
        opp_grade = candidate.get("opp_grade") or "?"
        consensus = candidate.get("consensus") or "n/a"
        form4_note = candidate.get("form4_note") or "n/a"
        late_share = candidate.get("late_share_pct")
        late_s = f"{late_share:.0f}%" if late_share is not None else "n/a"
        return (
            f"Oracle | pre-earnings build | {sym} earnings {earnings_date} "
            f"(+{lookahead}d) | +{pre_move:.1f}% {PRE_MOVE_LOOKBACK_DAYS}d build | "
            f"{shape} | late={late_s} | opp={opp_grade} | consensus={consensus} | {form4_note}"
        )

    def analyze(self, sa, stock) -> None:
        return


register(name="Oracle", python_class="Oracle")
