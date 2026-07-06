"""
Vulture advisor — damaged-quality recovery watchlist.

Daily flow (after US cash session):
  1. Weekly chronic-damage scan (Path 1, ~every 7 days) → diagnostic watch intake
  2. EOD large-drop scan + LLM triage (Path 2) → watchlist intake (monitor only)
  3. Re-score pending watchlist via health.diagnostic
  4. BUY READY persistence → rare discovery
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf
from django.conf import settings

from core.services.advisors.advisor import AdvisorBase, register
from core.services.financial import polygon as financial_polygon
from core.services.health.diagnostic import analyze_symbol, diagnostic_to_dict
from core.services.market import last_completed_trading_day, prior_trading_day, resolve_eod_session_date

logger = logging.getLogger(__name__)

# --- Advisor runtime ---
PROCESS_CUTOFF_HOUR_UTC = 21
SCAN_REBUILD_DAYS = 7
WATCHLIST_DAYS = 28
BUY_READY_STREAK_DAYS = 2
DISCOVERY_COOLDOWN_HOURS = 72
DISCOVERY_WEIGHT = 1.0

VULTURE_STOP_MULT = Decimal("0.88")
VULTURE_TARGET_MULT = Decimal("1.15")
VULTURE_MAX_HOLD_DAYS = 21

# --- Shared filters ---
DEFAULT_MIN_PRICE = 5.0
DEFAULT_MIN_VOLUME = 500_000
DEFAULT_MIN_DOLLAR_VOLUME = 25_000_000.0

ETF_EXCLUDE_TICKERS = frozenset(
    {
        "DIA", "EEM", "EFA", "GLD", "HYG", "IWM", "IVV", "LQD", "QQQ", "RSP", "SH",
        "SMH", "SOXL", "SOXS", "SOXX", "SPCX", "SPXL", "SPXS", "SPY", "SQQQ", "TLT",
        "TQQQ", "UPRO", "VCIT", "VCSH", "VOO", "VTI", "VXUS", "XBI", "XLE", "XLF",
        "XLI", "XLK", "XLP", "XLV",
    }
)

# --- Path 1: weekly chronic damage scan ---
SCAN_TOP = 50
SCAN_SEED_UNIVERSE = 1200
SCAN_BATCH_SIZE = 100
SCAN_MIN_3M_DAMAGE_PCT = 20.0
SCAN_MIN_52W_DAMAGE_PCT = 30.0
SCAN_MAX_52W_DAMAGE_PCT = 70.0
WATCHLIST_STAGES = frozenset({"WATCH", "WARM", "RECOVERY"})

# --- Path 2: EOD large-drop intake ---
EOD_TOP = 25
EOD_MIN_DAY_DROP_PCT = 7.0
EOD_LLM_BATCH_SIZE = 10
MIN_LLM_MONITOR_CONFIDENCE = 0.5

NON_EQUITY_QUOTE_TYPES = frozenset({"ETF", "MUTUALFUND", "TRUST"})
LEVERAGED_NAME_HINTS = (
    " 2x ", " 3x ", " -2x", " -3x", "leveraged", "ultra ", "ultra-", "daily ",
    "inverse", " bear ", " bull ", "+1x", "-1x", "+2x", "-2x", "+3x", "-3x", "single stock",
)


# --- Dataclasses ---


@dataclass(frozen=True)
class SeedRow:
    symbol: str
    polygon_price: float
    polygon_volume: int
    polygon_dollar_volume: float


@dataclass(frozen=True)
class VultureScanCandidate:
    rank: int
    symbol: str
    price: float
    polygon_price: float
    volume: int
    dollar_volume: float
    high_3m: Optional[float]
    high_52w: Optional[float]
    damage_3m_pct: Optional[float]
    damage_52w_pct: Optional[float]
    near_extreme_collapse: bool
    damage_score: float
    trigger: str


@dataclass(frozen=True)
class EodDropCandidate:
    rank: int
    symbol: str
    session_date: str
    close: float
    prior_close: float
    open: float
    volume: int
    dollar_volume: float
    day_change_pct: float
    session_change_pct: float
    llm_verdict: str = ""
    llm_damage_type: str = ""
    llm_reason: str = ""
    llm_confidence: Optional[float] = None


# --- Helpers ---


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _chunks(values: Sequence[Any], size: int) -> Iterable[list[Any]]:
    for idx in range(0, len(values), size):
        yield list(values[idx : idx + size])


# --- Path 1: weekly scan ---


def fetch_grouped_daily_rows(
    *,
    min_price: float,
    min_volume: int,
    scan_date: date,
) -> list[dict[str, Any]]:
    df = financial_polygon.get_filtered_stocks(
        min_price=min_price,
        min_volume=min_volume,
        test_date=scan_date.isoformat(),
    )
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


def build_seed_rows(
    rows: Iterable[dict[str, Any]],
    *,
    min_price: float,
    min_volume: int,
    min_dollar_volume: float,
    seed_universe: int,
    include_etfs: bool,
) -> list[SeedRow]:
    seeds: list[SeedRow] = []

    for row in rows:
        symbol = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
        if not symbol or "." in symbol or "-" in symbol or "/" in symbol:
            continue
        if not include_etfs and symbol in ETF_EXCLUDE_TICKERS:
            continue

        price = _safe_float(row.get("price"))
        volume = _safe_int(row.get("today_volume") or row.get("volume"))
        if price is None or price < min_price or volume < min_volume:
            continue

        dollar_volume = price * volume
        if dollar_volume < min_dollar_volume:
            continue

        seeds.append(
            SeedRow(
                symbol=symbol,
                polygon_price=price,
                polygon_volume=volume,
                polygon_dollar_volume=dollar_volume,
            )
        )

    seeds.sort(key=lambda row: row.polygon_dollar_volume, reverse=True)
    return seeds[:seed_universe]


def _download_history(symbols: Sequence[str], *, batch_size: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for batch in _chunks(list(symbols), batch_size):
        data = yf.download(
            batch,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
        if data is None or data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            for symbol in batch:
                if symbol not in data.columns.get_level_values(0):
                    continue
                hist = data[symbol].copy()
                hist.columns = [str(col).lower() for col in hist.columns]
                out[symbol] = hist.dropna(how="all")
        else:
            symbol = batch[0]
            hist = data.copy()
            hist.columns = [str(col).lower() for col in hist.columns]
            out[symbol] = hist.dropna(how="all")
    return out


def _history_price(hist: pd.DataFrame, fallback: float) -> float:
    if hist.empty or "close" not in hist.columns:
        return fallback
    price = _safe_float(hist["close"].dropna().iloc[-1] if not hist["close"].dropna().empty else None)
    return price if price is not None and price > 0 else fallback


def _down_from_high(price: float, high: Optional[float]) -> Optional[float]:
    if high is None or high <= 0 or price <= 0:
        return None
    return (price - high) / high * 100.0


def _trigger_label(
    damage_3m: Optional[float],
    damage_52w: Optional[float],
    min_3m: float,
    min_52w: float,
) -> str:
    parts = []
    if damage_3m is not None and damage_3m <= -min_3m:
        parts.append("3m")
    if damage_52w is not None and damage_52w <= -min_52w:
        parts.append("52w")
    return "+".join(parts) if parts else ""


def build_scan_candidates(
    seeds: Sequence[SeedRow],
    *,
    min_3m_damage_pct: float = SCAN_MIN_3M_DAMAGE_PCT,
    min_52w_damage_pct: float = SCAN_MIN_52W_DAMAGE_PCT,
    max_52w_damage_pct: float = SCAN_MAX_52W_DAMAGE_PCT,
    top: int = SCAN_TOP,
    batch_size: int = SCAN_BATCH_SIZE,
) -> list[VultureScanCandidate]:
    histories = _download_history([seed.symbol for seed in seeds], batch_size=batch_size)
    candidates: list[VultureScanCandidate] = []

    for seed in seeds:
        hist = histories.get(seed.symbol, pd.DataFrame())
        price = _history_price(hist, seed.polygon_price)

        high_3m = None
        high_52w = None
        if not hist.empty and "high" in hist.columns:
            highs = hist["high"].dropna().astype(float)
            if not highs.empty:
                high_3m = float(highs.tail(63).max())
                high_52w = float(highs.tail(252).max())

        damage_3m = _down_from_high(price, high_3m)
        damage_52w = _down_from_high(price, high_52w)
        trigger = _trigger_label(damage_3m, damage_52w, min_3m_damage_pct, min_52w_damage_pct)
        if not trigger:
            continue

        extreme = damage_52w is not None and damage_52w <= -max_52w_damage_pct
        if extreme:
            continue

        damage_score = max(abs(damage_3m or 0.0), abs(damage_52w or 0.0))
        candidates.append(
            VultureScanCandidate(
                rank=0,
                symbol=seed.symbol,
                price=price,
                polygon_price=seed.polygon_price,
                volume=seed.polygon_volume,
                dollar_volume=seed.polygon_dollar_volume,
                high_3m=high_3m,
                high_52w=high_52w,
                damage_3m_pct=damage_3m,
                damage_52w_pct=damage_52w,
                near_extreme_collapse=extreme,
                damage_score=damage_score,
                trigger=trigger,
            )
        )

    candidates.sort(key=lambda row: (row.damage_score, row.dollar_volume), reverse=True)
    return [
        VultureScanCandidate(**{**asdict(candidate), "rank": idx})
        for idx, candidate in enumerate(candidates[:top], start=1)
    ]


def build_weekly_scan_candidates(
    scan_date: Optional[date] = None,
    *,
    min_price: float = DEFAULT_MIN_PRICE,
    min_volume: int = DEFAULT_MIN_VOLUME,
    min_dollar_volume: float = DEFAULT_MIN_DOLLAR_VOLUME,
    seed_universe: int = SCAN_SEED_UNIVERSE,
    include_etfs: bool = False,
    min_3m_damage_pct: float = SCAN_MIN_3M_DAMAGE_PCT,
    min_52w_damage_pct: float = SCAN_MIN_52W_DAMAGE_PCT,
    max_52w_damage_pct: float = SCAN_MAX_52W_DAMAGE_PCT,
    top: int = SCAN_TOP,
    batch_size: int = SCAN_BATCH_SIZE,
) -> tuple[list[VultureScanCandidate], dict[str, int]]:
    session = scan_date or last_completed_trading_day()
    raw_rows = fetch_grouped_daily_rows(
        min_price=min_price,
        min_volume=min_volume,
        scan_date=session,
    )
    seeds = build_seed_rows(
        raw_rows,
        min_price=min_price,
        min_volume=min_volume,
        min_dollar_volume=min_dollar_volume,
        seed_universe=seed_universe,
        include_etfs=include_etfs,
    )
    candidates = build_scan_candidates(
        seeds,
        min_3m_damage_pct=min_3m_damage_pct,
        min_52w_damage_pct=min_52w_damage_pct,
        max_52w_damage_pct=max_52w_damage_pct,
        top=top,
        batch_size=batch_size,
    )
    stats = {
        "polygon_rows": len(raw_rows),
        "seeds": len(seeds),
        "candidates": len(candidates),
    }
    logger.info(
        "Vulture weekly scan %s: polygon=%s seeds=%s candidates=%s",
        session.isoformat(),
        stats["polygon_rows"],
        stats["seeds"],
        stats["candidates"],
    )
    return candidates, stats


def scan_candidate_to_meta(candidate: VultureScanCandidate) -> dict[str, Any]:
    return {
        "intake": "weekly_scan",
        "rank": candidate.rank,
        "trigger": candidate.trigger,
        "damage_3m_pct": candidate.damage_3m_pct,
        "damage_52w_pct": candidate.damage_52w_pct,
        "damage_score": candidate.damage_score,
        "dollar_volume": candidate.dollar_volume,
    }


def qualifies_for_weekly_watch(stage: str) -> bool:
    return (stage or "").strip().upper() in WATCHLIST_STAGES


# --- Path 2: EOD drop intake ---


def fetch_grouped_daily_map(session_date: date) -> dict[str, dict[str, Any]]:
    from polygon import RESTClient

    polygon_api_key = getattr(settings, "POLYGON_API_KEY", None) or os.getenv("POLYGON_API_KEY")
    if not polygon_api_key:
        raise RuntimeError("POLYGON_API_KEY not set in Django settings or environment")

    client = RESTClient(polygon_api_key)
    reference = session_date.isoformat()
    aggs = client.get_grouped_daily_aggs(locale="us", date=reference, adjusted=True)

    out: dict[str, dict[str, Any]] = {}
    for agg in aggs:
        symbol = str(agg.ticker or "").strip().upper()
        if not symbol:
            continue
        close = _safe_float(agg.close)
        if close is None or close <= 0:
            continue
        out[symbol] = {
            "open": _safe_float(agg.open) or close,
            "close": close,
            "volume": _safe_int(agg.volume),
        }
    return out


def _name_suggests_leveraged(info: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(info.get("longName") or ""),
            str(info.get("shortName") or ""),
            str(info.get("symbol") or ""),
        ]
    ).lower()
    padded = f" {text} "
    return any(hint in padded for hint in LEVERAGED_NAME_HINTS)


def is_common_equity(info: dict[str, Any]) -> bool:
    quote_type = (info.get("quoteType") or "").strip().upper()
    if quote_type in NON_EQUITY_QUOTE_TYPES:
        return False
    if quote_type and quote_type != "EQUITY":
        return False
    if _name_suggests_leveraged(info):
        return False
    return True


def filter_common_equity_symbols(symbols: Sequence[str]) -> set[str]:
    allowed: set[str] = set()
    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).info or {}
        except Exception:
            continue
        if is_common_equity(info):
            allowed.add(symbol.upper())
    return allowed


def build_eod_drop_candidates(
    session_date: date,
    *,
    min_price: float = DEFAULT_MIN_PRICE,
    min_volume: int = DEFAULT_MIN_VOLUME,
    min_dollar_volume: float = DEFAULT_MIN_DOLLAR_VOLUME,
    min_day_drop_pct: float = EOD_MIN_DAY_DROP_PCT,
    include_etfs: bool = False,
    equities_only: bool = True,
    top: int = EOD_TOP,
) -> tuple[list[EodDropCandidate], int]:
    prior_date = prior_trading_day(session_date)
    session_map = fetch_grouped_daily_map(session_date)
    prior_map = fetch_grouped_daily_map(prior_date)
    if not session_map:
        raise RuntimeError(f"No Polygon grouped daily rows for {session_date}")
    if not prior_map:
        raise RuntimeError(f"No Polygon grouped daily rows for prior session {prior_date}")

    rows: list[EodDropCandidate] = []
    for symbol, today in session_map.items():
        if not include_etfs and symbol in ETF_EXCLUDE_TICKERS:
            continue
        if "." in symbol or "-" in symbol or "/" in symbol:
            continue

        close = today["close"]
        prior_close = prior_map.get(symbol, {}).get("close")
        if prior_close is None or prior_close <= 0:
            continue

        volume = today["volume"]
        if close < min_price or volume < min_volume:
            continue

        dollar_volume = close * volume
        if dollar_volume < min_dollar_volume:
            continue

        day_change_pct = (close / prior_close - 1.0) * 100.0
        if day_change_pct > -min_day_drop_pct:
            continue

        open_px = today["open"]
        session_change_pct = (close / open_px - 1.0) * 100.0 if open_px and open_px > 0 else day_change_pct

        rows.append(
            EodDropCandidate(
                rank=0,
                symbol=symbol,
                session_date=session_date.isoformat(),
                close=close,
                prior_close=prior_close,
                open=open_px,
                volume=volume,
                dollar_volume=dollar_volume,
                day_change_pct=day_change_pct,
                session_change_pct=session_change_pct,
            )
        )

    raw_count = len(rows)
    if equities_only and not include_etfs and rows:
        equity_symbols = filter_common_equity_symbols([r.symbol for r in rows])
        rows = [r for r in rows if r.symbol in equity_symbols]

    rows.sort(key=lambda r: r.day_change_pct)
    ranked = [
        EodDropCandidate(**{**asdict(row), "rank": idx})
        for idx, row in enumerate(rows[:top], start=1)
    ]
    return ranked, raw_count


def build_llm_context_block(candidates: Sequence[EodDropCandidate]) -> str:
    lines = [
        f"Session date: {candidates[0].session_date}" if candidates else "Session date: unknown",
        "Tickers (large single-day drops; evaluate for Vulture damaged-quality recovery watchlist intake):",
    ]
    for row in candidates:
        lines.append(
            f"  {row.symbol}: close=${row.close:.2f} (prior ${row.prior_close:.2f}); "
            f"day={row.day_change_pct:+.1f}%; session(open-close)={row.session_change_pct:+.1f}%; "
            f"volume={row.volume:,}; dollar_volume=${row.dollar_volume:,.0f}"
        )
    return "\n".join(lines)


def build_vulture_drop_prompt(context_block: str) -> str:
    return f"""
You are a Vulture recovery watchlist triage assistant.

Large single-session price drops occurred for the tickers below. Your job is NOT to
recommend buying. Decide whether each name merits Vulture MONITORING (damaged but
potentially recoverable quality) vs EXCLUDE (terminal/dilution/no edge) vs DEFER (too
fresh or unclear — recheck in a few days).

Source quality policy:
- Primary sources (highest trust): Reuters, Bloomberg, Dow Jones Newswires, SEC filings
- Secondary: Benzinga, company press releases
- If credible recent evidence is missing, choose DEFER with lower confidence.

MONITOR (add to watchlist for structural recovery scoring later):
- Company-specific shock with plausible overreaction (earnings miss on intact franchise,
  guidance reset, isolated legal headline) where business survival does not appear impaired.
- Drop plausibly explained by news; recovery thesis is conceivable.

EXCLUDE (do not watch):
- Bankruptcy, going concern, fraud/accounting, clinical/regulatory terminal failure,
  massive dilutive offering, delisting risk, or thesis clearly broken.

DEFER:
- No clear catalyst found, macro sympathy only, or event too fresh to judge.

For each ticker return:
- verdict: "monitor" | "exclude" | "defer"
- damage_type: "overreaction" | "fundamental" | "terminal" | "unclear"
- confidence: 0.0–1.0
- reason: one concise sentence (max 30 words)
- sources_used: array of source names relied on (empty if none)

Rules:
- No prose outside JSON.
- When uncertain, default to defer with lower confidence.
- Do not predict future stock prices.

Context:
{context_block}

Return ONLY a single JSON object:
{{
  "TICKER": {{
    "verdict": "monitor|exclude|defer",
    "damage_type": "overreaction|fundamental|terminal|unclear",
    "confidence": 0.00,
    "reason": "short reason",
    "sources_used": ["Reuters"]
  }}
}}
"""


def _apply_triage_result(row: EodDropCandidate, data: dict[str, Any]) -> EodDropCandidate:
    verdict = str(data.get("verdict") or "").strip().lower()
    damage_type = str(data.get("damage_type") or "").strip().lower()
    reason = str(data.get("reason") or "").strip()
    try:
        confidence = float(data.get("confidence")) if data.get("confidence") is not None else None
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))
    return EodDropCandidate(
        rank=row.rank,
        symbol=row.symbol,
        session_date=row.session_date,
        close=row.close,
        prior_close=row.prior_close,
        open=row.open,
        volume=row.volume,
        dollar_volume=row.dollar_volume,
        day_change_pct=row.day_change_pct,
        session_change_pct=row.session_change_pct,
        llm_verdict=verdict,
        llm_damage_type=damage_type,
        llm_reason=reason,
        llm_confidence=confidence,
    )


def _lookup_triage_result(results: dict[str, Any], symbol: str) -> Optional[dict[str, Any]]:
    data = (
        results.get(symbol)
        or results.get(symbol.upper())
        or results.get(symbol.lower())
    )
    return data if isinstance(data, dict) else None


def triage_candidates(
    candidates: Sequence[EodDropCandidate],
    *,
    advisor_name: str = "vulture",
    backend: str = "router",
    use_search: bool = True,
    timeout: float = 180.0,
) -> tuple[Optional[str], list[EodDropCandidate]]:
    if not candidates:
        return None, []

    prompt = build_vulture_drop_prompt(build_llm_context_block(candidates))
    if backend == "gemini":
        from core.services.llm.gemini import ask_gemini

        model, results, _, _ = ask_gemini(
            prompt=prompt,
            advisor_name=advisor_name,
            gemini_model_index=0,
            gemini_key_index=0,
            timeout=timeout,
            use_search=use_search,
        )
    elif backend == "deepseek":
        from core.services.llm.deepseek import ask_deepseek

        model, results = ask_deepseek(prompt=prompt, advisor_name=advisor_name, timeout=timeout)
    else:
        from core.services.llm.router import ask_llm

        model, results, _, _ = ask_llm(
            prompt=prompt,
            advisor_name=advisor_name,
            gemini_model_index=0,
            gemini_key_index=0,
            timeout=timeout,
            use_search=use_search,
        )

    if not results or not isinstance(results, dict):
        logger.warning("%s EOD LLM triage: no usable JSON response", advisor_name)
        return model, list(candidates)

    updated: list[EodDropCandidate] = []
    for row in candidates:
        data = _lookup_triage_result(results, row.symbol)
        if not data:
            updated.append(row)
            continue
        updated.append(_apply_triage_result(row, data))
    return model, updated


def merge_triage_rows(
    all_rows: list[EodDropCandidate],
    triaged: Sequence[EodDropCandidate],
) -> list[EodDropCandidate]:
    by_symbol = {row.symbol: row for row in triaged}
    return [by_symbol.get(row.symbol, row) for row in all_rows]


def run_llm_triage(
    rows: list[EodDropCandidate],
    *,
    advisor_name: str = "vulture",
    backend: str = "router",
    rank_from: Optional[int] = None,
    rank_to: Optional[int] = None,
    batch_size: int = EOD_LLM_BATCH_SIZE,
    use_search: bool = True,
    quiet: bool = False,
) -> list[EodDropCandidate]:
    if not rows:
        return rows

    lo = rank_from if rank_from is not None else 1
    hi = rank_to if rank_to is not None else max(r.rank for r in rows)
    targets = [r for r in rows if lo <= r.rank <= hi]
    if not targets:
        if not quiet:
            logger.info("EOD LLM triage: no rows in rank range %s-%s", lo, hi)
        return rows

    merged = list(rows)
    for batch_idx, batch in enumerate(_chunks(targets, batch_size), start=1):
        if not quiet:
            logger.info(
                "EOD LLM triage batch %s: ranks %s-%s (%s)",
                batch_idx,
                batch[0].rank,
                batch[-1].rank,
                ", ".join(r.symbol for r in batch),
            )
        model, triaged = triage_candidates(
            batch,
            advisor_name=advisor_name,
            backend=backend,
            use_search=use_search,
        )
        if not quiet:
            logger.info("EOD LLM triage batch %s model: %s", batch_idx, model or "unknown")
        merged = merge_triage_rows(merged, triaged)
    return merged


def filter_monitor_candidates(
    rows: Sequence[EodDropCandidate],
    *,
    min_confidence: float = MIN_LLM_MONITOR_CONFIDENCE,
) -> list[EodDropCandidate]:
    out: list[EodDropCandidate] = []
    for row in rows:
        if row.llm_verdict != "monitor":
            continue
        if row.llm_confidence is not None and row.llm_confidence < min_confidence:
            continue
        out.append(row)
    return out


def _eod_triage_verdict_counts(rows: Sequence[EodDropCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = (row.llm_verdict or "").strip().lower() or "empty"
        counts[key] = counts.get(key, 0) + 1
    return counts


def eod_candidate_to_meta(candidate: EodDropCandidate) -> dict[str, Any]:
    return {
        "source": "eod_drop",
        "session_date": candidate.session_date,
        "day_change_pct": round(candidate.day_change_pct, 2),
        "session_change_pct": round(candidate.session_change_pct, 2),
        "close": candidate.close,
        "llm_verdict": candidate.llm_verdict,
        "llm_damage_type": candidate.llm_damage_type,
        "llm_reason": candidate.llm_reason,
        "llm_confidence": candidate.llm_confidence,
    }


# Lab script aliases
build_drop_candidates = build_eod_drop_candidates
filter_monitor_rows = filter_monitor_candidates
build_candidates = build_scan_candidates


# --- Advisor ---


class Vulture(AdvisorBase):
    """EOD drop intake, diagnostic watchlist, persistence-gated recovery discoveries."""

    sell_instructions = [
        ("TARGET_PERCENTAGE", VULTURE_TARGET_MULT, None),
        ("STOP_PERCENTAGE", VULTURE_STOP_MULT, None),
        ("AFTER_DAYS", VULTURE_MAX_HOLD_DAYS, None),
    ]

    def discover(self, sa) -> None:
        session_date = resolve_eod_session_date()
        target_date = session_date.isoformat()

        if not self.should_process_market_date_once(
            target_date=target_date,
            cutoff_hour_utc=PROCESS_CUTOFF_HOUR_UTC,
        ):
            return

        state = self._advisor_blob_state()
        streaks: Dict[str, int] = dict(state.get("buy_ready_streak") or {})

        financial_polygon.clear_polygon_cache()

        weekly_added = 0
        eod_added = 0
        rescored = 0
        discoveries = 0

        if self._needs_weekly_scan(state, target_date):
            try:
                weekly_added = self._weekly_scan_intake(target_date, session_date)
                state["last_universe_scan_date"] = target_date
            except Exception as exc:
                logger.exception("Vulture sa=%s: weekly scan intake failed: %s", sa.id, exc)

        try:
            eod_added = self._eod_intake(target_date, session_date)
        except Exception as exc:
            logger.exception("Vulture sa=%s: EOD intake failed: %s", sa.id, exc)

        rescored, discoveries, streaks = self._rescore_watchlist(sa, target_date, streaks)

        state["buy_ready_streak"] = streaks
        state["last_eod_session"] = target_date
        state["last_weekly_watches_added"] = weekly_added
        state["last_eod_watches_added"] = eod_added
        state["last_watches_added"] = weekly_added + eod_added
        state["last_rescored"] = rescored
        state["last_discoveries"] = discoveries
        self._save_advisor_blob_state(state)
        self.mark_market_date_processed(target_date)

        logger.info(
            "Vulture sa=%s session=%s weekly_added=%s eod_added=%s rescored=%s discoveries=%s",
            sa.id,
            target_date,
            weekly_added,
            eod_added,
            rescored,
            discoveries,
        )

    def _needs_weekly_scan(self, state: Dict, target_date: str) -> bool:
        built = (state.get("last_universe_scan_date") or "").strip()
        if not built:
            return True
        try:
            built_dt = datetime.strptime(built, "%Y-%m-%d").date()
            ref_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            return True
        return (ref_dt - built_dt).days >= SCAN_REBUILD_DAYS

    def _weekly_scan_intake(self, target_date: str, session_date) -> int:
        candidates, stats = build_weekly_scan_candidates(
            session_date,
            seed_universe=SCAN_SEED_UNIVERSE,
            top=SCAN_TOP,
        )
        logger.info(
            "Vulture weekly scan %s: polygon=%s seeds=%s candidates=%s",
            target_date,
            stats.get("polygon_rows"),
            stats.get("seeds"),
            stats.get("candidates"),
        )

        added = 0
        for row in candidates:
            if self.watched(row.symbol):
                logger.debug("Vulture skip weekly watch %s: already on watchlist", row.symbol)
                continue

            try:
                diag = analyze_symbol(row.symbol)
            except Exception as exc:
                logger.warning("Vulture weekly diagnostic failed for %s: %s", row.symbol, exc)
                continue

            if not qualifies_for_weekly_watch(diag.status):
                logger.debug(
                    "Vulture skip weekly watch %s: stage=%s decision=%s",
                    row.symbol,
                    diag.status,
                    diag.decision,
                )
                continue

            dmg = row.damage_52w_pct if row.damage_52w_pct is not None else row.damage_3m_pct
            dmg_str = f"{dmg:+.1f}" if dmg is not None else "n/a"
            explanation = (
                f"Vulture weekly scan {row.trigger} | "
                f"dmg {dmg_str}% | stage {diag.status} | "
                f"quality {diag.candidate_quality_pct}%"
            )[:500]
            meta = scan_candidate_to_meta(row)
            meta["intake_date"] = target_date
            meta["diagnostic"] = diagnostic_to_dict(diag)
            meta["recovery_stage"] = diag.status
            meta["decision"] = diag.decision
            if self.watch(row.symbol, explanation, days=WATCHLIST_DAYS, meta=meta):
                added += 1

        logger.info(
            "Vulture weekly intake %s: added=%s of %s candidates",
            target_date,
            added,
            len(candidates),
        )
        return added

    def _eod_intake(self, target_date: str, session_date) -> int:
        rows, raw_count = build_eod_drop_candidates(
            session_date,
            min_price=DEFAULT_MIN_PRICE,
            min_volume=DEFAULT_MIN_VOLUME,
            min_dollar_volume=DEFAULT_MIN_DOLLAR_VOLUME,
            min_day_drop_pct=EOD_MIN_DAY_DROP_PCT,
            top=EOD_TOP,
        )
        logger.info(
            "Vulture EOD scan %s: raw_drops=%s ranked=%s",
            target_date,
            raw_count,
            len(rows),
        )

        rows = run_llm_triage(
            rows,
            advisor_name="vulture",
            backend="router",
            batch_size=EOD_LLM_BATCH_SIZE,
            quiet=True,
        )
        monitor_rows = filter_monitor_candidates(rows)
        logger.info(
            "Vulture EOD LLM %s: monitor=%s of %s ranked",
            target_date,
            len(monitor_rows),
            len(rows),
        )
        if not monitor_rows and rows:
            logger.warning(
                "VULTURE_EOD_NO_MONITORS session=%s ranked=%s verdicts=%s "
                "(EOD triage needs Gemini with search; rate limits or DeepSeek fallback often defer all)",
                target_date,
                len(rows),
                _eod_triage_verdict_counts(rows),
            )

        added = 0
        for row in monitor_rows:
            if self.watched(row.symbol):
                logger.debug("Vulture skip watch %s: already on watchlist", row.symbol)
                continue
            explanation = (
                f"Vulture EOD drop {row.day_change_pct:+.1f}% | "
                f"{row.llm_damage_type or 'n/a'} | {row.llm_reason or 'monitor'}"
            )[:500]
            meta = eod_candidate_to_meta(row)
            meta["intake_date"] = target_date
            if self.watch(row.symbol, explanation, days=WATCHLIST_DAYS, meta=meta):
                added += 1
        return added

    def _rescore_watchlist(
        self,
        sa,
        target_date: str,
        streaks: Dict[str, int],
    ) -> Tuple[int, int, Dict[str, int]]:
        rescored = 0
        discoveries = 0

        for entry in self.watchlist():
            symbol = entry.stock.symbol.upper()
            try:
                diag = analyze_symbol(symbol)
            except Exception as exc:
                logger.warning("Vulture rescore failed for %s: %s", symbol, exc)
                continue

            meta = dict(entry.meta or {})
            meta["diagnostic"] = diagnostic_to_dict(diag)
            meta["last_scored"] = target_date
            meta["recovery_stage"] = diag.status
            meta["decision"] = diag.decision
            entry.meta = meta
            entry.save(update_fields=["meta"])
            rescored += 1

            if diag.decision == "BUY READY":
                streaks[symbol] = streaks.get(symbol, 0) + 1
                logger.info(
                    "Vulture %s BUY READY streak %s/%s (score %s/%s)",
                    symbol,
                    streaks[symbol],
                    BUY_READY_STREAK_DAYS,
                    diag.score,
                    diag.max_score,
                )
            else:
                if streaks.get(symbol):
                    logger.info(
                        "Vulture %s streak reset (decision=%s stage=%s)",
                        symbol,
                        diag.decision,
                        diag.status,
                    )
                streaks[symbol] = 0
                continue

            if streaks[symbol] < BUY_READY_STREAK_DAYS:
                continue
            if not self.allow_discovery(symbol, period=DISCOVERY_COOLDOWN_HOURS):
                continue

            explanation = self._discovery_explanation(symbol, diag, streaks[symbol])
            if self.discovered(
                sa,
                symbol,
                explanation,
                sell_instructions=list(self.sell_instructions),
                weight=DISCOVERY_WEIGHT,
            ):
                discoveries += 1
                streaks[symbol] = 0

        return rescored, discoveries, streaks

    def _discovery_explanation(self, symbol: str, diag, streak: int) -> str:
        return (
            f"Vulture recovery BUY READY x{streak}d | {symbol} "
            f"score {diag.score}/{diag.max_score} "
            f"quality {diag.candidate_quality_pct}% "
            f"recovery {diag.recovery_confidence_pct}% | "
            f"{diag.decision_reason[:120]}"
        )[:500]

    def analyze(self, sa, stock) -> None:
        return


register(name="Vulture", python_class="Vulture")
