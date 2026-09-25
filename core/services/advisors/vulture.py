"""
Vulture advisor v2 — cliff-event bounce monitors.

Flow:
  1. EOD cliff intake (once/session after Polygon prior-day bars are available):
     large single-day drop → LLM triage → dismiss structural/long-term damage;
     else watch with pre/post cliff prices (14 calendar days).
  2. RTH monitor (10:30–16:00 ET): discover when opens show early strength,
     price is still below the pre-cliff print, and analysts remain constructive.

Weekly chronic-damage helpers remain for lab CLIs only (not production discover).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd
import pytz
import yfinance as yf

from core.services.advisors.advisor import AdvisorBase, register
from core.services.financial import polygon as financial_polygon
from core.services.financial.polygon import fetch_grouped_daily_map
from core.services.health.consensus import score_consensus_health
from core.services.market import (
    last_completed_trading_day,
    market_open,
    prior_trading_day,
    resolve_eod_session_date,
)

logger = logging.getLogger(__name__)

# --- Advisor runtime (v2) ---
VULTURE_VERSION = 2
# EOD cliff intake after ~7 AM ET (Polygon prior-session grouped daily).
EOD_INTAKE_CUTOFF_HOUR_UTC = 11
WATCHLIST_DAYS = 14
MONITOR_START_MINUTES_AFTER_OPEN = 60  # 10:30 ET
DISCOVERY_COOLDOWN_HOURS = 72
DISCOVERY_WEIGHT = 1.0

# Still below the pre-cliff print / not fully recovered from the event drop.
MAX_PRICE_FRAC_OF_PRE_CLIFF = 0.95  # at least ~5% below pre-cliff close
MAX_CLIFF_DROP_RECOVERY_FRAC = 0.50  # recovered at most half of (pre - cliff)

# Analyst belief hard gates
MIN_CONSENSUS_UPSIDE_PCT = 15.0
BUY_RECOMMENDATION_KEYS = frozenset({"buy", "strong_buy"})

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

# --- Lab-only: weekly chronic damage scan ---
SCAN_REBUILD_DAYS = 7
SCAN_TOP = 50
SCAN_SEED_UNIVERSE = 1200
SCAN_BATCH_SIZE = 100
SCAN_MIN_3M_DAMAGE_PCT = 20.0
SCAN_MIN_52W_DAMAGE_PCT = 30.0
SCAN_MAX_52W_DAMAGE_PCT = 70.0
WATCHLIST_STAGES = frozenset({"WATCH", "WARM", "RECOVERY"})

# --- EOD cliff intake ---
EOD_TOP = 25
EOD_MIN_DAY_DROP_PCT = 7.0
EOD_LLM_BATCH_SIZE = 10
MIN_LLM_MONITOR_CONFIDENCE = 0.5
MONITOR_DAMAGE_TYPES = frozenset({"overreaction"})

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
    max_drop_date: Optional[str] = None
    max_drop_pct: Optional[float] = None


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


def _largest_session_drop(hist: pd.DataFrame, lookback: int = 90) -> tuple[Optional[str], Optional[float]]:
    """Worst single-session close-to-close return in the last `lookback` sessions."""
    if hist.empty or "close" not in hist.columns:
        return None, None
    closes = hist["close"].dropna().astype(float)
    if len(closes) < 2:
        return None, None
    window = closes.tail(lookback + 1)
    if len(window) < 2:
        return None, None
    rets = window.pct_change().dropna() * 100.0
    if rets.empty:
        return None, None
    idx = rets.idxmin()
    if hasattr(idx, "strftime"):
        drop_date = idx.strftime("%Y-%m-%d")
    elif hasattr(idx, "date"):
        drop_date = idx.date().isoformat()
    else:
        drop_date = str(idx)[:10]
    return drop_date, float(rets.min())


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
        max_drop_date, max_drop_pct = _largest_session_drop(hist)
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
                max_drop_date=max_drop_date,
                max_drop_pct=max_drop_pct,
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


def _weekly_collapse_reason(candidate: VultureScanCandidate) -> str:
    dmg = candidate.damage_52w_pct if candidate.damage_52w_pct is not None else candidate.damage_3m_pct
    dmg_str = f"{dmg:+.1f}" if dmg is not None else "n/a"
    if candidate.max_drop_date and candidate.max_drop_pct is not None:
        return (
            f"{candidate.max_drop_pct:+.1f}% session drop; "
            f"chronic {dmg_str}% from {candidate.trigger} highs"
        )
    return f"Chronic damage {dmg_str}% from {candidate.trigger} highs"


def _eod_collapse_reason(candidate: EodDropCandidate) -> str:
    if candidate.llm_reason:
        return candidate.llm_reason[:200]
    parts: list[str] = []
    if candidate.llm_damage_type:
        parts.append(candidate.llm_damage_type)
    parts.append(f"EOD drop {candidate.day_change_pct:+.1f}%")
    return " — ".join(parts)


def _collapse_context_from_meta(meta: dict[str, Any]) -> tuple[str, str]:
    meta = meta or {}
    raw_date = (
        meta.get("collapse_date")
        or meta.get("session_date")
        or meta.get("max_drop_date")
        or ""
    )
    date = str(raw_date).strip()[:10]

    reason = (meta.get("collapse_reason") or meta.get("llm_reason") or "").strip()
    if reason:
        return date, reason

    intake = meta.get("intake") or meta.get("source") or ""
    if intake == "weekly_scan":
        trigger = meta.get("trigger") or "?"
        dmg = meta.get("damage_52w_pct")
        if dmg is None:
            dmg = meta.get("damage_3m_pct")
        max_pct = meta.get("max_drop_pct")
        max_date = str(meta.get("max_drop_date") or "").strip()[:10]
        if max_pct is not None and dmg is not None and max_date:
            reason = f"{max_pct:+.1f}% on {max_date}; chronic {dmg:+.1f}% from {trigger} highs"
        elif dmg is not None:
            reason = f"Chronic damage {dmg:+.1f}% from {trigger} highs"
    elif intake == "eod_drop" or meta.get("source") == "eod_drop":
        dmg_type = meta.get("llm_damage_type") or "EOD drop"
        day_pct = meta.get("day_change_pct")
        if day_pct is not None:
            reason = f"{dmg_type} ({day_pct:+.1f}%)"
        else:
            reason = dmg_type
    return date, reason


def _format_collapse_clause(meta: dict[str, Any]) -> str:
    date, reason = _collapse_context_from_meta(meta)
    if not date and not reason:
        return ""
    parts: list[str] = []
    if date:
        parts.append(f"collapse {date}")
    if reason:
        parts.append(reason[:120])
    return " | ".join(parts)


def scan_candidate_to_meta(candidate: VultureScanCandidate) -> dict[str, Any]:
    return {
        "intake": "weekly_scan",
        "rank": candidate.rank,
        "trigger": candidate.trigger,
        "damage_3m_pct": candidate.damage_3m_pct,
        "damage_52w_pct": candidate.damage_52w_pct,
        "damage_score": candidate.damage_score,
        "dollar_volume": candidate.dollar_volume,
        "max_drop_date": candidate.max_drop_date,
        "max_drop_pct": candidate.max_drop_pct,
        "collapse_date": candidate.max_drop_date,
        "collapse_reason": _weekly_collapse_reason(candidate),
    }


def qualifies_for_weekly_watch(stage: str) -> bool:
    return (stage or "").strip().upper() in WATCHLIST_STAGES


# --- Path 2: EOD drop intake ---


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
    session_map, resolved_session = fetch_grouped_daily_map(session_date)
    prior_map, _prior_resolved = fetch_grouped_daily_map(prior_date)
    if resolved_session != session_date:
        logger.warning(
            "Vulture EOD scan requested %s; Polygon resolved session %s",
            session_date.isoformat(),
            resolved_session.isoformat(),
        )
    if not session_map:
        raise RuntimeError(f"No Polygon grouped daily rows for {resolved_session}")
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
                session_date=resolved_session.isoformat(),
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
You are a Vulture cliff-event triage assistant.

Large single-session price drops ("cliffs") occurred for the tickers below. Your job
is NOT to recommend buying. Decide whether each name merits SHORT-TERM MONITORING
for a bounce of an overreaction, vs EXCLUDE (structural / long-term damage), vs DEFER
(too unclear — do not monitor yet).

Source quality policy:
- Primary sources (highest trust): Reuters, Bloomberg, Dow Jones Newswires, SEC filings
- Secondary: Benzinga, company press releases
- If credible recent evidence is missing, choose DEFER with lower confidence.

MONITOR (temporary bounce watch — overreaction only):
- Plausible overreaction where the franchise looks intact: earnings miss on solid business,
  guidance reset without survival risk, sector/sympathy flush, temporary headline shock.
- damage_type MUST be "overreaction".

EXCLUDE (dismiss — do not watch):
- Bankruptcy, going concern, fraud/accounting, clinical/regulatory terminal failure,
  massive dilutive offering, delisting risk.
- Structural / multi-year regime risk: lasting liability, regulatory, or legislative
  changes that raise ongoing risk (e.g. utility wildfire-liability shield blocked).
- Clear fundamental thesis break (not a one-day overreaction).
- Use damage_type "fundamental" or "terminal" with verdict exclude.

DEFER:
- No clear catalyst found, or event too fresh to judge.
- damage_type "unclear" when appropriate.

For each ticker return:
- verdict: "monitor" | "exclude" | "defer"
- damage_type: "overreaction" | "fundamental" | "terminal" | "unclear"
- confidence: 0.0–1.0
- reason: one concise sentence (max 30 words)
- sources_used: array of source names relied on (empty if none)

Rules:
- No prose outside JSON.
- When uncertain, default to defer with lower confidence.
- Do not MONITOR fundamental or terminal damage.
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
    """Only overreaction MONITOR verdicts enter the cliff watchlist."""
    out: list[EodDropCandidate] = []
    for row in rows:
        if row.llm_verdict != "monitor":
            continue
        if (row.llm_damage_type or "").strip().lower() not in MONITOR_DAMAGE_TYPES:
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
    """Cliff-monitor meta: pre/post event prices are the discover ceiling anchors."""
    return {
        "vulture_version": VULTURE_VERSION,
        "intake": "cliff_v2",
        "source": "eod_drop",
        "cliff_date": candidate.session_date,
        "session_date": candidate.session_date,
        "pre_cliff_close": round(candidate.prior_close, 4),
        "cliff_close": round(candidate.close, 4),
        "cliff_open": round(candidate.open, 4),
        "day_change_pct": round(candidate.day_change_pct, 2),
        "session_change_pct": round(candidate.session_change_pct, 2),
        "close": candidate.close,
        "llm_verdict": candidate.llm_verdict,
        "llm_damage_type": candidate.llm_damage_type,
        "llm_reason": candidate.llm_reason,
        "llm_confidence": candidate.llm_confidence,
        "collapse_date": candidate.session_date,
        "collapse_reason": _eod_collapse_reason(candidate),
    }


# Lab script aliases
build_drop_candidates = build_eod_drop_candidates
filter_monitor_rows = filter_monitor_candidates
build_candidates = build_scan_candidates

# Backward-compatible name used by older call sites / docs.
PROCESS_CUTOFF_HOUR_UTC = EOD_INTAKE_CUTOFF_HOUR_UTC


# --- Cliff monitor helpers ---


def in_cliff_monitor_window(now: Optional[datetime] = None) -> bool:
    """True on a trading day from 10:30 ET through the regular-session close."""
    status = market_open()
    if status is None or status < 0:
        return False
    return status >= MONITOR_START_MINUTES_AFTER_OPEN


def _parse_iso_date(raw: Any) -> Optional[date]:
    text = str(raw or "").strip()[:10]
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def cliff_monitor_expired(meta: dict[str, Any], *, today: Optional[date] = None) -> bool:
    """True when cliff_date is older than WATCHLIST_DAYS calendar days."""
    cliff = _parse_iso_date(meta.get("cliff_date") or meta.get("session_date"))
    if cliff is None:
        return True
    ref = today or datetime.now(pytz.timezone("US/Eastern")).date()
    return (ref - cliff).days > WATCHLIST_DAYS


def _fetch_open_series(symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Return (open_today, open_prior, last_price).

    Today's open prefers live Yahoo regularMarketOpen; prior open is the previous
    completed daily bar.
    """
    sym = (symbol or "").strip().upper()
    open_today: Optional[float] = None
    last_price: Optional[float] = None
    try:
        info = yf.Ticker(sym).info or {}
        open_today = _safe_float(info.get("regularMarketOpen") or info.get("open"))
        last_price = _safe_float(
            info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
        )
    except Exception:
        pass

    open_prior: Optional[float] = None
    try:
        hist = yf.Ticker(sym).history(period="15d", interval="1d", auto_adjust=False)
        if hist is not None and not hist.empty and "Open" in hist.columns:
            opens = hist["Open"].dropna().astype(float)
            if not opens.empty:
                open_prior = float(opens.iloc[-1])
                # If history already includes today's partial bar, prior is the one before.
                et = pytz.timezone("US/Eastern")
                today = datetime.now(et).date()
                last_idx = opens.index[-1]
                last_day = last_idx.date() if hasattr(last_idx, "date") else None
                if last_day == today and len(opens) >= 2:
                    open_today = open_today or float(opens.iloc[-1])
                    open_prior = float(opens.iloc[-2])
                elif open_today is None and last_day == today:
                    open_today = float(opens.iloc[-1])
                if last_price is None and "Close" in hist.columns:
                    closes = hist["Close"].dropna().astype(float)
                    if not closes.empty:
                        last_price = float(closes.iloc[-1])
    except Exception:
        pass

    return open_today, open_prior, last_price


def two_concurrent_opening_highs(open_today: Optional[float], open_prior: Optional[float]) -> bool:
    """Today's open is higher than the prior session open (rising open pair)."""
    if open_today is None or open_prior is None:
        return False
    if open_today <= 0 or open_prior <= 0:
        return False
    return open_today > open_prior


def still_below_pre_cliff(
    price: Optional[float],
    pre_cliff: Optional[float],
    cliff_close: Optional[float],
) -> Tuple[bool, str]:
    """Price still discounted vs the pre-cliff print / not fully recovered."""
    if price is None or pre_cliff is None or pre_cliff <= 0:
        return False, "missing pre-cliff or price"
    if price > pre_cliff * MAX_PRICE_FRAC_OF_PRE_CLIFF:
        pct = (price / pre_cliff - 1.0) * 100.0
        return False, f"price {pct:+.1f}% vs pre-cliff (need <= {(MAX_PRICE_FRAC_OF_PRE_CLIFF - 1) * 100:.0f}%)"

    if cliff_close is not None and cliff_close > 0 and pre_cliff > cliff_close:
        drop = pre_cliff - cliff_close
        recovered = (price - cliff_close) / drop
        if recovered > MAX_CLIFF_DROP_RECOVERY_FRAC:
            return False, f"recovered {recovered:.0%} of cliff drop (max {MAX_CLIFF_DROP_RECOVERY_FRAC:.0%})"

    return True, "still below pre-cliff"


def consensus_supports_buy(symbol: str) -> Tuple[bool, str]:
    """Hard gate: Buy/Strong Buy and enough mean-target upside."""
    try:
        cons = score_consensus_health(symbol)
    except Exception as exc:
        return False, f"consensus error: {exc}"
    rec = (cons.recommendation_key or "").strip().lower()
    if rec not in BUY_RECOMMENDATION_KEYS:
        return False, f"rec={rec or 'n/a'} (need buy/strong_buy)"
    upside = cons.upside_to_mean_pct
    if upside is None or upside < MIN_CONSENSUS_UPSIDE_PCT:
        return False, f"upside {upside if upside is not None else 'n/a'}% (need >={MIN_CONSENSUS_UPSIDE_PCT:.0f}%)"
    return True, f"rec={rec} upside={upside:+.1f}%"


# --- Advisor ---


class Vulture(AdvisorBase):
    """Cliff-event intake + RTH bounce discovery (v2)."""

    def discover(self, sa) -> None:
        state = self._advisor_blob_state()
        cut = self._ensure_v2_hard_cut(state)

        session_date = resolve_eod_session_date()
        target_date = session_date.isoformat()

        eod_added = 0
        eod_ran = False
        eod_ok = True
        if self.should_process_market_date_once(
            target_date=target_date,
            cutoff_hour_utc=EOD_INTAKE_CUTOFF_HOUR_UTC,
        ):
            eod_ran = True
            financial_polygon.clear_polygon_cache()
            try:
                eod_added = self._cliff_intake(target_date, session_date)
            except Exception as exc:
                eod_ok = False
                logger.exception("Vulture sa=%s: cliff intake failed: %s", sa.id, exc)
            if eod_ok:
                self.mark_market_date_processed(target_date)
            else:
                logger.warning(
                    "Vulture sa=%s: skip mark processed for %s (cliff intake failed; will retry)",
                    sa.id,
                    target_date,
                )

        expired = 0
        discoveries = 0
        evaluated = 0
        if in_cliff_monitor_window():
            expired, evaluated, discoveries = self._monitor_cliffs(sa)
        else:
            logger.debug("Vulture sa=%s: outside 10:30–close ET monitor window", sa.id)

        state["last_eod_session"] = target_date
        state["last_eod_watches_added"] = eod_added
        state["last_watches_added"] = eod_added
        state["last_cliff_expired"] = expired
        state["last_cliff_evaluated"] = evaluated
        state["last_discoveries"] = discoveries
        state["last_v2_hard_cut"] = cut
        # Clear legacy streak blob noise
        state.pop("buy_ready_streak", None)
        self._save_advisor_blob_state(state)

        logger.info(
            "Vulture sa=%s session=%s eod_ran=%s eod_added=%s expired=%s evaluated=%s discoveries=%s hard_cut=%s",
            sa.id,
            target_date,
            eod_ran,
            eod_added,
            expired,
            evaluated,
            discoveries,
            cut,
        )

    def _ensure_v2_hard_cut(self, state: Dict) -> int:
        """One-time delete of all pending watches (legacy chronic + old EOD)."""
        if state.get("vulture_v2_cutover"):
            return 0
        from core.models import Watchlist

        qs = Watchlist.objects.filter(advisor=self.advisor, status="Pending")
        count = qs.count()
        if count:
            qs.delete()
            logger.warning(
                "Vulture v2 hard cut: deleted %s pending watchlist rows for advisor=%s",
                count,
                self.advisor.id,
            )
        state["vulture_v2_cutover"] = True
        state["vulture_v2_cutover_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        state["vulture_v2_cutover_deleted"] = count
        self._save_advisor_blob_state(state)
        return count

    def _has_active_cliff_watch(self, symbol: str) -> bool:
        sym = (symbol or "").strip().upper()
        for entry in self.watchlist():
            if entry.stock.symbol.upper() != sym:
                continue
            meta = entry.meta or {}
            if meta.get("vulture_version") == VULTURE_VERSION and meta.get("intake") == "cliff_v2":
                return True
        return False

    def _cliff_intake(self, target_date: str, session_date) -> int:
        rows, raw_count = build_eod_drop_candidates(
            session_date,
            min_price=DEFAULT_MIN_PRICE,
            min_volume=DEFAULT_MIN_VOLUME,
            min_dollar_volume=DEFAULT_MIN_DOLLAR_VOLUME,
            min_day_drop_pct=EOD_MIN_DAY_DROP_PCT,
            top=EOD_TOP,
        )
        logger.info(
            "Vulture cliff scan %s: raw_drops=%s ranked=%s",
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
            "Vulture cliff LLM %s: monitor=%s of %s ranked (overreaction only)",
            target_date,
            len(monitor_rows),
            len(rows),
        )
        if not monitor_rows and rows:
            logger.warning(
                "VULTURE_CLIFF_NO_MONITORS session=%s ranked=%s verdicts=%s",
                target_date,
                len(rows),
                _eod_triage_verdict_counts(rows),
            )

        added = 0
        for row in monitor_rows:
            if self._has_active_cliff_watch(row.symbol):
                logger.debug("Vulture skip cliff watch %s: already monitoring", row.symbol)
                continue
            meta = eod_candidate_to_meta(row)
            meta["intake_date"] = target_date
            explanation = (
                f"Vulture cliff {row.day_change_pct:+.1f}% on {row.session_date} | "
                f"pre ${row.prior_close:.2f} → ${row.close:.2f} | "
                f"{row.llm_damage_type or 'overreaction'} | {row.llm_reason or 'monitor'}"
            )[:500]
            if self.watch(row.symbol, explanation, days=WATCHLIST_DAYS, meta=meta):
                added += 1
        return added

    def _monitor_cliffs(self, sa) -> Tuple[int, int, int]:
        """Expire stale cliffs; discover when RTH bounce filters all pass."""
        expired = 0
        evaluated = 0
        discoveries = 0
        et_today = datetime.now(pytz.timezone("US/Eastern")).date()

        for entry in list(self.watchlist()):
            symbol = entry.stock.symbol.upper()
            meta = dict(entry.meta or {})

            if meta.get("vulture_version") != VULTURE_VERSION or meta.get("intake") != "cliff_v2":
                entry.status = "Excluded"
                entry.save(update_fields=["status"])
                expired += 1
                continue

            if cliff_monitor_expired(meta, today=et_today):
                entry.status = "Excluded"
                entry.meta = {**meta, "expire_reason": "cliff_age_gt_14d"}
                entry.save(update_fields=["status", "meta"])
                expired += 1
                logger.info("Vulture expire %s: cliff older than %sd", symbol, WATCHLIST_DAYS)
                continue

            evaluated += 1
            ok, reason = self._cliff_buy_ready(symbol, meta)
            meta["last_monitored"] = et_today.isoformat()
            meta["last_monitor_reason"] = reason
            entry.meta = meta
            entry.save(update_fields=["meta"])

            if not ok:
                logger.debug("Vulture %s not ready: %s", symbol, reason)
                continue
            if not self.allow_discovery(symbol, period=DISCOVERY_COOLDOWN_HOURS):
                continue

            explanation = self._cliff_discovery_explanation(symbol, meta, reason)
            if self.discovered(sa, symbol, explanation, weight=DISCOVERY_WEIGHT):
                discoveries += 1
                entry.status = "Executed"
                entry.save(update_fields=["status"])

        return expired, evaluated, discoveries

    def _cliff_buy_ready(self, symbol: str, meta: dict[str, Any]) -> Tuple[bool, str]:
        pre_cliff = _safe_float(meta.get("pre_cliff_close"))
        cliff_close = _safe_float(meta.get("cliff_close"))

        open_today, open_prior, price = _fetch_open_series(symbol)
        if not two_concurrent_opening_highs(open_today, open_prior):
            return False, (
                f"opens not rising (today={open_today}, prior={open_prior})"
            )

        below, below_detail = still_below_pre_cliff(price, pre_cliff, cliff_close)
        if not below:
            return False, below_detail

        cons_ok, cons_detail = consensus_supports_buy(symbol)
        if not cons_ok:
            return False, cons_detail

        return True, (
            f"higher open {open_prior:.2f}->{open_today:.2f}; "
            f"px {price:.2f} vs pre {pre_cliff:.2f}; {cons_detail}"
        )

    def _cliff_discovery_explanation(self, symbol: str, meta: dict[str, Any], reason: str) -> str:
        cliff = meta.get("cliff_date") or "?"
        pre = meta.get("pre_cliff_close")
        post = meta.get("cliff_close")
        dmg = meta.get("day_change_pct")
        llm = (meta.get("llm_reason") or meta.get("collapse_reason") or "")[:80]
        pre_s = f"${pre:.2f}" if isinstance(pre, (int, float)) else "?"
        post_s = f"${post:.2f}" if isinstance(post, (int, float)) else "?"
        dmg_s = f"{dmg:+.1f}%" if isinstance(dmg, (int, float)) else "?"
        return (
            f"Vulture cliff bounce | {symbol} cliff {cliff} {dmg_s} "
            f"({pre_s}→{post_s}) | {reason} | {llm}"
        )[:500]

    def analyze(self, sa, stock) -> None:
        return


register(name="Vulture", python_class="Vulture")
