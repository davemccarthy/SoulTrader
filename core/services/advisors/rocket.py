"""
Rocket advisor — opening-high tape (gaps Edgar never sees).

Stage A (+30m): watch CS/ADR names that gapped ~7.5–15% with real prior-day dvol.
Stage B (+45m): Gemini+search on the top 5 gaps — event reason + significance.
Stage C (+60m): discover those top 5. Explanation = LLM response + tape fields.

News LLM scores the event. Tape (gap, vs-open, prior-day, 52w) is separate.
Default SIs stay AdvisorBase (PEAKED / gated PERCENTAGE_REBUY / DESCENDING_TREND).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

import pytz
import yfinance as yf

from core.services.advisors.advisor import AdvisorBase, discovery_trade_explanation_lead, register
from core.services.financial import polygon as financial_polygon
from core.services.market.session import prior_trading_day

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")

WATCH_MIN_MINUTES = 30  # 10:00 ET
WATCH_MAX_MINUTES = 74  # through the 10:30 SA; 10:45 is +75
WATCH_DAYS = 1
LLM_MIN_MINUTES = 45  # 10:15 ET
DISCOVER_MIN_MINUTES = 60  # 10:30 ET
TOP_GAPS = 5
ROCKET_DISCOVERY_COOLDOWN_HOURS = 24
LLM_TIMEOUT_S = 180.0

GAP_MIN_PCT = 7.5
GAP_MAX_PCT = 15.0
MIN_PREV_CLOSE = 20.0
MIN_PRIOR_DVOL_M = 20.0
EQUITY_TYPES = frozenset({"CS", "ADRC", "ADR", "ADS"})
TYPE_FETCH_SLEEP_S = 0.25
CATALYST_TYPES = frozenset(
    {
        "earnings",
        "guidance",
        "analyst",
        "mna",
        "contract",
        "product",
        "legal",
        "sector",
        "macro",
        "continuation",
        "unexplained",
    }
)

ROCKET_LLM_PROMPT_TEMPLATE = """You are a buy-side equity analyst explaining why this stock gapped up at the regular-session open.

INPUT:
{{
  "ticker": "{ticker}",
  "company": "{company}",
  "session": "{session}",
  "gap_pct": {gap_pct}
}}

gap_pct is only so you know which open to explain. Do not use it to set significance.

Search the web for overnight / premarket / this-morning news and reputable coverage of THIS session's gap. Prefer Bloomberg, Reuters, WSJ, FT, CNBC, company filings/press, major brokers. Ignore older stories unless they clearly drove this open.

TASKS (in this order):
1. Name the most likely catalyst for the gap.
2. Judge the SIGNIFICANCE of that event for this company's equity value. Score the event, not the move.
3. Write a short summary suitable for a trade blotter.

SIGNIFICANCE LADDER (conservative; lean lower when uncertain):
5 = Company-defining, surprise event that rewrites the equity story. Canonical: unexpected pharma approval or pivotal trial success. Same bucket: takeout or another surprise binary that changes what the company is.
4 = Genuinely surprising, exceptional earnings result — strong beat AND meaningful guidance raise, with evidence the market underestimated the business. A token $0.0x tweak on a full-year number is not a meaningful raise. Backlog / RPO / qualitative color can support a 4; they do not create one on their own.
3 = Meaningful but ordinary good news: decent beat, solid results, normal guidance raise, or strong contract.
2 = Soft catalyst: analyst upgrade, sector sympathy, momentum, continuation of yesterday.
3 vs 4: if the raise is small relative to the full-year number, that is a 3 even with a strong EPS beat.
1 = No clear event, rumor-only, or no identifiable company/news catalyst.

Do NOT judge whether the event is already priced in. Do NOT assess entry quality, extension, or whether the stock will fade or hold. Do NOT use prior-day return, distance from highs, gap size, or opening behaviour as evidence for the score. Those are a separate tape layer.

If you cannot find a clear catalyst, insufficient_event_info=true and significance_score <= 2.

Return ONLY JSON:
{{
  "catalyst": "short phrase",
  "catalyst_type": "earnings|guidance|analyst|mna|contract|product|legal|sector|macro|continuation|unexplained",
  "significance_score": 1-5,
  "is_significant": true,
  "significance_reason": "max 2 sentences",
  "summary": "1-2 sentences, blotter English",
  "impact_direction": "bullish|bearish|neutral|not_inferable",
  "confidence": 0.0,
  "insufficient_event_info": false
}}

BOOLEAN: is_significant must be true only if significance_score >= 4.
"""


@dataclass(frozen=True)
class RocketCandidate:
    ticker: str
    prev_close: float
    open: float
    last: float
    gap_pct: float
    vs_open_pct: float
    prior_day_pct: Optional[float]
    near_52w_pct: Optional[float]
    prior_dvol_m: float
    ticker_type: str = ""


def keep_ticker_symbol(ticker: str) -> bool:
    symbol = (ticker or "").strip().upper()
    if not symbol or not symbol.isalpha():
        return False
    if len(symbol) > 5:
        return False
    if len(symbol) == 5 and symbol[-1] in {"W", "R", "U"}:
        return False
    return True


def _pct(numer: float, denom: float) -> Optional[float]:
    if denom <= 0:
        return None
    return (numer - denom) / denom * 100.0


def _bar_ohlcv(bar: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if not bar:
        return None
    if "open" in bar:
        open_px = float(bar["open"])
        high_px = float(bar.get("high") or open_px)
        low_px = float(bar.get("low") or open_px)
        close_px = float(bar["close"])
        volume = float(bar.get("volume") or 0)
    else:
        open_px = float(bar["o"])
        high_px = float(bar.get("h") or open_px)
        low_px = float(bar.get("l") or open_px)
        close_px = float(bar["c"])
        volume = float(bar.get("v") or 0)
    if min(open_px, high_px, low_px, close_px) <= 0:
        return None
    return {
        "open": open_px,
        "high": high_px,
        "low": low_px,
        "close": close_px,
        "volume": volume,
    }


def candidate_from_quote(
    *,
    ticker: str,
    open_px: float,
    last: float,
    prev_close: float,
    prior_dvol_m: float,
    prior_day_pct: Optional[float] = None,
    gap_min_pct: float = GAP_MIN_PCT,
    gap_max_pct: float = GAP_MAX_PCT,
    min_prev_close: float = MIN_PREV_CLOSE,
    min_prior_dvol_m: float = MIN_PRIOR_DVOL_M,
) -> Optional[RocketCandidate]:
    """Apply Stage A price/gap/dvol filters. Type and 52-week are filled later."""
    symbol = (ticker or "").strip().upper()
    if not keep_ticker_symbol(symbol):
        return None
    if prev_close < min_prev_close or open_px <= 0 or last <= 0:
        return None
    if prior_dvol_m < min_prior_dvol_m:
        return None
    gap_pct = _pct(open_px, prev_close)
    if gap_pct is None or gap_pct < gap_min_pct or gap_pct > gap_max_pct:
        return None
    vs_open_pct = _pct(last, open_px)
    if vs_open_pct is None:
        return None
    return RocketCandidate(
        ticker=symbol,
        prev_close=prev_close,
        open=open_px,
        last=last,
        gap_pct=gap_pct,
        vs_open_pct=vs_open_pct,
        prior_day_pct=prior_day_pct,
        near_52w_pct=None,
        prior_dvol_m=prior_dvol_m,
    )


def quotes_from_grouped(
    today: Dict[str, Dict[str, Any]],
    yesterday: Dict[str, Dict[str, Any]],
    day_before: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[RocketCandidate]:
    """Historical / dry-run quotes. Session close stands in for live last."""
    out: List[RocketCandidate] = []
    prior_map = day_before or {}
    for ticker, raw in today.items():
        day = _bar_ohlcv(raw)
        prev = _bar_ohlcv(yesterday.get(ticker) or {})
        if not day or not prev:
            continue
        t2 = _bar_ohlcv(prior_map.get(ticker) or {})
        if t2:
            prior_day_pct = _pct(prev["close"], t2["close"])
        else:
            prior_day_pct = _pct(prev["close"], prev["open"])
        prior_dvol_m = prev["close"] * prev["volume"] / 1_000_000.0
        cand = candidate_from_quote(
            ticker=ticker,
            open_px=day["open"],
            last=day["close"],
            prev_close=prev["close"],
            prior_dvol_m=prior_dvol_m,
            prior_day_pct=prior_day_pct,
        )
        if cand:
            out.append(cand)
    out.sort(key=lambda row: (-row.gap_pct, row.ticker))
    return out


def quotes_from_snapshots(
    snapshots: Iterable[Any],
    day_before: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[RocketCandidate]:
    prior_map = day_before or {}
    out: List[RocketCandidate] = []
    for snap in snapshots:
        ticker = str(getattr(snap, "ticker", "") or "").strip().upper()
        day = getattr(snap, "day", None)
        prev = getattr(snap, "prev_day", None)
        if not ticker or day is None or prev is None:
            continue
        open_px = float(getattr(day, "open", 0) or 0)
        day_close = float(getattr(day, "close", 0) or 0)
        prev_close = float(getattr(prev, "close", 0) or 0)
        prev_open = float(getattr(prev, "open", 0) or 0)
        prev_volume = float(getattr(prev, "volume", 0) or 0)
        last_trade = getattr(snap, "last_trade", None)
        last_px = float(getattr(last_trade, "price", 0) or 0) if last_trade else 0.0
        last = day_close if day_close > 0 else last_px
        t2 = _bar_ohlcv(prior_map.get(ticker) or {})
        if t2:
            prior_day_pct = _pct(prev_close, t2["close"])
        elif prev_open > 0:
            prior_day_pct = _pct(prev_close, prev_open)
        else:
            prior_day_pct = None
        cand = candidate_from_quote(
            ticker=ticker,
            open_px=open_px,
            last=last,
            prev_close=prev_close,
            prior_dvol_m=prev_close * prev_volume / 1_000_000.0,
            prior_day_pct=prior_day_pct,
        )
        if cand:
            out.append(cand)
    out.sort(key=lambda row: (-row.gap_pct, row.ticker))
    return out


def apply_equity_types(
    rows: List[RocketCandidate],
    types: Dict[str, str],
) -> List[RocketCandidate]:
    kept: List[RocketCandidate] = []
    for row in rows:
        kind = str(types.get(row.ticker) or "").upper()
        if kind not in EQUITY_TYPES:
            continue
        kept.append(
            RocketCandidate(
                ticker=row.ticker,
                prev_close=row.prev_close,
                open=row.open,
                last=row.last,
                gap_pct=row.gap_pct,
                vs_open_pct=row.vs_open_pct,
                prior_day_pct=row.prior_day_pct,
                near_52w_pct=row.near_52w_pct,
                prior_dvol_m=row.prior_dvol_m,
                ticker_type=kind,
            )
        )
    return kept


def week52_highs(symbols: List[str]) -> Dict[str, float]:
    tickers = sorted({(s or "").strip().upper() for s in symbols if (s or "").strip()})
    if not tickers:
        return {}
    try:
        data = yf.download(
            " ".join(tickers),
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception as exc:
        logger.warning("Rocket 52-week download failed: %s", exc)
        return {}
    if data is None or getattr(data, "empty", True):
        return {}

    out: Dict[str, float] = {}
    if len(tickers) == 1:
        high_col = data["High"] if "High" in data.columns else None
        if high_col is not None:
            high = float(high_col.max())
            if high > 0:
                out[tickers[0]] = high
        return out

    for symbol in tickers:
        try:
            high = float(data[symbol]["High"].max())
        except Exception:
            continue
        if high > 0:
            out[symbol] = high
    return out


def attach_52w(rows: List[RocketCandidate], highs: Dict[str, float]) -> List[RocketCandidate]:
    out: List[RocketCandidate] = []
    for row in rows:
        high = highs.get(row.ticker)
        near = (row.last / high * 100.0) if high and high > 0 else None
        out.append(
            RocketCandidate(
                ticker=row.ticker,
                prev_close=row.prev_close,
                open=row.open,
                last=row.last,
                gap_pct=row.gap_pct,
                vs_open_pct=row.vs_open_pct,
                prior_day_pct=row.prior_day_pct,
                near_52w_pct=near,
                prior_dvol_m=row.prior_dvol_m,
                ticker_type=row.ticker_type,
            )
        )
    return out


def watch_explanation(row: RocketCandidate) -> str:
    prior = (
        f"prior-day {row.prior_day_pct:+.1f}%"
        if row.prior_day_pct is not None
        else "prior-day n/a"
    )
    week = f"52w {row.near_52w_pct:.0f}%" if row.near_52w_pct is not None else "52w n/a"
    return (
        f"Rocket | gap {row.gap_pct:+.1f}% | vs-open {row.vs_open_pct:+.1f}% | "
        f"{prior} | {week} | prev ${row.prior_dvol_m:.0f}M dvol"
    )


def watch_meta(row: RocketCandidate, session: date) -> Dict[str, Any]:
    return {
        "stage": "watched",
        "session": session.isoformat(),
        "gap_pct": round(row.gap_pct, 3),
        "vs_open_pct": round(row.vs_open_pct, 3),
        "prior_day_pct": None if row.prior_day_pct is None else round(row.prior_day_pct, 3),
        "near_52w_pct": None if row.near_52w_pct is None else round(row.near_52w_pct, 2),
        "prior_dvol_m": round(row.prior_dvol_m, 2),
        "prev_close": round(row.prev_close, 4),
        "open": round(row.open, 4),
        "last": round(row.last, 4),
        "ticker_type": row.ticker_type,
    }


def _clean_segment(value: Any) -> str:
    return " ".join(str(value or "").replace("|", "/").split())


def _fmt_signed_pct(value: Any, digits: int = 1) -> Optional[str]:
    if value is None:
        return None
    try:
        return f"{float(value):+.{digits}f}%"
    except (TypeError, ValueError):
        return None


def tape_explanation_segments(meta: Dict[str, Any]) -> List[str]:
    gap = _fmt_signed_pct(meta.get("gap_pct"))
    vs_open = _fmt_signed_pct(meta.get("vs_open_pct"))
    prior = _fmt_signed_pct(meta.get("prior_day_pct"))
    week = meta.get("near_52w_pct")
    dvol = meta.get("prior_dvol_m")
    parts = [
        f"gap {gap}" if gap else "gap n/a",
        f"vs-open {vs_open}" if vs_open else "vs-open n/a",
        f"prior-day {prior}" if prior else "prior-day n/a",
        f"52w {float(week):.0f}%" if isinstance(week, (int, float)) else "52w n/a",
    ]
    try:
        parts.append(f"prev ${float(dvol):.0f}M dvol")
    except (TypeError, ValueError):
        parts.append("prev dvol n/a")
    return parts


def _score_1_5(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and 1 <= value <= 5:
        return value
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= 5 else None


def parse_rocket_llm(parsed: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        return None
    score = _score_1_5(parsed.get("significance_score"))
    if score is None:
        return None
    catalyst_type = str(parsed.get("catalyst_type") or "unexplained").strip().lower()
    if catalyst_type not in CATALYST_TYPES:
        catalyst_type = "unexplained"
    direction = str(parsed.get("impact_direction") or "not_inferable").strip().lower()
    if direction not in {"bullish", "bearish", "neutral", "not_inferable"}:
        direction = "not_inferable"
    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    is_significant = bool(parsed.get("is_significant"))
    if (score >= 4) != is_significant:
        is_significant = score >= 4
    summary = _clean_segment(parsed.get("summary"))
    catalyst = _clean_segment(parsed.get("catalyst"))
    reason = _clean_segment(parsed.get("significance_reason"))
    if not summary:
        summary = catalyst or "Rocket gap"
    return {
        "catalyst": catalyst,
        "catalyst_type": catalyst_type,
        "significance_score": score,
        "is_significant": is_significant,
        "significance_reason": reason,
        "summary": summary,
        "impact_direction": direction,
        "confidence": round(confidence, 3),
        "insufficient_event_info": bool(parsed.get("insufficient_event_info")),
    }


def discovery_explanation(meta: Dict[str, Any]) -> str:
    llm = meta.get("llm") if isinstance(meta.get("llm"), dict) else {}
    summary = _clean_segment(llm.get("summary") or llm.get("catalyst") or "Rocket gap")
    summary = discovery_trade_explanation_lead(summary)
    score = llm.get("significance_score")
    catalyst_type = llm.get("catalyst_type") or "unexplained"
    if score is None:
        sig_seg = "sig n/a"
    else:
        sig_seg = f"sig={score} {catalyst_type}"
    parts = [summary, sig_seg]
    reason = _clean_segment(llm.get("significance_reason"))
    if reason:
        parts.append(reason)
    parts.extend(tape_explanation_segments(meta))
    return " | ".join(parts)


def _watch_gap(entry) -> float:
    try:
        return float((entry.meta or {}).get("gap_pct") or 0)
    except (TypeError, ValueError):
        return 0.0


def top_gap_watches(entries: List[Any], limit: int = TOP_GAPS) -> List[Any]:
    return sorted(entries, key=lambda entry: (-_watch_gap(entry), entry.stock.symbol))[:limit]


class Rocket(AdvisorBase):
    """Opening-high watcher: Stage A watch, Stage B news LLM, Stage C top-5 discover."""

    def _session_date(self) -> date:
        return datetime.now(ET).date()

    def _blob_date_is(self, key: str, session: date) -> bool:
        state = self._advisor_blob_state()
        return str(state.get(key) or "") == session.isoformat()

    def _mark_blob_date(self, key: str, session: date, extra: Optional[Dict[str, Any]] = None) -> None:
        state = self._advisor_blob_state()
        state[key] = session.isoformat()
        state[f"{key}_at"] = datetime.now(ET).isoformat(timespec="seconds")
        if extra:
            state.update(extra)
        self._save_advisor_blob_state(state)

    def _already_watched_session(self, session: date) -> bool:
        return self._blob_date_is("rocket_watch_date", session)

    def _mark_watched_session(self, session: date, extra: Optional[Dict[str, Any]] = None) -> None:
        self._mark_blob_date("rocket_watch_date", session, extra)

    def _type_cache(self) -> Dict[str, str]:
        state = self._advisor_blob_state()
        raw = state.get("ticker_types") or {}
        if not isinstance(raw, dict):
            return {}
        return {str(k).upper(): str(v).upper() for k, v in raw.items() if k and v}

    def _resolve_types(self, tickers: List[str]) -> Dict[str, str]:
        cache = self._type_cache()
        resolved: Dict[str, str] = {}
        missing: List[str] = []
        for ticker in tickers:
            kind = cache.get(ticker, "")
            if kind:
                resolved[ticker] = kind
            else:
                missing.append(ticker)
        fetched = 0
        for i, ticker in enumerate(missing, start=1):
            if i > 1:
                time.sleep(TYPE_FETCH_SLEEP_S)
            try:
                kind = financial_polygon.fetch_ticker_type(ticker)
            except Exception as exc:
                logger.warning("Rocket type fetch failed %s: %s", ticker, exc)
                kind = ""
            if kind:
                resolved[ticker] = kind
                cache[ticker] = kind
                fetched += 1
            if i == 1 or i % 10 == 0 or i == len(missing):
                logger.info(
                    "Rocket types %s/%s missing (fetched=%s cache=%s)",
                    i,
                    len(missing),
                    fetched,
                    len(tickers) - len(missing),
                )
        if fetched:
            state = self._advisor_blob_state()
            state["ticker_types"] = cache
            self._save_advisor_blob_state(state)
        return resolved

    def _already_watched_symbol(self, symbol: str, session: date) -> bool:
        from core.models import Watchlist

        return Watchlist.objects.filter(
            advisor=self.advisor,
            stock__symbol=symbol,
            meta__session=session.isoformat(),
        ).exists()

    def _day_before_map(self, session: date) -> Dict[str, Dict[str, Any]]:
        t2 = prior_trading_day(prior_trading_day(session))
        try:
            bars, resolved = financial_polygon.fetch_grouped_daily_map(t2, adjusted=True)
            logger.info("Rocket T-2 grouped %s -> %s (%s symbols)", t2, resolved, len(bars))
            return bars
        except Exception as exc:
            logger.warning("Rocket T-2 grouped failed (%s): %s", t2, exc)
            return {}

    def _scan_live(self, session: date) -> List[RocketCandidate]:
        snapshots = financial_polygon.fetch_stock_snapshots()
        day_before = self._day_before_map(session)
        rows = quotes_from_snapshots(snapshots, day_before)
        logger.info("Rocket gap/dvol filter: %s names", len(rows))
        types = self._resolve_types([row.ticker for row in rows])
        rows = apply_equity_types(rows, types)
        highs = week52_highs([row.ticker for row in rows])
        return attach_52w(rows, highs)

    def _maybe_watch(self, sa) -> None:
        mins = self.market_open()
        if mins is None:
            logger.info("Rocket skip: market closed")
            return
        if mins < WATCH_MIN_MINUTES:
            logger.info("Rocket skip: before +%sm (market_open=%sm)", WATCH_MIN_MINUTES, mins)
            return

        session = self._session_date()
        if self._already_watched_session(session):
            logger.info("Rocket skip: Stage A already done session=%s", session)
            return
        if mins > WATCH_MAX_MINUTES:
            logger.info(
                "Rocket skip: after Stage A window (market_open=%sm, need <=%sm)",
                mins,
                WATCH_MAX_MINUTES,
            )
            return

        logger.info("Rocket Stage A sa=%s session=%s market_open=%sm", sa.id, session, mins)
        rows = self._scan_live(session)
        watched = 0
        skipped = 0
        for row in rows:
            if self._already_watched_symbol(row.ticker, session):
                skipped += 1
                continue
            entry = self.watch(
                row.ticker,
                watch_explanation(row),
                days=WATCH_DAYS,
                meta=watch_meta(row, session),
                status="Pending",
            )
            if entry is not None:
                watched += 1

        self._mark_watched_session(
            session,
            extra={
                "rocket_watch_count": watched,
                "rocket_watch_skipped": skipped,
                "rocket_watch_candidates": len(rows),
            },
        )
        logger.info(
            "Rocket Stage A done session=%s candidates=%s watched=%s skipped=%s",
            session,
            len(rows),
            watched,
            skipped,
        )

    def _session_watches(self, session: date):
        from core.models import Watchlist

        return list(
            Watchlist.objects.filter(
                advisor=self.advisor,
                meta__session=session.isoformat(),
                status="Pending",
            ).select_related("stock")
        )

    def _llm_prompt(self, entry) -> str:
        meta = entry.meta or {}
        stock = entry.stock
        company = (getattr(stock, "company", None) or "").strip() or stock.symbol
        try:
            gap_pct = float(meta.get("gap_pct"))
        except (TypeError, ValueError):
            gap_pct = 0.0
        return ROCKET_LLM_PROMPT_TEMPLATE.format(
            ticker=stock.symbol,
            company=company,
            session=str(meta.get("session") or ""),
            gap_pct=round(gap_pct, 2),
        )

    def _run_gap_llm(self, entry) -> Optional[Dict[str, Any]]:
        prompt = self._llm_prompt(entry)
        model, parsed = self.ask_gemini(prompt, timeout=LLM_TIMEOUT_S, use_search=True)
        cleaned = parse_rocket_llm(parsed)
        if cleaned is None:
            logger.info(
                "Rocket LLM no parse %s model=%s — trying ask_llm search",
                entry.stock.symbol,
                model,
            )
            model, parsed = self.ask_llm(prompt, use_search=True, timeout=LLM_TIMEOUT_S)
            cleaned = parse_rocket_llm(parsed)
        if cleaned is None:
            logger.warning("Rocket LLM failed %s", entry.stock.symbol)
            return None
        cleaned["model"] = model
        logger.info(
            "Rocket LLM %s sig=%s type=%s model=%s",
            entry.stock.symbol,
            cleaned.get("significance_score"),
            cleaned.get("catalyst_type"),
            model,
        )
        return cleaned

    def _save_watch_llm(self, entry, llm: Optional[Dict[str, Any]]) -> None:
        meta = dict(entry.meta or {})
        meta["stage"] = "llm"
        if llm:
            meta["llm"] = llm
        entry.meta = meta
        entry.explanation = discovery_explanation(meta)[:500]
        entry.save(update_fields=["meta", "explanation"])

    def _maybe_llm(self, sa) -> None:
        mins = self.market_open()
        if mins is None or mins < LLM_MIN_MINUTES:
            return
        session = self._session_date()
        if not self._already_watched_session(session):
            logger.info("Rocket LLM skip: Stage A not done session=%s", session)
            return
        if self._blob_date_is("rocket_llm_date", session):
            logger.info("Rocket skip: Stage B already done session=%s", session)
            return

        entries = top_gap_watches(self._session_watches(session), TOP_GAPS)
        logger.info(
            "Rocket Stage B sa=%s session=%s market_open=%sm names=%s",
            sa.id,
            session,
            mins,
            [entry.stock.symbol for entry in entries],
        )
        ok = 0
        for entry in entries:
            llm = self._run_gap_llm(entry)
            self._save_watch_llm(entry, llm)
            if llm:
                ok += 1

        self._mark_blob_date(
            "rocket_llm_date",
            session,
            extra={"rocket_llm_ok": ok, "rocket_llm_count": len(entries)},
        )
        logger.info("Rocket Stage B done session=%s llm_ok=%s of %s", session, ok, len(entries))

    def _maybe_discover(self, sa) -> None:
        mins = self.market_open()
        if mins is None or mins < DISCOVER_MIN_MINUTES:
            return
        session = self._session_date()
        if not self._blob_date_is("rocket_llm_date", session):
            logger.info("Rocket discover skip: Stage B not done session=%s", session)
            return
        if self._blob_date_is("rocket_discover_date", session):
            logger.info("Rocket skip: Stage C already done session=%s", session)
            return

        entries = top_gap_watches(self._session_watches(session), TOP_GAPS)
        logger.info(
            "Rocket Stage C sa=%s session=%s names=%s",
            sa.id,
            session,
            [entry.stock.symbol for entry in entries],
        )
        discoveries = 0
        skipped = 0
        for entry in entries:
            symbol = entry.stock.symbol
            if not self.allow_discovery(
                symbol,
                period=ROCKET_DISCOVERY_COOLDOWN_HOURS,
                headline_check=False,
            ):
                skipped += 1
                continue
            meta = dict(entry.meta or {})
            explanation = discovery_explanation(meta)
            if self.discovered(sa, symbol, explanation, meta=meta):
                discoveries += 1
                meta["stage"] = "scored"
                entry.meta = meta
                entry.status = "Executed"
                entry.explanation = explanation[:500]
                entry.save(update_fields=["meta", "status", "explanation"])

        self._mark_blob_date(
            "rocket_discover_date",
            session,
            extra={
                "rocket_discover_count": discoveries,
                "rocket_discover_skipped": skipped,
            },
        )
        logger.info(
            "Rocket Stage C done session=%s discoveries=%s skipped=%s",
            session,
            discoveries,
            skipped,
        )

    def discover(self, sa) -> None:
        self._maybe_watch(sa)
        self._maybe_llm(sa)
        self._maybe_discover(sa)

    def analyze(self, sa, stock) -> None:
        return


register(name="Rocket", python_class="Rocket")
