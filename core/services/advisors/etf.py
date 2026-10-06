"""
ETF advisor — discovers stocks newly added to tracked thematic ETF holdings.

Flow:
  1) Once/day after UTC cutoff: refresh snapshots → gate → watch() Pending
  2) Each SA while RTH ≥ +15m: open-check Pending watches → discovered()

Uses core.services.financial.etf_holdings for snapshot/diff/lookup.
Exit: PERCENTAGE_REBUY (no average-down) + AFTER_DAYS (profit-only) + PROFIT_FLAT.
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from core.services.advisors.advisor import AdvisorBase, register
from core.services.financial import etf_holdings

logger = logging.getLogger(__name__)

PROCESS_CUTOFF_HOUR_UTC = 12
ETF_DISCOVERY_COOLDOWN_HOURS = 24 * 30
ETF_WATCH_DAYS = 5
ETF_WATCH_SOURCE = "etf_entrant"
OPEN_CHECK_MIN_MINUTES = 15

# Open-check vs prior close (%): cliff / allow / rocket.
ETF_CLIFF_VS_CLOSE = -8.0
ETF_ROCKET_VS_CLOSE = 8.0

# Buy gates (blob may override).
ETF_SKIP_DEFAULT = frozenset({"XBI", "XAR"})
ETF_MIN_WEIGHT_PCT = 0.25
ETF_MIN_PRICE = 5.0
ETF_MAX_NEW_PER_DAY = 5

ETF_REBUY_DROP = Decimal("0.05")
ETF_REBUY_MAX_TRANCHES = Decimal("1")  # initial tranche only — no average-down
ETF_AFTER_DAYS = Decimal("15")
ETF_PROFIT_FLAT_RANGE = Decimal("0.05")
ETF_PROFIT_FLAT_DAYS = Decimal("15")

DEFAULT_SELL_INSTRUCTIONS = [
    ("PERCENTAGE_REBUY", ETF_REBUY_DROP, ETF_REBUY_MAX_TRANCHES),
    ("AFTER_DAYS", ETF_AFTER_DAYS, None),
    ("PROFIT_FLAT", ETF_PROFIT_FLAT_RANGE, ETF_PROFIT_FLAT_DAYS),
]

DEFAULT_TRADE_ETFS: Tuple[str, ...] = tuple(
    etf for etf in etf_holdings.DEFAULT_ETF_LIST if etf not in ETF_SKIP_DEFAULT
)


def build_etf_discovery_explanation(event: etf_holdings.EntrantEvent) -> str:
    weight = (
        f"{event.weight_pct:.2f}% fund weight"
        if event.weight_pct is not None
        else "Fund weight unavailable"
    )
    segments = [
        f"Added to {event.etf} ETF",
        etf_holdings.fund_character_sentence(event.theme, event.management_style),
        weight,
        f"First seen in holdings {event.holdings_date}",
        etf_holdings.inclusion_label(event.inclusion_type),
    ]
    return " | ".join(segments)


def classify_etf_open_bucket(vs_close_pct: Optional[float]) -> str:
    """cliff | allow | rocket | unclear from % vs prior close."""
    if vs_close_pct is None:
        return "unclear"
    if vs_close_pct <= ETF_CLIFF_VS_CLOSE:
        return "cliff"
    if vs_close_pct >= ETF_ROCKET_VS_CLOSE:
        return "rocket"
    return "allow"


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def etf_open_tape(symbol: str) -> Optional[Dict[str, Any]]:
    """Prior close vs live last for open-check (yfinance)."""
    sym = (symbol or "").strip().upper()
    if not sym:
        return None
    try:
        import yfinance as yf

        t = yf.Ticker(sym)
        last = None
        try:
            fi = getattr(t, "fast_info", None) or {}
            last = _safe_float(fi.get("last_price") or fi.get("lastPrice"))
        except Exception:
            last = None

        daily = t.history(period="10d", interval="1d", auto_adjust=True)
        prior = None
        if daily is not None and not daily.empty and "Close" in daily.columns:
            closes = daily["Close"].astype(float).dropna()
            if len(closes) >= 2:
                prior = _safe_float(closes.iloc[-2])
                if last is None:
                    last = _safe_float(closes.iloc[-1])
            elif len(closes) == 1:
                prior = _safe_float(closes.iloc[-1])

        if last is None or prior is None or prior == 0:
            return None
        vs_close = 100.0 * (last - prior) / prior
        return {
            "prior_close": prior,
            "last": last,
            "vs_close_pct": vs_close,
        }
    except Exception as exc:
        logger.warning("etf_open_tape(%s) failed: %s", sym, exc)
        return None


def _event_sort_key(event: etf_holdings.EntrantEvent) -> Tuple[int, float]:
    """Prefer active managed, then higher fund weight."""
    active_rank = 0 if event.management_style == "active" else 1
    weight = event.weight_pct if event.weight_pct is not None else -1.0
    return (active_rank, -float(weight))


class Etf(AdvisorBase):
    """Daily ETF holdings diff → watch → RTH open-check → discover."""

    sell_instructions = list(DEFAULT_SELL_INSTRUCTIONS)

    def _etf_settings(self) -> Dict[str, Any]:
        state = self._advisor_blob_state()
        skip = state.get("skip_etfs")
        if skip is None:
            skip_etfs = set(ETF_SKIP_DEFAULT)
        else:
            skip_etfs = {str(e).strip().upper() for e in skip if str(e).strip()}

        return {
            "min_weight_pct": float(state.get("min_weight_pct", ETF_MIN_WEIGHT_PCT)),
            "min_price": float(state.get("min_price", ETF_MIN_PRICE)),
            "max_new_per_day": int(state.get("max_new_per_day", ETF_MAX_NEW_PER_DAY)),
            "etfs": state.get("etfs"),
            "skip_etfs": skip_etfs,
        }

    def _trade_etf_list(self, settings: Dict[str, Any]) -> List[str]:
        skip = settings.get("skip_etfs") or set()
        if settings.get("etfs"):
            raw = [str(e).strip().upper() for e in settings["etfs"] if str(e).strip()]
            return [e for e in raw if e not in skip]
        return [e for e in DEFAULT_TRADE_ETFS if e not in skip]

    def _passes_filters(self, event: etf_holdings.EntrantEvent, settings: Dict[str, Any]) -> bool:
        etf = (event.etf or "").strip().upper()
        if etf in (settings.get("skip_etfs") or set()):
            logger.info("ETF skip %s %s: etf in skip list", event.symbol, etf)
            return False

        min_weight = float(settings.get("min_weight_pct") or 0.0)
        if min_weight > 0 and event.weight_pct is not None and event.weight_pct < min_weight:
            logger.info(
                "ETF skip %s %s: weight %.4f < min %.4f",
                event.symbol,
                etf,
                event.weight_pct,
                min_weight,
            )
            return False

        min_price = float(settings.get("min_price") or 0.0)
        if min_price > 0:
            stock = self.get_stock(event.symbol)
            if stock is None:
                return False
            stock.refresh()
            if float(stock.price or 0) < min_price:
                logger.info(
                    "ETF skip %s: price %.2f < min %.2f",
                    event.symbol,
                    float(stock.price or 0),
                    min_price,
                )
                return False
        return True

    def _watch_meta(self, event: etf_holdings.EntrantEvent, explanation: str) -> Dict[str, Any]:
        return {
            "source": ETF_WATCH_SOURCE,
            "etf": event.etf,
            "theme": event.theme,
            "name": event.name,
            "weight_pct": event.weight_pct,
            "holdings_date": event.holdings_date,
            "management_style": event.management_style,
            "issuer": event.issuer,
            "inclusion_type": event.inclusion_type,
            "discovery_explanation": explanation,
        }

    def _refresh_and_watch(self, sa) -> None:
        today = date.today().isoformat()
        if not self.should_process_market_date_once(
            target_date=today,
            cutoff_hour_utc=PROCESS_CUTOFF_HOUR_UTC,
        ):
            return

        settings = self._etf_settings()
        etf_list = self._trade_etf_list(settings)
        max_new = max(0, int(settings.get("max_new_per_day") or 0))

        logger.info("ETF sa=%s: refresh snapshots etfs=%s", sa.id, etf_list)
        results = etf_holdings.refresh_snapshots(etfs=etf_list, refresh=False)
        events = etf_holdings.entrants_from_refresh(results)

        ok_count = sum(1 for r in results if r.ok)
        stale_count = sum(1 for r in results if r.stale)
        fail_count = len(results) - ok_count
        watched = 0
        skipped_cooldown = 0
        skipped_filter = 0
        skipped_watched = 0
        skipped_cap = 0

        eligible: List[etf_holdings.EntrantEvent] = []
        for event in events:
            if self.watched(event.symbol):
                skipped_watched += 1
                continue
            if not self.allow_discovery(event.symbol, period=ETF_DISCOVERY_COOLDOWN_HOURS):
                skipped_cooldown += 1
                continue
            if not self._passes_filters(event, settings):
                skipped_filter += 1
                continue
            eligible.append(event)

        eligible.sort(key=_event_sort_key)
        if max_new > 0 and len(eligible) > max_new:
            skipped_cap = len(eligible) - max_new
            eligible = eligible[:max_new]

        for event in eligible:
            explanation = build_etf_discovery_explanation(event)
            entry = self.watch(
                event.symbol,
                explanation[:500],
                days=ETF_WATCH_DAYS,
                meta=self._watch_meta(event, explanation),
                status="Pending",
            )
            if entry is not None:
                watched += 1

        state = self._advisor_blob_state()
        state["last_refresh"] = {
            "date": today,
            "ok": ok_count,
            "stale": stale_count,
            "failed": fail_count,
            "entrants": len(events),
            "eligible": len(eligible) + skipped_cap,
            "watched": watched,
            "cooldown_skip": skipped_cooldown,
            "filter_skip": skipped_filter,
            "watched_skip": skipped_watched,
            "cap_skip": skipped_cap,
        }
        self._save_advisor_blob_state(state)
        self.mark_market_date_processed(today)

        logger.info(
            "ETF sa=%s: refresh ok=%d stale=%d failed=%d entrants=%d watched=%d "
            "cooldown_skip=%d filter_skip=%d watched_skip=%d cap_skip=%d",
            sa.id,
            ok_count,
            stale_count,
            fail_count,
            len(events),
            watched,
            skipped_cooldown,
            skipped_filter,
            skipped_watched,
            skipped_cap,
        )

    def _pending_etf_watches(self) -> List[Any]:
        return [
            w
            for w in self.watchlist()
            if isinstance(getattr(w, "meta", None), dict)
            and w.meta.get("source") == ETF_WATCH_SOURCE
        ]

    def _open_check_watches(self, sa) -> Dict[str, int]:
        """Promote Pending etf_entrant watches after RTH open-check."""
        counts = {
            "checked": 0,
            "allow": 0,
            "cliff": 0,
            "rocket": 0,
            "unclear": 0,
            "discovered": 0,
            "excluded": 0,
            "errors": 0,
        }

        mins = self.market_open()
        if mins is None:
            logger.info("ETF open-check: market closed — skip")
            return counts
        if mins < OPEN_CHECK_MIN_MINUTES:
            logger.info(
                "ETF open-check: market_open=%sm (need >= %sm) — skip",
                mins,
                OPEN_CHECK_MIN_MINUTES,
            )
            return counts

        watches = self._pending_etf_watches()
        if not watches:
            logger.info("ETF open-check: no Pending watches")
            return counts

        logger.info("ETF open-check: %d Pending (market_open=%sm)", len(watches), mins)

        for w in watches:
            symbol = (getattr(getattr(w, "stock", None), "symbol", None) or "").strip().upper()
            if not symbol:
                continue
            meta = dict(w.meta or {})
            try:
                tape = etf_open_tape(symbol)
                vs = tape.get("vs_close_pct") if tape else None
                bucket = classify_etf_open_bucket(vs)
                counts["checked"] += 1
                counts[bucket] = counts.get(bucket, 0) + 1

                logger.info(
                    "ETF open-check %s → %s (vs_close=%s last=%s prior=%s)",
                    symbol,
                    bucket,
                    f"{vs:+.2f}%" if vs is not None else "n/a",
                    tape.get("last") if tape else None,
                    tape.get("prior_close") if tape else None,
                )

                meta["open_bucket"] = bucket
                meta["open_check"] = {
                    "vs_close_pct": vs,
                    "last": tape.get("last") if tape else None,
                    "prior_close": tape.get("prior_close") if tape else None,
                }
                w.meta = meta

                if bucket == "cliff":
                    note = (
                        f"OPEN cliff ({vs:+.1f}% vs close)"
                        if vs is not None
                        else "OPEN cliff"
                    )
                    w.status = "Excluded"
                    w.explanation = f"{note} | {w.explanation}"[:500]
                    w.save(update_fields=["status", "meta", "explanation"])
                    counts["excluded"] += 1
                    continue

                if bucket == "rocket":
                    note = (
                        f"OPEN rocket ({vs:+.1f}% vs close) — no chase"
                        if vs is not None
                        else "OPEN rocket — no chase"
                    )
                    w.explanation = f"{note} | {w.explanation}"[:500]
                    w.save(update_fields=["meta", "explanation"])
                    continue

                if bucket == "unclear":
                    w.save(update_fields=["meta"])
                    continue

                # allow → discover
                if not self.allow_discovery(symbol, period=ETF_DISCOVERY_COOLDOWN_HOURS):
                    w.status = "Excluded"
                    w.explanation = f"allow_discovery false | {w.explanation}"[:500]
                    w.save(update_fields=["status", "meta", "explanation"])
                    counts["excluded"] += 1
                    continue

                explanation = (meta.get("discovery_explanation") or w.explanation or "").strip()
                try:
                    w.stock.refresh()
                except Exception as exc:
                    logger.warning("ETF open-check %s refresh failed: %s", symbol, exc)

                discovery_meta = {
                    "etf_entrant": {
                        "etf": meta.get("etf"),
                        "theme": meta.get("theme"),
                        "weight_pct": meta.get("weight_pct"),
                        "holdings_date": meta.get("holdings_date"),
                        "inclusion_type": meta.get("inclusion_type"),
                        "management_style": meta.get("management_style"),
                        "open_check": meta.get("open_check"),
                    }
                }
                stock = self.discovered(
                    sa,
                    symbol,
                    explanation,
                    sell_instructions=self.sell_instructions,
                    weight=1.0,
                    meta=discovery_meta,
                )
                if stock:
                    w.status = "Executed"
                    w.save(update_fields=["status", "meta"])
                    counts["discovered"] += 1
                    logger.info("ETF discovered %s from open-check allow", symbol)
                else:
                    w.save(update_fields=["meta"])
                    logger.warning("ETF discovered() returned None for %s; leave Pending", symbol)

            except Exception as exc:
                counts["errors"] += 1
                logger.error("ETF open-check %s failed: %s", symbol, exc)

        logger.info("ETF open-check done: %s", counts)
        return counts

    def discover(self, sa) -> None:
        self._refresh_and_watch(sa)
        self._open_check_watches(sa)

    def analyze(self, sa, stock) -> None:
        return


register(name="ETF", python_class="Etf")
