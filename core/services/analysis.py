"""
Stock Analysis Service
"""

import logging
from datetime import datetime, timedelta, timezone as dt_timezone
from decimal import Decimal
from typing import Any, Optional

import pandas as pd
from pytz import timezone as tz
from django.utils import timezone
from core.models import Holding, Discovery, Advisor, Profile
from core.services.execution import execute_buy, execute_sell
from core.services.intraday_stabilize import (
    STABILIZE_MINUTES_DEFAULT,
    price_above_minutes_ago,
)
from core.services.llm.gemini import ask_gemini as llm_ask_gemini
from core.services.health.risk_matrix import (
    discovery_axes,
    discovery_passes_risk_gate,
    so_gate_fail_display,
)
from core.services.market import in_opening_noise_window, market_open

logger = logging.getLogger(__name__)
DT_EXIT_CONFIDENCE_MIN = 0.70
REBUY_STABILIZE_MINUTES = STABILIZE_MINUTES_DEFAULT
PEAKED_MIN_MARKET_MINUTES = 60
RECENT_TP_LOOKBACK_HOURS = 4

# Runner: realtime impulse check inside analyse_intraday (shared 1m hist with IPC).
# Shadow mode (default): log target_moving but still run IPC. Flip to True after log replay.
RUNNER_SKIP_IPC = False
RUNNER_LOOKBACK_MINUTES = 30
RUNNER_MIN_PROFIT_PCT = 2.0
RUNNER_MIN_RET_30M_PCT = 1.0
RUNNER_MIN_VOL_RATIO = 2.0
RUNNER_MIN_CLOSE_POSITION = 0.6
RUNNER_MIN_SIGNALS = 4


def factor_sentiment(fund: Profile) -> Decimal:
    """
    Resolve fund sentiment to a scalar.
    - Manual presets: map via Profile.SENTIMENT (float values).
    - AUTO: derive from cash / wealth thresholds.
    Always returns a Decimal.
    """

    sentiment_key = fund.sentiment or "AUTO"
    if sentiment_key != "AUTO":
        return Decimal(Profile.SENTIMENT.get(sentiment_key, 1.0))

    cash_value = fund.cash or Decimal("0")
    holdings_value = Decimal("0")
    for holding in Holding.objects.filter(fund=fund).select_related("stock"):
        if holding.stock and holding.stock.price and holding.shares:
            holdings_value += Decimal(str(holding.stock.price)) * Decimal(str(holding.shares))

    wealth = cash_value + holdings_value
    if wealth <= 0:
        logger.warning(
            f"{fund.name}: AUTO sentiment fallback to STAG (wealth <= 0). "
            f"cash={cash_value}, holdings={holdings_value}"
        )
        return Decimal(1.0)

    cash_ratio = float(cash_value / wealth)
    if cash_ratio < 0.25:
        band = "STRONG_BEAR"
    elif cash_ratio < 0.50:
        band = "BEAR"
    else:
        band = "STAG"

    sentiment = Profile.SENTIMENT[band]

    if sentiment < 1.0:
        logger.info(
            f"{fund.name}: AUTO sentiment {band} ({sentiment:.2f}) "
            f"cash_ratio={cash_ratio:.3f}, cash={cash_value}, holdings={holdings_value}, wealth={wealth}"
        )

    return Decimal(str(sentiment))

def _recent_intraday_peak(stock, since_date, lookback_hours=RECENT_TP_LOOKBACK_HOURS):
    if not since_date:
        return None

    try:
        import yfinance as yf

        now = timezone.now()
        anchor = since_date
        if timezone.is_naive(anchor):
            anchor = timezone.make_aware(anchor, dt_timezone.utc)
        window_start = max(anchor, now - timedelta(hours=lookback_hours))

        ticker = yf.Ticker(stock.symbol)
        hist = ticker.history(period="1d", interval="1m")
        if hist.empty or "High" not in hist.columns:
            return None

        if hist.index.tz is None:
            window_start = timezone.make_naive(window_start, dt_timezone.utc)
        peak_window = hist[hist.index >= window_start]
        if peak_window.empty:
            return None
        return float(peak_window["High"].max())
    except Exception as exc:
        logger.debug("Could not check recent intraday peak for %s: %s", stock.symbol, exc)
        return None


def analyse_target(holding, target, sentiment, discovery=None):

    current = holding.stock.price
    buy_price = holding.average_price or Decimal(0.0)
    target = buy_price + (Decimal(str(target)) - buy_price) * sentiment

    # Case 1: Targets should only trigger sells at a profit, not at a loss.
    if current <= buy_price:
        return False

    # Case 2: Price < target, but a recent intraday spike already hit it.
    if current < target:
        if discovery is not None:
            peak = _recent_intraday_peak(holding.stock, discovery.created)
            if peak is not None and Decimal(str(peak)) >= target:
                logger.info(
                    "Taking TP SELL for %s: recent intraday peak %.2f hit target %s; current=%s avg=%s",
                    holding.stock.symbol,
                    peak,
                    target,
                    current,
                    buy_price,
                )
                return True
        return False

    # Case 3: At/above target, only hold if trend is clearly positive
    trend = holding.stock.calc_trend(
        period="1d",
        interval="15m",
        hours=1,
        latest_price=current,
    )
    if trend is None:
        return True  # in doubt sell

    if trend < Decimal("0.05"):
        return True  # weak/negative momentum => take profit

    logger.info(
        "Delaying TP SELL. %s trending upward (trend=%.4f >= 0.05, current=%s, target=%s)",
        holding.stock.symbol, float(trend), current, target,
    )
    return False


def _fetch_intraday_hist(stock):
    """Today's 1m bars for IPC peak and runner impulse (single fetch per analyse_intraday call)."""
    try:
        import yfinance as yf

        hist = yf.Ticker(stock.symbol).history(period="1d", interval="1m")
        if hist.empty or "High" not in hist.columns:
            return None
        return hist
    except Exception as exc:
        logger.debug("Could not fetch intraday hist for %s: %s", stock.symbol, exc)
        return None


def _hist_anchor_ts(since_date, hist: pd.DataFrame):
    anchor = since_date
    if timezone.is_naive(anchor):
        anchor = timezone.make_aware(anchor, dt_timezone.utc)
    if hist.index.tz is None:
        return timezone.make_naive(anchor, dt_timezone.utc)
    return pd.Timestamp(anchor).tz_convert(hist.index.tz)


def _hist_check_ts(hist: pd.DataFrame) -> pd.Timestamp:
    now = timezone.now()
    if hist.index.tz is None:
        return pd.Timestamp(now.replace(tzinfo=None))
    return pd.Timestamp(now.astimezone(hist.index.tz))


def _price_at_or_before(hist: pd.DataFrame, ts: pd.Timestamp) -> Optional[float]:
    if hist.empty or "Close" not in hist.columns:
        return None
    eligible = hist.index <= ts
    if not eligible.any():
        return None
    return float(hist.loc[eligible, "Close"].astype(float).iloc[-1])


def _high_in_hist_range(
    hist: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> Optional[float]:
    if hist.empty or "High" not in hist.columns:
        return None
    window = hist[(hist.index >= start_ts) & (hist.index < end_ts)]
    if window.empty:
        return None
    return float(window["High"].astype(float).max())


def _volume_sum_hist(
    hist: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> Optional[float]:
    if hist.empty or "Volume" not in hist.columns:
        return None
    window = hist[(hist.index >= start_ts) & (hist.index <= end_ts)]
    if window.empty:
        return None
    total = float(window["Volume"].astype(float).sum())
    return total if total > 0 else None


def _intraday_peak_from_hist(hist: pd.DataFrame, since_date) -> Optional[float]:
    if not since_date or hist is None or hist.empty:
        return None
    try:
        anchor = _hist_anchor_ts(since_date, hist)
        peak_window = hist[hist.index >= anchor]
        if peak_window.empty:
            return None
        return float(peak_window["High"].max())
    except Exception as exc:
        logger.debug("Could not compute intraday peak from hist: %s", exc)
        return None


def _target_is_moving(holding, hist: pd.DataFrame, discovery=None) -> tuple[bool, dict[str, Any]]:
    """
    Realtime runner impulse: 30m buying-trend bundle on shared 1m hist.
    Returns (is_moving, detail) for shadow logging.
    """
    buy_price = holding.average_price or (discovery.price if discovery else None)
    if not buy_price or hist is None or hist.empty:
        return False, {}

    entry_price = float(buy_price)
    if entry_price <= 0:
        return False, {}

    check_ts = _hist_check_ts(hist)
    price_now = _price_at_or_before(hist, check_ts)
    if price_now is None or price_now <= entry_price:
        return False, {}

    lookback = RUNNER_LOOKBACK_MINUTES
    start_30 = check_ts - pd.Timedelta(minutes=lookback)
    prior_start = start_30 - pd.Timedelta(minutes=lookback)
    window = hist[(hist.index >= start_30) & (hist.index <= check_ts)]
    if window.empty or len(window) < 2:
        return False, {}

    price_30_ago = _price_at_or_before(hist, start_30)
    profit_pct = (price_now / entry_price - 1.0) * 100.0
    ret_30m_pct = None
    if price_30_ago is not None and price_30_ago > 0:
        ret_30m_pct = (price_now / price_30_ago - 1.0) * 100.0

    highs = window["High"].astype(float)
    lows = window["Low"].astype(float) if "Low" in window.columns else window["Close"].astype(float)
    high_max = float(highs.max())
    low_min = float(lows.min())
    if high_max > low_min:
        close_position = (price_now - low_min) / (high_max - low_min)
    else:
        close_position = 0.5

    last_10_start = check_ts - pd.Timedelta(minutes=10)
    prev_10_start = check_ts - pd.Timedelta(minutes=20)
    hi_last10 = _high_in_hist_range(hist, last_10_start, check_ts + pd.Timedelta(minutes=1))
    hi_prev10 = _high_in_hist_range(hist, prev_10_start, last_10_start)

    vol_last = _volume_sum_hist(hist, start_30, check_ts)
    vol_prior = _volume_sum_hist(hist, prior_start, start_30)
    vol_ratio = None
    if vol_last is not None and vol_prior is not None and vol_prior > 0:
        vol_ratio = vol_last / vol_prior

    signals: list[str] = []
    if profit_pct >= RUNNER_MIN_PROFIT_PCT:
        signals.append("profit")
    if ret_30m_pct is not None and ret_30m_pct >= RUNNER_MIN_RET_30M_PCT:
        signals.append("ret30m")
    if hi_last10 is not None and hi_prev10 is not None and hi_last10 > hi_prev10:
        signals.append("hh")
    if vol_ratio is not None and vol_ratio >= RUNNER_MIN_VOL_RATIO:
        signals.append("vol")
    if close_position >= RUNNER_MIN_CLOSE_POSITION:
        signals.append("close")

    score = len(signals)
    detail = {
        "score": score,
        "signals": ",".join(signals),
        "profit_pct": round(profit_pct, 3),
        "ret_30m_pct": None if ret_30m_pct is None else round(ret_30m_pct, 3),
        "vol_ratio": None if vol_ratio is None else round(vol_ratio, 3),
        "close_position": round(close_position, 3),
        "price": round(price_now, 4),
    }
    return score >= RUNNER_MIN_SIGNALS, detail


def _ipc_should_sell(
    current,
    buy_price,
    activation_mult,
    giveback_pct,
    peak: Optional[float],
) -> bool:
    if peak is None or current <= buy_price:
        return False

    activation_px = Decimal(str(activation_mult)) * Decimal(str(buy_price))
    peak_d = Decimal(str(peak))
    if peak_d < activation_px:
        return False

    giveback = Decimal(str(giveback_pct or "0"))
    exit_px = peak_d * (Decimal("1.0") - giveback)
    return current <= exit_px


def analyse_intraday(holding, activation_mult, giveback_pct, discovery=None):
    current = holding.stock.price
    buy_price = holding.average_price or Decimal(0.0)
    if current <= buy_price:
        return False

    since_date = discovery.created if discovery else holding.created
    hist = _fetch_intraday_hist(holding.stock)
    if hist is None:
        return False

    moving, moving_detail = _target_is_moving(holding, hist, discovery)
    peak = _intraday_peak_from_hist(hist, since_date)
    ipc_would_sell = _ipc_should_sell(
        current, buy_price, activation_mult, giveback_pct, peak
    )

    if moving:
        logger.info(
            "target_moving %s score=%s signals=%s profit_pct=%s ret_30m=%s "
            "vol_ratio=%s close_pos=%s price=%s ipc_would_sell=%s runner_skip_ipc=%s",
            holding.stock.symbol,
            moving_detail.get("score"),
            moving_detail.get("signals"),
            moving_detail.get("profit_pct"),
            moving_detail.get("ret_30m_pct"),
            moving_detail.get("vol_ratio"),
            moving_detail.get("close_position"),
            moving_detail.get("price"),
            ipc_would_sell,
            RUNNER_SKIP_IPC,
        )
        if RUNNER_SKIP_IPC:
            return False

    if not ipc_would_sell:
        return False

    activation_px = Decimal(str(activation_mult)) * Decimal(str(buy_price))
    giveback = Decimal(str(giveback_pct or "0"))
    exit_px = Decimal(str(peak)) * (Decimal("1.0") - giveback)
    logger.info(
        "Taking intraday SELL for %s: peak %.2f hit activation %s; current=%s exit<=%s avg=%s",
        holding.stock.symbol,
        peak,
        activation_px,
        current,
        exit_px,
        buy_price,
    )
    return True


def _session_exit_threshold_px(value1, avg) -> Decimal | None:
    """
    Dollar sell threshold for END_DAY / END_WEEK.
    value1 is a multiplier (e.g. 1.01) applied to avg at sell-check time.
    END_WEEK value2 = optional min days held (holding.created); default 0.
    Legacy rows store a fixed dollar threshold when value1 > 3.
    """
    if value1 is None or not avg:
        return None
    mult = Decimal(str(value1))
    avg_d = Decimal(str(avg))
    if mult <= Decimal("3"):
        return mult * avg_d
    return mult


def _build_drop_prompt(context_block: str) -> str:
    return f"""
You are a live risk triage assistant for equity positions.

A descending-trend alert has triggered for the tickers below after a recent price drop.
Decide whether each position should be exited NOW due to a materially negative, very recent catalyst.

Source quality policy:
- Primary sources (highest trust): Reuters, Bloomberg, Dow Jones Newswires
- Secondary source: Benzinga
- Give most weight to primary sources.
- Use Benzinga alone only if the catalyst is clear and material.
- If credible recent evidence is missing or ambiguous, choose HOLD with lower confidence.

What qualifies for EXIT:
- A recent catalyst likely to impair near-term value materially, such as:
  earnings/guidance miss, major downgrade cluster, regulatory/legal action,
  financing/liquidity stress, or thesis-breaking company-specific news.
- The catalyst should plausibly explain the drop and suggest continued downside asymmetry.

What qualifies for HOLD:
- No credible material catalyst found in trusted sources, or
- Evidence appears stale, minor, speculative, or already absorbed.

Output requirements:
- For each ticker, return:
  - action: "EXIT" or "HOLD" only
  - confidence: number 0.0–1.0
  - reason: one concise sentence (max 25 words) naming the key catalyst/risk or lack of credible evidence
  - sources_used: array of source names you relied on (from Reuters/Bloomberg/Dow Jones Newswires/Benzinga; empty if none)

Rules:
- No prose outside JSON.
- If uncertain, default to HOLD with lower confidence.

Context:
{context_block}

Return ONLY a single JSON object in this exact shape:
{{
  "TICKER": {{
    "action": "EXIT|HOLD",
    "confidence": 0.00,
    "reason": "short reason",
    "sources_used": ["Reuters", "Bloomberg"]
  }}
}}
"""


def analyse_drop(sa, dropped_stocks):
    """
    Evaluate DT-triggered holdings via LLM and execute sells for EXIT decisions.
    """
    if not dropped_stocks:
        return

    # Deduplicate symbols for one batched LLM call, while preserving first-seen context.
    contexts_by_symbol = {}
    for item in dropped_stocks:
        holding = item["holding"]
        symbol = holding.stock.symbol.upper()
        if symbol in contexts_by_symbol:
            continue

        current_price = holding.stock.price or Decimal("0")
        buy_price = item.get("buy_price") or Decimal("0")
        pnl_pct = None
        if buy_price and buy_price > 0:
            pnl_pct = (Decimal(str(current_price)) - Decimal(str(buy_price))) / Decimal(str(buy_price)) * Decimal("100")

        trend = item.get("trend")
        threshold = item.get("threshold")
        company = (holding.stock.company or "").strip() or "unknown"
        fund = item["fund"]
        line = (
            f"  {symbol}: {company!r}, ${float(current_price):.2f}; "
            f"fund={fund.name!r}; trend={float(trend):.4f} < threshold={float(threshold):.4f}"
        )
        if pnl_pct is not None:
            line += (
                f"; buy_price=${float(buy_price):.2f}; "
                f"pnl_pct={float(pnl_pct):.2f}%"
            )
        contexts_by_symbol[symbol] = line

    context_block = "\n".join(contexts_by_symbol[s] for s in sorted(contexts_by_symbol.keys()))
    prompt = _build_drop_prompt(context_block)

    model, results, _next_model_idx, _next_key_idx = llm_ask_gemini(
        prompt=prompt,
        advisor_name="analyse_drop",
        gemini_model_index=0,
        gemini_key_index=0,
        timeout=120.0,
        use_search=True,
    )

    if not results or not isinstance(results, dict):
        logger.warning("analyse_drop: no usable JSON response from Gemini")
        return

    exit_by_symbol = {}
    for symbol in contexts_by_symbol.keys():
        data = results.get(symbol) or results.get(symbol.upper()) or results.get(symbol.lower())
        if not isinstance(data, dict):
            logger.info("analyse_drop: %s missing/invalid decision payload", symbol)
            continue

        action = (data.get("action") or "").strip().upper()
        raw_conf = data.get("confidence")
        try:
            confidence = float(raw_conf) if raw_conf is not None else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        reason = (data.get("reason") or "").strip()
        sources = data.get("sources_used") if isinstance(data.get("sources_used"), list) else []

        logger.info(
            "analyse_drop decision %s: action=%s confidence=%.2f sources=%s reason=%s",
            symbol,
            action or "N/A",
            confidence,
            ",".join(sources) if sources else "-",
            reason or "-",
        )

        if action == "EXIT" and confidence >= DT_EXIT_CONFIDENCE_MIN:
            exit_by_symbol[symbol] = {
                "confidence": confidence,
                "reason": reason,
                "sources_used": sources,
            }
            logger.info(
                "analyse_drop qualified EXIT %s (confidence %.2f >= %.2f)",
                symbol,
                confidence,
                DT_EXIT_CONFIDENCE_MIN,
            )
        else:
            logger.info(
                "analyse_drop HOLD/skip %s (action=%s, confidence=%.2f, threshold=%.2f)",
                symbol,
                action or "N/A",
                confidence,
                DT_EXIT_CONFIDENCE_MIN,
            )

    if not exit_by_symbol:
        logger.info("analyse_drop: no EXIT actions above confidence threshold (model=%s)", model)
        return

    logger.info(
        "analyse_drop: exiting %s symbols from %s DT candidates (model=%s)",
        len(exit_by_symbol),
        len(dropped_stocks),
        model,
    )

    for item in dropped_stocks:
        holding = item["holding"]
        fund = item["fund"]
        symbol = holding.stock.symbol.upper()
        decision = exit_by_symbol.get(symbol)
        if not decision:
            continue
        if not holding.pk or holding.shares <= 0:
            continue

        reason = decision["reason"] or "Recent material downside catalyst after descending trend."
        conf = decision["confidence"]
        logger.info(
            "analyse_drop executing SELL %s fund=%s holding_id=%s conf=%.2f",
            symbol,
            fund.name,
            holding.id,
            conf,
        )
        execute_sell(
            sa,
            fund,
            holding,
            reason[:240],
        )


def analyze_holdings(sa, funds):
    logger.info(f"Analyzing holdings for SA session {sa.id}")
    dropped_stocks = []
    market_status = market_open()
    peaked_allowed = market_status is not None and market_status >= PEAKED_MIN_MARKET_MINUTES

    # Check if we're in the last 30 minutes of trading day (3:30 PM ET onwards)
    et = tz('US/Eastern')
    now_et = timezone.now().astimezone(et)
    current_time = now_et.time()
    weekday = now_et.weekday()
    end_day = False
    end_week = False

    # Only check on weekdays (0=Monday, 4=Friday)
    if weekday < 5:
        # Check if after 3:30 PM ET (last 30 minutes of trading)
        end_day_check_time = datetime.strptime("15:30", "%H:%M").time()

        if current_time >= end_day_check_time:
            end_day = True

    # Check if it's Friday and market is open (anytime during trading hours)
    if weekday == 4:  # Friday
        market_open_time = datetime.strptime("09:30", "%H:%M").time()
        market_close_time = datetime.strptime("16:00", "%H:%M").time()

        # Check if within market hours (9:30 AM - 4:00 PM ET)
        if market_open_time <= current_time < market_close_time:
            end_week = True

    # Iterate thru funds
    for fund in funds:

        sentiment = factor_sentiment(fund)

        for holding in Holding.objects.filter(fund=fund).select_related(
            "stock", "discovery", "discovery__advisor"
        ):

            # Latest prices
            holding.stock.refresh()

            # Get most recent discovery for this stock
            # discovery = Discovery.objects.filter(stock=holding.stock).order_by('-created').first()
            discovery = holding.discovery

            if not discovery:
                logger.info("No holding.discovery for %s (fund=%s); skipping sell-instruction eval",
                            holding.stock.symbol, fund.name)
                continue

            if discovery:
                # Get all sell instructions for this discovery
                from core.models import SellInstruction
                instructions = SellInstruction.objects.filter(discovery=discovery)

                # Check sell conditions in priority order
                try:
                    # Calculate days_held and buy_price for dynamic instructions
                    days_held = (timezone.now() - discovery.created).days if discovery.created else 0
                    buy_price = holding.average_price if holding.average_price else discovery.price

                    for instruction in instructions:
                        if instruction.instruction == 'STOP_PRICE':
                            if instruction.value1 and holding.stock.price < instruction.value1:
                                execute_sell(sa, fund, holding, f"{holding.stock.symbol} fell to stop-loss of ${instruction.value1:.2f}")
                                break

                        elif instruction.instruction == 'STOP_PERCENTAGE':
                            adv = discovery.advisor
                            if (
                                adv
                                and adv.python_class == "Oracle"
                                and in_opening_noise_window(PEAKED_MIN_MARKET_MINUTES)
                            ):
                                continue
                            avg = holding.average_price or discovery.price
                            if instruction.value1 and avg:
                                mult = Decimal(str(instruction.value1))
                                # Legacy rows store dollar stop (price×mult at discovery); multipliers are ~0–2.
                                if mult <= Decimal("3"):
                                    stop_px = mult * Decimal(str(avg))
                                else:
                                    stop_px = mult
                                if holding.stock.price < stop_px:
                                    execute_sell(
                                        sa, fund, holding,
                                        f"{holding.stock.symbol} hit stop ${stop_px:.2f} "
                                        f"({instruction.value1}× avg ${avg:.2f})",
                                    )
                                    break

                        elif instruction.instruction == 'TARGET_PRICE':
                            if analyse_target(holding, instruction.value1, sentiment, discovery):
                                execute_sell(
                                    sa, fund, holding,
                                    f"{holding.stock.symbol} reached target price of ${instruction.value1:.2f}",
                                )
                                break

                        elif instruction.instruction == 'TARGET_PERCENTAGE':
                            avg = holding.average_price or discovery.price
                            if instruction.value1 and avg:
                                target_px = Decimal(str(instruction.value1)) * Decimal(str(avg))
                                if analyse_target(holding, target_px, sentiment, discovery):
                                    execute_sell(
                                        sa, fund, holding,
                                        f"{holding.stock.symbol} reached target "
                                        f"${target_px:.2f} ({instruction.value1}× avg ${avg:.2f})",
                                    )
                                    break

                        elif instruction.instruction == 'TARGET_INTRADAY':
                            if instruction.value1 and instruction.value2:
                                if analyse_intraday(holding, instruction.value1, instruction.value2, discovery):
                                    execute_sell(
                                        sa, fund, holding,
                                        f"{holding.stock.symbol} captured intraday target "
                                        f"({instruction.value1}× avg, {instruction.value2} giveback)",
                                    )
                                    break

                        elif instruction.instruction in ['TARGET_DIMINISHING', 'PERCENTAGE_DIMINISHING']:
                            # Calculate diminishing target: original_target → buy_price over max_days
                            # value1 = original target price, value2 = max_days
                            if instruction.value1 and buy_price:
                                max_days = int(instruction.value2) if instruction.value2 is not None else 14
                                if days_held <= max_days:
                                    progress = float(days_held) / float(max_days) if max_days > 0 else 1.0
                                    original_target = float(instruction.value1)
                                    current_target = original_target - progress * (original_target - float(buy_price))
                                else:
                                    current_target = float(buy_price)  # After max_days, target = buy_price (break-even)

                                if analyse_target(holding, Decimal(str(current_target)), sentiment, discovery):
                                    execute_sell(sa, fund, holding, f"{holding.stock.symbol} diminishing target ${current_target:.2f} (day {days_held}/{max_days})")
                                    break
                            else:
                                logger.warning(f"TARGET_DIMINISHING invalid threshold (value1={instruction.value1}, buy_price={buy_price})")

                        elif instruction.instruction == 'PROFIT_TARGET':
                            # Calculate target profit based on average spend and ratio
                            # value1 = ratio (e.g., 0.10 for 10% profit)
                            if instruction.value1 and holding.shares > 0:
                                ratio = Decimal(str(instruction.value1))
                                # Base allowance is derived from the fund's aspirational spread.
                                base_allowance = fund.average_spend()
                                target_value = base_allowance * (Decimal('1.0') + ratio)
                                target_price = target_value / Decimal(str(holding.shares))
                                
                                if analyse_target(holding, target_price, sentiment, discovery):
                                    execute_sell(sa, fund, holding, f"{holding.stock.symbol} reached profit target (target price: ${target_price:.2f})")
                                    break
                            else:
                                logger.warning(f"PROFIT_TARGET invalid threshold (value1={instruction.value1}, buy_price={buy_price})")

                        elif instruction.instruction in ['STOP_AUGMENTING', 'PERCENTAGE_AUGMENTING']:
                            # Calculate augmenting stop: original_stop → buy_price over max_days
                            # value1 = original stop price, value2 = max_days
                            if instruction.value1 and buy_price:
                                max_days = int(instruction.value2) if instruction.value2 is not None else 28
                                if days_held <= max_days:
                                    progress = float(days_held) / float(max_days) if max_days > 0 else 1.0
                                    original_stop = float(instruction.value1)
                                    current_stop = original_stop + progress * (float(buy_price) - original_stop)
                                else:
                                    current_stop = float(buy_price)  # After max_days, stop = buy_price (break-even)
                                
                                if holding.stock.price < Decimal(str(current_stop)):
                                    execute_sell(sa, fund, holding,
                                                f"{holding.stock.symbol} hit augmenting stop-loss of ${current_stop:.2f} (day {days_held}/{max_days})")
                                    break
                            else:
                                logger.warning(f"{instruction.instruction} invalid threshold (value1={instruction.value1}, buy_price={buy_price})")
                        elif instruction.instruction == 'PERCENTAGE_REBUY':
                            # value1 = drop fraction from average (e.g. 0.02 = 2%); add one fund tranche at current price.
                            # value2 = max tranche count cap (default 3 when null). value2 <= 0 = unlimited (cash only).
                            if instruction.value1 and holding.shares > 0 and buy_price:
                                drop_pct = Decimal(str(instruction.value1))
                                drop_threshold = Decimal(str(buy_price)) * (Decimal("1.0") - drop_pct)
                                if holding.stock.price <= drop_threshold:
                                    stabilized = price_above_minutes_ago(
                                        holding.stock, minutes=REBUY_STABILIZE_MINUTES
                                    )
                                    if stabilized is not True:
                                        if stabilized is False:
                                            logger.info(
                                                "%s rebuy deferred: price $%.2f not above %dm ago "
                                                "(wait for stabilization)",
                                                holding.stock.symbol,
                                                float(holding.stock.price),
                                                REBUY_STABILIZE_MINUTES,
                                            )
                                        else:
                                            logger.info(
                                                "%s rebuy deferred: no %dm intraday reference",
                                                holding.stock.symbol,
                                                REBUY_STABILIZE_MINUTES,
                                            )
                                        continue

                                    tranche_amount = fund.average_spend() * Decimal(str(sentiment))
                                    if instruction.value2 is not None:
                                        max_tranches = Decimal(str(instruction.value2))
                                    else:
                                        max_tranches = Decimal("3")

                                    if max_tranches > 0:
                                        current_book = Decimal(str(holding.average_price)) * Decimal(str(holding.shares))
                                        max_book = tranche_amount * max_tranches
                                        if current_book >= max_book:
                                            logger.info(
                                                "%s rebuy skipped: max tranches reached (book=$%.2f cap=$%.2f tranches=%s)",
                                                holding.stock.symbol,
                                                float(current_book),
                                                float(max_book),
                                                int(max_tranches),
                                            )
                                            continue
                                        rebuy_amount = min(tranche_amount, max_book - current_book)
                                    else:
                                        rebuy_amount = tranche_amount

                                    if rebuy_amount <= 0:
                                        continue
                                    execute_buy(
                                        sa,
                                        fund,
                                        holding.stock,
                                        rebuy_amount,
                                        f"Rebuy ${rebuy_amount:.0f} after {drop_pct * 100:.0f}% drop vs avg",
                                        force=True,
                                        discovery=holding.discovery,
                                    )
                            elif instruction.value1 and holding.shares > 0:
                                logger.warning(
                                    "PERCENTAGE_REBUY instruction %s missing buy_price",
                                    instruction.id,
                                )

                        elif instruction.instruction == 'PROFIT_FLAT':
                            # Sell if price is flat (low volatility) and in profit
                            # value1 = X (percentage threshold for price range, e.g., 0.05 for 5%)
                            # value2 = Y (evaluation period in days, e.g., 30)
                            if instruction.value1 and instruction.value2:
                                range_threshold_pct = Decimal(str(instruction.value1))
                                evaluation_days = int(instruction.value2)
                                current_price = holding.stock.price
                                
                                # Only check if stock has been held for at least the evaluation period
                                if days_held < evaluation_days:
                                    continue  # Skip check - not enough time has passed
                                
                                # Only check if in profit
                                if current_price >= buy_price:
                                    try:
                                        import yfinance as yf
                                        ticker = yf.Ticker(holding.stock.symbol)
                                        # Get enough days (add buffer for weekends/holidays)
                                        period_days = evaluation_days + 10
                                        hist = ticker.history(period=f"{period_days}d", interval="1d")
                                        
                                        if not hist.empty and len(hist) >= evaluation_days:
                                            # Get last Y days of close prices
                                            prices = hist['Close'].tail(evaluation_days).values
                                            
                                            max_price = Decimal(str(float(prices.max())))
                                            min_price = Decimal(str(float(prices.min())))
                                            avg_price = Decimal(str(float(prices.mean())))
                                            
                                            # Calculate price range
                                            price_range = max_price - min_price
                                            
                                            # Check if range is within X% of average (flat)
                                            if avg_price > 0:
                                                range_threshold = avg_price * range_threshold_pct
                                                
                                                if price_range <= range_threshold:
                                                    execute_sell(sa, fund, holding, f"Flat near {range_threshold_pct:.0f}% over {evaluation_days} days")
                                                    break
                                    except Exception as e:
                                        logger.warning(f"Error checking PROFIT_FLAT for {holding.stock.symbol}: {e}")
                                        continue
                            else:
                                logger.warning(f"PROFIT_FLAT instruction {instruction.id} missing required fields (value1 or value2)")

                        elif instruction.instruction == 'AFTER_DAYS':
                            days_held = (timezone.now() - discovery.created).days
                            if instruction.value1 and days_held >= int(instruction.value1):
                                execute_sell(sa, fund, holding, f"{holding.stock.symbol} after holding for {days_held} days (target: {int(instruction.value1)} days)")
                                break

                        elif instruction.instruction == 'DESCENDING_TREND':

                            trend = holding.stock.calc_trend(hours=2)

                            if instruction.value1 is not None and trend is not None and trend < instruction.value1:

                                dropped_stocks.append(
                                    {
                                        "fund": fund,
                                        "holding": holding,
                                        "discovery": discovery,
                                        "buy_price": buy_price,
                                        "current_price": holding.stock.price,
                                        "trend": trend,
                                        "threshold": instruction.value1,
                                    }
                                )
                                break

                        elif instruction.instruction == 'NOT_TRENDING':
                            trending = holding.stock.is_trending()
                            if trending is False:  # Explicitly False (not None)
                                execute_sell(sa, fund, holding, f"{holding.stock.symbol} no longer trending (low volume)")
                                break

                        elif instruction.instruction == 'PEAKED':
                            if not peaked_allowed:
                                continue
                            if holding.stock.downturned(
                                discovery.created,
                                buy_price=buy_price,
                                giveback_pct=float(instruction.value1),
                                min_peak_gain_pct=float(instruction.value2 or 5.0),
                            ):
                                execute_sell(sa, fund, holding,f"{holding.stock.symbol} down {instruction.value1}% from peak")
                                break

                        elif instruction.instruction == 'END_DAY' and end_day:
                            avg = holding.average_price or discovery.price
                            target_px = _session_exit_threshold_px(instruction.value1, avg)
                            if target_px and holding.stock.price >= target_px:
                                execute_sell(
                                    sa, fund, holding,
                                    f"{holding.stock.symbol} end of day "
                                    f"${holding.stock.price:.2f} >= ${target_px:.2f} "
                                    f"({instruction.value1}× avg ${avg:.2f})",
                                )
                                break

                        elif instruction.instruction == 'END_WEEK' and end_day and end_week:
                            min_days = int(instruction.value2) if instruction.value2 is not None else 0
                            days_in_holding = (
                                (timezone.now() - holding.created).days
                                if holding.created
                                else 0
                            )
                            if days_in_holding >= min_days:
                                avg = holding.average_price or discovery.price
                                target_px = _session_exit_threshold_px(instruction.value1, avg)
                                if target_px and holding.stock.price >= target_px:
                                    execute_sell(
                                        sa, fund, holding,
                                        f"{holding.stock.symbol} end of week "
                                        f"${holding.stock.price:.2f} >= ${target_px:.2f} "
                                        f"({instruction.value1}× avg ${avg:.2f}, held {days_in_holding}d)",
                                    )
                                    break

                except Exception as e:
                    logger.error(
                        f"Error checking sell instructions for {holding.stock.symbol} (holding {holding.id}): {e}",
                        exc_info=True
                    )
                    # Continue processing other holdings
                    continue

    if dropped_stocks:
        logger.info("DT candidates collected: %s", len(dropped_stocks))
        analyse_drop(sa, dropped_stocks)


# Discovery new stock
def analyze_discovery(sa, funds, advisors):
    logger.info(f"Analyzing discovery for SA session {sa.id}")

    # Clear Polygon cache at start of discovery run to ensure fresh data
    from core.services.advisors.advisor import AdvisorBase  # TODO: Review
    AdvisorBase.clear_polygon_cache()

    # 1. Look for new stock
    for a in advisors:
        logger.info(f"Discovery ------------- {a.advisor.name}")
        a.discover(sa)

    # 2. Filter stocks to buy on a per user basis
    for fund in funds:
        logger.info(f"Buying ------------- {fund.name}")

        sentiment = factor_sentiment(fund)
        allowed_advisors = list(fund.advisors or [])
        if not allowed_advisors:
            logger.info("%s: no advisors configured; skip discovery buys", fund.name)
            continue

        # 1. Filter discoveries by allowed advisors (if specified)
        discoveries_qs = Discovery.objects.filter(sa=sa)
        
        # Get Advisor objects for the allowed advisor python_class values
        allowed_advisor_objects = Advisor.objects.filter(python_class__in=allowed_advisors)

        if allowed_advisor_objects.exists():
            discoveries_qs = discoveries_qs.filter(advisor__in=allowed_advisor_objects)
        else:
            logger.warning(f"{fund.name}: no matching advisor")
            continue

        # 3. Get filtered discoveries
        filtered_discoveries = list(
            discoveries_qs.select_related('advisor', 'stock', 'assessment')
        )
        
        if not filtered_discoveries:
            continue

        # Deduplicate by stock - only process each stock once per user
        seen_stocks = set()
        unique_discoveries = []
        for discovery in filtered_discoveries:
            if discovery.stock_id not in seen_stocks:
                seen_stocks.add(discovery.stock_id)
                unique_discoveries.append(discovery)

        if not unique_discoveries:
            continue

        allowance = fund.average_spend()
        allowance *= sentiment

        for discovery in unique_discoveries:

            stability, opportunity = discovery_axes(discovery)
            if stability is None and opportunity is None:
                continue

            if discovery_passes_risk_gate(discovery, fund.risk):
                explanation = discovery.explanation.split(" | ")[0].strip()
                execute_buy(sa, fund, discovery.stock, allowance, explanation, discovery=discovery)
            else:
                logger.info(
                    "%s: %s discovery %s risk gate fail (SO %s)",
                    fund.name,
                    discovery.advisor.name,
                    discovery.stock.symbol,
                    so_gate_fail_display(
                        stability,
                        opportunity,
                        fund.risk,
                        weight=discovery.weight,
                    ),
                )


