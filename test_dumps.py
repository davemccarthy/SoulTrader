#!/usr/bin/env python
"""
Down-in-the-dumps scanner - 6M low proximity strategy.

Finds liquid, middle-of-the-road stocks that are:
- In a given price band (default: $8–$80)
- Have strong absolute liquidity (min average volume threshold)
- Are trading at or near their 6-month low
- Are not obviously distressed (basic profitability checks)
- Optionally undervalued vs a simple trailing-EPS fair value

Scoring: each candidate gets health_score (liquidity + stability + profitability),
valuation_score (cheap vs fair value), and low_score (proximity to 6M low).
Composite total_score = 0.35*health + 0.25*valuation + 0.25*low_6m + 0.15*low_2w.
Candidates with total_score >= SCORE_THRESHOLD (default 0.7) are eligible; up to
MAX_LLM_CANDIDATES (default 10) are sent to the LLM for consensus (Buy/Strong Buy pass).

Usage examples:
  python test_dumps.py
  python test_dumps.py --date 2026-03-10
  python test_dumps.py --date 2026-03-10 --candidates GRNY METU --verbose
  python test_dumps.py --top-n 10 --no-llm
"""

import os
import sys
import re
import json
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from polygon import RESTClient

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# ------------------------------
# CONFIG
# ------------------------------
MIN_PRICE = 8.0
MAX_PRICE = 80.0
MIN_AVG_VOLUME = 2_000_000  # 2M+ average daily volume

# 6M lookback: ~6 calendar months (~126 trading days)
SIX_MONTH_DAYS = 180

# How close to the 6M low we require the current price to be
MAX_DIST_FROM_6M_LOW_PCT = 0.15  # within 15% of 6M low (loosened)

# Optional filter: exclude names very near 6M high
MAX_DIST_FROM_6M_HIGH_PCT = 0.85  # price / high_6m must be <= 0.85 (not near highs)

# Valuation: standalone rough valuation current_price / fair_value
MAX_VALUATION_RATIO = 1.0   # <1.0 = undervalued, <0.75 = deep value

# LLM step: only send candidates with total_score >= this (0–1 scale)
SCORE_THRESHOLD = 0.7
# Max number of candidates to send to the LLM in a single request
MAX_LLM_CANDIDATES = 10


@dataclass
class DumpCandidate:
    ticker: str
    price: float
    low_6m: float
    high_6m: float
    ratio_to_low: float
    ratio_to_high: float
    avg_volume: float
    valuation_ratio: float
    profitable: bool
    rel_volume: float = 0.0
    health_score: float = 0.0
    valuation_score: float = 0.0
    low_score: float = 0.0
    total_score: float = 0.0


def get_last_trading_day_for_date(target_date: Optional[str] = None) -> Optional[str]:
    """
    Get the last trading day relative to a target date (or today) using simple weekday logic.

    For this standalone script we avoid importing Django/AdvisorBase; we mirror the logic from
    AdvisorBase.get_last_trading_day in a lightweight form.
    """
    if target_date:
        try:
            ref = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            return None
    else:
        ref = datetime.now().date()

    # Previous calendar day
    previous_day = ref - timedelta(days=1)
    # If yesterday was Sunday, go back to Friday
    if previous_day.weekday() == 6:  # Sunday
        previous_day = previous_day - timedelta(days=2)
    # If yesterday was Saturday, go back to Friday
    elif previous_day.weekday() == 5:  # Saturday
        previous_day = previous_day - timedelta(days=1)

    return previous_day.strftime("%Y-%m-%d")


def fetch_polygon_universe(reference_date: str,
                           min_price: float,
                           max_price: float,
                           min_volume: int,
                           verbose: bool = False) -> pd.DataFrame:
    """
    Fetch a broad US stock universe for a given date using Polygon grouped daily aggs.
    Mirrors the approach used in test_oscilla.build_candidates.
    """
    polygon_api_key = os.getenv("POLYGON_API_KEY")
    if not polygon_api_key:
        if verbose:
            print("❌ POLYGON_API_KEY not set; falling back to empty universe")
        return pd.DataFrame()

    try:
        client = RESTClient(polygon_api_key)
        if verbose:
            print(f"   Fetching all stocks for {reference_date} using Polygon (grouped daily aggs)...")

        aggs = client.get_grouped_daily_aggs(
            locale="us",
            date=reference_date,
            adjusted=False,
        )

        rows = []
        for agg in aggs:
            if (
                min_price <= agg.close <= max_price
                and agg.volume is not None
                and agg.volume >= min_volume
            ):
                rows.append(
                    {
                        "ticker": agg.ticker,
                        "price": float(agg.close),
                        "today_volume": int(agg.volume),
                    }
                )

        df = pd.DataFrame(rows)
        if verbose and not df.empty:
            print(
                f"   Found {len(df)} stocks in ${min_price:.2f}-${max_price:.2f} price range "
                f"with volume >= {min_volume:,}"
            )
        return df
    except Exception as e:
        if verbose:
            print(f"❌ Error fetching stocks from Polygon: {e}")
        return pd.DataFrame()


def get_yf_history(ticker: str, end_date: str, lookback_days: int) -> pd.DataFrame:
    """
    Simple yfinance history wrapper for 6M analysis.
    """
    try:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=lookback_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        t = yf.Ticker(ticker)
        hist = t.history(start=start_date, end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"), raise_errors=False)

        if hist is None or hist.empty:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "date": hist.index,
                "open": hist["Open"].values,
                "close": hist["Close"].values,
                "high": hist["High"].values,
                "low": hist["Low"].values,
                "volume": hist["Volume"].values,
            }
        )
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def is_profitable(info: dict) -> bool:
    """
    Mirror Oscilla's simple profitability test: any of trailing EPS, net income, or profit margin >= 0.
    """
    trailing_eps = info.get("trailingEps")
    net_income = info.get("netIncomeToCommon") or info.get("netIncome")
    profit_margin = info.get("profitMargins")

    return bool(
        (trailing_eps is not None and trailing_eps >= 0)
        or (net_income is not None and net_income >= 0)
        or (profit_margin is not None and profit_margin >= 0)
    )


def simple_valuation_ratio(info: dict) -> float:
    """
    Standalone rough valuation: current_price / (trailingEps * simple_PE).

    Falls back to 1.0 (neutral) if we can't compute anything sensible. This avoids
    pulling in AdvisorBase/Django while still giving a basic 'not obviously overvalued'
    filter aligned with the down-in-the-dumps idea.
    """
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    eps = info.get("trailingEps")
    if current_price is None or current_price <= 0 or eps is None or eps <= 0:
        return 1.0

    SIMPLE_PE = 15.0
    fair_value = eps * SIMPLE_PE
    if fair_value <= 0:
        return 1.0
    return float(current_price / fair_value)


def compute_scores(
    avg_volume: float,
    rel_volume: float,
    valuation_ratio: float,
    ratio_to_low: float,
    ratio_to_low_2w: float,
    min_avg_volume: int = MIN_AVG_VOLUME,
) -> tuple[float, float, float, float]:
    """
    Compute health (liquidity + stability + profitability), valuation, and proximity to lows.

    Components (all in [0, 1]):
      - health_score: liquidity + stability + profitability
      - valuation_score: cheaper vs fair value
      - low_score: proximity to 6M low
      - short-term low (2-week) proximity blended into total_score

    total_score = 0.35*health + 0.25*valuation + 0.25*low_6m + 0.15*low_2w.
    """
    # Health: liquidity (cap at 5M), stability (rel_volume near 1), profitability (already filtered = 1)
    liquidity = min(1.0, avg_volume / max(5_000_000, min_avg_volume))
    stability = 1.0 if 0.5 <= rel_volume <= 1.5 else max(0.0, 1.0 - abs(rel_volume - 1.0))
    health_score = (liquidity + stability + 1.0) / 3.0

    # Valuation: lower ratio = better value; score = max(0, 1 - ratio)
    valuation_score = max(0.0, 1.0 - valuation_ratio)

    # 6M low: ratio_to_low 1.0 = at low (best), 1.0 + MAX_DIST_FROM_6M_LOW_PCT = worst in band
    low_6m_score = max(
        0.0,
        1.0 - (ratio_to_low - 1.0) / max(MAX_DIST_FROM_6M_LOW_PCT, 1e-6),
    )

    # 2-week low: same structure but on short window
    low_2w_score = max(
        0.0,
        1.0 - (ratio_to_low_2w - 1.0) / max(MAX_DIST_FROM_6M_LOW_PCT, 1e-6),
    )

    total_score = (
        0.35 * health_score
        + 0.25 * valuation_score
        + 0.25 * low_6m_score
        + 0.15 * low_2w_score
    )
    return health_score, valuation_score, low_6m_score, total_score


# Optional Gemini for consensus step
try:
    import google.generativeai as genai
    from google.api_core import exceptions as genai_exceptions
except ImportError:
    genai = None
    genai_exceptions = None

LLM_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM response, handling markdown code blocks."""
    if not text or not text.strip():
        return None
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def fetch_consensus_llm(
    tickers: List[str],
    df_context: pd.DataFrame,
    timeout: float = 90.0,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Ask Gemini for analyst consensus (Strong Buy / Buy / Hold / Sell) per ticker and a confidence score (0–1).
    Returns dict: ticker -> {"consensus": str, "summary": str, "confidence": float} for every ticker in the response.
    If Gemini is unavailable or errors, returns empty dict.
    """
    if genai is None or genai_exceptions is None:
        if verbose:
            print("   LLM: google.generativeai not installed. Skipping consensus step.")
        return {}
    api_key = os.environ.get("GEMINI_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        if verbose:
            print("   LLM: GEMINI_KEY / GEMINI_API_KEY not set. Skipping consensus step.")
        return {}

    # Build one-line context per ticker from df_context
    lines = []
    for t in tickers:
        row = df_context.loc[df_context["ticker"] == t]
        if row.empty:
            lines.append(f"  {t}: (no context)")
        else:
            r = row.iloc[0]
            lines.append(
                f"  {t}: price=${r['price']:.2f}, ratio_to_6m_low={r['ratio_to_low']:.2f}, "
                f"valuation_ratio={r['valuation_ratio']:.2f}, total_score={r['total_score']:.3f}"
            )
    context_block = "\n".join(lines)

    prompt = f"""You are an equity research assistant. For each of the following stocks, determine the current analyst consensus recommendation (e.g. Strong Buy, Buy, Hold, Sell) based on typical analyst ratings and recent sentiment. These stocks are candidates in a "down in the dumps" screen (near 6-month lows, liquid, profitable).

Stocks and context:
{context_block}

TASK: For each ticker, return:
1) consensus: one of "Strong Buy", "Buy", "Hold", or "Sell" (use standard analyst wording).
2) summary: one short sentence on why this rating (catalysts, risks, or sentiment).
3) confidence: a number from 0.0 to 1.0 indicating your conviction in this rating (1.0 = high conviction, 0.5 = moderate, 0.0 = low conviction / many caveats).

OUTPUT FORMAT: Respond with ONLY a single JSON object. Keys are ticker symbols. Each value is an object with "consensus", "summary", and "confidence" (number).
Example:
{{"AAPL": {{"consensus": "Buy", "summary": "...", "confidence": 0.8}}, "MSFT": {{"consensus": "Strong Buy", "summary": "...", "confidence": 0.9}}}}

Return valid JSON only, no other text."""

    genai.configure(api_key=api_key)
    for model in LLM_MODELS:
        try:
            if verbose:
                print(f"   LLM: Calling {model} for consensus...")
            response = genai.GenerativeModel(model).generate_content(
                prompt,
                request_options={"timeout": timeout},
            )
            if not response.candidates or not response.candidates[0].content.parts:
                continue
            text = response.candidates[0].content.parts[0].text
            parsed = _extract_json(text)
            if not parsed or not isinstance(parsed, dict):
                continue
            # Normalize to ticker -> {consensus, summary, confidence} for ALL tickers in response
            result = {}
            for t in tickers:
                t_upper = t.upper()
                data = parsed.get(t) or parsed.get(t_upper)
                if not data or not isinstance(data, dict):
                    continue
                consensus = (data.get("consensus") or "").strip()
                raw_conf = data.get("confidence")
                try:
                    confidence = float(raw_conf) if raw_conf is not None else 0.75
                    confidence = max(0.0, min(1.0, confidence))
                except (TypeError, ValueError):
                    confidence = 0.75
                result[t_upper] = {
                    "consensus": consensus or "N/A",
                    "summary": (data.get("summary") or "")[:200],
                    "confidence": confidence,
                }
            return result
        except (genai_exceptions.ServiceUnavailable, genai_exceptions.ResourceExhausted,
                genai_exceptions.DeadlineExceeded, genai_exceptions.InternalServerError) as e:
            if verbose:
                print(f"   LLM: {model} unavailable ({e}). Trying next.")
            continue
        except Exception as e:
            if verbose:
                print(f"   LLM: Error {e}")
            return {}
    return {}


def analyze_dumps(reference_date: Optional[str] = None,
                  min_price: float = MIN_PRICE,
                  max_price: float = MAX_PRICE,
                  min_avg_volume: int = MIN_AVG_VOLUME,
                  rel_volume_min: float = 0.3,
                  rel_volume_max: float = 2.0,
                  valuation_threshold: float = MAX_VALUATION_RATIO,
                  candidate_symbols: Optional[List[str]] = None,
                  max_stocks: Optional[int] = None,
                  verbose: bool = False) -> pd.DataFrame:
    """
    Main 6M-low scanner.

    - Builds a universe (Polygon or candidate list).
    - Filters by 6M-low proximity and 6M-high distance.
    - Requires high average volume and reasonable today's volume vs average.
    - Keeps only profitable, non-distressed, reasonably valued names.
    """
    ref_date = get_last_trading_day_for_date(reference_date)
    if not ref_date:
        print("❌ No valid trading date available")
        return pd.DataFrame()

    if verbose:
        print(f"📅 Using reference date: {ref_date}")
        print(
            f"Settings:\n"
            f"  Price range: ${min_price:.2f} - ${max_price:.2f}\n"
            f"  Min avg volume: {min_avg_volume:,}\n"
            f"  6M low proximity: <= {MAX_DIST_FROM_6M_LOW_PCT*100:.1f}% above low\n"
            f"  6M high cap: <= {MAX_DIST_FROM_6M_HIGH_PCT*100:.1f}% of high\n"
            f"  Max valuation ratio: {valuation_threshold:.2f}"
        )

    # Build daily universe (Polygon or provided candidates)
    if candidate_symbols:
        if verbose:
            print(f"\n📊 Building universe from provided candidates: {', '.join(candidate_symbols)}")

        rows = []
        for ticker in candidate_symbols:
            t = ticker.upper()
            df_hist = get_yf_history(t, ref_date, lookback_days=40)
            if df_hist.empty:
                if verbose:
                    print(f"   {t}: ✗ No recent data")
                continue
            last_row = df_hist.sort_values("date").iloc[-1]
            rows.append(
                {
                    "ticker": t,
                    "price": float(last_row["close"]),
                    "today_volume": int(last_row["volume"]),
                }
            )
        universe = pd.DataFrame(rows)
    else:
        if verbose:
            print("\n📊 Building universe from Polygon...")
        universe = fetch_polygon_universe(ref_date, min_price, max_price, min_avg_volume, verbose=verbose)

    if universe.empty:
        if verbose:
            print("❌ No universe stocks available")
        return pd.DataFrame()

    if max_stocks and len(universe) > max_stocks:
        universe = universe.head(max_stocks)
        if verbose:
            print(f"   Limiting to first {max_stocks} stocks for analysis")

    candidates: List[DumpCandidate] = []

    for i, (_, row) in enumerate(universe.iterrows(), 1):
        ticker = row["ticker"]
        price = float(row["price"])
        today_volume = int(row["today_volume"])

        if verbose and i % 50 == 0:
            print(f"   Analyzed {i}/{len(universe)} stocks...")

        # 6M history
        df_6m = get_yf_history(ticker, ref_date, lookback_days=SIX_MONTH_DAYS)
        if df_6m.empty or len(df_6m) < 40:
            if verbose:
                print(f"   {ticker}: ✗ Insufficient 6M history ({len(df_6m)} points)")
            continue

        df_6m = df_6m.sort_values("date")
        low_6m = float(df_6m["low"].min())
        high_6m = float(df_6m["high"].max())
        if low_6m <= 0 or high_6m <= 0:
            continue

        current_price = float(df_6m["close"].iloc[-1])

        # Short-term (~2-week) low for additional proximity signal
        recent_window = df_6m.tail(10)
        low_2w = float(recent_window["low"].min())
        if low_2w <= 0:
            continue
        ratio_to_low_2w = current_price / low_2w
        avg_volume = float(df_6m["volume"].tail(40).mean())
        if avg_volume < min_avg_volume:
            if verbose:
                print(f"   {ticker}: ✗ Avg volume {avg_volume:,.0f} < MIN_AVG_VOLUME {min_avg_volume:,}")
            continue

        rel_volume = today_volume / avg_volume if avg_volume > 0 else 0.0
        if not (rel_volume_min <= rel_volume <= rel_volume_max):
            if verbose:
                print(
                    f"   {ticker}: ✗ Rel volume {rel_volume:.2f} not in "
                    f"[{rel_volume_min:.1f}, {rel_volume_max:.1f}] "
                    f"(avg={avg_volume:,.0f}, today={today_volume:,.0f})"
                )
            continue

        ratio_to_low = current_price / low_6m
        ratio_to_high = current_price / high_6m

        if ratio_to_low > (1.0 + MAX_DIST_FROM_6M_LOW_PCT):
            if verbose:
                print(
                    f"   {ticker}: ✗ Too far from 6M low (price {current_price:.2f}, "
                    f"low_6m {low_6m:.2f}, ratio {ratio_to_low:.2f})"
                )
            continue

        if ratio_to_high > MAX_DIST_FROM_6M_HIGH_PCT:
            if verbose:
                print(
                    f"   {ticker}: ✗ Too close to 6M high (price {current_price:.2f}, "
                    f"high_6m {high_6m:.2f}, ratio {ratio_to_high:.2f})"
                )
            continue

        # Fundamentals via yfinance
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info or {}
        except Exception:
            info = {}

        profitable = is_profitable(info)
        if not profitable:
            if verbose:
                print(f"   {ticker}: ✗ Not profitable by EPS/netIncome/margins")
            continue

        # Standalone valuation
        valuation_ratio = simple_valuation_ratio(info)

        if valuation_ratio > valuation_threshold:
            if verbose:
                print(
                    f"   {ticker}: ✗ Valuation ratio {valuation_ratio:.2f} > "
                    f"threshold {valuation_threshold:.2f}"
                )
            continue

        health_score, valuation_score, low_score, total_score = compute_scores(
            avg_volume,
            rel_volume,
            valuation_ratio,
            ratio_to_low,
            ratio_to_low_2w,
            min_avg_volume,
        )

        candidates.append(
            DumpCandidate(
                ticker=ticker,
                price=current_price,
                low_6m=low_6m,
                high_6m=high_6m,
                ratio_to_low=ratio_to_low,
                ratio_to_high=ratio_to_high,
                avg_volume=avg_volume,
                valuation_ratio=valuation_ratio,
                profitable=profitable,
                rel_volume=rel_volume,
                health_score=health_score,
                valuation_score=valuation_score,
                low_score=low_score,
                total_score=total_score,
            )
        )

    if not candidates:
        if verbose:
            print("\nNo 'down in the dumps' candidates found for given criteria.")
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            {
                "ticker": c.ticker,
                "price": round(c.price, 2),
                "low_6m": round(c.low_6m, 2),
                "high_6m": round(c.high_6m, 2),
                "ratio_to_low": round(c.ratio_to_low, 2),
                "ratio_to_high": round(c.ratio_to_high, 2),
                "avg_volume": int(c.avg_volume),
                "valuation_ratio": round(c.valuation_ratio, 2),
                "health_score": round(c.health_score, 3),
                "valuation_score": round(c.valuation_score, 3),
                "low_score": round(c.low_score, 3),
                "total_score": round(c.total_score, 3),
            }
            for c in candidates
        ]
    )

    df = df.sort_values(by="total_score", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Down-in-the-dumps 6M-low scanner for liquid, non-distressed stocks."
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Reference date (YYYY-MM-DD). Defaults to last trading day.",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=MIN_PRICE,
        help=f"Minimum stock price (default: {MIN_PRICE})",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=MAX_PRICE,
        help=f"Maximum stock price (default: {MAX_PRICE})",
    )
    parser.add_argument(
        "--min-avg-volume",
        type=int,
        default=MIN_AVG_VOLUME,
        help=f"Minimum average daily volume (default: {MIN_AVG_VOLUME})",
    )
    parser.add_argument(
        "--valuation-threshold",
        type=float,
        default=MAX_VALUATION_RATIO,
        help=f"Maximum valuation ratio current_price/fair_value (default: {MAX_VALUATION_RATIO}) "
             f"using a simple trailing EPS * PE model.",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        nargs="+",
        help="Optional specific tickers to test instead of full Polygon universe.",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        help="Maximum number of universe stocks to analyze (for speed).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=MAX_LLM_CANDIDATES,
        help=f"Max candidates to send to LLM (default: {MAX_LLM_CANDIDATES}). Only those with total_score >= --score-threshold are considered.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=SCORE_THRESHOLD,
        help=f"Minimum total_score (0–1) to be eligible for LLM (default: {SCORE_THRESHOLD}).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM consensus step; only print scored candidates.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    df = analyze_dumps(
        reference_date=args.date,
        min_price=args.min_price,
        max_price=args.max_price,
        min_avg_volume=args.min_avg_volume,
        valuation_threshold=args.valuation_threshold,
        candidate_symbols=args.candidates,
        max_stocks=args.max_stocks,
        verbose=args.verbose,
    )

    if df.empty:
        print("\nNo qualifying 'down in the dumps' candidates found.")
        return

    display_cols = [
        "ticker", "price", "low_6m", "high_6m", "ratio_to_low", "ratio_to_high",
        "avg_volume", "valuation_ratio", "health_score", "valuation_score", "low_score", "total_score",
    ]

    print("\n" + "=" * 80)
    print("DOWN-IN-THE-DUMPS CANDIDATES (6M LOW STRATEGY) — SCORED")
    print("=" * 80)
    print(df[display_cols].to_string(index=False))

    score_threshold = args.score_threshold
    df_eligible = df[df["total_score"] >= score_threshold]
    top_n = max(1, args.top_n)
    df_top = df_eligible.head(top_n)
    tickers_top = df_top["ticker"].tolist()

    if args.no_llm:
        print("\n" + "-" * 80)
        print(f"Top {top_n} by total score (score >= {score_threshold}, --no-llm: LLM step skipped)")
        print("-" * 80)
        print(df_top[display_cols].to_string(index=False))
        return

    if df_top.empty:
        print("\n" + "-" * 80)
        print(f"No candidates with total_score >= {score_threshold}. Nothing to send to LLM.")
        return

    n_send = len(tickers_top)
    print(f"\nSending {n_send} candidate(s) to LLM (total_score >= {score_threshold}, max {top_n}).")
    consensus_all = fetch_consensus_llm(tickers_top, df_top, verbose=args.verbose)
    if not consensus_all:
        print("\n" + "-" * 80)
        print("LLM consensus unavailable. Showing top N by score only.")
        print("-" * 80)
        print(df_top[display_cols].to_string(index=False))
        return

    # Filter to Buy/Strong Buy for the table
    consensus_map = {
        t: c for t, c in consensus_all.items()
        if (c.get("consensus") or "").strip().lower() in {"buy", "strong buy"}
    }

    df_after_llm = df_top[df_top["ticker"].str.upper().isin(consensus_map)].copy()
    if df_after_llm.empty:
        print("\n" + "-" * 80)
        print("No top-N candidates had consensus Buy or Strong Buy. Showing top N by score.")
        print("-" * 80)
        print(df_top[display_cols].to_string(index=False))
    else:
        print("\n" + "=" * 80)
        print("AFTER LLM CONSENSUS FILTER (BUY / STRONG BUY ONLY)")
        print("=" * 80)
        print(df_after_llm[display_cols].to_string(index=False))

    print("\nConsensus, confidence, and summary (all candidates sent to LLM):")
    for t in tickers_top:
        t_upper = t.upper()
        c = consensus_all.get(t_upper)
        if c:
            conf = c.get("confidence")
            conf_str = f" (confidence {conf:.2f})" if isinstance(conf, (int, float)) else ""
            print(f"  {t}: {c.get('consensus', 'N/A')}{conf_str} — {c.get('summary', '')}")
        else:
            print(f"  {t}: N/A (no response)")


if __name__ == "__main__":
    main()

