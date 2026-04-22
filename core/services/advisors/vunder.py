"""
Vunder Advisor - Down-in-the-dumps style value discovery (WIP)

New paradigm (high level):
- Build a liquid, mid-priced universe from Polygon via services.financial.polygon.get_filtered_stocks().
- For each stock, fetch ~6M daily history from yfinance and basic fundamentals.
- Require profitability and reasonable valuation via AdvisorBase.evaluate_stock().
- Score candidates by liquidity/health, valuation, and proximity to 6M/2-week lows.
- Pre-LLM long list: sort by total_score and keep the strongest names.
- LLM: use AdvisorBase.ask_gemini for consensus; keep Buy/Strong Buy only.
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from core.services.advisors.advisor import AdvisorBase, register
from core.services.financial import polygon as financial_polygon
from core.services.financial import yahoo as financial_yahoo

logger = logging.getLogger(__name__)

# ------------------------------
# CONFIG (aligned with test_dumps)
# ------------------------------
MIN_PRICE = 8.0
MAX_PRICE = 80.0
MIN_AVG_VOLUME = 2_000_000  # 2M+ average daily volume

# How close to the 6M low we require the current price to be
MAX_DIST_FROM_6M_LOW_PCT = 0.15  # within 15% of 6M low

# Optional filter: exclude names very near 6M high
MAX_DIST_FROM_6M_HIGH_PCT = 0.85  # price / high_6m must be <= 0.85 (not near highs)

# Valuation: ROE-based valuation ratio from AdvisorBase.evaluate_stock
MAX_VALUATION_RATIO = 1.0   # <1.0 = undervalued, <0.75 = deep value

# Scoring: only send candidates with total_score >= this (0–1 scale) to LLM step
SCORE_THRESHOLD = 0.7

# Caps
MAX_UNIVERSE = 500          # max Polygon names to analyze per run
MAX_LLM_CANDIDATES = 10     # max candidates to send to LLM per run


class Vunder(AdvisorBase):
    PROCESS_CUTOFF_HOUR_UTC = 9

    # ---------
    # Core helpers
    # ---------

    def _get_6m_history(self, ticker: str) -> pd.DataFrame:
        """
        Fetch ~6 months of daily history for a ticker from yfinance.
        Returns DataFrame with columns: date, open, close, high, low, volume.
        """
        return financial_yahoo.get_6m_history(ticker)

    def _compute_scores(
        self,
        avg_volume: float,
        rel_volume: float,
        valuation_ratio: float,
        ratio_to_low_6m: float,
        ratio_to_low_2w: float,
        min_avg_volume: int = MIN_AVG_VOLUME,
    ):
        """
        Compute health (liquidity + stability + profitability), valuation, and proximity to lows.
        Returns (health_score, valuation_score, low_6m_score, total_score), all in [0, 1].

        Mirrors the scoring weights from test_dumps:
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
            1.0 - (ratio_to_low_6m - 1.0) / max(MAX_DIST_FROM_6M_LOW_PCT, 1e-6),
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

    def _is_profitable(self, info: Dict[str, Any]) -> bool:
        """
        Simple profitability test: any of trailing EPS, net income, or profit margin >= 0.
        Mirrors test_dumps.is_profitable.
        """
        trailing_eps = info.get("trailingEps")
        net_income = info.get("netIncomeToCommon") or info.get("netIncome")
        profit_margin = info.get("profitMargins")

        return bool(
            (trailing_eps is not None and trailing_eps >= 0)
            or (net_income is not None and net_income >= 0)
            or (profit_margin is not None and profit_margin >= 0)
        )

    def _build_quant_static_list(self, sa) -> pd.DataFrame:
        """
        Pre-LLM long list builder (STATIC VERSION, no Polygon).

        - Starts from a fixed list of tickers.
        - For each stock, fetches ~6M history + basic fundamentals from yfinance.
        - Applies profitability, valuation, and 6M/2w low proximity filters.
        - Returns DataFrame sorted by total_score desc, filtered to total_score >= SCORE_THRESHOLD.
        """
        # Static test universe (adjust as you like)
        tickers = [
            "LYFT", "GIS", "GPK", "SYF", "CPB",
            "WFC", "UPWK", "WHR", "OBDC", "DXC", "GNTX",
        ]

        rows: List[Dict[str, Any]] = []

        for ticker in tickers:
            ticker = ticker.upper()

            df_6m = self._get_6m_history(ticker)
            if df_6m.empty or len(df_6m) < 40:
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

            ratio_to_low = current_price / low_6m
            ratio_to_high = current_price / high_6m
            ratio_to_low_2w = current_price / low_2w

            if ratio_to_low > (1.0 + MAX_DIST_FROM_6M_LOW_PCT):
                continue
            if ratio_to_high > MAX_DIST_FROM_6M_HIGH_PCT:
                continue

            avg_volume_6m = float(df_6m["volume"].tail(40).mean())
            if avg_volume_6m < MIN_AVG_VOLUME:
                continue

            # For static testing we don't have today_volume; approximate rel_volume ~ 1.0
            rel_volume = 1.0

            # Fundamentals + valuation via AdvisorBase.evaluate_stock
            info = financial_yahoo.get_ticker_info(ticker)

            if not self._is_profitable(info):
                continue

            valuation_ratio = self.evaluate_stock(ticker_symbol=ticker, info=info)
            if valuation_ratio > MAX_VALUATION_RATIO:
                continue

            health_score, valuation_score, low_score, total_score = self._compute_scores(
                avg_volume_6m,
                rel_volume,
                valuation_ratio,
                ratio_to_low,
                ratio_to_low_2w,
                MIN_AVG_VOLUME,
            )

            rows.append(
                {
                    "ticker": ticker,
                    "price": round(current_price, 2),
                    "low_6m": round(low_6m, 2),
                    "high_6m": round(high_6m, 2),
                    "ratio_to_low": round(ratio_to_low, 2),
                    "ratio_to_high": round(ratio_to_high, 2),
                    "avg_volume": int(avg_volume_6m),
                    "valuation_ratio": float(round(valuation_ratio, 2)),
                    "health_score": round(health_score, 3),
                    "valuation_score": round(valuation_score, 3),
                    "low_score": round(low_score, 3),
                    "total_score": round(total_score, 3),
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df[df["total_score"] >= SCORE_THRESHOLD]
        if df.empty:
            return df

        df = df.sort_values(by="total_score", ascending=False).reset_index(drop=True)
        return df

    def _build_quant_long_list(self, sa) -> pd.DataFrame:
        """
        Pre-LLM long list builder.

        - Uses services.financial.polygon.get_filtered_stocks() to fetch last-trading-day Polygon universe.
        - For each stock, fetches ~6M history + basic fundamentals from yfinance.
        - Applies profitability, valuation, and 6M/2w low proximity filters.
        - Returns DataFrame sorted by total_score desc, filtered to total_score >= SCORE_THRESHOLD.
        """
        df_universe = financial_polygon.get_filtered_stocks(
            min_price=MIN_PRICE,
            max_price=MAX_PRICE,
            min_volume=MIN_AVG_VOLUME,
        )
        if df_universe is None or df_universe.empty:
            logger.info("Vunder: no Polygon universe available")
            return pd.DataFrame()

        # Cap raw universe size for speed (highest volume first)
        if MAX_UNIVERSE and len(df_universe) > MAX_UNIVERSE:
            df_universe = df_universe.sort_values("today_volume", ascending=False).head(MAX_UNIVERSE)

        rows: List[Dict[str, Any]] = []

        for _, row in df_universe.iterrows():
            ticker = row["ticker"]
            price = float(row["price"])
            today_volume = int(row["today_volume"])

            df_6m = self._get_6m_history(ticker)
            if df_6m.empty or len(df_6m) < 40:
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

            ratio_to_low = current_price / low_6m
            ratio_to_high = current_price / high_6m
            ratio_to_low_2w = current_price / low_2w

            if ratio_to_low > (1.0 + MAX_DIST_FROM_6M_LOW_PCT):
                continue
            if ratio_to_high > MAX_DIST_FROM_6M_HIGH_PCT:
                continue

            avg_volume_6m = float(df_6m["volume"].tail(40).mean())
            if avg_volume_6m < MIN_AVG_VOLUME:
                continue
            rel_volume = today_volume / avg_volume_6m if avg_volume_6m > 0 else 0.0

            # Fundamentals + valuation via AdvisorBase.evaluate_stock
            info = financial_yahoo.get_ticker_info(ticker)

            if not self._is_profitable(info):
                continue

            valuation_ratio = self.evaluate_stock(ticker_symbol=ticker, info=info)
            if valuation_ratio > MAX_VALUATION_RATIO:
                continue

            health_score, valuation_score, low_score, total_score = self._compute_scores(
                avg_volume_6m,
                rel_volume,
                valuation_ratio,
                ratio_to_low,
                ratio_to_low_2w,
                MIN_AVG_VOLUME,
            )

            rows.append(
                {
                    "ticker": ticker,
                    "price": round(current_price, 2),
                    "low_6m": round(low_6m, 2),
                    "high_6m": round(high_6m, 2),
                    "ratio_to_low": round(ratio_to_low, 2),
                    "ratio_to_high": round(ratio_to_high, 2),
                    "avg_volume": int(avg_volume_6m),
                    "valuation_ratio": float(round(valuation_ratio, 2)),
                    "health_score": round(health_score, 3),
                    "valuation_score": round(valuation_score, 3),
                    "low_score": round(low_score, 3),
                    "total_score": round(total_score, 3),
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df[df["total_score"] >= SCORE_THRESHOLD]
        if df.empty:
            return df

        df = df.sort_values(by="total_score", ascending=False).reset_index(drop=True)
        return df

    def _llm_consensus(self, df_top: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Call Gemini via AdvisorBase.ask_gemini to get consensus and confidence for each ticker.
        Returns dict[ticker_upper] = {consensus, summary, confidence}.
        """
        tickers = df_top["ticker"].tolist()
        if not tickers:
            return {}

        # Build one-line context per ticker (mirrors test_dumps)
        lines = []
        for _, row in df_top.iterrows():
            lines.append(
                f"  {row['ticker']}: price=${row['price']:.2f}, "
                f"ratio_to_6m_low={row['ratio_to_low']:.2f}, "
                f"valuation_ratio={row['valuation_ratio']:.2f}, "
                f"total_score={row['total_score']:.3f}"
            )
        context_block = "\n".join(lines)

        prompt = f"""
You are an equity research assistant. For each of the following stocks, determine the current analyst-style
consensus recommendation (Strong Buy, Buy, Hold, Sell) based on typical analyst ratings and recent sentiment.

These stocks are candidates in a "down in the dumps" screen (near 6-month lows, liquid, profitable).

Stocks and context:
{context_block}

TASK: For each ticker, return:
1) consensus: one of "Strong Buy", "Buy", "Hold", or "Sell".
2) summary: one short sentence on why this rating (catalysts, risks, or sentiment).
3) confidence: a number from 0.0 to 1.0 indicating your conviction (1.0 = high, 0.5 = moderate, 0.0 = low).

OUTPUT FORMAT: Respond with ONLY a single JSON object.
Keys are ticker symbols. Each value is an object with "consensus", "summary", and "confidence" (number).

Example:
{{
  "AAPL": {{"consensus": "Buy", "summary": "...", "confidence": 0.8}},
  "MSFT": {{"consensus": "Strong Buy", "summary": "...", "confidence": 0.9}}
}}
"""

        model, results = self.ask_gemini(prompt, timeout=120.0, use_search=False)
        if not results or not isinstance(results, dict):
            logger.info("Vunder: no usable LLM consensus")
            return {}

        consensus_map: Dict[str, Dict[str, Any]] = {}
        for t in tickers:
            t_upper = t.upper()
            data = results.get(t) or results.get(t_upper)
            if not data or not isinstance(data, dict):
                continue
            consensus = (data.get("consensus") or "").strip()
            raw_conf = data.get("confidence")
            try:
                confidence = float(raw_conf) if raw_conf is not None else 0.75
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = 0.75
            consensus_map[t_upper] = {
                "consensus": consensus or "N/A",
                "summary": (data.get("summary") or "")[:200],
                "confidence": confidence,
                "model": model,
            }
        return consensus_map

    # ---------
    # Entry point
    # ---------

    def discover(self, sa):
        target_date = financial_polygon.get_last_trading_day()
        if not target_date:
            logger.info("Vunder: no valid target market date")
            return

        if not self.should_process_market_date_once(
            target_date=target_date,
            cutoff_hour_utc=self.PROCESS_CUTOFF_HOUR_UTC,
        ):
            return

        # Pre-LLM long list
        df_long = self._build_quant_long_list(sa)
        if df_long is None or df_long.empty:
            logger.info("Vunder: no quant candidates found")
            self.mark_market_date_processed(target_date)
            return

        # Limit to top N by total_score
        df_top = df_long.head(MAX_LLM_CANDIDATES).copy()
        if df_top.empty:
            return

        # Apply allow_discovery before LLM
        allowed_rows = []
        for _, row in df_top.iterrows():
            symbol = row["ticker"]
            # Example: 24h window, no price_decline filter yet
            if not self.allow_discovery(symbol, period=240, price_decline=None):
                continue
            allowed_rows.append(row)

        if not allowed_rows:
            logger.info("Vunder: all top candidates filtered out by allow_discovery")
            return

        df_allowed = pd.DataFrame(allowed_rows).reset_index(drop=True)

        # LLM consensus step
        consensus_map = self._llm_consensus(df_allowed)
        if not consensus_map:
            logger.info("Vunder: LLM consensus map empty; skipping discoveries")
            return
            # Map consensus to weights
        weight_map = {
            "strong buy": 1.25,
            "buy": 0.1,
        }

        # Sell instructions
        sell_instructions = [
            ("PERCENTAGE_DIMINISHING", 1.30, 60),
            ("PERCENTAGE_AUGMENTING", 0.85, 120),
            ("PEAKED", 7.0, None),
            ("DESCENDING_TREND", -0.20, None)
        ]

        # Final discoveries: only Buy / Strong Buy
        for _, row in df_allowed.iterrows():
            symbol = row["ticker"]
            data = consensus_map.get(symbol.upper())
            if not data:
                continue
            consensus = (data.get("consensus") or "").strip().lower()
            if consensus not in ("buy", "strong buy"):
                continue

            base_weight = weight_map.get(consensus, 1.0)
            # Optionally could factor in confidence here; for now, log only
            confidence = data.get("confidence", 1.0)
            weight = base_weight  # * confidence  # if you later want to include it

            explanation = (
                f"Under-valued {consensus} - confidence: {confidence} | "
                f"Quant: price={row['price']}, ratio_to_low={row['ratio_to_low']} | "
                f"valuation_ratio={row['valuation_ratio']}, total_score={row['total_score']} | "
                f"{data.get('model')}: {data.get('summary')}"
            )

            self.discovered(sa, symbol, explanation, sell_instructions, weight=weight)

        self.mark_market_date_processed(target_date)


register(name="Vunder", python_class="Vunder")
