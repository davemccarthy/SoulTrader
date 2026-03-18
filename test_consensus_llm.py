"""
Test consensus LLM: ask Gemini for an "advisor consensus" recommendation and target price
range for a publicly traded stock ticker passed via CLI.

Standalone script (like test_3ducks_llm / test_media_response_llm).
Calls Gemini directly (google.generativeai) with strict JSON output.

Usage:
    python test_consensus_llm.py --ticker AAPL
    python test_consensus_llm.py -t NVDA
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Keep consistent with other standalone Gemini scripts (fallback on rate limit / availability).
MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

PROMPT_TEMPLATE = """You are a financial research assistant. Provide a realistic buy/no-buy recommendation for a publicly traded stock.

- Stock: {ticker}
- Company Name: {company_name}
- Current Price: {current_price}
- 52-Week Low: {week52_low}, 52-Week High: {week52_high}
- Consider the company’s valuation (P/E, P/B), sector trends, operational performance, and macroeconomic risks.
- You must choose exactly one recommendation from: STRONG_BUY, BUY, NEUTRAL, AVOID, STRONG_AVOID.
- This is a buy gate (not a short). Prefer AVOID/STRONG_AVOID when valuation/risk dominates even if the company is profitable.
- If the current price is within 5–10% of the 52-week high and valuation appears stretched versus typical sector norms, lean NEUTRAL or AVOID unless there is a clear, durable catalyst.
- Reasoning must mention both positives (e.g., strong earnings, cash flow, stability) and negatives (e.g., high valuation, sector headwinds, declining growth).

Additional context (use if helpful):
- P/E (trailing): {pe_trailing}
- P/B: {pb}
- Sector: {sector}
- Industry: {industry}

Output a single JSON object only in this format:

{{
  "symbol": "{ticker}",
  "recommendation": "STRONG_BUY" | "BUY" | "NEUTRAL" | "AVOID" | "STRONG_AVOID",
  "explanation": "<2-3 sentences including positives and risks>"
}}

Respond with only a single valid JSON object, no other text.
"""


def _to_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _fetch_market_snapshot(ticker: str) -> dict:
    """
    Fetch market snapshot from yfinance. Returns possibly-partial dict.
    Keys: current_price, low_52w, high_52w, pe_trailing, sector, industry, company_name
    """
    info = {}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception:
        info = {}

    return {
        "company_name": (info.get("shortName") or info.get("longName") or "").strip() or None,
        "current_price": _to_float(info.get("currentPrice") or info.get("regularMarketPrice")),
        "low_52w": _to_float(info.get("fiftyTwoWeekLow")),
        "high_52w": _to_float(info.get("fiftyTwoWeekHigh")),
        "pe_trailing": _to_float(info.get("trailingPE")),
        "pb": _to_float(info.get("priceToBook")),
        "sector": (info.get("sector") or "").strip() or None,
        "industry": (info.get("industry") or "").strip() or None,
    }


def extract_json(text: str):
    """Extract JSON from Gemini response, handling markdown code blocks."""
    if not text:
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


def ask_gemini(prompt: str, timeout: float = 120.0):
    """
    Call Gemini API with retry across models.
    Returns (model, parsed_dict) or (None, None).
    """
    api_key = os.environ.get("GEMINI_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_KEY or GEMINI_API_KEY not set (e.g. in .env). Skipping Gemini.")
        return None, None

    genai.configure(api_key=api_key)
    model_index = 0

    for attempt in range(len(MODELS)):
        model = MODELS[model_index]
        try:
            retry_exceptions = (
                exceptions.ServiceUnavailable,
                exceptions.ResourceExhausted,
                exceptions.DeadlineExceeded,
                exceptions.InternalServerError,
            )

            print(f"Using model: {model}")
            response = genai.GenerativeModel(model).generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
                request_options={"timeout": timeout},
            )

            if not response.candidates or not response.candidates[0].content.parts:
                print("No content in Gemini response")
                model_index = (model_index + 1) % len(MODELS)
                continue

            response_text = response.candidates[0].content.parts[0].text
            if not response_text:
                print("Empty text in Gemini response")
                model_index = (model_index + 1) % len(MODELS)
                continue

            parsed = extract_json(response_text)
            if parsed:
                time.sleep(1)
                return model, parsed

            print("Could not parse JSON from response")
            model_index = (model_index + 1) % len(MODELS)

        except retry_exceptions as e:
            print(f"Attempt {attempt + 1}: {model} unavailable ({e}). Trying next model.")
            model_index = (model_index + 1) % len(MODELS)
        except Exception as e:
            print(f"Error: {e}")
            return None, None

    return None, None


def _validate_result(result: dict, ticker: str) -> dict:
    """Best-effort validation/normalization; returns result unchanged if already OK."""
    if not isinstance(result, dict):
        return result

    result.setdefault("symbol", ticker)

    rec = result.get("recommendation")
    if isinstance(rec, str):
        normalized = rec.strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "STRONG_BUY": "STRONG_BUY",
            "STRONGBUY": "STRONG_BUY",
            "BUY": "BUY",
            "NEUTRAL": "NEUTRAL",
            "AVOID": "AVOID",
            "STRONG_AVOID": "STRONG_AVOID",
            "STRONGAVOID": "STRONG_AVOID",
        }
        if normalized in aliases:
            result["recommendation"] = aliases[normalized]
    return result


def run(
    ticker: str,
    company_name: str | None = None,
    current_price: float | None = None,
    low_52w: float | None = None,
    high_52w: float | None = None,
    pe_trailing: float | None = None,
    pb: float | None = None,
    sector: str | None = None,
    industry: str | None = None,
):
    ticker = (ticker or "").strip().upper()
    if not ticker:
        print("Please provide --ticker (e.g. AAPL).")
        return

    snapshot = _fetch_market_snapshot(ticker)
    company_name = (company_name or snapshot.get("company_name") or ticker).strip()
    # Price is required by CLI (do not default to yfinance for this field).
    current_price = current_price if current_price is not None else None
    low_52w = low_52w if low_52w is not None else snapshot.get("low_52w")
    high_52w = high_52w if high_52w is not None else snapshot.get("high_52w")
    pe_trailing = pe_trailing if pe_trailing is not None else snapshot.get("pe_trailing")
    pb = pb if pb is not None else snapshot.get("pb")
    sector = (sector or snapshot.get("sector") or "Not provided").strip()
    industry = (industry or snapshot.get("industry") or "Not provided").strip()

    missing = []
    if current_price is None:
        missing.append("current_price")
    if low_52w is None:
        missing.append("low_52w")
    if high_52w is None:
        missing.append("high_52w")
    if missing:
        print(f"Missing required market snapshot fields: {', '.join(missing)}")
        print("Provide overrides via CLI (e.g. --price/--low52/--high52) or ensure yfinance can fetch ticker info.")
        return

    prompt = PROMPT_TEMPLATE.format(
        ticker=ticker,
        company_name=company_name,
        current_price=f"${current_price:.2f}",
        week52_low=f"${low_52w:.2f}",
        week52_high=f"${high_52w:.2f}",
        pe_trailing=(f"{pe_trailing:.2f}" if isinstance(pe_trailing, (int, float)) else "Not provided"),
        pb=(f"{pb:.2f}" if isinstance(pb, (int, float)) else "Not provided"),
        sector=sector,
        industry=industry,
    )
    print(f"Calling Gemini for consensus on {ticker}...")

    model, result = ask_gemini(prompt)
    if not result:
        print("No result from Gemini.")
        return

    result = _validate_result(result, ticker)
    print(f"\nModel: {model}")
    print("Result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ask Gemini for a consensus recommendation + target price range (JSON)"
    )
    parser.add_argument("--ticker", "-t", required=True, help="Stock ticker (e.g. AAPL, MSFT)")
    parser.add_argument("--company-name", "--company", default=None, help="Company name (optional; otherwise fetched)")
    parser.add_argument("--price", type=float, required=True, help="Current price (required)")
    parser.add_argument("--low52", type=float, default=None, help="Override 52-week low")
    parser.add_argument("--high52", type=float, default=None, help="Override 52-week high")
    parser.add_argument("--pe", type=float, default=None, help="Override trailing P/E")
    parser.add_argument("--pb", type=float, default=None, help="Override P/B")
    parser.add_argument("--sector", default=None, help="Override sector")
    parser.add_argument("--industry", default=None, help="Override industry")
    args = parser.parse_args()
    run(
        args.ticker,
        company_name=args.company_name,
        current_price=args.price,
        low_52w=args.low52,
        high_52w=args.high52,
        pe_trailing=args.pe,
        pb=args.pb,
        sector=args.sector,
        industry=args.industry,
    )

