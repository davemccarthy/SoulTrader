"""
Test consensus LLM: ask Gemini for institutional-style consensus (Strong Buy / Buy / Hold / Sell),
a sharp one-line summary, and confidence. Uses yfinance for company name and optional hints.

Standalone script. Calls Gemini directly (google.generativeai) with strict JSON output.

Usage:
    python test_consensus_llm.py -t TSM --price 370.5 --low52 351 --high52 600
    python test_consensus_llm.py -t TSM --price 370.5 --low52 351 --high52 600 \\
        --scenario "Portfolio review" --context "Weigh overvaluation vs growth."
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

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

PROMPT_TEMPLATE = """You are an equity research assistant producing institutional-quality summaries.

For each ticker below, infer the CURRENT consensus-style recommendation
("Strong Buy", "Buy", "Hold", "Sell") as seen in mainstream equity research.

IMPORTANT:
- Do NOT default to generic summaries.
- Explicitly weigh the dominant bull vs bear arguments before deciding.
- Reflect what would ACTUALLY drive an analyst's rating (growth, margins, valuation, macro sensitivity, etc.).
- Avoid vague phrasing like "mixed outlook" unless truly justified.

Use the FACT LINES only as supporting hints. If facts are sparse, infer from widely known characteristics of the company/sector and LOWER confidence accordingly.

Scenario: {scenario}

Stocks and context (one ticker per line):
{context_block}

TASK — for each ticker, return:
1) consensus: exactly one of "Strong Buy", "Buy", "Hold", "Sell".
2) summary: ONE sharp sentence that includes:
   - the PRIMARY driver of the rating
   - and the MAIN limiting factor or risk
3) confidence: number 0.0–1.0 based on:
   - 0.9+ = very strong, widely agreed consensus
   - 0.7–0.89 = solid but not unanimous
   - 0.5–0.69 = mixed or uncertain
   - <0.5 = weak/unclear consensus

CONSTRAINTS:
- No fluff or filler language
- No repetition of the ticker name
- No bullet points
- Keep summaries under 25 words
- Avoid trendy narratives unless clearly material (e.g., AI risk only if it meaningfully impacts fundamentals)

OUTPUT FORMAT:
Respond with ONLY a single JSON object.
Keys are ticker symbols. Each value is an object with:
- "consensus"
- "summary"
- "confidence" (number)

Example:
{{
  "AAPL": {{"consensus": "Buy", "summary": "Services growth supports margins but valuation limits upside.", "confidence": 0.82}},
  "MSFT": {{"consensus": "Strong Buy", "summary": "Cloud and AI leadership drive durable growth with minimal near-term risk.", "confidence": 0.9}}
}}
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


def _validate_consensus_result(result: dict, ticker: str) -> Optional[dict]:
    """Ensure keyed by ticker and has consensus/summary/confidence."""
    if not isinstance(result, dict):
        return None
    t_up = ticker.upper()
    entry = result.get(ticker) or result.get(t_up) or result.get(ticker.lower())
    if not isinstance(entry, dict):
        return None
    cons = (entry.get("consensus") or "").strip()
    if not cons:
        return None
    allowed = {"strong buy", "buy", "hold", "sell"}
    if cons.lower() not in allowed:
        return None
    return result


def run(
    ticker: str,
    *,
    company_name: str | None = None,
    current_price: float | None = None,
    low_52w: float | None = None,
    high_52w: float | None = None,
    pe_trailing: float | None = None,
    pb: float | None = None,
    sector: str | None = None,
    industry: str | None = None,
    scenario: str = "Portfolio holdings review",
    extra_context: str = "",  # merged into scenario (same line as stock_audit)
):
    ticker = (ticker or "").strip().upper()
    if not ticker:
        print("Please provide --ticker (e.g. AAPL).")
        return

    snapshot = _fetch_market_snapshot(ticker)
    company_name = (company_name or snapshot.get("company_name") or ticker).strip()
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

    pe_str = f"{pe_trailing:.2f}" if isinstance(pe_trailing, (int, float)) else "Not provided"
    pb_str = f"{pb:.2f}" if isinstance(pb, (int, float)) else "Not provided"

    context_block = (
        f"  {ticker}: {company_name!r}, ${current_price:.2f}\n"
        f"  52-week range: ${low_52w:.2f} – ${high_52w:.2f}; trailing P/E: {pe_str}; "
        f"P/B: {pb_str}; sector: {sector}; industry: {industry}"
    )

    scen = (scenario or "Portfolio holdings review").strip()
    ctx = (extra_context or "").strip()
    if ctx:
        scen = f"{scen} — {ctx}".strip()

    prompt = PROMPT_TEMPLATE.format(
        scenario=scen,
        context_block=context_block,
    )

    print("=== Prompt ===\n")
    print(prompt)
    print("\nCalling Gemini...")

    model, result = ask_gemini(prompt)
    if not result:
        print("No result from Gemini.")
        return

    validated = _validate_consensus_result(result, ticker)
    if not validated:
        print("Unexpected JSON shape (expected top-level key = ticker with consensus/summary/confidence).")
        print(json.dumps(result, indent=2))
        return

    print(f"\nModel: {model}")
    print("Result:")
    print(json.dumps(validated, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ask Gemini for Strong Buy/Buy/Hold/Sell consensus + summary + confidence (JSON)"
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
    parser.add_argument(
        "--scenario",
        default="Portfolio holdings review",
        help="Scenario line in the prompt",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Optional text appended to Scenario (same line)",
    )
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
        scenario=args.scenario,
        extra_context=args.context,
    )
