"""
Test media-response LLM: ask Gemini (with web search) for actual business/financial
media response to a company's earnings or 8-K filing.

Uses the google-genai package and the google_search tool (required for search grounding).
Usage:
    python test_media_response_llm.py --ticker RRR
    python test_media_response_llm.py --ticker EXEL
    python test_media_response_llm.py --ticker APP --date 2026-02-11

By default uses yesterday through today as the search window. With --date YYYY-MM-DD
uses that date through the next day.
"""

import json
import os
import re
import time
from datetime import date, timedelta

from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Same model order as test_3ducks_llm (fallback on rate limit)
MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
]

# Search window: by default yesterday through today; with --date D use D through D+1
def _search_start_date(anchor: date | None = None):
    if anchor is not None:
        return anchor
    return date.today() - timedelta(days=1)


def _search_end_date(anchor: date | None = None):
    if anchor is not None:
        return anchor + timedelta(days=1)
    return date.today()

PROMPT_TEMPLATE = """Search the web for business and financial news from {start_date} through {end_date} about {ticker}'s earnings announcement, quarterly results, or 8-K filing.

What was the media response (headlines, articles, analyst or press commentary)?
- Consider sources like Reuters, Bloomberg, CNBC, MarketWatch, Yahoo Finance, Seeking Alpha, and similar.

Respond with STRICT JSON only. No other text before or after. Use this structure:

{{
  "reaction": "positive" | "negative" | "neutral" | "no_coverage",
  "headlines_or_snippets": ["<quote or headline 1>", "<quote or headline 2>", ...],
  "reason": "<1-2 sentences summarizing media tone and key points>"
}}

If you find no relevant coverage in that window, set reaction to "no_coverage" and reason accordingly."""


def extract_json(text: str):
    """Extract JSON from response, handling markdown code blocks."""
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


def ask_gemini_with_search(prompt: str, timeout: float = 120.0):
    """
    Call Gemini with Google Search grounding (google_genai + google_search tool).
    Returns (model, parsed_dict) or (None, None).
    """
    api_key = os.environ.get("GEMINI_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_KEY or GEMINI_API_KEY not set (e.g. in .env). Skipping Gemini.")
        return None, None

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )
    model_index = 0

    for attempt in range(len(MODELS)):
        try:
            model = MODELS[model_index]
            print(f"Using model: {model} (with Google Search grounding)")

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            if not response.text:
                print("No content in Gemini response")
                model_index = (model_index + 1) % len(MODELS)
                continue

            parsed = extract_json(response.text)
            if parsed:
                time.sleep(1)
                return model, parsed
            print("Could not parse JSON from response")
            model_index = (model_index + 1) % len(MODELS)

        except Exception as e:
            print(f"Attempt {attempt + 1}: {model} error ({e}). Trying next model.")
            model_index = (model_index + 1) % len(MODELS)

    return None, None


def run(ticker: str, anchor_date: date | None = None):
    """Build prompt for ticker and search window; if anchor_date given use that day through next, else yesterday through today."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        print("Please provide --ticker (e.g. RRR, EXEL).")
        return

    start_date = _search_start_date(anchor_date)
    end_date = _search_end_date(anchor_date)
    prompt = PROMPT_TEMPLATE.format(
        ticker=ticker,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )
    print(f"Search window: {start_date} to {end_date} for {ticker}")
    print("Calling Gemini with web search...")

    model, result = ask_gemini_with_search(prompt)
    if not result:
        print("No result from Gemini.")
        return

    print(f"\nModel: {model}")
    print("Result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Fetch business media response to a ticker's earnings/filing via Gemini (web search)"
    )
    parser.add_argument("--ticker", "-t", required=True, help="Stock ticker (e.g. RRR, EXEL)")
    parser.add_argument(
        "--date", "-d",
        default=None,
        metavar="YYYY-MM-DD",
        help="Optional filing date for search window (default: yesterday through today)",
    )
    args = parser.parse_args()
    anchor = None
    if args.date:
        try:
            anchor = date.fromisoformat(args.date)
        except ValueError:
            print(f"Invalid --date '{args.date}'; use YYYY-MM-DD.")
            raise SystemExit(1)
    run(args.ticker, anchor)
