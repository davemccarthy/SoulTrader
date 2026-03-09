"""
Test 3-ducks LLM: grade EX-99.1 earnings filing on past_performance, future_performance, expectation_gap.

Reads EX-99.1 text from ex99_1_dump.txt (written by test_8k_inspector.py --analyse).
Calls Gemini directly (google.generativeai), no pydantic AI. Same pattern as advisor ask_gemini.
"""

# TODO (real advisor): Sector/industry filter. Real advisor has stock object with sector/industry (e.g. yfinance).
# Add separate 1-line LLM prompt: given sector + industry, return SECTOR_REGIME_SCORE in {-1, 0, +1}
# (headwind / neutral / tailwind). Then either: hard gate (if -1 require total score >= 80) or
# soft adjust (FINAL_SCORE = DUCK_SCORE + SECTOR_REGIME_SCORE * 10). Do not infer sector from EX-99.1.

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

EX99_1_DUMP_PATH = BASE_DIR / "ex99_1_dump.txt"

# Same model order as main advisor (fallback on rate limit)
MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

PROMPT_TEMPLATE = """You are an equity analyst.
Given an earnings press release (EX-99.1 from an 8-K), evaluate the following three metrics independently.
Use only the information in the document.
Ignore stock price movement.
Ignore analyst consensus unless explicitly mentioned in the text.
Do not speculate beyond the text.

TASK
For each of the four metrics below, output a single label: "negative", "neutral", "positive", or "strong_positive".

- strong_positive: Clearly strong; multiple positive themes, confident tone, materially better than prior/expectations.
- positive: Net positive; more supportive than concerning; above average or improved.
- neutral: Mixed or ambiguous; balanced positives and negatives, or no clear signal.
- negative: Net negative; more concerning than supportive; weak vs prior or vs expectations.

1) Past Performance
Assess historical results versus prior periods. Consider revenue, profitability, margins, cash flow, EPS trends, and balance sheet quality. Output one label.

2) Future Performance
Assess forward-looking guidance and management commentary. Consider growth outlook, margins, demand environment, confidence vs caution, and risks mentioned. Output one label.

3) Expectation Gap
Assess whether results and commentary are better or worse than a reasonable pre-announcement expectation. Use the tone and content of the release to infer positive surprise vs negative surprise. Output one label.

4) Market reaction
How does the market normally react typically to this kind of earnings press release. Consider the companies financial history and sector.

OUTPUT FORMAT (STRICT):
Respond with only a single valid JSON object, no other text. Use this structure:

{
  "past_performance": "negative" | "neutral" | "positive" | "strong_positive",
  "future_performance": "negative" | "neutral" | "positive" | "strong_positive",
  "expectation_gap": "negative" | "neutral" | "positive" | "strong_positive",
  "market_reaction": "negative" | "neutral" | "positive" | "strong_positive",
  "justifications": {
    "past_performance": "<1-2 sentences>",
    "future_performance": "<1-2 sentences>",
    "expectation_gap": "<1-2 sentences>",
    "market_reaction": "<1-2 sentences>"
  }
}

For past_performance, future_performance, and expectation_gap use exactly one of: "negative", "neutral", "positive", "strong_positive".
Replace the justification placeholder strings with brief text (1–2 sentences per metric).

----------------------------------------
BEGIN EX-99.1
----------------------------------------

<<<EX99_1_TEXT>>>

----------------------------------------
END EX-99.1
----------------------------------------
"""


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
    Call Gemini API with retry across models. Returns (model, parsed_dict) or (None, None).
    """
    api_key = os.environ.get("GEMINI_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_KEY or GEMINI_API_KEY not set (e.g. in .env). Skipping Gemini.")
        return None, None

    genai.configure(api_key=api_key)
    model_index = 0

    for attempt in range(len(MODELS)):
        try:
            retry_exceptions = (
                exceptions.ServiceUnavailable,
                exceptions.ResourceExhausted,
                exceptions.DeadlineExceeded,
                exceptions.InternalServerError,
            )
            model = MODELS[model_index]
            print(f"Using model: {model}")

            response = genai.GenerativeModel(model).generate_content(
                prompt,
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


def run(path: Path | None = None):
    """Load EX-99.1 from path (default ex99_1_dump.txt), call Gemini, print result."""
    p = path or EX99_1_DUMP_PATH
    if not p.exists():
        print(f"File not found: {p}")
        print("Run: python test_8k_inspector.py --analyse <accession> first to dump EX-99.1.")
        return

    text = p.read_text(encoding="utf-8")
    if not text.strip():
        print(f"File is empty: {p}")
        return

    prompt = PROMPT_TEMPLATE.replace("<<<EX99_1_TEXT>>>", text.strip())
    print(f"EX-99.1 length: {len(text)} chars")
    print("Calling Gemini...")

    model, result = ask_gemini(prompt)
    if not result:
        print("No result from Gemini.")
        return

    print(f"\nModel: {model}")
    print("Result:")
    print(json.dumps(result, indent=2))

    # Print 3-ducks labels (negative | neutral | positive | strong_positive)
    print("\n3-ducks labels:")
    for key in ("past_performance", "future_performance", "expectation_gap"):
        val = result.get(key)
        if isinstance(val, str):
            print(f"  {key}: {val}")
        elif isinstance(val, dict):
            print(f"  {key}: (legacy counts) {val}")
        else:
            print(f"  {key}: N/A")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Grade EX-99.1 on 3 dimensions via Gemini")
    parser.add_argument("--file", type=Path, default=None, help="Path to EX-99.1 text (default: ex99_1_dump.txt)")
    args = parser.parse_args()
    run(args.file)
