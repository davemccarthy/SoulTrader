"""
Test Gemini Opinion for Stock Health Check

Tests different prompt approaches for getting Gemini's independent opinion
on whether to buy a stock, given only company, symbol, and price.
"""

import yfinance as yf
import google.generativeai as genai
import json
import re
import time
from google.api_core import exceptions
from django.conf import settings
import os
import sys

# Setup Django if running standalone
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    import django
    django.setup()
    from django.conf import settings

genai.configure(api_key=getattr(settings, 'GEMINI_KEY', None))

# Gemini models (from advisor.py)
models = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.0-flash",
]


def extract_json(text):
    """Extract JSON from Gemini response, handling markdown code blocks."""
    if not text:
        return None

    # Remove markdown code blocks
    cleaned = re.sub(r'```json\s*', '', text)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object with regex
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def ask_gemini(prompt, timeout=120.0):
    """
    Call Gemini API with retry logic.
    Returns parsed JSON or None.
    """
    gemini_model_index = 0
    
    for attempt in range(len(models)):
        try:
            retry_exceptions = (
                exceptions.ServiceUnavailable,  # 503
                exceptions.ResourceExhausted,   # 429
                exceptions.DeadlineExceeded,    # 504
                exceptions.InternalServerError, # 500
            )

            model = models[gemini_model_index]
            print(f"\nUsing model: {model}")

            response = genai.GenerativeModel(model).generate_content(
                prompt, 
                request_options={"timeout": timeout}
            )

            # Extract text from nested structure
            if not response.candidates or len(response.candidates) == 0:
                print("No candidates in Gemini response")
                return None

            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                print("No content/parts in Gemini response")
                return None

            response_text = candidate.content.parts[0].text

            if not response_text:
                print("Empty text in Gemini response")
                return None

            print(f"\nRaw Response:\n{response_text}\n")

            # Extract JSON
            results = extract_json(response_text)
            return results

        except retry_exceptions as e:
            print(f"Attempt {attempt + 1}: Service {model} unavailable. Retrying...")
            gemini_model_index += 1
            gemini_model_index %= len(models)

        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    return None


def get_stock_info(symbol):
    """Get basic stock info needed for prompts."""
    try:
        t = yf.Ticker(symbol)
        info = t.info
        
        company = info.get('longName') or info.get('shortName') or symbol
        price = info.get('currentPrice') or info.get('regularMarketPrice') or 0.0
        
        return {
            'symbol': symbol,
            'company': company,
            'price': price
        }
    except Exception as e:
        print(f"Error fetching stock info for {symbol}: {e}")
        return None


def prompt_v1_simple(symbol, company, price):
    """Simple prompt: Would you recommend buying?"""
    prompt = f"""
Would you recommend buying {symbol} ({company}) @ ${price:.2f}?

Respond with: DEFINITELY_YES|YES|NO|DEFINITELY_NO

And explain your reasoning.

Return JSON format:
{{
    "recommendation": "DEFINITELY_YES|YES|NO|DEFINITELY_NO",
    "explanation": "Your reasoning here"
}}
"""
    return prompt


def prompt_v2_with_context(symbol, company, price):
    """Similar to v1 but with more context about being independent."""
    prompt = f"""
You are analyzing stocks independently. 

Would you recommend buying {symbol} ({company}) at the current price of ${price:.2f}?

Consider recent news, sentiment, market conditions, and your knowledge about this company.
Form your own independent opinion.

Respond in JSON format:
{{
    "recommendation": "DEFINITELY_YES|YES|NO|DEFINITELY_NO",
    "explanation": "Your reasoning explaining recent news, sentiment, and your assessment"
}}
"""
    return prompt


def prompt_v3_emphasis_news(symbol, company, price):
    """Emphasize recent media/news sentiment."""
    prompt = f"""
Would you recommend buying {symbol} ({company}) @ ${price:.2f}?

Focus on:
- Recent news and media coverage
- Market sentiment
- Any significant developments

Respond in JSON format:
{{
    "recommendation": "DEFINITELY_YES|YES|NO|DEFINITELY_NO",
    "explanation": "Your reasoning based on recent news and sentiment"
}}
"""
    return prompt


def prompt_v4_buy_decision(symbol, company, price):
    """More direct buy decision framing."""
    prompt = f"""
Should I buy {symbol} ({company}) at ${price:.2f}?

Consider recent news, company performance, and market conditions.

Return JSON:
{{
    "recommendation": "DEFINITELY_YES|YES|NO|DEFINITELY_NO",
    "reasoning": "Your explanation",
    "key_factors": ["factor1", "factor2", "factor3"]
}}
"""
    return prompt


def test_gemini_opinion(symbol, prompt_func, prompt_name):
    """Test a Gemini prompt approach."""
    print(f"\n{'='*60}")
    print(f"Testing: {prompt_name}")
    print(f"{'='*60}")
    
    # Get stock info
    stock_info = get_stock_info(symbol)
    if not stock_info:
        print(f"Failed to get stock info for {symbol}")
        return None
    
    print(f"Symbol: {stock_info['symbol']}")
    print(f"Company: {stock_info['company']}")
    print(f"Price: ${stock_info['price']:.2f}")
    
    # Create prompt
    prompt = prompt_func(
        stock_info['symbol'],
        stock_info['company'],
        stock_info['price']
    )
    
    print(f"\nPrompt:\n{prompt}\n")
    
    # Call Gemini
    result = ask_gemini(prompt)
    
    if result:
        print(f"\n✅ Success!")
        print(f"Recommendation: {result.get('recommendation', 'N/A')}")
        print(f"Explanation: {result.get('explanation', result.get('reasoning', 'N/A'))}")
        if 'key_factors' in result:
            print(f"Key Factors: {result['key_factors']}")
        return result
    else:
        print(f"\n❌ Failed to get response")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Gemini opinion prompts for stocks")
    parser.add_argument('symbol', help='Stock symbol to test')
    parser.add_argument('--prompt', choices=['v1', 'v2', 'v3', 'v4', 'all'], 
                       default='all', help='Which prompt version to test')
    
    args = parser.parse_args()
    
    prompts = {
        'v1': (prompt_v1_simple, "V1: Simple Recommendation"),
        'v2': (prompt_v2_with_context, "V2: With Independent Context"),
        'v3': (prompt_v3_emphasis_news, "V3: Emphasis on News/Sentiment"),
        'v4': (prompt_v4_buy_decision, "V4: Direct Buy Decision")
    }
    
    print("="*60)
    print("Gemini Opinion Test")
    print("="*60)
    
    if args.prompt == 'all':
        # Test all versions
        results = {}
        for key, (func, name) in prompts.items():
            result = test_gemini_opinion(args.symbol.upper(), func, name)
            results[key] = result
            time.sleep(2)  # Rate limiting
        
        # Summary
        print(f"\n{'='*60}")
        print("Summary of All Approaches")
        print(f"{'='*60}")
        for key, (_, name) in prompts.items():
            result = results[key]
            if result:
                rec = result.get('recommendation', 'N/A')
                print(f"{name}: {rec}")
            else:
                print(f"{name}: FAILED")
    else:
        # Test single version
        if args.prompt in prompts:
            func, name = prompts[args.prompt]
            test_gemini_opinion(args.symbol.upper(), func, name)
        else:
            print(f"Unknown prompt version: {args.prompt}")



