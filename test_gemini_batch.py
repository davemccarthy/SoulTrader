"""
Batch test Gemini V5 prompt on multiple stocks
Tests the nervous investor prompt across diverse stocks to see recommendation distribution
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
from collections import defaultdict

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

            response = genai.GenerativeModel(model).generate_content(
                prompt, 
                request_options={"timeout": timeout}
            )

            # Extract text from nested structure
            if not response.candidates or len(response.candidates) == 0:
                return None, None

            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                return None, None

            response_text = candidate.content.parts[0].text

            if not response_text:
                return None, None

            # Extract JSON
            results = extract_json(response_text)
            return model, results

        except retry_exceptions as e:
            gemini_model_index += 1
            gemini_model_index %= len(models)
            time.sleep(1)  # Brief pause before retry

        except Exception as e:
            print(f"Unexpected error: {e}")
            return None, None
    
    return None, None


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


def prompt_v5_nervous_investor(symbol, company, price):
    """Nervous investor perspective with 5-tier recommendation scale."""
    prompt = f"""
Starting afresh,

From the perspective of a nervous investor, would you recommend buying {symbol} ({company}) @ ${price:.2f}?

Considering:
- Recent news and media coverage
- Market sentiment
- Any significant developments

Respond in JSON format:
{{
    "recommendation": "STRONG_BUY|BUY|NEUTRAL|AVOID|STRONG_AVOID",
    "explanation": "Your reasoning based on recent news and sentiment"
}}

Thank you
"""
    return prompt


def test_stock(symbol):
    """Test a single stock with V5 prompt."""
    print(f"\n{'='*60}")
    print(f"Testing: {symbol}")
    print(f"{'='*60}")
    
    # Get stock info
    stock_info = get_stock_info(symbol)
    if not stock_info:
        print(f"❌ Failed to get stock info for {symbol}")
        return None
    
    symbol_upper = stock_info['symbol'].upper()
    company = stock_info['company']
    price = stock_info['price']
    
    if price == 0.0:
        print(f"⚠️  Warning: Price is $0.00 for {symbol_upper} - may be delisted or invalid")
        return None
    
    print(f"Symbol: {symbol_upper}")
    print(f"Company: {company}")
    print(f"Price: ${price:.2f}")
    
    # Create prompt
    prompt = prompt_v5_nervous_investor(symbol_upper, company, price)
    
    # Call Gemini
    model, result = ask_gemini(prompt)
    
    if not result:
        print(f"❌ Failed to get Gemini response")
        return None
    
    recommendation = result.get('recommendation', 'N/A')
    explanation = result.get('explanation', 'N/A')
    
    # Truncate explanation for display
    explanation_preview = explanation[:200] + "..." if len(explanation) > 200 else explanation
    
    print(f"\n✅ Success!")
    print(f"Recommendation: {recommendation}")
    print(f"Explanation (preview): {explanation_preview}")
    print(f"Model used: {model}")
    
    return {
        'symbol': symbol_upper,
        'company': company,
        'price': price,
        'recommendation': recommendation,
        'explanation': explanation,
        'model': model
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch test Gemini V5 prompt on multiple stocks")
    parser.add_argument('symbols', nargs='+', help='Stock symbols to test (space-separated)')
    parser.add_argument('--delay', type=float, default=2.0, 
                       help='Delay between requests in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Gemini Opinion Batch Test - V5 (Nervous Investor)")
    print("="*60)
    print(f"Testing {len(args.symbols)} stocks with {args.delay}s delay between requests")
    print()
    
    results = []
    failed = []
    
    for i, symbol in enumerate(args.symbols, 1):
        print(f"\n[{i}/{len(args.symbols)}] Processing {symbol.upper()}...")
        
        result = test_stock(symbol)
        
        if result:
            results.append(result)
        else:
            failed.append(symbol.upper())
        
        # Rate limiting (except for last item)
        if i < len(args.symbols):
            time.sleep(args.delay)
    
    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if results:
        # Group by recommendation
        by_recommendation = defaultdict(list)
        for r in results:
            by_recommendation[r['recommendation']].append(r)
        
        # Print grouped results
        recommendation_order = ['STRONG_BUY', 'BUY', 'NEUTRAL', 'AVOID', 'STRONG_AVOID']
        
        for rec in recommendation_order:
            if rec in by_recommendation:
                stocks = by_recommendation[rec]
                print(f"\n{rec}: {len(stocks)} stock(s)")
                for stock in stocks:
                    print(f"  • {stock['symbol']:6} ({stock['company'][:40]:40}) @ ${stock['price']:.2f}")
        
        # Print any unknown recommendations
        for rec, stocks in by_recommendation.items():
            if rec not in recommendation_order:
                print(f"\n{rec} (UNKNOWN): {len(stocks)} stock(s)")
                for stock in stocks:
                    print(f"  • {stock['symbol']:6} ({stock['company'][:40]:40}) @ ${stock['price']:.2f}")
        
        # Statistics
        print(f"\n{'='*60}")
        print("STATISTICS")
        print(f"{'='*60}")
        print(f"Total tested: {len(results)}")
        print(f"STRONG_BUY: {len(by_recommendation.get('STRONG_BUY', []))}")
        print(f"BUY: {len(by_recommendation.get('BUY', []))}")
        print(f"NEUTRAL: {len(by_recommendation.get('NEUTRAL', []))}")
        print(f"AVOID: {len(by_recommendation.get('AVOID', []))}")
        print(f"STRONG_AVOID: {len(by_recommendation.get('STRONG_AVOID', []))}")
    
    if failed:
        print(f"\n{'='*60}")
        print("FAILED")
        print(f"{'='*60}")
        for symbol in failed:
            print(f"  • {symbol}")
    
    print()

