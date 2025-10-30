from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from core.models import Discovery, Advisor, Stock, Recommendation
import logging
import re
import json
import requests

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are a financial news analyst. Read the article at the URL below (use your browsing/knowledge to infer content if needed) and return STRICT JSON.

Input:
- Headline: {headline}
- URL: {url}

Task:
- Identify if this article is market-moving (HIGH/MEDIUM/LOW impact)
- Extract the most relevant US stock tickers (max 6)
- Classify the event type: one of [FDA_APPROVAL, EARNINGS, MA, GUIDANCE, MACRO, OTHER]
- Provide a concise reason (<= 240 chars)
- Recommend keep=true if the article should lead to discovery; else keep=false

Return JSON only with fields:
{{
  "impact": "HIGH|MEDIUM|LOW",
  "category": "FDA_APPROVAL|EARNINGS|MA|GUIDANCE|MACRO|OTHER",
  "tickers": ["SYMBOL1", "SYMBOL2"],
  "keep": true,
  "reason": "string"
}}
"""


def call_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini using the same model/endpoint format as gemini.py.
    Returns the response text (string), or raises CommandError on failure.
    """
    try:
        import urllib.parse
        headers = {"Content-Type": "application/json"}
        base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        model_name = "gemma-3-12b-it"
        endpoint = f"{base_url}/{model_name}:generateContent"
        url = f"{endpoint}?key={urllib.parse.quote(api_key, safe='')}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            raise CommandError(f"Gemini error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        if 'candidates' in data and data['candidates']:
            return data['candidates'][0]['content']['parts'][0]['text']
        return json.dumps(data)
    except CommandError:
        raise
    except Exception as e:
        raise CommandError(f"Gemini call failed: {e}")


def parse_json_loose(text: str) -> dict:
    """Extract JSON from LLM response robustly: strip fences, find first {...}."""
    cleaned = text.strip().replace('```json', '').replace('```', '').strip()
    # Try direct
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Try to find first JSON object via regex
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise CommandError("Failed to parse JSON from Gemini response")


class Command(BaseCommand):
    help = "Analyze a core_discovery record via Gemini and print structured suggestions"

    def add_arguments(self, parser):
        parser.add_argument("discovery_id", type=int, help="ID of core_discovery record")

    def handle(self, *args, **options):
        discovery_id = options["discovery_id"]

        try:
            d = Discovery.objects.select_related("stock", "advisor", "sa").get(id=discovery_id)
        except Discovery.DoesNotExist:
            raise CommandError(f"Discovery id {discovery_id} not found")

        # Extract headline and URL from explanation like: "News: {title} ({url})"
        explanation = d.explanation or ""
        title = explanation
        url_found = None

        m = re.search(r"\((https?://[^\s)]+)\)", explanation)
        if m:
            url_found = m.group(1)
            t = re.sub(r"\(https?://[^\s)]+\)", "", explanation).strip()
            if t.lower().startswith("news:"):
                t = t[5:].strip()
            title = t

        if not url_found:
            self.stdout.write(self.style.WARNING("No URL found in discovery explanation; sending headline only"))

        # Get Gemini advisor config (for key only)
        gemini = Advisor.objects.filter(python_class="Gemini").first()
        if not gemini or not gemini.key:
            raise CommandError("Gemini advisor not configured with API key")

        # Call Gemini (aligned with gemini.py)
        response_text = call_gemini(PROMPT_TEMPLATE.format(headline=title[:300], url=url_found or "N/A"), gemini.key)

        # Parse response
        parsed = parse_json_loose(response_text)

        # Normalize output
        impact = str(parsed.get("impact", "")).upper()
        category = str(parsed.get("category", "OTHER")).upper()
        tickers = [str(t).upper() for t in (parsed.get("tickers") or []) if t]
        keep = bool(parsed.get("keep", False))
        reason = str(parsed.get("reason", "")).strip()

        # Print results
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(f"Discovery {d.id}: {d.stock.symbol} - {d.stock.company}"))
        self.stdout.write(f"Headline: {title[:200]}")
        if url_found:
            self.stdout.write(f"URL: {url_found}")
        self.stdout.write("")
        self.stdout.write("Gemini Suggestions:")
        self.stdout.write(f"- impact: {impact}")
        self.stdout.write(f"- category: {category}")
        self.stdout.write(f"- tickers: {tickers}")
        self.stdout.write(f"- keep: {keep}")
        self.stdout.write(f"- reason: {reason}")

        self.stdout.write("")
        if keep and tickers:
            self.stdout.write(self.style.SUCCESS("Suggested actions:"))
            for t in tickers[:6]:
                self.stdout.write(f"  - DISCOVER {t}: {reason[:120]}")
        
        # Show recent recommendations for the first suggested ticker (if any)
        if tickers:
            first_sym = tickers[0]
            self.stdout.write("")
            self.stdout.write(self.style.NOTICE(f"Inspecting recent recommendations for {first_sym}:"))
            stock = Stock.objects.filter(symbol=first_sym).first()
            if not stock:
                self.stdout.write(self.style.WARNING(f"No Stock record for {first_sym} yet"))
                return

            recs = (
                Recommendation.objects
                .select_related("advisor", "sa")
                .filter(stock=stock)
                .order_by("-id")[:10]
            )
            if not recs:
                self.stdout.write(self.style.WARNING("No recommendations found for this symbol"))
            else:
                for r in recs:
                    adv = r.advisor.name if r.advisor_id else "(unknown)"
                    sa_id = r.sa_id if r.sa_id else "?"
                    self.stdout.write(f"- SA {sa_id} | {adv}: {float(r.confidence):.2f} | {r.explanation[:120]}")
        else:
            self.stdout.write(self.style.WARNING("Suggested action: DROP this article for discovery"))
