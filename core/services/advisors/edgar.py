"""
Edgar advisor (ED-8): 8-K earnings–related filings, filters 1–6, analyze_8k.

Test independently: python manage.py run_edgar

Production: discover() will fetch 8-Ks, run filters (1–6, sector, reception),
and call self.discovered(...) for passing stocks.
"""

import html
import logging
import os
import re
import time
from collections import Counter
from typing import Dict, Optional, Tuple
from datetime import date, datetime, timedelta

import requests
import yfinance as yf
from django.conf import settings
from edgar import Company, find, set_identity, get_latest_filings

from core.services.advisors.advisor import AdvisorBase, register


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 8-K helpers and constants (used by analyze_8k)
# ---------------------------------------------------------------------------
_CIK_TO_TICKER_CACHE: Dict[str, Optional[str]] = {}
_CIK_TO_NAME_CACHE: Dict[str, Optional[str]] = {}

# -----------------------------
# 1️⃣ Set SEC identity (mandatory!)
# -----------------------------
# Replace with your real info for SEC compliance
set_identity("David McCarthy david@example.com")

def vprint(verbose: bool, text: str) -> None:
    if verbose:
        print(text)


def cik_to_ticker(cik: str) -> Optional[str]:
    """Map CIK to ticker symbol using edgar Company helper."""
    cik = str(cik).zfill(10)
    if cik in _CIK_TO_TICKER_CACHE:
        return _CIK_TO_TICKER_CACHE[cik]
    try:
        company = Company(cik)
        ticker = company.get_ticker()
        _CIK_TO_TICKER_CACHE[cik] = ticker
        name = getattr(company, "name", None) or getattr(company, "title", None)
        if name and isinstance(name, str):
            _CIK_TO_NAME_CACHE[cik] = name.strip() or None
        else:
            _CIK_TO_NAME_CACHE[cik] = None
        return ticker
    except Exception:
        _CIK_TO_TICKER_CACHE[cik] = None
        _CIK_TO_NAME_CACHE[cik] = None
        return None


def cik_to_company_name(cik: str) -> Optional[str]:
    """Company name for display/search. Uses cache; populates from Company(cik) if needed."""
    cik = str(cik).zfill(10)
    if cik in _CIK_TO_NAME_CACHE:
        return _CIK_TO_NAME_CACHE[cik]
    try:
        company = Company(cik)
        name = getattr(company, "name", None) or getattr(company, "title", None)
        if name and isinstance(name, str):
            name = name.strip() or None
        _CIK_TO_NAME_CACHE[cik] = name
        return name
    except Exception:
        _CIK_TO_NAME_CACHE[cik] = None
        return None


EARNINGS_KEYWORDS = [
    "results of operations", "financial condition", "period ended",
    "three months ended", "six months ended", "nine months ended",
    "fiscal quarter", "fiscal year", "furnished pursuant to",
    "press release announcing", "press release regarding", "attached as exhibit 99",
    "included as exhibit", "announced its results", "reported results",
    "released results", "announced financial results", "reported financial results",
    "provided guidance", "updated guidance", "outlook for", "expects for",
    "anticipated results", "quarter ended", "year ended", "period results",
    "financial results", "earnings release", "quarterly results", "full year results",
    "earnings per share", "eps", "net income per share", "guidance", "outlook",
    "expects", "forecast",
]
REG_FD_KEYWORDS = ["furnished pursuant to regulation fd", "Item 3.01", "Item 1.03"]
RED_FLAGS_SEVERE = {
    "chapter 11": -10, "bankruptcy filing": -10, "receivership": -10,
    "substantial doubt": -8, "ability to continue as a going concern": -8,
    "going concern": -8, "item 4.02": -8, "non-reliance on financial statements": -8,
    "restatement of previously issued financial statements": -7,
    "should no longer be relied upon": -7,
}
RED_FLAGS_MODERATE = {
    "resignation": -2, "ceo termination": -2, "cfo termination": -2,
    "termination of employment": -2, "departure": -2, "material weakness": -2,
    "internal control deficiency": -2, "sec investigation": -2, "sec inquiry": -2,
    "class action": -1, "material litigation": -2, "significant litigation": -2,
    "weak guidance": -2, "cautious outlook": -2, "cautious guidance": -2,
    "cautious stance": -2, "projected decline": -2, "expects decline": -2,
    "lower outlook": -2, "lowered expectations": -2, "decline in sales": -1,
    "anticipated softness": -2, "tempered outlook": -2, "mid-single-digit decline": -2,
    "single-digit decline": -1, "macroeconomic headwinds": -1, "promotional environment": -1,
    "disciplined inventory management": -1, "challenging macroeconomic environment": -2,
    "strategic alternatives": -2, "restructuring charges": -2, "liquidity constraints": -2,
    "breach of loan covenants": -2, "softer start": -2, "down approximately": -1,
    "net sales to be down": -1,
}
GREEN_FLAGS = {
    "raised guidance": +3, "increased outlook": +3, "increased guidance": +3,
    "initiated guidance": +1, "provided guidance for": +1, "dividend increase": +2,
    "increased dividend": +2, "share repurchase authorization": +2, "buyback program": +2,
    "above expectations": +2, "exceeded estimates": +2, "record revenue": +2,
    "record ebitda": +1, "record earnings": +1, "stronger than expected results": +2,
    "higher than expected revenue": +2, "return to profitability": +2, "driven by ai": +1,
    "eliminated debt": +1, "positive operating cash flow": +1, "strong operating leverage": +1,
    "meaningful margin expansion": +2, "revenue growth": +1, "revenue increase": +1,
    "exceptional revenue growth": +3, "robust cash generation": +1,
    "improved operating leverage": +1, "scalability and cost efficiency": +1,
    "successfully launched": +1, "growing demand ": +1, "adjusted ebitda profitability": +1,
    "backlog grew": +1, "in line with previous estimate": +1, "in line with filed estimates": +1,
    "listing compliance": +1, "compliance deadline": +1, "debt reduction": +1,
    "reduction of total debt": +1, "filed prior to": +1, "strongest year": +2,
    "strongest quater": +2,
}

# 3-ducks LLM (Filter 7): prompt for EX-99.1 grading
THREE_DUCKS_PROMPT = """You are an equity analyst.
Given an earnings press release (EX-99.1 from an 8-K), evaluate and grade the following three metrics independently.
Use only the information in the document.
Ignore stock price movement.
Ignore analyst consensus unless explicitly mentioned in the text.
Do not speculate beyond the text.

TASK
Grade the following three metrics:

1) Past Performance
Evaluate historical results versus prior periods.
Consider revenue, profitability, margins, cash flow, EPS trends, and balance sheet quality.

2) Future Performance
Evaluate forward-looking guidance and management commentary.
Consider growth outlook, margins, demand environment, confidence vs caution, and risks mentioned.

3) Expectation Gap
Evaluate whether the reported results and commentary are better or worse than what a reasonable market participant would have expected before the release (i.e. typical pre-announcement expectations). Use the tone and content of the release to infer whether the company is signaling a positive surprise, in line, or a negative surprise relative to those prior expectations.

Scoring:
-2 = strong negative | -1 = negative | 0 = neutral | +1 = positive | +2 = strong positive
Use +2 or -2 only when evidence is strong and unambiguous.
Default to 0 when signals are mixed.

OUTPUT FORMAT (STRICT):
Respond with only a single valid JSON object, no other text. Use this structure:

{
  "past_performance": <integer>,
  "future_performance": <integer>,
  "expectation_gap": <integer>,
  "justifications": {
    "past_performance": "<1-2 sentences>",
    "future_performance": "<1-2 sentences>",
    "expectation_gap": "<1-2 sentences>"
  }
}

Replace <integer> with -2, -1, 0, +1, or +2. Replace the placeholder strings with brief justifications (1–2 sentences per metric).

----------------------------------------
BEGIN EX-99.1
----------------------------------------

<<<EX99_1_TEXT>>>

----------------------------------------
END EX-99.1
----------------------------------------
"""

# Filter 8 (media reaction): same prompt as test_media_response_llm.py; company name for clarity.
MEDIA_RESPONSE_PROMPT_TEMPLATE = """Search the web for business and financial news from {start_date} through {end_date} regarding {company_name} ({ticker})'s earnings.

Analyze the gap between "Headline Results" and any "Market Reaction" (if the market has traded since the release).
- Identify if the company beat/missed analyst consensus for EPS and Revenue.
- Specifically look for forward-looking guidance, management's tone during the Q&A, and any cited "headwinds" (e.g., rising expenses, interest rates, or segment softness).
- Specifically look for mentions of interest expense, capital expenditures (CapEx), or operating margins, as these often drive post-earnings sell-offs.
- If the market has traded since the release, compare headlines to the actual stock price movement; otherwise set market_reaction_pct to null and note in reason that trading had not occurred yet in the search window.

Respond with STRICT JSON only. No other text before or after:
{{
  "reaction": "positive" | "negative" | "neutral" | "no_coverage",
  "headline_beat": {{ "eps": true/false, "revenue": true/false }},
  "market_reaction_pct": "<e.g. -2.5%> or null if no trading since release",
  "headlines_or_snippets": ["<quote 1>", "<quote 2>"],
  "reason": "<2-3 sentences: why the market reacted as it did (or that it had not yet traded). Mention specific guidance or expense figures if available.>"
}}

If you find no relevant coverage in that window, set reaction to "no_coverage" and reason accordingly."""


FILTER3_PE_MAX = 100
FILTER3_MIN_CAP = 300e6
FILTER3_PRICE_MIN = 5.0
FILTER3_PRICE_MAX = 1000.0
FILTER3_PREV_DAY_GAIN_WARN_PCT = 5.0
FILTER3_PREV_DAY_GAIN_FAIL_PCT = 10.0

# Sector beware (vprint WARNING only; matches yfinance sector/industry wording)
FILTER3_UNDERPERFORM_SECTORS = ("consumer cyclical", "real estate", "utilities")  # 🔴 Underperform
FILTER3_WATCH_SECTOR_INDUSTRY = (  # 🟡 Neutral/cautious or Watch
    ("technology", "software"),   # Tech (software): AI disruption, capex concerns
    ("financial services", "insurance"),
)


EARNINGS_CONTEXT = ["earnings", "financial results", "results of operations", "quarter ended", "fiscal quarter"]
TABLE_HINTS = ["gaap", "non-gaap", "q/q", "y/y", "%"]
EPS_NUMBER_PATTERN = re.compile(r"(earnings per share|eps|diluted).{0,60}?(-?\d+\.\d+)", re.IGNORECASE)


def _filter4_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _filter4_has_earnings_context(text: str) -> bool:
    return any(k in text for k in EARNINGS_CONTEXT)


def _filter4_has_eps_evidence(text: str) -> bool:
    return bool(EPS_NUMBER_PATTERN.search(text))


def _filter4_numeric_density(text: str) -> int:
    return len(re.findall(r"-?\$?\d+(?:,\d{3})*(?:\.\d+)?", text))


def _filter4_has_table_structure(text: str) -> bool:
    return sum(1 for k in TABLE_HINTS if k in text) >= 2


def _filter4_has_comparables(text: str) -> bool:
    words = re.findall(r"(revenue|earnings per share|eps|net income)", text, re.IGNORECASE)
    counts = Counter(w.lower() for w in words)
    return any(v >= 2 for v in counts.values())


def earnings_release_filter(raw_text: str) -> dict:
    text = _filter4_normalize(raw_text)
    checks = {
        "earnings_context": _filter4_has_earnings_context(text),
        "eps_evidence": _filter4_has_eps_evidence(text),
        "numeric_density": _filter4_numeric_density(text),
        "table_structure": _filter4_has_table_structure(text),
        "comparables": _filter4_has_comparables(text),
    }
    if not checks["earnings_context"]:
        return {"pass": False, "reason": "no earnings context", "checks": checks}
    if not checks["eps_evidence"]:
        return {"pass": False, "reason": "no EPS evidence", "checks": checks}
    if checks["numeric_density"] < 15:
        return {"pass": False, "reason": "low numeric density", "checks": checks}
    score = sum([checks["table_structure"], checks["comparables"]])
    return {"pass": score >= 1, "score": score, "reason": "ok" if score >= 1 else "soft fail", "checks": checks}


try:
    import pysentiment2 as ps
except ImportError:
    ps = None

# FILTER5: LM guidance sentiment (same as test_8k_inspector.py)
# Section headings / openers that start the disclaimer block (remove before parsing).
_FORWARD_LOOKING_PATTERN = re.compile(
    r"forward\s*[-]?\s*looking\s+statements",
    re.IGNORECASE,
)
_FORWARD_LOOKING_STARTS = [
    "Cautionary Statement",
    "Safe Harbor",
    "This press release contains forward-looking statements",
    "This release contains forward-looking statements",
]


def _filter5_strip_forward_looking_section(text: str) -> str:
    """
    Remove the Forward-Looking Statements (disclaimer) section from EX-99.1 text
    before parsing, so its negative wording does not affect FILTER5 LM.
    """
    if not text or not text.strip():
        return text
    text_lower = text.lower()
    earliest = len(text)
    m = _FORWARD_LOOKING_PATTERN.search(text)
    if m:
        earliest = min(earliest, m.start())
    for phrase in _FORWARD_LOOKING_STARTS:
        idx = text_lower.find(phrase.lower())
        if idx != -1 and idx < earliest:
            earliest = idx
    if earliest == len(text):
        return text
    return text[:earliest].strip()


GUIDANCE_KEYWORDS = [
    "guidance", "outlook", "financial outlook", "forecast", "projected", "projection",
    "expects", "expect", "we expect", "anticipate", "target", "targets", "goal", "goals",
    "estimate", "estimates", "future", "next quarter", "next year",
]

BOILERPLATE_PHRASES = [
    "forward-looking statements",
    "management believes that",
    "we do not provide a forward-looking reconciliation",
    "cannot be estimated at this time without unreasonable efforts",
    "cannot, without unreasonable effort",
    "not be meaningful to investors",
    "range of values so broad",
    "summarized in the table below",
]


def _filter5_split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[\.\?\!])\s+|(?<=\")\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _filter5_looks_like_table_fragment(sentence: str) -> bool:
    """Exclude table rows / financial grids from LM scoring."""
    s = sentence.strip()
    if not s:
        return True
    digit_count = len(re.findall(r"\d", s))
    dollar_count = len(re.findall(r"\$\d", s))
    lower = s.lower()
    if digit_count >= 12 and ("$" in s or "million" in lower or "non-gaap" in lower):
        return True
    if dollar_count >= 2 and ("non-gaap" in lower or "revenue" in lower or "operating income" in lower):
        return True
    return False


def _filter5_extract_guidance_sentences(text: str) -> list:
    sentences = _filter5_split_sentences(text)
    lower_kw = [k.lower() for k in GUIDANCE_KEYWORDS]
    return [s for s in sentences if any(k in s.lower() for k in lower_kw)]


def _filter5_is_boilerplate(sentence: str) -> bool:
    lower = sentence.lower()
    return any(phrase in lower for phrase in BOILERPLATE_PHRASES)


def _filter5_compute_lm_guidance(text: str):
    """
    Run LM on guidance sentences. Returns None if ps missing / no guidance after boilerplate.
    Else dict with passed, n_sentences, total_pos, total_neg, avg_polarity, net_polarity.
    """
    if ps is None or not (text or "").strip():
        return None
    guidance = _filter5_extract_guidance_sentences(text)
    candidates = [
        s for s in guidance
        if not _filter5_is_boilerplate(s) and not _filter5_looks_like_table_fragment(s)
    ]
    if not candidates:
        return None
    try:
        lm = ps.LM()
        total_pos = total_neg = 0
        polarities = []
        for s in candidates:
            tokens = lm.tokenize(s)
            score = lm.get_score(tokens)
            total_pos += score.get("Positive", 0)
            total_neg += score.get("Negative", 0)
            polarities.append(score.get("Polarity", 0.0))
        avg_polarity = sum(polarities) / len(polarities) if polarities else 0.0
        denom = total_pos + total_neg
        net_polarity = (total_pos - total_neg) / denom if denom > 0 else 0.0
        passed = not (total_neg > total_pos)
        return {
            "passed": passed,
            "n_sentences": len(candidates),
            "total_pos": total_pos,
            "total_neg": total_neg,
            "avg_polarity": round(avg_polarity, 4),
            "net_polarity": round(net_polarity, 4),
        }
    except Exception:
        return None


def _latest_quarter_end_for_date(d: date) -> date:
    """Return quarter-end date for the quarter this filing date reports. Jan/Feb/Mar → prev Dec 31; Apr/Jun → Mar 31; etc."""
    y, m = d.year, d.month
    if m in (1, 2, 3):
        return date(y - 1, 12, 31)
    if m in (4, 5, 6):
        return date(y, 3, 31)
    if m in (7, 8, 9):
        return date(y, 6, 30)
    return date(y, 9, 30)


def get_eps_for_report_quarter(ticker: str, report_date: date) -> Optional[Dict]:
    """
    Fetch Alpha Vantage EARNINGS; find the quarterly record for the latest quarter (by filing date).
    If no record for that quarter, or record has no EPS surprise, print and return None. Else return record.
    """
    api_key = getattr(settings, "ALPHAVANTAGE_API_KEY", None) or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return None
    latest_quarter_end = _latest_quarter_end_for_date(report_date)
    target = latest_quarter_end.isoformat()
    url = "https://www.alphavantage.co/query"
    params = {"function": "EARNINGS", "symbol": ticker.upper(), "apikey": api_key}
    try:
        time.sleep(0.25)
        resp = requests.get(url, params=params, timeout=15)

        if resp.status_code == 429:
            logger.warning("Alpha Vantage rate limit (429): request rejected for ticker=%s", ticker)
            return None
        resp.raise_for_status()
        data = resp.json()
        if "Error Message" in data or "Note" in data:
            return None
        if "Information" in data:
            print("Alpha Vantage (Information) for %s: %s" % (ticker, data["Information"][:250]))
            return None
        quarterly = data.get("quarterlyEarnings") or []
        record = None
        for r in quarterly:
            fd = r.get("fiscalDateEnding") or ""
            if fd[:10] == target if len(fd) >= 10 else fd == target:
                record = r
                break
        if record is None:
            return None
        return record
    except Exception:
        return None


# -------- 8-K actual EPS (for Filter 6 fallback when AV reportedEPS missing) --------

def _extract_eps_from_xbrl(filing) -> Optional[float]:
    """Try diluted EPS from 8-K XBRL if available."""
    try:
        xbrl = filing.xbrl()
    except Exception:
        return None
    if not xbrl:
        return None
    concepts = ("EarningsPerShareDiluted", "EarningsPerShare", "EarningsPerShareBasic")
    for concept in concepts:
        try:
            facts = xbrl.query().by_concept(concept, exact=False).execute()
            if not facts:
                continue
            duration_facts = [f for f in facts if f.get("period_type") == "duration"]
            if not duration_facts:
                duration_facts = facts
            sorted_facts = sorted(
                duration_facts,
                key=lambda f: f.get("period_end", "") or "",
                reverse=True,
            )
            val = sorted_facts[0].get("numeric_value") or sorted_facts[0].get("value")
            if val is not None:
                return float(val)
        except Exception:
            continue
    return None


_EPS_PATTERNS = [
    r"(?:diluted|basic)\s+EPS\s+(?:of\s+|were\s+)?\$?\s*([\d,]+\.?\d{2,})",
    r"(?:GAAP\s+)?EPS\s+(?:of\s+|were\s+)?\$?\s*([\d,]+\.?\d{2,})",
    r"earnings\s+per\s+(?:common\s+)?share\s+(?:\(?EPS\)?)?\s+(?:of\s+|were\s+)?\$?\s*([\d,]+\.?\d{2,})",
    r"\$([\d,]+\.?\d{2,})\s+per\s+(?:diluted|basic)?\s*(?:share|diluted share)",
    r"(?:Diluted|diluted)\s+.*?(?:per\s+share|operations).*?\$\s*([\d,]+\.?\d{2,})",
    r"Continuing\s+operations\s+\$\s*([\d,]+\.?\d{2,})",
]
_NONGAAP_PATTERNS = [
    r"(?:non[-\s]?GAAP|adjusted)\s+EPS\s+(?:of\s+)?\$?\s*([\d,]+\.?\d{2,})",
]


def _normalize_exhibit_text(raw: str) -> str:
    """Strip HTML and normalize whitespace for EPS regex matching."""
    if not raw:
        return ""
    s = raw
    try:
        s = html.unescape(s)
    except Exception:
        pass
    s = re.sub(r"<script[^>]*>.*?</script>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<style[^>]*>.*?</style>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_eps_from_text(text: str) -> Optional[float]:
    """Extract one EPS value from exhibit 99 text; prefer diluted/GAAP, then non-GAAP."""
    text = _normalize_exhibit_text(text)
    if not text:
        return None
    for pattern in _EPS_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if abs(val) < 1e6 and val != 0:
                    return val
            except ValueError:
                pass
    for pattern in _NONGAAP_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if abs(val) < 1e6 and val != 0:
                    return val
            except ValueError:
                pass
    return None


def get_actual_eps_from_8k(filing, verbose: bool = False) -> Tuple[Optional[float], str]:
    """
    Get reported EPS from 8-K: try XBRL first, then exhibit 99 text.
    Returns (eps_value, source) where source is "xbrl" or "text" or "none".
    """
    eps = _extract_eps_from_xbrl(filing)
    if eps is not None:
        if verbose:
            vprint(verbose, f"  [8-K XBRL] EPS ${eps:.2f}")
        return eps, "xbrl"
    text = _get_ex99_text(filing)
    eps = _extract_eps_from_text(text)
    if eps is not None:
        if verbose:
            vprint(verbose, f"  [8-K text] EPS ${eps:.2f}")
        return eps, "text"
    return None, "none"


def _get_ex99_text(filing) -> str:
    """Return concatenated EX-99.x exhibit text from filing, or empty string."""
    parts = []
    if hasattr(filing, "exhibits") and filing.exhibits:
        for ex in filing.exhibits:
            if "99." in str(ex).lower():
                try:
                    parts.append(ex.text())
                except Exception:
                    continue
    return " ".join(parts)


# ---------------------------------------------------------------------------
# ED-8 advisor class and command entry
# ---------------------------------------------------------------------------


class Edgar(AdvisorBase):
    """Advisor for 8-K earnings filings. Entry points and filters TBD."""

    def compute_filter1_pass(self, filing, verbose: bool = False) -> bool:
        form = getattr(filing, "form", None)
        if form != "8-K":
            vprint(verbose, "FILTER1: form not 8-K")
            return False
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(getattr(filing, "cik", ""))
        if not ticker:
            vprint(verbose, "FILTER1: no tradable ticker → fail")
            return False
        if not (hasattr(filing, "exhibits") and filing.exhibits):
            vprint(verbose, "FILTER1: no exhibits → fail")
            return False
        if not any("99." in str(ex).lower() for ex in filing.exhibits):
            vprint(verbose, "FILTER1: no exhibit 99.x → fail")
            return False
        filing_text = filing.text().lower()
        if "item 9.0" not in filing_text:
            vprint(verbose, "FILTER1: item 9.0.x → fail")
            return False
        for kw in REG_FD_KEYWORDS:
            if kw in filing_text:
                vprint(verbose, f"FILTER1: Reg keyword {kw} in main text → fail")
                return False
        if not any(kw in filing_text for kw in EARNINGS_KEYWORDS):
            vprint(verbose, "FILTER1: No earnings key phrase in main text")
            return False
        vprint(verbose, "FILTER1: (pass)")
        return True

    def compute_filter2_pass(self, filing, verbose: bool = False) -> bool:
        exhibit_99_parts = []
        if hasattr(filing, "exhibits") and filing.exhibits:
            for ex in filing.exhibits:
                if "99." in str(ex).lower():
                    try:
                        exhibit_99_parts.append(ex.text())
                    except Exception:
                        pass
        exhibit_99_text = " ".join(exhibit_99_parts) if exhibit_99_parts else ""
        filing_text = (filing.text() or "") if hasattr(filing, "text") else ""
        combined_text = (filing_text + " " + exhibit_99_text).lower()
        exhibit_99_lower = exhibit_99_text.lower()
        score = 0
        for keyword, penalty in RED_FLAGS_SEVERE.items():
            if keyword in combined_text:
                vprint(verbose, f"FILTER2: severe red flag '{keyword}' → {penalty}")
                score += penalty
        for keyword, penalty in RED_FLAGS_MODERATE.items():
            if keyword in combined_text:
                vprint(verbose, f"FILTER2: moderate red flag '{keyword}' → {penalty}")
                score += penalty
        for keyword, bonus in GREEN_FLAGS.items():
            if keyword in exhibit_99_lower:
                vprint(verbose, f"FILTER2: green flag '{keyword}' → +{bonus}")
                score += bonus
        vprint(verbose, f"FILTER2: {'(pass)' if score >= 0 else 'fail'} {score}")
        return score >= 0

    def compute_filter3_pass(self, filing, verbose: bool = False) -> bool:

        ticker = getattr(filing, "ticker", None)
        comments = []

        if not ticker:
            ticker = cik_to_ticker(getattr(filing, "cik", ""))
        try:
            time.sleep(0.05)
            stock = yf.Ticker(ticker)
            info = stock.info or {}
        except Exception as e:
            vprint(verbose, f"FILTER3: yfinance error → pass (skip checks): {e}")
            return True
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe is not None and isinstance(pe, (int, float)):
            if pe > FILTER3_PE_MAX:
                vprint(verbose, f"FILTER3: fail P/E {pe:.1f} > {FILTER3_PE_MAX} (overvalued)")
                return False
        cap = info.get("marketCap")
        if cap is not None and isinstance(cap, (int, float)):
            if cap < FILTER3_MIN_CAP:
                vprint(verbose, f"FILTER3: fail market_cap {cap/1e6:.1f}M < {FILTER3_MIN_CAP/1e6:.0f}M (nano/micro)")
                return False
        price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
        if price is not None and isinstance(price, (int, float)):
            if price < FILTER3_PRICE_MIN:
                vprint(verbose, f"FILTER3: fail price ${price:.2f} < ${FILTER3_PRICE_MIN} (below band)")
                return False
            if price > FILTER3_PRICE_MAX:
                vprint(verbose, f"FILTER3: fail price ${price:.2f} > ${FILTER3_PRICE_MAX} (above band)")
                return False
        fifty_two_high = info.get("fiftyTwoWeekHigh")
        if price is not None and fifty_two_high is not None and isinstance(fifty_two_high, (int, float)) and fifty_two_high > 0:
            pct_52 = 100.0 * price / fifty_two_high
            if pct_52 >= 90.0:
                comments.append(f"⚠️ BASIC FILTER WARNING: Price at {pct_52:.1f}% of 52-week high")
        fd = getattr(filing, "filing_date", None)
        if price is not None and fd is not None:
            try:
                if isinstance(fd, date):
                    filing_date = fd
                elif isinstance(fd, str):
                    filing_date = date.fromisoformat(fd[:10])
                elif hasattr(fd, "date") and callable(getattr(fd, "date")):
                    filing_date = fd.date()
                else:
                    filing_date = date.fromisoformat(str(fd)[:10])
                start = filing_date - timedelta(days=14)
                end = filing_date + timedelta(days=2)
                hist = stock.history(start=start, end=end, auto_adjust=True)
                if hist is not None and not hist.empty:
                    two_week_high = float(hist["High"].max())
                    last_close = float(hist["Close"].iloc[-1])
                    if two_week_high > 0 and last_close >= 0.90 * two_week_high:
                        pct_2w = 100.0 * last_close / two_week_high
                        comments.append(f"⚠️ BASIC FILTER WARNING: Filing price at {pct_2w:.1f}% of 2-week high")

                    # Previous trading day big price gain (last close before filing vs day before)
                    before_filing = hist[[(x.date() if hasattr(x, "date") else x) < filing_date for x in hist.index]]
                    if len(before_filing) >= 2:
                        close_prev = float(before_filing["Close"].iloc[-1])
                        close_prev2 = float(before_filing["Close"].iloc[-2])
                        if close_prev2 > 0:
                            prev_day_pct = (close_prev - close_prev2) / close_prev2 * 100
                            if prev_day_pct >= FILTER3_PREV_DAY_GAIN_FAIL_PCT:
                                logger.warning(f"Previous day price gain +{prev_day_pct:.1f}% (hard fail >= {FILTER3_PREV_DAY_GAIN_FAIL_PCT}%)")
                                return False, comments
                            if prev_day_pct >= FILTER3_PREV_DAY_GAIN_WARN_PCT:
                                comments.append(f"⚠️ BASIC FILTER WARNING: Previous day price gain +{prev_day_pct:.1f}%")

            except Exception as e:
                vprint(verbose, f"FILTER3: (skip 2-week high check: {e})")
        if verbose:
            sector = (info.get("sector") or "").strip().lower()
            industry = (info.get("industry") or "").strip().lower()
            if sector and sector in FILTER3_UNDERPERFORM_SECTORS:
                comments.append(f"⚠️ WARNING: Sector in underperform list")

            for watch_sector, watch_industry in FILTER3_WATCH_SECTOR_INDUSTRY:
                if sector and industry and watch_sector in sector and watch_industry in industry:
                    comments.append(f"⚠️* BASIC FILTER WARNING: Sector/industry in watch list **********")
                    break

        vprint(verbose, "FILTER3: (pass) (P/E, cap, price band ok)")
        return True, comments

    def compute_filter4_pass(self, filing, verbose: bool = False) -> bool:
        exhibit_99_parts = []
        if hasattr(filing, "exhibits") and filing.exhibits:
            for ex in filing.exhibits:
                if "99." in str(ex).lower():
                    try:
                        exhibit_99_parts.append(ex.text())
                    except Exception:
                        pass
        text = " ".join(exhibit_99_parts) if exhibit_99_parts else ""
        if not text:
            vprint(verbose, "FILTER4: no 99.x exhibit text → fail")
            return False
        result = earnings_release_filter(text)
        if not result["pass"]:
            vprint(verbose, f"FILTER4: {result['reason']} → fail")
            if verbose and result.get("checks"):
                vprint(verbose, f"FILTER4: checks={result['checks']}")
            return False
        if verbose and result.get("checks"):
            vprint(verbose, f"FILTER4: (pass) checks={result['checks']}")
        return True

    def compute_filter5_pass(self, filing, verbose: bool = False) -> bool:
        exhibit_99_parts = []
        if hasattr(filing, "exhibits") and filing.exhibits:
            for ex in filing.exhibits:
                if "99." in str(ex).lower():
                    try:
                        exhibit_99_parts.append(ex.text())
                    except Exception:
                        pass
        text = " ".join(exhibit_99_parts) if exhibit_99_parts else ""
        if not text:
            vprint(verbose, "FILTER5: (fail) no 99.x exhibit text")
            return False
        text = _filter5_strip_forward_looking_section(text)
        guidance_sents = _filter5_extract_guidance_sentences(text)
        if not guidance_sents:
            vprint(verbose, "FILTER5: (fail) no guidance sentences")
            return False
        non_boilerplate = [
            s for s in guidance_sents
            if not _filter5_is_boilerplate(s) and not _filter5_looks_like_table_fragment(s)
        ]
        vprint(verbose, f"FILTER5: guidance {len(guidance_sents)} total, {len(non_boilerplate)} after boilerplate + table filter")
        result = _filter5_compute_lm_guidance(text)
        if result is None:
            vprint(verbose, "FILTER5: (pass) no guidance after boilerplate or pysentiment2 unavailable")
            return False
        vprint(verbose, f"FILTER5: ({'pass' if result['passed'] else 'fail'}) n_sentences={result['n_sentences']} total_pos={result['total_pos']} total_neg={result['total_neg']} net_polarity={result['net_polarity']:+.3f}")
        return result["passed"]

    def compute_filter6_pass(self, filing, verbose: bool = False):
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(str(getattr(filing, "cik", "")))
        comments = []

        if not ticker:
            return None,[]
        fd = getattr(filing, "filing_date", None)
        if fd is None:
            return None,[]
        if isinstance(fd, str):
            try:
                report_date = date.fromisoformat(fd[:10])
            except ValueError:
                return None,[]
        else:
            report_date = getattr(fd, "date", lambda: fd)() if hasattr(fd, "date") else fd
        record = get_eps_for_report_quarter(ticker, report_date)
        if not record:
            logger.warning(f"No EPS record ticker={ticker or 'N/A'}")
            return None,[]
        surprise_str = record.get("surprisePercentage")
        reported_str = record.get("reportedEPS")
        try:
            surprise_val = float(surprise_str)
            reported_val = float(reported_str)
            if reported_val != 0:
                eps_score = min(surprise_val, 50) * (reported_val ** 0.5)
                if verbose:
                    comments.append(f"AV EPS surprise {surprise_str}% reported {reported_str} score {eps_score}")
                return eps_score >= 5,comments
        except (TypeError, ValueError):
            pass
        # Fallback: AV reported missing/0 → use 8-K actual + AV estimate (test_eps_8k_plus_av)
        actual_eps, source = get_actual_eps_from_8k(filing, verbose=verbose)
        estimated_eps = record.get("estimatedEPS")
        try:
            estimated_eps = float(estimated_eps) if estimated_eps not in (None, "", "None") else None
        except (TypeError, ValueError):
            estimated_eps = None
        if actual_eps is not None and estimated_eps is not None and estimated_eps != 0:
            surprise_pct = ((actual_eps - estimated_eps) / abs(estimated_eps)) * 100
            eps_score = min(surprise_pct, 50) * (actual_eps ** 0.5)
            if verbose:
                comments.append(f"8-K actual ${actual_eps:.2f} ({source}) + AV estimate ${estimated_eps:.2f} → surprise {surprise_pct:+.2f}% score {eps_score}")
            return eps_score >= 10,comments
        logger.warning(f"Missing EPS value ticker={ticker or 'N/A'} (AV reported missing/0 and 8-K+AV fallback failed)")
        return None,[]


    def compute_filter7_pass(self, filing, verbose: bool = False):
        """
        FILTER7: 3-ducks LLM (past/future/expectation). Uses AdvisorBase.ask_gemini.
        Strict pass: past_performance >= 0, future_performance >= 1, expectation_gap >= 1.
        Returns True (pass), False (fail), or None (no text / LLM error / inconclusive).
        """
        text = _get_ex99_text(filing)
        comments = []

        if not text.strip():
            vprint(verbose, "FILTER7: no EX-99.x text → None")
            return None,[]
        prompt = THREE_DUCKS_PROMPT.replace("<<<EX99_1_TEXT>>>", text.strip())
        model, results = self.ask_gemini(prompt, timeout=120.0)
        if not results:
            vprint(verbose, "FILTER7: ask_gemini returned no results → None")
            return None,[]
        try:
            past = results.get("past_performance")
            future = results.get("future_performance")
            gap = results.get("expectation_gap")
            past = int(past) if past is not None else None
            future = int(future) if future is not None else None
            gap = int(gap) if gap is not None else None

        except (TypeError, ValueError):
            vprint(verbose, "FILTER7: could not parse scores → None")
            return None,[]
        if past is None or future is None or gap is None:
            vprint(verbose, "FILTER7: missing score(s) → None")
            return None,[]
        # Strict: past >= 0, future >= 1, expectation_gap >= 1
        if past < 0 or future < 1 or gap < 1:
            return False,[]

        comments.append(f"Ex99.1 LLM interpretation: Past performance={past} guidance={future} expectation_gap={gap}")
        return True,comments

    @staticmethod
    def _filing_datetime_et(filing):
        """
        Return filing datetime in US/Eastern, or None.
        Uses header.acceptance_datetime / accepted if available; else filing_date at start-of-day ET.
        """
        import pytz
        et = pytz.timezone("US/Eastern")
        header = getattr(filing, "header", None)
        if header is not None:
            acc = getattr(header, "acceptance_datetime", None) or getattr(header, "accepted", None)
            if acc is not None:
                if hasattr(acc, "hour"):  # datetime-like
                    dt = acc
                elif hasattr(acc, "strftime") and hasattr(acc, "date"):
                    dt = acc
                elif isinstance(acc, str) and len(acc) >= 12:  # SEC YYYYMMDDHHMMSS
                    try:
                        dt = datetime(
                            int(acc[:4]), int(acc[4:6]), int(acc[6:8]),
                            int(acc[8:10]), int(acc[10:12]), int(acc[12:14]) if len(acc) >= 14 else 0,
                        )
                    except (ValueError, TypeError):
                        dt = None
                else:
                    dt = None
                if dt is not None:
                    if dt.tzinfo is None:
                        return et.localize(dt)
                    return dt.astimezone(et)
        fd = getattr(filing, "filing_date", None)
        if fd is None:
            return None
        try:
            if isinstance(fd, date) and not isinstance(fd, datetime):
                d = fd
            elif hasattr(fd, "date") and callable(getattr(fd, "date")):
                d = fd.date()
            elif isinstance(fd, str):
                d = date.fromisoformat(fd[:10])
            else:
                d = date.fromisoformat(str(fd)[:10])
        except (TypeError, ValueError):
            return None
        return et.localize(datetime.combine(d, datetime.min.time()))

    def compute_filter8_pass(self, filing, verbose: bool = False):
        """
        FILTER8: Third-party media reaction (company name + date required; no our intel).
        Uses ask_gemini(..., use_search=True). Pass: positive → True; negative/neutral/no_coverage → False.
        Returns True (pass), False (fail), or None (no ticker/date/LLM error).
        """
        cik = str(getattr(filing, "cik", "") or "")
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)
        comments = []

        if not ticker:
            vprint(verbose, "FILTER8: no ticker → None")
            return None,[]
        company_name = cik_to_company_name(cik) or ticker
        fd = getattr(filing, "filing_date", None)

        if fd is None:
            vprint(verbose, "FILTER8: no filing date → None")
            return None,[]
        try:
            if isinstance(fd, date):
                filing_date = fd
            elif isinstance(fd, str):
                filing_date = date.fromisoformat(fd[:10])
            elif hasattr(fd, "date") and callable(getattr(fd, "date")):
                filing_date = fd.date()
            else:
                filing_date = date.fromisoformat(str(fd)[:10])
        except (TypeError, ValueError):
            vprint(verbose, "FILTER8: invalid filing date → None")
            return None,[]
        # When market is closed, defer the LLM call until at least 1 hour after filing (save on calls).
        market_status = self.market_open()
        is_market_open = market_status is not None and market_status >= 0
        if not is_market_open:
            filing_dt_et = self._filing_datetime_et(filing)
            if filing_dt_et is not None:
                import pytz
                now_et = datetime.now(pytz.timezone("US/Eastern"))
                if now_et < filing_dt_et + timedelta(hours=1):
                    filing_time_str = filing_dt_et.strftime("%Y-%m-%d %H:%M ET")
                    logger.info(
                        "FILTER8: deferred (market closed and filing < 1h old), ticker=%s filing_time=%s",
                        ticker,
                        filing_time_str,
                    )
                    return None,[]
        start_date = filing_date - timedelta(days=1)
        end_date = filing_date + timedelta(days=1)
        prompt = MEDIA_RESPONSE_PROMPT_TEMPLATE.format(
            company_name=company_name,
            ticker=ticker,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        model, results = self.ask_gemini(prompt, timeout=120.0, use_search=True)
        if not results:
            vprint(verbose, "FILTER8: ask_gemini returned no results → None")
            return None,[]
        reaction = (results.get("reaction") or "").strip().lower()
        reason = results.get("reason")

        if not reaction:
            vprint(verbose, "FILTER8: missing reaction in response → None")
            return None,[]
        if verbose:
            vprint(verbose, f"FILTER8: reaction={reaction}")
        if reaction == "no_coverage":
            return None,[]
        if reaction == "positive":
            comments.append(f"LLM found media reaction {reaction}: {reason}")
            return True,comments
        return False,[]

    def analyze_8k_advanced(self, filing, meta_requirements: dict, verbose: bool = False):
        """
        Run advanced filters (6, 7, 8) for any missing in meta_requirements.
        Returns (result, state, advanced_comments) where state is (f7, f6, f8) and
        advanced_comments is a list of comment strings from the filters that ran.
        """
        cik = str(getattr(filing, "cik", None))
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)
        advanced_comments = []

        if meta_requirements.get("filter6") is None:
            f6 = self.compute_filter6_pass(filing, verbose)
            f6_pass = f6[0] if isinstance(f6, (list, tuple)) and len(f6) >= 1 else f6
            f6_comments = f6[1] if isinstance(f6, (list, tuple)) and len(f6) >= 2 else []
            if f6_pass is False or f6_pass is None:
                logger.info(f"compute_filter6_pass(EPS beat): ticker={ticker or 'N/A'}, returns {f6_pass}")
                return f6_pass, (None, f6_pass, None), (f6_comments or [])
            advanced_comments.extend(f6_comments or [])

        if meta_requirements.get("filter7") is None:
            f7 = self.compute_filter7_pass(filing, verbose)
            f7_pass = f7[0] if isinstance(f7, (list, tuple)) and len(f7) >= 1 else f7
            f7_comments = f7[1] if isinstance(f7, (list, tuple)) and len(f7) >= 2 else []
            if f7_pass is False or f7_pass is None:
                logger.info(f"compute_filter7_pass(3ducks): ticker={ticker or 'N/A'}, returns {f7_pass}")
                return f7_pass, (f7_pass, True, None), advanced_comments
            advanced_comments.extend(f7_comments or [])

        if meta_requirements.get("filter8") is None:
            f8 = self.compute_filter8_pass(filing, verbose)
            f8_pass = f8[0] if isinstance(f8, (list, tuple)) and len(f8) >= 1 else f8
            f8_comments = f8[1] if isinstance(f8, (list, tuple)) and len(f8) >= 2 else []
            if f8_pass is False or f8_pass is None:
                logger.info(f"compute_filter8_pass(media reaction): ticker={ticker or 'N/A'}, returns {f8_pass}")
                return f8_pass, (True, True, f8_pass), advanced_comments
            advanced_comments.extend(f8_comments or [])

        logger.info(f"analyze_8k_advanced: ticker={ticker or 'N/A'}, passed")
        logger.warning("NO NINTH GATE!!!!")

        return True, (True, True, True), advanced_comments

    def analyze_8k_basic(self, filing, verbose: bool = False):
        accession = getattr(filing, "accession_no", None) or getattr(filing, "accession_number", None)
        cik = str(getattr(filing, "cik", None))
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)

        if not ticker:
            return False, []
        if not self.compute_filter1_pass(filing, verbose):
            logger.info(f"{ticker} CIK={cik}, accession={accession} failed filter1")
            return False, []
        if not self.compute_filter2_pass(filing, verbose):
            logger.info(f"{ticker} CIK={cik}, accession={accession} failed filter2")
            return False, []
        if not self.compute_filter4_pass(filing, verbose):
            logger.info(f"{ticker} CIK={cik}, accession={accession} failed filter4")
            return False, []

        filter3, comments = self.compute_filter3_pass(filing, verbose)

        if not filter3:
            logger.info(f"{ticker} CIK={cik}, accession={accession} failed filter3")
            return False, []

        if not self.compute_filter5_pass(filing, verbose):
            logger.info(f"{ticker} CIK={cik}, accession={accession} failed filter5")
            return False, []

        logger.info(f"{ticker} CIK={cik}, accession={accession} passed basic filters")
        return True, comments

    def _meta_filters_complete(self, meta):
        """True if meta has all of filter6, filter7, filter8 set (non-None)."""
        m = meta or {}
        return (
            m.get("filter6") is not None
            and m.get("filter7") is not None
            and m.get("filter8") is not None
        )

    def _process_incomplete_watchlist_entry(self, entry, verbose=True):
        """
        Fetch filing for this watchlist entry, run advanced filters for any missing
        meta filter6/7/8, merge state into entry.meta and save. If result is False, set status Excluded.
        Skips if accession/cik missing or filing not found.
        """
        meta = entry.meta or {}
        accession = meta.get("accession")
        cik = meta.get("cik")
        if not accession or not cik:
            logger.warning(f"Watchlist entry {entry.id} missing accession or cik in meta, skip")
            return
        try:
            filing = find(accession)
        except Exception as e:
            logger.warning(f"Could not find filing {accession}: {e}")
            return
        if filing is None:
            logger.warning(f"Could not find filing {accession}")
            return
        result, state, advanced_comments = self.analyze_8k_advanced(filing, meta, verbose=verbose)
        if verbose:
            print(f"    Advanced: result={result} state={state}")
        new_meta = dict(meta)
        new_meta["filter7"] = state[0]
        new_meta["filter6"] = state[1]
        new_meta["filter8"] = state[2]
        new_meta["comments"] = (meta.get("comments") or []) + (advanced_comments or [])
        new_meta.setdefault("cik", cik)
        new_meta.setdefault("accession", accession)
        entry.meta = new_meta
        if result is False:
            entry.status = "Excluded"
            entry.explanation = "8-K dismissed"
        entry.save()

    def discover(self, sa):

        """
        Discovery entry point for the ED-8 advisor.
        """
        market_status = self.market_open()
        """
        logger.info("Fetching latest filings...")
        latest = get_latest_filings()

        try:
            filings = list(latest)
        except Exception as e:
            logger.warning(f"❌ Error converting latest filings to list: {e}")
            return

        # Filter to 8-Ks only
        filings_8k = [f for f in filings if getattr(f, "form", None) == "8-K"]
        if not filings_8k:
            logger.info("No latest 8-K filings found.")
            return
        """
        filing1 = find("0000764622-26-000010")
        filing2 = find("0001140361-26-006718")
        filing3 = find("0001193125-26-068512")

        filings_8k = [filing1]

        # Process pending watchlist entries with incomplete meta (filter6/7/8)
        pending = self.watchlist().order_by("created")
        for entry in pending:
            if self._meta_filters_complete(entry.meta):

                if True or market_status is not None and market_status >= 0:
                    comments = entry.meta.get("comments") or []
                    explanation = " | ".join(comments) if comments else "8-K passed"
                    self.discovered(sa, entry.stock.symbol, explanation, None)

                    entry.status = "Executed"
                    entry.save(update_fields=["status"])
                continue
            logger.info(f"Processing incomplete: {entry.stock.symbol} accession={(entry.meta or {}).get('accession')}")
            self._process_incomplete_watchlist_entry(entry, verbose=True)

        logger.info(f"Found {len(filings_8k)} 8-K filings. Running basic inspection (filters 1-5)...")

        for filing in filings_8k:
            try:
                cik = str(getattr(filing, "cik", ""))
                ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)

                if ticker and self.watched(ticker):
                    logger.info(f"Skip {ticker} (already watched)")
                    continue
                basic_passed, basic_comments = self.analyze_8k_basic(filing, False)
                if basic_passed:
                    cik = str(getattr(filing, "cik", ""))
                    ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)
                    accession = getattr(filing, "accession_no", None) or getattr(filing, "accession_number", None)

                    comments = [f"Positive 8-K earnings filing: {accession}"]

                    meta = {"filter7": None, "filter6": None, "filter8": None}
                    result, state, advanced_comments = self.analyze_8k_advanced(filing, meta, verbose=False)
                    all_comments = comments + (basic_comments or []) + (advanced_comments or [])
                    watch_meta = {
                        "cik": cik,
                        "accession": accession,
                        "filter7": state[0],
                        "filter6": state[1],
                        "filter8": state[2],
                        "comments": all_comments,
                    }
                    if result is True and ticker:
                        if market_status is None or market_status < 0:
                            self.watch(
                                ticker,
                                explanation="8-K passed; market closed",
                                days=1,
                                meta=watch_meta,
                            )
                        else:
                            explanation = " | ".join(all_comments) if all_comments else "8-K passed"
                            self.discovered(sa, ticker, explanation, None)
                            self.watch(
                                ticker,
                                explanation="8-K passed; discovered",
                                days=1,
                                meta=watch_meta,
                                status="Executed",
                            )
                    elif result is None and ticker:
                        self.watch(
                            ticker,
                            explanation=f"8-K pending filters {accession or ''}",
                            days=1,
                            meta=watch_meta,
                        )
                        logger.info(f"Added to watchlist (days=1)")
                    elif result is False and ticker:
                        self.watch(
                            ticker,
                            explanation="8-K dismissed",
                            days=1,
                            meta=watch_meta,
                            status="Excluded",
                        )
                        logger.info(f"Added to watchlist (Excluded, days=1)")

            except Exception as e:
                logger.error(f"⚠️ Error inspecting filing: {e}")

def run_edgar_standalone():
    """
    Minimal entry point for the `run_edgar` management command.

    No params yet; instantiates the advisor via python_class (same as production)
    and calls discover(sa) with a minimal SmartAnalysis().
    """
    from core.services import advisors as advisor_modules
    from core.models import Advisor, SmartAnalysis

    try:
        advisor_row = Advisor.objects.get(name="EDDIE-8")
    except Advisor.DoesNotExist:
        return None, "ED-8 advisor not found in Advisor table"

    # Resolve class via python_class (same pattern as smartanalyse)
    module_name = advisor_row.python_class.lower()
    module = getattr(advisor_modules, module_name)
    PythonClass = getattr(module, advisor_row.python_class)

    sa = SmartAnalysis()
    impl = PythonClass(advisor_row)
    impl.discover(sa)

    return "ED-8 discover() stub completed", None


register(name="ED-8", python_class="Edgar")

