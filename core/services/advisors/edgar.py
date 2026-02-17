"""
Edgar advisor (ED-8): 8-K earnings–related filings, filters 1–6, analyze_8k.

Test independently: python manage.py run_edgar

Production: discover() will fetch 8-Ks, run filters (1–6, sector, reception),
and call self.discovered(...) for passing stocks.
"""

import logging
import os
import re
import time
from collections import Counter
from typing import Dict, Optional
from datetime import date, timedelta

import requests
import yfinance as yf
from django.conf import settings
from edgar import Company, get_filings, get_latest_filings, set_identity

from core.services.advisors.advisor import AdvisorBase, register


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 8-K helpers and constants (used by analyze_8k)
# ---------------------------------------------------------------------------
_CIK_TO_TICKER_CACHE: Dict[str, Optional[str]] = {}

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
        return ticker
    except Exception:
        _CIK_TO_TICKER_CACHE[cik] = None
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


FILTER3_PE_MAX = 100
FILTER3_MIN_CAP = 300e6
FILTER3_PRICE_MIN = 5.0
FILTER3_PRICE_MAX = 1000.0

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
]


def _filter5_split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    return [p.strip() for p in parts if p.strip()]


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
    candidates = [s for s in guidance if not _filter5_is_boilerplate(s)]
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
        resp.raise_for_status()
        data = resp.json()
        if "Error Message" in data or "Note" in data or "Information" in data:
            return None
        quarterly = data.get("quarterlyEarnings") or []
        record = None
        for r in quarterly:
            fd = r.get("fiscalDateEnding") or ""
            if fd[:10] == target if len(fd) >= 10 else fd == target:
                record = r
                break
        if record is None:
            print("quarterly record not available")
            return None
        sp = record.get("surprisePercentage")
        if sp is None or sp == "" or str(sp).strip() == "None":
            print("quarterly record available but no EPSSurprise")
            return None
        try:
            float(sp)
        except (TypeError, ValueError):
            print("quarterly record available but no EPSSurprise")
            return None
        return record
    except Exception:
        return None


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
        if verbose and price is not None and fifty_two_high is not None and isinstance(fifty_two_high, (int, float)) and fifty_two_high > 0:
            pct_52 = 100.0 * price / fifty_two_high
            if pct_52 >= 90.0:
                vprint(verbose, f"********** WARNING: Price at {pct_52:.1f}% of 52-week high **********")
        fd = getattr(filing, "filing_date", None)
        if verbose and price is not None and fd is not None:
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
                        vprint(verbose, f"********** WARNING: Filing price at {pct_2w:.1f}% of 2-week high **********")
            except Exception as e:
                vprint(verbose, f"FILTER3: (skip 2-week high check: {e})")
        if verbose:
            sector = (info.get("sector") or "").strip().lower()
            industry = (info.get("industry") or "").strip().lower()
            if sector and sector in FILTER3_UNDERPERFORM_SECTORS:
                vprint(verbose, f"********** WARNING: Sector in underperform list **********")
            for watch_sector, watch_industry in FILTER3_WATCH_SECTOR_INDUSTRY:
                if sector and industry and watch_sector in sector and watch_industry in industry:
                    vprint(verbose, f"********** WARNING: Sector/industry in watch list **********")
                    break
        vprint(verbose, "FILTER3: (pass) (P/E, cap, price band ok)")
        return True

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
        guidance_sents = _filter5_extract_guidance_sentences(text)
        if not guidance_sents:
            vprint(verbose, "FILTER5: (fail) no guidance sentences")
            return False
        vprint(verbose, f"FILTER5: guidance {len(guidance_sents)} total, {len([s for s in guidance_sents if not _filter5_is_boilerplate(s)])} after boilerplate")
        result = _filter5_compute_lm_guidance(text)
        if result is None:
            vprint(verbose, "FILTER5: (pass) no guidance after boilerplate or pysentiment2 unavailable")
            return False
        vprint(verbose, f"FILTER5: ({'pass' if result['passed'] else 'fail'}) n_sentences={result['n_sentences']} net_polarity={result['net_polarity']:+.3f}")
        return result["passed"]

    def compute_filter6_pass(self, filing, verbose: bool = False):
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(str(getattr(filing, "cik", "")))
        if not ticker:
            return None
        fd = getattr(filing, "filing_date", None)
        if fd is None:
            return None
        if isinstance(fd, str):
            try:
                report_date = date.fromisoformat(fd[:10])
            except ValueError:
                return None
        else:
            report_date = getattr(fd, "date", lambda: fd)() if hasattr(fd, "date") else fd
        record = get_eps_for_report_quarter(ticker, report_date)
        if not record:
            return None
        surprise_str = record.get("surprisePercentage")
        reported_str = record.get("reportedEPS")
        try:
            surprise_val = float(surprise_str)
            reported_val = float(reported_str)

            if reported_val == 0:
                return None

            eps_score = min(surprise_val, 50) * (reported_val ** 0.5)
        except (TypeError, ValueError):
            return None
        if verbose:
            vprint(verbose, f"FILTER6: {ticker} EPS surprise {surprise_str}% reported {reported_str} score {eps_score}")
        return eps_score >= 10


    def compute_filter7_pass(self, filing, verbose: bool = False):
        """
        FILTER7: 3-ducks LLM (past/future/expectation). Uses AdvisorBase.ask_gemini.
        Strict pass: past_performance >= 0, future_performance >= 1, expectation_gap >= 1.
        Returns True (pass), False (fail), or None (no text / LLM error / inconclusive).
        """
        text = _get_ex99_text(filing)
        if not text.strip():
            vprint(verbose, "FILTER7: no EX-99.x text → None")
            return None
        prompt = THREE_DUCKS_PROMPT.replace("<<<EX99_1_TEXT>>>", text.strip())
        model, results = self.ask_gemini(prompt, timeout=120.0)
        if not results:
            vprint(verbose, "FILTER7: ask_gemini returned no results → None")
            return None
        try:
            past = results.get("past_performance")
            future = results.get("future_performance")
            gap = results.get("expectation_gap")
            past = int(past) if past is not None else None
            future = int(future) if future is not None else None
            gap = int(gap) if gap is not None else None
        except (TypeError, ValueError):
            vprint(verbose, "FILTER7: could not parse scores → None")
            return None
        if past is None or future is None or gap is None:
            vprint(verbose, "FILTER7: missing score(s) → None")
            return None
        if verbose:
            vprint(verbose, f"FILTER7: past={past} future={future} expectation_gap={gap}")
        # Strict: past >= 0, future >= 1, expectation_gap >= 1
        if past < 0 or future < 1 or gap < 1:
            return False
        return True

    def compute_filter8_pass(self, filing, verbose: bool = False):
        vprint(verbose, "FILTER8: (stub) media not implemented → None")
        return None

    def analyze_8k_advanced(self, filing, meta_requirements: dict, verbose: bool = False):
        if meta_requirements.get("filter7") is None:
            f7 = self.compute_filter7_pass(filing, verbose)
            if f7 is False or f7 is None:
                return (f7, (None, None, None))
        if meta_requirements.get("filter6") is None:
            f6 = self.compute_filter6_pass(filing, verbose)
            if f6 is False or f6 is None:
                return (f6, (True, None, None))
        if meta_requirements.get("filter8") is None:
            f8 = self.compute_filter8_pass(filing, verbose)
            if f8 is False or f8 is None:
                return (f8, (True, True, None))
        return (True, (True, True, True))

    def analyze_8k_basic(self, filing, verbose: bool = False):
        accession = getattr(filing, "accession_no", None) or getattr(filing, "accession_number", None)
        cik = str(getattr(filing, "cik", None))
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)

        logger.info(f"Inpecting: ticker={ticker or 'N/A'}, CIK={cik}, accession={accession}")

        if not ticker:
            return False
        if not self.compute_filter1_pass(filing, verbose):
            return False
        if not self.compute_filter2_pass(filing, verbose):
            return False
        if not self.compute_filter4_pass(filing, verbose):
            return False
        if not self.compute_filter3_pass(filing, verbose):
            return False
        if not self.compute_filter5_pass(filing, verbose):
            return False
        return True

    def discover(self, sa):

        """
        Discovery entry point for the ED-8 advisor.

        For now this is a no-op stub so we can:
          - register the advisor
          - invoke it independently via `run_edgar`
        """

        print("Fetching latest filings...")
        latest = get_latest_filings()

        try:
            filings = list(latest)
        except Exception as e:
            print(f"❌ Error converting latest filings to list: {e}")
            return

        # Filter to 8-Ks only
        filings_8k = [f for f in filings if getattr(f, "form", None) == "8-K"]
        if not filings_8k:
            print("No latest 8-K filings found. Using last day for tests")
            return

        print(f"Found {len(filings_8k)} 8-K filings. Running basic inspection (filters 1-5)...")
        for filing in filings_8k:
            try:
                if self.analyze_8k_basic(filing, True):
                    cik = str(getattr(filing, "cik", ""))
                    ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)
                    accession = getattr(filing, "accession_no", None) or getattr(filing, "accession_number", None)
                    company = getattr(filing, "company", None) or ""
                    print(f"\n--- Candidate: {ticker or 'N/A'} | CIK={cik} | accession={accession or 'N/A'} ---")
                    if company:
                        print(f"    Company: {company}")
                    print(f"    SEC: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K")
                    meta = {"filter7": None, "filter6": None, "filter8": None}
                    result, state = self.analyze_8k_advanced(filing, meta, verbose=True)
                    print(f"    Advanced: result={result} state={state}")
                    # TODO: if result is True → buy; if False → dismiss; if None → add to watchlist with state
                    pass

            except Exception as e:
                print(f"  ⚠️  Error inspecting filing: {e}")

        """
        TODO: Retrieve from watchlist oldest first
        
        if not 3ducks
            get 3ducks
            On fail delete from eatchlist
            On no LLM continue
            
        if not eps ...
        if not media ...
    
        if eps and 3ducks and media
            discover
        """


def run_edgar_standalone():
    """
    Minimal entry point for the `run_edgar` management command.

    No params yet; instantiates the advisor via python_class (same as production)
    and calls discover(sa) with a minimal SmartAnalysis().
    """
    from core.services import advisors as advisor_modules
    from core.models import Advisor, SmartAnalysis

    try:
        advisor_row = Advisor.objects.get(name="ED-8")
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

