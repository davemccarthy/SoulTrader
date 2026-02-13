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
from edgar import Company, get_filings, set_identity

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


def compute_filter1_pass(filing, verbose: bool = False) -> bool:
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


def compute_filter2_pass(filing, verbose: bool = False) -> bool:
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


def compute_filter3_pass(filing, verbose: bool = False) -> bool:
    """
    FILTER3: valuation, cap, and price band.
    - P/E: fail if trailing P/E > FILTER3_PE_MAX (overvalued definitive).
    - Cap: fail if market_cap < FILTER3_MIN_CAP (nano/micro).
    - Price: fail if price < FILTER3_PRICE_MIN or > FILTER3_PRICE_MAX.
    Returns True if pass, False if fail. If no ticker or yfinance data missing, passes (no block).
    """
    ticker = getattr(filing, "ticker", None)
    if not ticker:
        ticker = cik_to_ticker(getattr(filing, "cik", ""))

    try:
        time.sleep(0.05)  # Rate limit yfinance
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception as e:
        vprint(verbose, f"FILTER3: yfinance error → pass (skip checks): {e}")
        return True

    # P/E: overvalued definitive
    pe = info.get("trailingPE") or info.get("forwardPE")
    if pe is not None and isinstance(pe, (int, float)):
        if pe > FILTER3_PE_MAX:
            vprint(verbose, f"FILTER3: fail P/E {pe:.1f} > {FILTER3_PE_MAX} (overvalued)")
            return False

    # Cap: exclude nano/micro
    cap = info.get("marketCap")
    if cap is not None and isinstance(cap, (int, float)):
        if cap < FILTER3_MIN_CAP:
            vprint(verbose, f"FILTER3: fail market_cap {cap/1e6:.1f}M < {FILTER3_MIN_CAP/1e6:.0f}M (nano/micro)")
            return False

    # Price band: $5–$500
    price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
    if price is not None and isinstance(price, (int, float)):
        if price < FILTER3_PRICE_MIN:
            vprint(verbose, f"FILTER3: fail price ${price:.2f} < ${FILTER3_PRICE_MIN} (below band)")
            return False
        if price > FILTER3_PRICE_MAX:
            vprint(verbose, f"FILTER3: fail price ${price:.2f} > ${FILTER3_PRICE_MAX} (above band)")
            return False

    # Warnings (vprint only): 90%+ of 52-week high, 90%+ of 2-week lead-up high
    fifty_two_high = info.get("fiftyTwoWeekHigh")
    if (
        verbose
        and price is not None
        and fifty_two_high is not None
        and isinstance(fifty_two_high, (int, float))
        and fifty_two_high > 0
    ):
        pct_52 = 100.0 * price / fifty_two_high
        if pct_52 >= 90.0:
            vprint(
                verbose,
                f"********** WARNING: Price at {pct_52:.1f}% of 52-week high "
                f"(price ${price:.2f}, 52w high ${fifty_two_high:.2f}) **********",
            )

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
                    vprint(
                        verbose,
                        f"********** WARNING: Filing price at {pct_2w:.1f}% of 2-week high "
                        f"(filing close ${last_close:.2f}, 2w high ${two_week_high:.2f}) **********",
                    )
        except Exception as e:
            vprint(verbose, f"FILTER3: (skip 2-week high check: {e})")

    # Warnings (vprint only): sector/industry in beware lists
    if verbose:
        sector = (info.get("sector") or "").strip().lower()
        industry = (info.get("industry") or "").strip().lower()
        if sector and sector in FILTER3_UNDERPERFORM_SECTORS:
            vprint(
                verbose,
                f"********** WARNING: Sector in underperform list "
                f"(sector={info.get('sector') or 'N/A'}, industry={info.get('industry') or 'N/A'}) **********",
            )
        for watch_sector, watch_industry in FILTER3_WATCH_SECTOR_INDUSTRY:
            if sector and industry and watch_sector in sector and watch_industry in industry:
                vprint(
                    verbose,
                    f"********** WARNING: Sector/industry in watch list "
                    f"(sector={info.get('sector') or 'N/A'}, industry={info.get('industry') or 'N/A'}) **********",
                )
                break

    vprint(verbose, "FILTER3: (pass) (P/E, cap, price band ok)")
    return True


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


def compute_filter4_pass(filing, verbose: bool = False) -> bool:
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


def compute_filter5_pass(filing, verbose: bool = False) -> bool:
    """
    FILTER5: LM guidance sentiment (same logic as edgar.py).
    Pass when neg does not beat pos; skip (pass) when no text / no guidance / ps unavailable.
    """
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

    non_boilerplate = [s for s in guidance_sents if not _filter5_is_boilerplate(s)]
    vprint(verbose, f"FILTER5: guidance {len(guidance_sents)} total, {len(non_boilerplate)} after boilerplate")

    result = _filter5_compute_lm_guidance(text)
    if result is None:
        vprint(verbose, "FILTER5: (pass) no guidance after boilerplate or pysentiment2 unavailable")
        return False

    vprint(verbose, f"FILTER5: ({'pass' if result['passed'] else 'fail'}) n_sentences={result['n_sentences']} total_pos={result['total_pos']} total_neg={result['total_neg']} net_polarity={result['net_polarity']:+.3f}")
    return result["passed"]


def get_eps_for_report_date(ticker: str, report_date: date) -> Optional[Dict]:
    """Fetch Alpha Vantage EARNINGS; return quarter where reportedDate == report_date, or None."""
    api_key = getattr(settings, "ALPHAVANTAGE_API_KEY", None) or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return None
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
        target = report_date.isoformat()
        for record in quarterly:
            if record.get("reportedDate") == target:
                return record
        return None
    except Exception:
        return None


def compute_filter6_value(filing, verbose: bool = False) -> float:
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

    record = get_eps_for_report_date(ticker, report_date)
    if not record:
        return None

    surprise_str = record.get("surprisePercentage")
    try:
        surprise_val = float(surprise_str)
    except (TypeError, ValueError):
        # No usable surprise% → fail FILTER6
        return None

    if surprise_val <= 0:
        if verbose:
            vprint(verbose, f"FILTER6: {ticker} EPS surprise {surprise_val}% <= 0 → fail")
        return None

    if verbose:
        vprint(verbose, f"FILTER6: {ticker} EPS surprise {surprise_val}% → pass")

    return surprise_val


def analyze_8k(filing, verbose: bool = False):
    """
    Run FILTER1–6. If all pass, return (ticker, cik, accession, eps).
    eps is Alpha Vantage record for reportedDate == filing date, or None.
    Otherwise return None.
    """
    accession = getattr(filing, "accession_no", None) or getattr(filing, "accession_number", None)
    cik = str(getattr(filing, "cik", None))
    ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)
    if not ticker:
        return None
    print(f"Inspecting: ticker={ticker or 'N/A'}, CIK={cik}, accession={accession}")
    if not compute_filter1_pass(filing, verbose):
        return None
    if not compute_filter2_pass(filing, verbose):
        return None
    if not compute_filter4_pass(filing, verbose):
        return None
    if not compute_filter3_pass(filing, verbose):
        return None
    if not compute_filter5_pass(filing, verbose):
        return None
    if (eps_beat := compute_filter6_value(filing, verbose)) is None:
        return None

    vprint(verbose, f"SEC: https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude")
    return ticker, cik, accession, eps_beat


# ---------------------------------------------------------------------------
# ED-8 advisor class and command entry
# ---------------------------------------------------------------------------


class Edgar(AdvisorBase):
    """Advisor for 8-K earnings filings. Entry points and filters TBD."""

    def discover(self, sa):
        return
        """
        Discovery entry point for the ED-8 advisor.

        For now this is a no-op stub so we can:
          - register the advisor
          - invoke it independently via `run_edgar`
        """

        target_date = self.get_last_trading_day()

        filings = get_filings(form="8-K", filing_date=target_date)

        if not filings:
            print("No 8-K filings found for that date.")
            return

        candidates: list[tuple[str, str, str, Optional[Dict]]] = []  # (ticker, cik, accession, eps)

        print(f"Found {len(filings)} 8-K filings. Running FILTER1 + basic inspection...")
        for filing in filings:
            try:
                candidate = analyze_8k(filing)

                if candidate is not None:
                    candidates.append(candidate)

            except Exception as e:
                print(f"  ⚠️  Error inspecting filing: {e}")

        print(f"Passed {len(candidates)} candidates...")

        header = f"{'Symbol':<8} {'CIK':<8} {'Accession':<22} {'EPS':<10}"
        print(header)
        print("-" * 50)

        for candidate in candidates:
            ticker, cik, accession, eps_beat = candidate  # (ticker, cik, accession, eps)

            row = f"{ticker:<8} {cik[:8]:<8} {accession[:22]:<22} +{eps_beat:<10}"
            print(row)

        return


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

