import logging
import os
import re
from decimal import Decimal
from datetime import datetime, timedelta

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from django.utils import timezone

from django.conf import settings
from core.models import Discovery, SmartAnalysis
from core.services.advisors.advisor import AdvisorBase, register

logger = logging.getLogger(__name__)

FDA_REPORT_URL = "https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=report.page"
FDA_REPORT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}
OPENFIGI_ENDPOINT = "https://api.openfigi.com/v3/mapping"

REPORT_WINDOW_DAYS = 14
DISCOVERY_CONFIDENCE_THRESHOLD = 0.60
RECOMMENDATION_CONFIDENCE_THRESHOLD = 0.85
MAX_CONFIDENCE_SCORE = 1.5

_OPENFIGI_CACHE = {}
_MARKET_DATA_CACHE = {}

STATUS_WEIGHTS = {
    "approval": 0.30,
    "tentative approval": 0.15,
}

APPLICATION_TYPE_WEIGHTS = {
    "BLA": 0.20,
    "NDA": 0.18,
    "ANDA": 0.12,
}

SUBMISSION_PREFIX_WEIGHTS = {
    "ORIG": 0.20,
    "SUPPL": 0.08,
    "RESUB": 0.06,
    "EFFICACY": 0.10,
}

CLASSIFICATION_WEIGHTS = {
    "EFFICACY": 0.20,
    "PRIOR APPROVAL": 0.12,
    "MANUFACTURING": 0.10,
    "CMC": 0.08,
    "BIOEQUIVALENCE": 0.12,
    "PEDIATRIC": 0.08,
    "SAFETY": 0.08,
    "LABELING": 0.05,
    "REMS": 0.05,
    "BREAKTHROUGH THERAPY": 0.25,
    "FAST TRACK": 0.18,
    "PRIORITY REVIEW": 0.18,
    "ACCELERATED APPROVAL": 0.18,
}

DOSAGE_FORM_WEIGHTS = {
    "injectable": 0.08,
    "injection": 0.08,
    "tablet": 0.05,
    "capsule": 0.05,
    "oral solution": 0.04,
    "solution": 0.04,
    "suspension": 0.04,
    "topical": 0.03,
}

MATCH_CONFIDENCE_WEIGHTS = {
    "exact": 0.05,
    "normalized": 0.04,
    "openfigi": 0.04,
    "partial": 0.03,
}

MARKET_CAP_THRESHOLDS_ASC = [
    (500_000_000, 0.12),
    (2_000_000_000, 0.08),
    (10_000_000_000, 0.04),
]

MARKET_CAP_PENALTIES_DESC = [
    (200_000_000_000, -0.08),
    (100_000_000_000, -0.05),
    (50_000_000_000, -0.03),
]

PRICE_THRESHOLDS_ASC = [
    (5, 0.08),
    (20, 0.05),
    (50, 0.02),
]

PRICE_PENALTIES_DESC = [
    (300, -0.05),
    (150, -0.02),
]

FDA_SKIP_KEYWORDS = ("LABELING",)
FDA_ALLOW_IF_CONTAINS = (
    "EFFICACY",
    "PRIOR APPROVAL",
    "MANUFACTURING",
    "CMC",
    "BIOEQUIVALENCE",
    "PEDIATRIC",
    "SAFETY",
    "REMS",
    "BREAKTHROUGH",
    "FAST TRACK",
    "PRIORITY REVIEW",
    "ACCELERATED",
)


class FDA(AdvisorBase):
    def discover(self, sa):
        sa_end = _ensure_aware(sa.started)

        try:
            approvals = scrape_fda_approvals(days=REPORT_WINDOW_DAYS)
        except Exception as exc:
            logger.error("FDA advisor: failed to scrape approvals: %s", exc, exc_info=True)
            return

        if not approvals:
            logger.info("FDA advisor: no approvals returned from FDA report.")
            return

        prev_sa = (
            SmartAnalysis.objects.filter(username=sa.username, id__lt=sa.id)
            .order_by("-id")
            .first()
        )
        if prev_sa:
            sa_start = _ensure_aware(prev_sa.started)
        else:
            sa_start = sa_end - timedelta(days=REPORT_WINDOW_DAYS)

        start_date = sa_start.date()
        end_date = sa_end.date()

        logger.info(
            "FDA advisor: scanning approvals between %s and %s (inclusive)",
            start_date,
            end_date,
        )

        best_approvals = {}

        for approval in approvals:
            approval_date = _parse_approval_date(approval.get("approval_date"))
            if approval_date and (approval_date < start_date or approval_date > end_date):
                continue

            stock_symbol = approval.get("stock_symbol")
            score = approval.get("confidence_score") or 0.0

            if not stock_symbol:
                logger.debug(
                    "FDA advisor: skipping %s (%s) due to missing symbol.",
                    approval.get("drug_name"),
                    approval.get("company"),
                )
                continue

            print(f"FDA scores {stock_symbol} {score}")

            if score < DISCOVERY_CONFIDENCE_THRESHOLD:
                continue

            existing = best_approvals.get(stock_symbol)
            if not existing or score > existing.get("confidence_score", 0.0):
                best_approvals[stock_symbol] = approval

        for approval in best_approvals.values():
            stock_symbol = approval.get("stock_symbol")
            score = approval.get("confidence_score") or 0.0

            explanation = build_discovery_explanation(approval, score)
            stock = self.discovered(sa, stock_symbol, approval.get("company", ""), explanation)
            if not stock:
                continue

            if score >= RECOMMENDATION_CONFIDENCE_THRESHOLD:
                recommendation_note = build_recommendation_explanation(approval, score)
                confidence_value = Decimal(f"{min(score, MAX_CONFIDENCE_SCORE):.2f}")
                self.recommend(sa, stock, confidence_value, recommendation_note)


def scrape_fda_approvals(days=7):
    if days not in (7, 14):
        raise ValueError("days must be either 7 or 14 to match available FDA reports")

    try:
        response = requests.get(FDA_REPORT_URL, headers=FDA_REPORT_HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("FDA advisor: error fetching FDA report page: %s", exc)
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    tab_id = "example2-tab1" if days <= 7 else "example2-tab2"
    tab_content = soup.find("div", id=tab_id)

    if not tab_content:
        logger.warning("FDA advisor: could not locate the %s-day report tab on the FDA page.", days)
        return []

    approvals = []
    headings = tab_content.find_all("h4")

    if not headings:
        logger.warning("FDA advisor: no report sections found in the %s-day tab.", days)
        return []

    for heading in headings:
        report_date_str = heading.get_text(strip=True)
        table = heading.find_next("table")
        if not table:
            continue
        approvals.extend(parse_report_table(table, report_date_str))

    return approvals


def parse_report_table(table, report_date_str):
    approvals = []

    try:
        report_date = datetime.strptime(report_date_str, "%B %d, %Y").date()
    except ValueError:
        report_date = None

    tbody = table.find("tbody")
    if not tbody:
        return approvals

    rows = tbody.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 7:
            continue

        drug_name, app_type, app_number, link_url = extract_drug_info(cells[0])
        active_ingredients = cells[1].get_text(" ", strip=True)
        dosage_form = cells[2].get_text(" ", strip=True)
        submission = cells[3].get_text(strip=True)
        company = cells[4].get_text(strip=True)
        submission_classification = cells[5].get_text(strip=True)
        submission_status = cells[6].get_text(strip=True)
        normalized_status = "Approval" if submission_status.lower() == "approved" else submission_status

        classification_upper = submission_classification.upper() if submission_classification else ""
        if classification_upper and any(skip in classification_upper for skip in FDA_SKIP_KEYWORDS):
            has_other_signals = any(keyword in classification_upper for keyword in FDA_ALLOW_IF_CONTAINS)
            if not has_other_signals:
                continue

        stock_symbol, match_confidence = match_company_to_symbol(company)
        market_snapshot = get_market_snapshot(stock_symbol) if stock_symbol else None
        market_cap = market_snapshot.get("market_cap") if market_snapshot else None
        market_price = market_snapshot.get("price") if market_snapshot else None

        approval_date_str = report_date.strftime("%m/%d/%Y") if report_date else report_date_str
        approval_data = {
            "approval_date": approval_date_str,
            "report_date": report_date_str,
            "drug_name": drug_name,
            "application_type": app_type,
            "application_number": app_number,
            "dosage_form": dosage_form,
            "submission": submission,
            "active_ingredients": active_ingredients,
            "company": company,
            "stock_symbol": stock_symbol,
            "match_confidence": match_confidence,
            "market_cap": market_cap,
            "market_price": market_price,
            "submission_classification": submission_classification,
            "status": normalized_status,
            "raw_status": submission_status,
            "link": link_url,
            "month": report_date.month if report_date else None,
            "year": report_date.year if report_date else None,
        }

        approval_data["confidence_score"] = calculate_confidence_score(approval_data)
        approvals.append(approval_data)

    return approvals


def extract_drug_info(drug_cell):
    text = drug_cell.get_text(separator="\n", strip=True)
    parts = [part.strip() for part in text.split("\n") if part.strip()]

    drug_name = parts[0] if parts else ""
    app_info = " ".join(parts[1:]) if len(parts) > 1 else ""

    app_type = None
    app_number = None
    match = re.search(r"(NDA|ANDA|BLA)\s*#?\s*([0-9]+)", app_info, re.IGNORECASE)
    if match:
        app_type = match.group(1).upper()
        app_number = match.group(2).strip()

    link_url = None
    drug_link = drug_cell.find("a", href=True)
    if drug_link:
        href = drug_link.get("href", "")
        if href and not href.startswith("http"):
            link_url = "https://www.accessdata.fda.gov" + href
        else:
            link_url = href

    return drug_name, app_type, app_number, link_url


def match_company_to_symbol(company_name):
    if not company_name:
        return None, None

    company_map = _get_company_to_symbol_map()
    company_upper = company_name.upper().strip()

    if "PRIVATE LTD" in company_upper or "PRIVATE LIMITED" in company_upper:
        return None, None
    if company_upper in company_map:
        return company_map[company_upper], "exact"

    normalized = _normalize_company_name(company_name)
    if normalized in company_map:
        return company_map[normalized], "normalized"

    for key, symbol in company_map.items():
        if key in company_upper or company_upper in key:
            return symbol, "partial"

    openfigi_symbol = lookup_symbol_via_openfigi(company_name)
    if openfigi_symbol:
        return openfigi_symbol, "openfigi"

    return None, None


def lookup_symbol_via_openfigi(company_name, api_key=None):
    if not company_name:
        return None

    cache_key = company_name.strip().upper()
    if cache_key in _OPENFIGI_CACHE:
        return _OPENFIGI_CACHE[cache_key]

    api_key = getattr(settings, 'OPENFIGI_API_KEY', None)
    if not api_key:
        _OPENFIGI_CACHE[cache_key] = None
        return None

    payload = [
        {
            "idType": "NAME",
            "idValue": company_name,
            "securityType2": "Common Stock",
        }
    ]
    headers = {
        "Content-Type": "application/json",
        "X-OPENFIGI-APIKEY": api_key,
    }

    try:
        response = requests.post(OPENFIGI_ENDPOINT, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        logger.warning("FDA advisor: OpenFIGI lookup failed for '%s'", company_name)
        _OPENFIGI_CACHE[cache_key] = None
        return None

    try:
        results = response.json()
    except ValueError:
        logger.warning("FDA advisor: OpenFIGI returned invalid JSON for '%s'", company_name)
        _OPENFIGI_CACHE[cache_key] = None
        return None

    if not isinstance(results, list) or not results:
        _OPENFIGI_CACHE[cache_key] = None
        return None

    mapping_result = results[0] or {}
    data_entries = mapping_result.get("data") or []

    ticker = _select_best_openfigi_ticker(data_entries)
    _OPENFIGI_CACHE[cache_key] = ticker
    return ticker


def _select_best_openfigi_ticker(entries):
    if not entries:
        return None

    preferred_exchanges = {"XNYS", "XNAS", "ARCX", "BATS"}
    for entry in entries:
        ticker = entry.get("ticker")
        if not ticker:
            continue
        security_type = (entry.get("securityType2") or "").lower()
        if entry.get("exchCode") in preferred_exchanges and security_type.startswith("common stock"):
            return ticker

    for entry in entries:
        ticker = entry.get("ticker")
        if not ticker:
            continue
        security_type = (entry.get("securityType2") or "").lower()
        if "stock" in security_type or "equity" in security_type:
            return ticker

    for entry in entries:
        ticker = entry.get("ticker")
        if ticker:
            return ticker

    return None


def get_market_snapshot(symbol):
    if not symbol:
        return None

    cache_key = symbol.upper()
    if cache_key in _MARKET_DATA_CACHE:
        return _MARKET_DATA_CACHE[cache_key]

    try:
        ticker = yf.Ticker(symbol)
        market_cap = None
        price = None

        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            market_cap = getattr(fast_info, "market_cap", None)
            price = getattr(fast_info, "last_price", None)

        if market_cap is None or price is None:
            info = ticker.info
            if market_cap is None:
                market_cap = info.get("marketCap")
            if price is None:
                price = info.get("regularMarketPrice") or info.get("currentPrice")

        snapshot = {"market_cap": market_cap, "price": price}
        _MARKET_DATA_CACHE[cache_key] = snapshot
        return snapshot
    except Exception as exc:
        logger.warning("FDA advisor: unable to fetch market data for %s: %s", symbol, exc)
        _MARKET_DATA_CACHE[cache_key] = None
        return None


def calculate_confidence_score(approval):
    score = 0.0

    status = (approval.get("status") or "").lower()
    score += STATUS_WEIGHTS.get(status, 0.05)

    application_type = (approval.get("application_type") or "").upper()
    score += APPLICATION_TYPE_WEIGHTS.get(application_type, 0.08)

    submission = (approval.get("submission") or "").upper()
    prefix_weight = 0.0
    if submission:
        prefix = submission.split("-")[0]
        for key, weight in SUBMISSION_PREFIX_WEIGHTS.items():
            if prefix.startswith(key):
                prefix_weight = weight
                break
        if prefix_weight == 0.0:
            prefix_weight = 0.05
    score += prefix_weight

    classification = (approval.get("submission_classification") or "").upper()
    if classification:
        matched_any = False
        for key, weight in CLASSIFICATION_WEIGHTS.items():
            if key in classification:
                score += weight
                matched_any = True
        if not matched_any:
            score += 0.05

    dosage_form = (approval.get("dosage_form") or "").lower()
    for key, weight in DOSAGE_FORM_WEIGHTS.items():
        if key in dosage_form:
            score += weight
            break

    match_confidence = (approval.get("match_confidence") or "").lower()
    if match_confidence:
        score += MATCH_CONFIDENCE_WEIGHTS.get(match_confidence, 0.02)
    else:
        score -= 0.02

    market_cap = approval.get("market_cap")
    if market_cap:
        added = False
        for threshold, weight in MARKET_CAP_THRESHOLDS_ASC:
            if market_cap <= threshold:
                score += weight
                added = True
                break
        if not added:
            for threshold, penalty in MARKET_CAP_PENALTIES_DESC:
                if market_cap >= threshold:
                    score += penalty
                    break

    market_price = approval.get("market_price")
    if market_price:
        added = False
        for threshold, weight in PRICE_THRESHOLDS_ASC:
            if market_price <= threshold:
                score += weight
                added = True
                break
        if not added:
            for threshold, penalty in PRICE_PENALTIES_DESC:
                if market_price >= threshold:
                    score += penalty
                    break

    if not approval.get("stock_symbol"):
        score *= 0.4

    if application_type == "BLA" and "EFFICACY" in classification:
        score *= 1.15
    elif application_type == "ANDA" and "LABELING" in classification:
        score *= 0.9
    if "BREAKTHROUGH THERAPY" in classification:
        score *= 1.2
        if market_cap and market_cap <= 2_000_000_000:
            score += 0.08

    return max(0.0, min(score, MAX_CONFIDENCE_SCORE))


def build_discovery_explanation(approval, score):
    parts = [
        f"Confidence score {score:.2f}",
        f"{approval.get('company')} received {approval.get('status')} for {approval.get('drug_name')}",
    ]

    classification = approval.get("submission_classification")
    if classification:
        parts.append(f"Classification: {classification}")
    app_type = approval.get("application_type")
    if app_type:
        parts.append(f"Application: {app_type} #{approval.get('application_number')}")

    market_cap = approval.get("market_cap")
    if market_cap:
        parts.append(f"Market cap: {_humanize_number(market_cap)}")
    market_price = approval.get("market_price")
    if market_price:
        parts.append(f"Share price: ${market_price:.2f}")

    link = approval.get("link")
    if link:
        parts.append("Article: FDA approval details")
        parts.append(link)

    return " | ".join(filter(None, parts))


def build_recommendation_explanation(approval, score):
    pieces = [
        f"Confidence score {score:.2f}",
        f"{approval.get('status')} for {approval.get('drug_name')} ({approval.get('application_type')} #{approval.get('application_number')})",
    ]
    classification = approval.get("submission_classification")
    if classification:
        pieces.append(f"Classification: {classification}")
    return " | ".join(filter(None, pieces))


def _parse_approval_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%m/%d/%Y").date()
    except ValueError:
        return None


def _ensure_aware(dt):
    if not dt:
        return timezone.now()
    if timezone.is_naive(dt):
        return timezone.make_aware(dt)
    return dt


def _get_company_to_symbol_map():
    return {
        "PFIZER": "PFE",
        "JOHNSON & JOHNSON": "JNJ",
        "JANSSEN": "JNJ",
        "JANSSEN BIOTECH": "JNJ",
        "MERCK": "MRK",
        "MERCK & CO": "MRK",
        "ABBV": "ABBV",
        "ABBVIE": "ABBV",
        "GILEAD": "GILD",
        "GILEAD SCIENCES": "GILD",
        "BRISTOL MYERS SQUIBB": "BMY",
        "BMS": "BMY",
        "AMGEN": "AMGN",
        "ELI LILLY": "LLY",
        "LILLY": "LLY",
        "NOVARTIS": "NVS",
        "ROCHE": "RHHBY",
        "GLAXOSMITHKLINE": "GSK",
        "GSK": "GSK",
        "SANOFI": "SNY",
        "ASTRAZENECA": "AZN",
        "TEVA": "TEVA",
        "TEVA PHARMACEUTICAL": "TEVA",
        "TEVA PHARMS USA": "TEVA",
        "TEVA PHARMACEUTICALS": "TEVA",
        "MYLAN": "VTRS",
        "VIATRIS": "VTRS",
        "SUN PHARMA": "SUNPF",
        "SUN PHARMACEUTICAL": "SUNPF",
        "DR REDDYS": "RDY",
        "DR. REDDYS": "RDY",
        "DR REDDYS LABORATORIES": "RDY",
        "AUROBINDO": "AUPH",
        "AUROBINDO PHARMA": "AUPH",
        "AUROBINDO PHARMA LTD": "AUPH",
        "HOSPIRA": "HSP",
        "HOSPIRA INC": "HSP",
        "CENTOCOR": "JNJ",
        "CENTOCOR ORTHO BIOTECH": "JNJ",
        "PHARMACIA": "PFE",
        "PHARMACIA AND UPJOHN": "PFE",
        "UPJOHN": "PFE",
        "SCYNEXIS": "SCYX",
        "UCB": "UCB",
        "UCB INC": "UCB",
        "AMNEAL": "AMRX",
        "AMNEAL PHARMACEUTICALS": "AMRX",
        "AMNEAL PHARMS": "AMRX",
        "SAGENT": "SGNT",
        "SAGENT PHARMS": "SGNT",
        "SAGENT PHARMACEUTICALS": "SGNT",
        "BRECKENRIDGE": "BRX",
    }


def _normalize_company_name(company_name):
    if not company_name:
        return ""

    normalized = company_name.upper().strip()
    suffixes = [
        " INC",
        " INC.",
        " CORPORATION",
        " CORP",
        " CORP.",
        " LTD",
        " LTD.",
        " LIMITED",
        " LLC",
        " LLC.",
        " PHARMACEUTICALS",
        " PHARMA",
        " PHARMACEUTICAL",
        " PHARMS",
        " LABORATORIES",
        " LABS",
        " COMPANY",
        " CO",
    ]

    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()

    return normalized


def _humanize_number(value):
    if value is None:
        return None
    abs_value = abs(value)
    for divisor, suffix in ((1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M")):
        if abs_value >= divisor:
            return f"{value / divisor:.2f}{suffix}"
    if abs_value >= 1_000:
        return f"{value:,.0f}"
    return f"{value:.2f}"


register(name="FDA", python_class="FDA")