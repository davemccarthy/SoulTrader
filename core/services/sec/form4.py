from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from edgar import Company, get_filings, get_latest_filings, set_identity

logger = logging.getLogger(__name__)

_DEFAULT_SEC_IDENTITY = "Dave McCarthy dave@klynt.com"
_CIK_TO_TICKER_CACHE: Dict[str, Optional[str]] = {}


def configure_sec_identity() -> None:
    """Set the SEC identity required by the EDGAR API."""
    identity = (os.environ.get("SEC_EDGAR_IDENTITY") or _DEFAULT_SEC_IDENTITY).strip()
    set_identity(identity)


configure_sec_identity()


F4_IGNORE_CODES = frozenset({"A", "F", "G"})
F4_MIN_TRADE_VALUE_USD = 25_000.0
F4_WEIGHT_CODE = {"P": 4, "S": -1}
F4_DOLLAR_TIERS: Tuple[Tuple[int, int], ...] = (
    (1_000_000, 3),
    (250_000, 2),
    (50_000, 1),
)
F4_PCT_TIERS: Tuple[Tuple[float, int], ...] = (
    (0.30, 3),
    (0.15, 2),
    (0.05, 1),
)


@dataclass
class Form4LineScore:
    code: str
    date: str
    shares: int
    price: float
    value_usd: float
    post_remaining: int
    security: str
    components: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    skipped: bool = False
    raw_score: float = 0.0


@dataclass
class Form4FilingScore:
    accession: str
    cik: str
    issuer_cik: str
    ticker: str
    issuer_name: str
    insider_name: str
    role_label: str
    role_mod: int
    period: str
    lines: List[Form4LineScore] = field(default_factory=list)
    filing_subtotal: float = 0.0
    cluster_adj: float = 0.0
    total: float = 0.0


def form4_filing_score_to_dict(fs: Form4FilingScore) -> Dict[str, Any]:
    return asdict(fs)


def cik_to_ticker(cik: str) -> Optional[str]:
    """Map CIK to ticker symbol using the edgar Company helper."""
    cik = str(cik).zfill(10)
    if cik in _CIK_TO_TICKER_CACHE:
        return _CIK_TO_TICKER_CACHE[cik]
    try:
        ticker = Company(cik).get_ticker()
        _CIK_TO_TICKER_CACHE[cik] = ticker
        return ticker
    except Exception:
        _CIK_TO_TICKER_CACHE[cik] = None
        return None


def _f4_f_num(x: Any, default: float = 0.0) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _f4_f_int(x: Any, default: int = 0) -> int:
    v = _f4_f_num(x, float("nan"))
    if v != v:
        return default
    return int(abs(v))


def _f4_normalize_title(title: Optional[str]) -> str:
    if not title:
        return ""
    return re.sub(r"\s+", " ", title.strip().lower())


def _f4_insider_role_modifier(owner: Any) -> Tuple[int, str]:
    title = _f4_normalize_title(getattr(owner, "officer_title", None) or "")
    is_dir = bool(getattr(owner, "is_director", False))
    is_10 = bool(getattr(owner, "is_ten_pct_owner", False))
    if any(k in title for k in ("chief executive", "ceo")):
        return 2, "CEO-tier"
    if any(k in title for k in ("chief financial", "cfo")):
        return 2, "CFO-tier"
    if any(k in title for k in ("chief operating", "coo", "president")):
        return 1, "COO/President-tier"
    if is_dir:
        return 1, "Director"
    if is_10:
        return 0, "10% holder"
    if getattr(owner, "is_officer", False):
        return 0, "Officer (other)"
    return 0, "Other / unknown"


def _f4_dollar_weight(value_usd: float) -> int:
    av = abs(value_usd)
    for threshold, weight in F4_DOLLAR_TIERS:
        if av >= threshold:
            return weight
    return 0


def _f4_holdings_pct_weight(shares: int, post_remaining: int) -> int:
    if post_remaining <= 0 or shares <= 0:
        return 0
    pct = shares / float(post_remaining)
    for threshold, weight in F4_PCT_TIERS:
        if pct >= threshold:
            return weight
    return 0


def _f4_footnote_blob(form4: Any, row: pd.Series) -> str:
    raw = str(row.get("footnotes") or "")
    try:
        resolved = form4._resolve_footnotes(raw)  # noqa: SLF001
    except Exception:
        resolved = ""
    remarks = getattr(form4, "remarks", None) or ""
    return f"{resolved} {remarks}".lower()


def _f4_mentions_10b5_1(text: str) -> bool:
    if not text:
        return False
    return "10b5" in text or "10b-5" in text


def _f4_mentions_tax_withhold(text: str) -> bool:
    return "withhold" in text or "tax" in text


def _f4_dates_by_code(df: pd.DataFrame) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        code = str(row.get("Code") or "")
        d = str(row.get("Date") or "")
        if d:
            out[code].add(d)
    return out


def _f4_score_nonderivative_row(form4: Any, owner: Any, row: pd.Series) -> Form4LineScore:
    code = str(row.get("Code") or "").strip().upper()
    d = str(row.get("Date") or "")
    sec = str(row.get("Security") or "")
    shares = _f4_f_int(row.get("Shares"), 0)
    price = _f4_f_num(row.get("Price"), 0.0)
    remaining = _f4_f_int(row.get("Remaining"), 0)
    value = float(shares) * price if price > 0 else 0.0
    notes: List[str] = []

    if code in F4_IGNORE_CODES:
        return Form4LineScore(
            code, d, shares, price, value, remaining, sec, notes=["ignored code"], skipped=True
        )
    if code not in F4_WEIGHT_CODE and code not in ("M", "X"):
        notes.append("non-P/S code - not scored in v1")
        return Form4LineScore(code, d, shares, price, value, remaining, sec, notes=notes, skipped=True)
    if code in ("M", "X"):
        notes.append("exercise / conversion - not scored as open-market signal")
        return Form4LineScore(code, d, shares, price, value, remaining, sec, notes=notes, skipped=True)

    fn = _f4_footnote_blob(form4, row)
    acquired = str(row.get("AcquiredDisposed") or "").upper()
    is_buy = acquired == "A"
    is_sell = acquired == "D"
    if not is_buy and not is_sell:
        return Form4LineScore(code, d, shares, price, value, remaining, sec, notes=["no A/D"], skipped=True)
    if code == "P" and not is_buy:
        notes.append("code P but disposition - check filing")
    if code == "S" and not is_sell:
        notes.append("code S but acquisition - check filing")
    if value < F4_MIN_TRADE_VALUE_USD and price > 0:
        return Form4LineScore(
            code, d, shares, price, value, remaining, sec, notes=["below min notional"], skipped=True
        )

    role_mod, role_lbl = _f4_insider_role_modifier(owner)
    _ = role_lbl
    base = float(F4_WEIGHT_CODE.get(code, 0))
    dw = _f4_dollar_weight(value)
    pw = _f4_holdings_pct_weight(shares, remaining)
    sign = 1.0 if is_buy else -1.0
    comp: Dict[str, float] = {
        "type": base,
        "dollar": sign * dw,
        "holdings_pct": sign * pw,
        "role": float(role_mod) if is_buy else float(role_mod) * 0.5,
    }
    raw = sum(comp.values())
    if is_sell and _f4_mentions_10b5_1(fn):
        raw = raw + 1.0
        notes.append("10b5-1 language in footnotes - reduced bearish weight")
    tx_df = form4.non_derivative_table.transactions.data
    by_code_date = _f4_dates_by_code(tx_df)
    exercise_codes = by_code_date.get("M", set()) | by_code_date.get("X", set())
    if is_sell and d in exercise_codes:
        raw *= 0.25
        notes.append("same-date exercise row present - possible exercise+sale (attenuated)")
    if is_sell and _f4_mentions_tax_withhold(fn) and code == "S":
        notes.append("footnote mentions tax - verify not routine withhold")

    return Form4LineScore(
        code=code,
        date=d,
        shares=shares,
        price=price,
        value_usd=value,
        post_remaining=remaining,
        security=sec,
        components=comp,
        notes=notes,
        skipped=False,
        raw_score=raw,
    )


def _f4_accession(filing: Any) -> str:
    return str(
        getattr(filing, "accession_no", None)
        or getattr(filing, "accession_number", None)
        or ""
    )


def score_form4_filing(filing: Any) -> Optional[Form4FilingScore]:
    form = getattr(filing, "form", None)
    acc = _f4_accession(filing)
    if form not in ("4", "4/A"):
        logger.debug("Form 4 skip accession=%s: wrong form=%r", acc or "?", form)
        return None
    try:
        obj = filing.obj()
    except Exception as e:
        logger.warning("Form 4 parse fail accession=%s: %s", acc or "?", e)
        return None
    owners = getattr(obj.reporting_owners, "owners", None) or []
    if not owners:
        logger.warning("Form 4 skip accession=%s: no reporting owners", acc)
        return None
    owner = owners[0]
    role_mod, role_lbl = _f4_insider_role_modifier(owner)
    issuer_cik_raw = str(obj.issuer.cik or "").strip()
    issuer_cik = issuer_cik_raw.zfill(10) if issuer_cik_raw else ""
    ticker = (obj.issuer.ticker or "").strip().upper() or (cik_to_ticker(issuer_cik) or "")
    insider = obj.insider_name or owner.name
    lines: List[Form4LineScore] = []
    if obj.non_derivative_table.has_transactions:
        df = obj.non_derivative_table.transactions.data
        for _, row in df.iterrows():
            lines.append(_f4_score_nonderivative_row(obj, owner, row))
    active = [ln for ln in lines if not ln.skipped]
    subtotal = sum(ln.raw_score for ln in active)
    return Form4FilingScore(
        accession=acc,
        cik=str(filing.cik),
        issuer_cik=issuer_cik or str(filing.cik).zfill(10),
        ticker=ticker or "?",
        issuer_name=obj.issuer.name or "",
        insider_name=insider,
        role_label=role_lbl,
        role_mod=role_mod,
        period=str(obj.reporting_period or ""),
        lines=lines,
        filing_subtotal=subtotal,
        cluster_adj=0.0,
        total=subtotal,
    )


def _f4_week_key_issuer(issuer_cik: str, date_str: str) -> Optional[Tuple[str, int, int]]:
    cik = (issuer_cik or "").strip().zfill(10)
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    iso = d.isocalendar()
    return (cik, int(iso[0]), int(iso[1]))


def apply_form4_cluster_adjustments(results: List[Form4FilingScore]) -> None:
    p_insiders: Dict[Tuple[str, int, int], Set[str]] = defaultdict(set)
    s_insiders: Dict[Tuple[str, int, int], Set[str]] = defaultdict(set)
    for r in results:
        for ln in r.lines:
            if ln.skipped or not ln.date:
                continue
            wk = _f4_week_key_issuer(r.issuer_cik, ln.date)
            if not wk:
                continue
            if ln.code == "P":
                p_insiders[wk].add(r.insider_name)
            elif ln.code == "S":
                s_insiders[wk].add(r.insider_name)
    for r in results:
        adj = 0.0
        bullish_weeks: Set[Tuple[str, int, int]] = set()
        bearish_weeks: Set[Tuple[str, int, int]] = set()
        for ln in r.lines:
            if ln.skipped or not ln.date:
                continue
            wk = _f4_week_key_issuer(r.issuer_cik, ln.date)
            if not wk:
                continue
            if ln.code == "P":
                bullish_weeks.add(wk)
            elif ln.code == "S":
                bearish_weeks.add(wk)
        for wk in bullish_weeks:
            n = len(p_insiders.get(wk, ()))
            if n >= 3:
                adj += 3.0
            elif n == 2:
                adj += 1.0
        for wk in bearish_weeks:
            n = len(s_insiders.get(wk, ()))
            if n >= 3:
                adj -= 3.0
            elif n == 2:
                adj -= 1.0
        r.cluster_adj = adj
        r.total = r.filing_subtotal + adj


def score_form4_filings(
    filings: List[Any],
    *,
    apply_cluster: bool = True,
) -> List[Form4FilingScore]:
    results: List[Form4FilingScore] = []
    for filing in filings:
        fs = score_form4_filing(filing)
        if fs is not None:
            results.append(fs)
    if apply_cluster and results:
        apply_form4_cluster_adjustments(results)
    logger.info("Form 4 scored %d / %d filings (cluster=%s)", len(results), len(filings), apply_cluster)
    return results


def fetch_latest_form4_filings(page_size: int = 100) -> List[Any]:
    page = max(40, page_size)
    primary = list(get_latest_filings(form="4", page_size=page))
    amendments = list(get_latest_filings(form="4/A", page_size=min(page, 100)))
    merged: List[Any] = []
    seen: Set[str] = set()
    for filing in primary + amendments:
        aid = getattr(filing, "accession_no", None) or getattr(filing, "accession_number", None)
        if aid and aid in seen:
            continue
        if aid:
            seen.add(aid)
        merged.append(filing)
    logger.info("Form 4 fetched %d filings (4 + 4/A, page_size=%d)", len(merged), page)
    return merged


def fetch_form4_filings_for_date(filing_date: str, limit: int = 500) -> List[Any]:
    """SEC filing date YYYY-MM-DD - for backfill."""
    filings = list(get_filings(form="4", filing_date=filing_date))[:limit]
    if len(filings) < limit:
        need = limit - len(filings)
        filings.extend(list(get_filings(form="4/A", filing_date=filing_date))[:need])
    out = filings[:limit]
    logger.info("Form 4 fetched %d filings for filing_date=%s", len(out), filing_date)
    return out


def fetch_company_form4_filings(
    symbol: str,
    *,
    start_date: date,
    end_date: date,
    limit: int = 100,
) -> List[Any]:
    symbol = symbol.strip().upper()
    try:
        company = Company(symbol)
    except Exception as exc:
        raise ValueError(f"Could not resolve SEC company for {symbol}: {exc}") from exc

    date_range = f"{start_date.isoformat()}:{end_date.isoformat()}"
    filings = company.get_filings(form=["4", "4/A"], date=date_range)
    out = list(filings or [])[:limit]
    logger.info(
        "Form 4 fetched %d filings for symbol=%s range=%s..%s",
        len(out),
        symbol,
        start_date,
        end_date,
    )
    return out


def _resolve_date_range(
    *,
    days: int,
    start_date: Optional[date],
    end_date: Optional[date],
) -> Tuple[date, date]:
    end = end_date or date.today()
    start = start_date or (end - timedelta(days=max(days, 1)))
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    return start, end


def _score_sort_key(score: Form4FilingScore) -> str:
    line_dates = [line.date for line in score.lines if line.date]
    if line_dates:
        return max(line_dates)
    return score.period or ""


def _has_scoreable_lines(score: Form4FilingScore) -> bool:
    return any(not line.skipped for line in score.lines)


def get_form4_intel(
    symbol: str,
    *,
    days: int = 30,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = 100,
    min_abs_total: float = 0.0,
) -> Dict[str, Any]:
    """Return on-demand Form 4 intel for one stock over a date range."""
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("symbol is required")

    start, end = _resolve_date_range(days=days, start_date=start_date, end_date=end_date)
    filings = fetch_company_form4_filings(symbol, start_date=start, end_date=end, limit=limit)
    parsed_scores = score_form4_filings(filings, apply_cluster=True)
    scores = [score for score in parsed_scores if _has_scoreable_lines(score)]
    if min_abs_total > 0:
        scores = [score for score in scores if abs(float(score.total or 0.0)) >= min_abs_total]
    scores.sort(key=_score_sort_key, reverse=True)

    buy_scores = [score for score in scores if float(score.total or 0.0) > 0]
    sell_scores = [score for score in scores if float(score.total or 0.0) < 0]
    net_total = sum(float(score.total or 0.0) for score in scores)
    buy_total = sum(float(score.total or 0.0) for score in buy_scores)
    sell_total = sum(float(score.total or 0.0) for score in sell_scores)
    latest = scores[0] if scores else None

    return {
        "symbol": symbol,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "filing_count": len(filings),
        "parsed_count": len(parsed_scores),
        "scored_count": len(scores),
        "entry_count": len(scores),
        "watch_kind": ("form4_signal" if net_total >= 0 else "form4_sell") if scores else None,
        "total": net_total,
        "buy_total": buy_total,
        "sell_total": sell_total,
        "buy_count": len(buy_scores),
        "sell_count": len(sell_scores),
        "form4_accession": latest.accession if latest else None,
        "issuer_cik": latest.issuer_cik if latest else None,
        "insider_name": latest.insider_name if latest else None,
        "role_label": latest.role_label if latest else None,
        "filing_subtotal": latest.filing_subtotal if latest else None,
        "cluster_adj": latest.cluster_adj if latest else None,
        "period": latest.period if latest else None,
        "issuer_name": latest.issuer_name if latest else None,
        "matched_accessions": [score.accession for score in scores[:20]],
        "filings": [form4_filing_score_to_dict(score) for score in scores],
    }
