#!/usr/bin/env python3
"""
Probe ClinicalTrials.gov API v2 (no API key required).

Docs: https://clinicaltrials.gov/data-api
Search syntax: https://clinicaltrials.gov/data-api/about-api/search-areas

There is no "significance" field on ClinicalTrials.gov. The default preset uses a
**practical proxy**: Phase 3 + interventional + **industry** lead sponsor + **in-progress**
status (recruiting / active / not yet recruiting / enrolling by invitation). That
matches most users' "pivotal-style trial with a company behind it" without claiming
medical importance.

Usage:
  python test_clinicaltrials.py
  python test_clinicaltrials.py --preset phase3-in-progress --page-size 20
  python test_clinicaltrials.py --preset phase2-recruiting --page-size 5
  python test_clinicaltrials.py --cond "breast cancer" --page-size 3
  python test_clinicaltrials.py --query 'AREA[Phase]PHASE3 AND AREA[OverallStatus]RECRUITING'
  python test_clinicaltrials.py --schema
  python test_clinicaltrials.py --dump-first
  python test_clinicaltrials.py --preset phase2-recruiting --pages 2 --csv out.csv
  python test_clinicaltrials.py --csv out.csv --with-prices --event-date auto
  python test_clinicaltrials.py --csv out.csv --quiet   # summary + ASCII table only

With --csv, an ASCII table is printed after the file is written (disable with --no-table).
With --with-prices, hospital/university-style and some cooperative sponsors are skipped;
non-map matches require Yahoo name/token overlap (same idea as test_openfda_prices).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from calendar import monthrange
from datetime import date
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.services.market.yahoo_event_window import default_min_event_year

# query.term uses the search expression language (AREA[Field]VALUE, AND, OR, …)
_IN_PROGRESS_STATUSES = (
    "AREA[OverallStatus]RECRUITING OR AREA[OverallStatus]ACTIVE_NOT_RECRUITING "
    "OR AREA[OverallStatus]ENROLLING_BY_INVITATION OR AREA[OverallStatus]NOT_YET_RECRUITING"
)

PRESETS: dict[str, str] = {
    # Default intent: "significant" ≈ Phase 3 + company sponsor + actively ongoing
    "phase3-industry-in-progress": (
        "AREA[Phase]PHASE3 AND AREA[StudyType]INTERVENTIONAL AND "
        "AREA[LeadSponsorClass]INDUSTRY AND "
        f"({_IN_PROGRESS_STATUSES})"
    ),
    # Same phase/status as above, but any sponsor class (includes academic sites)
    "phase3-in-progress": (
        f"AREA[Phase]PHASE3 AND AREA[StudyType]INTERVENTIONAL AND ({_IN_PROGRESS_STATUSES})"
    ),
    "phase2-recruiting": "AREA[Phase]PHASE2 AND AREA[OverallStatus]RECRUITING",
    "phase3-recruiting": "AREA[Phase]PHASE3 AND AREA[OverallStatus]RECRUITING",
    "phase2-3-in-progress": (
        "(AREA[Phase]PHASE2 OR AREA[Phase]PHASE3) AND "
        f"({_IN_PROGRESS_STATUSES})"
    ),
    "diabetes-term": "diabetes",
}

EVENT_DATE_STRATEGIES = (
    "auto",
    "results_post",
    "primary_completion",
    "completion",
    "study_first_post",
)

CSV_BASE_FIELDS = [
    "nct_id",
    "brief_title",
    "overall_status",
    "phases",
    "lead_sponsor",
    "has_results",
    "start_date",
    "primary_completion_date",
    "primary_completion_type",
    "completion_date",
    "completion_type",
    "study_first_submit_date",
    "study_first_post_date",
    "results_first_post_date",
    "last_update_post_date",
    "status_verified_date",
    "event_date_strategy",
    "event_date_iso",
    "event_date_source",
    "event_date_raw",
]

CSV_PRICE_FIELDS = [
    "symbol",
    "match_type",
    "pct_6m_to_pre",
    "pct_pre_to_post",
    "pct_pre_to_5td",
    "price_error",
]

# Readable console table (subset); use --table-full for all CSV columns.
TABLE_PREVIEW_COLS: list[str] = [
    "nct_id",
    "overall_status",
    "phases",
    "lead_sponsor",
    "brief_title",
    "has_results",
    "event_date_iso",
    "event_date_source",
]

TABLE_PREVIEW_PRICE_COLS: list[str] = [
    "symbol",
    "match_type",
    "pct_6m_to_pre",
    "pct_pre_to_post",
    "pct_pre_to_5td",
    "price_error",
]

_TABLE_COL_LABELS: dict[str, str] = {
    "nct_id": "NCT",
    "overall_status": "Status",
    "phases": "Phases",
    "lead_sponsor": "Sponsor",
    "brief_title": "Title",
    "has_results": "Res?",
    "event_date_iso": "Event",
    "event_date_source": "Evt src",
    "symbol": "Sym",
    "match_type": "Match",
    "pct_6m_to_pre": "6m→pre%",
    "pct_pre_to_post": "pre→1st%",
    "pct_pre_to_5td": "pre→5td%",
    "price_error": "Px err",
}

# Lead sponsors on CT.gov are often sites, not issuers — avoid bogus yfinance tickers.
_ACADEMIC_SPONSOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bUNIVERSITY\b", re.I),
    re.compile(r"\bHOSPITAL\b", re.I),
    re.compile(r"AFFILIATED\s+HOSPITAL", re.I),
    re.compile(r"MEDICAL\s+UNIVERSITY", re.I),
    re.compile(r"\bCANCER\s+CENTER\b", re.I),
    re.compile(r"\bCANCER\s+INSTITUTE\b", re.I),
    re.compile(r"SCHOOL\s+OF\s+MEDICINE", re.I),
    re.compile(r"COLLEGE\s+OF\s+MEDICINE", re.I),
    re.compile(r"\bMAYO\s+CLINIC\b", re.I),
    re.compile(r"LANGONE", re.I),
    re.compile(r"\bNIH\b", re.I),
    re.compile(r"NATIONAL\s+INSTITUTES?\s+OF\s+HEALTH", re.I),
    re.compile(r"\bMEDICAL\s+CENTER\b", re.I),
    re.compile(r"\bRESEARCH\s+CENTER\b", re.I),
    re.compile(r"\bRESEARCH\s+NETWORK\b", re.I),
    re.compile(r"\bJAEB\b", re.I),
    # Dutch academic medical centers, cooperative trial groups, ambiguous AG names
    re.compile(r"\bACADEMISCH\s+MEDISCH\b", re.I),
    re.compile(r"\bGRUPO\s+ARGENTINO\b", re.I),
    re.compile(r"\bIDEA\s+AG\b", re.I),
)


def lead_sponsor_unlikely_listed_equity(name: str) -> bool:
    if not (name and name.strip()):
        return False
    return any(p.search(name) for p in _ACADEMIC_SPONSOR_PATTERNS)


def lead_sponsor_match_candidates(sponsor: str) -> list[str]:
    """
    Variants to try with match_company_to_symbol / Yahoo search (comma entities, mergers,
    trailing legal suffixes). Merger-derived issuer names are tried *before* the raw CT.gov
    string so e.g. Viatris is preferred over a partial Pfizer hit on narrative sponsor text.
    """
    if not sponsor or not sponsor.strip():
        return []
    s0 = sponsor.strip()
    priority: list[str] = []
    rest: list[str] = []
    seen: set[str] = set()

    def add_pri(t: str) -> None:
        t = (t or "").strip()
        if len(t) < 2:
            return
        k = t.upper()
        if k in seen:
            return
        seen.add(k)
        priority.append(t)

    def add_rest(t: str) -> None:
        t = (t or "").strip()
        if len(t) < 2:
            return
        k = t.upper()
        if k in seen:
            return
        seen.add(k)
        rest.append(t)

    low = s0.lower()
    u = s0.upper()

    if "viatris" in low or "merged with mylan" in low or "upjohn" in low:
        add_pri("Viatris Inc.")
        add_pri("Pfizer Inc.")
    if "subsidiary of pfizer" in low or ("wyeth" in low and "pfizer" in low):
        add_pri("Pfizer Inc.")
        add_pri("Wyeth LLC")
    if "merck sharp" in low and "dohme" in low:
        add_pri("Merck Sharp & Dohme LLC")
        add_pri("Merck & Co., Inc.")
    if "novartis vaccines" in low:
        add_pri("Novartis AG")

    add_rest(s0)

    if "," in s0:
        add_rest(s0.split(",", 1)[0].strip())

    for suf in (
        ", A SANOFI COMPANY",
        ", A WHOLLY OWNED SUBSIDIARY OF PFIZER",
        " IS NOW A WHOLLY OWNED SUBSIDIARY OF PFIZER",
        ", INC.",
        ", LLC",
        " LLC",
        " INC.",
        " INC",
        " S.A.",
        " SA",
        " LIMITED",
        " LTD.",
        " LTD",
        " PLC",
        " AG",
    ):
        if u.endswith(suf):
            add_rest(s0[: -len(suf)].rstrip(" ,"))

    if u.endswith(" RESEARCH"):
        add_rest(s0[: -len(" RESEARCH")].rstrip())

    return priority + rest


def resolve_symbol_for_ct_sponsor(
    sponsor: str,
    match_fn,
) -> tuple[str | None, str | None]:
    """
    Align with test_openfda_prices: trust exact/normalized advisor map; otherwise require
    token overlap with Yahoo long/short name; try yfinance_search when map heuristics misfire.
    Tries several sponsor string variants (legal suffix / merger boilerplate) before giving up.
    """
    from core.services.market.yahoo_event_window import (
        symbol_name_plausible,
        yfinance_search_us_equity,
    )

    if not sponsor.strip():
        return None, None
    if lead_sponsor_unlikely_listed_equity(sponsor):
        return None, "skip_academic"

    candidates = lead_sponsor_match_candidates(sponsor)

    for cand in candidates:
        if lead_sponsor_unlikely_listed_equity(cand):
            continue
        sym, how = match_fn(cand)
        trusted = how in ("exact", "normalized")
        if sym and (trusted or symbol_name_plausible(sym, cand)):
            if cand != sponsor.strip():
                return sym, f"{how} [{cand[:48]}]"
            return sym, how
        if sym:
            sym2, how2 = yfinance_search_us_equity(cand)
            if sym2 and symbol_name_plausible(sym2, cand):
                return sym2, how2 or "yfinance_search"

    for cand in candidates:
        if lead_sponsor_unlikely_listed_equity(cand):
            continue
        sym2, how2 = yfinance_search_us_equity(cand)
        if sym2 and symbol_name_plausible(sym2, cand):
            return sym2, how2 or "yfinance_search"

    return None, None


def fetch_studies(
    *,
    query_term: str | None = None,
    query_cond: str | None = None,
    page_size: int = 5,
    pages: int = 1,
    timeout: float = 60,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch up to `pages` of results using nextPageToken."""
    all_studies: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    token: str | None = None

    for _ in range(max(1, pages)):
        params: dict[str, str | int] = {"pageSize": min(page_size, 1000)}
        if query_term:
            params["query.term"] = query_term
        if query_cond:
            params["query.cond"] = query_cond
        if token:
            params["pageToken"] = token

        r = requests.get(BASE_URL, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        batch = data.get("studies") or []
        all_studies.extend(batch)
        meta = {
            "nextPageToken": data.get("nextPageToken"),
            "totalCountHint": data.get("totalCount"),
        }
        token = data.get("nextPageToken")
        if not token or not batch:
            break

    return all_studies, meta


def parse_ct_date_str(s: str | None) -> date | None:
    """Parse YYYY-MM-DD, YYYY-MM (last day of month), or YYYY (Dec 31)."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    parts = s.split("-")
    try:
        if len(parts) == 3:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            return date(y, m, d)
        if len(parts) == 2:
            y, m = int(parts[0]), int(parts[1])
            last = monthrange(y, m)[1]
            return date(y, m, last)
        if len(parts) == 1 and len(s) == 4:
            y = int(s)
            return date(y, 12, 31)
    except (ValueError, IndexError, OverflowError):
        return None
    return None


def _struct_date_raw(sm: dict[str, Any], struct_key: str) -> str:
    st = sm.get(struct_key)
    if isinstance(st, dict):
        return (st.get("date") or "").strip()
    return ""


def pick_event_date(
    study: dict[str, Any], strategy: str
) -> tuple[date | None, str, str]:
    """
    Return (parsed_date, source_key, raw_date_string).
    source_key is the statusModule field used (empty if none).
    """
    sm = (study.get("protocolSection") or {}).get("statusModule") or {}
    has_results = bool(study.get("hasResults"))

    def from_struct(struct_key: str) -> tuple[date | None, str]:
        raw = _struct_date_raw(sm, struct_key)
        return parse_ct_date_str(raw), raw

    if strategy == "results_post":
        d, raw = from_struct("resultsFirstPostDateStruct")
        return d, "resultsFirstPostDateStruct", raw
    if strategy == "primary_completion":
        d, raw = from_struct("primaryCompletionDateStruct")
        return d, "primaryCompletionDateStruct", raw
    if strategy == "completion":
        d, raw = from_struct("completionDateStruct")
        return d, "completionDateStruct", raw
    if strategy == "study_first_post":
        d, raw = from_struct("studyFirstPostDateStruct")
        return d, "studyFirstPostDateStruct", raw

    # auto
    if has_results:
        d, raw = from_struct("resultsFirstPostDateStruct")
        if d:
            return d, "resultsFirstPostDateStruct", raw
    d, raw = from_struct("primaryCompletionDateStruct")
    if d:
        return d, "primaryCompletionDateStruct", raw
    d, raw = from_struct("completionDateStruct")
    if d:
        return d, "completionDateStruct", raw
    d, raw = from_struct("studyFirstPostDateStruct")
    if d:
        return d, "studyFirstPostDateStruct", raw
    return None, "", ""


def study_csv_row(study: dict[str, Any], event_strategy: str) -> dict[str, str]:
    p = study.get("protocolSection") or {}
    idm = p.get("identificationModule") or {}
    sm = p.get("statusModule") or {}
    dm = p.get("designModule") or {}
    scm = p.get("sponsorCollaboratorsModule") or {}
    lead = (scm.get("leadSponsor") or {}).get("name") or ""
    phases = dm.get("phases") or []
    if isinstance(phases, list):
        phase_s = ";".join(str(x) for x in phases)
    else:
        phase_s = str(phases)

    def sd(struct_key: str) -> str:
        return _struct_date_raw(sm, struct_key)

    def stype(struct_key: str) -> str:
        st = sm.get(struct_key)
        if isinstance(st, dict):
            return (st.get("type") or "").strip()
        return ""

    ev_d, ev_src, ev_raw = pick_event_date(study, event_strategy)
    title = (idm.get("briefTitle") or "").replace("\r", " ").replace("\n", " ").strip()

    return {
        "nct_id": idm.get("nctId") or "",
        "brief_title": title,
        "overall_status": sm.get("overallStatus") or "",
        "phases": phase_s,
        "lead_sponsor": lead,
        "has_results": "1" if study.get("hasResults") else "0",
        "start_date": sd("startDateStruct"),
        "primary_completion_date": sd("primaryCompletionDateStruct"),
        "primary_completion_type": stype("primaryCompletionDateStruct"),
        "completion_date": sd("completionDateStruct"),
        "completion_type": stype("completionDateStruct"),
        "study_first_submit_date": (sm.get("studyFirstSubmitDate") or "").strip(),
        "study_first_post_date": sd("studyFirstPostDateStruct"),
        "results_first_post_date": sd("resultsFirstPostDateStruct"),
        "last_update_post_date": sd("lastUpdatePostDateStruct"),
        "status_verified_date": (sm.get("statusVerifiedDate") or "").strip(),
        "event_date_strategy": event_strategy,
        "event_date_iso": ev_d.isoformat() if ev_d else "",
        "event_date_source": ev_src,
        "event_date_raw": ev_raw,
    }


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.2f}"


def augment_row_with_prices(row: dict[str, str], *, min_year: int, match_fn) -> dict[str, str]:
    """match_fn: (company_name) -> tuple[str|None, str|None] (symbol, match_type)."""
    from core.services.market.yahoo_event_window import price_stats_for_event

    out = {**row}
    sponsor = row.get("lead_sponsor") or ""
    ev_iso = row.get("event_date_iso") or ""

    def blank_prices(msg: str, match_type: str = "") -> dict[str, str]:
        out.update(
            {
                "symbol": "",
                "match_type": match_type,
                "pct_6m_to_pre": "",
                "pct_pre_to_post": "",
                "pct_pre_to_5td": "",
                "price_error": msg,
            }
        )
        return out

    if not ev_iso:
        return blank_prices("no event date")

    try:
        ev_d = date.fromisoformat(ev_iso)
    except ValueError:
        return blank_prices("bad event_date_iso")

    sym, how = resolve_symbol_for_ct_sponsor(sponsor, match_fn)
    if how == "skip_academic":
        return blank_prices(
            "sponsor not mapped (hospital/university, cooperative trial group, or ambiguous name)",
            "skip_academic",
        )
    if not sym:
        return blank_prices("no plausible listed symbol for sponsor")

    px = price_stats_for_event(sym, ev_d, min_year=min_year)
    err = px.get("error") or ""
    out.update(
        {
            "symbol": sym,
            "match_type": how or "",
            "pct_6m_to_pre": _fmt_pct(px.get("pct_6m_to_pre")),
            "pct_pre_to_post": _fmt_pct(px.get("pct_pre_to_post")),
            "pct_pre_to_5td": _fmt_pct(px.get("pct_pre_to_5td")),
            "price_error": err,
        }
    )
    return out


def setup_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()


def summarize_study(study: dict[str, Any]) -> dict[str, Any]:
    p = study.get("protocolSection") or {}
    idm = p.get("identificationModule") or {}
    sm = p.get("statusModule") or {}
    dm = p.get("designModule") or {}
    scm = p.get("sponsorCollaboratorsModule") or {}
    lead = (scm.get("leadSponsor") or {}).get("name")
    title = idm.get("briefTitle") or ""
    if len(title) > 72:
        title = title[:69] + "..."
    return {
        "nct_id": idm.get("nctId"),
        "status": sm.get("overallStatus"),
        "phases": dm.get("phases") or [],
        "lead_sponsor": lead,
        "title": title,
        "has_results": study.get("hasResults"),
    }


def _norm_cell(val: str | None) -> str:
    if val is None:
        return ""
    return str(val).replace("\r", " ").replace("\n", " ").strip()


def print_results_table(
    rows: list[dict[str, str]],
    columns: list[str],
    *,
    max_cell: int = 42,
) -> None:
    """Print a fixed-width ASCII table to stdout (no extra dependencies)."""
    if not rows:
        print("(no rows to display)")
        return

    labels = [_TABLE_COL_LABELS.get(c, c.replace("_", " ")) for c in columns]

    def clip(s: str, w: int) -> str:
        if len(s) <= w:
            return s
        if w <= 2:
            return s[:w]
        return s[: w - 2] + ".."

    widths: list[int] = []
    for i, col in enumerate(columns):
        lab = labels[i]
        w = min(max_cell, len(lab))
        for row in rows:
            ln = min(len(_norm_cell(row.get(col))), max_cell)
            w = max(w, ln)
        widths.append(min(max(w, 1), max_cell))

    def row_line(texts: list[str]) -> str:
        parts = [t.ljust(widths[j]) for j, t in enumerate(texts)]
        return "| " + " | ".join(parts) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    print(sep)
    print(row_line([clip(labels[i], widths[i]) for i in range(len(columns))]))
    print(sep)
    for row in rows:
        cells = [
            clip(_norm_cell(row.get(col)), widths[i]) for i, col in enumerate(columns)
        ]
        print(row_line(cells))
    print(sep)
    print(f"({len(rows)} row(s) in table)")


def print_schema_sample(study: dict[str, Any]) -> None:
    print("--- top-level keys on one study record ---")
    print(sorted(study.keys()))
    p = study.get("protocolSection")
    if isinstance(p, dict):
        print("\n--- protocolSection sub-modules ---")
        for k in sorted(p.keys()):
            sub = p[k]
            n = len(sub) if isinstance(sub, dict) else "?"
            print(f"  {k}  (fields: {n if isinstance(n, int) else 'n/a'})")
    for section in ("resultsSection", "documentSection", "derivedSection"):
        sec = study.get(section)
        if sec:
            print(f"\n--- {section} present: keys ---")
            print(f"  {sorted(sec.keys()) if isinstance(sec, dict) else type(sec)}")


def write_csv(
    path: Path,
    studies: list[dict[str, Any]],
    *,
    event_strategy: str,
    with_prices: bool,
    min_year: int,
    print_table: bool = True,
    table_full: bool = False,
    table_max_cell: int = 42,
) -> None:
    fieldnames = list(CSV_BASE_FIELDS)
    if with_prices:
        fieldnames.extend(CSV_PRICE_FIELDS)

    match_fn = None
    if with_prices:
        setup_django()
        from dotenv import load_dotenv

        load_dotenv(BASE_DIR / ".env")
        from core.services.advisors.fda import match_company_to_symbol

        match_fn = match_company_to_symbol

    rows: list[dict[str, str]] = []
    for s in studies:
        row = study_csv_row(s, event_strategy)
        if with_prices and match_fn is not None:
            row = augment_row_with_prices(row, min_year=min_year, match_fn=match_fn)
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} row(s) to {path}")
    if print_table and rows:
        if table_full:
            table_cols = fieldnames
        else:
            table_cols = list(TABLE_PREVIEW_COLS)
            if with_prices:
                table_cols.extend(TABLE_PREVIEW_PRICE_COLS)
        print()
        print_results_table(rows, table_cols, max_cell=table_max_cell)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe ClinicalTrials.gov API v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Flag order: --query must be followed by the search expression. If you write "
            "`--query --csv FILE`, FILE is parsed as the query. Prefer `--csv FILE --query '…'` "
            "or `--query='…'`."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="phase3-industry-in-progress",
        help=(
            "Canned query.term preset (default: phase3-industry-in-progress = Phase 3 "
            "interventional, industry lead sponsor, in-progress statuses)"
        ),
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Raw query.term (overrides --preset when non-empty)",
    )
    parser.add_argument(
        "--cond",
        type=str,
        default="",
        metavar="TEXT",
        help="query.cond (condition keyword search, separate from query.term)",
    )
    parser.add_argument("--page-size", type=int, default=5, help="Studies per page (default 5)")
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Max pages to fetch via nextPageToken (default 1)",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Print module keys from the first returned study",
    )
    parser.add_argument(
        "--dump-first",
        action="store_true",
        help="Pretty-print first study JSON (truncated)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write flattened studies + key dates to CSV",
    )
    parser.add_argument(
        "--with-prices",
        action="store_true",
        help="Add ticker + yfinance window columns (requires --csv; Django + .env)",
    )
    parser.add_argument(
        "--event-date",
        choices=EVENT_DATE_STRATEGIES,
        default="auto",
        help="Which date anchors event_date_iso / price window (default: auto)",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        metavar="YEAR",
        help="Skip yfinance when event date is before this year (default: current year − 30)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Skip per-study console lines (still prints summary / CSV path / table)",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="With --csv, write file only (no ASCII table on stdout)",
    )
    parser.add_argument(
        "--table-full",
        action="store_true",
        help="With --csv, print all CSV columns in the table (very wide)",
    )
    parser.add_argument(
        "--table-width",
        type=int,
        default=42,
        metavar="N",
        help="Max characters per cell in the printed table (default: 42)",
    )
    args = parser.parse_args()

    if args.with_prices and not args.csv:
        parser.error("--with-prices requires --csv PATH")

    min_year = args.min_year if args.min_year is not None else default_min_event_year()

    query_term: str | None = None
    query_cond: str | None = None

    if args.query.strip():
        query_term = args.query.strip()
    elif args.cond.strip():
        query_cond = args.cond.strip()
    else:
        query_term = PRESETS[args.preset]

    print("GET", BASE_URL)
    print(
        "params:",
        {k: v for k, v in {"query.term": query_term, "query.cond": query_cond}.items() if v},
        "| pageSize:",
        args.page_size,
        "| pages:",
        args.pages,
    )
    print("(no API key required)\n")

    try:
        studies, meta = fetch_studies(
            query_term=query_term,
            query_cond=query_cond,
            page_size=args.page_size,
            pages=args.pages,
        )
    except requests.RequestException as e:
        print("Request failed:", e, file=sys.stderr)
        sys.exit(1)

    print(f"Retrieved {len(studies)} study record(s)")
    if meta.get("nextPageToken"):
        print("nextPageToken: present (more pages available)")
    else:
        print("nextPageToken: none (last page or empty)")
    print()

    if not args.quiet:
        for i, s in enumerate(studies):
            row = summarize_study(s)
            print(
                f"{i + 1:3}. {row['nct_id']} | {row['status']} | phases={row['phases']} | "
                f"results={row['has_results']}"
            )
            print(f"     sponsor: {row['lead_sponsor']}")
            print(f"     {row['title']}")
            print()

    if args.schema and studies:
        print_schema_sample(studies[0])

    if args.dump_first and studies:
        blob = json.dumps(studies[0], indent=2)
        print("\n--- first study JSON (up to 12000 chars) ---\n")
        print(blob[:12000])
        if len(blob) > 12000:
            print("\n... truncated ...")

    if args.csv:
        write_csv(
            args.csv,
            studies,
            event_strategy=args.event_date,
            with_prices=args.with_prices,
            min_year=min_year,
            print_table=not args.no_table,
            table_full=args.table_full,
            table_max_cell=max(8, args.table_width),
        )


if __name__ == "__main__":
    main()
