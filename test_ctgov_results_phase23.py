#!/usr/bin/env python3
"""
Minimal ClinicalTrials.gov probe for PHASE2/PHASE3 industry interventional trials
with posted results on/after a given date.

Defaults:
- since date: yesterday (UTC)
- page size: 20
- pages: 1

Usage:
  python test_ctgov_results_phase23.py
  python test_ctgov_results_phase23.py --since 2026-04-20
  python test_ctgov_results_phase23.py --page-size 20 --pages 2
  python test_ctgov_results_phase23.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import UTC, datetime, timedelta
from typing import Any

import requests
import yfinance as yf

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
BASE_DIR = Path(__file__).resolve().parent


def default_since_utc_yesterday() -> str:
    return (datetime.now(UTC).date() - timedelta(days=1)).isoformat()


def build_query_term(since_iso: str) -> str:
    return (
        "((AREA[Phase]PHASE2 OR AREA[Phase]PHASE3) "
        "AND AREA[StudyType]INTERVENTIONAL "
        "AND AREA[LeadSponsorClass]INDUSTRY "
        "AND AREA[HasResults]true "
        f"AND AREA[ResultsFirstPostDate]RANGE[{since_iso},MAX])"
    )


def fetch_studies(*, query_term: str, page_size: int, pages: int, timeout: float = 60.0) -> tuple[list[dict[str, Any]], bool]:
    studies: list[dict[str, Any]] = []
    token: str | None = None
    has_more = False

    for _ in range(max(1, pages)):
        params: dict[str, str | int] = {"query.term": query_term, "pageSize": min(max(1, page_size), 1000)}
        if token:
            params["pageToken"] = token

        resp = requests.get(BASE_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()

        batch = payload.get("studies") or []
        studies.extend(batch)

        token = payload.get("nextPageToken")
        has_more = bool(token)
        if not token or not batch:
            break

    return studies, has_more


def summarize(study: dict[str, Any]) -> dict[str, Any]:
    p = study.get("protocolSection") or {}
    idm = p.get("identificationModule") or {}
    sm = p.get("statusModule") or {}
    dm = p.get("designModule") or {}
    scm = p.get("sponsorCollaboratorsModule") or {}

    return {
        "nct_id": idm.get("nctId"),
        "results_first_post_date": (sm.get("resultsFirstPostDateStruct") or {}).get("date"),
        "overall_status": sm.get("overallStatus"),
        "phases": dm.get("phases") or [],
        "sponsor": (scm.get("leadSponsor") or {}).get("name"),
        "title": idm.get("briefTitle") or "",
    }


def setup_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()


def resolve_ticker_for_sponsor(sponsor: str | None) -> tuple[str | None, str | None]:
    if not sponsor or not sponsor.strip():
        return None, None

    from core.services.advisors.fda import match_company_to_symbol

    symbol, match_type = match_company_to_symbol(sponsor.strip())
    return symbol, match_type


def enrich_price_move(row: dict[str, Any]) -> None:
    ticker = row.get("ticker")
    event_date = row.get("results_first_post_date")
    if not ticker or not event_date:
        row["event_open_price"] = None
        row["latest_price"] = None
        row["pct_from_event_open_to_latest"] = None
        row["price_error"] = "missing ticker or event date"
        return

    try:
        dt = datetime.fromisoformat(str(event_date)).date()
    except ValueError:
        row["event_open_price"] = None
        row["latest_price"] = None
        row["pct_from_event_open_to_latest"] = None
        row["price_error"] = "invalid event date"
        return

    try:
        tk = yf.Ticker(str(ticker))
        hist = tk.history(start=dt.isoformat(), end=(dt + timedelta(days=1)).isoformat(), interval="1d", raise_errors=False)
        if hist is None or hist.empty:
            row["event_open_price"] = None
            row["latest_price"] = None
            row["pct_from_event_open_to_latest"] = None
            row["price_error"] = "no event-day bar"
            return

        event_open = float(hist["Open"].iloc[0])
        latest_price = None
        fast = getattr(tk, "fast_info", None)
        if fast is not None:
            latest_price = getattr(fast, "last_price", None)
        if latest_price is None:
            info = tk.info or {}
            latest_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if latest_price is None:
            latest_hist = tk.history(period="1d", interval="1d", raise_errors=False)
            if latest_hist is not None and not latest_hist.empty:
                latest_price = float(latest_hist["Close"].iloc[-1])

        if latest_price is None:
            row["event_open_price"] = event_open
            row["latest_price"] = None
            row["pct_from_event_open_to_latest"] = None
            row["price_error"] = "no latest price"
            return

        latest_price = float(latest_price)
        pct = ((latest_price - event_open) / event_open) * 100.0 if event_open > 0 else None
        row["event_open_price"] = round(event_open, 4)
        row["latest_price"] = round(latest_price, 4)
        row["pct_from_event_open_to_latest"] = round(pct, 2) if pct is not None else None
        row["price_error"] = None
    except Exception as exc:
        row["event_open_price"] = None
        row["latest_price"] = None
        row["pct_from_event_open_to_latest"] = None
        row["price_error"] = str(exc)


def enrich_window_deltas(row: dict[str, Any]) -> None:
    """Compute -30d/-10d/+10d/+30d(or now) percentages vs CT-date open."""
    ticker = row.get("ticker")
    event_date = row.get("results_first_post_date")
    row["pct_m30_vs_ct"] = None
    row["pct_m10_vs_ct"] = None
    row["pct_p10_vs_ct"] = None
    row["pct_p30_or_now_vs_ct"] = None
    row["window_error"] = None
    if not ticker or not event_date:
        row["window_error"] = "missing ticker or event date"
        return

    try:
        event_dt = datetime.fromisoformat(str(event_date)).date()
    except ValueError:
        row["window_error"] = "invalid event date"
        return

    try:
        tk = yf.Ticker(str(ticker))
        hist = tk.history(
            start=(event_dt - timedelta(days=120)).isoformat(),
            end=(datetime.now(UTC).date() + timedelta(days=1)).isoformat(),
            interval="1d",
            raise_errors=False,
        )
        if hist is None or hist.empty:
            row["window_error"] = "no history"
            return

        # Find first trading bar on/after CT date.
        event_pos = None
        for i, idx in enumerate(hist.index):
            d = idx.date()
            if d >= event_dt:
                event_pos = i
                break
        if event_pos is None:
            row["window_error"] = "no event bar"
            return

        ct_open = float(hist["Open"].iloc[event_pos])
        if ct_open <= 0:
            row["window_error"] = "invalid ct open"
            return

        def pct_at(offset: int) -> float | None:
            pos = event_pos + offset
            if pos < 0 or pos >= len(hist):
                return None
            px = float(hist["Open"].iloc[pos])
            return ((px - ct_open) / ct_open) * 100.0

        p_m30 = pct_at(-30)
        p_m10 = pct_at(-10)
        p_p10 = pct_at(10)
        p_p30 = pct_at(30)
        if p_p30 is None:
            latest_open = float(hist["Open"].iloc[-1])
            p_p30 = ((latest_open - ct_open) / ct_open) * 100.0

        row["pct_m30_vs_ct"] = round(p_m30, 2) if p_m30 is not None else None
        row["pct_m10_vs_ct"] = round(p_m10, 2) if p_m10 is not None else None
        row["pct_p10_vs_ct"] = round(p_p10, 2) if p_p10 is not None else None
        row["pct_p30_or_now_vs_ct"] = round(p_p30, 2) if p_p30 is not None else None
    except Exception as exc:
        row["window_error"] = str(exc)


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):+.2f}%"


def print_aligned_table(rows: list[dict[str, Any]]) -> None:
    headers = ["TICKER", "SPONSOR", "CT_DATE", "-30d", "-10d", "+10d", "+30d(now)"]
    table_data: list[list[str]] = []
    for r in rows:
        table_data.append(
            [
                str(r.get("ticker") or "n/a"),
                str(r.get("sponsor") or "n/a"),
                str(r.get("results_first_post_date") or "n/a"),
                _fmt_pct(r.get("pct_m30_vs_ct")),
                _fmt_pct(r.get("pct_m10_vs_ct")),
                _fmt_pct(r.get("pct_p10_vs_ct")),
                _fmt_pct(r.get("pct_p30_or_now_vs_ct")),
            ]
        )

    widths: list[int] = []
    for idx, header in enumerate(headers):
        max_cell = max((len(row[idx]) for row in table_data), default=0)
        widths.append(max(len(header), max_cell))

    def _line(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(_line(headers))
    print("-+-".join("-" * w for w in widths))
    for row in table_data:
        print(_line(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query CT.gov for PHASE2/PHASE3 industry interventional trials with posted results."
    )
    parser.add_argument("--since", default=default_since_utc_yesterday(), help="Cutoff date YYYY-MM-DD (default: yesterday UTC)")
    parser.add_argument("--page-size", type=int, default=20, help="Studies per page (default: 20)")
    parser.add_argument("--pages", type=int, default=1, help="Max pages to fetch (default: 1)")
    parser.add_argument("--ticker-required", action="store_true", help="Only print rows with resolved tickers")
    parser.add_argument("--json", action="store_true", help="Print normalized rows as JSON")
    args = parser.parse_args()

    query_term = build_query_term(args.since)
    print("GET", BASE_URL)
    print("query.term:", query_term)
    print("pageSize:", args.page_size, "| pages:", args.pages)
    print()

    try:
        studies, has_more = fetch_studies(query_term=query_term, page_size=args.page_size, pages=args.pages)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    rows = [summarize(s) for s in studies]
    if rows:
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        setup_django()
        for row in rows:
            ticker, ticker_match_type = resolve_ticker_for_sponsor(row.get("sponsor"))
            row["ticker"] = ticker
            row["ticker_match_type"] = ticker_match_type
            row["ticker_resolved"] = bool(ticker)
    else:
        for row in rows:
            row["ticker"] = None
            row["ticker_match_type"] = None
            row["ticker_resolved"] = False

    completed_rows = [r for r in rows if str(r.get("overall_status") or "").upper() == "COMPLETED"]
    resolved_rows = [r for r in rows if bool(r.get("ticker_resolved"))]
    completed_and_resolved_rows = [r for r in completed_rows if bool(r.get("ticker_resolved"))]
    for row in completed_and_resolved_rows:
        enrich_price_move(row)
        enrich_window_deltas(row)

    if args.ticker_required:
        rows = [r for r in rows if bool(r.get("ticker_resolved"))]

    print(f"Retrieved {len(rows)} study record(s)")
    print("nextPageToken:", "present" if has_more else "none")
    print(
        "summary:"
        f" total_raw={len(studies)}"
        f" completed={len(completed_rows)}"
        f" ticker_resolved={len(resolved_rows)}"
        f" completed_and_ticker_resolved={len(completed_and_resolved_rows)}"
    )
    print()

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    table_rows = [r for r in rows if bool(r.get("ticker_resolved")) and str(r.get("overall_status") or "").upper() == "COMPLETED"]
    if table_rows:
        print_aligned_table(table_rows)
        print()

    for i, row in enumerate(rows, start=1):
        print(
            f"{i:3}. {row['nct_id']} | {row['results_first_post_date']} | {row['overall_status']} | phases={row['phases']}"
        )
        print(f"     sponsor: {row['sponsor']}")
        print(f"     ticker: {row.get('ticker')} ({row.get('ticker_match_type')})")
        if row.get("ticker_resolved") and str(row.get("overall_status") or "").upper() == "COMPLETED":
            print(
                "     price:"
                f" event_open={row.get('event_open_price')}"
                f" latest={row.get('latest_price')}"
                f" pct={row.get('pct_from_event_open_to_latest')}%"
                f" err={row.get('price_error')}"
            )
            print(
                "     window:"
                f" -30d={_fmt_pct(row.get('pct_m30_vs_ct'))}"
                f" -10d={_fmt_pct(row.get('pct_m10_vs_ct'))}"
                f" +10d={_fmt_pct(row.get('pct_p10_vs_ct'))}"
                f" +30d(now)={_fmt_pct(row.get('pct_p30_or_now_vs_ct'))}"
                f" err={row.get('window_error')}"
            )
        title = row["title"]
        if len(title) > 100:
            title = title[:97] + "..."
        print(f"     {title}")
        print()


if __name__ == "__main__":
    main()

