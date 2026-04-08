#!/usr/bin/env python3
"""
Fetch studies from ClinicalTrials.gov API v2 using simple filters.

The registry calls recruitment/conduct state **overallStatus** (not "stage").
Phases are **Phase** (e.g. PHASE3). This script turns comma-separated CLI lists into
query.term expressions like:
  (AREA[Phase]PHASE1 OR AREA[Phase]PHASE2) AND (AREA[OverallStatus]RECRUITING OR ...)

Docs: https://clinicaltrials.gov/data-api/about-api/search-areas

Examples:
  python collect_clinicaltrials.py --phases phase1,phase2 --overall-status recruiting,not_yet_recruiting
  python collect_clinicaltrials.py --phases 3 --overall-status active_not_recruiting --pages 3 --csv out.csv
  python collect_clinicaltrials.py --phases phase4 --overall-status completed --csv done.csv
  python collect_clinicaltrials.py --phases 3 --overall-status completed \\
      --completion-after 2020-01-01 --last-update-after 2024-06-01
  # CSV and console include primary_completion_* and completion_* from statusModule (ACTUAL vs ESTIMATED).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date as date_type
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# API values for AREA[Phase]
_VALID_PHASES = frozenset({"PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"})

# API values for AREA[OverallStatus]
_VALID_OVERALL_STATUS = frozenset(
    {
        "RECRUITING",
        "NOT_YET_RECRUITING",
        "ACTIVE_NOT_RECRUITING",
        "ENROLLING_BY_INVITATION",
        "COMPLETED",
        "SUSPENDED",
        "TERMINATED",
        "WITHDRAWN",
        "UNKNOWN",
    }
)

_PHASE_ALIASES: dict[str, str] = {
    "1": "PHASE1",
    "2": "PHASE2",
    "3": "PHASE3",
    "4": "PHASE4",
    "PHASE1": "PHASE1",
    "PHASE2": "PHASE2",
    "PHASE3": "PHASE3",
    "PHASE4": "PHASE4",
    "PHASEI": "PHASE1",
    "PHASEII": "PHASE2",
    "PHASEIII": "PHASE3",
    "PHASEIV": "PHASE4",
    "NA": "NA",
    "N/A": "NA",
}

_STATUS_ALIASES: dict[str, str] = {
    "RECRUITING": "RECRUITING",
    "NOTYETRECRUITING": "NOT_YET_RECRUITING",
    "NOT_YET_RECRUITING": "NOT_YET_RECRUITING",
    "ACTIVENOTRECRUITING": "ACTIVE_NOT_RECRUITING",
    "ACTIVE_NOT_RECRUITING": "ACTIVE_NOT_RECRUITING",
    "ENROLLINGBYINVITATION": "ENROLLING_BY_INVITATION",
    "ENROLLING_BY_INVITATION": "ENROLLING_BY_INVITATION",
    "COMPLETED": "COMPLETED",
    "SUSPENDED": "SUSPENDED",
    "TERMINATED": "TERMINATED",
    "WITHDRAWN": "WITHDRAWN",
    "UNKNOWN": "UNKNOWN",
}


def _norm_token(s: str) -> str:
    return "".join(c for c in s.strip().upper() if c not in " -")  # phase-2 -> PHASE2 path below


def parse_phases(arg: str) -> list[str]:
    out: list[str] = []
    for raw in arg.split(","):
        t = raw.strip()
        if not t:
            continue
        u = t.upper().replace(" ", "")
        if u.startswith("PHASE") and u not in _PHASE_ALIASES:
            # phase1, phase2, ...
            key = u
        else:
            key = _norm_token(t)
            if key.startswith("PHASE") and len(key) > 5 and key[5:].isdigit():
                key = f"PHASE{key[5:]}"
            elif key.isdigit() and len(key) == 1:
                key = f"PHASE{key}"
        mapped = _PHASE_ALIASES.get(key) or _PHASE_ALIASES.get(u) or (
            u if u in _VALID_PHASES else None
        )
        if mapped is None and u in _VALID_PHASES:
            mapped = u
        if mapped is None or mapped not in _VALID_PHASES:
            raise argparse.ArgumentTypeError(
                f"unknown phase {raw!r}; use phase1..phase4, na, or PHASE1..PHASE4"
            )
        if mapped not in out:
            out.append(mapped)
    if not out:
        raise argparse.ArgumentTypeError("--phases must list at least one phase")
    return out


def parse_overall_statuses(arg: str) -> list[str]:
    out: list[str] = []
    for raw in arg.split(","):
        t = raw.strip()
        if not t:
            continue
        key = _norm_token(t)
        mapped = _STATUS_ALIASES.get(key) or _STATUS_ALIASES.get(t.upper().replace(" ", "_"))
        if mapped is None and t.upper() in _VALID_OVERALL_STATUS:
            mapped = t.upper()
        if mapped is None or mapped not in _VALID_OVERALL_STATUS:
            raise argparse.ArgumentTypeError(
                f"unknown overall status {raw!r}; examples: RECRUITING, ACTIVE_NOT_RECRUITING"
            )
        if mapped not in out:
            out.append(mapped)
    return out


def parse_yyyy_mm_dd(value: str) -> str:
    """Validate calendar date for API RANGE clauses; returns the same string."""
    v = (value or "").strip()
    try:
        date_type.fromisoformat(v)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected YYYY-MM-DD (ISO date), got {value!r}"
        ) from None
    return v


def build_query_term(
    phases: list[str],
    overall_statuses: list[str] | None,
    *,
    interventional_only: bool,
    industry_sponsor_only: bool,
    completion_after: str | None = None,
    last_update_after: str | None = None,
) -> str:
    if len(phases) == 1:
        phase_q = f"AREA[Phase]{phases[0]}"
    else:
        phase_q = "(" + " OR ".join(f"AREA[Phase]{p}" for p in phases) + ")"

    parts: list[str] = [phase_q]

    if overall_statuses:
        if len(overall_statuses) == 1:
            parts.append(f"AREA[OverallStatus]{overall_statuses[0]}")
        else:
            inner = " OR ".join(f"AREA[OverallStatus]{s}" for s in overall_statuses)
            parts.append(f"({inner})")

    if interventional_only:
        parts.append("AREA[StudyType]INTERVENTIONAL")
    if industry_sponsor_only:
        parts.append("AREA[LeadSponsorClass]INDUSTRY")

    if completion_after:
        parts.append(f"AREA[CompletionDate]RANGE[{completion_after},MAX]")
    if last_update_after:
        parts.append(f"AREA[LastUpdatePostDate]RANGE[{last_update_after},MAX]")

    return " AND ".join(parts)


def fetch_studies(
    *,
    query_term: str,
    page_size: int,
    pages: int,
    timeout: float = 120,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_studies: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    token: str | None = None

    for _ in range(max(1, pages)):
        params: dict[str, str | int] = {"pageSize": min(page_size, 1000)}
        params["query.term"] = query_term
        if token:
            params["pageToken"] = token

        r = requests.get(BASE_URL, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        batch = data.get("studies") or []
        all_studies.extend(batch)
        meta = {
            "nextPageToken": data.get("nextPageToken"),
            "totalCount": data.get("totalCount"),
        }
        token = data.get("nextPageToken")
        if not token or not batch:
            break

    return all_studies, meta


def _status_struct_date(sm: dict[str, Any], key: str) -> str:
    st = sm.get(key)
    if isinstance(st, dict):
        return (st.get("date") or "").strip()
    return ""


def _status_struct_type(sm: dict[str, Any], key: str) -> str:
    st = sm.get(key)
    if isinstance(st, dict):
        return (st.get("type") or "").strip()
    return ""


def study_brief_row(study: dict[str, Any]) -> dict[str, str]:
    p = study.get("protocolSection") or {}
    idm = p.get("identificationModule") or {}
    sm = p.get("statusModule") or {}
    dm = p.get("designModule") or {}
    scm = p.get("sponsorCollaboratorsModule") or {}
    lead = (scm.get("leadSponsor") or {}).get("name") or ""
    phases = dm.get("phases") or []
    ph = ";".join(str(x) for x in phases) if isinstance(phases, list) else str(phases)
    title = (idm.get("briefTitle") or "").replace("\n", " ").strip()
    return {
        "nct_id": idm.get("nctId") or "",
        "overall_status": sm.get("overallStatus") or "",
        "phases": ph,
        "primary_completion_date": _status_struct_date(sm, "primaryCompletionDateStruct"),
        "primary_completion_type": _status_struct_type(sm, "primaryCompletionDateStruct"),
        "completion_date": _status_struct_date(sm, "completionDateStruct"),
        "completion_type": _status_struct_type(sm, "completionDateStruct"),
        "last_update_post_date": _status_struct_date(sm, "lastUpdatePostDateStruct"),
        "lead_sponsor": lead,
        "brief_title": title,
    }


# CSV / console column order
_BRIEF_FIELDNAMES: list[str] = [
    "nct_id",
    "overall_status",
    "phases",
    "primary_completion_date",
    "primary_completion_type",
    "completion_date",
    "completion_type",
    "last_update_post_date",
    "lead_sponsor",
    "brief_title",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect ClinicalTrials.gov studies by phase and overall status (API: overallStatus).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phases",
        type=parse_phases,
        required=True,
        metavar="LIST",
        help="Comma-separated phases: phase1,phase2 or PHASE1,PHASE3 or 1,2",
    )
    parser.add_argument(
        "--overall-status",
        "--status",
        dest="overall_statuses",
        type=parse_overall_statuses,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated overallStatus values (registry term for recruitment/conduct), "
            "e.g. RECRUITING,ACTIVE_NOT_RECRUITING. Omit to not filter by status."
        ),
    )
    parser.add_argument(
        "--no-interventional-only",
        action="store_true",
        help="Do not restrict to AREA[StudyType]INTERVENTIONAL",
    )
    parser.add_argument(
        "--industry-sponsor",
        action="store_true",
        help="Restrict to AREA[LeadSponsorClass]INDUSTRY",
    )
    parser.add_argument(
        "--completion-after",
        type=parse_yyyy_mm_dd,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "AREA[CompletionDate]RANGE[date,MAX] — indexed study completion on/after this date "
            "(ISO YYYY-MM-DD)"
        ),
    )
    parser.add_argument(
        "--last-update-after",
        type=parse_yyyy_mm_dd,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "AREA[LastUpdatePostDate]RANGE[date,MAX] — record last updated/posted on/after this date "
            "(ISO YYYY-MM-DD)"
        ),
    )
    parser.add_argument("--page-size", type=int, default=50, help="Studies per request (max 1000)")
    parser.add_argument("--pages", type=int, default=1, help="Pages to fetch via pageToken")
    parser.add_argument("--csv", type=Path, default=None, metavar="PATH", help="Write brief CSV")
    parser.add_argument(
        "--print-query",
        action="store_true",
        help="Print built query.term and exit (no HTTP)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full study JSON records to stdout (large)",
    )
    args = parser.parse_args()

    query_term = build_query_term(
        args.phases,
        args.overall_statuses,
        interventional_only=not args.no_interventional_only,
        industry_sponsor_only=args.industry_sponsor,
        completion_after=args.completion_after,
        last_update_after=args.last_update_after,
    )

    print("query.term:", query_term)
    if args.print_query:
        return

    try:
        studies, meta = fetch_studies(
            query_term=query_term,
            page_size=args.page_size,
            pages=args.pages,
        )
    except requests.RequestException as e:
        print("Request failed:", e, file=sys.stderr)
        sys.exit(1)

    print(f"Retrieved {len(studies)} study record(s)")
    if meta.get("nextPageToken"):
        print("nextPageToken: present (more pages available)")
    print()

    if args.json:
        json.dump(studies, sys.stdout, indent=2)
        print()
        return

    if args.csv:
        fieldnames = list(_BRIEF_FIELDNAMES)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for s in studies:
                w.writerow(study_brief_row(s))
        print(f"Wrote {len(studies)} row(s) to {args.csv}")
        return

    for i, s in enumerate(studies):
        row = study_brief_row(s)
        comp = row["completion_date"] or row["primary_completion_date"] or "—"
        ctype = (row["completion_type"] or row["primary_completion_type"] or "").strip()
        ct_suffix = f" ({ctype})" if ctype else ""
        print(
            f"{i + 1:3}. {row['nct_id']} | {row['overall_status']} | {row['phases']} | "
            f"completed/end: {comp}{ct_suffix} | {row['lead_sponsor']}"
        )
        print(f"     {row['brief_title'][:100]}{'...' if len(row['brief_title']) > 100 else ''}")
        print()


if __name__ == "__main__":
    main()
