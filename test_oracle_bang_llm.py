"""
Oracle LLM triage test — same prompt/path as production Oracle discover().

Usage:
    python test_oracle_bang_llm.py --preset jul2
    python test_oracle_bang_llm.py --preset jul3
    python test_oracle_bang_llm.py --preset man
    python test_oracle_bang_llm.py MAN TRV --as-of 2026-07-02 --earn 2026-07-16 --days 14
    python test_oracle_bang_llm.py MAN --row "MAN|2026-07-16|14|+27.7|bang|53|9.95|C|none|buy(9)"
    python test_oracle_bang_llm.py --preset jul2 --no-search
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Preset pipe rows: sym|earn|d|b20|shape|late%|jmp%|opp|f4|cons
PRESETS: Dict[str, Tuple[str, List[str]]] = {
    "man": (
        "2026-07-02",
        ["MAN|2026-07-16|14|+27.7|bang|53|9.95|C|none|buy(9)"],
    ),
    "jul2": (
        "2026-07-02",
        [
            "MAN|2026-07-16|14|+27.7|bang|53|9.95|C|none|buy(9)",
            "CFG|2026-07-16|14|+15.4|creep|41|2.1|C|none|hold(18)",
            "BSVN|2026-07-16|14|+15.6|creep|38|1.8|C|none|hold(4)",
        ],
    ),
    "jul3": (
        "2026-07-03",
        [
            "TRV|2026-07-17|14|+18.6|creep|41|3.4|C|none|hold(23)",
            "IBN|2026-07-17|14|+15.7|creep|—|—|B|none|buy",
            "HDB|2026-07-17|14|+9.4|creep|—|—|B|none|buy",
        ],
    ),
}


def _django_setup() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()


def _candidate_from_live_row(
    symbol: str,
    *,
    as_of: date,
    earnings_date: str,
    days_to_print: int,
) -> Dict[str, Any]:
    from core.services.advisors.oracle import (
        PRE_MOVE_LOOKBACK_DAYS,
        _consensus_gate,
        _form4_gate,
        _so_gate,
        compute_pre_move_build,
    )

    sym = symbol.strip().upper()
    cache: Dict = {}
    build = compute_pre_move_build(sym, PRE_MOVE_LOOKBACK_DAYS, cache, as_of=as_of)
    if build.get("error"):
        raise ValueError(f"{sym}: {build.get('error')}")

    _, so = _so_gate(sym)
    _, cons = _consensus_gate(sym)
    _, _, _, f4 = _form4_gate(sym)

    return {
        **build,
        **so,
        **cons,
        **f4,
        "earnings_date": earnings_date,
        "lookahead_days": days_to_print,
    }


def _pipe_row_to_candidate(row: str) -> Dict[str, Any]:
    parts = row.split("|")
    if len(parts) != 10:
        raise ValueError(f"expected 10 pipe fields, got {len(parts)}: {row!r}")
    sym, earn, days, b20, shape, late, jmp, opp, f4, cons = parts
    pre_move = None if b20 == "—" else float(b20.replace("+", ""))
    late_share = None if late == "—" else float(late)
    max_jump = None if jmp == "—" else float(jmp)
    cons_rec, _, cons_tail = cons.partition("(")
    analyst_count = None
    if cons_tail.endswith(")"):
        try:
            analyst_count = int(cons_tail[:-1])
        except ValueError:
            pass
    return {
        "symbol": sym,
        "earnings_date": earn,
        "lookahead_days": int(days),
        "pre_move_pct": pre_move,
        "shape": shape,
        "late_share_pct": late_share,
        "max_1d_jump_pct": max_jump,
        "opp_grade": opp,
        "form4_entries": 0 if f4 == "none" else 1,
        "form4_kind": f4.split("(")[0] if f4 != "none" else None,
        "form4_total": None,
        "consensus_rec": cons_rec if cons_rec != "none" else None,
        "consensus_analyst_count": analyst_count,
    }


def run_llm(
    candidates: List[Dict[str, Any]],
    *,
    as_of: str,
    use_search: bool,
    timeout: float,
) -> Tuple[Optional[str], Any]:
    _django_setup()

    if use_search:
        from core.services.advisors.oracle import run_llm_triage_batch

        return run_llm_triage_batch(candidates, as_of=as_of, timeout=timeout)

    from core.services.llm.router import ask_llm
    from core.services.advisors.oracle import (
        build_llm_triage_prompt,
        build_pipe_row_from_candidate,
        _normalize_llm_triage_results,
    )

    pipe_rows = [build_pipe_row_from_candidate(c) for c in candidates]
    prompt = build_llm_triage_prompt(as_of, pipe_rows)
    model, results, _, _ = ask_llm(
        prompt=prompt,
        advisor_name="oracle_bang_triage_test",
        timeout=timeout,
        use_search=False,
    )
    return model, _normalize_llm_triage_results(results)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Test Oracle LLM triage (production prompt).")
    parser.add_argument("symbols", nargs="*", help="Live tickers (--as-of, --earn, --days).")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()))
    parser.add_argument("--row", action="append", default=[], dest="rows")
    parser.add_argument("--as-of", help="Scan date YYYY-MM-DD")
    parser.add_argument("--earn", help="Earnings date YYYY-MM-DD")
    parser.add_argument("--days", type=int, help="Days to earnings")
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--print-prompt", action="store_true")
    args = parser.parse_args(argv)

    _django_setup()
    from core.services.advisors.oracle import (
        build_llm_triage_prompt,
        build_pipe_row_from_candidate,
        _llm_triage_is_veto,
    )

    as_of: str
    candidates: List[Dict[str, Any]]

    if args.rows:
        as_of = args.as_of or date.today().isoformat()
        candidates = [_pipe_row_to_candidate(r) for r in args.rows]
    elif args.preset:
        as_of, rows = PRESETS[args.preset]
        candidates = [_pipe_row_to_candidate(r) for r in rows]
    elif args.symbols:
        if not args.as_of or not args.earn or args.days is None:
            parser.error("Live symbols require --as-of, --earn, and --days.")
        as_of = args.as_of
        as_of_dt = date.fromisoformat(args.as_of)
        candidates = [
            _candidate_from_live_row(
                sym,
                as_of=as_of_dt,
                earnings_date=args.earn,
                days_to_print=args.days,
            )
            for sym in args.symbols
        ]
    else:
        parser.error("Provide --preset, --row, or symbols with --as-of/--earn/--days.")

    pipe_rows = [build_pipe_row_from_candidate(c) for c in candidates]
    print("=== Pipe rows ===")
    for row in pipe_rows:
        print(row)
    print()

    if args.print_prompt:
        print(build_llm_triage_prompt(as_of, pipe_rows))
        return 0

    print("=== Calling ask_llm (Gemini → DeepSeek) ===")
    model, results = run_llm(
        candidates,
        as_of=as_of,
        use_search=not args.no_search,
        timeout=args.timeout,
    )
    if not results:
        print("No usable JSON response.", file=sys.stderr)
        return 1

    print(f"Model: {model or 'unknown'}\n")
    print(json.dumps(results, indent=2))

    items = results if isinstance(results, list) else [results]
    print("\n=== Summary ===")
    for item in items:
        if not isinstance(item, dict):
            continue
        sym = item.get("sym") or item.get("symbol") or "?"
        verdict = item.get("verdict", "?")
        skip = _llm_triage_is_veto(item)
        print(
            f"{sym}: {verdict} | {item.get('driver_cat', '?')} | "
            f"thesis={item.get('thesis_fit', '?')} | conf={item.get('conf', '?')}"
            f"{' → SKIP DISCOVER' if skip else ''}"
        )
        reason = item.get("reason")
        if reason:
            print(f"  {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
