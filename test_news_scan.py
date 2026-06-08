#!/usr/bin/env python3
"""
Standalone headline red-flag scan (no Django).

Two-stage safety net prototype:
  1. Yahoo headlines (recent titles)
  2. Regex keyword screen — ESCALATE on hit (run --llm for ALLOW/BLOCK)
  3. Optional Gemini gatekeeper (--llm) on keyword hits, or all symbols with --llm-all

Usage:
    source ~/Development/scratch/python/tutorial-env/bin/activate
    python test_news_scan.py LLY
    python test_news_scan.py PFE TSLA --days auto --limit 5
    python test_news_scan.py PVH --days 3          # fixed 3-day window
    python test_news_scan.py LLY --llm --trigger "Flux below_ma_up entry"
    python test_news_scan.py LLY --json

Requires GEMINI_KEY in env or .env for --llm.

Benchmark calibration set (see --benchmark):
  PVH  — outlook cut + plunge → expect HIGH regex + LLM risk ~80-95
  GTM  — ZoomInfo (use instead of ZI on Yahoo); downgrade/guidance → ~60-80
  META — product/monetization noise → false-positive test, LLM ~30-50 not 90
  AVGO — earnings miss + AI beat → nuance test, LLM reads context
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from core.services.financial.yahoo import (
    headline_lookback_slot,
    latest_headlines,
)

# Yahoo ticker aliases when primary symbol has stale/empty news feed.
HEADLINE_SYMBOL_ALIASES: Dict[str, str] = {
    "ZI": "GTM",  # ZoomInfo → GTM on Yahoo
}

# Calibration notes for --benchmark (human review targets, not assertions).
BENCHMARK_CASES: Dict[str, str] = {
    "PVH": "HIGH RISK — outlook cut, plunge; regex should hit; LLM risk ~80-95",
    "GTM": "MEDIUM — downgrade/guidance (ZoomInfo); LLM risk ~60-80 (ZI alias)",
    "ZI": "Same as GTM — Yahoo uses GTM for ZoomInfo headlines",
    "META": "FALSE-POSITIVE test — monetization/delay noise; LLM ~30-50 not 90",
    "AVGO": "NUANCE — ‘miss’ vs AI strength; LLM must read full headline context",
}

# ---------------------------------------------------------------------------
# Red-flag regexes — tune here after review (case-insensitive).
# Each entry: (label, pattern)
# ---------------------------------------------------------------------------
RED_FLAG_PATTERNS: List[Tuple[str, str]] = [
    ("bankruptcy", r"\bbankruptcy\b|\bchapter\s+11\b"),
    ("investigation", r"\binvestigation\b|\bsec probe\b|\bsubpoena\b|\bDOJ\b|\blitigation\b"),
    ("fraud", r"\bfraud\b|\baccounting irregularit"),
    ("restatement", r"\brestate(s|ment)\b|\brestate(s|ment) financial"),
    ("delisting", r"\bdelist(ing|ed)?\b|\bnon[- ]compliance notice\b"),
    ("offering", r"\b(public|secondary|registered) offering\b|\bstock offering\b|\bat-the-market\b|\bATM offering\b"),
    ("dilution", r"\bdilut(e|ion|ive)\b"),
    ("clinical_failure", r"\bclinical hold\b|\btrial (fail|halt|suspend|discontinu)\b|\bphase\s+[123].*(fail|miss|halt)\b"),
    ("fda_negative", r"\bcomplete response letter\b|\bcrl\b|\bfda reject"),
    ("leadership_exit", r"\bceo (resign|depart|step|exit|terminate|oust)\b|\bcfo (resign|depart|exit)\b"),
    (
        "outlook_cut",
        r"\bcut(s|ting)?\b.*\boutlook\b|\boutlook cut\b|\bcutting \d{4} revenue outlook\b"
        r"|\breduced (full[- ]year )?outlook\b",
    ),
    (
        "guidance_cut",
        r"\bcut(s|ting)? guidance\b|\blower(s|ed|ing)? (full[- ]year )?guidance\b"
        r"|\bwithdraw(s|ing)? guidance\b|\bguidance cut\b",
    ),
    ("downgrade", r"\bdowngrade(s|d)?\b|\bcut(s|ting)? (to )?(hold|sell|underperform|neutral)\b"),
    ("going_concern", r"\bgoing concern\b"),
    ("default_covenant", r"\b(default|covenant breach|liquidity crunch)\b"),
    ("halt", r"\btrading halt\b|\bsec suspend"),
    ("layoffs", r"\blayoff(s)?\b|\bworkforce reduction\b|\bjob cuts\b"),
    ("lawsuit", r"\bclass action\b|\bshareholder lawsuit\b|\bsecurities lawsuit\b"),
    ("recall", r"\bproduct recall\b|\brecall(s)? (all|product)\b"),
    ("audit", r"\baudit (fail|issue|concern|qualification)\b|\bqualified opinion\b"),
    (
        "earnings_miss",
        r"\bearnings miss\b|\b(results|report|quarter) miss\b"
        r"|\bmiss(ed|es)? (estimates|consensus|expectations|revenue targets|analyst)\b"
        r"|\bwhen \w+ missed\b|\b\w+ missed and\b",
    ),
    (
        "share_plunge",
        r"\bshares? (are )?(plunging|tumbling|crashing|sink(ing)?)\b"
        r"|\b(is )?down \d+\.?\d*%"
        r"|\bshed \d+\.?\d*%"
        r"|\bstock(s)? trade down\b",
    ),
    (
        "monetization_risk",
        r"\bmonetization (doubts|concerns|uncertainty)\b|\bdelay raises\b|\bcapex concern",
    ),
]

_COMPILED: List[Tuple[str, re.Pattern[str]]] = [
    (label, re.compile(pat, re.IGNORECASE)) for label, pat in RED_FLAG_PATTERNS
]

LLM_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]

LLM_SYSTEM = """You are a quantitative risk gatekeeper for short-term equity entries (1–3 session horizon).

Question: Does this news indicate elevated probability (>5%) of meaningful downside over the next 1-3 sessions?

Output STRICT JSON only:
{
  "action": "EXECUTE" | "BLOCK",
  "risk_score": 0-100,
  "severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "main_drivers": ["short bullet", "..."],
  "counterarguments": ["why risk may be overstated", "..."],
  "reasoning": "One clear sentence summary",
  "risk_type": "NOISE" | "EARNINGS_RISK" | "STRUCTURAL_REGIME_CHANGE"
}

Scoring guide:
- 80-100 CRITICAL/HIGH: guidance/outlook cuts with price collapse, fraud probes, offerings, halts, clinical failure.
- 60-80 HIGH/MEDIUM: downgrades + guidance cuts without catastrophe (e.g. ZoomInfo-style).
- 30-50 MEDIUM/LOW: product delays, monetization doubts, known concerns — concerning but not crash catalysts (Meta-style).
- 0-30 LOW: nuance headlines where scary words appear but context is mixed (e.g. earnings miss alongside strong AI revenue).

BLOCK when risk_score >= 70 or structural tail-risk is clear.
EXECUTE when risk_score < 70 and headlines are noise, already-priced concerns, or mixed/contextual."""


@dataclass
class KeywordHit:
    label: str
    headline: str
    match: str


@dataclass
class HeadlineScanResult:
    symbol: str
    headlines: List[str]
    keyword_hits: List[KeywordHit] = field(default_factory=list)
    stage: str = "clear"  # clear | keyword_hit | llm_allow | llm_block | llm_error
    allowed: bool = True
    llm: Optional[Dict[str, Any]] = None
    reason: str = ""
    lookback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "headlines": self.headlines,
            "keyword_hits": [
                {"label": h.label, "headline": h.headline, "match": h.match}
                for h in self.keyword_hits
            ],
            "stage": self.stage,
            "allowed": self.allowed,
            "llm": self.llm,
            "reason": self.reason,
            "lookback": self.lookback,
        }


def resolve_lookback(days_arg: str, *, benchmark: bool) -> Tuple[Optional[float], Optional[int], str]:
    """
    Parse --days: 'auto' (session-aware hours) or integer days.
    Benchmark mode uses 7d for calibration regardless of auto.
    """
    if benchmark:
        return None, 7, "7d (benchmark calibration)"
    if str(days_arg).strip().lower() == "auto":
        hours, slot = headline_lookback_slot()
        return hours, None, f"{hours:g}h auto ({slot})"
    days = int(days_arg)
    return None, days, f"{days}d"


def fetch_headlines(
    symbol: str,
    *,
    limit: int = 5,
    max_age_days: Optional[int] = None,
    max_age_hours: Optional[float] = None,
) -> Tuple[str, List[str]]:
    """
    Fetch headlines for symbol; fall back to HEADLINE_SYMBOL_ALIASES when feed is empty.
    Returns (resolved_symbol, headlines).
    """
    sym = (symbol or "").strip().upper()
    kwargs: Dict[str, Any] = {"limit": limit}
    if max_age_hours is not None:
        kwargs["max_age_hours"] = max_age_hours
        kwargs["max_age_days"] = 0
    else:
        kwargs["max_age_days"] = max_age_days if max_age_days is not None else 3

    headlines = latest_headlines(sym, **kwargs)
    empty = (
        not headlines
        or (len(headlines) == 1 and headlines[0].startswith("No headlines"))
        or (len(headlines) == 1 and headlines[0].startswith("No recent"))
    )
    alias = HEADLINE_SYMBOL_ALIASES.get(sym)
    if empty and alias:
        alt = latest_headlines(alias, **kwargs)
        if alt and not (len(alt) == 1 and alt[0].startswith("No ")):
            return alias, alt
    return sym, headlines


def scan_keywords(headlines: Sequence[str]) -> List[KeywordHit]:
    """Return regex matches against headline text."""
    hits: List[KeywordHit] = []
    seen: set[Tuple[str, str]] = set()
    for headline in headlines:
        if not headline or headline.startswith("No recent") or headline.startswith("Error"):
            continue
        for label, pattern in _COMPILED:
            m = pattern.search(headline)
            if m is None:
                continue
            key = (label, headline)
            if key in seen:
                continue
            seen.add(key)
            hits.append(KeywordHit(label=label, headline=headline, match=m.group(0)))
    return hits


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def run_llm_gatekeeper(
    symbol: str,
    headlines: Sequence[str],
    *,
    trigger: str = "",
    keyword_hits: Sequence[KeywordHit],
    requested_symbol: str = "",
) -> Optional[Dict[str, Any]]:
    api_key = os.environ.get("GEMINI_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  LLM skip: GEMINI_KEY not set")
        return None

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("  LLM skip: google-genai not installed")
        return None

    user_payload = {
        "ticker": symbol,
        "requested_ticker": requested_symbol or symbol,
        "technical_trigger": trigger or "unspecified entry signal",
        "live_headlines": list(headlines),
        "regex_flags": [{"label": h.label, "headline": h.headline} for h in keyword_hits],
    }
    prompt = f"{LLM_SYSTEM}\n\nInput:\n{json.dumps(user_payload, indent=2)}"

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(temperature=0.0, top_p=1.0)

    for attempt, model in enumerate(LLM_MODELS):
        try:
            print(f"  LLM: calling {model}...")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            parsed = _extract_json(response.text or "")
            if parsed:
                time.sleep(0.5)
                return parsed
            print(f"  LLM: could not parse JSON from {model}")
        except Exception as exc:
            print(f"  LLM: {model} failed ({exc})")
            if attempt + 1 >= len(LLM_MODELS):
                break
    return None


def scan_symbol(
    symbol: str,
    *,
    limit: int = 5,
    max_age_days: Optional[int] = None,
    max_age_hours: Optional[float] = None,
    lookback: str = "",
    use_llm: bool = False,
    llm_all: bool = False,
    trigger: str = "",
) -> HeadlineScanResult:
    requested = (symbol or "").strip().upper()
    resolved, headlines = fetch_headlines(
        requested,
        limit=limit,
        max_age_days=max_age_days,
        max_age_hours=max_age_hours,
    )
    hits = scan_keywords(headlines)

    result = HeadlineScanResult(symbol=requested, headlines=headlines, lookback=lookback)
    if resolved != requested:
        result.reason = f"Headlines from alias {resolved}"

    if not hits:
        result.stage = "clear"
        result.allowed = True
        if not result.reason:
            result.reason = "No red-flag keywords in recent headlines"
        if not (use_llm and llm_all):
            return result
    else:
        result.keyword_hits = hits
        result.stage = "keyword_hit"
        result.allowed = True
        labels = sorted({h.label for h in hits})
        prefix = f"Headlines from alias {resolved}; " if resolved != requested else ""
        result.reason = f"{prefix}Keyword hit: {', '.join(labels)}"

    if not use_llm:
        return result

    llm = run_llm_gatekeeper(
        resolved,
        headlines,
        trigger=trigger,
        keyword_hits=hits,
        requested_symbol=requested,
    )
    if llm is None:
        result.stage = "llm_error"
        result.reason = "LLM unavailable or unparseable; keyword hits remain"
        return result

    result.llm = llm
    action = str(llm.get("action", "")).strip().upper()
    risk_score = llm.get("risk_score")
    try:
        if risk_score is not None and float(risk_score) >= 70 and action != "EXECUTE":
            action = "BLOCK"
    except (TypeError, ValueError):
        pass

    if action == "BLOCK":
        result.stage = "llm_block"
        result.allowed = False
        result.reason = str(llm.get("reasoning") or "LLM BLOCK")
    elif action == "EXECUTE":
        result.stage = "llm_allow"
        result.allowed = True
        result.reason = str(llm.get("reasoning") or "LLM EXECUTE")
    else:
        result.stage = "llm_error"
        result.reason = f"Unexpected LLM action: {action!r}"

    return result


def _display_verdict(result: HeadlineScanResult) -> str:
    """Human verdict line: ESCALATE when regex fired but LLM has not run."""
    if result.stage == "keyword_hit":
        return "ESCALATE"
    if result.stage == "llm_block":
        return "BLOCK"
    if result.stage == "llm_allow":
        return "ALLOW"
    if result.stage == "llm_error" and result.keyword_hits:
        return "ESCALATE"
    if result.stage == "clear":
        return "ALLOW"
    return "BLOCK" if not result.allowed else "ALLOW"


def _print_result(result: HeadlineScanResult, *, benchmark_note: str = "") -> None:
    print(f"\n{'=' * 60}")
    print(f"News scan — {result.symbol}")
    print(f"{'=' * 60}")
    if benchmark_note:
        print(f"  Benchmark: {benchmark_note}")
    if result.lookback:
        print(f"  Lookback:  {result.lookback}")

    print(f"  Headlines ({len(result.headlines)}):")
    for i, h in enumerate(result.headlines, 1):
        print(f"    [{i}] {h}")

    if result.keyword_hits:
        print("\n  Keyword hits:")
        for hit in result.keyword_hits:
            snippet = hit.headline if len(hit.headline) <= 90 else f"{hit.headline[:90]}…"
            print(f"    • {hit.label}: “{hit.match}” in “{snippet}”")
    else:
        print("\n  Keyword screen: CLEAR")

    verdict = _display_verdict(result)
    print(f"\n  Stage:   {result.stage}")
    print(f"  Verdict: {verdict}")
    if verdict == "ESCALATE":
        print("  Hint:    run with --llm to resolve to ALLOW / BLOCK")
    print(f"  Reason:  {result.reason}")

    if result.llm:
        print("\n  LLM:")
        print(f"    action:      {result.llm.get('action')}")
        print(f"    risk_score:  {result.llm.get('risk_score')}")
        print(f"    severity:    {result.llm.get('severity')}")
        print(f"    risk_type:   {result.llm.get('risk_type')}")
        print(f"    reasoning:   {result.llm.get('reasoning')}")
        drivers = result.llm.get("main_drivers") or []
        if drivers:
            print("    drivers:")
            for d in drivers:
                print(f"      - {d}")
        counters = result.llm.get("counterarguments") or []
        if counters:
            print("    counterarguments:")
            for c in counters:
                print(f"      - {c}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headline red-flag scan: regex screen + optional LLM gatekeeper"
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Ticker symbol(s), e.g. LLY PFE (omit with --benchmark)",
    )
    parser.add_argument(
        "--days",
        default="auto",
        metavar="DAYS",
        help="Headline window: 'auto' (session-aware) or integer days (default auto)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max headlines per symbol (default 5)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Run Gemini gatekeeper when keywords hit (requires GEMINI_KEY)",
    )
    parser.add_argument(
        "--llm-all",
        action="store_true",
        help="With --llm, score every symbol even when regex is clear (nuance tests)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run calibration set: PVH ZI META AVGO (ZI uses GTM headlines)",
    )
    parser.add_argument(
        "--trigger",
        default="",
        help="Optional entry context for LLM, e.g. 'Flux below_ma_up'",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON results only",
    )
    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="Print configured regex labels and exit",
    )
    args = parser.parse_args()

    if args.list_patterns:
        print("Red-flag patterns (label → regex):")
        for label, pat in RED_FLAG_PATTERNS:
            print(f"  {label:20} {pat}")
        return

    symbols = list(args.symbols or [])
    benchmark = args.benchmark
    if benchmark:
        symbols = ["PVH", "ZI", "META", "AVGO"]
    if not symbols:
        parser.error("Provide ticker symbol(s) or use --benchmark")

    try:
        max_age_hours, max_age_days, lookback_label = resolve_lookback(
            args.days, benchmark=benchmark
        )
    except ValueError:
        parser.error(f"--days must be 'auto' or an integer, got {args.days!r}")

    if not args.json:
        print(f"Headline window: {lookback_label}")

    results = [
        scan_symbol(
            sym,
            limit=args.limit,
            max_age_days=max_age_days,
            max_age_hours=max_age_hours,
            lookback=lookback_label,
            use_llm=args.llm,
            llm_all=args.llm_all,
            trigger=args.trigger,
        )
        for sym in symbols
    ]

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    for result in results:
        note = BENCHMARK_CASES.get(result.symbol, "") if benchmark else ""
        _print_result(result, benchmark_note=note)
    print()


if __name__ == "__main__":
    main()
