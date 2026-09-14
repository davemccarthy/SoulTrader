#!/usr/bin/env python3
"""
Meyka v0 — standalone belief-mismatch shadow tester.

Detects tension between dominant market belief and contradictory evidence in a
news item or event summary. Does NOT output buy/sell; outputs conviction scaling
signals for shadow evaluation against real outcomes.

Uses shared core.services.meyka (Gemini first, DeepSeek fallback).

Usage:
  python test_meyka.py --ticker CRM --text "Salesforce shares rose after..."
  python test_meyka.py --ticker CRM --text-file article.txt
  python test_meyka.py --ticker CRM --text-file article.txt --log meyka_shadow.csv
  python test_meyka.py --ticker CRM   # runs built-in CRM sample
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

SAMPLE_TEXT = """
Salesforce shares rose after analysts noted accelerating adoption of AI-driven Agentforce tools.
However, concerns remain about long-term SaaS disruption from AI-native competitors.
Revenue growth slowed versus prior quarters while management raised full-year guidance.
"""


def _setup_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meyka v0 belief-mismatch shadow tester.")
    parser.add_argument("--ticker", "-t", required=True, help="Ticker symbol, e.g. CRM")
    parser.add_argument("--text", help="Article or event text to analyze.")
    parser.add_argument("--text-file", type=Path, help="Read article text from file.")
    parser.add_argument(
        "--log",
        type=Path,
        help="Append result row to CSV for shadow backtesting.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="LLM timeout seconds.")
    parser.add_argument("--sample", action="store_true", help="Use built-in CRM sample text.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    elif args.sample or not args.text:
        text = SAMPLE_TEXT
    else:
        print("ERROR: provide --text, --text-file, or --sample", file=sys.stderr)
        return 2

    if not text.strip():
        print("ERROR: empty input text", file=sys.stderr)
        return 2

    _setup_django()

    from core.services.meyka import analyze_opportunity, append_shadow_log

    model, output = analyze_opportunity(text, args.ticker, timeout=args.timeout)
    print(json.dumps(output, indent=2, ensure_ascii=False))

    if args.log:
        append_shadow_log(args.log, output, source="test_meyka")
        print(f"\nLogged to {args.log}", file=sys.stderr)

    if output.get("error"):
        return 1
    if not model:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
