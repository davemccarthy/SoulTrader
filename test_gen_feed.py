#!/usr/bin/env python3
"""
Standalone GEN (Genetic Engineering & Biotechnology News) RSS probe.

Usage:
  python test_gen_feed.py
  python test_gen_feed.py --limit 20
  python test_gen_feed.py --url "http://feeds.feedburner.com/GenGeneticEngineeringAndBiotechnologyNews"
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import feedparser


GEN_FEED = "http://feeds.feedburner.com/GenGeneticEngineeringAndBiotechnologyNews"


def _entry_timestamp(entry) -> str:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return "unknown-timestamp"
    try:
        return datetime(*parsed[:6], tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return "unknown-timestamp"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump latest GEN headlines.")
    parser.add_argument("--limit", type=int, default=15, help="How many entries to print (default: 15)")
    parser.add_argument("--url", type=str, default=GEN_FEED, help="RSS URL to query")
    args = parser.parse_args()

    feed = feedparser.parse(args.url)
    entries = list(feed.entries or [])

    print(f"Feed: {args.url}")
    print(f"Entries fetched: {len(entries)} | bozo={bool(getattr(feed, 'bozo', False))}")
    if getattr(feed, "bozo", False):
        err = getattr(feed, "bozo_exception", None)
        if err:
            print(f"bozo_exception: {err}")

    for i, e in enumerate(entries[: max(0, args.limit)], start=1):
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        print(f"{i:>2}. {_entry_timestamp(e)} | {title}")
        print(f"    {link}")


if __name__ == "__main__":
    main()
