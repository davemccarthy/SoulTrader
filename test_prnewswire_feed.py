#!/usr/bin/env python3
"""
Standalone PR Newswire feed probe: print latest headlines only.

Usage:
  python test_prnewswire_feed.py
  python test_prnewswire_feed.py --limit 20
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import feedparser

PRNEWSWIRE_HEALTH_FEED = "https://www.prnewswire.com/rss/health-latest-news-list.rss"


def _entry_timestamp(entry) -> str:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return "unknown-timestamp"
    try:
        return datetime(*parsed[:6], tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return "unknown-timestamp"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump latest PR Newswire health/pharma headlines.")
    parser.add_argument("--limit", type=int, default=15, help="How many entries to print (default: 15)")
    parser.add_argument("--url", type=str, default=PRNEWSWIRE_HEALTH_FEED, help="RSS URL to query")
    args = parser.parse_args()

    feed = feedparser.parse(args.url)
    entries = list(feed.entries or [])

    print(f"Feed: {args.url}")
    print(f"Entries fetched: {len(entries)} | bozo={getattr(feed, 'bozo', None)}")
    if getattr(feed, "bozo", False):
        print(f"bozo_exception: {getattr(feed, 'bozo_exception', None)}")

    for i, e in enumerate(entries[: max(0, args.limit)], start=1):
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        print(f"{i:>2}. {_entry_timestamp(e)} | {title}")
        print(f"    {link}")


if __name__ == "__main__":
    main()

