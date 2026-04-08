#!/usr/bin/env python3
"""
Standalone Business Wire feed probe: print latest headlines only.

Usage:
  python test_businesswire_feed.py
  python test_businesswire_feed.py --limit 20
  python test_businesswire_feed.py --url "https://www.businesswire.com/portal/site/home/rss/"
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import feedparser

# General Business Wire RSS endpoint.
# Can be overridden with --url if you have a preferred health/life-sciences channel URL.
BUSINESSWIRE_FEED = "https://www.businesswire.com/portal/site/home/rss/"


def _entry_timestamp(entry) -> str:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return "unknown-timestamp"
    try:
        return datetime(*parsed[:6], tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return "unknown-timestamp"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump latest Business Wire headlines.")
    parser.add_argument("--limit", type=int, default=15, help="How many entries to print (default: 15)")
    parser.add_argument("--url", type=str, default=BUSINESSWIRE_FEED, help="RSS URL to query")
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

