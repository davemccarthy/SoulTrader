#!/usr/bin/env python3
"""
Standalone BioSpace RSS probe: print latest headlines only.

Usage:
  python test_biospace_feed.py
  python test_biospace_feed.py --category drug-development --limit 20
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import feedparser


BIOSPACE_FEEDS: dict[str, str] = {
    "all-news": "https://www.biospace.com/all-news.rss",
    "business": "https://www.biospace.com/business.rss",
    "drug-development": "https://www.biospace.com/drug-development.rss",
    "fda": "https://www.biospace.com/FDA.rss",
    "deals": "https://www.biospace.com/deals.rss",
    "policy": "https://www.biospace.com/policy.rss",
}


def _entry_timestamp(entry) -> str:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return "unknown-timestamp"
    try:
        return datetime(*parsed[:6], tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return "unknown-timestamp"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump latest BioSpace headlines by RSS category.")
    parser.add_argument(
        "--category",
        choices=sorted(BIOSPACE_FEEDS.keys()),
        default="drug-development",
        help="BioSpace feed category (default: drug-development)",
    )
    parser.add_argument("--limit", type=int, default=15, help="How many entries to print (default: 15)")
    parser.add_argument("--url", type=str, default="", help="Optional explicit RSS URL override")
    args = parser.parse_args()

    url = args.url.strip() or BIOSPACE_FEEDS[args.category]
    feed = feedparser.parse(url)
    entries = list(feed.entries or [])

    print(f"Category: {args.category}")
    print(f"Feed: {url}")
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

