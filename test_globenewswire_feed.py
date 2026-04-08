#!/usr/bin/env python3
"""
Standalone GlobeNewswire feed probe: print latest headlines only.

GlobeNewswire removed /en/rss/industry/*.xml (404). This script uses RssFeed/subjectcode
URLs from https://www.globenewswire.com/rss/list .

Usage:
  python test_globenewswire_feed.py
  python test_globenewswire_feed.py --industry biotechnology --limit 20
"""

from __future__ import annotations

import argparse
import html
from datetime import UTC, datetime

import feedparser

_GNW_ALL_NEWS = (
    "https://www.globenewswire.com/RssFeed/orgclass/1/"
    "feedTitle/GlobeNewswire%20-%20All%20News"
)
_GNW_HEALTH = (
    "https://www.globenewswire.com/RssFeed/subjectcode/20-Health/"
    "feedTitle/GlobeNewswire%20-%20Health"
)
_GNW_CLINICAL = (
    "https://www.globenewswire.com/RssFeed/subjectcode/90-Clinical%20Study/"
    "feedTitle/GlobeNewswire%20-%20Clinical%20Study"
)
_GNW_PHARMA = (
    "https://www.globenewswire.com/RssFeed/subjectcode/3026-MedicalPharmaceutical/"
    "feedTitle/GlobeNewswire%20-%20Medical%20Pharmaceutical"
)

INDUSTRY_FEEDS: dict[str, str] = {
    "all-news": _GNW_ALL_NEWS,
    "life-sciences": _GNW_HEALTH,
    "healthcare": _GNW_HEALTH,
    "health": _GNW_HEALTH,
    "pharmaceuticals": _GNW_PHARMA,
    "biotechnology": _GNW_CLINICAL,
    "clinical-study": _GNW_CLINICAL,
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
    parser = argparse.ArgumentParser(
        description="Dump latest GlobeNewswire headlines (subject/org RssFeed URLs)."
    )
    parser.add_argument(
        "--industry",
        choices=sorted(INDUSTRY_FEEDS.keys()),
        default="all-news",
        help="Feed preset (default: all-news). biotechnology → Clinical Study subject.",
    )
    parser.add_argument("--url", type=str, default="", help="Optional explicit RSS URL override")
    parser.add_argument("--limit", type=int, default=15, help="How many entries to print (default: 15)")
    args = parser.parse_args()

    url = args.url.strip() or INDUSTRY_FEEDS[args.industry]
    feed = feedparser.parse(url)
    entries = list(feed.entries or [])

    print(f"Industry: {args.industry}")
    print(f"Feed: {url}")
    print(f"Entries fetched: {len(entries)} | bozo={getattr(feed, 'bozo', None)}")
    if getattr(feed, "bozo", False):
        print(f"bozo_exception: {getattr(feed, 'bozo_exception', None)}")

    for i, e in enumerate(entries[: max(0, args.limit)], start=1):
        title = html.unescape((e.get("title") or "").strip())
        link = (e.get("link") or "").strip()
        print(f"{i:>2}. {_entry_timestamp(e)} | {title}")
        print(f"    {link}")


if __name__ == "__main__":
    main()
