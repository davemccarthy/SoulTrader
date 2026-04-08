#!/usr/bin/env python3
"""
Standalone FierceBiotech RSS probe: print latest headlines only.

Usage:
  python test_fiercebiotech_feed.py
  python test_fiercebiotech_feed.py --section biotech --limit 20
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import html
import re

import feedparser


FIERCE_FEEDS: dict[str, str] = {
    "all": "https://www.fiercebiotech.com/rss/xml",
    "biotech": "https://www.fiercebiotech.com/rss/biotech/xml",
    "medtech": "https://www.fiercebiotech.com/rss/medtech/xml",
    "cro": "https://www.fiercebiotech.com/rss/cro/xml",
}


def _entry_timestamp(entry) -> str:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        # Fierce often provides a plain string like "Mar 30, 2026 6:23pm"
        raw = (entry.get("published") or entry.get("updated") or "").strip()
        if raw:
            for fmt in ("%b %d, %Y %I:%M%p", "%b %d, %Y %I:%M %p"):
                try:
                    dt = datetime.strptime(raw, fmt)
                    # Source string has no timezone; emit ISO without Z.
                    return dt.strftime("%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    continue
        return "unknown-timestamp"
    try:
        return datetime(*parsed[:6], tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return "unknown-timestamp"


def _clean_title(raw: str) -> str:
    text = html.unescape(raw or "")
    # Fierce feed sometimes wraps titles in <a ...>...</a>.
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.split()).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump latest FierceBiotech headlines by section.")
    parser.add_argument(
        "--section",
        choices=sorted(FIERCE_FEEDS.keys()),
        default="all",
        help="FierceBiotech RSS section (default: all)",
    )
    parser.add_argument("--limit", type=int, default=15, help="How many entries to print (default: 15)")
    parser.add_argument("--url", type=str, default="", help="Optional explicit RSS URL override")
    args = parser.parse_args()

    url = args.url.strip() or FIERCE_FEEDS[args.section]
    feed = feedparser.parse(url)
    entries = list(feed.entries or [])

    print(f"Section: {args.section}")
    print(f"Feed: {url}")
    print(f"Entries fetched: {len(entries)} | bozo={getattr(feed, 'bozo', None)}")
    if getattr(feed, "bozo", False):
        print(f"bozo_exception: {getattr(feed, 'bozo_exception', None)}")

    for i, e in enumerate(entries[: max(0, args.limit)], start=1):
        title = _clean_title(e.get("title") or "")
        link = (e.get("link") or "").strip()
        print(f"{i:>2}. {_entry_timestamp(e)} | {title}")
        print(f"    {link}")


if __name__ == "__main__":
    main()

