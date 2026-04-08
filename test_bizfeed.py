#!/usr/bin/env python3
"""
Standalone BIZFEED probe: headline dump for major press-release RSS sources.

Targets:
  - PR Newswire
  - GlobeNewswire
  - Newswire.com (ACCESS Newswire consumer newsroom; same network as accessnewswire.com)

Usage:
  python test_bizfeed.py
  python test_bizfeed.py --provider globenewswire --channel biotechnology --limit 20
  python test_bizfeed.py --all --limit 10
  python test_bizfeed.py --verify-timestamps --all --limit 5
  python test_bizfeed.py --list-channels --provider prnewswire

Notes:
  - Newswire.com: https://www.newswire.com/newsroom/rss — cloudscraper tried first when installed,
    then urllib/requests. pip install cloudscraper helps on many networks; use file://… if needed.
  - Older Newswire /newsroom/rss/custom/... paths often 403 for bots.
  - GlobeNewswire retired /en/rss/industry/*.xml (404). Use RssFeed/subjectcode URLs from
    https://www.globenewswire.com/rss/list — biotechnology maps to Clinical Study subject.

Timestamps (for BIZFEED advisor parity with PHARM):
  - Prefer entry.published_parsed / updated_parsed (9-tuple); interpreted as UTC, same as
    core.services.advisors.pharm._entry_datetime_utc.
  - feedparser normalizes RSS pubDate (RFC 822 with TZ) into *_parsed in GMT/UTC.
  - Fallback: strptime on published/updated string (PHARM formats); else datetime.now(UTC).
"""

from __future__ import annotations

import argparse
import html
import sys
from typing import Any
import urllib.error
import urllib.request
from datetime import UTC, datetime
from urllib.parse import urlparse

import feedparser

# Chrome-like UA; Cloudflare often blocks feedparser’s default urllib User-Agent on Newswire.com.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def _strict_newswire_headers() -> dict[str, str]:
    """Headers for Newswire.com (blocks feedparser’s default urllib client / Cloudflare)."""
    referer = "https://www.newswire.com/newsroom"
    return {
        "User-Agent": _BROWSER_UA,
        "Accept": "application/rss+xml, application/xml, application/atom+xml, text/xml, */*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        # Avoid claiming "br" — stdlib urllib does not decode Brotli.
        "Accept-Encoding": "gzip, deflate",
        "Referer": referer,
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }


def _needs_strict_fetch(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == "newswire.com" or host.endswith(".newswire.com")


def _parse_rss_url(url: str):
    """
    Parse RSS from URL. Newswire.com uses browser-like headers.
    When cloudscraper is installed, it is tried first (often the only thing that
    passes Cloudflare); then urllib, then requests.
    """
    if not _needs_strict_fetch(url):
        return feedparser.parse(url, agent=_BROWSER_UA)

    headers = _strict_newswire_headers()

    def _attach_status(parsed, code: int):
        parsed.status = code
        return parsed

    # 1) cloudscraper first — matches typical success on servers / many local setups
    try:
        import cloudscraper

        scraper = cloudscraper.create_scraper()
        r = scraper.get(url, headers=headers, timeout=45)
        alt = feedparser.parse(r.content)
        _attach_status(alt, r.status_code)
        if r.status_code == 200 and (alt.entries or []):
            return alt
    except ImportError:
        pass
    except Exception:
        pass

    body: bytes = b""
    status: int | None = None

    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read()
        status = e.code
    except Exception:
        pass

    parsed = feedparser.parse(body) if body else feedparser.parse(b"")
    if status is not None:
        parsed.status = status

    if status == 200 and (parsed.entries or []):
        return parsed

    try:
        import requests

        r = requests.get(url, headers=headers, timeout=45)
        alt = feedparser.parse(r.content)
        _attach_status(alt, r.status_code)
        if r.status_code == 200 and (alt.entries or []):
            return alt
        if status != 200 and r.status_code == 200:
            return alt
    except Exception:
        pass

    return parsed

# ---------------------------------------------------------------------------
# Feed registry (default channels are BIZ-wide, not PHARM-narrow)
# ---------------------------------------------------------------------------

PRNEWSWIRE_FEEDS: dict[str, str] = {
    "all-news": "https://www.prnewswire.com/rss/all-news-list.rss",
    "news-releases": "https://www.prnewswire.com/rss/news-releases-list.rss",
    "health": "https://www.prnewswire.com/rss/health-latest-news-list.rss",
    "biotech": "https://www.prnewswire.com/rss/health-latest-news/biotechnology-list.rss",
}

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

GLOBENEWSWIRE_FEEDS: dict[str, str] = {
    "all-news": _GNW_ALL_NEWS,
    # Aliases for old industry names → subject “Health” (see rss/list).
    "life-sciences": _GNW_HEALTH,
    "healthcare": _GNW_HEALTH,
    "health": _GNW_HEALTH,
    "pharmaceuticals": _GNW_PHARMA,
    # No industry biotech XML anymore; Clinical Study is the closest subject feed.
    "biotechnology": _GNW_CLINICAL,
    "clinical-study": _GNW_CLINICAL,
}

# Catalog index: https://www.newswire.com/rss — working main feed is /newsroom/rss .
_NEWSWIRE_NEWSROOM_RSS = "https://www.newswire.com/newsroom/rss"

NEWSWIRE_FEEDS: dict[str, str] = {
    "newsroom": _NEWSWIRE_NEWSROOM_RSS,
    # Same URL as newsroom; kept for backward compatibility with earlier BIZFEED defaults.
    "all-press-releases": _NEWSWIRE_NEWSROOM_RSS,
    # Legacy path; frequently 403 / HTML from bots — use --url only if you need to retry.
    "custom-all-press-releases-legacy": (
        "https://www.newswire.com/newsroom/rss/custom/all-press-releases"
    ),
    "banking-financial": "https://www.newswire.com/newsroom/rss/beat/banking-and-financial-services",
    "healthcare-pharma": "https://www.newswire.com/newsroom/rss/beat/healthcare-and-pharmaceutical",
    "technology": "https://www.newswire.com/newsroom/rss/beat/computers-technology-and-internet",
}

PROVIDERS: dict[str, dict[str, str]] = {
    "prnewswire": PRNEWSWIRE_FEEDS,
    "globenewswire": GLOBENEWSWIRE_FEEDS,
    "newswire": NEWSWIRE_FEEDS,
}

DEFAULT_CHANNEL: dict[str, str] = {
    "prnewswire": "all-news",
    "globenewswire": "all-news",
    "newswire": "newsroom",
}

ALL_PROVIDERS: tuple[str, ...] = ("prnewswire", "globenewswire", "newswire")


def _entry_datetime_utc(entry: Any) -> datetime:
    """
    Match PHARM `core.services.advisors.pharm._entry_datetime_utc` for advisor parity.
    *_parsed tuples are treated as UTC (feedparser convention for normalized dates).
    """
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=UTC)
        except Exception:
            pass
    raw = (entry.get("published") or entry.get("updated") or "").strip()
    if raw:
        for fmt in ("%b %d, %Y %I:%M%p", "%b %d, %Y %I:%M %p"):
            try:
                return datetime.strptime(raw, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
    return datetime.now(UTC)


def _entry_timestamp(entry: Any) -> str:
    return _entry_datetime_utc(entry).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp_source(entry: Any) -> str:
    if entry.get("published_parsed"):
        return "published_parsed→UTC"
    if entry.get("updated_parsed"):
        return "updated_parsed→UTC"
    raw = (entry.get("published") or entry.get("updated") or "").strip()
    if raw:
        return "raw string strptime→UTC (verify source TZ in string)"
    return "fallback now(UTC)"


def dump_timestamp_verification(
    *,
    provider: str,
    channel: str,
    url: str,
    limit: int,
) -> int:
    """Print per-entry date fields and UTC normalization; exit 1 if feed empty."""
    feed = _parse_rss_url(url)
    entries = list(feed.entries or [])
    status = getattr(feed, "status", None)

    print(f"=== Timestamp verification: {provider} | {channel} ===")
    print(f"URL: {url}")
    print(f"HTTP status: {status!r} | entries: {len(entries)}")
    print(
        f"Feed-level: updated={getattr(feed, 'updated', None)!r} | "
        f"updated_parsed={getattr(feed, 'updated_parsed', None)!r}"
    )
    print(
        "Policy: published_parsed/updated_parsed 6-tuple → datetime(*[:6], tzinfo=UTC) "
        "(same as PHARM advisor)."
    )
    print()

    if not entries:
        print("(no entries)\n")
        return 1

    n = max(0, min(limit, len(entries)))
    missing_parsed = 0
    for i, e in enumerate(entries[:n], start=1):
        pub_raw = (e.get("published") or "").strip() or None
        upd_raw = (e.get("updated") or "").strip() or None
        pp = e.get("published_parsed")
        up = e.get("updated_parsed")
        if not pp and not up:
            missing_parsed += 1
        title = html.unescape((e.get("title") or "").strip())[:72]
        dt_utc = _entry_datetime_utc(e)
        src = _timestamp_source(e)
        print(f"[{i}] {title}")
        print(f"    published (raw):     {pub_raw!r}")
        print(f"    updated (raw):       {upd_raw!r}")
        print(f"    published_parsed:    {pp!r}")
        print(f"    updated_parsed:      {up!r}")
        print(f"    --> UTC (PHARM rule): {dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}  ({src})")
        print()

    print(
        f"Summary: {n} entries shown; {missing_parsed} missing both *_parsed "
        f"(rely on string or now fallback)."
    )
    print()
    return 0


def _one_line_summary(entry: dict) -> str:
    raw = (entry.get("summary") or entry.get("description") or "").strip()
    return html.unescape(" ".join(raw.split()))[:240]


def dump_feed(
    *,
    provider: str,
    channel: str,
    url: str,
    limit: int,
    show_summary: bool,
) -> int:
    """Print feed header and up to `limit` entries. Returns process exit code hint (0 ok, 1 soft failure)."""
    feed = _parse_rss_url(url)
    entries = list(feed.entries or [])
    status = getattr(feed, "status", None)
    bozo = getattr(feed, "bozo", False)

    print(f"Provider: {provider} | channel: {channel}")
    print(f"URL: {url}")
    print(f"HTTP status: {status!r} | entries: {len(entries)} | bozo={bozo}")
    if bozo:
        print(f"bozo_exception: {getattr(feed, 'bozo_exception', None)!r}")

    if not entries:
        if _needs_strict_fetch(url) and status == 403:
            try:
                import cloudscraper  # noqa: F401
            except ImportError:
                print(
                    "Hint: pip install cloudscraper — often fixes Newswire over Cloudflare.",
                    file=sys.stderr,
                )
            else:
                print(
                    "Hint: cloudscraper returned no items; try another network/VPN (IP/geo rules vary).",
                    file=sys.stderr,
                )
        print("(no entries — check URL, network, or bot protection on the host)\n")
        return 1

    n = max(0, limit)
    for i, e in enumerate(entries[:n], start=1):
        title = html.unescape((e.get("title") or "").strip())
        link = (e.get("link") or "").strip()
        print(f"{i:>2}. {_entry_timestamp(e)} | {title}")
        print(f"    {link}")
        if show_summary:
            summ = _one_line_summary(e)
            if summ:
                print(f"    {summ}")
    print()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe BIZFEED RSS sources (PR Newswire, GlobeNewswire, Newswire.com)."
    )
    parser.add_argument(
        "--provider",
        choices=["prnewswire", "globenewswire", "newswire", "all"],
        default="prnewswire",
        help="Which source to query (default: prnewswire). Use 'all' for each provider's default channel.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Shorthand for --provider all.",
    )
    parser.add_argument(
        "--channel",
        default="",
        help="Sub-feed key for that provider (see --list-channels). Default is provider-specific.",
    )
    parser.add_argument("--limit", type=int, default=15, help="Max entries per feed (default: 15)")
    parser.add_argument(
        "--url",
        default="",
        help="Override RSS URL entirely (ignores --provider / --channel for that single fetch).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a one-line summary/description per entry when present.",
    )
    parser.add_argument(
        "--list-channels",
        action="store_true",
        help="List channel keys for --provider and exit.",
    )
    parser.add_argument(
        "--verify-timestamps",
        action="store_true",
        help="Print raw date fields, *_parsed tuples, and UTC time (PHARM-aligned) for each entry.",
    )
    args = parser.parse_args()
    if args.all:
        args.provider = "all"

    if args.list_channels:
        prov = args.provider
        if prov == "all":
            print("Use --list-channels with a specific --provider (not 'all').", file=sys.stderr)
            sys.exit(2)
        feeds = PROVIDERS[prov]
        print(f"Channels for {prov}:")
        for key in sorted(feeds.keys()):
            print(f"  {key}")
        sys.exit(0)

    exit_code = 0

    if args.verify_timestamps:
        lim = max(1, args.limit)

        def _run_verify(prov: str, ch: str, u: str) -> int:
            return dump_timestamp_verification(
                provider=prov, channel=ch, url=u, limit=lim
            )

        if args.url.strip():
            sys.exit(_run_verify("custom", "override", args.url.strip()))
        if args.provider == "all":
            for prov in ALL_PROVIDERS:
                ch = DEFAULT_CHANNEL[prov]
                url = PROVIDERS[prov][ch]
                exit_code = max(exit_code, _run_verify(prov, ch, url))
            sys.exit(exit_code)
        prov = args.provider
        ch = args.channel.strip() or DEFAULT_CHANNEL[prov]
        feeds = PROVIDERS[prov]
        if ch not in feeds:
            valid = ", ".join(sorted(feeds.keys()))
            print(f"Unknown channel {ch!r} for {prov}. Valid: {valid}", file=sys.stderr)
            sys.exit(2)
        sys.exit(_run_verify(prov, ch, feeds[ch]))

    if args.url.strip():
        rc = dump_feed(
            provider="custom",
            channel="override",
            url=args.url.strip(),
            limit=args.limit,
            show_summary=args.summary,
        )
        sys.exit(rc)

    if args.provider == "all":
        for prov in ALL_PROVIDERS:
            ch = DEFAULT_CHANNEL[prov]
            url = PROVIDERS[prov][ch]
            rc = dump_feed(
                provider=prov,
                channel=ch,
                url=url,
                limit=args.limit,
                show_summary=args.summary,
            )
            exit_code = max(exit_code, rc)
        sys.exit(exit_code)

    prov = args.provider
    ch = args.channel.strip() or DEFAULT_CHANNEL[prov]
    feeds = PROVIDERS[prov]
    if ch not in feeds:
        valid = ", ".join(sorted(feeds.keys()))
        print(f"Unknown channel {ch!r} for {prov}. Valid: {valid}", file=sys.stderr)
        sys.exit(2)
    url = feeds[ch]
    rc = dump_feed(
        provider=prov,
        channel=ch,
        url=url,
        limit=args.limit,
        show_summary=args.summary,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
