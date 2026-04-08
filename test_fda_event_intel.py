#!/usr/bin/env python3
"""
Feed-first FDA/pharma event intel probe (no Drugs@FDA scraping required).

What this does:
1) Pulls RSS/Atom feeds.
2) Detects likely FDA catalyst events (approval, CRL, label expansion, etc.).
3) Normalizes each event to a compact JSON payload.
4) Optionally sends each event to Gemini for structured surprise/significance scoring.
5) Writes machine-friendly output JSON.

Usage examples:
  python test_fda_event_intel.py
  python test_fda_event_intel.py --max-events 20 --no-llm
  python test_fda_event_intel.py --max-events 30 --out event_intel.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

try:
    import feedparser
except Exception:  # pragma: no cover - import guard for local envs without feedparser
    feedparser = None

try:
    import google.generativeai as genai
    from google.api_core import exceptions as genai_exceptions
except Exception:  # pragma: no cover
    genai = None
    genai_exceptions = None

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


FEEDS: dict[str, str] = {
    "fda_press_releases": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
}

# Keep this set focused on approval/surprise catalysts.
KEYWORDS = (
    "fda approves",
    "fda grants approval",
    "fda granted approval",
    "grants approval",
    "granted approval",
    "approved by the fda",
    "new drug application",
    "biologics license application",
    "nda approval",
    "bla approval",
    "supplemental nda",
    "label expansion",
    "complete response letter",
    "crl",
    "pdufa",
)

# Regex triggers for language variants where words may appear between key terms.
KEYWORD_PATTERNS = (
    re.compile(r"\bfda\s+grants\b.*\bapproval\b", re.IGNORECASE),
    re.compile(r"\bfda\s+granted\b.*\bapproval\b", re.IGNORECASE),
)

# Retry-friendly model list; first is fast/cost-effective.
MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
)


PROMPT_TEMPLATE = """You are a pharma market intelligence assistant.

Input: A newly detected potential FDA event (fresh news / press release), structured as JSON:

{event_json_here}

Task: Analyze the event and return a JSON object with the following fields:
1. "drug_name" -> extract the approved product/brand name from title/summary/press_release_text
2. "company_name" -> extract the sponsor/company from title/summary/press_release_text
3. "approval_type" -> one of: "NDA", "BLA", "SUPPL", "CRL", "unknown"
4. "surprise" -> "yes", "likely", or "no"
5. "significance" -> "high", "medium", or "low"
6. "rationale" -> 1-2 sentences explaining why you chose the values
7. "confidence" -> "high", "medium", "low"

Rules:
- Base your analysis only on the content in the input JSON (title + summary + press_release_text), plus general knowledge of FDA approval patterns.
- Do not hallucinate exact stock prices or NDA numbers.
- Always output valid JSON with all fields filled.
- If company/product cannot be extracted with confidence, output "unknown" (never null).
- Keep output concise and structured.
"""


@dataclass
class FeedEvent:
    source_name: str
    source_url: str
    title: str
    summary: str
    press_release_text: str | None
    approval_date: str
    link: str
    drug_name: str | None
    generic_name: str | None
    company_name: str | None
    approval_type: str
    fresh: bool

    def as_llm_input(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "press_release_text": (self.press_release_text or "")[:6000],
            "drug_name": self.drug_name,
            "generic_name": self.generic_name,
            "company_name": self.company_name,
            "approval_type": self.approval_type,
            "approval_date": self.approval_date,
            "fresh": self.fresh,
        }

    def as_record(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_url": self.source_url,
            "title": self.title,
            "summary": self.summary,
            "press_release_text": self.press_release_text,
            "approval_date": self.approval_date,
            "link": self.link,
            "drug_name": self.drug_name,
            "generic_name": self.generic_name,
            "company_name": self.company_name,
            "approval_type": self.approval_type,
            "fresh": self.fresh,
        }


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().split())


def _guess_approval_type(blob: str) -> str:
    b = blob.lower()
    if "complete response letter" in b or re.search(r"\bcrl\b", b):
        return "CRL"
    if "supplemental nda" in b or "snda" in b or "supplement" in b:
        return "SUPPL"
    if "biologics license application" in b or re.search(r"\bbla\b", b):
        return "BLA"
    if "new drug application" in b or re.search(r"\bnda\b", b):
        return "NDA"
    return "unknown"


def _naive_entities_from_title(title: str) -> tuple[str | None, str | None]:
    """
    Lightweight heuristic only; the LLM can refine when title/summary are descriptive.
    """
    t = _clean_text(title)
    tl = t.lower()

    # FDA-style title: "FDA Approves <drug/therapy> for <condition>"
    m = re.search(r"fda approves\s+(.+?)(?:\s+for\s+|$)", t, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip(" .,:;")
        # Trim generic lead words for better placeholder quality.
        candidate = re.sub(
            r"^(first|new|drug|treatment|therapy|product|labeling changes to)\s+",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        if candidate.lower() in {
            "treatment",
            "therapy",
            "product",
            "products",
            "device",
            "drug",
            "labeling changes",
            "labeling changes to",
        }:
            candidate = ""
        return candidate or None, None

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", t)
    if not words:
        return None, None
    cap_words = [w for w in words if w[:1].isupper() and len(w) >= 4 and w.lower() not in {"fda", "approves"}]
    drug = cap_words[0] if cap_words else None
    company = cap_words[-1] if len(cap_words) >= 2 else None
    return drug, company


def _extract_drug_from_summary(summary: str) -> tuple[str | None, str | None]:
    """
    Common FDA press pattern:
      "approved BrandName (generic-name) ..."
    Returns (brand_or_drug_name, generic_name).
    """
    s = _clean_text(summary)
    m = re.search(r"approved\s+([A-Za-z0-9][A-Za-z0-9'/-]+)\s*\(([^)]+)\)", s, flags=re.IGNORECASE)
    if m:
        brand = m.group(1).strip(" .,:;")
        generic = m.group(2).strip(" .,:;")
        return (brand or None), (generic or None)

    return None, None


def _extract_company_from_summary(summary: str) -> str | None:
    """
    Best-effort company extraction from common press-release phrasing.
    """
    s = _clean_text(summary)
    # Pattern set 1: "<Company> announced ..."
    m = re.search(
        r"([A-Z][A-Za-z0-9&'., -]{2,120}?)\s+(?:announced|today announced|has announced)\b",
        s,
    )
    if m:
        candidate = m.group(1).strip(" .,:;")
        if candidate and "U.S. Food and Drug Administration" not in candidate:
            return candidate

    # Pattern set 2: "... granted ... to <Company>."
    grant_triggers = (
        r"approval(?:\s+was)?\s+granted\s+to",
        r"granted\s+the\s+approval\s+to",
        r"granted\s+(?:accelerated\s+)?approval(?:\s+of\s+[A-Za-z0-9'(), -]+)?\s+to",
    )
    for trigger in grant_triggers:
        m = re.search(trigger + r"\s+([^.;]+)", s, flags=re.IGNORECASE)
        if not m:
            continue
        candidate = m.group(1).strip(" .,:;")
        # Drop trailing qualifier fragments if present.
        candidate = re.sub(r"\s+(?:for|in|under|with)\b.*$", "", candidate, flags=re.IGNORECASE).strip(" .,:;")
        if candidate and "U.S. Food and Drug Administration" not in candidate:
            return candidate

    # Pattern set 3: "... by/from <Company>"
    m = re.search(r"(?:by|from)\s+([A-Z][A-Za-z0-9&'., -]{2,120}?)(?:[.;]|$)", s)
    if m:
        candidate = m.group(1).strip(" .,:;")
        if candidate and "U.S. Food and Drug Administration" not in candidate:
            return candidate
    return None


def _fetch_press_release_text(url: str) -> str | None:
    if not url:
        return None
    try:
        r = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; soultrader-bot/1.0)"},
        )
        if r.status_code != 200 or not r.text:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        # Prefer article/body content blocks when available.
        for sel in ("article", "main", "div.field--name-body", "div.region-content"):
            node = soup.select_one(sel)
            if node:
                txt = " ".join(node.get_text(" ", strip=True).split())
                if len(txt) >= 120:
                    return txt
        txt = " ".join(soup.get_text(" ", strip=True).split())
        return txt if len(txt) >= 120 else None
    except Exception:
        return None


def _entry_date_iso(entry: Any) -> str:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_candidate_event(title: str, summary: str) -> bool:
    blob = f"{title} {summary}".lower()
    if any(k in blob for k in KEYWORDS):
        return True
    return any(p.search(blob) is not None for p in KEYWORD_PATTERNS)


def extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def ask_gemini_json(prompt: str, *, timeout: float = 120.0) -> tuple[str | None, dict[str, Any] | None, str | None]:
    if genai is None or genai_exceptions is None:
        return None, None, "google.generativeai not available"
    api_key = os.environ.get("GEMINI_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, None, "GEMINI_KEY/GEMINI_API_KEY missing"

    genai.configure(api_key=api_key)
    model_index = 0
    retry_exceptions = (
        genai_exceptions.ServiceUnavailable,
        genai_exceptions.ResourceExhausted,
        genai_exceptions.DeadlineExceeded,
        genai_exceptions.InternalServerError,
    )

    attempts: list[str] = []
    for _ in range(len(MODELS)):
        model = MODELS[model_index]
        try:
            response = genai.GenerativeModel(model).generate_content(
                prompt,
                request_options={"timeout": timeout},
                generation_config={"temperature": 0.0},
            )
            if not response.candidates or not response.candidates[0].content.parts:
                attempts.append(f"{model}: empty response candidates")
                model_index = (model_index + 1) % len(MODELS)
                continue
            text = response.candidates[0].content.parts[0].text or ""
            parsed = extract_json(text)
            if parsed:
                return model, parsed, None
            preview = text.strip().replace("\n", " ")[:140]
            attempts.append(f"{model}: response not valid JSON (preview: {preview!r})")
        except retry_exceptions:
            attempts.append(f"{model}: transient API error")
        except Exception as exc:
            return model, None, f"{model}: fatal error: {exc}"
        model_index = (model_index + 1) % len(MODELS)
    return None, None, ("; ".join(attempts) if attempts else "LLM unavailable")


def _has_gemini_key() -> bool:
    return bool(os.environ.get("GEMINI_KEY") or os.environ.get("GEMINI_API_KEY"))


def fetch_feed_events(source_name: str, feed_url: str, *, verbose: bool = False) -> list[FeedEvent]:
    if feedparser is None:
        raise RuntimeError("feedparser is not installed. Install via: pip install feedparser")

    parsed = feedparser.parse(feed_url)
    out: list[FeedEvent] = []
    skipped = 0
    for entry in parsed.entries:
        title = _clean_text(entry.get("title"))
        summary = _clean_text(entry.get("summary") or entry.get("description"))
        if not _is_candidate_event(title, summary):
            skipped += 1
            if verbose:
                print(f"[skip:{source_name}] {title[:140]}")
            continue

        press_release_text = _fetch_press_release_text(_clean_text(entry.get("link")))
        parse_blob = f"{summary} {(press_release_text or '')}"

        # Keep deterministic extraction conservative; LLM is responsible for final entity extraction.
        _title_drug, _title_company = _naive_entities_from_title(title)
        summary_drug, summary_generic = _extract_drug_from_summary(parse_blob)
        summary_company = _extract_company_from_summary(parse_blob)
        drug_name = summary_drug or None
        company_name = summary_company or None
        approval_type = _guess_approval_type(f"{title} {summary}")
        out.append(
            FeedEvent(
                source_name=source_name,
                source_url=feed_url,
                title=title,
                summary=summary,
                press_release_text=press_release_text,
                approval_date=_entry_date_iso(entry),
                link=_clean_text(entry.get("link")),
                drug_name=drug_name,
                generic_name=summary_generic,
                company_name=company_name,
                approval_type=approval_type,
                fresh=True,
            )
        )
    if verbose:
        print(f"[feed:{source_name}] matched={len(out)} skipped={skipped} total={len(parsed.entries)}")
    return out


def fetch_all_events(feed_map: dict[str, str], *, verbose: bool = False) -> list[FeedEvent]:
    rows: list[FeedEvent] = []
    for source_name, url in feed_map.items():
        try:
            rows.extend(fetch_feed_events(source_name, url, verbose=verbose))
        except Exception as exc:
            print(f"[warn] feed '{source_name}' failed: {exc}")
    return rows


def dedupe_events(events: list[FeedEvent]) -> list[FeedEvent]:
    seen: set[tuple[str, str, str]] = set()
    out: list[FeedEvent] = []
    for e in events:
        key = (e.approval_date, e.title.lower(), e.source_name)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def score_event(event: FeedEvent, *, use_llm: bool, verbose: bool = False) -> dict[str, Any]:
    base = event.as_record()
    if not use_llm:
        base.update(
            {
                "surprise": "n/a",
                "significance": "n/a",
                "rationale": "LLM disabled; no surprise/significance scoring performed.",
                "confidence": "n/a",
                "llm_model": None,
            }
        )
        return base

    prompt = PROMPT_TEMPLATE.replace(
        "{event_json_here}",
        json.dumps(event.as_llm_input(), ensure_ascii=True, indent=2),
    )
    if verbose:
        print("\n--- GEMINI PROMPT START ---")
        print(prompt)
        print("--- GEMINI PROMPT END ---\n")
    model, parsed, llm_error = ask_gemini_json(prompt)

    # Hard fallback for strict schema continuity in downstream pipelines.
    scored = {
        "drug_name": event.drug_name,
        "company_name": event.company_name,
        "approval_type": event.approval_type,
        "surprise": "n/a",
        "significance": "n/a",
        "rationale": "LLM result unavailable or unparseable; no surprise/significance scoring performed.",
        "confidence": "n/a",
    }
    if parsed:
        parsed_drug = parsed.get("drug_name", scored["drug_name"])
        parsed_company = parsed.get("company_name", scored["company_name"])
        scored.update(
            {
                "drug_name": parsed_drug or "unknown",
                "company_name": parsed_company or "unknown",
                "approval_type": parsed.get("approval_type", scored["approval_type"]),
                "surprise": parsed.get("surprise", scored["surprise"]),
                "significance": parsed.get("significance", scored["significance"]),
                "rationale": parsed.get("rationale", scored["rationale"]),
                "confidence": parsed.get("confidence", scored["confidence"]),
            }
        )
    # Never leave entity fields null in LLM mode.
    if not scored.get("drug_name"):
        scored["drug_name"] = "unknown"
    if not scored.get("company_name"):
        scored["company_name"] = "unknown"
    scored["llm_model"] = model
    if parsed is None:
        print(f"[warn] LLM scoring unavailable for '{event.title[:72]}': {llm_error}")
    return {**base, **scored}


def run_pipeline(*, max_events: int, use_llm: bool, feed_map: dict[str, str], verbose: bool = False) -> list[dict[str, Any]]:
    events = dedupe_events(fetch_all_events(feed_map, verbose=verbose))
    if use_llm and not _has_gemini_key():
        print("[warn] GEMINI_KEY/GEMINI_API_KEY not set; LLM scoring will be n/a.")
    events.sort(key=lambda e: (e.approval_date, e.source_name), reverse=True)
    if max_events > 0:
        events = events[:max_events]

    out: list[dict[str, Any]] = []
    for idx, event in enumerate(events, start=1):
        print(f"[{idx}/{len(events)}] {event.approval_date} | {event.title[:90]}")
        out.append(score_event(event, use_llm=use_llm, verbose=verbose and use_llm))
        if use_llm:
            time.sleep(0.4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Feed-first FDA catalyst intel probe (Gemini-scored JSON output).")
    parser.add_argument("--max-events", type=int, default=25, help="Maximum deduped events to score (default: 25)")
    parser.add_argument("--no-llm", action="store_true", help="Skip Gemini scoring and emit placeholders only")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "event_intel_output.json", help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Print skipped headlines and feed match stats")
    parser.add_argument(
        "--feed-url",
        action="append",
        default=[],
        help="Optional extra feed URL(s) to include. Can pass multiple times.",
    )
    args = parser.parse_args()

    feed_map = dict(FEEDS)
    for i, url in enumerate(args.feed_url, start=1):
        feed_map[f"custom_{i}"] = url
    rows = run_pipeline(
        max_events=max(0, args.max_events),
        use_llm=not args.no_llm,
        feed_map=feed_map,
        verbose=args.verbose,
    )
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {len(rows)} scored event(s) to: {args.out}")

    # Short console preview
    for r in rows[:10]:
        print(
            f"{r.get('approval_date')} | {r.get('drug_name')} | {r.get('company_name')} | "
            f"surprise={r.get('surprise')} | significance={r.get('significance')}"
        )


if __name__ == "__main__":
    main()

