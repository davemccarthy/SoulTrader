"""Discovery serialization for API and web (structured EDGAR meta)."""

from __future__ import annotations

import re
from typing import Any

from core.health_display import _edgar_template_extras

EARNINGS_DISCOVERY_LEAD = "8-K earnings filing"


def discovery_excerpt_from_text(explanation: str | None) -> str:
    """Short discovery copy from first meaningful explanation segment."""
    if not explanation:
        return ""
    normalized = " ".join(explanation.split()).strip()
    if not normalized:
        return ""
    segments = [segment.strip() for segment in normalized.split("|") if segment.strip()]
    for segment in segments:
        if segment.startswith("http://") or segment.startswith("https://"):
            continue
        lower = segment.lower()
        if lower.startswith("article:"):
            title = segment.split(":", 1)[1].strip()
            if title:
                return title
            continue
        return segment
    return ""


def discovery_excerpt(discovery) -> str:
    """List/summary line for a discovery (holdings, advisory rows)."""
    meta = discovery.meta if isinstance(getattr(discovery, "meta", None), dict) else {}
    if meta.get("render") == "edgar":
        return str(meta.get("lead") or EARNINGS_DISCOVERY_LEAD)
    return discovery_excerpt_from_text(getattr(discovery, "explanation", None))


def _edgar_structured_from_meta(meta: dict) -> dict[str, Any] | None:
    if not isinstance(meta, dict) or meta.get("render") != "edgar":
        return None

    ex99 = meta.get("ex99") if isinstance(meta.get("ex99"), dict) else {}
    media = meta.get("media") if isinstance(meta.get("media"), dict) else {}
    bonuses = meta.get("bonuses") if isinstance(meta.get("bonuses"), list) else []
    penalties = meta.get("penalties") if isinstance(meta.get("penalties"), list) else []
    justifications = (
        ex99.get("justifications") if isinstance(ex99.get("justifications"), dict) else {}
    )
    has_payload = bool(ex99 or media or bonuses or penalties or meta.get("accession"))
    if not has_payload:
        return None

    ctx: dict[str, Any] = {
        "render": "edgar",
        "lead": meta.get("lead") or EARNINGS_DISCOVERY_LEAD,
        "accession": meta.get("accession"),
        "weight": meta.get("weight"),
        "filing_dt": meta.get("filing_dt"),
        "sec_url": meta.get("sec_url"),
        "open": meta.get("open") if isinstance(meta.get("open"), dict) else {},
        "media_gate": meta.get("media_gate"),
        "open_bucket": meta.get("open_bucket"),
        "ex99": ex99,
        "media": media,
        "bonuses": bonuses,
        "penalties": penalties,
        "justifications": justifications,
    }
    ctx.update(_edgar_template_extras(ctx))
    return ctx


def discovery_meta_api(discovery) -> dict[str, Any] | None:
    """JSON payload for iOS/web structured EDGAR discovery cards."""
    meta = discovery.meta if isinstance(getattr(discovery, "meta", None), dict) else {}
    structured = _edgar_structured_from_meta(meta)
    if not structured:
        return None
    return {
        "render": "edgar",
        "lead": structured.get("lead"),
        "accession": structured.get("accession"),
        "weight": structured.get("weight"),
        "filing_dt": structured.get("filing_dt"),
        "sec_url": structured.get("sec_url"),
        "open": structured.get("open"),
        "media_gate": structured.get("media_gate"),
        "open_bucket": structured.get("open_bucket"),
        "ex99": structured.get("ex99"),
        "media": structured.get("media"),
        "bonuses": structured.get("bonuses"),
        "penalties": structured.get("penalties"),
    }


def discovery_template_context(discovery) -> dict[str, Any] | None:
    """Template context for server-rendered EDGAR discovery partials."""
    meta = discovery.meta if isinstance(getattr(discovery, "meta", None), dict) else {}
    return _edgar_structured_from_meta(meta)


def discovery_paragraphs(explanation: str | None) -> list[dict[str, str]]:
    """Render-ready explanation blocks (legacy pipe-delimited discoveries)."""
    if not explanation:
        return []
    segments = [
        segment.strip()
        for segment in re.split(r"\s*\|\s*|\n+", explanation)
        if segment and segment.strip()
    ]
    if not segments:
        return []

    blocks: list[dict[str, str]] = []
    i = 0
    while i < len(segments):
        segment = segments[i]
        lower = segment.lower()
        if lower.startswith("article:") and i + 1 < len(segments):
            title = segment.split(":", 1)[1].strip()
            next_segment = segments[i + 1].strip()
            if title and (next_segment.startswith("http://") or next_segment.startswith("https://")):
                blocks.append({"kind": "link", "label": title, "url": next_segment})
                i += 2
                continue

        if segment.startswith("http://") or segment.startswith("https://"):
            blocks.append({"kind": "link", "label": segment, "url": segment})
        else:
            display_text = segment
            if lower.startswith("article:"):
                display_text = segment.split(":", 1)[1].strip()
            blocks.append({"kind": "text", "text": display_text})
        i += 1

    return blocks
