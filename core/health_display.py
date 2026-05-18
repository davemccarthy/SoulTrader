"""Shared health-check serialization for API and web templates."""

from __future__ import annotations

from typing import Any


def format_health_score(score: float | None) -> str:
    if score is None:
        return "—"
    if abs(score) < 1e-9:
        return "AVOID"
    return f"{score:.1f}"


def health_record_payload(health) -> dict[str, Any]:
    """JSON shape aligned with iOS `HealthHistoryRecord` (API)."""
    ctx = _health_record_base(health)
    ctx["created"] = health.created.isoformat() if health.created else None
    return ctx


def health_record_template_context(health) -> dict[str, Any]:
    """Template context for discovery health card (web advisory)."""
    ctx = _health_record_base(health)
    ctx["created"] = health.created
    if ctx["render"] == "edgar":
        ctx.update(_edgar_template_extras(ctx))
    return ctx


def _health_record_base(health) -> dict[str, Any]:
    meta = health.meta or {}
    if not isinstance(meta, dict):
        meta = {}
    overlay = meta.get("overlay") if isinstance(meta.get("overlay"), dict) else {}
    ex99 = meta.get("ex99") if isinstance(meta.get("ex99"), dict) else {}
    media = meta.get("media") if isinstance(meta.get("media"), dict) else {}
    bonuses = meta.get("bonuses") if isinstance(meta.get("bonuses"), list) else []
    penalties = meta.get("penalties") if isinstance(meta.get("penalties"), list) else []
    justifications = ex99.get("justifications") if isinstance(ex99.get("justifications"), dict) else {}

    overlay_points = overlay.get("points")
    overlay_reasons = overlay.get("reasons") if isinstance(overlay.get("reasons"), list) else []

    render_kind = meta.get("render") or "advisor"
    has_edgar_payload = bool(ex99 or media or bonuses or penalties)

    score = float(health.score)
    piotroski = meta.get("piotroski")
    return {
        "id": health.id,
        "score": score,
        "score_display": format_health_score(score),
        "piotroski_display": _piotroski_display(piotroski),
        "render": render_kind if render_kind == "edgar" and has_edgar_payload else "advisor",
        "confidence_score": meta.get("confidence_score"),
        "health_score": meta.get("health_score"),
        "valuation_score": meta.get("valuation_score"),
        "piotroski": piotroski,
        "altman_z": meta.get("altman_z"),
        "gemini_weight": meta.get("gemini_weight"),
        "gemini_rec": meta.get("gemini_recommendation"),
        "gemini_explanation": meta.get("gemini_explanation"),
        "overlay_points": overlay_points,
        "overlay_points_label": _overlay_points_label(overlay_points),
        "overlay_reasons": overlay_reasons,
        "has_gemini": any(
            meta.get(k) is not None
            for k in ("gemini_weight", "gemini_recommendation", "gemini_explanation")
        ),
        "ex99": ex99,
        "media": media,
        "bonuses": bonuses,
        "penalties": penalties,
        "justifications": justifications,
    }


def _edgar_template_extras(ctx: dict[str, Any]) -> dict[str, Any]:
    ex99 = ctx.get("ex99") or {}
    media = ctx.get("media") or {}
    justifications = ctx.get("justifications") or {}

    justification_rows = []
    for key, label in (
        ("past_performance", "Past Performance"),
        ("guidance", "Guidance"),
        ("expectation", "Expectation"),
        ("market_reaction", "Market Reaction"),
    ):
        text = justifications.get(key)
        if text:
            justification_rows.append((label, text))

    ex99_rows = []
    for label, key in (
        ("Expectation", "expectation"),
        ("Guidance", "guidance"),
        ("Market Reaction", "market_reaction"),
        ("Past Performance", "past_performance"),
    ):
        value = ex99.get(key)
        if value:
            ex99_rows.append((label, value))

    media_rows = []
    for label, key in (
        ("Sentiment", "sentiment"),
        ("EPS", "eps"),
        ("Revenue", "revenue"),
        ("Broker", "broker"),
    ):
        value = media.get(key)
        if value is not None and str(value).strip():
            media_rows.append((label, value))

    media_lists = []
    headlines = media.get("headlines") if isinstance(media.get("headlines"), list) else []
    red_flags = media.get("red_flags") if isinstance(media.get("red_flags"), list) else []
    if headlines:
        media_lists.append(("Headlines", headlines[:4]))
    if red_flags:
        media_lists.append(("Red Flags", red_flags[:4]))

    summary = (media.get("summary") or "").strip()
    media_has_content = bool(
        summary
        or media_rows
        or media_lists
    )

    return {
        "justification_rows": justification_rows,
        "ex99_rows": ex99_rows,
        "media_rows": media_rows,
        "media_lists": media_lists,
        "media_has_content": media_has_content,
        "media": {**media, "summary": summary} if summary else media,
    }


def _piotroski_display(piotroski) -> str | None:
    if piotroski is None:
        return None
    text = str(piotroski).strip()
    if not text or text == "—":
        return None
    if "/" in text:
        return text
    return f"{text}/4"


def _overlay_points_label(overlay_points) -> str | None:
    if overlay_points is None:
        return None
    try:
        pts = float(overlay_points)
    except (TypeError, ValueError):
        return str(overlay_points)
    if pts >= 0:
        return f"+{pts:.1f}"
    return f"{pts:.1f}"
