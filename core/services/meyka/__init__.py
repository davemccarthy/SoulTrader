"""Meyka belief-tension engine (shadow + advisor late gate)."""

from core.services.meyka.scoring import (
    MeykaGateDecision,
    analyze_opportunity,
    append_shadow_log,
    article_text_from_parts,
    build_prompt,
    compute_derived_scores,
    default_shadow_log_path,
    evaluate_for_discovery,
    fetch_market_context,
    meyka_bizfeed_gate_enabled,
    meyka_max_prior_close_run_pct,
    meyka_sell_instructions,
    meyka_shadow_only,
    passes_discovery_score_gate,
    prior_close_run_pct,
    trading_hours_ok,
)

__all__ = [
    "MeykaGateDecision",
    "analyze_opportunity",
    "append_shadow_log",
    "article_text_from_parts",
    "build_prompt",
    "compute_derived_scores",
    "default_shadow_log_path",
    "evaluate_for_discovery",
    "fetch_market_context",
    "meyka_bizfeed_gate_enabled",
    "meyka_max_prior_close_run_pct",
    "meyka_sell_instructions",
    "meyka_shadow_only",
    "passes_discovery_score_gate",
    "prior_close_run_pct",
    "trading_hours_ok",
]
