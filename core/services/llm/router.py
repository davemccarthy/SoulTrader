"""LLM routing: Gemini first, DeepSeek fallback."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from core.services.llm.deepseek import ask_deepseek
from core.services.llm.gemini import ask_gemini

logger = logging.getLogger(__name__)


def ask_llm(
    prompt: str,
    *,
    advisor_name: str = "llm",
    use_search: bool = False,
    timeout: float = 120.0,
    gemini_model_index: int = 0,
    gemini_key_index: int = 0,
) -> Tuple[Optional[str], Optional[Any], int, int]:
    """
    Call Gemini (optional search grounding), then DeepSeek on failure.

    Returns:
        (model, parsed_dict, next_gemini_model_index, next_gemini_key_index)
        or (None, None, next_gemini_model_index, next_gemini_key_index) if both fail.
    """
    model, results, next_model_idx, next_key_idx = ask_gemini(
        prompt=prompt,
        advisor_name=advisor_name,
        gemini_model_index=gemini_model_index,
        gemini_key_index=gemini_key_index,
        timeout=timeout,
        use_search=use_search,
    )

    if results:
        return model, results, next_model_idx, next_key_idx

    logger.info("%s: Gemini unavailable; falling back to DeepSeek", advisor_name)
    model, results = ask_deepseek(
        prompt=prompt,
        advisor_name=advisor_name,
        timeout=timeout,
    )
    return model, results, next_model_idx, next_key_idx
