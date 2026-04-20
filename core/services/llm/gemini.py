import logging
import os
import time
from typing import Any, List, Optional, Tuple

from django.conf import settings
from google import genai
from google.genai import types

from core.services.llm.parsing import extract_json_from_text

logger = logging.getLogger(__name__)


MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


def get_gemini_keys() -> List[Optional[str]]:
    """Build list of Gemini API keys from settings (GEMINI_KEY, GEMINI_KEY_2, ...) and env."""
    keys: List[Optional[str]] = []
    primary = getattr(settings, "GEMINI_KEY", None) or os.environ.get("GEMINI_KEY")
    if primary:
        keys.append(primary)
    for i in range(2, 10):
        key = getattr(settings, f"GEMINI_KEY_{i}", None) or os.environ.get(f"GEMINI_KEY_{i}")
        if key:
            keys.append(key)
    return keys


def ask_gemini(
    *,
    prompt: str,
    advisor_name: str,
    gemini_model_index: int,
    gemini_key_index: int,
    timeout: float = 120.0,
    use_search: bool = False,
) -> Tuple[Optional[str], Optional[Any], int, int]:
    """
    Call Gemini API with retry over models and round-robin API keys.

    Returns:
        (model, parsed_results, next_gemini_model_index, next_gemini_key_index)
    """
    keys = get_gemini_keys()
    if not keys:
        logger.warning("No GEMINI_KEY (or GEMINI_KEY_2, ...) configured for ask_gemini")
        return None, None, gemini_model_index, gemini_key_index

    model_index = gemini_model_index % len(MODELS)
    key_index = gemini_key_index

    for attempt in range(len(MODELS)):
        model = MODELS[model_index]
        try:
            key = keys[key_index % len(keys)]
            key_index += 1
            logger.info("%s using %s", advisor_name, model)

            client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(timeout=int(timeout * 1000)),
            )

            if use_search:
                config = types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.0,
                    top_p=1.0,
                )
            else:
                config = types.GenerateContentConfig(
                    temperature=0.0,
                    top_p=1.0,
                )

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            response_text = getattr(response, "text", None) if response else None
            if not response_text:
                logger.warning("No text in Gemini response for %s", advisor_name)
                return None, None, model_index, key_index

            results = extract_json_from_text(response_text)
            if not results:
                logger.warning("Cannot parse response for %s", advisor_name)
                return None, None, model_index, key_index

            time.sleep(1)
            return model, results, model_index, key_index

        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str and "RESOURCE_EXHAUSTED" in err_str:
                short_error = "429 RESOURCE_EXHAUSTED"
            elif "403" in err_str and "PERMISSION_DENIED" in err_str:
                short_error = "403 PERMISSION_DENIED"
            elif "429" in err_str:
                short_error = "429 (quota exceeded)"
            elif "403" in err_str:
                short_error = "403 (permission denied)"
            else:
                short_error = err_str[:80] + ("..." if len(err_str) > 80 else "")

            logger.warning(
                "Attempt %s: %s %s for %s. Trying next model.",
                attempt + 1,
                model,
                short_error,
                advisor_name,
            )
            model_index = (model_index + 1) % len(MODELS)

    logger.error("All Gemini models exhausted for %s", advisor_name)
    return None, None, model_index, key_index
