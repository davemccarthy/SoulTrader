import logging
import os
import time
from typing import Any, Optional, Tuple

import requests

from core.services.llm.parsing import extract_json_from_text

logger = logging.getLogger(__name__)


def ask_deepseek(
    *,
    prompt: str,
    advisor_name: str,
    model: str = "deepseek-chat",
    timeout: float = 120.0,
) -> Tuple[Optional[str], Optional[Any]]:
    """
    Call DeepSeek OpenAI-compatible /v1/chat/completions over HTTP.

    Environment:
        DEEPSEEK_API_KEY (required)
        DEEPSEEK_BASE_URL (optional, default: https://api.deepseek.com)

    Returns:
        (model, parsed_dict) or (None, None) on failure.
    """
    api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    base_url = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip().rstrip("/")

    if not api_key:
        logger.warning("ask_deepseek: missing DEEPSEEK_API_KEY for %s", advisor_name)
        return None, None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": False,
    }
    url = f"{base_url}/v1/chat/completions"

    try:
        logger.info("%s ask_deepseek model=%s", advisor_name, model)
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        snippet = (
            (exc.response.text[:500] + "...")
            if exc.response is not None and len(exc.response.text) > 500
            else (exc.response.text if exc.response is not None else "")
        )
        logger.warning("ask_deepseek HTTP %s for %s: %s", status, advisor_name, snippet)
        return None, None
    except requests.RequestException as exc:
        logger.warning("ask_deepseek request error for %s: %s", advisor_name, exc)
        return None, None

    try:
        data = response.json()
    except ValueError:
        logger.warning("ask_deepseek: invalid JSON body for %s", advisor_name)
        return None, None

    try:
        response_text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logger.warning("ask_deepseek: missing choices[0].message.content for %s", advisor_name)
        return None, None

    if not response_text:
        logger.warning("ask_deepseek: empty content for %s", advisor_name)
        return None, None

    results = extract_json_from_text(response_text)
    if not results:
        logger.warning("ask_deepseek: cannot parse JSON for %s", advisor_name)
        return None, None

    time.sleep(1)
    return model, results
