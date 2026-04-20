import base64
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import requests
from django.conf import settings

from core.services.llm.parsing import extract_json_from_text

logger = logging.getLogger(__name__)


def ask_ollama(
    *,
    prompt: str,
    advisor_name: str,
    model: str = "qwen3:8b",
    timeout: float = 300.0,
) -> Tuple[Optional[str], Optional[Any]]:
    """
    Call Ollama /api/generate over HTTP with Basic Auth.

    Returns:
        (model, parsed_dict) or (None, None) on failure.
    """
    host = (os.getenv("OLLAMA_HOST") or "").strip().rstrip("/")
    username = os.getenv("OLLAMA_USERNAME") or ""
    password = os.getenv("OLLAMA_PASSWORD") or ""

    if not host or not username or not password:
        logger.warning(
            "ask_ollama: missing OLLAMA_HOST, OLLAMA_USERNAME, or OLLAMA_PASSWORD for %s",
            advisor_name,
        )
        return None, None

    credentials = f"{username}:{password}"
    token = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    headers = {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "prompt": prompt, "stream": False}
    url = f"{host}/api/generate"

    try:
        logger.info("%s ask_ollama model=%s", advisor_name, model)
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        snippet = (
            (exc.response.text[:500] + "...")
            if exc.response is not None and len(exc.response.text) > 500
            else (exc.response.text if exc.response is not None else "")
        )
        logger.warning("ask_ollama HTTP %s for %s: %s", status, advisor_name, snippet)
        return None, None
    except requests.RequestException as exc:
        logger.warning("ask_ollama request error for %s: %s", advisor_name, exc)
        return None, None

    try:
        data = response.json()
    except ValueError:
        logger.warning("ask_ollama: invalid JSON body for %s", advisor_name)
        return None, None

    response_text = data.get("response")
    if not response_text:
        logger.warning("ask_ollama: empty or missing 'response' for %s", advisor_name)
        return None, None

    log_flag = (os.getenv("OLLAMA_PROMPT_LOG") or "1").strip().lower()
    if log_flag not in ("0", "false", "no", "off"):
        try:
            log_path = Path(settings.BASE_DIR) / "ollama.txt"
            stamp = datetime.now().isoformat(timespec="seconds")
            block = (
                f"========== {stamp} advisor={advisor_name} model={model} ==========\n"
                f"--- PROMPT ---\n{prompt}\n"
                f"--- RESPONSE ---\n{response_text}\n\n"
            )
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(block)
        except OSError as exc:
            logger.warning("ask_ollama: could not append ollama.txt: %s", exc)

    results = extract_json_from_text(response_text)
    if not results:
        logger.warning("ask_ollama: cannot parse JSON for %s", advisor_name)
        return None, None

    time.sleep(1)
    return model, results
