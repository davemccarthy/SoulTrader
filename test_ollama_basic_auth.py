#!/usr/bin/env python3
"""
Quick connectivity/auth test for an Ollama endpoint behind Nginx Basic Auth.

Usage:
  export OLLAMA_USERNAME="dave"
  export OLLAMA_PASSWORD="your-password"
  python test_ollama_basic_auth.py
  python test_ollama_basic_auth.py --prompt-file /path/to/prompt.txt

Optional env vars:
  OLLAMA_HOST   (default: https://llm.mi.www.ro)
  OLLAMA_MODEL  (default: ministral-3:latest)
  OLLAMA_PROMPT_FILE (path to text file containing prompt)
  OLLAMA_PROMPT (default: Explain Basic Authentication in 3 simple sentences)
  OLLAMA_TIMEOUT (default: 300)
"""

from __future__ import annotations

import argparse
import base64
import os
import sys

import requests


def build_auth_header(username: str, password: str) -> str:
    credentials = f"{username}:{password}"
    token = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    return f"Basic {token}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Ollama /api/generate with Basic Auth."
    )
    parser.add_argument(
        "--prompt-file",
        help="Path to a text file containing the prompt. Overrides env vars.",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt text. Used when --prompt-file is not provided. Overrides env vars.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Request timeout in seconds. Overrides OLLAMA_TIMEOUT.",
    )
    return parser.parse_args()


def read_prompt_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    except OSError as exc:
        raise RuntimeError(f"Could not read prompt file '{path}': {exc}") from exc
    if not prompt:
        raise RuntimeError(f"Prompt file '{path}' is empty.")
    return prompt


def main() -> int:
    args = parse_args()
    username = os.getenv("OLLAMA_USERNAME")
    password = os.getenv("OLLAMA_PASSWORD")
    host = os.getenv("OLLAMA_HOST", "https://llm.mi.www.ro").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "ministral-3:latest")
    env_prompt_file = os.getenv("OLLAMA_PROMPT_FILE")
    env_prompt = os.getenv(
        "OLLAMA_PROMPT", "Explain Basic Authentication in 3 simple sentences"
    )
    env_timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))

    if not username or not password:
        print("ERROR: Set OLLAMA_USERNAME and OLLAMA_PASSWORD environment variables.")
        return 2

    prompt_file = args.prompt_file or env_prompt_file
    prompt = args.prompt if args.prompt is not None else env_prompt
    timeout = args.timeout if args.timeout is not None else env_timeout

    if prompt_file:
        try:
            prompt = read_prompt_file(prompt_file)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 2

    headers = {
        "Authorization": build_auth_header(username, password),
        "Content-Type": "application/json",
    }
    payload = {"model": model, "prompt": prompt, "stream": False}
    url = f"{host}/api/generate"

    print(f"Testing Ollama endpoint: {url}")
    print(f"Model: {model}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        print(f"HTTP ERROR: status={status}")
        if exc.response is not None:
            print(exc.response.text[:1000])
        return 1
    except requests.RequestException as exc:
        print(f"REQUEST ERROR: {exc}")
        return 1

    data = response.json()
    text = data.get("response")
    if not text:
        print("Request succeeded but no 'response' field found in JSON payload.")
        print(data)
        return 1

    print("\nSUCCESS: Received response from model.\n")
    print(text.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
