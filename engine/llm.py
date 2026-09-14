"""Single entry point for every LLM call in MeetingMind v2."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

_API_KEY = os.getenv("GEMINI_API_KEY")
if not _API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add GEMINI_API_KEY to your .env file."
    )

MODEL = "gemini-2.5-flash"
MIN_API_GAP_SECONDS = 7.0
_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "llm_cache.json"

_client = genai.Client(
    api_key=_API_KEY,
    http_options=types.HttpOptions(
        # Our module owns retry policy; disable the SDK's built-in retries.
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

_call_count = 0
_last_call_time: float | None = None

if _CACHE_PATH.exists():
    with _CACHE_PATH.open("r", encoding="utf-8") as f:
        _cache: dict[str, str] = json.load(f)
else:
    _cache = {}


def get_call_count() -> int:
    """Return how many real API calls this process has made."""
    return _call_count


def complete(
    prompt: str,
    system: str | None = None,
    json_mode: bool = False,
) -> str:
    """Complete a prompt, using disk cache and rate limiting."""
    cache_key = _make_cache_key(MODEL, system, prompt, json_mode)
    if cache_key in _cache:
        return _cache[cache_key]

    text = _call_api_with_retries(prompt, system, json_mode)
    if json_mode:
        text = _strip_markdown_fences(text)

    _cache[cache_key] = text
    _write_cache()
    return text


def _make_cache_key(
    model: str,
    system: str | None,
    prompt: str,
    json_mode: bool,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "system": system,
            "prompt": prompt,
            "json_mode": json_mode,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_cache() -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False, indent=2)


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return cleaned


def _is_rate_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, errors.APIError) and getattr(exc, "code", None) == 429:
        return True
    message = str(exc).lower()
    return (
        "429" in message
        or "resource_exhausted" in message
        or "rate limit" in message
        or "rate_limit" in message
        or "ratelimit" in message
    )


def _wait_for_rate_limit() -> None:
    global _last_call_time
    if _last_call_time is None:
        return
    elapsed = time.monotonic() - _last_call_time
    remaining = MIN_API_GAP_SECONDS - elapsed
    if remaining > 0:
        time.sleep(remaining)


def _raw_generate(prompt: str, system: str | None, json_mode: bool) -> str:
    global _call_count, _last_call_time

    config_kwargs: dict = {}
    if system is not None:
        config_kwargs["system_instruction"] = system
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    _wait_for_rate_limit()
    _last_call_time = time.monotonic()
    _call_count += 1

    response = _client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=config,
    )
    text = response.text
    if text is None:
        raise RuntimeError("Gemini returned an empty response (response.text is None).")
    return text


def _call_api_with_retries(prompt: str, system: str | None, json_mode: bool) -> str:
    rate_limit_retries = 0
    other_retries = 0

    while True:
        try:
            return _raw_generate(prompt, system, json_mode)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                if rate_limit_retries >= 5:
                    raise
                delay = (10.0 * (2**rate_limit_retries)) + random.uniform(0.0, 1.0)
                rate_limit_retries += 1
                time.sleep(delay)
                continue

            if other_retries >= 2:
                raise
            other_retries += 1
