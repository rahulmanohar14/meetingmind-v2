"""Single entry point for every LLM call in MeetingMind v2 (Groq)."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import APIStatusError, Groq, NotFoundError, RateLimitError

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip() or "openai/gpt-oss-20b"
MIN_API_GAP_SECONDS = 1.5
# gpt-oss models spend tokens on an internal reasoning channel before emitting
# content. At 2048 the larger extraction batches ran out mid-reasoning and came
# back with empty content, which Groq then rejected as invalid JSON.
#
# Do not raise this casually. Groq charges a request against the tokens-per-
# minute allowance as prompt + max_tokens, not prompt + actual completion, so
# this value is spent on every call whether or not the model uses it. Callers
# that need a different ceiling pass max_tokens to complete() instead.
MAX_TOKENS = 4096
_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "llm_cache.json"


def _load_api_keys() -> list[str]:
    raw = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY") or ""
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise RuntimeError(
            "GROQ_API_KEYS is not set. Add one or more Groq keys to .env, e.g.\n"
            "GROQ_API_KEYS=key1,key2,key3"
        )
    return keys


_API_KEYS = _load_api_keys()
_key_index = 0

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
    model: str | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Complete a prompt, using disk cache and rate limiting.

    `model` overrides MODEL for this call only. The cache key already includes
    the model, so responses from different models never collide. This exists so
    a caller can pick a model in code rather than through the environment; see
    scripts/generate_corpus.py.

    `max_tokens` overrides MAX_TOKENS for this call. It matters more than it
    looks: Groq bills a request against the tokens-per-minute limit as prompt
    plus max_tokens, not prompt plus what the model actually returns. A large
    reservation therefore fails as a hard 413 on a small TPM allowance even
    when the prompt is modest. It is deliberately not part of the cache key,
    being a cap on the response rather than an input to it.

    `reasoning_effort` ("low"/"medium"/"high") controls how much gpt-oss spends
    on its reasoning channel before answering. Unlike max_tokens it changes the
    response, so it IS part of the cache key. Callers that leave it unset keep
    their existing cache entries.
    """
    model = model or MODEL
    cache_key = _make_cache_key(model, system, prompt, json_mode, reasoning_effort)
    if cache_key in _cache:
        return _cache[cache_key]

    text = _call_api_with_retries(
        prompt, system, json_mode, model, max_tokens, reasoning_effort
    )
    if json_mode:
        text = _strip_markdown_fences(text)
        try:
            json.loads(text)
        except json.JSONDecodeError:
            # Never cache unparseable output for a json_mode caller: a bad
            # response would otherwise be replayed forever. Return it so the
            # caller's own repair path can run.
            print(
                "engine.llm: json_mode response was not valid JSON; not caching it",
                file=sys.stderr,
            )
            return text

    _cache[cache_key] = text
    _write_cache()
    return text


def _make_cache_key(
    model: str,
    system: str | None,
    prompt: str,
    json_mode: bool,
    reasoning_effort: str | None = None,
) -> str:
    payload_dict = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "json_mode": json_mode,
    }
    # Only added when set, so callers that never pass it keep their existing
    # cache entries rather than having the whole cache invalidated.
    if reasoning_effort is not None:
        payload_dict["reasoning_effort"] = reasoning_effort
    payload = json.dumps(payload_dict, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_cache() -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False, indent=2)


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    fence = re.match(
        r"^```(?:json)?\s*\n?(.*?)\n?```\s*$",
        cleaned,
        re.DOTALL | re.IGNORECASE,
    )
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
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) == 429:
        return True
    message = str(exc).lower()
    return (
        "429" in message
        or "resource_exhausted" in message
        or "rate limit" in message
        or "rate_limit" in message
        or "ratelimit" in message
        or "quota" in message
    )


def _is_key_failure(exc: BaseException) -> bool:
    """Errors that mean this API key should be skipped."""
    if _is_rate_limit_error(exc):
        return True
    if isinstance(exc, APIStatusError):
        code = getattr(exc, "status_code", None)
        if code in {401, 403, 429}:
            return True
    message = str(exc).lower()
    return (
        "invalid api key" in message
        or "authentication" in message
        or "unauthorized" in message
        or "forbidden" in message
    )


def _wait_for_rate_limit() -> None:
    global _last_call_time
    if _last_call_time is None:
        return
    elapsed = time.monotonic() - _last_call_time
    remaining = MIN_API_GAP_SECONDS - elapsed
    if remaining > 0:
        time.sleep(remaining)


def _current_key() -> str:
    return _API_KEYS[_key_index % len(_API_KEYS)]


def _advance_key() -> None:
    global _key_index
    _key_index = (_key_index + 1) % len(_API_KEYS)


def _is_json_validate_failure(exc: BaseException) -> bool:
    """Groq rejected the model's json_object output before returning it.

    gpt-oss reasoning models occasionally put everything in the reasoning
    channel and leave content empty, which fails Groq's server-side JSON
    validation with an empty failed_generation.
    """
    if isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) != 400:
        return False
    return "json_validate_failed" in str(exc)


def _raw_generate(
    prompt: str,
    system: str | None,
    json_mode: bool,
    model: str,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    use_response_format: bool = True,
) -> str:
    global _call_count, _last_call_time

    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    if json_mode:
        # Groq json_object mode requires the word "json" in the messages.
        user_content = prompt
        if "json" not in prompt.lower():
            user_content = prompt + "\n\nReturn valid JSON only."
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": prompt})

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens or MAX_TOKENS,
    }
    if json_mode and use_response_format:
        kwargs["response_format"] = {"type": "json_object"}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    _wait_for_rate_limit()
    _last_call_time = time.monotonic()
    _call_count += 1

    client = Groq(api_key=_current_key())
    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    text = message.content
    if not text and not json_mode:
        # Some Groq reasoning models occasionally leave content empty. Only
        # usable for prose: reasoning is never the JSON a json_mode caller
        # asked for, and substituting it here silently poisoned the cache with
        # 9k-character monologues.
        reasoning = getattr(message, "reasoning", None)
        if isinstance(reasoning, str) and reasoning.strip():
            text = reasoning.strip()
    if not text:
        raise RuntimeError("Groq returned an empty response (message.content is None).")
    return text


def _call_api_with_retries(
    prompt: str,
    system: str | None,
    json_mode: bool,
    model: str,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    keys_tried = 0
    other_retries = 0
    last_exc: BaseException | None = None
    use_response_format = True

    while keys_tried < len(_API_KEYS):
        try:
            return _raw_generate(
                prompt,
                system,
                json_mode,
                model,
                max_tokens,
                reasoning_effort,
                use_response_format,
            )
        except Exception as exc:
            last_exc = exc
            if json_mode and use_response_format and _is_json_validate_failure(exc):
                # Drop response_format and ask in plain text instead. The
                # prompt still demands JSON and complete() strips any fences,
                # so callers see the same contract.
                print(
                    "engine.llm: Groq rejected json_object output; "
                    "retrying without response_format",
                    file=sys.stderr,
                )
                use_response_format = False
                continue
            # Bad model id fails for every key — don't rotate through the list.
            if isinstance(exc, NotFoundError) or (
                isinstance(exc, APIStatusError)
                and getattr(exc, "status_code", None) == 404
            ):
                raise RuntimeError(
                    f"Groq model {model!r} is not available for this account. "
                    "Set GROQ_MODEL in .env to a live model "
                    "(e.g. openai/gpt-oss-20b)."
                ) from exc
            if _is_key_failure(exc):
                keys_tried += 1
                _advance_key()
                # Brief pause before trying the next key.
                time.sleep(0.5 + random.uniform(0.0, 0.5))
                continue

            if other_retries >= 2:
                raise
            other_retries += 1
            time.sleep(1.0 + random.uniform(0.0, 0.5))
            continue

    raise RuntimeError(
        f"All {len(_API_KEYS)} Groq API key(s) failed under rate/auth limits. "
        f"Last error: {last_exc}"
    )
