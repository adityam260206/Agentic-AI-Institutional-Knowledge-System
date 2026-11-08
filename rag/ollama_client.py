"""
rag/ollama_client.py
---------------------
Thin HTTP client for the Ollama REST API.

Responsibility boundary:
    This module knows about Ollama's wire format (NDJSON streaming,
    option names, error codes) and nothing else.  RAGEngine owns the
    BuiltPrompt → RAGResponse mapping; this client just moves bytes.

Modular swap note:
    Replace OllamaClient with any class that implements the same
    .chat() / .chat_stream() / .health_check() interface to switch
    models or providers without touching RAGEngine.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Iterator, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class OllamaConfig(BaseModel):
    """All parameters controlling the Ollama connection and generation."""

    # OLLAMA_BASE_URL lets Docker Compose point the app at the `ollama` service.
    base_url:       str   = Field(default_factory=lambda: os.environ.get(
                                      "OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"))
    model:          str   = Field(default="qwen3:8b")

    # Reasoning toggle. qwen3 is a hybrid reasoning model: by default newer
    # Ollama builds stream the chain-of-thought into a separate `thinking`
    # field and leave `message.content` empty until reasoning finishes — which
    # (a) yields zero text chunks on the streaming path and (b) blows past the
    # read timeout on the non-streaming path because the whole token budget is
    # spent thinking. We don't surface CoT to users, so disable it: content is
    # then emitted directly and immediately.
    think:          bool  = Field(default=False,
                                  description="Ollama `think` flag. False disables qwen3 reasoning.")

    # Generation options (mapped to Ollama's /api/chat `options` dict)
    temperature:    float = Field(default=0.7,  ge=0.0, le=2.0)
    max_tokens:     int   = Field(default=1024, ge=1,   le=32768,
                                  description="Ollama option: num_predict")
    top_p:          float = Field(default=0.9,  ge=0.0, le=1.0)
    repeat_penalty: float = Field(default=1.1,  ge=0.0, le=2.0)

    # HTTP / retry
    timeout:      float = Field(default=120.0, gt=0,
                                description="Read timeout in seconds (generation can be slow).")
    max_retries:  int   = Field(default=3, ge=0, le=10)
    retry_delay:  float = Field(default=1.0, gt=0,
                                description="Base delay (seconds) before first retry.")

    def to_options(self, overrides: Optional[dict] = None) -> dict:
        """Returns the Ollama `options` dict, with any per-call overrides applied."""
        opts = {
            "temperature":    self.temperature,
            "num_predict":    self.max_tokens,
            "top_p":          self.top_p,
            "repeat_penalty": self.repeat_penalty,
        }
        if overrides:
            # Map public names to Ollama names
            if "max_tokens" in overrides:
                overrides["num_predict"] = overrides.pop("max_tokens")
            opts.update(overrides)
        return opts


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OllamaError(RuntimeError):
    """Base for all Ollama client errors."""

class OllamaConnectionError(OllamaError):
    """Could not reach the Ollama server."""

class OllamaTimeoutError(OllamaError):
    """Request exceeded the configured timeout."""

class OllamaModelNotFoundError(OllamaError):
    """Model is not pulled / not available on this Ollama instance."""

class OllamaResponseError(OllamaError):
    """Unexpected or malformed response from the Ollama server."""


# ---------------------------------------------------------------------------
# Pure parsing helpers  (independently testable, no httpx dependency)
# ---------------------------------------------------------------------------

def _strip_thinking_tags(text: str) -> str:
    """
    Removes Qwen3 chain-of-thought blocks from the answer.

    Qwen3 may wrap its reasoning in <think>...</think> before the final
    answer.  When the model emits only a thinking block and no trailing
    text, `content` arrives as an empty string after stripping.  We
    preserve whatever comes after the closing tag so callers always
    receive the user-facing answer, not the internal reasoning.
    """
    import re
    # Strip all <think>...</think> blocks (including multiline, non-greedy)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _parse_chat_response(body: dict) -> dict:
    """
    Extracts structured fields from a non-streaming /api/chat response body.

    Returns a dict with keys:
        answer, prompt_tokens, completion_tokens, finish_reason,
        generation_time_ms, model_name
    """
    try:
        raw_answer = body["message"]["content"]
        model      = body.get("model", "unknown")
        reason     = body.get("done_reason", "stop")
        p_tokens   = body.get("prompt_eval_count", 0)
        c_tokens   = body.get("eval_count", 0)
        # eval_duration is nanoseconds
        gen_ns     = body.get("eval_duration", 0)
        gen_ms     = gen_ns / 1_000_000
    except (KeyError, TypeError) as exc:
        raise OllamaResponseError(f"Unexpected response structure: {exc}. body={body!r}") from exc

    answer = _strip_thinking_tags(raw_answer)

    return {
        "answer":              answer,
        "prompt_tokens":       int(p_tokens),
        "completion_tokens":   int(c_tokens),
        "finish_reason":       reason,
        "generation_time_ms":  round(gen_ms, 2),
        "model_name":          model,
    }


def _parse_stream_line(line: str) -> tuple[Optional[str], bool, Optional[dict]]:
    """
    Parses one NDJSON line from a streaming /api/chat response.

    Returns (text_chunk, is_done, final_stats_or_none).
        text_chunk   — incremental text content, or None on the final line
        is_done      — True on the terminal line (done=true)
        final_stats  — populated only when is_done=True; same keys as _parse_chat_response
    """
    line = line.strip()
    if not line:
        return None, False, None

    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None, False, None

    done = obj.get("done", False)

    if not done:
        # Incremental token
        text = obj.get("message", {}).get("content", "")
        return text or None, False, None

    # Terminal line — contains usage stats
    stats = {
        "prompt_tokens":      int(obj.get("prompt_eval_count", 0)),
        "completion_tokens":  int(obj.get("eval_count", 0)),
        "finish_reason":      obj.get("done_reason", "stop"),
        "generation_time_ms": round(obj.get("eval_duration", 0) / 1_000_000, 2),
        "model_name":         obj.get("model", "unknown"),
    }
    return None, True, stats


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _with_retry(fn, max_retries: int, base_delay: float):
    """
    Calls fn(); retries on OllamaConnectionError / OllamaTimeoutError only.
    Uses exponential backoff with ±10 % jitter.
    """
    retryable = (OllamaConnectionError, OllamaTimeoutError)
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retryable as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            sleep_time = base_delay * (2 ** attempt) + random.uniform(0, base_delay * 0.1)
            logger.warning(
                "[OLLAMA] %s (attempt %d/%d) — retrying in %.2fs",
                type(exc).__name__, attempt + 1, max_retries, sleep_time,
            )
            time.sleep(sleep_time)

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Thin wrapper around Ollama's /api/chat REST endpoint.

    Thread-safe: creates a new httpx.Client per request (stateless).
    Swap note: implement the same .chat() / .chat_stream() / .health_check()
    interface on a different class to target any OpenAI-compatible endpoint.
    """

    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        self._cfg = config or OllamaConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._cfg.model

    @property
    def config(self) -> OllamaConfig:
        return self._cfg

    def chat(
        self,
        messages:  list[dict],
        *,
        temperature:    Optional[float] = None,
        max_tokens:     Optional[int]   = None,
        top_p:          Optional[float] = None,
        repeat_penalty: Optional[float] = None,
    ) -> dict:
        """
        Non-streaming chat call.

        Returns the parsed dict from _parse_chat_response plus a wall-clock
        latency_ms key added by this method.

        Raises OllamaConnectionError, OllamaTimeoutError,
               OllamaModelNotFoundError, OllamaResponseError.
        """
        overrides = self._build_overrides(temperature, max_tokens, top_p, repeat_penalty)
        payload   = self._build_payload(messages, stream=False, overrides=overrides)

        def _call():
            return self._post_json(payload)

        t_start = time.perf_counter()
        body    = _with_retry(_call, self._cfg.max_retries, self._cfg.retry_delay)
        latency = (time.perf_counter() - t_start) * 1000

        result = _parse_chat_response(body)
        result["latency_ms"] = round(latency, 2)
        return result

    def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature:    Optional[float] = None,
        max_tokens:     Optional[int]   = None,
        top_p:          Optional[float] = None,
        repeat_penalty: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Streaming chat call. Yields text chunk strings as they arrive.
        Token counts are not available from this method.
        """
        overrides = self._build_overrides(temperature, max_tokens, top_p, repeat_penalty)
        payload   = self._build_payload(messages, stream=True, overrides=overrides)
        url       = f"{self._cfg.base_url}/api/chat"
        timeout   = httpx.Timeout(connect=10.0, read=self._cfg.timeout, write=10.0, pool=5.0)

        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", url, json=payload) as response:
                    self._check_status(response.status_code, url)
                    for line in response.iter_lines():
                        chunk, is_done, _ = _parse_stream_line(line)
                        if chunk:
                            yield chunk
                        if is_done:
                            break

        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self._cfg.base_url}. "
                "Is Ollama running? Run: ollama serve"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama stream timed out after {self._cfg.timeout}s."
            ) from exc

    def health_check(self) -> bool:
        """Returns True if Ollama is reachable (GET /), False otherwise."""
        try:
            with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
                r = client.get(self._cfg.base_url)
                return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_overrides(
        self,
        temperature:    Optional[float],
        max_tokens:     Optional[int],
        top_p:          Optional[float],
        repeat_penalty: Optional[float],
    ) -> dict:
        overrides: dict = {}
        if temperature    is not None: overrides["temperature"]    = temperature
        if max_tokens     is not None: overrides["max_tokens"]     = max_tokens
        if top_p          is not None: overrides["top_p"]          = top_p
        if repeat_penalty is not None: overrides["repeat_penalty"] = repeat_penalty
        return overrides

    def _build_payload(
        self, messages: list[dict], stream: bool, overrides: dict
    ) -> dict:
        return {
            "model":    self._cfg.model,
            "messages": messages,
            "stream":   stream,
            "think":    self._cfg.think,
            "options":  self._cfg.to_options(overrides or None),
        }

    def _post_json(self, payload: dict) -> dict:
        """Executes a non-streaming POST and returns the response body as dict."""
        url     = f"{self._cfg.base_url}/api/chat"
        timeout = httpx.Timeout(connect=10.0, read=self._cfg.timeout, write=10.0, pool=5.0)

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self._cfg.base_url}. "
                "Is Ollama running? Run: ollama serve"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama did not respond within {self._cfg.timeout}s."
            ) from exc

        self._check_status(response.status_code, url)

        try:
            return response.json()
        except Exception as exc:
            raise OllamaResponseError(
                f"Ollama returned non-JSON body: {response.text[:200]}"
            ) from exc

    @staticmethod
    def _check_status(status_code: int, url: str) -> None:
        if status_code == 200:
            return
        if status_code == 404:
            raise OllamaModelNotFoundError(
                f"Model not found (HTTP 404) at {url}. "
                "Pull the model first: ollama pull qwen3:8b"
            )
        raise OllamaResponseError(f"Ollama returned HTTP {status_code} for {url}.")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_client_instance: Optional[OllamaClient] = None


def get_ollama_client(config: Optional[OllamaConfig] = None) -> OllamaClient:
    """Returns the process-level OllamaClient singleton."""
    global _client_instance
    if _client_instance is None:
        _client_instance = OllamaClient(config)
    return _client_instance


def reset_ollama_client() -> None:
    """Clears the singleton — primarily for testing."""
    global _client_instance
    _client_instance = None
