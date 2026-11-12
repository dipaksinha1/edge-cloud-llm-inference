"""Cloud client: an OpenAI-compatible API (OpenRouter by default), streaming (ADR 0007, ARCHITECTURE §6/§7)."""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal, TypeVar, cast
from urllib.parse import urlsplit

import httpx
import openai

from eclm.clients.base import ClientError, ClientUnavailable
from eclm.config import Settings
from eclm.types import Generation

log = logging.getLogger("eclm.clients.cloud")

T = TypeVar("T")
_RETRYABLE = (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)  # 429, 5xx, network
_RETRYABLE_CODES = (408, 429)  # as a plain status, or as the error code of a stream that began with HTTP 200
# The account can't make any call, so nothing later in the run can succeed: stop instead of writing error rows.
_ACCOUNT_ERRORS = {
    401: "rejected the API key",
    402: "refused the call: the account is out of credit or the key's spending limit is reached (HTTP 402)",
}
_PROBE_PROMPT = "Hi"
_PROBE_MAX_TOKENS = 1


def _code(exc: Exception) -> int | None:
    """The HTTP status of a failed response, or the error code sent inside a stream that began with HTTP 200."""
    for value in (getattr(exc, "status_code", None), getattr(exc, "code", None)):
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE):
        return True
    code = _code(exc)
    return code is not None and (code in _RETRYABLE_CODES or code >= 500)


def _describe(exc: Exception) -> str:
    """Short error text without request details (the API key is only ever in a header, never in these)."""
    status = getattr(exc, "status_code", None)
    if status:
        return f"{type(exc).__name__} (HTTP {status})"
    code = _code(exc)
    return f"{type(exc).__name__}" + (f" (error {code} inside the stream)" if code else "")


class CloudClient:
    """Streams chat completions. TTFT = first chunk with non-empty content; tokens/s = usage / (total - TTFT)."""

    def __init__(
        self,
        model: str,
        settings: Settings,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.name = model
        self.ran_on: Literal["edge", "cloud"] = "cloud"
        self.base_url = settings.cloud.base_url
        self.host = urlsplit(self.base_url).hostname or self.base_url  # names the API in messages
        self.key_env = settings.cloud.api_key_env
        self.max_retries = settings.cloud.max_retries
        self.seed = settings.seed
        # OpenRouter's routing block: only these upstream providers, in this order, and only if they honour every
        # parameter we send (seed). Without it each call may be served by a different provider.
        self._extra_body: dict[str, Any] | None = None
        if settings.cloud.providers:
            self._extra_body = {
                "provider": {
                    "order": list(settings.cloud.providers),
                    "allow_fallbacks": False,
                    "require_parameters": True,
                }
            }
        self._sleep = sleep
        self._calls: defaultdict[str, int] = defaultdict(int)
        key = (os.environ.get(self.key_env) or "").strip()
        self._api: openai.OpenAI | None = None
        if key:
            self._api = openai.OpenAI(
                base_url=self.base_url,
                api_key=key,
                # openai >= 3.4x types this as httpx2.Client; it only uses build_request()/send()/close(), so the
                # httpx.Client the brief asks for (tests inject a MockTransport) works by duck typing.
                http_client=cast(Any, http),
                max_retries=0,  # back-off is ours (§7)
                timeout=settings.cloud.timeout_s,
            )

    def _with_backoff(self, what: str, call: Callable[[], T]) -> T:
        """Retry 408/429/5xx/connection errors after 1, 2, 4, 8, 16 s (max_retries retries), then ClientError.

        A 401 or 402 raises ClientUnavailable at once, which stops the run (§7).
        """
        attempt = 0
        while True:
            try:
                return call()
            except openai.APIError as exc:
                code = _code(exc)
                if code in _ACCOUNT_ERRORS:
                    raise ClientUnavailable(f"{self.host} {_ACCOUNT_ERRORS[code]}") from exc
                if not _is_retryable(exc):  # other 4xx, bad responses: not worth retrying
                    raise ClientError(
                        f"{self.host} {what} failed: {_describe(exc)}: {getattr(exc, 'message', '')}"
                    ) from exc
                if attempt >= self.max_retries:
                    raise ClientError(
                        f"{self.host} {what} failed after {attempt + 1} attempts: {_describe(exc)}"
                    ) from exc
                delay = float(2**attempt)
                log.warning("%s %s: %s, retrying in %.0f s", self.host, what, _describe(exc), delay)
                self._sleep(delay)
                attempt += 1

    def check(self) -> None:
        if self._api is None:
            raise ClientUnavailable(f"Set {self.key_env} in .env (the API key for {self.host})")
        api = self._api
        try:
            ids = self._with_backoff("model list", lambda: [m.id for m in api.models.list()])
            if self.name not in ids:
                raise ClientUnavailable(
                    f"Model {self.name} is not available on {self.host}. Check cloud.model in config.yaml."
                )
            # The model list can be public (OpenRouter's is), so only a real call proves that the key, the credit
            # and the pinned providers work. One token, not recorded anywhere.
            self._with_backoff(
                "test call", lambda: self._stream_once(_PROBE_PROMPT, 0.0, _PROBE_MAX_TOKENS, self.seed)
            )
        except ClientError as exc:
            cause = exc.__cause__
            if isinstance(cause, openai.APIConnectionError):
                raise ClientUnavailable(f"Cannot reach {self.base_url}: {_describe(cause)}") from exc
            raise ClientUnavailable(str(exc)) from exc

    def _seed_for(self, prompt: str, temperature: float) -> int:
        """settings.seed at temperature 0; seed + k for the k-th sampled call of the same prompt (§6)."""
        if temperature <= 0:
            return self.seed
        k = self._calls[prompt]
        self._calls[prompt] += 1
        return self.seed + k

    def _stream_once(self, prompt: str, temperature: float, max_tokens: int, seed: int) -> Generation:
        assert self._api is not None
        parts: list[str] = []
        ttft: float | None = None
        usage_tokens: int | None = None
        start = time.perf_counter()
        stream = self._api.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=self._extra_body,
        )
        for chunk in stream:
            if chunk.choices:
                text = chunk.choices[0].delta.content or ""
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    parts.append(text)
            if chunk.usage is not None and chunk.usage.completion_tokens is not None:
                usage_tokens = int(chunk.usage.completion_tokens)
        total = time.perf_counter() - start
        if ttft is None:
            ttft = total
        # §6: usage.completion_tokens; if the provider sent no usage, count content chunks instead.
        tokens = usage_tokens if usage_tokens is not None else len(parts)
        gen_time = total - ttft
        return Generation(
            text="".join(parts),
            ttft_s=ttft,
            total_s=total,
            output_tokens=tokens,
            tokens_per_s=tokens / gen_time if gen_time > 0 else 0.0,
            model=self.name,
            ran_on="cloud",
        )

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> Generation:
        """Timing covers the successful attempt only; back-off sleeps are rate limiting, not inference latency."""
        if self._api is None:
            raise ClientError(f"{self.key_env} is not set")
        seed = self._seed_for(prompt, temperature)
        return self._with_backoff("completion", lambda: self._stream_once(prompt, temperature, max_tokens, seed))
