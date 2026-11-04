"""Cloud client: Groq through the OpenAI-compatible API, streaming (ADR 0002, ARCHITECTURE §6/§7)."""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal, TypeVar, cast

import httpx
import openai

from eclm.clients.base import ClientError, ClientUnavailable
from eclm.config import Settings
from eclm.types import Generation

log = logging.getLogger("eclm.clients.cloud")

T = TypeVar("T")
_RETRYABLE = (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)  # 429, 5xx, network


def _describe(exc: Exception) -> str:
    """Short error text without request details (the API key is only ever in a header, never in these)."""
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__}" + (f" (HTTP {status})" if status else "")


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
        self.key_env = settings.cloud.api_key_env
        self.max_retries = settings.cloud.max_retries
        self.seed = settings.seed
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
        """Retry 429/5xx/connection errors after 1, 2, 4, 8, 16 s (max_retries retries), then ClientError."""
        attempt = 0
        while True:
            try:
                return call()
            except _RETRYABLE as exc:
                if attempt >= self.max_retries:
                    raise ClientError(f"Groq {what} failed after {attempt + 1} attempts: {_describe(exc)}") from exc
                delay = float(2**attempt)
                log.warning("Groq %s: %s, retrying in %.0f s", what, _describe(exc), delay)
                self._sleep(delay)
                attempt += 1
            except openai.APIError as exc:  # 4xx other than 429, bad responses: not worth retrying
                raise ClientError(f"Groq {what} failed: {_describe(exc)}: {getattr(exc, 'message', '')}") from exc

    def check(self) -> None:
        if self._api is None:
            raise ClientUnavailable(f"Set {self.key_env} in .env (get a key at console.groq.com/keys)")
        api = self._api
        try:
            ids = self._with_backoff("model list", lambda: [m.id for m in api.models.list()])
        except ClientError as exc:
            cause = exc.__cause__
            if isinstance(cause, openai.AuthenticationError):
                raise ClientUnavailable("Groq rejected the API key") from exc
            if isinstance(cause, openai.APIConnectionError):
                raise ClientUnavailable(f"Cannot reach Groq at {self.base_url}: {_describe(cause)}") from exc
            raise ClientUnavailable(str(exc)) from exc
        if self.name not in ids:
            raise ClientUnavailable(f"Model {self.name} is not available on Groq. Check cloud.model in config.yaml.")

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
