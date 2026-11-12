"""Edge client: Ollama's local HTTP API with streaming timing (ADR 0003, ARCHITECTURE §6)."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any, Literal

import httpx

from eclm.clients.base import ClientError, ClientUnavailable
from eclm.config import Settings
from eclm.types import Generation


def _same_model(wanted: str, listed: str) -> bool:
    """Ollama treats a name without a tag as `<name>:latest`."""

    def full(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    return full(wanted) == full(listed)


class OllamaClient:
    """Streams `POST {host}/api/generate`. TTFT = first chunk with non-empty `response`."""

    def __init__(self, model: str, settings: Settings, http: httpx.Client | None = None) -> None:
        self.name = model
        self.ran_on: Literal["edge", "cloud"] = "edge"
        self.host = settings.edge.host.rstrip("/")
        self.seed = settings.seed
        self.timeout = httpx.Timeout(settings.edge.timeout_s)
        self._http = http if http is not None else httpx.Client(timeout=self.timeout)
        self._calls: defaultdict[str, int] = defaultdict(int)

    def check(self) -> None:
        try:
            resp = self._http.get(f"{self.host}/api/tags", timeout=self.timeout)
        except httpx.TransportError as exc:
            raise ClientUnavailable(
                f"Ollama is not running at {self.host}. Start the Ollama app or run `ollama serve`."
            ) from exc
        if resp.status_code != 200:
            raise ClientUnavailable(f"Ollama at {self.host} answered HTTP {resp.status_code} to /api/tags")
        try:
            models = resp.json().get("models") or []
            names = {str(m.get(key)) for m in models for key in ("name", "model") if m.get(key)}
        except (ValueError, AttributeError) as exc:
            raise ClientUnavailable(f"Ollama at {self.host} returned an unexpected /api/tags response") from exc
        if not any(_same_model(self.name, n) for n in names):
            raise ClientUnavailable(f"Model {self.name} not found. Run: ollama pull {self.name}")

    def _seed_for(self, prompt: str, temperature: float) -> int:
        """settings.seed at temperature 0; seed + k for the k-th sampled call of the same prompt (§6)."""
        if temperature <= 0:
            return self.seed
        k = self._calls[prompt]
        self._calls[prompt] += 1
        return self.seed + k

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> Generation:
        options = {"temperature": temperature, "num_predict": max_tokens, "seed": self._seed_for(prompt, temperature)}
        body = {"model": self.name, "prompt": prompt, "stream": True, "options": options}
        parts: list[str] = []
        ttft: float | None = None
        final: dict[str, Any] | None = None
        start = time.perf_counter()
        try:
            with self._http.stream("POST", f"{self.host}/api/generate", json=body, timeout=self.timeout) as resp:
                if resp.status_code != 200:
                    detail = resp.read().decode("utf-8", "replace").strip()[:200]
                    raise ClientError(f"Ollama HTTP {resp.status_code} {resp.reason_phrase}: {detail}")
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError as exc:
                        raise ClientError(f"Ollama sent invalid JSON: {line[:100]!r}") from exc
                    if chunk.get("error"):
                        raise ClientError(f"Ollama error: {chunk['error']}")
                    text = chunk.get("response") or ""
                    if text:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        parts.append(text)
                    if chunk.get("done"):
                        final = chunk
                        break
        except httpx.TimeoutException as exc:
            raise ClientError(f"Ollama timeout after {self.timeout.read}s: {type(exc).__name__}") from exc
        except httpx.TransportError as exc:
            raise ClientError(f"Ollama connection failed: {type(exc).__name__}: {exc}") from exc
        total = time.perf_counter() - start
        if final is None:
            raise ClientError("Ollama stream ended without a final done chunk")
        if ttft is None:
            ttft = total
        tokens = int(final.get("eval_count") or 0)
        duration_ns = float(final.get("eval_duration") or 0)
        if duration_ns > 0:
            tps = tokens / (duration_ns / 1e9)
        else:
            tps = tokens / (total - ttft) if total > ttft else 0.0
        return Generation(
            text="".join(parts),
            ttft_s=ttft,
            total_s=total,
            output_tokens=tokens,
            tokens_per_s=tps,
            model=self.name,
            ran_on="edge",
        )
