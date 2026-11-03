"""Inference strategies: where each question runs (ARCHITECTURE §5)."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from eclm.clients.base import ClientError, ModelClient
from eclm.config import Settings
from eclm.metrics import MemorySampler
from eclm.prompts import build_prompt
from eclm.types import Generation, Question


class StrategyError(Exception):
    """A ClientError inside answer(). Keeps the calls that did complete so the runner can record them (§7)."""

    def __init__(self, cause: Exception, generations: list[Generation], extra: dict[str, Any]) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.generations = list(generations)
        self.extra = dict(extra)


@dataclass
class StrategyContext:
    """Everything a strategy factory may need. run.py builds the clients (cloud wrapped in BudgetedClient)."""

    settings: Settings
    edge_model: str | None
    network: str
    mock: bool
    results_dir: Path
    ratio: float | None
    edge: ModelClient | None
    cloud: ModelClient | None


StrategyFactory = Callable[[StrategyContext], "Strategy"]
STRATEGIES: dict[str, StrategyFactory] = {}


def register(name: str, factory: StrategyFactory) -> None:
    """Make a strategy available to `run --strategy <name>` and `run --all`."""
    STRATEGIES[name] = factory


def _need(client: ModelClient | None, side: str, strategy: str) -> ModelClient:
    if client is None:
        raise ValueError(f"strategy {strategy!r} needs an {side} client")
    return client


class Strategy(ABC):
    name: str = ""

    def __init__(self, settings: Settings, mock: bool = False) -> None:
        self.settings = settings
        self.mock = mock

    def prompt(self, q: Question) -> str:
        return build_prompt(q, mock=self.mock)

    def sampler(self) -> MemorySampler:
        """Memory sampler for one edge call: the Ollama processes, or this process in mock mode (§5/§6)."""
        pids = [os.getpid()] if self.mock else None
        return MemorySampler(hz=self.settings.edge.memory_sample_hz, pids=pids)

    def edge_call(self, client: ModelClient, prompt: str, *, temperature: float, max_tokens: int) -> Generation:
        """One edge generation with its peak memory filled in."""
        with self.sampler() as mem:
            gen = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        return replace(gen, peak_mem_gb=mem.peak_gb)

    def prepare(self, questions: list[Question]) -> None:
        """Optional hook called once before the run (RandomRouter overrides it)."""
        return None

    @abstractmethod
    def answer(self, q: Question) -> tuple[str, list[Generation], dict[str, Any]]:
        """Return (final text, all generations in call order, extra row fields)."""


class EdgeOnly(Strategy):
    name = "edge"

    def __init__(self, edge: ModelClient, settings: Settings, mock: bool = False) -> None:
        super().__init__(settings, mock)
        self.edge = edge

    def answer(self, q: Question) -> tuple[str, list[Generation], dict[str, Any]]:
        try:
            gen = self.edge_call(
                self.edge,
                self.prompt(q),
                temperature=self.settings.generation.temperature,
                max_tokens=self.settings.generation.max_tokens,
            )
        except ClientError as exc:
            raise StrategyError(exc, [], {}) from exc
        return gen.text, [gen], {}


class CloudOnly(Strategy):
    name = "cloud"

    def __init__(self, cloud: ModelClient, settings: Settings, mock: bool = False) -> None:
        super().__init__(settings, mock)
        self.cloud = cloud

    def answer(self, q: Question) -> tuple[str, list[Generation], dict[str, Any]]:
        try:
            gen = self.cloud.generate(
                self.prompt(q),
                temperature=self.settings.generation.temperature,
                max_tokens=self.settings.generation.max_tokens,
            )
        except ClientError as exc:
            raise StrategyError(exc, [], {}) from exc
        return gen.text, [gen], {}


register("edge", lambda ctx: EdgeOnly(_need(ctx.edge, "edge", "edge"), ctx.settings, mock=ctx.mock))
register("cloud", lambda ctx: CloudOnly(_need(ctx.cloud, "cloud", "cloud"), ctx.settings, mock=ctx.mock))
