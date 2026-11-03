"""Inference strategies: where each question runs (ARCHITECTURE §5)."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from eclm.clients.base import ClientError, ModelClient
from eclm.config import Settings
from eclm.grading import extract_answer, is_correct
from eclm.metrics import MemorySampler
from eclm.prompts import build_prompt
from eclm.router import Router, RouterModel, router_path
from eclm.types import Generation, Question


class StrategyError(Exception):
    """A ClientError inside answer(). Keeps the calls that did complete so the runner can record them (§7)."""

    def __init__(self, cause: Exception, generations: list[Generation], extra: dict[str, Any]) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.generations = list(generations)
        self.extra = dict(extra)


class StrategyUnavailable(Exception):
    """A strategy can't be built (e.g. no trained router). run.py prints the message and exits 1."""


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


def memory_sampler(settings: Settings, mock: bool) -> MemorySampler:
    """Memory sampler for one edge call: the Ollama processes, or this process in mock mode (§5/§6)."""
    pids = [os.getpid()] if mock else None
    return MemorySampler(hz=settings.edge.memory_sample_hz, pids=pids)


class MeteredClient:
    """An edge client whose every generation carries the peak memory measured around it (§6)."""

    def __init__(self, inner: ModelClient, settings: Settings, mock: bool = False) -> None:
        self.inner = inner
        self.settings = settings
        self.mock = mock
        self.name = inner.name
        self.ran_on: Literal["edge", "cloud"] = inner.ran_on

    def check(self) -> None:
        self.inner.check()

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> Generation:
        with memory_sampler(self.settings, self.mock) as mem:
            gen = self.inner.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        return replace(gen, peak_mem_gb=mem.peak_gb)


class Strategy(ABC):
    name: str = ""

    def __init__(self, settings: Settings, mock: bool = False) -> None:
        self.settings = settings
        self.mock = mock

    def prompt(self, q: Question) -> str:
        return build_prompt(q, mock=self.mock)

    def sampler(self) -> MemorySampler:
        """Memory sampler for one edge call: the Ollama processes, or this process in mock mode (§5/§6)."""
        return memory_sampler(self.settings, self.mock)

    def edge_call(self, client: ModelClient, prompt: str, *, temperature: float, max_tokens: int) -> Generation:
        """One edge generation with its peak memory filled in."""
        return MeteredClient(client, self.settings, self.mock).generate(
            prompt, temperature=temperature, max_tokens=max_tokens
        )

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


class Hybrid(Strategy):
    """Two-stage router (R4, ADR 0004): stage-1 cloud, or edge samples that agree, else stage-2 cloud.

    The router's edge client should be a MeteredClient so each sample carries its peak memory.
    Extra row fields: router_stage, difficulty, agreement ("k/n"), and when kept on the edge, predicted
    (the majority) and answer_index (the first sample whose answer equals the majority; §6 timing).
    """

    name = "hybrid"

    def __init__(
        self, edge: ModelClient, cloud: ModelClient, router: Router, settings: Settings, mock: bool = False
    ) -> None:
        super().__init__(settings, mock)
        self.edge = edge
        self.cloud = cloud
        self.router = router

    def answer(self, q: Question) -> tuple[str, list[Generation], dict[str, Any]]:
        decision = self.router.decide(q)
        edge_gens = list(decision.edge_generations)
        extra: dict[str, Any] = {
            "router_stage": "" if decision.stage is None else str(decision.stage),
            "difficulty": f"{decision.difficulty:.4f}",
            "agreement": "" if decision.agreement is None else f"{decision.agreement}/{self.settings.router.samples}",
        }
        if decision.route == "edge" and decision.edge_answer is not None:
            majority = decision.edge_answer
            index = next(
                i for i, g in enumerate(edge_gens)
                if is_correct(extract_answer(g.text, q.dataset, q.choices), majority, q.dataset)
            )
            extra["predicted"] = majority
            extra["answer_index"] = index
            return edge_gens[index].text, edge_gens, extra
        try:
            gen = self.cloud.generate(
                self.prompt(q),
                temperature=self.settings.generation.temperature,
                max_tokens=self.settings.generation.max_tokens,
            )
        except ClientError as exc:
            raise StrategyError(exc, edge_gens, extra) from exc
        return gen.text, [*edge_gens, gen], extra


def _hybrid(ctx: StrategyContext) -> Hybrid:
    edge = _need(ctx.edge, "edge", "hybrid")
    cloud = _need(ctx.cloud, "cloud", "hybrid")
    model_name = ctx.edge_model or edge.name
    path = router_path(ctx.settings, model_name)
    hint = f"train it: python -m eclm.router train --edge-model {model_name}"
    if not path.is_file():
        raise StrategyUnavailable(f"no trained router for {model_name} at {path}; {hint}")
    try:
        model = RouterModel.load(path)
    except (OSError, ValueError, KeyError) as exc:
        raise StrategyUnavailable(f"can't load router {path} ({exc}); {hint}") from exc
    if model.edge_model != model_name:
        raise StrategyUnavailable(f"{path} was trained for {model.edge_model}, not {model_name}; {hint}")
    router = Router(model, MeteredClient(edge, ctx.settings, ctx.mock), ctx.settings, mock=ctx.mock)
    return Hybrid(edge, cloud, router, ctx.settings, mock=ctx.mock)


register("edge", lambda ctx: EdgeOnly(_need(ctx.edge, "edge", "edge"), ctx.settings, mock=ctx.mock))
register("cloud", lambda ctx: CloudOnly(_need(ctx.cloud, "cloud", "cloud"), ctx.settings, mock=ctx.mock))
register("hybrid", _hybrid)
