"""Inference strategies: where each question runs (ARCHITECTURE §5)."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any

from eclm.clients.base import ModelClient
from eclm.config import Settings
from eclm.metrics import MemorySampler
from eclm.prompts import build_prompt
from eclm.types import Generation, Question


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
        gen = self.edge_call(
            self.edge,
            self.prompt(q),
            temperature=self.settings.generation.temperature,
            max_tokens=self.settings.generation.max_tokens,
        )
        return gen.text, [gen], {}
