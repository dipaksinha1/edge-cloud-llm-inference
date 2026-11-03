"""Inference strategies: where each question runs (ARCHITECTURE §5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from eclm.clients.base import ModelClient
from eclm.config import Settings
from eclm.prompts import build_prompt
from eclm.types import Generation, Question


class Strategy(ABC):
    name: str = ""

    def __init__(self, settings: Settings, mock: bool = False) -> None:
        self.settings = settings
        self.mock = mock

    def prompt(self, q: Question) -> str:
        return build_prompt(q, mock=self.mock)

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
        gen = self.edge.generate(
            self.prompt(q),
            temperature=self.settings.generation.temperature,
            max_tokens=self.settings.generation.max_tokens,
        )
        return gen.text, [gen], {}
