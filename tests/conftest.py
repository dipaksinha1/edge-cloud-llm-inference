from __future__ import annotations

import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp working directory containing a copy of the repo's config.yaml."""
    shutil.copy(REPO / "config.yaml", tmp_path / "config.yaml")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _no_dataset_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never hit the network: the datasets loader fails unless a test monkeypatches its own."""
    from eclm import data

    def offline(path: str, config: str, split: str) -> list[dict[str, object]]:
        raise ConnectionError("network disabled in tests")

    monkeypatch.setattr(data, "_load_split", offline)


class ScriptedClient:
    """Fake ModelClient that replays a script, one entry per generate() call (reused by the hybrid and random tests).

    Each entry is an answer string (-> text "Answer: <x>") or a ClientError instance to raise.
    `timings` gives (ttft_s, total_s) per call (default (0.1, 1.0)). Calls are recorded in `calls`.
    """

    def __init__(self, script: list[object], *, ran_on: str = "edge", name: str = "scripted",
                 timings: list[tuple[float, float]] | None = None, tokens: int = 10) -> None:
        from typing import Any

        self.script: list[Any] = list(script)
        self.ran_on = ran_on
        self.name = name
        self.timings = list(timings) if timings is not None else None
        self.tokens = tokens
        self.calls: list[tuple[str, float]] = []

    def check(self) -> None:
        return None

    def generate(self, prompt: str, *, temperature: float, max_tokens: int):  # type: ignore[no-untyped-def]
        from eclm.types import Generation

        i = len(self.calls)
        self.calls.append((prompt, temperature))
        item = self.script[i]
        if isinstance(item, Exception):
            raise item
        ttft, total = self.timings[i] if self.timings is not None else (0.1, 1.0)
        return Generation(f"Some reasoning.\nAnswer: {item}", ttft, total, self.tokens + i,
                          float(self.tokens + i), self.name, self.ran_on)  # type: ignore[arg-type]


class StubRouterModel:
    """Stands in for RouterModel: predict_hard returns a fixed difficulty."""

    def __init__(self, difficulty: float, edge_model: str = "scripted") -> None:
        self.difficulty = difficulty
        self.edge_model = edge_model

    def predict_hard(self, q: object) -> float:
        return self.difficulty
