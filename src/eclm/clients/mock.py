"""Deterministic fake model for tests and --mock runs (no network, no Ollama, no key)."""

from __future__ import annotations

import hashlib
import random
import re
import time
from collections import defaultdict
from typing import Literal

from eclm.types import Generation

_GOLD = re.compile(r"\[\[gold:([^\]]*)\]\]")
_LETTERS = "ABCD"


def mock_accuracy(name: str, ran_on: str) -> float:
    """Accuracy a mock model of this name gets (ARCHITECTURE §5)."""
    if ran_on == "cloud":
        return 0.9
    lowered = name.lower()
    if "0.5b" in lowered:
        return 0.45
    if "1b" in lowered:
        return 0.55
    return 0.65


def _rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _distractors(gold: str) -> tuple[str, str]:
    if gold in _LETTERS and len(gold) == 1:
        i = _LETTERS.index(gold)
        return _LETTERS[(i + 1) % 4], _LETTERS[(i + 2) % 4]
    try:
        n = float(gold)
    except ValueError:
        return "0", "1"
    return _num_text(n + 1), _num_text(n * 2 if n != 0 else 2)


def _num_text(v: float) -> str:
    return str(int(v)) if v == int(v) else str(v)


class MockClient:
    """Answers from the hidden gold marker added by build_prompt(q, mock=True).

    temperature == 0: the answer depends only on (prompt, seed) and is the same on every call.
    temperature > 0: the k-th call for the same prompt uses seed + k, so samples can disagree.
    Wrong answers come from two fixed distractors, so wrong answers can also agree.
    """

    def __init__(
        self,
        name: str,
        ran_on: Literal["edge", "cloud"],
        accuracy: float,
        seed: int,
        latency_s: float = 0.0,
    ) -> None:
        self.name = name
        self.ran_on: Literal["edge", "cloud"] = ran_on
        self.accuracy = accuracy
        self.seed = seed
        self.latency_s = latency_s
        self._calls: defaultdict[str, int] = defaultdict(int)

    def check(self) -> None:
        return None

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> Generation:
        if temperature > 0:
            k = self._calls[prompt]
            self._calls[prompt] += 1
            rng = _rng(self.seed + k, temperature, self.name, prompt)
        else:
            rng = _rng(self.seed, self.name, prompt)

        match = _GOLD.search(prompt)
        if match is None:
            answer_line = "I am not sure."
        else:
            gold = match.group(1)
            if rng.random() < self.accuracy:
                answer = gold
            else:
                answer = rng.choice(_distractors(gold))
            answer_line = f"Answer: {answer}"
        text = f"Let me think step by step about this.\n{answer_line}"

        tokens = min(max_tokens, 20 + rng.randint(0, 60))
        tps = 30.0 if self.ran_on == "edge" else 150.0
        ttft = (0.05 if self.ran_on == "edge" else 0.2) + rng.random() * 0.1
        total = ttft + tokens / tps
        if self.latency_s > 0:
            time.sleep(self.latency_s)
        return Generation(
            text=text,
            ttft_s=round(ttft, 6),
            total_s=round(total, 6),
            output_tokens=tokens,
            tokens_per_s=tps,
            model=self.name,
            ran_on=self.ran_on,
        )
