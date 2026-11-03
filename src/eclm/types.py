"""Shared data types. Field names and order are contracts (ARCHITECTURE §5)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

Dataset = Literal["gsm8k", "mmlu"]
Where = Literal["edge", "cloud"]


@dataclass(frozen=True)
class Question:
    qid: str
    dataset: Dataset
    question: str
    gold: str
    choices: tuple[str, ...] = ()
    subject: str = ""


@dataclass(frozen=True)
class Generation:
    """One model call."""

    text: str
    ttft_s: float
    total_s: float
    output_tokens: int
    tokens_per_s: float
    model: str
    ran_on: Where
    peak_mem_gb: float | None = None


@dataclass(frozen=True)
class Decision:
    """Router output."""

    route: Where
    stage: int | None
    difficulty: float
    agreement: int | None
    edge_generations: tuple[Generation, ...]
    edge_answer: str | None


@dataclass(frozen=True)
class Result:
    """One CSV row. Field order = CSV column order. Not-applicable values are ''."""

    run_id: str
    strategy: str
    edge_model: str
    cloud_model: str
    network: str
    dataset: str
    qid: str
    ran_on: str
    router_stage: str
    difficulty: str
    agreement: str
    edge_calls: int
    cloud_calls: int
    ttft_s: float
    total_s: float
    output_tokens: int
    tokens_per_s: float
    peak_mem_gb: str
    predicted: str
    gold: str
    correct: bool
    parse_failed: bool
    error: str
    timestamp: str


RESULT_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(Result))
