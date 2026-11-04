"""T015: one unrecorded edge warm-up call before a run's first question (mock mode, no network)."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml

from eclm import data, router, run, strategies
from eclm.clients import ClientError
from eclm.config import load_settings
from eclm.prompts import WARMUP_QUESTION, build_prompt
from eclm.types import Generation

EDGE = ["--strategy", "edge", "--edge-model", "qwen2.5:3b", "--mock"]
WARMUP_PROMPT = build_prompt(WARMUP_QUESTION, mock=True)


class _Recorder:
    """Wraps a real (mock) client and records every call; optionally raises on the warm-up prompt."""

    def __init__(self, inner: Any, sampling: list[bool], fail_warmup: bool) -> None:
        self.inner = inner
        self.name = inner.name
        self.ran_on = inner.ran_on
        self.sampling = sampling
        self.fail_warmup = fail_warmup
        self.calls: list[dict[str, Any]] = []

    def check(self) -> None:
        self.inner.check()

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> Generation:
        self.calls.append(
            {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens, "sampled": self.sampling[0]}
        )
        if self.fail_warmup and prompt == WARMUP_PROMPT:
            raise ClientError("Ollama timeout after 300s: ReadTimeout")
        gen: Generation = self.inner.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        return gen


@pytest.fixture
def mockdata(workdir: Path) -> Path:
    assert data.main(["--mock"]) == 0
    assert router.main(["train", "--edge-model", "qwen2.5:3b", "--mock"]) == 0
    return workdir


class _Built(dict[str, _Recorder]):
    """The clients run.py built in the latest run.main(), keyed '<kind>:<model>'."""

    fail_warmup = False


@pytest.fixture
def recorders(monkeypatch: pytest.MonkeyPatch) -> _Built:
    """Wrap every client run.py builds in a _Recorder, and track whether a MemorySampler is active."""
    built = _Built()
    sampling = [False]
    real_make = run.make_client
    real_sampler = strategies.memory_sampler

    def make(kind: str, settings: Any, *, model: str | None = None, mock: bool = False) -> Any:
        rec = _Recorder(real_make(kind, settings, model=model, mock=mock), sampling, built.fail_warmup)
        built[f"{kind}:{rec.name}"] = rec
        return rec

    @contextmanager
    def flagged(sampler: Any) -> Iterator[Any]:
        with sampler as s:
            sampling[0] = True
            try:
                yield s
            finally:
                sampling[0] = False

    monkeypatch.setattr(run, "make_client", make)
    monkeypatch.setattr(strategies, "memory_sampler", lambda settings, mock: flagged(real_sampler(settings, mock)))
    return built


def _set_edge(workdir: Path, key: str, value: Any) -> None:
    cfg = yaml.safe_load((workdir / "config.yaml").read_text())
    cfg["edge"][key] = value
    (workdir / "config.yaml").write_text(yaml.safe_dump(cfg))


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _n_test(workdir: Path) -> int:
    return len(data.load_questions(workdir / "data" / data.TEST_FILE))


def _warmups(rec: _Recorder) -> list[dict[str, Any]]:
    return [c for c in rec.calls if c["prompt"] == WARMUP_PROMPT]


# --- AC1: exactly one unrecorded, unsampled warm-up call first, for edge / hybrid / random ---------------


@pytest.mark.parametrize(
    ("argv", "csv_name"),
    [
        (EDGE, "edge__qwen2.5-3b__normal.csv"),
        (["--strategy", "hybrid", "--edge-model", "qwen2.5:3b", "--mock"], "hybrid__qwen2.5-3b__normal.csv"),
        (["--strategy", "random", "--edge-model", "qwen2.5:3b", "--ratio", "0.25", "--mock"],
         "random__qwen2.5-3b__normal.csv"),
    ],
)
def test_one_warmup_call_before_the_first_question(
    mockdata: Path, recorders: _Built, caplog: pytest.LogCaptureFixture, argv: list[str], csv_name: str
) -> None:
    caplog.set_level(logging.INFO, logger="eclm.run")
    assert run.main([*argv, "--network", "normal"]) == 0
    settings = load_settings()
    edge = recorders["edge:qwen2.5:3b"]
    first = edge.calls[0]
    assert first == {"prompt": WARMUP_PROMPT, "temperature": 0.0,
                     "max_tokens": settings.edge.warmup_max_tokens, "sampled": False}
    assert len(_warmups(edge)) == 1
    questions = data.load_questions(mockdata / "data" / data.TEST_FILE)
    test_prompts = {build_prompt(q, mock=True) for q in questions}
    assert all(c["prompt"] in test_prompts for c in edge.calls[1:])
    assert all(c["sampled"] for c in edge.calls[1:])  # the flag works: real question calls are sampled
    rows = _rows(mockdata / "results" / csv_name)
    assert [r["qid"] for r in rows] == [q.qid for q in questions]
    assert WARMUP_QUESTION.qid not in {r["qid"] for r in rows}
    assert any(r.getMessage().startswith("warm-up qwen2.5:3b done in ") for r in caplog.records)


def test_warmup_question_is_not_in_any_question_set(mockdata: Path) -> None:
    assert WARMUP_QUESTION.qid == "warmup" and WARMUP_QUESTION.dataset == "gsm8k"
    test, calibration = data.mock_questions()
    texts = {q.question for q in [*test, *calibration]}
    assert WARMUP_QUESTION.question not in texts
    assert WARMUP_QUESTION.qid not in {q.qid for q in [*test, *calibration]}


# --- AC2: no warm-up for cloud, for warmup_max_tokens <= 0, or for a complete run; a resume warms up again


def test_cloud_run_makes_no_warmup_and_spends_no_budget(mockdata: Path, recorders: _Built) -> None:
    n = _n_test(mockdata)
    cfg = yaml.safe_load((mockdata / "config.yaml").read_text())
    cfg["cloud"]["max_calls_per_run"] = n  # exactly one call per question: a warm-up call would hit the budget
    (mockdata / "config.yaml").write_text(yaml.safe_dump(cfg))
    assert run.main(["--strategy", "cloud", "--mock", "--network", "normal"]) == 0
    assert not [k for k in recorders if k.startswith("edge:")]
    cloud = next(r for k, r in recorders.items() if k.startswith("cloud:"))
    assert len(cloud.calls) == n and not _warmups(cloud)
    assert len(_rows(mockdata / "results" / "cloud__none__normal.csv")) == n


@pytest.mark.parametrize("value", [0, -1])
def test_warmup_off_when_max_tokens_not_positive(
    mockdata: Path, recorders: _Built, caplog: pytest.LogCaptureFixture, value: int
) -> None:
    _set_edge(mockdata, "warmup_max_tokens", value)
    caplog.set_level(logging.INFO, logger="eclm.run")
    assert run.main([*EDGE, "--network", "normal"]) == 0
    edge = recorders["edge:qwen2.5:3b"]
    assert not _warmups(edge) and len(edge.calls) == _n_test(mockdata)
    assert not any("warm-up" in r.getMessage() for r in caplog.records)


def test_complete_run_makes_no_call_and_resume_warms_up_again(
    mockdata: Path, recorders: _Built
) -> None:
    n = _n_test(mockdata)
    assert run.main([*EDGE, "--network", "normal", "--limit", "3"]) == 0
    assert len(_warmups(recorders["edge:qwen2.5:3b"])) == 1 and len(recorders["edge:qwen2.5:3b"].calls) == 4
    assert run.main([*EDGE, "--network", "normal"]) == 0  # resume: the model may be cold again
    resumed = recorders["edge:qwen2.5:3b"]
    assert resumed.calls[0]["prompt"] == WARMUP_PROMPT
    assert len(_warmups(resumed)) == 1 and len(resumed.calls) == 1 + (n - 3)
    assert run.main([*EDGE, "--network", "normal"]) == 0  # already complete
    assert recorders["edge:qwen2.5:3b"].calls == []
    assert len(_rows(mockdata / "results" / "edge__qwen2.5-3b__normal.csv")) == n


# --- AC3: a failing warm-up is a warning; every row is still written ------------------------------------


def test_failing_warmup_logs_a_warning_and_the_run_continues(
    mockdata: Path, recorders: _Built, caplog: pytest.LogCaptureFixture
) -> None:
    recorders.fail_warmup = True
    caplog.set_level(logging.INFO, logger="eclm.run")
    assert run.main([*EDGE, "--network", "normal"]) == 0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage() for r in warnings] == [
        "warm-up for qwen2.5:3b failed: Ollama timeout after 300s: ReadTimeout"
    ]
    rows = _rows(mockdata / "results" / "edge__qwen2.5-3b__normal.csv")
    assert len(rows) == _n_test(mockdata) and all(r["error"] == "" for r in rows)
    assert not any("warm-up qwen2.5:3b done" in r.getMessage() for r in caplog.records)


# --- AC4: --all --mock answers are the same with and without the warm-up --------------------------------


def test_all_mock_results_unchanged_by_warmup(mockdata: Path, recorders: _Built) -> None:
    assert run.main(["--all", "--network", "normal", "--mock", "--results-dir", "with_warmup"]) == 0
    warm_calls = {k: len(_warmups(r)) for k, r in recorders.items() if k.startswith("edge:")}
    # qwen2.5:3b serves the edge, hybrid and random runs: one warm-up per run
    assert warm_calls == {"edge:qwen2.5:3b": 3, "edge:qwen2.5:0.5b": 1, "edge:llama3.2:1b": 1}
    _set_edge(mockdata, "warmup_max_tokens", 0)
    assert run.main(["--all", "--network", "normal", "--mock", "--results-dir", "without_warmup"]) == 0
    with_files = sorted(p.name for p in (mockdata / "with_warmup").glob("*.csv"))
    assert with_files == sorted(p.name for p in (mockdata / "without_warmup").glob("*.csv"))
    assert len(with_files) == 6
    cols = ("qid", "ran_on", "predicted", "correct")
    for name in with_files:
        a = [tuple(r[c] for c in cols) for r in _rows(mockdata / "with_warmup" / name)]
        b = [tuple(r[c] for c in cols) for r in _rows(mockdata / "without_warmup" / name)]
        assert a == b, name
