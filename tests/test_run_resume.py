"""T007: resumable runner, error rows, --all, --limit and the cloud-call budget (mock mode, no network)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from eclm import data, run
from eclm.clients import BudgetedClient, BudgetExceeded, ClientError, MockClient
from eclm.config import load_settings
from eclm.results import done_qids
from eclm.strategies import STRATEGIES, CloudOnly, EdgeOnly, Strategy, StrategyContext, StrategyError
from eclm.types import RESULT_COLUMNS, Generation, Question

EDGE = ["--strategy", "edge", "--edge-model", "qwen2.5:3b", "--mock"]
CLOUD = ["--strategy", "cloud", "--mock"]


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == RESULT_COLUMNS
        return list(reader)


def _qids(workdir: Path) -> list[str]:
    return [q.qid for q in data.load_questions(workdir / "data" / data.TEST_FILE)]


def _set_config(workdir: Path, section: str, key: str, value: Any) -> None:
    cfg = yaml.safe_load((workdir / "config.yaml").read_text())
    cfg[section][key] = value
    (workdir / "config.yaml").write_text(yaml.safe_dump(cfg))


@pytest.fixture
def mockdata(workdir: Path) -> Path:
    assert data.main(["--mock"]) == 0
    return workdir


# --- AC1: interrupted run, then rerun -> one row per question -------------------------------------------


def test_interrupted_run_resumes_without_duplicates(mockdata: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = STRATEGIES["edge"]
    answered = [0]

    class Interrupting(EdgeOnly):
        def answer(self, q: Question) -> tuple[str, list[Generation], dict[str, Any]]:
            if answered[0] == 3:
                raise KeyboardInterrupt
            answered[0] += 1
            return super().answer(q)

    def factory(ctx: StrategyContext) -> Strategy:
        assert ctx.edge is not None
        return Interrupting(ctx.edge, ctx.settings, mock=ctx.mock)

    monkeypatch.setitem(STRATEGIES, "edge", factory)
    with pytest.raises(KeyboardInterrupt):
        run.main([*EDGE, "--network", "normal"])
    path = mockdata / "results" / "edge__qwen2.5-3b__normal.csv"
    qids = _qids(mockdata)
    assert [r["qid"] for r in _rows(path)] == qids[:3]
    first_run_id = _rows(path)[0]["run_id"]

    monkeypatch.setitem(STRATEGIES, "edge", original)
    assert run.main([*EDGE, "--network", "normal"]) == 0
    rows = _rows(path)
    assert [r["qid"] for r in rows] == qids  # exactly one row per question, in file order
    assert {r["run_id"] for r in rows} == {first_run_id}  # resumed rows keep the run id of the meta file
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["run_id"] == first_run_id

    assert run.main([*EDGE, "--network", "normal"]) == 0  # already complete: nothing appended
    assert len(_rows(path)) == len(qids)


# --- AC2: ClientError on one qid -> error row, run continues ----------------------------------------------


class _FailOn:
    """Wraps a client and raises ClientError for the prompt of one question."""

    def __init__(self, inner: Any, needle: str) -> None:
        self.inner, self.needle = inner, needle
        self.name, self.ran_on = inner.name, inner.ran_on

    def check(self) -> None:
        self.inner.check()

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> Generation:
        if self.needle in prompt:
            raise ClientError("HTTP 500 from backend")
        gen: Generation = self.inner.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        return gen


@pytest.mark.parametrize("kind", ["edge", "cloud"])
def test_client_error_on_one_qid_gives_error_row(
    mockdata: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    questions = data.load_questions(mockdata / "data" / data.TEST_FILE)
    target = questions[4]
    real = run.make_client

    def factory(k: str, s: Any, *, model: str | None = None, mock: bool = False) -> Any:
        return _FailOn(real(k, s, model=model, mock=mock), target.question)

    monkeypatch.setattr(run, "make_client", factory)
    argv = EDGE if kind == "edge" else CLOUD
    assert run.main([*argv, "--network", "normal"]) == 0
    name = "edge__qwen2.5-3b__normal.csv" if kind == "edge" else "cloud__none__normal.csv"
    rows = _rows(mockdata / "results" / name)
    assert [r["qid"] for r in rows] == [q.qid for q in questions]
    for r in rows:
        if r["qid"] == target.qid:
            assert r["error"] == "ClientError: HTTP 500 from backend"
            assert (r["correct"], r["parse_failed"], r["predicted"]) == ("False", "False", "")
            assert (r["edge_calls"], r["cloud_calls"]) == ("0", "0")
            assert float(r["total_s"]) == 0.0
        else:
            assert r["error"] == ""
            assert r[f"{kind}_calls"] == "1"


def test_error_row_counts_completed_calls(mockdata: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A strategy that completed one edge call before its cloud call failed records that call."""
    done_gen = Generation("Answer: 1", 0.25, 1.5, 7, 10.0, "qwen2.5:3b", "edge", 1.25)

    class EdgeThenFail(Strategy):
        name = "edge"

        def answer(self, q: Question) -> tuple[str, list[Generation], dict[str, Any]]:
            raise StrategyError(ClientError("timed out"), [done_gen], {"router_stage": 2})

    monkeypatch.setitem(STRATEGIES, "edge", lambda ctx: EdgeThenFail(ctx.settings, ctx.mock))
    assert run.main([*EDGE, "--network", "normal", "--limit", "2"]) == 0
    rows = _rows(mockdata / "results" / "edge__qwen2.5-3b__normal.csv")
    assert len(rows) == 2
    for r in rows:
        assert (r["edge_calls"], r["cloud_calls"], r["router_stage"]) == ("1", "0", "2")
        assert (float(r["ttft_s"]), float(r["total_s"]), r["peak_mem_gb"]) == (0.25, 1.5, "1.2500")
        assert r["error"] == "ClientError: timed out" and r["correct"] == "False" and r["parse_failed"] == "False"


@pytest.mark.parametrize("cls", [EdgeOnly, CloudOnly])
def test_strategies_wrap_client_error(workdir: Path, cls: type[Strategy]) -> None:
    settings = load_settings("config.yaml")
    q = Question("q1", "gsm8k", "2+2?", "4")
    failing = _FailOn(MockClient("m", "edge", 1.0, 1), "2+2?")
    with pytest.raises(StrategyError) as exc:
        cls(failing, settings).answer(q)  # type: ignore[call-arg]
    assert isinstance(exc.value.cause, ClientError) and exc.value.generations == []


# --- AC3: cloud budget -------------------------------------------------------------------------------------


def test_budgeted_client_refuses_before_the_call() -> None:
    inner = MockClient("llama", "cloud", 1.0, 1)
    client = BudgetedClient(inner, max_calls=2)
    assert (client.name, client.ran_on) == ("llama", "cloud")
    for _ in range(2):
        client.generate("Q [[gold:1]]", temperature=0.0, max_tokens=8)
    with pytest.raises(BudgetExceeded):
        client.generate("Q [[gold:1]]", temperature=0.0, max_tokens=8)
    assert client.calls == 2


def test_budget_exits_2_then_rerun_resumes(mockdata: Path) -> None:
    _set_config(mockdata, "cloud", "max_calls_per_run", 3)
    path = mockdata / "results" / "cloud__none__normal.csv"
    qids = _qids(mockdata)
    assert run.main([*CLOUD, "--network", "normal"]) == 2
    assert [r["qid"] for r in _rows(path)] == qids[:3]  # the 4th question has no row
    assert run.main([*CLOUD, "--network", "normal"]) == 2  # fresh budget per process: 3 more
    assert [r["qid"] for r in _rows(path)] == qids[:6]
    _set_config(mockdata, "cloud", "max_calls_per_run", 1500)
    assert run.main([*CLOUD, "--network", "normal"]) == 0
    rows = _rows(path)
    assert [r["qid"] for r in rows] == qids and all(r["error"] == "" for r in rows)


def test_budget_stops_all_and_is_shared_across_strategies(mockdata: Path) -> None:
    _set_config(mockdata, "cloud", "max_calls_per_run", 3)
    assert run.main(["--all", "--network", "normal", "--mock"]) == 2
    results = mockdata / "results"
    assert len(_rows(results / "cloud__none__normal.csv")) == 3
    assert all(len(_rows(results / f"edge__{m}__normal.csv")) == len(_qids(mockdata))
               for m in ("qwen2.5-3b", "qwen2.5-0.5b", "llama3.2-1b"))


# --- AC4: --all ------------------------------------------------------------------------------------------


def test_all_runs_every_edge_model_then_cloud(mockdata: Path) -> None:
    assert run.main(["--all", "--network", "normal", "--mock"]) == 0
    results = mockdata / "results"
    n = len(_qids(mockdata))
    expected = {"edge__qwen2.5-3b__normal", "edge__qwen2.5-0.5b__normal", "edge__llama3.2-1b__normal",
                "cloud__none__normal"}
    assert {p.name for p in results.iterdir()} == {f"{s}{ext}" for s in expected for ext in (".csv", ".meta.json")}
    for stem in expected:
        rows = _rows(results / f"{stem}.csv")
        assert len(rows) == n and len({r["qid"] for r in rows}) == n
    assert {r["edge_model"] for r in _rows(results / "edge__llama3.2-1b__normal.csv")} == {"llama3.2:1b"}


def test_all_includes_registered_strategies_in_order(mockdata: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[tuple[str, str | None, bool, bool]] = []

    class Fake(Strategy):
        def __init__(self, ctx: StrategyContext, name: str) -> None:
            super().__init__(ctx.settings, ctx.mock)
            self.name = name
            order.append((name, ctx.edge_model, ctx.edge is not None, ctx.cloud is not None))

        def answer(self, q: Question) -> tuple[str, list[Generation], dict[str, Any]]:
            return "Answer: 1", [], {}

    for name in ("random", "hybrid"):  # registration order must not matter
        monkeypatch.setitem(STRATEGIES, name, lambda ctx, n=name: Fake(ctx, n))
    assert run.main(["--all", "--network", "degraded", "--mock", "--limit", "2"]) == 0
    assert order == [("hybrid", "qwen2.5:3b", True, True), ("random", "qwen2.5:3b", True, True)]
    results = mockdata / "results"
    for name in ("edge__qwen2.5-0.5b", "cloud__none", "hybrid__qwen2.5-3b", "random__qwen2.5-3b"):
        assert len(_rows(results / f"{name}__degraded.csv")) == 2


def test_all_and_strategy_together_is_a_user_error(mockdata: Path) -> None:
    assert run.main(["--all", *EDGE, "--network", "normal"]) == 1
    assert run.main(["--all", "--edge-model", "qwen2.5:3b", "--network", "normal", "--mock"]) == 1
    assert not (mockdata / "results").exists()


# --- AC5: degraded never overwrites normal -------------------------------------------------------------


def test_degraded_never_overwrites_normal(mockdata: Path) -> None:
    assert run.main(["--all", "--network", "normal", "--mock"]) == 0
    results = mockdata / "results"
    before = {p.name: p.read_bytes() for p in results.iterdir()}
    assert run.main(["--all", "--network", "degraded", "--mock"]) == 0
    assert run.main([*EDGE, "--network", "degraded"]) == 0
    assert {p.name: p.read_bytes() for p in results.iterdir() if "__normal" in p.name} == before
    degraded = [p for p in results.iterdir() if p.name.endswith("__degraded.csv")]
    assert len(degraded) == 4
    for p in degraded:
        assert {r["network"] for r in _rows(p)} == {"degraded"}


# --- --limit ---------------------------------------------------------------------------------------------


def test_limit_takes_first_n_in_file_order(mockdata: Path) -> None:
    path = mockdata / "results" / "edge__qwen2.5-3b__normal.csv"
    assert run.main([*EDGE, "--network", "normal", "--limit", "5"]) == 0
    assert [r["qid"] for r in _rows(path)] == _qids(mockdata)[:5]
    assert run.main([*EDGE, "--network", "normal", "--limit", "7"]) == 0  # resumes up to 7
    assert [r["qid"] for r in _rows(path)] == _qids(mockdata)[:7]
    assert done_qids(path) == set(_qids(mockdata)[:7])


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_limit_below_one_is_a_user_error(mockdata: Path, bad: str) -> None:
    assert run.main([*EDGE, "--network", "normal", "--limit", bad]) == 1


def test_progress_bar_starts_from_done_rows(mockdata: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run.main([*EDGE, "--network", "normal", "--limit", "4"])
    capsys.readouterr()
    run.main([*EDGE, "--network", "normal"])
    err = capsys.readouterr().err
    n = len(_qids(mockdata))
    assert "edge qwen2.5:3b normal:" in err and f"4/{n}" in err and f"{n}/{n}" in err
