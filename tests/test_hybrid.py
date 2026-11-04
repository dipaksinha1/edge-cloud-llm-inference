"""T009: two-stage router and the hybrid strategy (scripted fakes and mock mode, no network)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from conftest import ScriptedClient, StubRouterModel

from eclm import data, router, run
from eclm.clients import ClientError
from eclm.config import Settings, load_settings
from eclm.router import Router
from eclm.strategies import STRATEGIES, Hybrid, StrategyError
from eclm.types import RESULT_COLUMNS, Question, Result

REPO = Path(__file__).resolve().parents[1]
GSM = Question("g1", "gsm8k", "Tom has 70 apples and buys 2 more. How many?", "72")
MMLU = Question("m1", "mmlu", "Which is a prime?", "B", ("4", "5", "6", "8"), "college_mathematics")
HYBRID = ["--strategy", "hybrid", "--edge-model", "qwen2.5:3b", "--mock"]


@pytest.fixture
def settings() -> Settings:
    return load_settings(REPO / "config.yaml")


def _hybrid(settings: Settings, difficulty: float, edge_script: list[object], cloud_script: list[object] | None = None,
            edge_timings: list[tuple[float, float]] | None = None,
            cloud_timings: list[tuple[float, float]] | None = None) -> tuple[Hybrid, ScriptedClient, ScriptedClient]:
    edge = ScriptedClient(edge_script, timings=edge_timings)
    cloud = ScriptedClient(cloud_script or ["72"], ran_on="cloud", name="cloud-model", timings=cloud_timings)
    r = Router(StubRouterModel(difficulty), edge, settings)  # type: ignore[arg-type]
    return Hybrid(edge, cloud, r, settings), edge, cloud


def _row(strategy: Hybrid, q: Question = GSM) -> Result:
    text, gens, extra = strategy.answer(q)
    return run.build_result(run_id="r", strategy="hybrid", edge_model="e", cloud_model="c", network="normal",
                            q=q, text=text, gens=gens, extra=extra)


# --- AC1: stage 1 ----------------------------------------------------------------------------------------


def test_stage1_hard_question_goes_straight_to_cloud(settings: Settings) -> None:
    h, edge, cloud = _hybrid(settings, 0.9, [])
    r = _row(h)
    assert (r.ran_on, r.router_stage, r.difficulty, r.agreement) == ("cloud", "1", "0.9000", "")
    assert (r.edge_calls, r.cloud_calls) == (0, 1)
    assert edge.calls == [] and len(cloud.calls) == 1
    assert cloud.calls[0][1] == settings.generation.temperature
    assert r.predicted == "72" and r.correct and r.peak_mem_gb == ""


def test_threshold_is_strict(settings: Settings) -> None:
    """A score equal to the threshold is not 'above' it, so it goes to stage 2 (same rule as router train)."""
    d = Router(StubRouterModel(settings.router.threshold), ScriptedClient(["1", "1", "1"]),  # type: ignore[arg-type]
               settings).decide(GSM)
    assert d.route == "edge" and d.stage is None and len(d.edge_generations) == 3


# --- AC2: stage 2 keeps the edge answer --------------------------------------------------------------------


def test_two_of_three_agree_stays_on_edge(settings: Settings) -> None:
    h, edge, cloud = _hybrid(settings, 0.1, ["72", "72", "68"])
    r = _row(h)
    assert (r.ran_on, r.router_stage, r.difficulty, r.agreement) == ("edge", "", "0.1000", "2/3")
    assert (r.edge_calls, r.cloud_calls) == (3, 0)
    assert r.predicted == "72" and r.correct and not r.parse_failed
    assert cloud.calls == []
    assert [t for _, t in edge.calls] == [settings.router.sample_temperature] * settings.router.samples
    assert len({p for p, _ in edge.calls}) == 1


def test_decision_fields_when_kept_on_edge(settings: Settings) -> None:
    d = Router(StubRouterModel(0.2), ScriptedClient(["72", "72.0", "68"]), settings).decide(GSM)  # type: ignore[arg-type]
    assert (d.route, d.stage, d.agreement, d.edge_answer) == ("edge", None, 2, "72")
    assert d.difficulty == 0.2 and len(d.edge_generations) == 3


def test_mmlu_majority_letter(settings: Settings) -> None:
    h, _, _ = _hybrid(settings, 0.1, ["C", "B", "B"])
    r = _row(h, MMLU)
    assert (r.ran_on, r.predicted, r.agreement, r.correct) == ("edge", "B", "2/3", True)


def test_three_agreeing_samples(settings: Settings) -> None:
    h, _, _ = _hybrid(settings, 0.1, ["68", "68", "68"])
    r = _row(h)
    assert (r.ran_on, r.predicted, r.agreement, r.correct) == ("edge", "68", "3/3", False)


# --- AC3: stage 2 escalates ----------------------------------------------------------------------------------


def test_three_different_answers_escalate(settings: Settings) -> None:
    h, edge, cloud = _hybrid(settings, 0.1, ["72", "68", "5"], ["72"],
                             edge_timings=[(0.1, 1.0), (0.2, 2.0), (0.3, 3.0)], cloud_timings=[(0.5, 4.0)])
    r = _row(h)
    assert (r.ran_on, r.router_stage, r.agreement) == ("cloud", "2", "1/3")
    assert (r.edge_calls, r.cloud_calls) == (3, 1)
    assert r.total_s == pytest.approx(10.0)
    assert r.predicted == "72" and r.correct
    assert r.output_tokens == 10  # the cloud call's (first and only cloud call)


def test_all_unparseable_samples_have_zero_agreement(settings: Settings) -> None:
    h, _, _ = _hybrid(settings, 0.1, ["no idea", "dunno", "?"])
    r = _row(h)
    assert (r.ran_on, r.router_stage, r.agreement) == ("cloud", "2", "0/3")


def test_edge_sample_error_counts_as_unparseable(settings: Settings) -> None:
    h, _, cloud = _hybrid(settings, 0.1, [ClientError("timeout"), "72", "72"])
    r = _row(h)
    assert (r.ran_on, r.agreement, r.edge_calls, r.cloud_calls, r.error) == ("edge", "2/3", 2, 0, "")
    h, _, cloud = _hybrid(settings, 0.1, [ClientError("timeout"), "72", "68"])
    r = _row(h)
    assert (r.ran_on, r.router_stage, r.agreement, r.edge_calls, r.cloud_calls) == ("cloud", "2", "1/3", 2, 1)


def test_cloud_error_keeps_the_edge_samples(settings: Settings) -> None:
    h, _, _ = _hybrid(settings, 0.1, ["1", "2", "3"], [ClientError("HTTP 503")])
    with pytest.raises(StrategyError) as exc:
        h.answer(GSM)
    assert len(exc.value.generations) == 3 and all(g.ran_on == "edge" for g in exc.value.generations)
    assert exc.value.extra["router_stage"] == "2" and exc.value.extra["agreement"] == "1/3"
    assert isinstance(exc.value.cause, ClientError)


def test_cloud_error_at_stage1_has_no_generations(settings: Settings) -> None:
    h, _, _ = _hybrid(settings, 0.95, [], [ClientError("HTTP 503")])
    with pytest.raises(StrategyError) as exc:
        h.answer(GSM)
    assert exc.value.generations == [] and exc.value.extra["router_stage"] == "1"


# --- AC4: TTFT per ARCHITECTURE §6 ---------------------------------------------------------------------------


def test_ttft_when_kept_on_edge_uses_first_majority_sample(settings: Settings) -> None:
    """Answers 68, 72, 72: the answering call is sample 2 (index 1), so TTFT = total(1) + ttft(2)."""
    h, _, _ = _hybrid(settings, 0.1, ["68", "72", "72"], edge_timings=[(0.1, 1.0), (0.2, 2.0), (0.3, 3.0)])
    text, gens, extra = h.answer(GSM)
    assert extra["answer_index"] == 1 and text == gens[1].text
    r = _row(_hybrid(settings, 0.1, ["68", "72", "72"], edge_timings=[(0.1, 1.0), (0.2, 2.0), (0.3, 3.0)])[0])
    assert r.ttft_s == pytest.approx(1.2)
    assert r.total_s == pytest.approx(6.0)
    assert (r.output_tokens, r.tokens_per_s) == (11, 11.0)  # the answering call's, not the last call's


def test_ttft_when_first_sample_answers(settings: Settings) -> None:
    h, _, _ = _hybrid(settings, 0.1, ["72", "68", "72"], edge_timings=[(0.1, 1.0), (0.2, 2.0), (0.3, 3.0)])
    r = _row(h)
    assert (r.ttft_s, r.total_s) == (pytest.approx(0.1), pytest.approx(6.0))


def test_ttft_when_escalated(settings: Settings) -> None:
    h, _, _ = _hybrid(settings, 0.1, ["1", "2", "3"], ["72"],
                      edge_timings=[(0.1, 1.0), (0.2, 2.0), (0.3, 3.0)], cloud_timings=[(0.5, 4.0)])
    r = _row(h)
    assert (r.ttft_s, r.total_s) == (pytest.approx(6.5), pytest.approx(10.0))


def test_ttft_stage1(settings: Settings) -> None:
    h, _, _ = _hybrid(settings, 0.9, [], ["72"], cloud_timings=[(0.5, 4.0)])
    r = _row(h)
    assert (r.ttft_s, r.total_s) == (pytest.approx(0.5), pytest.approx(4.0))


def test_combine_timings_default_is_last_call() -> None:
    from eclm.types import Generation

    gens = [Generation("a", 0.1, 1.0, 1, 1.0, "m", "edge"), Generation("b", 0.2, 2.0, 1, 1.0, "m", "edge")]
    assert run.combine_timings(gens) == (pytest.approx(1.2), pytest.approx(3.0))
    assert run.combine_timings(gens, 0) == (pytest.approx(0.1), pytest.approx(3.0))


# --- CLI / AC5: mock fixture -----------------------------------------------------------------------------------


@pytest.fixture
def mockdata(workdir: Path) -> Path:
    assert data.main(["--mock"]) == 0
    return workdir


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == RESULT_COLUMNS
        return list(reader)


def test_missing_router_exits_1_with_hint(mockdata: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run.main([*HYBRID, "--network", "normal"]) == 1
    err = capsys.readouterr().err
    assert "train it: python -m eclm.router train --edge-model qwen2.5:3b" in err
    assert not (mockdata / "results").exists()


def test_all_without_router_stops_before_any_question(mockdata: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run.main(["--all", "--network", "normal", "--mock"]) == 1
    assert "train it" in capsys.readouterr().err
    assert not (mockdata / "results").exists()


def test_router_for_another_model_exits_1(mockdata: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert router.main(["train", "--edge-model", "qwen2.5:0.5b", "--mock"]) == 0
    src = mockdata / "models" / "router_qwen2.5-0.5b.joblib"
    src.rename(mockdata / "models" / "router_qwen2.5-3b.joblib")
    assert run.main([*HYBRID, "--network", "normal"]) == 1
    assert "trained for qwen2.5:0.5b" in capsys.readouterr().err


def test_hybrid_mock_run_records_routing(mockdata: Path) -> None:
    assert router.main(["train", "--edge-model", "qwen2.5:3b", "--mock"]) == 0
    assert run.main([*HYBRID, "--network", "normal"]) == 0
    rows = _rows(mockdata / "results" / "hybrid__qwen2.5-3b__normal.csv")
    qids = [q.qid for q in data.load_questions(mockdata / "data" / data.TEST_FILE)]
    assert [r["qid"] for r in rows] == qids
    assert any(r["router_stage"] == "2" for r in rows), "mock samples at temperature 0.7 should disagree somewhere"
    for r in rows:
        assert r["error"] == "" and r["strategy"] == "hybrid" and r["edge_model"] == "qwen2.5:3b"
        assert len(r["difficulty"].split(".")[1]) == 4
        stage, calls = r["router_stage"], (r["edge_calls"], r["cloud_calls"])
        if stage == "1":
            assert (r["ran_on"], calls, r["agreement"], r["peak_mem_gb"]) == ("cloud", ("0", "1"), "", "")
        elif stage == "2":
            assert (r["ran_on"], calls) == ("cloud", ("3", "1")) and r["peak_mem_gb"] != ""
        else:
            assert (r["ran_on"], calls) == ("edge", ("3", "0")) and r["peak_mem_gb"] != ""
            assert r["agreement"] in ("2/3", "3/3")


def test_hybrid_is_registered_and_runs_under_all(mockdata: Path) -> None:
    assert "hybrid" in STRATEGIES
    assert router.main(["train", "--edge-model", "qwen2.5:3b", "--mock"]) == 0
    assert run.main(["--all", "--network", "normal", "--mock", "--limit", "3"]) == 0
    rows = _rows(mockdata / "results" / "hybrid__qwen2.5-3b__normal.csv")
    assert len(rows) == 3 and {r["edge_model"] for r in rows} == {"qwen2.5:3b"}
