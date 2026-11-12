"""Random-router baseline (scripted fakes and mock mode, no network)."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest
from conftest import ScriptedClient

from eclm import data, router, run
from eclm.clients import ClientError
from eclm.config import Settings, load_settings
from eclm.results import offload_ratio
from eclm.strategies import STRATEGIES, RandomRouter, StrategyError, StrategyUnavailable
from eclm.types import RESULT_COLUMNS, Question

REPO = Path(__file__).resolve().parents[1]
RANDOM = ["--strategy", "random", "--edge-model", "qwen2.5:3b", "--mock"]


@pytest.fixture
def settings() -> Settings:
    return load_settings(REPO / "config.yaml")


@pytest.fixture
def mockdata(workdir: Path) -> Path:
    assert data.main(["--mock"]) == 0
    return workdir


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == RESULT_COLUMNS
        return list(reader)


def _cloud_qids(path: Path) -> set[str]:
    return {r["qid"] for r in _rows(path) if r["ran_on"] == "cloud"}


def _questions(n: int) -> list[Question]:
    return [Question(f"q{i:02d}", "gsm8k", f"What is {i} + 1?", str(i + 1)) for i in range(n)]


def _write_hybrid(path: Path, ran_on: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        for i, where in enumerate(ran_on):
            w.writerow({c: "" for c in RESULT_COLUMNS} | {"qid": f"h{i}", "ran_on": where})


# --- AC1: --ratio 0.25 on 12 questions -> exactly 3 cloud rows, the same on rerun ------------------------


def test_ratio_selects_exact_seeded_subset(settings: Settings) -> None:
    qs = _questions(12)
    edge = ScriptedClient(["1"] * 12)
    cloud = ScriptedClient(["1"] * 12, ran_on="cloud", name="cloud-model")
    s = RandomRouter(edge, cloud, 0.25, settings.seed, settings)  # type: ignore[arg-type]
    s.prepare(list(reversed(qs)))  # input order must not matter: selection uses sorted qids
    expected = set(random.Random(settings.seed).sample(sorted(q.qid for q in qs), 3))
    assert s.cloud_qids == expected
    routed = {}
    for q in qs:
        _, gens, extra = s.answer(q)
        assert len(gens) == 1
        routed[q.qid] = (gens[0].ran_on, extra.get("router_stage", ""))
    assert {qid for qid, (where, _) in routed.items() if where == "cloud"} == expected
    assert all(stage == ("random" if where == "cloud" else "") for where, stage in routed.values())
    assert len(edge.calls) == 9 and len(cloud.calls) == 3


def test_cli_ratio_gives_3_cloud_rows_stable_on_rerun_and_resume(mockdata: Path) -> None:
    results = mockdata / "results"
    path = results / "random__qwen2.5-3b__normal.csv"
    assert run.main([*RANDOM, "--network", "normal", "--ratio", "0.25"]) == 0
    rows = _rows(path)
    assert len(rows) == 12
    first = _cloud_qids(path)
    assert len(first) == 3
    cloud_rows = [r for r in rows if r["ran_on"] == "cloud"]
    assert all(r["router_stage"] == "random" and r["cloud_calls"] == "1" and r["edge_calls"] == "0"
               for r in cloud_rows)
    assert all(r["edge_calls"] == "1" and r["cloud_calls"] == "0" and r["peak_mem_gb"]
               for r in rows if r["ran_on"] == "edge")
    # fresh rerun: same 3
    path.unlink()
    assert run.main([*RANDOM, "--network", "normal", "--ratio", "0.25"]) == 0
    assert _cloud_qids(path) == first
    # interrupted run (first 4 rows kept), then resume: still the same 3, no duplicates
    full = _rows(path)
    with open(path, "w", newline="") as f:  # keep the header and the first 4 rows only
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        w.writerows(full[:4])
    assert run.main([*RANDOM, "--network", "normal", "--ratio", "0.25"]) == 0
    rows = _rows(path)
    assert len(rows) == 12 and len({r["qid"] for r in rows}) == 12
    assert _cloud_qids(path) == first


@pytest.mark.parametrize("bad", ["-0.1", "1.5"])
def test_ratio_out_of_range_is_a_user_error(mockdata: Path, bad: str) -> None:
    assert run.main([*RANDOM, "--network", "normal", "--ratio", bad]) == 1
    assert not (mockdata / "results").exists()


# --- AC2: ratio read from the hybrid CSV ----------------------------------------------------------------


def test_offload_ratio_reads_cloud_share(tmp_path: Path) -> None:
    path = tmp_path / "h.csv"
    assert offload_ratio(path) is None
    _write_hybrid(path, [])
    assert offload_ratio(path) is None
    _write_hybrid(path, ["cloud", "edge", "edge", "", "cloud"])
    assert offload_ratio(path) == pytest.approx(0.4)


def test_ratio_defaults_to_handcrafted_hybrid_csv(mockdata: Path) -> None:
    results = mockdata / "results"
    # 1 of 3 hybrid rows on the cloud -> ratio 1/3 -> k = round(12/3) = 4
    _write_hybrid(results / "hybrid__qwen2.5-3b__degraded.csv", ["cloud", "edge", "edge"])
    _write_hybrid(results / "hybrid__qwen2.5-3b__normal.csv", ["edge"] * 3)  # other network: must be ignored
    assert run.main([*RANDOM, "--network", "degraded"]) == 0
    rows = _rows(results / "random__qwen2.5-3b__degraded.csv")
    assert len(rows) == 12
    assert sum(r["ran_on"] == "cloud" for r in rows) == 4
    qids = sorted(r["qid"] for r in rows)
    assert _cloud_qids(results / "random__qwen2.5-3b__degraded.csv") == set(
        random.Random(load_settings(mockdata / "config.yaml").seed).sample(qids, 4))


def test_explicit_ratio_overrides_hybrid_csv(mockdata: Path) -> None:
    results = mockdata / "results"
    _write_hybrid(results / "hybrid__qwen2.5-3b__normal.csv", ["cloud"] * 3)
    assert run.main([*RANDOM, "--network", "normal", "--ratio", "0.5"]) == 0
    assert len(_cloud_qids(results / "random__qwen2.5-3b__normal.csv")) == 6


def test_all_reads_ratio_from_the_hybrid_it_just_ran(mockdata: Path) -> None:
    assert router.main(["train", "--edge-model", "qwen2.5:3b", "--mock"]) == 0
    assert run.main(["--all", "--network", "normal", "--mock"]) == 0
    results = mockdata / "results"
    ratio = offload_ratio(results / "hybrid__qwen2.5-3b__normal.csv")
    assert ratio is not None
    rows = _rows(results / "random__qwen2.5-3b__normal.csv")
    assert len(rows) == 12
    assert sum(r["ran_on"] == "cloud" for r in rows) == round(ratio * 12)


# --- AC3: no ratio and no hybrid CSV -> exit 1 ----------------------------------------------------------


def test_no_ratio_no_hybrid_csv_exits_1(mockdata: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run.main([*RANDOM, "--network", "normal"]) == 1
    err = capsys.readouterr().err
    assert "--ratio" in err and "hybrid__qwen2.5-3b__normal.csv" in err
    results = mockdata / "results"
    assert not results.exists() or not any(results.iterdir())  # no CSV, no meta


def test_prepare_without_source_raises(settings: Settings) -> None:
    s = RandomRouter(ScriptedClient([]), ScriptedClient([], ran_on="cloud"), None, 1, settings)  # type: ignore[arg-type]
    with pytest.raises(StrategyUnavailable, match="--ratio"):
        s.prepare(_questions(4))


# --- errors and registration ----------------------------------------------------------------------------


def test_client_error_becomes_strategy_error(settings: Settings) -> None:
    qs = _questions(4)
    edge = ScriptedClient([ClientError("edge down")] * 4)
    cloud = ScriptedClient([ClientError("cloud down")] * 4, ran_on="cloud")
    s = RandomRouter(edge, cloud, 0.5, settings.seed, settings)  # type: ignore[arg-type]
    s.prepare(qs)
    assert s.cloud_qids is not None
    for q in qs:
        with pytest.raises(StrategyError) as exc:
            s.answer(q)
        assert exc.value.generations == []
        assert exc.value.extra == ({"router_stage": "random"} if q.qid in s.cloud_qids else {})


def test_answer_before_prepare_raises(settings: Settings) -> None:
    s = RandomRouter(ScriptedClient([]), ScriptedClient([], ran_on="cloud"), 0.5, 1, settings)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        s.answer(_questions(1)[0])


def test_constructor_rejects_bad_ratio(settings: Settings) -> None:
    with pytest.raises(ValueError):
        RandomRouter(ScriptedClient([]), ScriptedClient([]), 1.2, 1, settings)  # type: ignore[arg-type]


def test_random_is_registered() -> None:
    assert "random" in STRATEGIES
