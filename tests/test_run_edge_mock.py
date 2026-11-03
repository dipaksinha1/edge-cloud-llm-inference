from __future__ import annotations

import csv
from pathlib import Path

import pytest

from eclm import data, run
from eclm.types import RESULT_COLUMNS


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == RESULT_COLUMNS
        return list(reader)


def _n_test(workdir: Path) -> int:
    return len((workdir / "data" / data.TEST_FILE).read_text().splitlines())


def _edge(network: str = "normal", results_dir: str | None = None) -> int:
    argv = ["--strategy", "edge", "--edge-model", "qwen2.5:3b", "--network", network, "--mock"]
    if results_dir:
        argv += ["--results-dir", results_dir]
    return run.main(argv)


def test_mock_edge_run_writes_one_row_per_question(workdir: Path) -> None:
    assert data.main(["--mock"]) == 0
    assert _edge() == 0
    rows = _rows(workdir / "results" / "edge__qwen2.5-3b__normal.csv")
    assert len(rows) == _n_test(workdir) > 0
    for r in rows:
        assert r["ran_on"] == "edge" and r["strategy"] == "edge" and r["edge_model"] == "qwen2.5:3b"
        assert (r["edge_calls"], r["cloud_calls"]) == ("1", "0")
        assert 0 <= float(r["ttft_s"]) <= float(r["total_s"])
        assert r["correct"] == str(r["predicted"] == r["gold"])
        assert r["parse_failed"] == "False" and r["error"] == ""


def test_network_label_in_rows_and_file_name(workdir: Path) -> None:
    data.main(["--mock"])
    assert _edge("degraded") == 0
    rows = _rows(workdir / "results" / "edge__qwen2.5-3b__degraded.csv")
    assert rows and all(r["network"] == "degraded" for r in rows)
    assert not (workdir / "results" / "edge__qwen2.5-3b__normal.csv").exists()


def test_two_runs_give_identical_predictions(workdir: Path) -> None:
    data.main(["--mock"])
    _edge(results_dir="r1")
    _edge(results_dir="r2")
    a = [r["predicted"] for r in _rows(workdir / "r1" / "edge__qwen2.5-3b__normal.csv")]
    b = [r["predicted"] for r in _rows(workdir / "r2" / "edge__qwen2.5-3b__normal.csv")]
    assert a == b


def test_network_is_required(workdir: Path) -> None:
    data.main(["--mock"])
    with pytest.raises(SystemExit) as exc:
        run.main(["--strategy", "edge", "--mock"])
    assert exc.value.code == 1


def test_missing_data_file_exits_1(workdir: Path) -> None:
    assert _edge() == 1


def test_unimplemented_options_exit_1(workdir: Path) -> None:
    data.main(["--mock"])
    assert run.main(["--strategy", "cloud", "--network", "normal", "--mock"]) == 1
    assert run.main(["--all", "--network", "normal", "--mock"]) == 1
    assert data.main([]) == 1


def test_mock_split_is_disjoint_and_balanced() -> None:
    test, calibration = data.mock_questions()
    assert not {q.qid for q in test} & {q.qid for q in calibration}
    for part in (test, calibration):
        assert sum(q.dataset == "gsm8k" for q in part) == sum(q.dataset == "mmlu" for q in part) > 0


def test_save_load_round_trip(tmp_path: Path) -> None:
    test, _ = data.mock_questions()
    data.save_questions(test, tmp_path / "q.jsonl")
    assert data.load_questions(tmp_path / "q.jsonl") == test
