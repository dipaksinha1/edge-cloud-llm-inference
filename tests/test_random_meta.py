"""The random router's resolved ratio is recorded in meta.json and kept when a run resumes (mock mode)."""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Any

import pytest

from eclm import data, router, run
from eclm.config import load_settings
from eclm.results import offload_ratio

RANDOM = ["--strategy", "random", "--edge-model", "qwen2.5:3b", "--mock", "--network", "normal"]
CSV = "random__qwen2.5-3b__normal.csv"
META = "random__qwen2.5-3b__normal.meta.json"
HYBRID = "hybrid__qwen2.5-3b__normal.csv"


@pytest.fixture
def mockdata(workdir: Path) -> Path:
    assert data.main(["--mock"]) == 0
    assert router.main(["train", "--edge-model", "qwen2.5:3b", "--mock"]) == 0
    return workdir


def _qids(workdir: Path) -> list[str]:
    return [q.qid for q in data.load_questions(workdir / "data" / data.TEST_FILE)]


def _write_hybrid(path: Path, qids: list[str], n_cloud: int) -> None:
    """A minimal hybrid CSV whose offload ratio is n_cloud / len(qids)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "ran_on"])
        w.writeheader()
        for i, qid in enumerate(qids):
            w.writerow({"qid": qid, "ran_on": "cloud" if i < n_cloud else "edge"})


def _meta(workdir: Path, name: str = META) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((workdir / "results" / name).read_text())
    return meta


def _cloud_rows(workdir: Path) -> set[str]:
    with open(workdir / "results" / CSV, newline="") as f:
        return {r["qid"] for r in csv.DictReader(f) if r["ran_on"] == "cloud"}


def _interrupt_after(workdir: Path, n_rows: int) -> None:
    """Simulate a run stopped after n_rows: keep the header and the first n_rows rows (meta.json stays)."""
    path = workdir / "results" / CSV
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[: 1 + n_rows]))


def _expected_cloud(qids: list[str], ratio: float) -> set[str]:
    seed = load_settings().seed
    return set(random.Random(seed).sample(sorted(qids), round(ratio * len(qids))))


# --- AC1: meta.json records ratio, source, k and n ------------------------------------------------------


def test_meta_records_explicit_ratio(mockdata: Path) -> None:
    assert run.main([*RANDOM, "--ratio", "0.25"]) == 0
    n = len(_qids(mockdata))
    assert _meta(mockdata)["strategy"] == {
        "ratio": 0.25, "ratio_source": "--ratio", "cloud_questions": round(0.25 * n), "questions": n
    }


def test_meta_records_ratio_from_hybrid_csv(mockdata: Path, caplog: pytest.LogCaptureFixture) -> None:
    qids = _qids(mockdata)
    hybrid = mockdata / "results" / HYBRID
    _write_hybrid(hybrid, qids, 4)
    caplog.set_level(logging.INFO, logger="eclm.strategies")
    assert run.main(RANDOM) == 0
    strategy = _meta(mockdata)["strategy"]
    assert strategy == {
        "ratio": 4 / len(qids), "ratio_source": str(Path("results") / HYBRID),
        "cloud_questions": 4, "questions": len(qids),
    }
    assert any(f"-> {strategy['cloud_questions']} of {strategy['questions']} questions" in r.getMessage()
               for r in caplog.records)


# --- AC2: a resume keeps the recorded ratio, even if the hybrid CSV changed -----------------------------


def test_resume_keeps_recorded_ratio_and_warns(mockdata: Path, caplog: pytest.LogCaptureFixture) -> None:
    qids = _qids(mockdata)
    hybrid = mockdata / "results" / HYBRID
    _write_hybrid(hybrid, qids, 3)  # r0 = 3/12
    assert run.main(RANDOM) == 0
    _interrupt_after(mockdata, 5)
    r0 = offload_ratio(hybrid)
    assert r0 is not None
    _write_hybrid(hybrid, qids, 9)  # the hybrid file changes: r1 = 9/12
    caplog.set_level(logging.WARNING)
    assert run.main(RANDOM) == 0
    assert _cloud_rows(mockdata) == _expected_cloud(qids, r0)
    assert _cloud_rows(mockdata) != _expected_cloud(qids, 9 / len(qids))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("keeping the recorded ratio 0.25 " in w and "gives 0.75." in w for w in warnings), warnings
    assert _meta(mockdata)["strategy"]["ratio"] == r0  # the meta file is kept, not rewritten


def test_resume_with_same_source_does_not_warn(mockdata: Path, caplog: pytest.LogCaptureFixture) -> None:
    qids = _qids(mockdata)
    _write_hybrid(mockdata / "results" / HYBRID, qids, 3)
    assert run.main(RANDOM) == 0
    _interrupt_after(mockdata, 4)
    caplog.set_level(logging.WARNING)
    assert run.main(RANDOM) == 0
    assert _cloud_rows(mockdata) == _expected_cloud(qids, 3 / len(qids))
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_resume_keeps_recorded_ratio_when_hybrid_csv_is_gone(mockdata: Path) -> None:
    qids = _qids(mockdata)
    hybrid = mockdata / "results" / HYBRID
    _write_hybrid(hybrid, qids, 6)
    assert run.main(RANDOM) == 0
    _interrupt_after(mockdata, 4)
    hybrid.unlink()
    assert run.main(RANDOM) == 0  # no source any more: the recorded ratio still resumes the run
    assert _cloud_rows(mockdata) == _expected_cloud(qids, 6 / len(qids))


# --- AC3: old meta without "strategy" behaves as before; other strategies write {} ----------------------


def test_resume_with_pre_t017_meta_resolves_again(mockdata: Path, caplog: pytest.LogCaptureFixture) -> None:
    qids = _qids(mockdata)
    hybrid = mockdata / "results" / HYBRID
    _write_hybrid(hybrid, qids, 3)
    assert run.main(RANDOM) == 0
    _interrupt_after(mockdata, 4)
    meta_file = mockdata / "results" / META
    meta = json.loads(meta_file.read_text())
    del meta["strategy"]  # as written by older versions
    meta_file.write_text(json.dumps(meta))
    _write_hybrid(hybrid, qids, 6)
    caplog.set_level(logging.WARNING)
    assert run.main(RANDOM) == 0
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    cloud = _cloud_rows(mockdata)  # the first 4 rows used 3/12, the rest the re-resolved 6/12 (old behaviour)
    first4, rest = set(qids[:4]), set(qids[4:])
    assert cloud & first4 == _expected_cloud(qids, 3 / len(qids)) & first4
    assert cloud & rest == _expected_cloud(qids, 6 / len(qids)) & rest


def test_resume_with_another_question_count_warns(mockdata: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert run.main([*RANDOM, "--ratio", "0.25", "--limit", "4"]) == 0
    caplog.set_level(logging.WARNING)
    assert run.main([*RANDOM, "--ratio", "0.25"]) == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("started with 4 questions and now has 12" in w for w in warnings), warnings


def test_other_strategies_write_no_strategy_meta(mockdata: Path) -> None:
    assert run.main(["--all", "--network", "normal", "--mock", "--limit", "2"]) == 0
    for name in ["edge__qwen2.5-3b__normal", "cloud__none__normal", "hybrid__qwen2.5-3b__normal"]:
        assert "strategy" not in _meta(mockdata, f"{name}.meta.json")
    random_meta = _meta(mockdata)["strategy"]
    assert random_meta["ratio_source"] == str(Path("results") / HYBRID)
    assert random_meta["questions"] == 2


@pytest.mark.parametrize("bad", [1.5, -0.1, "0.25", float("nan")])
def test_resume_with_invalid_recorded_ratio_exits_1(
    mockdata: Path, capsys: pytest.CaptureFixture[str], bad: object
) -> None:
    assert run.main([*RANDOM, "--ratio", "0.25"]) == 0
    _interrupt_after(mockdata, 4)
    meta_file = mockdata / "results" / META
    meta = json.loads(meta_file.read_text())
    meta["strategy"]["ratio"] = bad
    meta_file.write_text(json.dumps(meta))
    assert run.main([*RANDOM, "--ratio", "0.25"]) == 1
    assert "invalid ratio" in capsys.readouterr().err
    assert len((mockdata / "results" / CSV).read_text().splitlines()) == 1 + 4  # nothing appended
