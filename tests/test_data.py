"""Dataset builder. The Hugging Face loader is monkeypatched to small in-memory tables."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from eclm import data
from eclm.config import Settings, load_settings
from eclm.types import Question

Row = dict[str, object]

COUNTS = {"gsm8k_test": 5, "mmlu_test": 5, "gsm8k_calibration": 7, "mmlu_calibration": 7}


def _gsm8k(n: int, tag: str) -> list[Row]:
    return [{"question": f"GSM {tag} question {i}?", "answer": f"Work {i}.\n#### {i * 1000 + 3:,}"} for i in range(n)]


def _mmlu(n: int, tag: str) -> list[Row]:
    return [
        {
            "question": f"MMLU {tag} question {i}?",
            "choices": [f"a{i}", f"b{i}", f"c{i}", f"d{i}"],
            "answer": i % 4,
            "subject": "philosophy" if i % 2 else "college_physics",
        }
        for i in range(n)
    ]


def _tables() -> dict[tuple[str, str, str], list[Row]]:
    gsm, mmlu = data.GSM8K_SOURCE, data.MMLU_SOURCE
    shared = _mmlu(3, "test")  # validation repeats some test question texts (with extra whitespace/case)
    for r in shared:
        r["question"] = "  " + str(r["question"]).upper() + "  "
    return {
        (*gsm, "test"): _gsm8k(20, "test"),
        (*gsm, "train"): _gsm8k(30, "train") + _gsm8k(5, "test"),  # train repeats test texts too
        (*mmlu, "test"): _mmlu(20, "test"),
        (*mmlu, "validation"): shared + _mmlu(5, "val"),
        (*mmlu, "dev"): _mmlu(6, "dev"),
    }


@pytest.fixture
def fake_hf(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    tables = _tables()
    calls: list[tuple[str, str, str]] = []

    def load(path: str, config: str, split: str) -> list[Row]:
        calls.append((path, config, split))
        return [dict(r) for r in tables[(path, config, split)]]

    monkeypatch.setattr(data, "_load_split", load)
    return calls


def _settings(workdir: Path, seed: int = 42, **counts: int) -> Settings:
    return load_settings("config.yaml", {"seed": seed, "dataset": {**COUNTS, **counts}})


def _texts(qs: list[Question]) -> set[str]:
    return {" ".join(q.question.split()).casefold() for q in qs}


def test_counts_halves_and_disjoint(workdir: Path, fake_hf: list[tuple[str, str, str]]) -> None:
    test, cal = data.build_datasets(_settings(workdir))
    assert [q.dataset for q in test] == ["gsm8k"] * 5 + ["mmlu"] * 5
    assert [q.dataset for q in cal] == ["gsm8k"] * 7 + ["mmlu"] * 7
    assert not {q.qid for q in test} & {q.qid for q in cal}
    assert not _texts(test) & _texts(cal)
    assert len({q.qid for q in test + cal}) == 24
    assert all(q.qid.startswith("gsm8k-test-") or q.qid.startswith("mmlu-test-") for q in test)
    assert {q.qid.rsplit("-", 1)[0] for q in cal} <= {"gsm8k-train", "mmlu-validation", "mmlu-dev"}


def test_text_overlap_is_excluded_even_when_every_test_question_is_taken(
    workdir: Path, fake_hf: list[tuple[str, str, str]]
) -> None:
    # all 20 test questions are taken, so the 5 repeated GSM8K and 3 repeated MMLU texts must be skipped
    settings = _settings(workdir, gsm8k_test=20, mmlu_test=20, gsm8k_calibration=30, mmlu_calibration=11)
    test, cal = data.build_datasets(settings)
    assert not _texts(test) & _texts(cal)
    assert len(cal) == 41
    with pytest.raises(data.DatasetError, match="mmlu calibration: need 12"):
        data.build_datasets(_settings(workdir, gsm8k_test=20, mmlu_test=20, mmlu_calibration=12))


def test_mmlu_dev_only_loaded_when_validation_is_short(workdir: Path, fake_hf: list[tuple[str, str, str]]) -> None:
    data.build_datasets(_settings(workdir, mmlu_calibration=4))
    assert (*data.MMLU_SOURCE, "dev") not in fake_hf
    fake_hf.clear()
    _, cal = data.build_datasets(_settings(workdir, mmlu_calibration=10))
    assert (*data.MMLU_SOURCE, "dev") in fake_hf
    assert sum(q.dataset == "mmlu" for q in cal) == 10


def test_question_fields(workdir: Path, fake_hf: list[tuple[str, str, str]]) -> None:
    test, _ = data.build_datasets(_settings(workdir, gsm8k_test=20, mmlu_test=20))
    by_qid = {q.qid: q for q in test}
    g = by_qid["gsm8k-test-7"]
    assert (g.question, g.gold, g.choices, g.subject) == ("GSM test question 7?", "7003", (), "")
    m = by_qid["mmlu-test-6"]
    assert (m.gold, m.choices, m.subject) == ("C", ("a6", "b6", "c6", "d6"), "college_physics")


@pytest.mark.parametrize(
    ("answer", "gold"),
    [("x\n#### 72", "72"), ("#### 1,234", "1234"), ("#### -5", "-5"), ("#### $18", "18"), ("#### 2.5 ", "2.5"),
     ("#### 10.0", "10"), ("#### 10.00", "10")],
)
def test_gsm8k_gold_normalised(answer: str, gold: str) -> None:
    assert data._gsm8k_gold(answer) == gold


@pytest.mark.parametrize("answer", ["#### nan", "#### inf", "#### seven", "no marker"])
def test_gsm8k_gold_rejects_non_numbers(answer: str) -> None:
    with pytest.raises(ValueError):
        data._gsm8k_gold(answer)


def test_bad_row_format_is_a_dataset_error(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "_load_split", lambda p, c, s: [{"question": "q", "answer": "no marker"}])
    with pytest.raises(data.DatasetError, match="gsm8k test row 0"):
        data.build_datasets(_settings(workdir))


def _write_config(workdir: Path, seed: int = 42) -> None:
    cfg = yaml.safe_load((workdir / "config.yaml").read_text())
    cfg["seed"] = seed
    cfg["dataset"] = dict(COUNTS)
    (workdir / "config.yaml").write_text(yaml.safe_dump(cfg))


def _run_cli(workdir: Path, seed: int) -> tuple[bytes, bytes]:
    _write_config(workdir, seed)
    assert data.main([]) == 0
    d = workdir / "data"
    return (d / data.TEST_FILE).read_bytes(), (d / data.CALIBRATION_FILE).read_bytes()


def test_cli_same_seed_identical_bytes_other_seed_differs(
    workdir: Path, fake_hf: list[tuple[str, str, str]]
) -> None:
    first = _run_cli(workdir, 42)
    second = _run_cli(workdir, 42)
    assert first == second
    assert len(first[0].splitlines()) == 10 and len(first[1].splitlines()) == 14
    assert _run_cli(workdir, 7) != first


def test_loader_failure_exit_1_names_dataset_and_keeps_files(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_config(workdir)
    d = workdir / "data"
    d.mkdir()
    (d / data.TEST_FILE).write_text("old test\n")
    (d / data.CALIBRATION_FILE).write_text("old cal\n")
    tables = _tables()

    def load(path: str, config: str, split: str) -> list[Row]:
        if path == data.MMLU_SOURCE[0]:
            raise ConnectionError("hub unreachable")
        return tables[(path, config, split)]

    monkeypatch.setattr(data, "_load_split", load)
    assert data.main([]) == 1
    err = capsys.readouterr().err
    assert "mmlu" in err and "cais/mmlu" in err and "hub unreachable" in err
    assert (d / data.TEST_FILE).read_text() == "old test\n"
    assert (d / data.CALIBRATION_FILE).read_text() == "old cal\n"
    assert sorted(p.name for p in d.iterdir()) == [data.CALIBRATION_FILE, data.TEST_FILE]


def test_round_trip_built_questions(workdir: Path, fake_hf: list[tuple[str, str, str]]) -> None:
    test, cal = data.build_datasets(_settings(workdir))
    for qs, name in ((test, "t.jsonl"), (cal, "c.jsonl")):
        data.save_questions(qs, workdir / name)
        assert data.load_questions(workdir / name) == qs
    unicode_q = [Question("x-1", "mmlu", "Qu\u2019est-ce que \u03c0?", "B", ("1", "2", "3", "4"), "philosophy")]
    data.save_questions(unicode_q, workdir / "u.jsonl")
    assert data.load_questions(workdir / "u.jsonl") == unicode_q


def test_mock_fixture_24_questions_and_split(workdir: Path) -> None:
    all_q = data._fixture_questions()
    assert len(all_q) == 24 and len({q.qid for q in all_q}) == 24
    assert sum(q.dataset == "gsm8k" for q in all_q) == 12
    assert len(_texts(all_q)) == 24
    assert data.main(["--mock"]) == 0
    test = data.load_questions(workdir / "data" / data.TEST_FILE)
    cal = data.load_questions(workdir / "data" / data.CALIBRATION_FILE)
    assert len(test) == len(cal) == 12
    assert not {q.qid for q in test} & {q.qid for q in cal}
    for part in (test, cal):
        assert sum(q.dataset == "gsm8k" for q in part) == 6
    mmlu = [q for q in all_q if q.dataset == "mmlu"]
    assert all(len(q.choices) == 4 and q.gold in "ABCD" and q.subject for q in mmlu)


# official MMLU categories (hendrycks/test categories.py) for the fixture's subjects
_FIXTURE_GROUPS = {
    "high_school_physics": "stem", "high_school_biology": "stem", "high_school_mathematics": "stem",
    "computer_security": "stem", "philosophy": "humanities", "high_school_world_history": "humanities",
    "world_religions": "humanities", "high_school_macroeconomics": "social", "high_school_psychology": "social",
    "nutrition": "other", "clinical_knowledge": "other", "marketing": "other",
}


def test_mock_fixture_has_router_signal() -> None:
    all_q = data._fixture_questions()
    mmlu = [q for q in all_q if q.dataset == "mmlu"]
    assert {_FIXTURE_GROUPS[q.subject] for q in mmlu} == {"stem", "humanities", "social", "other"}
    gsm = [q for q in all_q if q.dataset == "gsm8k"]
    lengths = sorted(len(q.question) for q in gsm)
    assert lengths[-1] > 3 * lengths[0]
    assert len({len(re.findall(r"\d+", q.question)) for q in gsm}) >= 3
