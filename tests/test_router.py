from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from eclm import router
from eclm.clients import ClientError, MockClient, make_client
from eclm.config import load_settings
from eclm.data import CALIBRATION_FILE, TEST_FILE, mock_questions, save_questions
from eclm.router import FEATURES, SUBJECT_GROUPS, RouterModel, edge_labels, features, train_router
from eclm.types import Generation, Question

REPO = Path(__file__).resolve().parents[1]


def gsm(qid: str, text: str, gold: str = "1") -> Question:
    return Question(qid=qid, dataset="gsm8k", question=text, gold=gold)


def mmlu(qid: str, text: str, subject: str, choices: tuple[str, ...] = ("a", "b", "c", "d")) -> Question:
    return Question(qid=qid, dataset="mmlu", question=text, gold="A", choices=choices, subject=subject)


# --- AC1: features ---------------------------------------------------------------------------------


def test_features_gsm8k_hand_checked() -> None:
    q = gsm("g", "Tom buys 3 apples at $1,250.5 each and pays 20 percent tax. Half of 12 is 6 - 2 + 1.")
    # words: whitespace tokens = 21; numbers: 3, 1,250.5, 20, 12, 6, 2, 1 = 7
    # ops: '-' '+' (2 symbols; ',' and '.' are not ops) + each, percent, Half (3 words) = 5
    # longest number 1,250.5 has 5 digits
    assert features(q) == [21.0, 7.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]


def test_features_mmlu_hand_checked() -> None:
    q = mmlu("m", "What is 2 × 3 per unit?", "high_school_physics", ("six units", "5", "four", "none of these"))
    # words: 7 in the question (× is a token) + 2 + 1 + 1 + 3 in the choices = 14; numbers in the question only: 2, 3
    # ops: × + per = 2; STEM group; longest number 1 digit
    assert features(q) == [14.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_features_mmlu_no_numbers_unknown_subject() -> None:
    q = mmlu("m", "Which philosopher wrote Leviathan?", "not_a_real_subject", ("Hobbes", "Locke", "Hume", "Kant"))
    assert features(q) == [8.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_features_length_and_types() -> None:
    for q in [*mock_questions()[0], *mock_questions()[1]]:
        f = features(q)
        assert len(f) == len(FEATURES)
        assert all(isinstance(v, float) for v in f)
        groups = f[4:8]
        assert sum(groups) == (1.0 if q.dataset == "mmlu" else 0.0)


def test_math_op_words_are_whole_words_case_insensitive() -> None:
    assert features(gsm("g", "TWICE per each Times percentage perhaps"))[2] == 4.0


@pytest.mark.parametrize(
    "subject, group",
    [
        ("abstract_algebra", "stem"),
        ("high_school_physics", "stem"),
        ("philosophy", "humanities"),
        ("professional_law", "humanities"),
        ("high_school_macroeconomics", "social"),
        ("human_sexuality", "social"),
        ("nutrition", "other"),
        ("marketing", "other"),
        ("global_facts", "other"),
    ],
)
def test_subject_groups_follow_official_mapping(subject: str, group: str) -> None:
    assert SUBJECT_GROUPS[subject] == group


def test_subject_groups_cover_all_57_subjects() -> None:
    assert len(SUBJECT_GROUPS) == 57
    assert set(SUBJECT_GROUPS.values()) == {"stem", "humanities", "social", "other"}
    counts = {g: list(SUBJECT_GROUPS.values()).count(g) for g in ("stem", "humanities", "social", "other")}
    assert counts == {"stem": 18, "humanities": 13, "social": 12, "other": 14}


# --- AC2: training -----------------------------------------------------------------------------------


class ScriptedEdge:
    """Wrong (answers 'Answer: 999999') on long questions, right on short ones."""

    name = "scripted-edge"
    ran_on = "edge"

    def __init__(self, golds: dict[str, str], long_words: int = 15) -> None:
        self.golds = golds
        self.long_words = long_words
        self.prompts: list[str] = []

    def check(self) -> None:
        return None

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> Generation:
        self.prompts.append(prompt)
        text = prompt.split("\n\n", 1)[1]
        gold = self.golds[text]
        answer = "999999" if len(text.split()) > self.long_words else gold
        return Generation(f"Answer: {answer}", 0.1, 0.2, 3, 15.0, self.name, "edge")


def _calibration() -> list[Question]:
    qs = []
    for i in range(10):
        qs.append(gsm(f"s{i}", f"Add {i} and {i + 1}.", str(2 * i + 1)))
        long = " ".join(["word"] * (20 + i)) + f" then add {i} and {i + 2}."
        qs.append(gsm(f"l{i}", long, str(2 * i + 2)))
    return qs


def test_training_scores_long_questions_harder() -> None:
    settings = load_settings(REPO / "config.yaml")
    cal = _calibration()
    edge = ScriptedEdge({q.question: q.gold for q in cal})
    model = train_router(settings, edge, cal)
    assert len(edge.prompts) == len(cal)
    assert model.calibration_accuracy == 0.5
    assert model.train_accuracy == 1.0
    assert model.edge_model == "scripted-edge"
    short, long = gsm("x", "Add 4 and 9."), gsm("y", " ".join(["word"] * 30) + " then add 4 and 9.")
    assert model.predict_hard(long) > model.predict_hard(short)
    assert 0.0 <= model.predict_hard(short) <= 1.0


def test_client_error_and_parse_failure_are_hard() -> None:
    settings = load_settings(REPO / "config.yaml")

    class Flaky:
        name, ran_on = "flaky", "edge"

        def check(self) -> None:
            return None

        def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> Generation:
            assert temperature == 0.0
            if "boom" in prompt:
                raise ClientError("timeout")
            text = "no idea" if "unparsable" in prompt else "Answer: 1"
            return Generation(text, 0.1, 0.2, 1, 1.0, "flaky", "edge")

    qs = [gsm("a", "easy one"), gsm("b", "boom"), gsm("c", "unparsable"), gsm("d", "easy two", gold="2")]
    assert edge_labels(settings, Flaky(), qs) == [0, 1, 1, 1]


def test_single_class_raises_clear_error() -> None:
    settings = load_settings(REPO / "config.yaml")
    cal = [gsm(f"s{i}", f"Add {i}.", "1") for i in range(4)]
    edge = ScriptedEdge({q.question: q.gold for q in cal})
    with pytest.raises(ValueError, match="all 4 calibration questions right"):
        train_router(settings, edge, cal)


def test_mock_calibration_has_both_classes() -> None:
    settings = load_settings(REPO / "config.yaml")
    _, cal = mock_questions()
    for model in settings.edge.models:
        labels = edge_labels(settings, make_client("edge", settings, model=model, mock=True), cal, mock=True)
        assert set(labels) == {0, 1}, model


# --- AC3: save/load ---------------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    settings = load_settings(REPO / "config.yaml")
    _, cal = mock_questions()
    edge = MockClient("qwen2.5:3b", "edge", 0.65, settings.seed)
    model = train_router(settings, edge, cal, mock=True)
    path = tmp_path / "models" / "router_x.joblib"
    model.save(path)
    loaded = RouterModel.load(path)
    probes = [*mock_questions()[0], *cal]
    assert [loaded.predict_hard(q) for q in probes] == [model.predict_hard(q) for q in probes]
    assert loaded.edge_model == "qwen2.5:3b"
    assert loaded.calibration_accuracy == model.calibration_accuracy
    assert loaded.trained_at == model.trained_at
    assert list(loaded.features) == FEATURES


def test_load_rejects_other_features(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "r.joblib"
    joblib.dump({"model": None, "features": ["n_words"], "edge_model": "m", "trained_at": "t",
                 "calibration_accuracy": 0.5}, path)
    with pytest.raises(ValueError, match="retrain"):
        RouterModel.load(path)


# --- AC4 + CLI ----------------------------------------------------------------------------------------


def _write_calibration_only(workdir: Path) -> None:
    _, cal = mock_questions()
    save_questions(cal, workdir / "data" / CALIBRATION_FILE)


def test_cli_trains_without_test_file(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_calibration_only(workdir)
    assert not (workdir / "data" / TEST_FILE).exists()
    assert router.main(["train", "--edge-model", "qwen2.5:3b", "--mock"]) == 0
    out = capsys.readouterr().out
    assert "calibration questions: 12" in out
    assert "edge accuracy on calibration" in out
    assert "router training accuracy" in out
    assert "share predicted hard (score > 0.6)" in out
    assert (workdir / "models" / "router_qwen2.5-3b.joblib").is_file()


def test_training_never_opens_test_file(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_calibration_only(workdir)
    (workdir / "data" / TEST_FILE).write_text("poison\n", encoding="utf-8")
    opened: list[str] = []
    real_load = router.load_questions

    def spy(path: Path) -> list[Question]:
        opened.append(Path(path).name)
        return real_load(path)

    monkeypatch.setattr(router, "load_questions", spy)
    assert router.main(["train", "--edge-model", "qwen2.5:3b", "--mock"]) == 0
    assert opened == [CALIBRATION_FILE]


def test_cli_missing_calibration_exits_1(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert router.main(["train", "--edge-model", "qwen2.5:3b", "--mock"]) == 1
    assert "run python -m eclm.data first" in capsys.readouterr().err
    assert not (workdir / "models").exists()


def test_cli_usage_errors_exit_1(workdir: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        router.main(["train"])
    assert exc.value.code == 1
    with pytest.raises(SystemExit) as exc:
        router.main([])
    assert exc.value.code == 1


def test_cli_edge_unavailable_exits_1(workdir: Path, monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    from eclm.clients import ClientUnavailable

    _write_calibration_only(workdir)

    class Down:
        name, ran_on = "m", "edge"

        def check(self) -> None:
            raise ClientUnavailable("Ollama is not running")

    monkeypatch.setattr(router, "make_client", lambda *a, **k: Down())
    assert router.main(["train", "--edge-model", "m"]) == 1
    assert "Ollama is not running" in capsys.readouterr().err


def test_build_parser_exposed() -> None:
    args = router.build_parser().parse_args(["train", "--edge-model", "m", "--mock"])
    assert (args.command, args.edge_model, args.mock, args.config) == ("train", "m", True, "config.yaml")


def test_module_runs_as_script(workdir: Path) -> None:
    _write_calibration_only(workdir)
    proc = subprocess.run([sys.executable, "-m", "eclm.router", "train", "--edge-model", "llama3.2:1b", "--mock"],
                          cwd=workdir, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert (workdir / "models" / "router_llama3.2-1b.joblib").is_file()
