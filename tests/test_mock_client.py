from __future__ import annotations

from pathlib import Path

from eclm.clients import MockClient, make_client, mock_accuracy
from eclm.config import load_settings
from eclm.grading import extract_answer
from eclm.prompts import build_prompt
from eclm.types import Question


def _q(i: int) -> Question:
    return Question(qid=f"t-{i}", dataset="gsm8k", question=f"What is {i} + {i}?", gold=str(2 * i))


def test_real_prompts_never_contain_the_gold_marker() -> None:
    q = _q(3)
    assert "[[gold:" not in build_prompt(q)
    assert "[[gold:6]]" in build_prompt(q, mock=True)


def test_temperature_zero_is_deterministic() -> None:
    a = MockClient("m", "edge", 0.5, seed=1)
    b = MockClient("m", "edge", 0.5, seed=1)
    for i in range(20):
        p = build_prompt(_q(i), mock=True)
        first = a.generate(p, temperature=0.0, max_tokens=512)
        assert first == a.generate(p, temperature=0.0, max_tokens=512)
        assert first == b.generate(p, temperature=0.0, max_tokens=512)


def test_samples_at_positive_temperature_can_disagree() -> None:
    c = MockClient("m", "edge", 0.5, seed=1)
    disagreements = 0
    for i in range(30):
        p = build_prompt(_q(i), mock=True)
        answers = {extract_answer(c.generate(p, temperature=0.7, max_tokens=512).text, "gsm8k") for _ in range(3)}
        disagreements += len(answers) > 1
    assert disagreements > 0


def test_accuracy_close_to_configured_value() -> None:
    for name, where in (("qwen2.5:3b", "edge"), ("qwen2.5:0.5b", "edge"), ("cloud-x", "cloud")):
        c = MockClient(name, where, mock_accuracy(name, where), seed=7)  # type: ignore[arg-type]
        n = 300
        right = sum(
            extract_answer(c.generate(build_prompt(_q(i), mock=True), temperature=0.0, max_tokens=512).text, "gsm8k")
            == str(2 * i)
            for i in range(n)
        )
        assert abs(right / n - mock_accuracy(name, where)) <= 0.1


def test_mock_accuracy_by_name() -> None:
    assert mock_accuracy("anything", "cloud") == 0.9
    assert mock_accuracy("qwen2.5:0.5b", "edge") == 0.45
    assert mock_accuracy("llama3.2:1b", "edge") == 0.55
    assert mock_accuracy("qwen2.5:3b", "edge") == 0.65


def test_without_marker_the_answer_is_unparseable() -> None:
    gen = MockClient("m", "cloud", 1.0, seed=1).generate("plain prompt", temperature=0.0, max_tokens=50)
    assert extract_answer(gen.text, "gsm8k") is None
    assert gen.ran_on == "cloud" and 0 <= gen.ttft_s <= gen.total_s


def test_make_client_names_mock_after_requested_model(workdir: Path) -> None:
    s = load_settings()
    edge = make_client("edge", s, model="llama3.2:1b", mock=True)
    cloud = make_client("cloud", s, mock=True)
    assert (edge.name, edge.ran_on) == ("llama3.2:1b", "edge")
    assert (cloud.name, cloud.ran_on) == ("llama-3.3-70b-versatile", "cloud")
