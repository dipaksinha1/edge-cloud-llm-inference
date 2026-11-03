from __future__ import annotations

from eclm.grading import answers_agree, extract_answer, is_correct


def test_gsm8k_answer_line() -> None:
    assert extract_answer("so 48 + 24\nAnswer: 72", "gsm8k") == "72"
    assert extract_answer("Answer: 1,200", "gsm8k") == "1200"
    assert extract_answer("Answer: 72.0", "gsm8k") == "72"


def test_mmlu_answer_line() -> None:
    assert extract_answer("Velocity has direction.\nAnswer: C", "mmlu") == "C"


def test_no_answer_is_none_and_incorrect() -> None:
    assert extract_answer("no idea", "gsm8k") is None
    assert is_correct(None, "72", "gsm8k") is False


def test_is_correct() -> None:
    assert is_correct("72", "72", "gsm8k")
    assert not is_correct("71", "72", "gsm8k")
    assert is_correct("c", "C", "mmlu")


def test_answers_agree() -> None:
    assert answers_agree(["72", "72", "68"], "gsm8k", 2) == "72"
    assert answers_agree(["72", "68", "5"], "gsm8k", 2) is None
    assert answers_agree(["A", None, None], "mmlu", 2) is None
