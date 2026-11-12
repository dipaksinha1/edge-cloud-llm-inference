from __future__ import annotations

import pytest

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


# ---- robust extraction ----

GSM8K_CASES = [
    ("48 + 24 = 72\nAnswer: 72", "72"),
    ("Total cost.\nanswer: $1,200.", "1200"),
    ("**Answer:** 72", "72"),
    ("Answer**: 72", "72"),
    ("She sold 48 then 24. The answer is 72.", "72"),
    ("so 48 and 24 make 72", "72"),
    ("Answer: 72.0", "72"),
    ("Answer: 72.50", "72.5"),
    ("He is short by 5 dollars.\nAnswer: -5", "-5"),
    ("Answer: 3/4", "0.75"),
    ("Answer: 50%", "50"),
    ("Answer: 10\nWait, recheck.\nAnswer: 12", "12"),
    ("Answer: 1,234,567 people", "1234567"),
    ("Answer:\n42", "42"),
    ("The final answer is $18.", "18"),
    ("Answer: 0.5", "0.5"),
    ("Answer: 1/3", "0.333333"),
    # review regressions: later prose must not override an explicit Answer: line
    ("Answer: 72\nThis answer is based on 3 boxes.", "72"),
    ("Answer: 72\nI hope this answer is clear: 3 boxes of 24.", "72"),
    # equations on the answer line
    ("Answer: 48 + 24 = 72", "72"),
    ("Answer: 5 x 12 = 60 minutes", "60"),
    ("Answer: 2/3 of 9 = 6", "6"),
    # signs and leading dots
    ("Answer: -$5", "-5"),
    ("Answer: \u22125", "-5"),
    ("Answer: .5", "0.5"),
    # review round 2: plain answer followed by working
    ("Answer: 18 (3 x 6 = 18 per day; 18 x 7 = 126 per week)", "18"),
    ("Answer: 12 hours = 0.5 days", "12"),
    ("Answer: 30 = 15 + 15", "30"),
    ("Answer: 2,000 - 500 = 1,500", "1500"),
    ("Pages 10\u201312 were read, so 3 pages", "3"),
]


@pytest.mark.parametrize("text,expected", GSM8K_CASES)
def test_gsm8k_extraction(text: str, expected: str) -> None:
    assert extract_answer(text, "gsm8k") == expected


CHOICES = ("Speed", "Mass", "Velocity", "Temperature")
MMLU_CASES = [
    ("Velocity has direction.\nAnswer: C", "C"),
    ("Answer: (C)", "C"),
    ("answer: c", "C"),
    ("The correct answer is C. Velocity", "C"),
    ("Speed has no direction but velocity does.\nC) Velocity", "C"),
    ("Answer: Velocity", "C"),
    ("Answer: velocity.", "C"),
    ("**Answer:** B", "B"),
    ("Answer: it is a B", "B"),
    ("Answer: A\nActually no.\nAnswer: D", "D"),
    ("I pick (B).", "B"),
    ("Answer: C. Velocity, because it has direction", "C"),
    # review regressions: the article "a" is not choice A, later prose doesn't override Answer:
    ("The answer is a vector quantity, so C", "C"),
    ("Answer: a vector quantity, which is velocity (C)", "C"),
    ("Answer: Not A, it is C", "C"),
    ("Answer: C\nThis answer is based on option B being wrong.", "C"),
    ("Answer: d) temperature", "D"),
    # review round 2: negated / rejected options are not the answer
    ("Answer: The best option is C, not A", "C"),
    ("Answer: Option C (not B)", "C"),
    ("Answer: I choose C over D", "C"),
]


@pytest.mark.parametrize("text,expected", MMLU_CASES)
def test_mmlu_extraction(text: str, expected: str) -> None:
    assert extract_answer(text, "mmlu", CHOICES) == expected


@pytest.mark.parametrize(
    "text,dataset",
    [
        ("", "gsm8k"),
        ("I am not sure.", "gsm8k"),
        ("I am not sure.", "mmlu"),
        ("Answer: none of these", "mmlu"),
        ("The answer is a vector, velocity.", "mmlu"),
        ("Answer: maybe C or D", "mmlu"),
    ],
)
def test_unparseable_is_none_and_wrong(text: str, dataset: str) -> None:
    got = extract_answer(text, dataset, CHOICES)
    assert got is None
    assert is_correct(got, "C" if dataset == "mmlu" else "72", dataset) is False


def test_numeric_correctness() -> None:
    assert is_correct("72", "72.0", "gsm8k")
    assert is_correct("0.75", "0.750", "gsm8k")
    assert not is_correct("abc", "72", "gsm8k")


def test_agreement_rules() -> None:
    assert answers_agree(["72", "72.0", "68"], "gsm8k", 2) == "72"
    assert answers_agree(["A", None, None], "mmlu", 2) is None
    assert answers_agree(["5", "7", "5", "7"], "gsm8k", 2) == "5"  # tie -> earliest
    assert answers_agree([None, None, None], "gsm8k", 2) is None
    assert answers_agree(["b", "B", "C"], "mmlu", 2) == "b"
