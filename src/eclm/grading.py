"""Answer extraction and grading. Minimal version (T001); T002 makes it robust."""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation

_ANSWER_LINE = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_LETTER = re.compile(r"\b([A-D])\b")


def _normalise_number(raw: str) -> str | None:
    try:
        value = Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def extract_answer(text: str, dataset: str, choices: tuple[str, ...] = ()) -> str | None:
    """Return the normalised final answer, or None if it can't be found."""
    matches = _ANSWER_LINE.findall(text)
    if not matches:
        return None
    tail = matches[-1].strip()
    if dataset == "gsm8k":
        num = _NUMBER.search(tail)
        return _normalise_number(num.group(0)) if num else None
    letter = _LETTER.search(tail.upper())
    return letter.group(1) if letter else None


def is_correct(predicted: str | None, gold: str, dataset: str) -> bool:
    if predicted is None:
        return False
    if dataset == "gsm8k":
        try:
            return abs(Decimal(predicted) - Decimal(gold)) < Decimal("1e-6")
        except InvalidOperation:
            return False
    return predicted.strip().upper() == gold.strip().upper()


def answers_agree(answers: list[str | None], dataset: str, min_agree: int) -> str | None:
    """Majority answer if at least min_agree answers are equal; None never counts.

    Ties pick the answer that appeared first.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    best = max(counts.values())
    if best < min_agree:
        return None
    for a in valid:
        if counts[a] == best:
            return a
    return None
