"""Answer extraction and grading (SPEC R8, assumption A3; ARCHITECTURE §5).

Small models format answers messily, so extraction is deliberately tolerant:
Markers: the last colon-style "Answer:" wins; prose "answer is" is used only when there is no "Answer:" at all.
- GSM8K: the first number after the marker; if the answer line is an equation ("48 + 24 = 72"), the number after
  the last "="; with no marker, the last number in the text.
  "$1,200." -> "1200", "72.0" -> "72", "3/4" -> "0.75", "50%" -> "50", "-$5" -> "-5".
- MMLU: a letter leading the answer line ("C", "(C)", "c", "C) Velocity"), else an answer naming a choice's text,
  else the one capital A-D on the line that isn't negated ("not A", "over D"); several candidates -> None.
  With no marker, the last "C)" / "(C)" / "C." / lone "C".
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation

from eclm.prompts import LETTERS

# Colon markers ("Answer: 72", "**Answer:** 72", "Answer**: 72") always win over prose markers
# ("final answer is 72", "The correct answer is C."), so later prose can't override an explicit Answer: line.
_COLON_MARKER = re.compile(r"answer\**\s*[:：]\s*\**\s*", re.IGNORECASE)
_IS_MARKER = re.compile(r"answer\**\s*\bis\b\s*\**\s*", re.IGNORECASE)
_NUMBER = re.compile(r"(-?)\$?(\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*/\s*(\d+(?:\.\d+)?))?")
_UNICODE_MINUS = re.compile(r"\u2212|(?<!\d)\u2013")  # en dash only when it isn't a range like 10–12
_STARTS_WITH_OPERATOR = re.compile(r"\s*(?:[+\-x×*/÷]|of\b)", re.IGNORECASE)
_NEGATED = re.compile(
    r"(?:\bnot|\bover|\bunlike|\bexcept|\brather than|\binstead of)\s*(?:option\s*)?\(?$", re.IGNORECASE
)
# uppercase letter as a token, or a lowercase letter only when it is the whole token ("c", "c.", "(c)")
# so the article "a" in "Answer: a vector ..." is not read as choice A
_LEADING_LETTER = re.compile(r"^[\s*\"'`]*(?:\(?([A-D])\)?(?![A-Za-z0-9])|\(?([a-d])(?:\)|[.:]|\s*$))")
_CAPITAL_LETTER = re.compile(r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])")
_TRAILING_LETTER = re.compile(r"(?:^|[\s(\[])\(?([A-D])(?:[).:\]]|\s*$)", re.MULTILINE)


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def _normalise(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def _number_from(match: re.Match[str]) -> str | None:
    num = _to_decimal(match.group(1) + match.group(2))
    if num is None:
        return None
    if match.group(3):
        den = _to_decimal(match.group(3))
        if den is None or den == 0:
            return None
        num = num / den
        # keep a sensible precision for repeating fractions (1/3 -> 0.333333)
        num = num.quantize(Decimal("0.000001")) if num != num.to_integral_value() else num
    return _normalise(num)


def _after_last_marker(text: str) -> str | None:
    for marker in (_COLON_MARKER, _IS_MARKER):
        matches = list(marker.finditer(text))
        if matches:
            return text[matches[-1].end():]
    return None


def _first_line(tail: str) -> str:
    stripped = tail.strip()
    return stripped.splitlines()[0] if stripped else ""


def _extract_gsm8k(text: str) -> str | None:
    text = _UNICODE_MINUS.sub("-", text)
    tail = _after_last_marker(text)
    if tail is not None:
        line = _first_line(tail)
        first = _NUMBER.search(line)
        # "Answer: 48 + 24 = 72" -> 72, but "Answer: 18 (3 x 6 = 18; ...)" -> 18
        if "=" in line and first and _STARTS_WITH_OPERATOR.match(line, first.end()):
            m = _NUMBER.search(line.rsplit("=", 1)[1])
            if m:
                return _number_from(m)
        m = _NUMBER.search(tail)
        if m:
            return _number_from(m)
    matches = list(_NUMBER.finditer(text))
    return _number_from(matches[-1]) if matches else None


def _clean(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _extract_mmlu(text: str, choices: tuple[str, ...]) -> str | None:
    tail = _after_last_marker(text)
    if tail is not None:
        first_line = _first_line(tail)
        m = _LEADING_LETTER.match(first_line)
        if m:
            return (m.group(1) or m.group(2)).upper()
        if choices and first_line:
            said = _clean(first_line)
            for i, choice in enumerate(choices[: len(LETTERS)]):
                c = _clean(choice)
                if c and (said == c or said.startswith(c + " ") or said.rstrip(" ") == c):
                    return LETTERS[i]
        candidates = {
            m3.group(1)
            for m3 in _CAPITAL_LETTER.finditer(first_line)
            if not _NEGATED.search(first_line[: m3.start()])
        }
        if len(candidates) == 1:  # "Not A, it is C" -> C; "C or D" -> None (ambiguous)
            return candidates.pop()
        if candidates:
            return None
    found = _TRAILING_LETTER.findall(text)
    return found[-1] if found else None


def extract_answer(text: str, dataset: str, choices: tuple[str, ...] = ()) -> str | None:
    """Return the normalised final answer, or None if it can't be found."""
    if not text:
        return None
    if dataset == "gsm8k":
        return _extract_gsm8k(text)
    return _extract_mmlu(text, choices)


def is_correct(predicted: str | None, gold: str, dataset: str) -> bool:
    if predicted is None:
        return False
    if dataset == "gsm8k":
        p, g = _to_decimal(predicted), _to_decimal(gold)
        if p is None or g is None:
            return False
        return abs(p - g) < Decimal("1e-6")
    return predicted.strip().upper() == gold.strip().upper()


def answers_agree(answers: list[str | None], dataset: str, min_agree: int) -> str | None:
    """Majority answer if at least min_agree answers are equal; None never counts.

    GSM8K answers are compared numerically ("72" == "72.0"). Ties pick the answer that appeared first.
    """
    keyed: list[tuple[str, str]] = []
    for a in answers:
        if a is None:
            continue
        key = a.strip().upper()
        if dataset == "gsm8k":
            num = _to_decimal(a)
            if num is None:
                continue
            key = _normalise(num)
        keyed.append((key, a))
    if not keyed:
        return None
    counts = Counter(k for k, _ in keyed)
    best = max(counts.values())
    if best < min_agree:
        return None
    for key, original in keyed:
        if counts[key] == best:
            return key if dataset == "gsm8k" else original
    return None
