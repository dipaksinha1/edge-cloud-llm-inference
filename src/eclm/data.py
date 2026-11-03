"""Question sets: build (download + seeded split), load/save JSONL and the `python -m eclm.data` command.

Sources (Hugging Face `datasets`): GSM8K `openai/gsm8k` (config `main`): test from `test`, calibration from
`train`. MMLU `cais/mmlu` (config `all`): test from `test`, calibration from `validation` (+ `dev` if needed).
qid = `<dataset>-<split>-<row>`. Selection is `random.Random(seed).sample(sorted indices, k)`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Callable
from importlib import resources
from pathlib import Path

from eclm.config import ConfigError, Settings, load_settings
from eclm.grading import _normalise, _to_decimal
from eclm.types import Dataset, Question

TEST_FILE = "test_questions.jsonl"
CALIBRATION_FILE = "calibration_questions.jsonl"

GSM8K_SOURCE = ("openai/gsm8k", "main")
MMLU_SOURCE = ("cais/mmlu", "all")
_LETTERS = "ABCD"
_GSM8K_GOLD = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)


class DatasetError(Exception):
    """A dataset could not be loaded or does not have enough usable questions."""


Row = dict[str, object]
_Convert = Callable[[str, int, Row], Question]


def _load_split(path: str, config: str, split: str) -> list[Row]:
    """Download one split with Hugging Face `datasets`. Tests monkeypatch this function."""
    from datasets import load_dataset  # imported lazily: slow, and not needed for --mock

    return [dict(r) for r in load_dataset(path, config, split=split)]


def _to_dict(q: Question) -> dict[str, object]:
    return {
        "qid": q.qid,
        "dataset": q.dataset,
        "question": q.question,
        "gold": q.gold,
        "choices": list(q.choices),
        "subject": q.subject,
    }


def _from_dict(d: dict[str, object]) -> Question:
    choices = d.get("choices") or []
    if not isinstance(choices, list):
        raise ValueError(f"question {d.get('qid')}: 'choices' must be a list")
    return Question(
        qid=str(d["qid"]),
        dataset=str(d["dataset"]),  # type: ignore[arg-type]
        question=str(d["question"]),
        gold=str(d["gold"]),
        choices=tuple(str(c) for c in choices),
        subject=str(d.get("subject") or ""),
    )


def load_questions(path: Path) -> list[Question]:
    with open(path, encoding="utf-8") as f:
        return [_from_dict(json.loads(line)) for line in f if line.strip()]


def save_questions(qs: list[Question], path: Path) -> None:
    """Write JSONL atomically (temp file, then rename) so a failure leaves old files intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for q in qs:
            f.write(json.dumps(_to_dict(q), sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _fixture_questions() -> list[Question]:
    text = resources.files("eclm").joinpath("fixtures/mock_questions.jsonl").read_text(encoding="utf-8")
    return [_from_dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def mock_questions() -> tuple[list[Question], list[Question]]:
    """Fixed, disjoint (test, calibration) split of the fixture: half each, each half GSM8K/half MMLU."""
    test: list[Question] = []
    calibration: list[Question] = []
    for dataset in ("gsm8k", "mmlu"):
        group = [q for q in _fixture_questions() if q.dataset == dataset]
        half = len(group) // 2
        test.extend(group[:half])
        calibration.extend(group[half:])
    return test, calibration


def _gsm8k_gold(answer: str) -> str:
    m = _GSM8K_GOLD.search(answer)
    if m is None:
        raise ValueError("no '####' in answer")
    value = _to_decimal(m.group(1).replace("$", "").strip())
    if value is None or not value.is_finite():
        raise ValueError(f"gold is not a number: {m.group(1)!r}")
    return _normalise(value)  # same normalisation as graded predictions


def _gsm8k_question(split: str, row: int, r: Row) -> Question:
    return Question(
        qid=f"gsm8k-{split}-{row}",
        dataset="gsm8k",
        question=str(r["question"]).strip(),
        gold=_gsm8k_gold(str(r["answer"])),
    )


def _mmlu_question(split: str, row: int, r: Row) -> Question:
    choices = r["choices"]
    answer = r["answer"]
    if not isinstance(choices, (list, tuple)) or len(choices) != 4 or not isinstance(answer, int):
        raise ValueError("expected 4 choices and an integer answer")
    return Question(
        qid=f"mmlu-{split}-{row}",
        dataset="mmlu",
        question=str(r["question"]).strip(),
        gold=_LETTERS[answer],
        choices=tuple(str(c) for c in choices),
        subject=str(r.get("subject") or ""),
    )


def _text_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def _pool(
    dataset: Dataset,
    source: tuple[str, str],
    splits: tuple[str, ...],
    k: int,
    convert: _Convert,
    exclude: set[str],
) -> list[Question]:
    """Usable questions from `splits` in order, stopping once there are at least k.

    Questions whose normalised text is in `exclude`, or repeats an earlier one, are dropped.
    """
    out: list[Question] = []
    seen = set(exclude)
    for split in splits:
        if len(out) >= k:
            break
        try:
            rows = _load_split(source[0], source[1], split)
        except Exception as exc:  # any loader/network failure
            raise DatasetError(f"cannot load {dataset} ({source[0]}/{source[1]}, split {split}): {exc}") from exc
        for i, r in enumerate(rows):
            try:
                q = convert(split, i, r)
            except (KeyError, ValueError, IndexError, TypeError) as exc:
                raise DatasetError(f"{dataset} {split} row {i}: unexpected format ({exc})") from exc
            key = _text_key(q.question)
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
    return out


def _select(pool: list[Question], k: int, seed: int, what: str) -> list[Question]:
    if k > len(pool):
        raise DatasetError(f"{what}: need {k} questions but only {len(pool)} are usable")
    picked = random.Random(seed).sample(range(len(pool)), k)
    return [pool[i] for i in sorted(picked)]


def build_datasets(settings: Settings) -> tuple[list[Question], list[Question]]:
    """(test, calibration): seeded, GSM8K first then MMLU, no qid or question text in both sets."""
    counts = settings.dataset
    seed = settings.seed
    test: list[Question] = []
    calibration: list[Question] = []
    plan: tuple[tuple[Dataset, tuple[str, str], str, tuple[str, ...], int, int, _Convert], ...] = (
        ("gsm8k", GSM8K_SOURCE, "test", ("train",), counts.gsm8k_test, counts.gsm8k_calibration, _gsm8k_question),
        ("mmlu", MMLU_SOURCE, "test", ("validation", "dev"), counts.mmlu_test, counts.mmlu_calibration, _mmlu_question),
    )
    for dataset, source, test_split, cal_splits, n_test, n_cal, convert in plan:
        test_pool = _pool(dataset, source, (test_split,), n_test, convert, set())
        chosen = _select(test_pool, n_test, seed, f"{dataset} test")
        taken = {_text_key(q.question) for q in chosen}
        cal_pool = _pool(dataset, source, cal_splits, n_cal, convert, taken)
        test.extend(chosen)
        calibration.extend(_select(cal_pool, n_cal, seed, f"{dataset} calibration"))
    return test, calibration


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m eclm.data", description="Build the question sets.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--mock", action="store_true", help="write the bundled fixture split (no network)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.mock:
        test, calibration = mock_questions()
    else:
        try:
            test, calibration = build_datasets(settings)
        except DatasetError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    save_questions(test, settings.data_dir / TEST_FILE)
    save_questions(calibration, settings.data_dir / CALIBRATION_FILE)
    print(f"wrote {len(test)} test and {len(calibration)} calibration questions to {settings.data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
