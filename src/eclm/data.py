"""Question sets: load/save JSONL and the `python -m eclm.data` command.

T001: only --mock is implemented (writes the bundled fixture split). T003 adds the real download.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

from eclm.config import ConfigError, load_settings
from eclm.types import Question

TEST_FILE = "test_questions.jsonl"
CALIBRATION_FILE = "calibration_questions.jsonl"


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
    if not args.mock:
        print("error: downloading the real datasets is not implemented yet (T003)", file=sys.stderr)
        return 1
    test, calibration = mock_questions()
    save_questions(test, settings.data_dir / TEST_FILE)
    save_questions(calibration, settings.data_dir / CALIBRATION_FILE)
    print(f"wrote {len(test)} test and {len(calibration)} calibration questions to {settings.data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
