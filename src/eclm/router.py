"""Stage-1 router: cheap question features, the difficulty model and `python -m eclm.router train` (R5, ADR 0004).

The difficulty model is a logistic regression trained on the calibration set only, labelled 1 when the
edge model answers the question wrongly (wrong answer, parse failure or a failed call) and 0 otherwise.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import joblib
from sklearn.linear_model import LogisticRegression

from eclm.clients import ClientError, ClientUnavailable, ModelClient, make_client
from eclm.config import ConfigError, Settings, load_settings
from eclm.data import CALIBRATION_FILE, load_questions
from eclm.grading import extract_answer, is_correct
from eclm.prompts import build_prompt
from eclm.results import safe_name
from eclm.types import Question

log = logging.getLogger("eclm.router")

FEATURES = [
    "n_words",
    "n_numbers",
    "n_math_ops",
    "is_mmlu",
    "subject_stem",
    "subject_humanities",
    "subject_social",
    "subject_other",
    "max_number_digits",
]

# MMLU subject -> group, following the official mapping in hendrycks/test categories.py
# (subcategories -> categories "STEM", "humanities", "social sciences", "other (business, health, misc.)").
SUBJECT_GROUPS: dict[str, str] = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social",
    "high_school_government_and_politics": "social",
    "high_school_macroeconomics": "social",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social",
    "high_school_physics": "stem",
    "high_school_psychology": "social",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social",
    "public_relations": "social",
    "security_studies": "social",
    "sociology": "social",
    "us_foreign_policy": "social",
    "virology": "other",
    "world_religions": "humanities",
}
_GROUPS = ("stem", "humanities", "social", "other")

# A number: digits with optional thousands separators and decimals ("1,250.75", "3.5", "40").
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_OP_SYMBOLS = re.compile(r"[+\-*/%×]")
_OP_WORDS = re.compile(r"\b(?:percent|twice|half|times|each|per)\b", re.IGNORECASE)


def subject_group(subject: str) -> str:
    """stem | humanities | social | other; unknown subjects are `other`."""
    return SUBJECT_GROUPS.get(subject.strip().lower(), "other")


def features(q: Question) -> list[float]:
    """Cheap difficulty features of a question, in FEATURES order (no model calls)."""
    text = q.question
    is_mmlu = q.dataset == "mmlu"
    words = len(text.split()) + (sum(len(c.split()) for c in q.choices) if is_mmlu else 0)
    numbers = _NUMBER.findall(text)
    ops = len(_OP_SYMBOLS.findall(text)) + len(_OP_WORDS.findall(text))
    group = subject_group(q.subject) if is_mmlu else None
    longest = max((sum(ch.isdigit() for ch in n) for n in numbers), default=0)
    return [
        float(words),
        float(len(numbers)),
        float(ops),
        1.0 if is_mmlu else 0.0,
        *(1.0 if group == g else 0.0 for g in _GROUPS),
        float(longest),
    ]


@dataclass
class RouterModel:
    """The trained stage-1 difficulty model. predict_hard(q) = P(edge model gets q wrong)."""

    model: LogisticRegression
    edge_model: str
    trained_at: str
    calibration_accuracy: float
    features: tuple[str, ...] = tuple(FEATURES)
    train_accuracy: float | None = None  # accuracy on its own training labels; reported at train time, not saved

    def predict_hard(self, q: Question) -> float:
        hard_col = list(self.model.classes_).index(1)
        return float(self.model.predict_proba([features(q)])[0][hard_col])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "features": list(self.features),
            "edge_model": self.edge_model,
            "trained_at": self.trained_at,
            "calibration_accuracy": self.calibration_accuracy,
        }
        tmp = path.with_name(path.name + ".tmp")
        joblib.dump(payload, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> RouterModel:
        payload: dict[str, Any] = joblib.load(path)
        if list(payload["features"]) != FEATURES:
            raise ValueError(f"{path} was trained on features {payload['features']}, expected {FEATURES}; retrain it")
        return cls(
            model=payload["model"],
            edge_model=str(payload["edge_model"]),
            trained_at=str(payload["trained_at"]),
            calibration_accuracy=float(payload["calibration_accuracy"]),
            features=tuple(payload["features"]),
        )


def router_path(settings: Settings, edge_model: str) -> Path:
    """models/router_<safe name>.joblib"""
    return Path(settings.models_dir) / f"router_{safe_name(edge_model)}.joblib"


def edge_labels(settings: Settings, edge: ModelClient, calibration: list[Question], *, mock: bool = False) -> list[int]:
    """1 = the edge model got the question wrong (incl. parse failure or ClientError), 0 = right."""
    labels: list[int] = []
    for q in calibration:
        try:
            gen = edge.generate(
                build_prompt(q, mock=mock),
                temperature=settings.generation.temperature,
                max_tokens=settings.generation.max_tokens,
            )
        except ClientError as exc:
            log.warning("edge call failed on %s (labelled hard): %s", q.qid, exc)
            labels.append(1)
            continue
        predicted = extract_answer(gen.text, q.dataset, q.choices)
        labels.append(0 if is_correct(predicted, q.gold, q.dataset) else 1)
    return labels


def train_router(
    settings: Settings, edge: ModelClient, calibration: list[Question], *, mock: bool = False
) -> RouterModel:
    """Have the edge model answer each calibration question once, then fit the difficulty model."""
    if not calibration:
        raise ValueError("the calibration set is empty; run python -m eclm.data first")
    labels = edge_labels(settings, edge, calibration, mock=mock)
    if len(set(labels)) < 2:
        outcome = "wrong" if labels[0] else "right"
        raise ValueError(
            f"the edge model {edge.name} got all {len(labels)} calibration questions {outcome}; "
            "the router needs both easy and hard examples (use a larger or different calibration set)"
        )
    x = [features(q) for q in calibration]
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(x, labels)
    predicted = model.predict(x)
    return RouterModel(
        model=model,
        edge_model=edge.name,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        calibration_accuracy=labels.count(0) / len(labels),
        train_accuracy=sum(int(p) == y for p, y in zip(predicted, labels, strict=True)) / len(labels),
    )


class _Parser(argparse.ArgumentParser):
    """Usage errors exit 1 (ARCHITECTURE §5)."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="python -m eclm.router", description="Train the stage-1 router for an edge model.")
    sub = p.add_subparsers(dest="command", required=True, parser_class=_Parser)
    train = sub.add_parser("train", help="train and save models/router_<edge model>.joblib")
    train.add_argument("--edge-model", required=True)
    train.add_argument("--config", default="config.yaml")
    train.add_argument("--mock", action="store_true", help="fake edge model, no Ollama")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    cal_file = Path(settings.data_dir) / CALIBRATION_FILE
    if not cal_file.is_file():
        print(f"error: {cal_file} not found; run python -m eclm.data first", file=sys.stderr)
        return 1
    calibration = load_questions(cal_file)
    edge = make_client("edge", settings, model=args.edge_model, mock=args.mock)
    try:
        edge.check()
    except ClientUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        router = train_router(settings, edge, calibration, mock=args.mock)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    path = router_path(settings, args.edge_model)
    router.save(path)
    print(f"calibration questions: {len(calibration)}")
    print(f"edge accuracy on calibration ({router.edge_model}): {router.calibration_accuracy:.3f}")
    if router.train_accuracy is not None:
        print(f"router training accuracy: {router.train_accuracy:.3f}")
    threshold = settings.router.threshold
    hard = sum(router.predict_hard(q) > threshold for q in calibration) / len(calibration)
    print(f"share predicted hard (score > {threshold}): {hard:.3f}")
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
