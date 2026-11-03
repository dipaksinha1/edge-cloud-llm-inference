"""`python -m eclm.run`: answer every test question with one strategy and write one CSV row each."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from tqdm import tqdm

from eclm.clients import ClientUnavailable, make_client
from eclm.config import ConfigError, Settings, load_settings
from eclm.data import TEST_FILE, load_questions
from eclm.grading import extract_answer, is_correct
from eclm.metrics import system_info
from eclm.results import ResultWriter, result_path
from eclm.strategies import EdgeOnly, Strategy
from eclm.types import Generation, Question, Result

log = logging.getLogger("eclm.run")

STRATEGY_NAMES = ("edge", "cloud", "hybrid", "random")
NOT_YET = {"cloud": "T006", "hybrid": "T009", "random": "T010"}


class _Parser(argparse.ArgumentParser):
    """Usage errors exit 1 (ARCHITECTURE §5: exit 2 is reserved for 'budget reached')."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="python -m eclm.run", description="Run one strategy over the test set.")
    p.add_argument("--strategy", choices=STRATEGY_NAMES)
    p.add_argument("--edge-model")
    p.add_argument("--network", choices=("normal", "degraded"), required=True)
    p.add_argument("--ratio", type=float)
    p.add_argument("--limit", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--mock", action="store_true", help="fake models, no Ollama/API key/network")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--results-dir")
    return p


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def combine_timings(gens: list[Generation]) -> tuple[float, float]:
    """(ttft_s, total_s) for a row, per ARCHITECTURE §6: the answering call is the last one."""
    if not gens:
        return 0.0, 0.0
    before = sum(g.total_s for g in gens[:-1])
    return before + gens[-1].ttft_s, before + gens[-1].total_s


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def build_result(
    *,
    run_id: str,
    strategy: str,
    edge_model: str,
    cloud_model: str,
    network: str,
    q: Question,
    text: str,
    gens: list[Generation],
    extra: dict[str, Any],
    error: str = "",
) -> Result:
    ttft, total = combine_timings(gens)
    final = gens[-1] if gens else None
    predicted = extra.get("predicted")
    if predicted is None and text:
        predicted = extract_answer(text, q.dataset, q.choices)
    mems = [g.peak_mem_gb for g in gens if g.peak_mem_gb is not None]
    return Result(
        run_id=run_id,
        strategy=strategy,
        edge_model=edge_model,
        cloud_model=cloud_model,
        network=network,
        dataset=q.dataset,
        qid=q.qid,
        ran_on=final.ran_on if final else "",
        router_stage=str(extra.get("router_stage", "")),
        difficulty=str(extra.get("difficulty", "")),
        agreement=str(extra.get("agreement", "")),
        edge_calls=sum(1 for g in gens if g.ran_on == "edge"),
        cloud_calls=sum(1 for g in gens if g.ran_on == "cloud"),
        ttft_s=round(ttft, 6),
        total_s=round(total, 6),
        output_tokens=final.output_tokens if final else 0,
        tokens_per_s=round(final.tokens_per_s, 4) if final else 0.0,
        peak_mem_gb=_fmt(max(mems)) if mems else "",
        predicted=predicted or "",
        gold=q.gold,
        correct=False if error else is_correct(predicted, q.gold, q.dataset),
        parse_failed=False if error else predicted is None,
        error=error,
        timestamp=now_utc(),
    )


def meta_path(result_file: Path) -> Path:
    """results/edge__qwen2.5-3b__normal.csv -> results/edge__qwen2.5-3b__normal.meta.json"""
    return result_file.with_suffix(".meta.json")


def write_meta(path: Path, *, run_id: str, started_at: str, args: argparse.Namespace, settings: Settings) -> None:
    """Run metadata (§7). The config holds only the API key's env var name, never its value."""
    meta = {
        "run_id": run_id,
        "started_at": started_at,
        "args": vars(args),
        "config": asdict(settings),
        "system_info": system_info(settings, ollama=not args.mock),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _make_strategy(args: argparse.Namespace, settings: Settings) -> tuple[Strategy, str, str]:
    edge_model = args.edge_model or settings.edge.hybrid_model
    edge = make_client("edge", settings, model=edge_model, mock=args.mock)
    edge.check()
    return EdgeOnly(edge, settings, mock=args.mock), edge.name, ""


def run_one(args: argparse.Namespace, settings: Settings, questions: list[Question]) -> int:
    try:
        strategy, edge_model, cloud_model = _make_strategy(args, settings)
    except ClientUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    path = result_path(settings.results_dir, strategy.name, edge_model or None, args.network)
    started_at = now_utc()
    run_id = f"{started_at}_{strategy.name}"
    write_meta(meta_path(path), run_id=run_id, started_at=started_at, args=args, settings=settings)
    strategy.prepare(questions)
    log.info("run %s -> %s (%d questions)", run_id, path, len(questions))
    with ResultWriter(path) as writer:
        desc = f"{strategy.name} {edge_model or cloud_model} {args.network}"
        for q in tqdm(questions, desc=desc, file=sys.stderr):
            text, gens, extra = strategy.answer(q)
            writer.write(
                build_result(
                    run_id=run_id,
                    strategy=strategy.name,
                    edge_model=edge_model,
                    cloud_model=cloud_model,
                    network=args.network,
                    q=q,
                    text=text,
                    gens=gens,
                    extra=extra,
                )
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    args = build_parser().parse_args(argv)
    for flag, task in (("all", "T007"), ("limit", "T007"), ("ratio", "T010")):
        if getattr(args, flag) not in (None, False):
            print(f"error: --{flag} is not implemented yet ({task})", file=sys.stderr)
            return 1
    if args.strategy is None:
        print("error: --strategy is required (or --all, from T007)", file=sys.stderr)
        return 1
    if args.strategy in NOT_YET:
        print(f"error: --strategy {args.strategy} is not implemented yet ({NOT_YET[args.strategy]})", file=sys.stderr)
        return 1
    overrides = {"results_dir": args.results_dir} if args.results_dir else None
    try:
        settings = load_settings(args.config, overrides)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    test_file = Path(settings.data_dir) / TEST_FILE
    if not test_file.is_file():
        print(f"error: {test_file} not found; run python -m eclm.data first", file=sys.stderr)
        return 1
    return run_one(args, settings, load_questions(test_file))


if __name__ == "__main__":
    sys.exit(main())
