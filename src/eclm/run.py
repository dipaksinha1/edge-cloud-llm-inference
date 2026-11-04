"""`python -m eclm.run`: answer the test questions with one strategy (or --all) and write one CSV row each.

Runs are resumable: qids already in the CSV are skipped and new rows are appended."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from eclm import cli
from eclm.clients import BudgetedClient, BudgetExceeded, ClientError, ClientUnavailable, ModelClient, make_client
from eclm.config import ConfigError, Settings, load_settings
from eclm.data import TEST_FILE, load_questions
from eclm.grading import extract_answer, is_correct
from eclm.metrics import system_info
from eclm.prompts import WARMUP_QUESTION, build_prompt
from eclm.results import ResultWriter, done_qids, result_path
from eclm.strategies import STRATEGIES, Strategy, StrategyContext, StrategyError, StrategyUnavailable
from eclm.types import Generation, Question, Result

log = logging.getLogger("eclm.run")

STRATEGY_NAMES = ("edge", "cloud", "hybrid", "random")


def build_parser() -> argparse.ArgumentParser:
    p = cli.ArgumentParser(prog="python -m eclm.run", description="Run one strategy over the test set.")
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


def combine_timings(gens: list[Generation], answer_index: int | None = None) -> tuple[float, float]:
    """(ttft_s, total_s) for a row, per ARCHITECTURE §6.

    total = sum of every call's total. ttft = totals of the calls before the answering call + its TTFT.
    The answering call is the last one unless `answer_index` says otherwise (hybrid kept on the edge).
    """
    if not gens:
        return 0.0, 0.0
    i = len(gens) - 1 if answer_index is None else answer_index
    return sum(g.total_s for g in gens[:i]) + gens[i].ttft_s, sum(g.total_s for g in gens)


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
    answer_index = extra.get("answer_index") if not error else None
    ttft, total = combine_timings(gens, answer_index)
    final = (gens[-1] if answer_index is None else gens[answer_index]) if gens else None
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


def write_meta(
    path: Path,
    *,
    run_id: str,
    started_at: str,
    args: argparse.Namespace,
    settings: Settings,
    strategy_meta: dict[str, Any] | None = None,
) -> None:
    """Run metadata (§7). The config holds only the API key's env var name, never its value."""
    meta: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "args": vars(args),
        "config": asdict(settings),
        "system_info": system_info(settings, ollama=not args.mock),
    }
    if strategy_meta:  # only strategies with something to record (random); others keep the T005 key set
        meta["strategy"] = strategy_meta
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


# Clients each strategy needs. Strategies not listed (hybrid, random) need both sides.
CLIENT_SIDES: dict[str, tuple[str, ...]] = {"edge": ("edge",), "cloud": ("cloud",)}
ALL_ORDER = ("edge", "cloud", "hybrid", "random")  # ARCHITECTURE §10


@dataclass(frozen=True)
class Combo:
    """One (strategy, edge model) cell of the experiment; edge_model is None for cloud-only."""

    strategy: str
    edge_model: str | None


class _Clients:
    """Builds each client once per process. The cloud client is wrapped in ONE BudgetedClient (§7)."""

    def __init__(self, settings: Settings, mock: bool) -> None:
        self.settings = settings
        self.mock = mock
        self._edge: dict[str, ModelClient] = {}
        self._cloud: BudgetedClient | None = None

    def edge(self, model: str) -> ModelClient:
        if model not in self._edge:
            client = make_client("edge", self.settings, model=model, mock=self.mock)
            client.check()
            self._edge[model] = client
        return self._edge[model]

    def cloud(self) -> BudgetedClient:
        if self._cloud is None:
            client = make_client("cloud", self.settings, mock=self.mock)
            client.check()
            self._cloud = BudgetedClient(client, self.settings.cloud.max_calls_per_run)
        return self._cloud


def plan_combos(args: argparse.Namespace, settings: Settings) -> list[Combo]:
    """Single run, or the --all matrix: edge x each model, cloud, then hybrid/random if registered."""
    if not args.all:
        if args.strategy == "cloud":
            return [Combo("cloud", None)]
        return [Combo(args.strategy, args.edge_model or settings.edge.hybrid_model)]
    combos: list[Combo] = []
    for name in ALL_ORDER:
        if name not in STRATEGIES:
            continue
        if name == "edge":
            combos += [Combo("edge", m) for m in settings.edge.models]
        elif name == "cloud":
            combos.append(Combo("cloud", None))
        else:
            combos.append(Combo(name, settings.edge.hybrid_model))
    return combos


def _context(combo: Combo, args: argparse.Namespace, settings: Settings, clients: _Clients) -> StrategyContext:
    sides = CLIENT_SIDES.get(combo.strategy, ("edge", "cloud"))
    edge = clients.edge(combo.edge_model) if "edge" in sides and combo.edge_model else None
    cloud = clients.cloud() if "cloud" in sides else None
    return StrategyContext(
        settings=settings,
        edge_model=combo.edge_model,
        network=args.network,
        mock=args.mock,
        results_dir=Path(settings.results_dir),
        ratio=args.ratio,
        edge=edge,
        cloud=cloud,
    )


def _read_meta(meta_file: Path) -> dict[str, Any] | None:
    """The run's meta.json with a usable run_id, or None if it is missing or unreadable."""
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    run_id = meta.get("run_id")
    return meta if isinstance(run_id, str) and run_id else None


def warm_up(edge: ModelClient, settings: Settings, mock: bool) -> None:
    """One edge call before a run's first question, so row 1 doesn't include loading the model (ARCH §6).

    Not memory-sampled, writes no row, counted nowhere. A failure is only a warning: question 1 then
    fails or succeeds on its own.
    """
    start = time.perf_counter()
    try:
        edge.generate(
            build_prompt(WARMUP_QUESTION, mock=mock), temperature=0.0, max_tokens=settings.edge.warmup_max_tokens
        )
    except ClientError as exc:
        log.warning("warm-up for %s failed: %s", edge.name, exc)
        return
    log.info("warm-up %s done in %.2f s", edge.name, time.perf_counter() - start)


def run_combo(
    combo: Combo,
    strategy: Strategy,
    ctx: StrategyContext,
    args: argparse.Namespace,
    settings: Settings,
    questions: list[Question],
) -> int:
    """Answer every question of `questions` not yet in this combo's CSV. Returns 0, or 2 if the budget ran out."""
    edge_model = ctx.edge.name if ctx.edge is not None else (combo.edge_model or "")
    cloud_model = ctx.cloud.name if ctx.cloud is not None else ""
    path = result_path(settings.results_dir, strategy.name, combo.edge_model, args.network)
    done = done_qids(path)
    todo = [q for q in questions if q.qid not in done]
    meta = meta_path(path)
    # A resumed run keeps its original metadata and run_id, so every row of the CSV matches its meta.json.
    recorded = _read_meta(meta) if done else None
    if todo:  # before any file is written: e.g. random without a ratio raises StrategyUnavailable here
        if recorded is not None:
            strategy_meta = recorded.get("strategy")
            strategy.restore(strategy_meta if isinstance(strategy_meta, dict) else {})
        strategy.prepare(questions)
    run_id: str | None = recorded["run_id"] if recorded is not None else None
    if done and run_id is None:
        log.warning("%s has rows but no readable %s; writing new metadata", path, meta.name)
    if run_id is None:
        started_at = now_utc()
        run_id = f"{started_at}_{strategy.name}"
        write_meta(meta, run_id=run_id, started_at=started_at, args=args, settings=settings,
                   strategy_meta=strategy.meta())
    if not todo:
        log.info("run %s -> %s already complete (%d questions)", run_id, path, len(questions))
        return 0
    log.info("run %s -> %s (%d of %d questions to do)", run_id, path, len(todo), len(questions))
    if ctx.edge is not None and settings.edge.warmup_max_tokens > 0:
        warm_up(ctx.edge, settings, args.mock)  # before tqdm, which draws 0/N as soon as it is created
    desc = f"{strategy.name} {edge_model or cloud_model} {args.network}"
    with ResultWriter(path) as writer, tqdm(
        total=len(questions), initial=len(questions) - len(todo), desc=desc, file=sys.stderr
    ) as bar:
        for q in todo:
            error = ""
            try:
                text, gens, extra = strategy.answer(q)
            except StrategyError as exc:
                text, gens, extra = "", exc.generations, exc.extra
                error = f"{type(exc.cause).__name__}: {exc.cause}"
                log.warning("%s failed on %s: %s", desc, q.qid, error)
            except BudgetExceeded as exc:
                log.warning("%s: %s; stopping before %s. Rerun the same command to resume.", desc, exc, q.qid)
                return 2
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
                    error=error,
                )
            )
            bar.update(1)
    log.info("run %s -> %s finished", run_id, path)
    return 0


def run_all(args: argparse.Namespace, settings: Settings, questions: list[Question]) -> int:
    combos = plan_combos(args, settings)
    clients = _Clients(settings, args.mock)
    try:  # check every backend (and e.g. the hybrid's router file) before the first question
        contexts = [_context(c, args, settings, clients) for c in combos]
        strategies = [STRATEGIES[c.strategy](ctx) for c, ctx in zip(combos, contexts, strict=True)]
    except (ClientUnavailable, StrategyUnavailable) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for combo, strategy, ctx in zip(combos, strategies, contexts, strict=True):
        try:
            code = run_combo(combo, strategy, ctx, args, settings, questions)
        except StrategyUnavailable as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if code != 0:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    args = build_parser().parse_args(argv)
    if args.ratio is not None and not 0.0 <= args.ratio <= 1.0:
        print("error: --ratio must be between 0 and 1", file=sys.stderr)
        return 1
    if args.all and (args.strategy is not None or args.edge_model is not None):
        print("error: --all runs every configured model; don't combine it with --strategy or --edge-model",
              file=sys.stderr)
        return 1
    if not args.all and args.strategy is None:
        print("error: --strategy is required (or --all)", file=sys.stderr)
        return 1
    if args.strategy is not None and args.strategy not in STRATEGIES:
        print(f"error: --strategy {args.strategy} is not implemented yet", file=sys.stderr)
        return 1
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be at least 1", file=sys.stderr)
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
    questions = load_questions(test_file)
    if args.limit is not None:
        questions = questions[: args.limit]
    return run_all(args, settings, questions)


if __name__ == "__main__":
    sys.exit(main())
