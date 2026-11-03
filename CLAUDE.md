# edge-cloud-llm-inference

A Python CLI experiment comparing edge-only (Ollama), cloud-only (Groq, Llama 3.3 70B), hybrid (two-stage router) and random-router LLM inference on GSM8K + MMLU. For the COSC 5756 Cloud Computing final project.
Spec: `plan/SPEC.md` · Architecture & exact contracts: `plan/ARCHITECTURE.md` · Decisions: `plan/decisions/` · Tasks: `plan/tasks.json` + `plan/tasks/` · Log: `plan/PROGRESS.md`

## Commands
- Setup: `python -m pip install -e ".[dev]"`
- Full gate: `scripts/verify.sh` → last line `VERIFY: PASS` / `VERIFY: FAIL (<step>)`
- Single test: `python -m pytest tests/test_grading.py -q -k <name>`
- Lint / types: `ruff check src tests` · `mypy src`
- Pipeline in mock mode (no Ollama, no key, no network):
  `python -m eclm.data --mock && python -m eclm.router train --edge-model qwen2.5:3b --mock && python -m eclm.run --all --network normal --mock && python -m eclm.analyze`

## Conventions
- Code in `src/eclm/`, tests in `tests/test_<module>.py`. Python ≥3.11, type hints everywhere, dataclasses from `eclm.types`.
- Follow the contracts in ARCHITECTURE §5 exactly (names, fields, CSV column order, CLI flags).
- All settings come from `config.yaml` via `eclm.config.load_settings`. No magic numbers in code.
- Never hit the network in tests: use `httpx.MockTransport` for Ollama and Groq, and monkeypatch the datasets loader.
- Ollama and Groq are NOT reachable from cloud sessions. Prove behaviour with fake transports and `--mock` runs.
- Secrets live only in `.env` (git-ignored). Never print or write the API key.
- When a task adds a CLI step, extend the mock smoke test in `scripts/verify.sh` (ARCH §8). Never weaken earlier assertions.
- Result and router file names use `results.safe_name()` (`qwen2.5:3b` → `qwen2.5-3b`), so they stay valid on Windows.

## Session rules
- One task per session, via `/next-task`. Start with `git fetch origin build && git checkout build && git pull --ff-only origin build`.
- During a task, change only your task's `status` and `notes` in `plan/tasks.json`. New ideas go in as `"proposed"` tasks.
- Don't edit SPEC, ARCHITECTURE, decisions or task briefs during a task. If they're wrong, mark the task blocked. Use `/replan` to change them.
- Search the codebase before creating anything, and reuse what exists.
- Never edit, skip or delete existing tests to make them pass (unless the brief lists them).
- Commit only on `build` (`TNNN: <title>`) and push only `build` (`git push -u origin build`). Never push to `main`, force-push or rewrite history. Cloud sessions are discarded, so unpushed work is lost.
- Done = `scripts/verify.sh` passes, every acceptance criterion is checked, and task-reviewer says PASS.

## Gotchas
- (none yet; add one line per gotcha every future session must know)
