# T001 · Walking skeleton: edge-only run end to end in mock mode

## Goal
`python -m eclm.data --mock && python -m eclm.run --strategy edge --edge-model qwen2.5:3b --network normal --mock` answers the bundled questions with a fake edge model and writes a correct results CSV. The toolchain and `scripts/verify.sh` work and are meaningful.

## Context
SPEC R2, R7, R9, R12 · ARCHITECTURE §3 (layout), §4 (config), §5 (contracts), §6 (measurement rules), §8 (verify.sh) · ADR 0001.

## Read first
`plan/ARCHITECTURE.md` §3–§8, `CLAUDE.md`.

## Contracts provided (exact names per ARCHITECTURE §5)
- `pyproject.toml`: package `eclm` in `src/`, deps from ARCHITECTURE §2, extra `[dev]` (pytest, ruff, mypy). Ruff + mypy config (`ignore_missing_imports = true`).
- `config.yaml` with every key in ARCHITECTURE §4. `.env.example` containing `GROQ_API_KEY=`.
- `eclm/config.py`: `Settings` (frozen dataclasses) and `load_settings(path="config.yaml", overrides=None)`. Relative paths in config resolve against the current working directory.
- `eclm/types.py`: `Question`, `Generation`, `Decision`, `Result` exactly as specified.
- `eclm/prompts.py`: `build_prompt(q)`, both formats.
- `eclm/grading.py`: the three functions with their final signatures. **Minimal** behaviour is enough here (GSM8K: number after the last `Answer:`; MMLU: letter after the last `Answer:`; `answers_agree` majority). T002 makes them robust.
- `eclm/clients/base.py`, `clients/mock.py`, `clients/__init__.py` (`make_client`; with `mock=True` it returns `MockClient`. Non-mock kinds may raise `NotImplementedError("T004")`/`("T006")` until those tasks land, and that's the ONLY allowed placeholder).
- `MockClient` + `mock_accuracy` exactly as in ARCHITECTURE §5 (gold read from the hidden `[[gold:<x>]]` marker that `build_prompt(q, mock=True)` appends; real prompts never contain it; temperature-0 determinism; per-call variation at temperature > 0; distractor answers). In mock mode, `make_client` names the client after the requested model (e.g. `qwen2.5:3b`). It returns plausible timings (`latency_s`) and token counts.
- `eclm/fixtures/mock_questions.jsonl`: **8** realistic questions (4 GSM8K, 4 MMLU) in the `Question` schema (T003 expands it to 24). `eclm.data.mock_questions() -> (test, calibration)` returns a fixed disjoint split (half each, each half GSM8K/half MMLU), loaded via `importlib.resources`. Tests must derive expected counts from the fixture/data files, never hard-code them.
- `eclm/data.py` (minimal): `load_questions`, `save_questions` and the CLI `python -m eclm.data --mock`, which writes the mock split to `data_dir/test_questions.jsonl` and `calibration_questions.jsonl`. Without `--mock` it exits 1 with "not implemented yet (T003)". `run.py` always loads `data_dir/test_questions.jsonl` (missing → exit 1, "run python -m eclm.data first").
- `eclm/strategies.py`: `Strategy` ABC + `EdgeOnly`.
- `eclm/results.py`: `safe_name`, `result_path`, `done_qids`, `ResultWriter` (header = `Result` field order).
- `build_parser()` in `run.py` and `data.py`.
- `eclm/run.py`: CLI with the flags in ARCHITECTURE §5. In this task only `--strategy edge`, `--edge-model`, `--network`, `--mock`, `--config`, `--results-dir` must work. The others parse but exit 1 with "not implemented yet (TNNN)".
- `scripts/verify.sh` steps 1–4 from ARCHITECTURE §8. The smoke step wipes and recreates `.verify_tmp/`, copies `config.yaml` in, runs `data --mock` then the edge run from there, and asserts `results/edge__qwen2.5-3b__normal.csv` has as many data rows as `data/test_questions.jsonl` has lines.
- `tests/test_types_config.py`, `tests/test_mock_client.py`, `tests/test_run_edge_mock.py`.

## Acceptance criteria
1. (R12) On a clean clone, `python -m pip install -e ".[dev]" && scripts/verify.sh` prints `VERIFY: PASS`. Verify: run it.
2. (R2, R7) The mock edge run writes one row per test question with every `Result` column in order, `ran_on=edge`, `edge_calls=1`, `cloud_calls=0`, and `ttft_s <= total_s`. Verify: test reads the CSV.
3. (R9) `--network` is required, and its value appears on every row and in the file name `results/edge__qwen2.5-3b__<network>.csv` (via `safe_name`). Verify: test with `degraded`.
4. (R8 minimal) `correct` matches `predicted == gold` on the mock run. Verify: test. (MockClient's accuracy is checked separately over ≥200 synthetic prompts: within ±0.1 of `mock_accuracy`.)
5. Running the same mock command twice produces identical `predicted` columns (determinism). Verify: test.
6. `verify.sh` fails with `VERIFY: FAIL (pytest)` when a test fails. Verify: temporarily add a failing test locally, observe, remove it (don't commit it).

## Out of scope
Real Ollama/Groq, resume/--all/--limit (T007), memory sampling (T005), dataset download (T003).

## Existing tests that may change
None (first task).

## Notes for the implementer
Keep `verify.sh` fast (<60 s). Use `python -m` everywhere so the venv's interpreter is used. `Result.timestamp` is UTC ISO-8601.
