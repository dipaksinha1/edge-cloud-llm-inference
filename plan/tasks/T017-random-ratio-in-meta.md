# T017 · Record the random router's resolved ratio in meta.json and keep it on resume

## Goal
A random run's cloud set must stay the same from its start to its last row. Today `RandomRouter.prepare()` resolves the ratio again on every (re)start: from `--ratio`, or from the hybrid CSV for the same edge model and network. If that hybrid CSV changed between the start and a resume (for example a hybrid run redone after rate-limit errors), k and the seeded cloud set change half-way, so the file mixes two selections. The resolved ratio is also not recorded anywhere, so the report can't say which ratio the baseline used.

## Context
SPEC R6, R10, R13 · ARCHITECTURE §5 (`Strategy.meta`/`Strategy.restore`, `write_meta`), §7 (run metadata) · follow-up from T010 (PROGRESS).

## Read first
`src/eclm/strategies.py` (`Strategy`, `RandomRouter`, `_random`), `src/eclm/run.py` (`run_combo`, `write_meta`, `_read_run_id`), `tests/test_random_router.py`, `tests/test_run_resume.py` (resume tests).

## Contracts
- `Strategy` (base class) gets two non-abstract methods:
  - `meta(self) -> dict[str, Any]`, default `{}`. Called after `prepare()`. `run_combo` writes it into `<csv>.meta.json` under the key `"strategy"` when a run starts (new meta file).
  - `restore(self, recorded: dict[str, Any]) -> None`, default no-op. On a resume (the CSV has rows and its meta file is readable), `run_combo` calls `strategy.restore(meta.get("strategy") or {})` **before** `prepare()`.
- `RandomRouter.meta()` returns `{"ratio": <float>, "ratio_source": "--ratio" | "<hybrid csv path>", "cloud_questions": k, "questions": n}`.
- `RandomRouter.restore(recorded)`: if `recorded` has a `ratio`, the resumed run uses it (so k and the cloud set are the same as at the start). If the ratio the run would otherwise resolve now differs by more than 1e-9, log a warning: `random: keeping the recorded ratio <r0> from <meta file>; the current source gives <r1>. Delete <csv> and its meta.json to start over with the new ratio.` A meta file without a `strategy` key (written before T017) restores nothing, and the ratio is resolved as before.
- `write_meta(path, *, run_id, started_at, args, settings, strategy_meta: dict[str, Any] | None = None)` adds `"strategy": strategy_meta` when given. `_read_run_id` becomes `_read_meta(meta_file) -> dict[str, Any] | None` (None if missing or unreadable), and the run_id is taken from it as before.
- Edge, cloud and hybrid keep the default `meta()` (`{}`) and write `"strategy": {}`.

## Acceptance criteria
1. (R13) A random run's meta.json has `strategy.ratio`, `strategy.ratio_source`, `strategy.cloud_questions` and `strategy.questions`, matching the INFO log line. Verify: a test for `--ratio 0.25` (source `--ratio`) and one with the ratio read from a handcrafted hybrid CSV (source = that path).
2. (R6, R10) Resume keeps the cloud set: start a random run with the ratio from a hybrid CSV, stop it after some rows (`--limit`), rewrite the hybrid CSV so it gives a different ratio, resume. The resumed rows use the recorded ratio (the full file's cloud rows equal `Random(seed).sample(sorted(qids), round(r0·n))`), and the warning names both ratios. Verify: test.
3. A resume whose meta has no `strategy` key behaves as before (ratio re-resolved, no warning). Edge/cloud/hybrid meta files get `"strategy": {}`. Verify: tests.
4. `scripts/verify.sh` passes, and its smoke checks that `results/random__qwen2.5-3b__normal.meta.json` contains a `"ratio"` key. Verify: verify.sh.

## Verification
`scripts/verify.sh`, then a manual mock run (in a scratch folder with `config.yaml`, after `data --mock` and `router train --mock`): `run --strategy random --edge-model qwen2.5:3b --ratio 0.25 --network normal --mock` and read the meta file.

## Out of scope
Changing how the ratio is computed, or recording anything for the other strategies.

## Existing tests that may change
None. Existing tests read meta keys (`run_id`, `args`, `config`, `system_info`) that stay as they are.

## Notes
`run_combo` today calls `prepare()` before reading the meta file. Read the meta first (only when `done` is non-empty), then `restore`, then `prepare`, then write a new meta only for a new run, exactly as now.
