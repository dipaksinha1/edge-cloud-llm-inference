# T007 · Resumable runner, error rows, --all, --limit and cloud budget

## Goal
Real runs on the Mac survive interruptions, single failures and rate limits, and `--all` runs the whole matrix.

## Context
SPEC R9, R10, N3, N6 · ARCHITECTURE §5 (`results.py`, CLI), §7 (errors, budget), §10 (`--all` order).

## Read first
`src/eclm/run.py`, `src/eclm/results.py`, `src/eclm/strategies.py`.

## Contracts
- Restarting with the same args skips qids in `done_qids(path)` and appends.
- Add `StrategyError`, `StrategyContext` and the `STRATEGIES` factory registry exactly as in ARCHITECTURE §5. Refactor EdgeOnly/CloudOnly to register factories and to raise `StrategyError(cause, completed_generations, extra)` on `ClientError`.
- `StrategyError` on a question → the runner writes a row per ARCHITECTURE §7: `error="<type>: <msg>"`, `correct=False`, `parse_failed=False`, timings and `edge_calls`/`cloud_calls` from the completed generations (0 if none), then continues.
- `--limit N` = only the first N test questions (in file order).
- Add `BudgetedClient` and `BudgetExceeded` (ARCHITECTURE §5, §7). run.py wraps the cloud client. On `BudgetExceeded` it writes no row for that question, logs, flushes and exits 2. The budget covers only this process (a restart starts a fresh budget, which is intended).
- `--all --network X`: edge for each model in `edge.models`, then cloud, then hybrid and random **only if registered in `STRATEGIES`** (T009/T010 register themselves). Each combination gets its own CSV + meta.json.
- tqdm progress bar `<strategy> <model> <network>: n/total`, starting from already-done rows.

## Acceptance criteria
1. (R10) A mock run interrupted after 3 rows (simulate by raising `KeyboardInterrupt` from a patched strategy) then rerun → exactly one row per test question, no duplicate qids. Verify: test.
2. (R10) A client raising `ClientError` on one specific qid → that row has `error` set, call counts from completed calls, and all other rows are normal. Verify: test.
3. (N3) With `max_calls_per_run=3`, a mock cloud run exits 2 after 3 rows (the 4th question has no row). A rerun completes the rest. Verify: test.
4. (R10) `--all --network normal --mock` creates one CSV per configured edge model plus cloud. verify.sh smoke switches to `--all` and also runs the `--network degraded` edge step from ARCH §8, asserting all those files. Verify: verify.sh + test.
5. (R9) `--network degraded` results never overwrite `normal` results. Verify: test.
6. `scripts/verify.sh` passes.

## Out of scope
Hybrid/random strategies themselves.

## Existing tests that may change
None. verify.sh keeps every earlier assertion (`edge__qwen2.5-3b__normal.csv`, `cloud__none__normal.csv`) and adds new ones. EdgeOnly/CloudOnly tests may change only in how strategies are constructed (factory), not in what they assert.

## Notes
In mock mode, `make_client` already names clients after the requested model (T001), so file names match real runs. The three mock edge models differ through `mock_accuracy`.
