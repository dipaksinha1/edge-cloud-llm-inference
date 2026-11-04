# T018 · Count a failed stage-2 edge sample's time in the hybrid row's latency

## Goal
A hybrid row's latency must include time spent on stage-2 edge samples that failed. Today a sample that raises `ClientError` (an Ollama timeout after `edge.timeout_s` = 300 s, a dropped connection) counts as an unparseable answer and its time is dropped. So the row's `total_s`/`ttft_s` understate what the user waited, and hybrid looks faster than it was exactly when the edge struggles.

## Context
SPEC R4, R7 · ARCHITECTURE §5 (`Decision.failed_edge_s`), §6 (strategy totals), §7 (errors) · follow-up from T009 (PROGRESS). Decision (approved by Dk with the other open items): count the time, and keep `edge_calls` = completed calls (§7 is unchanged).

## Read first
`src/eclm/router.py` (`Router.decide`), `src/eclm/types.py` (`Decision`), `src/eclm/strategies.py` (`Hybrid.answer`), `src/eclm/run.py` (`combine_timings`, `build_result`), `tests/test_hybrid.py`, `tests/conftest.py` (`ScriptedClient`, which can raise per call).

## Contracts
- `Decision` gets a last field `failed_edge_s: tuple[tuple[int, float], ...] = ()`. Each entry is `(completed_before, elapsed_s)`: how many stage-2 samples had completed before this failed one, and the seconds from just before the `generate` call to the exception (`time.perf_counter()`).
- `Router.decide` times every sample call and fills `failed_edge_s`. Nothing else in the decision changes (a failed sample still counts as an unparseable answer).
- `Hybrid.answer` puts `extra["failed_edge_s"] = decision.failed_edge_s` when it is non-empty (also on the `StrategyError` path).
- `combine_timings(gens, answer_index=None, failed=())`: `total_s` = sum of every call's `total_s` + every failed sample's time. `ttft_s` = as now (calls before the answering call + its TTFT) + the time of each failed sample with `completed_before <= i`, where `i` is the answering call's index (the last call unless `answer_index` is given). With no generations, it returns `(0.0, sum of failed)`. `build_result` passes `extra.get("failed_edge_s", ())`.
- `edge_calls` still counts completed calls only (§7). `failed_edge_s` is not a CSV column.

## Acceptance criteria
1. (R7) Edge-kept row with a failure before the answering sample: script `[ClientError, "72", "72"]` with fixed timings, where the failure takes a known time (use a fake clock or make `ScriptedClient` sleep-free by patching `time.perf_counter` in `eclm.router`). Then `total_s` = failure + both samples, `ttft_s` = failure + the answering sample's TTFT, `edge_calls` = 2, `agreement` = `2/3`. Verify: test with exact numbers.
2. Edge-kept row with the failure after the answering sample (`["72", "72", ClientError]`): the failure counts in `total_s` but not in `ttft_s`. Verify: test.
3. Stage-2 → cloud row with one failed sample: the failure counts in both `total_s` and `ttft_s` (it happened before the cloud call). Verify: test.
4. Rows without failures are unchanged: the existing hybrid timing tests pass unchanged, and `--all --mock` CSVs are identical apart from run_id/timestamps (MockClient never fails). Verify: test suite.
5. `scripts/verify.sh` passes.

## Verification
`scripts/verify.sh`.

## Out of scope
Counting failed calls in `edge_calls`. Failed calls in other strategies (their failure is an error row already). Retrying failed samples.

## Existing tests that may change
None.

## Notes
`ScriptedClient` returns the timings you give it for completed calls; for the failed call's duration, patch `eclm.router.time.perf_counter` with an iterator of fixed values, or have the scripted client advance a fake clock. Keep `router.py` importing `time` as a module so the patch works.
