# T010 · Random-router baseline

## Goal
`--strategy random` sends a seeded random subset of questions to the cloud at the hybrid's offload ratio, so the report can show the router beats chance.

## Context
SPEC R6 · ARCHITECTURE §5 (`RandomRouter`), §7 (determinism), §10 (`--all` order).

## Read first
`src/eclm/strategies.py`, `src/eclm/run.py`, `src/eclm/results.py`.

## Contracts
`RandomRouter(edge, cloud, ratio, seed, settings)`: before the run, `k = round(ratio * n_questions)`, and the cloud set = `Random(seed).sample(sorted(qids), k)`. It needs the full question list, so take it via a `prepare(questions)` method called by the runner before the loop (add an optional no-op `prepare` to the `Strategy` base). Edge questions get 1 edge call, cloud questions 1 cloud call. `router_stage` = "random" for cloud rows. CLI: `--ratio R` overrides. Otherwise the ratio is read from `results/hybrid__<edge model>__<network>.csv` (rows with `ran_on=cloud` ÷ rows). Missing both → exit 1 with a message. Registered in `STRATEGIES` as a factory, run by `--all` after hybrid.

## Acceptance criteria
1. (R6) `--ratio 0.25` on 12 mock questions → exactly 3 cloud rows, the same 3 on rerun. Verify: test.
2. (R6) Without `--ratio`, the ratio equals the hybrid CSV's offload ratio for the same model and network. Verify: test with a handcrafted hybrid CSV.
3. (R6) No ratio and no hybrid CSV → exit 1 with a clear message. Verify: test.
4. The `--all` mock smoke produces `results/random__qwen2.5-3b__normal.csv`, and the smoke's explicit `--ratio 0.25 --network degraded` step produces `random__qwen2.5-3b__degraded.csv`. verify asserts both. Verify: verify.sh.
5. `scripts/verify.sh` passes.

## Out of scope
Multiple random seeds (possible follow-up).

## Existing tests that may change
None.

## Notes
Resume must keep the same cloud set: the selection depends only on seed + sorted qids + k.
