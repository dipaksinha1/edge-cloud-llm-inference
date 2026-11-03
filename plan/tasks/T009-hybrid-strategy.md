# T009 · Hybrid strategy with two-stage router

## Goal
`python -m eclm.run --strategy hybrid --edge-model M --network X` answers each question via the two-stage router and records the routing details.

## Context
SPEC R4, R7, A3, A5 · ARCHITECTURE §5 (`Router`, `Decision`, `Hybrid`), §6 (strategy totals) · ADR 0004.

## Read first
`src/eclm/router.py`, `src/eclm/strategies.py` (registry from T007), `src/eclm/grading.py`.

## Contracts
`Router(model, edge, settings).decide(q) -> Decision`:
- Stage 1: `d = model.predict_hard(q)`. If `d > router.threshold` → `Decision(route="cloud", stage=1, difficulty=d, agreement=None, edge_generations=(), edge_answer=None)`.
- Stage 2: `router.samples` edge calls at `router.sample_temperature`. Extract answers, then `answers_agree(…, min_agree)`. Majority found → `route="edge", stage=None, edge_answer=<majority>`. Otherwise → `route="cloud", stage=2`. `agreement` = size of the largest agreeing group (0 if all unparseable).
- `ClientError` on an edge sample counts as an unparseable answer (no crash).
`Hybrid(edge, cloud, router, settings)`: on cloud route, calls the cloud once. The row gets `router_stage` ("1"/"2"/""), `difficulty` (4 decimals), `agreement` ("k/n" or ""), `edge_calls`, `cloud_calls`. Timing per ARCHITECTURE §6 (TTFT = sum of earlier calls' totals + the answering call's TTFT). When it stays on the edge, the answering call is the first sample whose extracted answer equals the majority, and `predicted` = majority. `peak_mem_gb` = max over edge calls. Registered in `STRATEGIES` as a factory (hybrid uses `edge.hybrid_model` under `--all`). A `ClientError` from the cloud call → `StrategyError` carrying the edge samples. A missing router file → exit 1, "train it: python -m eclm.router train --edge-model M".

## Acceptance criteria
1. (R4) A router model stubbed to return 0.9 → stage-1 cloud, `edge_calls=0`, `cloud_calls=1`. Verify: test.
2. (R4) Low difficulty + edge answers 72, 72, 68 → edge, predicted 72, agreement "2/3", `edge_calls=3`. Verify: test with a scripted client.
3. (R4) Low difficulty + 3 different answers → stage-2 cloud, `edge_calls=3`, `cloud_calls=1`, `total_s` = sum of the 4 calls. Verify: test.
4. (A5) TTFT is computed per ARCHITECTURE §6 for both the edge-kept and escalated cases. Verify: test with fixed fake timings.
5. The verify.sh `--all` smoke now also writes `results/hybrid__qwen2.5-3b__normal.csv` (one row per test question), and verify asserts it. A test on the mock fixture shows at least one stage-2 escalation (mock samples can disagree at temperature 0.7). Verify: verify.sh + test.
6. `scripts/verify.sh` passes.

## Out of scope
Threshold tuning (open question Q1).

## Existing tests that may change
None.

## Notes
Scripted fake clients belong in `tests/conftest.py` for reuse in T010.
