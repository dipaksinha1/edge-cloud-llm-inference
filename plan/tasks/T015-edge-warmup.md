# T015 · Warm up the edge model before a run's first timed question

## Goal
Row 1 of an edge, hybrid or random run must not include Ollama loading the model. Today nothing loads the model before the first timed call, so the first row's `ttft_s`/`total_s` include the load time (seconds). This hits the first row of every run in `--all` too, because the hybrid model can be unloaded during the long cloud run (Ollama's default keep-alive is 5 minutes). Fix: one untimed warm-up call per run, before the first question.

## Context
SPEC R2, R7 (correct per-question timings) · ARCHITECTURE §4 (`edge.warmup_max_tokens`), §5 (`WARMUP_QUESTION`, `warm_up`), §6 (warm-up rule), §8 (verify.sh) · found in the T013 review. Agreed with Dk: warm up **edge models only**; the cloud's first-row connection setup (about 0.1–0.3 s) is accepted.

## Read first
ARCHITECTURE §4 and §6, `src/eclm/run.py` (`run_combo`, `_context`, `_Clients`), `src/eclm/prompts.py`, `src/eclm/config.py` (`EdgeSettings`), `src/eclm/clients/mock.py` (temperature 0 calls keep no state), `src/eclm/strategies.py` (`memory_sampler`, `MeteredClient`), `scripts/verify.sh`, `docs/MAC_CHECKLIST.md` steps 3 and 6.

## Contracts
- `config.yaml` and `EdgeSettings`: new key `edge.warmup_max_tokens: 1` (int). Any value `<= 0` turns the warm-up off. No extra validation (config.py checks only missing/unknown keys). Add the key to both in the same change: `load_settings` rejects a missing or unknown key.
- `src/eclm/prompts.py`: `WARMUP_QUESTION: Question`, a fixed, made-up GSM8K-style question with qid `warmup` that isn't in any dataset or fixture. The warm-up prompt is `build_prompt(WARMUP_QUESTION, mock=...)`, so the prompt template, and with it Ollama's prompt-prefix cache, is in the same state for question 1 as for the later questions.
- `src/eclm/run.py`: `warm_up(edge: ModelClient, settings: Settings, mock: bool) -> None` makes one `edge.generate(prompt, temperature=0.0, max_tokens=settings.edge.warmup_max_tokens)` call. It isn't wrapped in a `MemorySampler`, writes no row, and isn't counted anywhere. A `ClientError` is logged as a warning (`warm-up for <model> failed: ...`) and the run continues; question 1 then fails or succeeds on its own. It logs one INFO line, `warm-up <model> done in <s> s`.
- `run_combo` calls `warm_up(ctx.edge, ...)` once per combo, right after the `log.info("run %s -> %s (%d of %d questions to do)", ...)` line and before the `with ResultWriter(...) as writer, tqdm(...)` block (tqdm draws 0/N as soon as it is created, so the warm-up log line comes before the bar), only when `ctx.edge is not None` and `warmup_max_tokens > 0`. Cloud-only runs never warm up. A resumed run warms up again (the model may be cold).

## Acceptance criteria
1. (R7) Before the first question of an edge, hybrid and random run, exactly one warm-up call reaches the edge client, with the `WARMUP_QUESTION` prompt, temperature 0 and `max_tokens == warmup_max_tokens`. No row is written for it and no `MemorySampler` wraps it. Verify: a test with a recording wrapper around the mock edge client (via monkeypatching `run.make_client`, as in `tests/test_run_resume.py`): the first recorded call is the warm-up prompt, every later call is a test-question prompt, and the CSV has one row per question. For "not sampled": monkeypatch `eclm.strategies.memory_sampler` with a wrapper that sets a flag while its sampler is entered, and have the recording client assert the flag is off during the warm-up call (a recorder alone sits inside `MeteredClient` and can't see sampling).
2. A cloud-only run makes no warm-up call and spends no budget on it. With `warmup_max_tokens: 0` no warm-up call is made at all. A run that is already complete makes no warm-up call. Verify: tests.
3. A failing warm-up (`ClientError`) logs a warning and the run still writes every row. Verify: a test whose wrapper raises only on the warm-up prompt.
4. Mock results are unchanged: with warm-up on, the `--all --mock` CSVs have the same `qid, ran_on, predicted, correct` columns as with `warmup_max_tokens: 0`. Verify: a test running both into two results dirs and comparing those columns.
5. `docs/MAC_CHECKLIST.md` step 6 no longer says the first edge row can include the model load. Step 3 gains a check that the log shows `warm-up qwen2.5:3b done` and that row 1's `ttft_s` is under 2× the median `ttft_s` of rows 2–5. Verify: read.
6. `scripts/verify.sh` passes, and its smoke checks that the `run --all --network normal --mock` output contains `warm-up qwen2.5:3b done` and no `warm-up for` (the `step` helper keeps the output in `$LOG`; capture it before the next step overwrites it). Verify: verify.sh, plus temporarily setting `warmup_max_tokens: 0` in the copied config makes that check fail.

## Verification
`scripts/verify.sh`. By hand, in a scratch folder: `cp ../config.yaml .`, `python -m eclm.data --mock`, then `python -m eclm.run --strategy edge --edge-model qwen2.5:3b --network normal --mock --limit 3` shows the `warm-up ... done` line before the progress bar.

## Out of scope
Warming up the cloud client. Passing Ollama `keep_alive`. Warm-up in `router train` (its labels don't use timings). Changing how rows are timed.

## Existing tests that may change
None expected: strategy-level tests use `ScriptedClient` directly (no runner, so no warm-up), and runner tests use `MockClient`, whose temperature-0 calls keep no state. If one does break, stop and mark the task blocked with the test name.

## Notes
The warm-up call goes to the raw edge client from `ctx.edge`. Hybrid's `MeteredClient` wraps the same client, so the model is warm for the router's samples too.
