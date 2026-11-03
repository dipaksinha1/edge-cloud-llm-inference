# Progress

## Project state
- Phase: **M1 complete** (T001–T003), **M2 complete** (T004–T007). M3 started: T008 done. Next: T009 (hybrid strategy). PRs build → main for M1/M2 are still pending.
- Spec: plan/SPEC.md (R1–R13, N1–N6). Architecture: plan/ARCHITECTURE.md. ADRs 0001–0006.
- 13 tasks in 4 milestones: M1 mock pipeline (T001–T003) · M2 real backends (T004–T007) · M3 hybrid + random (T008–T010) · M4 analysis + run guide (T011–T013).
- Workflow: one task per fresh Claude Code cloud session via /next-task, commits on `build`, pushed after each task. Milestones merged to `main` via PR reviewed by Dk.
- Real Ollama/Groq can't be reached from cloud sessions. They're tested via fake transports and verified on Dk's Mac with docs/MAC_CHECKLIST.md (T013).
- Deadlines: cloud credit expires 2026-11-05 02:59 EST · final presentation 2026-12-02/09 · report due 2026-12-10.
- Open question Q1 (router threshold tuning) is deferred until real results exist.

---

## 2026-11-03 · PLAN · Spec, architecture and task breakdown · DONE
- Did: adapted Dk's kickoff/architect/next-task/replan commands + task-reviewer agent for Claude Code cloud sessions. Wrote SPEC from the proposal, proposal report and agreed decisions (Groq, 3 Ollama models, LR + 3-sample router, random baseline, tool-agnostic network degradation). Wrote ARCHITECTURE with exact contracts, 6 ADRs, CLAUDE.md, 13 task briefs.
- Decisions: integration branch `build` (ADR 0006). Local `run-tasks.sh` dropped (it would bill a local plan, not the cloud credit). `.claude/settings.json` allow-list left for Dk to add (auto-mode blocked writing it).
- Verified: cold-audit subagent reviewed the plan and the fixes were applied (see the next entry).

## 2026-11-03 · PLAN · Cold audit fixes · DONE
- Did: an independent subagent audited SPEC/ARCH/tasks/briefs and found 14 issues. All real ones were fixed.
- Fixed: one smoke command and file set fixed in ARCH §8 (router trained as qwen2.5:3b, `.verify_tmp` wiped each run). `data --mock` now feeds run/router through the normal loader. Mock stage-2 samples can disagree. `safe_name()` for Windows-safe file names. Fixture 8 → 24 (T001 → T003). qid format unified. `prepare` is a non-abstract no-op. Retries: 5 retries, 6 attempts. Added StrategyError/StrategyContext/factory registry/BudgetedClient contracts. T006 now depends on T005. `--mock` limited to data/router/run. Seeds passed to backends. SUBJECT_GROUPS source named. fig3 side-file. `build_parser()` everywhere.
- Not changed: T001 stays size M (the fixture was cut to 8 questions to keep it within one session).

## 2026-11-03 · T001 · Walking skeleton: edge-only run end to end in mock mode · DONE
- Did: pyproject (src layout, ruff/mypy/pytest), config.yaml + Settings loader, types, prompts, minimal grading, MockClient + mock_accuracy + make_client, 8-question fixture + `data --mock`, results (safe_name, writer), EdgeOnly, run CLI, scripts/verify.sh (ruff, mypy, pytest, mock smoke).
- Decisions: ruff line length 120. argparse usage errors exit 1 (2 is reserved for budget). `build_prompt(q, mock=True)` appends the gold marker for MockClient only.
- Gotchas: running Python from `src/eclm/` shadows stdlib `types` (added to CLAUDE.md).
- Verified: verify.sh PASS. AC2–5 by tests in tests/test_run_edge_mock.py and test_mock_client.py. AC6 by adding a failing test temporarily → `VERIFY: FAIL (pytest)`. Independent review: PASS, no blocking issues.
- Follow-ups: T002 should cover "Answer: it is a B" (the minimal MMLU extractor returns "A").

## 2026-11-03 · T002 · Robust answer grading for GSM8K and MMLU · DONE
- Did: rewrote grading.py with colon-marker priority over "answer is", equation handling on the answer line, sign/$/Unicode-minus/leading-dot numbers, fractions, MMLU leading letter / choice text / single non-negated capital, numeric agreement. 63 tests in tests/test_grading.py.
- Decisions: ambiguous MMLU answers ("C or D") return None (counted as a parse failure, not a guess). A lowercase letter counts only as a whole token, so the article "a" isn't choice A.
- Tried & failed: round 1 (last marker wins, last capital wins) was confidently wrong on later prose and negations. Round 2 fixed those but over-applied the "=" rule. Both were caught by reviewer probes and kept as regression tests.
- Verified: verify.sh PASS. AC1 17+ GSM8K cases, AC2 12+ MMLU cases, AC3/AC4 dedicated tests. Two review rounds; round-2 findings fixed and tested (the review limit was reached, so no third round).
- Follow-ups: with no marker, a plain-hyphen range ("10-12") still reads as -12 (rare in GSM8K).

## 2026-11-03 · T003 · Dataset builder with seeded, disjoint test and calibration sets · DONE
- Did: `build_datasets` (GSM8K test/train, MMLU test/validation, plus dev only if validation is short), `DatasetError`, CLI without `--mock` now downloads. Fixture grew to 24 questions (all 4 MMLU groups, varied lengths and number counts). verify.sh asserts both data files have 12 lines. 22 tests in tests/test_data.py.
- Decisions: question text is compared after collapsing whitespace and casefolding. Calibration skips test texts, and each pool drops repeated texts. Output order is GSM8K then MMLU, sorted by row. GSM8K gold reuses grading's `_to_decimal`/`_normalise`. An autouse conftest fixture makes `eclm.data._load_split` raise, so no test can download.
- Gotchas: `test_run_edge_mock.py::test_unimplemented_options_exit_1` (`data.main([]) == 1`) now passes through the offline loader guard. It tests loader failure, not an unimplemented option, and should be renamed in a /replan.
- Verified: verify.sh PASS. AC1 is covered by the counts/disjoint and text-overlap tests, AC2 by a CLI byte-compare with the same and a different seed, AC3 by a failing-loader test (exit 1, names cais/mmlu, old files untouched). A real run here was blocked by a Hugging Face 403 and exited 1 naming gsm8k. AC4 by a fixture test plus verify.sh, AC5 by a round-trip test. Reviewer: PASS, no blocking issues.
- Follow-ups: the real download is unverified in a cloud session, so check it on the Mac (T013). The two files are renamed one after the other, so if the second write fails they can end up mismatched.

## 2026-11-03 · T004 · Ollama client with streaming timing · DONE
- Did: `clients/ollama.py` OllamaClient: streams `POST /api/generate` with `iter_lines()`. `check()` calls `GET /api/tags` and treats a name without a tag as `:latest`. `make_client("edge", mock=False)` returns it. 14 tests in tests/test_ollama.py.
- Decisions: `generate()` has no seed parameter, so it sends `options.seed` the way MockClient does (settings.seed at temperature 0, seed + k for the k-th sampled call of a prompt). total_s is taken at the `done` chunk. If `eval_duration` is missing, tokens/s falls back to tokens / (total − TTFT). A mid-stream `{"error"}` chunk, invalid JSON or a missing done chunk raise ClientError.
- Gotchas: run.py already called `check()` and caught ClientUnavailable, so AC5 only needed a test. This session ran T003 and T004 back to back because the user asked for it.
- Verified: verify.sh PASS. AC1 covered by a streaming test (text, 30 tokens, 20 tok/s, ttft ≤ total). AC2 by a test with a controlled perf_counter and an empty first chunk (ttft 2.0, not 0.5). AC3 by tests comparing the exact messages. AC4 by HTTP 500 and ReadTimeout tests. AC5 by run.main with a patched factory (exit 1, only /api/tags requested, no rows). A real non-mock run here exits 1 with the "not running" message. Reviewer: PASS, no blocking issues.
- Follow-ups: T009 must count stage-2 samples the same way (the first sample reuses settings.seed). Real Ollama timing still needs checking on the Mac (T013).

## 2026-11-03 · T005 · Memory sampler and run metadata · DONE
- Did: `metrics.py` with MemorySampler (one sample on enter, a daemon thread at `edge.memory_sample_hz`, one on exit) and system_info / ollama_version. `Strategy.sampler()` + `edge_call()` wrap every edge call and fill `peak_mem_gb`, using `pids=[os.getpid()]` in mock mode. run.py writes `results/<csv stem>.meta.json` (run_id, started_at, args, config, system_info) atomically before the first question. verify.sh checks that the meta file exists. 13 tests in tests/test_metrics.py.
- Decisions: GB = 1024**3 bytes. system_info gained optional `ollama=` / `http=` keywords, so mock runs never request /api/version and tests use MockTransport. The version request ignores proxy env vars (localhost). On Linux, cpu comes from /proc/cpuinfo.
- Verified: verify.sh PASS. AC1 covered by fake process_iter tests (summed peak, case-insensitive match, AccessDenied/NoSuchProcess skipped, a spike caught by the thread). AC2 by an EdgeOnly test, a build_result max/blank test and the mock-run CSV. AC3 by a meta test with GROQ_API_KEY set (value absent, exact keys). AC4 by 10 start/stop cycles at 10 Hz averaging < 50 ms. Reviewer: PASS, no blocking issues.
- Follow-ups: T007 should decide whether a resumed run keeps the original meta (it is currently overwritten on every run). N5 overhead in process_iter mode still needs checking on the Mac (T013).

## 2026-11-03 · T006 · Groq cloud client and cloud-only strategy · DONE
- Did: `clients/cloud.py` CloudClient: `openai.OpenAI(max_retries=0)`, streams with `stream_options.include_usage` and sends a seed like the other clients. Its own back-off (1, 2, 4, 8, 16 s, injectable sleep) covers 429/5xx/connection errors; other 4xx fail at once. `check()`: missing key, 401 → "Groq rejected the API key", model not listed. CloudOnly strategy, `--strategy cloud`, `make_client("cloud")`. verify.sh smoke runs cloud and checks its rows and meta. 16 tests in tests/test_cloud.py.
- Decisions: timing covers the successful attempt only (back-off sleeps are rate limiting, not latency). With no usage chunk, content chunks are counted as tokens. `check()` uses the back-off too.
- Existing test changed (user approved in-session): test_run_edge_mock.py:73 now asserts `--strategy hybrid` exits 1 instead of cloud. The brief should have listed it, because AC4 makes cloud succeed.
- Gotchas: installed openai (3.43) uses `httpx2` and types `http_client` as `httpx2.Client`, so the injected `httpx.Client` is passed with `cast(Any, ...)` and works by duck typing. A stale `.mypy_cache` can hide the type error (cold cache passed, warm failed).
- Verified: verify.sh PASS (twice). AC1 covered by an SSE test with a controlled clock (40 tokens, tps = 40/(1.7-0.5), empty first chunk). AC2 by recorded sleeps [1,2] and six 429/503s → ClientError. AC3 by an exact-message test, run exit 1 with no CSV, and a sentinel key absent from stdout, stderr, logs, CSV and meta. AC4 by a mock run test plus smoke. Reviewer: PASS, no blocking issues.
- Follow-ups: real Groq (model id, rate limits) still needs checking on the Mac (T013).

## 2026-11-03 · T007 · Resumable runner, error rows, --all, --limit and cloud budget · DONE
- Did: `StrategyError`, `StrategyContext` and the `STRATEGIES` registry (edge/cloud register factories and wrap ClientError). `BudgetedClient`/`BudgetExceeded` in clients/base.py. run.py: `plan_combos` (--all in ARCH §10 order, hybrid/random only if registered), one shared client cache, resume via `done_qids`, error rows, `--limit`, exit 2 on budget, tqdm starting from done rows. verify.sh smoke now runs `--all` plus the degraded edge step. 19 tests in tests/test_run_resume.py.
- Decisions: a resumed run keeps the existing meta.json and reuses its run_id, so all rows match the meta (it warns if the meta is missing). `--all` checks every backend before the first question, and one BudgetedClient covers the whole process. `--all` combined with `--strategy`/`--edge-model`, or `--limit < 1`, exits 1. Every attempted cloud call counts against the budget, failed ones included.
- Existing test changed (user approved in-session): test_run_edge_mock.py:74 now asserts `--strategy random` exits 1 instead of `--all`, because AC4 makes `--all` succeed. The brief should have listed it.
- Verified: verify.sh PASS. AC1 by a KeyboardInterrupt-after-3 test, then a rerun with exact qids. AC2 by a ClientError-on-one-qid test (edge and cloud) plus a completed-calls StrategyError test. AC3 by budget=3 → exit 2 with 3 rows, then a rerun, then completion; a manual CLI run agreed. AC4 by smoke and a file-set test. AC5 by a byte-compare of the normal files. Reviewer: PASS, and non-blocking items 2 and 3 were fixed.
- Follow-ups: `prepare()` receives the full (limited) question list, not only the remaining ones, which T010 needs to know. Earlier PROGRESS entries have a duplicated T001/PLAN block inside T003 (left as is).

## 2026-11-03 · T008 · Router features and training command · DONE
- Did: `router.py` with FEATURES/`features()`, `SUBJECT_GROUPS` (all 57 MMLU subjects per hendrycks categories.py, unknown → other), `edge_labels`, `train_router`, `RouterModel` (joblib save/load, `predict_hard`), `router_path`, and the `python -m eclm.router train` CLI. verify.sh smoke trains the qwen2.5:3b router and checks the joblib file. 28 tests in tests/test_router.py.
- Decisions: `train_router` takes an extra keyword-only `mock=False` so MockClient prompts carry the gold marker (ARCH §5 should be updated in a /replan). n_numbers, n_math_ops and max_number_digits use the question text only; n_words adds the MMLU choices. Op symbols are counted literally, so the '-' in "well-known" counts too. `RouterModel.train_accuracy` is reported but not saved. `load()` rejects a file trained on a different FEATURES list. "Share predicted hard" uses score > threshold (strict).
- Verified: verify.sh PASS. AC1 by 3 hand-checked vectors plus group-count tests. AC2 by a scripted fake edge that is wrong on long questions. AC3 by a save/load test over 24 probes. AC4 by a CLI run with no test file, and a spy test with a poison test file that shows only calibration is opened. AC5 by verify.sh. The mock calibration has both classes for all 3 edge models (tested). Reviewer: PASS, no blocking issues.
- Follow-ups: T009's Router.decide should use the same strict `>` against the threshold. Mock training accuracy for qwen2.5:3b is 0.75, with 25% of questions predicted hard.
