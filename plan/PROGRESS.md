# Progress

## Project state
- Phase: **M1 in progress.** T001, T002 done. Next: T003 (dataset builder), which completes M1 (then PR build → main).
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
