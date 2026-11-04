# T016 · Run the Mac real-backend checklist (Dk, on the Mac)

## Goal
Prove the real Ollama and Groq paths work on the Mac that produces the official results, on the final code (after T014 and T015), before the full experiment.

**This task is done by a person on the Mac, not by a cloud session.** Cloud sessions can't reach Ollama or Groq. Its status in `plan/tasks.json` stays `blocked` ("waiting on Dk's Mac run"), so `/next-task` never picks it: with only T016 left it finds nothing eligible and reports `RESULT: NEEDS_HUMAN`. A cloud session must not change T016's status or commit anything for it. Dk sets it to `done` (or leaves it `blocked` with the problem) in the same commit as the PROGRESS entry.

## Context
SPEC R1–R3, R7, R13, N5 · ARCHITECTURE §10, §11 · `docs/MAC_CHECKLIST.md` (written in T013).

## Read first
`README.md` (Setup, Smoke test, Degrading the network), `docs/MAC_CHECKLIST.md`.

## Contracts
None (no code). Outputs: a PROGRESS entry and, for every mismatch that needs a code change, a new `proposed` task in `plan/tasks.json`.

## Acceptance criteria
1. Every box in `docs/MAC_CHECKLIST.md` sections 0–9 is ticked, or its mismatch is written down. Verify: the PROGRESS entry lists the unticked boxes (or says there were none).
2. (R2, R3, R7, R13, N5) The PROGRESS entry `## YYYY-MM-DD · T016 · Run the Mac real-backend checklist · DONE|BLOCKED` records: Mac model and chip, RAM, macOS, Ollama version, edge tokens/s, `peak_mem_gb` vs `ollama ps` SIZE, cloud TTFT normal vs degraded, the N5 overhead (total and generation only), router calibration accuracy and share predicted hard, and the MMLU parse check (step 7b).
3. Each mismatch that needs code has a `proposed` task with the evidence (command, output).
4. `data/test_questions.jsonl` and `data/calibration_questions.jsonl` are committed (`git add data`), and smoke outputs are deleted.

## Verification
Read the PROGRESS entry and `git log` on `build`.

## Out of scope
The full experiment (README "Full experiment"). That follows once this is DONE.

## Existing tests that may change
None.

## Notes
If a box fails in a way that makes the official results wrong (for example `peak_mem_gb` far below `ollama ps`, or the N5 overhead still over 5 % after retrying at `memory_sample_hz: 5` as step 7 says), keep this task `blocked` with the problem in its notes and don't start the full experiment until a `/replan` has decided what to do.
