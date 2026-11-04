# T016 · Run the Mac real-backend checklist (Dk, on the Mac)

## Goal
Prove the real Ollama and Groq paths work on the Mac that produces the official results, on the final code (after T014 and T015), before the full experiment.

**This task is done by a person on the Mac, not by a cloud session.** Cloud sessions can't reach Ollama or Groq. A `/next-task` session that picks this task must not attempt it: it reports `RESULT: NEEDS_HUMAN` and says "T016 is Dk's Mac checklist run".

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
If a box fails in a way that makes the official results wrong (for example `peak_mem_gb` far below `ollama ps`, or overhead over 5 %), mark this task `blocked` and don't start the full experiment until a `/replan` has decided what to do.
