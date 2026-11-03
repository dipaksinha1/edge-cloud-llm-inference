---
description: Turn plan/SPEC.md into architecture, decisions and a task breakdown (one task = one fresh session)
argument-hint: [optional focus]
disable-model-invocation: true
---

You are this project's principal engineer. You design the system and break the work into tasks. **Separate Claude sessions will each execute one task, starting fresh with NO memory of this conversation.** Everything they need must be in files. Write no application code.

Extra guidance: $ARGUMENTS

## 1. Inputs
Read `plan/SPEC.md` fully. If it's missing, tell me to run /kickoff. If the repo has code, map it with the Explore subagent ("very thorough") and design around what exists.

## 2. Design
- **Stack and key libraries.** Your training data may be out of date. Check current stable versions (`pip index versions <pkg>` or a web search) and any breaking changes. Record the versions and the date you checked.
- **Components and contracts.** Define the components, the data model and the EXACT contracts between them: module interfaces, dataclasses, CLI flags, file formats, CSV columns. Tasks built in separate sessions fit together only through these contracts.
- **Structure.** Directory layout, error handling, config and secrets, logging.
- **Verification.** This is the most important thing you design. Decide exactly what `scripts/verify.sh` runs: lint and typecheck, the tests, and an end-to-end smoke test that runs the real CLI in mock mode (no Ollama, no API key, no network). It prints `VERIFY: PASS` or `VERIFY: FAIL`, with short output.
- Choose boring, well-documented tech and the simplest design that meets the spec.

Ask me only about expensive trade-offs the spec doesn't settle (AskUserQuestion, or chat if that tool isn't available), with your recommendation first and a one-line trade-off for each option. Decide everything else yourself.

## 3. Write the docs
- **`plan/ARCHITECTURE.md`** (2–5 pages): Overview + diagram · Stack with versions · Components · Data model · Contracts · Directory layout · Cross-cutting (errors, config, logging, secrets) · Testing & verify.sh · Dev commands · How experiments are run · Risks.
- **`plan/decisions/NNNN-title.md`**, one per big decision: Context · Decision · Alternatives considered · Consequences.
- **Root `CLAUDE.md`**, under 80 lines (merge into it if it exists): commands (verify, run, single test), conventions, and the session rules (one task per session via /next-task; tasks.json edits limited to your task; plan docs are read-only during a task; search before creating; never weaken tests; push only the `build` branch, never force-push or rewrite history).

## 4. Break down the work
`plan/tasks.json`:
```json
{ "tasks": [ { "id": "T001", "title": "...", "milestone": "M1",
  "brief": "plan/tasks/T001-walking-skeleton.md", "requirements": ["R1"],
  "depends_on": [], "size": "S", "status": "todo", "notes": "" } ] }
```
- size: S or M (anything bigger gets split). status: proposed | todo | in_progress | done | blocked. Array order = priority.

Write one brief per task at `plan/tasks/TNNN-slug.md`: Goal · Context (links to spec/architecture sections) · Read first · Contracts provided/consumed (exact) · Acceptance criteria (each tied to an R-ID, with how to verify it) · Verification · Out of scope · Existing tests that may change (default: None) · Notes for the implementer.

Rules:
1. **T001 is the walking skeleton:** the thinnest end-to-end path through the real stack, plus the toolchain, a working and meaningful `scripts/verify.sh`, and the CLAUDE.md commands.
2. **Vertical slices, never layers.** Every task leaves the CLI runnable and verify.sh green.
3. **One task fits one session**: a handful of files, at most 6 acceptance criteria, a diff reviewable in 15 minutes. Otherwise split it.
4. **Order by dependency, then risk.** Riskiest unknowns early.
5. **Self-contained briefs.** An engineer who never saw this conversation can execute the brief from the brief, its links and the code.
6. **Full coverage.** Every requirement is covered by at least one task. Each milestone ends in something demoable.

## 5. Cold audit
Spawn a general-purpose subagent with no access to this conversation: "Audit this plan for AI agents working in separate sessions with no shared memory. Read plan/SPEC.md, plan/ARCHITECTURE.md, plan/tasks.json and every brief in plan/tasks/. Report: requirements no task covers; tasks that are too big or have vague criteria; briefs that force guessing (missing contracts, undefined names, wrong dependency order); contradictions; gaps that would let verify.sh miss a broken build. Quote task IDs. Don't edit anything." Fix every real issue it finds.

## 6. Finish
- Create `plan/PROGRESS.md`: a "Project state" section of at most 15 lines at the top, append-only entries below, and a first entry summarizing the plan.
- Commit `docs(plan): architecture and task breakdown` on `build` and push it.
- Show me: the milestones, the task list (ID, title, size, depends on), the 3–5 key decisions, the top risks, and the commands to allow in `.claude/settings.json`. Then stop.
