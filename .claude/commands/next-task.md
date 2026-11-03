---
description: Do exactly ONE task from plan/tasks.json — orient, build, verify, review, record, commit, push, stop
argument-hint: [optional task ID, e.g. T007]
disable-model-invocation: true
---

You are a senior engineer joining this project for ONE task. You have no memory of earlier sessions. The repo (code, git history, `plan/`) is the only source of truth. Do exactly one task, to a standard you'd sign off in code review, then stop.

Task: $ARGUMENTS (if empty, pick the next eligible task)

This project runs in **Claude Code cloud sessions**: each session starts from a fresh clone and is thrown away afterwards, so **work that isn't pushed is lost**.

## Rules
- One task per session. Never start a second one.
- Never edit, skip, delete or weaken existing tests unless the brief lists them under "Existing tests that may change". If a test seems wrong, stop and mark the task blocked.
- Don't edit plan/SPEC.md, plan/ARCHITECTURE.md, the decisions or the briefs. If one is wrong, record it and mark the task blocked.
- In plan/tasks.json, change only your task's status and notes. You may append new tasks with status "proposed".
- Search before you build and reuse existing code. Stay in scope; extras go under follow-ups.
- Commit only on the `build` branch and push only `build`. Never push to `main`, force-push, rewrite history or commit secrets (`.env`, API keys).
- Done means verified, not "looks right".

## 1. Orient
1. `git fetch origin build && git checkout build && git pull --ff-only origin build` (if `build` doesn't exist on the remote, create it from `origin/main`). Then `git status` and `git log --oneline -15`. Read the "Project state" section and the last 3 entries of plan/PROGRESS.md, and plan/tasks.json.
2. Pick the task:
   - If a task is `in_progress`, a previous session stopped mid-task. Read its notes and PARTIAL entry, then continue it. Say so.
   - Otherwise take the task ID I gave (stop if its dependencies aren't `done`). If none was given, take the first `todo` task whose `depends_on` are all `done`.
   - If nothing is eligible: report `RESULT: ALL_DONE` if every task is done, otherwise explain what's blocked and report `RESULT: NEEDS_HUMAN`. Stop.
3. If there are uncommitted changes that don't belong to an in-progress task, stop and report `RESULT: NEEDS_HUMAN`.
4. Set the task's status to `in_progress`.

## 2. Baseline
Set up the environment if needed (`python -m pip install -e ".[dev]"`), then run `scripts/verify.sh` before changing anything. If it fails, fixing the baseline becomes your task (expected only for T001). Note it in PROGRESS.

## 3. Understand
- Read the brief, then only the sections and files it links. Hand broad code searches to the Explore subagent to keep your context lean.
- Check the brief's contracts against the real code. Small mismatch: follow the code and note it. A mismatch that changes promised behavior, or ambiguity that could waste the session: mark the task blocked with a precise question.

## 4. Implement
- Small steps. Where practical, write the test for an acceptance criterion first.
- Follow CLAUDE.md conventions. Add no new dependencies unless the brief or architecture names them.
- No stubs, hard-coded results or TODOs standing in for required behavior. Real Ollama and Groq are NOT reachable from cloud sessions: test them through fake transports (e.g. `httpx.MockTransport`), never by faking the code under test.

## 5. Verify
- `scripts/verify.sh` must print `VERIFY: PASS`.
- Then check every acceptance criterion the way the brief says: actually run the CLI (in `--mock` mode where real backends are needed).
- After 3 genuinely different failed approaches: `git stash push -u`, mark the task blocked, record what you tried. A clear BLOCKED beats a fake DONE.

## 6. Independent review
`git add -A`, then use the task-reviewer subagent: "Review task <ID>." Fix every blocking issue and verify again. At most 2 review rounds; put unresolved disagreement in the notes.

## 7. Record, commit, push
1. In tasks.json set status to `done` or `blocked`, with a one-line note (if blocked, the exact question).
2. Append an entry of at most 15 lines to plan/PROGRESS.md. Heading: `## YYYY-MM-DD · TNNN · <title> · DONE|BLOCKED|PARTIAL`. Bullets: Did · Decisions · Gotchas · Tried & failed · Verified (how each criterion was checked) · Follow-ups. Update the "Project state" section if it changed.
3. Add one line to the CLAUDE.md "Gotchas" section only if every future session needs to know it.
4. One commit: `TNNN: <title>`, then `git push -u origin build` (retry up to 4 times on network errors, waiting 2s, 4s, 8s, 16s).

## 8. Stop
Reply in at most 10 lines: what was done, how it was verified, anything I should look at, and the next eligible task. The last line must be exactly one of:
`RESULT: DONE TNNN` · `RESULT: BLOCKED TNNN` · `RESULT: NEEDS_HUMAN` · `RESULT: ALL_DONE`

If you're running out of context mid-task: stop coding, leave the tree sane, write a PARTIAL entry with exact next steps, keep the status `in_progress`, commit `TNNN: WIP <title>`, push, and end with `RESULT: NEEDS_HUMAN`.
