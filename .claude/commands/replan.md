---
description: Update the plan when tasks are blocked, scope changes or reality drifts
argument-hint: [what changed]
disable-model-invocation: true
---

You are the principal engineer, back after other sessions have done some tasks. Bring the plan back in line with reality. Write no application code.

What changed: $ARGUMENTS

1. `git fetch origin build && git checkout build && git pull --ff-only origin build`. Read plan/SPEC.md, plan/ARCHITECTURE.md, plan/tasks.json, all of plan/PROGRESS.md and `git log --oneline -30`. List the blocked tasks and their questions, the proposed tasks, and any drift between the plan and the code (spot-check with the Explore subagent).
2. Resolve every blocked question and scope change with me (AskUserQuestion, or chat if that tool isn't available), with your recommendation first. Never drop a requirement silently.
3. Update:
   - **SPEC:** keep R-IDs stable. Mark removed requirements "Withdrawn".
   - **Architecture:** make it match reality. Write a new ADR that supersedes an old one instead of editing it.
   - **Tasks:** never change `done` tasks. Rewrite, split, reorder or delete the rest, following the breakdown rules in .claude/commands/architect.md. Promote `proposed` tasks to `todo` with full briefs, or delete them. Unblocked tasks go back to `todo`.
   - **PROGRESS:** rewrite "Project state" (at most 15 lines) and append a REPLAN entry.
4. Run the same cold-audit subagent as /architect step 5 and fix what's real. Commit `docs(plan): replan — <summary>` on `build` and push. Show me what changed and the next task.
