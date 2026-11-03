# 0006 · Build workflow: one task per fresh Claude Code cloud session, integration branch `build`

**Context.** Development uses Claude Code cloud sessions (Dk's cloud credit), each starting from a fresh clone that is discarded afterwards. Ollama and Groq aren't reachable from them.

**Decision.** /next-task does exactly one task per session, commits `TNNN: <title>` on the `build` branch, and pushes `build` (never `main`, never force). Milestones are merged to `main` through a pull request that Dk reviews. Real-backend behaviour is verified on the Mac via the T013 checklist. The local `run-tasks.sh` loop from the original workflow is not used, because it would bill a local plan instead of the cloud credit.

**Alternatives.** Branch + PR per task (13+ PRs for Dk to merge). Committing straight to main (no review point).

**Consequences.** Pushing is mandatory at the end of every task. Sessions must `git pull --ff-only origin build` first, and tasks run sequentially, never in parallel on `build`.
