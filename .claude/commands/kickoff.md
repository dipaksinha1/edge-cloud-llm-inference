---
description: Interview me like a principal engineer, then write plan/SPEC.md
argument-hint: [project idea or change]
disable-model-invocation: true
---

You are a principal engineer with strong product sense. This session has ONE output: an approved product spec at `plan/SPEC.md`. Do not design the architecture, create tasks or write code. Later sessions do that, each starting fresh.

Project idea: $ARGUMENTS (if empty, ask me for a short description first)

> This project already has an approved spec (`plan/SPEC.md`, written from the course proposal). Run /kickoff again only to restart the spec from scratch. For changes to an existing spec, use /replan.

## 1. Ground yourself
Read `plan/SPEC.md` if it exists, and the proposal summary in it. If the repo has code, use the Explore subagent to summarize the stack, the main modules and how tests run, so your questions are about the real system.

## 2. Interview me
Use the AskUserQuestion tool. If it isn't available (for example in a cloud thread), ask the same questions in chat and wait.
- Ask in rounds of 1–4 related questions. Each round builds on my previous answers.
- Give concrete options for every question, with your recommendation first, marked "(Recommended)".
- Skip the obvious. Dig into: what "done" means for the experiment, data and its lifecycle, failure modes, integrations (Ollama, cloud API), budget and rate limits, privacy of API keys, tech I want or forbid, time, non-goals.
- Stop when nothing expensive is left to guess (usually 3–6 rounds). If I say "you decide", decide and record it as an assumption.

## 3. Confirm the shape
Show me an outline of at most 20 lines: the goal in one sentence, the requirements by ID (R1, R2…), the non-goals and your top assumptions. Apply my corrections.

## 4. Write plan/SPEC.md
Sections: Problem & users · Goals (v1) · Non-goals · Requirements · Non-functional requirements · Constraints · Assumptions · Open questions · Glossary.
- Cover WHAT and WHY only. Technology appears only as a constraint I set.
- Give every requirement a stable ID (R1…). Write its acceptance criteria as "WHEN <situation> THE SYSTEM SHALL <observable behavior>", including the failure paths that matter. Each criterion must be testable using only the running software.
- Make non-functional requirements measurable, or leave them out.
- Keep it to 1–4 pages. Every later session pays for the spec's length.

## 5. Finish
Commit on the `build` branch with the message `docs(plan): product spec` and push it (`git push -u origin build`). Then tell me to review the spec, start a fresh session, and run `/architect`.
