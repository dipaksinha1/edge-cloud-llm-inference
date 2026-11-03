# T002 · Robust answer grading for GSM8K and MMLU

## Goal
`grading.py` handles the messy ways real small models format answers, so accuracy numbers are trustworthy.

## Context
SPEC R8, assumption A3 · ARCHITECTURE §5 (`grading.py` contract).

## Read first
`src/eclm/grading.py`, `src/eclm/prompts.py`, `tests/` for existing grading tests.

## Contracts
Provides (unchanged signatures): `extract_answer(text, dataset) -> str | None`, `is_correct(predicted, gold, dataset) -> bool`, `answers_agree(answers, dataset, min_agree) -> str | None`.

## Acceptance criteria
1. (R8) GSM8K extraction handles: `Answer: 72`, `answer: $1,200.`, `**Answer:** 72`, `The answer is 72.` (no "Answer:" line → last number), `72.0`→`72`, `-5`, `3/4`→`0.75`, `50%`→`50`, and multiple `Answer:` lines (the last wins). Verify: parametrised tests, ≥15 cases.
2. (R8) MMLU extraction handles: `Answer: C`, `Answer: (C)`, `answer: c`, `The correct answer is C. Velocity`, `C) Velocity` on the last line, and `Answer: Velocity` matched to the choice text when choices are given (add an optional `choices` parameter with default `()`, backward compatible). Verify: ≥10 cases.
3. (R8) Unparseable text returns `None`, and `is_correct(None, …)` is False. Verify: tests.
4. (A3) `answers_agree(["72","72.0","68"],"gsm8k",2) == "72"`. `answers_agree(["A",None,None],"mmlu",2) is None`. Ties at min_agree pick the earliest. Verify: tests.
5. `scripts/verify.sh` passes.

## Out of scope
Any change to prompts (only if a case can't be graded otherwise, and then note it).

## Existing tests that may change
T001's minimal grading tests may be extended, not weakened.

## Notes
Normalise numbers with `Decimal` to avoid float noise. Strip commas, `$`, trailing `.`.
