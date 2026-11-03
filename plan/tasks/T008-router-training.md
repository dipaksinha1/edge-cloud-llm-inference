# T008 · Router features and training command

## Goal
`python -m eclm.router train --edge-model M` trains and saves the stage-1 difficulty model from the calibration set.

## Context
SPEC R5, A4 · ARCHITECTURE §5 (`router.py`) · ADR 0004.

## Read first
`src/eclm/data.py`, `src/eclm/grading.py`, `src/eclm/clients/`, `src/eclm/run.py` (client creation + `check()` pattern).

## Contracts
- `FEATURES` and `features(q)` exactly as in ARCHITECTURE §5. Definitions: `n_words` = whitespace tokens of question (+choices for MMLU). `n_numbers` = numeric tokens in the question. `n_math_ops` = count of `+ - * / % ×` and the words "percent, twice, half, times, each, per". `is_mmlu` 0/1. Subject groups: one-hot over STEM / humanities / social sciences / other via `SUBJECT_GROUPS`, a dict of all 57 MMLU subjects following the official MMLU category mapping (hendrycks/test `categories.py`: STEM, humanities, social sciences, other). An unknown subject maps to `other`. All four are 0 for GSM8K. `max_number_digits` = digits in the longest number.
- `train_router(settings, edge, calibration) -> RouterModel`: the edge model answers each calibration question once (temperature 0). label = 1 if wrong (incl. parse failure or `ClientError`), else 0. Fit `LogisticRegression(class_weight="balanced", max_iter=1000)`. If only one class is present, raise `ValueError` with a clear message.
- `RouterModel.save(path)` / `RouterModel.load(path)` (joblib dict: sklearn model, FEATURES, edge model name, train date, calibration accuracy). `predict_hard(q) -> float`.
- CLI (`build_parser()` exposed): `python -m eclm.router train --edge-model M [--config] [--mock]` loads `data_dir/calibration_questions.jsonl` (in mock mode too, as written by `data --mock`), prints n, the edge accuracy on calibration, router training accuracy, and the share predicted hard at the threshold, then saves `models/router_<safe name>.joblib`. A missing calibration file → exit 1, "run python -m eclm.data first".

## Acceptance criteria
1. (R5) `features()` returns `len(FEATURES)` floats with the expected values for 3 hand-checked questions. Verify: test.
2. (R5) Training with a scripted fake edge client (wrong on long questions) yields `predict_hard(long) > predict_hard(short)`. Verify: test.
3. (R5) Save → load round-trip gives identical predictions. Verify: test.
4. (R5) Training never reads `test_questions.jsonl`. Verify: test that points data_dir at a dir without it.
5. The verify.sh smoke runs `python -m eclm.router train --edge-model qwen2.5:3b --mock` and asserts `models/router_qwen2.5-3b.joblib` exists. Verify: verify.sh.
6. `scripts/verify.sh` passes.

## Out of scope
Using the router in a strategy (T009).

## Existing tests that may change
None.

## Notes
Keep features cheap (no model calls). With 12 mock calibration questions and mock accuracy 0.65, a single-class outcome is possible. If it happens with the default seed, adjust the fixture (T003 content), not the seed, and add a test that asserts both classes occur.
