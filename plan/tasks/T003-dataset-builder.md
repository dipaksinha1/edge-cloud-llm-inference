# T003 · Dataset builder with seeded, disjoint test and calibration sets

## Goal
`python -m eclm.data` downloads GSM8K and MMLU and writes `data/test_questions.jsonl` (100+100) and `data/calibration_questions.jsonl` (150+150), reproducibly. `--mock` (already built in T001) keeps working, and the bundled fixture grows to 24 questions.

## Context
SPEC R1, R12 · ARCHITECTURE §4 (`dataset.*`, `seed`, `data_dir`), §5 (`data.py`, `Question`).

## Read first
`src/eclm/data.py` (mock_questions from T001), `src/eclm/types.py`.

## Contracts
Provides `build_datasets(settings)`, `load_questions(path)`, `save_questions(qs, path)`, CLI `python -m eclm.data [--config] [--mock]`.
Sources (Hugging Face `datasets`): GSM8K `openai/gsm8k` config `main`. Test questions come from split `test`, calibration from split `train`. Gold = the number after `####`, normalised. MMLU `cais/mmlu` config `all`. Test questions from `test`, calibration from `validation` (+`dev` if needed). Gold = `"ABCD"[answer]`, choices tuple, subject. qid = `gsm8k-<split>-<row>` / `mmlu-<split>-<row>`.

## Acceptance criteria
1. (R1) With the loader monkeypatched to small in-memory tables, the builder returns the configured counts, half from each dataset, with no qid **and no question text** shared between test and calibration. Verify: test.
2. (R1) Same seed → byte-identical files. A different seed → a different selection. Verify: test.
3. (R1) A loader exception → exit code 1, a message naming the dataset, and existing files untouched (write to temp, then rename). Verify: test.
4. (R12) Expand `eclm/fixtures/mock_questions.jsonl` to 24 questions (12 GSM8K + 12 MMLU, varied subjects including at least one of each MMLU subject group, varied lengths and number counts, so the router has signal). `data --mock` still writes a 12/12 disjoint split. verify.sh additionally asserts both data files exist and have 12 lines each. Verify: verify.sh.
5. `load_questions(save_questions(x))` round-trips equal objects. Verify: test.

## Out of scope
Running models on the data. Committing the real data files (Dk generates them on the Mac, see T013).

## Existing tests that may change
None (T001 tests derive counts from the fixture).

## Notes
Selection: `random.Random(seed).sample(sorted indices, k)`. Never call the network in tests.
