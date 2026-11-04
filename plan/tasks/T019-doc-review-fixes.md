# T019 · Fix the open README, checklist and doc-test findings from the T013 code review

## Goal
The run guide and checklist are what Dk follows on the Mac, so they must be exactly right. A code review of T013 found seven small problems that are still open (an eighth, mean vs median in checklist step 6, was fixed in T015).

## Context
SPEC R9–R11 · ARCHITECTURE §10 · README.md · docs/MAC_CHECKLIST.md · tests/test_readme_commands.py (T013).

## Read first
`README.md` (Quick test, Smoke test, Troubleshooting), `docs/MAC_CHECKLIST.md` (steps 7, 8), `tests/test_readme_commands.py`, `src/eclm/strategies.py` (`memory_sampler`, `RandomRouter.resolve_ratio`).

## Contracts
No code contracts change. Edits:
1. README Troubleshooting, rate-limit row: if you delete and rerun a **hybrid** CSV, also delete and rerun the **random** CSV (and meta) for the same edge model and network, because random's ratio comes from hybrid. (T017 makes a resumed random keep its recorded ratio; a complete random file is never re-resolved.)
2. README Quick test: the block starts with `rm -rf mock_test` so a second try starts clean (the expected output assumes a fresh folder). Update the Windows sentence below the block: the first four lines in PowerShell are `Remove-Item -Recurse -Force mock_test -ErrorAction SilentlyContinue`, `mkdir mock_test`, `copy config.yaml mock_test\` and `cd mock_test`.
3. README Smoke test intro: building data needs internet (Hugging Face), router training needs Ollama running; neither needs the Groq key.
4. Checklist step 7 (N5 snippet): build the sampler with `eclm.strategies.memory_sampler(s, mock=False)`, the same function real runs use, instead of `MemorySampler(...)` directly.
5. Checklist step 8: replace its `analyze --results-dir smoke_results ...` command with "run the README quick test again (about a minute) and open `mock_test/figures/*.png`" (MockClient uses the real model names, so all three edge models are on the axes; `smoke_figures` has only one), then delete `mock_test`. Keep step 6's `analyze` command and step 9's `rm -rf smoke_results smoke_figures` as they are.
6. `tests/test_readme_commands.py`: (a) words stop at the first shell redirection or background operator as shlex (with `punctuation_chars`) emits them (`>`, `>>`, `<`, `>&`, `&>`, `<&`, `&`), and an all-digit word right before the operator is dropped too (it is the file-descriptor number: `2>&1` lexes as `'2'`, `'>&'`, `'1'`), so `python -m eclm.run ... 2>&1 | tee log` parses as the run command; (b) a command is found when `-m eclm...` appears after a prefix (`sudo`, `VAR=value`, `time`, `.venv/bin/python`, `$PY`); (c) heredoc bodies (`<<'EOF'` … `EOF`) are skipped, so Python lines are never lexed as shell.

## Acceptance criteria
1. Edits 1–5 are in the docs. Verify: read; `tests/test_readme_commands.py` still finds and parses every command.
2. Extractor unit tests cover 6a (a redirect and a pipe to `tee`), 6b (each prefix form, plus a non-eclm `-m` such as `python -m pip` that is ignored) and 6c (a heredoc with an unbalanced quote and an `eclm` import is skipped, and a command after the heredoc is still found). Verify: tests.
3. The step-7 snippet still runs: with `make_client(..., mock=True)` and `memory_sampler(s, mock=True)` substituted, it prints the three expected lines. Verify: run it in a mock scratch folder.
4. The README Quick test block runs as written in a fresh shell, twice in a row, with the same expected output both times. Verify: run it.
5. `scripts/verify.sh` passes.

## Out of scope
Rewriting other README sections, or any code change outside the doc test.

## Existing tests that may change
`tests/test_readme_commands.py::test_extractor_finds_commands_in_fences_only` may be extended (not weakened).

## Notes
Keep the README written for non-developers: short steps, expected output after each. For 6c: the extractor currently skips lines without "eclm" before lexing, so detect the heredoc start line (`... <<'EOF'` or `<<EOF`) before that filter, and skip until the line that is exactly the terminator.
