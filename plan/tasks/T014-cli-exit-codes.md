# T014 · Usage errors exit 1 in every CLI

## Goal
A mistyped option exits with code 1 in all four commands, so exit code 2 means only "cloud budget reached" (ARCHITECTURE §5). Today `eclm.data` and `eclm.analyze` use stock argparse, which exits 2 on a usage error. `eclm.run` and `eclm.router` each have their own private `_Parser` copy that exits 1.

## Context
ARCHITECTURE §3 (`cli.py`), §5 (`cli.ArgumentParser`, CLI contracts, exit codes: 2 = budget reached, the resumable stop of SPEC N3), §8 (verify.sh) · found in the T013 review.

## Read first
`src/eclm/run.py` (`_Parser`, `build_parser`), `src/eclm/router.py` (`_Parser`, `build_parser`), `src/eclm/data.py` and `src/eclm/analyze.py` (`build_parser`), README.md "Troubleshooting" (the exit-code line).

## Contracts
- New module `src/eclm/cli.py` (ARCHITECTURE §3):
  ```python
  class ArgumentParser(argparse.ArgumentParser):
      """Usage errors print the usage and exit 1 (exit 2 is reserved for 'budget reached')."""
      def error(self, message: str) -> NoReturn
  ```
  It behaves exactly like the current `run._Parser`: print the usage to stderr, then `self.exit(1, f"{self.prog}: error: {message}\n")`.
- All four `build_parser()` functions build an `eclm.cli.ArgumentParser`, including the `train` subparser in router.py (`add_subparsers(parser_class=...)` if needed, so `router train --bogus` also exits 1). Delete both private `_Parser` classes; there is one copy left.
- `--help` still exits 0. All flags, defaults, help texts and `prog` names stay the same.

## Acceptance criteria
1. (ARCH §5) For each of `eclm.data`, `eclm.router train`, `eclm.run` and `eclm.analyze`, an unknown option exits 1, and so does a bad value where one exists (`run --network fast`). Verify: `tests/test_cli.py`, parametrized over the four modules, calls `main([...])` or `build_parser().parse_args([...])` and asserts `SystemExit.code == 1` and that stderr contains `error:`.
2. `--help` exits 0 for each module. Verify: same test file.
3. Only one parser class: `grep -rn "class _Parser" src` prints nothing. Verify: run it.
4. The README exit-code line in "Troubleshooting" now holds for every command: `0` done, `1` a problem to fix (including a mistyped option), `2` cloud budget reached. Remove the sentence saying `eclm.data`/`eclm.analyze` exit 2 on a mistyped option. Verify: read it. `tests/test_readme_commands.py` still passes.
5. `scripts/verify.sh` passes, and its smoke adds checks that `python -m eclm.data --bogus` and `python -m eclm.analyze --bogus` exit 1 (not 2). Verify: verify.sh, plus temporarily reverting data.py to stock argparse makes that check fail.

## Verification
`scripts/verify.sh`, then by hand: `python -m eclm.data --bogus; echo $?` and `python -m eclm.analyze --bogus; echo $?` both print `1`.

## Out of scope
Changing any other exit path, adding flags, or changing messages.

## Existing tests that may change
None. The existing tests that run and router usage errors exit 1 must still pass unchanged.

## Notes
Small task. router.py already passes `parser_class=_Parser` to `add_subparsers`; point it at the new class (argparse would default to the parent's class anyway). Keep `build_parser()` as the public entry of every CLI module (README tests and the T013 doc test call it).
