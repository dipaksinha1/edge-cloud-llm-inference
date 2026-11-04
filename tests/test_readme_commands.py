"""Every `python -m eclm...` command in the README and the Mac checklist parses with the module's build_parser().

Commands are only parsed, never executed.
"""

from __future__ import annotations

import argparse
import re
import shlex
from collections.abc import Callable
from pathlib import Path

import pytest

from eclm import analyze, data, router, run

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "docs" / "MAC_CHECKLIST.md"]
PARSERS: dict[str, Callable[[], argparse.ArgumentParser]] = {
    "eclm.data": data.build_parser,
    "eclm.router": router.build_parser,
    "eclm.run": run.build_parser,
    "eclm.analyze": analyze.build_parser,
}
FENCE = re.compile(r"^[ \t]*```[^\n]*\n(.*?)^[ \t]*```", re.MULTILINE | re.DOTALL)


def eclm_commands(text: str) -> list[list[str]]:
    """argv lists (starting with the module name) of the `python -m eclm.*` commands in fenced code blocks."""
    commands: list[list[str]] = []
    for block in FENCE.findall(text):
        for line in block.splitlines():
            if "eclm" not in line:
                continue
            lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            lexer.commenters = "#"
            words: list[str] = []
            for token in [*lexer, ";"]:
                if token not in ("&&", "||", ";", "|"):
                    words.append(token)
                    continue
                if len(words) >= 3 and words[0] in ("python", "python3", "py") and words[1] == "-m":
                    if words[2].startswith("eclm"):
                        commands.append(words[2:])
                words = []
    return commands


def all_commands() -> list[tuple[str, list[str]]]:
    return [(doc.name, cmd) for doc in DOCS for cmd in eclm_commands(doc.read_text(encoding="utf-8"))]


def test_extractor_finds_commands_in_fences_only() -> None:
    text = "python -m eclm.run --all\n```bash\ncd x && python -m eclm.data --mock  # note\nls\n```\n"
    assert eclm_commands(text) == [["eclm.data", "--mock"]]


def test_docs_cover_every_cli() -> None:
    modules = {cmd[0] for _, cmd in all_commands()}
    assert modules == set(PARSERS)
    readme = eclm_commands((ROOT / "README.md").read_text(encoding="utf-8"))
    assert ["eclm.run", "--all", "--network", "normal", "--mock"] in readme
    assert ["eclm.run", "--all", "--network", "degraded"] in readme


@pytest.mark.parametrize(("doc", "argv"), all_commands(), ids=lambda v: " ".join(v) if isinstance(v, list) else v)
def test_command_parses(doc: str, argv: list[str]) -> None:
    module, args = argv[0], argv[1:]
    assert module in PARSERS, f"{doc}: unknown module {module}"
    try:
        PARSERS[module]().parse_args(args)
    except SystemExit as exc:  # argparse exits on a bad command line
        pytest.fail(f"{doc}: `python -m {' '.join(argv)}` does not parse (exit {exc.code})")
