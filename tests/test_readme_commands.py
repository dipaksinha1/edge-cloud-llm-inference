"""Every `python -m eclm...` command in the README parses with the module's build_parser().

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
DOCS = [ROOT / "README.md"]
PARSERS: dict[str, Callable[[], argparse.ArgumentParser]] = {
    "eclm.data": data.build_parser,
    "eclm.router": router.build_parser,
    "eclm.run": run.build_parser,
    "eclm.analyze": analyze.build_parser,
}
FENCE = re.compile(r"^[ \t]*```[^\n]*\n(.*?)^[ \t]*```", re.MULTILINE | re.DOTALL)


SEPARATORS = {"&&", "||", ";", "|"}
# redirection and background operators as shlex(punctuation_chars=True) emits them; words after one aren't argv
REDIRECTS = {">", ">>", "<", "<<", ">&", "&>", "&>>", "<&", ">|", "&"}
HEREDOC = re.compile(r"(?<!<)<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")  # not a <<< here-string
ECLM_MODULE = re.compile(r"eclm(\.\w+)*")


def _eclm_argv(words: list[str]) -> list[str] | None:
    """The argv after `-m` when the module is eclm[.x]: any prefix (sudo, VAR=x, time, a python path) is allowed."""
    for i in range(len(words) - 1):
        if words[i] == "-m" and ECLM_MODULE.fullmatch(words[i + 1]):
            return words[i + 1 :]
    return None


def _line_commands(line: str) -> list[list[str]]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    commands: list[list[str]] = []
    words: list[str] = []
    redirected = False
    for token in [*lexer, ";"]:
        if token in SEPARATORS:
            argv = _eclm_argv(words)
            if argv is not None:
                commands.append(argv)
            words, redirected = [], False
        elif redirected:
            continue
        elif token in REDIRECTS:
            if words and words[-1].isdigit():  # the file-descriptor number of e.g. 2>&1
                words.pop()
            redirected = True
        else:
            words.append(token)
    return commands


def eclm_commands(text: str) -> list[list[str]]:
    """argv lists (starting with the module name) of the `... -m eclm.*` commands in fenced code blocks.

    Heredoc bodies (`<<'EOF'` ... `EOF`) are skipped: they are input text, not shell commands.
    """
    commands: list[list[str]] = []
    for block in FENCE.findall(text):
        terminator: str | None = None
        for line in block.splitlines():
            if terminator is not None:
                if line.strip() == terminator:
                    terminator = None
                continue
            heredoc = HEREDOC.search(line)
            if heredoc is not None:
                terminator = heredoc.group(2)
                line = line[: heredoc.start()]
            if "eclm" in line:
                commands += _line_commands(line)
    return commands


def all_commands() -> list[tuple[str, list[str]]]:
    return [(doc.name, cmd) for doc in DOCS for cmd in eclm_commands(doc.read_text(encoding="utf-8"))]


def test_extractor_finds_commands_in_fences_only() -> None:
    text = "python -m eclm.run --all\n```bash\ncd x && python -m eclm.data --mock  # note\nls\n```\n"
    assert eclm_commands(text) == [["eclm.data", "--mock"]]


def _fence(*lines: str) -> str:
    return "```bash\n" + "\n".join(lines) + "\n```\n"


def test_extractor_stops_at_redirects_and_pipes() -> None:
    text = _fence(
        "python -m eclm.run --all --network normal 2>&1 | tee run.log",
        "python -m eclm.analyze > out.txt",
        "python -m eclm.data --mock &> data.log",
        "python -m eclm.run --strategy cloud --network normal >> log 2>> err &",
    )
    assert eclm_commands(text) == [
        ["eclm.run", "--all", "--network", "normal"],
        ["eclm.analyze"],
        ["eclm.data", "--mock"],
        ["eclm.run", "--strategy", "cloud", "--network", "normal"],
    ]


def test_extractor_finds_prefixed_commands_only_for_eclm_modules() -> None:
    text = _fence(
        "sudo python -m eclm.data --mock",
        "PYTHONPATH=src python3 -m eclm.analyze",
        "time python -m eclm.data",
        ".venv/bin/python -m eclm.router train --edge-model qwen2.5:3b",
        "$PY -m eclm.run --all --network normal",
        "python -m pip install -e .",
        'git commit -m "eclm notes"',
        "python -m eclm_tools.x --flag",
    )
    assert eclm_commands(text) == [
        ["eclm.data", "--mock"],
        ["eclm.analyze"],
        ["eclm.data"],
        ["eclm.router", "train", "--edge-model", "qwen2.5:3b"],
        ["eclm.run", "--all", "--network", "normal"],
    ]


def test_extractor_skips_heredoc_bodies() -> None:
    text = _fence(
        "python - <<'EOF'",
        "from eclm.run import main  # don't lex this as shell: it has an unbalanced quote",
        'print("python -m eclm.run --bogus")',
        "EOF",
        "python -m eclm.data --mock",
        'cat <<EOF > notes.txt',
        "it's python -m eclm.analyze --bogus",
        "EOF",
        "python -m eclm.analyze",
        "cat <<<word",  # a here-string has no body: the next line is still a command
        "python -m eclm.run --all --network normal",
    )
    assert eclm_commands(text) == [
        ["eclm.data", "--mock"], ["eclm.analyze"], ["eclm.run", "--all", "--network", "normal"]
    ]


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
