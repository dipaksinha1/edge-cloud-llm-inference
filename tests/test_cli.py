"""Usage errors exit 1 in every CLI (exit 2 is reserved for 'cloud budget reached'), --help exits 0 (T014)."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import pytest

from eclm import analyze, cli, data, router, run

PARSERS: dict[str, Callable[[], argparse.ArgumentParser]] = {
    "data": data.build_parser,
    "router": router.build_parser,
    "run": run.build_parser,
    "analyze": analyze.build_parser,
}
MAINS: dict[str, Callable[[list[str]], int]] = {
    "data": data.main,
    "router": router.main,
    "run": run.main,
    "analyze": analyze.main,
}
# argv with an unknown option; the rest of the command line is valid so only the option is wrong
UNKNOWN_OPTION = {
    "data": ["--bogus"],
    "router": ["train", "--edge-model", "qwen2.5:3b", "--bogus"],
    "run": ["--strategy", "edge", "--network", "normal", "--bogus"],
    "analyze": ["--bogus"],
}
BAD_VALUES = [
    ("run", ["--strategy", "edge", "--network", "fast"]),
    ("run", ["--strategy", "fog", "--network", "normal"]),
    ("run", ["--strategy", "edge", "--network", "normal", "--limit", "many"]),
    ("run", ["--strategy", "edge"]),  # missing required --network
    ("router", ["fly"]),  # unknown subcommand
    ("router", ["train"]),  # missing required --edge-model
]
HELP = {
    "data": ["--help"],
    "router": ["train", "--help"],
    "run": ["--help"],
    "analyze": ["--help"],
}


def _exit_code(fn: Callable[[], object]) -> int | str | None:
    with pytest.raises(SystemExit) as exc:
        fn()
    return exc.value.code


@pytest.mark.parametrize("name", sorted(PARSERS))
def test_every_cli_uses_the_shared_parser(name: str) -> None:
    assert isinstance(PARSERS[name](), cli.ArgumentParser)


@pytest.mark.parametrize("name", sorted(PARSERS))
def test_unknown_option_exits_1(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert _exit_code(lambda: PARSERS[name]().parse_args(UNKNOWN_OPTION[name])) == 1
    err = capsys.readouterr().err
    assert "error:" in err and "--bogus" in err
    assert err.startswith("usage:")


@pytest.mark.parametrize("name", sorted(MAINS))
def test_main_unknown_option_exits_1(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert _exit_code(lambda: MAINS[name](UNKNOWN_OPTION[name])) == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize(("name", "argv"), BAD_VALUES, ids=lambda v: " ".join(v) if isinstance(v, list) else v)
def test_bad_value_exits_1(name: str, argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert _exit_code(lambda: PARSERS[name]().parse_args(argv)) == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("name", sorted(PARSERS))
def test_help_exits_0(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert _exit_code(lambda: PARSERS[name]().parse_args(HELP[name])) == 0
    assert "usage:" in capsys.readouterr().out


def test_error_message_format(capsys: pytest.CaptureFixture[str]) -> None:
    p = cli.ArgumentParser(prog="python -m eclm.x")
    assert _exit_code(lambda: p.error("broken")) == 1
    assert capsys.readouterr().err.endswith("python -m eclm.x: error: broken\n")
