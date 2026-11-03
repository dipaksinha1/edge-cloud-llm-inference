"""Result files: naming, resume support and the CSV writer (ARCHITECTURE §5)."""

from __future__ import annotations

import csv
import re
from dataclasses import astuple
from pathlib import Path
from types import TracebackType

from eclm.types import RESULT_COLUMNS, Result

_UNSAFE = re.compile(r"[:/\\\s]+")


def safe_name(model: str | None) -> str:
    """'qwen2.5:3b' -> 'qwen2.5-3b'; None/'' -> 'none'. Valid on Windows, macOS and Linux."""
    if not model:
        return "none"
    return _UNSAFE.sub("-", model.strip())


def result_path(results_dir: Path, strategy: str, edge_model: str | None, network: str) -> Path:
    return Path(results_dir) / f"{strategy}__{safe_name(edge_model)}__{network}.csv"


def done_qids(path: Path) -> set[str]:
    """qids already recorded in a result file (empty if it doesn't exist)."""
    if not Path(path).is_file():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["qid"] for row in csv.DictReader(f) if row.get("qid")}


class ResultWriter:
    """Appends rows, writing the header if the file is new, and flushes after every row."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.is_file() or self.path.stat().st_size == 0
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._csv = csv.writer(self._file)
        if new:
            self._csv.writerow(RESULT_COLUMNS)
            self._file.flush()

    def write(self, r: Result) -> None:
        self._csv.writerow(astuple(r))
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> ResultWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
