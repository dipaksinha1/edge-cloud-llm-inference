from __future__ import annotations

import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp working directory containing a copy of the repo's config.yaml."""
    shutil.copy(REPO / "config.yaml", tmp_path / "config.yaml")
    monkeypatch.chdir(tmp_path)
    return tmp_path
