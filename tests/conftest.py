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


@pytest.fixture(autouse=True)
def _no_dataset_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never hit the network: the datasets loader fails unless a test monkeypatches its own."""
    from eclm import data

    def offline(path: str, config: str, split: str) -> list[dict[str, object]]:
        raise ConnectionError("network disabled in tests")

    monkeypatch.setattr(data, "_load_split", offline)
