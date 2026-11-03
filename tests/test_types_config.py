from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from eclm.config import ConfigError, load_settings
from eclm.types import RESULT_COLUMNS

CONTRACT_COLUMNS = (
    "run_id strategy edge_model cloud_model network dataset qid ran_on router_stage difficulty agreement "
    "edge_calls cloud_calls ttft_s total_s output_tokens tokens_per_s peak_mem_gb predicted gold correct "
    "parse_failed error timestamp"
).split()


def test_result_columns_match_contract() -> None:
    assert list(RESULT_COLUMNS) == CONTRACT_COLUMNS


def test_load_settings_reads_every_section(workdir: Path) -> None:
    s = load_settings()
    assert s.seed == 42
    assert s.data_dir == Path("data")
    assert s.edge.models == ("qwen2.5:3b", "qwen2.5:0.5b", "llama3.2:1b")
    assert s.cloud.model == "llama-3.3-70b-versatile"
    assert s.router.min_agree == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.seed = 1  # type: ignore[misc]


def test_overrides_are_merged(workdir: Path) -> None:
    s = load_settings(overrides={"results_dir": "elsewhere", "router": {"threshold": 0.9}})
    assert s.results_dir == Path("elsewhere")
    assert s.router.threshold == 0.9
    assert s.router.samples == 3


def test_missing_and_unknown_keys_are_errors(workdir: Path) -> None:
    text = (workdir / "config.yaml").read_text()
    (workdir / "bad.yaml").write_text(text.replace("seed: 42\n", "") + "extra_key: 1\n")
    with pytest.raises(ConfigError, match="missing keys"):
        load_settings("bad.yaml")
    with pytest.raises(ConfigError, match="not found"):
        load_settings("nope.yaml")
