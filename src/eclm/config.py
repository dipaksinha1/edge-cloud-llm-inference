"""Load config.yaml (+ .env) into a frozen Settings tree. See ARCHITECTURE §4."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class DatasetSettings:
    gsm8k_test: int
    mmlu_test: int
    gsm8k_calibration: int
    mmlu_calibration: int


@dataclass(frozen=True)
class EdgeSettings:
    host: str
    models: tuple[str, ...]
    hybrid_model: str
    timeout_s: float
    memory_sample_hz: float


@dataclass(frozen=True)
class CloudSettings:
    base_url: str
    model: str
    api_key_env: str
    timeout_s: float
    max_retries: int
    max_calls_per_run: int


@dataclass(frozen=True)
class GenerationSettings:
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class RouterSettings:
    threshold: float
    samples: int
    sample_temperature: float
    min_agree: int


@dataclass(frozen=True)
class Settings:
    seed: int
    data_dir: Path
    results_dir: Path
    models_dir: Path
    figures_dir: Path
    dataset: DatasetSettings
    edge: EdgeSettings
    cloud: CloudSettings
    generation: GenerationSettings
    router: RouterSettings


class ConfigError(Exception):
    """config.yaml is missing, unreadable or has missing/unknown keys."""


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls: type, raw: Any, where: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping")
    names = {f.name for f in fields(cls)}
    missing = names - raw.keys()
    unknown = raw.keys() - names
    if missing:
        raise ConfigError(f"{where}: missing keys {sorted(missing)}")
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        value = raw[f.name]
        ftype = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
        sub = _SECTIONS.get(f.name)
        if sub is not None and is_dataclass(sub):
            kwargs[f.name] = _build(sub, value, f"{where}.{f.name}")
        elif ftype == "Path":
            kwargs[f.name] = Path(str(value))
        elif ftype.startswith("tuple"):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


_SECTIONS: dict[str, type] = {
    "dataset": DatasetSettings,
    "edge": EdgeSettings,
    "cloud": CloudSettings,
    "generation": GenerationSettings,
    "router": RouterSettings,
}


def load_settings(path: str | Path = "config.yaml", overrides: dict[str, Any] | None = None) -> Settings:
    """Read config.yaml and apply nested overrides.

    Relative paths in the config resolve against the current working directory.
    Also loads `.env` from the current directory (if present) into the environment.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse {p}: {exc}") from exc
    if overrides:
        raw = _merge(raw, overrides)
    load_dotenv(Path.cwd() / ".env", override=False)
    settings: Settings = _build(Settings, raw, "config")
    return settings
