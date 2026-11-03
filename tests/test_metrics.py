"""MemorySampler, system_info and run metadata (T005)."""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from pathlib import Path

import httpx
import psutil
import pytest

from eclm import data, metrics, run
from eclm.config import Settings, load_settings
from eclm.metrics import BYTES_PER_GB, MemorySampler, system_info
from eclm.strategies import EdgeOnly
from eclm.types import Generation, Question

GB = BYTES_PER_GB


class _Mem:
    def __init__(self, rss: int) -> None:
        self.rss = rss


class FakeProc:
    def __init__(self, name: str, rss: list[int] | None = None, error: type[Exception] | None = None) -> None:
        self._name = name
        self._rss = list(rss or [0])
        self._i = 0
        self.error = error

    def name(self) -> str:
        return self._name

    def memory_info(self) -> _Mem:
        if self.error is not None:
            raise self.error(1)
        value = self._rss[min(self._i, len(self._rss) - 1)]
        self._i += 1
        return _Mem(value)


def _patch(monkeypatch: pytest.MonkeyPatch, procs: list[FakeProc]) -> None:
    monkeypatch.setattr(metrics.psutil, "process_iter", lambda *a, **k: iter(procs))


def test_peak_of_summed_rss_over_time(monkeypatch: pytest.MonkeyPatch) -> None:
    procs = [
        FakeProc("ollama", [1 * GB, 2 * GB, 1 * GB]),
        FakeProc("Ollama Helper", [GB // 2, 2 * GB, 3 * GB]),  # case-insensitive match
        FakeProc("python", [50 * GB]),  # not matched
        FakeProc("ollama runner", error=psutil.AccessDenied),  # skipped silently
        FakeProc("ollama gone", error=psutil.NoSuchProcess),
    ]
    _patch(monkeypatch, procs)
    s = MemorySampler(hz=10)
    for _ in range(3):
        s.sample()
    assert s.peak_gb == pytest.approx(4.0)  # tick sums: 1.5, 4.0, 4.0


def test_no_matching_process_gives_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [FakeProc("python", [GB])])
    with MemorySampler(hz=50) as s:
        time.sleep(0.05)
    assert s.peak_gb is None


def test_thread_catches_a_spike_during_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    busy = threading.Event()

    class Spiky(FakeProc):
        def memory_info(self) -> _Mem:
            return _Mem(6 * GB if busy.is_set() else GB)

    _patch(monkeypatch, [Spiky("ollama")])
    with MemorySampler(hz=100) as s:
        busy.set()
        time.sleep(0.1)
        busy.clear()
    assert s.peak_gb == pytest.approx(6.0)


def test_pids_mode_measures_this_process() -> None:
    with MemorySampler(hz=10, pids=[os.getpid()]) as s:
        pass
    expected = psutil.Process().memory_info().rss / GB
    assert s.peak_gb is not None and 0.5 * expected < s.peak_gb < 2 * expected


def test_start_stop_overhead_under_50ms() -> None:
    n = 10
    t0 = time.perf_counter()
    for _ in range(n):
        with MemorySampler(hz=10, pids=[os.getpid()]):
            pass
    assert (time.perf_counter() - t0) / n < 0.05


class _FakeEdge:
    name = "qwen2.5:3b"
    ran_on = "edge"

    def check(self) -> None: ...

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> Generation:
        return Generation("Answer: 4", 0.1, 0.2, 5, 25.0, self.name, "edge")


def test_edge_only_fills_peak_mem(workdir: Path) -> None:
    settings = load_settings("config.yaml")
    q = Question("q1", "gsm8k", "2+2?", "4")
    _, gens, _ = EdgeOnly(_FakeEdge(), settings, mock=True).answer(q)
    assert gens[0].peak_mem_gb is not None and gens[0].peak_mem_gb > 0


def _gen(where: str, mem: float | None) -> Generation:
    return Generation("Answer: 4", 0.1, 0.2, 5, 25.0, "m", where, mem)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("gens", "expected"),
    [([_gen("edge", 1.5), _gen("edge", 2.25), _gen("cloud", None)], "2.2500"), ([_gen("cloud", None)], "")],
)
def test_row_peak_is_max_over_edge_calls_blank_for_cloud(gens: list[Generation], expected: str) -> None:
    r = run.build_result(run_id="r", strategy="s", edge_model="m", cloud_model="c", network="normal",
                         q=Question("q1", "gsm8k", "2+2?", "4"), text="Answer: 4", gens=gens, extra={})
    assert r.peak_mem_gb == expected


def _mock_run(workdir: Path) -> None:
    assert data.main(["--mock"]) == 0
    assert run.main(["--strategy", "edge", "--edge-model", "qwen2.5:3b", "--network", "normal", "--mock"]) == 0


def test_mock_run_rows_have_peak_mem(workdir: Path) -> None:
    _mock_run(workdir)
    with open(workdir / "results" / "edge__qwen2.5-3b__normal.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and all(float(r["peak_mem_gb"]) > 0 for r in rows)


def test_meta_json_written_without_secrets(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "gsk_test_SECRET_value_123"
    monkeypatch.setenv("GROQ_API_KEY", secret)
    _mock_run(workdir)
    path = workdir / "results" / "edge__qwen2.5-3b__normal.meta.json"
    raw = path.read_text()
    assert secret not in raw
    meta = json.loads(raw)
    assert set(meta) == {"run_id", "started_at", "args", "config", "system_info"}
    assert meta["run_id"].startswith(meta["started_at"]) and meta["run_id"].endswith("_edge")
    assert meta["args"]["strategy"] == "edge" and meta["args"]["mock"] is True
    assert meta["config"]["cloud"]["api_key_env"] == "GROQ_API_KEY"
    assert meta["config"]["seed"] == 42 and meta["config"]["edge"]["models"][0] == "qwen2.5:3b"
    info = meta["system_info"]
    assert set(info) == {"os", "os_version", "machine", "cpu", "total_ram_gb", "python", "ollama_version"}
    assert info["ollama_version"] is None and info["total_ram_gb"] > 0  # mock runs don't ask Ollama


@pytest.fixture
def settings(workdir: Path) -> Settings:
    return load_settings("config.yaml")


def test_system_info_reads_ollama_version(settings: Settings) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert str(req.url) == "http://localhost:11434/api/version"
        return httpx.Response(200, json={"version": "0.12.3"})

    info = system_info(settings, http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert info["ollama_version"] == "0.12.3"
    assert info["python"].count(".") == 2 and info["cpu"]


def test_system_info_ollama_unreachable_is_none(settings: Settings) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=req)

    info = system_info(settings, http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert info["ollama_version"] is None


def test_macos_cpu_from_sysctl(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    monkeypatch.setattr(metrics.sys, "platform", "darwin")
    monkeypatch.setattr(metrics.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="Apple M2 Pro\n"))
    assert metrics._cpu_name() == "Apple M2 Pro"

    def fail(*a: object, **k: object) -> None:
        raise FileNotFoundError("sysctl")

    monkeypatch.setattr(metrics.subprocess, "run", fail)
    monkeypatch.setattr(metrics.platform, "processor", lambda: "arm")
    assert metrics._cpu_name() == "arm"
