"""Edge memory sampling and machine information (R7, R13, ARCHITECTURE §5/§6)."""

from __future__ import annotations

import platform
import subprocess
import sys
import threading
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx
import psutil

from eclm.config import Settings

BYTES_PER_GB = 1024**3
_VERSION_TIMEOUT_S = 5.0  # GET /api/version is local; don't stall a run start on a hung server
_PROCESS_ERRORS = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess)


class MemorySampler:
    """Context manager: peak summed RSS (GB, 1024**3 bytes) of the model processes during a call.

    `pids` given: exactly those processes (mock mode passes [os.getpid()]). Otherwise every process whose
    name contains `process_match` (case-insensitive). Samples once on enter, at `hz` on a daemon thread,
    and once on exit, so even a very short call gets a reading. `.peak_gb` is None if nothing matched.
    """

    def __init__(self, hz: float = 10.0, process_match: str = "ollama", pids: list[int] | None = None) -> None:
        if hz <= 0:
            raise ValueError("hz must be > 0")
        self.interval = 1.0 / hz
        self.process_match = process_match.lower()
        self.pids = list(pids) if pids is not None else None
        self.peak_gb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _rss_bytes(self) -> int | None:
        total = 0
        seen = False
        if self.pids is not None:
            procs: list[Any] = []
            for pid in self.pids:
                try:
                    procs.append(psutil.Process(pid))
                except _PROCESS_ERRORS:
                    continue
        else:
            procs = []
            for p in psutil.process_iter():
                try:
                    if self.process_match in (p.name() or "").lower():
                        procs.append(p)
                except _PROCESS_ERRORS:
                    continue
        for p in procs:
            try:
                total += int(p.memory_info().rss)
                seen = True
            except _PROCESS_ERRORS:
                continue
        return total if seen else None

    def sample(self) -> None:
        rss = self._rss_bytes()
        if rss is None:
            return
        gb = rss / BYTES_PER_GB
        with self._lock:
            if self.peak_gb is None or gb > self.peak_gb:
                self.peak_gb = gb

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def __enter__(self) -> MemorySampler:
        self.sample()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="eclm-memory-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.sample()


def _cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _os_version() -> str:
    if sys.platform == "darwin":
        mac = platform.mac_ver()[0]
        if mac:
            return mac
    return platform.release()


def ollama_version(settings: Settings, http: httpx.Client | None = None) -> str | None:
    """Version from `GET {edge.host}/api/version`, or None if Ollama isn't reachable."""
    url = f"{settings.edge.host.rstrip('/')}/api/version"
    try:
        if http is not None:
            resp = http.get(url, timeout=_VERSION_TIMEOUT_S)
        else:
            resp = httpx.get(url, timeout=_VERSION_TIMEOUT_S, trust_env=False)  # localhost: ignore proxy env vars
        if resp.status_code != 200:
            return None
        version = resp.json().get("version")
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    return str(version) if version else None


def system_info(settings: Settings, *, ollama: bool = True, http: httpx.Client | None = None) -> dict[str, Any]:
    """Machine info for the run metadata. `ollama=False` (mock runs) skips the version request."""
    return {
        "os": platform.system(),
        "os_version": _os_version(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "total_ram_gb": round(psutil.virtual_memory().total / BYTES_PER_GB, 2),
        "python": platform.python_version(),
        "ollama_version": ollama_version(settings, http) if ollama else None,
    }
