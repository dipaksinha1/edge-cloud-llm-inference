# T005 · Memory sampler and run metadata

## Goal
Edge rows carry `peak_mem_gb`, and every run writes `<result file>.meta.json` with machine info and config.

## Context
SPEC R7, R13, N5, assumption A6 · ARCHITECTURE §5 (`metrics.py`), §6 (memory rule), §7 (run metadata).

## Read first
`src/eclm/strategies.py` (EdgeOnly), `src/eclm/run.py`, `src/eclm/results.py`.

## Contracts
`MemorySampler(hz=10.0, process_match="ollama", pids=None)`: a context manager running a daemon thread that sums the RSS of the given `pids`, or else of all processes whose name contains `process_match` (case-insensitive), at each tick, and keeps the max as `.peak_gb` (None if no matching process was ever seen). `system_info(settings) -> dict` with keys `os, os_version, machine, cpu, total_ram_gb, python, ollama_version` (from `GET /api/version`, None if unreachable). On macOS, `cpu` comes from `sysctl -n machdep.cpu.brand_string` when available, else `platform.processor()`.

## Acceptance criteria
1. (R7) With `psutil.process_iter` monkeypatched to fake processes whose RSS changes over time, the sampler reports the peak of the summed RSS in GB. Verify: test.
2. (R7) EdgeOnly wraps each edge call in a sampler, and the row's `peak_mem_gb` = max over that question's calls (blank for cloud rows). In mock mode, the sampler is created with `pids=[os.getpid()]` so the column is filled. Verify: mock run test.
3. (R13) Each run writes `results/<name>.meta.json` with `run_id, started_at, args, config` (no secrets: assert the key value isn't present when the env var is set) and `system_info`. Verify: test.
4. (N5) Sampler thread start/stop adds <50 ms per call in a test with a 10 Hz sampler. Verify: test timing with tolerance.
5. `scripts/verify.sh` passes.

## Out of scope
GPU-specific APIs, energy.

## Existing tests that may change
T001 run tests may gain assertions for `peak_mem_gb`.

## Notes
Handle `psutil.NoSuchProcess` / `AccessDenied` silently per process.
