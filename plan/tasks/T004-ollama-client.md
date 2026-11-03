# T004 · Ollama client with streaming timing

## Goal
A real `OllamaClient` that streams from `POST {edge.host}/api/generate` and returns a `Generation` with correct TTFT, total time and tokens/s. `make_client("edge", …, mock=False)` returns it.

## Context
SPEC R2, R7 · ARCHITECTURE §5 (`ModelClient`, `ClientError`, `ClientUnavailable`), §6 (measurement rules), §7 (errors) · ADR 0003.

## Read first
`src/eclm/clients/`, `src/eclm/types.py`. Ollama API: request `{"model","prompt","stream":true,"options":{"temperature","num_predict"}}`. The response is newline-delimited JSON chunks `{"response": "...", "done": false}`, and the final chunk has `done: true` plus `eval_count` and `eval_duration` (ns).

## Contracts
`OllamaClient(model: str, settings: Settings, http: httpx.Client | None = None)` (inject `http` for tests). `check()` calls `GET /api/tags`. Connection refused → `ClientUnavailable("Ollama is not running at <host>. Start the Ollama app or run `ollama serve`.")`. Model not listed → `ClientUnavailable("Model <m> not found. Run: ollama pull <m>")`.

## Acceptance criteria
1. (R7) With `httpx.MockTransport` streaming 3 content chunks + a final chunk (`eval_count=30`, `eval_duration=1.5e9`), `generate` returns the joined text, `output_tokens=30`, `tokens_per_s=20.0`, `0 <= ttft_s <= total_s`, `ran_on="edge"`. Verify: test.
2. (R7) TTFT is taken at the first chunk with non-empty `response`, not at an empty first chunk. Verify: test with an empty first chunk and a controlled clock (monkeypatch `time.perf_counter`).
3. (R2) `check()` raises `ClientUnavailable` with the messages above for connection refused and missing model. Verify: tests.
4. HTTP 500 or a timeout during `generate` raises `ClientError` with the status/reason. Verify: tests.
5. `run.py` (non-mock) calls `check()` before the first question, and exits 1 printing the message on `ClientUnavailable`. Verify: test via the CLI entry function with a patched client factory.
6. `scripts/verify.sh` passes.

## Out of scope
Memory sampling (T005), retries for Ollama (local, not needed).

## Existing tests that may change
None.

## Notes
Use `http.stream("POST", …)` and `iter_lines()`. Timeout from `edge.timeout_s`.
