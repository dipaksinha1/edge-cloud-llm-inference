# T006 · Groq cloud client and cloud-only strategy

## Goal
A real `CloudClient` for Groq (OpenAI-compatible, streaming) and a working `--strategy cloud` (mock and real).

## Context
SPEC R3, R7, N2 · ARCHITECTURE §4 (`cloud.*`), §5, §6 (cloud tokens/s), §7 (back-off, secrets) · ADR 0002.

## Read first
`src/eclm/clients/`, `src/eclm/strategies.py`, `src/eclm/run.py`.

## Contracts
`CloudClient(model: str, settings: Settings, http: httpx.Client | None = None)` builds `openai.OpenAI(base_url=settings.cloud.base_url, api_key=<env var>, http_client=http, max_retries=0)` and calls `chat.completions.create(model, messages=[{"role":"user","content":prompt}], temperature, max_tokens, stream=True, stream_options={"include_usage": True})`. It implements its own back-off (1, 2, 4, 8, 16 s; injectable `sleep` for tests) on 429/5xx/connection errors, up to `max_retries`, then `ClientError`. `check()`: a missing env var → `ClientUnavailable("Set GROQ_API_KEY in .env (get a key at console.groq.com/keys)")`. A 401 on `GET /models` → `ClientUnavailable("Groq rejected the API key")`. A model not in the list → `ClientUnavailable` naming it. `CloudOnly(cloud, settings)` strategy. `make_client("cloud", …, mock=False)` returns it.

## Acceptance criteria
1. (R7) With a `MockTransport` serving an SSE stream (`data: {...}` lines, a final usage chunk `completion_tokens=40`, `data: [DONE]`), `generate` returns the joined text, `output_tokens=40`, `tokens_per_s = 40/(total-ttft)`, `ran_on="cloud"`. Verify: test.
2. (R3) A 429 twice then 200 → succeeds after 2 sleeps of 1 s and 2 s. Six 429s → `ClientError`. Verify: test with recorded sleeps.
3. (R3, N2) A missing key → `ClientUnavailable` with the message above, and the run exits 1 before any row. The key value never appears in logs, CSV or meta.json. Verify: tests (set a sentinel key, grep outputs).
4. (R3) `python -m eclm.run --strategy cloud --network normal --mock` writes `results/cloud__none__normal.csv` with `ran_on=cloud`, `cloud_calls=1`, `edge_calls=0`, blank `peak_mem_gb`, `edge_model` blank, `cloud_model` filled. It's added to the verify.sh smoke. Verify: verify.sh + test.
5. `scripts/verify.sh` passes.

## Out of scope
Cloud call budget (T007). Other providers (config only).

## Existing tests that may change
None.

## Notes
Never print the key. Strip whitespace from it. If usage is missing, count content chunks as tokens and note it in the code.
