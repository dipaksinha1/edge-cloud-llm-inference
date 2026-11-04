"""CloudClient (Groq, OpenAI-compatible) and CloudOnly (T006), through httpx.MockTransport; no real Groq."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from eclm import data, run
from eclm.clients import ClientError, ClientUnavailable, CloudClient, make_client
from eclm.clients import cloud as cloud_mod
from eclm.config import Settings, load_settings

Handler = Callable[[httpx.Request], httpx.Response]
BASE = "https://api.groq.com/openai/v1"
MODEL = "llama-3.3-70b-versatile"
KEY = "gsk_SENTINEL_key_9f8e7d"


@pytest.fixture
def settings(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("GROQ_API_KEY", f"  {KEY}\n")  # whitespace is stripped
    return load_settings("config.yaml")


def _chunk(content: str | None = None, usage: int | None = None) -> bytes:
    body: dict[str, object] = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": MODEL}
    body["choices"] = [] if content is None else [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
    if usage is not None:
        body["usage"] = {"prompt_tokens": 12, "completion_tokens": usage, "total_tokens": 12 + usage}
    return b"data: " + json.dumps(body).encode() + b"\n\n"


DONE = b"data: [DONE]\n\n"
SSE = {"content-type": "text/event-stream"}


def _sse(parts: list[bytes]) -> httpx.Response:
    return httpx.Response(200, headers=SSE, content=b"".join(parts))


def _models(*ids: str) -> httpx.Response:
    return httpx.Response(200, json={"object": "list", "data": [
        {"id": i, "object": "model", "created": 0, "owned_by": "x"} for i in ids]})


def _client(settings: Settings, handler: Handler, sleeps: list[float] | None = None) -> CloudClient:
    rec = sleeps if sleeps is not None else []
    return CloudClient(MODEL, settings, http=httpx.Client(transport=httpx.MockTransport(handler)), sleep=rec.append)


def test_generate_streams_text_usage_and_tps(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    monkeypatch.setattr(cloud_mod.time, "perf_counter", lambda: now[0])
    timeline = [(0.3, _chunk("")), (0.5, _chunk("Answer")), (0.9, _chunk(": ")), (1.5, _chunk("42")),
                (1.6, _chunk(usage=40)), (1.7, DONE)]
    seen: list[httpx.Request] = []

    def body() -> Iterator[bytes]:
        for t, part in timeline:
            now[0] = t
            yield part

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, headers=SSE, content=body())

    gen = _client(settings, handler).generate("What is 6*7?", temperature=0.0, max_tokens=64)
    assert gen.text == "Answer: 42"
    assert gen.output_tokens == 40
    assert gen.ttft_s == pytest.approx(0.5) and gen.total_s == pytest.approx(1.7)
    assert gen.tokens_per_s == pytest.approx(40 / (1.7 - 0.5))
    assert (gen.ran_on, gen.model, gen.peak_mem_gb) == ("cloud", MODEL, None)
    req = seen[0]
    assert str(req.url) == f"{BASE}/chat/completions"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    sent = json.loads(req.content)
    assert sent["messages"] == [{"role": "user", "content": "What is 6*7?"}]
    assert (sent["model"], sent["temperature"], sent["max_tokens"], sent["seed"]) == (MODEL, 0.0, 64, 42)
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}


def test_missing_usage_counts_content_chunks(settings: Settings) -> None:
    gen = _client(settings, lambda r: _sse([_chunk("a"), _chunk("b"), _chunk("c"), DONE])).generate(
        "p", temperature=0.0, max_tokens=8)
    assert (gen.text, gen.output_tokens) == ("abc", 3)


def test_429_twice_then_ok_backs_off_1_then_2(settings: Settings) -> None:
    calls = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        calls[0] += 1
        if calls[0] <= 2:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return _sse([_chunk("ok"), _chunk(usage=1), DONE])

    sleeps: list[float] = []
    gen = _client(settings, handler, sleeps).generate("p", temperature=0.0, max_tokens=8)
    assert gen.text == "ok" and sleeps == [1.0, 2.0] and calls[0] == 3


@pytest.mark.parametrize(
    "fail",
    [
        lambda req: httpx.Response(429, json={"error": {"message": "rate limited"}}),
        lambda req: httpx.Response(503, json={"error": {"message": "over capacity"}}),
    ],
)
def test_six_retryable_failures_raise_client_error(settings: Settings, fail: Handler) -> None:
    sleeps: list[float] = []
    with pytest.raises(ClientError, match="after 6 attempts"):
        _client(settings, fail, sleeps).generate("p", temperature=0.0, max_tokens=8)
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_connection_error_is_retried(settings: Settings) -> None:
    calls = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        calls[0] += 1
        if calls[0] == 1:
            raise httpx.ConnectError("network unreachable", request=req)
        return _sse([_chunk("ok"), DONE])

    sleeps: list[float] = []
    assert _client(settings, handler, sleeps).generate("p", temperature=0.0, max_tokens=8).text == "ok"
    assert sleeps == [1.0]


def test_400_is_not_retried(settings: Settings) -> None:
    sleeps: list[float] = []
    with pytest.raises(ClientError, match="HTTP 400"):
        _client(settings, lambda r: httpx.Response(400, json={"error": {"message": "bad"}}), sleeps).generate(
            "p", temperature=0.0, max_tokens=8)
    assert sleeps == []


def test_sampled_calls_use_seed_plus_k(settings: Settings) -> None:
    seeds: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seeds.append(json.loads(req.content)["seed"])
        return _sse([_chunk("x"), DONE])

    c = _client(settings, handler)
    for _ in range(2):
        c.generate("same", temperature=0.7, max_tokens=8)
    c.generate("same", temperature=0.0, max_tokens=8)
    assert seeds == [42, 43, 42]


def test_check_ok(settings: Settings) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET" and str(req.url) == f"{BASE}/models"
        return _models("whisper-large-v3", MODEL)

    _client(settings, handler).check()


def test_check_missing_key(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "   ")
    c = CloudClient(MODEL, load_settings("config.yaml"))
    with pytest.raises(ClientUnavailable) as exc:
        c.check()
    assert str(exc.value) == "Set GROQ_API_KEY in .env (get a key at console.groq.com/keys)"
    with pytest.raises(ClientError):
        c.generate("p", temperature=0.0, max_tokens=8)


def test_check_401(settings: Settings) -> None:
    sleeps: list[float] = []
    with pytest.raises(ClientUnavailable) as exc:
        _client(settings, lambda r: httpx.Response(401, json={"error": {"message": "Invalid API Key"}}),
                sleeps).check()
    assert str(exc.value) == "Groq rejected the API key" and sleeps == []


def test_check_model_not_listed(settings: Settings) -> None:
    with pytest.raises(ClientUnavailable, match=MODEL):
        _client(settings, lambda r: _models("llama-3.1-8b-instant")).check()


def test_make_client_cloud_real(settings: Settings) -> None:
    c = make_client("cloud", settings)
    assert isinstance(c, CloudClient) and c.name == MODEL and c.ran_on == "cloud"


def test_mock_cloud_run_rows(workdir: Path) -> None:
    assert data.main(["--mock"]) == 0
    assert run.main(["--strategy", "cloud", "--network", "normal", "--mock"]) == 0
    n = len((workdir / "data" / data.TEST_FILE).read_text().splitlines())
    with open(workdir / "results" / "cloud__none__normal.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n > 0
    for r in rows:
        assert (r["strategy"], r["ran_on"], r["cloud_calls"], r["edge_calls"]) == ("cloud", "cloud", "1", "0")
        assert r["peak_mem_gb"] == "" and r["edge_model"] == "" and r["cloud_model"] == MODEL
        assert r["error"] == ""


def test_missing_key_run_exits_1_before_any_row(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    data.main(["--mock"])
    assert run.main(["--strategy", "cloud", "--network", "normal"]) == 1
    assert "Set GROQ_API_KEY in .env" in capsys.readouterr().err
    assert not (workdir / "results" / "cloud__none__normal.csv").exists()


def test_key_never_written_to_logs_csv_or_meta(
    settings: Settings, workdir: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture,
) -> None:
    data.main(["--mock"])
    calls = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return _models(MODEL)
        calls[0] += 1
        if calls[0] == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})  # exercises the warning log
        return _sse([_chunk("Answer: 4"), _chunk(usage=3), DONE])

    def factory(kind: str, s: Settings, *, model: str | None = None, mock: bool = False) -> CloudClient:
        assert kind == "cloud" and not mock
        return _client(s, handler)

    monkeypatch.setattr(run, "make_client", factory)
    with caplog.at_level(logging.DEBUG):
        assert run.main(["--strategy", "cloud", "--network", "normal"]) == 0
    out = capsys.readouterr()
    results = workdir / "results"
    texts = [out.out, out.err, caplog.text] + [p.read_text() for p in results.iterdir()]
    assert any("retrying" in t for t in texts)
    assert sorted(p.name for p in results.iterdir()) == ["cloud__none__normal.csv", "cloud__none__normal.meta.json"]
    for t in texts:
        assert KEY not in t
