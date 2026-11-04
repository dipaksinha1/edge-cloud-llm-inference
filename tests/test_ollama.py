"""OllamaClient (T004) through httpx.MockTransport; no real Ollama."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from eclm import data, run
from eclm.clients import ClientError, ClientUnavailable, OllamaClient, make_client
from eclm.clients import ollama as ollama_mod
from eclm.config import Settings, load_settings

Handler = Callable[[httpx.Request], httpx.Response]
HOST = "http://localhost:11434"


@pytest.fixture
def settings(workdir: Path) -> Settings:
    return load_settings("config.yaml")


def _client(settings: Settings, handler: Handler, model: str = "qwen2.5:3b") -> OllamaClient:
    return OllamaClient(model, settings, http=httpx.Client(transport=httpx.MockTransport(handler)))


def _ndjson(chunks: list[dict[str, object]]) -> bytes:
    return b"".join(json.dumps(c).encode() + b"\n" for c in chunks)


FINAL = {"response": "", "done": True, "eval_count": 30, "eval_duration": 1.5e9}
CONTENT = [{"response": "Six ", "done": False}, {"response": "times ", "done": False}, {"response": "7", "done": False}]


def test_generate_streams_text_and_timing(settings: Settings) -> None:
    seen: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST" and str(req.url) == f"{HOST}/api/generate"
        seen.append(json.loads(req.content))
        return httpx.Response(200, content=_ndjson([*CONTENT, FINAL]))

    gen = _client(settings, handler).generate("What is 6*7?", temperature=0.0, max_tokens=64)
    assert gen.text == "Six times 7"
    assert gen.output_tokens == 30
    assert gen.tokens_per_s == pytest.approx(20.0)
    assert 0 <= gen.ttft_s <= gen.total_s
    assert (gen.ran_on, gen.model, gen.peak_mem_gb) == ("edge", "qwen2.5:3b", None)
    body = seen[0]
    assert body["model"] == "qwen2.5:3b" and body["prompt"] == "What is 6*7?" and body["stream"] is True
    assert body["options"] == {"temperature": 0.0, "num_predict": 64, "seed": settings.seed}


def test_ttft_is_first_non_empty_chunk(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    monkeypatch.setattr(ollama_mod.time, "perf_counter", lambda: now[0])
    timeline = [  # (clock when the chunk arrives, chunk)
        (0.5, {"response": "", "done": False}),
        (2.0, {"response": "Answer", "done": False}),
        (2.5, {"response": ": 4", "done": False}),
        (3.0, FINAL),
    ]

    def body() -> Iterator[bytes]:
        for t, chunk in timeline:
            now[0] = t
            yield json.dumps(chunk).encode() + b"\n"
        now[0] = 3.25  # stream closes

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())

    gen = _client(settings, handler).generate("p", temperature=0.0, max_tokens=8)
    assert gen.ttft_s == pytest.approx(2.0)
    assert gen.total_s == pytest.approx(3.0)
    assert gen.text == "Answer: 4"


def test_sampled_calls_use_seed_plus_k(settings: Settings) -> None:
    seeds: list[object] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seeds.append(json.loads(req.content)["options"]["seed"])
        return httpx.Response(200, content=_ndjson([*CONTENT, FINAL]))

    c = _client(settings, handler)
    for _ in range(3):
        c.generate("same", temperature=0.7, max_tokens=8)
    c.generate("same", temperature=0.0, max_tokens=8)
    c.generate("other", temperature=0.7, max_tokens=8)
    s = settings.seed
    assert seeds == [s, s + 1, s + 2, s, s]


def test_check_ok_accepts_listed_model_and_latest_tag(settings: Settings) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET" and str(req.url) == f"{HOST}/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b", "model": "qwen2.5:3b"},
                                                    {"name": "llama3.2:latest"}]})

    _client(settings, handler).check()
    _client(settings, handler, model="llama3.2").check()


def test_check_connection_refused(settings: Settings) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused", request=req)

    with pytest.raises(ClientUnavailable) as exc:
        _client(settings, handler).check()
    assert str(exc.value) == f"Ollama is not running at {HOST}. Start the Ollama app or run `ollama serve`."


def test_check_model_missing(settings: Settings) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:0.5b"}]})

    with pytest.raises(ClientUnavailable) as exc:
        _client(settings, handler).check()
    assert str(exc.value) == "Model qwen2.5:3b not found. Run: ollama pull qwen2.5:3b"


def test_generate_http_500_is_client_error(settings: Settings) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "model runner crashed"})

    with pytest.raises(ClientError, match=r"500 Internal Server Error.*model runner crashed"):
        _client(settings, handler).generate("p", temperature=0.0, max_tokens=8)


def test_generate_timeout_is_client_error(settings: Settings) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=req)

    with pytest.raises(ClientError, match=r"timeout after 300(\.0)?s: ReadTimeout"):
        _client(settings, handler).generate("p", temperature=0.0, max_tokens=8)


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        ([{"error": "out of memory"}], "Ollama error: out of memory"),
        (CONTENT, "without a final done chunk"),
    ],
)
def test_generate_stream_errors(settings: Settings, chunks: list[dict[str, object]], message: str) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_ndjson(chunks))

    with pytest.raises(ClientError, match=message):
        _client(settings, handler).generate("p", temperature=0.0, max_tokens=8)


def test_make_client_edge_real_is_ollama(settings: Settings) -> None:
    c = make_client("edge", settings, model="llama3.2:1b")
    assert isinstance(c, OllamaClient) and c.name == "llama3.2:1b" and c.ran_on == "edge"


def test_run_checks_before_first_question_and_exits_1(
    workdir: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data.main(["--mock"])
    requests: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req.url.path)
        raise httpx.ConnectError("Connection refused", request=req)

    def factory(kind: str, s: Settings, *, model: str | None = None, mock: bool = False) -> OllamaClient:
        assert kind == "edge" and not mock
        return _client(s, handler, model=model or s.edge.hybrid_model)

    monkeypatch.setattr(run, "make_client", factory)
    assert run.main(["--strategy", "edge", "--edge-model", "qwen2.5:3b", "--network", "normal"]) == 1
    assert "Ollama is not running at http://localhost:11434" in capsys.readouterr().err
    assert requests == ["/api/tags"]  # checked, and no question was sent
    assert not (workdir / "results").exists() or not any((workdir / "results").iterdir())


def test_tokens_per_s_falls_back_without_eval_duration(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    monkeypatch.setattr(ollama_mod.time, "perf_counter", lambda: now[0])

    def body() -> Iterator[bytes]:
        for t, chunk in ((1.0, CONTENT[0]), (3.0, {"response": "", "done": True, "eval_count": 10})):
            now[0] = t
            yield json.dumps(chunk).encode() + b"\n"

    gen = _client(settings, lambda req: httpx.Response(200, content=body())).generate(
        "p", temperature=0.0, max_tokens=8
    )
    assert gen.tokens_per_s == pytest.approx(5.0)  # 10 tokens over (total 3.0 - ttft 1.0)


def test_check_non_200_is_unavailable(settings: Settings) -> None:
    with pytest.raises(ClientUnavailable, match="HTTP 503"):
        _client(settings, lambda req: httpx.Response(503)).check()
