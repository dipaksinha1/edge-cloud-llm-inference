"""A failed stage-2 edge sample's time counts in the hybrid row's latency (scripted clients, fake clock)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import ScriptedClient, StubRouterModel

from eclm import router, run
from eclm.clients import ClientError
from eclm.config import Settings, load_settings
from eclm.router import Router
from eclm.strategies import Hybrid, StrategyError
from eclm.types import Generation, Question, Result

REPO = Path(__file__).resolve().parents[1]
GSM = Question("g1", "gsm8k", "Tom has 70 apples and buys 2 more. How many?", "72")
EASY = 0.1  # below the router threshold: stage 2 runs


@pytest.fixture
def settings() -> Settings:
    return load_settings(REPO / "config.yaml")


def _fake_clock(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    """The router reads the clock before every sample and again when one fails."""
    it = iter(values)
    monkeypatch.setattr(router, "_clock", lambda: next(it))


def _hybrid(settings: Settings, edge_script: list[object], edge_timings: list[tuple[float, float]],
            cloud_script: list[object] | None = None) -> Hybrid:
    edge = ScriptedClient(edge_script, timings=edge_timings)
    cloud = ScriptedClient(cloud_script or ["72"], ran_on="cloud", name="cloud-model", timings=[(0.3, 2.0)])
    r = Router(StubRouterModel(EASY), edge, settings)  # type: ignore[arg-type]
    return Hybrid(edge, cloud, r, settings)


def _row(answer: Callable[[], tuple[str, list[Generation], dict[str, object]]], error: str = "") -> Result:
    text, gens, extra = answer()
    return run.build_result(run_id="r", strategy="hybrid", edge_model="e", cloud_model="c", network="normal",
                            q=GSM, text=text, gens=gens, extra=dict(extra), error=error)


# --- AC1: failure before the answering sample counts in total and ttft --------------------------------


def test_failure_before_answering_sample(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    # sample 1 fails after 2.5 s; samples 2 and 3 agree on 72 (answering call = sample 2)
    _fake_clock(monkeypatch, [10.0, 12.5, 20.0, 30.0])
    h = _hybrid(settings, [ClientError("Ollama timeout"), "72", "72"], [(0.0, 0.0), (0.4, 1.0), (0.5, 1.5)])
    text, gens, extra = h.answer(GSM)
    assert extra["failed_edge_s"] == ((0, 2.5),)
    r = run.build_result(run_id="r", strategy="hybrid", edge_model="e", cloud_model="c", network="normal",
                         q=GSM, text=text, gens=gens, extra=extra)
    assert r.ran_on == "edge" and r.router_stage == "" and r.agreement == "2/3"
    assert r.edge_calls == 2 and r.cloud_calls == 0
    assert r.total_s == pytest.approx(2.5 + 1.0 + 1.5)
    assert r.ttft_s == pytest.approx(2.5 + 0.4)
    assert r.output_tokens == 11  # the answering call's (ScriptedClient: tokens + call index 1)


# --- AC2: failure after the answering sample counts in total only ------------------------------------


def test_failure_after_answering_sample(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    # samples 1 and 2 agree (answering call = sample 1); sample 3 fails after 4.0 s
    _fake_clock(monkeypatch, [0.0, 1.0, 2.0, 6.0])
    h = _hybrid(settings, ["72", "72", ClientError("connection reset")], [(0.2, 1.0), (0.3, 1.2), (0.0, 0.0)])
    r = _row(lambda: h.answer(GSM))
    assert r.ran_on == "edge" and r.edge_calls == 2 and r.agreement == "2/3"
    assert r.total_s == pytest.approx(1.0 + 1.2 + 4.0)
    assert r.ttft_s == pytest.approx(0.2)


# --- AC3: stage 2 -> cloud with a failed sample: counts in both ---------------------------------------


def test_failure_then_disagreement_goes_to_cloud(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    # sample 1 = 70, sample 2 fails after 3.0 s, sample 3 = 68: no majority -> cloud (TTFT 0.3, total 2.0)
    _fake_clock(monkeypatch, [0.0, 5.0, 8.0, 9.0])
    h = _hybrid(settings, ["70", ClientError("Ollama timeout"), "68"], [(0.2, 1.0), (0.0, 0.0), (0.4, 1.5)])
    r = _row(lambda: h.answer(GSM))
    assert r.ran_on == "cloud" and r.router_stage == "2" and r.agreement == "1/3"
    assert (r.edge_calls, r.cloud_calls) == (2, 1)
    assert r.total_s == pytest.approx(1.0 + 3.0 + 1.5 + 2.0)
    assert r.ttft_s == pytest.approx(1.0 + 3.0 + 1.5 + 0.3)


def test_failed_cloud_call_after_failed_sample_keeps_the_time(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # error row: the failed sample's time is still part of total_s (completed calls + failed sample)
    _fake_clock(monkeypatch, [0.0, 2.0, 3.0, 4.0])
    h = _hybrid(settings, [ClientError("timeout"), "70", "68"], [(0.0, 0.0), (0.2, 1.0), (0.3, 1.5)],
                cloud_script=[ClientError("HTTP 500")])
    with pytest.raises(StrategyError) as exc:
        h.answer(GSM)
    gens, extra = exc.value.generations, exc.value.extra
    r = run.build_result(run_id="r", strategy="hybrid", edge_model="e", cloud_model="c", network="normal",
                         q=GSM, text="", gens=gens, extra=extra, error="ClientError: HTTP 500")
    assert extra["failed_edge_s"] == ((0, 2.0),)
    assert r.total_s == pytest.approx(2.0 + 1.0 + 1.5) and r.edge_calls == 2


def test_every_sample_failed_and_cloud_failed(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_clock(monkeypatch, [0.0, 1.0, 1.0, 3.0, 3.0, 6.0])
    h = _hybrid(settings, [ClientError("a"), ClientError("b"), ClientError("c")], [(0.0, 0.0)] * 3,
                cloud_script=[ClientError("HTTP 500")])
    with pytest.raises(StrategyError) as exc:
        h.answer(GSM)
    assert exc.value.extra["failed_edge_s"] == ((0, 1.0), (0, 2.0), (0, 3.0))
    assert run.combine_timings([], None, exc.value.extra["failed_edge_s"]) == (0.0, pytest.approx(6.0))


# --- AC4: no failures -> unchanged ---------------------------------------------------------------------


def test_no_failure_adds_nothing(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_clock(monkeypatch, [0.0, 1.0, 2.0])
    h = _hybrid(settings, ["72", "72", "72"], [(0.2, 1.0), (0.3, 1.2), (0.4, 1.4)])
    text, gens, extra = h.answer(GSM)
    assert "failed_edge_s" not in extra
    assert run.combine_timings(gens, 0) == run.combine_timings(gens, 0, ())


def test_combine_timings_failed_positions() -> None:
    def g(ttft: float, total: float) -> Generation:
        return Generation("Answer: 1", ttft, total, 1, 1.0, "m", "edge")

    gens = [g(0.1, 1.0), g(0.2, 2.0), g(0.3, 3.0)]
    failed = ((0, 0.5), (2, 0.7), (3, 0.9))  # before call 0, before call 2, after the last call
    # answering call 1: only the failure before call 0 is before it
    assert run.combine_timings(gens, 1, failed) == (pytest.approx(0.5 + 1.0 + 0.2), pytest.approx(6.0 + 2.1))
    # answering call = last (2): failures at positions 0 and 2 come before it
    assert run.combine_timings(gens, None, failed) == (pytest.approx(0.5 + 0.7 + 3.0 + 0.3), pytest.approx(8.1))
