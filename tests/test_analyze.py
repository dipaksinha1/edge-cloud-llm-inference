"""T011: analysis summary table (handcrafted CSVs and the mock pipeline, no network)."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from eclm import analyze, data, router, run
from eclm.types import RESULT_COLUMNS


def _row(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {c: "" for c in RESULT_COLUMNS}
    base |= {"run_id": "r", "strategy": "edge", "edge_model": "m1", "network": "normal", "dataset": "gsm8k",
             "ran_on": "edge", "edge_calls": 1, "cloud_calls": 0, "ttft_s": 0.0, "total_s": 0.0,
             "output_tokens": 1, "tokens_per_s": 0.0, "correct": False, "parse_failed": False}
    return base | kw


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        w.writerows(rows)


EDGE_ROWS = [  # edge m1 normal: 4 rows, 1 error
    _row(qid="a", dataset="gsm8k", ttft_s=1.0, total_s=2.0, tokens_per_s=10.0, peak_mem_gb="1.0", correct=True),
    _row(qid="b", dataset="gsm8k", ttft_s=3.0, total_s=6.0, tokens_per_s=20.0, peak_mem_gb="3.0",
         parse_failed=True),
    _row(qid="c", dataset="mmlu", ttft_s=2.0, total_s=10.0, tokens_per_s=30.0, peak_mem_gb="2.0", correct=True),
    _row(qid="d", dataset="mmlu", ttft_s=99.0, total_s=99.0, tokens_per_s=99.0, peak_mem_gb="9.0",
         error="ClientError: timeout"),
]
CLOUD_ROWS = [  # cloud-only: blank edge model and memory
    _row(qid="a", strategy="cloud", edge_model="", ran_on="cloud", edge_calls=0, cloud_calls=1, ttft_s=0.5,
         total_s=1.0, tokens_per_s=100.0, correct=True),
    _row(qid="c", strategy="cloud", edge_model="", ran_on="cloud", dataset="mmlu", edge_calls=0, cloud_calls=1,
         ttft_s=1.5, total_s=3.0, tokens_per_s=200.0, correct=True),
]
HYBRID_ROWS = [
    _row(qid="a", strategy="hybrid", ran_on="cloud", total_s=4.0, ttft_s=1.0, correct=True),
    _row(qid="b", strategy="hybrid", ran_on="edge", total_s=2.0, ttft_s=1.0, peak_mem_gb="2.5"),
    _row(qid="c", strategy="hybrid", ran_on="edge", dataset="mmlu", total_s=6.0, ttft_s=3.0, peak_mem_gb="1.5",
         correct=True),
]


@pytest.fixture
def handmade(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    _write(results / "hybrid__m1__normal.csv", HYBRID_ROWS)
    _write(results / "edge__m1__normal.csv", EDGE_ROWS)
    _write(results / "cloud__none__normal.csv", CLOUD_ROWS)
    _write(results / "edge__m1__degraded.csv", [])  # header only: contributes nothing
    return results


def _by(summary: pd.DataFrame, strategy: str) -> pd.Series:
    return summary[summary["strategy"] == strategy].iloc[0]


# --- AC1: every column against hand-computed values -----------------------------------------------------


def test_load_results_skips_summary_and_types_columns(handmade: Path) -> None:
    _write(handmade / "summary.csv", [_row(qid="zzz")])
    df = analyze.load_results(handmade)
    assert len(df) == 9 and "zzz" not in set(df["qid"])
    assert df["correct"].dtype == bool and df["ttft_s"].dtype == float
    assert df.loc[df["strategy"] == "cloud", "peak_mem_gb"].isna().all()
    assert set(df.loc[df["strategy"] == "cloud", "edge_model"]) == {""}


def test_summary_columns_and_order(handmade: Path) -> None:
    s = analyze.summarize(analyze.load_results(handmade))
    assert tuple(s.columns) == analyze.SUMMARY_COLUMNS == (
        "strategy", "edge_model", "network", "n", "accuracy", "acc_gsm8k", "acc_mmlu", "ttft_mean", "ttft_median",
        "total_mean", "total_median", "tps_mean", "mem_peak_mean", "offload_ratio", "parse_fail_rate",
        "error_rate")
    assert list(s["strategy"]) == ["edge", "cloud", "hybrid"]  # ARCH §10 order, not file order


def test_edge_group_matches_hand_computed(handmade: Path) -> None:
    e = _by(analyze.summarize(analyze.load_results(handmade)), "edge")
    assert (e["edge_model"], e["network"], e["n"]) == ("m1", "normal", 4)
    assert e["accuracy"] == pytest.approx(2 / 4)
    assert e["acc_gsm8k"] == pytest.approx(1 / 2)
    assert e["acc_mmlu"] == pytest.approx(1 / 2)
    assert e["ttft_mean"] == pytest.approx(2.0)      # (1+3+2)/3, error row excluded
    assert e["ttft_median"] == pytest.approx(2.0)
    assert e["total_mean"] == pytest.approx(6.0)     # (2+6+10)/3
    assert e["total_median"] == pytest.approx(6.0)
    assert e["tps_mean"] == pytest.approx(20.0)
    assert e["mem_peak_mean"] == pytest.approx(2.0)
    assert e["offload_ratio"] == 0.0
    assert e["parse_fail_rate"] == pytest.approx(1 / 4)
    assert e["error_rate"] == pytest.approx(1 / 4)


def test_cloud_and_hybrid_groups_match_hand_computed(handmade: Path) -> None:
    s = analyze.summarize(analyze.load_results(handmade))
    c = _by(s, "cloud")
    assert (c["edge_model"], c["n"], c["accuracy"]) == ("", 2, 1.0)
    assert c["ttft_mean"] == pytest.approx(1.0) and c["total_median"] == pytest.approx(2.0)
    assert c["tps_mean"] == pytest.approx(150.0)
    assert math.isnan(c["mem_peak_mean"])
    assert c["offload_ratio"] == 1.0 and c["error_rate"] == 0.0 and c["parse_fail_rate"] == 0.0
    h = _by(s, "hybrid")
    assert h["n"] == 3 and h["accuracy"] == pytest.approx(2 / 3)
    assert h["acc_gsm8k"] == pytest.approx(1 / 2) and h["acc_mmlu"] == 1.0
    assert h["offload_ratio"] == pytest.approx(1 / 3)
    assert h["total_mean"] == pytest.approx(4.0) and h["ttft_median"] == pytest.approx(1.0)
    assert h["mem_peak_mean"] == pytest.approx(2.0)  # mean of the edge rows that have a peak


def test_dataset_missing_from_group_is_nan() -> None:
    df = pd.DataFrame([_row(qid="x", correct=True)])
    df = df.astype({"ttft_s": float, "total_s": float, "tokens_per_s": float})
    df["peak_mem_gb"] = math.nan
    s = analyze.summarize(df)
    assert s.iloc[0]["acc_gsm8k"] == 1.0 and math.isnan(s.iloc[0]["acc_mmlu"])


# --- AC2: error rows ------------------------------------------------------------------------------------


def test_error_rows_wrong_in_accuracy_out_of_timing() -> None:
    rows = [_row(qid="a", ttft_s=1.0, total_s=1.0, tokens_per_s=5.0, peak_mem_gb="1.0", correct=True),
            _row(qid="b", ttft_s=50.0, total_s=50.0, tokens_per_s=50.0, peak_mem_gb="50.0",
                 error="ClientError: x")]
    df = pd.DataFrame(rows).astype({"ttft_s": float, "total_s": float, "tokens_per_s": float})
    df["peak_mem_gb"] = pd.to_numeric(df["peak_mem_gb"])
    r = analyze.summarize(df).iloc[0]
    assert r["accuracy"] == 0.5 and r["error_rate"] == 0.5
    assert r["ttft_mean"] == 1.0 and r["total_mean"] == 1.0 and r["total_median"] == 1.0
    assert r["tps_mean"] == 5.0 and r["mem_peak_mean"] == 1.0


def test_all_error_group_has_blank_timings(tmp_path: Path) -> None:
    _write(tmp_path / "edge__m1__normal.csv", [_row(qid="a", error="ClientError: x", total_s=3.0)])
    r = analyze.summarize(analyze.load_results(tmp_path)).iloc[0]
    assert r["error_rate"] == 1.0 and r["accuracy"] == 0.0 and math.isnan(r["total_mean"])
    assert analyze.main(["--results-dir", str(tmp_path)]) == 0
    md = (tmp_path / "summary.md").read_text()
    assert "| edge | m1 | normal | 1 | 0.000 |" in md


# --- AC3: empty results dir -----------------------------------------------------------------------------


@pytest.mark.parametrize("setup", ["missing", "empty", "header_only", "summary_only"])
def test_no_results_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str], setup: str) -> None:
    results = tmp_path / "results"
    if setup != "missing":
        results.mkdir()
    if setup == "header_only":
        _write(results / "edge__m1__normal.csv", [])
    if setup == "summary_only":
        (results / "summary.csv").write_text("strategy\nedge\n")
    assert analyze.main(["--results-dir", str(results)]) == 1
    assert "no results found" in capsys.readouterr().err


# --- outputs and CLI ------------------------------------------------------------------------------------


def test_cli_writes_csv_and_markdown(handmade: Path) -> None:
    assert analyze.main(["--results-dir", str(handmade)]) == 0
    with open(handmade / "summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == analyze.SUMMARY_COLUMNS
        rows = list(reader)
    assert [r["strategy"] for r in rows] == ["edge", "cloud", "hybrid"]
    assert rows[1]["mem_peak_mean"] == "" and float(rows[0]["accuracy"]) == 0.5
    lines = (handmade / "summary.md").read_text().splitlines()
    assert lines[0] == "| " + " | ".join(analyze.SUMMARY_COLUMNS) + " |"
    assert len(lines) == 2 + 3
    assert lines[2].startswith("| edge | m1 | normal | 4 | 0.500 | 0.500 | 0.500 | 2.000 |")
    assert "| cloud |  | normal | 2 | 1.000 |" in lines[3]
    # summary.csv is skipped on input, so a rerun gives the same file
    before = (handmade / "summary.csv").read_bytes()
    assert analyze.main(["--results-dir", str(handmade)]) == 0
    assert (handmade / "summary.csv").read_bytes() == before


def test_default_results_dir_from_config_after_mock_pipeline(workdir: Path) -> None:
    assert data.main(["--mock"]) == 0
    assert router.main(["train", "--edge-model", "qwen2.5:3b", "--mock"]) == 0
    assert run.main(["--all", "--network", "normal", "--mock"]) == 0
    assert analyze.main([]) == 0
    summary = pd.read_csv(workdir / "results" / "summary.csv", keep_default_na=False)
    n_csvs = len([p for p in (workdir / "results").glob("*.csv") if p.name != "summary.csv"])
    assert len(summary) == n_csvs == 6
    assert set(summary["n"]) == {12}
    assert summary.loc[summary["strategy"] == "cloud", "offload_ratio"].tolist() == [1.0]
    assert summary.loc[summary["strategy"] == "edge", "offload_ratio"].tolist() == [0.0, 0.0, 0.0]


def test_csv_without_result_columns_exits_1(handmade: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (handmade / "notes.csv").write_text("qid,foo\na,1\n")
    assert analyze.main(["--results-dir", str(handmade)]) == 1
    err = capsys.readouterr().err
    assert "notes.csv is not a result file" in err and "strategy" in err


# --- T012: figures --------------------------------------------------------------------------------------

FIGURE_FILES = {f"{n}.{e}" for n in analyze.FIGURE_NAMES for e in ("png", "pdf")}


def _figures_ok(figdir: Path) -> None:
    assert FIGURE_FILES | {analyze.FIG3_SIDE_FILE} <= {p.name for p in figdir.iterdir()}
    for name in FIGURE_FILES:
        assert (figdir / name).stat().st_size > 0
    for name in analyze.FIGURE_NAMES:
        assert (figdir / f"{name}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert (figdir / f"{name}.pdf").read_bytes()[:5] == b"%PDF-"


def test_figures_from_full_mock_pipeline(workdir: Path) -> None:
    assert data.main(["--mock"]) == 0
    assert router.main(["train", "--edge-model", "qwen2.5:3b", "--mock"]) == 0
    assert run.main(["--all", "--network", "normal", "--mock"]) == 0
    assert run.main(["--strategy", "edge", "--edge-model", "qwen2.5:3b", "--network", "degraded", "--mock"]) == 0
    assert analyze.main([]) == 0
    _figures_ok(workdir / "figures")  # default figures_dir from config.yaml


def test_png_is_300_dpi_single_column(handmade: Path, tmp_path: Path) -> None:
    from matplotlib import image

    df = analyze.load_results(handmade)
    analyze.make_figures(analyze.summarize(df), df, tmp_path / "figs")
    height, width = image.imread(tmp_path / "figs" / "fig1_accuracy.png").shape[:2]
    assert width == pytest.approx(analyze.FIG_WIDTH_IN * analyze.FIG_DPI, abs=2)
    assert height == pytest.approx(analyze.FIG_HEIGHT_IN * analyze.FIG_DPI, abs=2)


# --- AC2: missing degraded data or strategy -> figures still produced, warning logged ------------------


def test_missing_degraded_and_strategies_warns_but_writes(handmade: Path, tmp_path: Path,
                                                           caplog: pytest.LogCaptureFixture) -> None:
    df = analyze.load_results(handmade)  # normal only, no random
    with caplog.at_level("WARNING", logger="eclm.analyze"):
        paths = analyze.make_figures(analyze.summarize(df), df, tmp_path / "figs")
    _figures_ok(tmp_path / "figs")
    assert len(paths) == 9
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "fig2_latency: no degraded-network results" in messages
    assert "no normal results for random" in messages


def test_only_degraded_cloud_results_still_writes_every_figure(tmp_path: Path,
                                                                caplog: pytest.LogCaptureFixture) -> None:
    _write(tmp_path / "results" / "cloud__none__degraded.csv",
           [_row(qid="a", strategy="cloud", edge_model="", network="degraded", ran_on="cloud", total_s=2.0)])
    with caplog.at_level("WARNING", logger="eclm.analyze"):
        assert analyze.main(["--results-dir", str(tmp_path / "results"), "--figures-dir", str(tmp_path / "f")]) == 0
    _figures_ok(tmp_path / "f")
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "fig1_accuracy: no normal-network results" in messages
    assert "fig4_memory: no edge-only memory samples" in messages
    assert "fig2_latency: no normal-network results" in messages
    assert pd.read_csv(tmp_path / "f" / analyze.FIG3_SIDE_FILE).empty


# --- AC3: fig3 side-file equals the summary -------------------------------------------------------------


def test_fig3_side_file_matches_summary(handmade: Path, tmp_path: Path) -> None:
    _write(handmade / "random__m1__normal.csv",
           [_row(qid="a", strategy="random", ran_on="cloud", correct=True), _row(qid="b", strategy="random")])
    _write(handmade / "edge__m1__degraded.csv", [_row(qid="a", network="degraded", correct=True)])
    df = analyze.load_results(handmade)
    summary = analyze.summarize(df)
    analyze.make_figures(summary, df, tmp_path / "figs")
    side = pd.read_csv(tmp_path / "figs" / analyze.FIG3_SIDE_FILE, keep_default_na=False)
    assert list(side.columns) == ["label", "strategy", "edge_model", "x", "y"]
    normal = summary[summary["network"] == "normal"].reset_index(drop=True)
    assert list(side["strategy"]) == list(normal["strategy"]) == ["edge", "cloud", "hybrid", "random"]
    assert list(side["edge_model"]) == list(normal["edge_model"])
    assert side["x"].tolist() == pytest.approx(normal["offload_ratio"].tolist())
    assert side["y"].tolist() == pytest.approx(normal["accuracy"].tolist())
    assert list(side["label"]) == ["edge m1", "cloud", "hybrid m1", "random m1"]
    assert side.loc[side["strategy"] == "edge", "x"].tolist() == [0.0]
    assert side.loc[side["strategy"] == "cloud", "x"].tolist() == [1.0]
    assert side.loc[side["strategy"] == "random", "x"].tolist() == [0.5]
