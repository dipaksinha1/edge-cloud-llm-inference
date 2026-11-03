"""`python -m eclm.analyze`: turn every result CSV into one summary table (SPEC R11a, ARCHITECTURE §5/§6).

Writes results/summary.csv and results/summary.md. Figures (R11b) come in T012."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from eclm.config import ConfigError, load_settings
from eclm.types import RESULT_COLUMNS

SUMMARY_CSV = "summary.csv"
SUMMARY_MD = "summary.md"
GROUP_KEYS = ("strategy", "edge_model", "network")
SUMMARY_COLUMNS: tuple[str, ...] = (
    *GROUP_KEYS,
    "n", "accuracy", "acc_gsm8k", "acc_mmlu",
    "ttft_mean", "ttft_median", "total_mean", "total_median",
    "tps_mean", "mem_peak_mean", "offload_ratio", "parse_fail_rate", "error_rate",
)  # stable order: the report tables are built from it
STRATEGY_ORDER = ("edge", "cloud", "hybrid", "random")  # ARCHITECTURE §10
NETWORK_ORDER = ("normal", "degraded")
_FLOATS = ("ttft_s", "total_s", "tokens_per_s", "peak_mem_gb")
_BOOLS = ("correct", "parse_failed")


def load_results(results_dir: Path) -> pd.DataFrame:
    """All rows of every results/*.csv except summary.csv, with numeric and boolean columns typed.

    Blank peak_mem_gb becomes NaN; blank edge_model stays '' (cloud-only). A CSV without the result
    columns raises ValueError."""
    paths = sorted(p for p in Path(results_dir).glob("*.csv") if p.name != SUMMARY_CSV)
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = [c for c in RESULT_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{path} is not a result file (missing columns: {', '.join(missing)})")
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(RESULT_COLUMNS))
    df = pd.concat(frames, ignore_index=True)
    for col in _FLOATS:
        df[col] = pd.to_numeric(df[col].replace("", math.nan), errors="coerce")
    for col in _BOOLS:
        df[col] = df[col].str.strip().str.lower() == "true"
    return df


def _mean(s: pd.Series) -> float:
    return float(s.mean()) if s.notna().any() else math.nan


def _median(s: pd.Series) -> float:
    return float(s.median()) if s.notna().any() else math.nan


def _summary_row(g: pd.DataFrame) -> dict[str, float]:
    errored = g["error"].astype(str).str.len() > 0
    ok = g[~errored]  # error rows count as wrong but are left out of timing and memory means
    acc = {d: _mean(g.loc[g["dataset"] == d, "correct"].astype(float)) for d in ("gsm8k", "mmlu")}
    return {
        "n": float(len(g)),
        "accuracy": _mean(g["correct"].astype(float)),
        "acc_gsm8k": acc["gsm8k"],
        "acc_mmlu": acc["mmlu"],
        "ttft_mean": _mean(ok["ttft_s"]),
        "ttft_median": _median(ok["ttft_s"]),
        "total_mean": _mean(ok["total_s"]),
        "total_median": _median(ok["total_s"]),
        "tps_mean": _mean(ok["tokens_per_s"]),
        "mem_peak_mean": _mean(ok["peak_mem_gb"]),
        "offload_ratio": _mean((g["ran_on"] == "cloud").astype(float)),
        "parse_fail_rate": _mean(g["parse_failed"].astype(float)),
        "error_rate": _mean(errored.astype(float)),
    }


def _rank(value: str, order: tuple[str, ...]) -> int:
    return order.index(value) if value in order else len(order)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (strategy, edge_model, network), columns in SUMMARY_COLUMNS order."""
    rows = []
    for key, g in df.groupby(list(GROUP_KEYS), sort=False, dropna=False):
        rows.append(dict(zip(GROUP_KEYS, key, strict=True)) | _summary_row(g))
    out = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    if out.empty:
        return out
    out["n"] = out["n"].astype(int)
    order = out.apply(
        lambda r: (_rank(r["strategy"], STRATEGY_ORDER), r["edge_model"], _rank(r["network"], NETWORK_ORDER)),
        axis=1,
    )
    return out.iloc[sorted(range(len(out)), key=lambda i: order.iloc[i])].reset_index(drop=True)


def _cell(value: object) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.3f}"
    return str(value)


def to_markdown(summary: pd.DataFrame) -> str:
    """A GitHub Markdown table, floats at 3 decimals, missing values blank."""
    cols = list(summary.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in summary.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_summary(summary: pd.DataFrame, results_dir: Path) -> tuple[Path, Path]:
    results_dir = Path(results_dir)
    csv_path, md_path = results_dir / SUMMARY_CSV, results_dir / SUMMARY_MD
    summary.to_csv(csv_path, index=False, float_format="%.6f")
    md_path.write_text(to_markdown(summary), encoding="utf-8")
    return csv_path, md_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m eclm.analyze", description="Summarise the result CSVs.")
    p.add_argument("--results-dir", help="default: results_dir from the config")
    p.add_argument("--figures-dir", help="default: figures_dir from the config (figures: T012)")
    p.add_argument("--config", default="config.yaml")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        try:
            results_dir = Path(load_settings(args.config).results_dir)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    try:
        df = load_results(results_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if df.empty:
        print(f"error: no results found in {results_dir}; run python -m eclm.run first", file=sys.stderr)
        return 1
    summary = summarize(df)
    csv_path, md_path = write_summary(summary, results_dir)
    print(f"summarised {len(df)} rows into {len(summary)} groups: {csv_path}, {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
