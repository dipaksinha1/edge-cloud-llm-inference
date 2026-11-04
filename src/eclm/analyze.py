"""`python -m eclm.analyze`: summary table and report figures from every result CSV (SPEC R11).

Writes results/summary.csv, results/summary.md and figures/fig1..fig4 as PNG (300 dpi) and PDF."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")  # before pyplot: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from eclm.config import ConfigError, load_settings  # noqa: E402
from eclm.types import RESULT_COLUMNS  # noqa: E402

log = logging.getLogger("eclm.analyze")

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


# --- figures (R11b) --------------------------------------------------------------------------------------

FIGURE_NAMES = ("fig1_accuracy", "fig2_latency", "fig3_tradeoff", "fig4_memory")
FIG3_SIDE_FILE = "fig3_tradeoff.csv"
FIG_WIDTH_IN = 3.5  # IEEE single column
FIG_HEIGHT_IN = 2.6
FIG_DPI = 300
FONT_PT = 8
# Okabe-Ito colour-blind-safe palette
OKABE_ITO = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
             "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000"}
STRATEGY_STYLE = {
    "edge": (OKABE_ITO["blue"], "o"),
    "cloud": (OKABE_ITO["vermillion"], "s"),
    "hybrid": (OKABE_ITO["green"], "D"),
    "random": (OKABE_ITO["orange"], "^"),
}
_RC: dict[str, Any] = {
    "font.size": FONT_PT, "axes.labelsize": FONT_PT, "xtick.labelsize": FONT_PT - 1,
    "ytick.labelsize": FONT_PT, "legend.fontsize": FONT_PT - 1, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
}


def _label(strategy: str, edge_model: str, sep: str = " ") -> str:
    """'edge' + 'qwen2.5:3b' -> 'edge qwen2.5:3b'; cloud-only (no edge model) -> 'cloud'."""
    return f"{strategy}{sep}{edge_model}" if edge_model else strategy


def _network(summary: pd.DataFrame, network: str, figure: str) -> pd.DataFrame:
    rows = summary[summary["network"] == network]
    missing = [s for s in STRATEGY_ORDER if s not in set(rows["strategy"])]
    if missing:
        log.warning("%s: no %s results for %s; plotting what exists", figure, network, ", ".join(missing))
    return rows


def _empty(ax: Any, figure: str, why: str) -> None:
    log.warning("%s: %s; writing an empty figure", figure, why)
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _save(fig: Any, figures_dir: Path, name: str) -> list[Path]:
    paths = [figures_dir / f"{name}.png", figures_dir / f"{name}.pdf"]
    fig.tight_layout(pad=0.3)
    fig.savefig(paths[0], dpi=FIG_DPI)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def _bars(ax: Any, labels: list[str], series: list[tuple[str, list[float], str]], ylabel: str) -> None:
    """Grouped bars: one group per label, one bar per (name, values, colour) series. NaN values are skipped."""
    width = 0.8 / len(series)
    for i, (name, values, colour) in enumerate(series):
        xs = [x + (i - (len(series) - 1) / 2) * width for x in range(len(labels))]
        pts = [(x, v) for x, v in zip(xs, values, strict=True) if not math.isnan(v)]
        ax.bar([x for x, _ in pts], [v for _, v in pts], width, label=name, color=colour, edgecolor="none")
    _xlabels(ax, labels)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        _legend_above(ax, len(series))


def _xlabels(ax: Any, labels: list[str]) -> None:
    """Category labels, slanted so long model names don't overlap at single-column width."""
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")


def _legend_above(ax: Any, ncol: int) -> None:
    ax.legend(frameon=False, ncol=ncol, loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)


def _fig1_accuracy(summary: pd.DataFrame, ax: Any) -> None:
    rows = _network(summary, "normal", "fig1_accuracy")
    if rows.empty:
        _empty(ax, "fig1_accuracy", "no normal-network results")
        return
    labels = [_label(r.strategy, r.edge_model) for r in rows.itertuples()]
    _bars(ax, labels, [("GSM8K", [float(v) for v in rows["acc_gsm8k"]], OKABE_ITO["blue"]),
                       ("MMLU", [float(v) for v in rows["acc_mmlu"]], OKABE_ITO["orange"])], "Accuracy")
    ax.set_ylim(0, 1.05)


def _fig2_latency(summary: pd.DataFrame, ax: Any) -> None:
    if summary.empty:
        _empty(ax, "fig2_latency", "no results")
        return
    networks = [n for n in NETWORK_ORDER if n in set(summary["network"])]
    for n in NETWORK_ORDER:
        if n not in networks:
            log.warning("fig2_latency: no %s-network results; plotting %s only", n, ", ".join(networks))
    keys = list(dict.fromkeys(zip(summary["strategy"], summary["edge_model"], strict=True)))
    colours = {"normal": OKABE_ITO["blue"], "degraded": OKABE_ITO["vermillion"]}
    series = []
    for n in networks:
        rows = summary[summary["network"] == n]
        by_key = {(r.strategy, r.edge_model): float(r.total_mean) for r in rows.itertuples()}
        series.append((n, [by_key.get(k, math.nan) for k in keys], colours[n]))
    _bars(ax, [_label(s, m) for s, m in keys], series, "Mean total latency (s)")


def fig3_points(summary: pd.DataFrame) -> pd.DataFrame:
    """The exact points of fig3 (normal network): label, strategy, edge_model, x=offload_ratio, y=accuracy."""
    rows = _network(summary, "normal", "fig3_tradeoff")
    return pd.DataFrame(
        {"label": [_label(r.strategy, r.edge_model) for r in rows.itertuples()],
         "strategy": list(rows["strategy"]), "edge_model": list(rows["edge_model"]),
         "x": [float(v) for v in rows["offload_ratio"]], "y": [float(v) for v in rows["accuracy"]]},
        columns=["label", "strategy", "edge_model", "x", "y"],
    )


def _fig3_tradeoff(points: pd.DataFrame, ax: Any) -> None:
    if points.empty:
        _empty(ax, "fig3_tradeoff", "no normal-network results")
        return
    for strategy in STRATEGY_ORDER:
        pts = points[points["strategy"] == strategy]
        if pts.empty:
            continue
        colour, marker = STRATEGY_STYLE[strategy]
        ax.scatter(pts["x"], pts["y"], color=colour, marker=marker, s=18, label=strategy, zorder=3)
        if strategy in ("hybrid", "random"):
            for p in pts.itertuples():
                ax.annotate(p.label, (p.x, p.y), xytext=(4, -8), textcoords="offset points",
                            fontsize=FONT_PT - 2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Offload ratio (share of questions answered by the cloud)")
    ax.set_ylabel("Accuracy")
    _legend_above(ax, len(set(points["strategy"])))


def _fig4_memory(results: pd.DataFrame, ax: Any) -> None:
    edge = results[(results["strategy"] == "edge") & (results["error"].astype(str) == "")]
    if "normal" in set(edge["network"]):
        edge = edge[edge["network"] == "normal"]
    edge = edge[edge["peak_mem_gb"].notna()]
    if edge.empty:
        _empty(ax, "fig4_memory", "no edge-only memory samples")
        return
    stats = edge.groupby("edge_model", sort=True)["peak_mem_gb"].agg(["mean", "max"])
    ax.bar(range(len(stats)), stats["mean"], 0.6, color=OKABE_ITO["blue"], edgecolor="none", label="mean")
    ax.scatter(range(len(stats)), stats["max"], color=OKABE_ITO["black"], marker="_", s=60, label="max", zorder=3)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(list(stats.index))
    ax.set_ylabel("Peak edge memory (GB)")
    _legend_above(ax, 2)


def make_figures(summary_df: pd.DataFrame, results_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Write the four report figures as PNG + PDF, and fig3's plotted points to fig3_tradeoff.csv.

    Missing strategies or networks log a warning and the figure shows what exists."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    points = fig3_points(summary_df)
    side = figures_dir / FIG3_SIDE_FILE
    points.to_csv(side, index=False, float_format="%.6f")
    with plt.rc_context(cast(Any, _RC)):
        for name in FIGURE_NAMES:
            fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
            if name == "fig1_accuracy":
                _fig1_accuracy(summary_df, ax)
            elif name == "fig2_latency":
                _fig2_latency(summary_df, ax)
            elif name == "fig3_tradeoff":
                _fig3_tradeoff(points, ax)
            else:
                _fig4_memory(results_df, ax)
            written += _save(fig, figures_dir, name)
    return [*written, side]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m eclm.analyze", description="Summarise the result CSVs.")
    p.add_argument("--results-dir", help="default: results_dir from the config")
    p.add_argument("--figures-dir", help="default: figures_dir from the config")
    p.add_argument("--config", default="config.yaml")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    args = build_parser().parse_args(argv)
    results_dir = Path(args.results_dir) if args.results_dir else None
    figures_dir = Path(args.figures_dir) if args.figures_dir else None
    if results_dir is None or figures_dir is None:
        try:
            settings = load_settings(args.config)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        results_dir = results_dir or Path(settings.results_dir)
        figures_dir = figures_dir or Path(settings.figures_dir)
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
    figures = make_figures(summary, df, figures_dir)
    print(f"wrote {len(figures)} figure files to {figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
