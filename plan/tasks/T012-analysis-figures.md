# T012 · Analysis: report figures

## Goal
`python -m eclm.analyze` also writes the four report figures as PNG (300 dpi) and PDF.

## Context
SPEC R11(b) · ARCHITECTURE §5, §10.

## Read first
`src/eclm/analyze.py` (`summarize`).

## Contracts
`make_figures(summary_df, results_df, figures_dir) -> list[Path]`, matplotlib only (Agg backend), sized for an IEEE single column (3.5 in wide, readable 8 pt fonts), colour-blind-safe palette, no titles inside figures (captions go in LaTeX):
1. `fig1_accuracy` grouped bars per strategy/edge model, GSM8K vs MMLU, `normal` network.
2. `fig2_latency` mean total latency per strategy, normal vs degraded side by side (degraded bars omitted if there's no degraded data, with no crash).
3. `fig3_tradeoff` accuracy (y) vs offload ratio (x): edge-only at x=0, cloud at x=1, hybrid and random points labelled, `normal` network.
4. `fig4_memory` peak edge memory per edge model (edge-only rows).

## Acceptance criteria
1. (R11) From the mock smoke results, all 4 figures exist as `.png` and `.pdf` and are non-empty. Verify: verify.sh smoke asserts the files.
2. (R11) Missing degraded data or a missing strategy doesn't crash. The figure is produced with what exists and a warning is logged. Verify: test.
3. (R11) `make_figures` also writes `figures/fig3_tradeoff.csv` (the exact x/y/label plotted). Its values equal the summary's offload_ratio/accuracy. Verify: test reads the side-file.
4. `scripts/verify.sh` passes.

## Out of scope
Report text, LaTeX.

## Existing tests that may change
T011 smoke assertions may be extended.

## Notes
`matplotlib.use("Agg")` before importing pyplot.
