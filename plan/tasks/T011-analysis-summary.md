# T011 · Analysis: summary table

## Goal
`python -m eclm.analyze` turns all result CSVs into one summary table for the report.

## Context
SPEC R11(a) · ARCHITECTURE §5 (Result columns), §6 (offload ratio).

## Read first
`src/eclm/results.py`, a sample CSV produced by the mock smoke.

## Contracts
`analyze.py`: `load_results(results_dir) -> pandas.DataFrame` (all `*.csv` except `summary.csv`). `summarize(df) -> DataFrame` grouped by `strategy, edge_model, network` with columns `n, accuracy, acc_gsm8k, acc_mmlu, ttft_mean, ttft_median, total_mean, total_median, tps_mean, mem_peak_mean, offload_ratio, parse_fail_rate, error_rate` (error rows count as incorrect and are excluded from timing means). Writes `results/summary.csv` and `results/summary.md` (Markdown table, 3 decimals). CLI flags `--results-dir`, `--figures-dir` (figures in T012).

## Acceptance criteria
1. (R11) For a handcrafted DataFrame, every summary column matches hand-computed values. Verify: test.
2. (R11) Error rows are excluded from timing means but included in accuracy (as wrong) and `error_rate`. Verify: test.
3. (R11) With an empty results dir → exit 1, "no results found". Verify: test.
4. The verify.sh smoke runs analyze and asserts `results/summary.csv` has one row per CSV produced. Verify: verify.sh.
5. `scripts/verify.sh` passes.

## Out of scope
Figures (T012). Statistical tests.

## Existing tests that may change
None.

## Notes
Keep the column order stable: the report tables are built from it.
