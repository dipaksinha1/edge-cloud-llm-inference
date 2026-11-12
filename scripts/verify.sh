#!/usr/bin/env bash
# Full quality gate (ARCHITECTURE §8). Last line: VERIFY: PASS or VERIFY: FAIL (<step>).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
PY="${PYTHON:-python}"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

step() {  # step <name> <command...>: run quietly, show the tail of the output on failure
  local name="$1"; shift
  if "$@" >"$LOG" 2>&1; then
    echo "ok   $name"
  else
    echo "FAIL $name"; tail -n 30 "$LOG"
    echo "VERIFY: FAIL ($name)"; exit 1
  fi
}

check() {  # check <name> <condition...>
  local name="$1"; shift
  if "$@"; then echo "ok   $name"; else echo "FAIL $name"; echo "VERIFY: FAIL (smoke)"; exit 1; fi
}

rows() { echo $(( $(wc -l < "$1") - 1 )); }   # data rows in a CSV (minus header)
in_log() { grep -q "$1" "$LOG"; }        # the last step's output contains $1
not_in_log() { ! grep -q "$1" "$LOG"; }  # ... and doesn't contain $1
usage_error() {  # usage_error <command...>: exits 1 with argparse's "unrecognized arguments: --bogus" (not a crash)
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  [ "$rc" -eq 1 ] && grep -q "error: unrecognized arguments: --bogus" <<<"$out"
}

step ruff   "$PY" -m ruff check src tests
step mypy   "$PY" -m mypy src
step pytest "$PY" -m pytest -q

# --- smoke: real CLI, mock mode, fresh temp dir ---
rm -rf "$ROOT/.verify_tmp" && mkdir -p "$ROOT/.verify_tmp"
cp "$ROOT/config.yaml" "$ROOT/.verify_tmp/"
cd "$ROOT/.verify_tmp" || exit 1
step "smoke: data --mock" "$PY" -m eclm.data --mock
for D in data/test_questions.jsonl data/calibration_questions.jsonl; do
  check "smoke: $D has 12 lines" test -f "$D" -a "$(wc -l < "$D" 2>/dev/null || echo 0)" -eq 12
done
step "smoke: router train --mock" "$PY" -m eclm.router train --edge-model qwen2.5:3b --mock
check "smoke: models/router_qwen2.5-3b.joblib" test -s models/router_qwen2.5-3b.joblib
step "smoke: run --all normal" "$PY" -m eclm.run --all --network normal --mock
# edge models are warmed up before a run's first question, and no warm-up failed
check "smoke: run --all warmed up qwen2.5:3b" in_log "warm-up qwen2.5:3b done"
check "smoke: run --all had no failed warm-up" not_in_log "warm-up for"
step "smoke: run edge degraded" "$PY" -m eclm.run --strategy edge --edge-model qwen2.5:3b --network degraded --mock
step "smoke: run random degraded" "$PY" -m eclm.run --strategy random --edge-model qwen2.5:3b --ratio 0.25 \
  --network degraded --mock
N_TEST=$(wc -l < data/test_questions.jsonl)
check "smoke: test set not empty" test "$N_TEST" -gt 0
for F in results/edge__qwen2.5-3b__normal.csv results/edge__qwen2.5-0.5b__normal.csv \
         results/edge__llama3.2-1b__normal.csv results/cloud__none__normal.csv \
         results/hybrid__qwen2.5-3b__normal.csv results/random__qwen2.5-3b__normal.csv \
         results/edge__qwen2.5-3b__degraded.csv results/random__qwen2.5-3b__degraded.csv; do
  check "smoke: $F rows" test -f "$F" -a "$(rows "$F" 2>/dev/null || echo 0)" -eq "$N_TEST"
  check "smoke: ${F%.csv}.meta.json" test -s "${F%.csv}.meta.json"
done

# the random run records the ratio it used, so a resume keeps the same cloud set
# (parse the JSON: args.ratio is in every meta.json too, as null)
check "smoke: random meta records its ratio" "$PY" -c \
  'import json, sys; r = json.load(open(sys.argv[1]))["strategy"]["ratio"]; sys.exit(not isinstance(r, float))' \
  results/random__qwen2.5-3b__normal.meta.json

N_CSV=$(find results -maxdepth 1 -name '*.csv' ! -name summary.csv | wc -l)
step "smoke: analyze" "$PY" -m eclm.analyze
check "smoke: results/summary.csv has $N_CSV rows" test -f results/summary.csv -a "$(rows results/summary.csv 2>/dev/null || echo 0)" -eq "$N_CSV"
check "smoke: results/summary.md" test -s results/summary.md
for FIG in fig1_accuracy fig2_latency fig3_tradeoff fig4_memory; do
  for EXT in png pdf; do
    check "smoke: figures/$FIG.$EXT" test -s "figures/$FIG.$EXT"
  done
done
check "smoke: figures/fig3_tradeoff.csv" test -s figures/fig3_tradeoff.csv
# usage errors exit 1 in every CLI; 2 means only "cloud budget reached"
for M in data "router train --edge-model qwen2.5:3b" "run --strategy edge --network normal" analyze; do
  # shellcheck disable=SC2086  # $M is split on purpose into module + valid arguments
  check "smoke: $M --bogus exits 1" usage_error "$PY" -m eclm.$M --bogus
done

echo "VERIFY: PASS"
