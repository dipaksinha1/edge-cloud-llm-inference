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
step "smoke: run edge degraded" "$PY" -m eclm.run --strategy edge --edge-model qwen2.5:3b --network degraded --mock
N_TEST=$(wc -l < data/test_questions.jsonl)
check "smoke: test set not empty" test "$N_TEST" -gt 0
for F in results/edge__qwen2.5-3b__normal.csv results/edge__qwen2.5-0.5b__normal.csv \
         results/edge__llama3.2-1b__normal.csv results/cloud__none__normal.csv \
         results/edge__qwen2.5-3b__degraded.csv; do
  check "smoke: $F rows" test -f "$F" -a "$(rows "$F" 2>/dev/null || echo 0)" -eq "$N_TEST"
  check "smoke: ${F%.csv}.meta.json" test -s "${F%.csv}.meta.json"
done

echo "VERIFY: PASS"
