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
step "smoke: run edge"    "$PY" -m eclm.run --strategy edge --edge-model qwen2.5:3b --network normal --mock
N_TEST=$(wc -l < data/test_questions.jsonl)
check "smoke: test set not empty" test "$N_TEST" -gt 0
F=results/edge__qwen2.5-3b__normal.csv
check "smoke: $F rows" test -f "$F" -a "$(rows "$F" 2>/dev/null || echo 0)" -eq "$N_TEST"
check "smoke: ${F%.csv}.meta.json" test -s "${F%.csv}.meta.json"
step "smoke: run cloud"   "$PY" -m eclm.run --strategy cloud --network normal --mock
F=results/cloud__none__normal.csv
check "smoke: $F rows" test -f "$F" -a "$(rows "$F" 2>/dev/null || echo 0)" -eq "$N_TEST"
check "smoke: ${F%.csv}.meta.json" test -s "${F%.csv}.meta.json"

echo "VERIFY: PASS"
