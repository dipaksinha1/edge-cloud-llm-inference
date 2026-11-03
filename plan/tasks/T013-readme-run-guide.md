# T013 · README run guide and Mac real-backend checklist

## Goal
A teammate can go from a fresh laptop to finished results using only the README. The checklist proves the real Ollama/Groq paths work on the Mac (things cloud sessions can't test).

## Context
SPEC R2, R3, R9, N4–N6 · ARCHITECTURE §9, §10 · ADR 0005.

## Read first
`CLAUDE.md`, all CLI `--help` outputs.

## Contracts
`README.md` sections: What this is · Requirements · Setup (Python venv, `pip install -e .`, Ollama install + `ollama pull` for each model, Groq key into `.env`) · Quick test (`--mock` pipeline) · Smoke test on real backends (`--limit 5` for edge, cloud, hybrid) · Full experiment (exact command sequence from ARCHITECTURE §10) · Degrading the network (+200 ms RTT, 5 Mbit/s): macOS (Network Link Conditioner via Xcode Additional Tools, or `dnctl`/`pfctl` commands, and how to undo), Linux (`tc qdisc … netem delay 100ms rate 5mbit` on egress, with the RTT note, and undo), Windows (clumsy: lag + bandwidth settings) · Outputs (where CSVs and figures land, `git add -f` to commit them) · Troubleshooting (Ollama not running, model not pulled, 401/429 from Groq, resume after interruption, budget exit code 2).
`docs/MAC_CHECKLIST.md`: a numbered checklist with expected outputs: real edge `--limit 5` (tokens/s plausible, peak_mem_gb filled and close to `ollama ps`), real cloud `--limit 5`, real hybrid `--limit 5`, degraded run shows higher cloud latency, N5 overhead check (edge latency with vs without sampler, within 5%), and where to record findings (PROGRESS entry).

## Acceptance criteria
1. Every command in the README's mock section runs successfully as written. Verify: copy-paste them in a fresh shell.
2. Every `python -m eclm…` command in the README parses. Verify: `tests/test_readme_commands.py` extracts them from fenced code blocks and calls the module's `build_parser().parse_args(argv)` (every CLI module exposes `build_parser()` per ARCH §3), with no execution.
3. (R9) The network section describes the condition generically (+200 ms RTT, 5 Mbit/s) and gives an undo step for each OS. Verify: read.
4. `scripts/verify.sh` passes.

## Out of scope
Actually running on the Mac (Dk does that with the checklist).

## Existing tests that may change
None.

## Notes
Write for students who aren't professional developers: short steps, expected output after each.
