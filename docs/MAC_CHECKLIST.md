# Mac real-backend checklist

Cloud sessions can't reach Ollama or Groq, so the real backends were only tested with fake HTTP transports. This checklist proves they work on the Mac that produces the official results. Do it once before the full experiment, and again if the code, Ollama or the models change.

Work through it in order. Tick each box, and write down anything that doesn't match the "Expected" line. All commands run from the repository folder with the virtual environment on (see the README, Setup).

## 0. Before you start

- [ ] `python --version` shows 3.11 or newer.
- [ ] `ollama --version` works, and `ollama list` shows `qwen2.5:3b`, `qwen2.5:0.5b` and `llama3.2:1b`.
- [ ] `.env` contains `GROQ_API_KEY=gsk_...`.
- [ ] No network degradation is on (Network Link Conditioner OFF, no `dnctl` pipes).
- [ ] The mock quick test in the README works (ends with `wrote 9 figure files to figures`).
- [ ] Laptop plugged in, other heavy apps closed.

## 1. Real question sets (Hugging Face download)

```bash
python -m eclm.data
```

Expected: `wrote 200 test and 300 calibration questions to data`. Then:

- [ ] `wc -l data/test_questions.jsonl data/calibration_questions.jsonl` shows 200 and 300.
- [ ] Running `python -m eclm.data` again and then `git status data/` shows no change (same seed → byte-identical files).

## 2. Router training on the real edge model

```bash
python -m eclm.router train --edge-model qwen2.5:3b
```

Expected (takes 20–60 minutes): a progress bar over 300 calibration questions, then

```
calibration questions: 300
edge accuracy on calibration (qwen2.5:3b): 0.xxx
router training accuracy: 0.xxx
share predicted hard (score > 0.6): 0.xxx
saved models/router_qwen2.5-3b.joblib
```

- [ ] Edge accuracy is plausible for a 3B model (roughly 0.5–0.8).
- [ ] "share predicted hard" is between 0 and 1, not exactly 0 or 1. Write it down (threshold tuning, open question Q1).

## 3. Real edge, 5 questions

```bash
python -m eclm.run --strategy edge --edge-model qwen2.5:3b --network normal --limit 5 --results-dir smoke_results
```

While it runs, in a second terminal run `ollama ps` and write down the `SIZE` shown for `qwen2.5:3b`.

Expected: `INFO run ... (5 of 5 questions to do)`, a progress bar to 5/5, then `INFO run ... finished`. Open `smoke_results/edge__qwen2.5-3b__normal.csv`:

- [ ] 5 rows, `error` empty, `ran_on` = `edge`, `edge_calls` = 1, `cloud_calls` = 0.
- [ ] `tokens_per_s` plausible for Apple Silicon: roughly 20–100 for qwen2.5:3b (not 0, not thousands).
- [ ] `ttft_s` is smaller than `total_s` on every row; `output_tokens` > 0.
- [ ] `peak_mem_gb` is filled on every row and close to the `ollama ps` SIZE (within about 30 %). If it is far below (for example under 0.5 GB), the sampler isn't seeing the model's memory (on Apple Silicon the weights may sit in GPU/unified memory that process RSS doesn't count): record this as a limitation.
- [ ] `predicted` looks like a number (GSM8K) or a letter A–D (MMLU), and `parse_failed` is `False` on most rows.
- [ ] `smoke_results/edge__qwen2.5-3b__normal.meta.json` has `system_info` with your chip, RAM, macOS version and an Ollama version (not null).

## 4. Real cloud, 5 questions

```bash
python -m eclm.run --strategy cloud --network normal --limit 5 --results-dir smoke_results
```

Expected: `INFO run ... finished`. Open `smoke_results/cloud__none__normal.csv`:

- [ ] 5 rows, `error` empty, `ran_on` = `cloud`, `cloud_calls` = 1, `peak_mem_gb` empty.
- [ ] `correct` is `True` on most rows (the 70B model should get 4–5 of 5).
- [ ] `ttft_s` is usually well under 2 s on a normal network; `tokens_per_s` is high (Groq is often in the hundreds).
- [ ] The API key does not appear anywhere: `grep -r gsk_ smoke_results` prints nothing.

## 5. Real hybrid, 5 questions

```bash
python -m eclm.run --strategy hybrid --edge-model qwen2.5:3b --network normal --limit 5 --results-dir smoke_results
```

Expected: `INFO run ... finished`. Open `smoke_results/hybrid__qwen2.5-3b__normal.csv`:

- [ ] 5 rows, `error` empty, `difficulty` (the router score) filled on every row.
- [ ] Rows with `ran_on` = `cloud` and `router_stage` = `1` (sent straight to the cloud) have `edge_calls` = 0, `cloud_calls` = 1 and an empty `agreement`.
- [ ] Rows with `ran_on` = `edge` (blank `router_stage`) have `edge_calls` = 3, `cloud_calls` = 0 and `agreement` of `2/3` or `3/3`.
- [ ] Rows with `ran_on` = `cloud` and `router_stage` = `2` (the edge samples disagreed) have `edge_calls` = 3, `cloud_calls` = 1 and `agreement` of `1/3` or `0/3`.
- [ ] Edge-kept rows have `peak_mem_gb` filled.

## 6. Degraded network raises cloud latency

Turn on the degradation (README, "Degrading the network", macOS). Check `ping -c 5 8.8.8.8` is about 200 ms slower than without it. Then:

```bash
python -m eclm.run --strategy cloud --network degraded --limit 5 --results-dir smoke_results
python -m eclm.run --strategy edge --edge-model qwen2.5:3b --network degraded --limit 5 --results-dir smoke_results
```

Turn the degradation **off** again and check `ping` is back to normal. Then compare with the normal runs from steps 3 and 4:

- [ ] Cloud `ttft_s` and `total_s` (in `smoke_results/cloud__none__degraded.csv`) are clearly higher than in `cloud__none__normal.csv` (at least about 0.2 s more on average).
- [ ] Edge `total_s` (in `edge__qwen2.5-3b__degraded.csv`) is about the same as in `edge__qwen2.5-3b__normal.csv`. If it is much higher, the shaping is also slowing `localhost`: use the `dnctl`/`pfctl` option, which skips `lo0`.

A quick way to compare averages (needs no extra installs):

```bash
python -m eclm.analyze --results-dir smoke_results --figures-dir smoke_figures
```

and read the `ttft_mean` and `total_mean` columns in `smoke_results/summary.md`.

## 7. Memory sampler overhead (N5: under 5 %)

This times the same 10 edge calls with and without the memory sampler, alternating so that warm-up and background noise affect both equally. It needs Ollama and the step-1 data, and takes a few minutes.

```bash
python - <<'EOF'
import statistics
from pathlib import Path
from eclm.clients import make_client
from eclm.config import load_settings
from eclm.data import load_questions
from eclm.metrics import MemorySampler
from eclm.prompts import build_prompt

s = load_settings()
edge = make_client("edge", s, model="qwen2.5:3b")
edge.check()
qs = load_questions(Path(s.data_dir) / "test_questions.jsonl")[:10]
gen = dict(temperature=s.generation.temperature, max_tokens=s.generation.max_tokens)
edge.generate(build_prompt(qs[0]), **gen)  # warm-up: loads the model
plain, sampled, peaks = [], [], []
for q in qs:
    p = build_prompt(q)
    plain.append(edge.generate(p, **gen).total_s)
    with MemorySampler(hz=s.edge.memory_sample_hz) as m:
        sampled.append(edge.generate(p, **gen).total_s)
    peaks.append(m.peak_gb or 0.0)
a, b = statistics.mean(plain), statistics.mean(sampled)
print(f"without sampler {a:.3f} s, with sampler {b:.3f} s, overhead {100 * (b - a) / a:+.1f} %")
print(f"peak memory {max(peaks):.2f} GB at {s.edge.memory_sample_hz} Hz")
EOF
```

Expected: two lines, for example `without sampler 3.412 s, with sampler 3.430 s, overhead +0.5 %` and `peak memory 2.4 GB at 10 Hz`.

- [ ] Overhead is under +5 % (a small negative number is noise and also passes). If it is over 5 %, run it once more; if it stays over, try `memory_sample_hz: 5` in `config.yaml` (the minimum N5 allows) and record that.
- [ ] Peak memory is close to the `ollama ps` SIZE from step 3.

## 8. Check the figures with real model names

```bash
python -m eclm.analyze --results-dir smoke_results --figures-dir smoke_figures
```

- [ ] Open `smoke_figures/*.png`: the x-axis labels (model names) don't overlap or get cut off.

## 9. Clean up and record

- [ ] Delete the smoke outputs: `rm -rf smoke_results smoke_figures`. Keep `data/` and `models/`.
- [ ] Network degradation is OFF.
- [ ] Record the findings: add an entry to `plan/PROGRESS.md` with the heading `## YYYY-MM-DD · MAC · Real-backend checklist · DONE` (or `BLOCKED`), listing the Mac model, macOS, Ollama version, tokens/s, peak_mem_gb vs `ollama ps`, cloud latency normal vs degraded, the N5 overhead, router accuracy and share predicted hard, and every box that didn't match. Commit and push it on `build`.
- [ ] Anything that needs a code change goes into `plan/tasks.json` as a new task with status `proposed`.

When every box is ticked, run the full experiment (README, "Full experiment").
