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
- [ ] Same seed → byte-identical files: run `shasum data/*.jsonl`, run `python -m eclm.data` again, run `shasum data/*.jsonl` again. Both checksum pairs are the same.

## 2. Router training on the real edge model

```bash
python -m eclm.router train --edge-model qwen2.5:3b
```

Expected (takes 20–60 minutes): **no output while it works** (there is no progress bar; only a `WARNING edge call failed ...` line if a call fails), then

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

Expected: `INFO run ... (5 of 5 questions to do)`, then `INFO warm-up qwen2.5:3b done in <s> s`, a progress bar to 5/5, then `INFO run ... finished`. Open `smoke_results/edge__qwen2.5-3b__normal.csv`:

- [ ] The warm-up line appears before the progress bar, with no `WARNING warm-up for ... failed`. (If the model wasn't loaded yet, `<s>` is the load time, often several seconds; that time is not in any row.)
- [ ] 5 rows, `error` empty, `ran_on` = `edge`, `edge_calls` = 1, `cloud_calls` = 0.
- [ ] Row 1's `ttft_s` is under 2× the median `ttft_s` of rows 2–5 (the warm-up took the model load, so row 1 isn't an outlier).
- [ ] `tokens_per_s` plausible for Apple Silicon: roughly 20–100 for qwen2.5:3b (not 0, not thousands).
- [ ] `ttft_s` is smaller than `total_s` on every row; `output_tokens` > 0.
- [ ] `peak_mem_gb` is filled on every row and close to the `ollama ps` SIZE (within about 30 %). If it is far below (for example under 0.5 GB), the sampler isn't seeing the model's memory (on Apple Silicon the weights may sit in GPU/unified memory that process RSS doesn't count): record this as a limitation.
- [ ] `predicted` looks like a number and `parse_failed` is `False` on most rows. (The test set lists GSM8K first, so `--limit 5` only covers GSM8K; MMLU is checked in step 7b.)
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
- [ ] Edge `ttft_median` for `edge qwen2.5:3b degraded` is about the same as for `normal` in `smoke_results/summary.md` (made by the `analyze` command below), less than about 0.1 s apart. If it is 0.1 s or more higher, the shaping is also slowing `localhost`: use the `dnctl`/`pfctl` option, which skips `lo0`, and repeat this step.

A quick way to compare averages and medians (needs no extra installs):

```bash
python -m eclm.analyze --results-dir smoke_results --figures-dir smoke_figures
```

and read the `ttft_mean`, `ttft_median` and `total_mean` columns in `smoke_results/summary.md`.

## 7. Memory sampler overhead (N5: under 5 %)

This times the same 10 edge calls with and without the memory sampler (built with `memory_sampler`, the same function real runs use). The order flips on every question (plain first, then sampled first), so Ollama's prompt cache and background noise favour neither side, and it also compares generation time only (`total_s - ttft_s`). It needs Ollama and the step-1 data, and takes a few minutes.

```bash
python - <<'EOF'
import statistics
from pathlib import Path
from eclm.clients import make_client
from eclm.config import load_settings
from eclm.data import load_questions
from eclm.prompts import build_prompt
from eclm.strategies import memory_sampler

s = load_settings()
edge = make_client("edge", s, model="qwen2.5:3b")
edge.check()
qs = load_questions(Path(s.data_dir) / "test_questions.jsonl")[:10]
gen = dict(temperature=s.generation.temperature, max_tokens=s.generation.max_tokens)
edge.generate(build_prompt(qs[0]), **gen)  # warm-up: loads the model
plain, sampled, peaks = [], [], []


def timed(prompt, sampler):
    if sampler is None:
        g = edge.generate(prompt, **gen)
    else:
        with sampler:
            g = edge.generate(prompt, **gen)
        peaks.append(sampler.peak_gb or 0.0)
    return g.total_s, g.total_s - g.ttft_s


for i, q in enumerate(qs):
    p = build_prompt(q)
    if i % 2 == 0:
        plain.append(timed(p, None))
        sampled.append(timed(p, memory_sampler(s, mock=False)))
    else:
        sampled.append(timed(p, memory_sampler(s, mock=False)))
        plain.append(timed(p, None))
for k, what in enumerate(["total", "generation only"]):
    a = statistics.mean(t[k] for t in plain)
    b = statistics.mean(t[k] for t in sampled)
    print(f"{what}: without sampler {a:.3f} s, with sampler {b:.3f} s, overhead {100 * (b - a) / a:+.1f} %")
print(f"peak memory {max(peaks):.2f} GB at {s.edge.memory_sample_hz} Hz")
EOF
```

Expected: three lines, for example `total: without sampler 3.412 s, with sampler 3.430 s, overhead +0.5 %`, a `generation only: ...` line, and `peak memory 2.4 GB at 10 Hz`.

- [ ] Both overheads are under +5 % (a small negative number is noise and also passes). If it is over 5 %, run it once more; if it stays over, try `memory_sample_hz: 5` in `config.yaml` (the minimum N5 allows) and record that.
- [ ] Peak memory is close to the `ollama ps` SIZE from step 3.

## 7b. MMLU answers from the real models

The 5-question smoke runs only cover GSM8K. This asks each edge model and the cloud model the first 3 MMLU test questions and prints the parsed letter:

```bash
python - <<'EOF'
from pathlib import Path
from eclm.clients import make_client
from eclm.config import load_settings
from eclm.data import load_questions
from eclm.grading import extract_answer
from eclm.prompts import build_prompt

s = load_settings()
qs = [q for q in load_questions(Path(s.data_dir) / "test_questions.jsonl") if q.dataset == "mmlu"][:3]
gen = dict(temperature=s.generation.temperature, max_tokens=s.generation.max_tokens)
for kind, model in [("edge", m) for m in s.edge.models] + [("cloud", None)]:
    client = make_client(kind, s, model=model)
    client.check()
    for q in qs:
        letter = extract_answer(client.generate(build_prompt(q), **gen).text, q.dataset, q.choices)
        print(f"{client.name:28} {q.qid:14} predicted {letter} gold {q.gold}")
EOF
```

Expected: 12 lines (3 edge models + cloud, 3 questions each).

- [ ] `predicted` is a letter A–D on nearly every line, not `None`. Several `None` for one model means its answer format isn't parsed: record an example in the PROGRESS entry.

## 8. Check the figures with real model names

The smoke runs above use only one edge model, so check the labels on the mock figures instead: the mock models carry the real names, so all three edge models are on the axes. Run the README quick test again (about a minute, it starts with `rm -rf mock_test`), then:

- [ ] Open `mock_test/figures/*.png`: the x-axis labels (`qwen2.5:3b`, `qwen2.5:0.5b`, `llama3.2:1b`, cloud, hybrid, random) don't overlap or get cut off.
- [ ] Delete the folder afterwards: `rm -rf mock_test`.

## 9. Clean up and record

- [ ] Delete the smoke outputs: `rm -rf smoke_results smoke_figures`. Keep `data/` and `models/`, and commit the question sets: `git add data`.
- [ ] Network degradation is OFF.
- [ ] Record the findings: add an entry to `plan/PROGRESS.md` with the heading `## YYYY-MM-DD · T016 · Run the Mac real-backend checklist · DONE` (or `BLOCKED`), listing the Mac model and chip, RAM, macOS, Ollama version, edge tokens/s, peak_mem_gb vs `ollama ps` SIZE, cloud TTFT normal vs degraded, the N5 overhead (total and generation only), router accuracy and share predicted hard, the step-7b MMLU result, and every box that didn't match. In `plan/tasks.json` set T016 to `done` (or keep it `blocked` with the problem in its notes). Commit and push on `build`.
- [ ] Anything that needs a code change goes into `plan/tasks.json` as a new task with status `proposed`.

When every box is ticked, run the full experiment (README, "Full experiment").
