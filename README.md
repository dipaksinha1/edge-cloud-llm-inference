# edge-cloud-llm-inference

Edge vs cloud vs hybrid LLM inference experiment (COSC 5756 Cloud Computing, Algoma University).

## What this is

A command-line experiment that answers the same 200 questions (100 GSM8K maths + 100 MMLU multiple choice) in four ways and measures each one:

| Strategy | What it does |
|---|---|
| `edge` | A small model on your laptop (Ollama: `qwen2.5:3b`, `qwen2.5:0.5b`, `llama3.2:1b`) answers everything. |
| `cloud` | A large cloud model (Groq, `llama-3.3-70b-versatile`) answers everything. |
| `hybrid` | A two-stage router: easy-looking questions stay on the laptop, hard ones go to the cloud. |
| `random` | Sends the same share of questions to the cloud as `hybrid` did, but picked at random (the baseline). |

Every question gets one row in a CSV: latency (time to first token and total), tokens per second, edge memory, where the answer came from, and whether it was right. Each run is done twice, on a normal and on a degraded network (+200 ms round-trip time, 5 Mbit/s). `python -m eclm.analyze` then makes a summary table and four report figures.

There are four commands, run in this order:

| Command | Job |
|---|---|
| `python -m eclm.data` | Download and fix the question sets (test + calibration). |
| `python -m eclm.router train` | Train the hybrid router for an edge model. |
| `python -m eclm.run` | Run a strategy over the test questions. |
| `python -m eclm.analyze` | Write the summary table and figures. |

Every command has `--help`. Settings (models, question counts, router threshold, cloud budget) are in `config.yaml`.

## Requirements

- macOS, Linux or Windows, with Python 3.11 or newer (`python --version`). On some systems the command is `python3`.
- About 6 GB of free disk space for the three Ollama models.
- [Ollama](https://ollama.com/download) for the edge models.
- A free Groq API key from [console.groq.com/keys](https://console.groq.com/keys) for the cloud model.
- Git, and an internet connection for setup and the cloud runs.

You don't need Ollama or a Groq key for the [quick test](#quick-test-no-ollama-no-key).

## Setup

All commands are run from the repository folder.

1. Get the code:

   ```bash
   git clone https://github.com/dipaksinha1/edge-cloud-llm-inference.git
   cd edge-cloud-llm-inference
   ```

2. Make a virtual environment and turn it on.

   macOS / Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   Windows (PowerShell):

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   Your prompt now starts with `(.venv)`. Turn it on again in every new terminal.

3. Install the project:

   ```bash
   python -m pip install -e .
   ```

   Expected: the last line starts with `Successfully installed ... eclm-0.1.0 ...`. (Developers: `python -m pip install -e ".[dev]"` also installs pytest, ruff and mypy.)

4. Install Ollama from [ollama.com/download](https://ollama.com/download) and start it (open the Ollama app, or run `ollama serve` in a separate terminal). Then pull the three models:

   ```bash
   ollama pull qwen2.5:3b
   ollama pull qwen2.5:0.5b
   ollama pull llama3.2:1b
   ```

   Expected: each ends with `success`. Check with `ollama list`, which should show all three.

5. Put your Groq key in a file called `.env` (it is git-ignored and is never printed or written to results):

   ```bash
   cp .env.example .env
   ```

   Open `.env` in a text editor and fill in the line `GROQ_API_KEY=gsk_...`. On Windows use `copy .env.example .env`.

## Quick test (no Ollama, no key)

`--mock` replaces the models with a built-in fake and uses 24 bundled questions, so this needs no Ollama, no key and no network. It takes about a minute. Run it in a scratch folder so it can't overwrite your real data and results.

macOS / Linux:

```bash
mkdir -p mock_test
cp config.yaml mock_test/
cd mock_test
python -m eclm.data --mock
python -m eclm.router train --edge-model qwen2.5:3b --mock
python -m eclm.run --all --network normal --mock
python -m eclm.run --strategy edge --edge-model qwen2.5:3b --network degraded --mock
python -m eclm.analyze
cd ..
```

On Windows (PowerShell) the first three lines are `mkdir mock_test`, `copy config.yaml mock_test\` and `cd mock_test`. The rest is the same.

Expected output, step by step:

- `data --mock` → `wrote 12 test and 12 calibration questions to data`
- `router train` → four lines (calibration questions, edge accuracy, router training accuracy, share predicted hard), then `saved models/router_qwen2.5-3b.joblib`
- `run --all` → a progress bar and `INFO run ... finished` for each of 6 runs (3 edge models, cloud, hybrid, random)
- `run --strategy edge ... degraded` → one more run
- `analyze` → `summarised 84 rows into 7 groups: results/summary.csv, results/summary.md`, then `wrote 9 figure files to figures` (the many font lines like `INFO GDEF subsetted` or `INFO name pruned` are normal)

You now have `mock_test/results/*.csv`, `mock_test/results/summary.md` and `mock_test/figures/*.png`. The numbers are fake. Delete the folder when you're done (`rm -rf mock_test`).

## Smoke test on real backends

Before the full experiment, check that Ollama and Groq really work, with 5 questions each. Make sure Ollama is running and `.env` has your key.

First build the real question sets and train the router (this needs both; training asks `qwen2.5:3b` all 300 calibration questions, so allow 20–60 minutes):

```bash
python -m eclm.data
python -m eclm.router train --edge-model qwen2.5:3b
```

Expected: `wrote 200 test and 300 calibration questions to data`, then the router's accuracy lines and `saved models/router_qwen2.5-3b.joblib`.

Then the 5-question runs. `--results-dir smoke_results` keeps them out of the real `results/` folder (a run resumes whatever is already in its file, so smoke rows there would leak into the real results):

```bash
python -m eclm.run --strategy edge --edge-model qwen2.5:3b --network normal --limit 5 --results-dir smoke_results
python -m eclm.run --strategy cloud --network normal --limit 5 --results-dir smoke_results
python -m eclm.run --strategy hybrid --edge-model qwen2.5:3b --network normal --limit 5 --results-dir smoke_results
```

Expected: each prints `INFO run ... (5 of 5 questions to do)` and `INFO run ... finished`, and writes a CSV with 5 rows to `smoke_results/`. Open them: the `error` column should be empty, `ttft_s` and `total_s` should be filled, `peak_mem_gb` should be filled for edge rows, and `correct` should be `True` for most rows. The step-by-step checks are in [docs/MAC_CHECKLIST.md](docs/MAC_CHECKLIST.md).

## Full experiment

This is the exact sequence for the official results (ARCHITECTURE §10). The test set and router from the smoke test are reused, so skip steps 1–2 if you already ran them.

1. Build the question sets (once):

   ```bash
   python -m eclm.data
   ```

2. Train the router:

   ```bash
   python -m eclm.router train --edge-model qwen2.5:3b
   ```

3. Normal network. Turn off any network degradation, close other heavy apps, keep the laptop plugged in, then:

   ```bash
   python -m eclm.run --all --network normal
   ```

   This runs, in order: edge with each of the three models, cloud, hybrid, then random. Expect a few hours. Expected: 6 CSVs in `results/` with 200 rows each.

4. Degraded network. Turn on the degradation from [Degrading the network](#degrading-the-network), check it with `ping`, then:

   ```bash
   python -m eclm.run --all --network degraded
   ```

   Then **turn the degradation off** (see the undo step for your OS).

5. Analyse:

   ```bash
   python -m eclm.analyze
   ```

   Expected: `results/summary.csv`, `results/summary.md` and the figures in `figures/`.

6. Commit the results (see [Outputs](#outputs)).

If a run stops for any reason, run the same command again: it skips the questions already done (see [Troubleshooting](#troubleshooting)).

## Degrading the network

The degraded condition is **+200 ms round-trip time (RTT) and a 5 Mbit/s bandwidth cap** on the internet link. Any tool that gives this condition is fine; below is one way per OS. The code doesn't change the network itself: `--network degraded` only labels the rows and the file names, so you must switch the degradation on before and off after.

Two rules for every OS:

- **Don't slow down `localhost`.** Ollama talks to the experiment over `localhost:11434`. Only the internet link should be slower, otherwise the edge numbers are wrong.
- **Check it before the run.** Run `ping -c 5 8.8.8.8` (Windows: `ping -n 5 8.8.8.8`) with and without the degradation. The average time should be about 200 ms higher when it is on.

### macOS

**Option A: Network Link Conditioner (graphical).**

1. Download "Additional Tools for Xcode" from [developer.apple.com/download/all](https://developer.apple.com/download/all/) (free Apple ID). Open the `.dmg`, go to the `Hardware` folder and double-click `Network Link Conditioner.prefPane` to install it.
2. Open it from System Settings. Click "Manage Profiles", add a profile called `eclm-degraded` with: downlink bandwidth 5 Mbps, downlink delay 100 ms, uplink bandwidth 5 Mbps, uplink delay 100 ms, packet loss 0 %. (100 ms each way = +200 ms RTT.)
3. Pick `eclm-degraded` and switch it **ON**. Check with `ping`.

Network Link Conditioner may also slow `localhost`, which would slow Ollama too. Step 6 of [docs/MAC_CHECKLIST.md](docs/MAC_CHECKLIST.md) checks for this; if it happens, use Option B.

Undo: switch Network Link Conditioner **OFF**, and check that `ping` is back to normal.

**Option B: `dnctl` + `pfctl` (Terminal).** This shapes every interface except loopback (`lo0`), so Ollama is not slowed down:

```bash
sudo dnctl pipe 1 config bw 5Mbit/s delay 100
sudo dnctl pipe 2 config bw 5Mbit/s delay 100
printf 'no dummynet quick on lo0 all\ndummynet out all pipe 1\ndummynet in all pipe 2\n' | sudo pfctl -f -
sudo pfctl -E
```

Expected: `pfctl -E` prints `pf enabled` and a line `Token : <number>` (warnings about `ALTQ` or flushing rules are normal). Write down the token number. Check with `ping`.

Undo (restores Apple's default firewall rules, releases your hold on pf and removes the pipes; use the token you wrote down):

```bash
sudo pfctl -f /etc/pf.conf
TOKEN=1234567890        # replace with your token number
sudo pfctl -X "$TOKEN"
sudo dnctl -q flush
```

If you lost the token, `sudo pfctl -d` switches pf off completely instead (fine unless you use another firewall tool that needs pf).

### Linux

`tc netem` adds delay and a rate limit to packets **leaving** an interface (egress). Note on RTT: a delay on egress only is paid once per round trip, so 100 ms on egress adds only 100 ms to the RTT. The commands below therefore put 100 ms + 5 Mbit/s on egress and another 100 ms + 5 Mbit/s on incoming traffic (through the `ifb0` helper interface), which gives +200 ms RTT and caps both directions. It shapes only your internet interface, so `localhost` (Ollama) is not affected.

```bash
IFACE=$(ip route show default | awk '{print $5; exit}')
echo "$IFACE"
sudo tc qdisc add dev "$IFACE" root netem delay 100ms rate 5mbit
sudo modprobe ifb numifs=1
sudo ip link set dev ifb0 up
sudo tc qdisc add dev "$IFACE" handle ffff: ingress
sudo tc filter add dev "$IFACE" parent ffff: protocol all u32 match u32 0 0 action mirred egress redirect dev ifb0
sudo tc qdisc add dev ifb0 root netem delay 100ms rate 5mbit
```

`echo` should print your interface (e.g. `wlan0`, `eth0`, `enp3s0`). Check with `ping`. (If you can't use `ifb`, a single `sudo tc qdisc add dev "$IFACE" root netem delay 200ms rate 5mbit` also gives +200 ms RTT, but only caps upload.)

Undo (in the same terminal, so `$IFACE` is still set):

```bash
sudo tc qdisc del dev "$IFACE" root
sudo tc qdisc del dev "$IFACE" ingress
sudo tc qdisc del dev ifb0 root
```

### Windows

Use [clumsy](https://jagt.github.io/clumsy/) (free, run it as Administrator):

1. Filter: `(inbound or outbound) and not loopback` (this leaves Ollama on `localhost` alone).
2. Tick **Lag**, set the delay to `100` ms, and tick both **Inbound** and **Outbound** (100 ms each way = +200 ms RTT).
3. Tick **Bandwidth**, tick both **Inbound** and **Outbound**, and set the limit to `610` KB/s (5 Mbit/s ÷ 8 ÷ 1024).
4. Click **Start**. Check with `ping -n 5 8.8.8.8`.

Undo: click **Stop** and close clumsy. Check that `ping` is back to normal.

## Outputs

| Path | What | Committed? |
|---|---|---|
| `data/test_questions.jsonl`, `data/calibration_questions.jsonl` | the fixed question sets | yes (`git add data`) |
| `models/router_qwen2.5-3b.joblib` | the trained router | no |
| `results/<strategy>__<edge model>__<network>.csv` | one row per question, e.g. `results/edge__qwen2.5-3b__normal.csv`, `results/cloud__none__normal.csv` | with `git add -f` |
| `results/<same name>.meta.json` | machine info and settings for that run | with `git add -f` |
| `results/summary.csv`, `results/summary.md` | the summary table (one row per strategy × edge model × network) | with `git add -f` |
| `figures/fig1_accuracy` … `fig4_memory` (`.png` + `.pdf`), `figures/fig3_tradeoff.csv` | the report figures | with `git add -f` |

Model names in file names have `:` replaced by `-` so they work on Windows. `results/` and `figures/` are git-ignored so test runs don't get committed by accident. Commit the final ones on purpose:

```bash
git add data
git add -f results figures
git commit -m "Add experiment results"
git push
```

## Troubleshooting

| Problem | What you see | Fix |
|---|---|---|
| Ollama isn't running | ``error: Ollama is not running at http://localhost:11434. Start the Ollama app or run `ollama serve`.`` | Open the Ollama app, or run `ollama serve` in another terminal, then rerun. |
| Model not pulled | `error: Model qwen2.5:3b not found. Run: ollama pull qwen2.5:3b` | Run the `ollama pull` command it shows. |
| No Groq key | `error: Set GROQ_API_KEY in .env (get a key at console.groq.com/keys)` | Create `.env` as in [Setup](#setup) step 5, in the folder you run the commands from. |
| Wrong Groq key (HTTP 401) | `error: Groq rejected the API key` | Copy the key again from console.groq.com/keys into `.env` (no quotes, no spaces). |
| Rate limit (HTTP 429) | the run slows down: it waits 1, 2, 4, 8, 16 s and retries, 5 retries per call. If all fail, that question's row gets an `error` and the run goes on. | Usually nothing. If many rows have errors (see `error_rate` in `results/summary.md`), check your limits on console.groq.com, then delete that CSV and its `.meta.json` and rerun it later. |
| Cloud model renamed | `error: Model llama-3.3-70b-versatile is not available on Groq. Check cloud.model in config.yaml.` | Check the model list on console.groq.com and update `cloud.model`. |
| Run interrupted (Ctrl+C, crash, laptop asleep) | the CSV stops part-way | Run the **same command** again. It prints `(N of 200 questions to do)` and continues where it stopped. Use the same `--network` label. |
| Cloud budget reached, exit code 2 | `WARNING ... cloud-call budget of 1500 calls for this run is used up; stopping before ... Rerun the same command to resume.` The command exits with code 2 (`echo $?` shows `2`). | The run stopped cleanly after `cloud.max_calls_per_run` (1500) cloud calls in this command. Run the same command again: it resumes with a fresh budget. |
| Hybrid says no router | `error: no trained router for qwen2.5:3b at models/router_qwen2.5-3b.joblib; ...` | Run `python -m eclm.router train --edge-model qwen2.5:3b` first. |
| Random says no ratio | `error: random needs an offload ratio: pass --ratio R, or run --strategy hybrid ... first` | Run hybrid first for the same network (or give `--ratio`, a number between 0 and 1). |
| `test_questions.jsonl not found` | `error: data/test_questions.jsonl not found; run python -m eclm.data first` | Run `python -m eclm.data` from the repository folder. |
| `config file not found` | `error: config file not found: config.yaml` | Run the commands from the repository folder (where `config.yaml` is). |

Exit codes of `python -m eclm.run`: `0` = done, `1` = a problem to fix first (message above), `2` = cloud budget reached (rerun to resume). (A mistyped option to `eclm.data` or `eclm.analyze` also exits with `2`; that is just the option error, not the budget.)

## For developers

```bash
python -m pip install -e ".[dev]"
scripts/verify.sh
```

`scripts/verify.sh` runs ruff, mypy, pytest and the mock pipeline, and prints `VERIFY: PASS` at the end. Plan and design notes are in `plan/`.
