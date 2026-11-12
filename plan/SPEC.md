# Product spec: Edge vs Cloud vs Hybrid LLM inference experiment

Status: **approved** (written 2026-11-03 from the course proposal, the proposal report and decisions agreed with Dk; no separate interview). Owner: Dk (Dipak Kumar Sinha). Team: Shobi Navaranchan, Ajanthy Jayarajan, Dipak Kumar Sinha. Course: COSC 5756 Cloud Computing, Algoma University.

## Problem & users
LLM requests usually go to the cloud, which costs latency, money and privacy. Small models on the user's own device avoid this but are less accurate. A hybrid answers easy questions locally and sends hard ones to the cloud. The team needs a **command-line experiment tool** that measures all three strategies on the same questions, on a real laptop and a real network, and produces the numbers and graphs for a ≥6-page IEEE-format report (due 2026-12-10).

Users: the three team members, running Python on their laptops (official results on one machine: Dk's Apple Silicon Mac). They're comfortable with a terminal but aren't professional developers.

## Goals (v1)
1. Run the same fixed question set through edge-only, cloud-only, hybrid and random-router strategies.
2. Record latency, throughput, accuracy, edge memory and cloud usage for every question.
3. Repeat runs under a normal and a degraded network condition.
4. Produce a summary table and report-ready figures from the recorded data.

## Non-goals
Energy measurement · training or fine-tuning LLMs · a web or graphical UI · task division / mixture-style collaboration (only *task assignment*: each question goes whole to one model) · automating the network degradation itself (done with OS tools, outside the code) · running models on a phone or Raspberry Pi · multi-user or hosted deployment.

## Requirements

**R1 Question sets.** WHEN the user builds the dataset with a seed THE SYSTEM SHALL save a test set of 100 GSM8K + 100 MMLU questions and a separate calibration set (150 + 150) with no question in both sets. WHEN it's run twice with the same seed THE SYSTEM SHALL produce byte-identical files. WHEN the dataset source is unreachable THE SYSTEM SHALL exit non-zero with a clear message and leave existing files untouched.

**R2 Edge-only strategy.** WHEN the user runs the edge strategy with an edge model name THE SYSTEM SHALL answer every test question with that local model only and record one result row per question. WHEN the local model server isn't running or the model isn't pulled THE SYSTEM SHALL stop before the first question with a message naming the fix (e.g. `ollama pull <model>`).

**R3 Cloud-only strategy.** WHEN the user runs the cloud strategy THE SYSTEM SHALL answer every test question with the cloud model only. WHEN no API key is configured THE SYSTEM SHALL stop before the first question with a message saying where to put it. WHEN the provider rate-limits a request THE SYSTEM SHALL wait and retry (up to 5 retries) before recording an error for that question.

**R4 Hybrid strategy.** WHEN the user runs the hybrid strategy THE SYSTEM SHALL, per question: (stage 1) compute a difficulty score from cheap question features with a trained model and send the question straight to the cloud if the score exceeds the configured threshold; otherwise (stage 2) ask the edge model 3 times and keep the edge answer if at least 2 answers agree, else send the question to the cloud. THE SYSTEM SHALL record which stage decided, the score, the agreement count, and the number of edge and cloud calls. WHEN no trained router exists for the edge model THE SYSTEM SHALL stop with a message telling the user to train it.

**R5 Router training.** WHEN the user trains the router for an edge model THE SYSTEM SHALL have that edge model answer every calibration question, label each as hard (edge wrong) or easy, fit the difficulty model on question features, save it, and print its calibration accuracy. THE SYSTEM SHALL never use test questions for training.

**R6 Random-router baseline.** WHEN the user runs the random strategy THE SYSTEM SHALL send a randomly chosen (seeded) fraction of questions to the cloud and the rest to the edge (one call each), where the fraction equals the offload ratio of a completed hybrid run (or a value the user gives).

**R7 Per-question measurements.** For every answered question THE SYSTEM SHALL record: time to first token, total response time (sum over all calls the strategy made), output tokens, tokens per second, peak edge-model memory during the question (edge calls only), where the final answer came from (edge/cloud), the predicted and correct answers, correctness, and any error, in the CSV format defined in the architecture.

**R8 Grading.** WHEN a model answers a GSM8K question THE SYSTEM SHALL extract the final number and compare it numerically with the correct answer. WHEN it answers an MMLU question THE SYSTEM SHALL extract the letter A–D. WHEN no answer can be extracted THE SYSTEM SHALL mark it incorrect and flag it as a parse failure.

**R9 Network condition label.** WHEN the user starts a run THE SYSTEM SHALL require a network label (`normal` or `degraded`) and store it on every row and in the output file name, so the same strategy can be run under both conditions without overwriting.

**R10 Resumable, robust runs.** WHEN a run is interrupted and restarted with the same arguments THE SYSTEM SHALL skip questions already recorded and continue. WHEN a single question fails (timeout, HTTP error) THE SYSTEM SHALL record the error on that row and continue with the next question. WHEN `--all` is given THE SYSTEM SHALL run every configured strategy/model combination in sequence.

**R11 Analysis.** WHEN the user runs the analysis THE SYSTEM SHALL read all result files and write (a) a summary table per strategy × edge model × network (accuracy overall and per dataset, mean and median TTFT and total latency, mean tokens/s, mean peak memory, offload ratio, parse-failure rate) as CSV and Markdown, and (b) four figures as PNG and PDF: accuracy per strategy, latency normal vs degraded, accuracy vs offload ratio (hybrid vs random), and edge memory per model.

**R12 Mock mode.** WHEN any command that downloads data or calls a model (data, router, run) is given `--mock` THE SYSTEM SHALL run end to end with a built-in fake model and a small bundled question set, without a local model server, API key or network access, producing outputs in the same formats.

**R13 Run metadata.** WHEN a run starts THE SYSTEM SHALL save machine information (OS, CPU/chip, total RAM, Python version, local model server version if reachable) and the run's configuration next to the results.

## Non-functional requirements
- **N1 Reproducibility:** all random choices are seeded from config. Model calls use temperature 0 and pass a seed where the backend supports it (Ollama `options.seed`, OpenRouter `seed`). Stage-2 samples use temperature 0.7 with seed + sample index, so they can differ but are repeatable.
- **N2 Secrets:** the API key is read from a git-ignored `.env` file or the environment, and is never printed or written to results.
- **N3 Cost control:** a configurable maximum number of cloud calls per run (default 1,500). When it's reached, the run stops cleanly and can be resumed.
- **N4 Portability:** runs on macOS, Linux and Windows with Python 3.11+.
- **N5 Measurement overhead:** the memory sampler polls at ≥5 Hz and adds no more than 5% to edge total latency (checked once on the Mac).
- **N6 Usability:** one command per step, with a progress bar showing questions done out of total.

## Constraints (set by the team / course)
- Edge models are served by **Ollama**: `qwen2.5:0.5b`, `qwen2.5:3b`, `llama3.2:1b`. The hybrid's edge model defaults to `qwen2.5:3b`.
- Cloud model: **Llama 3.3 70B Instruct on OpenRouter** (OpenAI-compatible API).
- Datasets: GSM8K (`openai/gsm8k`, test split for test questions) and MMLU (`cais/mmlu`, test split for test questions). Calibration comes from other splits.
- Python. Stage-1 router is **logistic regression** (as stated in the proposal report).
- Network degradation is described in the report only as "+200 ms RTT, 5 Mbit/s cap", not tied to a tool.
- Tests never need Ollama or OpenRouter (mocks only); real runs happen on the Mac.
- Code complete before the team's cloud credit expires (2026-11-05 02:59 EST) where possible. Final report due 2026-12-10.

## Assumptions
- A1: The Mac has ≥8 GB RAM, enough for `qwen2.5:3b` (~2 GB quantized) alongside macOS.
- A2: The OpenRouter key has enough credit for ~2,000 calls (paid per token). Otherwise another OpenAI-compatible provider is set in config.
- A3: "Agree" in stage 2 means equal extracted answers (numbers equal after normalisation, or the same letter). Unparseable answers never agree.
- A4: Stage-1 threshold default is 0.6. It's tunable in config without retraining.
- A5: For hybrid, TTFT is the TTFT of the call that produced the final answer, plus the total time of all calls before it.
- A6: Peak memory is the resident memory of the Ollama processes (on Apple Silicon this includes the model, since memory is unified).

## Open questions
- Q1: Should the official hybrid run use a threshold tuned on calibration data (e.g. targeting about 40% offload), or the fixed default? Default for now: fixed 0.6, tuned later if results call for it.

## Glossary
**Edge**: the user's laptop. **Cloud**: a hosted model reached over the internet. **SLM**: small language model (0.5–3B parameters). **TTFT**: time to first token. **Offload ratio**: share of questions whose final answer came from the cloud. **Calibration set**: questions used only to train the router. **Stage 1 / Stage 2**: the router's difficulty check and its 3-sample agreement check. **Mock mode**: running with a fake model for testing.
