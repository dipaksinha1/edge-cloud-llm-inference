# Architecture

Spec: [SPEC.md](SPEC.md) · Decisions: [decisions/](decisions/) · Versions checked: 2026-11-03 (`pip index versions`).

## 1. Overview
A Python package `eclm` ("edge-cloud LLM") with four CLI entry points: `eclm.data`, `eclm.router`, `eclm.run`, `eclm.analyze`. It is layered. Each layer only calls the one below it, and every model is behind one `ModelClient` interface, so a `MockClient` can replace real backends in tests and in `--mock` mode.

```
 COMMANDS      data.py      router.py (train)      run.py      analyze.py
                  │               │                   │             │
 STRATEGIES       │          Router ◄──── EdgeOnly · CloudOnly · Hybrid · RandomRouter
                  │               │                   │
 MEASUREMENT      │        grading.py  ·  metrics.py (Timer inside clients, MemorySampler, system_info)
                  │               │                   │
 CLIENTS          │        OllamaClient · CloudClient · MockClient   (ModelClient protocol)
                  ▼               ▼                   ▼
 FILES     data/*.jsonl    models/*.joblib     results/*.csv + *.meta.json      figures/*, results/summary.*
 EXTERNAL  Hugging Face    Ollama localhost:11434 (Mac)      Groq API (internet, normal/degraded link)
```

## 2. Stack (versions current on 2026-11-03)
| Purpose | Library | Pin |
|---|---|---|
| Runtime | Python | ≥3.11 |
| Ollama HTTP + streaming | httpx | `>=0.28,<1` |
| Groq (OpenAI-compatible) | openai | `>=2,<4` (3.42.1 current) |
| Datasets | datasets | `>=3,<6` (5.1.0 current) |
| Router | scikit-learn, joblib, numpy | `scikit-learn>=1.5,<2`, `joblib>=1.4`, `numpy>=2` |
| Memory, system info | psutil | `>=6,<8` |
| Analysis | pandas, matplotlib | `pandas>=2.2,<4`, `matplotlib>=3.9,<4` |
| Config / secrets | pyyaml, python-dotenv | `pyyaml>=6`, `python-dotenv>=1` |
| Progress | tqdm | `>=4.66` |
| Dev | pytest, ruff, mypy | `pytest>=8`, `ruff>=0.6`, `mypy>=1.10` |

Packaging: `pyproject.toml` (setuptools, `src/` layout), extras `[dev]`. Install: `python -m pip install -e ".[dev]"`.

## 3. Directory layout
```
pyproject.toml   config.yaml   .env.example   README.md   CLAUDE.md
scripts/verify.sh
src/eclm/
  __init__.py
  config.py          Settings loader (config.yaml + .env + CLI overrides)
  cli.py             ArgumentParser whose usage errors exit 1, used by every CLI module (T014)
  types.py           Question, Generation, Result, Decision dataclasses (§5)
  prompts.py         build_prompt(question) -> str
  data.py            dataset builder CLI (R1)
  grading.py         extract_answer / is_correct / answers_agree (R8)
  metrics.py         MemorySampler, system_info (R7, R13)
  clients/__init__.py  make_client(kind, settings, mock) factory
  clients/base.py    ModelClient protocol, ClientError, ClientUnavailable
  clients/mock.py    MockClient (R12)
  clients/ollama.py  OllamaClient (R2)
  clients/cloud.py   CloudClient (R3)
  router.py          features, train CLI, Router (R4, R5)
  strategies.py      Strategy base + EdgeOnly, CloudOnly, Hybrid, RandomRouter (R2–R6)
  results.py         ResultWriter (CSV append/resume), result file naming (R7, R9, R10)
  run.py             run CLI (R2–R10)
  analyze.py         analysis CLI (R11)
  fixtures/mock_questions.jsonl   bundled questions for --mock (R12): 8 in T001 (4 GSM8K + 4 MMLU), expanded to 24 (12 + 12) in T003
  run.py / router.py / data.py / analyze.py each expose build_parser() -> argparse.ArgumentParser (an eclm.cli.ArgumentParser from T014)
data/  models/  results/  figures/   (generated; only data/ is committed)
docs/MAC_CHECKLIST.md               (T013)
tests/  test_*.py (+ tests/conftest.py)
```

## 4. Configuration — `config.yaml` (exact keys)
```yaml
seed: 42
data_dir: data
results_dir: results
models_dir: models
figures_dir: figures
dataset:
  gsm8k_test: 100
  mmlu_test: 100
  gsm8k_calibration: 150
  mmlu_calibration: 150
edge:
  host: http://localhost:11434
  models: [qwen2.5:3b, qwen2.5:0.5b, llama3.2:1b]
  hybrid_model: qwen2.5:3b
  timeout_s: 300
  memory_sample_hz: 10
  warmup_max_tokens: 1        # one unrecorded edge call before a run's first question; <= 0 = off (T015)
cloud:
  base_url: https://api.groq.com/openai/v1
  model: llama-3.3-70b-versatile
  api_key_env: GROQ_API_KEY
  timeout_s: 120
  max_retries: 5
  max_calls_per_run: 1500
generation:
  max_tokens: 512
  temperature: 0.0
router:
  threshold: 0.6
  samples: 3
  sample_temperature: 0.7
  min_agree: 2
```
`Settings` is a frozen dataclass tree mirroring these keys. `load_settings(path="config.yaml", overrides: dict | None = None) -> Settings`. `.env` is loaded with python-dotenv. The API key is read with `os.environ.get(settings.cloud.api_key_env)` only inside `CloudClient`.

## 5. Data model and contracts (exact)

```python
# types.py
@dataclass(frozen=True)
class Question:
    qid: str                      # "gsm8k-<split>-<row>" or "mmlu-<split>-<row>" (row = index in the source split); fixture qids use "mock-gsm8k-<n>" / "mock-mmlu-<n>"
    dataset: Literal["gsm8k", "mmlu"]
    question: str
    gold: str                     # GSM8K: normalised number string e.g. "72"; MMLU: "A".."D"
    choices: tuple[str, ...] = () # MMLU only, 4 items
    subject: str = ""             # MMLU only

@dataclass(frozen=True)
class Generation:                 # one model call
    text: str
    ttft_s: float                 # request sent -> first non-empty content chunk
    total_s: float                # request sent -> stream finished
    output_tokens: int
    tokens_per_s: float           # output_tokens / generation time (see §6)
    model: str
    ran_on: Literal["edge", "cloud"]
    peak_mem_gb: float | None = None   # edge only, set by strategies via MemorySampler

@dataclass(frozen=True)
class Decision:                   # router output
    route: Literal["edge", "cloud"]
    stage: int | None             # 1, 2, or None when kept on edge
    difficulty: float
    agreement: int | None         # number of agreeing edge samples (stage 2 only)
    edge_generations: tuple[Generation, ...]
    edge_answer: str | None       # majority answer when route == "edge"
    failed_edge_s: tuple[tuple[int, float], ...] = ()   # (T018) per failed stage-2 sample: (samples completed before it, seconds it took)

@dataclass(frozen=True)
class Result:                     # one CSV row (field order = CSV column order)
    run_id: str; strategy: str; edge_model: str; cloud_model: str; network: str
    dataset: str; qid: str; ran_on: str; router_stage: str; difficulty: str; agreement: str
    edge_calls: int; cloud_calls: int
    ttft_s: float; total_s: float; output_tokens: int; tokens_per_s: float; peak_mem_gb: str
    predicted: str; gold: str; correct: bool; parse_failed: bool; error: str; timestamp: str
```
Empty/not-applicable values are written as the empty string. `timestamp` is ISO-8601 UTC.

```python
# clients/base.py
class ClientError(Exception): ...          # a single call failed (recorded on the row, run continues)
class ClientUnavailable(Exception): ...    # backend unusable (stop before first question), message says how to fix
class BudgetExceeded(Exception): ...       # raised by BudgetedClient BEFORE a call once the run's cloud-call budget is used up
class BudgetedClient:                      # wraps the cloud client; counts calls; same ModelClient interface
    def __init__(self, inner: ModelClient, max_calls: int): ...
    calls: int
class ModelClient(Protocol):
    name: str                              # model name
    ran_on: Literal["edge", "cloud"]
    def check(self) -> None: ...           # raises ClientUnavailable
    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> Generation: ...

# clients/__init__.py
def make_client(kind: Literal["edge", "cloud"], settings: Settings, *, model: str | None = None, mock: bool = False) -> ModelClient

# clients/mock.py
class MockClient:  # needs the hidden gold marker from build_prompt(q, mock=True)
    def __init__(self, name: str, ran_on: str, accuracy: float, seed: int, latency_s: float = 0.0): ...
    # accuracy: mock_accuracy(name, ran_on) = cloud 0.9; edge names containing '0.5b' 0.45, '1b' 0.55, anything else 0.65
    # temperature == 0: answer depends only on (prompt, seed) -> identical on every call
    # temperature > 0: the k-th call for the same prompt uses seed + k, so stage-2 samples can disagree;
    #   wrong answers are drawn from 2 fixed distractors per question (gold+1, gold*2 / next letters) so wrong answers can also agree
def mock_accuracy(name: str, ran_on: str) -> float

# prompts.py
def build_prompt(q: Question, mock: bool = False) -> str   # mock=True appends a hidden '[[gold:<x>]]' marker read only by MockClient
WARMUP_QUESTION: Question   # (T015) fixed made-up GSM8K-style question, qid 'warmup', in no dataset or fixture
#  GSM8K: "...Solve the problem. Show brief reasoning, then end with a final line 'Answer: <number>'."
#  MMLU:  question + "A. ..\nB. ..\nC. ..\nD. .." + "End with a final line 'Answer: <letter>'."

# grading.py
def extract_answer(text: str, dataset: str, choices: tuple[str, ...] = ()) -> str | None  # GSM8K: number after last "Answer:" else last number; strip $ , %; "72.0" -> "72". MMLU: letter after last "Answer:" else last standalone A-D
def is_correct(predicted: str | None, gold: str, dataset: str) -> bool   # GSM8K numeric compare (abs diff < 1e-6)
def answers_agree(answers: list[str | None], dataset: str, min_agree: int) -> str | None  # majority answer if count >= min_agree else None; None never counts

# metrics.py
class MemorySampler:   # context manager; polls summed RSS of matching processes at hz; .peak_gb: float | None
    def __init__(self, hz: float = 10.0, process_match: str = "ollama", pids: list[int] | None = None): ...
    # pids given -> sample exactly those pids (mock mode passes [os.getpid()]); else match process names containing process_match
def system_info(settings: Settings) -> dict   # os, os_version, machine, cpu/chip, total_ram_gb, python, ollama_version|None

# data.py
def build_datasets(settings: Settings) -> tuple[list[Question], list[Question]]  # (test, calibration), seeded, disjoint
def load_questions(path: Path) -> list[Question]
def save_questions(qs: list[Question], path: Path) -> None   # JSONL, sorted keys, one per line
# files: data/test_questions.jsonl, data/calibration_questions.jsonl

# router.py
FEATURES = ["n_words", "n_numbers", "n_math_ops", "is_mmlu", "subject_stem", "subject_humanities", "subject_social", "subject_other", "max_number_digits"]
def features(q: Question) -> list[float]                     # order = FEATURES
SUBJECT_GROUPS: dict[str, str]   # all 57 MMLU subjects -> stem|humanities|social|other, per the official MMLU categories (hendrycks/test categories.py); unknown subject -> other
def train_router(settings, edge: ModelClient, calibration: list[Question], *, mock: bool = False) -> RouterModel  # mock: prompts carry the gold marker
class RouterModel:  # wraps sklearn LogisticRegression (class_weight="balanced"); save(path)/load(path) via joblib; predict_hard(q) -> float
class Router:
    def __init__(self, model: RouterModel, edge: ModelClient, settings: Settings, *, mock: bool = False): ...
    def decide(self, q: Question) -> Decision
# model file: models/router_<safe_name(edge_model)>.joblib

# strategies.py
class Strategy(ABC):
    name: str   # "edge" | "cloud" | "hybrid" | "random"
    @abstractmethod
    def prepare(self, questions: list[Question]) -> None:                # NOT abstract: default no-op (RandomRouter overrides)
        return None
    def meta(self) -> dict:              # (T017) NOT abstract, default {}: written to meta.json["strategy"] (if non-empty) when a run starts; {} if prepare() didn't run
    def restore(self, recorded: dict) -> None:   # (T017) NOT abstract, default no-op: called with meta.json["strategy"] before prepare() on a resume
    @abstractmethod
    def answer(self, q: Question) -> tuple[str, list[Generation], dict]   # (final text, all generations in call order, extra row fields)
class StrategyError(Exception):          # a ClientError inside answer(); keeps the calls that did complete
    def __init__(self, cause: Exception, generations: list[Generation], extra: dict): ...
@dataclass
class StrategyContext:
    settings: Settings; edge_model: str | None; network: str; mock: bool; results_dir: Path; ratio: float | None
    edge: ModelClient | None; cloud: ModelClient | None    # built by run.py (cloud wrapped in BudgetedClient)
STRATEGIES: dict[str, Callable[[StrategyContext], Strategy]]   # factories; each module-level strategy registers itself, run.py --all iterates the registered names in ARCH §10 order
EdgeOnly(edge, settings) · CloudOnly(cloud, settings) · Hybrid(edge, cloud, router, settings) · RandomRouter(edge, cloud, ratio, seed, settings)

# cli.py (T014)
class ArgumentParser(argparse.ArgumentParser):   # error(): print usage to stderr, then exit 1 with '<prog>: error: <msg>'

# run.py (T015)
def warm_up(edge: ModelClient, settings: Settings, mock: bool) -> None   # one call build_prompt(WARMUP_QUESTION, mock), temperature 0, max_tokens=edge.warmup_max_tokens; ClientError -> warning

# results.py
def safe_name(model: str | None) -> str   # None -> 'none'; ':' '/' '\\' and spaces -> '-'  (qwen2.5:3b -> qwen2.5-3b); used for result AND router file names
def result_path(results_dir, strategy, edge_model, network) -> Path   # results/<strategy>__<safe_name(edge_model)>__<network>.csv
def done_qids(path) -> set[str]
class ResultWriter:  # opens in append mode, writes header if new, flushes after every row
    def write(self, r: Result) -> None
```

### CLI contracts
```
python -m eclm.data   [--config config.yaml] [--mock]
python -m eclm.router train --edge-model M [--config ...] [--mock]
python -m eclm.run    --strategy {edge,cloud,hybrid,random} [--edge-model M] --network {normal,degraded}
                      [--ratio R] [--limit N] [--all] [--mock] [--config ...] [--results-dir DIR]
python -m eclm.analyze [--results-dir DIR] [--figures-dir DIR] [--config ...]     # no --mock: it only reads CSVs
```
Exit codes: 0 success, 1 user error (including argparse usage errors, via `eclm.cli.ArgumentParser`) or backend unavailable, 2 budget reached (resumable).
**Mock mode:** `python -m eclm.data --mock` writes the fixture's fixed disjoint split (`eclm.data.mock_questions() -> (test, calibration)`, half GSM8K/half MMLU each) to `data_dir` in the normal file format. `run --mock` and `router --mock` then read `data_dir` exactly like real runs (same loader), but build `MockClient`s named after the requested models (`mock_accuracy` above), and edge memory is sampled on the current process pid.

## 6. Measurement rules
- **TTFT / total:** `time.perf_counter()` immediately before sending the HTTP request. TTFT is taken at the first chunk with non-empty content, total when the stream ends.
- **Tokens/s:** Ollama uses its final chunk: `eval_count / (eval_duration / 1e9)`. Cloud uses `usage.completion_tokens` (request `stream_options={"include_usage": True}`) divided by (total − TTFT). If usage is missing, it falls back to counting chunks.
- **Strategy totals:** a row's `total_s` is the sum of all its calls' `total_s`. Its `ttft_s` is the sum of `total_s` of all calls before the answering call plus that call's `ttft_s`. `output_tokens` and `tokens_per_s` are the answering call's (for an edge-kept hybrid row: the first stage-2 sample whose answer equals the majority). A failed stage-2 sample's time (`Decision.failed_edge_s`) is added to `total_s`, and to `ttft_s` when it happened before the answering call; `edge_calls` still counts completed calls (T018).
- **Warm-up (T015):** before the first question of a run with an edge client (edge, hybrid, random), the runner makes one call `build_prompt(WARMUP_QUESTION, mock=mock)` at temperature 0 with `max_tokens = edge.warmup_max_tokens` (<= 0 = off). It writes no row, isn't memory-sampled and isn't counted anywhere (its duration is only logged), and a failure is only a warning. Cloud clients aren't warmed up, so a cloud run's first row includes connection setup (accepted).
- **Seeds:** every call passes a seed (Ollama `options.seed`, Groq `seed`): `settings.seed` for temperature-0 calls, `settings.seed + k` for the k-th stage-2 sample.
- **Memory:** `MemorySampler` wraps every edge call. The row's `peak_mem_gb` is the max over the question's edge calls. It's blank for cloud-only rows.
- **Offload ratio** (analysis): rows with `ran_on == "cloud"` ÷ rows.

## 7. Cross-cutting
- **Errors:** `ClientUnavailable` → message + exit 1 before any question. Inside a strategy, a `ClientError` is re-raised as `StrategyError(cause, completed_generations, extra)` (except stage-2 samples, which count as unparseable). The runner writes the row from the completed generations (timings summed per §6, `edge_calls`/`cloud_calls` = completed calls, `error="<type>: <msg>"`, `correct=False`) and continues. Cloud HTTP 429/5xx/connection errors → back-off sleeps 1, 2, 4, 8, 16 s (`max_retries`=5 retries, 6 attempts in total), then `ClientError`.
- **Budget:** `run.py` wraps the cloud client in `BudgetedClient(max_calls_per_run)`, which raises `BudgetExceeded` *before* a call that would exceed the budget. The runner then discards the unfinished question (no row), flushes and exits 2. A restart resumes with a fresh budget. A question never makes a partial over-budget call.
- **Logging:** `logging` at INFO to stderr, one line per run start/end plus warnings. tqdm progress bar to stderr. The API key is never logged.
- **Secrets:** `.env` (git-ignored), with `.env.example` committed.
- **Run metadata:** next to each CSV, `<same name>.meta.json` = `{run_id, started_at, args, config snapshot (no secrets), system_info}`, plus `strategy` = `Strategy.meta()` when it is non-empty: only random has one, `{ratio, ratio_source, cloud_questions, questions}` (T017). A resumed random run keeps its recorded ratio (an invalid one exits 1), so its cloud set stays the same as long as the question count is unchanged; a resume with another count (a different `--limit`) warns.
- **Determinism:** `random.Random(settings.seed)` for selection. The random router uses `Random(seed)` over sorted qids.

## 8. Testing & `scripts/verify.sh`
`verify.sh` (bash, `set -uo pipefail`) runs, in order, stopping at the first failure:
1. `ruff check src tests`
2. `mypy src` (config in pyproject: `ignore_missing_imports = true`)
3. `python -m pytest -q`
4. **Smoke (e2e, mock):** `rm -rf .verify_tmp && mkdir .verify_tmp`, copy `config.yaml` in, `cd` there, then run the steps that exist so far. The final form:
   ```
   python -m eclm.data --mock
   python -m eclm.router train --edge-model qwen2.5:3b --mock
   python -m eclm.run --all --network normal --mock
   python -m eclm.run --strategy edge --edge-model qwen2.5:3b --network degraded --mock
   python -m eclm.run --strategy random --edge-model qwen2.5:3b --ratio 0.25 --network degraded --mock
   python -m eclm.analyze
   ```
   Then assert row counts (= number of test questions in `data/test_questions.jsonl`) for `results/edge__qwen2.5-3b__normal.csv`, `edge__qwen2.5-0.5b__normal.csv`, `edge__llama3.2-1b__normal.csv`, `cloud__none__normal.csv`, `hybrid__qwen2.5-3b__normal.csv`, `random__qwen2.5-3b__normal.csv`, `edge__qwen2.5-3b__degraded.csv`, `random__qwen2.5-3b__degraded.csv`, plus `results/summary.csv` and the 8 figure files. Each task adds its own steps and assertions, and never removes or weakens earlier ones.

It prints `VERIFY: PASS` or `VERIFY: FAIL (<step>)` as its last line. Exit 0/1. Output is kept short (`-q`, tails of failing output only).

Test rules: HTTP clients are tested with `httpx.MockTransport` (Ollama) and an `httpx.Client(transport=MockTransport)` passed to `openai.OpenAI(http_client=...)` (Groq). The datasets download is tested by monkeypatching the loader to return small in-memory tables. No test touches the network.

## 9. Dev commands
```
python -m pip install -e ".[dev]"    # setup
scripts/verify.sh                    # full gate
python -m pytest tests/test_grading.py -q -k gsm8k   # single test
python -m eclm.run --strategy edge --edge-model qwen2.5:3b --network normal --limit 5   # on the Mac
```

## 10. How the experiment is run (on the Mac)
1. Install Ollama and pull the three models. Put `GROQ_API_KEY` in `.env`.
2. `python -m eclm.data` (once), then `python -m eclm.router train --edge-model qwen2.5:3b`.
3. Normal network: `python -m eclm.run --all --network normal`.
4. Degrade the link (+200 ms, 5 Mbit/s) with the OS tool from the README, then `python -m eclm.run --all --network degraded`.
5. `python -m eclm.analyze`, then commit `results/` and `figures/` with `git add -f`.

`--all` order: edge × each model, cloud, hybrid (hybrid_model), random (ratio read from that hybrid CSV for the same network).

## 11. Risks
| Risk | Mitigation |
|---|---|
| Groq rate limits or model rename | Back-off + resume. Model and base_url are in config. Verify the model id at console.groq.com before runs |
| Small models break the answer format | Tolerant extraction plus a parse-failure rate column |
| Memory sampler misses the model process on some OS | Match by process name, verified on the Mac in the T013 smoke checklist. `ollama ps` noted as cross-check |
| Router overfits or sends everything to the cloud | Balanced class weights, threshold in config, calibration accuracy printed at train time |
| Cloud sessions can't reach Ollama/Groq | Everything testable through mocks. Real-backend checks are a Mac checklist (T013) |
