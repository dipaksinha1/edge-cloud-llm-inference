# 0003 · Edge runtime: Ollama over its local HTTP API

**Context.** The proposal uses quantized models via Ollama, following Husom et al. Ollama is already installed on Dk's Mac.

**Decision.** Call `POST /api/generate` (stream=true) with `httpx`. Read `eval_count` and `eval_duration` from the final chunk for throughput. Models: qwen2.5:3b, qwen2.5:0.5b, llama3.2:1b, with the hybrid using qwen2.5:3b.

**Alternatives.** The `ollama` Python package (thin wrapper, hides raw timing). llama.cpp or MLX directly (more setup, diverges from the cited method).

**Consequences.** Memory is measured as the RSS of the Ollama processes (the model runs in a separate process). On Apple Silicon this includes the model, because memory is unified.
