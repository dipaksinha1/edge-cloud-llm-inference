# 0007 · Cloud model: Llama 3.3 70B on OpenRouter (supersedes 0002)

**Context.** On 2026-11-12 the first real cloud smoke test on the Mac failed: Groq no longer lists `llama-3.3-70b-versatile`. The proposal fixes Llama 3.3 70B Instruct "through a hosted API", so the model stays and the provider changes. Dk chose OpenRouter and created a key with a $5 limit.

**Decision.** Use OpenRouter (`base_url=https://openrouter.ai/api/v1`, model `meta-llama/llama-3.3-70b-instruct`, key in `OPENROUTER_API_KEY`) through the official `openai` SDK with streaming, as before. Provider, model and key-variable name still live in `config.yaml`. Groq is removed from the code, tests, config and run guides.

**Alternatives.** Stay on Groq with a model it still serves: `openai/gpt-oss-120b` (hit the 512-token cap on 1 of 5 smoke questions) or `qwen/qwen3.8-27b` (a much smaller model, and four HTTP 429 retries in 5 questions). Both would change the proposal's cloud model.

**Consequences.** The cloud model is paid per token (listed at about $0.71 per million tokens on 2026-11-12). OpenRouter can route calls to different upstream providers (the first smoke test ranged from 15 to 115 tokens/s over 5 calls), so `cloud.providers` pins one: `coreweave/fp16`, which serves the model unquantized and supports `seed`. The report must name the pinned provider, in place of ADR 0002's "Groq is unusually fast" note. If that provider is down, calls fail instead of moving to another one (back-off, then an error row). Back-off and resumable runs stay as they are. The model id must be re-checked on openrouter.ai/models before the real runs.
