# 0002 · Cloud model: Llama 3.3 70B on Groq via the OpenAI-compatible API

**Context.** The proposal fixes Llama 3.3 70B Instruct "through a hosted API". The team has no budget, and Dk confirmed Groq on 2026-11-03.

**Decision.** Use Groq (`base_url=https://api.groq.com/openai/v1`, model `llama-3.3-70b-versatile`) through the official `openai` SDK with streaming. Provider, model and key-variable name live in `config.yaml`.

**Alternatives.** Together AI or OpenRouter (paid or credit-based, same API shape, kept as config-only fallbacks). Self-hosting 70B (impossible on a laptop).

**Consequences.** Groq is unusually fast, so cloud latency is optimistic compared with typical providers. This must be stated as a limitation in the report. Free-tier rate limits need back-off and resumable runs. The model id must be re-checked on console.groq.com before the real runs.
