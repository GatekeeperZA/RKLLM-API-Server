# Future Features

Improvements identified through community research (August 2026).
Items are ordered by estimated ROI. Items marked **Done** are already implemented.

---

## Implemented

| Feature | File | Notes |
|---|---|---|
| OpenAI `/v1/chat/completions` | `api.py` | Streaming + non-streaming |
| OpenAI `/v1/models` | `api.py` | Lists all configured models |
| Streaming SSE + reasoning_content | `api.py` | ThinkTagParser splits `<think>` blocks |
| VL multimodal (`/v1/chat/completions`) | `api.py` | RKNN vision encoder + RKLLM decoder |
| Tool calling (Qwen3 native) | `api.py` | `<tool_call>` format injected as system prompt |
| KV cache + incremental history | `api.py` | Skips re-prefill when history unchanged |
| Prompt cache (`.bin` file) | `api.py` | Saves/loads KV state across restarts |
| Document RAG pipeline | `api.py` | Paragraph scoring, dedup, follow-up detection |
| Open WebUI shortcircuits | `api.py` | Query gen, title gen, tag gen — instant, no inference |
| Home Assistant special handling | `api.py` | Thinking disabled, fast path for HA prompts |
| SearXNG web search support | `api.py` | Context injection from search results |
| Prometheus `/metrics` | `api.py` | Flask + RKLLM inference timings |
| Model-aware sampling profiles | `api.py` | Per-model temperature/top_k/top_p |
| Repetition loop detection | `api.py` | Aborts runaway generation |
| Sliding window context | `api.py` | Keeps system prompt + recent N turns |
| Stop sequences | `api.py` | Sliding buffer match, content trimmed at hit point |
| Context overflow protection | `api.py` | 400 error with clear message |
| `/v1/completions` legacy shim | `api.py` | Maps to chat_completions |
| `/health` endpoint | `api.py` | Model status, loaded model name |
| Idle auto-unload | `api.py` | Unloads model after N minutes idle |
| CPU affinity (big cores) | `api.py` | Pins to cores 4-7 on RK3588 |
| **`/v1/embeddings`** | `embed_api.py` | Qwen3-Embedding-0.6B, L2-normalised, batched |
| **`/v1/rerank`** | `embed_api.py` | Qwen3-Reranker-0.6B, Cohere-compatible |

---

## Pending — High Priority

### API Key Authentication
**Effort:** Low (1–2 hours)
**Impact:** Allows safe exposure beyond LAN without a full reverse-proxy auth layer.

Add `RKLLM_API_KEY` env var. Before any route handler runs, check
`Authorization: Bearer <key>`. Return 401 if missing or wrong.
Apply to both `api.py` and `embed_api.py`.

---

### NPU Utilisation in `/metrics`
**Effort:** Low (1–2 hours)
**Impact:** Shows actual NPU core load % in Grafana. Useful for spotting thermal throttling.

The `rknpu` driver exposes load via sysfs:
```
/sys/kernel/debug/rknpu/load
```
Format on RK3588: `NPU load:  Core0: 74%, Core1: 0%, Core2: 0%`

Add a Prometheus gauge `rkllm_npu_load_percent{core="0"}` etc., updated on
each Prometheus scrape via a `@app.before_request` or a background thread.
Requires the process to have read access to the sysfs path (usually readable
by root; check if `armbian` user can read it without sudo).

---

### Ollama API Compatibility Shim
**Effort:** Medium (4–6 hours)
**Impact:** Unlocks Continue (VS Code), AnythingLLM, Msty, Jan — any tool
with native Ollama mode — without those tools needing OpenAI base-URL config.

Minimum viable routes to add to `api.py`:
- `GET  /api/tags`     → map from `/v1/models` response
- `GET  /api/ps`       → currently-loaded model + memory info
- `POST /api/chat`     → translate Ollama message format → existing `chat_completions`
- `POST /api/generate` → translate Ollama completion format → `chat_completions`
- `GET  /api/version`  → `{"version": "0.3.0"}` (static string, clients just check existence)

The Ollama chat format is close to OpenAI but uses `"message"` instead of
`"messages"` and `"done"` instead of `"finish_reason"`. The streaming format
uses newline-delimited JSON instead of SSE `data:` lines.

Do NOT implement `POST /api/pull` — downloading models from Ollama's registry
won't work for RKLLM format.

Reference: [NotPunchnox/rkllama](https://github.com/NotPunchnox/rkllama)

---

### LoRA Adapter Support
**Effort:** Low (2–3 hours, structs already defined)
**Impact:** Allows domain-specific fine-tuned adapters without storing full model copies.

The ctypes structs `RKLLMLoraAdapter` and `RKLLMLoraParam` are already defined
in `api.py` (lines 890–944). What's missing:

1. Add optional `lora_path` and `lora_scale` fields to model config in `models.yaml`.
2. In `init_model()`, call `rkllm_load_lora()` after `rkllm_init()` if configured.
3. Pass the adapter name via `RKLLMLoraParam` in `RKLLMInferParam` at inference time.

Constraint: LoRA adapter must match the base model architecture. Loading adds
50–200 MB RAM depending on rank. Hot-swap of adapters per-request is possible
but expensive (requires model reload) — not recommended.

Official demo: [airockchip/rknn-llm server demo](https://github.com/airockchip/rknn-llm)

---

## Pending — Medium Priority

### Audio Transcription (`/v1/audio/transcriptions`)
**Effort:** High (2–3 days, separate service)
**Impact:** Voice input in Open WebUI without sending audio to Google/cloud.

Run as a separate service on port 8002. Options:
- Whisper INT8 on NPU via RKNN: [jianglu/whisper_RK3588](https://github.com/jianglu/whisper_RK3588)
- OmniASR-CTC via RKNN: [suharvest/rkvoice-stream](https://github.com/suharvest/rkvoice-stream)
  — 145ms latency, 52 languages, WebSocket streaming

Must be a separate process — audio decoding is CPU-heavy and would stall
the main inference queue if in-process.

---

### Text-to-Speech (`/v1/audio/speech`)
**Effort:** High (2–3 days, separate service)
**Impact:** Responses read aloud offline; voice assistants on mobile.

Piper TTS with RKNN decoder is 4.3× faster than CPU:
[marty1885/paroli](https://github.com/marty1885/paroli)

Run as a separate service on port 8003. Coordinate startup with the
voice-to-voice pipeline if both audio features are added together.

---

### Voice-to-Voice WebSocket Pipeline (`/dialogue`)
**Effort:** Very High (depends on ASR + TTS both being done first)
**Impact:** End-to-end voice assistant at ~700ms latency.

Chains ASR → LLM → TTS in a single WebSocket session.
Reference: [suharvest/rkvoice-stream](https://github.com/suharvest/rkvoice-stream)

---

## Pending — Low Priority / Nice-to-Have

### Swagger UI (`/docs`)
**Effort:** Very Low (flask-swagger-ui, ~30 min)
**Impact:** Auto-generated API docs for the endpoint surface.

Add `flask-swagger-ui` package and a static `openapi.yaml` describing all
routes. Mount at `/docs`. No server logic changes needed.

---

### Anthropic Claude API Compatibility (`POST /v1/messages`)
**Effort:** Low (2–3 hours shim)
**Impact:** Lets clients using the Anthropic Python SDK target this server.

The Anthropic messages format uses `"content"` as a list of blocks rather
than a plain string. Map to the existing `chat_completions` handler.

Reference: [huonwe/rkllm_openai_like_api](https://github.com/huonwe/rkllm_openai_like_api)

---

### HuggingFace Model Pull (`/v1/models/pull`)
**Effort:** Medium
**Impact:** Download pre-converted `.rkllm` files from HuggingFace without SSH.

Accept `{"repo": "dulimov/Qwen3-Embedding-0.6B-rk3588-1.2.1"}` and use
`huggingface_hub` to download into `MODELS_ROOT`. Stream download progress
as SSE. Auto-register the new model in config on completion.

---

### Image Generation (`/v1/images/generations`)
**Effort:** Very High
**Performance note:** MUST run as a completely separate service — SD inference
holds the NPU for several seconds and would severely stall chat generation.

LCM Stable Diffusion 1.5 on RK3588 RKNN — 512×512 in ~3–5 seconds:
[darkautism/LCM-Dreamshaper-V7-rs](https://github.com/darkautism/LCM-Dreamshaper-V7-rs)

---

## Research / Tracking

### Multi-batch / Multi-instance Inference (RKLLM v1.2.2+)
The runtime supports multiple model instances simultaneously since v1.2.2.
Could enable genuine request concurrency on the NPU rather than serialising
into one queue. No community server has implemented this yet.
Track: [airockchip/rknn-llm releases](https://github.com/airockchip/rknn-llm/releases)

### Distributed Inference Across Multiple Boards
Split large models (7B+) across two RK3588 boards using `start_layer/end_layer`
partial loading. Requested upstream but not yet supported in the runtime.
Track: [airockchip/rknn-llm#489](https://github.com/airockchip/rknn-llm/issues/489)
