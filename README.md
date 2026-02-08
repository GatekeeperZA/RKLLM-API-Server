# RKLLM API Server

**OpenAI-compatible API server for Rockchip NPU (RK3588/RK3576) running RKLLM models, designed as a drop-in backend for [Open WebUI](https://github.com/open-webui/open-webui).**

Built for single-board computers like the **Orange Pi 5 Plus**, this server bridges the gap between the `rkllm` C++ runtime and any OpenAI-compatible frontend — enabling local, private LLM inference on ARM hardware with zero cloud dependency.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Setup](#model-setup)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Open WebUI Configuration](#open-webui-configuration)
- [SearXNG Configuration](#searxng-configuration)
- [RAG Pipeline](#rag-pipeline)
- [Reasoning Models](#reasoning-models)
- [Configuration Reference](#configuration-reference)
- [Logging](#logging)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

### Core
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/models` endpoints work with any OpenAI client
- **Streaming & non-streaming** responses with proper SSE (Server-Sent Events) format
- **Auto-detection** of all `.rkllm` models in `~/models` directory
- **Context length auto-detection** from filename patterns (2k/4k/8k/16k/32k)
- **Auto-generated aliases** — short names like `qwen`, `deepseek` resolve automatically
- **Process reuse** for fast follow-up requests (no reload between turns)
- **Model hot-switching** — request a different model and it loads automatically
- **On-demand loading** via `/v1/models/select` for warm-up
- **Explicit unloading** via `/v1/models/unload` to free NPU memory

### Robustness
- **Request tracking** with automatic stale-request cleanup (prevents deadlocks)
- **Process health monitoring** with auto-recovery (background thread)
- **Idle auto-unload** — frees NPU memory after configurable idle period
- **Graceful shutdown** on SIGTERM/SIGINT with process cleanup
- **Bleed-through prevention** — process killed on interrupted generation to prevent leftover output contaminating the next response
- **RLock-based locking** — fixes model switch deadlock scenarios
- **SIGKILL for reliability** — aggressive unloading ensures NPU is fully released
- **Residual data guard** — discards output arriving faster than NPU prefill time

### RAG (Retrieval-Augmented Generation)
- **Automatic RAG detection** when Open WebUI injects web search results
- **Smart prompt restructuring** — reading comprehension format optimized for small models
- **4-pass web content cleaning** — strips navigation, boilerplate, cookie banners
- **Score-based paragraph selection** — jusText-inspired content quality scoring
- **Near-duplicate removal** — Jaccard similarity deduplication across sources
- **Quality floor** — drops irrelevant search results instead of confusing the model
- **Follow-up detection** — 3-layer system prevents RAG on conversational replies
- **Response caching** — LRU cache eliminates redundant inference for repeated questions
- **Context-dependent thinking** — disables reasoning on small context models to save tokens

### Reasoning Model Support
- **`<think>` tag parsing** for Qwen3, DeepSeek-R1 and similar models
- **`reasoning_content`** field in both streaming deltas and non-streaming responses
- **Streaming state machine** handles tags split across output chunks
- **Open WebUI integration** — reasoning appears as collapsible thinking blocks

### Standards Compliance
- **`stream_options.include_usage`** — streaming token counts per OpenAI spec
- **`system_fingerprint`** in all responses
- **`max_tokens` / `max_completion_tokens`** support
- **Request body size limit** (16 MB)
- **Proper error responses** matching OpenAI error format

---

## Architecture

```
┌──────────────┐    HTTP/SSE     ┌──────────────────┐    stdin/stdout    ┌──────────────┐
│              │  ──────────────▶│                  │  ────────────────▶│              │
│  Open WebUI  │    OpenAI API   │  RKLLM API       │    plain text      │  rkllm       │
│  (Frontend)  │  ◀──────────────│  Server (Flask)  │  ◀────────────────│  (C++ binary)│
│              │                 │                  │                    │              │
└──────────────┘                 └──────────────────┘                    └──────┬───────┘
                                        │                                      │
                                        │ monitor thread                       │
                                        │ health checks                   ┌────▼────┐
                                        └─────────────────────────────────│  RK3588  │
                                                                          │   NPU    │
                                                                          └──────────┘
```

**Key design decisions:**

1. **Plain text only** — The `rkllm` binary applies chat templates internally using actual token IDs. Special tokens (`<|im_start|>`, `<start_of_turn>`, etc.) are stripped from the text vocabulary during model conversion. Sending them as literal text causes the model to see garbage.

2. **Single worker** — The NPU can only load one model at a time. The server enforces `-w 1` (one gunicorn worker) and rejects concurrent requests with HTTP 503.

3. **Kill on interrupt** — The `rkllm` binary uses synchronous `rkllm_run()` which cannot be cancelled. On client disconnect or timeout, the process is killed and restarted to prevent output contamination.

---

## Requirements

### Hardware
- **Rockchip RK3588 or RK3576** SBC (Orange Pi 5 Plus, Rock 5B, etc.)
- **NPU driver** installed and functional
- Minimum **8GB RAM** recommended (16GB for larger models)

### Software
- **Linux** (ARM64) — tested on Ubuntu/Debian
- **Python 3.8+**
- **rkllm binary** — the C++ inference runtime, must be in `$PATH` or set via `RKLLM_BINARY` env var
- **RKLLM models** (`.rkllm` format) placed in `~/models/`

### Python Dependencies
```bash
pip install flask flask-cors gunicorn gevent
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/GatekeeperZA/RKLLM-API-Server.git
cd RKLLM-API-Server

# Install Python dependencies
pip install flask flask-cors gunicorn gevent

# Ensure rkllm binary is in PATH
which rkllm  # should return a path

# Create models directory
mkdir -p ~/models
```

---

## Model Setup

Place each `.rkllm` model in its own subfolder under `~/models/`:

```
~/models/
├── Qwen2.5-1.5B-Instruct-4K/
│   └── Qwen2.5-1.5B-Instruct-4K.rkllm
├── Qwen3-4B-16K/
│   └── Qwen3-4B-16K.rkllm
└── DeepSeek-R1-Distill-Qwen-1.5B-4K/
    └── DeepSeek-R1-Distill-Qwen-1.5B-4K.rkllm
```

### Context Length Detection

The server auto-detects context length from the filename or folder name:

| Pattern in name | Detected context |
|----------------|-----------------|
| `-2k` or `_2k` | 2,048 tokens |
| `-4k` or `_4k` | 4,096 tokens |
| `-8k` or `_8k` | 8,192 tokens |
| `-16k` or `_16k` | 16,384 tokens |
| `-32k` or `_32k` | 32,768 tokens |
| *(none found)* | 4,096 (default) |

### Auto-Generated Aliases

Model folder names are converted to IDs (lowercase, hyphens). Aliases are auto-generated:

| Model ID | Auto-Aliases |
|----------|-------------|
| `qwen2.5-1.5b-instruct-4k` | `qwen`, `qwen2`, `qwen2.5`, `qwen2.5-1.5b` |
| `deepseek-r1-distill-qwen-1.5b-4k` | `deepseek`, `deepseek-r1` |

Aliases are only created when unambiguous (one model claims the alias). If two models share a prefix, that alias is skipped.

---

## Running the Server

### Production (recommended)

```bash
gunicorn -w 1 -k gevent --timeout 300 -b 0.0.0.0:8000 api:app
```

> **Critical:** Always use `-w 1` (single worker). The NPU can only load one model at a time.

### Development

```bash
python api.py
```

This starts Flask's built-in server on `0.0.0.0:8000` with threading enabled.

### Systemd Service

```ini
[Unit]
Description=RKLLM API Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/RKLLM-API-Server
ExecStart=/usr/bin/gunicorn -w 1 -k gevent --timeout 300 -b 0.0.0.0:8000 api:app
Restart=always
RestartSec=5
Environment=RKLLM_API_LOG_LEVEL=INFO

[Install]
WantedBy=multi-user.target
```

---

## API Endpoints

### `GET /v1/models`

List all detected models.

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen2.5-1.5b-instruct-4k",
      "object": "model",
      "created": 1738972800,
      "owned_by": "rkllm"
    }
  ]
}
```

### `POST /v1/chat/completions`

OpenAI-compatible chat completions (streaming and non-streaming).

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "stream": true,
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

**Supported parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | *required* | Model ID or alias |
| `messages` | array | *required* | OpenAI messages format |
| `stream` | bool | `false` | Enable SSE streaming |
| `max_tokens` | int | `2048` | Max completion tokens |
| `stream_options.include_usage` | bool | `false` | Include token counts in stream |

### `POST /v1/models/select`

Pre-load a model without generating (warm-up).

```bash
curl -X POST http://localhost:8000/v1/models/select \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen"}'
```

### `POST /v1/models/unload`

Explicitly unload the current model to free NPU memory.

```bash
curl -X POST http://localhost:8000/v1/models/unload
```

### `GET /health`

Health check endpoint.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "current_model": "qwen2.5-1.5b-instruct-4k",
  "process_healthy": true,
  "active_request": null,
  "models_available": 3
}
```

---

## Open WebUI Configuration

### Connection

In Open WebUI **Admin > Settings > Connections**, add the API as an OpenAI-compatible endpoint:

| Setting | Value |
|---------|-------|
| API Base URL | `http://<device-ip>:8000/v1` |
| API Key | *(any non-empty string — the server has no auth)* |

### Web Search (for RAG)

**Admin > Settings > Web Search:**

| Setting | Value | Reason |
|---------|-------|--------|
| Search Engine | `searxng` | Self-hosted, JSON API |
| SearXNG Query URL | `http://searxng:8080/search` | Docker service name |
| Web Search Result Count | `3` (4k models) / `5` (16k models) | Balances coverage vs. context |
| Bypass Web Loader | **ON** | Snippets > raw page scraping for small models |
| Bypass Embedding | **ON** | No embedding model on ARM/NPU |

### Documents (RAG Template)

**Admin > Settings > Documents:**

| Setting | Value | Reason |
|---------|-------|--------|
| RAG Template | `{{CONTEXT}}` | The default template wastes 300+ tokens on meta-instructions |

### Per-Model Settings

**Workspace > Models > Edit > Capabilities:**

| Setting | Value | Reason |
|---------|-------|--------|
| Builtin Tools | **OFF** | Small models can't do function-calling |
| File Context | **ON** | Enables document/search result injection |

---

## SearXNG Configuration

The included `settings.yml` is optimized for Open WebUI on ARM hardware. Key settings:

```yaml
use_default_settings:
  engines:
    keep_only:
      - google
      - google news
      - duckduckgo
      - bing
      - brave
      - wikipedia

search:
  formats:
    - html
    - json    # REQUIRED for Open WebUI API access
```

**Installation:**
```bash
cp settings.yml ~/Downloads/searxng-docker/searxng/settings.yml
cd ~/Downloads/searxng-docker
docker compose down && docker compose up -d
```

---

## RAG Pipeline

When Open WebUI performs a web search, the results are injected into the system message as `<source>` tags. The server detects this and activates a specialized RAG pipeline:

### Processing Steps

1. **Detection** — `<source>` tags in the system message trigger RAG mode
2. **Extraction** — Content extracted from between `<source>...</source>` tags
3. **Web Content Cleaning** (4-pass):
   - Pass 1: Remove known boilerplate phrases (cookies, sign-in, privacy policy, etc.)
   - Pass 2: Remove navigation patterns (CamelCase runs, title-case-heavy lines, URL clusters)
   - Pass 3: Collapse consecutive short-line menus (4+ short lines = navigation)
   - Pass 4: Keep only lines with data signals (digits, prose punctuation, ≥40 chars)
4. **Deduplication** — Exact prefix key + Jaccard word-similarity removal
5. **Score-based selection** — jusText-inspired paragraph scoring:
   - Stopword density (prose ≥ 30%, boilerplate < 15%)
   - Length, sentence count, data presence
   - Query keyword matching (3x weight)
   - Negative signals: short fragments, navigation patterns, boilerplate keywords
6. **Quality floor** — If best paragraph scores below threshold, RAG is dropped entirely
7. **Prompt construction** — SQuAD-style reading comprehension format:
   ```
   {reference data}

   According to the above, {question}. Answer in detail with specific facts and examples
   ```

### Follow-Up Detection (3 Layers)

Open WebUI searches SearXNG with the raw user message. Short follow-ups produce garbage results:

| Layer | Trigger | Example |
|-------|---------|---------|
| Layer 1: Word list | Exact match to known conversational words | "yes", "thanks", "tell me more" |
| Layer 2: Short query | ≤3 words with conversation history | "south africa", "another one" |
| Layer 3: Topical overlap | < 30% query keywords found in reference text | Off-topic search results |

When any layer fires, RAG is skipped and the model uses normal conversation mode.

### Response Cache

RAG responses are cached in an LRU cache (key: model + question hash) to avoid redundant NPU inference:

| Setting | Default | Description |
|---------|---------|-------------|
| `RAG_CACHE_TTL` | 300s | Cache lifetime |
| `RAG_CACHE_MAX_ENTRIES` | 50 | Max cached responses |

---

## Reasoning Models

Models like **Qwen3** and **DeepSeek-R1** output chain-of-thought wrapped in `<think>...</think>` tags.

The server:
- Parses these tags from the token stream using a state machine (handles tags split across chunks)
- Sends `reasoning_content` in streaming deltas (Open WebUI displays these as collapsible thinking blocks)
- Returns `reasoning_content` in non-streaming responses
- **Context-dependent thinking for RAG**: On small context models (< 8k), thinking is disabled via `/no_think` to save tokens for the actual answer

---

## Configuration Reference

All configuration is at the top of `api.py`:

### Timeouts

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_LOAD_TIMEOUT` | 180s | Max time to wait for model initialization |
| `GENERATION_TIMEOUT` | 600s | Max total generation time |
| `FIRST_TOKEN_TIMEOUT` | 120s | Max wait for first token (includes prefill) |
| `FALLBACK_SILENCE` | 12s | Max silence between tokens after first |
| `MINIMUM_PREFILL_TIME` | 0.3s | Discard data arriving faster than this (residual guard) |

### Defaults

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_TOKENS_DEFAULT` | 2048 | Default max completion tokens |
| `CONTEXT_LENGTH_DEFAULT` | 4096 | Fallback when not detected from filename |

### RAG Controls

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_MIN_QUALITY_SCORE` | 2 | Minimum score for paragraph inclusion |
| `RAG_MAX_PARAGRAPHS` | 10 | Max paragraphs (prevents "lost in the middle") |
| `RAG_QUALITY_FLOOR_THRESHOLD` | 3 | Below this, RAG is dropped entirely |
| `RAG_DEDUP_SIMILARITY` | 0.70 | Jaccard threshold for near-duplicate removal |
| `RAG_CACHE_TTL` | 300 | Cache lifetime in seconds (0 to disable) |
| `RAG_CACHE_MAX_ENTRIES` | 50 | Max cached responses |
| `DISABLE_THINK_FOR_RAG_BELOW_CTX` | 8192 | Disable thinking for RAG when context < this |

### Process Management

| Variable | Default | Description |
|----------|---------|-------------|
| `REQUEST_STALE_TIMEOUT` | 30s | Auto-clear tracked request after this idle time |
| `MONITOR_INTERVAL` | 10s | Health check frequency |
| `IDLE_UNLOAD_TIMEOUT` | 300s | Auto-unload model after idle (0 to disable) |

### Protocol

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MULTILINE_PROTOCOL` | `False` | Send `\n__END__\n` delimited prompts (requires recompiled binary) |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `RKLLM_BINARY` | Path to rkllm binary (default: `rkllm` from `$PATH`) |
| `RKLLM_API_LOG_LEVEL` | Python API log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `RKLLM_LOG_LEVEL` | rkllm C runtime log level (integer 0-2, set to `1` for stats) |

---

## Logging

Logs are written to both **stderr** and a rotating log file (`api.log` in the script directory):

- **Max file size:** 10 MB
- **Backup count:** 3 rotated files
- **Default level:** `DEBUG` (set `RKLLM_API_LOG_LEVEL=INFO` for production)

### Log Examples

```
2026-02-08 17:45:12 [INFO] Detected: qwen2.5-1.5b-instruct-4k (context=4096)
2026-02-08 17:45:12 [INFO] Models: ['qwen2.5-1.5b-instruct-4k']
2026-02-08 17:45:12 [INFO] Aliases: {'qwen': 'qwen2.5-1.5b-instruct-4k', 'qwen2': 'qwen2.5-1.5b-instruct-4k'}
2026-02-08 17:45:30 [INFO] >>> NEW REQUEST chatcmpl-a1b2c3d4e5f6
2026-02-08 17:45:30 [INFO] Resolved alias 'qwen' -> 'qwen2.5-1.5b-instruct-4k'
2026-02-08 17:45:30 [INFO] REUSING process for qwen2.5-1.5b-instruct-4k (PID: 1234)
2026-02-08 17:45:31 [DEBUG] First token in 0.87s
2026-02-08 17:45:45 [INFO] Request ENDED: chatcmpl-a1b2c3d4e5f6
```

---

## Security

> **This server has NO authentication.** It is designed to run on a trusted local network.

- Binds to `0.0.0.0:8000` — accessible from all network interfaces
- No API key validation (any non-empty string works for Open WebUI)
- Request body limited to 16 MB
- **Do NOT expose directly to the public internet**
- Place behind a reverse proxy (nginx, Caddy) with authentication if external access is needed

---

## Troubleshooting

### "Model not found"
- Ensure the `.rkllm` file is inside a subfolder of `~/models/` (not directly in `~/models/`)
- Check folder naming — spaces become hyphens, underscores become hyphens
- Run `curl http://localhost:8000/v1/models` to see detected models

### "Failed to load model"
- Check that the `rkllm` binary is in `$PATH`: `which rkllm`
- Verify NPU driver is loaded: `dmesg | grep -i npu`
- Check `api.log` for init failure messages — may indicate corrupt `.rkllm` file or version mismatch

### "Another request is currently being processed" (503)
- NPU is single-task — only one request at a time
- Previous request may be stuck — check `/health` endpoint
- Stale requests auto-clear after 30s idle

### Streaming stops mid-response
- Check `FALLBACK_SILENCE` timeout (default 12s) — increase if model is slow
- Large prompts near context limit may cause long prefill — increase `FIRST_TOKEN_TIMEOUT`
- Check process health via `/health`

### RAG returns irrelevant answers
- Verify SearXNG is returning JSON: add `json` to `search.formats` in SearXNG settings
- Check if "Bypass Web Loader" is **ON** (recommended for small models)
- Set `RAG_QUALITY_FLOOR_THRESHOLD` higher to drop poor search results
- Check logs for "RAG SKIP" and "Quality floor triggered" messages

### High memory usage
- Set `IDLE_UNLOAD_TIMEOUT` to auto-unload after idle periods
- Use `/v1/models/unload` to manually free NPU memory
- Smaller quantized models (W4A16) use less memory

---

## Project Structure

```
RKLLM-API-Server/
├── api.py           # Main API server (Flask + rkllm process management)
├── settings.yml     # SearXNG configuration (optimized for Open WebUI)
├── .gitignore       # Excludes logs, __pycache__, etc.
└── README.md        # This file
```

---

## License

This project is provided as-is for personal and educational use. The `rkllm` runtime and model files are subject to their respective licenses from Rockchip and model authors.
