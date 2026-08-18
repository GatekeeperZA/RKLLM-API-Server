# RKLLM API Server

[![Build and Push Docker Image](https://github.com/GatekeeperZA/RKLLM-API-Server/actions/workflows/docker.yml/badge.svg)](https://github.com/GatekeeperZA/RKLLM-API-Server/actions/workflows/docker.yml)

**OpenAI-compatible API server for Rockchip NPU (RK3588/RK3576) running RKLLM models, designed as a drop-in backend for [Open WebUI](https://github.com/open-webui/open-webui).**

Built for single-board computers like the **Orange Pi 5 Plus**, this server bridges the gap between the `librkllmrt.so` C runtime and any OpenAI-compatible frontend — enabling local, private LLM inference on ARM hardware with zero cloud dependency.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [API Keys & Secrets to Configure](#api-keys--secrets-to-configure)
- [Installation](#installation)
  - [Docker (Recommended)](#docker-recommended)
  - [Automated Setup (systemd)](#automated-setup-recommended)
  - [Manual Installation](#manual-installation)
- [Pre-Built Models](#pre-built-models)
- [Model Setup](#model-setup)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Open WebUI Configuration](#open-webui-configuration)
  - [Docker Setup](#docker-setup)
  - [Embedding Model](#embedding-model-recommendation)
  - [Document RAG Settings](#document-rag-settings-recommended-for-pdfdocument-upload)
  - [VL / Image Upload Settings](#vl--image-upload-settings)
- [Home Assistant Integration](#home-assistant-integration)
- [SearXNG Configuration](#searxng-configuration)
- [RAG Pipeline](#rag-pipeline)
- [Reasoning Models](#reasoning-models)
- [KV Cache Strategy](#kv-cache-strategy)
- [Configuration Reference](#configuration-reference)
- [Logging](#logging)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)
- [File Structure](#file-structure)
- [V1 (Subprocess) vs V2 (ctypes) — Why We Migrated](#v1-subprocess-vs-v2-ctypes--why-we-migrated)
- [Tested Hardware](#tested-hardware)
- [Tested Models](#tested-models)
- [VL Model Evaluation](#vl-model-evaluation)
- [Vision Encoder Resolution Comparison](#vision-encoder-resolution-comparison)
- [Re-Exporting VL Models at Higher Resolution](#re-exporting-vl-models-at-higher-resolution)
- [Benchmarks](#benchmarks)
- [Ollama API Compatibility](#ollama-api-compatibility)
- [LoRA Hot-Swap](#lora-hot-swap)
- [Hermes Agent Integration](#hermes-agent-integration)
- [Known Issues & Fixes](#known-issues--fixes)
- [Git Tags & Branches](#git-tags--branches)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

### Core
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/models`, `/v1/models/<id>`, `/v1/completions` endpoints work with any OpenAI client
- **Stop sequences** — `stop` parameter fully implemented (string or array); generation aborts at first match
- **Direct NPU access** via ctypes binding to `librkllmrt.so` (no subprocess overhead)
- **KV cache incremental mode** — follow-up turns only prefill the new message (~50ms vs ~500ms)
- **Prompt cache preloading** — saves KV state to disk after first inference; subsequent model loads restore it instantly, skipping system prompt re-prefill
- **Model-aware sampling profiles** — per-family tuned sampling parameters (Qwen3, Gemma, Phi, DeepSeek) with `model_config.json` override support
- **Context-aware sliding window** — automatically trims oldest conversation turns when history exceeds context length, keeping the most recent exchange intact
- **Context overflow hard-rejection** — if the built prompt exceeds 110% of the model's context length (after sliding window), the request is rejected with a `400 context_length_exceeded` error instead of sending an oversized prompt to the NPU runtime
- **Streaming & non-streaming** responses with proper SSE (Server-Sent Events) format
- **Auto-detection** of all `.rkllm` models in `~/models` directory
- **Context length auto-detection** from filename patterns (2k/4k/8k/16k/32k)
- **Auto-generated aliases** — short names like `qwen`, `phi` resolve automatically
- **Multi-turn conversation history** — full chat context preserved via KV cache across turns
- **Model hot-switching** — request a different model and it loads automatically
- **On-demand loading** via `/v1/models/select` for warm-up
- **Explicit unloading** via `/v1/models/unload` to free NPU memory

### Robustness
- **Request tracking** with automatic stale-request cleanup (prevents deadlocks)
- **Idle auto-unload** — frees NPU memory after configurable idle period (default 5 min)
- **Clean abort** — native `rkllm_abort()` for instant cancellation (no SIGKILL needed)
- **Graceful shutdown** on SIGTERM/SIGINT with model cleanup
- **RLock-based locking** — prevents model switch deadlock scenarios
- **Repetition loop detection** — aborts generation when the model enters paragraph-level repetition loops (configurable window/threshold)
- **SSE heartbeats during prefill** — sends keep-alive comments every 15s during long prefill to prevent HTTP proxy/client timeouts
- **Error callback state** — detects C library errors and surfaces them as proper HTTP responses

### RAG (Retrieval-Augmented Generation)
- **Automatic RAG detection** when Open WebUI injects web search or document results
- **Document/PDF RAG** — works with Open WebUI's document upload and embedding pipeline
- **Summarization detection** — detects "summarize" queries and adds stronger multi-paragraph instructions
- **Smart prompt restructuring** — reading comprehension format optimized for small models
- **5-pass web content cleaning** — strips navigation, boilerplate, cookie banners, stale dates
- **Score-based paragraph selection** — jusText-inspired content quality scoring
- **Near-duplicate removal** — Jaccard similarity deduplication across sources
- **Quality floor** — drops irrelevant search results instead of confusing the model
- **Follow-up detection** — 3-layer system prevents RAG on conversational replies
- **Response caching** — LRU cache eliminates redundant inference for repeated questions
- **Context-dependent thinking** — disables reasoning on small context models to save tokens
- **Auto-capability detection** — infers model type (thinking, instruct, VL, OCR) from folder names; only reasoning models get `<think>` enabled

### Reasoning Model Support
- **`<think>` tag parsing** for Qwen3 and similar reasoning models
- **`reasoning_content`** field in both streaming deltas and non-streaming responses
- **Streaming state machine** handles tags split across output chunks
- **Thinking block stripping** — `<think>...</think>` blocks are automatically stripped from assistant history before re-sending to the model (per Qwen3 docs: "historical output should only include the final output")
- **Open WebUI integration** — reasoning appears as collapsible thinking blocks

### Open WebUI Meta-Task Shortcuts
- **Query generation shortcircuit** — Open WebUI asks the model to generate search queries for retrieval; instead of wasting 5s of inference, the server extracts the user's actual question from the chat history and returns it as the query instantly (~0ms). For vague follow-ups ("can you verify that?"), it enriches the query with entities extracted from the assistant's previous response (bold text, quoted strings, capitalized phrases)
- **Title generation shortcircuit** — extracts the first user message as the chat title (~0ms instead of 5-10s inference)
- **Tag generation shortcircuit** — returns a default tag instantly (~0ms instead of 5-10s inference)
- **Meta-task thinking disabled** — auto-detects Open WebUI internal tasks (query gen, title gen, tags, autocomplete) and disables `<think>` reasoning to avoid wasting 20+ seconds on trivial tasks
- **No JSON leakage** — query generation shortcircuit prevents raw JSON from appearing in the chat display

### Home Assistant
- **Auto-detection** — recognizes Home Assistant requests by system prompt signatures (`smart home manager`, `Available Devices:`, `execute_services`)
- **Thinking auto-disabled** — skips `<think>` reasoning for HA requests, cutting response time in half
- **Compatible with Extended OpenAI Conversation** (HACS) — works as a drop-in conversation agent for Assist

### Monitoring
- **Prometheus metrics** (optional) — `rkllm_tokens_generated`, `rkllm_prefill_duration`, `rkllm_decode_duration`, `rkllm_tokens_per_request`, `rkllm_queue_wait_seconds`, `rkllm_active_requests`, `rkllm_model_load_seconds`, `rkllm_current_model` — exposed at `/metrics`
- **NPU hardware metrics** — `rknpu_core_load_percent` (per-core %), `rknpu_freq_mhz`, `rknpu_power_on` — polled every 5s from `/sys/kernel/debug/rknpu/` via a background thread
- **Graceful degradation** — metrics are disabled automatically if `prometheus-flask-exporter` is not installed

### Standards Compliance
- **`stream_options.include_usage`** — streaming token counts per OpenAI spec
- **`system_fingerprint`** in all responses
- **`x-ratelimit-*` headers** — `limit-requests`, `limit-tokens`, `remaining-requests`, `remaining-tokens`, `reset-requests` on all completions responses
- **Request queuing** — concurrent requests wait up to `RKLLM_QUEUE_TIMEOUT` seconds (default 30s) instead of getting an immediate 503. `x-queue-wait-seconds` header is returned on queued responses
- **Tool call streaming** — role delta is emitted immediately when tool calling is active, keeping the connection alive during silent buffering
- **`/ready` / `/v1/ready`** — readiness probe (503 when no model loaded or busy)
- **`/v1/tokenize`** — BPE token estimation without inference
- **`/v1/count_tokens`** — Anthropic SDK alias for token counting
- **`/v1/rerank`** — reranker proxy to NPU embed service
- **`/v1/messages`** — Anthropic Messages API shim (Claude-format requests)
- **`max_tokens` / `max_completion_tokens`** support
- **Request body size limit** (50 MB)
- **Proper error responses** matching OpenAI error format

### Ollama API Compatibility
- **`GET /api/version`** — returns `{"version": "0.3.0-rkllm"}` so Ollama-native clients auto-connect
- **`GET /api/tags`** — lists all loaded models in Ollama format (name, size, digest, details)
- **`POST /api/chat`** — Ollama chat completions; streaming returns NDJSON, non-streaming returns Ollama JSON envelope
- **`POST /api/generate`** — wraps a raw prompt as a user message and delegates to `/api/chat`
- **`GET /api/show`** and **`GET /api/ps`** — stub endpoints for Ollama-compatible dashboards
- Allows tools like [Open WebUI in Ollama mode](https://docs.openwebui.com) to connect without the OpenAI adapter

### LoRA Adapters
- **`GET /v1/lora`** — list all loaded LoRA adapters
- **`POST /v1/lora/load`** — load a LoRA adapter from disk at runtime without model restart (calls `rkllm_load_lora()` via ctypes)
- **`DELETE /v1/lora/<name>`** — unregister an adapter from the server's adapter table
- Adapter scale (α) is configurable per request via the `scale` field
- Requires `rkllm_load_lora` to be exported by the installed `librkllmrt.so`

### Vision-Language (VL) / Multimodal
- **Dual-model architecture** — text model (e.g. Qwen3-1.7B) + VL model (e.g. Qwen3-VL-2B) loaded simultaneously
- **Automatic image routing** — requests with images route to VL model, text-only to text model
- **Base64 image support** — accepts `image_url` with `data:image/...;base64,...` format (Open WebUI compatible)
- **Direct NPU vision encoding** via ctypes binding to `librknnrt.so` (no Python RKNN toolkit needed)
- **Image preprocessing** — auto square-pad (128,128,128 background) and resize to encoder input size
- **Multiple VL model support** — auto-detects `.rknn` vision encoder alongside `.rkllm` decoder
- **Configurable special tokens** — `VL_MODEL_CONFIGS` maps model families to their image tokens
- **Multi-turn VL context** — follow-up questions, RAG/web search data, and conversation history are included in VL prompts (not just the original image caption)
- **Seamless Open WebUI experience** — paste/upload images in chat, responses stream normally

---

## Architecture

```
┌──────────────┐     HTTP/SSE      ┌──────────────────────────────────┐
│  Open WebUI  │ ◄──────────────── │   api.py (Flask + gunicorn)      │
│  or any      │ ─────────────────►│   gthread, -w 1                  │
│  OpenAI      │                   │                                  │
│  client      │                   │  ┌──────────────────────────┐    │
└──────────────┘                   │  │  VL Auto-Router          │    │
                                   │  │  (image → VL, text → LLM)│    │
        ┌──────────┐               │  └────┬─────────────┬───────┘    │
        │ SearXNG  │ ◄──── Open    │       │ text        │ image      │
        │ (search) │  WebUI injects│       ▼             ▼            │
        └──────────┘  results      │  ┌─────────┐  ┌─────────────┐   │
                                   │  │ Prompt  │  │ Vision Enc. │   │
        ┌──────────┐               │  │ Builder │  │ librknnrt.so│   │
        │  Ollama  │               │  │ + RAG   │  │ (.rknn NPU) │   │
        │  (CPU)   │               │  └────┬────┘  └──────┬──────┘   │
        └──────────┘               │       │              │           │
         optional                  │       ▼              ▼           │
                                   │  ┌──────────────────────────┐   │
                                   │  │  librkllmrt.so v1.2.3    │   │
                                   │  │  Text: RKLLMWrapper      │   │
                                   │  │  VL:   RKLLMWrapper #2   │   │
                                   │  │  C callback → Queue      │   │
                                   │  └────────────┬─────────────┘   │
                                   │               │                 │
                                   │  ┌────────────▼─────────────┐   │
                                   │  │  RK3588 NPU (3 cores)    │   │
                                   │  │  6 TOPS per core         │   │
                                   │  └──────────────────────────┘   │
                                   │                                  │
                                   │  ┌──────────────────────────┐   │
                                   │  │  ThinkTagParser          │   │
                                   │  │  (reasoning_content)     │   │
                                   │  └──────────────────────────┘   │
                                   └──────────────────────────────────┘
```

**Key design decisions:**

1. **Plain text only** — The rkllm runtime applies chat templates internally using actual token IDs. Special tokens (`<|im_start|>`, `<start_of_turn>`, etc.) are stripped from the text vocabulary during model conversion. Sending them as literal text causes the model to see garbage.

2. **Single worker** — The NPU can only load one model at a time. The server enforces `-w 1` (one gunicorn worker) and rejects concurrent generation with HTTP 503.

3. **ctypes + callback** — The C library's `rkllm_run()` is blocking, so it runs in a worker thread. A C callback pushes tokens to a `queue.Queue`, which the main thread reads and yields as SSE chunks. This keeps the KV cache in-process across turns.

4. **gthread, not gevent** — `rkllm_run()` is a blocking C function that freezes gevent's event loop. Using `-k gthread` with real OS threads avoids this.

5. **Dual-model VL** — Text and VL models are loaded simultaneously into separate `RKLLMWrapper` instances. The vision encoder runs on a third ctypes binding (`librknnrt.so`). Image requests are auto-routed to the VL pipeline; text requests use the primary model. A shared `_token_queue` serialized by `PROCESS_LOCK` prevents interleaving. Default VL model: **Qwen3-VL-2B** (replaced DeepSeekOCR-3B — smaller, saves ~1.1 GB RAM).

---

## Requirements

### Tested System

This project was developed and tested on:

| Component | Details |
|-----------|--------|
| **Board** | Orange Pi 5 Plus (16 GB RAM) |
| **SoC** | Rockchip RK3588 (3 NPU cores) |
| **OS** | [Armbian Pelochus 24.11.0](https://github.com/Pelochus/armbian-build-rknpu-updates/releases) — `Armbian-Pelochus_24.11.0-OrangePi5-plus_jammy_vendor.7z` |
| **Kernel NPU Driver** | 0.9.8 (**included in the Pelochus image** — no driver build required) |
| **RKLLM Runtime** | v1.2.3 (only the runtime library needs to be installed) |

> **Why Pelochus Armbian?** The standard Armbian images ship with an older RKNPU driver (0.9.6 or earlier). The [Pelochus builds](https://github.com/Pelochus/armbian-build-rknpu-updates/releases) bundle **RKNPU driver 0.9.8** in the kernel, so you only need to install the RKLLM runtime — no kernel module compilation required.

### Hardware
- **Rockchip RK3588 or RK3576** SBC (Orange Pi 5 Plus, Rock 5B, etc.)
- **NPU driver** installed and functional
- Minimum **8 GB RAM** recommended (16 GB for larger models)

### Software
- **Linux** (ARM64) — tested on Ubuntu/Debian (Armbian)
- **Python 3.8+**
- **RKNPU driver ≥ 0.9.6** (0.9.8 recommended — see [Installation](#installation))
- **RKLLM Runtime ≥ v1.2.0** (tested with v1.2.3) — `librkllmrt.so` shared library (see [Installation](#installation))
- **RKLLM models** (`.rkllm` format) placed in `~/models/`
- **RKNN Runtime** (optional) — `librknnrt.so` shared library (only needed for VL/multimodal models with `.rknn` vision encoders)

> **SDK Version Coupling:** The ctypes struct definitions in `api.py` target the RKLLM SDK v1.2.x C header (`rkllm.h`). Older SDK versions used a flat 112-byte reserved blob in `RKLLMExtendParam` and lacked fields like `n_keep`, `n_batch`, `use_cross_attn`, and `enable_thinking`. Running this server against an older `librkllmrt.so` (pre-1.2) will cause **silent struct-offset misalignment** — the parameter block passed to `rkllm_init()` would be corrupted, producing wrong sampling behaviour rather than a crash. Always use the runtime from the [v1.2.x release](https://github.com/airockchip/rknn-llm) or later.
>
> **Do not upgrade to v1.3.0 without updating `api.py` first.** RKLLM v1.3.0 adds new fields to `RKLLMInferParam` (`max_new_tokens`, `sampling_params`) and `RKLLMParam` (`ignore_eos_token`), and changes the multimodal input interface. Swapping the `.so` without updating the ctypes struct definitions will cause silent struct-offset corruption or crashes. A complete v1.3.0 upgrade attempt is available on the [`rkllm-v1.3.0-wip`](https://github.com/GatekeeperZA/RKLLM-API-Server/tree/rkllm-v1.3.0-wip) branch — it is rolled back and not production-ready due to a phase-2 early EOS bug. See [Git Tags & Branches](#git-tags--branches) for details.

### Python Dependencies
```bash
# Core (required)
pip install flask flask-cors gunicorn

# Prometheus monitoring (optional — metrics at /metrics endpoint)
pip install prometheus-flask-exporter prometheus-client

# VL / multimodal support (optional — needed only for vision-language models)
pip install numpy Pillow
```

---

## API Keys & Secrets to Configure

This project uses several API keys. The table below shows every key, where to set it, and what it's for.

| Key | Where to set | Required? | Purpose |
|-----|-------------|-----------|---------|
| `OPENROUTER_API_KEY` | `/etc/rkllm-hermes.env` or `export` before `setup.sh` | Optional | Hermes agent — primary cloud LLM (Nemotron 550B free tier) |
| `GROQ_API_KEY` | `/etc/rkllm-hermes.env` or `export` before `setup.sh` | Optional | Hermes agent — fallback cloud LLM (Llama 3.3 70B free tier) |
| `HERMES_API_KEY` | Auto-generated by `setup.sh`, written to `~/.hermes/.env` | Auto | Gateway auth key — added to OpenWebUI as the Hermes connection key |
| `OWUI_API_KEY` | OpenWebUI → Settings → Account → API Keys | For tests only | Used by `tests/test_openwebui.py` — set via `OWUI_API_KEY` env var |

### /etc/rkllm-hermes.env (created by setup.sh)

`setup.sh` writes this file automatically. If you need to update keys after install:

```bash
sudo nano /etc/rkllm-hermes.env
# Contents:
# API_SERVER_KEY=<hermes-gateway-key>        # auto-generated by setup.sh
# OPENROUTER_API_KEY=<your-openrouter-key>   # optional
# GROQ_API_KEY=<your-groq-key>               # optional
sudo chmod 600 /etc/rkllm-hermes.env
sudo systemctl restart rkllm-hermes
```

### Running tests

```bash
# OpenWebUI integration tests — requires OWUI API key
OWUI_HOST=192.168.2.180 OWUI_API_KEY=sk-owui-xxxx python tests/test_openwebui.py

# Direct API tests — no key required (RKLLM API has no auth)
python tests/e2e_test.py
```

> The RKLLM API server (`api.py` on port 8000) and the embed service (`embed_api.py` on port 8001) have **no authentication** — they are designed to be LAN-only services. Do not expose them directly to the internet.

---

## Installation

### Docker (Recommended)

The fastest way to get running. A pre-built ARM64 image is published automatically to GitHub Container Registry on every push to `main`.

**Prerequisites:** Docker installed, NPU driver 0.9.8+ on the host (the container shares the host kernel — the driver runs on the host, not inside Docker).

```bash
docker run -d \
  --name rkllm \
  --privileged \
  -p 8000:8000 \
  -v /home/armbian/models:/root/models \
  --restart unless-stopped \
  ghcr.io/gatekeeperza/rkllm-api-server:latest
```

> **`--privileged` is required** — the NPU kernel driver (`/dev/rknpu`, `/dev/dri`) requires elevated device access.

**With Open WebUI as a compose stack:**

```yaml
# docker-compose.yml
services:
  rkllm:
    image: ghcr.io/gatekeeperza/rkllm-api-server:latest
    privileged: true
    ports:
      - "8000:8000"
    volumes:
      - /home/armbian/models:/root/models
    restart: unless-stopped

  openwebui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      OPENAI_API_BASE_URL: http://rkllm:8000/v1
      OPENAI_API_KEY: none
    volumes:
      - open-webui:/app/backend/data
    depends_on:
      - rkllm
    restart: unless-stopped

volumes:
  open-webui:
```

```bash
docker compose up -d
```

**Environment variables** (all optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `RKLLM_MODELS_ROOT` | `/root/models` | Path to models directory inside the container |
| `RKLLM_API_LOG_LEVEL` | `DEBUG` | Log verbosity (`DEBUG`, `INFO`, `WARNING`) |
| `RKLLM_API_LOG_FILE` | *(stderr only in container)* | Set to a path to also write a log file |
| `RKLLM_LIB_PATH` | auto-detected | Override path to `librkllmrt.so` |
| `RKNN_LIB_PATH` | auto-detected | Override path to `librknnrt.so` (VL models) |
| `GUNICORN_WORKERS` | `1` | Always keep at 1 — NPU is single-instance |
| `GUNICORN_THREADS` | `4` | Request handler threads |
| `GUNICORN_TIMEOUT` | `300` | Request timeout in seconds |
| `GUNICORN_BIND` | `0.0.0.0:8000` | Bind address |
| `RKLLM_QUEUE_TIMEOUT` | `240` | Seconds a queued request waits for the NPU slot before returning 503 |

**Check health:**
```bash
docker logs rkllm -f
curl http://localhost:8000/v1/models
```

---

### Automated Setup (Recommended)

A zero-configuration setup script is included that handles **everything** — system packages, Python venv, RKLLM runtime installation, kernel module/driver verification, udev rules, systemd service, and NPU frequency fix:

```bash
git clone https://github.com/GatekeeperZA/RKLLM-API-Server.git
cd RKLLM-API-Server
chmod +x setup.sh
./setup.sh
```

> **Do NOT run as root.** The script uses `sudo` internally only where needed (installing system packages, copying libraries, creating the systemd service). User-level files (venv, models directory) are owned by your normal account.

The script is **idempotent** — safe to run multiple times. It detects what's already installed and skips those steps.

**What it installs / verifies:**
- System packages: `python3`, `python3-venv`, `python3-pip`, `build-essential`, `git`, `git-lfs`
- RKNPU kernel module check (`lsmod`, `modinfo`, `/dev/rknpu`, udev rules, `render` group)
- RKLLM Runtime: `librkllmrt.so` → `/usr/lib/`
- Python venv (`.venv`) with `flask`, `flask-cors`, `gunicorn`
- Systemd services: `rkllm-api` (API server) + `fix-freq` (NPU/CPU frequency governor)

After setup, download a model and start the service:
```bash
# Download Qwen3-1.7B (recommended)
mkdir -p ~/models/Qwen3-1.7B && cd ~/models/Qwen3-1.7B
git lfs install && git clone https://huggingface.co/GatekeeperZA/Qwen3-1.7B-RKLLM-v1.2.3 .

# Start the server
sudo systemctl start rkllm-api

# Check status
sudo systemctl status rkllm-api
curl http://localhost:8000/v1/models
```

---

### Docker Installation

A pre-built docker image is provided at ghcr.io, use following command to run rkllm-api-server:
```bash
docker run -d \
   --privileged \
   -p 8000:8000 \
   -v /path/to/models:/root/models \
   ghcr.io/gatekeeperza/rkllm-api-server:latest

```

---

### Manual Installation

<details>
<summary>Click to expand manual step-by-step instructions</summary>

#### 1. Clone This Repository

```bash
git clone https://github.com/GatekeeperZA/RKLLM-API-Server.git
cd RKLLM-API-Server

# Install Python dependencies
pip install flask flask-cors gunicorn numpy Pillow

# Create models directory
mkdir -p ~/models
```

#### 2. RKNPU Driver 0.9.8

The RKNPU kernel driver enables communication with the NPU hardware. Some board images ship with an older driver — you need **≥ 0.9.6** (0.9.8 recommended).

**Check your current driver version:**
```bash
dmesg | grep -i rknpu
# Look for a line like: "RKNPU driver loaded version 0.9.8"
# or:
cat /sys/kernel/debug/rknpu/version 2>/dev/null || echo "Check dmesg"
```

**If you need to update:**

The driver source is included in the [rknn-llm](https://github.com/airockchip/rknn-llm) repository as a pre-built tarball. It must be compiled against your running kernel's headers.

```bash
# Clone the rknn-llm repo (if not already done)
git clone https://github.com/airockchip/rknn-llm.git
cd rknn-llm/rknpu-driver

# Extract the driver source
tar xjf rknpu_driver_0.9.8_20241009.tar.bz2
cd rknpu_driver_0.9.8

# Install kernel headers (required for compilation)
sudo apt update
sudo apt install -y linux-headers-$(uname -r) build-essential

# Build the driver module
make -C /lib/modules/$(uname -r)/build M=$(pwd)/drivers/rknpu modules

# Install the new driver
sudo cp drivers/rknpu/rknpu.ko /lib/modules/$(uname -r)/kernel/drivers/rknpu/
sudo depmod -a

# Load the new driver (or reboot)
sudo modprobe -r rknpu 2>/dev/null  # unload old
sudo modprobe rknpu                  # load new

# Verify
dmesg | tail -5 | grep -i rknpu
```

> **Note:** Many Armbian and Orange Pi images already include RKNPU driver 0.9.8. Check before building. If `dmesg | grep rknpu` shows `0.9.8`, you're good.

> **Recommended:** The [Pelochus Armbian builds](https://github.com/Pelochus/armbian-build-rknpu-updates/releases) ship with RKNPU driver 0.9.8 pre-installed — no manual driver compilation needed. Use `Armbian-Pelochus_24.11.0-OrangePi5-plus_jammy_vendor.7z` (or the latest release for your board) and skip straight to the runtime setup.

#### 3. RKLLM Runtime ≥ v1.2.0 (tested with v1.2.3)

The RKLLM runtime provides the `librkllmrt.so` shared library that this API server loads via ctypes. The ctypes struct layouts in `api.py` require **SDK v1.2.0 or later** — see [Requirements](#requirements) for details on version coupling.

```bash
# Clone the rknn-llm repo (if not already done)
git clone https://github.com/airockchip/rknn-llm.git
cd rknn-llm

# --- Install the runtime library ---
sudo cp rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so /usr/lib/
sudo ldconfig

# Verify the library is findable
ldconfig -p | grep rkllm
# Should show: librkllmrt.so => /usr/lib/librkllmrt.so
```

**Verify everything works:**
```bash
# Check RKNPU driver
dmesg | grep -i rknpu

# Check runtime library
ldconfig -p | grep rkllm
```

#### 4. Fix NPU Frequency (Recommended)

For consistent performance, pin the NPU and CPU frequencies. The rknn-llm repo includes scripts for this:

```bash
cd rknn-llm/scripts

# RK3588
sudo bash fix_freq_rk3588.sh

# RK3576 (if using that platform)
sudo bash fix_freq_rk3576.sh
```

> Run this after each reboot, or use the setup script which creates a systemd service for automatic frequency pinning.

</details>

---

## Pre-Built Models

Ready-to-run `.rkllm` models converted by the author for RK3588 NPU are available on HuggingFace:

**Text / Instruct models:**

| Model | Parameters | Quant | Context | Speed | RAM | Thinking | Link |
|-------|-----------|-------|---------|-------|-----|----------|------|
| **xLAM-1b-fc-r** | 1.1B | w8a8 | 8,192 | ~16 tok/s | ~1.6 GB | ❌ No | [Download](https://huggingface.co/GatekeeperZA/xLAM-1b-fc-r-RKLLM-v1.2.3) |
| **Qwen3-1.7B** | 1.7B | w8a8 | 4,096 | ~13.6 tok/s | ~2 GB | ✅ Yes | [Download](https://huggingface.co/GatekeeperZA/Qwen3-1.7B-RKLLM-v1.2.3) |
| **Llama-3.2-3B-Instruct** | 3B | w8a8 | 8,192 | — | ~3.5 GB | ❌ No | [Download](https://huggingface.co/GatekeeperZA/Llama-3.2-3B-Instruct-RKLLM-v1.2.3) |
| **Qwen3-4B-Instruct-2507** | 4B | w8a8 | 16,384 | — | ~5.5 GB | ❌ No | [Download](https://huggingface.co/GatekeeperZA/Qwen3-4B-Instruct-2507-RKLLM-v1.2.3) |
| ~~Qwen2.5-1.5B-Instruct~~ | 1.5B | w8a8 | 8,192 | ~14 tok/s | ~2.1 GB | ❌ No | [Download](https://huggingface.co/GatekeeperZA/Qwen2.5-1.5B-Instruct-RKLLM-v1.2.3) *(deprecated — superseded by Qwen3-1.7B)* |

**Vision-Language models (VLM):**

| Model | Parameters | Quant | Type | RAM | Link |
|-------|-----------|-------|------|-----|------|
| **Qwen3-VL-2B-Instruct** | 2B | w8a8 | VLM | ~3 GB | [Download](https://huggingface.co/GatekeeperZA/Qwen3-VL-2B-Instruct-RKLLM-v1.2.3) |
| **InternVL3.5-4B-Instruct** | 4B | w8a8 | VLM | ~5.5 GB | [Download](https://huggingface.co/GatekeeperZA/InternVL3.5-4B-Instruct-RKLLM-v1.2.3) |
| **Qwen3-VL-4B-Instruct** | 4B | w8a8 | VLM | ~5.5 GB | [Download](https://huggingface.co/GatekeeperZA/Qwen3-VL-4B-Instruct-RKLLM-v1.2.3) |

**Retrieval / Embedding models:**

| Model | Parameters | Quant | Type | RAM | Link |
|-------|-----------|-------|------|-----|------|
| **Qwen3-Embedding-0.6B** | 0.6B | w8a8 | Embedding | ~1 GB | [Download](https://huggingface.co/GatekeeperZA/Qwen3-Embedding-0.6B-RKLLM-v1.2.3) |
| **Qwen3-Reranker-0.6B** | 0.6B | w8a8 | Reranker | ~1 GB | [Download](https://huggingface.co/GatekeeperZA/Qwen3-Reranker-0.6B-RKLLM-v1.2.3) |

> Browse all models: **[huggingface.co/GatekeeperZA](https://huggingface.co/GatekeeperZA)**

All models are converted with **RKLLM Toolkit v1.2.3**, targeting **RK3588 (3 NPU cores)**, and tested on an **Orange Pi 5 Plus** (16 GB RAM, RKNPU driver 0.9.8).

> **⚠️ DeepSeek-R1 on NPU — Currently Not Usable**
>
> DeepSeek-R1 (including distilled variants like DeepSeek-R1-Distill-Qwen-1.5B) **does not work correctly** with RKLLM Runtime v1.2.3. The model converts without errors but produces garbage output — repeating `[PAD151935]` tokens instead of real text ([rknn-llm#424](https://github.com/airockchip/rknn-llm/issues/424)). The Airockchip team has acknowledged this is a known issue and stated it will be fixed in a future runtime version.
>
> **For NPU reasoning, use Qwen3-1.7B instead** — it supports `<think>` tags, runs at ~13.6 tok/s on the NPU, and works reliably with RKLLM v1.2.3.
>
> If you need DeepSeek-R1, run `deepseek-r1:7b` via **Ollama on CPU** — it works correctly (just slower, ~2-3 tok/s on RK3588 ARM cores). See [Using Ollama Alongside](#using-ollama-alongside-cpu-models) below.

### Quick Download

```bash
# Install git-lfs (required for large files)
sudo apt install git-lfs
git lfs install

# Qwen3-1.7B (thinking/reasoning model — recommended)
mkdir -p ~/models/Qwen3-1.7B
cd ~/models/Qwen3-1.7B
git clone https://huggingface.co/GatekeeperZA/Qwen3-1.7B-RKLLM-v1.2.3 .

# Phi-3-mini (3.8B — strong at math/code, MIT licensed)
mkdir -p ~/models/Phi-3-mini-4k-instruct
cd ~/models/Phi-3-mini-4k-instruct
git clone https://huggingface.co/GatekeeperZA/Phi-3-mini-4k-instruct-w8a8 .
```

### Model Notes

**Qwen3-1.7B** — Hybrid thinking model. Produces `<think>...</think>` reasoning blocks that this API server parses into `reasoning_content` for Open WebUI's collapsible thinking display. Supports English and Chinese.

**Phi-3-mini-4k-instruct** — Microsoft's 3.8B parameter model excelling at reasoning, math (85.7% GSM8K), and code generation (57.3% HumanEval). English-primary. No thinking mode — this is a standard instruct model. MIT licensed.

---

## Model Setup

Place each `.rkllm` model in its own subfolder under `~/models/`:

```
~/models/
├── Qwen3-1.7B/
│   └── Qwen3-1.7B-w8a8-rk3588.rkllm
├── Qwen3-4B-Instruct-2507/
│   └── Qwen3-4B-Instruct-16k-w8a8-rk3588.rkllm
├── Gemma-3-4B-IT/
│   └── Gemma-3-4B-IT-w8a8-rk3588.rkllm
└── Phi-3-Mini-4K-Instruct/
    └── Phi-3-Mini-4K-Instruct-w8a8-rk3588.rkllm
```

### VL (Vision-Language) Model Setup

VL models require **two** files in the same folder: a `.rkllm` decoder and a `.rknn` vision encoder.

```
~/models/
├── Qwen3-1.7B/                          # Text-only model
│   └── Qwen3-1.7B-w8a8-rk3588.rkllm
└── Qwen3-VL-2b/                          # VL model (text + vision)
    ├── Qwen3-VL-2b-w8a8-rk3588.rkllm     # LLM decoder
    └── Qwen3-VL-2b-vision-encoder.rknn   # Vision encoder
```

**How it works:**
1. The server auto-detects `.rknn` files alongside `.rkllm` files
2. The folder name is matched against `VL_MODEL_CONFIGS` (supports DeepSeekOCR, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, InternVL3, MiniCPM)
3. When a chat request includes an image (base64 `image_url`), it auto-routes to the VL model
4. Text-only requests continue using the text model normally

**Supported VL models** (model folder name must contain):
| Pattern | Model Family | Notes |
|---------|-------------|-------|
| `qwen3-vl` | Qwen3-VL | **Recommended** — best OCR, fastest |
| `qwen2.5-vl` | Qwen2.5-VL | Lower 392×392 encoder |
| `qwen2-vl` | Qwen2-VL | Lower 392×392 encoder |
| `internvl3` | InternVL3 / InternVL3.5 | W8A8 precision loss, poor OCR |
| `deepseekocr` | DeepSeekOCR | Severe hallucination in RKNN conversion |
| `minicpm` | MiniCPM-V | Untested |

**Requirements:**
- `numpy` and `Pillow` Python packages (installed by `setup.sh`)
- `librknnrt.so` (RKNN runtime library, usually at `/usr/lib/librknnrt.so`)
- Sufficient RAM for both models (~4.4 GB for Qwen3-1.7B + Qwen3-VL-2B)

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

### Model Capabilities Detection

The server auto-detects each model's capabilities from its folder name. This controls whether `<think>` reasoning is enabled and what metadata is exposed via `/v1/models`.

**Auto-detected capabilities by model family:**

| Folder pattern | Detected capabilities | Thinking |
|---|---|---|
| `qwen3` (not VL) | `instruct`, `thinking` | Yes |
| `deepseek*r1` / `deepseek*r2` | `instruct`, `thinking` | Yes |
| `qwq` | `instruct`, `thinking` | Yes |
| `deepseekocr` | `instruct`, `ocr` | No |
| `qwen*vl` | `instruct`, `vl` | No |
| `phi` | `instruct` | No |
| `gemma` | `instruct` | No |
| `llama` | `instruct` | No |
| `mistral` / `mixtral` | `instruct` | No |
| `internvl` / `minicpm` | `instruct` | No |
| *(contains `instruct` or `-it`)* | `instruct` | No |
| *(no match)* | `base` | No |

Any model with a `.rknn` vision encoder file automatically gains the `vl` capability.

**Override with `model_config.json`:** Place a JSON file in the model folder to override any auto-detected value:

```json
{
  "context_length": 8192,
  "capabilities": ["thinking", "instruct"],
  "sampling": {
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "repeat_penalty": 1.1,
    "presence_penalty": 1.5
  }
}
```

| Key | Effect |
|-----|--------|
| `context_length` | Override the auto-detected context window (tokens) |
| `capabilities` | Override capability list (`instruct`, `thinking`, `vl`, etc.) |
| `sampling` | Override model-family sampling defaults for this specific model |

**Effect on thinking:** Only models with the `thinking` capability get `enable_thinking=True` on the RKLLM runtime. Non-thinking models always run with `enable_thinking=False`, preventing wasted tokens on models that don't support `<think>` blocks.

### Auto-Generated Aliases

Model folder names are converted to IDs (lowercase, hyphens). Aliases are auto-generated:

| Model ID | Auto-Aliases |
|----------|-------------|
| `qwen3-1.7b` | `qwen`, `qwen3` |
| `qwen3-4b-instruct-2507` | `qwen3-4b`, `qwen3-4b-instruct` |
| `qwen2.5-1.5b-instruct` | `qwen2`, `qwen2.5`, `qwen2.5-1.5b` |
| `xlam-1b-fc-r` | `xlam`, `xlam-1b`, `xlam-1b-fc` |
| `internvl3.5-4b` | `internvl`, `internvl3`, `internvl3.5` |

Aliases are only created when unambiguous (one model claims the alias). If two models share a prefix, that alias is skipped.

---

## Running the Server

There are two services. They run on separate ports and never share the NPU queue:

| Service | File | Port | Purpose |
|---|---|---|---|
| Main API | `api.py` | 8000 | Chat completions, VL, RAG, tool calling |
| Embed & Rerank | `embed_api.py` | 8001 | `/v1/embeddings` + `/v1/rerank` |

### Production (Recommended)

```bash
# Main chat API
gunicorn -w 1 -k gthread --threads 4 --timeout 300 -b 0.0.0.0:8000 api:app

# Embedding & reranker service (separate terminal / service unit)
gunicorn -w 1 -k gthread --threads 2 --timeout 120 -b 0.0.0.0:8001 embed_api:app
```

> **Critical:** Always use `-w 1` (single worker). The NPU can only load one model at a time per service.
>
> **Critical:** Always use `-k gthread`, NOT `-k gevent`. `rkllm_run()` is a blocking C call that freezes gevent's event loop.

### Development

```bash
python api.py        # port 8000
python embed_api.py  # port 8001
```

### Systemd Service

The setup script creates the main service automatically. For the embedding service, add a second unit:

```ini
[Unit]
Description=RKLLM API Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/RKLLM-API-Server
ExecStart=/path/to/.venv/bin/gunicorn -w 1 -k gthread --threads 4 --timeout 300 -b 0.0.0.0:8000 api:app
Restart=always
RestartSec=5
Environment=RKLLM_API_LOG_LEVEL=INFO

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/rkllm-embed.service
[Unit]
Description=RKLLM Embedding & Reranker Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/RKLLM-API-Server
ExecStart=/path/to/.venv/bin/gunicorn -w 1 -k gthread --threads 2 --timeout 120 -b 0.0.0.0:8001 embed_api:app
Restart=always
RestartSec=5
Environment=EMBED_MODEL_PATH=/home/your-user/models/qwen3-embedding-0.6b
Environment=RERANK_MODEL_PATH=/home/your-user/models/qwen3-reranker-0.6b

[Install]
WantedBy=multi-user.target
```

```bash
# Start/stop/restart
sudo systemctl start rkllm-api
sudo systemctl stop rkllm-api
sudo systemctl restart rkllm-api

# View logs
sudo journalctl -u rkllm-api -f

# Enable/disable auto-start on boot
sudo systemctl enable rkllm-api
sudo systemctl disable rkllm-api
```

---

## API Endpoints

### `GET /v1/models`

List all detected models with capabilities and context length.

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-1.7b",
      "object": "model",
      "created": 1738972800,
      "owned_by": "rkllm",
      "capabilities": ["instruct", "thinking"],
      "context_length": 4096
    },
    {
      "id": "gemma-3-4b-it",
      "object": "model",
      "created": 1738972800,
      "owned_by": "rkllm",
      "capabilities": ["instruct"],
      "context_length": 4096
    },
    {
      "id": "qwen3-vl-2b",
      "object": "model",
      "created": 1738972800,
      "owned_by": "rkllm",
      "capabilities": ["instruct", "vl"],
      "context_length": 4096
    }
  ]
}
```

**Capability values:**

| Capability | Meaning |
|---|---|
| `thinking` | Native `<think>` reasoning support (Qwen3, DeepSeek-R1, QwQ) |
| `instruct` | Instruction-tuned / chat model |
| `vl` | Vision-language model (image understanding) |
| `ocr` | Specialised for document OCR |
| `base` | Base / completion-only model (no chat template) |

### `POST /v1/chat/completions`

OpenAI-compatible chat completions (streaming and non-streaming).

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-1.7b",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-1.7b",
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
| `stop` | string or array | `null` | Stop sequences — generation halts when any string is matched |
| `max_tokens` | int | `2048` | Max completion tokens (accepted; rkllm uses model-compiled value) |
| `temperature` | float | *(ignored)* | Accepted but has no effect — rkllm uses model-compiled sampling |
| `top_p` | float | *(ignored)* | Accepted but has no effect |
| `frequency_penalty` | float | *(ignored)* | Accepted but has no effect |
| `presence_penalty` | float | *(ignored)* | Accepted but has no effect |
| `stream_options.include_usage` | bool | `false` | Include token counts in stream |

> **Note:** When `temperature`, `top_p`, `frequency_penalty`, or `presence_penalty` are sent with non-default values, the response will include an `X-RKLLM-Warning` header explaining that these are not honoured by the rkllm runtime.

### `POST /v1/completions`

Legacy text completions endpoint — wraps a `prompt` string into chat completions format and forwards to `/v1/chat/completions`. Accepts all the same parameters as chat completions plus `prompt` in place of `messages`.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-1.7b", "prompt": "The capital of France is"}'
```

### `GET /v1/models/<model_id>`

Retrieve a single model by ID or alias.

```bash
curl http://localhost:8000/v1/models/qwen3-1.7b
curl http://localhost:8000/v1/models/phi   # alias works too
```

### `POST /v1/models/select`

Pre-load a model without generating (warm-up).

```bash
curl -X POST http://localhost:8000/v1/models/select \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-1.7b"}'
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
  "current_model": "qwen3-1.7b",
  "model_loaded": true,
  "vl_model": {
    "model": "qwen3-vl-2b",
    "encoder_loaded": true,
    "llm_loaded": true
  },
  "active_request": null,
  "models_available": 4
}
```

> The `vl_model` field is `null` when no VL model is loaded.

### `GET /metrics`

Prometheus metrics endpoint (only available when `prometheus-flask-exporter` is installed).

```bash
curl http://localhost:8000/metrics
```

Exposes counters, histograms, and gauges for tokens generated, prefill/decode duration, tokens per request, queue wait time, active requests, model load time, and current model state.

**NPU hardware gauges** (polled every 5 s from `/sys/kernel/debug/rknpu/`):

```
rknpu_core_load_percent{core="core0"} 42.0
rknpu_core_load_percent{core="core1"} 38.0
rknpu_core_load_percent{core="core2"} 41.0
rknpu_freq_mhz 1000.0
rknpu_power_on 1.0
```

Reading `/sys/kernel/debug/rknpu/` requires root. A sudoers rule is deployed by `setup.sh`:
```
armbian ALL=(root) NOPASSWD: /usr/bin/cat /sys/kernel/debug/rknpu/load, ...
```

### `GET /ready`, `GET /v1/ready`

Readiness probe — returns 200 only when a model is loaded and no inference is in progress. Returns 503 otherwise. Useful for load balancers and health-gating.

```bash
curl -i http://localhost:8000/ready
# HTTP/1.1 200 OK   — model loaded and idle
# HTTP/1.1 503 Service Unavailable  — no model or busy
```

```json
{"status": "ready", "model": "qwen3-1.7b"}
```

### `POST /v1/tokenize`, `POST /tokenize`

Estimate token count for a prompt or messages array without running inference. Uses a BPE estimator tuned per model family with CJK-awareness.

```bash
# From a prompt string
curl -X POST http://localhost:8000/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-1.7b", "prompt": "Hello world"}'

# From a messages array
curl -X POST http://localhost:8000/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-1.7b", "messages": [{"role": "user", "content": "Hello"}]}'
```

```json
{
  "model": "qwen3-1.7b",
  "tokens": 8,
  "context_length": 4096,
  "remaining": 4088,
  "truncated": false
}
```

`POST /v1/count_tokens` is an alias for Anthropic SDK compatibility.

### `POST /v1/rerank`

Proxy to the embed service reranker (`embed_api.py` on port 8001). Re-scores a list of documents against a query using the Qwen3-Reranker NPU model.

```bash
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-reranker-0.6b",
    "query": "What is the capital of France?",
    "documents": ["Paris is the capital of France.", "Berlin is the capital of Germany."]
  }'
```

```json
{
  "object": "list",
  "data": [
    {"index": 0, "relevance_score": 0.97},
    {"index": 1, "relevance_score": 0.03}
  ],
  "model": "qwen3-reranker-0.6b"
}
```

### `POST /v1/messages` (Anthropic / Claude API)

Anthropic Messages API shim. Translates Claude-format requests to OpenAI chat completions internally, so any SDK targeting `api.anthropic.com` can point at this server instead.

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "qwen3-1.7b",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "Hello! How can I help you?"}],
  "model": "qwen3-1.7b",
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 5, "output_tokens": 9}
}
```

System prompts, multi-turn history, and streaming (`"stream": true`) are all supported.

### Ollama-Compatible Endpoints

The server exposes an Ollama-compatible API surface so any tool that speaks Ollama (dashboards, CLI clients, some frontends) can connect without configuration changes.

```bash
# Version probe
curl http://localhost:8000/api/version
# {"version":"0.3.0-rkllm"}

# List models
curl http://localhost:8000/api/tags

# Chat (non-streaming)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"/no_think Hello"}],"stream":false}'

# Chat (streaming — returns NDJSON)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"/no_think Hello"}],"stream":true}'

# Generate (wraps prompt as user message)
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-1.7b","prompt":"Hello"}'
```

| Ollama endpoint | Behaviour |
|---|---|
| `GET /api/version` | Returns `{"version":"0.3.0-rkllm"}` |
| `GET /api/tags` | Lists all detected models in Ollama format |
| `POST /api/chat` | Translates Ollama → OpenAI internally; streaming returns NDJSON |
| `POST /api/generate` | Wraps `prompt` as user message, delegates to `/api/chat` |
| `GET /api/show` | Returns model info stub |
| `GET /api/ps` | Returns running model stub |

### LoRA Hot-Swap Endpoints

Load and manage LoRA adapters at runtime without restarting the server. The adapter is applied to the currently loaded text model.

```bash
# List loaded adapters
curl http://localhost:8000/v1/lora
# {"adapters":[]}

# Load a LoRA adapter
curl -X POST http://localhost:8000/v1/lora/load \
  -H "Content-Type: application/json" \
  -d '{"path":"/home/armbian/models/lora/my-adapter.bin","name":"my-adapter","scale":1.0}'
# {"status":"loaded","name":"my-adapter"}

# Unregister an adapter
curl -X DELETE http://localhost:8000/v1/lora/my-adapter
# {"status":"unloaded","name":"my-adapter"}
```

> **Runtime requirement:** `rkllm_load_lora` must be exported by your `librkllmrt.so`. The endpoint returns HTTP 500 (`lora_load_failed`) if the call fails, e.g. the symbol is absent. LoRA adapter support was introduced in RKLLM SDK ≥ v1.2.3.

---

## Open WebUI Configuration

### Docker Setup

Open WebUI runs as a Docker container on the same Orange Pi (or any machine on the network). All optimized settings are hardcoded as environment variables so they persist across container recreations.

**Option A: Docker Compose (recommended)**

A `docker-compose.yml` is included in this repo with all settings pre-configured:

```bash
# Copy docker-compose.yml to the Orange Pi and start:
docker compose up -d

# Update to latest Open WebUI image:
docker compose pull && docker compose up -d

# Update with full backup + rollback (recommended):
# Uses the automated backup script — backs up DB + uploads,
# verifies integrity, pulls latest image, health checks,
# and auto-rolls back on failure. See Backup & Update section below.
/home/armbian/scripts/openwebui_full_backup_update.sh

# Full reset (deletes all data + settings, env vars re-apply):
docker compose down -v && docker compose up -d
```

**Option B: Docker Run**

Equivalent single command with all env vars:

```bash
docker run -d \
  --name open-webui \
  --restart always \
  --add-host=host.docker.internal:host-gateway \
  -p 3000:8080 \
  -v open-webui:/app/backend/data \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=sk-unused \
  -e RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
  -e RAG_RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
  -e RAG_EMBEDDING_BATCH_SIZE=10 \
  -e ENABLE_ASYNC_EMBEDDING=True \
  -e RAG_SYSTEM_CONTEXT=True \
  -e RAG_TOP_K=5 \
  -e RAG_TOP_K_RERANKER=3 \
  -e RAG_RELEVANCE_THRESHOLD=0.0 \
  -e ENABLE_RAG_HYBRID_SEARCH=True \
  -e ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS=True \
  -e RAG_HYBRID_BM25_WEIGHT=0.1 \
  -e CHUNK_SIZE=1000 \
  -e CHUNK_OVERLAP=0 \
  -e CHUNK_MIN_SIZE_TARGET=400 \
  -e ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=True \
  -e ENABLE_RETRIEVAL_QUERY_GENERATION=True \
  -e 'RAG_TEMPLATE=### Task:
Answer the user'"'"'s question using ONLY the provided context. Be thorough and detailed.

### Guidelines:
- If the answer is in the context, provide a comprehensive response with all relevant details.
- If the context doesn'"'"'t contain the answer, say so clearly.
- Respond in the same language as the user'"'"'s query.
- Do not use XML tags in your response.

<context>
{{CONTEXT}}
</context>

<user_query>
{{QUERY}}
</user_query>' \
  -e ENABLE_WEB_SEARCH=True \
  -e ENABLE_WEB_SEARCH_DEFAULT=False \
  -e WEB_SEARCH_ENGINE=searxng \
  -e SEARXNG_QUERY_URL=http://host.docker.internal:8080/search?q=<query> \
  -e WEB_SEARCH_RESULT_COUNT=5 \
  -e WEB_SEARCH_CONCURRENT_REQUESTS=1 \
  -e BYPASS_WEB_SEARCH_WEB_LOADER=True \
  -e BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True \
  -e FILE_IMAGE_COMPRESSION_WIDTH=672 \
  -e FILE_IMAGE_COMPRESSION_HEIGHT=672 \
  -e PDF_EXTRACT_IMAGES=True \
  -e ENABLE_CODE_EXECUTION=False \
  -e ENABLE_CODE_INTERPRETER=False \
  -e ENABLE_CHANNELS=True \
  -e ENABLE_MEMORIES=True \
  -e ENABLE_NOTES=True \
  -e ANONYMIZED_TELEMETRY=false \
  -e DO_NOT_TRACK=true \
  ghcr.io/open-webui/open-webui:main
```

**Environment variables explained:**

**Connection:**

| Variable | Value | Reason |
|----------|-------|--------|
| `OPENAI_API_BASE_URL` | `http://host.docker.internal:8000/v1` | Auto-connects to the RKLLM API server on the host. No manual UI setup needed — models appear immediately |
| `OPENAI_API_KEY` | `sk-unused` | The RKLLM server has no auth, but Open WebUI requires a non-empty key for OpenAI-compatible endpoints |

**RAG Pipeline:**

| Variable | Value | Reason |
|----------|-------|--------|
| `ENABLE_RETRIEVAL_QUERY_GENERATION` | `True` | Enables retrieval query generation — Open WebUI sends a query gen request, which the API server shortcircuits instantly (~0ms) with a context-enriched query instead of wasting 5-10s on inference |
| `RAG_SYSTEM_CONTEXT` | `True` | Injects retrieved document/search content into the system message instead of user message, enabling KV prefix caching for faster follow-up turns |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Best retrieval-quality embedding model that runs efficiently on ARM CPU (see [Embedding Model](#embedding-model-recommendation) below) |
| `RAG_RERANKING_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Lightweight cross-encoder reranker (22M params, ~88MB RAM) — re-scores Top K results for much better precision. Open WebUI's sigmoid normalization is specifically designed for MS MARCO models |
| `RAG_EMBEDDING_BATCH_SIZE` | `10` | Processes 10 text chunks per embedding batch — speeds up document ingestion without excessive memory use on ARM |
| `ENABLE_ASYNC_EMBEDDING` | `True` | Embeds documents asynchronously — prevents blocking the UI during file uploads |
| `ENABLE_RAG_HYBRID_SEARCH` | `True` | Combines semantic (vector) + keyword (BM25) search for significantly better retrieval than vector-only |
| `RAG_HYBRID_BM25_WEIGHT` | `0.1` | 10% keyword / 90% semantic — heavily semantic-leaning since bge-small-en-v1.5 delivers strong retrieval. Higher values dilute precision |
| `ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS` | `True` | Enriches BM25 index with document filenames, titles, and section headers — improves keyword recall for metadata-based queries |
| `ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER` | `True` | Splits documents by Markdown headers (H1-H6) first, preserving document structure. The character splitter only runs as a secondary pass on oversized sections |
| `RAG_RELEVANCE_THRESHOLD` | `0.0` | Threshold is disabled by default — the reranker handles quality filtering. Set to 0.3 to pre-filter chunks below 0.3 semantic similarity |
| `CHUNK_SIZE` | `1000` | Maximum characters per chunk. Balanced for 4K context models — large enough for coherent passages, small enough to fit multiple chunks |
| `CHUNK_OVERLAP` | `0` | Zero overlap — Chroma Research showed overlap actively hurts retrieval IoU by returning redundant tokens. With Hybrid Search, overlap is unnecessary |
| `CHUNK_MIN_SIZE_TARGET` | `400` | Merges tiny fragments (<400 chars) with neighbors, preventing low-quality micro-chunks. Works with Markdown Header Splitter to reduce chunk count by up to 90% |
| `RAG_TOP_K` | `5` | Retrieves 5 candidate chunks, then reranker narrows to best 3 (Top K Reranker). Good funnel ratio for 4K context models |
| `RAG_TOP_K_RERANKER` | `3` | Reranker keeps top 3 from the 5 retrieved chunks — only the most relevant content reaches the model |
| `RAG_TEMPLATE` | *(custom)* | Custom reading-comprehension prompt that instructs the model to answer from context only. See `docker-compose.yml` for the full template |

**Web Search (SearXNG):**

| Variable | Value | Reason |
|----------|-------|--------|
| `ENABLE_WEB_SEARCH` | `True` | Enables the web search toggle in the chat UI |
| `WEB_SEARCH_ENGINE` | `searxng` | Uses the self-hosted SearXNG instance for privacy and JSON API support |
| `SEARXNG_QUERY_URL` | `http://host.docker.internal:8080/search?q=<query>` | SearXNG instance URL. Uses `host.docker.internal` to reach the host-side SearXNG container. Change if your SearXNG is on a different host or port |
| `WEB_SEARCH_RESULT_COUNT` | `5` | Number of search results to fetch. 5 gives good coverage — the API server's quality-floor filtering drops irrelevant results automatically |
| `WEB_SEARCH_CONCURRENT_REQUESTS` | `1` | One search request at a time — prevents hammering engines simultaneously, which triggers rate limits and CAPTCHAs |
| `ENABLE_WEB_SEARCH_DEFAULT` | `False` | Globe toggle is off by default — users enable it per-query when they need live data. Reduces engine query volume by ~80% |
| `BYPASS_WEB_SEARCH_WEB_LOADER` | `True` | Uses search engine snippets instead of scraping full pages — cleaner, faster, and more reliable for small models |
| `BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL` | `True` | Sends search snippets directly to the model without embedding/retrieving — the API server builds its own optimized prompt internally |

**File Upload / Image Compression:**

| Variable | Value | Reason |
|----------|-------|--------|
| `FILE_IMAGE_COMPRESSION_WIDTH` | `672` | Compresses uploaded images to 672px width — matches the active VL encoder resolution. Other options: `448` (faster, less detail), `896` (slower, more detail) |
| `FILE_IMAGE_COMPRESSION_HEIGHT` | `672` | Compresses uploaded images to 672px height — must match the width value. See [Vision Encoder Resolution Comparison](#vision-encoder-resolution-comparison) |

**Document Processing:**

| Variable | Value | Reason |
|----------|-------|--------|
| `PDF_EXTRACT_IMAGES` | `True` | Extracts text from scanned images inside PDFs using OCR (Tesseract inside the container) |

**Code Execution:**

| Variable | Value | Reason |
|----------|-------|--------|
| `ENABLE_CODE_EXECUTION` | `False` | Small NPU models (1.7B–4B) generate unreliable code — running it wastes time or produces wrong results |
| `ENABLE_CODE_INTERPRETER` | `False` | Same reason — disable to prevent unreliable code interpretation |

**Features:**

| Variable | Value | Reason |
|----------|-------|--------|
| `ENABLE_CHANNELS` | `True` | Enables group chat channels |
| `ENABLE_MEMORIES` | `True` | Enables persistent user memories across conversations |
| `ENABLE_NOTES` | `True` | Enables the notes feature for saving snippets |

**Privacy:**

| Variable | Value | Reason |
|----------|-------|--------|
| `ANONYMIZED_TELEMETRY` | `false` | Disables telemetry (optional, recommended for privacy) |
| `DO_NOT_TRACK` | `true` | Disables tracking (optional, recommended for privacy) |

> **Port mapping:** `3000:8080` — access Open WebUI at `http://<device-ip>:3000`. Change `3000` to any port you prefer.

> **`--add-host` flag (Linux-specific, required):** On Linux, Docker does not resolve `host.docker.internal` by default — this is a Docker Desktop feature for macOS/Windows only. The `--add-host=host.docker.internal:host-gateway` flag maps it to the host's gateway IP, allowing the container to reach services running on the host (the RKLLM API server, Ollama, etc.). Without this flag, Open WebUI's default Ollama connection (`http://host.docker.internal:11434`) and any OpenAI connections using `host.docker.internal` will fail with `ClientConnectorDNSError: Cannot connect to host host.docker.internal`.

### What's Hardcoded and Why

The goal is **minimal user configuration** — a fresh install should work correctly out of the box with zero manual settings. The `docker-compose.yml` file and database setup scripts together achieve this by hardcoding every setting that can be automated.

**Three hardcoding layers are used:**

| Layer | How | Survives Container Recreate | Survives Volume Delete |
|-------|-----|:---------------------------:|:----------------------:|
| **Docker env vars** (`docker-compose.yml`) | `PersistentConfig` — env var sets the initial default, DB value takes precedence once changed in UI | Yes | Yes (re-applies) |
| **Database scripts** (`tests/set_model_prompts.py`, `tests/db_fix.py`) | Directly write to `webui.db` inside the container | Yes (data on Docker volume) | No (must re-run) |
| **Admin UI only** | No env var or script — must configure manually | Yes (data on Docker volume) | No (must redo) |

**Settings hardcoded via Docker env vars** (auto-restore on fresh install):

| Category | Settings |
|----------|----------|
| Connection | API base URL, API key |
| RAG pipeline | Embedding model, reranking model, batch size, async embedding, hybrid search, BM25 weight, enriched texts, relevance threshold, top_k, top_k_reranker, custom RAG template |
| Chunking | Chunk size, overlap, min size target, markdown header splitter |
| Document processing | PDF image extraction via OCR (`PDF_EXTRACT_IMAGES=True`) — extracts text from scanned images inside PDFs |
| Web search | Engine, SearXNG URL, result count, concurrent requests, bypass modes |
| File upload | Image compression (672×672 for VL model — matches active encoder resolution) |
| Code execution | Disabled (`ENABLE_CODE_EXECUTION=False`) — small NPU models (1.7B–4B) generate unreliable code |
| Code interpreter | Disabled (`ENABLE_CODE_INTERPRETER=False`) — same reason; wastes time or produces wrong results |
| Features | Channels, memories, notes |
| Privacy | Telemetry disabled |

**Settings hardcoded via database scripts** (must re-run after volume reset):

| Script | What it sets |
|--------|-------------|
| `tests/set_model_prompts.py` | System prompt + full capability flags on all RKLLM models (web_search, memory, vision, etc.) |
| `tests/db_fix.py` | Writes 14 settings directly to the config table — task model, compaction, query prompts, and more |
| `tests/fix_owui_config.py` | Audits the config table for invalid JSON values (empty strings, raw nulls) that crash `Config.get()` — run this if the UI breaks after any direct DB manipulation |

**Settings only configurable via Admin UI** (no env var available, must redo manually after volume reset):

| Setting | Current Value | Where to Set |
|---------|---------------|-------------|
| Web search domain filter list | `!reddit.com`, `!twitter.com`, `!x.com`, `!linkedin.com`, `!facebook.com`, `!instagram.com`, `!tripadvisor.com`, `!timeanddate.com` | Admin > Settings > Web Search > Domain Filter List |
| Model display order | qwen3-1.7b, qwen3-4b, phi-3-mini, gemma-3, deepseek-r1:7b, qwen3:8b, qwen3-vl-2b | Admin > Settings > Interface > Model Order |
| Prompt suggestions | 16 custom suggestions (study, coding, travel, etc.) | Admin > Settings > Interface > Prompt Suggestions |

**Full recovery after volume reset (`docker compose down -v`):**

```bash
# 1. Re-create container (env vars auto-apply)
docker compose up -d

# 2. Create admin account in browser, then run DB scripts:
docker cp tests/set_model_prompts.py open-webui:/tmp/
docker exec open-webui python3 /tmp/set_model_prompts.py
docker cp tests/db_fix.py open-webui:/tmp/
docker exec open-webui python3 /tmp/db_fix.py

# Optional: sanitize any invalid JSON left by manual DB edits
docker cp tests/fix_owui_config.py open-webui:/tmp/
docker exec open-webui python3 /tmp/fix_owui_config.py

# 3. Re-configure Admin UI-only settings manually (domain filters, model order, prompt suggestions)
```

> **Config DB safety note:** OpenWebUI's `config` table stores all values as JSON-encoded strings.
> If a value is written as a raw empty string `''` (not the JSON null `'null'`), every chat
> request will crash with `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` before
> the model is even called. This happens when config keys are written via direct SQL without
> proper JSON encoding. Always use `'null'` (not `''`) for unset optional string config values.

### Connection

The RKLLM API server connection is **auto-configured** via `OPENAI_API_BASE_URL` and `OPENAI_API_KEY` env vars — no manual setup needed. Models appear in the dropdown immediately after container startup.

If you need to change the connection later: **Admin > Settings > Connections** > edit the OpenAI-compatible endpoint:

| Setting | Value |
|---------|-------|
| API Base URL | `http://host.docker.internal:8000/v1` (default via env var) |
| API Key | `sk-unused` (default via env var) |

### Using Ollama Alongside (CPU Models)

Ollama can be installed on the same board and added as a **second connection** in Open WebUI. This gives you access to CPU-only models (e.g., larger models that don't have RKLLM conversions) alongside your NPU models — both appear in the model selector.

```bash
# Install Ollama on the same ARM board
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull gemma2:2b
```

**Admin > Settings > Connections:**

Add Ollama as an additional connection (don't remove the RKLLM one):

| Setting | Value |
|---------|-------|
| Ollama API URL | `http://localhost:11434` |

Both backends appear in Open WebUI's model dropdown:
- **NPU models** (fast, via this RKLLM API server) — use for everyday chat and web search
- **CPU models** (slower, via Ollama) — use for larger models or architectures not yet supported by RKLLM

> **Note:** NPU and CPU inference don't conflict — they use different hardware. You can have an NPU model loaded via this server while Ollama runs a CPU model simultaneously.

**Recommended Ollama models for RK3588:**

```bash
# DeepSeek-R1 reasoning (works on CPU, broken on NPU — see Pre-Built Models note)
ollama pull deepseek-r1:7b

# Other useful CPU models
ollama pull gemma2:2b
ollama pull phi3:3.8b
```

> **CPU models (Ollama) do NOT need the NPU-specific settings below.** The system prompt, disabled "Builtin Tools", and other restrictions apply only to small NPU models served by this RKLLM API.

### System Prompt (Pre-Configured)

The system prompt is set at the **model level** in the database — it applies to all users automatically with **zero user configuration required**. New users get the correct prompt immediately without needing to set anything up.

**Current prompt (set on all models):**

```
Today is {{CURRENT_DATE}} ({{CURRENT_WEEKDAY}}), {{CURRENT_TIME}}. This is the ONLY correct current date. Ignore any conflicting dates from search results.
```

Open WebUI resolves the template variables server-side before sending to the model:
- `{{CURRENT_DATE}}` → e.g. "February 10, 2026"
- `{{CURRENT_WEEKDAY}}` → e.g. "Tuesday"
- `{{CURRENT_TIME}}` → e.g. "14:30:00"

**How it works:** The prompt is stored in each model's `params.system` field in the database. Open WebUI injects it server-side on every request via `apply_system_prompt_to_body()`. This is enforced regardless of which user sends the message.

**To change the prompt:** Edit `SYSTEM_PROMPT` in `tests/set_model_prompts.py` and re-run:

```bash
# Edit the prompt in the script, then:
docker cp tests/set_model_prompts.py open-webui:/tmp/
docker exec open-webui python3 /tmp/set_model_prompts.py
```

**Users can optionally add their own prompt** in Settings > General > System Prompt. If set, both prompts are sent (they stack). Leave the user-level prompt **empty** to use only the model-level default.

> **Why model-level instead of user-level?** User-level prompts must be configured manually by each user — if a new user joins and forgets to set it, models won't know today's date. Model-level prompts are server-enforced, zero-configuration, and apply to everyone.

> **Why "Ignore any conflicting dates"?** Web search results often contain stale "current date" claims from cached pages (e.g. a time zone site showing "Today is October 26, 2025"). Small models (1.7B) can latch onto these and output the wrong date. This instruction, combined with the API server's date cleanup (see below), significantly reduces false dates.

### Date Accuracy for Web Search

When web search results contain date/time information, three layers work together to help the model use the correct date:

1. **System prompt** — Explicitly states the current date with "This is the ONLY correct current date"
2. **Stale date cleanup** (`api.py`) — The API server automatically strips misleading "current date is X" and "today is Y" claims from web search snippets before they reach the model. Only factual date references (like DST transition dates) are preserved.
3. **Date anchor injection** (`api.py`) — A `[Current date: February 10, 2026. Any conflicting dates below are outdated.]` line is prepended to all RAG context, placing the correct date immediately before the web content.

These are pure preprocessing steps — zero inference overhead. They significantly improve date accuracy, especially on larger models (4B+). The 1.7B model may still occasionally get confused with heavily date-laden content; for time-sensitive web search queries, prefer the 4B model.

### Web Search (SearXNG)

Web search is **auto-configured** via Docker env vars — no manual UI setup needed. The search icon appears in the chat UI immediately.

> **Prerequisite:** SearXNG must be running as a Docker container named `searxng` on the same host (see [SearXNG Configuration](#searxng-configuration) below). If your SearXNG container has a different name or IP, update the `SEARXNG_QUERY_URL` env var.

All web search settings are hardcoded (see [Docker Setup](#docker-setup) env var table above). To verify or adjust: **Admin > Settings > Web Search.**

| Setting | Value | Hardcoded |
|---------|-------|----------|
| Web Search | **ON** | `ENABLE_WEB_SEARCH=True` |
| Search Engine | `searxng` | `WEB_SEARCH_ENGINE=searxng` |
| SearXNG Query URL | `http://host.docker.internal:8080/search?q=<query>` | `SEARXNG_QUERY_URL` |
| Result Count | `5` | `WEB_SEARCH_RESULT_COUNT=5` |
| Bypass Web Loader | **ON** | `BYPASS_WEB_SEARCH_WEB_LOADER=True` |
| Bypass Embedding & Retrieval | **ON** | `BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True` |

> **Why Bypass Web Loader?** Search engine snippets are cleaner and faster than raw page scraping. Small models handle structured snippets better than noisy full-page HTML.

> **Why Bypass Embedding & Retrieval?** The API server builds its own optimized reading-comprehension prompt internally. Sending snippets directly avoids unnecessary embedding overhead.

### Embedding Model Recommendation

The embedding model determines how well Open WebUI finds the right document chunks when you ask a question.

#### Option A — NPU Embedding via embed_api.py (Best Quality)

If `embed_api.py` is running with `Qwen3-Embedding-0.6B`, point Open WebUI at it:

**Admin > Settings > Documents > Embedding Model Backend:** set to `OpenAI`  
**Embedding Model API Base URL:** `http://<your-device-ip>:8001`  
**Embedding Model:** `Qwen3-Embedding-0.6B`

This runs embeddings on the NPU — faster than CPU and semantically aligned with your Qwen3 chat models (both trained in the same embedding space), which improves retrieval quality.

> After switching, go to **Admin > Settings > Documents > Danger Zone** and click **Reindex Knowledge Base Vectors** to rebuild all embeddings.

#### Option B — CPU Embedding (Default, No Extra Service Needed)

**Recommended: `BAAI/bge-small-en-v1.5`** — set via the Docker `RAG_EMBEDDING_MODEL` env var above.

| Model | Params | MTEB Avg | Retrieval Score | RAM | Verdict |
|-------|--------|---------|----------------|-----|--------|
| **BAAI/bge-small-en-v1.5** | 33M | **62.17** | **51.68** | ~150 MB | **Best for RAG on ARM** |
| sentence-transformers/all-MiniLM-L6-v2 | 22.7M | 56.08 | 41.95 | ~90 MB | Decent but lower retrieval quality |
| minishlab/potion-base-8M | 8M | 50.54 | 31.71 | ~30 MB | Ultra-fast but poor retrieval — wrong chunks = worse answers |
| TaylorAI/bge-micro-v2 | ~4M | ~45 | ~28 | ~16 MB | Too small for reliable RAG |
| BAAI/bge-base-en-v1.5 | 109M | 63.55 | 53.25 | ~450 MB | Marginal gain, 3× more RAM — overkill for ARM |

> **Why not a faster/smaller model?** Embedding speed is not the bottleneck — embedding 5 chunks takes <100ms even with transformer models on ARM CPU. The NPU generation at 13 tok/s is the actual bottleneck. Trading retrieval quality for embedding speed is a bad trade: the model gets worse context and gives worse answers.

> **Changing the model:** If you switch embedding models, go to **Admin > Settings > Documents > Danger Zone** and click **Reindex Knowledge Base Vectors** to re-embed all existing documents with the new model.

### Document RAG Settings (Recommended for PDF/Document Upload)

**Admin > Settings > Documents:**

These settings control how Open WebUI chunks, embeds, and retrieves uploaded documents. Most are hardcoded via Docker env vars (see [Docker Setup](#docker-setup)) so they persist across container recreations. The values below are tuned for 1.5-4B parameter models on constrained ARM hardware, backed by [Chroma Research](https://research.trychroma.com/evaluating-chunking) and Open WebUI best practices:

| Setting | Value | Hardcoded | Reason |
|---------|-------|-----------|--------|
| **Text Splitter** | `Default (Character)` | default | RecursiveCharacterTextSplitter outperforms TokenTextSplitter (Chroma Research). Tokenizer-agnostic — no mismatch between tiktoken and BERT tokenizer |
| **Markdown Header Splitter** | **ON** | `ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=True` | Splits by H1-H6 headers first, preserving document structure. Character splitter only runs on oversized sections |
| **Chunk Size** | `1000` | default | ~200-250 tokens. Chroma Research found 200 tokens optimal; 1000 chars is a good character equivalent |
| **Chunk Overlap** | `0` | `CHUNK_OVERLAP=0` | Overlap actively hurts retrieval IoU (Chroma Research). With Hybrid Search + BM25, overlap is unnecessary |
| **Min Chunk Size Target** | `400` | `CHUNK_MIN_SIZE_TARGET=400` | Merges tiny fragments with neighbors, reducing chunk count by up to 90% while improving accuracy |
| **Embedding Model** | `BAAI/bge-small-en-v1.5` | `RAG_EMBEDDING_MODEL` | 62.17 MTEB avg, 33M params, ~150MB RAM (see [Embedding Model](#embedding-model-recommendation)) |
| **Reranking Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `RAG_RERANKING_MODEL` | 22M params, ~88MB RAM. Re-scores Top K candidates for much better precision. Sigmoid normalization built-in for MS MARCO models |
| **Top K** | `5` | `RAG_TOP_K=5` | Retrieves 5 chunks, reranker narrows to best 3. Good funnel ratio |
| **Top K Reranker** | `3` | default | Keeps the 3 highest-scored chunks after reranking |
| **Full Context Mode** | **OFF** | default | Injecting the entire document overflows the 4K context window |
| **Hybrid Search** | **ON** | `ENABLE_RAG_HYBRID_SEARCH=True` | Combines semantic (vector) + keyword (BM25) search |
| **Enrich Hybrid Search Text** | **ON** | `ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS=True` | Enriches BM25 index with filenames, titles, and section headers |
| **BM25 Weight** | `0.1` | `RAG_HYBRID_BM25_WEIGHT=0.1` | 10% keyword / 90% semantic — heavily semantic-leaning since bge-small-en-v1.5 delivers strong retrieval |
| **Relevance Threshold** | `0.3` | `RAG_RELEVANCE_THRESHOLD=0.3` | Filters chunks below 0.3 semantic similarity. The reranker handles final quality selection from the passing chunks |

> **RAG Template:** Use the **default template** (clear the field) — it includes inline citation support with `[id]` format and comprehensive guidelines. The API server's RAG pipeline works with the default template.

> **Image Compression:** Set to `672x672` to match the active VL encoder resolution. Available resolutions: 448 (fast/low detail), 672 (balanced, default), 896 (slow/high detail). Change via Admin > Settings > Documents or the `tests/owui_set_compression.py` script.

> **After changing settings:** Click **"Reindex Knowledge Base Vectors"** at the bottom of the Documents page to rebuild all embeddings with the new chunking/embedding configuration.

### VL / Image Upload Settings

For vision-language (VL) models like Qwen3-VL-2B to work with Open WebUI image uploads and OCR:

**Admin > Settings > Images:**

| Setting | Value | Reason |
|---------|-------|--------|
| **Image Generation (Engine)** | **OFF** (leave unset) | Do NOT enable image generation — it interferes with image upload routing. VL/OCR is handled by the chat API, not the image generation pipeline |

**Workspace > Models > Edit** (for **all** NPU models):

| Setting | Value | Reason |
|---------|-------|--------|
| **Vision** capability | **ON** | Enables the image upload button in chat for every model. The API server auto-routes image requests to the VL pipeline regardless of which model is selected — users don't need to manually switch to the VL model |
| **Builtin Tools** | **OFF** | Small NPU models (1.5B–4B) cannot do function-calling |

**How VL works:** When you upload an image in chat, Open WebUI sends it as a base64-encoded `image_url` in the OpenAI multimodal content array format. The RKLLM API server auto-detects image content and routes the request to the VL model pipeline (vision encoder → NPU embedding → LLM decoder). No special configuration is needed beyond enabling the Vision capability on the model.

**Supported image formats:** JPEG, PNG, WebP, BMP, GIF (first frame). Images are automatically resized to the VL encoder's input resolution.

> **Tip:** For OCR tasks (extracting text from screenshots, documents, photos), use the VL model (Qwen3-VL-2B). Simply upload an image in chat — it auto-routes to the VL pipeline.

### Per-Model Capabilities (Required)

**Workspace > Models > Edit > Capabilities** — configure for **each** NPU model:

| Setting | Value | Reason |
|---------|-------|--------|
| **Vision** | **ON** | Enables image upload button in chat. The API server auto-routes images to the VL model — set this on **all** models, not just the VL model |
| **Builtin Tools** | **OFF** ⚠️ | **Required.** Small NPU models (1.5B-4B) cannot do function-calling. Leaving this on injects tool-use instructions that confuse the model |
| File Context | **ON** | Enables document and search result injection |

### Interface Settings (Recommended)

**Admin > Settings > Interface > Generation Settings:**

| Setting | Value | Reason |
|---------|-------|--------|
| **Show Generation Settings** | **OFF** | The RKLLM runtime handles sampling internally. UI sliders are ignored by the API |

---

## Home Assistant Integration

The API server works as a conversation agent for **Home Assistant** via the [Extended OpenAI Conversation](https://github.com/jekalmin/extended_openai_conversation) HACS integration. This enables voice and text control of smart home devices using your local NPU.

### Setup

1. Install **HACS** in Home Assistant if not already installed
2. Install **Extended OpenAI Conversation** from HACS
3. Add the integration in **Settings > Devices & Services**:

| Field | Value |
|---|---|
| **Name** | `RKLLM Orange Pi` (or any name) |
| **API Key** | `sk-no-key-required` (any dummy value) |
| **Base Url** | `http://<ORANGE_PI_IP>:8000/v1` |
| **Skip Authentication** | Checked |
| **Api Provider** | `OpenAI` |

4. Configure the conversation agent:

| Setting | Recommended Value |
|---|---|
| **chat_model** | `qwen3-1.7b` (fast) or `qwen3-4b-instruct-2507` (smarter) |
| **Max tokens** | `2048` |
| **Temperature** | `0.3` (low for reliable device control) |
| **Top P** | `0.9` |
| **Max function calls** | `3` |
| **Context Threshold** | `3500` (for 1.7B) or `13000` (for 4B) |

5. Create a Voice Assistant in **Settings > Voice Assistants** using the new conversation agent
6. Expose entities in the **Expose** tab (start with 5-10 lights/switches)

### Model Choice

| Model | Prefill | Generate | Total | Best For |
|---|---|---|---|---|
| **qwen3-1.7b** | ~3s | ~0.5s | **~3.5s** | Simple commands, fast response |
| **qwen3-4b-instruct-2507** | ~19s | ~8s | **~27s** | Complex queries, more entities |

The 1.7B model is recommended for HA — simple commands like "turn off the living room light" work reliably and respond in under 5 seconds. Keep exposed entities under 20 to stay within the 4096 context window.

### Automatic Optimizations

The API server automatically detects Home Assistant requests and:
- **Disables thinking** — skips `<think>` reasoning tokens, cutting latency significantly
- **Detection** is based on system prompt signatures (`smart home manager of home assistant`, `available devices:`, `execute_services function`)
- This does **not** affect Open WebUI or other clients

### Limitations

- **Complex multi-step commands** may fail on 1.7B (e.g., "turn off all lights except the kitchen")
- **Entity count** affects prompt size — more entities = slower prefill and less room for response
- **No native tool calling** — relies on Extended OpenAI Conversation's prompt-based function calling

---

## SearXNG Configuration

The included `settings.yml` is optimized for Open WebUI on ARM hardware with a home/residential IP. Engine selection matters significantly — several widely-recommended engines are now broken or will block automated queries within minutes.

> **Important:** Many SearXNG engines are `disabled: true` in SearXNG's bundled defaults. Using `use_default_settings: keep_only` inherits those flags unless you explicitly override them with `disabled: false`. Without this, engines silently stay off even though they appear in the `keep_only` list.

```yaml
use_default_settings:
  engines:
    keep_only:
      - mojeek
      - brave
      - qwant
      - mwmbl
      - wikipedia
      - duckduckgo

server:
  secret_key: "change-me-to-a-random-string"
  limiter: false
  image_proxy: false
  request_timeout: 10

search:
  formats: [html, json]    # json REQUIRED for Open WebUI API access
  default_lang: en
  safe_search: 0
  autocomplete: ""

outgoing:
  request_timeout: 8.0
  max_request_timeout: 15.0
  pool_connections: 100
  pool_maxsize: 20
  retries: 2
  keepalive_expiry: 5.0

redis:
  url: redis://redis:6379/0

engines:
  - name: mojeek
    weight: 1.5
    shortcut: mj
    disabled: false           # Own independent index — lowest block rate of any engine
    categories: [general, web]
  - name: brave
    weight: 1.4
    shortcut: brave
    disabled: false           # Own index — good quality, moderate rate limits
    categories: [general, web]
  - name: qwant
    weight: 1.3
    shortcut: qw
    disabled: false           # French search engine — generally stable
    categories: [general, web]
  - name: mwmbl
    weight: 1.0
    shortcut: mw
    disabled: false           # Non-profit, bot-friendly, never blocks
    categories: [general, web]
  - name: wikipedia
    weight: 1.2
    shortcut: wp
    disabled: false           # Always reliable, no rate limiting
    categories: [general]
  - name: duckduckgo
    weight: 1.0
    shortcut: ddg
    disabled: true            # Gets CAPTCHAed quickly on automated queries — re-enable
    categories: [general, web] # manually after 24-48h of no traffic if desired
```

**Engine reliability reference:**

| Engine | Status | Notes |
|--------|--------|-------|
| **Mojeek** | ✅ Best | Own crawl index, lowest block rate, best long-term choice |
| **Brave** | ✅ Good | Own index, suspends for ~3 min under heavy load, recovers quickly |
| **Qwant** | ✅ Good | Generally stable, occasional CAPTCHA under burst load |
| **mwmbl** | ✅ Reliable | Non-profit index, never blocks, smaller coverage |
| **Wikipedia** | ✅ Always | No rate limiting, excellent for factual queries |
| DuckDuckGo | ⚠️ Fragile | CAPTCHAed within hours of automated use — keep disabled |
| **Bing** | ❌ Broken | Scraper API **retired August 11, 2025** — returns garbage results, do not use |
| Startpage | ❌ Broken | Proxies Google, CAPTCHAed immediately, 1h suspension |
| Google | ❌ Broken | Blocks headless requests, times out |
| Yahoo | ❌ Broken | Unreliable for automated queries |

> **Search query quality:** The API server's shortcircuit automatically strips conversational openers from web search queries. "Tell me a random fun fact about South Africa" becomes "facts about South Africa" before it reaches SearXNG, preventing engines from misinterpreting casual words ("fun") as topic keywords.

**OpenWebUI setting to prevent engine hammering:**

Set `WEB_SEARCH_CONCURRENT_REQUESTS=1` in your OpenWebUI environment (docker-compose or systemd). The default of 3 fires parallel queries at all engines simultaneously and triggers rate limits quickly. Also consider setting web search **off by default** in OpenWebUI settings (Admin > Settings > Web Search) so users only enable it when they genuinely need live data — this reduces query volume by ~80%.

**Installation:**
```bash
cp settings.yml ~/Downloads/searxng-docker/searxng/settings.yml
cd ~/Downloads/searxng-docker
docker compose down && docker compose up -d

# Verify engines are active:
curl -s http://localhost:8080/config | python3 -c "
import sys, json
for e in json.load(sys.stdin).get('engines', []):
    if e.get('enabled'):
        print(e['name'], '- cats:', e.get('categories'))
"

# Test search quality (should return relevant results from multiple engines):
curl -s 'http://localhost:8080/search?q=facts+about+South+Africa&format=json' | python3 -c "
import sys, json; d = json.load(sys.stdin)
from collections import Counter
print('Total:', len(d.get('results', [])), '| Engines:', dict(Counter(r.get('engine') for r in d.get('results', []))))
"
```

**Open WebUI Globe Toggle (web_search feature):**

> **Important — OpenWebUI v0.11+ requires `function_calling: legacy` on models for the web search globe toggle to work.** By default, OpenWebUI v0.11+ skips the forced RAG web search path and expects the model to call a `search_web` built-in tool via native function calling instead. Small NPU models cannot reliably do this. Setting `function_calling: legacy` forces OpenWebUI to use the RAG-style web search injection (query SearXNG → inject results as context) regardless of model capability.
>
> This is set automatically by `tests/set_model_prompts.py`. If you add new models, re-run the script.
>
> To set it manually via the API:
> ```python
> import requests, json
> BASE = "http://192.168.2.180:3000"
> HEADERS = {"Authorization": "Bearer <your-api-key>", "Content-Type": "application/json"}
> model_id = "qwen3-1.7b"
> model = requests.get(f"{BASE}/api/v1/models/model?id={model_id}", headers=HEADERS).json()
> model.setdefault("params", {})["function_calling"] = "legacy"
> requests.post(f"{BASE}/api/v1/models/model/update", headers=HEADERS, json=model)
> ```

**OpenWebUI Interface Settings for Single-NPU Hardware:**

Every background generation task (tags, follow-up suggestions, autocomplete) fires an additional LLM call on the same NPU. On a single-NPU device, these cause 503 collisions and add latency after every chat. The `docker-compose.yml` hardens these as env vars that survive container recreates:

| Setting | Value | Reason |
|---|---|---|
| `TASK_MODEL` | `qwen3-1.7b` | Fastest model handles all background tasks |
| `ENABLE_TAGS_GENERATION` | `False` | Auto-labels chats — not worth an NPU call |
| `ENABLE_FOLLOW_UP_GENERATION` | `False` | Suggests next questions — NPU waste |
| `ENABLE_AUTOCOMPLETE_GENERATION` | `False` | Fires every keystroke — most expensive |
| `ENABLE_TITLE_GENERATION` | `True` | Once per chat at natural pause — worth keeping |
| `ENABLE_CONTEXT_COMPACTION` | `True` | Essential for phi-3-mini (4K context) |
| `CONTEXT_COMPACTION_MODEL` | `qwen3-1.7b` | Prevent phi-3-mini summarising itself (4K window) |
| `CONTEXT_COMPACTION_TOKEN_THRESHOLD` | `3000` | Fires before phi-3-mini's 4K limit |
| `CONTEXT_COMPACTION_TOKEN_CAP` | `32000` | Matches largest model (Qwen3 4B) |
| `CONTEXT_COMPACTION_RETENTION_PERCENTAGE` | `30` | Leaves room for summary + new messages |

> **PersistentConfig gotcha:** OpenWebUI env vars only take effect on a **fresh volume** (`docker compose down -v`). Once a setting is saved via the admin UI, the database value takes precedence over the env var. After a fresh install the env vars apply automatically. For an existing install, settings must be updated via UI or via the DB fix script below.

**DB fix script** (run after any in-place upgrade to re-apply settings without wiping data):
```bash
docker cp tests/db_fix.py open-webui:/tmp/db_fix.py
docker exec open-webui python3 /tmp/db_fix.py
```

**Custom prompts hardcoded in `docker-compose.yml`:**

- **`QUERY_GENERATION_PROMPT_TEMPLATE`** — The default (200+ words) causes small models to add commentary before the JSON, breaking query parsing. Replaced with a minimal version that reliably returns `{ "queries": [...] }`.
- **`CONTEXT_COMPACTION_PROMPT_TEMPLATE`** — Default is good but omits technical detail preservation. Added one line: *"Preserve exact technical details: command syntax, file paths, IP addresses, model names, version numbers, error messages, and configuration values."* — critical for config/debugging sessions.

---

## Backup & Update

### Automated Weekly Backup + Update

A cron job runs every **Sunday at 3 AM** to back up Open WebUI data and pull the latest image:

```
0 3 * * 0  /home/armbian/scripts/openwebui_full_backup_update.sh >> /var/log/openwebui_backup.log 2>&1
```

**What the script does:**

1. Stops the Open WebUI container
2. Backs up `webui.db` (SQLite) and `uploads/` + `vector_db/` directories
3. Verifies backup integrity (`sqlite3 PRAGMA integrity_check` + `tar -tzf`)
4. Rotates backups — keeps the last **5** (oldest deleted automatically)
5. Pulls the latest Open WebUI image
6. Recreates the container via `docker compose up -d`
7. Runs an HTTP health check (up to 180 s timeout)
8. On failure: **auto-rollback** — restores DB + files, tags the old image, restarts with it

**Backup location:** `/home/armbian/backups/openwebui/`

**Manual trigger:**
```bash
/home/armbian/scripts/openwebui_full_backup_update.sh
```

### Rollback Support

The `docker-compose.yml` uses an environment variable for the image tag so the rollback path can override it:

```yaml
image: ${OPEN_WEBUI_IMAGE:-ghcr.io/open-webui/open-webui:main}
```

During normal operation `OPEN_WEBUI_IMAGE` is unset, so Docker pulls `main`. During rollback the script sets `OPEN_WEBUI_IMAGE` to the previously-tagged image and runs `docker compose up -d`, which picks up the old version.

### Model Enforcement

A separate hourly cron runs `enforce-owui-models.sh` to ensure the Open WebUI model list stays in sync with the RKLLM API server:

```
0 * * * *  /home/armbian/scripts/enforce-owui-models.sh >> /var/log/enforce-owui-models.log 2>&1
```

---

## RAG Pipeline

When Open WebUI performs a web search or retrieves document chunks, the results are injected into the system message as `<source>` tags (or via a custom RAG template). The server detects this and activates a specialized RAG pipeline:

### Processing Steps

1. **Detection** — `<source>` tags in the system message trigger RAG mode
2. **Extraction** — Content extracted from between `<source>...</source>` tags
3. **Web Content Cleaning** (5-pass):
   - Pass 0: Strip misleading "current date/time" claims from cached web pages
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
8. **Summarization boost** — When the query contains "summarize", "summary", "overview", or "outline", a stronger instruction is appended: *"Cover all major points, sections, and key details. Use multiple paragraphs."*

### Follow-Up Detection (3 Layers)

Open WebUI searches SearXNG with the raw user message. Short follow-ups produce garbage results:

| Layer | Trigger | Example |
|-------|---------|---------|
| Layer 0: Document-referential bypass | Query contains document-related words/phrases — **forces RAG mode** | "summarize this", "the attached file" |
| Layer 1: Word list | Exact match to known conversational words | "yes", "thanks", "tell me more" |
| Layer 2: Zero-overlap check | Zero query content-words found in reference text (w/ conversation history) | Off-topic follow-up after RAG |

Layer 0 fires first and overrides the other layers (document-referential queries always use RAG). When Layer 1 or 2 fires, RAG is skipped and the model uses normal conversation mode.

### Multi-Turn Conversation History

The server preserves full conversation context across turns within a chat session. Open WebUI sends the entire message history (system, user, and assistant messages) with each request, and the server formats them into a multi-turn prompt:

```
User: What is the capital of France?
Assistant: The capital of France is Paris.
User: What is its population?
```

The model sees all previous turns and can answer follow-up questions in context (e.g., "its" refers to Paris). With KV cache incremental mode, only the new user message is prefilled — prior turns are already in the NPU's KV cache.

### Response Cache

RAG responses are cached in an LRU cache (key: model + question hash) to avoid redundant NPU inference:

| Setting | Default | Description |
|---------|---------|-------------|
| `RAG_CACHE_TTL` | 300s | Cache lifetime |
| `RAG_CACHE_MAX_ENTRIES` | 50 | Max cached responses |

---

## Reasoning Models

Models like **Qwen3** output chain-of-thought wrapped in `<think>...</think>` tags.

The server:
- Parses these tags from the token stream using a state machine (handles tags split across chunks)
- Sends `reasoning_content` in streaming deltas (Open WebUI displays these as collapsible thinking blocks)
- Returns `reasoning_content` in non-streaming responses
- **Thinking blocks stripped from history** — prior assistant responses have `<think>...</think>` removed before re-sending to the model, per Qwen3 docs ("historical output should only include the final output part"). This saves tokens and prevents the model from mimicking its own chain-of-thought
- **Context-dependent thinking for RAG**: On small context models (< 8k), thinking is disabled via `enable_thinking = false` to save tokens for the actual answer

### `/no_think` Soft-Switch

For Qwen3 models, prefix any message with `/no_think` to suppress the thinking phase for that request. The server detects the prefix, strips it before sending to the model, and disables the `enable_thinking` flag — the model responds directly without a `<think>` block:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"/no_think What is 2+2?"}]}'
```

This is useful when you need a fast factual answer and the thinking overhead is not wanted. Works on the OpenAI endpoint, the Ollama `/api/chat` endpoint, and directly through Open WebUI (just type `/no_think` at the start of your message).

> **Note:** DeepSeek-R1 is currently **not usable on the NPU** with RKLLM Runtime v1.2.3 (produces `[PAD]` garbage tokens). Use **Qwen3-1.7B** for NPU reasoning, or run `deepseek-r1:7b` via Ollama on CPU. See the [Pre-Built Models](#pre-built-models) section for details.

---

## KV Cache Strategy

The NPU runtime maintains an internal KV cache. With `keep_history=1`, prior conversation turns are preserved, so follow-up messages only need to prefill the new tokens:

| Scenario | Strategy | Prefill Time | What's Sent |
|----------|----------|-------------|-------------|
| New conversation | `clear_kv_cache()` + `keep_history=1` | ~90ms (full) | Full prompt |
| Follow-up turn | `keep_history=1` | ~50ms (incremental) | Only new user message |
| RAG query | `keep_history=0` | ~90ms (full) | RAG context + question |
| Model switch | New model loaded | ~90ms (full) | Full prompt |

### How It Works

1. **First turn** — The server calls `rkllm_clear_kv_cache()` then sends the full prompt with `keep_history=1`. After generation, the KV cache contains the full conversation.
2. **Follow-up turns** — The server compares the list of prior user messages against what the KV cache already contains. If the lists match (same conversation, same model), only the new user message is sent with `keep_history=1`. The NPU appends to the existing KV cache.
3. **New conversation** — List mismatch triggers `rkllm_clear_kv_cache()` + full prompt resend.
4. **RAG queries** — Always use `keep_history=0` (standalone, no history needed).

This makes multi-turn conversations significantly faster — Turn 2+ take ~50ms to prefill regardless of total conversation length.

### Prompt Cache Preloading

When `PROMPT_CACHE_ENABLED = True` (default), the server saves the KV state to disk after the first inference on a freshly loaded model. On subsequent model loads (e.g., after a model swap or service restart), this cache is restored automatically, pre-populating the system prompt tokens so the first turn starts faster.

- Cache file: `<model_dir>/prompt_cache.bin` (e.g., `~/models/Qwen3-1.7B/prompt_cache.bin`)
- **Save**: Triggered on the first KV reset (new conversation) after model load, if no cache file exists yet
- **Load**: Called automatically during `load_model()` if a valid cache file is found
- **Staleness guard**: A `.meta` sidecar file records the model file's mtime + size at save time. On load, the guard re-checks — if the model file changed (update/replace), the stale cache is automatically deleted and regenerated on the next request
- Uses the RKLLM SDK's `rkllm_load_prompt_cache()` / `rkllm_release_prompt_cache()` API
- Graceful fallback: if the SDK version doesn't support the cache API, the feature is silently disabled

### Context-Aware Sliding Window

When conversation history exceeds the model's context window, the server automatically trims the oldest turns to make room. This prevents context overflow errors while preserving the most recent exchange:

- Reserves `HISTORY_CONTEXT_RESERVE` (35%) of context for the current turn + generation output
- Caps each prior assistant message at `ASSISTANT_HISTORY_CAP` (1500 chars) to prevent single long responses from dominating history
- Trims from the oldest turn first, always keeping at least the most recent user/assistant pair
- Strips `<think>...</think>` blocks from assistant history before inclusion (per Qwen3 guidelines)

---

## Configuration Reference

All configuration is at the top of `api.py`. Key values are overridable via environment variable for Docker deployments:

### Paths & Logging (env var overridable)

| Variable | Env Var | Default | Description |
|----------|---------|---------|-------------|
| `MODELS_ROOT` | `RKLLM_MODELS_ROOT` | `~/models` | Directory scanned for `.rkllm` model folders |
| `LOG_FILE` | `RKLLM_API_LOG_FILE` | `api.log` (next to api.py) | Log file path; falls back to stderr-only if not writable |
| `RKLLM_LIB_PATH` | `RKLLM_LIB_PATH` | auto-detected | Path to `librkllmrt.so` |
| `RKNN_LIB_PATH` | `RKNN_LIB_PATH` | auto-detected | Path to `librknnrt.so` (VL models) |

### Timeouts

| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATION_TIMEOUT` | 600s | Max total generation time |
| `FIRST_TOKEN_TIMEOUT` | 300s | Max wait for first token (includes prefill) |
| `FALLBACK_SILENCE` | 20s | Max silence between tokens after first |
| `REPETITION_WINDOW` | 200 chars | Sliding window size for repetition loop detection |
| `REPETITION_MAX_HITS` | 2 | Abort after this many repeated windows detected |

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

### VL (Vision-Language) Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `VL_RAG_CONTEXT_CAP` | 2000 chars | Max RAG reference text in VL multi-turn prompts |
| `VL_ASSISTANT_HISTORY_CAP` | 500 chars | Max chars per prior assistant answer in VL prompts |

### History & Sliding Window

| Variable | Default | Description |
|----------|---------|-------------|
| `HISTORY_CONTEXT_RESERVE` | 0.35 | Fraction of context reserved for current turn + output |
| `ASSISTANT_HISTORY_CAP` | 1500 chars | Max chars per prior assistant answer in text prompts |

### Prompt Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMPT_CACHE_ENABLED` | `True` | Enable/disable KV state save/load on model init |

### Sampling Profiles

Model-aware sampling is configured via `MODEL_SAMPLING_PROFILES` in `api.py`. Each model family gets tuned defaults:

| Family | top_k | top_p | temp | repeat_penalty | presence_penalty |
|--------|-------|-------|------|----------------|------------------|
| `qwen3` | 20 | 0.8 | 0.7 | 1.1 | 1.5 |
| `gemma` | 40 | 0.95 | 0.7 | 1.1 | 0.0 |
| `phi` | 40 | 0.9 | 0.6 | 1.1 | 0.0 |
| `deepseek` | 20 | 0.9 | 0.6 | 1.1 | 1.0 |

**Override per model**: Add a `"sampling"` key in `model_config.json` inside the model directory:

```json
{
  "sampling": {
    "top_k": 30,
    "temperature": 0.5
  }
}
```

Only specified fields are overridden; unset fields use the family profile defaults.

### Process Management

| Variable | Default | Description |
|----------|---------|-------------|
| `REQUEST_STALE_TIMEOUT` | 180s | Auto-clear tracked request after this idle time |
| `MONITOR_INTERVAL` | 10s | Health check / idle monitoring frequency |
| `IDLE_UNLOAD_TIMEOUT` | 300s | Auto-unload text model after idle (0 to disable) |
| `VL_IDLE_UNLOAD_TIMEOUT` | 300s | Auto-unload VL model after idle (0 to disable) |
| `RKLLM_QUEUE_TIMEOUT` | 240s | How long a queued request waits for the NPU slot |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `RKLLM_LIB_PATH` | Path to `librkllmrt.so` (auto-detected from `/usr/lib/` by default) |
| `RKNN_LIB_PATH` | Path to `librknnrt.so` for VL vision encoder (auto-detected from `/usr/lib/` by default) |
| `RKLLM_API_LOG_LEVEL` | Python API log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Logging

Logs are written to both **stderr** and a rotating log file (`api.log` in the script directory):

- **Max file size:** 10 MB
- **Backup count:** 3 rotated files
- **Default level:** `DEBUG` (set `RKLLM_API_LOG_LEVEL=INFO` for production)

### Log Examples

```
2026-02-08 17:45:12 [INFO] Detected: qwen3-1.7b (context=4096)
2026-02-08 17:45:12 [INFO] Models: ['qwen3-1.7b']
2026-02-08 17:45:12 [INFO] Aliases: {'qwen': 'qwen3-1.7b', 'qwen3': 'qwen3-1.7b'}
2026-02-08 17:45:30 [INFO] >>> NEW REQUEST chatcmpl-a1b2c3d4e5f6
2026-02-08 17:45:30 [INFO] Resolved alias 'qwen' -> 'qwen3-1.7b'
2026-02-08 17:45:30 [INFO] Loading model: qwen3-1.7b
2026-02-08 17:45:33 [INFO] Model loaded in 3.2s
2026-02-08 17:45:33 [DEBUG] KV incremental: sending only new user message (hash match)
2026-02-08 17:45:33 [DEBUG] First token in 0.05s
2026-02-08 17:45:45 [INFO] Request ENDED: chatcmpl-a1b2c3d4e5f6
```

---

## Security

> **This server has NO authentication.** It is designed to run on a trusted local network.

- Binds to `0.0.0.0:8000` — accessible from all network interfaces
- No API key validation (any non-empty string works for Open WebUI)
- Request body limited to 50 MB
- **Do NOT expose directly to the public internet**
- Place behind a reverse proxy (nginx, Caddy) with authentication if external access is needed

---

## Troubleshooting

### "Model not found"
- Ensure the `.rkllm` file is inside a subfolder of `~/models/` (not directly in `~/models/`)
- Check folder naming — spaces become hyphens, underscores become hyphens
- Run `curl http://localhost:8000/v1/models` to see detected models

### "Failed to load model"
- Check that `librkllmrt.so` is in `/usr/lib/`: `ldconfig -p | grep rkllm`
- Verify NPU driver is loaded: `dmesg | grep -i npu`
- Check `api.log` for init failure messages — may indicate corrupt `.rkllm` file or version mismatch

### VL model fails with "max_context must be less than max_context_limit"
The full error in `api.log`: `E rkllm: max_context[8192] must be less than the model's max_context_limit[4096]`

The `context_length` in `model_config.json` exceeds what the model was compiled with. Fix:
```json
{ "context_length": 4096 }
```
The compiled context limit is reported in the load log line: `rkllm-toolkit version: ..., max_context_limit: 4096`

### "Another request is currently being processed" (503)
- NPU is single-task — only one request at a time
- Previous request may be stuck — check `/health` endpoint
- Stale requests auto-clear after 180s idle

### Streaming stops mid-response
- Check `FALLBACK_SILENCE` timeout (default 20s) — increase if model is slow
- Large prompts near context limit may cause long prefill — increase `FIRST_TOKEN_TIMEOUT` (default 300s)
- Check `/health` endpoint for status

### Server freezes on requests
- Ensure you are using `-k gthread`, **not** `-k gevent`. `rkllm_run()` is a blocking C call that freezes gevent's event loop
- Check `gunicorn` command: `gunicorn -w 1 -k gthread --threads 4 --timeout 300 -b 0.0.0.0:8000 api:app`

### RAG returns irrelevant answers
- Verify SearXNG is returning JSON: add `json` to `search.formats` in SearXNG settings
- Check if "Bypass Web Loader" is **ON** in Open WebUI
- Set `RAG_QUALITY_FLOOR_THRESHOLD` higher to drop poor search results
- Check logs for "RAG SKIP" and "Quality floor triggered" messages

### High memory usage
- Set `IDLE_UNLOAD_TIMEOUT` to auto-unload after idle periods
- Use `/v1/models/unload` to manually free NPU memory
- Smaller quantized models (W4A16) use less memory

---

## Testing

Four test suites verify every code path against a live server — from unit-level parsing to full end-to-end integration across all models.

### Diagnostic Test (`tests/diagnostic_test.py`)

Section-by-section diagnostic covering 17 areas of the codebase — **91 tests total**. Designed for copy-paste output analysis.

```bash
python tests/diagnostic_test.py                # Run all 17 sections
python tests/diagnostic_test.py --skip-vl      # Skip VL tests (faster)
python tests/diagnostic_test.py --section 4    # Run only section 4
```

| Section | Coverage |
|---|---|
| 1 | Server connectivity & health endpoint structure |
| 2 | Model detection, `/v1/models` listing, response format |
| 3 | Alias generation & model name resolution |
| 4 | Error handling: bad body, empty messages, invalid types, bad base64 |
| 5 | Text generation (non-streaming): response structure, usage stats |
| 6 | Text generation (streaming): SSE format, chunk structure, `include_usage` |
| 7 | Think tag parsing (`reasoning_content` in SSE) |
| 8 | KV cache tracking & incremental mode (multi-turn memory) |
| 9 | Model select, unload, switch, idle state |
| 10 | Concurrent request rejection (single-NPU guard) |
| 11 | RAG pipeline: `<source>` extraction, boilerplate cleaning, skip detection |
| 12 | RAG response cache (generate vs cached timing) |
| 13 | Content normalization (multimodal arrays with text only) |
| 14 | VL auto-routing, image processing, streaming, model name in response |
| 15 | Text-after-VL (dual-model isolation) |
| 16 | Route variants (`/chat/completions` vs `/v1/...`), edge cases |
| 17 | Final system state consistency |

### Integration Test (`tests/vl_test.py`)

Focused integration tests across 17 categories — **68 assertions**. Tests text generation, VL multimodal, streaming, error handling, model lifecycle, and concurrent rejection.

```bash
python tests/vl_test.py all          # Run all tests
python tests/vl_test.py complete     # Non-streaming tests only
python tests/vl_test.py stream       # Streaming tests only
```

### Test Results (Orange Pi 5 Plus, March 2026)

| Suite | Total | Pass | Fail | Warn | Time |
|---|---|---|---|---|---|
| `tests/diagnostic_test.py` | 91 | 91 | 0 | 0 | ~12 min |
| `tests/vl_test.py` | 68 | 68 | 0 | 0 | ~5 min |
| `tests/e2e_test.py` | 78 | 78 | 0 | 0 | ~11 min |
| `tests/deep_diagnostic.py` | 84 | 84 | 0 | 1 | ~7 min |
| `tests/realworld_smoke.py` | 47 | 47 | 0 | 0 | ~4 min |
| `tests/test_features.py` | 27 | 27 | 0 | 0 | ~4 min |

### Real-World Smoke Test (`tests/realworld_smoke.py`)

Ad-hoc real-world smoke test with 15 scenarios — **47 checks**. Exercises the API with natural prompts to verify end-to-end behavior including Q&A, streaming, reasoning, multi-turn context, RAG, shortcircuits, date awareness, Home Assistant, model switching, concurrent rejection, long output, and error handling.

```bash
python tests/realworld_smoke.py              # Run all 15 scenarios
```

### End-to-End Test (`tests/e2e_test.py`)

Full integration test that exercises every model with real inference — **78 checks across 9 sections**, run against all 4 text models. Covers streaming, non-streaming, shortcircuits, RAG, cache, web search, KV multi-turn, prompt building, Open WebUI database config, and API compliance. WebUI/SearXNG URLs are auto-derived from `RKLLM_API` host.

```bash
python tests/e2e_test.py                # Run all 9 sections against all models
python tests/e2e_test.py --section 3    # Run only section 3
python tests/e2e_test.py --fast          # Skip slow models (gemma, phi)
```

| Section | Coverage |
|---|---|
| 1 | Per-model text generation (streaming + non-streaming, all 4 models) |
| 2 | Shortcircuit detection (title gen, tag gen, query gen) |
| 3 | RAG pipeline with real document context |
| 4 | RAG response cache (hit/miss timing) |
| 5 | Web search flow (SearXNG integration) |
| 6 | KV cache multi-turn memory |
| 7 | Prompt building & detection (date, HA, summarization) |
| 8 | Open WebUI database config |
| 9 | API compliance & edge cases (alias resolution, stream_options) |

### Deep Diagnostic Test (`tests/deep_diagnostic.py`)

Targeted deep-dive into 12 under-tested areas identified via gap analysis — **84 checks** covering protocol compliance, edge cases, and stress scenarios.

```bash
python tests/deep_diagnostic.py                # Run all 12 sections
python tests/deep_diagnostic.py --section 4    # Run only section 4
```

| Section | Coverage |
|---|---|
| 1 | SSE stream format strict compliance (Content-Type, JSON validity, [DONE], role, finish_reason) |
| 2 | Concurrent request rejection (503 on second request, recovery after) |
| 3 | Model hot-swap correctness (swap timing, response model field, health state) |
| 4 | Unicode / special character handling (emoji, CJK, Arabic, HTML entities in stream) |
| 5 | ThinkTagParser edge cases (partial tags, char-by-char, multi-block, regex parity, 10K blocks) |
| 6 | CORS headers & preflight (OPTIONS, Access-Control-Allow-Origin, methods) |
| 7 | Token usage accuracy (prompt+completion=total, streaming include_usage, ranges) |
| 8 | Select / unload endpoints (invalid model, double unload, invalid JSON) |
| 9 | VL / OCR pipeline (image inference, streaming, invalid base64, URL-based image rejection) |
| 10 | Context overflow / large input (22K chars, 50-turn conversation, server recovery) |
| 11 | Message normalization (list content, integer coercion, null content, non-dict filtering) |
| 12 | Shortcircuit streaming SSE compliance (chunk count, usage block, system_fingerprint) |

All suites default to `http://localhost:8000`. To target a remote server, set the `RKLLM_API` environment variable:

```bash
RKLLM_API=http://192.168.x.x:8000 python tests/diagnostic_test.py
```

### OpenWebUI Feature Test (`tests/test_features.py`)

End-to-end feature verification suite for OpenWebUI + RKLLM — **27 checks across 8 sections**. Connects directly to the OpenWebUI API and verifies every major capability: settings correctness, memory (ChromaDB store/retrieve/isolation), RAG (file upload → chunking → embedding → retrieval), web search (SearXNG), tool calling, and multi-turn context continuity.

```bash
python tests/test_features.py                              # Run against default host
python tests/test_features.py --host 192.168.2.180 --port 3000
```

| Section | Coverage |
|---|---|
| SETTINGS — task config | Task model, tag gen, follow-up gen, autocomplete, title gen, query gen prompts |
| SETTINGS — context compaction | Compaction model set, health check |
| MEMORY | Add memory via API, DB persistence, context injection, isolation when feature OFF |
| RAG | File upload with sync processing, file ID assigned, sources returned, answer retrieved, answer absent without file |
| WEB SEARCH | SearXNG correct answer, model answers from training without search |
| TOOL CALLING | `finish_reason=tool_calls`, correct function name, arguments extracted |
| CONTEXT CONTINUITY | Code remembered across turns |
| LOGS | Health endpoint, all 5 models listed |

> **Requires:** OpenWebUI running with API key auth enabled and `db_fix.py` + `set_model_prompts.py` applied.
>
> **Note on 503 errors in logs:** During a test run, OpenWebUI logs will show `HTTP 503 Another request is currently being processed` entries. These are **expected** — the retry logic in the test client deliberately re-sends requests when the NPU is busy. All retries resolve cleanly; no real errors occur.
>
> **RAG API format:** Files must be passed in the top-level `files` key of the request body, not nested inside `metadata.files`. OpenWebUI's `main.py` reads `body.get('files')` before reconstructing `body['metadata']`, so anything sent in `body.metadata.files` is silently discarded.

### OpenWebUI Integration Test (`tests/test_openwebui.py`)

End-to-end integration test that verifies the full stack through the OpenWebUI API layer — **24 checks across 9 sections**. Connects to OpenWebUI at `http://192.168.2.180:3000` using an API key and exercises every local model through the `/openai/chat/completions` proxy.

```bash
python tests/test_openwebui.py                          # Run against default host
python tests/test_openwebui.py --host 192.168.2.180 --port 3000
```

| Section | Coverage |
|---|---|
| Health | `/health` endpoint responds |
| Models | All 5 local models listed via `/api/models` |
| Basic chat | PONG response from all 5 models |
| Streaming | Multi-chunk SSE streaming |
| VL model | Image + text response (inline base64 PNG) |
| RAG | Document upload (`/api/v1/files/`), context retrieval, grounded answer |
| Tool calling | `finish_reason=tool_calls`, correct function name and arguments |
| System prompt | Prompt respected in response |
| Multi-turn | Name remembered across turns |
| Error handling | 4xx on bad model and empty messages |

> **Requires:** OpenWebUI running with API key auth enabled. The test uses the admin API key stored in `webui.db`.

### Benchmark Tool (`tests/benchmark_test.py`)

Automated NPU benchmark tool that measures cold load time, warm TTFT, generation speed (tok/s), and NPU memory usage. Fetches real perf stats from the server log via SSH.

```bash
python tests/benchmark_test.py                                     # Benchmark all models
python tests/benchmark_test.py --models qwen3-1.7b phi-3-mini-4k-instruct  # Specific models only
python tests/benchmark_test.py --skip-vl                            # Skip VL model tests
python tests/benchmark_test.py --runs 3                             # Average over 3 runs
python tests/benchmark_test.py --remote-log                         # Fetch NPU perf via SSH
```

Results are saved to `tests/benchmark_results.json` and printed as formatted markdown tables.

### DB Maintenance Scripts (`tests/`)

Three scripts write directly to the OpenWebUI SQLite database inside the container. Run them after initial setup and after any volume reset:

| Script | Purpose | When to run |
|--------|---------|-------------|
| `tests/set_model_prompts.py` | Sets system prompt + capability flags (`web_search`, `memory`, `file_context`, `function_calling=legacy`, `vision`) on all 5 RKLLM models | After setup or volume reset |
| `tests/db_fix.py` | Writes 14 config keys (task model, context compaction settings, query-gen prompts, memory enable) | After setup or volume reset |
| `tests/fix_owui_config.py` | Audits all config rows for invalid JSON values (empty strings that crash `Config.get()`). Safe to run any time; dry-run with `--dry-run` | After any direct DB manipulation |

```bash
# Run all three inside the container:
docker cp tests/set_model_prompts.py open-webui:/tmp/ && docker exec open-webui python3 /tmp/set_model_prompts.py
docker cp tests/db_fix.py open-webui:/tmp/ && docker exec open-webui python3 /tmp/db_fix.py
docker cp tests/fix_owui_config.py open-webui:/tmp/ && docker exec open-webui python3 /tmp/fix_owui_config.py
```

### Test Environment Variables

All test files respect environment variables for connection config:

| Variable | Used By | Default |
|----------|---------|---------|
| `RKLLM_API` | `realworld_smoke.py`, `e2e_test.py`, `diagnostic_test.py` | `http://localhost:8000` |
| `RKLLM_HOST` | `realworld_smoke.py` | `192.168.2.180` |
| `RKLLM_PORT` | `realworld_smoke.py` | `8000` |
| `OWUI_HOST` | `test_features.py`, `test_openwebui.py` | `192.168.2.180` |
| `OWUI_PORT` | `test_features.py`, `test_openwebui.py` | `3000` |
| `OWUI_API_KEY` | `test_features.py`, `test_openwebui.py` | (dev key — set this on any other install) |

---

## OpenWebUI Admin Utilities (`owui_tools/`)

A collection of standalone scripts for managing and diagnosing the OpenWebUI instance. These operate via the OpenWebUI API or direct DB access. They are not part of the automated test suite.

| Script | What it does |
|--------|-------------|
| `check_owui_config.py` | Print current OpenWebUI config values via the API |
| `check_owui_config2.py` | Updated version using current OWUI DB schema |
| `check_owui_models.py` | List all models visible via `/api/models` |
| `dump_owui_config.py` | Full JSON dump of the config table |
| `dump_owui_connections.py` | Dump all API connection entries |
| `dump_owui_models_quick.py` | Fast model listing with capability flags |
| `dump_owui_settings.py` | Dump all OpenWebUI settings |
| `dump_owui_apiconfigs2.py` | Dump OpenAI-compatible API connection configs (current schema) |
| `diag_config.py` | Diagnostic dump highlighting unusual config values |
| `fix_capabilities.py` | Fix/re-apply capability flags on RKLLM models via the API |
| `fix_owui_models.py` | Fix stale model metadata in the DB |
| `clear_model_filters.py` | Clear model filter settings |
| `enforce_free_models.py` / `set_owui_free_models.py` | Set free-access flags on all models |
| `set_owui_model_filters.py` | Configure model filter allowlist/blocklist |
| `owui_set_compression.py` | Set image compression width/height (matched to VL model input size) |
| `remove_openrouter.py` | Remove any OpenRouter connection entries |
| `remove_stale_models.py` | Remove stale model entries left by a previous RKLLM API instance |
| `enforce-owui-models.sh` / `test_providers.sh` | Shell wrappers for bulk enforcement and provider connectivity testing |

> **Note:** `owui_tools/set_model_prompts.py` is an older copy of `tests/set_model_prompts.py`. Use the `tests/` version — it is kept up to date.

---

## File Structure

```
RKLLM-API-Server/
├── api.py                          # Main API server — chat, VL, RAG, tools (port 8000)
├── embed_api.py                    # Embedding + reranker service (port 8001)
├── FUTURE_FEATURES.md              # Planned improvements with effort/impact notes
├── Dockerfile                      # Container image (ARM64, python:3.12-slim)
├── gunicorn.config.py              # Gunicorn config (env-var driven)
├── healthcheck.py                  # Docker HEALTHCHECK script
├── docker-compose.yml              # Open WebUI Docker config (all settings hardcoded)
├── setup.sh                        # Zero-config installer (includes Hermes Agent setup)
├── settings.yml                    # SearXNG configuration for Open WebUI
├── hermes/
│   ├── config.yaml                 # Hermes Agent config template (providers, fallback chain, API server)
│   └── .env.example                # Environment variable template (API keys for cloud providers)
├── README.md                       # This file
├── .github/workflows/docker.yml    # CI: builds and pushes ARM64 image to GHCR on main push
├── tests/
│   ├── diagnostic_test.py          # Section-by-section diagnostic (17 sections, 91 tests)
│   ├── e2e_test.py                 # End-to-end integration (9 sections, 78 checks, all models)
│   ├── deep_diagnostic.py          # Deep diagnostic (12 sections, 84 checks, edge cases)
│   ├── vl_test.py                  # Integration test suite (17 categories, 68 tests)
│   ├── realworld_smoke.py           # Real-world smoke test (15 scenarios, 47 checks)
│   ├── test_openwebui.py           # OpenWebUI end-to-end integration test (24 checks)
│   ├── benchmark_test.py           # NPU model benchmark tool (tok/s, TTFT, memory)
│   ├── benchmark_results.json      # Latest benchmark results
│   ├── set_model_prompts.py        # Set system prompts + capabilities on all RKLLM models (DB script)
│   ├── db_fix.py                   # Write 14 OpenWebUI settings directly to SQLite (DB script)
│   ├── test_features.py            # OpenWebUI feature test — 27 checks (memory, RAG, web search, tools)
│   ├── fix_owui_config.py          # Audit/fix invalid JSON in config table (run after direct DB edits)
│   ├── test_capabilities.py        # Quick capability detection and thinking-control verification
│   ├── vl_multi_image_test.py      # Multi-image VL model integration test
│   └── vl_multiturn_test.py        # VL multi-turn context + RAG integration test
├── owui_tools/                     # Standalone OpenWebUI admin utilities (see below)
│   ├── check_owui_config.py        # Print current OpenWebUI config via API (legacy schema)
│   ├── check_owui_config2.py       # Updated version using current OWUI DB schema
│   ├── check_owui_models.py        # List models visible from OpenWebUI API
│   ├── clear_model_filters.py      # Clear model filter settings
│   ├── diag_config.py              # Diagnostic dump of all config keys
│   ├── dump_owui_config.py         # Dump full config JSON
│   ├── dump_owui_connections.py    # Dump model connection entries
│   ├── dump_owui_models_quick.py   # Fast model list
│   ├── dump_owui_settings.py       # Dump all settings
│   ├── dump_owui_apiconfigs.py     # Dump API connection configs (legacy schema)
│   ├── dump_owui_apiconfigs2.py    # Updated version using current OWUI DB schema
│   ├── enforce_free_models.py      # Set free-access flags on models
│   ├── fix_capabilities.py         # Fix capability flags on RKLLM models via API
│   ├── fix_owui_models.py          # Fix stale model metadata in DB
│   ├── owui_set_compression.py     # Set image compression dimensions
│   ├── remove_openrouter.py        # Remove OpenRouter connection entries
│   ├── remove_stale_models.py      # Remove stale model entries
│   ├── set_model_prompts.py        # (older copy — use tests/set_model_prompts.py instead)
│   ├── set_owui_free_models.py     # Set free-access flags
│   ├── set_owui_model_filters.py   # Configure model filter settings
│   ├── enforce-owui-models.sh      # Shell wrapper to enforce model config
│   └── test_providers.sh           # Test provider connectivity
├── archive/
│   ├── api_v1_subprocess.py        # Original subprocess version (archived)
│   └── CTYPES_MIGRATION_PLAN.md    # V1→V2 migration planning document
└── .gitignore
```

---

## V1 (Subprocess) vs V2 (ctypes) — Why We Migrated

The original server (`archive/api_v1_subprocess.py`) worked by spawning a separate C++ binary and communicating via stdin/stdout pipes. While functional, this architecture had significant limitations. The current version (`api.py`) uses direct ctypes bindings to the shared library, eliminating the process boundary entirely.

### Architecture Comparison

| Aspect | V1 — Subprocess | V2 — ctypes (current) |
|--------|-----------------|----------------------|
| **NPU communication** | Pipes stdin/stdout to a C++ binary | Direct C library calls via ctypes |
| **Token delivery** | Parse stdout line-by-line | C callback pushes to `queue.Queue` |
| **KV cache** | Lost on every turn (binary restarts) | Preserved across turns (`keep_history=1`) |
| **Prefill (Turn 2+)** | ~500ms (re-process entire conversation) | ~50ms (only new user message) |
| **Abort / cancel** | `SIGKILL` the process | `rkllm_abort()` — clean, instant |
| **Performance stats** | Parsed from stdout text | Native `RKLLMResult.perf` struct |
| **Thinking mode toggle** | Append `/no_think` to prompt text | `RKLLMInput.enable_thinking` flag |
| **Error handling** | Detect process crash / timeout | C return codes + error callback state |
| **Process management** | ~500 lines (spawn, monitor, kill, restart) | 0 lines (no process to manage) |
| **VL / multimodal** | Not supported | Dual-model architecture with RKNN vision encoder |
| **Code size** | 2682 lines | ~3700 lines (text + VL + RAG) |

### Why the Change Matters

**The biggest win is KV cache retention.** In the subprocess architecture, every turn killed and restarted the C++ binary, destroying the NPU's key-value cache. This meant the model had to re-prefill the entire conversation history from scratch on every single message — growing linearly with conversation length.

With ctypes, the library stays loaded in-process. The KV cache persists between calls. On a 10-turn conversation, Turn 1 takes ~90ms to prefill. All subsequent turns take ~50ms regardless of conversation length, because only the new message is processed.

**Performance impact (measured on Orange Pi 5 Plus, Qwen3-1.7B):**

| Metric | V1 (Subprocess) | V2 (ctypes) | Improvement |
|--------|-----------------|-------------|-------------|
| Turn 1 prefill | ~90ms | ~90ms | Same |
| Turn 2 prefill | ~500ms | ~50ms | **10x faster** |
| Turn 5 prefill | ~1200ms | ~50ms | **24x faster** |
| Turn 10 prefill | ~2000ms+ | ~50ms | **40x faster** |
| Model switch | ~5s (kill + restart + reload) | ~3s (destroy + init) | ~40% faster |
| Cancel generation | ~1s (SIGKILL + wait) | instant (`rkllm_abort()`) | Near-instant |

### V1 Subprocess Code (Archived)

The original subprocess version is preserved at [`archive/api_v1_subprocess.py`](archive/api_v1_subprocess.py) (2682 lines, fully functional). You can also access it via the git tag:

```bash
# View the last working subprocess version
git checkout v1.0-subprocess -- api.py

# Return to current ctypes version
git checkout main -- api.py
```

The V1 code may be useful as a reference if:
- You need to run on a system where ctypes binding is not possible
- You want to see how stdout parsing / process management was implemented
- You're porting to a different inference runtime that only provides a CLI binary

---

## Tested Hardware

| Board | RAM | NPU Driver | Runtime | Status |
|-------|-----|-----------|---------|--------|
| Orange Pi 5 Plus | 16 GB | 0.9.8 | v1.2.3 | Fully tested, production use |

## Tested Models

### Text Models

| Model | Quantization | Context | File Size | Speed | Status |
|-------|-------------|---------|-----------|-------|--------|
| Qwen3-1.7B | W8A8 | 4K | ~1.7 GB | **13.0 tok/s** avg | Fully benchmarked |
| Phi-3-Mini-4K-Instruct | W8A8 | 4K | ~3.8 GB | **6.8 tok/s** avg | Fully benchmarked |
| Qwen3-4B-Instruct | W8A8 | 16K | ~4 GB | ~6 tok/s | Tested |
| Gemma-3-4B-IT | W8A8 | 4K | ~4 GB | ~6 tok/s | Tested |
| **Qwen2.5-1.5B-Instruct** | W8A8 | 8K | ~2.1 GB | **~14 tok/s** | Tested — fast general chat, Hermes local fallback |
| **xLAM-1b-fc-r** | W8A8 | 8K | ~1.6 GB | **~16 tok/s** | Tested — best sub-2B tool-calling model (78.94% BFCL v3) |

> Pre-built RKLLM models available on HuggingFace: [Qwen3-1.7B-RKLLM-v1.2.3](https://huggingface.co/GatekeeperZA/Qwen3-1.7B-RKLLM-v1.2.3) · [Phi-3-mini-4k-instruct-w8a8](https://huggingface.co/GatekeeperZA/Phi-3-mini-4k-instruct-w8a8) · [Qwen3-VL-2B-Instruct-RKLLM-v1.2.3](https://huggingface.co/GatekeeperZA/Qwen3-VL-2B-Instruct-RKLLM-v1.2.3) · [Qwen2.5-1.5B-Instruct-RKLLM-v1.2.3](https://huggingface.co/GatekeeperZA/Qwen2.5-1.5B-Instruct-RKLLM-v1.2.3) · [xLAM-1b-fc-r-RKLLM-v1.2.3](https://huggingface.co/GatekeeperZA/xLAM-1b-fc-r-RKLLM-v1.2.3)

### VL (Vision-Language) Models

| Model | Quantization | Encoder Res | Decode Speed | Encoder Time | Peak RAM | Status |
|-------|-------------|-------------|-------------|-------------|----------|--------|
| **Qwen3-VL-2B** | W8A8 | **672×672** | ~15 tok/s | ~4s | ~6.5 GB | **Active (recommended)** |
| Qwen3-VL-2B | W8A8 | 448×448 | ~15 tok/s | ~2s | ~5.5 GB | Available (default export) |
| Qwen3-VL-2B | W8A8 | 896×896 | ~15 tok/s | ~12s | ~8.5 GB | Available (high-detail) |
| **Qwen3-VL-4B** | W8A8 | 448×448 | ~6 tok/s | ~2s | ~10 GB | **Tested — higher reasoning quality** |
| InternVL3.5-2B | W8A8 | 448×448 | ~12.1 tok/s | ~2.0s | ~3.0 GB | **Tested — poor OCR accuracy** |
| DeepSeekOCR-3B | W8A8 | 448×448 | ~31.8 tok/s | ~2.1s | ~3.0 GB | **Tested — severe hallucination** |
| Qwen2.5-VL-3B | W8A8 | 392×392 | ~8.7 tok/s | ~2.9s | ~5.3 GB | Supported (lower resolution) |
| Qwen2-VL-2B | W8A8 | 392×392 | ~16.6 tok/s | ~3.3s | ~3.0 GB | Supported |
| InternVL3-1B | W8A8 | 448×448 | ~TBD | ~TBD | ~TBD | Supported |
| MiniCPM-V-2.6 | W8A8 | 448×448 | ~TBD | ~TBD | ~TBD | Supported |

> **Qwen3-VL-2B is the recommended VL model** with the vision encoder re-exported at **672×672** for 2.25× more visual detail than the default 448×448. Three encoder resolutions (448/672/896) are available — see [Vision Encoder Resolution Comparison](#vision-encoder-resolution-comparison). To switch encoders, rename the `.rknn` files (only one should have the `.rknn` extension; others use `.rknn.alt`).
>
> **Qwen3-VL-4B memory note:** The 4B VL model requires ~10 GB of NPU IOVA (IOMMU domain 1 on RK3588). If a text model is loaded when the first VL-4B image request arrives, the server automatically unloads the text model before initialising the VL decoder to stay within the 4 GB domain-1 limit. The text model reloads on the next text-only request. This hot-swap is transparent but adds ~3 s latency on the first mixed workload request.
>
> All Qwen3-VL-2B files (LLM + all 3 vision encoders) are on HuggingFace: [GatekeeperZA/Qwen3-VL-2B-Instruct-RKLLM-v1.2.3](https://huggingface.co/GatekeeperZA/Qwen3-VL-2B-Instruct-RKLLM-v1.2.3). Pre-converted models for other architectures available in the [RKLLM official model zoo](https://console.box.lenovo.com/l/l0tXb8) (fetch code: `rkllm`).

---

## Benchmarks

Measured on **Orange Pi 5 Plus (16 GB)** — RK3588, 3 NPU cores, RKNPU driver 0.9.8, `librkllmrt.so` v1.2.3. All measurements are server-side NPU timing from the `RKLLMResult.perf` struct (not client-side estimates). Both text and VL models remain loaded in NPU memory simultaneously during normal operation.

### Qwen3-1.7B (W8A8, ctx=4096)

| Prompt | Prefill | Prefill Tokens | Generate Time | Output Tokens | tok/s | Cold TTFT |
|--------|---------|---------------|--------------|--------------|-------|----------|
| Short Q&A | 95 ms | 15 | 39.6s | 539 | **13.6** | 4.1s |
| Medium explanation | 128 ms | 26 | 145.3s | 1,829 | **12.6** | 4.1s |
| Long generation | 176 ms | 49 | 134.6s | 1,701 | **12.6** | 4.1s |
| Reasoning (step-by-step) | 140 ms | 33 | 67.0s | 889 | **13.3** | 4.2s |
| **Average** | **135 ms** | **31** | — | **1,240** | **13.0** | **4.1s** |

- **Model load time:** 2.8s
- **NPU memory:** 2,343 MB (standalone) · 7,308 MB (with VL model co-loaded)
- **HuggingFace:** [GatekeeperZA/Qwen3-1.7B-RKLLM-v1.2.3](https://huggingface.co/GatekeeperZA/Qwen3-1.7B-RKLLM-v1.2.3)

### Phi-3-Mini-4K-Instruct (W8A8, ctx=4096)

| Prompt | Prefill | Prefill Tokens | Generate Time | Output Tokens | tok/s | Cold TTFT |
|--------|---------|---------------|--------------|--------------|-------|----------|
| Short Q&A | 204 ms | 10 | 9.4s | 68 | **7.2** | 6.8s |
| Medium explanation | 258 ms | 25 | 141.3s | 913 | **6.5** | 6.9s |
| Long generation | 454 ms | 47 | 149.9s | 963 | **6.4** | 6.6s |
| Reasoning (step-by-step) | 263 ms | 29 | 29.4s | 207 | **7.0** | 6.9s |
| **Average** | **295 ms** | **28** | — | **538** | **6.8** | **6.8s** |

- **Model load time:** 4.5s (avg)
- **NPU memory:** 7,308 MB (with text model co-loaded)
- **HuggingFace:** [GatekeeperZA/Phi-3-mini-4k-instruct-w8a8](https://huggingface.co/GatekeeperZA/Phi-3-mini-4k-instruct-w8a8)

### Key Observations

- **Qwen3-1.7B generates ~2× faster** than Phi-3-Mini despite similar quantization — smaller parameter count means fewer operations per token
- **Prefill scales linearly** with input token count (~6–10 ms per token for Qwen3, ~10–15 ms for Phi-3)
- **Cold TTFT includes model load** — Qwen3 loads in 2.8s, Phi-3 in ~4.5s. Warm TTFT (model already loaded) is ~1.3s for Qwen3 and ~2.0s for Phi-3
- **Generation speed is stable** across prompt lengths — the NPU maintains consistent tok/s regardless of output length
- **Reasoning prompts generate fewer tokens** but at slightly higher tok/s (less KV cache pressure from shorter context)

---

## KV Cache & Inference Performance

### How KV Cache Works on This Server

The KV (Key-Value) cache stores the attention matrices computed during the prefill phase. Instead of recomputing them for every token in the prompt, the NPU reads them from cache — saving significant time on long conversations.

```
Turn 1: "What is South Africa known for?"
  → Full prefill: compute KV for all tokens (~95ms for 15 tokens)
  → Generate answer (kept in KV cache)

Turn 2: "Tell me more about the wildlife."
  → Incremental prefill: only new tokens computed (~20ms for 8 tokens)
  → Prior conversation already in KV cache — no re-prefill needed
```

**The speedup compounds with conversation length** — a 10-turn conversation would take ~10× longer per turn without KV cache.

### What This Server Does

| Mechanism | Status | Effect |
|---|---|---|
| `keep_history=1` between turns | ✅ Active | Each follow-up only prefills new tokens — ~20× prefill speedup on long conversations |
| System prompt preserved on KV reset | ✅ Active | When a new chat starts, system prompt tokens stay cached — only user/assistant turns are cleared |
| Disk prompt cache (`save_prompt_cache`) | ✅ Active | KV state for common system prompts is saved to disk and loaded on next server start — skips re-prefill of long system prompts |
| KV reset on new conversation | ✅ Active | `rkllm_clear_kv_cache()` called when OpenWebUI starts a new chat window — frees NPU memory |
| RAG uses `keep_history=0` | ✅ Correct | RAG always starts fresh so stale conversation context doesn't pollute retrieval answers |

### What KV Cache Does NOT Do

- **Asking the same question in a new chat is not faster.** Each new chat window does a full KV reset — there is no cross-conversation cache.
- **Generation speed is unaffected.** KV cache only speeds up the prefill phase (reading the prompt). Token generation speed (tok/s) is fixed by the NPU clock and model size.
- **The 57-second "Thought for 57s" in OpenWebUI is not cache-related.** That is Qwen3's built-in reasoning mode computing an internal chain of thought before answering. It runs at the same 13 tok/s as regular output — it's just generating a lot of reasoning tokens. Use `/no_think` at the start of your message to skip it for simple queries.

### What RKLLM Does Not Expose (Current Limits)

| Feature | Status | Notes |
|---|---|---|
| Prefix sharing across users | ❌ Not possible | Single NPU context — only one inference at a time |
| Content-hash KV reuse | ❌ No SDK API | Would require hashing prompt prefixes and matching across conversations |
| KV cache quantization | ❌ Not exposed | Would allow longer effective context at lower memory cost |
| Partial cache invalidation | ❌ Not exposed | Can only clear full cache or preserve system prompt — no per-turn granularity |

**Conclusion:** KV cache is being used to its full extent given what RKLLM v1.3.0 exposes. Future SDK versions may expose finer-grained control.

### Reducing Inference Latency

| Technique | Effect | How |
|---|---|---|
| `/no_think` prefix | Eliminates reasoning overhead (~30–60s on Qwen3) | Type `/no_think` before your message in OpenWebUI |
| Shorter system prompts | Faster prefill on new conversations | Reduce system prompt length in OpenWebUI settings |
| Keep conversations going | More turns benefit from KV cache | Avoid opening new chats for follow-up questions |
| Use smaller models | Higher tok/s | Qwen3-1.7B (~13 tok/s) vs Phi-3-Mini (~7 tok/s) |

---

## VL Model Evaluation

We tested all available pre-converted VL models to find the best option for real-world OCR tasks (reading gas meters from phone photos). All models use the same 448×448 (or lower) vision encoder, which crushes high-resolution phone photos (14-15 MP) down to a tiny thumbnail — a fundamental bottleneck for OCR accuracy.

> **RKLLM Version Compatibility:** `.rkllm` model files are compiled for a specific RKLLM runtime version and are **not cross-compatible**. If your model fails with `"Model failed to initialize"`, the model was compiled for a different runtime version than what is installed. Symptom: running `strings model.rkllm | grep rkllm_spilt` returns results — the old weight naming convention (`_spilt_N`, a typo of "split") was used in v1.1.x; v1.2.x uses `_split_N`. Download the model again from a source that provides RKLLM v1.2.x builds.

### Test Setup

- **Hardware:** Orange Pi 5 Plus (16 GB), RK3588, RKNPU driver 0.9.8, rkllm-runtime v1.2.3
- **Test images:** Two real gas meter photos (14-15 MB JPEG, taken with phone camera)
- **Task:** Read the numeric meter display and identify the brand name printed on the meter

### Results

| Model | Source | Meter Reading | Brand Detection | Speed | Verdict |
|-------|--------|--------------|----------------|-------|--------|
| **Qwen3-VL-2B** | [RKLLM model zoo](https://console.box.lenovo.com/l/l0tXb8) | Produces plausible numbers (wrong but in range) | Detected similar text | 5-10s | **Best available** |
| InternVL3.5-2B | [happyme531/InternVL3_5-2B-RKLLM](https://huggingface.co/happyme531/InternVL3_5-2B-RKLLM) | Completely wrong ("200.0", "48") | Hallucinated ("Bundix") | ~30s | Not usable for OCR |
| DeepSeekOCR-3B | [RKLLM model zoo](https://console.box.lenovo.com/l/l0tXb8) | Completely wrong ("1234") or empty | Hallucinated entire scenes (car dashboards, industrial panels) | 5-15s | Not usable — severe hallucination |
| Qwen2.5-VL-3B | [vuong1/Qwen2.5-VL-3B-Instruct-RKLLM](https://huggingface.co/vuong1/Qwen2.5-VL-3B-Instruct-RKLLM) | Not tested (lower 392×392 resolution) | — | ~8.7 tok/s | Rejected — lower resolution than current |

### Key Findings

1. **Qwen3-VL-2B is the best pre-converted option** — while not perfectly accurate for OCR, it produces contextually plausible results and is the fastest
2. **InternVL3.5-2B** has a larger language model (Qwen2.5-1.5B) but W8A8 quantization destroyed its vision capability — also generates Chinese chain-of-thought gibberish
3. **DeepSeekOCR-3B** was specifically designed for OCR but the RKNN conversion is fundamentally broken — it hallucinates entirely different scenes
4. **All models share the same root problem:** the 448×448 (or 392×392) vision encoder crushes phone photos too aggressively for reliable text/number reading
5. **The real solution is re-exporting the vision encoder at higher resolution** — see [Vision Encoder Resolution Comparison](#vision-encoder-resolution-comparison) for results

---

## Vision Encoder Resolution Comparison

After identifying that the 448×448 default vision encoder was the bottleneck, we re-exported the Qwen3-VL-2B vision encoder at 672×672 and 896×896 using the rknn-llm export scripts on an x86 host (Ubuntu, 15GB RAM + 36GB swap, CPU-only — no GPU required).

### File Layout on Orange Pi

```
~/models/Qwen3-VL-2b/
    qwen3-vl-2b-instruct_w8a8_rk3588.rkllm   # LLM decoder (shared)
    qwen3-vl-2b_vision_672_rk3588.rknn        # Active encoder (672×672)
    qwen3-vl-2b_vision_448_rk3588.rknn.alt    # Fast, low-detail (inactive)
    qwen3-vl-2b_vision_896_rk3588.rknn.alt    # Slow, high-detail (inactive)
```

To switch encoder: rename the active `.rknn` to `.rknn.alt` and the desired one from `.rknn.alt` to `.rknn`, then `sudo systemctl restart rkllm-api`.

### Resolution Benchmark Results

Tested on Orange Pi 5 Plus (16GB) with real 14-15MB JPEG gas meter photos:

| Resolution | Visual Tokens | RKNN Size | Encoder Time | Total Response | Peak RAM |
|---|---|---|---|---|---|
| **448×448** | 196 (14×14) | 812 MB | ~2s | 5–10s | ~5.5 GB |
| **672×672** ⭐ | 441 (21×21) | 854 MB | ~4s | 9–11s | ~6.5 GB |
| **896×896** | 784 (28×28) | 923 MB | ~12s | 25–28s | ~8.5 GB |

### OCR Test Results (Gas Meter Photos)

| Resolution | meter1.jpg | meter2.jpg | Consistency |
|---|---|---|---|
| 448×448 | `975648` | `3700211` | Unique readings |
| 672×672 | `37866` | `3709217` | — |
| 896×896 | `37866` | `57709217` | 672 & 896 agree on meter1 |

**Key conclusions:**
- **672×672 is the sweet spot** — 2.25× more visual detail with only ~1s extra latency vs 448
- **896×896 is 3× slower** (25-28s vs 9-11s) for marginal benefit
- **The 2B LLM is the accuracy bottleneck**, not the vision encoder — different resolutions produce different (all incorrect) readings, suggesting the model size limits OCR reliability
- **Text decode speed is unaffected** (~15 tok/s) — only the vision encode + prefill time increases
- All files are on HuggingFace: [GatekeeperZA/Qwen3-VL-2B-Instruct-RKLLM-v1.2.3](https://huggingface.co/GatekeeperZA/Qwen3-VL-2B-Instruct-RKLLM-v1.2.3)

---

## Re-Exporting VL Models at Higher Resolution

The RKNN export scripts accept `--height` and `--width` parameters, so you can re-export the Qwen3-VL-2B vision encoder at a higher resolution (e.g., 672×672 or 896×896) to improve OCR accuracy. This only affects the `.rknn` vision encoder — the `.rkllm` language model stays the same.

### Requirements

- **x86 Linux machine** with Python 3.9-3.12 (the toolkits do **not** run on ARM)
- **No GPU required** — CPU-only export works (tested on Ubuntu 22.04, 15GB RAM)
- ~20 GB RAM+swap for 672×672, ~35 GB for 896×896 (use `fallocate` + `mkswap` for extra swap)
- `rknn-toolkit2` v2.3.2 (`pip install rknn-toolkit2`)
- `torch==2.4.0`, `torchvision==0.19.0`
- `transformers>=4.57.0`, `onnx>=1.18.0`

### Step-by-Step: Re-Export Qwen3-VL at Higher Resolution

```bash
# Clone the export scripts
git clone https://github.com/airockchip/rknn-llm.git
cd rknn-llm/examples/multimodal_model_demo

# Install dependencies
pip install transformers==4.57.0 torch rkllm-toolkit rknn-toolkit2

# Download the original Qwen3-VL-2B HuggingFace model
# (needed as source for the vision encoder weights)
git clone https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

# Step 1: Export vision encoder to ONNX at custom resolution
# Height/width must be divisible by (merge_size * patch_size) = 2 * 16 = 32 for Qwen3-VL
python export/export_vision.py \
  --path ./Qwen3-VL-2B-Instruct \
  --model_name qwen3-vl \
  --height 672 --width 672
# Output: ./onnx/qwen3-vl_vision.onnx

# Step 2: Convert ONNX to RKNN for RK3588
python export/export_vision_rknn.py \
  --path ./onnx/qwen3-vl_vision.onnx \
  --model_name qwen3-vl \
  --target-platform rk3588 \
  --height 672 --width 672
# Output: ./rknn/qwen3-vl_vision_rk3588.rknn
```

### Resolution Options for Qwen3-VL

Qwen3-VL uses `patch_size=16` and `merge_size=2`, so resolution must be divisible by 32:

| Resolution | Visual Tokens | Encoder Time (measured) | RAM Impact | Notes |
|-----------|--------------|--------------------|-----------|---------|
| 448×448 | 196 | ~2s | Baseline (5.5 GB) | Default from model zoo |
| 672×672 | 441 | ~4s | +1 GB (6.5 GB) | ⭐ **Recommended** — 2.25× more pixels |
| 896×896 | 784 | ~12s | +3 GB (8.5 GB) | 4× more pixels, noticeably slower |
| 1120×1120 | 1225 | ~20s+ | +5 GB+ | May OOM on 16 GB devices |

> **Note:** The `.rkllm` language model file does NOT need to be re-exported — only the `.rknn` vision encoder changes. Copy the new `.rknn` file to the model folder on the Orange Pi alongside the existing `.rkllm` file.

### Important Constraints

- Height and width must be divisible by `patch_size × merge_size` (32 for Qwen3-VL, 28 for Qwen2/2.5-VL)
- Higher resolution = more visual tokens = longer prefill time and more NPU memory
- The vision encoder runs on the NPU — very large resolutions may cause OOM on 16 GB devices
- The RKLLM LLM decoder has a fixed `max_context_len` — ensure visual tokens + text tokens fit within it

---

## Ollama API Compatibility

The server exposes an Ollama-compatible API surface alongside the OpenAI API. Any client that speaks the Ollama wire protocol can connect without configuration changes.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/version` | Version probe — returns `{"version":"0.3.0-rkllm"}` |
| `GET` | `/api/tags` | List all models in Ollama format |
| `POST` | `/api/chat` | Chat completions — streaming (NDJSON) or non-streaming |
| `POST` | `/api/generate` | Raw prompt completion — wraps prompt as a user message |
| `GET` | `/api/show` | Model info stub |
| `GET` | `/api/ps` | Running-model stub |

### Quick Test

```bash
# Verify Ollama handshake
curl http://localhost:8000/api/version

# List models
curl http://localhost:8000/api/tags | python3 -c "import sys,json; [print(m['name']) for m in json.load(sys.stdin)['models']]"

# Non-streaming chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"/no_think Hello"}],"stream":false}'

# Streaming chat (NDJSON)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"/no_think Count to 3"}],"stream":true}'
```

### Connecting Open WebUI in Ollama Mode

If you want to use Open WebUI's native Ollama connection instead of the OpenAI adapter:

**Admin > Settings > Connections > Ollama:**

| Setting | Value |
|---------|-------|
| Ollama API URL | `http://host.docker.internal:8000` (no `/v1`) |

All models will appear in the model dropdown exactly as they do via the OpenAI adapter. Both connection types can be active simultaneously.

---

## LoRA Hot-Swap

Load and unload LoRA adapters at runtime without restarting the server. The adapter is applied to the currently loaded text model via the `rkllm_load_lora()` C API.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/lora` | List all currently registered adapters |
| `POST` | `/v1/lora/load` | Load an adapter from disk |
| `DELETE` | `/v1/lora/<name>` | Unregister an adapter |

### Usage

```bash
# List loaded adapters
curl http://localhost:8000/v1/lora
# {"adapters":[]}

# Load an adapter
curl -X POST http://localhost:8000/v1/lora/load \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/home/armbian/models/lora/my-adapter.bin",
    "name": "my-adapter",
    "scale": 1.0
  }'
# {"status":"loaded","name":"my-adapter"}

# List again to confirm
curl http://localhost:8000/v1/lora
# {"adapters":[{"name":"my-adapter","path":"/home/armbian/models/lora/my-adapter.bin","scale":1.0}]}

# Unregister (does not unload from NPU — reload model to fully clear)
curl -X DELETE http://localhost:8000/v1/lora/my-adapter
# {"status":"unloaded","name":"my-adapter"}
```

### Request Body for `/v1/lora/load`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | ✅ | Absolute path to the adapter `.bin` file on the Orange Pi |
| `name` | string | ✅ | Identifier used for listing and deletion |
| `scale` | float | ❌ | Adapter weight scale α (default `1.0`) |

### Notes

- **Runtime requirement:** `rkllm_load_lora` must be exported by your `librkllmrt.so`. `POST /v1/lora/load` returns `HTTP 500` (`lora_load_failed`) if the call fails, e.g. the symbol is absent.
- LoRA is applied to the **currently loaded text model** — the adapter must be compatible with that model's architecture.
- Unregistering an adapter via `DELETE` removes it from the server's tracking table. To fully clear the adapter from the NPU weights, unload and reload the model (`POST /v1/models/unload` then `/v1/models/select`).
- Multiple adapters can be loaded simultaneously if supported by the runtime.

---

## Hermes Agent Integration

> **Branch:** `hermes-agent-integration`

[Hermes Agent](https://github.com/nousresearch/hermes) is an autonomous AI agent that runs a full tool-calling loop — web search, code execution, memory, and more. This branch integrates Hermes as a selectable model inside Open WebUI, appearing alongside the local NPU models in the model dropdown.

### How It Works

```
Open WebUI ──► Hermes Gateway (port 8642) ──► Nemotron 550B (OpenRouter free)
                                         ├──► llama-3.3-70b (Groq free, fallback)
                                         └──► qwen3-vl-4b (local NPU, vision tasks)
```

When a user selects **Hermes Agent** in Open WebUI, requests go to the Hermes gateway instead of the RKLLM API server. Hermes runs its autonomous agent loop — searching the web, using tools, maintaining memory — powered by cloud LLMs. Vision tasks (image analysis) are routed to the local `qwen3-vl-4b` NPU model via the `auxiliary.vision` provider. Local NPU models are available directly in OWUI for fast single-turn chat; Hermes requires ≥64K context which exceeds what the NPU models support.

### Provider Fallback Chain

Hermes tries each provider in order until one succeeds:

| Priority | Provider | Model | Notes |
|----------|----------|-------|-------|
| 1st | OpenRouter | `nvidia/nemotron-3-ultra-550b-a55b:free` | Free tier, 550B parameter model, 1M context |
| 2nd | Groq | `llama-3.3-70b-versatile` | ~14,400 req/day free, fast inference |

### Setup

The `setup.sh` script handles Hermes installation automatically when cloud API keys are provided:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."   # Free at openrouter.ai (primary)
export GROQ_API_KEY="gsk_..."              # Free at console.groq.com (fallback)

git clone https://github.com/GatekeeperZA/RKLLM-API-Server.git
cd RKLLM-API-Server
./setup.sh
```

**What setup.sh installs:**
- Hermes Agent v0.19.0 in a dedicated Python 3.11 venv (`/home/armbian/rkllm_api/.venv-hermes`)
- `rkllm-hermes.service` — systemd service exposing the Hermes gateway on **port 8642**
- Hermes config at `/home/armbian/.hermes/config.yaml` with provider + fallback chain
- API key env vars in the systemd service unit (not stored in config.yaml)
- `tests/db_fix.py` automatically adds the Hermes gateway as a connection in Open WebUI

**Manual configuration** (`hermes/config.yaml` template):
```yaml
providers:
  openrouter:
    api_key: "${OPENROUTER_API_KEY}"
    base_url: "https://openrouter.ai/api/v1"
    request_timeout_seconds: 120
  groq:
    api_key: "${GROQ_API_KEY}"
    base_url: "https://api.groq.com/openai/v1"
    request_timeout_seconds: 60
  local:
    base_url: "http://127.0.0.1:8000/v1"
    api_key: "na"
    request_timeout_seconds: 300

model:
  provider: openrouter
  model: nvidia/nemotron-3-ultra-550b-a55b:free
  context_length: 131072    # override reported window — Hermes requires ≥64K

fallback_providers:
  - provider: openrouter
    model: nvidia/nemotron-3-ultra-550b-a55b:free
    key_env: OPENROUTER_API_KEY
  - provider: groq
    model: llama-3.3-70b-versatile
    key_env: GROQ_API_KEY

# Vision tasks routed to local VLM — avoids cloud roundtrip for image analysis
auxiliary:
  vision:
    provider: "local"
    model: "qwen3-vl-4b"

api_server:
  enabled: true
  host: 0.0.0.0
  port: 8642
  key: "${HERMES_API_KEY}"
```

### Open WebUI Integration

The Hermes gateway appears in Open WebUI as a separate OpenAI-compatible connection. `tests/db_fix.py` adds it automatically:

| Setting | Value |
|---------|-------|
| API Base URL | `http://host.docker.internal:8642/v1` |
| API Key | *(generated during setup)* |
| Model | `hermes-agent` |

Users see **hermes-agent** in the model dropdown alongside `qwen3-1.7b`, `qwen3-4b`, etc. Selecting it routes the request through the Hermes autonomous agent loop instead of direct NPU inference.

### Key Differences vs Direct NPU Models

| Feature | Direct NPU (qwen3-4b) | Hermes Agent |
|---------|----------------------|--------------|
| Web search | Via Open WebUI + SearXNG | Native agent tool (autonomous) |
| Tool calling | Limited (via legacy function_calling) | Full agentic loop with retries |
| Model size | 4B NPU | 550B cloud (free) |
| Response time | ~2-5s | ~10-30s (cloud roundtrip) |
| Privacy | Fully local | Cloud LLM sees queries |
| Offline | ✅ Yes | ❌ Requires cloud connection |

### Service Notes

- **`StartLimitIntervalSec` / `StartLimitBurst`** must be in the `[Unit]` section of the systemd unit file — not `[Service]`. They are silently ignored in `[Service]` and the service will restart without limit. The `rkllm-hermes.service` included in this repo has the correct placement.
- **`rkllm-embed.service`** depends on `rkllm-api.service` starting first (shared NPU memory). `After=rkllm-api.service` is set, and `RestartSec=30` gives the NPU time to free before a restart attempt.
- **Multiple system messages** from Open WebUI pipelines are silently concatenated by `api.py` — this is expected behavior, not an error.

### Service Management

```bash
# Start / stop / restart
sudo systemctl start rkllm-hermes
sudo systemctl stop rkllm-hermes
sudo systemctl restart rkllm-hermes

# View logs
sudo journalctl -u rkllm-hermes -f

# Test the gateway
curl http://localhost:8642/v1/models \
  -H "Authorization: Bearer <your-api-key>"
```

### Small Model Tool Calling (Experimental)

Two sub-2B NPU models have been converted for fast local tool calling:

| Model | BFCL v3 | Speed | Best For |
|-------|---------|-------|----------|
| **xLAM-1b-fc-r** | **78.94%** | ~16 tok/s | Structured function calls (highest sub-2B accuracy) |
| **Qwen2.5-1.5B-Instruct** | ~40-50% | ~14 tok/s | General instruct + light tool use |

Both models are on HuggingFace ([xLAM](https://huggingface.co/GatekeeperZA/xLAM-1b-fc-r-RKLLM-v1.2.3) · [Qwen2.5-1.5B](https://huggingface.co/GatekeeperZA/Qwen2.5-1.5B-Instruct-RKLLM-v1.2.3)) and auto-discovered by the RKLLM API server. For production tool calling, use the cloud-backed Hermes Agent — the sub-2B local models are best suited to simple, well-structured function calls.

---

## Roadmap & Known Gaps

### Planned (research + docs only)

| Item | Notes |
|---|---|
| **W4A16 quantization guide** | W4A16 models run ~1.5–2× faster than W8A8/W8A16 at the cost of 1–3% accuracy loss on benchmarks. Requires re-converting models with `rkllm-toolkit`. Guide + accuracy tradeoff table to be added. |
| **Prompt caching research** | RKLLM exposes `save_prompt_cache` / `load_prompt_cache` for disk-based KV state persistence. Currently implemented for system prompts. Research whether per-user prefix caching is feasible given single-NPU constraints. |
| **Vision auto-load on demand** | VL model currently stays resident in NPU memory once loaded (~2–3 GB). Auto-unload after idle + reload on next image request would free memory for text model but adds 3–5s latency on first image after idle. Evaluating whether the tradeoff is worth it. |

### Not Implemented (decided against)

| Item | Reason |
|---|---|
| **JSON schema / structured output enforcement** | Adds latency: schema injection increases prefill, validation adds CPU work, retries can 2–3× total request time. Not worth the cost for a single-user NPU server. |
| **Batch inference** | RKLLM processes one request at a time — the NPU has a single execution context. Queuing (already implemented) is the correct approach. |
| **NPU overclock beyond 1GHz** | Stock max is 1GHz. Going higher requires DTB modification and the PMIC voltage rail is already at its ceiling (850mV). Community reports show instability under sustained load. Not safe. |
| **Prefix sharing across users** | Requires multiple simultaneous NPU contexts — not possible on RK3588. |

### Missing API Surface

| Endpoint | Status | Notes |
|---|---|---|
| `POST /v1/completions` | ❌ Not implemented | Legacy non-chat text completion — some older tools (Continue.dev, llama.cpp clients) use this |
| `POST /v1/messages` streaming | ⚠️ Partial | Anthropic shim translates to OpenAI format but does not yet support `"stream": true` |
| `xlam-1b-fc-r` tool routing | ⚠️ Partial | Model is detected and listed but has `base` capability — no automatic tool-call prompt injection for base models |

### Undocumented Tools

The `tools/` directory contains helper scripts that are not yet covered in the README:

| Script | Purpose |
|---|---|
| `tools/convert_rkllm.py` | Convert HuggingFace models to `.rkllm` format using `rkllm-toolkit` |
| `tools/hf_upload.py` | Upload converted models to HuggingFace Hub |
| `tools/hf_upload_pi_models.py` | Upload Orange Pi compiled models directly from the device |

Documentation for these scripts is planned.

---

## Known Issues & Fixes

Bugs found during code audits, with the commit where each was resolved. Listed so other users running this codebase can identify whether they are affected and apply the fix manually if they are on an older commit.

---

### [FIXED `ec3cfc6`] Shutdown cleanup skipped on SIGTERM/SIGINT — NPU handle never released

**Severity:** Critical  
**Affects:** All versions before commit `ec3cfc6`

`shutdown()` used `SHUTDOWN_EVENT.is_set()` as a double-call guard. The signal handler sets `SHUTDOWN_EVENT` first, then delegates cleanup to `model_monitor` which calls `shutdown()`. Because the event was already set, the guard fired immediately and returned — skipping `unload_current()` and `_unload_vl_model()`. The NPU RKLLM handle was never destroyed on clean shutdown.

**Fix:** Replaced the guard with a separate `_SHUTDOWN_DONE = Event()` that is only set inside `shutdown()` itself, not by the signal handler.

---

### [FIXED `ec3cfc6`] Stale request timeout (30s) shorter than NPU model load time (30–90s)

**Severity:** High  
**Affects:** All versions before commit `ec3cfc6`

`REQUEST_STALE_TIMEOUT = 30` meant a second concurrent request arriving while a slow model was loading (cold NPU, large model) would see the first slot as stale, clear it, and start a second `rkllm_init()` — two concurrent NPU inits produces undefined behaviour.

**Fix:** `REQUEST_STALE_TIMEOUT` increased to `120` (seconds). `update_request_activity()` is also called immediately after `load_model()` returns to reset the stale clock.

---

### [FIXED `ec3cfc6`] Ollama `/api/tags` listed embedding/reranker models

**Severity:** Medium  
**Affects:** Deployments using `skip_chat: true` in `model_config.json` before commit `ec3cfc6`

`/v1/models` correctly filtered models with `skip_chat: true` (embedding/reranker models). The Ollama-compatibility endpoint `/api/tags` did not, so Ollama-compatible clients (Continue, AnythingLLM) showed those models in their picker and returned a confusing 400 error when selected for chat.

**Fix:** `ollama_tags()` now applies the same `skip_chat` filter as `list_models()`.

---

### [FIXED `ec3cfc6`] `setup.sh` Hermes config block aborted mid-install with no cloud API keys

**Severity:** Medium  
**Affects:** Users running `setup.sh` without `OPENROUTER_API_KEY` or `GROQ_API_KEY` set, before commit `ec3cfc6`

The embedded Python heredoc raised `RuntimeError('No cloud API keys provided...')` when neither cloud key was set. This caused `setup.sh` to exit non-zero mid-run, leaving the hermes venv installed but the systemd service not registered. There was no recovery guidance printed.

**Fix:** The no-key path now writes a placeholder `~/.hermes/config.yaml` with instructions, prints a clear warning, and exits `sys.exit(0)` so the rest of `setup.sh` continues cleanly.

---

### [FIXED `ec3cfc6`] `setup.sh` PYEOF heredoc expanded shell variables into Python source

**Severity:** Medium  
**Affects:** All versions before commit `ec3cfc6` on any deployment where `OPENROUTER_API_KEY` or `GROQ_API_KEY` contain shell metacharacters (`$`, `\`, `'`)

The heredoc delimiter was unquoted (`<< PYEOF`), so the shell expanded `${OPENROUTER_API_KEY}` inline into the Python source. API keys containing single-quotes or dollar signs would produce a Python `SyntaxError` or silently embed a truncated key.

**Fix:** Delimiter is now quoted (`<< 'PYEOF'`); API keys are read inside the script via `os.environ.get('OPENROUTER_API_KEY', '')`, which is safe for all key formats.

---

### [FIXED `ec3cfc6`] Ollama streaming response iterated outside its request context

**Severity:** Medium  
**Affects:** All versions before commit `ec3cfc6` using the Ollama `/api/chat` streaming path

`_ollama_stream()` called `chat_completions()` inside `with app.request_context(environ):`, then iterated `oai_resp.response` (a lazy generator) after the `with` block exited and tore down the context. Subsequent chunks would attempt to push `stream_with_context`'s inner context into a torn-down application context, raising `RuntimeError('Working outside of application context')`.

**Fix:** Replaced `with` block with manual `ctx.push()` / `ctx.pop()` in a `try/finally` wrapping both the `chat_completions()` call and the full generator iteration.

---

### [FIXED `ec3cfc6`] Agent framework heuristic silently disabled thinking for long user system prompts

**Severity:** Medium  
**Affects:** Thinking-capable models (Qwen3) with user system prompts longer than 3000 characters, before commit `ec3cfc6`

The heuristic `if len(system_text) > 3000: enable_thinking = False` was intended to detect agent frameworks (Hermes, AutoGen) which inject large tool schemas. It also fired for any user with a long system prompt, silently degrading thinking models with no response header or warning logged at a visible level.

**Fix:** The heuristic now also checks for tool-related content markers (`<tools>`, `tool_call`, `"type": "function"`, etc.) before disabling thinking. A long system prompt without tool markers keeps thinking enabled; this case is logged at DEBUG level.

---

### [FIXED `ec3cfc6`] embed service `StartLimitBurst=40` could storm the NPU driver

**Severity:** Low  
**Affects:** Deployments using `rkllm-embed.service` before commit `ec3cfc6`

`StartLimitBurst=40` in a 600s window permitted 40 rapid systemd restarts if the embed service failed fast (e.g. NPU busy). With `RestartSec=30`, all 40 attempts fire within 20 minutes, hammering the NPU driver and flooding the journal rather than backing off.

**Fix:** Reduced to `StartLimitBurst=5` in a `StartLimitIntervalSec=300s` window — 5 attempts over 5 minutes is sufficient for transient NPU contention recovery.

---

### [FIXED `ec3cfc6`] `disable_tools` strip removed user system message if injected prompt was absent

**Severity:** Medium  
**Affects:** Edge case: models with `disable_tools: true` in `model_config.json`, before commit `ec3cfc6`

`messages = messages[1:]` was always executed when `_has_tools=True` and `disable_tools=True`, on the assumption that `messages[0]` was always the server-injected tool system prompt. If that assumption was ever violated (e.g. a future code path where injection didn't happen), the first user or system message would be silently dropped.

**Fix:** Added a guard: `messages[0]` is only stripped if its `role == 'system'` and its content starts with `# Tools\n` (the known injection prefix). A `logger.warning` fires if the expected prefix is absent.

---

### [FIXED `audit-2`] Hardcoded API keys committed to repository

**Severity:** Critical  
**Affects:** All versions before this audit — `tools/rkllm-hermes.service` and `tests/test_openwebui.py`

`tools/rkllm-hermes.service` contained a live `API_SERVER_KEY` value hardcoded in an `Environment=` line. `tests/test_openwebui.py` contained a live Open WebUI session key as the default for `OWUI_API_KEY`. Both were committed to git history and readable by anyone with repo access.

**Fix:** Service file now uses `EnvironmentFile=-/etc/rkllm-hermes.env` (mode 0600, root-owned). Test file now requires `OWUI_API_KEY` to be set as an environment variable — no default. Rotate any keys that were exposed.

---

### [FIXED `audit-2`] OWUI RAG used CPU sentence-transformers instead of rkllm-embed NPU service

**Severity:** High  
**Affects:** All Docker deployments before this audit

`docker-compose.yml` configured Open WebUI to download and run `BAAI/bge-small-en-v1.5` inside the container for RAG embeddings, ignoring the `rkllm-embed` service already running on port 8001. This caused a vector space mismatch (RAG indexed with bge, direct embeddings used Qwen3-Embedding-0.6B) and wasted CPU on a task the NPU was already handling.

**Fix:** `docker-compose.yml` now sets `RAG_EMBEDDING_ENGINE=openai` pointing to `http://host.docker.internal:8001/v1`, using the same Qwen3-Embedding-0.6B model for both RAG and direct embedding calls.

---

### [FIXED `audit-2`] NPU metrics poller used `sudo cat` — privilege escalation risk

**Severity:** High  
**Affects:** All versions before this audit using Prometheus metrics

The NPU load metric poller executed `sudo /usr/bin/cat /sys/kernel/debug/rknpu/load` on every poll cycle. Granting passwordless `sudo cat` is a file-read primitive for any process running as the service user. The fallback `open()` path already worked for world-readable debugfs files.

**Fix:** Removed the `sudo cat` path entirely. The `open()` fallback is now the only read method. If `rknpu` debugfs requires elevated access on your kernel, add a udev ACL rule instead.

---

### [FIXED `audit-2`] `/v1/lora/load` accepted arbitrary filesystem paths

**Severity:** High  
**Affects:** All versions before this audit on unauthenticated deployments

The LoRA load endpoint accepted any file path from the request body and passed it to `rkllm_load_lora()` without validating it was inside the models directory. Any network client could use this as a filesystem existence probe or to attempt loading arbitrary files via the C library.

**Fix:** Added `os.path.realpath()` check — path must be inside `MODELS_ROOT`. Returns HTTP 403 otherwise.

---

### [FIXED `audit-2`] `rkllm-embed.service` missing security hardening directives

**Severity:** High  
**Affects:** Users who deployed `tools/rkllm-embed.service` directly (not via `setup.sh`)

The committed embed service file lacked `NoNewPrivileges`, `ProtectSystem`, `ProtectHome`, and `ReadWritePaths` directives that `setup.sh` generated. It also had inconsistent `RestartSec` (30s vs 15s in setup.sh) and `StartLimitBurst` values.

**Fix:** Service file now includes full security hardening block matching setup.sh output, `RestartSec=15`, and the correct start limit values.

---

### [FIXED `audit-2`] `CONTEXT_COMPACTION_TOKEN_CAP=32000` exceeded qwen3-1.7b context window

**Severity:** Medium  
**Affects:** All Docker deployments before this audit using context compaction

The compaction cap was set to 32,000 tokens but qwen3-1.7b (the compaction model) has a 4K–8K context window. Compaction summaries would be rejected or silently truncated.

**Fix:** `CONTEXT_COMPACTION_TOKEN_CAP` reduced to `2048` — within the model's safe context window.

---

### [FIXED `audit-2`] RAG cache stored reasoning differently in streaming vs non-streaming paths

**Severity:** Medium  
**Affects:** Users with thinking-capable models (Qwen3) using RAG before this audit

The streaming path stored `<think>reasoning</think>content` in the cache when reasoning was present. The non-streaming path always stored only `content`, discarding reasoning. A cache miss on the non-streaming path would return a response without reasoning; a subsequent miss on the streaming path would return one with it — inconsistent behaviour.

**Fix:** Non-streaming path now stores `<think>reasoning</think>content` when reasoning is present, matching the streaming path.

---

### [FIXED `audit-2`] Title shortcircuit could intercept real user queries

**Severity:** Low  
**Affects:** Users sending messages containing words like "title", "concise", and "chat" before this audit

The OWUI title-generation shortcircuit matched purely on keywords in the last user message. A real query like "Give me a concise title for this chat" would be intercepted and return the first 60 characters of the conversation instead of being sent to the model.

**Fix:** Shortcircuit now also requires the OWUI task signature to be present in a system message before firing — it no longer triggers on user messages alone.

---

### [FIXED `audit-2`] `embed_api.py` embedding extraction used slow Python ctypes loop

**Severity:** Medium  
**Affects:** All embed service deployments before this audit

The embedding callback extracted the final token's hidden state vector using a Python `for i in range(d)` loop over ctypes pointer indexing. For Qwen3-Embedding-0.6B (d=1024+), this ran entirely in Python inside the `_LOCK` critical section, blocking all concurrent embedding requests.

**Fix:** Replaced with a single `np.frombuffer()` copy from a cast ctypes pointer — O(1) Python, O(d) C memcopy.

---

### [FIXED `audit-2`] `embed_api.py` missing null handle check and content length limit

**Severity:** High  
**Affects:** All embed service deployments before this audit

`rkllm_init()` returning 0 with a null handle (NPU OOM) was not detected — subsequent `rkllm_run()` calls would segfault. Additionally, no `MAX_CONTENT_LENGTH` was set, allowing arbitrarily large embedding requests that would silently truncate at the model's 2048-token context limit.

**Fix:** Null handle check added after `rkllm_init()`. `MAX_CONTENT_LENGTH = 1 MB` set on the Flask app.

---

## Git Tags & Branches

| Tag / Branch | Description |
|---|---|
| `v1.0-subprocess-stable` | Last working subprocess version (V1) |
| `v1.1-ctypes-text-only` | Text-only ctypes version before VL additions |
| `subprocess-legacy` | Branch preserving the subprocess architecture |
| `main` | Current stable: RKLLM v1.2.3 runtime — ctypes + VL multimodal + meta-task shortcircuits + context-enriched query gen + document RAG + model-aware sampling + prompt cache + sliding window + NPU benchmarks |
| `hermes-agent-integration` | Hermes Agent v0.19.0 integration — autonomous agent as an OpenWebUI model; Ollama API compat layer (`/api/chat`, `/api/tags`, `/api/generate`); NPU hardware Prometheus metrics (`rknpu_core_load_percent`, `rknpu_freq_mhz`, `rknpu_power_on`); LoRA hot-swap endpoints (`/v1/lora`); `/no_think` soft-switch for Qwen3; VL 4B domain-1 IOVA OOM fix; VL embed cache key collision fix; cloud provider fallback chain (OpenRouter Nemotron 550B → Groq llama-3.3-70b); setup.sh automation |
| `rkllm-v1.3.0-wip` | Work-in-progress v1.3.0 upgrade — **archived, not stable**. Contains updated ctypes structs for v1.3.0 ABI and a two-phase thinking implementation. Rolled back due to phase-2 early EOS bug (13–27 tokens then mid-sentence EOS). See [ISSUES_v1.3.0.md](ISSUES_v1.3.0.md) on that branch for full bug tracker. Revisit when a newer v1.3.x runtime is released. |

### RKLLM v1.3.0 Status

> **v1.3.0 is currently not recommended for production.** The `rkllm-v1.3.0-wip` branch contains a complete upgrade attempt including updated struct layouts for the new ABI and a two-phase thinking implementation. It was rolled back to v1.2.3 on 2026-08-09 due to an unresolved bug where phase-2 inference (after `</think>`) generates only 13–27 tokens then terminates mid-sentence, even with `role=None`. No `ignore_eos_token` field exists in v1.3.0's `RKLLMSamplingParam`, making a per-call workaround impossible.
>
> **To try v1.3.0:** Check out the `rkllm-v1.3.0-wip` branch and replace `~/librkllmrt.so` with the v1.3.0 binary. Read `ISSUES_v1.3.0.md` for the full list of known issues and what was attempted.

---

## License

This project is provided as-is for personal and educational use. The rkllm runtime and model files are subject to their respective licenses from Rockchip and model authors.

## Acknowledgements

- [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm) — RKLLM runtime, toolkit, and multimodal demo
- [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) — RKNN runtime (`librknnrt.so`) for vision encoder NPU inference
- [Pelochus/armbian-build-rknpu-updates](https://github.com/Pelochus/armbian-build-rknpu-updates) — Armbian builds with RKNPU driver
- [Open WebUI](https://github.com/open-webui/open-webui) — Web interface for LLM interaction
