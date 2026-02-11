# RKLLM API Server

**OpenAI-compatible API server for Rockchip NPU (RK3588/RK3576) running RKLLM models, designed as a drop-in backend for [Open WebUI](https://github.com/open-webui/open-webui).**

Built for single-board computers like the **Orange Pi 5 Plus**, this server bridges the gap between the `librkllmrt.so` C runtime and any OpenAI-compatible frontend — enabling local, private LLM inference on ARM hardware with zero cloud dependency.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
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
- [Git Tags & Branches](#git-tags--branches)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

### Core
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/models` endpoints work with any OpenAI client
- **Direct NPU access** via ctypes binding to `librkllmrt.so` (no subprocess overhead)
- **KV cache incremental mode** — follow-up turns only prefill the new message (~50ms vs ~500ms)
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
- **Error callback state** — detects C library errors and surfaces them as proper HTTP responses

### RAG (Retrieval-Augmented Generation)
- **Automatic RAG detection** when Open WebUI injects web search or document results
- **Document/PDF RAG** — works with Open WebUI's document upload and embedding pipeline
- **Summarization detection** — detects "summarize" queries and adds stronger multi-paragraph instructions
- **Smart prompt restructuring** — reading comprehension format optimized for small models
- **4-pass web content cleaning** — strips navigation, boilerplate, cookie banners
- **Score-based paragraph selection** — jusText-inspired content quality scoring
- **Near-duplicate removal** — Jaccard similarity deduplication across sources
- **Quality floor** — drops irrelevant search results instead of confusing the model
- **Follow-up detection** — 3-layer system prevents RAG on conversational replies
- **Response caching** — LRU cache eliminates redundant inference for repeated questions
- **Context-dependent thinking** — disables reasoning on small context models to save tokens

### Reasoning Model Support
- **`<think>` tag parsing** for Qwen3 and similar reasoning models
- **`reasoning_content`** field in both streaming deltas and non-streaming responses
- **Streaming state machine** handles tags split across output chunks
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

### Standards Compliance
- **`stream_options.include_usage`** — streaming token counts per OpenAI spec
- **`system_fingerprint`** in all responses
- **`max_tokens` / `max_completion_tokens`** support
- **Request body size limit** (16 MB)
- **Proper error responses** matching OpenAI error format

### Vision-Language (VL) / Multimodal
- **Dual-model architecture** — text model (e.g. Qwen3-1.7B) + VL model (e.g. Qwen3-VL-2B) loaded simultaneously
- **Automatic image routing** — requests with images route to VL model, text-only to text model
- **Base64 image support** — accepts `image_url` with `data:image/...;base64,...` format (Open WebUI compatible)
- **Direct NPU vision encoding** via ctypes binding to `librknnrt.so` (no Python RKNN toolkit needed)
- **Image preprocessing** — auto square-pad (128,128,128 background) and resize to encoder input size
- **Multiple VL model support** — auto-detects `.rknn` vision encoder alongside `.rkllm` decoder
- **Configurable special tokens** — `VL_MODEL_CONFIGS` maps model families to their image tokens
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

### Python Dependencies
```bash
# Core (required)
pip install flask flask-cors gunicorn

# VL / multimodal support (optional — needed only for vision-language models)
pip install numpy Pillow
```

---

## Installation

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

### Manual Installation

<details>
<summary>Click to expand manual step-by-step instructions</summary>

#### 1. Clone This Repository

```bash
git clone https://github.com/GatekeeperZA/RKLLM-API-Server.git
cd RKLLM-API-Server

# Install Python dependencies
pip install flask flask-cors gunicorn

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

| Model | Parameters | Quant | Context | Speed | RAM | Thinking | Link |
|-------|-----------|-------|---------|-------|-----|----------|------|
| **Qwen3-1.7B** | 1.7B | w8a8 | 4,096 | ~13.6 tok/s | ~2 GB | ✅ Yes | [Download](https://huggingface.co/GatekeeperZA/Qwen3-1.7B-RKLLM-v1.2.3) |
| **Phi-3-mini-4k-instruct** | 3.82B | w8a8 | 4,096 | ~6.8 tok/s | ~3.7 GB | ❌ No | [Download](https://huggingface.co/GatekeeperZA/Phi-3-mini-4k-instruct-w8a8) |

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

### Auto-Generated Aliases

Model folder names are converted to IDs (lowercase, hyphens). Aliases are auto-generated:

| Model ID | Auto-Aliases |
|----------|-------------|
| `qwen3-1.7b` | `qwen`, `qwen3` |
| `qwen3-4b-instruct-2507` | `qwen3-4b`, `qwen3-4b-instruct` |
| `gemma-3-4b-it` | `gemma`, `gemma-3`, `gemma-3-4b` |
| `phi-3-mini-4k-instruct` | `phi`, `phi-3`, `phi-3-mini` |

Aliases are only created when unambiguous (one model claims the alias). If two models share a prefix, that alias is skipped.

---

## Running the Server

### Production (Recommended)

```bash
gunicorn -w 1 -k gthread --threads 4 --timeout 300 -b 0.0.0.0:8000 api:app
```

> **Critical:** Always use `-w 1` (single worker). The NPU can only load one model at a time.
>
> **Critical:** Always use `-k gthread`, NOT `-k gevent`. `rkllm_run()` is a blocking C call that freezes gevent's event loop.

### Development

```bash
python api.py
```

This starts Flask's built-in server on `0.0.0.0:8000` with threading enabled.

### Systemd Service

The setup script creates this automatically. Manual setup:

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

List all detected models.

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
| `max_tokens` | int | `2048` | Max completion tokens |
| `temperature` | float | *(ignored)* | Accepted but has no effect — rkllm uses model-compiled sampling |
| `top_p` | float | *(ignored)* | Accepted but has no effect |
| `frequency_penalty` | float | *(ignored)* | Accepted but has no effect |
| `presence_penalty` | float | *(ignored)* | Accepted but has no effect |
| `stream_options.include_usage` | bool | `false` | Include token counts in stream |

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
  -e WEB_SEARCH_ENGINE=searxng \
  -e SEARXNG_QUERY_URL=http://host.docker.internal:8080/search?q=<query> \
  -e WEB_SEARCH_RESULT_COUNT=5 \
  -e WEB_SEARCH_CONCURRENT_REQUESTS=3 \
  -e BYPASS_WEB_SEARCH_WEB_LOADER=True \
  -e BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True \
  -e FILE_IMAGE_COMPRESSION_WIDTH=448 \
  -e FILE_IMAGE_COMPRESSION_HEIGHT=448 \
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
| `RAG_RELEVANCE_THRESHOLD` | `0.0` | **Critical.** Must be `0.0` — higher values filter out valid results because cross-encoder sigmoid scores are often below 0.1 for cross-lingual or loosely-related content. The reranker handles quality filtering instead |
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
| `WEB_SEARCH_CONCURRENT_REQUESTS` | `3` | Limits concurrent web search requests to 3 — prevents overwhelming SearXNG while keeping searches fast |
| `BYPASS_WEB_SEARCH_WEB_LOADER` | `True` | Uses search engine snippets instead of scraping full pages — cleaner, faster, and more reliable for small models |
| `BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL` | `True` | Sends search snippets directly to the model without embedding/retrieving — the API server builds its own optimized prompt internally |

**File Upload / Image Compression:**

| Variable | Value | Reason |
|----------|-------|--------|
| `FILE_IMAGE_COMPRESSION_WIDTH` | `448` | Compresses uploaded images to 448px width — matches the VL model input resolution |
| `FILE_IMAGE_COMPRESSION_HEIGHT` | `448` | Compresses uploaded images to 448px height — reduces upload size without quality loss for vision tasks |

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
| **Database scripts** (`tests/set_model_prompts.py`, `tests/fix_owui_models.py`) | Directly write to `webui.db` inside the container | Yes (data on Docker volume) | No (must re-run) |
| **Admin UI only** | No env var or script — must configure manually | Yes (data on Docker volume) | No (must redo) |

**Settings hardcoded via Docker env vars** (auto-restore on fresh install):

| Category | Settings |
|----------|----------|
| Connection | API base URL, API key |
| RAG pipeline | Embedding model, reranking model, batch size, async embedding, hybrid search, BM25 weight, enriched texts, relevance threshold, top_k, top_k_reranker, custom RAG template |
| Chunking | Chunk size, overlap, min size target, markdown header splitter |
| Document processing | PDF image extraction via OCR (`PDF_EXTRACT_IMAGES=True`) — extracts text from scanned images inside PDFs |
| Web search | Engine, SearXNG URL, result count, concurrent requests, bypass modes |
| File upload | Image compression (448×448 for VL model) |
| Code execution | Disabled (`ENABLE_CODE_EXECUTION=False`) — small NPU models (1.7B–4B) generate unreliable code |
| Code interpreter | Disabled (`ENABLE_CODE_INTERPRETER=False`) — same reason; wastes time or produces wrong results |
| Features | Channels, memories, notes |
| Privacy | Telemetry disabled |

**Settings hardcoded via database scripts** (must re-run after volume reset):

| Script | What it sets |
|--------|-------------|
| `tests/set_model_prompts.py` | System prompt on all models (date/time context) |
| `tests/fix_owui_models.py` | Model capability flags (vision, image_gen, code_interpreter, etc.) |

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
docker cp tests/fix_owui_models.py open-webui:/tmp/
docker exec open-webui python3 /tmp/fix_owui_models.py

# 3. Re-configure Admin UI-only settings manually (domain filters, model order, prompt suggestions)
```

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

The embedding model determines how well Open WebUI finds the right document chunks when you ask a question. This runs on CPU (not NPU), so it needs to be small enough for ARM hardware.

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
| **Relevance Threshold** | `0` | `RAG_RELEVANCE_THRESHOLD=0.0` | **Must be 0.** Cross-encoder sigmoid scores are often below 0.1 for valid content — any threshold filters out real results. Let the reranker handle quality |

> **RAG Template:** Use the **default template** (clear the field) — it includes inline citation support with `[id]` format and comprehensive guidelines. The API server's RAG pipeline works with the default template.

> **Image Compression:** Set to `448x448` to match the VL encoder input resolution.

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

When Open WebUI performs a web search or retrieves document chunks, the results are injected into the system message as `<source>` tags (or via a custom RAG template). The server detects this and activates a specialized RAG pipeline:

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
8. **Summarization boost** — When the query contains "summarize", "summary", "overview", or "outline", a stronger instruction is appended: *"Cover all major points, sections, and key details. Use multiple paragraphs."*

### Follow-Up Detection (3 Layers)

Open WebUI searches SearXNG with the raw user message. Short follow-ups produce garbage results:

| Layer | Trigger | Example |
|-------|---------|---------|
| Layer 1: Word list | Exact match to known conversational words | "yes", "thanks", "tell me more" |
| Layer 2: Short query | ≤3 words with conversation history | "south africa", "another one" |
| Layer 3: Topical overlap | < 30% query keywords found in reference text | Off-topic search results |

When any layer fires, RAG is skipped and the model uses normal conversation mode.

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
- **Context-dependent thinking for RAG**: On small context models (< 8k), thinking is disabled via `enable_thinking = false` to save tokens for the actual answer

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
2. **Follow-up turns** — The server computes the hash of the conversation prefix. If it matches the previous turn's hash (same conversation, same model), only the new user message is sent with `keep_history=1`. The NPU appends to the existing KV cache.
3. **New conversation** — Hash mismatch triggers `rkllm_clear_kv_cache()` + full prompt resend.
4. **RAG queries** — Always use `keep_history=0` (standalone, no history needed).

This makes multi-turn conversations significantly faster — Turn 2+ take ~50ms to prefill regardless of total conversation length.

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
| `MONITOR_INTERVAL` | 10s | Health check / idle monitoring frequency |
| `IDLE_UNLOAD_TIMEOUT` | 300s | Auto-unload model after idle (0 to disable) |

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
- Check that `librkllmrt.so` is in `/usr/lib/`: `ldconfig -p | grep rkllm`
- Verify NPU driver is loaded: `dmesg | grep -i npu`
- Check `api.log` for init failure messages — may indicate corrupt `.rkllm` file or version mismatch

### "Another request is currently being processed" (503)
- NPU is single-task — only one request at a time
- Previous request may be stuck — check `/health` endpoint
- Stale requests auto-clear after 30s idle

### Streaming stops mid-response
- Check `FALLBACK_SILENCE` timeout (default 12s) — increase if model is slow
- Large prompts near context limit may cause long prefill — increase `FIRST_TOKEN_TIMEOUT`
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

Section-by-section diagnostic covering 17 areas of the codebase — **108 tests total**. Designed for copy-paste output analysis.

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

### Test Results (Orange Pi 5 Plus, Feb 2026)

| Suite | Total | Pass | Fail | Warn | Time |
|---|---|---|---|---|---|
| `tests/diagnostic_test.py` | 108 | 108 | 0 | 0 | ~12 min |
| `tests/vl_test.py` | 68 | 68 | 0 | 0 | ~8 min |
| `tests/e2e_test.py` | 85 | 85 | 0 | 0 | ~25 min |
| `tests/deep_diagnostic.py` | 72 | 71 | 0 | 1 | ~18 min |

### End-to-End Test (`tests/e2e_test.py`)

Full integration test that exercises every model with real inference — **85 checks across 9 sections**, run against all 4 text models. Covers streaming, non-streaming, shortcircuits, RAG, cache, web search, KV multi-turn, per-model system prompts, database queries, and API compliance.

```bash
python tests/e2e_test.py                # Run all 9 sections against all models
python tests/e2e_test.py --section 3    # Run only section 3
```

| Section | Coverage |
|---|---|
| 1 | Health, models endpoint, model listing |
| 2 | Per-model text generation (streaming + non-streaming, all 4 models) |
| 3 | Shortcircuit detection (title gen, tag gen, query gen) |
| 4 | RAG pipeline with real document context |
| 5 | Response cache (hit/miss timing) |
| 6 | Web search result handling |
| 7 | KV cache multi-turn memory |
| 8 | Per-model system prompts |
| 9 | Database / API compliance edge cases |

### Deep Diagnostic Test (`tests/deep_diagnostic.py`)

Targeted deep-dive into 12 under-tested areas identified via gap analysis — **72 checks** covering protocol compliance, edge cases, and stress scenarios.

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

---

## File Structure

```
RKLLM-API-Server/
├── api.py                          # Main API server (ctypes, v2.0)
├── docker-compose.yml              # Open WebUI Docker config (all settings hardcoded)
├── setup.sh                        # Zero-config installer (761 lines)
├── settings.yml                    # SearXNG configuration for Open WebUI
├── README.md                       # This file
├── tests/
│   ├── diagnostic_test.py          # Section-by-section diagnostic (17 sections, 108 tests)
│   ├── e2e_test.py                 # End-to-end integration (9 sections, 85 checks, all models)
│   ├── deep_diagnostic.py          # Deep diagnostic (12 sections, 72 checks, edge cases)
│   ├── vl_test.py                  # Integration test suite (17 categories, 68 tests)
│   ├── benchmark_test.py           # NPU model benchmark tool (tok/s, TTFT, memory)
│   ├── benchmark_results.json      # Latest benchmark results
│   ├── set_model_prompts.py        # Set system prompts on all OWUI models (DB script)
│   ├── fix_owui_models.py          # Set model capabilities: vision, tools, etc. (DB script)
│   ├── remove_stale_models.py      # Mark old/removed models as inactive in OWUI DB
│   ├── dump_owui_models_quick.py   # Quick dump of all OWUI model records
│   └── dump_owui_settings.py       # Dump all OWUI admin settings from DB
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
| **Code size** | 2682 lines | ~3200 lines (text + VL + RAG) |

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

> Pre-built RKLLM models available on HuggingFace: [Qwen3-1.7B-RKLLM-v1.2.3](https://huggingface.co/GatekeeperZA/Qwen3-1.7B-RKLLM-v1.2.3) · [Phi-3-mini-4k-instruct-w8a8](https://huggingface.co/GatekeeperZA/Phi-3-mini-4k-instruct-w8a8) · [Qwen3-VL-2B-Instruct-RKLLM-v1.2.3](https://huggingface.co/GatekeeperZA/Qwen3-VL-2B-Instruct-RKLLM-v1.2.3)

### VL (Vision-Language) Models

| Model | Quantization | Encoder Res | Decode Speed | Encoder Time | Peak RAM | Status |
|-------|-------------|-------------|-------------|-------------|----------|--------|
| **Qwen3-VL-2B** | W8A8 | **672×672** | ~15 tok/s | ~4s | ~6.5 GB | **Active (recommended)** |
| Qwen3-VL-2B | W8A8 | 448×448 | ~15 tok/s | ~2s | ~5.5 GB | Available (default export) |
| Qwen3-VL-2B | W8A8 | 896×896 | ~15 tok/s | ~12s | ~8.5 GB | Available (high-detail) |
| InternVL3.5-2B | W8A8 | 448×448 | ~12.1 tok/s | ~2.0s | ~3.0 GB | **Tested — poor OCR accuracy** |
| DeepSeekOCR-3B | W8A8 | 448×448 | ~31.8 tok/s | ~2.1s | ~3.0 GB | **Tested — severe hallucination** |
| Qwen2.5-VL-3B | W8A8 | 392×392 | ~8.7 tok/s | ~2.9s | ~5.3 GB | Supported (lower resolution) |
| Qwen2-VL-2B | W8A8 | 392×392 | ~16.6 tok/s | ~3.3s | ~3.0 GB | Supported |
| InternVL3-1B | W8A8 | 448×448 | ~TBD | ~TBD | ~TBD | Supported |
| MiniCPM-V-2.6 | W8A8 | 448×448 | ~TBD | ~TBD | ~TBD | Supported |

> **Qwen3-VL-2B is the recommended VL model** with the vision encoder re-exported at **672×672** for 2.25× more visual detail than the default 448×448. Three encoder resolutions (448/672/896) are available — see [Vision Encoder Resolution Comparison](#vision-encoder-resolution-comparison). To switch encoders, rename the `.rknn` files (only one should have the `.rknn` extension; others use `.rknn.alt`).
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

## VL Model Evaluation

We tested all available pre-converted VL models to find the best option for real-world OCR tasks (reading gas meters from phone photos). All models use the same 448×448 (or lower) vision encoder, which crushes high-resolution phone photos (14-15 MP) down to a tiny thumbnail — a fundamental bottleneck for OCR accuracy.

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

## Git Tags & Branches

| Tag / Branch | Description |
|---|---|
| `v1.0-subprocess-stable` | Last working subprocess version (V1) |
| `v1.1-ctypes-text-only` | Text-only ctypes version before VL additions |
| `subprocess-legacy` | Branch preserving the subprocess architecture |
| `main` | Current: ctypes + VL multimodal + meta-task shortcircuits + context-enriched query gen + document RAG + NPU benchmarks + full test suites (333 checks, 0 failures) |

---

## License

This project is provided as-is for personal and educational use. The rkllm runtime and model files are subject to their respective licenses from Rockchip and model authors.

## Acknowledgements

- [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm) — RKLLM runtime, toolkit, and multimodal demo
- [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) — RKNN runtime (`librknnrt.so`) for vision encoder NPU inference
- [Pelochus/armbian-build-rknpu-updates](https://github.com/Pelochus/armbian-build-rknpu-updates) — Armbian builds with RKNPU driver
- [Open WebUI](https://github.com/open-webui/open-webui) — Web interface for LLM interaction
