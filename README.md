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
  - [Document RAG Settings](#document-rag-settings-recommended-for-pdfdocument-upload)
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
- **Query generation shortcircuit** — Open WebUI asks the model to generate search queries for retrieval; instead of wasting 5s of inference, the server extracts the user's actual question from the chat history and returns it as the query instantly (~0ms)
- **Title generation shortcircuit** — extracts the first user message as the chat title (~0ms instead of 5-10s inference)
- **Tag generation shortcircuit** — returns a default tag instantly (~0ms instead of 5-10s inference)
- **Meta-task thinking disabled** — auto-detects Open WebUI internal tasks (query gen, title gen, tags, autocomplete) and disables `<think>` reasoning to avoid wasting 20+ seconds on trivial tasks
- **No JSON leakage** — query generation shortcircuit prevents raw JSON from appearing in the chat display

### Standards Compliance
- **`stream_options.include_usage`** — streaming token counts per OpenAI spec
- **`system_fingerprint`** in all responses
- **`max_tokens` / `max_completion_tokens`** support
- **Request body size limit** (16 MB)
- **Proper error responses** matching OpenAI error format

### Vision-Language (VL) / Multimodal
- **Dual-model architecture** — text model (e.g. Qwen3-1.7B) + VL model (e.g. DeepSeekOCR-3B) loaded simultaneously
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

5. **Dual-model VL** — Text and VL models are loaded simultaneously into separate `RKLLMWrapper` instances. The vision encoder runs on a third ctypes binding (`librknnrt.so`). Image requests are auto-routed to the VL pipeline; text requests use the primary model. A shared `_token_queue` serialized by `PROCESS_LOCK` prevents interleaving.

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
└── DeepSeekOCR-3B/                       # VL model (text + vision)
    ├── DeepSeekOCR-3B-w8a8-rk3588.rkllm  # LLM decoder
    └── DeepSeekOCR-3B-vision-encoder.rknn # Vision encoder
```

**How it works:**
1. The server auto-detects `.rknn` files alongside `.rkllm` files
2. The folder name is matched against `VL_MODEL_CONFIGS` (supports DeepSeekOCR, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, InternVL3, MiniCPM)
3. When a chat request includes an image (base64 `image_url`), it auto-routes to the VL model
4. Text-only requests continue using the text model normally

**Supported VL models** (model folder name must contain):
| Pattern | Model Family |
|---------|-------------|
| `deepseekocr` | DeepSeekOCR (recommended for OCR tasks) |
| `qwen3-vl` | Qwen3-VL |
| `qwen2.5-vl` | Qwen2.5-VL |
| `qwen2-vl` | Qwen2-VL |
| `internvl3` | InternVL3 |
| `minicpm` | MiniCPM-V |

**Requirements:**
- `numpy` and `Pillow` Python packages (installed by `setup.sh`)
- `librknnrt.so` (RKNN runtime library, usually at `/usr/lib/librknnrt.so`)
- Sufficient RAM for both models (~5.5 GB for Qwen3-1.7B + DeepSeekOCR-3B)

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
    "model": "deepseekocr-3b",
    "encoder_loaded": true,
    "llm_loaded": true
  },
  "active_request": null,
  "models_available": 3
}
```

> The `vl_model` field is `null` when no VL model is loaded.

---

## Open WebUI Configuration

### Connection

In Open WebUI **Admin > Settings > Connections**, add the API as an OpenAI-compatible endpoint:

| Setting | Value |
|---------|-------|
| API Base URL | `http://<device-ip>:8000/v1` |
| API Key | *(any non-empty string — the server has no auth)* |

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

### System Prompt (Required for All NPU Models)

**Workspace > Models > Edit** each NPU model, and set the **System Prompt** to:

```
Today is {{CURRENT_DATE}} ({{CURRENT_WEEKDAY}}), {{CURRENT_TIME}}.
```

> **Why this is required:** NPU models have no built-in awareness of the current date or time. Without this, any question like "what day is it?" gets a hallucinated answer. Open WebUI replaces the `{{...}}` variables with live values before sending the request.

> **Do NOT add generic instructions** like "You are a helpful assistant" — these get sent as part of the user prompt and cause the model to respond with a greeting instead of answering the question.

### Web Search (Required for RAG/SearXNG)

**Admin > Settings > Web Search:**

| Setting | Value | Reason |
|---------|-------|--------|
| Search Engine | `searxng` | Self-hosted, JSON API |
| SearXNG Query URL | `http://searxng:8080/search` | Docker service name |
| Web Search Result Count | `3` (4k models) / `5` (16k models) | Balances coverage vs. context |
| **Bypass Web Loader** | **ON** ⚠️ | **Required.** Snippets are cleaner than raw page scraping for small models |
| **Bypass Embedding and Retrieval** | **ON** ⚠️ | **Required.** No embedding model available on ARM/NPU. Sends docs directly to the model |

### Documents / RAG Template (Required for SearXNG)

**Admin > Settings > Documents:**

| Setting | Value | Reason |
|---------|-------|--------|
| **RAG Template** | `{{CONTEXT}}` | **Required.** Just the variable, nothing else. The default template wastes 300+ tokens of meta-instructions |

> **Important:** The RAG Template must be exactly `{{CONTEXT}}` — no extra text. The API server builds its own optimized reading-comprehension prompt format internally.

### Document RAG Settings (Recommended for PDF/Document Upload)

**Admin > Settings > Documents:**

These settings control how Open WebUI chunks, embeds, and retrieves uploaded documents. The defaults are too aggressive for small models — these values are tuned for 1.5-4B parameter models on constrained hardware:

| Setting | Value | Reason |
|---------|-------|--------|
| **Chunk Size** | `1500` | Larger chunks give the model more context per retrieval hit |
| **Chunk Overlap** | `200` | Prevents important information from being split between chunks |
| **Min Chunk Size** | `200` | Filters out tiny useless fragments |
| **Top K** | `5` | Retrieves 5 chunks — balances coverage vs context limit |
| **Full Context Mode** | **OFF** | Injecting the entire document overflows the 4K context window |
| **Hybrid Search** | **ON** ⚠️ | **Recommended.** Combines semantic + keyword search for much better retrieval |
| **Enrich Hybrid Search Text** | **ON** | Adds surrounding context to search results |
| **BM25 Weight** | `0.5` | Equal weighting of keyword (BM25) and semantic search |
| **Relevance Threshold** | `0` | Let the model see all retrieved chunks rather than filtering too aggressively |

> **RAG Template** should be `{{CONTEXT}}` (set in the Documents section above) — the API server builds its own optimized prompt internally.

> **Image Compression:** If uploading documents with images, set the compression resolution to `448x448` to match typical VL encoder input sizes.

### Per-Model Capabilities (Required)

**Workspace > Models > Edit > Capabilities** — configure for **each** NPU model:

| Setting | Value | Reason |
|---------|-------|--------|
| **Builtin Tools** | **OFF** ⚠️ | **Required.** Small NPU models (1.5B-4B) cannot do function-calling. Leaving this on injects tool-use instructions that confuse the model |
| File Context | **ON** | Enables document and search result injection |

### Interface Settings (Recommended)

**Admin > Settings > Interface > Generation Settings:**

| Setting | Value | Reason |
|---------|-------|--------|
| **Show Generation Settings** | **OFF** | The RKLLM runtime handles sampling internally. UI sliders are ignored by the API |

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

Two comprehensive test suites verify every code path against a live server. Both use only Python stdlib (`urllib`, `json`, `base64`) — no extra dependencies.

### Diagnostic Test (`diagnostic_test.py`)

Section-by-section diagnostic covering 17 areas of the codebase — **108 tests total**. Designed for copy-paste output analysis.

```bash
python diagnostic_test.py                # Run all 17 sections
python diagnostic_test.py --skip-vl      # Skip VL tests (faster)
python diagnostic_test.py --section 4    # Run only section 4
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

### Integration Test (`vl_test.py`)

Focused integration tests across 17 categories — **68 assertions**. Tests text generation, VL multimodal, streaming, error handling, model lifecycle, and concurrent rejection.

```bash
python vl_test.py all          # Run all tests
python vl_test.py complete     # Non-streaming tests only
python vl_test.py stream       # Streaming tests only
```

### Test Results (Orange Pi 5 Plus, Feb 2026)

| Suite | Total | Pass | Fail | Time |
|---|---|---|---|---|
| `diagnostic_test.py` | 108 | 108 | 0 | ~12 min |
| `vl_test.py` | 68 | 68 | 0 | ~8 min |

Both suites target `http://192.168.2.180:8000` by default — edit the `API` constant at the top to match your server IP.

---

## File Structure

```
RKLLM-API-Server/
├── api.py                          # Main API server (ctypes, v2.0)
├── diagnostic_test.py              # Section-by-section diagnostic (17 sections, 108 tests)
├── vl_test.py                      # Integration test suite (17 categories, 68 tests)
├── setup.sh                        # Zero-config installer (761 lines)
├── settings.yml                    # SearXNG configuration for Open WebUI
├── README.md                       # This file
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
| Qwen3-1.7B | W8A8 | 4K | ~1.7 GB | ~13.6 tok/s | Fully tested |
| Qwen3-4B-Instruct | W8A8 | 16K | ~4 GB | ~6 tok/s | Tested |
| Gemma-3-4B-IT | W8A8 | 4K | ~4 GB | ~6 tok/s | Tested |
| Phi-3-Mini-4K-Instruct | W8A8 | 4K | ~3.8 GB | ~6.8 tok/s | Tested |

### VL (Vision-Language) Models

| Model | Quantization | Img Encoder | Decode Speed | Encoder Time | Status |
|-------|-------------|-------------|-------------|-------------|--------|
| DeepSeekOCR-3B | W8A8 | 448×448 | ~31.8 tok/s | ~2.1s | Supported |
| Qwen2.5-VL-3B | W8A8 | 392×392 | ~8.7 tok/s | ~2.9s | Supported |
| Qwen2-VL-2B | W8A8 | 392×392 | ~16.6 tok/s | ~3.3s | Supported |
| Qwen3-VL-2B | W8A8 | varies | ~TBD | ~TBD | Supported |
| InternVL3-1B | W8A8 | 448×448 | ~TBD | ~TBD | Supported |
| MiniCPM-V-2.6 | W8A8 | 448×448 | ~TBD | ~TBD | Supported |

> Encoder times and decode speeds from [RKLLM official benchmarks](https://github.com/airockchip/rknn-llm/blob/main/benchmark.md) on RK3588 W8A8 with all 3 NPU cores.

---

## Git Tags & Branches

| Tag / Branch | Description |
|---|---|
| `v1.0-subprocess-stable` | Last working subprocess version (V1) |
| `v1.1-ctypes-text-only` | Text-only ctypes version before VL additions |
| `subprocess-legacy` | Branch preserving the subprocess architecture |
| `main` | Current: ctypes + VL multimodal + meta-task shortcircuits + document RAG + full test suites (108/108 pass) |

---

## License

This project is provided as-is for personal and educational use. The rkllm runtime and model files are subject to their respective licenses from Rockchip and model authors.

## Acknowledgements

- [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm) — RKLLM runtime, toolkit, and multimodal demo
- [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) — RKNN runtime (`librknnrt.so`) for vision encoder NPU inference
- [Pelochus/armbian-build-rknpu-updates](https://github.com/Pelochus/armbian-build-rknpu-updates) — Armbian builds with RKNPU driver
- [Open WebUI](https://github.com/open-webui/open-webui) — Web interface for LLM interaction
