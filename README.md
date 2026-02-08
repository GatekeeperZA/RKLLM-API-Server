# RKLLM API Server

An OpenAI-compatible API server for running LLMs on Rockchip RK3588 NPU hardware. Drop in `.rkllm` model files, run the setup script, and connect from [Open WebUI](https://github.com/open-webui/open-webui) or any OpenAI-compatible client.

Built for the **Orange Pi 5 Plus** (16 GB RAM) but works on any RK3588-based board with the RKNPU driver.

## Features

- **OpenAI-compatible API** — works with Open WebUI, LiteLLM, or any OpenAI client
- **Direct NPU access** via ctypes binding to `librkllmrt.so` (no subprocess overhead)
- **KV cache incremental mode** — follow-up turns only prefill the new message (~50ms vs ~500ms)
- **Multi-model support** — auto-detects all `.rkllm` files in `~/models`, with automatic aliases
- **Built-in RAG pipeline** — cleans web content, scores paragraphs, deduplicates, caches responses
- **Streaming & non-streaming** — full SSE support with `<think>` tag parsing for reasoning models
- **Zero-config setup** — one script installs everything and creates a systemd service
- **Auto-unload** — idle models are freed after 5 minutes to reclaim NPU memory
- **Clean abort** — native `rkllm_abort()` for instant cancellation (no SIGKILL)

## Requirements

| Component | Requirement |
|-----------|-------------|
| Board | RK3588-based (Orange Pi 5 Plus, Rock 5B, etc.) |
| OS | Armbian / Ubuntu (aarch64) |
| RAM | 8 GB minimum, 16 GB recommended |
| NPU Driver | RKNPU driver 0.9.8+ ([Pelochus Armbian builds](https://github.com/Pelochus/armbian-build-rknpu-updates/releases)) |
| Runtime | `librkllmrt.so` v1.2.3 from [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm) |
| Models | `.rkllm` files converted with rkllm-toolkit v1.2.3 |

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/GatekeeperZA/RKLLM-API-Server.git
cd RKLLM-API-Server

# 2. Run the setup script (installs everything)
chmod +x setup.sh
./setup.sh

# 3. Place your .rkllm model files
mkdir -p ~/models/Qwen3-1.7B
cp /path/to/Qwen3-1.7B-w8a8-rk3588.rkllm ~/models/Qwen3-1.7B/

# 4. Start the server
source .venv/bin/activate
gunicorn -w 1 -k gthread --threads 4 --timeout 300 -b 0.0.0.0:8000 api:app
```

The setup script handles: system packages, RKNPU driver check, rkllm runtime installation, Python venv creation, NPU frequency fix, and systemd service configuration. It is idempotent — safe to run multiple times.

## Model Setup

Place each model in its own folder under `~/models`:

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

The server auto-detects models on startup and generates short aliases:
- `qwen3-1.7b` → `qwen3-1.7b` (already short)
- `qwen3-4b-instruct-2507` → `qwen3-4b`, `qwen3-4b-instruct`
- `gemma-3-4b-it` → `gemma`, `gemma-3`, `gemma-3-4b`
- `phi-3-mini-4k-instruct` → `phi`, `phi-3`, `phi-3-mini`

Context length is auto-detected from filenames (e.g., `16k` in the name → 16384 tokens).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming & non-streaming) |
| `GET`  | `/v1/models` | List available models |
| `POST` | `/v1/models/select` | Pre-load a model (warm-up) |
| `POST` | `/v1/models/unload` | Unload current model to free NPU memory |
| `GET`  | `/health` | Server health check |

### Chat Completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-1.7b",
    "stream": true,
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

### Open WebUI Connection

In Open WebUI settings → Connections → OpenAI API:
- **URL**: `http://<orange-pi-ip>:8000/v1`
- **API Key**: `not-needed` (any non-empty string)

## Architecture

```
┌──────────────┐     HTTP/SSE      ┌──────────────────────┐
│  Open WebUI  │ ◄──────────────── │   api.py (Flask)     │
│  or any      │ ─────────────────►│   gunicorn gthread   │
│  OpenAI      │                   │                      │
│  client      │                   │  ┌────────────────┐  │
└──────────────┘                   │  │  KV Cache       │  │
                                   │  │  Tracking       │  │
                                   │  └────────┬───────┘  │
                                   │           │ ctypes    │
                                   │  ┌────────▼───────┐  │
                                   │  │ librkllmrt.so  │  │
                                   │  │ (C library)    │  │
                                   │  └────────┬───────┘  │
                                   │           │          │
                                   │  ┌────────▼───────┐  │
                                   │  │  RK3588 NPU    │  │
                                   │  │  (3 cores)     │  │
                                   │  └────────────────┘  │
                                   └──────────────────────┘
```

### How It Works

1. **Flask** receives OpenAI-format requests on port 8000
2. **Prompt builder** converts the messages array to plain text (the rkllm runtime applies chat templates internally using actual token IDs)
3. **KV cache tracker** decides: send only the new message (incremental) or clear cache and send the full conversation (reset)
4. **Worker thread** calls `rkllm_run()` (blocking C function) while the main thread reads tokens from a `queue.Queue`
5. **Callback** pushes tokens from the C library to the queue, which are yielded as SSE chunks
6. **ThinkTagParser** routes `<think>...</think>` content to `reasoning_content` and visible text to `content`

### KV Cache Strategy

The NPU runtime maintains an internal KV cache. With `keep_history=1`, prior conversation turns are preserved, so follow-up messages only need to prefill the new tokens:

| Scenario | Strategy | Prefill Time | What's Sent |
|----------|----------|-------------|-------------|
| New conversation | `clear_kv_cache()` + `keep_history=1` | ~90ms (full) | Full prompt |
| Follow-up turn | `keep_history=1` | ~50ms (incremental) | Only new user message |
| RAG query | `keep_history=0` | ~90ms (full) | RAG context + question |
| Model switch | New model loaded | ~90ms (full) | Full prompt |

## Configuration

Key settings in `api.py` (top of file):

| Setting | Default | Description |
|---------|---------|-------------|
| `MODELS_ROOT` | `~/models` | Directory to scan for `.rkllm` files |
| `MAX_TOKENS_DEFAULT` | `2048` | Default max generation tokens |
| `CONTEXT_LENGTH_DEFAULT` | `4096` | Fallback context length if not detected from filename |
| `IDLE_UNLOAD_TIMEOUT` | `300` | Seconds before idle model is auto-unloaded (0 = disabled) |
| `GENERATION_TIMEOUT` | `600` | Max total generation time |
| `FIRST_TOKEN_TIMEOUT` | `120` | Max wait for first token (includes prefill) |
| `RAG_CACHE_TTL` | `300` | RAG response cache lifetime in seconds |
| `RAG_MIN_QUALITY_SCORE` | `2` | Minimum paragraph score for RAG inclusion |
| `DISABLE_THINK_FOR_RAG_BELOW_CTX` | `8192` | Disable thinking for RAG on small-context models |

Environment variables:
- `RKLLM_LIB_PATH` — Override library path (auto-detected by default)
- `RKLLM_API_LOG_LEVEL` — Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`

## RAG Pipeline

When Open WebUI sends web search results in the system message, the server automatically:

1. **Extracts** reference data from the system prompt (detects Open WebUI's RAG format)
2. **Cleans** web content — strips navigation, cookie banners, boilerplate (4-pass filtering)
3. **Scores** paragraphs by relevance to the user's question (word overlap, data signals, sentence structure)
4. **Deduplicates** similar paragraphs (Jaccard similarity threshold)
5. **Truncates** to fit within context length while keeping the highest-quality paragraphs
6. **Caches** responses for 5 minutes (avoids re-generating identical RAG answers)
7. **Detects follow-ups** — short replies like "tell me more" or "why?" skip RAG and use conversation context instead

## Running as a Service

The setup script creates a systemd service automatically. Manual management:

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

### Important: Gunicorn Settings

```bash
gunicorn -w 1 -k gthread --threads 4 --timeout 300 -b 0.0.0.0:8000 api:app
```

- **`-w 1`** — Single worker only. The NPU can only load one model at a time.
- **`-k gthread`** — Must use gthread, NOT gevent. `rkllm_run()` is a blocking C call that freezes gevent's event loop.
- **`--threads 4`** — Allows concurrent request handling (one generates, others queue).
- **`--timeout 300`** — Large models can take time to load.

## SearXNG Configuration

The included `settings.yml` is an optimized SearXNG configuration for use with Open WebUI's web search feature. It whitelists only fast, reliable search engines (Google, DuckDuckGo, Bing, Brave, Wikipedia) and tunes timeouts for the RK3588.

```bash
# Install: copy to your SearXNG docker instance
cp settings.yml ~/Downloads/searxng-docker/searxng/settings.yml
cd ~/Downloads/searxng-docker && docker compose down && docker compose up -d
```

## File Structure

```
├── api.py                          # Main API server (ctypes, v2.0)
├── setup.sh                        # Zero-config installer (761 lines)
├── settings.yml                    # SearXNG configuration for Open WebUI
├── README.md                       # This file
├── CTYPES_MIGRATION_PLAN.md        # Migration planning document
├── archive/
│   └── api_v1_subprocess.py        # Original subprocess version (archived)
└── .gitignore
```

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
| **Code size** | 2682 lines | 2376 lines (−306, despite adding features) |

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

## Tested Hardware

| Board | RAM | NPU Driver | Runtime | Status |
|-------|-----|-----------|---------|--------|
| Orange Pi 5 Plus | 16 GB | 0.9.8 | v1.2.3 | Fully tested, production use |

## Tested Models

| Model | Quantization | Context | File Size | Status |
|-------|-------------|---------|-----------|--------|
| Qwen3-1.7B | W8A8 | 4K | ~1.7 GB | Fully tested |
| Qwen3-4B-Instruct | W8A8 | 16K | ~4 GB | Tested |
| Gemma-3-4B-IT | W8A8 | 4K | ~4 GB | Tested |
| Phi-3-Mini-4K-Instruct | W8A8 | 4K | ~3.8 GB | Tested |

## License

This project is provided as-is for personal and educational use with Rockchip NPU hardware.

## Acknowledgements

- [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm) — RKLLM runtime and toolkit
- [Pelochus/armbian-build-rknpu-updates](https://github.com/Pelochus/armbian-build-rknpu-updates) — Armbian builds with RKNPU driver
- [Open WebUI](https://github.com/open-webui/open-webui) — Web interface for LLM interaction
