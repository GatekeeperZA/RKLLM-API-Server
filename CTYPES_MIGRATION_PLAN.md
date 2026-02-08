# RKLLM API Server — ctypes Migration Plan

**Date:** February 8, 2026
**Repository:** https://github.com/GatekeeperZA/RKLLM-API-Server
**Rollback Tag:** `v1.0-subprocess` (pushed to origin)
**Backup File:** `api_subprocess_backup.py` (2322 lines, 122798 bytes, verified intact)

---

## Session History

### 1. Starting Point — Multi-Turn Conversation Visibility
User asked if the README documented multi-turn conversation history support.
A dedicated section was added and committed (commit `2615474`).

### 2. KV Cache Question
User asked: *"Is there caching to speed up follow-up responses?"*

**Finding:** No KV cache persistence exists. The subprocess-based architecture
re-processes the ENTIRE conversation from scratch on every turn. The C++ binary
(`llm_demo.cpp`) hardcodes `keep_history = 0`, which discards the KV cache after
each `rkllm_run()` call. Process reuse only saves the ~10s model load time, NOT
the prefill computation.

**Impact on a 4-turn conversation (4096 ctx):**
| Turn | Tokens to prefill | Time (~250 tok/s) |
|------|-------------------|-------------------|
| 1    | 50                | 200ms             |
| 2    | 261               | 1044ms            |
| 3    | 661               | 2644ms            |
| 4    | 1061              | 4244ms            |

With `keep_history=1`, turn 4 would only prefill the ~50 NEW tokens (~200ms) —
a **~20x speedup** on follow-up turns.

### 3. Research Request
User asked: *"Is this the most efficient way? Can you do some research on this?"*

**Research performed:**
- Fetched `rkllm.h` header from `airockchip/rknn-llm` GitHub repository
- Studied `flask_server.py` and `gradio_server.py` official demo code
- Read `llm_demo.cpp` C++ demo for API usage patterns
- Analyzed all struct definitions, function signatures, and callback patterns

**Key discoveries in the rkllm C API:**

1. **`RKLLMInferParam.keep_history`** — `1` = keep KV cache between calls,
   `0` = discard. The binary hardcodes `0`.

2. **`rkllm_clear_kv_cache(handle, keep_system, system_prompt, system_prompt_len)`**
   — Clear KV cache while optionally keeping system prompt cached.

3. **`rkllm_abort(handle)`** — Gracefully abort running inference (replaces
   our SIGKILL approach).

4. **`RKLLMInput.role`** — Native role field (`"user"`, `"system"`, `"tool"`)
   for proper multi-turn support. The runtime applies chat templates internally
   using actual token IDs.

5. **`RKLLMResult.perf`** — Real performance stats: `prefill_time_ms`,
   `prefill_tokens`, `generate_time_ms`, `generate_tokens`, `memory_usage_mb`.
   Replaces all our stdout stats parsing.

6. **`rkllm_load_prompt_cache()` / `RKLLMPromptCacheParam`** — Disk-based
   prompt caching for system prompts.

7. **`RKLLMInput.enable_thinking`** — Native thinking mode toggle for Qwen3.

**Three options presented:**

| Option | Approach | Speedup | Effort |
|--------|----------|---------|--------|
| 1      | ctypes migration (direct C API binding) | ~20x prefill | High |
| 2      | Prompt cache (disk-based) | ~5-10x first turn only | Medium |
| 3      | Modify C++ binary | ~20x prefill | Medium |

### 4. User Decision
User chose **Option 1 (ctypes migration)** and requested:
> *"Do extensive research to make sure everything will still work, add a
> rollback point in git in case we need to rollback, then implement."*

### 5. Deep Research Phase

#### 5.1 Current api.py Architecture Analysis
Read entire 2682-line file. Identified sections to preserve vs. replace:

**PRESERVE (unchanged logic):**
- Configuration constants (lines 1-120)
- RAG response cache + LRU logic (lines 180-245)
- Logging setup (lines 250-270)
- Model detection & auto-aliases (lines 275-395)
- Request tracking (lines 418-530)
- `_clean_web_content()` — 4-pass web scraping cleanup (lines 610-730)
- `_score_paragraph()` — jusText-inspired scoring (lines 730-830)
- `_strip_system_fluff()`, `_is_date_only_system()`, `_extract_rag_reference()` (lines 830-970)
- `build_prompt()` — entire RAG pipeline + follow-up detection (lines 971-1390)
- `ThinkTagParser` class + `parse_think_tags()` (lines 2010-2115)
- SSE helpers (`make_sse_chunk`, `make_error_response`) (lines 1710-1750)
- All routes: `/v1/models`, `/v1/models/select`, `/v1/models/unload`,
  `/health`, `/v1/chat/completions` (lines 1760-2000)
- Startup block (lines 2650-2682)

**REPLACE (subprocess → ctypes):**

| Old (subprocess) | New (ctypes) |
|-----------------|--------------|
| `import subprocess, select, codecs` | `import ctypes, queue` |
| `drain_stdout()` | REMOVE entirely |
| `is_stats_line()`, `_parse_stats_line()` | REMOVE — use `RKLLMResult.perf` |
| `is_noise_line()`, `clean_line()` | REMOVE — no stdout to parse |
| `load_model()` — subprocess.Popen + init wait | `rkllm_init()` + `rkllm_set_chat_template()` |
| `unload_current()` — SIGKILL + pipe close | `rkllm_destroy()` |
| `is_process_healthy()` | `is_model_loaded()` — check handle != None |
| `_generate_stream()` — select/os.read stdout loop | Callback → Queue → yield SSE |
| `_generate_complete()` — select/os.read collect | Callback → Queue → collect |
| `USE_MULTILINE_PROTOCOL` | REMOVE — native prompt handling |
| `RKLLM_BINARY` config | `RKLLM_LIB_PATH` config |
| Residual data guard (MINIMUM_PREFILL_TIME) | REMOVE — no stdout contamination |
| Prefix stripping (`robot:`, `LLM:`) | REMOVE — no text prefix in callback |
| Process monitor (poll/restart) | Simplified health check |
| SIGKILL on abort/disconnect | `rkllm_abort(handle)` |
| Kill after RAG for KV clear | `rkllm_clear_kv_cache()` |

#### 5.2 rkllm ctypes API Research

**Critical finding: `rkllm_run()` is BLOCKING/synchronous.**
The callback fires from the same thread that called `rkllm_run()`.
Official demos use `threading.Thread` to run inference in background.

**Architecture pattern (from official flask_server.py):**
```
Main thread:     Flask request handler
                      │
                      ▼
Worker thread:   threading.Thread(target=rkllm_run, ...)
                      │
                      ▼ (callback fires per token)
Callback:        pushes token text to queue.Queue
                      │
                      ▼
Main thread:     reads from queue, yields SSE chunks
```

**Struct definitions verified from official Python demos:**

```python
# From airockchip/rknn-llm flask_server.py
class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput)
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", RKLLMInputType),
        ("input_data", RKLLMInputUnion)
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int)
    ]

class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat)
    ]
```

**API function signatures:**

```python
# rkllm_init(handle*, param*, callback) -> int
rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
rkllm_init.restype = ctypes.c_int

# rkllm_run(handle, input*, infer_params*, userdata) -> int  [BLOCKING]
rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
rkllm_run.restype = ctypes.c_int

# rkllm_destroy(handle) -> int
rkllm_destroy.argtypes = [RKLLM_Handle_t]
rkllm_destroy.restype = ctypes.c_int

# rkllm_abort(handle) -> int
rkllm_abort.argtypes = [RKLLM_Handle_t]
rkllm_abort.restype = ctypes.c_int

# rkllm_set_chat_template(handle, system_prefix, user_prefix, assistant_prefix) -> int
rkllm_set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
rkllm_set_chat_template.restype = ctypes.c_int

# Callback type: int callback(RKLLMResult*, void* userdata, int state)
#   Return 0 = continue, 1 = pause
callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
```

**Callback states:**
```
RKLLM_RUN_NORMAL  = 0  → result->text has the next token(s)
RKLLM_RUN_WAITING = 1  → waiting for complete UTF-8 char (ignore)
RKLLM_RUN_FINISH  = 2  → generation complete, result->perf has stats
RKLLM_RUN_ERROR   = 3  → error occurred
```

### 6. Rollback Point Created
- Created git tag `v1.0-subprocess` on current HEAD
- Pushed tag to GitHub origin
- To rollback: `git checkout v1.0-subprocess -- api.py`

### 7. Backup Created
- Renamed `api.py` → `api_subprocess_backup.py`
- Verified: 122798 bytes, 2322 lines, intact

### 8. Current State
- `api.py` exists but is **empty (0 bytes)** — needs to be written
- All research complete, all safety measures in place

---

## Implementation Plan

### New Architecture Overview

```
┌──────────────────────────────────────────────────┐
│                  Flask Server                     │
│         (same routes, same API contract)          │
├──────────────────────────────────────────────────┤
│                                                   │
│   /v1/chat/completions                           │
│        │                                          │
│        ▼                                          │
│   build_prompt()  ← PRESERVED (RAG pipeline)     │
│        │                                          │
│        ▼                                          │
│   ┌─────────────────────────┐                    │
│   │  RKLLMWrapper class     │  ← NEW             │
│   │  ├─ init(model_path)    │  rkllm_init()      │
│   │  ├─ run(prompt, role,   │  rkllm_run() in    │
│   │  │      keep_history)   │  worker thread      │
│   │  ├─ abort()             │  rkllm_abort()      │
│   │  ├─ clear_kv_cache()    │  rkllm_clear_kv..  │
│   │  └─ destroy()           │  rkllm_destroy()    │
│   └─────────┬───────────────┘                    │
│             │                                     │
│    callback │ pushes to queue.Queue               │
│             ▼                                     │
│   ┌─────────────────────────┐                    │
│   │  Token Queue            │                    │
│   │  (thread-safe)          │                    │
│   └─────────┬───────────────┘                    │
│             │                                     │
│             ▼                                     │
│   _generate_stream() / _generate_complete()      │
│   reads queue → SSE chunks / JSON response       │
│                                                   │
└──────────────────────────────────────────────────┘
```

### Key Design Decisions

#### 1. KV Cache Strategy
- **Normal mode (multi-turn):** `keep_history=1` — KV cache retained between
  `rkllm_run()` calls. Follow-up turns only prefill NEW tokens.
- **RAG mode:** `keep_history=0` + `rkllm_clear_kv_cache()` — RAG queries get
  fresh context each time (different web search results).
- **Model switch:** `rkllm_destroy()` old + `rkllm_init()` new.

#### 2. Threading Model
```
Flask request thread:
    1. build_prompt()
    2. Create queue.Queue for this request
    3. Start threading.Thread → calls rkllm_run() (blocking)
    4. Read tokens from queue in a loop, yield SSE chunks
    5. Thread finishes when rkllm_run() returns

Callback (called from worker thread):
    1. RKLLM_RUN_NORMAL: queue.put(("token", text))
    2. RKLLM_RUN_FINISH: queue.put(("finish", perf_stats))
    3. RKLLM_RUN_ERROR:  queue.put(("error", None))
```

#### 3. Abort Handling
- Client disconnect → `rkllm_abort(handle)` (graceful, no process kill)
- The worker thread's `rkllm_run()` returns after abort
- No process restart needed — model stays loaded

#### 4. Native Features Gained
- **`RKLLMInput.role = "user"`** — Proper role labeling. The runtime applies
  the correct chat template tokens internally.
- **`enable_thinking`** — Native toggle replaces our `/no_think` text suffix.
- **`RKLLMResult.perf`** — Real NPU performance metrics instead of parsing
  stdout text lines.
- **`keep_history=1`** — THE key feature. ~20x prefill speedup on follow-ups.

### File Structure (new api.py)

```
Lines ~1-35:     Module docstring (updated for ctypes)
Lines ~36-55:    Imports (subprocess/select/codecs removed, ctypes/queue added)
Lines ~56-70:    Flask app setup
Lines ~71-180:   Configuration (RKLLM_LIB_PATH replaces RKLLM_BINARY,
                 remove USE_MULTILINE_PROTOCOL, MINIMUM_PREFILL_TIME,
                 READ_BUFFER_SIZE, DRAIN_BUFFER_SIZE, etc.)
Lines ~180-250:  RAG cache (PRESERVED)
Lines ~250-270:  Logging (PRESERVED)
Lines ~270-395:  Model detection + aliases (PRESERVED)
Lines ~395-420:  ctypes struct definitions (NEW)
Lines ~420-500:  RKLLMWrapper class (NEW)
Lines ~500-540:  Callback function (NEW)
Lines ~540-580:  Globals (simplified — no CURRENT_PROCESS, just wrapper ref)
Lines ~580-700:  Request tracking (PRESERVED)
Lines ~700-730:  Utility: resolve_model (PRESERVED)
                 NOTE: is_stats_line, _parse_stats_line, is_noise_line,
                 clean_line, drain_stdout ALL REMOVED
Lines ~730-850:  _clean_web_content (PRESERVED)
Lines ~850-960:  _score_paragraph (PRESERVED)
Lines ~960-1000: _strip_system_fluff, _is_date_only_system (PRESERVED)
Lines ~1000-1050: _extract_rag_reference (PRESERVED)
Lines ~1050-1450: build_prompt (PRESERVED — entire RAG pipeline)
Lines ~1450-1550: Model management: load_model, unload_current (REWRITTEN)
Lines ~1550-1600: Process monitor (SIMPLIFIED)
Lines ~1600-1640: Shutdown (SIMPLIFIED — rkllm_destroy instead of SIGKILL)
Lines ~1640-1700: SSE helpers (PRESERVED)
Lines ~1700-1800: ThinkTagParser + parse_think_tags (PRESERVED)
Lines ~1800-1950: Routes (PRESERVED, minor adjustments)
Lines ~1950-2100: _generate_stream (REWRITTEN — queue-based)
Lines ~2100-2200: _generate_complete (REWRITTEN — queue-based)
Lines ~2200-2230: Startup (PRESERVED)
```

**Estimated new file size:** ~1800-2000 lines (vs 2322 old)
— ~500 lines of stdout parsing removed, ~200 lines of ctypes code added.

### What Gets Removed (~500 lines)

1. **`drain_stdout()`** (~30 lines) — No stdout pipe to drain
2. **`is_stats_line()`** (~20 lines) — Stats come from `RKLLMResult.perf`
3. **`_parse_stats_line()`** (~30 lines) — Same
4. **`is_noise_line()`** (~15 lines) — No noise in callbacks
5. **`clean_line()`** (~5 lines) — No control chars from C library
6. **Residual data guard** (~15 lines) — No stdout contamination possible
7. **Prefix stripping** (`robot:`, `LLM:`, `user: robot:`) — No prefixes in callback text
8. **Prompt indicator detection** (`user:`, `robot:` regex) — No prompt indicators
9. **stdout select/read loops** in both `_generate_stream` and `_generate_complete`
   (~300 lines total) — Replaced by queue reads
10. **`USE_MULTILINE_PROTOCOL`** logic — Native prompt handling
11. **`subprocess.Popen` + pipe management** — Direct API calls
12. **`codecs.getincrementaldecoder`** — Callback gives clean UTF-8 strings

### What Gets Added (~200 lines)

1. **ctypes struct definitions** (~80 lines) — All rkllm types
2. **`RKLLMWrapper` class** (~80 lines) — init, run, abort, destroy, clear_kv
3. **Callback function** (~20 lines) — Token → queue routing
4. **Queue-based generation** (~20 lines net addition over removed code)

### Trade-offs Acknowledged

| Pro | Con |
|-----|-----|
| ~20x prefill speedup on follow-ups | Library crash takes down Python |
| Native abort (no SIGKILL) | Must match exact struct layout |
| Real perf stats from NPU | Can't redirect rkllm C library stderr |
| ~500 fewer lines of code | New dependency on exact .so path |
| Native role/thinking support | Less isolation than subprocess |

### Testing Plan (on device)

1. **Smoke test:** Load model, single-turn generation
2. **Multi-turn:** Verify KV cache retention (check prefill_time_ms decreases)
3. **RAG mode:** Verify KV cache clears between RAG queries
4. **Model switch:** Load model A → Load model B → verify clean switch
5. **Abort:** Send request, disconnect client, verify no crash
6. **Idle unload:** Wait for timeout, verify `rkllm_destroy()` runs
7. **Streaming + non-streaming:** Both paths work
8. **Think tags:** Qwen3 `<think>` parsing still works
9. **Performance benchmark:** Compare prefill times before/after

### Rollback Procedure

If anything goes wrong on the device:

```bash
# Option A: Restore from backup file
cp api_subprocess_backup.py api.py

# Option B: Restore from git tag
git checkout v1.0-subprocess -- api.py

# Option C: Full rollback to tagged state
git checkout v1.0-subprocess
```

---

## Progress Tracker

- [x] Research current api.py architecture (2682 lines fully analyzed)
- [x] Research rkllm ctypes API (header + official demos studied)
- [x] Map current features to ctypes equivalents
- [x] Create git rollback tag `v1.0-subprocess`
- [x] Create backup file `api_subprocess_backup.py` (verified)
- [ ] **Write new ctypes-based api.py** ← NEXT STEP
- [ ] Verify syntax compiles (`python -m py_compile api.py`)
- [ ] Commit and push to GitHub
- [ ] Deploy and test on Orange Pi 5 Plus
