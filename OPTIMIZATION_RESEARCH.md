# Actionable Optimization Guide: Small LLMs on RK3588 NPU
**Research Date: February 11, 2026 | Hardware: Orange Pi 5 Plus (16GB RAM) | SDK: RKLLM v1.2.3**

---

## Table of Contents
1. [Sampling Parameter Optimization](#1-sampling-parameter-optimization)
2. [Prompt Engineering for Small Models](#2-prompt-engineering-for-small-models)
3. [RKLLM SDK-Specific Optimizations](#3-rkllm-sdk-specific-optimizations)
4. [API Server Code Changes](#4-api-server-code-changes)
5. [Open WebUI Configuration](#5-open-webui-configuration)
6. [Model Selection & Upgrades](#6-model-selection--upgrades)
7. [Quantization Strategy](#7-quantization-strategy)
8. [Implementation Priority](#8-implementation-priority)

---

## 1. Sampling Parameter Optimization

### Current Problem
The current `api.py` hardcodes sampling at model init time:
```python
param.top_k = 1          # Greedy — bad for Qwen3!
param.top_p = 0.9
param.temperature = 0.8
param.repeat_penalty = 1.1
param.frequency_penalty = 0.0
param.presence_penalty = 0.0
```
These are set once at model load and **cannot be changed per-request** (RKLLM bakes them into the model session). This is a major limitation.

### Official Recommendations by Model

| Model | Mode | Temperature | Top-P | Top-K | MinP | Presence Penalty | Notes |
|-------|------|-------------|-------|-------|------|-----------------|-------|
| **Qwen3-4B (thinking)** | `enable_thinking=True` | **0.6** | **0.95** | **20** | 0 | 0-1.5 if repetition | DO NOT use greedy (top_k=1) — causes degradation & infinite loops |
| **Qwen3-4B (non-thinking)** | `enable_thinking=False` | **0.7** | **0.8** | **20** | 0 | 0-1.5 if repetition | |
| **Qwen3-1.7B** | Same as 4B | 0.6/0.7 | 0.95/0.8 | 20 | 0 | 1.5 recommended | Smaller models need presence_penalty more |
| **Phi-3-Mini-4K** | General | 0.7 | 0.9 | 40 | 0 | 0.0 | Conservative works best |
| **Gemma-3-4B** | General | 0.7 | 0.9 | 40 | 0 | 0.0 | |
| **MiniCPM4** | General | 0.7 | 0.7 | — | 0 | 0.0 | repetition_penalty: 1.02 |

### Actionable Changes

**CRITICAL FIX — Change init-time sampling parameters:**
```python
# For Qwen3 models (thinking mode default):
param.top_k = 20           # Was 1 (greedy) — this is WRONG for Qwen3
param.top_p = 0.95         # Was 0.9
param.temperature = 0.6    # Was 0.8
param.repeat_penalty = 1.1 # Keep
param.frequency_penalty = 0.0
param.presence_penalty = 1.5  # Was 0.0 — prevents repetition loops

# For non-Qwen3 models:
param.top_k = 40
param.top_p = 0.9
param.temperature = 0.7
param.repeat_penalty = 1.1
param.frequency_penalty = 0.0
param.presence_penalty = 0.0
```

**KEY INSIGHT from community:** Qwen3 models are "extra sensitive to quantization and sampling parameters. Q4 feels significantly dumber than Q8, and that recommended presence_penalty is well-advised." Using `top_k=1` (greedy) on Qwen3 "can lead to performance degradation and endless repetitions" — this is officially documented by Qwen team.

### Per-Model Default Profiles (implement in api.py)
```python
_MODEL_SAMPLING_PROFILES = {
    'qwen3': {
        'thinking': {'top_k': 20, 'top_p': 0.95, 'temperature': 0.6, 'presence_penalty': 1.5},
        'non_thinking': {'top_k': 20, 'top_p': 0.8, 'temperature': 0.7, 'presence_penalty': 1.5},
    },
    'phi3': {
        'default': {'top_k': 40, 'top_p': 0.9, 'temperature': 0.7, 'presence_penalty': 0.0},
    },
    'gemma3': {
        'default': {'top_k': 40, 'top_p': 0.9, 'temperature': 0.7, 'presence_penalty': 0.0},
    },
    'minicpm': {
        'default': {'top_k': 40, 'top_p': 0.7, 'temperature': 0.7, 'repeat_penalty': 1.02},
    },
}
```

---

## 2. Prompt Engineering for Small Models

### Key Principles (validated by community & research)

1. **One task per prompt** — "Single-goal prompts: 89% satisfaction vs 41% for multi-goal" (Reddit community testing)

2. **Keep system prompts SHORT** — Small models have limited attention. A long elaborate system prompt steals context from the actual task. For sub-4B models:
   - System prompt should be < 100 tokens
   - Avoid persona fluff like "You are an incredibly helpful, knowledgeable..."
   - Use terse directives: "Answer concisely. If unsure, say 'I don't know'."

3. **Put instructions at START or END** — "LLMs often forget what's in the middle (especially in long prompts)" — position critical instructions first or last, never buried in the middle.

4. **Use explicit format instructions** — Instead of "make it nice", say "Reply in 3 bullet points" or "Respond with a JSON object containing 'answer' field."

5. **Use positive instructions** — "Do X" works better than "Don't do Y" for small models.

6. **Avoid asking for conciseness on Qwen3** — Community finding: "Qwen3 does not like my usual system prompt where I ask it to answer in a concise manner without preambles — it gives short answers as requested, but the answers become wrong." Let the model think and explain.

7. **Let thinking models think** — "Even Qwen 0.6B delivers surprisingly good results when you let it think." For Qwen3 with `enable_thinking=True`, don't constrain output length unnecessarily.

### Actionable System Prompt Templates

**For General Chat (Qwen3):**
```
You are a helpful assistant. Answer the user's question directly. If you are unsure about something, say so.
```

**For RAG Queries:**
```
Answer based on the provided context. If the context doesn't contain the answer, say "I don't have enough information to answer that."

Context:
{rag_context}
```

**For Code Tasks:**
```
Write clean, working code. Include brief comments. No explanations outside the code block unless asked.
```

### Actionable Changes for api.py

- **Trim the RAG system prompt** — if the current RAG prompt is verbose, shorten it for sub-4B models
- **For meta-tasks** (search query gen, title gen), the existing `_OPENWEBUI_META_TASK_SIGNATURES` detection is good — these should use non-thinking mode with minimal prompting
- **Strip thinking content from history** — The Qwen3 docs explicitly say: "In multi-turn conversations, the historical model output should only include the final output part and does not need to include the thinking content." Verify this is implemented correctly.

---

## 3. RKLLM SDK-Specific Optimizations

### v1.2.3 Features (Current SDK - Nov 2025)
From the CHANGELOG, key features available now:
- **Automatic cache reuse for embedding input** — NEW in v1.2.3
- **Chat template loading from external file** — NEW in v1.2.3
- **Multi-instance inference** — v1.2.2
- **LongRoPE support** — v1.2.2
- **Multi-batch inference** — v1.2.1
- **Function calling capability** — v1.2.1
- **Thinking mode selection** — v1.2.1
- **Automatic prompt cache reuse** — v1.2.0
- **Embedding flash storage** (reduces memory) — v1.2.0
- **Max context length up to 16K** — v1.2.0
- **GRQ Int4 quantization** — v1.2.0
- **Group-wise quantization** (w4a16 g32/g64/g128, w8a8 g128/g256/g512) — v1.1.0
- **LoRA model loading** — v1.1.0
- **Prompt cache storage and preloading** — v1.1.0
- **Mixed quantization** (combo of grouped + non-grouped) — v1.1.0

### Benchmark Numbers (RK3588, w8a8, prefill=128, decode=64)

| Model | Params | Prefill (ms) | Decode (tok/s) | Memory (MB) |
|-------|--------|-------------|---------------|-------------|
| Qwen3 | 0.6B | 213.50 | 32.16 | 773.77 |
| Qwen2.5 | 1.5B | 412.27 | 16.32 | 1659.15 |
| InternLM2 | 1.8B | 374.00 | 15.58 | 1765.71 |
| Gemma2 | 2B | 679.90 | 9.80 | 2765.30 |
| **Gemma3n** | **2B** | **1220.40** | **9.46** | **2709.25** |
| TeleChat2 | 3B | 649.60 | 10.22 | 2777.00 |
| Phi3 | 3.8B | 1022.00 | 7.50 | 3747.73 |
| MiniCPM3 | 4B | 1385.92 | 5.99 | 4339.61 |
| ChatGLM3 | 6B | 1395.34 | 4.94 | 5976.43 |
| **Qwen3-VL** | **2B** | **391** | **15.12** | **1892.13** |
| DeepSeekOCR | 3B(A570M) | 696.21 | 31.81 | 3028.66 |

### Multimodal Benchmarks (RK3588)

| Model | Encoder (img) | Prefill | Decode |
|-------|--------------|---------|--------|
| Qwen3-VL-2B | 2.08s (448x448) | 649ms (196 tokens) | 14.91 tok/s |
| Qwen2.5-VL-3B | 2.93s (392x392) | 1120ms (196 tokens) | 8.66 tok/s |
| SmolVLM-256M | 842ms (512x512) | 77.3ms (128 tokens) | 78 tok/s |
| InternVL3-1B | — | — | — |

### Actionable Init Parameter Changes

```python
# Current code:
param.extend_param.embed_flash = 1        # Good — reduces memory
param.extend_param.n_batch = 1
param.extend_param.use_cross_attn = 0
param.extend_param.enabled_cpus_num = 4
param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)  # Big cores

# Recommendations:
# 1. embed_flash = 1 is correct — keep it (stores embeddings on flash, saves RAM)
# 2. enabled_cpus_mask targets big cores 4-7 — this is correct for RK3588
# 3. Consider increasing n_batch if multi-batch is needed (but for single-user, 1 is fine)
```

### Critical: optimization_level During Conversion
From the benchmark page: **"All models should be converted with `optimization_level` set to 0 to enable optimized runtime performance."** Verify your .rkllm models were converted with `optimization_level=0` in the RKLLM-Toolkit conversion script.

### Performance Monitoring
```bash
# Set environment variable for perf logging:
export RKLLM_LOG_LEVEL=1

# Run frequency-setting script from rknn-llm/scripts/ to max NPU/CPU frequencies:
# This ensures benchmarks and usage hit peak performance
```

---

## 4. API Server Code Changes

### Change 1: Model-Aware Sampling Defaults (HIGH PRIORITY)
The current `param.top_k = 1` is causing Qwen3 quality degradation. Implement model-family detection to set appropriate defaults:

```python
def _get_sampling_params(model_name_lower):
    """Return optimal sampling params based on model family."""
    if 'qwen3' in model_name_lower or 'qwen2.5' in model_name_lower:
        return {
            'top_k': 20,
            'top_p': 0.95,
            'temperature': 0.6,
            'repeat_penalty': 1.1,
            'frequency_penalty': 0.0,
            'presence_penalty': 1.5,
        }
    elif 'phi' in model_name_lower:
        return {
            'top_k': 40,
            'top_p': 0.9,
            'temperature': 0.7,
            'repeat_penalty': 1.1,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
        }
    elif 'gemma' in model_name_lower:
        return {
            'top_k': 40,
            'top_p': 0.9,
            'temperature': 0.7,
            'repeat_penalty': 1.1,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
        }
    elif 'minicpm' in model_name_lower:
        return {
            'top_k': 40,
            'top_p': 0.7,
            'temperature': 0.7,
            'repeat_penalty': 1.02,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
        }
    else:
        # Safe general defaults
        return {
            'top_k': 40,
            'top_p': 0.9,
            'temperature': 0.7,
            'repeat_penalty': 1.1,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
        }
```

### Change 2: Thinking Mode Toggling for Meta-Tasks (ALREADY IMPLEMENTED - verify)
The existing `_OPENWEBUI_META_TASK_SIGNATURES` detection is good. Verify that when thinking is disabled for these tasks, the sampling params also adjust (temperature 0.7, top_p 0.8 for non-thinking Qwen3).

### Change 3: Strip Thinking Content from Multi-Turn History
Qwen3 docs explicitly state: "the historical model output should only include the final output part and does not need to include the thinking content." Verify that `<think>...</think>` blocks are stripped from assistant messages before they're fed back as history.

### Change 4: Context Length Management
- For Qwen3-4B: native 32K context, but on RK3588 with limited memory, keep practical limit at 4K-8K
- For Qwen3-1.7B: can afford slightly more context since model is smaller
- **Don't enable YaRN/RoPE scaling** unless context regularly exceeds 32K — it degrades short-text performance

### Change 5: Log the Actual Sampling Params Being Used
Currently the code logs "Ignored sampling params" when users send non-defaults. Instead, log what params ARE being used so debugging is possible.

---

## 5. Open WebUI Configuration

### System Prompt Management
1. **Keep system prompts concise** for small models — under 100 tokens. Open WebUI allows per-model system prompts in Admin > Models.
2. **Don't use the default verbose Open WebUI system prompt** — set a custom short one per model.
3. **For Qwen3**, avoid prompts requesting "be concise" — it degrades answer quality.

### Recommended Per-Model Settings in Open WebUI

**For Qwen3-4B / Qwen3-1.7B:**
- System prompt: `"You are a helpful assistant."`
- Temperature: 0.6 (if adjustable via UI)
- Max tokens: 4096 (sufficient for most responses, saves time)
- **Don't request streaming stop** on thinking tokens — let the model complete thinking

**For Gemma-3-4B:**
- System prompt: `"You are a helpful assistant. Be clear and accurate."`
- Temperature: 0.7

**For VL models (Qwen3-VL-2B):**
- Lower max tokens (1024-2048) since image encoding takes time
- System prompt: `"Describe what you see accurately. If unsure, say so."`

### Context Length Settings
- In Open WebUI, set context length per model to match what was used during RKLLM conversion
- Don't let the UI send more context than the model was compiled for — it causes silent truncation or errors

### Chat History Management
- Open WebUI sends increasingly long conversation history. For small models, this fills context fast.
- Consider implementing a sliding window in the API that keeps only the last N turns
- The current `VL_ASSISTANT_HISTORY_CAP = 500` chars is good for VL; consider a similar cap for text models at 2000-4000 chars

---

## 6. Model Selection & Upgrades

### Recommended Models for RKLLM on RK3588

#### Best Overall Chat (under 4B)
1. **Qwen3-4B** — Best in class for sub-4B general chat. Community consensus: "When did small models get so smart?" with Qwen3 4B being the standout. 5.99 tok/s on RK3588 w8a8.
2. **Qwen3-1.7B** — Excellent for its size, especially with thinking mode. Great for tool calling. 
3. **Gemma-3-4B / Gemma-3-QAT-4B** — Community says "comparable to Qwen3" for non-reasoning tasks. The QAT variant is quantization-aware trained.

#### New Models to Try (supported by RKLLM v1.2.3)
| Model | Why Try It | RKLLM Support | Expected Performance |
|-------|-----------|---------------|---------------------|
| **Gemma-3n-E2B** | 2B effective params, 5B raw but runs like 2B via selective activation. Multimodal (text+image+audio). Very new (mid-2025). | Yes (v1.2.3 added Gemma3n) | ~9.46 tok/s (benchmark shows Gemma3n 2B) |
| **MiniCPM4-0.5B** | Ultra-efficient, designed for edge. Trainable sparse attention. | Yes (v1.2.3) | 45.13 tok/s — blazing fast |
| **MiniCPM3-4B** | Strong 4B model with good benchmark scores. | Yes | 5.99 tok/s |
| **DeepSeek-R1-Distill** (small variants) | Reasoning distilled into smaller models. | Yes (v1.2.3) | Varies |
| **InternVL3-1B** | Very small VL model, could replace Qwen3-VL-2B for faster image tasks | Yes (v1.2.3) | Should be faster than Qwen3-VL-2B |
| **SmolVLM-256M** | Tiny VL model, 78 tok/s decode, great for quick image descriptions | Yes | 78 tok/s |
| **RWKV7-1.5B** | Linear attention — different architecture, constant memory. 13.33 tok/s | Yes (v1.2.1) | 13.33 tok/s, 1450 MB RAM |
| **TeleChat2-3B** | Good middle ground, 10.22 tok/s | Yes | 10.22 tok/s |

#### Models NOT to Bother With
- **ChatGLM3-6B**: Too large for RK3588, only 4.94 tok/s, uses 6GB RAM
- **Gemma2-2B**: Superseded by Gemma-3n-E2B
- **TinyLLAMA-1.1B**: Old, quality below Qwen3-0.6B

### Speed vs Quality Tradeoff

For the Orange Pi 5 Plus with 16GB RAM, the practical sweet spots are:

| Slot | Model | Decode Speed | RAM | Use Case |
|------|-------|-------------|-----|----------|
| **Fast responder** | Qwen3-1.7B w8a8 | ~15-16 tok/s | ~1.7 GB | Quick chat, tool calls, meta-tasks |
| **Quality responder** | Qwen3-4B w8a8 | ~6 tok/s | ~4.3 GB | Complex questions, reasoning |
| **Vision** | Qwen3-VL-2B w8a8 | ~15 tok/s | ~1.9 GB | Image understanding |
| **Ultrafast utility** | MiniCPM4-0.5B w8a8 | ~45 tok/s | ~525 MB | Search query gen, title gen, tags |

**Idea**: Use MiniCPM4-0.5B or Qwen3-0.6B for Open WebUI meta-tasks (search query generation, title generation, tag generation) — these are currently going through the main model and wasting time on thinking mode. A tiny model at 45 tok/s would return results in <1 second.

---

## 7. Quantization Strategy

### w8a8 vs w4a16 vs w8a8_g128

| Quantization | Quality | Speed | Memory | Best For |
|-------------|---------|-------|--------|----------|
| **w8a8** | Best | Fastest decode | Highest | **RK3588 (recommended)** |
| w4a16 | Lower | Fast prefill, slower decode | Lowest | Memory-constrained platforms |
| w4a16_g128 | Better than w4a16 | Medium | Medium | Compromise |
| w8a8_g128 | Good | Fast | Medium-high | Good balance |

**For RK3588 with 16GB RAM: Always use w8a8** — it's both the fastest and highest quality on this platform. The benchmark shows RK3588 only lists w8a8 results (w4a16 variants are shown for RK3576/RV1126B where memory is tighter).

### Critical Community Finding
"Qwen3 models are extra sensitive to quantization. Q4 feels significantly dumber than Q8." — Use w8a8 for all Qwen3 models on RK3588.

### Conversion Tips
```python
# When converting models with rkllm-toolkit:
rkllm.build(
    do_quantization=True,
    optimization_level=0,     # CRITICAL: must be 0 for optimized runtime
    quantized_dtype='w8a8',   # Best for RK3588
    target_platform='rk3588',
)
```

---

## 8. Implementation Priority

### P0 — Do Immediately (biggest quality impact)
1. **Fix sampling params for Qwen3**: Change `top_k=1` → `top_k=20`, add `presence_penalty=1.5`. This is likely causing repetition loops and degraded output quality RIGHT NOW.
2. **Verify models were converted with `optimization_level=0`** — huge performance impact if wrong.
3. **Set NPU/CPU to max frequency** — run the frequency-setting scripts from rknn-llm/scripts/.

### P1 — Do Soon (meaningful improvement)
4. **Implement model-family-aware sampling defaults** — different params for Qwen3 vs Phi3 vs Gemma.
5. **Shorten system prompts** for all models to under 100 tokens.
6. **Verify thinking content is stripped from multi-turn history** for Qwen3.
7. **Set Open WebUI per-model system prompts** to be concise, not verbose.

### P2 — Nice to Have (optimization)
8. **Try Gemma-3n-E2B** as possible Gemma-3-4B replacement (similar quality, 2B effective params).
9. **Try MiniCPM4-0.5B** for meta-tasks (search query gen, title gen) — 45 tok/s ultrafast.
10. **Add sliding window for text model chat history** to prevent context overflow.
11. **Explore InternVL3-1B or SmolVLM** as faster VL alternatives.

### P3 — Future Exploration
12. **RWKV7** models — linear attention means constant memory regardless of context length, interesting for long conversations.
13. **Prompt cache preloading** (RKLLM v1.1.0+) — pre-cache common system prompts for faster first response.
14. **LoRA fine-tuning** — fine-tune Qwen3-1.7B on your specific use cases for better quality.
15. **Multi-instance inference** (v1.2.2) — run a tiny model alongside the main model for meta-tasks.

---

## Quick Reference: Single Most Impactful Change

**In `api.py` line ~1207, change:**
```python
# FROM (current — WRONG for Qwen3):
param.top_k = 1
param.top_p = 0.9
param.temperature = 0.8
param.repeat_penalty = 1.1
param.frequency_penalty = 0.0
param.presence_penalty = 0.0

# TO (correct for Qwen3 with thinking mode):
param.top_k = 20
param.top_p = 0.95
param.temperature = 0.6
param.repeat_penalty = 1.1
param.frequency_penalty = 0.0
param.presence_penalty = 1.5
```

This single change will likely produce the most noticeable quality improvement across all Qwen3 models.
