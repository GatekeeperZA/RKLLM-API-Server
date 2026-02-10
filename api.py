"""
RKLLM API Server — ctypes Direct API Version (Feb 08, 2026)
============================================================================
Major rewrite: subprocess → ctypes binding to librkllmrt.so

Key changes from subprocess version:
- Direct C library binding via ctypes (no process management / stdout parsing)
- KV cache preserved between turns (keep_history=1) → ~20x prefill speedup
- Native rkllm_abort() replaces SIGKILL for clean cancellation
- Real NPU performance stats from RKLLMResult.perf (no stdout parsing)
- Native enable_thinking toggle replaces /no_think text suffix
- ~500 lines of stdout parsing removed, ~200 lines of ctypes code added

Preserved unchanged from subprocess version:
- All RAG pipeline logic (build_prompt, _clean_web_content, _score_paragraph)
- Model detection, aliases, context-length auto-detection
- Request tracking, RAG response cache, ThinkTagParser
- SSE helpers, all API routes, OpenAI-compatible response format

Rollback: git checkout v1.0-subprocess -- api.py
Archive:  archive/api_v1_subprocess.py (2682 lines, fully functional)

MINIMUM SDK: librkllmrt.so ≥ v1.2.0 (RKLLM Runtime from airockchip/rknn-llm)
    The ctypes struct definitions (RKLLMExtendParam, RKLLMParam, RKLLMInput,
    etc.) match the rkllm.h C header shipped with SDK v1.2.x.  Older versions
    used a 112-byte reserved blob in RKLLMExtendParam and lacked fields like
    n_keep, n_batch, use_cross_attn, and enable_thinking.  Running against an
    older librkllmrt.so will cause silent struct-offset misalignment — the
    parameter block passed to rkllm_init() would be corrupted, producing wrong
    sampling behaviour rather than a crash.  Tested against v1.2.3.

Usage:
    gunicorn -w 1 -k gthread --threads 4 --timeout 300 -b 0.0.0.0:8000 api:app

IMPORTANT: Use -w 1 (single worker) - NPU can only load one model!
NOTE: rkllm_run() is BLOCKING — inference runs in a worker thread.
      The callback pushes tokens to a queue.Queue; the main thread
      reads from the queue and yields SSE chunks.
"""
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
import ctypes
import queue
import os
import time
import logging
from logging.handlers import RotatingFileHandler
import re
import json
import uuid
import hashlib
import base64
import io
import struct as pystruct
from collections import OrderedDict
from threading import Lock, RLock, Thread, Event
import atexit
import sys
import signal
from datetime import datetime, timezone

try:
    import numpy as np
    from PIL import Image
    _VL_DEPS_AVAILABLE = True
except ImportError:
    _VL_DEPS_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
CORS(app)

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, 'api.log')
MODELS_ROOT = os.path.expanduser("~/models")
SYSTEM_FINGERPRINT = "rkllm-v2.0.0-ctypes"

# Path to rkllm shared library — auto-detected or set via environment
RKLLM_LIB_PATH = os.environ.get('RKLLM_LIB_PATH', '')
if not RKLLM_LIB_PATH:
    for _candidate in ['/usr/lib/librkllmrt.so', 'lib/librkllmrt.so',
                        '/usr/local/lib/librkllmrt.so', 'librkllmrt.so']:
        if os.path.exists(_candidate):
            RKLLM_LIB_PATH = _candidate
            break
    if not RKLLM_LIB_PATH:
        RKLLM_LIB_PATH = 'librkllmrt.so'  # Let ctypes search LD_LIBRARY_PATH

# Path to rknn runtime library (for vision encoder) — auto-detected
RKNN_LIB_PATH = os.environ.get('RKNN_LIB_PATH', '')
if not RKNN_LIB_PATH:
    for _candidate in ['/usr/lib/librknnrt.so', 'lib/librknnrt.so',
                        '/usr/local/lib/librknnrt.so', 'librknnrt.so']:
        if os.path.exists(_candidate):
            RKNN_LIB_PATH = _candidate
            break
    if not RKNN_LIB_PATH:
        RKNN_LIB_PATH = 'librknnrt.so'

# VL model configuration: maps model folder name patterns to their special tokens.
# These tokens tell rkllm where image embeddings go in the prompt.
# Key: substring matched against the model folder name (lowercase).
VL_MODEL_CONFIGS = {
    'deepseekocr': {
        'img_start': '',
        'img_end': '',
        'img_content': '<\uff5c\u2581pad\u2581\uff5c>',  # <\uff5c\u2581pad\u2581\uff5c>
        'base_domain_id': 1,
        'image_tag': '<image>',
    },
    'qwen3-vl': {
        'img_start': '<|vision_start|>',
        'img_end': '<|vision_end|>',
        'img_content': '<|image_pad|>',
        'base_domain_id': 1,
        'image_tag': '<image>',
    },
    'qwen2.5-vl': {
        'img_start': '<|vision_start|>',
        'img_end': '<|vision_end|>',
        'img_content': '<|image_pad|>',
        'base_domain_id': 1,
        'image_tag': '<image>',
    },
    'qwen2-vl': {
        'img_start': '<|vision_start|>',
        'img_end': '<|vision_end|>',
        'img_content': '<|image_pad|>',
        'base_domain_id': 1,
        'image_tag': '<image>',
    },
    'internvl3': {
        'img_start': '<img>',
        'img_end': '</img>',
        'img_content': '<IMG_CONTEXT>',
        'base_domain_id': 1,
        'image_tag': '<image>',
    },
    'minicpm': {
        'img_start': '<image>',
        'img_end': '</image>',
        'img_content': 'slice_placeholder',
        'base_domain_id': 1,
        'image_tag': '<image>',
    },
}

# Timeouts
GENERATION_TIMEOUT = 600
FIRST_TOKEN_TIMEOUT = 120     # Max wait for first token (includes prefill)
FALLBACK_SILENCE = 12         # Max silence between tokens after first token

# Defaults
MAX_TOKENS_DEFAULT = 2048
CONTEXT_LENGTH_DEFAULT = 4096
CHARS_PER_TOKEN_ESTIMATE = 4  # ~4 chars/token for English (industry standard)

# OpenAI sampling parameter defaults — rkllm uses model-compiled sampling,
# but we log when callers send non-default values so they know.
_SAMPLING_DEFAULTS = {
    'temperature': 1.0,
    'top_p': 1.0,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0,
}

# Context-dependent thinking for RAG queries.
DISABLE_THINK_FOR_RAG_BELOW_CTX = 8192

# Open WebUI internal task signatures — these are meta-tasks (search query
# generation, title generation, tag generation, etc.) sent by Open WebUI
# before/after the actual user request.  Thinking mode wastes 20+ seconds
# on these trivial JSON/text generation tasks, so we auto-disable it.
_OPENWEBUI_META_TASK_SIGNATURES = (
    'generate search queries',
    'generating search queries',
    'search_queries',
    'generate 1-3 broad',
    'create a concise, 3-5 word',
    'title for the chat',
    'title for the prompt',
    'generate 1-3 tags',
    'categorize the chat',
    'autocomplete generation',
    'emoji as a title',
)

# Home Assistant detection — HA sends a system prompt with specific signatures.
# Thinking mode adds latency and reasoning tokens that are wasted on
# smart-home commands, so we auto-disable it for HA requests.
_HOME_ASSISTANT_SIGNATURES = (
    'smart home manager of home assistant',
    'available devices:',
    'execute_services function',
)

# RAG quality controls
RAG_MIN_QUALITY_SCORE = 2
RAG_MAX_PARAGRAPHS = 10
RAG_QUALITY_FLOOR_THRESHOLD = 3
RAG_DEDUP_SIMILARITY = 0.70

# Response caching for RAG queries
RAG_CACHE_TTL = 300
RAG_CACHE_MAX_ENTRIES = 50

# Request tracking
REQUEST_STALE_TIMEOUT = 30

# Monitoring
MONITOR_INTERVAL = 10
IDLE_UNLOAD_TIMEOUT = 300
VL_IDLE_UNLOAD_TIMEOUT = 300  # Auto-unload VL model after idle (frees NPU memory for large text models)

# Stopword list for content-vs-boilerplate detection (jusText-inspired).
ENGLISH_STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
    'or', 'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these',
    'those', 'it', 'its', 'he', 'she', 'they', 'them', 'their', 'his',
    'her', 'we', 'you', 'your', 'my', 'i', 'me', 'what', 'which', 'who',
})

# =============================================================================
# RAG RESPONSE CACHE
# =============================================================================
_rag_cache = OrderedDict()
_rag_cache_lock = Lock()


def _rag_cache_key(model_name, question):
    """Generate a deterministic cache key from model + question."""
    raw = f"{model_name}:{question.strip().lower()}"
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]


def _rag_cache_get(model_name, question):
    """Return cached (prompt, response) if still valid, else None."""
    if RAG_CACHE_TTL <= 0:
        return None
    key = _rag_cache_key(model_name, question)
    with _rag_cache_lock:
        entry = _rag_cache.get(key)
        if entry is None:
            return None
        ts, prompt, response = entry
        if time.time() - ts > RAG_CACHE_TTL:
            _rag_cache.pop(key, None)
            return None
        _rag_cache.move_to_end(key)
        return (prompt, response)


def _rag_cache_put(model_name, question, prompt, response):
    """Store a RAG response in the cache."""
    if RAG_CACHE_TTL <= 0:
        return
    key = _rag_cache_key(model_name, question)
    with _rag_cache_lock:
        _rag_cache[key] = (time.time(), prompt, response)
        _rag_cache.move_to_end(key)
        while len(_rag_cache) > RAG_CACHE_MAX_ENTRIES:
            _rag_cache.popitem(last=False)


def _jaccard_similarity(text_a, text_b):
    """Word-level Jaccard similarity between two strings."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


# =============================================================================
# LOGGING
# =============================================================================
_log_env = os.environ.get('RKLLM_API_LOG_LEVEL', 'DEBUG')
LOG_LEVEL = getattr(logging, _log_env.upper(), logging.DEBUG)

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=3)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# MODEL DETECTION
# =============================================================================
MODELS = {}


def detect_context_length(path_or_name, default=4096):
    s = path_or_name.lower()

    for k in (32768, 16384, 8192, 4096, 2048):
        suffix = f"{k // 1024}k"
        if f"-{suffix}" in s or f"_{suffix}" in s or f"-{suffix}-" in s or f"_{suffix}_" in s:
            return k

    m = re.search(r'[\-_](\d+)k[\-_\.]', s)
    if m:
        try:
            return int(m.group(1)) * 1024
        except (ValueError, OverflowError):
            pass

    return default


if os.path.exists(MODELS_ROOT):
    for root, dirs, files in os.walk(MODELS_ROOT):
        # Skip hidden/disabled directories (prefixed with '.')
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        rkllm_file = None

        rkllm_files = sorted(f for f in files if f.endswith(".rkllm"))
        if len(rkllm_files) > 1:
            logger.warning(f"Multiple .rkllm files in {root}: {rkllm_files} — using '{rkllm_files[0]}'")
        if rkllm_files:
            rkllm_file = os.path.join(root, rkllm_files[0])

        if not rkllm_file:
            continue

        model_id = os.path.basename(root).lower().replace(" ", "-").replace("_", "-")
        if not model_id:
            continue

        context_len = detect_context_length(rkllm_file, default=CONTEXT_LENGTH_DEFAULT)

        # Check for vision encoder (.rknn) in same folder -> VL model
        rknn_files = sorted(f for f in files if f.endswith(".rknn"))
        vision_encoder_path = None
        vl_config = None
        if rknn_files:
            vision_encoder_path = os.path.join(root, rknn_files[0])
            if len(rknn_files) > 1:
                logger.warning(f"Multiple .rknn files in {root}: {rknn_files} -- using '{rknn_files[0]}'")
            # Match VL config by folder name
            for vl_key, vl_cfg in VL_MODEL_CONFIGS.items():
                if vl_key in model_id:
                    vl_config = vl_cfg
                    break
            if vl_config is None:
                logger.warning(f"VL model {model_id}: .rknn found but no VL_MODEL_CONFIGS match -- "
                              f"treating as text-only (add config for '{model_id}' to VL_MODEL_CONFIGS)")
                vision_encoder_path = None

        config = {
            "path": rkllm_file,
            "context_length": context_len,
            "max_tokens": MAX_TOKENS_DEFAULT,
        }

        if vision_encoder_path and vl_config:
            config["vision_encoder_path"] = vision_encoder_path
            config["vl_config"] = vl_config
            logger.info(f"Detected VL: {model_id} (context={context_len}, "
                       f"encoder={os.path.basename(vision_encoder_path)})")
        else:
            logger.info(f"Detected: {model_id} (context={context_len})")

        MODELS[model_id] = config

# =============================================================================
# ALIASES
# =============================================================================


def generate_aliases(model_ids):
    """Auto-generate short aliases from detected model folder names."""
    candidates = {}

    for model_id in model_ids:
        parts = model_id.split('-')

        for i in range(1, len(parts)):
            candidate = '-'.join(parts[:i])
            if candidate != model_id:
                candidates.setdefault(candidate, set()).add(model_id)

        family_match = re.match(r'^([a-zA-Z]+)', parts[0])
        if family_match:
            family = family_match.group(1).lower()
            if family != model_id and family != parts[0].lower():
                candidates.setdefault(family, set()).add(model_id)

        if '.' in parts[0]:
            truncated = parts[0].split('.')[0]
            if truncated != model_id and truncated != parts[0].lower():
                candidates.setdefault(truncated, set()).add(model_id)

    aliases = {}
    for alias, claimants in sorted(candidates.items(), key=lambda x: len(x[0])):
        if len(claimants) == 1 and alias not in model_ids:
            aliases[alias] = next(iter(claimants))

    return aliases


ALIASES = generate_aliases(MODELS.keys())

_vl_count = sum(1 for c in MODELS.values() if 'vl_config' in c)
logger.info(f"Models: {list(MODELS.keys())}")
logger.info(f"Aliases: {ALIASES}")
if _vl_count:
    logger.info(f"VL deps: {'available (numpy, Pillow)' if _VL_DEPS_AVAILABLE else 'MISSING — install numpy Pillow for VL support'}")

# =============================================================================
# RKLLM CTYPES DEFINITIONS  (SDK ≥ v1.2.0 — see header docstring)
# =============================================================================
# All struct definitions match the rkllm.h C header from airockchip/rknn-llm
# SDK v1.2.x (verified against the Linux header and the official Python demos).
# Do NOT use with librkllmrt.so older than v1.2.0 — struct offsets will be wrong.

RKLLM_Handle_t = ctypes.c_void_p

# Callback states (mirrors rkllm.h LLMCallState enum)
RKLLM_RUN_NORMAL  = 0   # result->text has the next token(s)
RKLLM_RUN_WAITING = 1   # Waiting for complete UTF-8 char (ignore)
RKLLM_RUN_FINISH  = 2   # Generation complete, result->perf has stats
RKLLM_RUN_ERROR   = 3   # Error occurred

# Input types (mirrors rkllm.h — only PROMPT used; others kept for completeness)
RKLLM_INPUT_PROMPT     = 0
RKLLM_INPUT_TOKEN      = 1
RKLLM_INPUT_EMBED      = 2
RKLLM_INPUT_MULTIMODAL = 3

# Inference modes (mirrors rkllm.h — only GENERATE used; others kept for completeness)
RKLLM_INFER_GENERATE              = 0
RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLM_INFER_GET_LOGITS            = 2

# Integer types used as enums in struct fields
RKLLMInputType = ctypes.c_int
RKLLMInferMode = ctypes.c_int


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
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


class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float),
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput),
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", RKLLMInputType),
        ("input_data", RKLLMInputUnion),
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p),
    ]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]


# Callback type: int callback(RKLLMResult*, void* userdata, LLMCallState state)
#   Return 0 = continue inference, 1 = pause inference
callback_type = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(RKLLMResult),
    ctypes.c_void_p,
    ctypes.c_int,
)

# =============================================================================
# RKLLM CALLBACK AND WRAPPER
# =============================================================================
# Global token queue: the C callback pushes tokens here, the generation
# loop reads from it.  Safe because NPU is single-task — only one
# inference runs at a time.
_token_queue = queue.Queue()


def _rkllm_callback_impl(result_ptr, userdata, state):
    """C callback invoked by rkllm_run() for each generated token.

    Called from the SAME thread as rkllm_run() (the worker thread).
    Pushes (type, data) tuples to the global _token_queue.
    """
    try:
        if state == RKLLM_RUN_NORMAL:
            if result_ptr:
                text = result_ptr.contents.text
                if text:
                    decoded = text.decode('utf-8', errors='replace')
                    if decoded:
                        _token_queue.put(("token", decoded))
        elif state == RKLLM_RUN_FINISH:
            stats = {}
            if result_ptr:
                try:
                    perf = result_ptr.contents.perf
                    stats = {
                        "prefill_time_ms": perf.prefill_time_ms,
                        "prefill_tokens": perf.prefill_tokens,
                        "generate_time_ms": perf.generate_time_ms,
                        "generate_tokens": perf.generate_tokens,
                        "memory_usage_mb": perf.memory_usage_mb,
                    }
                except (ValueError, AttributeError):
                    pass  # Perf stats unavailable
            _token_queue.put(("finish", stats))
        elif state == RKLLM_RUN_ERROR:
            error_msg = "unknown error"
            if result_ptr:
                try:
                    text = result_ptr.contents.text
                    if text:
                        error_msg = text.decode('utf-8', errors='replace')
                except (ValueError, AttributeError):
                    pass
            _token_queue.put(("error", error_msg))
        # RKLLM_RUN_WAITING (state=1): incomplete UTF-8 — ignore
    except Exception as _cb_exc:
        # Callback must never raise into C land — but signal the error
        # so the consumer doesn't hang for GENERATION_TIMEOUT seconds.
        try:
            _token_queue.put(("error", f"callback exception: {_cb_exc}"))
        except Exception:
            pass  # Last resort: swallow to protect C caller
    return 0


# Must keep a reference to prevent garbage collection of the callback
_rkllm_callback = callback_type(_rkllm_callback_impl)


# =============================================================================
# RKNN VISION ENCODER (for VL models)
# =============================================================================
# Mirrors the C++ image_enc.cc from airockchip/rknn-llm multimodal_model_demo.
# Uses librknnrt.so via ctypes to run the .rknn vision encoder on the NPU.
#
# Flow: image (uint8 RGB HWC) -> rknn_inputs_set -> rknn_run -> rknn_outputs_get
#       -> float32 embedding array -> feed to rkllm via RKLLM_INPUT_MULTIMODAL

# RKNN constants (from rknn_api.h — verified against rknpu2 v2.x official header)
RKNN_SUCC = 0
# rknn_tensor_type enum: FLOAT32=0, FLOAT16=1, INT8=2, UINT8=3, INT16=4, UINT16=5
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_UINT8 = 3
# rknn_tensor_format enum: NCHW=0, NHWC=1, NC1HWC2=2
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1
# rknn_core_mask enum: AUTO=0, CORE_0=1, CORE_1=2, CORE_2=4, 0_1=3, 0_1_2=7
RKNN_NPU_CORE_AUTO = 0
RKNN_NPU_CORE_0_1_2 = 7       # All 3 NPU cores (CORE_0|CORE_1|CORE_2)
# rknn_query_cmd enum
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2


class RKNNInputOutputNum(ctypes.Structure):
    _fields_ = [
        ("n_input", ctypes.c_uint32),
        ("n_output", ctypes.c_uint32),
    ]


class RKNNTensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16),
        ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int),
        ("type", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class RKNNInput(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_int),
        ("fmt", ctypes.c_int),
    ]


class RKNNOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


class RKNNVisionEncoder:
    """Wraps librknnrt.so to run a .rknn vision encoder model on the NPU."""

    def __init__(self, lib_path):
        self.lib = ctypes.CDLL(lib_path)
        self._setup_functions()
        self.ctx = ctypes.c_uint64(0)
        self._loaded = False
        self.model_width = 0
        self.model_height = 0
        self.model_channel = 0
        self.model_image_token = 0
        self.model_embed_size = 0
        self.n_output = 0

    def _setup_functions(self):
        self.lib.rknn_init.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
        ]
        self.lib.rknn_init.restype = ctypes.c_int
        self.lib.rknn_set_core_mask.argtypes = [ctypes.c_uint64, ctypes.c_int]
        self.lib.rknn_set_core_mask.restype = ctypes.c_int
        self.lib.rknn_query.argtypes = [
            ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
        ]
        self.lib.rknn_query.restype = ctypes.c_int
        self.lib.rknn_inputs_set.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(RKNNInput),
        ]
        self.lib.rknn_inputs_set.restype = ctypes.c_int
        self.lib.rknn_run.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
        self.lib.rknn_run.restype = ctypes.c_int
        self.lib.rknn_outputs_get.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32,
            ctypes.POINTER(RKNNOutput), ctypes.c_void_p,
        ]
        self.lib.rknn_outputs_get.restype = ctypes.c_int
        self.lib.rknn_outputs_release.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(RKNNOutput),
        ]
        self.lib.rknn_outputs_release.restype = ctypes.c_int
        self.lib.rknn_destroy.argtypes = [ctypes.c_uint64]
        self.lib.rknn_destroy.restype = ctypes.c_int

    def init_model(self, model_path, core_num=3):
        """Load a .rknn vision encoder model. Returns True on success."""
        self.ctx = ctypes.c_uint64(0)
        ret = self.lib.rknn_init(
            ctypes.byref(self.ctx), ctypes.c_char_p(model_path.encode('utf-8')),
            0, 0, None,
        )
        if ret != RKNN_SUCC:
            logger.error(f"rknn_init failed: ret={ret} model={model_path}")
            return False

        mask = RKNN_NPU_CORE_0_1_2 if core_num >= 3 else RKNN_NPU_CORE_AUTO
        self.lib.rknn_set_core_mask(self.ctx, mask)

        io_num = RKNNInputOutputNum()
        ret = self.lib.rknn_query(self.ctx, RKNN_QUERY_IN_OUT_NUM,
                                  ctypes.byref(io_num), ctypes.sizeof(io_num))
        if ret != RKNN_SUCC:
            logger.error(f"rknn_query IN_OUT_NUM failed: ret={ret}")
            self.destroy()
            return False
        self.n_output = io_num.n_output
        logger.info(f"Vision encoder: {io_num.n_input} inputs, {io_num.n_output} outputs")

        input_attr = RKNNTensorAttr()
        ctypes.memset(ctypes.byref(input_attr), 0, ctypes.sizeof(input_attr))
        input_attr.index = 0
        ret = self.lib.rknn_query(self.ctx, RKNN_QUERY_INPUT_ATTR,
                                  ctypes.byref(input_attr), ctypes.sizeof(input_attr))
        if ret != RKNN_SUCC:
            logger.error(f"rknn_query INPUT_ATTR failed: ret={ret}")
            self.destroy()
            return False

        if input_attr.fmt == RKNN_TENSOR_NCHW:
            self.model_channel = input_attr.dims[1]
            self.model_height = input_attr.dims[2]
            self.model_width = input_attr.dims[3]
        else:
            self.model_height = input_attr.dims[1]
            self.model_width = input_attr.dims[2]
            self.model_channel = input_attr.dims[3]
        logger.info(f"Vision encoder input: {self.model_width}x{self.model_height}x{self.model_channel}")

        output_attr = RKNNTensorAttr()
        ctypes.memset(ctypes.byref(output_attr), 0, ctypes.sizeof(output_attr))
        output_attr.index = 0
        ret = self.lib.rknn_query(self.ctx, RKNN_QUERY_OUTPUT_ATTR,
                                  ctypes.byref(output_attr), ctypes.sizeof(output_attr))
        if ret != RKNN_SUCC:
            logger.error(f"rknn_query OUTPUT_ATTR failed: ret={ret}")
            self.destroy()
            return False

        for i in range(4):
            if output_attr.dims[i] > 1:
                self.model_image_token = output_attr.dims[i]
                self.model_embed_size = output_attr.dims[i + 1]
                break
        logger.info(f"Vision encoder output: image_token={self.model_image_token}, "
                    f"embed_size={self.model_embed_size}, n_output={self.n_output}")

        self._loaded = True
        return True

    def run(self, image_data_uint8):
        """Run vision encoder. Returns numpy float32 array of embeddings, or None."""
        if not self._loaded:
            return None

        rknn_input = RKNNInput()
        ctypes.memset(ctypes.byref(rknn_input), 0, ctypes.sizeof(rknn_input))
        rknn_input.index = 0
        rknn_input.type = RKNN_TENSOR_UINT8
        rknn_input.fmt = RKNN_TENSOR_NHWC
        rknn_input.size = self.model_width * self.model_height * self.model_channel
        rknn_input.buf = image_data_uint8.ctypes.data_as(ctypes.c_void_p)

        ret = self.lib.rknn_inputs_set(self.ctx, 1, ctypes.byref(rknn_input))
        if ret != RKNN_SUCC:
            logger.error(f"rknn_inputs_set failed: ret={ret}")
            return None

        ret = self.lib.rknn_run(self.ctx, None)
        if ret != RKNN_SUCC:
            logger.error(f"rknn_run failed: ret={ret}")
            return None

        outputs = (RKNNOutput * self.n_output)()
        for j in range(self.n_output):
            ctypes.memset(ctypes.byref(outputs[j]), 0, ctypes.sizeof(RKNNOutput))
            outputs[j].want_float = 1

        ret = self.lib.rknn_outputs_get(self.ctx, self.n_output, outputs, None)
        if ret != RKNN_SUCC:
            logger.error(f"rknn_outputs_get failed: ret={ret}")
            return None

        embed_len = self.model_image_token * self.model_embed_size * self.n_output
        result = np.zeros(embed_len, dtype=np.float32)

        if self.n_output == 1:
            ctypes.memmove(result.ctypes.data, outputs[0].buf, outputs[0].size)
        else:
            for i in range(self.model_image_token):
                for j in range(self.n_output):
                    offset = (i * self.n_output * self.model_embed_size +
                              j * self.model_embed_size)
                    src = ctypes.cast(outputs[j].buf, ctypes.POINTER(ctypes.c_float))
                    src_offset = i * self.model_embed_size
                    ctypes.memmove(
                        ctypes.cast(result.ctypes.data + offset * 4, ctypes.c_void_p),
                        ctypes.cast(ctypes.addressof(src.contents) + src_offset * 4, ctypes.c_void_p),
                        self.model_embed_size * 4,
                    )

        self.lib.rknn_outputs_release(self.ctx, self.n_output, outputs)
        return result

    def destroy(self):
        if self._loaded:
            try:
                self.lib.rknn_destroy(self.ctx)
            except Exception:
                pass
            self._loaded = False
            self.ctx = ctypes.c_uint64(0)

    @property
    def is_loaded(self):
        return self._loaded


# =============================================================================
# IMAGE HELPER FUNCTIONS
# =============================================================================


def _extract_images_from_messages(messages):
    """Extract image data from OpenAI-format multimodal messages.

    Returns (list_of_image_bytes, text_prompt).
    Only processes the LAST user message that contains images.
    """
    images = []
    text_parts = []

    for msg in reversed(messages):
        if msg.get('role') != 'user':
            continue
        content = msg.get('content')
        if not isinstance(content, list):
            continue

        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get('type') == 'image_url':
                url = part.get('image_url', {}).get('url', '')
                if url.startswith('data:'):
                    try:
                        _, b64data = url.split(',', 1)
                        img_bytes = base64.b64decode(b64data)
                        images.append(img_bytes)
                    except Exception as e:
                        logger.warning(f"Failed to decode base64 image: {e}")
                elif url.startswith('http'):
                    logger.warning(f"URL-based images not supported (only base64): {url[:80]}...")
            elif part.get('type') == 'text':
                text = part.get('text', '').strip()
                if text:
                    text_parts.append(text)

        if images:
            break

    text_prompt = ' '.join(text_parts) if text_parts else "Describe this image."
    return images, text_prompt


def _has_images_in_messages(messages):
    """Quick check: do any messages contain image content?"""
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'image_url':
                    return True
    return False


def _preprocess_image(image_bytes, target_width, target_height):
    """Preprocess image for vision encoder: decode, RGB, expand-to-square, resize."""
    if not _VL_DEPS_AVAILABLE:
        raise RuntimeError("VL dependencies not available (numpy, Pillow)")

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    w, h = img.size
    if w != h:
        size = max(w, h)
        square = Image.new('RGB', (size, size), (128, 128, 128))
        paste_x = (size - w) // 2
        paste_y = (size - h) // 2
        square.paste(img, (paste_x, paste_y))
        img = square

    img = img.resize((target_width, target_height), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


class RKLLMWrapper:
    """Wraps librkllmrt.so for direct ctypes access to the rkllm C API.

    Lifecycle: create once (loads .so), then call init_model/destroy
    repeatedly for model switches.  The library stays loaded.
    """

    def __init__(self, lib_path):
        self.lib = ctypes.CDLL(lib_path)
        self._setup_functions()
        self.handle = ctypes.c_void_p()
        self._model_loaded = False

    def _setup_functions(self):
        """Define argtypes/restype for all rkllm C functions."""
        # rkllm_init(handle*, param*, callback) -> int
        self.lib.rkllm_init.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(RKLLMParam),
            callback_type,
        ]
        self.lib.rkllm_init.restype = ctypes.c_int

        # rkllm_run(handle, input*, infer_param*, userdata) -> int  [BLOCKING]
        self.lib.rkllm_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.c_void_p,
        ]
        self.lib.rkllm_run.restype = ctypes.c_int

        # rkllm_destroy(handle) -> int
        self.lib.rkllm_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rkllm_destroy.restype = ctypes.c_int

        # rkllm_abort(handle) -> int
        self.lib.rkllm_abort.argtypes = [ctypes.c_void_p]
        self.lib.rkllm_abort.restype = ctypes.c_int

        # rkllm_clear_kv_cache(handle, keep_system_prompt, start_pos*, end_pos*) -> int
        try:
            self.lib.rkllm_clear_kv_cache.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ]
            self.lib.rkllm_clear_kv_cache.restype = ctypes.c_int
        except AttributeError:
            pass  # May not exist in older library versions

        # rkllm_set_chat_template(handle, system_prompt, prompt_prefix, prompt_postfix) -> int
        try:
            self.lib.rkllm_set_chat_template.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
            ]
            self.lib.rkllm_set_chat_template.restype = ctypes.c_int
        except AttributeError:
            pass  # May not exist in all library versions

    def init_model(self, model_path, ctx_len, max_tokens, vl_config=None):
        """Initialize a model.  Returns True on success.

        Args:
            vl_config: if provided, dict with img_start/img_end/img_content/base_domain_id
                       for VL model initialization.
        """
        param = RKLLMParam()
        param.model_path = model_path.encode('utf-8')
        param.max_context_len = ctx_len
        param.max_new_tokens = max_tokens
        param.top_k = 1
        param.n_keep = -1
        param.top_p = 0.9
        param.temperature = 0.8
        param.repeat_penalty = 1.1
        param.frequency_penalty = 0.0
        param.presence_penalty = 0.0
        param.mirostat = 0
        param.mirostat_tau = 5.0
        param.mirostat_eta = 0.1
        param.skip_special_token = True
        param.is_async = False
        if vl_config:
            param.img_start = vl_config.get('img_start', '').encode('utf-8')
            param.img_end = vl_config.get('img_end', '').encode('utf-8')
            param.img_content = vl_config.get('img_content', '').encode('utf-8')
            param.extend_param.base_domain_id = vl_config.get('base_domain_id', 1)
        else:
            param.img_start = b""
            param.img_end = b""
            param.img_content = b""
            param.extend_param.base_domain_id = 0
        param.extend_param.embed_flash = 1
        param.extend_param.n_batch = 1
        param.extend_param.use_cross_attn = 0
        param.extend_param.enabled_cpus_num = 4
        # RK3588 big cores (4-7)
        param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)

        self.handle = ctypes.c_void_p()
        ret = self.lib.rkllm_init(
            ctypes.byref(self.handle), ctypes.byref(param), _rkllm_callback
        )
        if ret != 0:
            self._model_loaded = False
            return False
        self._model_loaded = True
        return True

    def run(self, prompt, role="user", keep_history=1, enable_thinking=False):
        """Run inference (BLOCKING).  Must be called from a worker thread.

        Returns the rkllm_run return code (0 = success).
        """
        if not self._model_loaded:
            return -1

        rkllm_input = RKLLMInput()
        rkllm_input.role = role.encode('utf-8')
        rkllm_input.enable_thinking = ctypes.c_bool(enable_thinking)
        rkllm_input.input_type = RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode('utf-8'))

        infer_param = RKLLMInferParam()
        ctypes.memset(ctypes.byref(infer_param), 0, ctypes.sizeof(RKLLMInferParam))
        infer_param.mode = RKLLM_INFER_GENERATE
        infer_param.keep_history = keep_history

        return self.lib.rkllm_run(
            self.handle,
            ctypes.byref(rkllm_input),
            ctypes.byref(infer_param),
            None,
        )

    def run_multimodal(self, prompt, image_embed, n_image_tokens, n_image,
                        image_width, image_height, role="user", keep_history=0):
        """Run multimodal inference (BLOCKING) with image embeddings.

        Returns the rkllm_run return code (0 = success).
        """
        if not self._model_loaded:
            return -1

        # Keep reference to prevent GC during rkllm_run
        embed_ptr = image_embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        rkllm_input = RKLLMInput()
        ctypes.memset(ctypes.byref(rkllm_input), 0, ctypes.sizeof(RKLLMInput))
        rkllm_input.role = role.encode('utf-8')
        rkllm_input.enable_thinking = ctypes.c_bool(False)
        rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL
        rkllm_input.input_data.multimodal_input.prompt = prompt.encode('utf-8')
        rkllm_input.input_data.multimodal_input.image_embed = embed_ptr
        rkllm_input.input_data.multimodal_input.n_image_tokens = n_image_tokens
        rkllm_input.input_data.multimodal_input.n_image = n_image
        rkllm_input.input_data.multimodal_input.image_width = image_width
        rkllm_input.input_data.multimodal_input.image_height = image_height

        infer_param = RKLLMInferParam()
        ctypes.memset(ctypes.byref(infer_param), 0, ctypes.sizeof(RKLLMInferParam))
        infer_param.mode = RKLLM_INFER_GENERATE
        infer_param.keep_history = keep_history

        return self.lib.rkllm_run(
            self.handle,
            ctypes.byref(rkllm_input),
            ctypes.byref(infer_param),
            None,
        )

    def abort(self):
        """Abort running inference.  Non-blocking, safe to call anytime."""
        if not self._model_loaded:
            return
        try:
            self.lib.rkllm_abort(self.handle)
        except Exception:
            pass

    def clear_kv_cache(self):
        """Clear the runtime's internal KV cache.

        Call before a new conversation when using keep_history=1 so that
        stale context from a prior conversation is discarded.  The next
        rkllm_run with keep_history=1 starts from a clean slate AND
        retains the new turn's state for incremental follow-ups.
        """
        if not self._model_loaded:
            return
        try:
            ret = self.lib.rkllm_clear_kv_cache(
                self.handle, ctypes.c_int(0), None, None
            )
            if ret != 0:
                logger.warning(f"rkllm_clear_kv_cache returned error code {ret}")
        except Exception as e:
            logger.warning(f"rkllm_clear_kv_cache exception: {e}")

    def destroy(self):
        """Destroy the model and free NPU resources."""
        if not self._model_loaded:
            return
        try:
            self.lib.rkllm_destroy(self.handle)
        except Exception:
            pass
        self._model_loaded = False
        self.handle = ctypes.c_void_p()

    @property
    def is_loaded(self):
        return self._model_loaded


# =============================================================================
# GLOBALS
# =============================================================================
ACTIVE_REQUEST = {
    "id": None,
    "start_time": 0,
    "last_activity": 0,
    "model": None,
}
ACTIVE_LOCK = Lock()
ABORT_EVENT = Event()

_rkllm_wrapper = None     # RKLLMWrapper instance (singleton — one NPU)
_worker_thread = None      # Current inference worker thread
CURRENT_MODEL = None
PROCESS_LOCK = RLock()

# VL model state (separate from text model — dual model architecture)
_vision_encoder = None       # RKNNVisionEncoder instance (persistent)
_vl_rkllm_wrapper = None     # RKLLMWrapper for VL LLM decoder
VL_CURRENT_MODEL = None      # Name of loaded VL model (None = not loaded)
VL_LAST_REQUEST_TIME = 0     # Timestamp of last VL request (for idle auto-unload)

SHUTDOWN_EVENT = Event()
GENERATION_COMPLETE = Event()
GENERATION_COMPLETE.set()  # Initially set — no generation is running
SERVER_START_TIME = int(time.time())
LAST_REQUEST_TIME = 0

# KV Cache History Tracking
# Tracks what the rkllm runtime has in its internal KV cache so we can
# send only the new user message on follow-up turns (incremental mode)
# instead of the full concatenated conversation.  With keep_history=1,
# the runtime keeps all prior turns — we just need to add the new one.
_KV_LOCK = Lock()
_kv_cache_state = {
    "model": None,          # Model name currently in KV cache
    "user_messages": [],    # User messages (in order) the KV cache covers
}


def _check_kv_incremental(model_name, messages):
    """Check if KV cache allows incremental mode (send only last user message).

    Returns the last user message text if incremental is possible, else None.
    """
    with _KV_LOCK:
        if _kv_cache_state["model"] != model_name:
            return None

        user_msgs = [m['content'] for m in messages
                     if m.get('role') == 'user' and m.get('content')]
        if not user_msgs:
            return None

        # History = all user messages except the latest
        history = user_msgs[:-1]

        if history == _kv_cache_state["user_messages"]:
            return user_msgs[-1]
        return None


def _update_kv_tracking(model_name, messages, is_reset):
    """Update KV cache tracking after successful generation."""
    user_msgs = [m['content'] for m in messages
                 if m.get('role') == 'user' and m.get('content')]
    with _KV_LOCK:
        _kv_cache_state["model"] = model_name
        if is_reset:
            # Full reset — track all user messages from this conversation
            _kv_cache_state["user_messages"] = list(user_msgs)
        else:
            # Incremental — append only the latest user message
            if user_msgs:
                _kv_cache_state["user_messages"].append(user_msgs[-1])


def _reset_kv_tracking():
    """Reset KV tracking (model unloaded or KV cache cleared)."""
    with _KV_LOCK:
        _kv_cache_state["model"] = None
        _kv_cache_state["user_messages"] = []


# =============================================================================
# REQUEST TRACKING
# =============================================================================


def is_request_active():
    with ACTIVE_LOCK:
        if ACTIVE_REQUEST["id"] is None:
            return False

        elapsed = time.time() - ACTIVE_REQUEST["last_activity"]
        if elapsed > REQUEST_STALE_TIMEOUT:
            logger.warning(f"Request {ACTIVE_REQUEST['id']} stale ({elapsed:.0f}s idle) - auto-clearing")
            ACTIVE_REQUEST["id"] = None
            ACTIVE_REQUEST["start_time"] = 0
            ACTIVE_REQUEST["last_activity"] = 0
            ACTIVE_REQUEST["model"] = None
            return False
        return True


def try_start_request(request_id, model):
    """Atomic check-and-set: reject if busy, else claim the request slot."""
    with ACTIVE_LOCK:
        if ACTIVE_REQUEST["id"] is not None:
            elapsed = time.time() - ACTIVE_REQUEST["last_activity"]
            if elapsed <= REQUEST_STALE_TIMEOUT:
                return False
            logger.warning(f"Request {ACTIVE_REQUEST['id']} stale ({elapsed:.0f}s idle) - auto-clearing")
        ACTIVE_REQUEST["id"] = request_id
        ACTIVE_REQUEST["start_time"] = time.time()
        ACTIVE_REQUEST["last_activity"] = time.time()
        ACTIVE_REQUEST["model"] = model
        logger.info(f"Request STARTED: {request_id} for model {model}")
        return True


def force_clear_if_orphaned():
    """Clear orphaned request tracking if model changed or unloaded."""
    loaded = is_model_loaded()
    with ACTIVE_LOCK:
        if ACTIVE_REQUEST["id"] is None:
            return
        if CURRENT_MODEL != ACTIVE_REQUEST["model"]:
            # Model switched — only clear if request is stale
            elapsed = time.time() - ACTIVE_REQUEST["last_activity"]
            if elapsed <= REQUEST_STALE_TIMEOUT:
                return  # Model may still be loading
            logger.warning(f"Clearing orphaned request {ACTIVE_REQUEST['id']} "
                          f"- model mismatch ({ACTIVE_REQUEST['model']} vs "
                          f"{CURRENT_MODEL}) after {elapsed:.0f}s")
        elif not loaded:
            logger.warning(f"Clearing orphaned request {ACTIVE_REQUEST['id']} "
                          f"- model not loaded")
        else:
            return  # Model matches and is loaded — not orphaned
        ACTIVE_REQUEST["id"] = None
        ACTIVE_REQUEST["start_time"] = 0
        ACTIVE_REQUEST["last_activity"] = 0
        ACTIVE_REQUEST["model"] = None


def update_request_activity():
    with ACTIVE_LOCK:
        if ACTIVE_REQUEST["id"]:
            ACTIVE_REQUEST["last_activity"] = time.time()


def end_request(request_id):
    global LAST_REQUEST_TIME
    with ACTIVE_LOCK:
        if ACTIVE_REQUEST["id"] == request_id:
            logger.info(f"Request ENDED: {request_id}")
            ACTIVE_REQUEST["id"] = None
            ACTIVE_REQUEST["start_time"] = 0
            ACTIVE_REQUEST["last_activity"] = 0
            ACTIVE_REQUEST["model"] = None
            LAST_REQUEST_TIME = time.time()


def get_active_request_info():
    with ACTIVE_LOCK:
        if ACTIVE_REQUEST["id"]:
            return {
                "id": ACTIVE_REQUEST["id"],
                "model": ACTIVE_REQUEST["model"],
                "elapsed": time.time() - ACTIVE_REQUEST["start_time"],
                "idle": time.time() - ACTIVE_REQUEST["last_activity"],
            }
        return None


# =============================================================================
# UTILITIES
# =============================================================================


def resolve_model(requested_name):
    """Resolve a model name (including aliases) to (model_name, config)."""
    if not requested_name:
        return None, None
    name = requested_name.lower().strip()
    if name in ALIASES:
        logger.debug(f"Resolved alias '{requested_name}' -> '{ALIASES[name]}'")
        name = ALIASES[name]
    config = MODELS.get(name)
    if config is None:
        available = list(MODELS.keys()) + list(ALIASES.keys())
        logger.warning(f"Model '{requested_name}' not found. Available: {available}")
    return name, config


def is_model_loaded():
    """Check if a model is currently loaded and ready."""
    return _rkllm_wrapper is not None and _rkllm_wrapper.is_loaded


# =============================================================================
# RAG CONTENT CLEANING
# =============================================================================


def _strip_stale_date_claims(text):
    """Remove misleading 'current date/time is X' claims from web content.

    Web pages (especially time/date sites) embed phrases like
    'Current date is October 26, 2025' from cached or outdated pages.
    Small models (1.7B-4B) latch onto these and output them as the answer,
    ignoring the correct date in the system prompt.

    This strips such claims WITHOUT removing factual date references
    (e.g. 'DST starts on March 29, 2026' is kept).
    """
    # Patterns that claim to state what the "current" date/time is
    # These are the exact patterns that confuse small models
    _STALE_PATTERNS = [
        # "Current date is October 26, 2025" / "Current time is 12:00 PM"
        r'(?:the\s+)?current\s+(?:date|time|day)(?:\s+and\s+(?:date|time))?\s*(?:is|:)\s*[^.\n]{5,60}[.\n]?',
        # "Today is October 26, 2025" / "Today's date is..."
        r"today(?:'s\s+date)?\s+is\s*:?\s*[^.\n]{5,60}[.\n]?",
        # "Right now it is October 26, 2025"
        r'right\s+now\s+it\s+is\s*:?\s*[^.\n]{5,60}[.\n]?',
        # "The date today is..."
        r'the\s+date\s+today\s+is\s*:?\s*[^.\n]{5,60}[.\n]?',
        # "Local Time: Sunday, October 26, 2025, 12:00 PM"
        r'local\s+time\s*:\s*[^.\n]{5,60}[.\n]?',
    ]
    result = text
    for pat in _STALE_PATTERNS:
        matches = list(re.finditer(pat, result, re.IGNORECASE))
        if matches:
            for m in reversed(matches):
                stripped = m.group().strip()
                logger.debug(f"Stripped stale date claim: '{stripped[:80]}'")
            result = re.sub(pat, '', result, flags=re.IGNORECASE)
    if result != text:
        # Clean up any leftover double blank lines
        result = re.sub(r'\n{3,}', '\n\n', result)
        logger.info(f"Stale date cleanup: {len(text)} -> {len(result)} chars "
                    f"({len(text) - len(result)} chars removed)")
    return result


def _clean_web_content(text):
    """Strip web page navigation/boilerplate from scraped content.

    Open WebUI's SafeWebBaseLoader returns soup.get_text() — raw text with
    ALL navigation menus, cookie banners, sidebar links, footer cruft, and
    JS-framework text.

    Strategy — five-pass line-level filtering:
    Pass 0: Strip misleading "current date/time" claims from web pages
    Pass 1: Remove known boilerplate phrases
    Pass 2: Remove concatenated navigation (CamelCase, title-case, URL-heavy)
    Pass 3: Collapse consecutive short-line navigation menus
    Pass 4: Keep only lines with data signals
    """
    # --- Pass 0: Remove misleading date/time claims from web snippets ---
    # Web pages often contain stale "current date is X" or "today is Y" text
    # from cached pages that contradicts the actual current date. These confuse
    # small models into outputting wrong dates.
    text = _strip_stale_date_claims(text)

    lines = text.split('\n')
    pass1 = []

    for line in lines:
        s = line.strip()
        if not s:
            pass1.append('')
            continue

        # --- Pass 1: Known boilerplate phrases ---
        s_lower = s.lower()
        if len(s) < 150 and any(kw in s_lower for kw in (
                'sign in', 'log in', 'sign up', 'create account',
                'subscribe now', 'subscribe to', 'newsletter signup',
                'cookie policy', 'privacy policy', 'terms of service',
                'terms of use', 'terms & conditions', 'terms and conditions',
                'all rights reserved', 'skip to content',
                'skip to main', 'skip to navigation',
                'advertisement', 'sponsored content', 'sponsored by',
                'accept cookies', 'accept all', 'cookie settings',
                'manage preferences', 'cookie consent',
                'follow us on', 'share on facebook', 'share on twitter',
                'tweet this', 'pin it', 'share this',
                'back to top', 'read more', 'show more', 'load more',
                'related articles', 'you may also like',
                'recommended for you', 'trending now',
                'download the app', 'get the app',
                'unsubscribe', 'no thanks', 'dismiss',
                'powered by', 'built with')):
            continue

        # --- Pass 2: Navigation pattern detection ---
        if len(re.findall(r'[a-z][A-Z]', s)) >= 2 and not re.search(r'[.!?]', s):
            continue

        words = s.split()
        if len(words) >= 4 and not re.search(r'[.!?]', s):
            cap_words = sum(1 for w in words if w and w[0].isupper())
            if cap_words / len(words) > 0.65:
                continue

        url_count = len(re.findall(r'https?://', s))
        if url_count >= 2:
            continue
        if url_count == 1 and len(s) < 80 and not re.search(r'[.!?]', s):
            continue

        pass1.append(s)

    # --- Pass 3: Collapse consecutive short-line runs (menus) ---
    pass2 = []
    run_start = -1
    run_count = 0
    for i, line in enumerate(pass1):
        is_short_nav = (line and len(line) < 30 and not re.search(r'[.!?]', line))
        is_empty = not line
        if is_short_nav or (is_empty and run_count > 0):
            if run_count == 0:
                run_start = i
            run_count += 1
        else:
            pending_start = run_start
            pending_count = run_count
            run_count = 0
            run_start = -1
            if pending_count >= 4:
                pass  # Drop navigation menu
            elif pending_start >= 0:
                for j in range(pending_start, pending_start + pending_count):
                    if j < len(pass1):
                        pass2.append(pass1[j])
            pass2.append(line)
    if run_count >= 4:
        pass
    elif run_start >= 0:
        for j in range(run_start, run_start + run_count):
            if j < len(pass1):
                pass2.append(pass1[j])

    # --- Pass 4: Content signal checks ---
    cleaned = []
    prev_empty = False
    for s in pass2:
        if not s:
            if not prev_empty and cleaned:
                cleaned.append('')
                prev_empty = True
            continue
        prev_empty = False

        if re.search(r'\d', s):
            cleaned.append(s)
            continue
        if re.search(r'[.!?;:]', s):
            cleaned.append(s)
            continue
        if len(s) >= 40:
            cleaned.append(s)
            continue

    result = '\n'.join(cleaned).strip()
    return result if len(result) > 100 else text


def _score_paragraph(para, query_words=None):
    """Score a paragraph for content relevance (higher = more useful)."""
    score = 0
    words = para.split()
    word_count = len(words)

    if word_count >= 5:
        sw_count = sum(1 for w in words if w.lower() in ENGLISH_STOPWORDS)
        sw_density = sw_count / word_count
        if sw_density >= 0.30:
            score += 3
        elif sw_density >= 0.20:
            score += 1
        elif word_count >= 15:
            score -= 2

    if len(para) >= 200:
        score += 3
    elif len(para) >= 100:
        score += 2
    elif len(para) >= 50:
        score += 1

    sentences = [s.strip() for s in re.split(r'[.!?]', para)
                 if len(s.strip().split()) >= 4]
    score += min(len(sentences), 3)

    digit_groups = re.findall(r'\d[\d,.:%/\-]*', para)
    if sentences:
        score += min(len(digit_groups), 4)
    elif digit_groups and word_count >= 8:
        score += min(len(digit_groups), 2)

    if query_words:
        para_lower = para.lower()
        matches = sum(1 for w in query_words if w in para_lower)
        score += matches * 3

    if len(para) < 20:
        score -= 4
    elif len(para) < 40:
        score -= 2

    if word_count >= 3:
        cap_words = sum(1 for w in words if w and w[0].isupper())
        if cap_words / word_count > 0.7 and not re.search(r'[.!?]', para):
            score -= 3

    emoji_count = len(re.findall(r'[\U0001f300-\U0001f9ff⏱⏰🌍🌐🏙🏴🗺🌦🌅🌇🕰🎉📆🔁🌬]', para))
    if emoji_count >= 2:
        score -= 3

    lower = para.lower()
    if any(kw in lower for kw in (
            'copyright', 'all rights reserved', 'privacy policy',
            'terms of service', 'cookie policy', 'sign in',
            'subscribe', 'advertisement', 'sponsored',
            'free widget', 'webmaster', 'full screen clock',
            'atomic-clock', 'advert free')):
        score -= 4

    return score


# =============================================================================
# STRING HELPERS
# =============================================================================


def _strip_system_fluff(text):
    """Remove generic assistant instructions from system messages."""
    generic_phrases = [
        r'you are a helpful assistant\s*(?:[.!?]\s*|$)',
        r'you are an? ai assistant\s*(?:[.!?]\s*|$)',
        r'you are an? assistant\s*(?:[.!?]\s*|$)',
        r'you are an? helpful ai\s*(?:[.!?]\s*|$)',
        r'as an ai assistant\s*(?:[.!?]\s*|$)',
        r'as a helpful assistant\s*(?:[.!?]\s*|$)',
    ]

    result = text
    for pat in generic_phrases:
        result = re.sub(pat, '', result, flags=re.IGNORECASE)

    result = result.strip()
    if result != text.strip():
        logger.debug(f"Stripped system fluff: {len(text)} -> {len(result)} chars")
    return result


def _is_date_only_system(text):
    """Check if system message contains ONLY date/time info."""
    s = text.strip()
    if re.match(r'^today is \d{4}-\d{2}-\d{2}\b.*$', s, re.IGNORECASE):
        lines = [l.strip() for l in s.split('\n') if l.strip()]
        if len(lines) <= 1:
            return True
    return False


# =============================================================================
# RAG EXTRACTION
# =============================================================================


def _extract_rag_reference(system_text):
    """Extract reference data from a RAG-injected system message."""
    has_source = '<source' in system_text
    has_context_tags = '<context>' in system_text.lower()
    has_ref_preamble = 'reference information' in system_text.lower()

    if not has_source and not has_ref_preamble and not has_context_tags:
        return None

    if has_source:
        source_start = system_text.find('<source')
        instructions = system_text[:source_start].strip()

        last_close = system_text.rfind('</source>')
        if last_close >= 0:
            last_close += len('</source>')
        else:
            last_close = len(system_text)

        reference_data = system_text[source_start:last_close]

        trailing = system_text[last_close:].strip()
        rag_question = ""
        if trailing:
            q_match = re.search(
                r'(?:answer\s+(?:the\s+following|this)\s+question\s*[:\-]\s*)(.+?)$',
                trailing, re.IGNORECASE | re.DOTALL
            )
            if q_match:
                rag_question = q_match.group(1).strip()
    else:
        instructions = ""
        reference_data = system_text
        rag_question = ""

    instructions = re.sub(r'(?:Here is some )?reference information\s*:?\s*$', '', instructions, flags=re.IGNORECASE).strip()
    instructions = re.sub(r'###\s*Task:.*$', '', instructions, flags=re.IGNORECASE | re.DOTALL).strip()
    instructions = re.sub(r'</?context>\s*', '', instructions, flags=re.IGNORECASE).strip()

    raw_len = len(reference_data)
    reference_data = re.sub(r'</?source[^>]*>', '', reference_data)
    reference_data = re.sub(r'</?context>\s*', '', reference_data, flags=re.IGNORECASE)
    reference_data = _clean_web_content(reference_data)
    reference_data = re.sub(r'\n{3,}', '\n\n', reference_data)
    reference_data = re.sub(r'[ \t]{2,}', ' ', reference_data)
    logger.debug(f"RAG content cleaned: {raw_len} -> {len(reference_data)} chars "
                 f"({100 - len(reference_data) * 100 // max(raw_len, 1)}% removed)")

    return (instructions, reference_data, rag_question)


# =============================================================================
# PROMPT BUILDER
# =============================================================================


def build_prompt(messages, model_name):
    """Convert OpenAI messages array to plain text for rkllm.

    Returns (prompt, is_rag, enable_thinking).

    IMPORTANT: The rkllm runtime applies chat templates INTERNALLY using
    actual token IDs. We send ONLY plain text.

    enable_thinking: True = model can use <think> reasoning mode,
                     False = thinking disabled (set on RKLLMInput.enable_thinking)
    """
    model_cfg = MODELS.get(model_name, {})
    ctx = model_cfg.get('context_length', CONTEXT_LENGTH_DEFAULT) if model_cfg else CONTEXT_LENGTH_DEFAULT

    system_parts = []
    conversation = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        # OpenAI multimodal: content can be a list of {"type":"text","text":"..."}
        if isinstance(content, list):
            content = " ".join(
                part.get('text', '') for part in content
                if isinstance(part, dict) and part.get('type') == 'text'
            )
        if not isinstance(content, str):
            content = str(content) if content else ''
        if not content:
            continue
        if role == 'system':
            system_parts.append(content)
        else:
            conversation.append((role, content))
    if len(system_parts) > 1:
        logger.warning(f"Multiple system messages ({len(system_parts)}) — concatenating")
    system_text = "\n".join(system_parts)

    user_question = ""
    for role, content in reversed(conversation):
        if role == 'user':
            user_question = content
            break

    rag_parts = _extract_rag_reference(system_text) if system_text else None

    prompt = ""
    enable_thinking = True  # Default: allow thinking

    # =====================================================================
    # FOLLOW-UP / IRRELEVANT-RAG DETECTION
    # =====================================================================
    _has_assistant_turn = any(r == 'assistant' for r, _ in conversation)

    if rag_parts:
        _skip_reason = None
        _query_normalized = user_question.strip().lower().rstrip('?!.,')

        # --- Layer 0: Document-referential bypass ---
        # Queries that clearly reference the uploaded document should ALWAYS
        # use RAG mode — they are never irrelevant follow-ups.
        _DOC_REF_WORDS = {
            'summarize', 'summary', 'summarise', 'explain', 'describe',
            'review', 'analyze', 'analyse', 'outline', 'extract', 'list',
            'translate', 'read', 'parse', 'interpret', 'breakdown',
            'attached', 'document', 'file', 'pdf', 'upload', 'uploaded',
            'above', 'content', 'data', 'table', 'report', 'invoice',
            'receipt', 'payslip', 'letter', 'contract', 'page', 'text',
        }
        _DOC_REF_PHRASES = (
            'see attached', 'the above', 'this document', 'this file',
            'the document', 'the file', 'what does', 'what is',
            'break down', 'break it down', 'tell me about this',
            'what about', 'how much', 'how many',
        )
        _query_word_set = set(_query_normalized.split())
        _is_doc_ref = bool(_query_word_set & _DOC_REF_WORDS)
        if not _is_doc_ref:
            _is_doc_ref = any(p in _query_normalized for p in _DOC_REF_PHRASES)
        if _is_doc_ref:
            logger.info(f"Document-referential query '{user_question}' — forcing RAG mode")

        # --- Layer 1: Known conversational words / phrases ---
        if not _is_doc_ref:
            _SHORT_REPLY_WORDS = {
                'yes', 'no', 'yeah', 'nah', 'yep', 'nope', 'ok', 'okay',
                'sure', 'thanks', 'thank you', 'please', 'go ahead',
                'continue', 'more', 'why', 'how', 'what', 'really',
                'cool', 'nice', 'wow', 'great', 'interesting',
                'tell me more', 'go on', 'and', 'also',
            }
            if _query_normalized in _SHORT_REPLY_WORDS:
                _skip_reason = f"short conversational reply '{user_question}'"

        # --- Layer 2: Query-to-reference topical overlap ---
        # Only skip RAG when the query is clearly unrelated to the reference
        # data (none of the content words appear in the document).
        if not _skip_reason and not _is_doc_ref and _has_assistant_turn:
            _qcw = {w.lower().strip('?!.,;:\'"()') for w in user_question.split()
                    if w.lower().strip('?!.,;:\'"()') not in ENGLISH_STOPWORDS and len(w.strip('?!.,;:\'"()')) > 2}
            if _qcw and len(_qcw) <= 8:
                _, ref_text, _ = rag_parts
                ref_lower = ref_text.lower()
                hits = sum(1 for w in _qcw if w in ref_lower)
                overlap = hits / len(_qcw)
                if overlap == 0:
                    _skip_reason = (f"zero query-reference overlap ({len(_qcw)} "
                                    f"content words, none in document: {_qcw})")

        if _skip_reason:
            logger.info(f"RAG SKIP: {_skip_reason} — using normal mode with chat history")
            rag_parts = None
            system_text = ""

    if rag_parts and user_question:
        # === RAG MODE ===
        instructions, reference_data, rag_question = rag_parts

        logger.info(f"RAG mode: instructions={len(instructions)} chars, "
                     f"reference={len(reference_data)} chars, "
                     f"rag_question='{rag_question[:80]}', "
                     f"user_question='{user_question[:80]}'")

        max_ref_chars = int(ctx * 1.5) - len(user_question) - 200
        max_ref_chars = max(500, max_ref_chars)

        _rag_best_score = None

        if len(reference_data) > max_ref_chars:
            original_ref = len(reference_data)
            raw_paras = [p.strip() for p in reference_data.split('\n\n') if p.strip()]
            merged = []
            current_block = []
            for p in raw_paras:
                if len(p) < 60 and not re.search(r'[.!?]$', p):
                    current_block.append(p)
                else:
                    if current_block:
                        merged.append(' '.join(current_block))
                        current_block = []
                    merged.append(p)
            if current_block:
                merged.append(' '.join(current_block))
            paras = [p for p in merged if p.strip()]

            # Deduplicate
            seen = set()
            unique_paras = []
            for p in paras:
                key = re.sub(r'[\s\d:]+', '', p.lower())[:80]
                if key not in seen:
                    seen.add(key)
                    unique_paras.append(p)
            deduped = []
            for p in unique_paras:
                is_dup = False
                for kept in deduped:
                    if _jaccard_similarity(p, kept) >= RAG_DEDUP_SIMILARITY:
                        is_dup = True
                        break
                if not is_dup:
                    deduped.append(p)
            if len(deduped) < len(unique_paras):
                logger.info(f"Similarity dedup: {len(unique_paras)} -> {len(deduped)} paragraphs "
                            f"(threshold={RAG_DEDUP_SIMILARITY})")
            paras = deduped

            if len(paras) >= 3:
                stop = {'the', 'a', 'an', 'is', 'in', 'of', 'at', 'to',
                        'for', 'and', 'or', 'on', 'it', 'be', 'as', 'do',
                        'by', 'this', 'that', 'what', 'when', 'where',
                        'how', 'who', 'which', 'my', 'your', 'can'}
                qwords = {w.lower() for w in user_question.split()
                          if w.lower() not in stop and len(w) > 1}

                scored = [(i, _score_paragraph(p, qwords), p)
                          for i, p in enumerate(paras)]
                scored.sort(key=lambda x: x[1], reverse=True)
                _rag_best_score = scored[0][1] if scored else 0

                if logger.isEnabledFor(logging.DEBUG):
                    for i, (idx, sc, p) in enumerate(scored[:5]):
                        logger.debug(f"  TOP-{i+1} score={sc}: {p[:100]}")
                    for i, (idx, sc, p) in enumerate(scored[-3:]):
                        logger.debug(f"  BOT-{i+1} score={sc}: {p[:100]}")

                picked = []
                budget = max_ref_chars
                for idx, sc, para in scored:
                    if budget <= 0:
                        break
                    if len(picked) >= RAG_MAX_PARAGRAPHS:
                        break
                    if sc < RAG_MIN_QUALITY_SCORE:
                        continue
                    cost = len(para) + 2
                    if cost <= budget:
                        picked.append(idx)
                        budget -= cost

                if picked:
                    picked.sort()
                    reference_data = '\n\n'.join(paras[i] for i in picked)
                    logger.info(f"Smart truncation: {len(raw_paras)} raw -> "
                                f"{len(paras)} merged/deduped -> "
                                f"{len(picked)} picked ({len(reference_data)} "
                                f"chars) [min_score={RAG_MIN_QUALITY_SCORE}, "
                                f"max_paras={RAG_MAX_PARAGRAPHS}, "
                                f"{len(qwords)} keywords]")
                else:
                    top_n = scored[:10]
                    picked_idx = sorted([idx for idx, sc, p in top_n])
                    reference_data = '\n\n'.join(paras[i] for i in picked_idx)
                    if len(reference_data) > max_ref_chars:
                        reference_data = reference_data[:max_ref_chars]
                        last_nl = reference_data.rfind('\n')
                        if last_nl > max_ref_chars // 2:
                            reference_data = reference_data[:last_nl]
                    logger.info(f"Score fallback: took top {len(picked_idx)} paras "
                                f"({len(reference_data)} chars)")
            else:
                reference_data = reference_data[:max_ref_chars]
                last_nl = reference_data.rfind('\n')
                if last_nl > max_ref_chars // 2:
                    reference_data = reference_data[:last_nl]
            logger.warning(f"Reference data truncated: {original_ref} -> {len(reference_data)} chars")

        # Quality floor check
        if (_rag_best_score is not None
                and _rag_best_score < RAG_QUALITY_FLOOR_THRESHOLD):
            logger.warning(f"Quality floor triggered: best_score={_rag_best_score} < "
                          f"threshold={RAG_QUALITY_FLOOR_THRESHOLD}. "
                          f"Dropping RAG context, falling back to model knowledge.")
            rag_parts = None
            system_text = ""

    # Re-check after quality floor may have cleared rag_parts
    if rag_parts and user_question:
        # enable_thinking for RAG: disabled on small context models
        enable_thinking = ctx >= DISABLE_THINK_FOR_RAG_BELOW_CTX
        abstention = ". If not answered above, say you don't know" if enable_thinking else ''
        logger.info(f"RAG thinking: ctx={ctx}, threshold={DISABLE_THINK_FOR_RAG_BELOW_CTX}, "
                    f"thinking={'enabled' if enable_thinking else 'disabled'}")

        # --- Date anchor injection ---
        # Prepend a clear date statement to the RAG reference data so the
        # model sees the correct date IMMEDIATELY before the web content.
        # This counteracts stale dates that may remain in search snippets.
        _today = datetime.now().strftime('%B %d, %Y')
        _date_anchor = f"[Current date: {_today}. Any conflicting dates below are outdated.]"
        reference_data = f"{_date_anchor}\n\n{reference_data}"
        logger.debug(f"Injected date anchor: {_date_anchor}")

        # Clean trailing punctuation from user question to avoid double-period
        _clean_q = user_question.rstrip(' .!?;:')

        # Detect summarization-style queries for stronger prompting
        _summ_keywords = ('summarize', 'summary', 'summarise', 'overview',
                          'outline', 'describe', 'explain', 'what is this',
                          'what does this', 'tell me about', 'break down')
        _is_summary = any(kw in _clean_q.lower() for kw in _summ_keywords)

        if _is_summary:
            _instruction = (
                "Provide a thorough and comprehensive answer. "
                "Cover all major points, sections, and key details from the document. "
                "Use multiple paragraphs"
            )
        else:
            _instruction = (
                "Answer in detail with specific facts and examples"
            )

        prompt = (
            f"{reference_data}\n\n"
            f"According to the above, {_clean_q}. "
            f"{_instruction}"
            f"{abstention}."
        )
        logger.debug(f"RAG prompt built ({len(prompt)} chars, ctx={ctx})")
        if len(prompt) > 500:
            logger.debug(f"RAG prompt START: {prompt[:250]}")
            logger.debug(f"RAG prompt END: {prompt[-250:]}")
        else:
            logger.debug(f"RAG prompt FULL: {prompt}")
        approx_tokens = len(prompt) // CHARS_PER_TOKEN_ESTIMATE
        if approx_tokens > ctx * 0.85:
            logger.warning(
                f"RAG prompt (~{approx_tokens} tokens) approaches context limit ({ctx}). "
                f"Reference data may be truncated by the model."
            )
        return prompt, True, enable_thinking

    if not rag_parts or not user_question:
        # === NORMAL MODE (no RAG) ===
        parts = []
        turn_count = sum(1 for r, _ in conversation if r in ('user', 'assistant'))
        multi_turn = turn_count > 1

        if system_text:
            cleaned_sys = _strip_system_fluff(system_text)
            if cleaned_sys:
                if _is_date_only_system(cleaned_sys):
                    date_related = bool(re.search(
                        r'\b(?:date|time|today|tonight|tomorrow|yesterday'
                        r'|day|week|month|year|when|clock|hour|minute|schedule)\b',
                        user_question, re.IGNORECASE))
                    if date_related:
                        parts.append(cleaned_sys)
                    else:
                        logger.debug(f"Omitting date-only system prompt for non-date question")
                else:
                    parts.append(cleaned_sys)

        for role, content in conversation:
            if multi_turn:
                if role == 'user':
                    parts.append(f"User: {content}")
                elif role == 'assistant':
                    parts.append(f"Assistant: {content}")
            else:
                parts.append(content)

        prompt = "\n".join(parts)
        enable_thinking = True  # Default: allow thinking in normal mode

        # --- Open WebUI meta-task detection ---
        # Open WebUI sends internal tasks (search query gen, title gen, tag gen)
        # as regular chat completions. These are simple JSON/text tasks where
        # thinking mode wastes 20+ seconds and can confuse result parsing.
        _user_lower = user_question.lower()
        _is_meta = any(sig in _user_lower for sig in _OPENWEBUI_META_TASK_SIGNATURES)
        if _is_meta:
            enable_thinking = False
            logger.info(f"Meta-task detected — thinking disabled for speed")

        # --- Home Assistant request detection ---
        # HA's Extended OpenAI Conversation sends a system prompt with
        # smart-home device lists.  Disable thinking to cut latency in half.
        if enable_thinking and system_text:
            _sys_lower = system_text.lower()
            if any(sig in _sys_lower for sig in _HOME_ASSISTANT_SIGNATURES):
                enable_thinking = False
                logger.info("Home Assistant request detected — thinking disabled for speed")

    logger.debug(f"Prompt built ({len(prompt)} chars, ctx={ctx}, "
                 f"rag={'yes' if rag_parts else 'no'})")

    if len(prompt) > 500:
        logger.debug(f"Prompt START: {prompt[:250]}")
        logger.debug(f"Prompt END: {prompt[-250:]}")
    else:
        logger.debug(f"Prompt FULL: {prompt}")

    approx_tokens = len(prompt) // CHARS_PER_TOKEN_ESTIMATE
    if approx_tokens > ctx * 0.85:
        logger.warning(
            f"Prompt (~{approx_tokens} tokens) approaches context limit ({ctx}). "
            f"Model may truncate input."
        )

    return prompt, False, enable_thinking


# =============================================================================
# MODEL MANAGEMENT
# =============================================================================


def load_model(model_name, config):
    """Load an rkllm model via ctypes.  Reuses existing if same model."""
    global _rkllm_wrapper, CURRENT_MODEL, _worker_thread

    # If a previous generation is still running, abort it first
    if not GENERATION_COMPLETE.is_set():
        logger.warning("Previous generation still active — aborting for clean state")
        if _rkllm_wrapper:
            _rkllm_wrapper.abort()
        with PROCESS_LOCK:
            if _worker_thread and _worker_thread.is_alive():
                _worker_thread.join(timeout=10)
            _worker_thread = None
        GENERATION_COMPLETE.set()

    with PROCESS_LOCK:
        logger.info(f"load_model called for {model_name} (current loaded: {CURRENT_MODEL})")

        # Reuse if same model already loaded
        if CURRENT_MODEL == model_name and _rkllm_wrapper and _rkllm_wrapper.is_loaded:
            logger.info(f"REUSING loaded model {model_name}")
            return True

        # Unload different model if needed
        if CURRENT_MODEL and CURRENT_MODEL != model_name:
            logger.info(f"Model switch detected: unloading {CURRENT_MODEL}")
            unload_current("model switch")

        # Create wrapper (load library) if needed — only happens once
        if _rkllm_wrapper is None:
            try:
                logger.info(f"Loading rkllm library from {RKLLM_LIB_PATH}")
                _rkllm_wrapper = RKLLMWrapper(RKLLM_LIB_PATH)
            except OSError as e:
                logger.error(f"Failed to load rkllm library from {RKLLM_LIB_PATH}: {e}")
                return False

        # Initialize model
        model_path = config['path']
        ctx_len = config.get('context_length', CONTEXT_LENGTH_DEFAULT)
        max_tokens = config.get('max_tokens', MAX_TOKENS_DEFAULT)

        logger.info(f"STARTING LOAD of {model_name} (ctx={ctx_len}, max_tok={max_tokens})...")
        load_start = time.time()

        if not _rkllm_wrapper.init_model(model_path, ctx_len, max_tokens):
            # NPU memory may be exhausted by VL model — try unloading VL and retrying
            if VL_CURRENT_MODEL:
                logger.warning(f"Model {model_name} init failed with VL loaded — "
                              f"unloading VL model to free NPU memory and retrying")
                _unload_vl_model("NPU memory pressure — text model init failed")
                if _rkllm_wrapper.init_model(model_path, ctx_len, max_tokens):
                    logger.info(f"Model {model_name} loaded successfully after VL unload")
                    elapsed = time.time() - load_start
                    logger.info(f"{model_name} LOADED successfully in {elapsed:.1f}s (ctx={ctx_len})")
                    CURRENT_MODEL = model_name
                    return True
            logger.error(f"Model {model_name} failed to initialize "
                        f"(check model file integrity and rkllm version compatibility)")
            return False

        elapsed = time.time() - load_start
        logger.info(f"{model_name} LOADED successfully in {elapsed:.1f}s (ctx={ctx_len})")
        CURRENT_MODEL = model_name
        return True


def unload_current(reason="requested"):
    """Unload the current model and free NPU resources."""
    global CURRENT_MODEL, _worker_thread

    with PROCESS_LOCK:
        if not _rkllm_wrapper or not _rkllm_wrapper.is_loaded:
            logger.debug("No model to unload - skipping")
            CURRENT_MODEL = None
            return

        logger.info(f">>> START UNLOADING {CURRENT_MODEL} - reason: {reason}")
        unload_start = time.time()

        # Abort any running inference
        _rkllm_wrapper.abort()

        # Wait for worker thread to finish
        if _worker_thread and _worker_thread.is_alive():
            _worker_thread.join(timeout=5)
            if _worker_thread.is_alive():
                logger.warning("Worker thread did not finish after abort — proceeding with destroy")
        _worker_thread = None

        # Destroy the model (frees NPU memory)
        _rkllm_wrapper.destroy()
        CURRENT_MODEL = None
        _reset_kv_tracking()

        logger.info(f">>> UNLOAD COMPLETE in {time.time() - unload_start:.1f}s")


# =============================================================================
# VL MODEL MANAGEMENT
# =============================================================================


def _find_vl_model():
    """Find the first available VL model. Returns (name, config) or (None, None)."""
    for name, config in MODELS.items():
        if 'vl_config' in config and 'vision_encoder_path' in config:
            return name, config
    return None, None


def _load_vl_model(vl_name, vl_config):
    """Load VL model (vision encoder + LLM decoder). Returns True on success."""
    global _vision_encoder, _vl_rkllm_wrapper, VL_CURRENT_MODEL

    if not _VL_DEPS_AVAILABLE:
        logger.error("VL dependencies not available -- install numpy and Pillow")
        return False

    with PROCESS_LOCK:
        if (VL_CURRENT_MODEL == vl_name
                and _vl_rkllm_wrapper and _vl_rkllm_wrapper.is_loaded
                and _vision_encoder and _vision_encoder.is_loaded):
            logger.info(f"REUSING loaded VL model {vl_name}")
            return True

        load_start = time.time()
        logger.info(f"Loading VL model {vl_name}...")

        if _vision_encoder is None or not _vision_encoder.is_loaded:
            try:
                _vision_encoder = RKNNVisionEncoder(RKNN_LIB_PATH)
                encoder_path = vl_config['vision_encoder_path']
                if not _vision_encoder.init_model(encoder_path):
                    logger.error(f"Vision encoder failed to load: {encoder_path}")
                    _vision_encoder = None
                    return False
                logger.info(f"Vision encoder loaded: {os.path.basename(encoder_path)} "
                           f"({_vision_encoder.model_width}x{_vision_encoder.model_height}, "
                           f"tokens={_vision_encoder.model_image_token}, "
                           f"embed={_vision_encoder.model_embed_size})")
            except OSError as e:
                logger.error(f"Failed to load RKNN library from {RKNN_LIB_PATH}: {e}")
                _vision_encoder = None
                return False

        if VL_CURRENT_MODEL and VL_CURRENT_MODEL != vl_name:
            if _vl_rkllm_wrapper and _vl_rkllm_wrapper.is_loaded:
                logger.info(f"VL model switch: unloading {VL_CURRENT_MODEL}")
                _vl_rkllm_wrapper.destroy()

        if _vl_rkllm_wrapper is None:
            try:
                _vl_rkllm_wrapper = RKLLMWrapper(RKLLM_LIB_PATH)
            except OSError as e:
                logger.error(f"Failed to load rkllm library for VL: {e}")
                return False

        if not _vl_rkllm_wrapper.is_loaded:
            model_path = vl_config['path']
            ctx_len = vl_config.get('context_length', CONTEXT_LENGTH_DEFAULT)
            max_tokens = vl_config.get('max_tokens', MAX_TOKENS_DEFAULT)
            vl_cfg = vl_config.get('vl_config')
            if not _vl_rkllm_wrapper.init_model(model_path, ctx_len, max_tokens, vl_config=vl_cfg):
                logger.error(f"VL model {vl_name} failed to initialize")
                return False

        elapsed = time.time() - load_start
        VL_CURRENT_MODEL = vl_name
        logger.info(f"VL model {vl_name} LOADED in {elapsed:.1f}s")
        return True


def _unload_vl_model(reason="requested"):
    """Unload VL model resources."""
    global _vision_encoder, _vl_rkllm_wrapper, VL_CURRENT_MODEL

    with PROCESS_LOCK:
        if _vl_rkllm_wrapper and _vl_rkllm_wrapper.is_loaded:
            logger.info(f"Unloading VL model {VL_CURRENT_MODEL} -- reason: {reason}")
            _vl_rkllm_wrapper.abort()
            _vl_rkllm_wrapper.destroy()

        if _vision_encoder and _vision_encoder.is_loaded:
            _vision_encoder.destroy()
            _vision_encoder = None

        VL_CURRENT_MODEL = None
        logger.info("VL model unloaded")


# =============================================================================
# MODEL MONITOR
# =============================================================================


def model_monitor():
    """Background thread to monitor model health and auto-unload on idle."""
    logger.info("Model monitor started")
    while not SHUTDOWN_EVENT.is_set():
        try:
            force_clear_if_orphaned()

            # Auto-unload text model after idle timeout
            if (IDLE_UNLOAD_TIMEOUT > 0 and is_model_loaded()
                    and not is_request_active() and LAST_REQUEST_TIME > 0
                    and (time.time() - LAST_REQUEST_TIME) > IDLE_UNLOAD_TIMEOUT):
                logger.info(f"Auto-unloading {CURRENT_MODEL} after "
                            f"{int(time.time() - LAST_REQUEST_TIME)}s idle")
                unload_current("idle timeout")

            # Auto-unload VL model after idle timeout (frees NPU memory)
            if (VL_IDLE_UNLOAD_TIMEOUT > 0 and VL_CURRENT_MODEL
                    and not is_request_active() and VL_LAST_REQUEST_TIME > 0
                    and (time.time() - VL_LAST_REQUEST_TIME) > VL_IDLE_UNLOAD_TIMEOUT):
                logger.info(f"Auto-unloading VL model {VL_CURRENT_MODEL} after "
                            f"{int(time.time() - VL_LAST_REQUEST_TIME)}s idle")
                _unload_vl_model("idle timeout")
        except Exception as e:
            logger.error(f"Monitor error: {e}")

        SHUTDOWN_EVENT.wait(MONITOR_INTERVAL)
    logger.info("Model monitor stopped")


_monitor_thread = Thread(target=model_monitor, daemon=True)
_monitor_thread.start()


# =============================================================================
# SHUTDOWN
# =============================================================================


def shutdown():
    """Clean shutdown - destroy text model, VL model, stop monitor."""
    logger.info("Shutting down RKLLM API...")
    SHUTDOWN_EVENT.set()
    ABORT_EVENT.set()
    try:
        unload_current("shutdown")
    except Exception as e:
        logger.error(f"Error during shutdown unload: {e}")
    try:
        _unload_vl_model("shutdown")
    except Exception as e:
        logger.error(f"Error during VL shutdown unload: {e}")
    logger.info("Shutdown complete")


atexit.register(shutdown)


def signal_handler(signum, frame):
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    logger.info(f"Signal {sig_name} received - shutting down")
    shutdown()
    sys.exit(0)


# =============================================================================
# SSE HELPERS
# =============================================================================


def make_sse_chunk(request_id, model, created, delta=None, finish_reason=None, usage=None):
    """Build one SSE data line in OpenAI streaming format."""
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
    }
    if usage is not None and delta is None and finish_reason is None:
        chunk["choices"] = []
        chunk["usage"] = usage
    else:
        chunk["choices"] = [{
            "index": 0,
            "delta": delta if delta is not None else {},
            "logprobs": None,
            "finish_reason": finish_reason,
        }]
        if usage is not None:
            chunk["usage"] = usage
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def make_error_response(message, status_code=500, error_type="server_error"):
    """Build an OpenAI-compatible error response."""
    return jsonify({
        "error": {
            "message": message,
            "type": error_type,
            "code": status_code,
        }
    }), status_code


# =============================================================================
# THINK TAG PARSER
# =============================================================================


class ThinkTagParser:
    """State machine to parse <think>...</think> tags from a token stream."""

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self):
        self.in_thinking = False
        self.buffer = ""
        self.saw_think_block = False

    def feed(self, text):
        """Feed text into the parser. Yields (kind, text) tuples."""
        self.buffer += text

        while self.buffer:
            if self.in_thinking:
                close_pos = self.buffer.find(self.CLOSE_TAG)
                if close_pos != -1:
                    thinking_text = self.buffer[:close_pos]
                    if thinking_text:
                        yield ('thinking', thinking_text)
                    self.buffer = self.buffer[close_pos + len(self.CLOSE_TAG):]
                    self.in_thinking = False
                    if self.buffer.startswith('\n'):
                        self.buffer = self.buffer[1:]
                    continue
                else:
                    safe = len(self.buffer)
                    for i in range(1, len(self.CLOSE_TAG)):
                        if self.buffer.endswith(self.CLOSE_TAG[:i]):
                            safe = len(self.buffer) - i
                            break
                    if safe > 0:
                        yield ('thinking', self.buffer[:safe])
                        self.buffer = self.buffer[safe:]
                    break
            else:
                open_pos = self.buffer.find(self.OPEN_TAG)
                if open_pos != -1:
                    content_text = self.buffer[:open_pos]
                    if content_text:
                        yield ('content', content_text)
                    self.buffer = self.buffer[open_pos + len(self.OPEN_TAG):]
                    self.in_thinking = True
                    self.saw_think_block = True
                    if self.buffer.startswith('\n'):
                        self.buffer = self.buffer[1:]
                    continue
                else:
                    safe = len(self.buffer)
                    for i in range(1, len(self.OPEN_TAG)):
                        if self.buffer.endswith(self.OPEN_TAG[:i]):
                            safe = len(self.buffer) - i
                            break
                    if safe > 0:
                        yield ('content', self.buffer[:safe])
                        self.buffer = self.buffer[safe:]
                    break

    def flush(self):
        """Flush any remaining buffered text. Call at end of stream."""
        if self.buffer:
            kind = 'thinking' if self.in_thinking else 'content'
            text = self.buffer
            self.buffer = ""
            return (kind, text)
        return None


def parse_think_tags(text):
    """Parse <think>...</think> from complete text. Returns (reasoning_content, content).

    Handles unclosed <think> tags consistently with the streaming ThinkTagParser:
    any content after an unclosed <think> is treated as reasoning.
    """
    pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    thinking_parts = pattern.findall(text)
    content = pattern.sub('', text)

    # Handle unclosed <think> tag (matches ThinkTagParser.flush() behavior)
    unclosed = re.search(r'<think>(.*?)$', content, re.DOTALL)
    if unclosed:
        thinking_parts.append(unclosed.group(1))
        content = content[:unclosed.start()]

    content = content.strip()
    reasoning = '\n'.join(part.strip() for part in thinking_parts if part.strip())
    return (reasoning if reasoning else None, content)


# =============================================================================
# ROUTES
# =============================================================================


@app.route('/v1/models', methods=['GET'])
@app.route('/models', methods=['GET'])
def list_models():
    """OpenAI-compatible model listing endpoint."""
    logger.debug(f"/v1/models called - current_model: {CURRENT_MODEL}")
    data = []
    for model_id in MODELS:
        data.append({
            "id": model_id,
            "object": "model",
            "created": SERVER_START_TIME,
            "owned_by": "rkllm",
        })
    return jsonify({"object": "list", "data": data})


@app.route('/v1/models/select', methods=['POST'])
def select_model():
    """Pre-load a model without generating (useful for warm-up)."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return make_error_response("Request body must be a JSON object", 400, "invalid_request")
    model_name = body.get('model', '')

    name, config = resolve_model(model_name)
    if config is None:
        return make_error_response(f"Model '{model_name}' not found", 404, "not_found")

    # Reject if a generation is currently in flight
    with ACTIVE_LOCK:
        if ACTIVE_REQUEST["id"] is not None:
            elapsed = time.time() - ACTIVE_REQUEST["last_activity"]
            if elapsed <= REQUEST_STALE_TIMEOUT:
                return make_error_response(
                    "Cannot switch models while a request is in progress",
                    503, "service_unavailable")

    if not load_model(name, config):
        return make_error_response(f"Failed to load model '{name}'", 500)

    return jsonify({"status": "ok", "model": name})


@app.route('/v1/models/unload', methods=['POST'])
def unload_model():
    """Explicitly unload the current model to free NPU memory."""
    # Reject if a generation is currently in flight
    with ACTIVE_LOCK:
        if ACTIVE_REQUEST["id"] is not None:
            elapsed = time.time() - ACTIVE_REQUEST["last_activity"]
            if elapsed <= REQUEST_STALE_TIMEOUT:
                return make_error_response(
                    "Cannot unload model while a request is in progress",
                    503, "service_unavailable")

    if CURRENT_MODEL:
        model = CURRENT_MODEL
        unload_current("explicit unload request")
        return jsonify({"status": "ok", "unloaded": model})
    return jsonify({"status": "ok", "message": "no model loaded"})


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    active = get_active_request_info()
    vl_info = None
    if VL_CURRENT_MODEL:
        vl_info = {
            "model": VL_CURRENT_MODEL,
            "encoder_loaded": _vision_encoder is not None and _vision_encoder.is_loaded,
            "llm_loaded": _vl_rkllm_wrapper is not None and _vl_rkllm_wrapper.is_loaded,
        }
    return jsonify({
        "status": "ok",
        "current_model": CURRENT_MODEL,
        "model_loaded": is_model_loaded(),
        "vl_model": vl_info,
        "active_request": active,
        "models_available": len(MODELS),
    })


@app.route('/v1/chat/completions', methods=['POST'])
@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint (streaming and non-streaming)."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return make_error_response("Request body must be a JSON object", 400, "invalid_request")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    logger.info(f">>> NEW REQUEST {request_id}")

    # Parse request
    requested_model = body.get('model', '')
    messages = body.get('messages', [])
    if not isinstance(messages, list):
        return make_error_response("'messages' must be a JSON array", 400, "invalid_request")

    # Validate message elements are dicts (rejects e.g. messages: ["hello"])
    messages = [m for m in messages if isinstance(m, dict)]

    # === VL AUTO-ROUTING: Check for images BEFORE normalizing content ===
    _vl_has_images = _has_images_in_messages(messages)
    _vl_images = None
    _vl_text_prompt = None
    if _vl_has_images:
        _vl_images, _vl_text_prompt = _extract_images_from_messages(messages)
        if not _vl_images:
            _vl_has_images = False
            logger.warning(f"[{request_id}] Image parts found but no images could be decoded "
                          f"(check base64 encoding)")
            return make_error_response(
                "Image content detected but all images failed to decode. "
                "Ensure images are valid base64-encoded data URIs "
                "(e.g., data:image/png;base64,...).", 400, "invalid_request")
        else:
            logger.info(f"[{request_id}] VL AUTO-ROUTE: {len(_vl_images)} image(s) detected, "
                       f"text='{_vl_text_prompt[:80]}...' -- routing to VL model")

    # Normalize message content: OpenAI allows content to be a list of
    # multimodal parts (e.g., [{"type":"text","text":"..."}]).  Convert to
    # plain strings once so all downstream code gets clean string content.
    for msg in messages:
        content = msg.get('content')
        if content is None:
            msg['content'] = ''
        elif isinstance(content, list):
            msg['content'] = " ".join(
                part.get('text', '') for part in content
                if isinstance(part, dict) and part.get('type') == 'text'
            )
        elif not isinstance(content, str):
            msg['content'] = str(content)

    stream = bool(body.get('stream', False))
    stream_options = body.get('stream_options')
    if not isinstance(stream_options, dict):
        stream_options = {}
    include_usage = stream_options.get('include_usage', False) if stream else False

    # Per-request max_tokens — rkllm uses the model-config value set at load
    # time; log when the caller's requested value differs so they know.
    req_max_tokens = body.get('max_tokens') or body.get('max_completion_tokens')
    if req_max_tokens is not None:
        logger.debug(f"[{request_id}] max_tokens={req_max_tokens} requested "
                     f"(rkllm uses model-config value; per-request override not supported)")

    # Log ignored sampling parameters
    ignored_params = {k: body[k] for k, default in _SAMPLING_DEFAULTS.items()
                      if body.get(k) is not None and body[k] != default}
    if ignored_params:
        summary = ', '.join(f'{k}={v}' for k, v in ignored_params.items())
        logger.debug(f"[{request_id}] Ignored sampling params: {summary} (rkllm uses model-compiled sampling)")

    logger.info(f"Request {request_id} model: '{requested_model}' stream: {stream}")

    # === DIAGNOSTIC: Dump all messages from Open WebUI ===
    for i, msg in enumerate(messages):
        role = msg.get('role', '?')
        content = msg.get('content', '')
        preview = content[:500] if len(content) <= 500 else content[:500] + f"... [{len(content)} chars total]"
        logger.debug(f"[{request_id}] MSG[{i}] role={role} len={len(content)}: {preview}")

    if not messages:
        return make_error_response("No messages provided", 400, "invalid_request")

    # === SHORTCIRCUIT helper: return instant response without inference ===
    def _make_shortcircuit_response(content, label):
        """Return a chat completion response instantly, no model needed."""
        logger.info(f"[{request_id}] SHORTCIRCUIT {label}")
        if stream:
            def _sc_gen():
                c = {
                    "id": request_id, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": requested_model,
                    "system_fingerprint": SYSTEM_FINGERPRINT,
                    "choices": [{"index": 0, "delta": {"role": "assistant",
                                 "content": content}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(c)}\n\n"
                d = {
                    "id": request_id, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": requested_model,
                    "system_fingerprint": SYSTEM_FINGERPRINT,
                    "choices": [{"index": 0, "delta": {},
                                 "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(d)}\n\n"
                yield "data: [DONE]\n\n"
            return Response(
                stream_with_context(_sc_gen()),
                mimetype='text/event-stream',
                headers={'Cache-Control': 'no-cache',
                         'X-Accel-Buffering': 'no',
                         'Connection': 'keep-alive'},
            )
        return jsonify({
            "id": request_id, "object": "chat.completion",
            "created": int(time.time()), "model": requested_model,
            "system_fingerprint": SYSTEM_FINGERPRINT,
            "choices": [{"index": 0, "message": {"role": "assistant",
                         "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 1,
                      "total_tokens": 1}
        })

    # ---- Detect Open WebUI meta-task type from last user message ----
    _last_user_content = ''
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            _last_user_content = msg.get('content', '')
            break
    _luc_lower = _last_user_content.lower()

    # === SHORTCIRCUIT: Retrieval / web-search query generation ===
    # Open WebUI asks the model to generate search queries for document
    # retrieval or web search.  On small models this wastes ~5s and the
    # JSON output often leaks into the chat display.
    #
    # Strategy: extract the LAST user message from the embedded
    # <chat_history>.  If it's a short/vague follow-up (e.g.
    # "can you verify that course exists"), enrich it with key
    # entities/topics extracted from the previous ASSISTANT response
    # so the search engine gets a contextual query.
    _is_query_gen = (
        'generate search queries' in _luc_lower
        or 'generating search queries' in _luc_lower
        or ('search queries' in _luc_lower and '"queries"' in _luc_lower)
        or ('generate 1-3 broad' in _luc_lower and 'search' in _luc_lower)
    )
    if _is_query_gen:

        # --- 1. Parse every USER/ASSISTANT turn from the chat_history ---
        _ch_match = re.search(
            r'<chat_history>(.*?)</chat_history>',
            _last_user_content, re.DOTALL
        )
        _chat_block = _ch_match.group(1) if _ch_match else _last_user_content

        _all_user = re.findall(
            r'(?:USER|user):\s*(.+?)(?=\s*(?:ASSISTANT|assistant):|</chat_history>|\s*$)',
            _chat_block, re.DOTALL
        )
        _all_asst = re.findall(
            r'(?:ASSISTANT|assistant):\s*(.+?)(?=\s*(?:USER|user):|</chat_history>|\s*$)',
            _chat_block, re.DOTALL
        )

        _real_q = _all_user[-1].strip() if _all_user else 'general query'
        _prev_asst = _all_asst[-1].strip() if _all_asst else ''

        # --- 2. Context-enrich short/vague follow-up questions ---
        # A question is considered "vague" when it's short and uses
        # pronouns / demonstratives instead of naming things directly.
        _VAGUE_WORDS = {'it', 'its', 'this', 'that', 'these', 'those',
                        'they', 'them', 'one', 'the course', 'the document',
                        'the file', 'the article', 'the data'}
        _q_lower = _real_q.lower()
        _is_vague = (
            len(_real_q.split()) < 12
            and any(v in _q_lower for v in _VAGUE_WORDS)
        )

        if _is_vague and _prev_asst:
            # Extract key entities from the assistant's last response.
            # Use capitalized words/phrases and quoted terms as signals
            # of important nouns (course names, product names, etc.).
            _entities = []

            # Quoted strings (single or double)
            _quoted = re.findall(r'["\u201c]([^"\u201d]{3,60})["\u201d]', _prev_asst)
            _entities.extend(q.strip() for q in _quoted[:3])

            # **Bold** text
            _bold = re.findall(r'\*\*(.{3,60}?)\*\*', _prev_asst)
            _entities.extend(b.strip() for b in _bold[:3])

            # Capitalized multi-word phrases (e.g. "Python for Data Science")
            _caps = re.findall(
                r'\b([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|for|and|the|of|in|with|&)){1,6})\b',
                _prev_asst
            )
            _entities.extend(c.strip() for c in _caps[:5])

            # Deduplicate while preserving order; skip entries that
            # are substrings of an already-accepted entity.
            _seen = set()
            _unique_ents = []
            for e in _entities:
                _e_low = e.lower().strip()
                if len(_e_low) < 3:
                    continue
                # Skip if this entity is a substring of one already kept
                if any(_e_low in prev for prev in _seen):
                    continue
                # Remove previously-kept entries that are substrings
                # of this (longer) entity
                _seen = {p for p in _seen if p not in _e_low}
                _unique_ents = [u for u in _unique_ents
                                if u.lower() not in _e_low]
                _seen.add(_e_low)
                _unique_ents.append(e)

            if _unique_ents:
                # Build an enriched query: user question + top entities
                _context_str = ' '.join(_unique_ents[:4])
                _enriched_q = f"{_real_q} {_context_str}"
                # Cap length for search engine friendliness
                if len(_enriched_q) > 200:
                    _enriched_q = _enriched_q[:200].rsplit(' ', 1)[0]
                logger.info(f"SHORTCIRCUIT query gen — enriched vague query: "
                            f"'{_real_q}' → '{_enriched_q}'")
                return _make_shortcircuit_response(
                    json.dumps({"queries": [_enriched_q]}),
                    f"query gen — enriched: '{_enriched_q[:80]}'"
                )

        # Not vague or no assistant context — use the raw question
        return _make_shortcircuit_response(
            json.dumps({"queries": [_real_q]}),
            f"query gen — search query: '{_real_q[:80]}'"
        )

    # === SHORTCIRCUIT: Title generation ===
    # Open WebUI asks for a concise 3-5 word chat title.
    _is_title_gen = (
        ('title' in _luc_lower and 'chat' in _luc_lower
         and ('3-5 word' in _luc_lower or 'concise' in _luc_lower))
        or 'emoji as a title' in _luc_lower
        or 'title for the prompt' in _luc_lower
    )
    if _is_title_gen:
        _first_user_msg = ''
        for msg in messages:
            if msg.get('role') == 'user' and not any(
                sig in msg.get('content', '').lower()
                for sig in ('title for the chat', 'emoji as a title',
                            'title for the prompt', '3-5 word')
            ):
                _first_user_msg = msg.get('content', '')[:60].strip()
                break
        if not _first_user_msg:
            _m2 = re.search(
                r'(?:USER|user):\s*(.+?)(?:\s*(?:ASSISTANT|assistant)'
                r'|\s*</chat_history>|\s*$)',
                _last_user_content, re.DOTALL
            )
            _first_user_msg = (_m2.group(1).strip()[:60]
                               if _m2 else 'New Chat')
        _title = _first_user_msg.split('\n')[0].strip()
        if len(_title) > 50:
            _title = _title[:47] + '...'
        return _make_shortcircuit_response(_title,
                                           f"title gen — '{_title}'")

    # === SHORTCIRCUIT: Tag generation ===
    # Open WebUI asks the model to generate 1-3 tags.
    _is_tag_gen = (
        ('generate 1-3 tags' in _luc_lower
         or 'categorize the chat' in _luc_lower)
        and 'tag' in _luc_lower
    )
    if _is_tag_gen:
        return _make_shortcircuit_response('general',
                                           "tag gen — 'general'")

    # Resolve model
    name, config = resolve_model(requested_model)
    logger.info(f"Resolved '{requested_model}' -> '{name}'. Current: {CURRENT_MODEL}")

    if config is None:
        return make_error_response(f"Model '{requested_model}' not found", 404, "not_found")

    # Atomic check-and-set: reject if another generation is in progress
    if not try_start_request(request_id, name):
        logger.warning(f"[{request_id}] Rejected - request already in progress")
        return make_error_response(
            "Another request is currently being processed. Please retry shortly.",
            503, "service_unavailable"
        )

    ABORT_EVENT.clear()
    created = int(time.time())

    try:
        # =================================================================
        # VL PATH -- image detected, route to vision-language model
        # =================================================================
        if _vl_has_images and _vl_images:
            vl_name, vl_config = _find_vl_model()
            if vl_name is None:
                end_request(request_id)
                return make_error_response(
                    "Image detected but no VL model available. "
                    "Place a VL model (.rkllm + .rknn) in ~/models/", 500)

            logger.info(f"[{request_id}] VL path: loading {vl_name}")
            if not _load_vl_model(vl_name, vl_config):
                end_request(request_id)
                return make_error_response(f"Failed to load VL model '{vl_name}'", 500)
            update_request_activity()
            global VL_LAST_REQUEST_TIME
            VL_LAST_REQUEST_TIME = time.time()

            try:
                image_np = _preprocess_image(
                    _vl_images[0],
                    _vision_encoder.model_width,
                    _vision_encoder.model_height,
                )
            except Exception as e:
                logger.error(f"[{request_id}] Image preprocessing failed: {e}")
                end_request(request_id)
                return make_error_response(f"Image preprocessing failed: {e}", 400)
            update_request_activity()

            logger.info(f"[{request_id}] Running vision encoder...")
            with PROCESS_LOCK:
                image_embed = _vision_encoder.run(image_np)
            if image_embed is None:
                end_request(request_id)
                return make_error_response("Vision encoder inference failed", 500)
            update_request_activity()
            logger.info(f"[{request_id}] Vision encoder done: "
                       f"{len(image_embed)} floats "
                       f"({_vision_encoder.model_image_token} tokens x "
                       f"{_vision_encoder.model_embed_size} embed)")

            vl_image_tag = vl_config.get('vl_config', {}).get('image_tag', '<image>')
            vl_prompt = f"{vl_image_tag}{_vl_text_prompt}"

            vl_data = {
                'image_embed': image_embed,
                'n_image_tokens': _vision_encoder.model_image_token,
                'n_image': 1,
                'image_width': _vision_encoder.model_width,
                'image_height': _vision_encoder.model_height,
            }

            if stream:
                return Response(
                    stream_with_context(_generate_stream(
                        vl_prompt, request_id, vl_name, created,
                        keep_history=0, enable_thinking=False,
                        include_usage=include_usage, messages=messages,
                        is_rag=False, rag_cache_info=None,
                        kv_is_reset=True, vl_data=vl_data,
                    )),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no',
                             'Connection': 'keep-alive'},
                )
            else:
                return _generate_complete(
                    vl_prompt, request_id, vl_name, created,
                    keep_history=0, enable_thinking=False,
                    is_rag=False, messages=messages,
                    rag_cache_info=None, kv_is_reset=True, vl_data=vl_data,
                )

        # =================================================================
        # TEXT PATH -- normal text-only request (existing logic unchanged)
        # =================================================================
        # Load model
        logger.info(f"Loading model {name} for request {request_id}")
        if not load_model(name, config):
            end_request(request_id)
            return make_error_response(f"Failed to load model '{name}'", 500)
        update_request_activity()

        # Build prompt
        prompt, is_rag, enable_thinking = build_prompt(messages, name)
        if not prompt:
            end_request(request_id)
            return make_error_response("Failed to build prompt from messages", 400, "invalid_request")

        # KV cache strategy:
        # - RAG: always keep_history=0 (fresh context each time)
        # - Normal incremental: send only last user message, keep_history=1
        #   (runtime already has prior turns in KV cache)
        # - Normal reset: clear_kv_cache() + keep_history=1
        #   IMPORTANT: keep_history=0 discards the KV cache after the run,
        #   so Turn 2 would find an empty cache.  We use keep_history=1
        #   with an explicit clear_kv_cache() call instead.
        kv_is_reset = False
        if is_rag:
            keep_history = 0
            kv_is_reset = True
            # Explicitly clear KV cache before RAG inference.
            # keep_history=0 discards the cache AFTER the run, but stale
            # context from a prior non-RAG conversation would still be
            # present DURING the run, contaminating the RAG prompt.
            if _rkllm_wrapper and _rkllm_wrapper.is_loaded:
                _rkllm_wrapper.clear_kv_cache()
            _reset_kv_tracking()
        else:
            incremental_msg = _check_kv_incremental(name, messages)
            if incremental_msg is not None:
                logger.info(f"[{request_id}] KV INCREMENTAL — sending only latest "
                            f"user message ({len(incremental_msg)} chars "
                            f"vs {len(prompt)} full)")
                prompt = incremental_msg
                keep_history = 1
            else:
                logger.info(f"[{request_id}] KV RESET — clearing KV cache for "
                            f"new conversation")
                # Clear stale KV cache from prior conversation, then use
                # keep_history=1 so THIS turn's state IS retained for
                # incremental follow-ups.  With keep_history=0 the runtime
                # discards the cache after generation, breaking Turn 2.
                if _rkllm_wrapper and _rkllm_wrapper.is_loaded:
                    _rkllm_wrapper.clear_kv_cache()
                keep_history = 1
                kv_is_reset = True
                _reset_kv_tracking()

        # Extract user question for cache key
        user_question = ""
        for msg in reversed(messages):
            if msg.get('role') == 'user' and msg.get('content'):
                user_question = msg['content']
                break

        # === RAG Response Cache Check ===
        if is_rag and RAG_CACHE_TTL > 0:
            cached = _rag_cache_get(name, user_question)
            if cached:
                cached_prompt, cached_response = cached
                # Invalidate if web context changed (different search results)
                if cached_prompt != prompt:
                    logger.info(f"[{request_id}] RAG cache STALE \u2014 web context changed, regenerating")
                    cached = None
            if cached:
                logger.info(f"[{request_id}] RAG cache HIT for '{user_question[:60]}' "
                            f"({len(cached_response)} chars)")
                end_request(request_id)
                created = int(time.time())
                prompt_tokens = max(1, len(cached_prompt) // CHARS_PER_TOKEN_ESTIMATE)
                completion_tokens = max(1, len(cached_response) // CHARS_PER_TOKEN_ESTIMATE)
                reasoning_content, cleaned_content = parse_think_tags(cached_response)
                if stream:
                    def _cached_stream():
                        yield make_sse_chunk(request_id, name, created, delta={"role": "assistant"})
                        if reasoning_content:
                            yield make_sse_chunk(request_id, name, created, delta={"reasoning_content": reasoning_content})
                        # Progressive chunking for natural streaming feel
                        _chunk_size = 80
                        for _ci in range(0, len(cleaned_content), _chunk_size):
                            yield make_sse_chunk(request_id, name, created,
                                                 delta={"content": cleaned_content[_ci:_ci + _chunk_size]})
                        yield make_sse_chunk(request_id, name, created, finish_reason="stop")
                        if include_usage:
                            yield make_sse_chunk(request_id, name, created, usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens})
                        yield "data: [DONE]\n\n"
                    return Response(
                        stream_with_context(_cached_stream()),
                        mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no',
                                 'Connection': 'keep-alive'},
                    )
                else:
                    message_obj = {"role": "assistant", "content": cleaned_content}
                    if reasoning_content:
                        message_obj["reasoning_content"] = reasoning_content
                    return jsonify({
                        "id": request_id, "object": "chat.completion",
                        "created": created, "model": name,
                        "system_fingerprint": SYSTEM_FINGERPRINT,
                        "choices": [{"index": 0, "message": message_obj,
                                     "logprobs": None, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": prompt_tokens,
                                  "completion_tokens": completion_tokens,
                                  "total_tokens": prompt_tokens + completion_tokens},
                    })

        # Prepare cache info for RAG queries
        rag_cache_info = (name, user_question, prompt) if is_rag and RAG_CACHE_TTL > 0 else None

        if stream:
            return Response(
                stream_with_context(_generate_stream(
                    prompt, request_id, name, created,
                    keep_history=keep_history,
                    enable_thinking=enable_thinking,
                    include_usage=include_usage,
                    messages=messages,
                    is_rag=is_rag,
                    rag_cache_info=rag_cache_info,
                    kv_is_reset=kv_is_reset,
                )),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                    'Connection': 'keep-alive',
                },
            )
        else:
            return _generate_complete(
                prompt, request_id, name, created,
                keep_history=keep_history,
                enable_thinking=enable_thinking,
                is_rag=is_rag,
                messages=messages,
                rag_cache_info=rag_cache_info,
                kv_is_reset=kv_is_reset,
            )

    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        end_request(request_id)
        return make_error_response(f"Internal error: {e}", 500)


# =============================================================================
# GENERATION — STREAMING
# =============================================================================


def _generate_stream(prompt, request_id, model_name, created,
                     keep_history=1, enable_thinking=True,
                     include_usage=False, messages=None,
                     is_rag=False, rag_cache_info=None,
                     kv_is_reset=False, vl_data=None):
    """Generator that yields SSE chunks from rkllm token callback queue."""
    global _worker_thread

    _is_vl = vl_data is not None
    _active_wrapper = _vl_rkllm_wrapper if _is_vl else _rkllm_wrapper

    logger.info(f"[{request_id}] Starting STREAMING generation "
                f"(rag={is_rag}, keep_history={keep_history}, thinking={enable_thinking}"
                f"{', VL=True' if _is_vl else ''})")

    # Clear any stale items from the token queue
    while not _token_queue.empty():
        try:
            _token_queue.get_nowait()
        except queue.Empty:
            break

    # GENERATION_COMPLETE and the first yield are inside try so that an
    # early client disconnect (GeneratorExit) is caught by the finally
    # block, which resets GENERATION_COMPLETE and calls end_request().

    # Start rkllm_run in a worker thread (it blocks until generation completes)
    def _worker():
        try:
            if _is_vl:
                ret = _active_wrapper.run_multimodal(
                    prompt, vl_data['image_embed'],
                    vl_data['n_image_tokens'], vl_data['n_image'],
                    vl_data['image_width'], vl_data['image_height'],
                    role="user", keep_history=keep_history,
                )
            else:
                ret = _active_wrapper.run(prompt, role="user",
                                          keep_history=keep_history,
                                          enable_thinking=enable_thinking)
            if ret != 0:
                logger.error(f"[{request_id}] rkllm_run returned error code {ret}")
                _token_queue.put(("error", f"rkllm_run returned {ret}"))
        except Exception as e:
            logger.error(f"[{request_id}] Worker thread error: {e}")
            _token_queue.put(("error", str(e)))

    with PROCESS_LOCK:
        _worker_thread = Thread(target=_worker, daemon=True)
        _worker_thread.start()

    generation_start = time.time()
    got_first_token = False
    total_content = ""
    total_reasoning = ""
    generation_clean = False
    request_ended = False
    stats_data = {}
    think_parser = ThinkTagParser()

    try:
        GENERATION_COMPLETE.clear()

        # First chunk: role
        yield make_sse_chunk(request_id, model_name, created, delta={"role": "assistant"})

        while True:
            # Check abort
            if ABORT_EVENT.is_set():
                logger.info(f"[{request_id}] Abort signal received")
                _active_wrapper.abort()
                break

            # Check overall timeout
            if time.time() - generation_start > GENERATION_TIMEOUT:
                logger.warning(f"[{request_id}] Generation timeout ({GENERATION_TIMEOUT}s)")
                _active_wrapper.abort()
                break

            # Determine queue read timeout
            if not got_first_token:
                remaining = GENERATION_TIMEOUT - (time.time() - generation_start)
                get_timeout = min(FIRST_TOKEN_TIMEOUT, max(0.1, remaining))
            else:
                get_timeout = FALLBACK_SILENCE

            try:
                msg_type, msg_data = _token_queue.get(timeout=get_timeout)
            except queue.Empty:
                label = "first token" if not got_first_token else "silence"
                logger.warning(f"[{request_id}] Timeout waiting for {label}")
                _active_wrapper.abort()
                break

            if msg_type == "token":
                if not got_first_token:
                    got_first_token = True
                    logger.debug(f"[{request_id}] First token in {time.time() - generation_start:.2f}s")

                update_request_activity()

                # Route through think-tag parser
                for kind, chunk_text in think_parser.feed(msg_data):
                    if kind == 'thinking':
                        total_reasoning += chunk_text
                        if chunk_text.strip():
                            yield make_sse_chunk(request_id, model_name, created,
                                                 delta={"reasoning_content": chunk_text})
                    else:
                        total_content += chunk_text
                        yield make_sse_chunk(request_id, model_name, created,
                                             delta={"content": chunk_text})

            elif msg_type == "finish":
                stats_data = msg_data or {}
                generation_clean = True
                if stats_data:
                    logger.info(f"[{request_id}] Perf: "
                                f"prefill={stats_data.get('prefill_time_ms', 0):.0f}ms "
                                f"({stats_data.get('prefill_tokens', 0)} tok), "
                                f"generate={stats_data.get('generate_time_ms', 0):.0f}ms "
                                f"({stats_data.get('generate_tokens', 0)} tok), "
                                f"mem={stats_data.get('memory_usage_mb', 0):.0f}MB")
                break

            elif msg_type == "error":
                logger.error(f"[{request_id}] rkllm_run error: {msg_data}")
                break

        # End request tracking immediately
        end_request(request_id)
        request_ended = True

        # Wait for worker thread to finish
        with PROCESS_LOCK:
            if _worker_thread and _worker_thread.is_alive():
                _worker_thread.join(timeout=5)
            _worker_thread = None

        # Flush think-tag parser
        flushed = think_parser.flush()
        if flushed:
            kind, flush_text = flushed
            if flush_text:
                if kind == 'thinking':
                    total_reasoning += flush_text
                    if flush_text.strip():
                        yield make_sse_chunk(request_id, model_name, created,
                                             delta={"reasoning_content": flush_text})
                else:
                    total_content += flush_text
                    yield make_sse_chunk(request_id, model_name, created,
                                         delta={"content": flush_text})

        # Final stop chunk
        yield make_sse_chunk(request_id, model_name, created, finish_reason="stop")

        # Usage chunk (stream_options.include_usage)
        if include_usage:
            prompt_text = "".join(m.get("content", "") for m in (messages or [])
                                  if isinstance(m.get("content"), str))
            prompt_tokens = max(1, len(prompt_text) // CHARS_PER_TOKEN_ESTIMATE)
            total_output_chars = len(total_content) + len(total_reasoning)
            completion_tokens = stats_data.get(
                'generate_tokens', max(1, total_output_chars // CHARS_PER_TOKEN_ESTIMATE))
            yield make_sse_chunk(request_id, model_name, created, usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            })

        yield "data: [DONE]\n\n"

        # Post-stream bookkeeping — each operation is independent so a
        # failure in one doesn't skip the other.
        try:
            if rag_cache_info and generation_clean and total_content:
                cache_response = total_content
                if total_reasoning:
                    cache_response = f"<think>{total_reasoning}</think>{total_content}"
                _model, _question, _prompt = rag_cache_info
                _rag_cache_put(_model, _question, _prompt, cache_response)
                logger.info(f"[{request_id}] RAG cache STORE ({len(cache_response)} chars)")
        except Exception as _cache_exc:
            logger.error(f"[{request_id}] RAG cache store error: {_cache_exc}")

        try:
            if generation_clean and not is_rag and messages:
                _update_kv_tracking(model_name, messages, is_reset=kv_is_reset)
        except Exception as _kv_exc:
            logger.error(f"[{request_id}] KV tracking update error: {_kv_exc}")

    except GeneratorExit:
        logger.warning(f"[{request_id}] Client DISCONNECTED / stopped")
        if _active_wrapper:
            _active_wrapper.abort()
    except Exception as e:
        logger.error(f"[{request_id}] Stream error: {e}", exc_info=True)
        try:
            yield make_sse_chunk(request_id, model_name, created, finish_reason="stop")
            yield "data: [DONE]\n\n"
        except Exception:
            pass
    finally:
        if not request_ended:
            end_request(request_id)
        # Ensure worker thread is cleaned up
        with PROCESS_LOCK:
            if _worker_thread and _worker_thread.is_alive():
                if _active_wrapper:
                    _active_wrapper.abort()
                _worker_thread.join(timeout=5)
                _worker_thread = None
        GENERATION_COMPLETE.set()
        if total_reasoning:
            logger.info(f"[{request_id}] Stream ended ({len(total_content)} content + "
                        f"{len(total_reasoning)} reasoning chars)")
        else:
            logger.info(f"[{request_id}] Stream ended ({len(total_content)} chars total)")


# =============================================================================
# GENERATION — NON-STREAMING
# =============================================================================


def _generate_complete(prompt, request_id, model_name, created,
                       keep_history=1, enable_thinking=True,
                       is_rag=False, messages=None, rag_cache_info=None,
                       kv_is_reset=False, vl_data=None):
    """Collect all output and return a non-streaming JSON response."""
    global _worker_thread

    _is_vl = vl_data is not None
    _active_wrapper = _vl_rkllm_wrapper if _is_vl else _rkllm_wrapper

    logger.info(f"[{request_id}] Starting NON-STREAMING generation "
                f"(rag={is_rag}, keep_history={keep_history}, thinking={enable_thinking}"
                f"{', VL=True' if _is_vl else ''})")

    # Clear stale queue items
    while not _token_queue.empty():
        try:
            _token_queue.get_nowait()
        except queue.Empty:
            break

    # Start worker thread
    def _worker():
        try:
            if _is_vl:
                ret = _active_wrapper.run_multimodal(
                    prompt, vl_data['image_embed'],
                    vl_data['n_image_tokens'], vl_data['n_image'],
                    vl_data['image_width'], vl_data['image_height'],
                    role="user", keep_history=keep_history,
                )
            else:
                ret = _active_wrapper.run(prompt, role="user",
                                          keep_history=keep_history,
                                          enable_thinking=enable_thinking)
            if ret != 0:
                logger.error(f"[{request_id}] rkllm_run returned error code {ret}")
                _token_queue.put(("error", f"rkllm_run returned {ret}"))
        except Exception as e:
            logger.error(f"[{request_id}] Worker thread error: {e}")
            _token_queue.put(("error", str(e)))

    with PROCESS_LOCK:
        _worker_thread = Thread(target=_worker, daemon=True)
        _worker_thread.start()

    generation_start = time.time()
    got_first_token = False
    content_parts = []
    generation_clean = False
    stats_data = {}

    try:
        GENERATION_COMPLETE.clear()
        while True:
            if ABORT_EVENT.is_set():
                _active_wrapper.abort()
                break

            if time.time() - generation_start > GENERATION_TIMEOUT:
                logger.warning(f"[{request_id}] Generation timeout")
                _active_wrapper.abort()
                break

            if not got_first_token:
                remaining = GENERATION_TIMEOUT - (time.time() - generation_start)
                get_timeout = min(FIRST_TOKEN_TIMEOUT, max(0.1, remaining))
            else:
                get_timeout = FALLBACK_SILENCE

            try:
                msg_type, msg_data = _token_queue.get(timeout=get_timeout)
            except queue.Empty:
                logger.warning(f"[{request_id}] Silence timeout")
                _active_wrapper.abort()
                break

            if msg_type == "token":
                if not got_first_token:
                    got_first_token = True
                update_request_activity()
                content_parts.append(msg_data)

            elif msg_type == "finish":
                stats_data = msg_data or {}
                generation_clean = True
                if stats_data:
                    logger.info(f"[{request_id}] Perf: "
                                f"prefill={stats_data.get('prefill_time_ms', 0):.0f}ms "
                                f"({stats_data.get('prefill_tokens', 0)} tok), "
                                f"generate={stats_data.get('generate_time_ms', 0):.0f}ms "
                                f"({stats_data.get('generate_tokens', 0)} tok), "
                                f"mem={stats_data.get('memory_usage_mb', 0):.0f}MB")
                break

            elif msg_type == "error":
                logger.error(f"[{request_id}] rkllm_run error: {msg_data}")
                break

        # Wait for worker thread
        with PROCESS_LOCK:
            if _worker_thread and _worker_thread.is_alive():
                _worker_thread.join(timeout=5)
            _worker_thread = None

        full_content = "".join(content_parts).rstrip()

        # Parse think tags
        reasoning_content, cleaned_content = parse_think_tags(full_content)

        # Token counts: use real NPU stats if available, else approximate
        prompt_text = "".join(m.get("content", "") for m in (messages or [])
                              if isinstance(m.get("content"), str))
        prompt_tokens = max(1, len(prompt_text) // CHARS_PER_TOKEN_ESTIMATE)
        completion_tokens = stats_data.get('generate_tokens', max(1, len(full_content) // CHARS_PER_TOKEN_ESTIMATE))

        # Determine finish_reason: "stop" for clean generation, "length" for
        # timeout/silence, "stop" with empty content for abort/error.
        finish_reason = "stop" if generation_clean else "length"

        message_obj = {
            "role": "assistant",
            "content": cleaned_content,
        }
        if reasoning_content:
            message_obj["reasoning_content"] = reasoning_content

        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "system_fingerprint": SYSTEM_FINGERPRINT,
            "choices": [{
                "index": 0,
                "message": message_obj,
                "logprobs": None,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        # Post-generation bookkeeping — each operation is independent so a
        # failure in one doesn't skip the other.
        try:
            if rag_cache_info and generation_clean and cleaned_content:
                cache_text = cleaned_content
                if reasoning_content:
                    cache_text = f"<think>{reasoning_content}</think>{cleaned_content}"
                _model, _question, _prompt = rag_cache_info
                _rag_cache_put(_model, _question, _prompt, cache_text)
                logger.info(f"[{request_id}] RAG cache STORE ({len(cache_text)} chars)")
        except Exception as _cache_exc:
            logger.error(f"[{request_id}] RAG cache store error: {_cache_exc}")

        try:
            if generation_clean and not is_rag and messages:
                _update_kv_tracking(model_name, messages, is_reset=kv_is_reset)
        except Exception as _kv_exc:
            logger.error(f"[{request_id}] KV tracking update error: {_kv_exc}")

        if reasoning_content:
            logger.info(f"[{request_id}] Completed ({len(cleaned_content)} content + "
                        f"{len(reasoning_content)} reasoning chars)")
        else:
            logger.info(f"[{request_id}] Completed ({len(cleaned_content)} chars)")
        return jsonify(response)

    except Exception as e:
        logger.error(f"[{request_id}] Generation error: {e}", exc_info=True)
        return make_error_response(f"Generation failed: {e}", 500)
    finally:
        end_request(request_id)
        # Ensure worker cleanup
        with PROCESS_LOCK:
            if _worker_thread and _worker_thread.is_alive():
                if _active_wrapper:
                    _active_wrapper.abort()
                _worker_thread.join(timeout=5)
                _worker_thread = None
        GENERATION_COMPLETE.set()


# =============================================================================
# STARTUP
# =============================================================================

if __name__ == "__main__":
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, signal_handler)
        except (OSError, ValueError):
            pass

    _vl_models = [n for n, c in MODELS.items() if 'vl_config' in c]
    _text_models = [n for n, c in MODELS.items() if 'vl_config' not in c]

    logger.info("=" * 60)
    logger.info("RKLLM API starting (ctypes direct API)")
    logger.info(f"  RKLLM Library : {RKLLM_LIB_PATH}")
    logger.info(f"  RKNN Library  : {RKNN_LIB_PATH}")
    logger.info(f"  Text models   : {len(_text_models)} detected")
    logger.info(f"  VL models     : {len(_vl_models)} detected")
    logger.info(f"  VL deps       : {'available' if _VL_DEPS_AVAILABLE else 'MISSING (install numpy Pillow)'}")
    logger.info(f"  Aliases       : {len(ALIASES)} generated")
    logger.info(f"  Port          : 8000")
    logger.info("=" * 60)
    logger.info(f"Text models: {_text_models}")
    if _vl_models:
        logger.info(f"VL models: {_vl_models} (auto-routing enabled)")
    logger.info(f"Aliases: {ALIASES}")
    app.run(host="0.0.0.0", port=8000, threaded=True)
