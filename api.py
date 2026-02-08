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
from collections import OrderedDict
from threading import Lock, RLock, Thread, Event
import atexit
import sys
import signal

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
    for root, _, files in os.walk(MODELS_ROOT):
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

        config = {
            "path": rkllm_file,
            "context_length": context_len,
            "max_tokens": MAX_TOKENS_DEFAULT,
        }

        MODELS[model_id] = config
        logger.info(f"Detected: {model_id} (context={context_len})")

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

logger.info(f"Models: {list(MODELS.keys())}")
logger.info(f"Aliases: {ALIASES}")

# =============================================================================
# RKLLM CTYPES DEFINITIONS
# =============================================================================
# All struct definitions match the official airockchip/rknn-llm Python demos
# (flask_server.py / gradio_server.py) and the rkllm.h C header.

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

    def init_model(self, model_path, ctx_len, max_tokens):
        """Initialize a model.  Returns True on success."""
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


def _clean_web_content(text):
    """Strip web page navigation/boilerplate from scraped content.

    Open WebUI's SafeWebBaseLoader returns soup.get_text() — raw text with
    ALL navigation menus, cookie banners, sidebar links, footer cruft, and
    JS-framework text.

    Strategy — four-pass line-level filtering:
    Pass 1: Remove known boilerplate phrases
    Pass 2: Remove concatenated navigation (CamelCase, title-case, URL-heavy)
    Pass 3: Collapse consecutive short-line navigation menus
    Pass 4: Keep only lines with data signals
    """
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

        # --- Layer 1: Known conversational words / phrases ---
        _SHORT_REPLY_WORDS = {
            'yes', 'no', 'yeah', 'nah', 'yep', 'nope', 'ok', 'okay', 'sure',
            'thanks', 'thank you', 'please', 'go ahead', 'continue', 'more',
            'why', 'how', 'what', 'really', 'cool', 'nice', 'wow', 'great',
            'interesting', 'tell me more', 'go on', 'and', 'also',
        }
        if _query_normalized in _SHORT_REPLY_WORDS:
            _skip_reason = f"short conversational reply '{user_question}'"

        # --- Layer 2: Short query (<=3 words) with conversation history ---
        if not _skip_reason and _has_assistant_turn:
            _query_words = _query_normalized.split()
            if len(_query_words) <= 3:
                _skip_reason = (f"short follow-up ({len(_query_words)} words) "
                                f"with conversation history")

        # --- Layer 3: Query-to-reference topical overlap ---
        if not _skip_reason and _has_assistant_turn:
            _stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                     'been', 'of', 'in', 'to', 'for', 'and', 'or', 'but',
                     'not', 'on', 'at', 'by', 'it', 'its', 'this', 'that',
                     'with', 'from', 'as', 'do', 'does', 'did', 'has', 'have',
                     'had', 'will', 'would', 'can', 'could', 'should', 'may',
                     'me', 'my', 'i', 'you', 'your', 'we', 'our', 'they',
                     'them', 'their', 'he', 'she', 'him', 'her', 'about',
                     'what', 'which', 'who', 'how', 'when', 'where', 'why',
                     'tell', 'give', 'some', 'any', 'all', 'more', 'much',
                     'many', 'very', 'just', 'also', 'so', 'if', 'than',
                     'then', 'like', 'know', 'get', 'make', 'go', 'see',
                     'no', 'yes', 'up', 'out', 'new', 'one', 'two'}
            _qcw = {w.lower().strip('?!.,;:\'"()') for w in user_question.split()
                    if w.lower().strip('?!.,;:\'"()') not in _stop and len(w.strip('?!.,;:\'"()')) > 2}
            if _qcw and len(_qcw) <= 8:
                _, ref_text, _ = rag_parts
                ref_lower = ref_text.lower()
                hits = sum(1 for w in _qcw if w in ref_lower)
                overlap = hits / len(_qcw)
                if overlap < 0.30:
                    _skip_reason = (f"low query-reference overlap ({hits}/{len(_qcw)} "
                                    f"= {overlap:.0%}, words={_qcw})")

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
        prompt = (
            f"{reference_data}\n\n"
            f"According to the above, {user_question}. "
            f"Answer in detail with specific facts and examples"
            f"{abstention}"
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
        enable_thinking = True  # Always allow thinking in normal mode

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
# MODEL MONITOR
# =============================================================================


def model_monitor():
    """Background thread to monitor model health and auto-unload on idle."""
    logger.info("Model monitor started")
    while not SHUTDOWN_EVENT.is_set():
        try:
            force_clear_if_orphaned()

            # Auto-unload after idle timeout
            if (IDLE_UNLOAD_TIMEOUT > 0 and is_model_loaded()
                    and not is_request_active() and LAST_REQUEST_TIME > 0
                    and (time.time() - LAST_REQUEST_TIME) > IDLE_UNLOAD_TIMEOUT):
                logger.info(f"Auto-unloading {CURRENT_MODEL} after "
                            f"{int(time.time() - LAST_REQUEST_TIME)}s idle")
                unload_current("idle timeout")
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
    """Clean shutdown - destroy model, stop monitor."""
    logger.info("Shutting down RKLLM API...")
    SHUTDOWN_EVENT.set()
    ABORT_EVENT.set()
    try:
        unload_current("shutdown")
    except Exception as e:
        logger.error(f"Error during shutdown unload: {e}")
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
    return jsonify({
        "status": "ok",
        "current_model": CURRENT_MODEL,
        "model_loaded": is_model_loaded(),
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

    # Normalize message content: OpenAI allows content to be a list of
    # multimodal parts (e.g., [{"type":"text","text":"..."}]).  Convert to
    # plain strings once so all downstream code gets clean string content.
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, list):
            msg['content'] = " ".join(
                part.get('text', '') for part in content
                if isinstance(part, dict) and part.get('type') == 'text'
            )
        elif content is not None and not isinstance(content, str):
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
                        yield make_sse_chunk(request_id, name, created, delta={"content": cleaned_content})
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
                     kv_is_reset=False):
    """Generator that yields SSE chunks from rkllm token callback queue."""
    global _worker_thread

    logger.info(f"[{request_id}] Starting STREAMING generation "
                f"(rag={is_rag}, keep_history={keep_history}, thinking={enable_thinking})")

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
            ret = _rkllm_wrapper.run(prompt, role="user",
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
                _rkllm_wrapper.abort()
                break

            # Check overall timeout
            if time.time() - generation_start > GENERATION_TIMEOUT:
                logger.warning(f"[{request_id}] Generation timeout ({GENERATION_TIMEOUT}s)")
                _rkllm_wrapper.abort()
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
                _rkllm_wrapper.abort()
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
        if _rkllm_wrapper:
            _rkllm_wrapper.abort()
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
                if _rkllm_wrapper:
                    _rkllm_wrapper.abort()
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
                       kv_is_reset=False):
    """Collect all output and return a non-streaming JSON response."""
    global _worker_thread

    logger.info(f"[{request_id}] Starting NON-STREAMING generation "
                f"(rag={is_rag}, keep_history={keep_history}, thinking={enable_thinking})")
    GENERATION_COMPLETE.clear()

    # Clear stale queue items
    while not _token_queue.empty():
        try:
            _token_queue.get_nowait()
        except queue.Empty:
            break

    # Start worker thread
    def _worker():
        try:
            ret = _rkllm_wrapper.run(prompt, role="user",
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
        while True:
            if ABORT_EVENT.is_set():
                _rkllm_wrapper.abort()
                break

            if time.time() - generation_start > GENERATION_TIMEOUT:
                logger.warning(f"[{request_id}] Generation timeout")
                _rkllm_wrapper.abort()
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
                _rkllm_wrapper.abort()
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
            if rag_cache_info and generation_clean and full_content:
                cache_text = full_content
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
                if _rkllm_wrapper:
                    _rkllm_wrapper.abort()
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

    logger.info("=" * 60)
    logger.info("RKLLM API starting (ctypes direct API)")
    logger.info(f"  Library: {RKLLM_LIB_PATH}")
    logger.info(f"  Models : {len(MODELS)} detected")
    logger.info(f"  Aliases: {len(ALIASES)} generated")
    logger.info(f"  Port   : 8000")
    logger.info("=" * 60)
    logger.info(f"Models: {list(MODELS.keys())}")
    logger.info(f"Aliases: {ALIASES}")
    app.run(host="0.0.0.0", port=8000, threaded=True)
