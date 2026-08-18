#!/usr/bin/env python3
"""
embed_api.py — RKLLM Embedding & Reranker Service
Companion to api.py. Runs on port 8001 (never shares port 8000).

Endpoints:
  POST /v1/embeddings   OpenAI-compatible text embeddings
  POST /v1/rerank       Cohere-compatible document reranking
  GET  /health          Health / readiness check

Required models (RKLLM v1.2.1+, place in ~/models/ or set env vars):
  EMBED_MODEL_PATH   — Qwen3-Embedding-0.6B  .rkllm
  RERANK_MODEL_PATH  — Qwen3-Reranker-0.6B   .rkllm

Pre-converted models for RK3588:
  https://huggingface.co/dulimov/Qwen3-Embedding-0.6B-rk3588-1.2.1
  https://huggingface.co/dulimov/Qwen3-Reranker-0.6B-rk3588-1.2.1

If only one model is available the other endpoint returns 503; the service
still starts and serves what it has.
"""

import ctypes
import logging
import math
import os
import sys
import threading
import time
import uuid

import numpy as np
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get('RKLLM_EMBED_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('embed_api')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_ROOT = os.path.expanduser(os.environ.get('RKLLM_MODELS_ROOT', '~/models'))
PORT        = int(os.environ.get('RKLLM_EMBED_PORT', 8001))

# Model paths — can be absolute or relative to MODELS_ROOT
_EMBED_PATH_ENV  = os.environ.get('EMBED_MODEL_PATH', '')
_RERANK_PATH_ENV = os.environ.get('RERANK_MODEL_PATH', '')

def _resolve_model_path(env_val: str, glob_hints: list) -> str:
    """Return the first existing path from env override or hint list."""
    if env_val and os.path.exists(env_val):
        return env_val
    for hint in glob_hints:
        p = os.path.join(MODELS_ROOT, hint) if not os.path.isabs(hint) else hint
        if os.path.exists(p):
            return p
    return ''

EMBED_MODEL_PATH = _resolve_model_path(_EMBED_PATH_ENV, [
    'qwen3-embedding-0.6b',
    'Qwen3-Embedding-0.6B',
    'qwen3_embedding_0.6b',
])
RERANK_MODEL_PATH = _resolve_model_path(_RERANK_PATH_ENV, [
    'qwen3-reranker-0.6b',
    'Qwen3-Reranker-0.6B',
    'qwen3_reranker_0.6b',
])

# Qwen3 tokenizer token IDs for reranker yes/no classification
YES_TOKEN_ID = int(os.environ.get('RERANK_YES_TOKEN_ID', 9693))
NO_TOKEN_ID  = int(os.environ.get('RERANK_NO_TOKEN_ID',  2152))

# Default reranker instruction shown to the model
RERANK_INSTRUCTION = os.environ.get(
    'RERANK_INSTRUCTION',
    'Retrieve relevant passages that answer the query.'
)

# Max tokens to generate — embeddings need 0, reranker needs 1
EMBED_MAX_NEW_TOKENS  = 0
RERANK_MAX_NEW_TOKENS = 1

# RK3588 big-core mask (cores 4-7) for NPU-adjacent CPU work
CPU_MASK = int(os.environ.get('RKLLM_CPU_MASK', '0xF0'), 16)

# ---------------------------------------------------------------------------
# RKLLM shared library
# ---------------------------------------------------------------------------
RKLLM_LIB_PATH = os.environ.get('RKLLM_LIB_PATH', '')
if not RKLLM_LIB_PATH:
    for _c in [os.path.expanduser('~/librkllmrt.so'),
               '/usr/lib/librkllmrt.so', 'lib/librkllmrt.so',
               '/usr/local/lib/librkllmrt.so', 'librkllmrt.so']:
        if os.path.exists(_c):
            RKLLM_LIB_PATH = _c
            break
    if not RKLLM_LIB_PATH:
        RKLLM_LIB_PATH = 'librkllmrt.so'

try:
    _lib = ctypes.CDLL(RKLLM_LIB_PATH)
    logger.info(f'Loaded RKLLM runtime: {RKLLM_LIB_PATH}')
except OSError as e:
    logger.critical(f'Cannot load {RKLLM_LIB_PATH}: {e}')
    sys.exit(1)

# ---------------------------------------------------------------------------
# ctypes struct definitions — must match RKLLM v1.2.x ABI exactly
# These are identical to the definitions in api.py; both files target the
# same librkllmrt.so, so the layouts must stay in sync.
# ---------------------------------------------------------------------------
RKLLM_Handle_t = ctypes.c_void_p

RKLLM_RUN_NORMAL  = 0
RKLLM_RUN_WAITING = 1
RKLLM_RUN_FINISH  = 2
RKLLM_RUN_ERROR   = 3

RKLLM_INPUT_PROMPT = 0
RKLLMInputType     = ctypes.c_int
RKLLMInferMode     = ctypes.c_int

RKLLM_INFER_GENERATE              = 0
RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLM_INFER_GET_LOGITS            = 2


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ('base_domain_id',   ctypes.c_int32),
        ('embed_flash',      ctypes.c_int8),
        ('enabled_cpus_num', ctypes.c_int8),
        ('enabled_cpus_mask',ctypes.c_uint32),
        ('n_batch',          ctypes.c_uint8),
        ('use_cross_attn',   ctypes.c_int8),
        ('reserved',         ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ('model_path',       ctypes.c_char_p),
        ('max_context_len',  ctypes.c_int32),
        ('max_new_tokens',   ctypes.c_int32),
        ('top_k',            ctypes.c_int32),
        ('n_keep',           ctypes.c_int32),
        ('top_p',            ctypes.c_float),
        ('temperature',      ctypes.c_float),
        ('repeat_penalty',   ctypes.c_float),
        ('frequency_penalty',ctypes.c_float),
        ('presence_penalty', ctypes.c_float),
        ('mirostat',         ctypes.c_int32),
        ('mirostat_tau',     ctypes.c_float),
        ('mirostat_eta',     ctypes.c_float),
        ('skip_special_token',ctypes.c_bool),
        ('is_async',         ctypes.c_bool),
        ('img_start',        ctypes.c_char_p),
        ('img_end',          ctypes.c_char_p),
        ('img_content',      ctypes.c_char_p),
        ('extend_param',     RKLLMExtendParam),
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [('lora_adapter_name', ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ('save_prompt_cache', ctypes.c_int),
        ('prompt_cache_path', ctypes.c_char_p),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [('prompt_input', ctypes.c_char_p)]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ('role',          ctypes.c_char_p),
        ('enable_thinking',ctypes.c_bool),
        ('input_type',    RKLLMInputType),
        ('input_data',    RKLLMInputUnion),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ('mode',                RKLLMInferMode),
        ('lora_params',         ctypes.POINTER(RKLLMLoraParam)),
        ('prompt_cache_params', ctypes.POINTER(RKLLMPromptCacheParam)),
        ('keep_history',        ctypes.c_int),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ('hidden_states', ctypes.POINTER(ctypes.c_float)),
        ('embd_size',     ctypes.c_int),
        ('num_tokens',    ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ('logits',     ctypes.POINTER(ctypes.c_float)),
        ('vocab_size', ctypes.c_int),
        ('num_tokens', ctypes.c_int),
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ('prefill_time_ms',  ctypes.c_float),
        ('prefill_tokens',   ctypes.c_int),
        ('generate_time_ms', ctypes.c_float),
        ('generate_tokens',  ctypes.c_int),
        ('memory_usage_mb',  ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ('text',              ctypes.c_char_p),
        ('token_id',          ctypes.c_int),
        ('last_hidden_layer', RKLLMResultLastHiddenLayer),
        ('logits',            RKLLMResultLogits),
        ('perf',              RKLLMPerfStat),
    ]


callback_type = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(RKLLMResult),
    ctypes.c_void_p,
    ctypes.c_int,
)

# ---------------------------------------------------------------------------
# Thread-local callback state — avoids globals with a per-call dict
# ---------------------------------------------------------------------------
_cb_state = threading.local()


def _embed_callback(result_ptr, _userdata, state):
    """Callback for RKLLM_INFER_GET_LAST_HIDDEN_LAYER.
    The hidden_states pointer is freed after this function returns — copy immediately.
    """
    if state == RKLLM_RUN_NORMAL:
        lhl = result_ptr.contents.last_hidden_layer
        if lhl.hidden_states and lhl.embd_size > 0 and lhl.num_tokens > 0:
            n, d = lhl.num_tokens, lhl.embd_size
            # Last-token pooling: take the final token's hidden state row.
            start = (n - 1) * d
            ptr = ctypes.cast(lhl.hidden_states, ctypes.POINTER(ctypes.c_float * (start + d)))
            _cb_state.embed_vec = np.frombuffer(ptr.contents, dtype=np.float32)[start:start + d].copy()
    elif state == RKLLM_RUN_FINISH:
        if not hasattr(_cb_state, 'embed_vec'):
            _cb_state.embed_vec = None
        _cb_state.embed_done = True
    elif state == RKLLM_RUN_ERROR:
        _cb_state.embed_vec  = None
        _cb_state.embed_done = True
    return 0


def _rerank_callback(result_ptr, _userdata, state):
    """Callback for RKLLM_INFER_GET_LOGITS.
    Extracts yes/no token logits from the last token position.
    """
    if state == RKLLM_RUN_NORMAL:
        lg = result_ptr.contents.logits
        if lg.logits and lg.vocab_size > 0 and lg.num_tokens > 0:
            last = (lg.num_tokens - 1) * lg.vocab_size
            yes_l = lg.logits[last + YES_TOKEN_ID]
            no_l  = lg.logits[last + NO_TOKEN_ID]
            # Numerically stable softmax over {yes, no}
            m = max(yes_l, no_l)
            yes_e = math.exp(yes_l - m)
            no_e  = math.exp(no_l  - m)
            _cb_state.rerank_score = yes_e / (yes_e + no_e)
    elif state == RKLLM_RUN_FINISH:
        if not hasattr(_cb_state, 'rerank_score'):
            _cb_state.rerank_score = 0.0
        _cb_state.rerank_done = True
    elif state == RKLLM_RUN_ERROR:
        _cb_state.rerank_score = 0.0
        _cb_state.rerank_done = True
    return 0


_embed_callback_fn  = callback_type(_embed_callback)
_rerank_callback_fn = callback_type(_rerank_callback)

# ---------------------------------------------------------------------------
# Library function signatures — required for correct argument marshaling on ARM
# Without argtypes, ctypes may pass structs by reference incorrectly on AArch64.
# ---------------------------------------------------------------------------
_lib.rkllm_init.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(RKLLMParam),
    callback_type,
]
_lib.rkllm_init.restype = ctypes.c_int

_lib.rkllm_run.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(RKLLMInput),
    ctypes.POINTER(RKLLMInferParam),
    ctypes.c_void_p,
]
_lib.rkllm_run.restype = ctypes.c_int

_lib.rkllm_destroy.argtypes = [ctypes.c_void_p]
_lib.rkllm_destroy.restype = ctypes.c_int

# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class _ModelWrapper:
    """Thin wrapper around rkllm_init / rkllm_run / rkllm_destroy."""

    def __init__(self, model_path: str, callback_fn, max_new_tokens: int, name: str):
        self.name = name
        self.handle = ctypes.c_void_p(None)
        self.loaded = False

        param = RKLLMParam()
        param.model_path       = model_path.encode()
        param.max_context_len  = 2048
        param.max_new_tokens   = max_new_tokens
        param.top_k            = 1
        param.n_keep           = -1
        param.top_p            = 0.9
        param.temperature      = 0.0
        param.repeat_penalty   = 1.0
        param.frequency_penalty= 0.0
        param.presence_penalty = 0.0
        param.mirostat         = 0
        param.mirostat_tau     = 5.0
        param.mirostat_eta     = 0.1
        param.skip_special_token = True
        param.is_async           = False
        param.img_start          = b""
        param.img_end            = b""
        param.img_content        = b""

        ext = RKLLMExtendParam()
        ext.base_domain_id    = 0
        ext.embed_flash       = 1
        ext.enabled_cpus_num  = 4
        ext.enabled_cpus_mask = CPU_MASK
        ext.n_batch           = 1
        ext.use_cross_attn    = 0
        param.extend_param = ext

        logger.info(f'Loading {name} from {model_path} …')
        t0 = time.time()
        ret = _lib.rkllm_init(
            ctypes.byref(self.handle), ctypes.byref(param), callback_fn
        )
        if ret != 0:
            raise RuntimeError(f'rkllm_init failed for {name} (ret={ret})')
        if not self.handle:
            raise RuntimeError(f'rkllm_init returned 0 but handle is null for {name} — NPU OOM?')

        self.loaded = True
        logger.info(f'{name} loaded in {time.time()-t0:.1f}s')

    def run(self, prompt: str, mode: int) -> None:
        inp = RKLLMInput()
        inp.role           = b'user'
        inp.enable_thinking = False
        inp.input_type     = RKLLM_INPUT_PROMPT
        inp.input_data.prompt_input = prompt.encode('utf-8')

        params = RKLLMInferParam()
        params.mode                = mode
        params.lora_params         = None
        params.prompt_cache_params = None
        params.keep_history        = 0  # every call is independent

        ret = _lib.rkllm_run(
            self.handle, ctypes.byref(inp), ctypes.byref(params), None
        )
        if ret != 0:
            raise RuntimeError(f'rkllm_run failed for {self.name} (ret={ret})')

    def destroy(self):
        if self.loaded and self.handle:
            _lib.rkllm_destroy(self.handle)
            self.loaded = False
            self.handle = ctypes.c_void_p(None)


# ---------------------------------------------------------------------------
# Load models at startup
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()   # serialise all RKLLM calls — NPU is single-task

_embed_wrapper  = None
_rerank_wrapper = None

def _load_models():
    global _embed_wrapper, _rerank_wrapper

    if EMBED_MODEL_PATH:
        try:
            _embed_wrapper = _ModelWrapper(
                EMBED_MODEL_PATH, _embed_callback_fn,
                EMBED_MAX_NEW_TOKENS, 'Qwen3-Embedding-0.6B'
            )
        except Exception as exc:
            logger.error(f'Embedding model failed to load: {exc}')
    else:
        logger.warning(
            'EMBED_MODEL_PATH not set or not found — '
            '/v1/embeddings will return 503'
        )

    if RERANK_MODEL_PATH:
        try:
            _rerank_wrapper = _ModelWrapper(
                RERANK_MODEL_PATH, _rerank_callback_fn,
                RERANK_MAX_NEW_TOKENS, 'Qwen3-Reranker-0.6B'
            )
        except Exception as exc:
            logger.error(f'Reranker model failed to load: {exc}')
    else:
        logger.warning(
            'RERANK_MODEL_PATH not set or not found — '
            '/v1/rerank will return 503'
        )

    # Exit if any configured model failed — systemd will retry after RestartSec.
    # NPU may be temporarily occupied by rkllm-api; a retry after 10s usually succeeds.
    _load_failed = (
        (EMBED_MODEL_PATH and not _embed_wrapper) or
        (RERANK_MODEL_PATH and not _rerank_wrapper)
    )
    if _load_failed:
        logger.critical('One or more models failed to load — exiting for systemd restart')
        sys.exit(1)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def _embed_text(text: str) -> list:
    """Return L2-normalised embedding vector for text."""
    _cb_state.embed_vec  = None
    _cb_state.embed_done = False

    _embed_wrapper.run(text, RKLLM_INFER_GET_LAST_HIDDEN_LAYER)

    vec = _cb_state.embed_vec
    if vec is None:
        raise RuntimeError('Embedding extraction returned no hidden states')

    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm
    return arr.tolist()


def _rerank_pair(query: str, document: str, instruction: str) -> float:
    """Return relevance score in [0, 1] for a query-document pair."""
    _cb_state.rerank_score = 0.0
    _cb_state.rerank_done  = False

    prompt = (
        f'<Instruct>: {instruction}\n'
        f'<Query>: {query}\n'
        f'<Document>: {document}'
    )
    _rerank_wrapper.run(prompt, RKLLM_INFER_GET_LOGITS)

    return _cb_state.rerank_score


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1 MB — sufficient for text embedding requests

# ---------------------------------------------------------------------------
# Prometheus metrics (optional — degrades gracefully if not installed)
# ---------------------------------------------------------------------------
try:
    from prometheus_flask_exporter import PrometheusMetrics
    from prometheus_client import Counter, Histogram

    _prom = PrometheusMetrics(app, path="/metrics")

    embed_requests_total = Counter(
        "embed_requests_total",
        "Total embedding requests",
        ["status"]
    )
    embed_latency_seconds = Histogram(
        "embed_latency_seconds",
        "Embedding inference latency per item",
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, float("inf"))
    )
    embed_batch_size = Histogram(
        "embed_batch_size",
        "Number of texts per embedding request",
        buckets=(1, 2, 4, 8, 16, 32, float("inf"))
    )
    rerank_requests_total = Counter(
        "rerank_requests_total",
        "Total reranking requests",
        ["status"]
    )
    rerank_latency_seconds = Histogram(
        "rerank_latency_seconds",
        "Reranking inference latency per document",
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, float("inf"))
    )
    rerank_doc_count = Histogram(
        "rerank_doc_count",
        "Number of documents per rerank request",
        buckets=(1, 2, 4, 8, 16, 32, float("inf"))
    )
    _EMBED_PROMETHEUS = True
    logger.info("Prometheus metrics enabled at /metrics")
except ImportError:
    _EMBED_PROMETHEUS = False
    logger.info("prometheus-flask-exporter not installed — /metrics disabled")


def _error(msg: str, code: int) -> tuple:
    return jsonify({'error': {'message': msg, 'code': code}}), code


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'embed_model':  _embed_wrapper.name  if _embed_wrapper  else None,
        'rerank_model': _rerank_wrapper.name if _rerank_wrapper else None,
    })


@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    if not _embed_wrapper:
        return _error('Embedding model not loaded', 503)

    body = request.get_json(silent=True)
    if not body:
        return _error('Request body must be JSON', 400)

    raw_input = body.get('input')
    if raw_input is None:
        return _error('"input" field required', 400)

    model_name = body.get('model', 'Qwen3-Embedding-0.6B')

    # Normalise to list
    if isinstance(raw_input, str):
        texts = [raw_input]
    elif isinstance(raw_input, list):
        texts = raw_input
    else:
        return _error('"input" must be a string or array of strings', 400)

    if not texts:
        return _error('"input" must not be empty', 400)

    # Optional: prefix queries with instruction for better retrieval quality.
    # Callers can pass input_type="query" to enable the instruction prefix.
    input_type  = body.get('input_type', 'document')
    task        = body.get('task', 'Retrieve relevant passages that answer the query.')
    use_prefix  = (input_type == 'query')

    data = []
    total_tokens = 0

    if _EMBED_PROMETHEUS:
        embed_batch_size.observe(len(texts))

    with _LOCK:
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                if _EMBED_PROMETHEUS:
                    embed_requests_total.labels(status="error").inc()
                return _error(f'input[{i}] must be a string', 400)

            prompt = f'Instruct: {task}\nQuery: {text}' if use_prefix else text
            t0 = time.time()
            vec = _embed_text(prompt)
            elapsed = time.time() - t0

            if _EMBED_PROMETHEUS:
                embed_latency_seconds.observe(elapsed)

            # Rough token estimate (4 chars ≈ 1 token)
            est_tokens = max(1, len(text) // 4)
            total_tokens += est_tokens

            logger.debug(
                f'embed[{i}] len={len(text)} dim={len(vec)} {elapsed*1000:.0f}ms'
            )
            data.append({'object': 'embedding', 'index': i, 'embedding': vec})

    if _EMBED_PROMETHEUS:
        embed_requests_total.labels(status="ok").inc()

    return jsonify({
        'object': 'list',
        'data':   data,
        'model':  model_name,
        'usage':  {'prompt_tokens': total_tokens, 'total_tokens': total_tokens},
    })


@app.route('/v1/rerank', methods=['POST'])
def rerank():
    if not _rerank_wrapper:
        return _error('Reranker model not loaded', 503)

    body = request.get_json(silent=True)
    if not body:
        return _error('Request body must be JSON', 400)

    query     = body.get('query')
    documents = body.get('documents')

    if not query or not isinstance(query, str):
        return _error('"query" field required (string)', 400)
    if not documents or not isinstance(documents, list):
        return _error('"documents" field required (array)', 400)

    model_name       = body.get('model', 'Qwen3-Reranker-0.6B')
    instruction      = body.get('instruction', RERANK_INSTRUCTION)
    return_documents = body.get('return_documents', False)
    top_n            = body.get('top_n', len(documents))

    if _EMBED_PROMETHEUS:
        rerank_doc_count.observe(len(documents))

    scored = []
    with _LOCK:
        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                if _EMBED_PROMETHEUS:
                    rerank_requests_total.labels(status="error").inc()
                return _error(f'documents[{i}] must be a string', 400)
            t0    = time.time()
            score = _rerank_pair(query, doc, instruction)
            elapsed = time.time() - t0
            if _EMBED_PROMETHEUS:
                rerank_latency_seconds.observe(elapsed)
            logger.debug(
                f'rerank[{i}] score={score:.4f} {elapsed*1000:.0f}ms'
            )
            entry = {'index': i, 'relevance_score': score}
            if return_documents:
                entry['document'] = {'text': doc}
            scored.append(entry)

    # Sort by descending relevance, then truncate to top_n
    scored.sort(key=lambda x: x['relevance_score'], reverse=True)
    scored = scored[:top_n]

    if _EMBED_PROMETHEUS:
        rerank_requests_total.labels(status="ok").inc()

    return jsonify({
        'id':      f'rerank-{uuid.uuid4().hex[:12]}',
        'results': scored,
        'model':   model_name,
        'meta': {
            'api_version':  {'version': '1'},
            'billed_units': {'search_units': 1},
        },
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Load models at module level so gunicorn workers initialize on startup.
# __name__ check is intentionally skipped — this must run under gunicorn too.
_load_models()

if __name__ == '__main__':
    logger.info(f'Starting embed_api on port {PORT}')
    app.run(host='0.0.0.0', port=PORT, threaded=False)
