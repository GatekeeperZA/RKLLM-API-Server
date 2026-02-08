"""
RKLLM API Server - Open WebUI Managed Version (Feb 09, 2026)
============================================================================
Features:
- Open WebUI manages all prompts via OpenAI API messages format
- Auto-detects all .rkllm models in ~/models
- Context length auto-detection from filename (4k/8k/16k/32k)
- Plain text prompt only (rkllm applies chat templates internally via token IDs)
- Process reuse for fast follow-up requests
- Model switching interrupts current request and loads new model
- Robust request tracking with automatic timeout
- On-demand model loading via /v1/models/select
- Explicit model unloading
- Process health monitoring with auto-recovery
- Graceful shutdown on signals and exit
- Fixed model switch deadlock with RLock
- Aggressive unloading with immediate SIGKILL for reliability on NPU
- Process killed on interrupted generation to prevent output bleed-through
- stream_options.include_usage support for streaming token counts
- system_fingerprint in all responses
- Request body size limit (16 MB)
- <think> tag parsing for reasoning models (Qwen3, DeepSeek-R1) →
  reasoning_content in streaming delta and non-streaming message

IMPORTANT: Use -w 1 (single worker) - NPU can only load one model!
NOTE: The rkllm binary uses synchronous rkllm_run() - generation cannot be
cancelled via stdin. On client disconnect/timeout, the process is killed and
restarted for the next request to prevent leftover output contamination.

Usage:
    gunicorn -w 1 -k gevent --timeout 300 -b 0.0.0.0:8000 api:app
"""
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
import subprocess
import os
import time
import logging
from logging.handlers import RotatingFileHandler
import re
import json
import uuid
import select
import codecs
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
RKLLM_BINARY = os.environ.get('RKLLM_BINARY', 'rkllm')
SYSTEM_FINGERPRINT = "rkllm-v1.2.3"  # Sent in all API responses

# Timeouts
MODEL_LOAD_TIMEOUT = 180
GENERATION_TIMEOUT = 600
FIRST_TOKEN_TIMEOUT = 120     # Max wait for first token (includes prefill)
FALLBACK_SILENCE = 12         # Max silence between tokens after first token
MINIMUM_PREFILL_TIME = 0.3    # Discard data arriving faster than this after prompt send (residual guard)

# Defaults
MAX_TOKENS_DEFAULT = 2048
CONTEXT_LENGTH_DEFAULT = 4096

# Multi-line prompt protocol.
# When True, sends prompts with \n__END__\n delimiter (preserves paragraph
# structure in RAG prompts).  Requires the C++ binary to be recompiled
# with __END__ marker support.  When False, newlines are flattened to spaces
# (works with the stock binary but loses paragraph formatting).
# Set to True ONLY after deploying the updated llm_demo.cpp with __END__
# delimiter support.
USE_MULTILINE_PROTOCOL = False

# Context-dependent thinking for RAG queries.
# Thinking models (Qwen3, DeepSeek-R1) spend hundreds of tokens on
# reasoning.  On small context models (4k) this leaves too few tokens
# for the actual answer — so thinking is disabled via /no_think.
# On larger context models (8k+) there is room for both reasoning AND
# a detailed answer, and thinking acts as a verification mechanism:
# the model reasons about whether sources agree/contradict before
# answering.  Set to 0 to always allow thinking, or 999999 to never.
DISABLE_THINK_FOR_RAG_BELOW_CTX = 8192  # /no_think only when ctx < this value

# RAG quality controls
RAG_MIN_QUALITY_SCORE = 2       # Paragraphs scoring below this are skipped
RAG_MAX_PARAGRAPHS = 10         # Max paragraphs to include (prevents "lost in the middle")
RAG_QUALITY_FLOOR_THRESHOLD = 3 # If best paragraph scores below this, drop RAG entirely
RAG_DEDUP_SIMILARITY = 0.70     # Jaccard word-overlap threshold for near-duplicate removal

# Response caching for RAG queries (avoids redundant inference on repeated questions)
RAG_CACHE_TTL = 300             # Cache lifetime in seconds (0 to disable)
RAG_CACHE_MAX_ENTRIES = 50      # Max cached responses (LRU eviction)

# Request tracking
REQUEST_STALE_TIMEOUT = 30  # seconds idle before auto-clearing tracked request

# Monitoring
MONITOR_INTERVAL = 10  # seconds
IDLE_UNLOAD_TIMEOUT = 300  # seconds idle before auto-unloading model (0 to disable)

# =============================================================================
# OPEN WEBUI RECOMMENDED SETTINGS (Admin Panel)
# =============================================================================
# These settings affect how web search results are delivered to this API.
# For small NPU models (1.5B-4B, 4096 context), the optimal config is:
#
# Admin > Settings > Web Search:
#   - Search Engine: searxng
#   - SearXNG Query URL: http://searxng:8080/search
#   - Web Search Result Count: 3  (for 4k models; use 5 for 16k models)
#       * More results = more content to process/truncate/deduplicate
#       * 3 results is optimal for small context; 5 gives better coverage on 16k
#   - Bypass Web Loader: ON  (use search snippets, not full pages)
#       * SafeWebBaseLoader returns raw get_text() with all navigation/footer
#         boilerplate — overwhelming for small context windows.
#       * Snippets are pre-extracted by search engines and already relevant.
#       * Full pages often have JS-rendered content that SafeWebBaseLoader
#         cannot capture (time displays, weather widgets, etc.)
#   - Bypass Embedding: ON  (skip ChromaDB, send docs directly)
#       * No embedding model available on ARM/NPU
#       * Faster (skips embed + vector search loop)
#       * Better quality — full snippet text goes to LLM, not top-k chunks
#
# Admin > Settings > Documents:
#   - RAG Template: {{CONTEXT}}   (just the context variable, nothing else)
#       * The default template is 300+ tokens of meta-instructions that
#         waste context and encourage the model to use "own knowledge"
#
# Per-Model (Workspace > Models > Edit > Capabilities):
#   - Disable "Builtin Tools" for all NPU models
#       * Small models cannot do function-calling / tool-use
#   - Keep "File Context" ON
#
# SearXNG settings.yml:
#   search:
#     formats:
#       - html
#       - json    # REQUIRED for Open WebUI API access
#
# For Qwen3-4B (16K context), Bypass Web Loader OFF is viable IF the
# web loader is switched to Playwright or Firecrawl (both handle JS).
# =============================================================================

# Stopword list for content-vs-boilerplate detection (jusText-inspired).
# Prose has ~30%+ stopwords; navigation menus / link lists have < 15%.
# Used by _score_paragraph() to distinguish real content from boilerplate.
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
# Simple LRU cache: key = hash(model + question), value = (timestamp, response_text).
# Eliminates redundant NPU inference when the same question is asked again
# within the TTL window.  Web search results don't change second-by-second.
# TRADE-OFF: The cache key is (model, question) only — it does NOT include the
# RAG reference data.  If SearXNG returns different results for the same query
# within the TTL window, the cached answer (built from the old context) is served.
# This is acceptable because: (a) search results rarely change within 5 minutes,
# and (b) including the full reference text would effectively disable caching.
_rag_cache = OrderedDict()  # {cache_key: (timestamp, prompt, response_text)}
_rag_cache_lock = Lock()


def _rag_cache_key(model_name, question):
    """Generate a deterministic cache key from model + question."""
    raw = f"{model_name}:{question.strip().lower()}"
    return hashlib.md5(raw.encode('utf-8'), usedforsecurity=False).hexdigest()


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
        # Move to end (LRU)
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
        # Evict oldest if over limit
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
# Use RKLLM_API_LOG_LEVEL for the Python API's own logging.
# RKLLM_LOG_LEVEL is reserved for the rkllm C runtime (expects integer 0-2).
# Fallback chain: RKLLM_API_LOG_LEVEL → RKLLM_LOG_LEVEL → DEBUG
_log_env = os.environ.get('RKLLM_API_LOG_LEVEL') or os.environ.get('RKLLM_LOG_LEVEL', 'DEBUG')
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

        for f in files:
            if f.endswith(".rkllm"):
                rkllm_file = os.path.join(root, f)
                break

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

        # NOTE: System prompts are managed by Open WebUI and sent via the
        # OpenAI API messages array. Modelfile SYSTEM lines are NOT loaded
        # here to avoid duplicate system prompts in the context.

        MODELS[model_id] = config
        logger.info(f"Detected: {model_id} (context={context_len})")

# =============================================================================
# ALIASES (auto-generated from detected model IDs)
# =============================================================================


def generate_aliases(model_ids):
    """Auto-generate short aliases from detected model folder names.

    For each model ID (e.g. "qwen2.5-3b-instruct"), generates candidate
    aliases from:
      1. Progressive hyphen-delimited prefixes: "qwen2.5", "qwen2.5-3b"
      2. Alphabetic family name: "qwen" (letters-only from first segment)
      3. Integer-truncated version: "qwen2" (from "qwen2.5")

    An alias is only created when exactly ONE model claims it and it
    doesn't collide with an actual model ID. Ambiguous aliases (claimed
    by multiple models) are silently skipped.

    This means adding a new model folder to ~/models/ automatically
    creates its aliases — no code changes needed.
    """
    candidates = {}  # candidate_alias -> set of model_ids that claim it

    for model_id in model_ids:
        parts = model_id.split('-')

        # 1. Progressive prefix aliases (skip full ID — not an alias)
        #    e.g. "deepseek-r1-distill" -> "deepseek", "deepseek-r1"
        for i in range(1, len(parts)):
            candidate = '-'.join(parts[:i])
            if candidate != model_id:
                candidates.setdefault(candidate, set()).add(model_id)

        # 2. Family name alias (letters-only prefix of first segment)
        #    e.g. "qwen2.5" -> "qwen", "llama" -> "llama" (skip, same)
        family_match = re.match(r'^([a-zA-Z]+)', parts[0])
        if family_match:
            family = family_match.group(1).lower()
            if family != model_id and family != parts[0].lower():
                candidates.setdefault(family, set()).add(model_id)

        # 3. Integer-truncated version alias
        #    e.g. "qwen2.5" -> "qwen2"
        if '.' in parts[0]:
            truncated = parts[0].split('.')[0]
            if truncated != model_id and truncated != parts[0].lower():
                candidates.setdefault(truncated, set()).add(model_id)

    # Only create alias when exactly one model claims it
    aliases = {}
    for alias, claimants in sorted(candidates.items(), key=lambda x: len(x[0])):
        if len(claimants) == 1 and alias not in model_ids:
            aliases[alias] = next(iter(claimants))

    return aliases


ALIASES = generate_aliases(MODELS.keys())

logger.info(f"Models: {list(MODELS.keys())}")
logger.info(f"Aliases: {ALIASES}")

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
ABORT_EVENT = Event()  # Thread-safe abort signaling

CURRENT_PROCESS = None
CURRENT_MODEL = None
PROCESS_LOCK = RLock()  # Reentrant lock - fixes deadlock during model switch

SHUTDOWN_EVENT = Event()
GENERATION_COMPLETE = Event()
GENERATION_COMPLETE.set()  # Initially set — no generation is running
SERVER_START_TIME = int(time.time())
LAST_REQUEST_TIME = 0  # Track last request completion for idle auto-unload


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
    """Atomic check-and-set: reject if busy, else claim the request slot.

    Returns True if the request was successfully started, False if another
    request is already active.  Eliminates the race window between
    is_request_active() and start_request() — safe even if future code
    adds I/O (gevent yield points) between the check and the call.
    """
    with ACTIVE_LOCK:
        if ACTIVE_REQUEST["id"] is not None:
            elapsed = time.time() - ACTIVE_REQUEST["last_activity"]
            if elapsed <= REQUEST_STALE_TIMEOUT:
                return False
            # Stale — clear and reclaim
            logger.warning(f"Request {ACTIVE_REQUEST['id']} stale ({elapsed:.0f}s idle) - auto-clearing")
        ACTIVE_REQUEST["id"] = request_id
        ACTIVE_REQUEST["start_time"] = time.time()
        ACTIVE_REQUEST["last_activity"] = time.time()
        ACTIVE_REQUEST["model"] = model
        logger.info(f"Request STARTED: {request_id} for model {model}")
        return True


def force_clear_if_orphaned():
    # Check process health BEFORE acquiring ACTIVE_LOCK to avoid
    # lock-ordering issues (PROCESS_LOCK is read inside is_process_healthy)
    healthy = is_process_healthy()
    with ACTIVE_LOCK:
        if ACTIVE_REQUEST["id"] is None:
            return
        # During model loading CURRENT_MODEL hasn't been set yet, so the
        # process appears "unhealthy".  Don't clear the request until the
        # requested model is actually loaded (CURRENT_MODEL matches).
        if CURRENT_MODEL != ACTIVE_REQUEST["model"]:
            return
        if not healthy:
            logger.warning(f"Clearing orphaned request {ACTIVE_REQUEST['id']} - process died")
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


def is_stats_line(line):
    """Detect rkllm statistics/summary lines that signal end of generation."""
    if not line or not line.strip():
        return False
    s = line.strip()
    # rkllm bracketed stats: [Token/s]: 9.25, [Tokens]: 560, [Seconds]: 60.57
    if re.match(r'\[(?:Token/s|Tokens|Seconds)\]\s*:', s, re.IGNORECASE):
        return True
    # Alternative stats patterns: "N tokens, X ms, Y token/s" or "prefill ... generate ..."
    if re.search(r'\d+\s*tokens?', s, re.IGNORECASE) and \
       re.search(r'(tok(en)?s?/s|ms\b|speed)', s, re.IGNORECASE):
        return True
    # rkllm v1.2.x table-format stats (RKLLM_LOG_LEVEL=1):
    #   I rkllm:  Generate      68167.06         928       73.46    13.61
    #   I rkllm:  Prefill       137.74           18        7.65     130.69
    if s.startswith('I rkllm:') and re.search(
            r'(Generate|Prefill)\s+[\d.]+\s+\d+\s+[\d.]+\s+[\d.]+', s):
        return True
    return False


def _parse_stats_line(line, stats_data):
    """Parse rkllm statistics values from a stats line into stats_data dict."""
    s = line.strip()
    m = re.match(r'\[Tokens\]\s*:\s*(\d+)', s, re.IGNORECASE)
    if m:
        stats_data['completion_tokens'] = int(m.group(1))
        return
    m = re.match(r'\[Token/s\]\s*:\s*([\d.]+)', s, re.IGNORECASE)
    if m:
        stats_data['tokens_per_sec'] = float(m.group(1))
        return
    m = re.match(r'\[Seconds\]\s*:\s*([\d.]+)', s, re.IGNORECASE)
    if m:
        stats_data['seconds'] = float(m.group(1))
        return
    # Alternative format: "prefill X tokens ... generate Y tokens ..."
    m = re.search(r'generate\s+(\d+)\s+tokens?', s, re.IGNORECASE)
    if m:
        stats_data['completion_tokens'] = int(m.group(1))
        return
    # rkllm v1.2.x table format:
    #   I rkllm:  Generate  68167.06  928  73.46  13.61
    #   Fields: Stage  TotalTime(ms)  Tokens  TimePerToken(ms)  TokensPerSec
    m = re.search(r'I rkllm:\s*Generate\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)', s)
    if m:
        stats_data['seconds'] = float(m.group(1)) / 1000.0
        stats_data['completion_tokens'] = int(m.group(2))
        stats_data['tokens_per_sec'] = float(m.group(4))
        return


def is_noise_line(line):
    """Detect rkllm runtime debug output that should be filtered."""
    if line is None:
        return True
    s = line.strip()
    # Blank lines are NOT noise — they are paragraph separators
    # in the model's output and must be preserved for formatting.
    if not s:
        return False
    # rkllm runtime info/warning/error/debug prefixes
    if s.startswith(('I rkllm:', 'W rkllm:', 'E rkllm:', 'D rkllm:')):
        return True
    if s in ('RKLLM starting, please wait...', 'rkllm init success'):
        return True
    # rkllm binary prompt indicators (printed between turns)
    if re.match(r'^(user|robot)\s*:\s*$', s, re.IGNORECASE):
        return True
    return False


def clean_line(line):
    """Clean output line, removing control characters but keeping whitespace."""
    if not line:
        return ""
    # Remove control chars except tab, newline, carriage return
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', line)
    return cleaned


def _clean_web_content(text):
    """Strip web page navigation/boilerplate from scraped content.

    Open WebUI's SafeWebBaseLoader returns soup.get_text() — raw text with
    ALL navigation menus, cookie banners, sidebar links, footer cruft, and
    JS-framework text.  These overwhelm small NPU models (1.5B-4B) and
    waste precious context tokens.

    Strategy — four-pass line-level filtering:
    Pass 1: Remove known boilerplate phrases (copyright, sign-in, cookies, etc.)
    Pass 2: Remove concatenated navigation (CamelCase runs, title-case-heavy
            sequences, URL-heavy lines)
    Pass 3: Collapse consecutive short-line navigation menus (4+ short lines
            in a row with no sentence punctuation → all dropped)
    Pass 4: Keep only lines with data signals (digits, prose punctuation,
            or substantial length ≥ 40 chars)
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
        # CamelCase transitions (lowercase→uppercase without space):
        # e.g. "NewsHome", "TimeZone", "NetherlandsSee"
        if len(re.findall(r'[a-z][A-Z]', s)) >= 2 and not re.search(r'[.!?]', s):
            continue

        # Title-case-heavy lines without sentence structure:
        # e.g. "News Home Astronomy News Time Zone Calendar"
        words = s.split()
        if len(words) >= 4 and not re.search(r'[.!?]', s):
            cap_words = sum(1 for w in words if w and w[0].isupper())
            if cap_words / len(words) > 0.65:
                continue

        # URL-heavy lines (navigation sidebars, link lists)
        url_count = len(re.findall(r'https?://', s))
        if url_count >= 2:
            continue
        # Single URL on a very short line = standalone link
        if url_count == 1 and len(s) < 80 and not re.search(r'[.!?]', s):
            continue

        pass1.append(s)

    # --- Pass 3: Collapse consecutive short-line runs (menus) ---
    # 4+ lines in a row all < 30 chars with no sentence punctuation = menu
    pass2 = []
    run_start = -1
    run_count = 0
    for i, line in enumerate(pass1):
        is_short_nav = (line and len(line) < 30 and not re.search(r'[.!?]', line))
        # Treat empty lines as transparent — don't break short-line runs.
        # Navigation menus extracted by get_text() can have blank lines
        # between items; breaking the run would preserve those menus.
        is_empty = not line
        if is_short_nav or (is_empty and run_count > 0):
            if run_count == 0:
                run_start = i
            run_count += 1
        else:
            if run_count >= 4:
                # Drop the entire run — it's a navigation menu
                pass
            else:
                # Keep the short run (< 4 lines, probably legitimate)
                for j in range(run_start, run_start + run_count):
                    if run_start >= 0 and j < len(pass1):
                        pass2.append(pass1[j])
            run_count = 0
            run_start = -1
            pass2.append(line)
    # Handle trailing run
    if run_count >= 4:
        pass  # drop
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

        # Keep lines with digits — times, dates, statistics, tables
        if re.search(r'\d', s):
            cleaned.append(s)
            continue

        # Keep lines with sentence punctuation — actual prose
        if re.search(r'[.!?;:]', s):
            cleaned.append(s)
            continue

        # Keep lines 40+ chars — likely paragraph fragments
        if len(s) >= 40:
            cleaned.append(s)
            continue

        # Everything else is likely short UI labels / cruft — skip

    result = '\n'.join(cleaned).strip()
    # Fall back to original if aggressive cleaning removed everything
    return result if len(result) > 100 else text


def _score_paragraph(para, query_words=None):
    """Score a paragraph for content relevance (higher = more useful).

    Used during smart truncation to select the most informative paragraphs
    from scraped web pages, replacing naive positional selection (first /
    middle / last) which grabs navigation headers and copyright footers.

    Signals (jusText-inspired + query-aware):
    + Stopword density > 0.30 (prose text has high stopword ratio)
    + Length, sentence count, digit/data presence, query keyword matches
    - Stopword density < 0.15 on long text (boilerplate, link lists)
    - Short fragments, navigation patterns, boilerplate keywords
    """
    score = 0
    words = para.split()
    word_count = len(words)

    # --- Stopword density (the #1 jusText signal) ---
    # Real prose has ~30-45% stopwords. Navigation/menus have < 15%.
    if word_count >= 5:
        sw_count = sum(1 for w in words if w.lower() in ENGLISH_STOPWORDS)
        sw_density = sw_count / word_count
        if sw_density >= 0.30:
            score += 3  # Strong prose indicator
        elif sw_density >= 0.20:
            score += 1  # Moderate prose
        elif word_count >= 15:
            score -= 2  # Long text with few stopwords = boilerplate/links

    # --- Positive signals ---
    if len(para) >= 200:
        score += 3
    elif len(para) >= 100:
        score += 2
    elif len(para) >= 50:
        score += 1

    # Complete sentences (4+ words before sentence ender)
    sentences = [s.strip() for s in re.split(r'[.!?]', para)
                 if len(s.strip().split()) >= 4]
    score += min(len(sentences), 3)

    # Data presence — only reward if within prose-like context
    # (prevents clock/timezone sites from scoring high on bare numbers)
    digit_groups = re.findall(r'\d[\d,.:%/\-]*', para)
    if sentences:  # digits + sentences = real data
        score += min(len(digit_groups), 4)
    elif digit_groups and word_count >= 8:
        score += min(len(digit_groups), 2)  # less reward without sentences
    # bare digits without context: no bonus

    # Query keyword matches (biggest relevance signal)
    if query_words:
        para_lower = para.lower()
        matches = sum(1 for w in query_words if w in para_lower)
        score += matches * 3

    # --- Negative signals ---
    if len(para) < 20:
        score -= 4  # very short fragment — almost certainly UI label
    elif len(para) < 40:
        score -= 2  # short fragment

    # High capitalisation without sentence structure (navigation)
    if word_count >= 3:
        cap_words = sum(1 for w in words if w and w[0].isupper())
        if cap_words / word_count > 0.7 and not re.search(r'[.!?]', para):
            score -= 3

    # Heavy emoji/symbol content (UI elements, widget labels)
    emoji_count = len(re.findall(r'[\U0001f300-\U0001f9ff⏱⏰🌍🌐🏙🏴🗺🌦🌅🌇🕰🎉📆🔁🌬]', para))
    if emoji_count >= 2:
        score -= 3

    # Boilerplate keywords
    lower = para.lower()
    if any(kw in lower for kw in (
            'copyright', 'all rights reserved', 'privacy policy',
            'terms of service', 'cookie policy', 'sign in',
            'subscribe', 'advertisement', 'sponsored',
            'free widget', 'webmaster', 'full screen clock',
            'atomic-clock', 'advert free')):
        score -= 4

    return score


def _strip_system_fluff(text):
    """Remove generic assistant instructions from system messages.

    The rkllm binary sends ALL text as a single user turn with the model's
    internal chat template.  Generic instructions like "You are a helpful
    assistant" appearing in the user turn trigger the model to respond with
    a greeting ("Hello! How can I assist you?") instead of answering the
    actual question.

    Also strips bare date/time-only system prompts that Open WebUI sends
    (e.g. "Today is 2026-02-07 (Saturday), 17:45:54.").  These are useful
    context for date-related questions but are handled by returning them
    separately.  When they are the ONLY content in the system prompt,
    leaving them in a single-turn prompt makes the model respond about
    the date instead of the actual question.

    Returns the cleaned text.  May return empty string if only fluff.
    """
    # Match ONLY standalone generic phrases — not domain-specific instructions.
    # "You are a helpful assistant." → strip
    # "You are a helpful cooking assistant who specializes in Italian cuisine." → KEEP
    # Strategy: match the generic phrase only when followed by sentence-end
    # punctuation or end-of-string.  If it continues with more words
    # (e.g. "who specializes in..."), leave the entire sentence intact.
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
    """Check if system message contains ONLY date/time info and nothing else.

    Open WebUI injects 'Today is YYYY-MM-DD (DayName), HH:MM:SS.' as the
    system prompt.  For questions unrelated to date/time, this should be
    omitted to prevent the model from latching onto it.
    """
    s = text.strip()
    # Match patterns like "Today is 2026-02-07 (Saturday), 17:45:54."
    # with possible trailing punctuation/whitespace variations
    if re.match(r'^today is \d{4}-\d{2}-\d{2}\b.*$', s, re.IGNORECASE):
        # Check there's nothing else substantial after the date line
        lines = [l.strip() for l in s.split('\n') if l.strip()]
        if len(lines) <= 1:
            return True
    return False


def _extract_rag_reference(system_text):
    """Extract reference data from a RAG-injected system message.

    Supports multiple Open WebUI RAG template formats:
    1. Custom: "Here is some reference information: ... answer the question: X"
    2. Default v0.7.2: "### Task: ... <context>{{CONTEXT}}</context>"
    3. Minimal: just "{{CONTEXT}}" (recommended for small NPU models)

    All formats inject <source id="N" name="...">content</source> tags.
    This function uses those tags as reliable anchors for extraction,
    making it format-agnostic.

    Returns: (instructions, reference_data, rag_question) or None if not RAG.
    """
    # Detect RAG injection by <source> tags (always present in Open WebUI RAG)
    has_source = '<source' in system_text
    has_context_tags = '<context>' in system_text.lower()
    has_ref_preamble = 'reference information' in system_text.lower()

    if not has_source and not has_ref_preamble and not has_context_tags:
        return None

    # === Anchor-based extraction using <source> tag positions ===
    # This works regardless of which RAG template is configured.
    if has_source:
        source_start = system_text.find('<source')
        # Everything before the first <source> tag = system prompt + template preamble
        instructions = system_text[:source_start].strip()

        # Find end of last </source> tag
        last_close = system_text.rfind('</source>')
        if last_close >= 0:
            last_close += len('</source>')
        else:
            last_close = len(system_text)

        # Reference data = content within/between source tags
        reference_data = system_text[source_start:last_close]

        # Trailing text after last source tag may contain embedded question
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
        # Fallback for templates without <source> tags (unlikely but safe)
        instructions = ""
        reference_data = system_text
        rag_question = ""

    # Strip template framing from instructions (keep only system prompt)
    # Remove "Here is some reference information:", "### Task:", <context>, etc.
    instructions = re.sub(r'(?:Here is some )?reference information\s*:?\s*$', '', instructions, flags=re.IGNORECASE).strip()
    instructions = re.sub(r'###\s*Task:.*$', '', instructions, flags=re.IGNORECASE | re.DOTALL).strip()
    instructions = re.sub(r'</?context>\s*', '', instructions, flags=re.IGNORECASE).strip()

    # Clean the reference data
    raw_len = len(reference_data)
    reference_data = re.sub(r'</?source[^>]*>', '', reference_data)
    reference_data = re.sub(r'</?context>\s*', '', reference_data, flags=re.IGNORECASE)
    reference_data = _clean_web_content(reference_data)
    reference_data = re.sub(r'\n{3,}', '\n\n', reference_data)
    reference_data = re.sub(r'[ \t]{2,}', ' ', reference_data)
    logger.debug(f"RAG content cleaned: {raw_len} -> {len(reference_data)} chars "
                 f"({100 - len(reference_data) * 100 // max(raw_len, 1)}% removed)")

    return (instructions, reference_data, rag_question)


def build_prompt(messages, model_name):
    """Convert OpenAI messages array to plain text for rkllm.

    IMPORTANT: The rkllm runtime applies chat templates INTERNALLY using
    actual token IDs (not text). The model's special tokens (e.g.
    <|im_start|>, <|begin_of_text|>, <start_of_turn>) are stripped from
    the text vocabulary during RKLLM model conversion - sending them as
    literal text causes the model to see them as garbage content.

    We send ONLY plain text. The rkllm binary wraps it with the correct
    chat template at the token level.

    RAG mode (web search results injected):
    - Detects when Open WebUI injects search results into the system message
    - Restructures as reading comprehension: raw data first, question last
    - Strips <source> XML tags, removes navigation junk
    - Score-based truncation selects most relevant paragraphs (Fix 17)
    - NO headers/labels/framing — small models need pure data-first format

    Normal mode (no RAG injection):
    - Standard concatenation of system/user/assistant messages
    - Multi-turn conversations get lightweight role labels
    """
    # Get model context length for smart truncation
    model_cfg = MODELS.get(model_name, {})
    ctx = model_cfg.get('context_length', CONTEXT_LENGTH_DEFAULT) if model_cfg else CONTEXT_LENGTH_DEFAULT

    # Collect messages by role
    system_parts = []  # accumulate all system messages (Open WebUI usually sends one)
    conversation = []  # list of (role, content)
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if not content:
            continue
        if role == 'system':
            system_parts.append(content)
        else:
            conversation.append((role, content))
    if len(system_parts) > 1:
        logger.warning(f"Multiple system messages ({len(system_parts)}) — concatenating")
    system_text = "\n".join(system_parts)

    # Get the actual user question (last user message)
    user_question = ""
    for role, content in reversed(conversation):
        if role == 'user':
            user_question = content
            break

    # Try to detect RAG injection in system message
    rag_parts = _extract_rag_reference(system_text) if system_text else None

    prompt = ""  # Safety default — overwritten by RAG or normal-mode branch below

    # =====================================================================
    # FOLLOW-UP / IRRELEVANT-RAG DETECTION
    # =====================================================================
    # Open WebUI searches SearXNG with the raw user message.  Short follow-up
    # replies ("yes", "tell me more", "why?") produce garbage search results.
    # Three layers of defense, checked in order:
    #
    #   Layer 1: Exact-match word list (cheapest — catches the obvious ones)
    #   Layer 2: Short query + conversation history (generalizes layer 1)
    #   Layer 3: Query-to-reference topical overlap (catches subtle mismatches)
    #
    # If any layer fires, RAG is skipped and the model uses normal
    # conversation mode with full chat history intact.

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

        # --- Layer 2: Short query (≤3 words) with conversation history ---
        # A 1-3 word message after an assistant turn is almost certainly a
        # follow-up, not a new research question.  "another one", "explain that",
        # "south africa" etc. all produce bad SearXNG results.
        if not _skip_reason and _has_assistant_turn:
            _query_words = _query_normalized.split()
            if len(_query_words) <= 3:
                _skip_reason = (f"short follow-up ({len(_query_words)} words) "
                                f"with conversation history")

        # --- Layer 3: Query-to-reference topical overlap ---
        # Check whether the search results are actually about the user's topic.
        # Extract content words from the query and measure what fraction appear
        # anywhere in the reference text.  If < 30%, the search returned
        # off-topic results (e.g. "yes" → Yes the prog rock band).
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
            if _qcw and len(_qcw) <= 8:  # Only for focused queries, not essays
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
            system_text = ""  # Drop the injected RAG system text

    if rag_parts and user_question:
        # === RAG MODE ===
        # Web search data detected — restructure prompt for small NPU models.
        # See Fix 16 comment below for rationale.
        instructions, reference_data, rag_question = rag_parts

        logger.info(f"RAG mode: instructions={len(instructions)} chars, "
                     f"reference={len(reference_data)} chars, "
                     f"rag_question='{rag_question[:80]}', "
                     f"user_question='{user_question[:80]}'")

        # Budget: keep reference data compact so truncation fires and
        # score-based selection picks only the most relevant paragraphs.
        # ctx*1.5 chars ≈ 37.5% of context tokens — leaves plenty for
        # chat template overhead + generation output.
        # For 4K ctx: ~5900 chars (~1475 tokens) → forces smart truncation
        # For 16K ctx: ~24K chars → generous but truncation still helps
        max_ref_chars = int(ctx * 1.5) - len(user_question) - 200
        max_ref_chars = max(500, max_ref_chars)  # minimum useful reference

        _rag_best_score = None  # Track best paragraph score for quality floor

        if len(reference_data) > max_ref_chars:
            original_ref = len(reference_data)
            # Score-based truncation: rank paragraphs by content quality
            # and query relevance, then select the best ones within budget.
            # Replaces naive positional (first/middle/last) which grabbed
            # navigation headers and copyright footers from web pages.

            # Step 1: Merge tiny consecutive paragraphs into blocks.
            # web get_text() creates hundreds of single-line "paragraphs"
            # from <div>, <span>, <td> etc. — merge them so scoring sees
            # coherent context instead of isolated fragments.
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

            # Step 2: Deduplicate near-identical paragraphs.
            # Time/weather sites repeat city times, zone info, etc.
            # Two-pass: (a) exact prefix key, (b) Jaccard word similarity.
            seen = set()
            unique_paras = []
            for p in paras:
                key = re.sub(r'[\s\d:]+', '', p.lower())[:80]
                if key not in seen:
                    seen.add(key)
                    unique_paras.append(p)
            # Pass 2: remove paraphrased duplicates from different sources
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
                # Extract query keywords for relevance scoring
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

                # Log top 5 and bottom 5 for debugging
                if logger.isEnabledFor(logging.DEBUG):
                    for i, (idx, sc, p) in enumerate(scored[:5]):
                        logger.debug(f"  TOP-{i+1} score={sc}: {p[:100]}")
                    for i, (idx, sc, p) in enumerate(scored[-3:]):
                        logger.debug(f"  BOT-{i+1} score={sc}: {p[:100]}")

                # Greedily pick top-scoring paragraphs within budget.
                # RAG_MIN_QUALITY_SCORE (default 2) = at least one positive
                # signal (prevents filling budget with low-value fragments).
                # RAG_MAX_PARAGRAPHS caps count to avoid "lost in the middle"
                # where models ignore content buried in long contexts.
                picked = []
                budget = max_ref_chars
                for idx, sc, para in scored:
                    if budget <= 0:
                        break
                    if len(picked) >= RAG_MAX_PARAGRAPHS:
                        break
                    if sc < RAG_MIN_QUALITY_SCORE:
                        continue  # require at least moderate quality
                    cost = len(para) + 2  # +2 for \n\n separator
                    if cost <= budget:
                        picked.append(idx)
                        budget -= cost

                if picked:
                    picked.sort()  # restore original document order
                    reference_data = '\n\n'.join(paras[i] for i in picked)
                    logger.info(f"Smart truncation: {len(raw_paras)} raw -> "
                                f"{len(paras)} merged/deduped -> "
                                f"{len(picked)} picked ({len(reference_data)} "
                                f"chars) [min_score={RAG_MIN_QUALITY_SCORE}, "
                                f"max_paras={RAG_MAX_PARAGRAPHS}, "
                                f"{len(qwords)} keywords]")
                else:
                    # All paragraphs scored poorly — take top 10 by score
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
                # Few paragraphs — simple hard cut at limit
                reference_data = reference_data[:max_ref_chars]
                last_nl = reference_data.rfind('\n')
                if last_nl > max_ref_chars // 2:
                    reference_data = reference_data[:last_nl]
            logger.warning(f"Reference data truncated: {original_ref} -> {len(reference_data)} chars")

        # === Quality Floor Check ===
        # If the best-scoring paragraph is below RAG_QUALITY_FLOOR_THRESHOLD,
        # the web search returned irrelevant results.  Drop RAG entirely
        # and let the model answer from training data — "garbage in, garbage
        # out" wastes context and confuses small models.
        # Only applies when scoring was performed (truncation fired).
        if (_rag_best_score is not None
                and _rag_best_score < RAG_QUALITY_FLOOR_THRESHOLD):
            logger.warning(f"Quality floor triggered: best_score={_rag_best_score} < "
                          f"threshold={RAG_QUALITY_FLOOR_THRESHOLD}. "
                          f"Dropping RAG context, falling back to model knowledge.")
            # Fall through to normal mode by clearing rag_parts.
            # Also clear system_text so normal mode doesn't see RAG cruft.
            rag_parts = None
            system_text = ""

    # Re-check after quality floor may have cleared rag_parts
    if rag_parts and user_question:
        # Note: reference_data was already modified by truncation/dedup above.
        # Do NOT re-extract from rag_parts here — that would undo truncation.

        # Fix 16: Revert to proven Fix 13 format (data-first, no header).
        # Fix 17: Improved _clean_web_content + score-based truncation
        #         for full web pages (Bypass Web Loader OFF).
        #
        # Fix history on RAG prompt format:
        #   Fix 12: {data}\n\nAccording to the above, {Q} + instructions → 2/4
        #   Fix 13: Same but instructions OMITTED → 3/4 (BEST EVER)
        #   Fix 14: Query:/Answer: + /no_think → 0/4 REGRESSION
        #   Fix 15: "Retrieved from the web:" header → 0/4 REGRESSION
        #   Fix 16: Revert to Fix 13 format → 4/4 PERFECT (snippets)
        #   Fix 17: Better web cleaning + scored truncation (full pages)
        #
        # Key insight: small NPU models (1.5B-4B) need the SIMPLEST
        # possible reading comprehension format. Data must come FIRST
        # with ZERO preamble. Any meta-framing ("retrieved from web",
        # "not prior knowledge", Query:/Answer:) breaks the flow.
        #
        # This format mirrors SQuAD reading comprehension training:
        # raw passage followed by a question about it.
        # Context-dependent /no_think for RAG:
        # - Small context (< DISABLE_THINK_FOR_RAG_BELOW_CTX): disable
        #   thinking so all tokens go to the answer.  On 4k models,
        #   reasoning consumed 2336 chars and left only 319 for content.
        # - Large context (>= threshold): allow thinking.  The model
        #   reasons about whether sources agree/contradict, acting as
        #   a lightweight verification step (Thread-of-Thought pattern).
        # "Answer in detail" nudge: small models (1.7B) tend to generate
        # very short answers (~70 tokens) and stop.  A brief instruction
        # nudges them to use more of the reference data.
        # Abstention clause: "If not answered above, say you don't know"
        # prevents the model from fabricating answers when search results
        # don't contain relevant information (RAG Failure Point FP1).
        disable_think = ctx < DISABLE_THINK_FOR_RAG_BELOW_CTX
        no_think_suffix = ' /no_think' if disable_think else ''
        # Abstention clause only for large-context models where thinking
        # is enabled — the model can reason about source sufficiency.
        # On small models this clause competes with "Answer in detail"
        # and makes responses shorter (same regression pattern as Fix 14/15).
        abstention = ". If not answered above, say you don't know" if not disable_think else ''
        logger.info(f"RAG thinking: ctx={ctx}, threshold={DISABLE_THINK_FOR_RAG_BELOW_CTX}, "
                    f"thinking={'disabled' if disable_think else 'enabled'}")
        prompt = (
            f"{reference_data}\n\n"
            f"According to the above, {user_question}. "
            f"Answer in detail with specific facts and examples"
            f"{abstention}{no_think_suffix}"
        )
        logger.debug(f"RAG prompt built ({len(prompt)} chars, ctx={ctx})")
        if len(prompt) > 500:
            logger.debug(f"RAG prompt START: {prompt[:250]}")
            logger.debug(f"RAG prompt END: {prompt[-250:]}")
        else:
            logger.debug(f"RAG prompt FULL: {prompt}")
        approx_tokens = len(prompt) // 3
        if approx_tokens > ctx * 0.85:
            logger.warning(
                f"RAG prompt (~{approx_tokens} tokens) approaches context limit ({ctx}). "
                f"Reference data may be truncated by the model."
            )
        return prompt, True  # is_rag=True

    if not rag_parts or not user_question:
        # === NORMAL MODE (no RAG) ===
        parts = []
        turn_count = sum(1 for r, _ in conversation if r in ('user', 'assistant'))
        multi_turn = turn_count > 1

        if system_text:
            # Strip generic "You are a helpful assistant" etc — these
            # go into rkllm's user turn and trigger greeting mode
            cleaned_sys = _strip_system_fluff(system_text)
            if cleaned_sys:
                # Check if system prompt is ONLY date/time info.
                # For non-date questions, omit it — small models latch
                # onto the date and respond with "Today is..." instead
                # of answering the actual question.
                if _is_date_only_system(cleaned_sys):
                    # Only include date context if question seems date/time related.
                    # Use word-boundary regex to avoid substring false positives:
                    # "update" contains "date", "holiday" contains "day",
                    # "bedtime" contains "time", etc.
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

    logger.debug(f"Prompt built ({len(prompt)} chars, ctx={ctx}, "
                 f"rag={'yes' if rag_parts else 'no'})")

    # Log first and last ~200 chars for debugging what model actually sees
    if len(prompt) > 500:
        logger.debug(f"Prompt START: {prompt[:250]}")
        logger.debug(f"Prompt END: {prompt[-250:]}")
    else:
        logger.debug(f"Prompt FULL: {prompt}")

    # Warn if prompt likely exceeds model context window.
    # Token estimate: English ≈ len//4, CJK ≈ len//1.5.  We use len//3 as
    # a conservative middle-ground that avoids silent over-stuffing for
    # Chinese/Japanese/Korean queries (common with Qwen models).
    approx_tokens = len(prompt) // 3
    if approx_tokens > ctx * 0.85:
        logger.warning(
            f"Prompt (~{approx_tokens} tokens) approaches context limit ({ctx}). "
            f"Model may truncate input — web search results or long conversations could be cut off."
        )

    # If we reach here, we're in normal mode — is_rag is always False.
    # (RAG mode returns early above with is_rag=True.)
    return prompt, False


def drain_stdout(proc, timeout=0.5):
    """Drain any buffered stdout from the process before sending a new prompt.

    Reads in a loop until the pipe has been quiet for 'timeout' seconds.
    Increased buffer size (8 KB) and default timeout (0.5s) to ensure all
    post-generation output (stats, prompt indicators, late flushes) is fully
    consumed before we send the next prompt.
    """
    drained = 0
    while True:
        try:
            ready, _, _ = select.select([proc.stdout], [], [], timeout)
        except (ValueError, OSError):
            break
        if ready:
            try:
                data = os.read(proc.stdout.fileno(), 8192)
                if data:
                    drained += len(data)
                else:
                    break
            except Exception:
                break
        else:
            break
    if drained:
        logger.debug(f"Drained {drained} bytes from stdout")
    return drained


# =============================================================================
# PROCESS MANAGEMENT
# =============================================================================


def is_process_healthy():
    """Check if the current rkllm process is alive."""
    if not CURRENT_PROCESS:
        return False
    proc = CURRENT_PROCESS.get('process')
    if not proc:
        return False
    return proc.poll() is None


def unload_current(reason="requested"):
    """Kill and clean up the current rkllm process."""
    global CURRENT_PROCESS, CURRENT_MODEL
    with PROCESS_LOCK:
        if not CURRENT_PROCESS:
            logger.debug("No process to unload - skipping")
            CURRENT_MODEL = None
            return

        pid = CURRENT_PROCESS['process'].pid
        logger.info(f">>> START UNLOADING {CURRENT_MODEL} (PID: {pid}) - reason: {reason}")

        proc = CURRENT_PROCESS['process']
        unload_start = time.time()

        # Kill process FIRST — before closing pipes.
        # With gevent, closing a pipe fd that libev is still watching via
        # epoll triggers an assertion crash:
        #   "libev: I/O watcher with invalid fd found in epoll_ctl"
        # Killing the process first ensures no I/O watchers are active.
        try:
            proc.kill()
            logger.debug("SIGKILL sent")
            proc.wait(timeout=5)
            logger.info("Process killed with SIGKILL")
        except subprocess.TimeoutExpired:
            logger.error("SIGKILL timed out - process may be zombie")
        except Exception as e:
            logger.error(f"SIGKILL failed: {e}")

        # Close pipes AFTER kill (safe — no active gevent watchers)
        for pipe_name, pipe in [("stdin", proc.stdin), ("stdout", proc.stdout)]:
            try:
                if pipe and not pipe.closed:
                    pipe.close()
                    logger.debug(f"{pipe_name} closed")
            except Exception as e:
                logger.debug(f"{pipe_name} close error (ignored): {e}")

        final_poll = proc.poll()
        if final_poll is None:
            logger.error(f"UNLOAD FAILED - Process (PID {pid}) STILL ALIVE!")
        else:
            logger.info(f"UNLOAD SUCCESS - exited with code {final_poll}")

        CURRENT_PROCESS = None
        CURRENT_MODEL = None
        logger.info(f">>> UNLOAD COMPLETE in {time.time() - unload_start:.1f}s - current_model now None")


def load_model(model_name, config):
    """Load an rkllm model subprocess. Reuses existing process if same model."""
    global CURRENT_PROCESS, CURRENT_MODEL

    # If a previous generator greenlet is still alive (stuck in gevent
    # select/yield), it holds the stdout fd.  Kill the process now so
    # the next load gets a clean pipe.
    # Checked OUTSIDE PROCESS_LOCK — the generator's finally block may
    # need PROCESS_LOCK for its own unload_current() call.
    if not GENERATION_COMPLETE.is_set():
        logger.warning("Previous generator still active — killing process for clean state")
        with PROCESS_LOCK:
            unload_current("stale generator")

    with PROCESS_LOCK:
        logger.info(f"load_model called for {model_name} (current loaded: {CURRENT_MODEL})")

        # Unload different model if needed
        if CURRENT_MODEL and CURRENT_MODEL != model_name:
            logger.info(f"Model switch detected: unloading {CURRENT_MODEL}")
            unload_current("model switch")

        # Reuse existing process if healthy and stdout is usable
        if CURRENT_MODEL == model_name and CURRENT_PROCESS and is_process_healthy():
            # Verify stdout is still functional before reusing.
            # With gevent, a previous streaming generator can leave stdout
            # in a broken state (EOF / closed fd) even though the process
            # is technically alive (poll() returns None).
            proc = CURRENT_PROCESS['process']
            stdout_ok = True
            try:
                ready, _, _ = select.select([proc.stdout], [], [], 0)
                if ready:
                    peek = os.read(proc.stdout.fileno(), 8192)
                    if not peek:
                        stdout_ok = False
                        logger.warning("Stdout returned EOF during reuse check")
                    else:
                        logger.debug(f"Drained {len(peek)} leftover bytes during reuse check")
            except (OSError, ValueError) as e:
                stdout_ok = False
                logger.warning(f"Stdout broken during reuse check: {e}")

            if stdout_ok:
                logger.info(f"REUSING process for {model_name} (PID: {proc.pid})")
                return CURRENT_PROCESS
            else:
                unload_current("stdout broken on reuse")
                # Fall through to fresh load

        # Fresh load
        model_path = config['path']
        max_tokens = config.get('max_tokens', MAX_TOKENS_DEFAULT)
        context_length = config.get('context_length', CONTEXT_LENGTH_DEFAULT)

        cmd = [RKLLM_BINARY, model_path, str(max_tokens), str(context_length)]
        logger.info(f"STARTING LOAD of new process for {model_name}...")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            # Override RKLLM_LOG_LEVEL for the subprocess so the rkllm C runtime
            # prints performance stats ([Token/s], [Tokens], [Seconds]) after
            # each generation.  Level 1 = stats only.  The API's own Python
            # logging reads RKLLM_API_LOG_LEVEL instead (see logging section).
            rkllm_env = os.environ.copy()
            rkllm_env['RKLLM_LOG_LEVEL'] = '1'
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=rkllm_env,
            )
        except FileNotFoundError:
            logger.error(f"rkllm binary '{RKLLM_BINARY}' not found in PATH")
            return None
        except Exception as e:
            logger.error(f"Failed to start rkllm process: {e}")
            return None

        logger.info(f"Process started - PID: {proc.pid}")

        # Wait for init success
        init_start = time.time()
        init_success = False

        while time.time() - init_start < MODEL_LOAD_TIMEOUT:
            if proc.poll() is not None:
                logger.error(f"Process died during init (exit code: {proc.returncode})")
                break

            try:
                ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            except (ValueError, OSError):
                break

            if ready:
                try:
                    line = proc.stdout.readline()
                    if not line:
                        logger.error("EOF during init - process may have crashed")
                        break
                    text = line.decode('utf-8', errors='replace').rstrip()
                    logger.debug(f"Init output: {text}")
                    if 'init success' in text.lower():
                        init_success = True
                        break
                    if 'init failed' in text.lower():
                        logger.error(f"rkllm reported init failure for {model_name} — model file may be corrupt or incompatible")
                        break
                except Exception as e:
                    logger.error(f"Error reading init output: {e}")
                    break

        if not init_success:
            logger.error(f"Model {model_name} failed to initialize (check model file integrity and rkllm version compatibility)")
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
            CURRENT_PROCESS = None
            CURRENT_MODEL = None
            return None

        elapsed = time.time() - init_start
        logger.info(f"{model_name} LOADED successfully in {elapsed:.1f}s (PID: {proc.pid}, ctx={context_length})")

        CURRENT_PROCESS = {
            'process': proc,
            'model': model_name,
            'config': config,
            'context_length': context_length,
        }
        CURRENT_MODEL = model_name

        return CURRENT_PROCESS


# =============================================================================
# PROCESS MONITOR
# =============================================================================


def process_monitor():
    """Background thread to monitor process health and clean up orphans."""
    logger.info("Process monitor started")
    while not SHUTDOWN_EVENT.is_set():
        try:
            force_clear_if_orphaned()

            if CURRENT_PROCESS and not is_process_healthy():
                logger.warning(f"Process for {CURRENT_MODEL} died unexpectedly - cleaning up")
                unload_current("process died")

            # Auto-unload model after idle timeout to free NPU memory.
            # LAST_REQUEST_TIME is read without ACTIVE_LOCK here which is safe:
            # CPython's GIL makes float reads/writes atomic, and the worst case
            # (stale value) only delays unload by one monitor cycle.
            elif (IDLE_UNLOAD_TIMEOUT > 0 and CURRENT_PROCESS and is_process_healthy()
                  and not is_request_active() and LAST_REQUEST_TIME > 0
                  and (time.time() - LAST_REQUEST_TIME) > IDLE_UNLOAD_TIMEOUT):
                logger.info(f"Auto-unloading {CURRENT_MODEL} after "
                            f"{int(time.time() - LAST_REQUEST_TIME)}s idle")
                unload_current("idle timeout")
        except Exception as e:
            logger.error(f"Monitor error: {e}")

        SHUTDOWN_EVENT.wait(MONITOR_INTERVAL)
    logger.info("Process monitor stopped")


# Start monitor thread
_monitor_thread = Thread(target=process_monitor, daemon=True)
_monitor_thread.start()


# =============================================================================
# SHUTDOWN
# =============================================================================


def shutdown():
    """Clean shutdown - kill process, stop monitor."""
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


# NOTE: Signal handlers are registered in the __main__ block below.
# Under gunicorn, the arbiter manages SIGTERM/SIGINT for workers;
# overriding them at import time would conflict with worker lifecycle.


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
    # Per OpenAI spec: the usage-only final chunk has "choices": []
    # (empty array), not a regular choices entry with empty delta.
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
    body = request.get_json(silent=True) or {}
    model_name = body.get('model', '')

    name, config = resolve_model(model_name)
    if config is None:
        return make_error_response(f"Model '{model_name}' not found", 404, "not_found")

    result = load_model(name, config)
    if result is None:
        return make_error_response(f"Failed to load model '{name}'", 500)

    return jsonify({"status": "ok", "model": name, "pid": result['process'].pid})


@app.route('/v1/models/unload', methods=['POST'])
def unload_model():
    """Explicitly unload the current model to free NPU memory."""
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
        "process_healthy": is_process_healthy(),
        "active_request": active,
        "models_available": len(MODELS),
    })


@app.route('/v1/chat/completions', methods=['POST'])
@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint (streaming and non-streaming)."""
    body = request.get_json(silent=True) or {}

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    logger.info(f">>> NEW REQUEST {request_id}")

    # Parse request
    requested_model = body.get('model', '')
    messages = body.get('messages', [])
    stream = body.get('stream', False)
    stream_options = body.get('stream_options') or {}
    include_usage = stream_options.get('include_usage', False) if stream else False
    max_tokens = body.get('max_tokens') or body.get('max_completion_tokens') or MAX_TOKENS_DEFAULT

    logger.info(f"Request {request_id} model: '{requested_model}' stream: {stream}")

    # === DIAGNOSTIC: Dump all messages from Open WebUI ===
    for i, msg in enumerate(messages):
        role = msg.get('role', '?')
        content = msg.get('content', '')
        preview = content[:500] if len(content) <= 500 else content[:500] + f"... [{len(content)} chars total]"
        logger.info(f"[{request_id}] MSG[{i}] role={role} len={len(content)}: {preview}")
    # === END DIAGNOSTIC ===

    if not messages:
        return make_error_response("No messages provided", 400, "invalid_request")

    # Resolve model
    name, config = resolve_model(requested_model)
    logger.info(f"Resolved '{requested_model}' -> '{name}'. Current: {CURRENT_MODEL}")

    if config is None:
        return make_error_response(f"Model '{requested_model}' not found", 404, "not_found")

    # Atomic check-and-set: reject if another generation is in progress.
    # NPU is single-task — only one request can run at a time.
    if not try_start_request(request_id, name):
        logger.warning(f"[{request_id}] Rejected - request already in progress")
        return make_error_response(
            "Another request is currently being processed. Please retry shortly.",
            503, "service_unavailable"
        )

    # Override max_tokens if provided
    effective_config = dict(config)
    try:
        max_tok = int(max_tokens)
        if max_tok != MAX_TOKENS_DEFAULT:
            effective_config['max_tokens'] = min(max_tok, config['context_length'])
    except (ValueError, TypeError):
        pass  # Keep default if max_tokens is non-numeric

    # try_start_request() already set ACTIVE_REQUEST above
    ABORT_EVENT.clear()
    created = int(time.time())

    try:
        # Load model
        logger.info(f"Loading model {name} for request {request_id}")
        proc_info = load_model(name, effective_config)
        if proc_info is None:
            end_request(request_id)
            return make_error_response(f"Failed to load model '{name}'", 500)
        update_request_activity()  # Model loaded — refresh stale timer

        proc = proc_info['process']

        # Build prompt
        prompt, is_rag = build_prompt(messages, name)

        # Extract user question for cache key (last user message)
        user_question = ""
        for msg in reversed(messages):
            if msg.get('role') == 'user' and msg.get('content'):
                user_question = msg['content']
                break

        # === RAG Response Cache Check ===
        # If this is a RAG query and we have a cached response for the same
        # model+question, return it immediately without NPU inference.
        if is_rag and RAG_CACHE_TTL > 0:
            cached = _rag_cache_get(name, user_question)
            if cached:
                cached_prompt, cached_response = cached
                logger.info(f"[{request_id}] RAG cache HIT for '{user_question[:60]}' "
                            f"({len(cached_response)} chars)")
                end_request(request_id)
                created = int(time.time())
                prompt_tokens = max(1, len(cached_prompt) // 3)
                completion_tokens = max(1, len(cached_response) // 3)
                reasoning_content, cleaned_content = parse_think_tags(cached_response)
                if stream:
                    # Synthesize SSE stream from cached response using make_sse_chunk
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

        # Drain any buffered output (stats lines, prompt indicators from previous turn)
        drain_stdout(proc, timeout=0.5)

        # Health-check: ensure the process didn't die during drain
        if proc.poll() is not None:
            logger.error(f"[{request_id}] Model process exited (code {proc.returncode}) after drain")
            end_request(request_id)
            unload_current("process died after drain")
            return make_error_response("Model process died unexpectedly", 500)

        # Send prompt
        # The C++ binary reads input via std::getline (splits on newlines).
        # With USE_MULTILINE_PROTOCOL=True (requires recompiled binary):
        #   Send prompt as-is with \n__END__\n delimiter — preserves paragraph
        #   structure in RAG prompts for better model comprehension.
        # With USE_MULTILINE_PROTOCOL=False (stock binary):
        #   Flatten newlines to spaces — prevents duplicate inference turns
        #   but loses paragraph formatting.
        try:
            if USE_MULTILINE_PROTOCOL:
                prompt_bytes = (prompt + "\n__END__\n").encode('utf-8')
            else:
                flat_prompt = prompt.replace('\n', ' ').replace('\r', ' ')
                prompt_bytes = (flat_prompt + "\n").encode('utf-8')
            proc.stdin.write(prompt_bytes)
            proc.stdin.flush()
            update_request_activity()  # Prompt sent — refresh stale timer
            logger.info(f"Sending prompt to {name} for request {request_id}")
        except (BrokenPipeError, OSError) as e:
            logger.error(f"[{request_id}] Failed to send prompt: {e}")
            end_request(request_id)
            unload_current("broken pipe")
            return make_error_response("Model process died", 500)

        # Prepare cache info for RAG queries (store result after generation)
        rag_cache_info = (name, user_question, prompt) if is_rag and RAG_CACHE_TTL > 0 else None

        if stream:
            return Response(
                stream_with_context(_generate_stream(proc, request_id, name, created,
                                                     include_usage=include_usage, messages=messages,
                                                     is_rag=is_rag, rag_cache_info=rag_cache_info)),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                    'Connection': 'keep-alive',
                },
            )
        else:
            return _generate_complete(proc, request_id, name, created, is_rag=is_rag, messages=messages,
                                      rag_cache_info=rag_cache_info)

    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        end_request(request_id)
        return make_error_response(f"Internal error: {e}", 500)


# =============================================================================
# THINK TAG PARSER
# =============================================================================


class ThinkTagParser:
    """State machine to parse <think>...</think> tags from a token stream.

    Reasoning models (Qwen3, DeepSeek-R1) output chain-of-thought wrapped
    in <think>...</think> tags.  Open WebUI displays `reasoning_content`
    as collapsible thinking blocks when returned in the streaming delta.

    The parser buffers incoming text to handle tags split across chunks:
    - While buffering, checks if the buffer could be a prefix of <think> or </think>
    - Once a full tag is detected, switches state accordingly
    - Emits (kind, text) tuples: kind is 'thinking' or 'content'
    """

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self):
        self.in_thinking = False
        self.buffer = ""
        self.saw_think_block = False  # True once any <think> was found

    def feed(self, text):
        """Feed text into the parser. Yields (kind, text) tuples.

        kind: 'thinking' for text inside <think> tags,
              'content' for text outside <think> tags.
        """
        self.buffer += text

        while self.buffer:
            if self.in_thinking:
                # Look for </think> in buffer
                close_pos = self.buffer.find(self.CLOSE_TAG)
                if close_pos != -1:
                    # Emit everything before the close tag as thinking
                    thinking_text = self.buffer[:close_pos]
                    if thinking_text:
                        yield ('thinking', thinking_text)
                    self.buffer = self.buffer[close_pos + len(self.CLOSE_TAG):]
                    self.in_thinking = False
                    # Strip a single leading newline after </think>
                    if self.buffer.startswith('\n'):
                        self.buffer = self.buffer[1:]
                    continue
                else:
                    # Could the end of buffer be a partial </think>?
                    # Check if any suffix of buffer is a prefix of CLOSE_TAG
                    safe = len(self.buffer)
                    for i in range(1, len(self.CLOSE_TAG)):
                        if self.buffer.endswith(self.CLOSE_TAG[:i]):
                            safe = len(self.buffer) - i
                            break
                    if safe > 0:
                        yield ('thinking', self.buffer[:safe])
                        self.buffer = self.buffer[safe:]
                    # Either buffer is now a partial tag or empty — wait for more
                    break
            else:
                # Look for <think> in buffer
                open_pos = self.buffer.find(self.OPEN_TAG)
                if open_pos != -1:
                    # Emit everything before the open tag as content
                    content_text = self.buffer[:open_pos]
                    if content_text:
                        yield ('content', content_text)
                    self.buffer = self.buffer[open_pos + len(self.OPEN_TAG):]
                    self.in_thinking = True
                    self.saw_think_block = True
                    # Strip a single leading newline after <think>
                    if self.buffer.startswith('\n'):
                        self.buffer = self.buffer[1:]
                    continue
                else:
                    # Could the end of buffer be a partial <think>?
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
    """Parse <think>...</think> from complete text. Returns (reasoning_content, content)."""
    pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    thinking_parts = pattern.findall(text)
    content = pattern.sub('', text).strip()
    reasoning = '\n'.join(part.strip() for part in thinking_parts if part.strip())
    return (reasoning if reasoning else None, content)


# =============================================================================
# GENERATION
# =============================================================================


def _generate_stream(proc, request_id, model_name, created, include_usage=False, messages=None, is_rag=False, rag_cache_info=None):
    """Generator that yields SSE chunks from rkllm output."""
    logger.info(f"[{request_id}] Starting STREAMING generation (rag={is_rag})")
    GENERATION_COMPLETE.clear()

    # First chunk: role
    yield make_sse_chunk(request_id, model_name, created, delta={"role": "assistant"})

    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    line_buffer = ""
    last_activity = time.time()
    generation_start = time.time()
    got_first_token = False
    prefix_stripped = False
    total_content = ""
    total_reasoning = ""
    finished = False
    generation_clean = False  # True when stats detected — never reset by except
    request_ended = False     # True once end_request() has been called
    process_killed = False    # True once unload_current() has been called
    stats_data = {}           # Parsed rkllm stats: completion_tokens, tokens_per_sec, seconds
    think_parser = ThinkTagParser()  # State machine for <think>...</think> parsing

    try:
        while not finished:
            # Check abort
            if ABORT_EVENT.is_set():
                logger.info(f"[{request_id}] Abort signal received")
                break

            # Check overall timeout
            if time.time() - generation_start > GENERATION_TIMEOUT:
                logger.warning(f"[{request_id}] Generation timeout ({GENERATION_TIMEOUT}s)")
                break

            # Check process health
            if proc.poll() is not None:
                logger.warning(f"[{request_id}] Process exited during generation")
                break

            # Wait for data
            try:
                ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            except (ValueError, OSError):
                logger.warning(f"[{request_id}] select() failed - stdout closed")
                break

            if ready:
                try:
                    raw = os.read(proc.stdout.fileno(), 4096)
                except (OSError, ValueError):
                    logger.warning(f"[{request_id}] os.read failed - stdout closed")
                    break

                if not raw:
                    logger.info(f"[{request_id}] EOF on stdout")
                    break

                text = decoder.decode(raw)
                if not text:
                    continue

                # Guard: discard residual data from previous generation.
                # Real model output can't arrive this fast — NPU prefill
                # takes 500ms+ even for short prompts.  Data arriving
                # sooner is leftover from the previous turn's stdout.
                # Exception: if the text contains '<think>', it's real model
                # output (thinking mode) even if it arrives quickly after the
                # prompt prefix (e.g. "robot: <think>" can arrive in one burst).
                if not got_first_token and (time.time() - generation_start) < MINIMUM_PREFILL_TIME:
                    if '<think>' not in text:
                        logger.debug(f"[{request_id}] Discarding residual data ({len(raw)} bytes): {text[:80]!r}")
                        line_buffer = ""  # Clear any accumulated residual
                        last_activity = time.time()
                        continue
                    else:
                        logger.debug(f"[{request_id}] Early data contains <think> — treating as real output")

                last_activity = time.time()
                update_request_activity()
                line_buffer += text

                # Process complete lines
                while '\n' in line_buffer:
                    line, line_buffer = line_buffer.split('\n', 1)

                    if is_stats_line(line):
                        logger.debug(f"[{request_id}] Stats: {line.strip()}")
                        _parse_stats_line(line, stats_data)
                        generation_clean = True
                        continue  # Collect all stats lines from buffer
                    elif generation_clean:
                        # Post-stats line (e.g. "user:" prompt) — done
                        finished = True
                        break
                    elif is_noise_line(line):
                        # Standalone user:/robot: prompt = binary waiting for next input.
                        # Treat as end-of-generation fallback when stats weren't printed.
                        s = line.strip()
                        if got_first_token and s and re.match(r'^(user|robot)\s*:\s*$', s, re.IGNORECASE):
                            logger.debug(f"[{request_id}] Prompt indicator '{s}' = generation done (no stats)")
                            generation_clean = True
                            finished = True
                            break
                        logger.debug(f"[{request_id}] Noise: {line.strip()}")
                        continue
                    else:
                        # Real content line - include the newline
                        content = clean_line(line + '\n')
                        if content:
                            # Strip rkllm role prefix from first output line.
                            # Handles: "robot: ", "LLM: ", "user: robot: "
                            # (compound prefix appears on reused processes).
                            if not prefix_stripped:
                                new_content = re.sub(r'^(user:\s*)?(LLM|robot):\s*', '', content)
                                if new_content != content:
                                    content = new_content
                                    prefix_stripped = True
                            if not got_first_token:
                                got_first_token = True
                                logger.debug(f"[{request_id}] First token in {time.time() - generation_start:.2f}s")
                            # Route through think-tag parser
                            for kind, chunk_text in think_parser.feed(content):
                                if kind == 'thinking':
                                    total_reasoning += chunk_text
                                    # Only send non-whitespace reasoning (suppresses
                                    # empty <think></think> from /no_think mode)
                                    if chunk_text.strip():
                                        yield make_sse_chunk(request_id, model_name, created,
                                                             delta={"reasoning_content": chunk_text})
                                else:
                                    total_content += chunk_text
                                    yield make_sse_chunk(request_id, model_name, created,
                                                         delta={"content": chunk_text})

                # All stats consumed from buffer — generation complete
                if generation_clean and not finished:
                    finished = True

                if finished:
                    break

                # Send partial line buffer for streaming responsiveness
                if line_buffer:
                    stripped = line_buffer.strip()
                    # Prompt indicator in partial buffer (no trailing \n) = generation done
                    if got_first_token and stripped and \
                       re.match(r'^(user|robot)\s*:\s*$', stripped, re.IGNORECASE):
                        logger.debug(f"[{request_id}] Prompt indicator '{stripped}' in buffer = generation done")
                        generation_clean = True
                        finished = True
                        line_buffer = ""
                        break
                    if stripped and not is_noise_line(stripped) and \
                       not is_stats_line(stripped) and \
                       not stripped.startswith(('I rkllm', 'W rkllm', 'E rkllm', '[Token', '[Tokens', '[Seconds')) and \
                       not re.match(r'^\[(?:Tok|Sec)', stripped):
                        content = clean_line(line_buffer)
                        if content:
                            if not prefix_stripped:
                                new_content = re.sub(r'^(user:\s*)?(LLM|robot):\s*', '', content)
                                if new_content != content:
                                    content = new_content
                                    prefix_stripped = True
                            if not got_first_token:
                                got_first_token = True
                                logger.debug(f"[{request_id}] First token in {time.time() - generation_start:.2f}s")
                            # Route through think-tag parser
                            for kind, chunk_text in think_parser.feed(content):
                                if kind == 'thinking':
                                    total_reasoning += chunk_text
                                    if chunk_text.strip():
                                        yield make_sse_chunk(request_id, model_name, created,
                                                             delta={"reasoning_content": chunk_text})
                                else:
                                    total_content += chunk_text
                                    yield make_sse_chunk(request_id, model_name, created,
                                                         delta={"content": chunk_text})
                        line_buffer = ""

            else:
                # No data available - check silence timeout
                silence = time.time() - last_activity
                effective_timeout = FIRST_TOKEN_TIMEOUT if not got_first_token else FALLBACK_SILENCE

                if silence > effective_timeout:
                    # Before giving up, check partial buffer for prompt indicator
                    if got_first_token and line_buffer.strip():
                        s = line_buffer.strip()
                        if re.match(r'^(user|robot)\s*:\s*$', s, re.IGNORECASE):
                            logger.debug(f"[{request_id}] Prompt indicator '{s}' in buffer at timeout = done")
                            generation_clean = True
                            finished = True
                            break
                    label = "first token" if not got_first_token else "silence"
                    logger.warning(f"[{request_id}] Timeout waiting for {label} ({silence:.1f}s)")
                    break

        # End request tracking IMMEDIATELY — don't defer to finally.
        # With gevent, a streaming generator's finally block may not run
        # until garbage collection (minutes later), leaving the request
        # tracker blocked and stdout fd watched by stale gevent watchers.
        end_request(request_id)
        request_ended = True

        # If generation wasn't clean, kill process now to prevent bleed-through
        if not generation_clean:
            logger.warning(f"[{request_id}] Unclean end - killing process to prevent bleed-through")
            unload_current("generation interrupted")
            process_killed = True
        elif is_rag:
            # RAG requests get different reference data each time.
            # Kill the process to clear rkllm's KV cache and prevent
            # the previous conversation from contaminating the next response.
            logger.info(f"[{request_id}] Killing process after RAG to clear KV cache")
            unload_current("RAG KV cache clear")
            process_killed = True

        # Flush remaining data only if generation didn't end cleanly via stats
        if not finished:
            final_text = decoder.decode(b'', final=True)
            if final_text and not is_noise_line(final_text) and not is_stats_line(final_text):
                content = clean_line(final_text)
                if content:
                    for kind, chunk_text in think_parser.feed(content):
                        if kind == 'thinking':
                            total_reasoning += chunk_text
                            if chunk_text.strip():
                                yield make_sse_chunk(request_id, model_name, created,
                                                     delta={"reasoning_content": chunk_text})
                        else:
                            total_content += chunk_text
                            yield make_sse_chunk(request_id, model_name, created,
                                                 delta={"content": chunk_text})

            if line_buffer.strip() and not is_noise_line(line_buffer) and not is_stats_line(line_buffer):
                content = clean_line(line_buffer)
                if content:
                    for kind, chunk_text in think_parser.feed(content):
                        if kind == 'thinking':
                            total_reasoning += chunk_text
                            if chunk_text.strip():
                                yield make_sse_chunk(request_id, model_name, created,
                                                     delta={"reasoning_content": chunk_text})
                        else:
                            total_content += chunk_text
                            yield make_sse_chunk(request_id, model_name, created,
                                                 delta={"content": chunk_text})

        # Flush any remaining text in the think-tag parser buffer
        flushed = think_parser.flush()
        if flushed:
            kind, flush_text = flushed
            # Strip trailing role-prefix fragments the model may generate
            # as continuation (e.g. "heritage.\nuser:" or "heritage. ua").
            flush_text = re.sub(r'\s*\b(?:user|robot)\s*:?\s*$', '', flush_text)
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

        # stream_options.include_usage — send a final chunk with token counts
        if include_usage:
            prompt_text = "".join(m.get("content", "") for m in (messages or []) if isinstance(m.get("content"), str))
            prompt_tokens = max(1, len(prompt_text) // 3)
            # Fallback token count includes both content and reasoning tokens
            total_output_chars = len(total_content) + len(total_reasoning)
            completion_tokens = stats_data.get('completion_tokens', max(1, total_output_chars // 3))
            yield make_sse_chunk(request_id, model_name, created, usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            })

        yield "data: [DONE]\n\n"

        # Store in RAG cache if generation was clean
        if rag_cache_info and generation_clean and total_content:
            cache_response = total_content
            if total_reasoning:
                cache_response = f"<think>{total_reasoning}</think>{total_content}"
            _model, _question, _prompt = rag_cache_info
            _rag_cache_put(_model, _question, _prompt, cache_response)
            logger.info(f"[{request_id}] RAG cache STORE ({len(cache_response)} chars)")

    except GeneratorExit:
        logger.warning(f"[{request_id}] Client DISCONNECTED / stopped")
    except Exception as e:
        logger.error(f"[{request_id}] Stream error: {e}", exc_info=True)
        try:
            yield make_sse_chunk(request_id, model_name, created, finish_reason="stop")
            yield "data: [DONE]\n\n"
        except Exception:
            pass
    finally:
        # Safety net — handles exceptions that fire before inline cleanup
        # (e.g. GeneratorExit during yields, errors during drain/flush)
        if not request_ended:
            end_request(request_id)
        if not generation_clean and not process_killed:
            logger.warning(f"[{request_id}] Unclean end - killing process to prevent bleed-through")
            unload_current("generation interrupted")
        GENERATION_COMPLETE.set()
        if total_reasoning:
            logger.info(f"[{request_id}] Stream ended ({len(total_content)} content + {len(total_reasoning)} reasoning chars)")
        else:
            logger.info(f"[{request_id}] Stream ended ({len(total_content)} chars total)")


def _generate_complete(proc, request_id, model_name, created, is_rag=False, messages=None, rag_cache_info=None):
    """Collect all output and return a non-streaming response."""
    logger.info(f"[{request_id}] Starting NON-STREAMING generation (rag={is_rag})")
    GENERATION_COMPLETE.clear()

    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    line_buffer = ""
    last_activity = time.time()
    generation_start = time.time()
    got_first_token = False
    prefix_stripped = False
    content_parts = []
    finished = False
    generation_clean = False
    stats_data = {}

    try:
        while not finished:
            if ABORT_EVENT.is_set():
                break

            if time.time() - generation_start > GENERATION_TIMEOUT:
                logger.warning(f"[{request_id}] Generation timeout")
                break

            if proc.poll() is not None:
                logger.warning(f"[{request_id}] Process exited during generation")
                break

            try:
                ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            except (ValueError, OSError):
                break

            if ready:
                try:
                    raw = os.read(proc.stdout.fileno(), 4096)
                except (OSError, ValueError):
                    break

                if not raw:
                    break

                text = decoder.decode(raw)
                if not text:
                    continue

                # Guard: discard residual data from previous generation
                if not got_first_token and (time.time() - generation_start) < MINIMUM_PREFILL_TIME:
                    if '<think>' not in text:
                        logger.debug(f"[{request_id}] Discarding residual data ({len(raw)} bytes)")
                        line_buffer = ""
                        last_activity = time.time()
                        continue
                    else:
                        logger.debug(f"[{request_id}] Early data contains <think> — treating as real output")

                last_activity = time.time()
                update_request_activity()
                line_buffer += text

                # Process complete lines
                while '\n' in line_buffer:
                    line, line_buffer = line_buffer.split('\n', 1)

                    if is_stats_line(line):
                        logger.debug(f"[{request_id}] Stats: {line.strip()}")
                        _parse_stats_line(line, stats_data)
                        generation_clean = True
                        continue
                    elif generation_clean:
                        finished = True
                        break
                    elif is_noise_line(line):
                        # Standalone user:/robot: prompt = binary waiting for next input.
                        # Treat as end-of-generation fallback when stats weren't printed.
                        s = line.strip()
                        if got_first_token and s and re.match(r'^(user|robot)\s*:\s*$', s, re.IGNORECASE):
                            generation_clean = True
                            finished = True
                            break
                        continue
                    else:
                        content = clean_line(line + '\n')
                        if content:
                            if not prefix_stripped:
                                new_content = re.sub(r'^(user:\s*)?(LLM|robot):\s*', '', content)
                                if new_content != content:
                                    content = new_content
                                    prefix_stripped = True
                            if not got_first_token:
                                got_first_token = True
                            content_parts.append(content)

                if generation_clean and not finished:
                    finished = True

            else:
                silence = time.time() - last_activity
                effective_timeout = FIRST_TOKEN_TIMEOUT if not got_first_token else FALLBACK_SILENCE
                if silence > effective_timeout:
                    # Before giving up, check partial buffer for prompt indicator
                    if got_first_token and line_buffer.strip():
                        s = line_buffer.strip()
                        if re.match(r'^(user|robot)\s*:\s*$', s, re.IGNORECASE):
                            generation_clean = True
                            finished = True
                            break
                    logger.warning(f"[{request_id}] Silence timeout")
                    break

        # Flush remaining data only if generation didn't end cleanly via stats
        if not finished:
            final_text = decoder.decode(b'', final=True)
            if final_text and not is_noise_line(final_text) and not is_stats_line(final_text):
                content_parts.append(clean_line(final_text))

            if line_buffer.strip() and not is_noise_line(line_buffer) and not is_stats_line(line_buffer):
                content_parts.append(clean_line(line_buffer))

        full_content = "".join(content_parts).rstrip()

        # Strip trailing role-prefix fragments the model may generate
        # as continuation (e.g. "heritage.\nuser:" or "heritage. ua").
        full_content = re.sub(r'\s*\b(?:user|robot)\s*:?\s*$', '', full_content)

        # Parse <think>...</think> tags from reasoning models (Qwen3, DeepSeek-R1)
        reasoning_content, cleaned_content = parse_think_tags(full_content)

        # Token counts: use real stats if available, otherwise approximate (~3 chars/token)
        prompt_text = "".join(m.get("content", "") for m in (messages or []) if isinstance(m.get("content"), str))
        prompt_tokens = max(1, len(prompt_text) // 3)
        completion_tokens = stats_data.get('completion_tokens', max(1, len(full_content) // 3))

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
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        # Store in RAG cache if generation was clean
        if rag_cache_info and generation_clean and full_content:
            _model, _question, _prompt = rag_cache_info
            _rag_cache_put(_model, _question, _prompt, full_content)
            logger.info(f"[{request_id}] RAG cache STORE ({len(full_content)} chars)")

        if reasoning_content:
            logger.info(f"[{request_id}] Completed ({len(cleaned_content)} content + {len(reasoning_content)} reasoning chars)")
        else:
            logger.info(f"[{request_id}] Completed ({len(cleaned_content)} chars)")
        return jsonify(response)

    except Exception as e:
        logger.error(f"[{request_id}] Generation error: {e}", exc_info=True)
        return make_error_response(f"Generation failed: {e}", 500)
    finally:
        end_request(request_id)
        if not generation_clean:
            logger.warning(f"[{request_id}] Unclean end - killing process to prevent bleed-through")
            unload_current("generation interrupted")
        elif is_rag:
            logger.info(f"[{request_id}] Killing process after RAG to clear KV cache")
            unload_current("RAG KV cache clear")
        GENERATION_COMPLETE.set()


# =============================================================================
# STARTUP
# =============================================================================

if __name__ == "__main__":
    # Register signal handlers only when running standalone (not under gunicorn)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, signal_handler)
        except (OSError, ValueError):
            pass

    logger.info("=" * 60)
    logger.info("RKLLM API starting (Open WebUI managed)")
    logger.info(f"  Models : {len(MODELS)} detected")
    logger.info(f"  Aliases: {len(ALIASES)} generated")
    logger.info(f"  Port   : 8000")
    logger.info("=" * 60)
    logger.info(f"Models: {list(MODELS.keys())}")
    logger.info(f"Aliases: {ALIASES}")
    # SECURITY NOTE: This server binds to 0.0.0.0 with NO authentication.
    # It is intended to run on a trusted local network behind a firewall or
    # reverse-proxy.  Do NOT expose directly to the public internet.
    app.run(host="0.0.0.0", port=8000, threaded=True)
