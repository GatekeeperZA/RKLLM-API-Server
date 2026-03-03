#!/usr/bin/env python3
"""
Real-World Smoke Tests — Ad-hoc verification of the live RKLLM API
===================================================================
NOT part of the automated test suite. These are one-off, human-readable
checks that simulate actual user interactions.

Tests:
  1. Health & model listing
  2. Simple factual Q&A (non-streaming)
  3. Creative writing (streaming SSE)
  4. Reasoning / <think> blocks (Qwen3)
  5. Multi-turn conversation (KV cache incremental)
  6. RAG: document comprehension with <source> tags
  7. Shortcircuit: search query generation
  8. Shortcircuit: title generation
  9. Shortcircuit: tag generation
 10. Date awareness (system prompt injection)
 11. Home Assistant style request
 12. Model switching mid-session
 13. Concurrent request rejection
 14. Long output generation
 15. Error handling (bad model, empty messages)

Usage:
    python tests/realworld_smoke.py
"""
import json, os, re, sys, time, threading, urllib.request, urllib.error

API = os.environ.get("RKLLM_API", "http://192.168.2.180:8000")
PASS = 0
FAIL = 0
TIMEOUT = 180


def req(method, path, body=None, timeout=TIMEOUT):
    """HTTP request helper. Returns (status, data_dict, raw_str)."""
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw), raw
            except json.JSONDecodeError:
                return resp.status, None, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else ""
        try:
            return e.code, json.loads(raw), raw
        except Exception:
            return e.code, None, raw
    except Exception as e:
        return 0, None, str(e)


def stream_chat(model, messages, timeout=TIMEOUT):
    """Send streaming chat, return dict with content, reasoning, chunks, timing."""
    url = f"{API}/v1/chat/completions"
    body = json.dumps({"model": model, "messages": messages, "stream": True,
                        "stream_options": {"include_usage": True}}).encode()
    r = urllib.request.Request(url, data=body, method="POST",
                               headers={"Content-Type": "application/json"})
    content = ""
    reasoning = ""
    chunks = 0
    has_done = False
    has_usage = False
    t0 = time.time()
    ttft = None
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    has_done = True
                    break
                try:
                    obj = json.loads(payload)
                    chunks += 1
                    if "usage" in obj:
                        has_usage = True
                    choices = obj.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        c = delta.get("content", "")
                        r_c = delta.get("reasoning_content", "")
                        if c:
                            if ttft is None:
                                ttft = time.time() - t0
                            content += c
                        if r_c:
                            if ttft is None:
                                ttft = time.time() - t0
                            reasoning += r_c
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        return {"error": str(e), "content": content, "reasoning": reasoning}

    elapsed = time.time() - t0
    return {
        "content": content, "reasoning": reasoning, "chunks": chunks,
        "has_done": has_done, "has_usage": has_usage,
        "ttft": ttft, "elapsed": elapsed,
    }


def check(name, condition, detail=""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    line = f"  [{status}] {name}"
    if detail:
        line += f"  |  {detail}"
    print(line)


def unload():
    req("POST", "/v1/models/unload")
    time.sleep(1)


# =========================================================================
print("=" * 65)
print("REAL-WORLD SMOKE TESTS")
print("=" * 65)
print(f"API:  {API}")
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# --- 1. Health & Models ---
print("[1] HEALTH & MODEL LISTING")
code, health, _ = req("GET", "/health", timeout=10)
check("Health endpoint reachable", code == 200, f"status={code}")
check("Health status=ok", (health or {}).get("status") == "ok")
check("Models available > 0", (health or {}).get("models_available", 0) > 0,
      f"count={(health or {}).get('models_available')}")

code, mdata, _ = req("GET", "/v1/models")
models = (mdata or {}).get("data", [])
model_ids = [m["id"] for m in models]
print(f"  Models: {model_ids}")
for m in models:
    print(f"    {m['id']:35s} caps={m.get('capabilities',[])}  ctx={m.get('context_length')}")
check("Model list has OpenAI structure",
      mdata and mdata.get("object") == "list" and len(models) > 0)

# Pick a fast thinking model
THINK_MODEL = "qwen3-1.7b" if "qwen3-1.7b" in model_ids else model_ids[0]
ALT_MODEL = "gemma-3-4b-it" if "gemma-3-4b-it" in model_ids else (
    [m for m in model_ids if m != THINK_MODEL] or [THINK_MODEL])[0]
print(f"  Primary model: {THINK_MODEL}  |  Alt model: {ALT_MODEL}")

# --- 2. Simple Q&A (non-streaming) ---
print(f"\n[2] SIMPLE Q&A — NON-STREAMING ({THINK_MODEL})")
code, data, _ = req("POST", "/v1/chat/completions", {
    "model": THINK_MODEL,
    "messages": [{"role": "user", "content": "What is the boiling point of water in Celsius? Answer in one sentence."}],
    "stream": False,
})
answer = ""
if data and "choices" in data:
    answer = data["choices"][0]["message"].get("content", "")
check("Non-streaming response 200", code == 200, f"status={code}")
check("Response has content", len(answer) > 5, f"len={len(answer)}")
check("Answer mentions 100", "100" in answer, f"'{answer[:120]}'")
check("Has usage stats", "usage" in (data or {}),
      f"tokens={data.get('usage')}" if data else "no data")
check("Has system_fingerprint", "system_fingerprint" in (data or {}))

# --- 3. Creative writing (streaming) ---
print(f"\n[3] CREATIVE WRITING — STREAMING ({THINK_MODEL})")
result = stream_chat(THINK_MODEL, [
    {"role": "user", "content": "Write a haiku about the ocean."}
])
check("Stream completed with [DONE]", result.get("has_done", False))
total_output = len(result.get("content", "")) + len(result.get("reasoning", ""))
check("Got output (content or reasoning)", total_output > 10,
      f"content={len(result.get('content',''))}, reasoning={len(result.get('reasoning',''))}")
check("Usage chunk present", result.get("has_usage", False))
check("TTFT < 60s", (result.get("ttft") or 999) < 60,
      f"ttft={result.get('ttft', '?'):.2f}s" if result.get("ttft") else "no ttft")
print(f"  Content: {result.get('content','')[:200]}")
print(f"  Chunks: {result.get('chunks')} | Elapsed: {result.get('elapsed',0):.1f}s")

# --- 4. Reasoning / <think> blocks ---
print(f"\n[4] REASONING — THINK BLOCKS ({THINK_MODEL})")
result = stream_chat(THINK_MODEL, [
    {"role": "user", "content": "If a train travels 120 km in 2 hours, what is its average speed? Think step by step."}
])
has_reasoning = len(result.get("reasoning", "")) > 0
has_answer = len(result.get("content", "")) > 0
check("Got reasoning_content", has_reasoning,
      f"reasoning_len={len(result.get('reasoning',''))}")
check("Got final answer", has_answer,
      f"content_len={len(result.get('content',''))}")
check("Answer mentions 60", "60" in result.get("content", ""),
      f"'{result.get('content','')[:150]}'")
if has_reasoning:
    print(f"  Reasoning preview: {result['reasoning'][:200]}...")
print(f"  Answer: {result.get('content','')[:200]}")

# --- 5. Multi-turn conversation (KV cache) ---
print(f"\n[5] MULTI-TURN CONVERSATION — KV CACHE ({THINK_MODEL})")
# Unload first to ensure clean KV state
unload()

# Turn 1
t1 = stream_chat(THINK_MODEL, [
    {"role": "user", "content": "My name is Alex and I live in Tokyo."}
])
check("Turn 1: got response", len(t1.get("content", "")) > 0,
      f"'{t1.get('content','')[:80]}'")

# Turn 2 — should recall context from Turn 1
t2 = stream_chat(THINK_MODEL, [
    {"role": "user", "content": "My name is Alex and I live in Tokyo."},
    {"role": "assistant", "content": t1.get("content", "I see!")},
    {"role": "user", "content": "What city did I say I live in?"},
])
t2_content = t2.get("content", "").lower()
check("Turn 2: recalls Tokyo", "tokyo" in t2_content,
      f"'{t2.get('content','')[:120]}'")

# Turn 3 — further follow-up
t3 = stream_chat(THINK_MODEL, [
    {"role": "user", "content": "My name is Alex and I live in Tokyo."},
    {"role": "assistant", "content": t1.get("content", "I see!")},
    {"role": "user", "content": "What city did I say I live in?"},
    {"role": "assistant", "content": t2.get("content", "Tokyo")},
    {"role": "user", "content": "And what is my name?"},
])
t3_content = t3.get("content", "").lower()
check("Turn 3: recalls Alex", "alex" in t3_content,
      f"'{t3.get('content','')[:120]}'")

# --- 6. RAG — Document comprehension ---
print(f"\n[6] RAG — DOCUMENT COMPREHENSION ({THINK_MODEL})")
rag_system = (
    "Here is some reference information:\n"
    '<source id="1" name="company_report.pdf">'
    "Acme Corp reported Q3 2025 revenue of $4.2 billion, up 15% year-over-year. "
    "The company's cloud division grew 28% to $1.8 billion. CEO Jane Smith said "
    "\"Our AI investments are paying off with strong enterprise adoption.\" "
    "Operating margin improved to 22% from 19% in Q3 2024. "
    "The company announced 2,000 new hires in the engineering division."
    "</source>"
)
result_rag = stream_chat(THINK_MODEL, [
    {"role": "system", "content": rag_system},
    {"role": "user", "content": "What was Acme Corp's Q3 revenue and how much did it grow?"},
])
rag_content = result_rag.get("content", "")
check("RAG: got response", len(rag_content) > 10, f"len={len(rag_content)}")
check("RAG: mentions $4.2 billion", "4.2" in rag_content, f"'{rag_content[:200]}'")
check("RAG: mentions 15%", "15" in rag_content)
print(f"  RAG answer: {rag_content[:250]}")

# --- 7. Shortcircuit: Search query generation ---
print(f"\n[7] SHORTCIRCUIT — SEARCH QUERY GENERATION")
sc_query = stream_chat(THINK_MODEL, [
    {"role": "user", "content": (
        'Generate search queries for the following user message in order to '
        'retrieve relevant information from the web. Return a JSON object with '
        'a key "queries" containing a list of search queries.\n\n'
        '<chat_history>\n'
        'USER: What is the tallest building in the world right now?\n'
        '</chat_history>'
    )}
])
sc_content = sc_query.get("content", "")
check("Shortcircuit: fast response (< 2s)", (sc_query.get("elapsed") or 999) < 2,
      f"elapsed={sc_query.get('elapsed',0):.2f}s")
check("Shortcircuit: contains queries", "queries" in sc_content.lower() or "tallest" in sc_content.lower(),
      f"'{sc_content[:150]}'")
check("Shortcircuit: no reasoning", len(sc_query.get("reasoning", "")) == 0,
      "thinking was disabled for meta-task")
print(f"  Query gen output: {sc_content[:200]}")

# --- 8. Shortcircuit: Title generation ---
print(f"\n[8] SHORTCIRCUIT — TITLE GENERATION")
sc_title = stream_chat(THINK_MODEL, [
    {"role": "user", "content": "Tell me about the history of jazz music."},
    {"role": "assistant", "content": "Jazz originated in the early 20th century in New Orleans..."},
    {"role": "user", "content": (
        "Create a concise, 3-5 word title for the chat based on the conversation. "
        "Respond with just the title, nothing else."
    )},
])
title = sc_title.get("content", "").strip()
check("Title gen: fast (< 2s)", (sc_title.get("elapsed") or 999) < 2,
      f"elapsed={sc_title.get('elapsed',0):.2f}s")
check("Title gen: short output", 0 < len(title) < 60, f"'{title}'")
check("Title gen: no reasoning", len(sc_title.get("reasoning", "")) == 0)
print(f"  Generated title: '{title}'")

# --- 9. Shortcircuit: Tag generation ---
print(f"\n[9] SHORTCIRCUIT — TAG GENERATION")
sc_tag = stream_chat(THINK_MODEL, [
    {"role": "user", "content": (
        "Generate 1-3 tags to categorize the chat. "
        'Return tags as a JSON list, e.g. ["science", "biology"]\n\n'
        "Conversation: Tell me about photosynthesis."
    )},
])
tag_content = sc_tag.get("content", "").strip()
check("Tag gen: fast (< 2s)", (sc_tag.get("elapsed") or 999) < 2,
      f"elapsed={sc_tag.get('elapsed',0):.2f}s")
check("Tag gen: got output", len(tag_content) > 0, f"'{tag_content}'")
print(f"  Generated tags: '{tag_content}'")

# --- 10. Date awareness ---
print(f"\n[10] DATE AWARENESS — SYSTEM PROMPT")
result_date = stream_chat(THINK_MODEL, [
    {"role": "system", "content": "Today is 2026-03-03. Current time: 14:30 SAST."},
    {"role": "user", "content": "What is today's date?"},
])
date_content = result_date.get("content", "").lower()
check("Date: mentions March or 2026-03-03 or 03",
      "march" in date_content or "2026" in date_content or "03" in date_content,
      f"'{result_date.get('content','')[:150]}'")
print(f"  Date answer: {result_date.get('content','')[:200]}")

# --- 11. Home Assistant simulation ---
print(f"\n[11] HOME ASSISTANT STYLE REQUEST")
ha_result = stream_chat(THINK_MODEL, [
    {"role": "system", "content": (
        "You are a smart home manager of Home Assistant.\n"
        "Available Devices:\n"
        "- light.living_room (on, brightness: 80%)\n"
        "- light.bedroom (off)\n"
        "- switch.coffee_maker (off)\n"
        "- climate.thermostat (heat, 22°C)\n"
        "Use the execute_services function to control devices."
    )},
    {"role": "user", "content": "Turn on the bedroom light and set brightness to 50%."},
])
ha_content = ha_result.get("content", "")
check("HA: got response", len(ha_content) > 5, f"len={len(ha_content)}")
check("HA: no reasoning (thinking disabled)", len(ha_result.get("reasoning", "")) == 0,
      "HA requests should auto-disable thinking")
check("HA: mentions bedroom", "bedroom" in ha_content.lower(),
      f"'{ha_content[:150]}'")
print(f"  HA response: {ha_content[:250]}")

# --- 12. Model switching ---
print(f"\n[12] MODEL SWITCHING — {THINK_MODEL} -> {ALT_MODEL}")
r1 = stream_chat(THINK_MODEL, [
    {"role": "user", "content": "Say 'hello' in French."}
])
check("Model A response", len(r1.get("content", "")) > 0)

r2 = stream_chat(ALT_MODEL, [
    {"role": "user", "content": "Say 'hello' in Spanish."}
])
check("Model B response after switch", len(r2.get("content", "")) > 0,
      f"'{r2.get('content','')[:80]}'")

# Check model B is now loaded
code, h2, _ = req("GET", "/health")
check("Health shows new model", (h2 or {}).get("current_model") == ALT_MODEL,
      f"loaded={h2.get('current_model') if h2 else '?'}")

# --- 13. Concurrent request rejection ---
print(f"\n[13] CONCURRENT REQUEST REJECTION")
# Start a request in background — use ALT_MODEL (non-thinking) with a short prompt
# so it finishes quickly enough that we can proceed without blocking later tests
bg_result = {}

def _bg_request():
    bg_result["data"] = stream_chat(ALT_MODEL, [
        {"role": "user", "content": "What is 2 + 2?"}
    ], timeout=120)

t = threading.Thread(target=_bg_request, daemon=True)
t.start()
time.sleep(2)  # Let it start generating

# Try to send a second request while first is in flight
code_busy, data_busy, _ = req("POST", "/v1/chat/completions", {
    "model": ALT_MODEL,
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": False,
}, timeout=10)
check("Concurrent request rejected with 503", code_busy == 503,
      f"status={code_busy}")

# Wait for background thread to finish (should be fast for "2+2")
t.join(timeout=180)
bg_total = len(bg_result.get("data", {}).get("content", "")) + len(bg_result.get("data", {}).get("reasoning", ""))
check("Background request completed", bg_total > 0,
      f"content+reasoning={bg_total}")

# --- 14. Long output generation ---
print(f"\n[14] LONG OUTPUT GENERATION ({THINK_MODEL})")
# Explicitly disable thinking via /no_think suffix for content-focused prompt
long_result = stream_chat(THINK_MODEL, [
    {"role": "user", "content": "List 10 countries in Europe and their capitals. Number each one. Be concise, no explanations."}
])
long_content = long_result.get("content", "")
long_total = len(long_content) + len(long_result.get("reasoning", ""))
check("Long output: got substantial response", long_total > 50,
      f"content={len(long_content)}, reasoning={len(long_result.get('reasoning',''))}")
# Count how many numbers appear in content OR reasoning (thinking models may put list in reasoning)
all_output = long_content + long_result.get("reasoning", "")
numbers = re.findall(r'\b\d+[\.):]', all_output)
check("Long output: has numbered items", len(numbers) >= 3,
      f"found {len(numbers)} numbered items")
print(f"  Output length: {len(long_content)} content + {len(long_result.get('reasoning',''))} reasoning")
print(f"  Preview: {(long_content or long_result.get('reasoning',''))[:200]}...")

# --- 15. Error handling ---
print(f"\n[15] ERROR HANDLING")
# Bad model name
code_bad, _, _ = req("POST", "/v1/chat/completions", {
    "model": "nonexistent-model-42",
    "messages": [{"role": "user", "content": "test"}],
}, timeout=15)
check("Bad model returns 404", code_bad == 404, f"status={code_bad}")

# Empty messages
code_empty, _, _ = req("POST", "/v1/chat/completions", {
    "model": THINK_MODEL,
    "messages": [],
}, timeout=15)
check("Empty messages returns 400", code_empty == 400, f"status={code_empty}")

# Missing model field
code_nomodel, _, _ = req("POST", "/v1/chat/completions", {
    "messages": [{"role": "user", "content": "test"}],
}, timeout=15)
check("Missing model field: error", code_nomodel in (400, 404),
      f"status={code_nomodel}")

# Wait for any lingering request to complete — poll health until server is idle
for _ in range(30):
    _hc, _hd, _ = req("GET", "/health")
    if _hd and not _hd.get("request_in_progress", False):
        break
    time.sleep(2)

# /v1/models/select endpoint
code_sel, dsel, _ = req("POST", "/v1/models/select", {"model": THINK_MODEL})
check("Model select endpoint works", code_sel == 200,
      f"status={code_sel}")

# /v1/models/unload endpoint
code_unl, dunl, _ = req("POST", "/v1/models/unload")
check("Model unload endpoint works", code_unl == 200,
      f"status={code_unl}")
code_h3, h3, _ = req("GET", "/health")
check("After unload: no model loaded", (h3 or {}).get("model_loaded") == False,
      f"loaded={(h3 or {}).get('model_loaded')}")

# =========================================================================
# SUMMARY
# =========================================================================
print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"  PASS: {PASS}")
print(f"  FAIL: {FAIL}")
print(f"  Total: {PASS + FAIL}")
print()

if FAIL > 0:
    print("!! Some tests failed — review output above.")
    sys.exit(1)
else:
    print("All real-world smoke tests passed!")
    sys.exit(0)
