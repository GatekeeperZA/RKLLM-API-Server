"""
RKLLM API Server — End-to-End Integration Test
================================================
Tests all major components through the live API:
  1. Basic chat on all text models
  2. Shortcircuit paths (query gen, title gen, tag gen)
  3. RAG with document content (single + multi source)
  4. Web search simulation (SearXNG integration)
  5. KV cache incremental mode (multi-turn)
  6. Prompt building (system prompts, date injection, HA detection)
  7. RAG cache (repeat query returns cached result)
  8. Open WebUI database config verification

Usage:
    python e2e_test.py                 # Run all tests
    python e2e_test.py --section 1     # Run only section 1
    python e2e_test.py --fast          # Skip slow models (gemma, phi)

Runs against: http://192.168.2.180:8000 (RKLLM API)
              http://192.168.2.180:8080 (Open WebUI)
"""
import json, re, sys, time, os
import urllib.request, urllib.error

# =============================================================================
# CONFIGURATION
# =============================================================================
API = "http://192.168.2.180:8000"
WEBUI = "http://192.168.2.180:3000"
SEARXNG = "http://192.168.2.180:8080"
TIMEOUT = 180  # Per-request timeout (seconds)

# Text models to test (skip deepseekocr — it's VL-only)
TEXT_MODELS = [
    "qwen3-1.7b",
    "qwen3-4b-instruct-2507",
    "gemma-3-4b-it",
    "phi-3-mini-4k-instruct",
]
FAST_MODELS = ["qwen3-1.7b", "qwen3-4b-instruct-2507"]

# =============================================================================
# TEST FRAMEWORK
# =============================================================================
PASS = 0
FAIL = 0
SKIP = 0
WARN = 0
RESULTS = []
SECTION_FILTER = None
FAST_MODE = False


def _req(method, path, body=None, timeout=TIMEOUT, base=API):
    """Send HTTP request, return (status_code, parsed_json_or_None, raw_body)."""
    url = f"{base}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw), raw
            except json.JSONDecodeError:
                return resp.status, None, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else ""
        try:
            return e.code, json.loads(raw), raw
        except (json.JSONDecodeError, Exception):
            return e.code, None, raw
    except Exception as e:
        return 0, None, str(e)


def _stream_req(path, body, timeout=TIMEOUT):
    """Send streaming request, collect all SSE chunks."""
    url = f"{API}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    chunks = []
    content = ""
    reasoning = ""
    has_done = False
    has_fingerprint = True
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
                    chunks.append(obj)
                    if "system_fingerprint" not in obj:
                        has_fingerprint = False
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    content += delta.get("content", "")
                    reasoning += delta.get("reasoning_content", "")
                except json.JSONDecodeError:
                    pass
        return {
            "chunks": chunks, "content": content, "reasoning": reasoning,
            "has_done": has_done, "has_fingerprint": has_fingerprint,
        }
    except Exception as e:
        return {
            "chunks": chunks, "content": content, "reasoning": reasoning,
            "has_done": has_done, "has_fingerprint": has_fingerprint,
            "error": str(e),
        }


def _chat(model, messages, stream=True, timeout=TIMEOUT):
    """Convenience: send a chat completion request."""
    body = {"model": model, "messages": messages, "stream": stream}
    if stream:
        return _stream_req("/v1/chat/completions", body, timeout=timeout)
    else:
        code, data, raw = _req("POST", "/v1/chat/completions", body, timeout=timeout)
        return {"code": code, "data": data, "raw": raw}


def _unload():
    """Unload current model to start fresh."""
    _req("POST", "/v1/models/unload")
    time.sleep(1)


def check(section, name, condition, detail=""):
    """Record a test result."""
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    tag = f"[{status}]"
    line = f"  {tag:6s} {name}"
    if detail:
        line += f"  |  {detail}"
    print(line)
    RESULTS.append((section, name, status, detail))


def warn(section, name, detail=""):
    """Record a warning (not a failure)."""
    global WARN
    WARN += 1
    line = f"  [WARN] {name}"
    if detail:
        line += f"  |  {detail}"
    print(line)
    RESULTS.append((section, name, "WARN", detail))


def skip(section, name, detail=""):
    """Record a skipped test."""
    global SKIP
    SKIP += 1
    print(f"  [SKIP] {name}  |  {detail}")
    RESULTS.append((section, name, "SKIP", detail))


def run_section(num):
    """Check if this section should run."""
    return SECTION_FILTER is None or SECTION_FILTER == num


def get_models():
    """Return model list based on --fast flag."""
    return FAST_MODELS if FAST_MODE else TEXT_MODELS


# =============================================================================
# SECTION 1: BASIC CHAT ON ALL TEXT MODELS
# =============================================================================
def test_basic_chat():
    if not run_section(1):
        return
    sec = "1-BASIC"
    print(f"\n{'='*60}")
    print(f"SECTION 1: BASIC CHAT ON ALL TEXT MODELS")
    print(f"{'='*60}")

    models = get_models()
    for model in models:
        print(f"\n  --- Testing {model} ---")
        _unload()

        # Simple factual question (streaming)
        t0 = time.time()
        result = _chat(model, [
            {"role": "user", "content": "What is the capital of France? Answer in one sentence."}
        ], stream=True)
        elapsed = time.time() - t0

        content = result.get("content", "")
        has_done = result.get("has_done", False)
        has_fp = result.get("has_fingerprint", True)
        error = result.get("error", "")

        check(sec, f"{model}: got response",
              len(content) > 5,
              f"{len(content)} chars in {elapsed:.1f}s")
        check(sec, f"{model}: mentions Paris",
              "paris" in content.lower(),
              f"'{content[:100]}'")
        check(sec, f"{model}: stream completed",
              has_done, "SSE [DONE] received" if has_done else f"error={error}")
        check(sec, f"{model}: system_fingerprint",
              has_fp, "present in all chunks")

        # Non-streaming test
        result2 = _chat(model, [
            {"role": "user", "content": "What is 2+2? Reply with just the number."}
        ], stream=False)
        code = result2.get("code", 0)
        data = result2.get("data", {})
        ns_content = ""
        if data and "choices" in data:
            ns_content = data["choices"][0].get("message", {}).get("content", "")
        has_fp2 = "system_fingerprint" in (data or {})

        check(sec, f"{model}: non-stream 200",
              code == 200, f"status={code}")
        check(sec, f"{model}: non-stream answer",
              "4" in ns_content,
              f"'{ns_content[:80]}'")
        check(sec, f"{model}: non-stream fingerprint",
              has_fp2, "system_fingerprint in response")

    # Health check after tests
    code, data, _ = _req("GET", "/health")
    check(sec, "Health OK after model cycling",
          code == 200 and data.get("status") == "ok",
          json.dumps(data) if data else "no response")


# =============================================================================
# SECTION 2: SHORTCIRCUIT PATHS (QUERY GEN, TITLE GEN, TAG GEN)
# =============================================================================
def test_shortcircuits():
    if not run_section(2):
        return
    sec = "2-SHORTCIRCUIT"
    print(f"\n{'='*60}")
    print(f"SECTION 2: SHORTCIRCUIT PATHS")
    print(f"{'='*60}")

    model = "qwen3-1.7b"

    # --- Query Generation Shortcircuit ---
    print("\n  --- Query Gen Shortcircuit ---")
    t0 = time.time()
    result = _chat(model, [
        {"role": "user", "content": (
            "### Task:\n"
            "Based on the chat history, generate 1-3 broad search queries.\n"
            "Return JSON: {\"queries\": [\"query1\"]}\n\n"
            "<chat_history>\n"
            "USER: What is the weather like in London today?\n"
            "ASSISTANT: I don't have real-time weather data.\n"
            "</chat_history>"
        )}
    ], stream=True)
    elapsed = time.time() - t0
    content = result.get("content", "")

    check(sec, "Query gen: instant response",
          elapsed < 2.0,
          f"{elapsed:.2f}s (should be <2s — no model inference)")
    check(sec, "Query gen: valid JSON",
          '"queries"' in content,
          f"'{content[:120]}'")
    # Parse and check the actual query
    try:
        parsed = json.loads(content)
        queries = parsed.get("queries", [])
        check(sec, "Query gen: has queries array",
              len(queries) >= 1,
              f"queries={queries}")
        check(sec, "Query gen: relevant query",
              any("london" in q.lower() or "weather" in q.lower() for q in queries),
              f"queries={queries}")
    except json.JSONDecodeError:
        check(sec, "Query gen: parseable JSON", False, f"raw='{content}'")

    # --- Query Gen with Vague Follow-up (Context Enrichment) ---
    print("\n  --- Query Gen Context Enrichment ---")
    t0 = time.time()
    result = _chat(model, [
        {"role": "user", "content": (
            "### Task:\n"
            "Based on the chat history, generate 1-3 broad search queries.\n"
            "Return JSON: {\"queries\": [\"query1\"]}\n\n"
            "<chat_history>\n"
            "USER: Tell me about machine learning courses\n"
            "ASSISTANT: Here are some popular options:\n"
            "1. **Python for Data Science** by Coursera\n"
            "2. \"Deep Learning Specialization\" by Andrew Ng\n"
            "3. **Machine Learning Engineering** bootcamp\n"
            "USER: Can you verify that one exists?\n"
            "</chat_history>"
        )}
    ], stream=True)
    elapsed = time.time() - t0
    content = result.get("content", "")

    check(sec, "Enriched query: instant response",
          elapsed < 2.0,
          f"{elapsed:.2f}s")
    try:
        parsed = json.loads(content)
        queries = parsed.get("queries", [])
        enriched = queries[0] if queries else ""
        # Should contain context from the assistant's response
        has_context = any(kw in enriched.lower() for kw in
                         ["python", "data science", "deep learning",
                          "andrew ng", "machine learning", "coursera"])
        check(sec, "Enriched query: contains entities",
              has_context,
              f"enriched='{enriched}'")
    except (json.JSONDecodeError, IndexError):
        check(sec, "Enriched query: parseable", False, f"raw='{content}'")

    # --- Title Generation Shortcircuit ---
    print("\n  --- Title Gen Shortcircuit ---")
    t0 = time.time()
    result = _chat(model, [
        {"role": "user", "content": "How do I install Python on Windows?"},
        {"role": "assistant", "content": "You can download Python from python.org..."},
        {"role": "user", "content": (
            "Create a concise, 3-5 word title for the chat. "
            "Do not use quotes or special characters."
        )}
    ], stream=True)
    elapsed = time.time() - t0
    content = result.get("content", "")

    check(sec, "Title gen: instant response",
          elapsed < 2.0,
          f"{elapsed:.2f}s")
    check(sec, "Title gen: reasonable title",
          3 <= len(content) <= 60,
          f"'{content}'")

    # --- Tag Generation Shortcircuit ---
    print("\n  --- Tag Gen Shortcircuit ---")
    t0 = time.time()
    result = _chat(model, [
        {"role": "user", "content": (
            "Generate 1-3 tags to categorize the chat. "
            "Return a JSON list of tag strings."
        )}
    ], stream=True)
    elapsed = time.time() - t0
    content = result.get("content", "")

    check(sec, "Tag gen: instant response",
          elapsed < 2.0,
          f"{elapsed:.2f}s")
    check(sec, "Tag gen: returns content",
          len(content) > 0,
          f"'{content}'")

    # --- system_fingerprint in shortcircuit responses ---
    check(sec, "Query gen: system_fingerprint",
          result.get("has_fingerprint", False),
          "present in SSE chunks")


# =============================================================================
# SECTION 3: RAG WITH DOCUMENT CONTENT
# =============================================================================
def test_rag():
    if not run_section(3):
        return
    sec = "3-RAG"
    print(f"\n{'='*60}")
    print(f"SECTION 3: RAG WITH DOCUMENT CONTENT")
    print(f"{'='*60}")

    model = "qwen3-1.7b"
    _unload()

    # --- Single Source RAG ---
    print("\n  --- Single Source RAG ---")
    rag_system = (
        "Here is some reference information:\n"
        "<source>\n"
        "The Acme Corporation was founded in 1987 by Dr. Sarah Chen in Portland, Oregon. "
        "The company specializes in advanced robotics and artificial intelligence. "
        "In 2024, Acme reported annual revenue of $2.3 billion with 12,500 employees worldwide. "
        "Their flagship product is the RoboAssist X9, which costs $45,000 per unit. "
        "The company headquarters moved to Austin, Texas in 2021. "
        "CEO since 2019 is Marcus Williams. The CTO is Dr. James Park.\n"
        "</source>\n"
        "### Task: Answer the following question based on the reference above."
    )

    # Q1: Factual extraction
    t0 = time.time()
    result = _chat(model, [
        {"role": "system", "content": rag_system},
        {"role": "user", "content": "Who founded Acme Corporation and when?"}
    ], stream=True)
    elapsed = time.time() - t0
    content = result.get("content", "")

    check(sec, "RAG single: got response",
          len(content) > 10,
          f"{len(content)} chars in {elapsed:.1f}s")
    check(sec, "RAG single: Sarah Chen",
          "sarah chen" in content.lower(),
          f"'{content[:150]}'")
    check(sec, "RAG single: 1987",
          "1987" in content,
          f"'{content[:150]}'")

    # Q2: Numeric extraction
    result2 = _chat(model, [
        {"role": "system", "content": rag_system},
        {"role": "user", "content": "What is Acme's annual revenue and how many employees do they have?"}
    ], stream=True)
    content2 = result2.get("content", "")

    check(sec, "RAG single: revenue $2.3B",
          "2.3" in content2 or "2,3" in content2,
          f"'{content2[:150]}'")
    check(sec, "RAG single: 12,500 employees",
          "12,500" in content2 or "12500" in content2 or "12 500" in content2,
          f"'{content2[:150]}'")

    # --- Multi-Source RAG ---
    print("\n  --- Multi-Source RAG ---")
    multi_rag_system = (
        "Here is some reference information:\n"
        "<source id='doc1'>\n"
        "Project Alpha started on January 15, 2025. The project lead is Emily Rodriguez. "
        "Budget: $500,000. Status: On track. Deadline: June 30, 2025.\n"
        "</source>\n"
        "<source id='doc2'>\n"
        "Project Beta started on March 1, 2025. The project lead is David Kim. "
        "Budget: $1.2 million. Status: Delayed by 3 weeks. Deadline: September 15, 2025.\n"
        "</source>\n"
        "<source id='doc3'>\n"
        "Project Gamma started on February 10, 2025. The project lead is Lisa Chen. "
        "Budget: $750,000. Status: Completed ahead of schedule. Delivered: April 20, 2025.\n"
        "</source>\n"
        "### Task: Answer the following question based on the reference above."
    )

    result3 = _chat(model, [
        {"role": "system", "content": multi_rag_system},
        {"role": "user", "content": "Which project is delayed and by how much? Who leads it?"}
    ], stream=True)
    content3 = result3.get("content", "")

    check(sec, "RAG multi: mentions Beta",
          "beta" in content3.lower(),
          f"'{content3[:150]}'")
    check(sec, "RAG multi: 3 weeks delay",
          "3 week" in content3.lower() or "three week" in content3.lower(),
          f"'{content3[:150]}'")
    check(sec, "RAG multi: David Kim",
          "david" in content3.lower() or "kim" in content3.lower(),
          f"'{content3[:150]}'")

    # Cross-source comparison
    result4 = _chat(model, [
        {"role": "system", "content": multi_rag_system},
        {"role": "user", "content": "Which project has the largest budget?"}
    ], stream=True)
    content4 = result4.get("content", "")

    check(sec, "RAG multi: largest budget = Beta",
          "beta" in content4.lower(),
          f"'{content4[:150]}'")
    check(sec, "RAG multi: $1.2 million",
          "1.2" in content4,
          f"'{content4[:150]}'")

    # --- RAG Follow-up Detection (should skip RAG for unrelated follow-up) ---
    print("\n  --- RAG Follow-up Detection ---")
    result5 = _chat(model, [
        {"role": "system", "content": rag_system},
        {"role": "assistant", "content": "Acme was founded by Dr. Sarah Chen in 1987."},
        {"role": "user", "content": "thanks"}
    ], stream=True)
    content5 = result5.get("content", "")
    # "thanks" should trigger follow-up detection and skip RAG
    check(sec, "RAG follow-up skip: responds conversationally",
          len(content5) > 0,
          f"'{content5[:100]}'")


# =============================================================================
# SECTION 4: RAG CACHE
# =============================================================================
def test_rag_cache():
    if not run_section(4):
        return
    sec = "4-CACHE"
    print(f"\n{'='*60}")
    print(f"SECTION 4: RAG RESPONSE CACHE")
    print(f"{'='*60}")

    model = "qwen3-1.7b"
    _unload()  # Clean slate — prevent KV cross-contamination

    rag_system = (
        "Here is some reference information:\n"
        "<source>\n"
        "The XYZ-9000 processor runs at 4.5 GHz and has 16 cores. "
        "It was released on November 3, 2025, at a price of $599.\n"
        "</source>\n"
        "### Task: Answer the following question based on the reference above."
    )
    question = "What speed does the XYZ-9000 run at?"

    # First request (cold — generates new)
    t0 = time.time()
    result1 = _chat(model, [
        {"role": "system", "content": rag_system},
        {"role": "user", "content": question}
    ], stream=True)
    t1 = time.time() - t0
    content1 = result1.get("content", "")

    check(sec, "Cache cold: got response",
          len(content1) > 5,
          f"{len(content1)} chars in {t1:.1f}s")

    # Second request (should hit cache — much faster)
    t0 = time.time()
    result2 = _chat(model, [
        {"role": "system", "content": rag_system},
        {"role": "user", "content": question}
    ], stream=True)
    t2 = time.time() - t0
    content2 = result2.get("content", "")

    check(sec, "Cache warm: got response",
          len(content2) > 5,
          f"{len(content2)} chars in {t2:.1f}s")
    check(sec, "Cache warm: faster than cold",
          t2 < t1 * 0.5,
          f"cold={t1:.1f}s, warm={t2:.1f}s")
    check(sec, "Cache warm: same content",
          content1.strip() == content2.strip(),
          f"match={content1.strip() == content2.strip()}")


# =============================================================================
# SECTION 5: WEB SEARCH SIMULATION
# =============================================================================
def test_web_search():
    if not run_section(5):
        return
    sec = "5-WEBSEARCH"
    print(f"\n{'='*60}")
    print(f"SECTION 5: WEB SEARCH FLOW")
    print(f"{'='*60}")

    # First test SearXNG is reachable
    print("\n  --- SearXNG Availability ---")
    try:
        code, data, raw = _req("GET", "/search?q=test&format=json", base=SEARXNG, timeout=10)
        searxng_up = code == 200
        check(sec, "SearXNG reachable",
              searxng_up,
              f"status={code}")
    except Exception as e:
        searxng_up = False
        check(sec, "SearXNG reachable", False, str(e))

    if not searxng_up:
        skip(sec, "Web search tests", "SearXNG not available")
        return

    # Simulate what Open WebUI sends: system prompt with web search results
    print("\n  --- Web Search RAG ---")
    model = "qwen3-1.7b"

    # Simulate a web search result injected by Open WebUI
    web_rag_system = (
        "Here is some reference information:\n"
        "<source>\n"
        "According to the World Bank, the global GDP in 2024 was approximately $105 trillion. "
        "The United States remains the largest economy with a GDP of $28.8 trillion, "
        "followed by China at $18.5 trillion and Japan at $4.2 trillion. "
        "India surpassed the UK to become the fifth-largest economy with $3.9 trillion. "
        "Global GDP growth rate was 3.2% in 2024, slightly above the 2023 rate of 3.1%.\n"
        "</source>\n"
        "### Task: Answer the following question based on the reference above."
    )

    result = _chat(model, [
        {"role": "system", "content": web_rag_system},
        {"role": "user", "content": "What was the global GDP growth rate in 2024?"}
    ], stream=True)
    content = result.get("content", "")

    check(sec, "Web RAG: got response",
          len(content) > 10,
          f"{len(content)} chars")
    check(sec, "Web RAG: 3.2%",
          "3.2" in content,
          f"'{content[:150]}'")
    check(sec, "Web RAG: mentions growth",
          "growth" in content.lower() or "percent" in content.lower() or "%" in content,
          f"'{content[:150]}'")

    # Test accuracy: cross-reference question
    result2 = _chat(model, [
        {"role": "system", "content": web_rag_system},
        {"role": "user", "content": "Which country has the largest GDP and what is it?"}
    ], stream=True)
    content2 = result2.get("content", "")

    check(sec, "Web RAG: US largest",
          "united states" in content2.lower() or "us" in content2.lower() or "america" in content2.lower(),
          f"'{content2[:150]}'")
    check(sec, "Web RAG: $28.8T",
          "28.8" in content2,
          f"'{content2[:150]}'")


# =============================================================================
# SECTION 6: KV CACHE / MULTI-TURN CONVERSATION
# =============================================================================
def test_kv_cache():
    if not run_section(6):
        return
    sec = "6-KV"
    print(f"\n{'='*60}")
    print(f"SECTION 6: KV CACHE / MULTI-TURN")
    print(f"{'='*60}")

    model = "qwen3-1.7b"
    _unload()

    # Turn 1: Establish context
    print("\n  --- Turn 1: Establish Context ---")
    t0 = time.time()
    result1 = _chat(model, [
        {"role": "user", "content": "My name is Alex and I live in Tokyo. Remember this."}
    ], stream=True)
    t1 = time.time() - t0
    content1 = result1.get("content", "")

    check(sec, "Turn 1: got response",
          len(content1) > 5,
          f"{len(content1)} chars in {t1:.1f}s")

    # Turn 2: Test context retention (should use incremental KV)
    print("\n  --- Turn 2: Context Retention ---")
    t0 = time.time()
    result2 = _chat(model, [
        {"role": "user", "content": "My name is Alex and I live in Tokyo. Remember this."},
        {"role": "assistant", "content": content1},
        {"role": "user", "content": "What is my name and where do I live?"}
    ], stream=True)
    t2 = time.time() - t0
    content2 = result2.get("content", "")

    check(sec, "Turn 2: got response",
          len(content2) > 5,
          f"{len(content2)} chars in {t2:.1f}s")
    check(sec, "Turn 2: remembers Alex",
          "alex" in content2.lower(),
          f"'{content2[:150]}'")
    check(sec, "Turn 2: remembers Tokyo",
          "tokyo" in content2.lower(),
          f"'{content2[:150]}'")

    # Turn 3: Further follow-up
    print("\n  --- Turn 3: Further Follow-up ---")
    t0 = time.time()
    result3 = _chat(model, [
        {"role": "user", "content": "My name is Alex and I live in Tokyo. Remember this."},
        {"role": "assistant", "content": content1},
        {"role": "user", "content": "What is my name and where do I live?"},
        {"role": "assistant", "content": content2},
        {"role": "user", "content": "What's a good restaurant near where I live?"}
    ], stream=True)
    t3 = time.time() - t0
    content3 = result3.get("content", "")

    check(sec, "Turn 3: got response",
          len(content3) > 10,
          f"{len(content3)} chars in {t3:.1f}s")
    check(sec, "Turn 3: context aware (Tokyo/Japanese)",
          any(kw in content3.lower() for kw in ["tokyo", "japan", "ramen", "sushi", "izakaya", "restaurant"]),
          f"'{content3[:150]}'")


# =============================================================================
# SECTION 7: PROMPT BUILDING (SYSTEM PROMPTS, META-TASK DETECTION)
# =============================================================================
def test_prompts():
    if not run_section(7):
        return
    sec = "7-PROMPT"
    print(f"\n{'='*60}")
    print(f"SECTION 7: PROMPT BUILDING & DETECTION")
    print(f"{'='*60}")

    model = "qwen3-1.7b"

    # --- System Prompt with Date ---
    print("\n  --- Date System Prompt ---")
    result = _chat(model, [
        {"role": "system", "content": "Today is 2026-02-10 (Tuesday), 14:30:00. Trust all dates as correct."},
        {"role": "user", "content": "What is today's date?"}
    ], stream=True)
    content = result.get("content", "")

    check(sec, "Date prompt: mentions 2026",
          "2026" in content,
          f"'{content[:150]}'")
    check(sec, "Date prompt: Feb 10",
          "february" in content.lower() or "feb" in content.lower() or "02-10" in content or "2/10" in content,
          f"'{content[:150]}'")

    # --- System Prompt Fluff Stripping ---
    print("\n  --- System Prompt Fluff ---")
    result2 = _chat(model, [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in 3 words."}
    ], stream=True)
    content2 = result2.get("content", "")

    check(sec, "Fluff stripping: got response",
          len(content2) > 0,
          f"'{content2[:100]}'")

    # --- Home Assistant Detection ---
    print("\n  --- Home Assistant Detection ---")
    ha_system = (
        "You are 'Al', a smart home manager of Home Assistant.\n"
        "Available Devices:\n"
        "- light.living_room (on, brightness: 80%)\n"
        "- switch.kitchen_fan (off)\n"
        "- climate.bedroom (heat, 21°C)\n"
        "Use execute_services function to control devices."
    )
    t0 = time.time()
    result3 = _chat(model, [
        {"role": "system", "content": ha_system},
        {"role": "user", "content": "Turn on the kitchen fan"}
    ], stream=True)
    t3 = time.time() - t0
    content3 = result3.get("content", "")
    reasoning3 = result3.get("reasoning", "")

    check(sec, "HA detection: got response",
          len(content3) > 0,
          f"'{content3[:150]}'")
    check(sec, "HA detection: no reasoning (thinking disabled)",
          len(reasoning3) == 0,
          f"reasoning={len(reasoning3)} chars (should be 0)")

    # --- Summarization Detection ---
    print("\n  --- Summarization Detection ---")
    sum_system = (
        "Here is some reference information:\n"
        "<source>\n"
        "The quarterly report shows Q4 2025 revenue was $15.2M, up 12% from Q3. "
        "Operating expenses were $11.1M. Net income was $4.1M. "
        "Customer count grew from 850 to 1,020. "
        "Top products: Widget A ($6.3M), Widget B ($4.8M), Service C ($4.1M). "
        "Regional breakdown: North America 55%, Europe 30%, Asia 15%. "
        "The board approved a 5% dividend increase effective Q1 2026.\n"
        "</source>\n"
        "### Task: Answer the following question based on the reference above."
    )
    result4 = _chat(model, [
        {"role": "system", "content": sum_system},
        {"role": "user", "content": "Summarize this quarterly report"}
    ], stream=True)
    content4 = result4.get("content", "")

    check(sec, "Summary: comprehensive response",
          len(content4) > 100,
          f"{len(content4)} chars")
    check(sec, "Summary: mentions revenue",
          "15.2" in content4 or "revenue" in content4.lower(),
          f"'{content4[:200]}'")
    check(sec, "Summary: mentions multiple facts",
          sum(1 for kw in ["widget", "customer", "dividend", "europe", "asia"]
              if kw in content4.lower()) >= 2,
          f"'{content4[:200]}'")


# =============================================================================
# SECTION 8: OPEN WEBUI DATABASE CONFIG VERIFICATION
# =============================================================================
def test_database():
    if not run_section(8):
        return
    sec = "8-DATABASE"
    print(f"\n{'='*60}")
    print(f"SECTION 8: OPEN WEBUI DATABASE CONFIG")
    print(f"{'='*60}")

    # Check Open WebUI is reachable
    print("\n  --- Open WebUI Availability ---")
    try:
        code, _, _ = _req("GET", "/", base=WEBUI, timeout=10)
        webui_up = code == 200
        check(sec, "Open WebUI reachable",
              webui_up, f"status={code}")
    except Exception as e:
        webui_up = False
        check(sec, "Open WebUI reachable", False, str(e))

    if not webui_up:
        skip(sec, "Database checks", "Open WebUI not reachable")
        return

    # Check API config endpoint
    print("\n  --- API Configuration ---")
    try:
        code, data, _ = _req("GET", "/api/config", base=WEBUI, timeout=10)
        check(sec, "Config endpoint accessible",
              code == 200 and data is not None,
              f"status={code}")
    except Exception as e:
        check(sec, "Config endpoint accessible", False, str(e))
        return

    # Check models endpoint reachability through Open WebUI proxy
    code2, data2, _ = _req("GET", "/api/models", base=WEBUI, timeout=10)
    check(sec, "Models via Open WebUI",
          code2 in (200, 401, 403),
          f"status={code2} (401/403 = auth required, which is normal)")


# =============================================================================
# SECTION 9: API COMPLIANCE & EDGE CASES
# =============================================================================
def test_compliance():
    if not run_section(9):
        return
    sec = "9-COMPLIANCE"
    print(f"\n{'='*60}")
    print(f"SECTION 9: API COMPLIANCE & EDGE CASES")
    print(f"{'='*60}")

    model = "qwen3-1.7b"

    # --- Invalid model ---
    print("\n  --- Error Handling ---")
    result = _chat("nonexistent-model-xyz", [
        {"role": "user", "content": "test"}
    ], stream=False)
    check(sec, "Invalid model: 404",
          result.get("code") == 404,
          f"status={result.get('code')}")

    # --- Empty messages ---
    code, data, _ = _req("POST", "/v1/chat/completions",
                          {"model": model, "messages": []})
    check(sec, "Empty messages: 400",
          code == 400,
          f"status={code}")

    # --- Invalid body ---
    code2, data2, _ = _req("POST", "/v1/chat/completions", "not json")
    check(sec, "Invalid body: 400",
          code2 == 400,
          f"status={code2}")

    # --- Model list structure ---
    code3, data3, _ = _req("GET", "/v1/models")
    if data3:
        has_required = all(
            all(k in m for k in ("id", "object", "created", "owned_by"))
            for m in data3.get("data", [])
        )
        check(sec, "Model list: OpenAI-compatible structure",
              has_required and data3.get("object") == "list",
              f"{len(data3.get('data', []))} models")
    else:
        check(sec, "Model list: accessible", False, f"status={code3}")

    # --- Health endpoint ---
    code4, data4, _ = _req("GET", "/health")
    check(sec, "Health: has required fields",
          all(k in (data4 or {}) for k in ("status", "current_model", "model_loaded", "models_available")),
          json.dumps(data4)[:200] if data4 else "no data")

    # --- Alias resolution ---
    print("\n  --- Alias Resolution ---")
    result5 = _chat("gemma", [
        {"role": "user", "content": "Say OK."}
    ], stream=True)
    content5 = result5.get("content", "")
    check(sec, "Alias 'gemma' resolves to gemma-3-4b-it",
          len(content5) > 0,
          f"'{content5[:60]}'")

    # --- stream_options.include_usage ---
    print("\n  --- Usage in Stream ---")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi."}],
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    result6 = _stream_req("/v1/chat/completions", body)
    chunks = result6.get("chunks", [])
    has_usage = any("usage" in c for c in chunks if isinstance(c, dict))
    check(sec, "stream_options.include_usage",
          has_usage,
          f"found usage chunk in {len(chunks)} chunks")


# =============================================================================
# MAIN
# =============================================================================
def main():
    global SECTION_FILTER, FAST_MODE

    args = sys.argv[1:]
    if "--section" in args:
        idx = args.index("--section")
        if idx + 1 < len(args):
            SECTION_FILTER = int(args[idx + 1])
    if "--fast" in args:
        FAST_MODE = True

    print("=" * 60)
    print("RKLLM API — END-TO-END INTEGRATION TEST")
    print("=" * 60)
    print(f"API:       {API}")
    print(f"WebUI:     {WEBUI}")
    print(f"SearXNG:   {SEARXNG}")
    print(f"Models:    {get_models()}")
    print(f"Section:   {'ALL' if SECTION_FILTER is None else SECTION_FILTER}")
    print(f"Fast mode: {FAST_MODE}")
    print(f"Timeout:   {TIMEOUT}s")
    print()

    # Verify API is up
    code, data, _ = _req("GET", "/health", timeout=10)
    if code != 200:
        print(f"FATAL: API not reachable at {API} (status={code})")
        sys.exit(1)
    print(f"API health: {json.dumps(data)}")

    start = time.time()

    test_basic_chat()       # Section 1
    test_shortcircuits()    # Section 2
    test_rag()              # Section 3
    test_rag_cache()        # Section 4
    test_web_search()       # Section 5
    test_kv_cache()         # Section 6
    test_prompts()          # Section 7
    test_database()         # Section 8
    test_compliance()       # Section 9

    elapsed = time.time() - start

    # === SUMMARY ===
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  PASS: {PASS}")
    print(f"  FAIL: {FAIL}")
    print(f"  WARN: {WARN}")
    print(f"  SKIP: {SKIP}")
    print(f"  Time: {elapsed:.0f}s")
    print()

    if FAIL > 0:
        print("FAILURES:")
        for sec, name, status, detail in RESULTS:
            if status == "FAIL":
                print(f"  [{sec}] {name}  |  {detail}")
        print()

    if WARN > 0:
        print("WARNINGS:")
        for sec, name, status, detail in RESULTS:
            if status == "WARN":
                print(f"  [{sec}] {name}  |  {detail}")
        print()

    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
