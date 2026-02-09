"""
RKLLM API Server — Comprehensive Section-by-Section Diagnostic Test
====================================================================
Tests every major code section of api.py through the API surface.
Run from any machine with network access to the Orange Pi.

Usage:
    python diagnostic_test.py                # Run all sections
    python diagnostic_test.py --section 5    # Run only section 5
    python diagnostic_test.py --skip-vl      # Skip VL tests (faster)

Output is designed to be copy-pasted for analysis.
"""
import base64, hashlib, io, json, re, struct, sys, time, threading
import urllib.request, urllib.error

# =============================================================================
# CONFIGURATION
# =============================================================================
API = "http://192.168.2.180:8000"
TIMEOUT = 180
VL_TIMEOUT = 300

# =============================================================================
# TEST FRAMEWORK
# =============================================================================
PASS = 0
FAIL = 0
SKIP = 0
RESULTS = []  # (section, test_name, status, detail)


def _req(method, path, body=None, timeout=TIMEOUT):
    """Send HTTP request, return (status_code, parsed_json_or_None, raw_body)."""
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw), raw
            except json.JSONDecodeError:
                return resp.status, None, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw), raw
        except json.JSONDecodeError:
            return e.code, None, raw
    except Exception as e:
        return 0, None, str(e)


def _stream_req(path, body, timeout=TIMEOUT):
    """Send streaming request, return (chunks_list, content_str, final_chunk_or_None)."""
    url = f"{API}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    chunks = []
    content = ""
    reasoning = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                line = line.decode().strip()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        chunks.append("[DONE]")
                        break
                    try:
                        obj = json.loads(payload)
                        chunks.append(obj)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        content += delta.get("content", "")
                        reasoning += delta.get("reasoning_content", "")
                    except json.JSONDecodeError:
                        chunks.append({"_raw": payload})
            return chunks, content, reasoning
    except Exception as e:
        return chunks, content, str(e)


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


def skip(section, name, reason=""):
    global SKIP
    SKIP += 1
    line = f"  [SKIP] {name}"
    if reason:
        line += f"  |  {reason}"
    print(line)
    RESULTS.append((section, name, "SKIP", reason))


def section_header(num, title):
    print(f"\n{'='*70}")
    print(f"  SECTION {num}: {title}")
    print(f"{'='*70}")


# =============================================================================
# HELPERS
# =============================================================================

def _make_test_image(width=64, height=64):
    """Create a minimal valid PNG image (red square) as bytes."""
    # Build raw RGBA data
    raw = b''
    for y in range(height):
        raw += b'\x00'  # filter byte
        for x in range(width):
            raw += b'\xff\x00\x00\xff'  # RGBA red

    import zlib
    def _chunk(chunk_type, data):
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xffffffff
        return struct.pack('>I', len(data)) + c + struct.pack('>I', crc)

    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
    compressed = zlib.compress(raw)

    png = b'\x89PNG\r\n\x1a\n'
    png += _chunk(b'IHDR', ihdr_data)
    png += _chunk(b'IDAT', compressed)
    png += _chunk(b'IEND', b'')
    return png


def _make_base64_image():
    """Return a base64 data URI of a small test image."""
    img_bytes = _make_test_image()
    b64 = base64.b64encode(img_bytes).decode()
    return f"data:image/png;base64,{b64}"


# =============================================================================
# SECTION 1: SERVER CONNECTIVITY & HEALTH
# =============================================================================
def test_section_1():
    section_header(1, "SERVER CONNECTIVITY & HEALTH ENDPOINT")

    # 1a: Basic connectivity
    status, data, _ = _req("GET", "/health")
    check(1, "Server reachable", status == 200,
          f"status={status}")

    if status != 200:
        print("  >>> CRITICAL: Server unreachable. Remaining tests will fail.")
        return

    # 1b: Health response structure
    check(1, "Health has 'status' field", data.get("status") == "ok",
          f"status={data.get('status')}")
    check(1, "Health has 'models_available'", isinstance(data.get("models_available"), int),
          f"models_available={data.get('models_available')}")
    check(1, "Health has 'current_model'", "current_model" in data,
          f"current_model={data.get('current_model')}")
    check(1, "Health has 'model_loaded'", "model_loaded" in data,
          f"model_loaded={data.get('model_loaded')}")
    check(1, "Health has 'active_request'", "active_request" in data,
          f"active_request={data.get('active_request')}")

    # 1c: VL model info in health
    vl_info = data.get("vl_model")
    if vl_info is not None:
        check(1, "VL info has 'model'", "model" in vl_info,
              f"vl_model={vl_info}")
        check(1, "VL info has 'encoder_loaded'", "encoder_loaded" in vl_info,
              f"encoder_loaded={vl_info.get('encoder_loaded')}")
    else:
        check(1, "VL info present (None=no VL loaded)", True,
              "No VL model loaded (expected if no VL request sent yet)")

    # 1d: No active request at idle
    check(1, "No active request at idle", data.get("active_request") is None,
          f"active_request={data.get('active_request')}")

    # 1e: System fingerprint present
    print(f"  [INFO] models_available={data.get('models_available')}, "
          f"current_model={data.get('current_model')}, "
          f"model_loaded={data.get('model_loaded')}")


# =============================================================================
# SECTION 2: MODEL DETECTION & LISTING
# =============================================================================
def test_section_2():
    section_header(2, "MODEL DETECTION & /v1/models LISTING")

    # 2a: /v1/models
    status, data, _ = _req("GET", "/v1/models")
    check(2, "/v1/models returns 200", status == 200,
          f"status={status}")

    if status != 200:
        return

    check(2, "Response has 'data' array", isinstance(data.get("data"), list),
          f"type={type(data.get('data'))}")
    check(2, "Response has 'object'='list'", data.get("object") == "list",
          f"object={data.get('object')}")

    models = data.get("data", [])
    model_ids = [m.get("id") for m in models]
    check(2, "At least 1 model detected", len(models) >= 1,
          f"count={len(models)}")

    # 2b: Model entry structure
    if models:
        m = models[0]
        check(2, "Model has 'id'", "id" in m, f"id={m.get('id')}")
        check(2, "Model has 'object'='model'", m.get("object") == "model",
              f"object={m.get('object')}")
        check(2, "Model has 'owned_by'='rkllm'", m.get("owned_by") == "rkllm",
              f"owned_by={m.get('owned_by')}")
        check(2, "Model has 'created' (int)", isinstance(m.get("created"), int),
              f"created={m.get('created')}")

    # 2c: Check for expected models
    print(f"  [INFO] Models detected: {model_ids}")

    # 2d: Also check /models (no /v1 prefix)
    status2, data2, _ = _req("GET", "/models")
    check(2, "/models (no prefix) also works", status2 == 200,
          f"status={status2}")

    return model_ids


# =============================================================================
# SECTION 3: ALIAS GENERATION & RESOLUTION
# =============================================================================
def test_section_3(model_ids):
    section_header(3, "ALIAS GENERATION & MODEL RESOLUTION")

    if not model_ids:
        skip(3, "Alias tests", "No models detected")
        return

    # 3a: Test that a known model is resolvable by full name
    test_model = model_ids[0]
    status, data, _ = _req("POST", "/v1/chat/completions", {
        "model": test_model,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 5,
    })
    check(3, f"Full name '{test_model}' resolves", status == 200,
          f"status={status}")

    # Wait for generation
    time.sleep(3)

    # 3b: Test unknown model returns 404
    status, data, _ = _req("POST", "/v1/chat/completions", {
        "model": "nonexistent-model-xyz",
        "messages": [{"role": "user", "content": "test"}],
    })
    check(3, "Unknown model returns 404", status == 404,
          f"status={status}, error={data.get('error', {}).get('message', '') if data else 'N/A'}")

    # 3c: Test alias if we can derive one (first part of first model name)
    parts = test_model.split('-')
    if len(parts) >= 2:
        alias_candidate = parts[0]
        status, data, _ = _req("POST", "/v1/chat/completions", {
            "model": alias_candidate,
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5,
        })
        resolved_model = data.get("model", "") if data else ""
        # Either 200 (alias works) or 404 (alias doesn't exist for this model)
        check(3, f"Alias '{alias_candidate}' tested",
              status in (200, 404),
              f"status={status}, resolved_to={resolved_model}")
        if status == 200:
            time.sleep(3)
    else:
        skip(3, "Alias test", "Model name has no dashes for alias derivation")

    print(f"  [INFO] Available model IDs for alias testing: {model_ids}")


# =============================================================================
# SECTION 4: ERROR HANDLING & VALIDATION
# =============================================================================
def test_section_4():
    section_header(4, "ERROR HANDLING & INPUT VALIDATION")

    # 4a: No body
    status, data, _ = _req("POST", "/v1/chat/completions")
    check(4, "No body => 400", status == 400,
          f"status={status}")

    # 4b: Empty messages
    status, data, _ = _req("POST", "/v1/chat/completions", {
        "model": "qwen3-1.7b",
        "messages": [],
    })
    check(4, "Empty messages => 400", status == 400,
          f"status={status}")

    # 4c: Invalid messages type
    status, data, _ = _req("POST", "/v1/chat/completions", {
        "model": "qwen3-1.7b",
        "messages": "not an array",
    })
    check(4, "Invalid messages type => 400", status == 400,
          f"status={status}")

    # 4d: Missing model
    status, data, _ = _req("POST", "/v1/chat/completions", {
        "model": "",
        "messages": [{"role": "user", "content": "test"}],
    })
    check(4, "Empty model name => 404", status == 404,
          f"status={status}")

    # 4e: Bad base64 image (should be 400, NOT 200 fallthrough)
    bad_b64_body = {
        "model": "qwen3-1.7b",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,NOT_VALID_BASE64!!!"}},
                {"type": "text", "text": "What is this?"},
            ]
        }]
    }
    status, data, _ = _req("POST", "/v1/chat/completions", bad_b64_body)
    check(4, "Bad base64 image => 400 (not 200 fallthrough)",
          status == 400,
          f"status={status}, msg={data.get('error', {}).get('message', '')[:80] if data else 'N/A'}")
    if status == 200:
        print("  >>> BUG: Bad base64 fell through to text path (old code deployed?)")

    # 4f: No JSON body
    try:
        url = f"{API}/v1/chat/completions"
        req = urllib.request.Request(url, data=b"not json", method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    except Exception:
        status = 0
    check(4, "Non-JSON body => 400", status == 400,
          f"status={status}")


# =============================================================================
# SECTION 5: TEXT GENERATION (NON-STREAMING)
# =============================================================================
def test_section_5(model_ids):
    section_header(5, "TEXT GENERATION — NON-STREAMING")

    if not model_ids:
        skip(5, "All text gen tests", "No models")
        return

    # Find a text model (not VL)
    text_model = None
    for mid in model_ids:
        if 'deepseekocr' not in mid.lower():
            text_model = mid
            break
    if not text_model:
        text_model = model_ids[0]

    # 5a: Basic non-streaming generation
    body = {
        "model": text_model,
        "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        "stream": False,
    }
    t0 = time.time()
    status, data, _ = _req("POST", "/v1/chat/completions", body)
    elapsed = time.time() - t0
    check(5, "Non-stream text gen status=200", status == 200,
          f"status={status}, elapsed={elapsed:.1f}s")

    if status != 200:
        print(f"  >>> Error: {data}")
        return

    # 5b: Response structure
    check(5, "Has 'id' field", "id" in data, f"id={data.get('id')}")
    check(5, "Has 'object'='chat.completion'", data.get("object") == "chat.completion",
          f"object={data.get('object')}")
    check(5, "Has 'model' field", "model" in data, f"model={data.get('model')}")
    check(5, "Has 'created' (int)", isinstance(data.get("created"), int),
          f"created={data.get('created')}")
    check(5, "Has 'system_fingerprint'", "system_fingerprint" in data,
          f"fp={data.get('system_fingerprint')}")
    check(5, "system_fingerprint starts with 'rkllm'",
          str(data.get("system_fingerprint", "")).startswith("rkllm"),
          f"fp={data.get('system_fingerprint')}")

    # 5c: Choices structure
    choices = data.get("choices", [])
    check(5, "Has choices array", len(choices) >= 1, f"len={len(choices)}")
    if choices:
        c = choices[0]
        check(5, "Choice has 'index'=0", c.get("index") == 0, f"index={c.get('index')}")
        check(5, "Choice has 'finish_reason'", c.get("finish_reason") in ("stop", "length"),
              f"finish_reason={c.get('finish_reason')}")
        msg = c.get("message", {})
        check(5, "Message has 'role'='assistant'", msg.get("role") == "assistant",
              f"role={msg.get('role')}")
        check(5, "Message has 'content' (non-empty)", bool(msg.get("content")),
              f"content_len={len(msg.get('content', ''))}")
        content = msg.get("content", "")
        print(f"  [INFO] Response ({len(content)} chars): {content[:150]}...")

    # 5d: Usage stats
    usage = data.get("usage", {})
    check(5, "Has usage.prompt_tokens", isinstance(usage.get("prompt_tokens"), int),
          f"prompt_tokens={usage.get('prompt_tokens')}")
    check(5, "Has usage.completion_tokens", isinstance(usage.get("completion_tokens"), int),
          f"completion_tokens={usage.get('completion_tokens')}")
    check(5, "Has usage.total_tokens", isinstance(usage.get("total_tokens"), int),
          f"total_tokens={usage.get('total_tokens')}")
    check(5, "total = prompt + completion",
          usage.get("total_tokens") == usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
          f"{usage.get('total_tokens')} == {usage.get('prompt_tokens', 0)} + {usage.get('completion_tokens', 0)}")

    # 5e: Model in response matches requested
    check(5, "Response model matches requested",
          data.get("model") == text_model,
          f"requested={text_model}, got={data.get('model')}")

    return text_model


# =============================================================================
# SECTION 6: TEXT GENERATION (STREAMING)
# =============================================================================
def test_section_6(text_model):
    section_header(6, "TEXT GENERATION — STREAMING + SSE FORMAT")

    if not text_model:
        skip(6, "All streaming tests", "No text model")
        return

    time.sleep(3)  # Allow previous request to fully clean up

    body = {
        "model": text_model,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "stream": True,
    }
    chunks, content, reasoning = _stream_req("/v1/chat/completions", body)

    # 6a: Got chunks
    check(6, "Received SSE chunks", len(chunks) >= 3,
          f"chunk_count={len(chunks)}")

    # 6b: First non-DONE chunk structure
    first_data_chunks = [c for c in chunks if isinstance(c, dict) and "_raw" not in c]
    if first_data_chunks:
        fc = first_data_chunks[0]
        check(6, "Chunk has 'id'", "id" in fc, f"keys={list(fc.keys())}")
        check(6, "Chunk has 'object'='chat.completion.chunk'",
              fc.get("object") == "chat.completion.chunk",
              f"object={fc.get('object')}")
        check(6, "Chunk has 'model'", "model" in fc, f"model={fc.get('model')}")
        check(6, "Chunk model matches requested", fc.get("model") == text_model,
              f"requested={text_model}, got={fc.get('model')}")
        check(6, "Chunk has 'system_fingerprint'", "system_fingerprint" in fc,
              f"fp={fc.get('system_fingerprint')}")

    # 6c: First chunk has role
    if first_data_chunks:
        fc_delta = first_data_chunks[0].get("choices", [{}])[0].get("delta", {})
        check(6, "First chunk delta has 'role'='assistant'",
              fc_delta.get("role") == "assistant",
              f"delta={fc_delta}")

    # 6d: Last data chunk has finish_reason=stop
    stop_chunks = [c for c in chunks if isinstance(c, dict)
                   and c.get("choices", [{}])[0].get("finish_reason") == "stop"]
    check(6, "Has finish_reason='stop' chunk", len(stop_chunks) >= 1,
          f"stop_chunks={len(stop_chunks)}")

    # 6e: [DONE] sentinel
    check(6, "Stream ends with [DONE]", "[DONE]" in chunks,
          f"last={chunks[-1] if chunks else 'none'}")

    # 6f: Content accumulated
    check(6, "Streamed content non-empty", len(content) > 0,
          f"content_len={len(content)}")
    print(f"  [INFO] Streamed content ({len(content)} chars): {content[:150]}...")

    # 6g: Streaming with include_usage
    time.sleep(3)
    body_usage = {
        "model": text_model,
        "messages": [{"role": "user", "content": "Say OK."}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    chunks2, content2, _ = _stream_req("/v1/chat/completions", body_usage)
    usage_chunks = [c for c in chunks2 if isinstance(c, dict) and "usage" in c and c.get("usage")]
    check(6, "include_usage: usage chunk present", len(usage_chunks) >= 1,
          f"usage_chunks={len(usage_chunks)}")
    if usage_chunks:
        u = usage_chunks[-1].get("usage", {})
        check(6, "Usage has prompt_tokens", isinstance(u.get("prompt_tokens"), int),
              f"prompt_tokens={u.get('prompt_tokens')}")
        check(6, "Usage has completion_tokens", isinstance(u.get("completion_tokens"), int),
              f"completion_tokens={u.get('completion_tokens')}")


# =============================================================================
# SECTION 7: THINK TAG PARSING
# =============================================================================
def test_section_7(text_model):
    section_header(7, "THINK TAG PARSING (reasoning_content)")

    if not text_model:
        skip(7, "Think tag tests", "No text model")
        return

    time.sleep(3)

    # 7a: Trigger thinking mode (some models emit <think> tags)
    body = {
        "model": text_model,
        "messages": [{"role": "user", "content": "Think step by step: what is 15 * 17?"}],
        "stream": True,
    }
    chunks, content, reasoning = _stream_req("/v1/chat/completions", body)

    has_reasoning = len(reasoning) > 0
    check(7, "Stream completed", "[DONE]" in chunks,
          f"chunks={len(chunks)}")
    check(7, "Content generated", len(content) > 0,
          f"content_len={len(content)}")

    if has_reasoning:
        # Model supports thinking
        reasoning_chunks = [c for c in chunks if isinstance(c, dict)
                           and c.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")]
        check(7, "reasoning_content in SSE chunks", len(reasoning_chunks) > 0,
              f"reasoning_chunk_count={len(reasoning_chunks)}")
        print(f"  [INFO] Reasoning ({len(reasoning)} chars): {reasoning[:100]}...")
        print(f"  [INFO] Content ({len(content)} chars): {content[:100]}...")
    else:
        print(f"  [INFO] Model did not emit <think> tags (normal for non-thinking models)")
        check(7, "No reasoning = content only (valid)", len(content) > 0,
              f"content_len={len(content)}")


# =============================================================================
# SECTION 8: KV CACHE & INCREMENTAL MODE
# =============================================================================
def test_section_8(text_model):
    section_header(8, "KV CACHE TRACKING & INCREMENTAL MODE")

    if not text_model:
        skip(8, "KV cache tests", "No text model")
        return

    time.sleep(3)

    # 8a: First message (KV RESET expected)
    body1 = {
        "model": text_model,
        "messages": [{"role": "user", "content": "My name is DiagBot. Remember that."}],
        "stream": False,
    }
    status1, data1, _ = _req("POST", "/v1/chat/completions", body1)
    check(8, "Turn 1 (KV reset) status=200", status1 == 200,
          f"status={status1}")

    if status1 != 200:
        return

    asst_reply = data1.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"  [INFO] Turn 1 reply: {asst_reply[:100]}...")

    time.sleep(2)

    # 8b: Second message (KV INCREMENTAL expected — same user history prefix)
    body2 = {
        "model": text_model,
        "messages": [
            {"role": "user", "content": "My name is DiagBot. Remember that."},
            {"role": "assistant", "content": asst_reply},
            {"role": "user", "content": "What is my name?"},
        ],
        "stream": False,
    }
    status2, data2, _ = _req("POST", "/v1/chat/completions", body2)
    check(8, "Turn 2 (KV incremental) status=200", status2 == 200,
          f"status={status2}")

    if status2 == 200:
        reply2 = data2.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Check if model remembered the name (tests KV cache retention)
        has_name = "diagbot" in reply2.lower() or "diag" in reply2.lower()
        check(8, "Model remembers context from Turn 1",
              has_name,
              f"reply={reply2[:100]}...")
        print(f"  [INFO] Turn 2 reply: {reply2[:100]}...")


# =============================================================================
# SECTION 9: MODEL SWITCHING & LOAD/UNLOAD
# =============================================================================
def test_section_9(model_ids):
    section_header(9, "MODEL SWITCHING, SELECT, & UNLOAD")

    if not model_ids or len(model_ids) < 1:
        skip(9, "Model switching", "Need at least 1 model")
        return

    time.sleep(3)

    # 9a: Explicit model select (warm-up)
    test_model = model_ids[0]
    status, data, _ = _req("POST", "/v1/models/select", {"model": test_model})
    check(9, f"Select '{test_model}' => 200", status == 200,
          f"status={status}")

    # 9b: Health shows loaded model
    status, data, _ = _req("GET", "/health")
    if status == 200:
        check(9, "Health shows loaded model after select",
              data.get("current_model") == test_model,
              f"current={data.get('current_model')}, expected={test_model}")
        check(9, "model_loaded=True", data.get("model_loaded") == True,
              f"model_loaded={data.get('model_loaded')}")

    # 9c: Unload model
    time.sleep(1)
    status, data, _ = _req("POST", "/v1/models/unload")
    check(9, "Unload => 200", status == 200,
          f"status={status}")
    if status == 200:
        check(9, "Unload response has 'unloaded'",
              "unloaded" in (data or {}),
              f"data={data}")

    # 9d: Health shows no model after unload
    time.sleep(1)
    status, data, _ = _req("GET", "/health")
    if status == 200:
        check(9, "No model loaded after unload",
              data.get("current_model") is None,
              f"current={data.get('current_model')}")
        check(9, "model_loaded=False after unload",
              data.get("model_loaded") == False,
              f"model_loaded={data.get('model_loaded')}")

    # 9e: Select unknown model => 404
    status, data, _ = _req("POST", "/v1/models/select", {"model": "nonexistent"})
    check(9, "Select unknown model => 404", status == 404,
          f"status={status}")

    # 9f: Unload when nothing loaded
    status, data, _ = _req("POST", "/v1/models/unload")
    check(9, "Unload when empty => 200 (no-op)", status == 200,
          f"status={status}")


# =============================================================================
# SECTION 10: CONCURRENT REQUEST REJECTION
# =============================================================================
def test_section_10(text_model):
    section_header(10, "CONCURRENT REQUEST REJECTION (SINGLE-NPU GUARD)")

    if not text_model:
        skip(10, "Concurrent tests", "No text model")
        return

    time.sleep(3)

    # Start a long-running request
    results = {"first_status": None, "second_status": None}

    def long_request():
        s, d, _ = _req("POST", "/v1/chat/completions", {
            "model": text_model,
            "messages": [{"role": "user", "content": "Write a 200-word essay about clouds."}],
            "stream": False,
        }, timeout=300)
        results["first_status"] = s

    t = threading.Thread(target=long_request, daemon=True)
    t.start()
    time.sleep(3)  # Give it time to start generating

    # Try a second request while first is running
    status, data, _ = _req("POST", "/v1/chat/completions", {
        "model": text_model,
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": False,
    }, timeout=10)
    results["second_status"] = status

    check(10, "Concurrent request rejected => 503",
          results["second_status"] == 503,
          f"status={results['second_status']}")
    if data and data.get("error"):
        print(f"  [INFO] Rejection message: {data['error'].get('message', '')[:100]}")

    # Wait for first to complete
    t.join(timeout=120)
    check(10, "First request completed successfully",
          results["first_status"] == 200,
          f"first_status={results['first_status']}")


# =============================================================================
# SECTION 11: RAG PIPELINE (PROMPT BUILDING)
# =============================================================================
def test_section_11(text_model):
    section_header(11, "RAG PIPELINE (build_prompt, cleaning, scoring)")

    if not text_model:
        skip(11, "RAG tests", "No text model")
        return

    time.sleep(3)

    # 11a: RAG-style message with <source> tags (Open WebUI format)
    rag_system = (
        "Here is some reference information:\n"
        "<source id='1' title='Test Article' url='https://example.com'>\n"
        "The capital of France is Paris. Paris is known for the Eiffel Tower, "
        "which was built in 1889 for the World's Fair. The tower stands at "
        "330 meters tall and attracts millions of visitors each year. "
        "Paris is also famous for the Louvre Museum, Notre-Dame Cathedral, "
        "and the Champs-Élysées boulevard. The city has a population of "
        "approximately 2.1 million people in the city proper.\n"
        "</source>\n"
        "### Task: Answer the following question using the above reference."
    )
    body = {
        "model": text_model,
        "messages": [
            {"role": "system", "content": rag_system},
            {"role": "user", "content": "What is the capital of France and how tall is the Eiffel Tower?"},
        ],
        "stream": False,
    }
    status, data, _ = _req("POST", "/v1/chat/completions", body)
    check(11, "RAG query status=200", status == 200,
          f"status={status}")

    if status == 200:
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        has_paris = "paris" in content.lower()
        has_tower = "330" in content or "eiffel" in content.lower() or "tower" in content.lower()
        check(11, "Response mentions Paris", has_paris,
              f"content[:100]={content[:100]}")
        check(11, "Response uses reference data (Eiffel/330)", has_tower,
              f"content[:150]={content[:150]}")
        print(f"  [INFO] RAG response: {content[:200]}...")

    time.sleep(3)

    # 11b: RAG with boilerplate-heavy content (tests _clean_web_content)
    rag_boilerplate = (
        "<source id='1' title='News' url='https://example.com'>\n"
        "Sign in to continue\nCookie Policy\nSubscribe Now\n"
        "Accept All Cookies\nSkip to content\n"
        "Privacy Policy | Terms of Service | All Rights Reserved\n"
        "\n"
        "The weather in London today is partly cloudy with temperatures "
        "reaching 18 degrees Celsius. The forecast predicts rain in the "
        "afternoon with winds picking up to 25 km/h from the southwest. "
        "Tomorrow is expected to be sunny with temperatures rising to "
        "22 degrees Celsius. The humidity level is currently at 65%.\n"
        "\n"
        "Follow us on Twitter | Share on Facebook | Pin It\n"
        "Related Articles | You May Also Like | Trending Now\n"
        "Download the App | Powered by WordPress\n"
        "</source>"
    )
    body2 = {
        "model": text_model,
        "messages": [
            {"role": "system", "content": f"Reference information:\n{rag_boilerplate}"},
            {"role": "user", "content": "What is the weather in London?"},
        ],
        "stream": False,
    }
    status2, data2, _ = _req("POST", "/v1/chat/completions", body2)
    check(11, "RAG with boilerplate status=200", status2 == 200,
          f"status={status2}")
    if status2 == 200:
        content2 = data2.get("choices", [{}])[0].get("message", {}).get("content", "")
        check(11, "Response mentions weather/temperature",
              any(w in content2.lower() for w in ("weather", "temperature", "cloudy", "celsius", "18")),
              f"content[:100]={content2[:100]}")

    time.sleep(3)

    # 11c: RAG skip detection (short follow-up after assistant turn)
    body3 = {
        "model": text_model,
        "messages": [
            {"role": "system", "content": rag_system},
            {"role": "user", "content": "Tell me about France"},
            {"role": "assistant", "content": "France is a beautiful country."},
            {"role": "user", "content": "thanks"},
        ],
        "stream": False,
    }
    status3, data3, _ = _req("POST", "/v1/chat/completions", body3)
    check(11, "RAG skip (conversational follow-up) status=200", status3 == 200,
          f"status={status3}")
    # The "thanks" should trigger RAG skip and fall to normal mode


# =============================================================================
# SECTION 12: RAG RESPONSE CACHE
# =============================================================================
def test_section_12(text_model):
    section_header(12, "RAG RESPONSE CACHE")

    if not text_model:
        skip(12, "RAG cache tests", "No text model")
        return

    time.sleep(3)

    # Send same RAG query twice — second should be faster (cached)
    rag_sys = (
        "<source id='1' title='Cache Test'>\n"
        "The Great Wall of China is approximately 21,196 kilometers long. "
        "It was built over many centuries, with the most well-known sections "
        "built during the Ming Dynasty (1368-1644). The wall stretches across "
        "northern China and is a UNESCO World Heritage Site.\n"
        "</source>"
    )
    body = {
        "model": text_model,
        "messages": [
            {"role": "system", "content": rag_sys},
            {"role": "user", "content": "How long is the Great Wall of China?"},
        ],
        "stream": False,
    }

    # First call (should generate)
    t0 = time.time()
    status1, data1, _ = _req("POST", "/v1/chat/completions", body)
    time1 = time.time() - t0
    check(12, "RAG query 1 (generate) status=200", status1 == 200,
          f"status={status1}, time={time1:.1f}s")

    content1 = ""
    if status1 == 200:
        content1 = data1.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"  [INFO] Response 1 ({time1:.1f}s): {content1[:100]}...")

    time.sleep(2)

    # Second call (should be cached)
    t0 = time.time()
    status2, data2, _ = _req("POST", "/v1/chat/completions", body)
    time2 = time.time() - t0
    check(12, "RAG query 2 (cache) status=200", status2 == 200,
          f"status={status2}, time={time2:.1f}s")

    if status2 == 200:
        content2 = data2.get("choices", [{}])[0].get("message", {}).get("content", "")
        check(12, "Cached response matches original", content1 == content2,
              f"match={content1 == content2}, len1={len(content1)}, len2={len(content2)}")
        check(12, "Cache response faster than generation", time2 < time1 * 0.5,
              f"gen={time1:.1f}s, cache={time2:.1f}s, speedup={time1/max(time2,0.01):.1f}x")
        print(f"  [INFO] Response 2 ({time2:.1f}s): {content2[:100]}...")


# =============================================================================
# SECTION 13: CONTENT NORMALIZATION
# =============================================================================
def test_section_13(text_model):
    section_header(13, "CONTENT NORMALIZATION (multimodal content arrays)")

    if not text_model:
        skip(13, "Normalization tests", "No text model")
        return

    time.sleep(3)

    # Send message with content as array of text parts (no images)
    body = {
        "model": text_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Say hello."},
            ]
        }],
        "stream": False,
    }
    status, data, _ = _req("POST", "/v1/chat/completions", body)
    check(13, "Array content (text-only) => 200", status == 200,
          f"status={status}")
    if status == 200:
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        check(13, "Generated content not empty", len(content) > 0,
              f"content_len={len(content)}")


# =============================================================================
# SECTION 14: VL AUTO-ROUTING & IMAGE PROCESSING
# =============================================================================
def test_section_14(model_ids):
    section_header(14, "VL AUTO-ROUTING & IMAGE PROCESSING")

    vl_models = [m for m in model_ids if 'deepseekocr' in m.lower()
                 or 'vl' in m.lower() or 'vision' in m.lower()]
    if not vl_models:
        skip(14, "All VL tests", "No VL model detected")
        return

    vl_model = vl_models[0]
    time.sleep(3)

    img_uri = _make_base64_image()

    # 14a: VL non-streaming
    body = {
        "model": "qwen3-1.7b",  # Request text model — should auto-route to VL
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_uri}},
                {"type": "text", "text": "What color is this image?"},
            ]
        }],
        "stream": False,
    }
    t0 = time.time()
    status, data, _ = _req("POST", "/v1/chat/completions", body, timeout=VL_TIMEOUT)
    elapsed = time.time() - t0
    check(14, "VL non-stream auto-route => 200", status == 200,
          f"status={status}, elapsed={elapsed:.1f}s")

    if status == 200:
        resp_model = data.get("model", "")
        check(14, "Response model is VL model (not text model)",
              resp_model.lower() != "qwen3-1.7b",
              f"requested=qwen3-1.7b, got={resp_model}")
        check(14, "Response model is the detected VL model",
              resp_model.lower() in [v.lower() for v in vl_models],
              f"expected one of {vl_models}, got={resp_model}")

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        check(14, "VL response has content", len(content) > 0,
              f"content_len={len(content)}")
        has_color = any(c in content.lower() for c in ("red", "color", "image", "square", "block"))
        check(14, "VL response describes image", has_color,
              f"content[:100]={content[:100]}")
        print(f"  [INFO] VL response: {content[:200]}...")

    time.sleep(5)

    # 14b: VL streaming
    body_stream = dict(body)
    body_stream["stream"] = True
    chunks, content, reasoning = _stream_req("/v1/chat/completions", body_stream, timeout=VL_TIMEOUT)
    check(14, "VL streaming: got chunks", len(chunks) >= 3,
          f"chunks={len(chunks)}")
    check(14, "VL streaming: [DONE]", "[DONE]" in chunks,
          f"last={chunks[-1] if chunks else 'none'}")
    check(14, "VL streaming: content", len(content) > 0,
          f"content_len={len(content)}")

    # Check model name in stream chunks
    data_chunks = [c for c in chunks if isinstance(c, dict) and "model" in c]
    if data_chunks:
        stream_model = data_chunks[0].get("model", "")
        check(14, "VL stream model is VL model",
              stream_model.lower() in [v.lower() for v in vl_models],
              f"expected one of {vl_models}, got={stream_model}")

    time.sleep(5)

    # 14c: VL with no text (default prompt "Describe this image.")
    body_notext = {
        "model": "qwen3-1.7b",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_uri}},
            ]
        }],
        "stream": False,
    }
    status3, data3, _ = _req("POST", "/v1/chat/completions", body_notext, timeout=VL_TIMEOUT)
    check(14, "VL with no text prompt => 200", status3 == 200,
          f"status={status3}")
    if status3 == 200:
        content3 = data3.get("choices", [{}])[0].get("message", {}).get("content", "")
        check(14, "VL no-text: content generated", len(content3) > 0,
              f"content_len={len(content3)}")

    # 14d: Health shows VL state after VL request
    time.sleep(2)
    status_h, data_h, _ = _req("GET", "/health")
    if status_h == 200:
        vl_info = data_h.get("vl_model")
        check(14, "Health shows VL model loaded after VL request",
              vl_info is not None,
              f"vl_model={vl_info}")
        if vl_info:
            check(14, "VL encoder_loaded=True",
                  vl_info.get("encoder_loaded") == True,
                  f"encoder_loaded={vl_info.get('encoder_loaded')}")
            check(14, "VL llm_loaded=True",
                  vl_info.get("llm_loaded") == True,
                  f"llm_loaded={vl_info.get('llm_loaded')}")


# =============================================================================
# SECTION 15: TEXT AFTER VL (DUAL-MODEL ISOLATION)
# =============================================================================
def test_section_15(text_model):
    section_header(15, "TEXT AFTER VL (DUAL-MODEL ISOLATION)")

    if not text_model:
        skip(15, "Text-after-VL", "No text model")
        return

    time.sleep(3)

    # After VL tests, text model should still work (separate wrappers)
    body = {
        "model": text_model,
        "messages": [{"role": "user", "content": "What is 3+3? Reply with just the number."}],
        "stream": False,
    }
    status, data, _ = _req("POST", "/v1/chat/completions", body)
    check(15, "Text gen after VL tests => 200", status == 200,
          f"status={status}")
    if status == 200:
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        check(15, "Text response has content", len(content) > 0,
              f"content_len={len(content)}")
        check(15, "Response model is text model (not VL)",
              data.get("model") == text_model,
              f"model={data.get('model')}")
        print(f"  [INFO] Text-after-VL reply: {content[:100]}...")


# =============================================================================
# SECTION 16: ROUTE VARIANTS & EDGE CASES
# =============================================================================
def test_section_16():
    section_header(16, "ROUTE VARIANTS & EDGE CASES")

    # 16a: /chat/completions (no /v1 prefix)
    body = {
        "model": "qwen3-1.7b",
        "messages": [{"role": "user", "content": "Say OK"}],
        "stream": False,
    }
    status, data, _ = _req("POST", "/chat/completions", body)
    check(16, "/chat/completions (no /v1) works", status == 200,
          f"status={status}")

    time.sleep(3)

    # 16b: Multiple system messages
    body2 = {
        "model": "qwen3-1.7b",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say hello."},
        ],
        "stream": False,
    }
    status2, data2, _ = _req("POST", "/v1/chat/completions", body2)
    check(16, "Multiple system messages => 200", status2 == 200,
          f"status={status2}")

    time.sleep(3)

    # 16c: Content as integer (should be converted to string)
    body3 = {
        "model": "qwen3-1.7b",
        "messages": [
            {"role": "user", "content": 42},  # int, not string
        ],
        "stream": False,
    }
    status3, data3, _ = _req("POST", "/v1/chat/completions", body3)
    check(16, "Integer content => 200 (auto-converted)", status3 == 200,
          f"status={status3}")

    time.sleep(3)

    # 16d: Very long message (context limit test)
    long_text = "Hello. " * 500  # ~3500 chars, ~875 tokens
    body4 = {
        "model": "qwen3-1.7b",
        "messages": [{"role": "user", "content": long_text + "What did I say?"}],
        "stream": False,
    }
    status4, data4, _ = _req("POST", "/v1/chat/completions", body4)
    check(16, "Long message (3500+ chars) => 200", status4 == 200,
          f"status={status4}")


# =============================================================================
# SECTION 17: SYSTEM STATE CONSISTENCY
# =============================================================================
def test_section_17():
    section_header(17, "SYSTEM STATE CONSISTENCY (final health check)")

    time.sleep(2)

    status, data, _ = _req("GET", "/health")
    check(17, "Final health check => 200", status == 200,
          f"status={status}")
    if status == 200:
        check(17, "No active request (idle)", data.get("active_request") is None,
              f"active_request={data.get('active_request')}")
        print(f"  [INFO] Final state: model={data.get('current_model')}, "
              f"loaded={data.get('model_loaded')}, "
              f"vl={data.get('vl_model')}, "
              f"models={data.get('models_available')}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    global PASS, FAIL, SKIP

    skip_vl = "--skip-vl" in sys.argv
    target_section = None
    for arg in sys.argv[1:]:
        if arg.startswith("--section"):
            if "=" in arg:
                target_section = int(arg.split("=")[1])
            elif sys.argv.index(arg) + 1 < len(sys.argv):
                target_section = int(sys.argv[sys.argv.index(arg) + 1])

    print("=" * 70)
    print("  RKLLM API — COMPREHENSIVE DIAGNOSTIC TEST")
    print(f"  Target: {API}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if skip_vl:
        print("  Mode: --skip-vl (VL sections skipped)")
    if target_section:
        print(f"  Mode: --section {target_section} (single section)")
    print("=" * 70)

    start_time = time.time()
    model_ids = None
    text_model = None

    def should_run(n):
        return target_section is None or target_section == n

    # Section 1: Connectivity & Health
    if should_run(1):
        test_section_1()

    # Section 2: Model Detection
    if should_run(2) or model_ids is None:
        model_ids = test_section_2() or []

    # Section 3: Aliases
    if should_run(3):
        test_section_3(model_ids)

    # Section 4: Error Handling
    if should_run(4):
        test_section_4()

    # Section 5: Text Gen (non-streaming)
    if should_run(5) or text_model is None:
        text_model = test_section_5(model_ids)

    # Section 6: Text Gen (streaming)
    if should_run(6):
        test_section_6(text_model)

    # Section 7: Think Tags
    if should_run(7):
        test_section_7(text_model)

    # Section 8: KV Cache
    if should_run(8):
        test_section_8(text_model)

    # Section 9: Model Switching
    if should_run(9):
        test_section_9(model_ids)

    # Section 10: Concurrent Rejection
    if should_run(10):
        test_section_10(text_model)

    # Section 11: RAG Pipeline
    if should_run(11):
        test_section_11(text_model)

    # Section 12: RAG Cache
    if should_run(12):
        test_section_12(text_model)

    # Section 13: Content Normalization
    if should_run(13):
        test_section_13(text_model)

    # Section 14: VL Auto-Routing
    if should_run(14):
        if skip_vl:
            section_header(14, "VL AUTO-ROUTING & IMAGE PROCESSING")
            skip(14, "All VL tests", "Skipped via --skip-vl")
        else:
            test_section_14(model_ids)

    # Section 15: Text after VL
    if should_run(15):
        if skip_vl:
            section_header(15, "TEXT AFTER VL (DUAL-MODEL ISOLATION)")
            skip(15, "Text-after-VL", "Skipped via --skip-vl")
        else:
            test_section_15(text_model)

    # Section 16: Route Variants
    if should_run(16):
        test_section_16()

    # Section 17: Final State
    if should_run(17):
        test_section_17()

    # =================================================================
    # SUMMARY
    # =================================================================
    total_time = time.time() - start_time
    total = PASS + FAIL + SKIP
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  Total: {total}  |  PASS: {PASS}  |  FAIL: {FAIL}  |  SKIP: {SKIP}")
    print(f"  Time: {total_time:.1f}s")
    print()

    if FAIL > 0:
        print("  FAILURES:")
        for section, name, status, detail in RESULTS:
            if status == "FAIL":
                print(f"    Section {section}: {name}")
                if detail:
                    print(f"      Detail: {detail}")
        print()

    if SKIP > 0:
        print("  SKIPPED:")
        for section, name, status, detail in RESULTS:
            if status == "SKIP":
                print(f"    Section {section}: {name} — {detail}")
        print()

    # Version detection hint
    print("  DEPLOY CHECK:")
    print("    If Section 4 'Bad base64 image => 400' shows FAIL (got 200),")
    print("    the Orange Pi is running OLD code. Deploy commit d3537a1.")
    print("    If Section 14 VL model name shows 'qwen3-1.7b' instead of")
    print("    the VL model name, same issue — old code.")
    print()

    print("=" * 70)
    if FAIL == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {FAIL} TEST(S) FAILED — see details above")
    print("=" * 70)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
