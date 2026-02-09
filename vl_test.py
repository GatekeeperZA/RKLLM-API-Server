"""Comprehensive integration tests for RKLLM API Server.

Tests: health, models, aliases, model switching, VL auto-routing,
streaming, error handling, KV cache, concurrent rejection, and more.

Usage: python vl_test.py [complete|stream|all]
"""
import base64, json, struct, sys, time, urllib.request, urllib.error, threading

API = "http://192.168.2.180:8000"
PASS = 0
FAIL = 0
SKIP = 0


def _req(method, path, body=None, timeout=180):
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


def _stream_req(path, body, timeout=180):
    """Send streaming request, return list of parsed SSE chunks + final status."""
    url = f"{API}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    chunks = []
    content = ""
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
                    except json.JSONDecodeError:
                        chunks.append(payload)
            return 200, chunks, content
    except urllib.error.HTTPError as e:
        return e.code, [], e.read().decode()
    except Exception as e:
        return 0, [], str(e)


def _make_test_image_b64():
    """Create a 64x64 BMP with 4 colored quadrants, return base64 string."""
    w, h = 64, 64
    row_size = (w * 3 + 3) & ~3
    pds = row_size * h
    bmp = bytearray(b'BM')
    bmp += struct.pack('<I', 54 + pds) + b'\x00\x00\x00\x00' + struct.pack('<I', 54)
    bmp += struct.pack('<I', 40) + struct.pack('<i', w) + struct.pack('<i', h)
    bmp += struct.pack('<HH', 1, 24) + struct.pack('<I', 0) + struct.pack('<I', pds)
    bmp += struct.pack('<ii', 2835, 2835) + struct.pack('<II', 0, 0)
    for y in range(h):
        row = bytearray()
        for x in range(w):
            if x < 32 and y >= 32: row += bytes([0, 0, 255])
            elif x >= 32 and y >= 32: row += bytes([0, 255, 0])
            elif x < 32 and y < 32: row += bytes([255, 0, 0])
            else: row += bytes([0, 255, 255])
        while len(row) % 4:
            row += b'\x00'
        bmp += row
    return base64.b64encode(bytes(bmp)).decode()


def _chat(model, content, stream=False, max_tokens=100, **kwargs):
    """Shorthand for chat completions."""
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "stream": stream, "max_tokens": max_tokens, **kwargs}
    if stream:
        return _stream_req("/v1/chat/completions", body)
    return _req("POST", "/v1/chat/completions", body)


def _vl_chat(model, text, image_b64, stream=False, max_tokens=100):
    """Shorthand for VL chat with image."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/bmp;base64,{image_b64}"}},
            {"type": "text", "text": text},
        ]}],
        "stream": stream, "max_tokens": max_tokens,
    }
    if stream:
        return _stream_req("/v1/chat/completions", body, timeout=180)
    return _req("POST", "/v1/chat/completions", body, timeout=180)


def report(name, passed, detail=""):
    global PASS, FAIL
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS += 1
    else:
        FAIL += 1
    icon = "+" if passed else "!"
    det = f" -- {detail}" if detail else ""
    print(f"  [{icon}] {name}: {status}{det}")


def skip(name, reason=""):
    global SKIP
    SKIP += 1
    print(f"  [~] {name}: SKIP -- {reason}")


# =========================================================================
# TEST SUITES
# =========================================================================

def test_health():
    print("\n=== 1. HEALTH ENDPOINT ===")
    code, data, _ = _req("GET", "/health")
    report("GET /health returns 200", code == 200)
    report("status is ok", data and data.get("status") == "ok")
    report("models_available >= 1", data and data.get("models_available", 0) >= 1)
    report("has expected fields",
           data and all(k in data for k in ["current_model", "model_loaded", "vl_model"]),
           f"keys={list(data.keys()) if data else 'N/A'}")


def test_models_list():
    print("\n=== 2. MODEL LISTING ===")
    code, data, _ = _req("GET", "/v1/models")
    report("GET /v1/models returns 200", code == 200)
    report("object is 'list'", data and data.get("object") == "list")
    models = [m["id"] for m in data.get("data", [])] if data else []
    report("deepseekocr in model list", "deepseekocr" in models, f"models={models}")
    report("qwen3-1.7b in model list", "qwen3-1.7b" in models)
    report("all models have required fields",
           data and all(
               all(k in m for k in ["id", "object", "created", "owned_by"])
               for m in data.get("data", [])
           ))
    # Also test /models alias
    code2, _, _ = _req("GET", "/models")
    report("/models alias works", code2 == 200)


def test_model_aliases():
    print("\n=== 3. MODEL ALIAS RESOLUTION ===")
    # Test with a short alias - phi should resolve to phi-3-mini-4k-instruct
    code, data, _ = _req("POST", "/v1/models/select", {"model": "phi"})
    report("alias 'phi' resolves", code == 200,
           f"loaded={data.get('model') if data else 'N/A'}")
    if data:
        report("resolves to phi-3-mini-4k-instruct",
               data.get("model") == "phi-3-mini-4k-instruct")

    # Test qwen3-4b alias
    code, data, _ = _req("POST", "/v1/models/select", {"model": "qwen3-4b"})
    report("alias 'qwen3-4b' resolves", code == 200,
           f"loaded={data.get('model') if data else 'N/A'}")

    # Test invalid model
    code, data, _ = _req("POST", "/v1/models/select", {"model": "nonexistent-model-xyz"})
    report("invalid model returns 404", code == 404)


def test_text_generation():
    print("\n=== 4. TEXT GENERATION (NON-STREAMING) ===")
    code, data, _ = _chat("qwen3-1.7b", "Reply with only the word 'hello'.", max_tokens=50)
    report("returns 200", code == 200)
    if data:
        msg = data.get("choices", [{}])[0].get("message", {})
        report("has content", bool(msg.get("content")),
               f"content='{msg.get('content', '')[:80]}'")
        report("model field correct", data.get("model") == "qwen3-1.7b")
        report("has system_fingerprint", "system_fingerprint" in data)
        usage = data.get("usage", {})
        report("has usage stats",
               all(k in usage for k in ["prompt_tokens", "completion_tokens", "total_tokens"]),
               f"usage={usage}")
        report("finish_reason is 'stop'",
               data["choices"][0].get("finish_reason") == "stop")
    else:
        report("response parsed", False, "no data")


def test_text_streaming():
    print("\n=== 5. TEXT GENERATION (STREAMING) ===")
    code, chunks, content = _stream_req("/v1/chat/completions", {
        "model": "qwen3-1.7b",
        "messages": [{"role": "user", "content": "Say 'streaming works' and nothing else."}],
        "stream": True, "max_tokens": 50,
    })
    report("returns 200", code == 200)
    report("received chunks", len(chunks) > 1, f"chunk_count={len(chunks)}")
    report("ends with [DONE]", chunks and chunks[-1] == "[DONE]")
    report("has content", len(content) > 0, f"content='{content[:80]}'")
    # Check chunk structure
    if chunks and isinstance(chunks[0], dict):
        report("chunks have delta format",
               "delta" in chunks[0].get("choices", [{}])[0])
        report("object is chat.completion.chunk",
               chunks[0].get("object") == "chat.completion.chunk")


def test_streaming_usage():
    print("\n=== 6. STREAMING WITH USAGE ===")
    code, chunks, content = _stream_req("/v1/chat/completions", {
        "model": "qwen3-1.7b",
        "messages": [{"role": "user", "content": "Say 'usage test' only."}],
        "stream": True, "max_tokens": 50,
        "stream_options": {"include_usage": True},
    })
    report("returns 200", code == 200)
    # Last chunk before [DONE] should have usage
    real_chunks = [c for c in chunks if isinstance(c, dict)]
    if real_chunks:
        last = real_chunks[-1]
        has_usage = "usage" in last and last["usage"] is not None
        report("final chunk has usage", has_usage,
               f"usage={last.get('usage')}" if has_usage else "no usage in final chunk")
    else:
        report("final chunk has usage", False, "no chunks")


def test_model_switching():
    print("\n=== 7. MODEL SWITCHING ===")
    # Load qwen3-1.7b
    code1, data1, _ = _req("POST", "/v1/models/select", {"model": "qwen3-1.7b"})
    report("load qwen3-1.7b", code1 == 200)

    # Check health shows it loaded
    _, health1, _ = _req("GET", "/health")
    report("health shows qwen3-1.7b",
           health1 and health1.get("current_model") == "qwen3-1.7b")

    # Switch to phi
    code2, data2, _ = _req("POST", "/v1/models/select", {"model": "phi"})
    report("switch to phi", code2 == 200)

    _, health2, _ = _req("GET", "/health")
    report("health shows phi-3-mini-4k-instruct",
           health2 and health2.get("current_model") == "phi-3-mini-4k-instruct")

    # Generate with phi to confirm it works
    code3, data3, _ = _chat("phi", "Say 'phi works'.", max_tokens=30)
    report("phi generates text", code3 == 200 and data3 and
           bool(data3.get("choices", [{}])[0].get("message", {}).get("content")))


def test_model_unload():
    print("\n=== 8. MODEL UNLOAD ===")
    # First ensure a model is loaded
    _req("POST", "/v1/models/select", {"model": "qwen3-1.7b"})
    time.sleep(1)

    code, data, _ = _req("POST", "/v1/models/unload")
    report("unload returns 200", code == 200)
    report("unload confirms model",
           data and data.get("unloaded") == "qwen3-1.7b",
           f"response={data}")

    _, health, _ = _req("GET", "/health")
    report("health shows no model",
           health and health.get("current_model") is None and not health.get("model_loaded"))

    # Unload again (no model loaded)
    code2, data2, _ = _req("POST", "/v1/models/unload")
    report("unload when empty returns 200", code2 == 200)


def test_vl_non_streaming():
    print("\n=== 9. VL NON-STREAMING ===")
    img = _make_test_image_b64()
    code, data, raw = _vl_chat("deepseekocr", "What colors do you see?", img, stream=False, max_tokens=200)
    report("returns 200", code == 200, f"code={code}" if code != 200 else "")
    if code != 200:
        report("error detail", False, raw[:200])
        return
    if data:
        msg = data.get("choices", [{}])[0].get("message", {})
        report("has content", bool(msg.get("content")),
               f"len={len(msg.get('content', ''))}")
        report("model is deepseekocr", data.get("model") == "deepseekocr")
        report("finish_reason is stop",
               data["choices"][0].get("finish_reason") == "stop")

    # Check health shows VL model
    _, health, _ = _req("GET", "/health")
    report("health shows VL model loaded",
           health and health.get("vl_model") is not None,
           f"vl_model={health.get('vl_model') if health else 'N/A'}")


def test_vl_streaming():
    print("\n=== 10. VL STREAMING ===")
    img = _make_test_image_b64()
    code, chunks, content = _vl_chat("deepseekocr", "Describe this image briefly.", img,
                                      stream=True, max_tokens=150)
    report("returns 200", code == 200)
    report("received chunks", len(chunks) > 1, f"chunk_count={len(chunks)}")
    report("ends with [DONE]", chunks and chunks[-1] == "[DONE]")
    report("has content", len(content) > 10, f"content_len={len(content)}")


def test_vl_auto_routing():
    print("\n=== 11. VL AUTO-ROUTING ===")
    img = _make_test_image_b64()
    # Send image request with model=qwen3-1.7b — should auto-route to VL
    body = {
        "model": "qwen3-1.7b",  # text model requested, but image present
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/bmp;base64,{img}"}},
            {"type": "text", "text": "What is this?"},
        ]}],
        "stream": False, "max_tokens": 100,
    }
    code, data, _ = _req("POST", "/v1/chat/completions", body, timeout=180)
    report("auto-routes to VL (returns 200)", code == 200)
    if data:
        report("model in response is deepseekocr",
               data.get("model") == "deepseekocr",
               f"model={data.get('model')}")


def test_vl_no_text():
    print("\n=== 12. VL WITH NO TEXT (DEFAULT PROMPT) ===")
    img = _make_test_image_b64()
    # Send image with no text part — should use default "Describe this image."
    body = {
        "model": "deepseekocr",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/bmp;base64,{img}"}},
        ]}],
        "stream": False, "max_tokens": 100,
    }
    code, data, _ = _req("POST", "/v1/chat/completions", body, timeout=180)
    report("VL with no text returns 200", code == 200)
    if data:
        report("has content", bool(
            data.get("choices", [{}])[0].get("message", {}).get("content")))


def test_error_handling():
    print("\n=== 13. ERROR HANDLING ===")

    # Empty messages
    code, data, _ = _req("POST", "/v1/chat/completions",
                          {"model": "qwen3-1.7b", "messages": []})
    report("empty messages returns 400", code == 400)

    # Missing model
    code, data, _ = _req("POST", "/v1/chat/completions",
                          {"model": "fake-model-xyz", "messages": [{"role": "user", "content": "hi"}]})
    report("invalid model returns 404", code == 404)

    # Missing messages field
    code, data, _ = _req("POST", "/v1/chat/completions", {"model": "qwen3-1.7b"})
    report("missing messages returns 400", code == 400)

    # Non-JSON body
    req = urllib.request.Request(f"{API}/v1/chat/completions",
                                 data=b"not json", method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            report("non-JSON body returns 400", resp.status == 400)
    except urllib.error.HTTPError as e:
        report("non-JSON body returns 400", e.code == 400)

    # messages as non-array
    code, data, _ = _req("POST", "/v1/chat/completions",
                          {"model": "qwen3-1.7b", "messages": "hello"})
    report("messages as string returns 400", code == 400)

    # Bad base64 image
    body = {
        "model": "deepseekocr",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,NOT_VALID_BASE64!!!"}},
            {"type": "text", "text": "test"},
        ]}],
        "stream": False,
    }
    code, data, _ = _req("POST", "/v1/chat/completions", body, timeout=60)
    # Should return 400 since image was explicitly sent but couldn't be decoded
    report("bad base64 returns 400", code == 400,
           f"code={code}")


def test_text_after_vl():
    print("\n=== 14. TEXT AFTER VL (MODEL COEXISTENCE) ===")
    # After VL tests, verify text models still work (dual-model arch)
    code, data, _ = _chat("qwen3-1.7b", "Say 'text still works'.", max_tokens=30)
    report("text model works after VL", code == 200)
    if data:
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        report("got response content", len(content) > 0, f"content='{content[:60]}'")

    # Check health — both text and VL can be tracked
    _, health, _ = _req("GET", "/health")
    report("health still responds", health and health.get("status") == "ok")
    if health:
        report("text model loaded", health.get("model_loaded") is True,
               f"current={health.get('current_model')}")


def test_select_load_unload_cycle():
    print("\n=== 15. SELECT/LOAD/UNLOAD LIFECYCLE ===")
    # Unload everything
    _req("POST", "/v1/models/unload")
    time.sleep(1)

    _, h1, _ = _req("GET", "/health")
    report("starts unloaded", h1 and h1.get("model_loaded") is False)

    # Select (pre-load) a model
    code, _, _ = _req("POST", "/v1/models/select", {"model": "qwen3-1.7b"})
    report("select qwen3-1.7b returns 200", code == 200)

    _, h2, _ = _req("GET", "/health")
    report("model loaded after select",
           h2 and h2.get("model_loaded") is True and h2.get("current_model") == "qwen3-1.7b")

    # Unload
    _req("POST", "/v1/models/unload")
    time.sleep(1)
    _, h3, _ = _req("GET", "/health")
    report("model unloaded", h3 and h3.get("model_loaded") is False)


def test_multimodal_content_normalization():
    print("\n=== 16. MULTIMODAL CONTENT NORMALIZATION ===")
    # Send text-only but using multipart content format (like Open WebUI does)
    body = {
        "model": "qwen3-1.7b",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Say 'normalized'."}
        ]}],
        "stream": False, "max_tokens": 30,
    }
    code, data, _ = _req("POST", "/v1/chat/completions", body, timeout=120)
    report("multipart text-only returns 200", code == 200)
    if data:
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        report("got response", len(content) > 0, f"content='{content[:60]}'")


def test_concurrent_rejection():
    print("\n=== 17. CONCURRENT REQUEST REJECTION ===")
    # Start a long-running streaming request in background
    results = {"bg_code": None, "fg_code": None}

    def bg_request():
        code, chunks, _ = _stream_req("/v1/chat/completions", {
            "model": "qwen3-1.7b",
            "messages": [{"role": "user", "content": "Write a 200 word essay about space."}],
            "stream": True, "max_tokens": 300,
        }, timeout=180)
        results["bg_code"] = code

    t = threading.Thread(target=bg_request)
    t.start()
    time.sleep(3)  # Let it start generating

    # Try concurrent request — should be rejected with 503
    code, data, _ = _chat("qwen3-1.7b", "This should fail.", max_tokens=10)
    results["fg_code"] = code

    t.join(timeout=180)
    report("background request succeeded", results["bg_code"] == 200)
    report("concurrent request rejected with 503", results["fg_code"] == 503)


# =========================================================================
# MAIN
# =========================================================================

def run_all():
    """Run the complete test suite."""
    global PASS, FAIL, SKIP
    PASS = FAIL = SKIP = 0

    print("=" * 60)
    print("RKLLM API Server -- Comprehensive Test Suite")
    print(f"Target: {API}")
    print("=" * 60)

    start = time.time()

    # Quick connectivity check
    code, _, _ = _req("GET", "/health", timeout=5)
    if code != 200:
        print(f"\nFATAL: Cannot reach server at {API} (code={code})")
        print("Make sure gunicorn is running on the Orange Pi.")
        return

    test_health()
    test_models_list()
    test_error_handling()
    test_model_aliases()
    test_text_generation()
    test_text_streaming()
    time.sleep(2)  # Allow previous streaming request to fully complete
    test_streaming_usage()
    test_model_switching()
    test_model_unload()
    test_select_load_unload_cycle()
    test_multimodal_content_normalization()
    test_vl_non_streaming()
    test_vl_streaming()
    test_vl_auto_routing()
    test_vl_no_text()
    test_text_after_vl()
    test_concurrent_rejection()

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print(f"Time: {elapsed:.1f}s")
    print("=" * 60)

    if FAIL == 0:
        print("\nAll tests passed!")
    else:
        print(f"\n{FAIL} test(s) FAILED -- review output above.")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "complete":
        img = _make_test_image_b64()
        code, data, raw = _vl_chat("deepseekocr", "Describe the colors you see.", img, max_tokens=200)
        print(json.dumps(data, indent=2) if data else raw)
    elif mode == "stream":
        img = _make_test_image_b64()
        code, chunks, content = _vl_chat("deepseekocr", "Describe this image briefly.", img,
                                          stream=True, max_tokens=200)
        print(content)
        if chunks and chunks[-1] == "[DONE]":
            print("\n[DONE]")
    elif mode == "all":
        run_all()
    else:
        print(f"Usage: python {sys.argv[0]} [complete|stream|all]")
