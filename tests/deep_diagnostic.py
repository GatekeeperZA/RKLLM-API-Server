#!/usr/bin/env python3
"""
RKLLM API — Deep Diagnostic Tests
===================================
Covers areas NOT tested by e2e_test.py:

  Section 1:  SSE stream format strict compliance
  Section 2:  Concurrent request rejection (503)
  Section 3:  Model hot-swap correctness
  Section 4:  Unicode / special character handling
  Section 5:  ThinkTagParser edge cases (local unit tests)
  Section 6:  CORS headers & preflight
  Section 7:  Token usage accuracy
  Section 8:  Select / unload endpoint
  Section 9:  VL/OCR pipeline (image inference)
  Section 10: Context overflow / large input
  Section 11: Message normalization edge cases
  Section 12: Shortcircuit streaming SSE compliance

Usage:
    python tests/deep_diagnostic.py                  # all sections
    python tests/deep_diagnostic.py --section 5      # just section 5
    python tests/deep_diagnostic.py --section 1-6    # range
"""

import argparse, json, os, re, sys, time, threading, base64, struct, zlib
import requests

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
API = os.environ.get("RKLLM_API", "http://localhost:8000")
MODEL = os.environ.get("RKLLM_MODEL", "qwen3-1.7b")
MODEL_ALT = os.environ.get("RKLLM_MODEL_ALT", "phi-3-mini-4k-instruct")
VL_MODEL = os.environ.get("RKLLM_VL_MODEL", "qwen3-vl-2b")
TIMEOUT = int(os.environ.get("RKLLM_TIMEOUT", "180"))

PASS = FAIL = WARN = SKIP = 0
FAILURES = []


def check(section, name, ok, detail="", warn=False):
    global PASS, FAIL, WARN
    if ok:
        PASS += 1
        tag = "PASS"
    elif warn:
        WARN += 1
        tag = "WARN"
    else:
        FAIL += 1
        tag = "FAIL"
        FAILURES.append((section, name, detail))
    print(f"  [{tag}] {name}" + (f"  |  {detail[:120]}" if detail else ""))


def _req(method, path, json_body=None, headers=None, timeout=None):
    """Raw HTTP request. Returns (status, data_or_None, response)."""
    url = f"{API}{path}"
    try:
        r = requests.request(method, url, json=json_body, headers=headers,
                             timeout=timeout or TIMEOUT)
        try:
            data = r.json()
        except Exception:
            data = None
        return r.status_code, data, r
    except requests.exceptions.Timeout:
        return 0, None, "timeout"
    except Exception as e:
        return 0, None, str(e)


def _stream_raw(path, body, timeout=None):
    """Return raw SSE lines + parsed chunks."""
    url = f"{API}{path}"
    lines = []
    chunks = []
    content = ""
    try:
        r = requests.post(url, json=body, stream=True, timeout=timeout or TIMEOUT)
        for raw_line in r.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            lines.append(raw_line)
            if raw_line.startswith("data: "):
                payload = raw_line[6:]
                if payload.strip() == "[DONE]":
                    chunks.append({"_done": True})
                else:
                    try:
                        obj = json.loads(payload)
                        chunks.append(obj)
                        choices = obj.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            c = delta.get("content", "")
                            if c:
                                content += c
                    except json.JSONDecodeError:
                        chunks.append({"_parse_error": payload})
        return {"status": r.status_code, "lines": lines, "chunks": chunks,
                "content": content, "headers": dict(r.headers)}
    except Exception as e:
        return {"status": 0, "lines": lines, "chunks": chunks,
                "content": content, "error": str(e), "headers": {}}


def _chat(model, messages, stream=True, timeout=None, **kwargs):
    """High-level chat helper."""
    body = {"model": model, "messages": messages, "stream": stream, **kwargs}
    if stream:
        result = _stream_raw("/v1/chat/completions", body, timeout=timeout)
        return result
    else:
        code, data, resp = _req("POST", "/v1/chat/completions", body, timeout=timeout)
        return {"status": code, "data": data,
                "content": (data or {}).get("choices", [{}])[0].get("message", {}).get("content", ""),
                "headers": dict(resp.headers) if hasattr(resp, 'headers') else {}}


def _make_tiny_png(width=4, height=4, color=(255, 0, 0)):
    """Generate a minimal valid PNG in pure Python (no PIL needed)."""
    def _chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)

    raw = b""
    for y in range(height):
        raw += b"\x00"  # filter byte
        for x in range(width):
            raw += bytes(color)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr)
    png += _chunk(b"IDAT", zlib.compress(raw))
    png += _chunk(b"IEND", b"")
    return png


# ============================================================
# SECTION 1: SSE STREAM FORMAT STRICT COMPLIANCE
# ============================================================
def test_sse_compliance():
    sec = "1-SSE"
    print(f"\n{'='*60}")
    print(f"SECTION 1: SSE STREAM FORMAT STRICT COMPLIANCE")
    print(f"{'='*60}\n")

    result = _chat(MODEL, [{"role": "user", "content": "Say hello."}], stream=True)
    chunks = result.get("chunks", [])
    lines = result.get("lines", [])

    # 1. Content-Type header
    ct = result.get("headers", {}).get("Content-Type", "")
    check(sec, "Content-Type is text/event-stream",
          "text/event-stream" in ct, ct)

    # 2. Cache-Control header
    cc = result.get("headers", {}).get("Cache-Control", "")
    check(sec, "Cache-Control: no-cache",
          "no-cache" in cc, cc)

    # 3. Every non-empty, non-[DONE] data line is valid JSON
    parse_errors = [c for c in chunks if "_parse_error" in c]
    check(sec, "All data lines are valid JSON",
          len(parse_errors) == 0,
          f"{len(parse_errors)} parse errors" if parse_errors else "all valid")

    # 4. [DONE] terminator present
    has_done = any(c.get("_done") for c in chunks)
    check(sec, "[DONE] terminator present", has_done)

    # 5. First real chunk has delta.role = "assistant"
    real_chunks = [c for c in chunks if "_done" not in c and "_parse_error" not in c]
    first_role = ""
    if real_chunks:
        first_delta = real_chunks[0].get("choices", [{}])[0].get("delta", {})
        first_role = first_delta.get("role", "")
    check(sec, "First chunk has delta.role='assistant'",
          first_role == "assistant", f"role='{first_role}'")

    # 6. Every real chunk has required fields
    required_fields = {"id", "object", "created", "model", "system_fingerprint", "choices"}
    missing_fields = []
    for i, c in enumerate(real_chunks):
        missing = required_fields - set(c.keys())
        if missing:
            missing_fields.append((i, missing))
    check(sec, "All chunks have required fields",
          len(missing_fields) == 0,
          f"{len(missing_fields)} chunks missing fields" if missing_fields else f"checked {len(real_chunks)} chunks")

    # 7. object field is always "chat.completion.chunk"
    wrong_object = [c for c in real_chunks if c.get("object") != "chat.completion.chunk"]
    check(sec, "object='chat.completion.chunk' in all",
          len(wrong_object) == 0,
          f"{len(wrong_object)} wrong" if wrong_object else "all correct")

    # 8. Last real chunk (before DONE) has finish_reason="stop"
    stop_chunk = None
    for c in reversed(real_chunks):
        choices = c.get("choices", [])
        if choices and choices[0].get("finish_reason") is not None:
            stop_chunk = c
            break
    fr = stop_chunk.get("choices", [{}])[0].get("finish_reason") if stop_chunk else None
    check(sec, "Final chunk has finish_reason='stop'",
          fr == "stop", f"finish_reason='{fr}'")

    # 9. stop chunk has empty delta
    stop_delta = stop_chunk.get("choices", [{}])[0].get("delta", {}) if stop_chunk else {}
    check(sec, "Stop chunk has empty delta",
          stop_delta == {} or (not stop_delta.get("content") and not stop_delta.get("role")),
          f"delta={stop_delta}")

    # 10. No extraneous non-data lines (except empty lines between events)
    bad_lines = [l for l in lines if l.strip() and not l.startswith("data: ")]
    check(sec, "No extraneous non-data lines",
          len(bad_lines) == 0,
          f"found: {bad_lines[:3]}" if bad_lines else "clean")

    # 11. Consistent request ID across all chunks
    ids = set(c.get("id") for c in real_chunks if "id" in c)
    check(sec, "Consistent request ID across chunks",
          len(ids) == 1,
          f"ids={ids}" if len(ids) != 1 else f"id={ids.pop()}")

    # 12. Content was actually generated
    content = result.get("content", "")
    check(sec, "Response content is non-empty",
          len(content) > 0,
          f"'{content[:60]}'" if content else "empty")


# ============================================================
# SECTION 2: CONCURRENT REQUEST REJECTION
# ============================================================
def test_concurrent_rejection():
    sec = "2-CONCURRENT"
    print(f"\n{'='*60}")
    print(f"SECTION 2: CONCURRENT REQUEST REJECTION")
    print(f"{'='*60}\n")

    # Strategy: Start a long streaming request in a thread, then immediately
    # fire a second request. The second should get 503.

    first_status = [0]
    first_started = threading.Event()
    first_done = threading.Event()

    def _long_request():
        """Make a request that generates enough tokens to hold the lock."""
        url = f"{API}/v1/chat/completions"
        body = {"model": MODEL,
                "messages": [{"role": "user",
                              "content": "Write a detailed 200-word essay about the history of computing."}],
                "stream": True}
        try:
            r = requests.post(url, json=body, stream=True, timeout=TIMEOUT)
            first_status[0] = r.status_code
            first_started.set()
            # Consume just a few lines to keep connection alive
            count = 0
            for line in r.iter_lines(decode_unicode=True):
                count += 1
                if count >= 5:
                    # Wait a bit to hold the lock
                    time.sleep(3)
                    break
            # Let it drain
            for _ in r.iter_lines(decode_unicode=True):
                pass
        except Exception as e:
            first_started.set()
        finally:
            first_done.set()

    t = threading.Thread(target=_long_request, daemon=True)
    t.start()

    # Wait for first request to begin
    first_started.wait(timeout=30)
    time.sleep(0.5)  # give it a moment to be registered

    # Now fire second request
    code2, data2, _ = _req("POST", "/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hi."}],
        "stream": False,
    }, timeout=10)

    check(sec, "First request started OK",
          first_status[0] == 200, f"status={first_status[0]}")

    check(sec, "Second request rejected with 503",
          code2 == 503,
          f"status={code2}, expected 503" + (f" msg={json.dumps(data2)[:80]}" if data2 else ""))

    # Wait for first to finish
    first_done.wait(timeout=120)

    # After drain, a new request should succeed
    time.sleep(2)
    code3, data3, _ = _req("POST", "/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say OK."}],
        "stream": False,
    }, timeout=TIMEOUT)
    content3 = (data3 or {}).get("choices", [{}])[0].get("message", {}).get("content", "")
    check(sec, "Request succeeds after first completes",
          code3 == 200 and len(content3) > 0,
          f"status={code3}, content='{content3[:40]}'")


# ============================================================
# SECTION 3: MODEL HOT-SWAP CORRECTNESS
# ============================================================
def test_model_hotswap():
    sec = "3-HOTSWAP"
    print(f"\n{'='*60}")
    print(f"SECTION 3: MODEL HOT-SWAP CORRECTNESS")
    print(f"{'='*60}\n")

    # Request 1: primary model
    r1 = _chat(MODEL, [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
               stream=False)
    check(sec, f"Model {MODEL} responds",
          r1["status"] == 200 and len(r1["content"]) > 0,
          f"'{r1['content'][:40]}'")

    # Request 2: different model — triggers hot-swap
    swap_start = time.time()
    r2 = _chat(MODEL_ALT, [{"role": "user", "content": "What is 3+3? Answer with just the number."}],
               stream=False)
    swap_time = time.time() - swap_start
    check(sec, f"Hot-swap to {MODEL_ALT} responds",
          r2["status"] == 200 and len(r2["content"]) > 0,
          f"'{r2['content'][:40]}' swap_time={swap_time:.1f}s")

    # Verify the model field in response matches requested model
    resp_model = (r2.get("data") or {}).get("model", "")
    check(sec, "Response model field matches requested",
          MODEL_ALT in resp_model or resp_model == MODEL_ALT,
          f"model='{resp_model}'")

    # Request 3: swap back — verify it still works
    r3 = _chat(MODEL, [{"role": "user", "content": "What is 5+5? Answer with just the number."}],
               stream=False)
    check(sec, f"Swap back to {MODEL} works",
          r3["status"] == 200 and len(r3["content"]) > 0,
          f"'{r3['content'][:40]}'")

    # Health should show current model
    _, health, _ = _req("GET", "/health")
    current = (health or {}).get("current_model", "")
    check(sec, "Health shows correct model after swaps",
          current == MODEL,
          f"current_model='{current}'")


# ============================================================
# SECTION 4: UNICODE / SPECIAL CHARACTER HANDLING
# ============================================================
def test_unicode():
    sec = "4-UNICODE"
    print(f"\n{'='*60}")
    print(f"SECTION 4: UNICODE / SPECIAL CHARACTER HANDLING")
    print(f"{'='*60}\n")

    # Test 1: Emoji in prompt
    r1 = _chat(MODEL, [{"role": "user", "content": "What does this emoji mean: 🎉? One sentence."}],
               stream=False)
    check(sec, "Emoji in prompt handled",
          r1["status"] == 200 and len(r1["content"]) > 0,
          f"'{r1['content'][:60]}'")

    # Test 2: CJK characters
    r2 = _chat(MODEL, [{"role": "user", "content": "翻译成英文: 你好世界"}],
               stream=False)
    check(sec, "CJK characters handled",
          r2["status"] == 200 and len(r2["content"]) > 0,
          f"'{r2['content'][:60]}'")

    # Test 3: Mixed script (Arabic + Latin)
    r3 = _chat(MODEL, [{"role": "user", "content": "Translate: مرحبا means what in English? One word."}],
               stream=False)
    check(sec, "Arabic script handled",
          r3["status"] == 200 and len(r3["content"]) > 0,
          f"'{r3['content'][:60]}'")

    # Test 4: Special characters / escaping
    r4 = _chat(MODEL, [{"role": "user", "content": "Repeat exactly: <tag>foo</tag> & \"bar\" 'baz'"}],
               stream=False)
    check(sec, "HTML entities / quotes in prompt",
          r4["status"] == 200 and len(r4["content"]) > 0,
          f"'{r4['content'][:60]}'")

    # Test 5: Emoji in streaming mode — verify SSE JSON stays valid
    r5 = _chat(MODEL, [{"role": "user", "content": "Reply with exactly: Hello 🌍"}],
               stream=True)
    parse_errors = [c for c in r5.get("chunks", []) if "_parse_error" in c]
    check(sec, "Emoji in stream: SSE JSON stays valid",
          len(parse_errors) == 0 and r5["status"] == 200,
          f"errors={len(parse_errors)}")


# ============================================================
# SECTION 5: THINKTAG PARSER EDGE CASES (LOCAL UNIT TESTS)
# ============================================================
def test_thinktag_parser():
    """Test ThinkTagParser locally (re-implemented from api.py)."""
    sec = "5-THINKTAG"
    print(f"\n{'='*60}")
    print(f"SECTION 5: THINKTAG PARSER EDGE CASES")
    print(f"{'='*60}\n")

    # Re-implement parser locally to test independently
    class ThinkTagParser:
        OPEN_TAG = "<think>"
        CLOSE_TAG = "</think>"

        def __init__(self):
            self.in_thinking = False
            self.buffer = ""
            self.saw_think_block = False

        def feed(self, text):
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
            if self.buffer:
                kind = 'thinking' if self.in_thinking else 'content'
                text = self.buffer
                self.buffer = ""
                return (kind, text)
            return None

    def _run(token_list):
        """Feed tokens one-by-one. Return collected (thinking, content)."""
        p = ThinkTagParser()
        thinking = ""
        content = ""
        for tok in token_list:
            for kind, text in p.feed(tok):
                if kind == "thinking":
                    thinking += text
                else:
                    content += text
        flushed = p.flush()
        if flushed:
            kind, text = flushed
            if kind == "thinking":
                thinking += text
            else:
                content += text
        return thinking, content

    # Test 1: Basic think block
    t, c = _run(["<think>", "reason", "ing", "</think>", "answer"])
    check(sec, "Basic think block",
          t == "reasoning" and c == "answer",
          f"thinking='{t}', content='{c}'")

    # Test 2: No think tags — all content
    t, c = _run(["Hello ", "world"])
    check(sec, "No think tags = all content",
          t == "" and c == "Hello world",
          f"thinking='{t}', content='{c}'")

    # Test 3: Unclosed think tag (rest is thinking)
    t, c = _run(["<think>", "forever", " thinking"])
    check(sec, "Unclosed think tag -> remaining is thinking",
          "forever thinking" in t and c == "",
          f"thinking='{t}', content='{c}'")

    # Test 4: Partial open tag split across tokens
    t, c = _run(["before", "<thi", "nk>", "inside", "</think>", "after"])
    check(sec, "Partial open tag split across tokens",
          "inside" in t and "after" in c and "before" in c,
          f"thinking='{t}', content='{c}'")

    # Test 5: Partial close tag split across tokens
    t, c = _run(["<think>", "data", "</thi", "nk>", "out"])
    check(sec, "Partial close tag split across tokens",
          "data" in t and "out" in c,
          f"thinking='{t}', content='{c}'")

    # Test 6: Empty think block
    t, c = _run(["<think>", "</think>", "answer"])
    check(sec, "Empty think block",
          t == "" and "answer" in c,
          f"thinking='{t}', content='{c}'")

    # Test 7: Multiple think blocks
    t, c = _run(["<think>", "A", "</think>", "mid", "<think>", "B", "</think>", "end"])
    check(sec, "Multiple think blocks",
          "A" in t and "B" in t and "mid" in c and "end" in c,
          f"thinking='{t}', content='{c}'")

    # Test 8: Think tags character-by-character (worst case)
    chars = list("<think>reasoning</think>content")
    t, c = _run(chars)
    check(sec, "Character-by-character feeding",
          "reasoning" in t and "content" in c,
          f"thinking='{t}', content='{c}'")

    # Test 9: Angle brackets that are NOT think tags
    t, c = _run(["<div>hello</div>"])
    check(sec, "Non-think angle brackets = content",
          t == "" and "<div>hello</div>" in c,
          f"thinking='{t}', content='{c}'")

    # Test 10: Think tag immediately at start followed by content
    t, c = _run(["<think>\nI should think\n</think>\nHere's my answer."])
    check(sec, "Think at start with newlines",
          "I should think" in t and "my answer" in c,
          f"thinking='{t[:30]}...', content='{c[:30]}...'")

    # Also test the regex-based parse_think_tags equivalent
    def parse_think_tags(text):
        pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        thinking_parts = pattern.findall(text)
        content = pattern.sub('', text)
        unclosed = re.search(r'<think>(.*?)$', content, re.DOTALL)
        if unclosed:
            thinking_parts.append(unclosed.group(1))
            content = content[:unclosed.start()]
        content = content.strip()
        reasoning = '\n'.join(part.strip() for part in thinking_parts if part.strip())
        return (reasoning if reasoning else None, content)

    # Test 11: Streaming vs regex consistency on multi-block
    # Note: regex joins thinking parts with \n, streaming concatenates directly.
    # Both capture the same CONTENT correctly — the thinking join style differs.
    test_text = "<think>step 1</think>answer 1<think>step 2</think>answer 2"
    regex_t, regex_c = parse_think_tags(test_text)
    stream_t, stream_c = _run([test_text])
    # Compare thinking parts set and content (ignore join whitespace)
    regex_parts = set((regex_t or "").split("\n"))
    stream_parts = {"step 1", "step 2"}  # known from input
    check(sec, "Streaming vs regex parity (multi-block)",
          all(p.strip() in stream_t for p in regex_parts if p.strip())
          and regex_c.strip() == stream_c.strip(),
          f"regex=({regex_t}, {regex_c}), stream=({stream_t}, {stream_c})")

    # Test 12: Unclosed tag consistency
    test_text2 = "prefix <think>unclosed"
    regex_t2, regex_c2 = parse_think_tags(test_text2)
    stream_t2, stream_c2 = _run([test_text2])
    check(sec, "Unclosed tag: streaming vs regex parity",
          "unclosed" in stream_t2 and "prefix" in stream_c2,
          f"stream=({stream_t2}, {stream_c2}), regex=({regex_t2}, {regex_c2})")

    # Test 13: Large think block (performance sanity)
    big_thinking = "A" * 10000
    t, c = _run([f"<think>{big_thinking}</think>done"])
    check(sec, "Large think block (10K chars)",
          len(t) == 10000 and c == "done",
          f"thinking_len={len(t)}, content='{c}'")


# ============================================================
# SECTION 6: CORS HEADERS & PREFLIGHT
# ============================================================
def test_cors():
    sec = "6-CORS"
    print(f"\n{'='*60}")
    print(f"SECTION 6: CORS HEADERS & PREFLIGHT")
    print(f"{'='*60}\n")

    # Test 1: OPTIONS preflight on chat endpoint
    try:
        r = requests.options(f"{API}/v1/chat/completions",
                             headers={"Origin": "http://localhost:3000",
                                      "Access-Control-Request-Method": "POST",
                                      "Access-Control-Request-Headers": "content-type"},
                             timeout=10)
        acl = r.headers.get("Access-Control-Allow-Origin", "")
        check(sec, "OPTIONS preflight returns CORS origin header",
              acl != "",
              f"Access-Control-Allow-Origin: {acl}")

        check(sec, "OPTIONS returns 200",
              r.status_code == 200,
              f"status={r.status_code}")

        # Check allowed methods
        methods = r.headers.get("Access-Control-Allow-Methods", "")
        check(sec, "CORS allows POST method",
              "POST" in methods.upper() or acl == "*",
              f"methods={methods}")
    except Exception as e:
        check(sec, "OPTIONS preflight", False, str(e))
        check(sec, "OPTIONS returns 200", False, "skipped")
        check(sec, "CORS allows POST method", False, "skipped")

    # Test 2: Regular GET includes CORS header
    code, data, resp = _req("GET", "/v1/models")
    acl2 = resp.headers.get("Access-Control-Allow-Origin", "") if hasattr(resp, 'headers') else ""
    check(sec, "GET /v1/models has CORS header",
          acl2 != "",
          f"Access-Control-Allow-Origin: {acl2}")

    # Test 3: POST includes CORS header
    r3 = _chat(MODEL, [{"role": "user", "content": "Say OK."}], stream=False)
    acl3 = r3.get("headers", {}).get("Access-Control-Allow-Origin", "")
    check(sec, "POST chat/completions has CORS header",
          acl3 != "",
          f"Access-Control-Allow-Origin: {acl3}")


# ============================================================
# SECTION 7: TOKEN USAGE ACCURACY
# ============================================================
def test_usage_accuracy():
    sec = "7-USAGE"
    print(f"\n{'='*60}")
    print(f"SECTION 7: TOKEN USAGE ACCURACY")
    print(f"{'='*60}\n")

    prompt_text = "What is two plus two? Answer with just the number."

    # --- Non-streaming: check usage block ---
    r1 = _chat(MODEL, [{"role": "user", "content": prompt_text}], stream=False)
    usage = (r1.get("data") or {}).get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    tt = usage.get("total_tokens", 0)

    check(sec, "Non-streaming has usage block",
          pt > 0 and ct > 0 and tt > 0,
          f"prompt={pt}, completion={ct}, total={tt}")

    check(sec, "total_tokens = prompt + completion",
          tt == pt + ct,
          f"{tt} != {pt} + {ct}")

    # Prompt tokens should be reasonable (not 0, not millions)
    check(sec, "Prompt tokens in reasonable range (3-200)",
          3 < pt < 200,
          f"prompt_tokens={pt}")

    # Completion should be small for "4"
    check(sec, "Completion tokens reasonable for short answer",
          0 < ct < 500,
          f"completion_tokens={ct}")

    # --- Streaming with include_usage ---
    r2 = _stream_raw("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    chunks = r2.get("chunks", [])

    # Find usage chunk (has usage field, empty choices)
    usage_chunks = [c for c in chunks
                    if isinstance(c, dict) and "usage" in c and "_done" not in c]
    check(sec, "Streaming has usage chunk when include_usage=true",
          len(usage_chunks) > 0,
          f"found {len(usage_chunks)} usage chunks")

    if usage_chunks:
        u = usage_chunks[-1].get("usage", {})
        spt = u.get("prompt_tokens", 0)
        sct = u.get("completion_tokens", 0)
        stt = u.get("total_tokens", 0)
        check(sec, "Streaming usage has valid counts",
              spt > 0 and sct > 0 and stt > 0,
              f"prompt={spt}, completion={sct}, total={stt}")

        check(sec, "Streaming total = prompt + completion",
              stt == spt + sct,
              f"{stt} != {spt} + {sct}")

    # --- Without include_usage: no usage chunk should appear ---
    r3 = _stream_raw("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hi."}],
        "stream": True,
    })
    chunks3 = r3.get("chunks", [])
    usage_chunks3 = [c for c in chunks3
                     if isinstance(c, dict) and "usage" in c and "_done" not in c]
    check(sec, "No usage chunk when include_usage not set",
          len(usage_chunks3) == 0,
          f"found {len(usage_chunks3)} unexpected usage chunks")


# ============================================================
# SECTION 8: SELECT / UNLOAD ENDPOINTS
# ============================================================
def test_select_unload():
    sec = "8-SELECT"
    print(f"\n{'='*60}")
    print(f"SECTION 8: SELECT / UNLOAD ENDPOINTS")
    print(f"{'='*60}\n")

    # Test 1: Select a model (pre-load)
    code1, data1, _ = _req("POST", "/v1/models/select", {"model": MODEL})
    check(sec, f"Select {MODEL}: 200",
          code1 == 200 and (data1 or {}).get("status") == "ok",
          f"status={code1}, data={json.dumps(data1)[:80]}")

    # Health should show model loaded
    _, health1, _ = _req("GET", "/health")
    check(sec, "Health: model loaded after select",
          (health1 or {}).get("model_loaded") == True,
          f"loaded={(health1 or {}).get('model_loaded')}")

    # Test 2: Select invalid model
    code2, data2, _ = _req("POST", "/v1/models/select", {"model": "nonexistent-99"})
    check(sec, "Select invalid model: 404",
          code2 == 404,
          f"status={code2}")

    # Test 3: Unload model
    code3, data3, _ = _req("POST", "/v1/models/unload", {})
    check(sec, "Unload model: 200",
          code3 == 200 and (data3 or {}).get("status") == "ok",
          f"status={code3}, data={json.dumps(data3)[:80]}")

    # Health should show model not loaded
    _, health2, _ = _req("GET", "/health")
    check(sec, "Health: model_loaded=False after unload",
          (health2 or {}).get("model_loaded") == False,
          f"loaded={(health2 or {}).get('model_loaded')}")

    # Test 4: Unload when nothing loaded
    code4, data4, _ = _req("POST", "/v1/models/unload", {})
    check(sec, "Unload again: 200 (no model loaded)",
          code4 == 200,
          f"status={code4}")

    # Test 5: Reload model for subsequent tests
    code5, _, _ = _req("POST", "/v1/models/select", {"model": MODEL})
    check(sec, f"Re-select {MODEL}: 200",
          code5 == 200,
          f"status={code5}")

    # Test 6: Select with invalid body
    try:
        r = requests.post(f"{API}/v1/models/select",
                          data="not json",
                          headers={"Content-Type": "application/json"},
                          timeout=10)
        check(sec, "Select with invalid JSON: 400",
              r.status_code == 400,
              f"status={r.status_code}")
    except Exception as e:
        check(sec, "Select with invalid JSON", False, str(e))


# ============================================================
# SECTION 9: VL / OCR PIPELINE
# ============================================================
def test_vl_pipeline():
    sec = "9-VL"
    print(f"\n{'='*60}")
    print(f"SECTION 9: VL / OCR PIPELINE")
    print(f"{'='*60}\n")

    # Check if VL model is available
    _, models_data, _ = _req("GET", "/v1/models")
    model_ids = [m["id"] for m in (models_data or {}).get("data", [])]
    if VL_MODEL not in model_ids:
        global SKIP
        SKIP += 1
        print(f"  [SKIP] VL model '{VL_MODEL}' not available — skipping section")
        return

    # Generate a tiny red 4x4 PNG
    png_bytes = _make_tiny_png(16, 16, (255, 0, 0))
    b64_img = base64.b64encode(png_bytes).decode('ascii')
    data_uri = f"data:image/png;base64,{b64_img}"

    # Test 1: VL auto-routing with valid base64 image
    print("  --- Sending image to VL model (may take a while for model swap) ---")
    body = {
        "model": VL_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what you see in this image in one sentence."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        }],
        "stream": False,
    }
    code1, data1, _ = _req("POST", "/v1/chat/completions", body, timeout=300)
    content1 = (data1 or {}).get("choices", [{}])[0].get("message", {}).get("content", "")
    check(sec, "VL image inference returns response",
          code1 == 200 and len(content1) > 0,
          f"status={code1}, content='{content1[:80]}'")

    if code1 == 200:
        # Test 2: VL response has required OpenAI fields
        choice = (data1 or {}).get("choices", [{}])[0]
        check(sec, "VL response has finish_reason",
              choice.get("finish_reason") is not None,
              f"finish_reason={choice.get('finish_reason')}")

        check(sec, "VL response has model field",
              VL_MODEL in (data1 or {}).get("model", ""),
              f"model={(data1 or {}).get('model')}")

    # Test 3: VL with streaming
    body["stream"] = True
    result2 = _stream_raw("/v1/chat/completions", body, timeout=300)
    check(sec, "VL streaming works",
          result2["status"] == 200 and len(result2["content"]) > 0,
          f"content='{result2['content'][:80]}'")

    # Test 4: Invalid base64 should get 400
    bad_body = {
        "model": VL_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,NOT_VALID!!!"}},
            ]
        }],
        "stream": False,
    }
    code3, data3, _ = _req("POST", "/v1/chat/completions", bad_body, timeout=30)
    check(sec, "Invalid base64 image returns 400",
          code3 == 400,
          f"status={code3}")

    # Test 5: Image URL (not base64) should fail gracefully
    url_body = {
        "model": VL_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ]
        }],
        "stream": False,
    }
    code4, data4, _ = _req("POST", "/v1/chat/completions", url_body, timeout=30)
    check(sec, "URL-based image fails gracefully (400)",
          code4 == 400,
          f"status={code4}")

    # Reload text model for other tests
    print("  --- Reloading text model ---")
    _req("POST", "/v1/models/select", {"model": MODEL}, timeout=120)


# ============================================================
# SECTION 10: CONTEXT OVERFLOW / LARGE INPUT
# ============================================================
def test_context_overflow():
    sec = "10-OVERFLOW"
    print(f"\n{'='*60}")
    print(f"SECTION 10: CONTEXT OVERFLOW / LARGE INPUT")
    print(f"{'='*60}\n")

    # qwen3-1.7b has ctx=4096. ~4 chars per token, so
    # 4096 * 4 = 16384 chars should approach the limit.
    # Let's send ~22000 chars and see what happens.
    large_text = "The quick brown fox jumps over the lazy dog. " * 500  # ~22500 chars

    r1 = _chat(MODEL, [{"role": "user", "content": f"Summarize this in one word: {large_text}"}],
               stream=False, timeout=300)
    check(sec, "Large input (~22K chars) doesn't crash server",
          r1["status"] in (200, 400, 500),
          f"status={r1['status']}, content_len={len(r1['content'])}")

    # If 200, model managed it (maybe truncated, or context full = no output)
    if r1["status"] == 200:
        if len(r1["content"]) > 0:
            check(sec, "Large input produces some output",
                  True, f"output_len={len(r1['content'])}")
        else:
            # Context overflow → 200 with empty content is acceptable
            check(sec, "Large input returned empty output (context overflow, expected)",
                  False, f"output_len=0, ctx=4096 tokens, input=~22K chars", warn=True)

    # Verify server is still alive
    _, health, _ = _req("GET", "/health")
    check(sec, "Server still healthy after large input",
          (health or {}).get("status") == "ok",
          f"health={json.dumps(health)[:80] if health else 'unreachable'}")

    # Test 2: Many messages (deep conversation)
    many_msgs = []
    for i in range(50):
        many_msgs.append({"role": "user", "content": f"Message {i}: what is {i}+{i}?"})
        many_msgs.append({"role": "assistant", "content": f"The answer is {i*2}."})
    many_msgs.append({"role": "user", "content": "What was the first question?"})

    r2 = _chat(MODEL, many_msgs, stream=False, timeout=300)
    check(sec, "50-turn conversation doesn't crash",
          r2["status"] in (200, 400, 500),
          f"status={r2['status']}, content='{r2['content'][:60]}'")

    # Verify still alive
    _, health2, _ = _req("GET", "/health")
    check(sec, "Server healthy after deep conversation",
          (health2 or {}).get("status") == "ok")


# ============================================================
# SECTION 11: MESSAGE NORMALIZATION EDGE CASES
# ============================================================
def test_message_normalization():
    sec = "11-NORMALIZE"
    print(f"\n{'='*60}")
    print(f"SECTION 11: MESSAGE NORMALIZATION EDGE CASES")
    print(f"{'='*60}\n")

    # Test 1: content as list of text parts (multimodal format without images)
    body1 = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is"},
                {"type": "text", "text": " 2+2?"},
            ]
        }],
        "stream": False,
    }
    code1, data1, _ = _req("POST", "/v1/chat/completions", body1)
    check(sec, "List content (text parts) normalized OK",
          code1 == 200,
          f"status={code1}")

    # Test 2: content as integer (edge case)
    body2 = {
        "model": MODEL,
        "messages": [{"role": "user", "content": 42}],
        "stream": False,
    }
    code2, data2, _ = _req("POST", "/v1/chat/completions", body2)
    check(sec, "Integer content coerced to string",
          code2 == 200,
          f"status={code2}")

    # Test 3: content as null
    body3 = {
        "model": MODEL,
        "messages": [{"role": "user", "content": None}],
        "stream": False,
    }
    code3, data3, _ = _req("POST", "/v1/chat/completions", body3)
    # Should either work (empty string) or return 400, not 500
    check(sec, "null content doesn't crash (200 or 400, not 500)",
          code3 in (200, 400),
          f"status={code3}")

    # Test 4: Mixed valid/invalid messages array (non-dict elements filtered)
    body4 = {
        "model": MODEL,
        "messages": [
            "not a dict",
            {"role": "user", "content": "Say OK."},
            123,
        ],
        "stream": False,
    }
    code4, data4, _ = _req("POST", "/v1/chat/completions", body4)
    content4 = (data4 or {}).get("choices", [{}])[0].get("message", {}).get("content", "")
    check(sec, "Non-dict messages filtered, valid msg processed",
          code4 == 200 and len(content4) > 0,
          f"status={code4}, content='{content4[:40]}'")

    # Test 5: System message + user message
    body5 = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a pirate. Always say 'Arrr'."},
            {"role": "user", "content": "Hello"},
        ],
        "stream": False,
    }
    code5, data5, _ = _req("POST", "/v1/chat/completions", body5)
    content5 = (data5 or {}).get("choices", [{}])[0].get("message", {}).get("content", "")
    check(sec, "System message + user message works",
          code5 == 200 and len(content5) > 0,
          f"'{content5[:60]}'")

    # Test 6: Empty string content
    body6 = {
        "model": MODEL,
        "messages": [{"role": "user", "content": ""}],
        "stream": False,
    }
    code6, data6, _ = _req("POST", "/v1/chat/completions", body6)
    check(sec, "Empty string content: doesn't crash",
          code6 in (200, 400),
          f"status={code6}")

    # Test 7: Very long single message (but within context)
    long_msg = "Hello " * 200  # ~1200 chars, well within limits
    body7 = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Count the words: {long_msg}"}],
        "stream": False,
    }
    code7, data7, _ = _req("POST", "/v1/chat/completions", body7)
    check(sec, "Moderate-length single message (1200 chars)",
          code7 == 200,
          f"status={code7}")


# ============================================================
# SECTION 12: SHORTCIRCUIT STREAMING SSE COMPLIANCE
# ============================================================
def test_shortcircuit_sse():
    sec = "12-SC-SSE"
    print(f"\n{'='*60}")
    print(f"SECTION 12: SHORTCIRCUIT STREAMING SSE COMPLIANCE")
    print(f"{'='*60}\n")

    # Trigger a shortcircuit: title generation (must match Open WebUI pattern)
    result = _stream_raw("/v1/chat/completions", {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Hello there!"},
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "Create a concise, 3-5 word chat title with an emoji."},
        ],
        "stream": True,
    })

    chunks = result.get("chunks", [])
    real_chunks = [c for c in chunks if isinstance(c, dict) and "_done" not in c and "_parse_error" not in c]

    # Test 1: Has content
    check(sec, "Shortcircuit stream has content",
          len(result["content"]) > 0,
          f"'{result['content'][:60]}'")

    # Test 2: Has [DONE]
    has_done = any(c.get("_done") for c in chunks)
    check(sec, "Shortcircuit has [DONE] terminator",
          has_done)

    # Test 3: All chunks have required fields
    required = {"id", "object", "created", "model", "system_fingerprint"}
    missing = []
    for c in real_chunks:
        m = required - set(c.keys())
        if m:
            missing.append(m)
    check(sec, "All shortcircuit chunks have required fields",
          len(missing) == 0,
          f"{len(missing)} chunks missing fields" if missing else f"checked {len(real_chunks)} chunks")

    # Test 4: Has stop chunk
    stop_chunks = [c for c in real_chunks
                   if c.get("choices", [{}])[0].get("finish_reason") == "stop"]
    check(sec, "Shortcircuit has finish_reason='stop' chunk",
          len(stop_chunks) > 0)

    # Test 5: Shortcircuit non-streaming includes usage
    code2, data2, _ = _req("POST", "/v1/chat/completions", {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Hello there!"},
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "Create a concise, 3-5 word chat title with an emoji."},
        ],
        "stream": False,
    })
    usage = (data2 or {}).get("usage", {})
    check(sec, "Shortcircuit non-streaming has usage block",
          "prompt_tokens" in usage and "completion_tokens" in usage,
          f"usage={usage}")

    # Test 6: system_fingerprint present
    fps = set(c.get("system_fingerprint") for c in real_chunks if "system_fingerprint" in c)
    check(sec, "system_fingerprint present in shortcircuit",
          len(fps) > 0 and all(f is not None for f in fps),
          f"fingerprints={fps}")

    # Test 7: Shortcircuit chunks are compact: content chunk + stop chunk + [DONE]
    # (no role-only first chunk, no intermediate token chunks)
    check(sec, "Shortcircuit stream is compact (2 real + DONE)",
          len(real_chunks) == 2 and has_done,
          f"real_chunks={len(real_chunks)}, done={has_done}")

    # Test 8: Shortcircuit non-streaming usage is hardcoded {0, 1, 1}
    sc_usage = (data2 or {}).get("usage", {})
    check(sec, "Shortcircuit usage: prompt=0, completion=1",
          sc_usage.get("prompt_tokens") == 0 and sc_usage.get("completion_tokens") == 1,
          f"usage={sc_usage}")


# ============================================================
# RUNNER
# ============================================================

SECTIONS = {
    1: ("SSE Stream Compliance", test_sse_compliance),
    2: ("Concurrent Request Rejection", test_concurrent_rejection),
    3: ("Model Hot-Swap", test_model_hotswap),
    4: ("Unicode / Special Chars", test_unicode),
    5: ("ThinkTag Parser", test_thinktag_parser),
    6: ("CORS Headers", test_cors),
    7: ("Token Usage Accuracy", test_usage_accuracy),
    8: ("Select / Unload", test_select_unload),
    9: ("VL / OCR Pipeline", test_vl_pipeline),
    10: ("Context Overflow", test_context_overflow),
    11: ("Message Normalization", test_message_normalization),
    12: ("Shortcircuit SSE", test_shortcircuit_sse),
}


def main():
    global PASS, FAIL, WARN, SKIP
    parser = argparse.ArgumentParser(description="RKLLM Deep Diagnostic Tests")
    parser.add_argument("--section", type=str, default=None,
                        help="Run specific section(s): '5', '1-6', 'all'")
    parser.add_argument("--skip-vl", action="store_true",
                        help="Skip VL/OCR tests (section 9)")
    args = parser.parse_args()

    # Determine which sections to run
    sections_to_run = sorted(SECTIONS.keys())
    if args.section and args.section != "all":
        if "-" in args.section:
            a, b = args.section.split("-", 1)
            sections_to_run = list(range(int(a), int(b) + 1))
        else:
            sections_to_run = [int(args.section)]
    if args.skip_vl:
        sections_to_run = [s for s in sections_to_run if s != 9]

    print("=" * 60)
    print("RKLLM API — DEEP DIAGNOSTIC TESTS")
    print("=" * 60)
    print(f"API:       {API}")
    print(f"Model:     {MODEL}")
    print(f"Alt Model: {MODEL_ALT}")
    print(f"VL Model:  {VL_MODEL}")
    print(f"Sections:  {sections_to_run}")
    print(f"Timeout:   {TIMEOUT}s")

    # Health check
    code, health, _ = _req("GET", "/health", timeout=10)
    if code != 200:
        print(f"\n  FATAL: API not reachable (status={code})")
        sys.exit(1)
    print(f"\nAPI health: {json.dumps(health)}")

    start = time.time()
    for s in sections_to_run:
        if s in SECTIONS:
            _, fn = SECTIONS[s]
            fn()

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  PASS: {PASS}")
    print(f"  FAIL: {FAIL}")
    print(f"  WARN: {WARN}")
    print(f"  SKIP: {SKIP}")
    print(f"  Time: {elapsed:.0f}s")

    if FAILURES:
        print(f"\nFAILURES:")
        for sec, name, detail in FAILURES:
            print(f"  [{sec}] {name}  |  {detail[:120]}")

    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
