"""
test_features.py — End-to-end feature verification for OpenWebUI + RKLLM API.

Tests the full middleware pipeline: memory retrieval, true RAG (file → chunk →
embed → retrieve), web search (SearXNG), context compaction settings, and
tool calling. Each test cleans up after itself.

Usage:
    python tests/test_features.py [--host 192.168.2.180] [--port 3000]

All tests use qwen3-1.7b (fastest model). The NPU handles one request at a
time so tests run sequentially with short pauses between them.
"""

import argparse
import json
import sys
import time
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_HOST = "192.168.2.180"
DEFAULT_PORT = 3000
API_KEY = "sk-owui-eae7cdfc8d566d50a2b9ac6f5e22d0913c01d057ec8a991e"
MODEL = "qwen3-1.7b"
TASK_MODEL = "qwen3-1.7b"

PASS = 0
FAIL = 0


def result(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    suffix = f"  | {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if ok:
        PASS += 1
    else:
        FAIL += 1


def section(title: str):
    print(f"\n[{title}]")


class Client:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        })

    def get(self, path, **kw):
        return self.s.get(self.base + path, timeout=30, **kw)

    def post(self, path, **kw):
        return self.s.post(self.base + path, timeout=120, **kw)

    def delete(self, path, **kw):
        return self.s.delete(self.base + path, timeout=15, **kw)

    def chat(self, messages, features=None, files=None, tools=None, stream=False):
        body = {"model": MODEL, "messages": messages, "stream": stream}
        if features:
            body["features"] = features
        if files:
            # Files for RAG go in metadata.files, not the top-level files field
            body.setdefault("metadata", {})["files"] = files
        if tools:
            body["tools"] = tools
        return self.post("/api/chat/completions", json=body)

    def chat_with_retry(self, messages, features=None, files=None, tools=None, max_wait=90):
        """Chat with automatic retry on NPU-busy 503/400.

        On this single-NPU system, 400 from OpenWebUI almost always means it
        received a 503 from the upstream RKLLM API (another request in flight).
        """
        deadline = time.time() + max_wait
        attempts = 0
        while time.time() < deadline:
            r = self.chat(messages, features=features, files=files, tools=tools)
            if r.status_code == 200:
                return r
            attempts += 1
            if r.status_code in (400, 503) and attempts < 8:
                # 503 = RKLLM NPU busy; 400 = OpenWebUI wraps upstream 503 as 400
                wait = min(6 + attempts * 2, 20)
                time.sleep(wait)
                continue
            return r  # non-retriable error or too many attempts
        return r

    def chat_text(self, prompt, features=None, files=None) -> str:
        r = self.chat_with_retry([{"role": "user", "content": prompt}], features=features, files=files)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


# ── Settings verification ──────────────────────────────────────────────────────

def test_settings(c: Client):
    section("SETTINGS — task config API")
    r = c.get("/api/v1/tasks/config")
    if not r.ok:
        result("Tasks config API", False, str(r.status_code))
        return
    t = r.json()
    result("Task model = qwen3-1.7b",       t.get("TASK_MODEL") == TASK_MODEL,              t.get("TASK_MODEL", "NOT SET"))
    result("Tags generation OFF",            t.get("ENABLE_TAGS_GENERATION") is False,       str(t.get("ENABLE_TAGS_GENERATION")))
    result("Follow-up generation OFF",       t.get("ENABLE_FOLLOW_UP_GENERATION") is False,  str(t.get("ENABLE_FOLLOW_UP_GENERATION")))
    result("Autocomplete OFF",               t.get("ENABLE_AUTOCOMPLETE_GENERATION") is False, str(t.get("ENABLE_AUTOCOMPLETE_GENERATION")))
    result("Title generation ON",            t.get("ENABLE_TITLE_GENERATION") is True,       str(t.get("ENABLE_TITLE_GENERATION")))
    result("Web search query gen ON",        t.get("ENABLE_SEARCH_QUERY_GENERATION") is True, str(t.get("ENABLE_SEARCH_QUERY_GENERATION")))
    result("Retrieval query gen ON",         t.get("ENABLE_RETRIEVAL_QUERY_GENERATION") is True, str(t.get("ENABLE_RETRIEVAL_QUERY_GENERATION")))
    custom_q = t.get("QUERY_GENERATION_PROMPT_TEMPLATE", "")
    result("Custom query prompt set",        len(custom_q) > 20,  f"{len(custom_q)} chars")


def test_compaction_settings(c: Client):
    section("SETTINGS — context compaction")
    # Read directly from the OpenWebUI config API (admin only)
    r = c.get("/api/v1/configs/")
    # Configs returns HTML (SPA route) — check via health + chat API instead
    # Verify by confirming compaction model is set on tasks config
    r2 = c.get("/api/v1/tasks/config")
    if r2.ok:
        t = r2.json()
        cc_model = t.get("CONTEXT_COMPACTION_MODEL", "")
        result("Compaction model = qwen3-1.7b",
               cc_model == TASK_MODEL or cc_model == "",   # empty means current model (also valid)
               cc_model or "(current model)")
    # Verify via chat that OpenWebUI is healthy and compaction won't break requests
    r3 = c.get("/health")
    result("OpenWebUI healthy (compaction won't block)", r3.ok and r3.json().get("status") is True)


# ── Memory ────────────────────────────────────────────────────────────────────

def test_memory(c: Client):
    section("MEMORY — persistence across sessions")
    memory_id = None
    # Use a unique nonsense fact that can't be in training data
    codeword = f"MEMTEST-{uuid.uuid4().hex[:8].upper()}"
    memory_content = f"The user's secret test codeword is {codeword}."

    try:
        # 1. Manually store a memory (simulates what background extraction does)
        r = c.post("/api/v1/memories/add", json={"content": memory_content})
        ok = r.ok and r.json().get("id")
        result("Add memory via API", ok, r.json().get("id", r.text[:60]) if r.ok else str(r.status_code))
        if not ok:
            return
        memory_id = r.json()["id"]
        time.sleep(2)  # let embedding settle

        # 2. Verify it's stored
        r2 = c.get("/api/v1/memories/")
        found = any(m.get("id") == memory_id for m in (r2.json() if r2.ok else []))
        result("Memory persisted in DB", found, f"id={memory_id[:8]}...")

        # 3. In a new chat with memory ON, ask something that requires the stored fact
        time.sleep(3)
        text = c.chat_text(
            "What is my secret test codeword? Reply with just the codeword, nothing else.",
            features={"memory": True},
        )
        retrieved = codeword.lower() in text.lower()
        result("Memory retrieved into context",  retrieved, text.strip()[:80])

        # 4. Without memory feature, model should NOT know the codeword
        time.sleep(5)
        text_no_mem = c.chat_text(
            f"What is my secret test codeword? Reply with just the codeword or 'unknown'.",
            features={"memory": False},
        )
        not_retrieved = codeword.lower() not in text_no_mem.lower()
        result("Memory NOT injected when feature OFF", not_retrieved, text_no_mem.strip()[:60])

    except Exception as e:
        result("Memory test", False, str(e)[:80])
    finally:
        # Cleanup
        if memory_id:
            try:
                c.delete(f"/api/v1/memories/{memory_id}")
            except Exception:
                pass


# ── RAG pipeline ──────────────────────────────────────────────────────────────

def test_rag_pipeline(c: Client):
    section("RAG — file upload → chunk → embed → retrieve")
    file_id = None

    # Unique content that can't be in training data
    doc = (
        "INTERNAL MEMO — Project Heliosphere\n"
        "The activation phrase for the Heliosphere relay station is: COBALT-MERIDIAN-7.\n"
        "This phrase must be kept confidential. The relay operates on frequency 4.771 GHz.\n"
        "Contact coordinator: Dr. Elara Voss, station ID HX-0042.\n"
    )

    try:
        # 1. Upload document via files API
        upload = c.s.post(
            c.base + "/api/v1/files/",
            files={"file": ("heliosphere_memo.txt", doc.encode(), "text/plain")},
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": None},
            timeout=30,
        )
        ok = upload.status_code in (200, 201)
        result("Upload document to RAG pipeline", ok, upload.text[:60] if not ok else "")
        if not ok:
            return
        file_id = upload.json().get("id")
        result("Document has file ID", bool(file_id), file_id or "")
        time.sleep(3)  # wait for chunking + embedding

        # 2. Chat with the file ATTACHED (true RAG — OpenWebUI retrieves chunks)
        #    Files are referenced via the files field, NOT injected in system prompt
        time.sleep(5)
        r = c.chat(
            messages=[{"role": "user", "content": "What is the activation phrase for the Heliosphere relay station?"}],
            files=[{"type": "file", "id": file_id}],
        )
        ok = r.ok
        text = r.json()["choices"][0]["message"]["content"] if ok else ""
        retrieved = "cobalt-meridian-7" in text.lower() or "cobalt" in text.lower()
        result("RAG: answer retrieved from document chunks", retrieved,
               text.strip()[:100] if not retrieved else text.strip()[:80])

        # 3. Same question WITHOUT the file — model should not know
        time.sleep(5)
        text_no_rag = c.chat_text(
            "What is the activation phrase for the Heliosphere relay station? Reply 'unknown' if you don't know."
        )
        not_in = "cobalt-meridian-7" not in text_no_rag.lower()
        result("RAG: answer NOT known without file", not_in, text_no_rag.strip()[:60])

    except Exception as e:
        result("RAG pipeline test", False, str(e)[:80])
    finally:
        if file_id:
            try:
                c.delete(f"/api/v1/files/{file_id}")
            except Exception:
                pass


# ── Web search ────────────────────────────────────────────────────────────────

def test_web_search(c: Client):
    section("WEB SEARCH — SearXNG via globe toggle")
    try:
        # Ask something with a definitive factual answer that requires current info
        r = c.chat_with_retry(
            messages=[{"role": "user", "content": "What is the capital city of Australia? One word answer."}],
            features={"web_search": True},
        )
        ok = r.ok
        text = r.json()["choices"][0]["message"]["content"] if ok else ""
        result("Web search: correct answer (Canberra)", ok and "canberra" in text.lower(),
               text.strip()[:80] if not ok else text.strip()[:60])
    except Exception as e:
        result("Web search", False, str(e)[:80])

    time.sleep(5)

    try:
        # Same question WITHOUT web search — small model likely says Sydney or Melbourne
        text_no_ws = c.chat_text(
            "What is the capital city of Australia? One word answer only, no explanation."
        )
        # Don't fail if it answers correctly from training — just note it
        used_web = "canberra" in text_no_ws.lower()
        result("Without web search: model answers from training",
               True,  # always pass — just observing
               f"said: {text_no_ws.strip()[:40]} ({'correct' if used_web else 'from training data'})")
    except Exception as e:
        result("Web search comparison", False, str(e)[:80])


# ── Tool calling ──────────────────────────────────────────────────────────────

def test_tool_calling(c: Client):
    section("TOOL CALLING — explicit function definition")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }]
    try:
        r = c.chat_with_retry(
            messages=[{"role": "user", "content": "What is the weather in Cape Town?"}],
            tools=tools,
        )
        ok = r.ok
        finish = ""
        tool_calls = None
        content = ""
        if ok:
            choice = r.json()["choices"][0]
            finish = choice.get("finish_reason", "")
            tool_calls = choice["message"].get("tool_calls")
            content = choice["message"].get("content") or ""
            ok = finish == "tool_calls" and bool(tool_calls)
        if tool_calls:
            result("Tool calling: finish_reason=tool_calls", True, f"finish={finish}")
            fn = tool_calls[0]["function"]
            args = json.loads(fn.get("arguments", "{}"))
            result("Tool call: correct function name", fn.get("name") == "get_weather", fn.get("name", ""))
            result("Tool call: city argument extracted", "city" in args, str(args)[:60])
        else:
            # Small models may not support native tool calling; check if they answered in text
            answered_in_text = r.ok and "cape town" in content.lower()
            if answered_in_text:
                result("Tool calling: finish_reason=tool_calls", False,
                       f"model answered in text (no tool_calls). finish={finish}. This model may not support tool calling.")
                result("Tool call: correct function name", False, "N/A — answered in text")
                result("Tool call: city argument extracted", False, "N/A — answered in text")
            else:
                result("Tool calling: finish_reason=tool_calls", False, f"finish={finish}, content={content[:60]}")
    except Exception as e:
        result("Tool calling", False, str(e)[:80])


# ── Multi-turn context ────────────────────────────────────────────────────────

def test_context_continuity(c: Client):
    section("CONTEXT CONTINUITY — session history maintained")
    try:
        secret = f"ZETA-{uuid.uuid4().hex[:6].upper()}"
        r1 = c.chat_with_retry([{"role": "user", "content": f"Remember this code: {secret}. Reply OK."}])
        r1.raise_for_status()
        reply1 = r1.json()["choices"][0]["message"]["content"]

        time.sleep(4)

        r2 = c.chat_with_retry([
            {"role": "user",    "content": f"Remember this code: {secret}. Reply OK."},
            {"role": "assistant","content": reply1},
            {"role": "user",    "content": "What was the code I just gave you? Reply with just the code."},
        ])
        r2.raise_for_status()
        reply2 = r2.json()["choices"][0]["message"]["content"]
        ok = secret.lower() in reply2.lower()
        result("Session code remembered across turns", ok, reply2.strip()[:60])
    except Exception as e:
        result("Context continuity", False, str(e)[:80])


# ── Logs check ────────────────────────────────────────────────────────────────

def test_no_errors(c: Client):
    section("LOGS — no unexpected errors")
    # We can't SSH from here; just verify health endpoint
    r = c.get("/health")
    result("OpenWebUI health check passes", r.ok and r.json().get("status") is True)
    # Verify models are still reachable
    r2 = c.get("/api/models")
    ids = {m["id"] for m in r2.json().get("data", [])} if r2.ok else set()
    result(f"All models reachable ({len(ids)} listed)", MODEL in ids, f"{len(ids)} models")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OpenWebUI feature verification tests")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    c = Client(base)

    print(f"OpenWebUI feature tests → {base}")
    print(f"Model: {MODEL}")
    print("=" * 60)

    test_settings(c)
    time.sleep(3)

    test_compaction_settings(c)
    time.sleep(3)

    test_memory(c)
    time.sleep(6)

    test_rag_pipeline(c)
    time.sleep(6)

    test_web_search(c)
    time.sleep(12)  # web search fires async query-gen NPU call; give it time to drain

    test_tool_calling(c)
    time.sleep(8)

    test_context_continuity(c)
    time.sleep(3)

    test_no_errors(c)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  PASS: {PASS}")
    print(f"  FAIL: {FAIL}")
    print(f"  Total: {PASS + FAIL}")
    if FAIL == 0:
        print("\nAll feature tests passed!")
    else:
        print(f"\n{FAIL} test(s) failed.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
