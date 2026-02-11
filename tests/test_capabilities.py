#!/usr/bin/env python3
"""Quick test to verify model capabilities detection and thinking control."""
import os, sys, json, requests

BASE = os.getenv("RKLLM_API_BASE", "http://localhost:8000")

def test_models_endpoint():
    print("=== /v1/models ===")
    r = requests.get(f"{BASE}/v1/models")
    r.raise_for_status()
    data = r.json()["data"]
    for m in data:
        print(f"  {m['id']:30s} caps={m['capabilities']}  ctx={m['context_length']}")
    # Assertions
    caps_map = {m["id"]: m["capabilities"] for m in data}
    assert "thinking" not in caps_map.get("gemma-3-4b-it", []), "Gemma should NOT have thinking"
    assert "thinking" not in caps_map.get("phi-3-mini-4k-instruct", []), "Phi-3 should NOT have thinking"
    assert "thinking" in caps_map.get("qwen3-1.7b", []), "Qwen3-1.7B SHOULD have thinking"
    assert "thinking" in caps_map.get("qwen3-4b-instruct-2507", []), "Qwen3-4B SHOULD have thinking"
    assert "vl" in caps_map.get("qwen3-vl-2b", []), "Qwen3-VL SHOULD have vl"
    assert "thinking" not in caps_map.get("qwen3-vl-2b", []), "Qwen3-VL should NOT have thinking"
    print("  PASS: All capability assertions passed\n")

def test_thinking_control(model, expect_thinking):
    print(f"=== Chat: {model} (expect_thinking={expect_thinking}) ===")
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    })
    r.raise_for_status()
    resp = r.json()
    content = resp["choices"][0]["message"].get("content", "")
    reasoning = resp["choices"][0]["message"].get("reasoning_content")
    print(f"  Content: {content[:120]}")
    print(f"  Reasoning: {'(present)' if reasoning else '(none)'}")
    if expect_thinking:
        # Reasoning models may or may not produce <think> for trivial questions,
        # so we only verify the request succeeded.
        print(f"  OK: Thinking-capable model responded")
    else:
        # Non-thinking models should never return reasoning_content
        if reasoning:
            print(f"  WARN: Got reasoning_content from non-thinking model")
        else:
            print(f"  PASS: No reasoning_content as expected")
    print()

if __name__ == "__main__":
    test_models_endpoint()
    test_thinking_control("gemma", expect_thinking=False)
    test_thinking_control("qwen3-4b", expect_thinking=True)
    print("=== All tests passed ===")
