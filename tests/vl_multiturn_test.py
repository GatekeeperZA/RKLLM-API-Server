#!/usr/bin/env python3
"""Quick test: VL multi-turn context — verify follow-up questions + RAG
context are included in VL prompt (not just the original image message text).

This simulates what Open WebUI sends when:
1. User uploads image + asks "what is this"
2. Model answers "Venice..."
3. User corrects + does web search -> RAG context injected in system msg
4. OWUI re-sends full conversation -> our API should use the follow-up text

We check the log to confirm VL multi-turn mode was activated.
"""
import requests
import base64
import time
import subprocess
import sys

API = "http://192.168.2.180:8000"

# Generate a small valid test image
from PIL import Image
import io
img_buf = io.BytesIO()
Image.new('RGB', (64, 64), (200, 100, 50)).save(img_buf, format='PNG')
B64_IMG = f"data:image/png;base64,{base64.b64encode(img_buf.getvalue()).decode()}"

# Simulate the exact Open WebUI multi-turn request with RAG
payload = {
    "model": "qwen3-1.7b",  # OWUI sends the text model; VL auto-route kicks in
    "stream": True,
    "messages": [
        {
            "role": "system",
            "content": (
                "User Context:\n\n### Task:\n"
                "Answer the user's question using ONLY the provided context.\n\n"
                "<context>\n"
                '<source id="1" name="test query">'
                "A'DAM LOOKOUT is the iconic observation deck in Amsterdam. "
                "It features a panoramic observation deck, a revolving restaurant, "
                "and the famous Over The Edge swing at 100 meters above the IJ river. "
                "The A'DAM Tower is located in Amsterdam-Noord, directly across the "
                "IJ river from Amsterdam Centraal station."
                "</source>\n"
                "</context>"
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": B64_IMG}},
                {"type": "text", "text": "what is this"},
            ],
        },
        {
            "role": "assistant",
            "content": (
                "This is a detailed scale model of the city of Venice, Italy. "
                "The model showcases the iconic Venetian architecture."
            ),
        },
        {
            "role": "user",
            "content": "this is a model at adam lookout amsterdam, confirm",
        },
    ],
}

print("Sending multi-turn VL request with RAG context...")
resp = requests.post(f"{API}/v1/chat/completions", json=payload, stream=True, timeout=120)
print(f"Status: {resp.status_code}")

if resp.status_code != 200:
    print(f"ERROR: {resp.text[:500]}")
    sys.exit(1)

# Collect streamed response
full_text = ""
for line in resp.iter_lines(decode_unicode=True):
    if line.startswith("data: ") and line != "data: [DONE]":
        import json
        try:
            chunk = json.loads(line[6:])
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            full_text += delta
            print(delta, end="", flush=True)
        except json.JSONDecodeError:
            pass
print(f"\n\nFull response ({len(full_text)} chars):\n{full_text}")

# Check logs for the VL multi-turn marker
time.sleep(1)
print("\n--- Checking logs for VL multi-turn activation ---")
result = subprocess.run(
    ["ssh", "orangepi", "grep", "-c", "VL multi-turn", "/home/armbian/rkllm_api/rkllm_api.log"],
    capture_output=True, text=True,
)
count = result.stdout.strip()
print(f"VL multi-turn log entries: {count}")

result2 = subprocess.run(
    ["ssh", "orangepi", "tail", "-20", "/home/armbian/rkllm_api/rkllm_api.log"],
    capture_output=True, text=True,
)
print("\nRecent log lines:")
print(result2.stdout[-2000:] if len(result2.stdout) > 2000 else result2.stdout)
