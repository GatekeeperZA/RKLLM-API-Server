#!/usr/bin/env python3
"""Test: multi-image VL support.

Sends 2 tiny synthetic images (red square + blue square) to the
/v1/chat/completions endpoint and checks that the model responds.
"""

import base64
import io
import json
import os
import sys
import time

# pip install Pillow requests
from PIL import Image
import requests

API_BASE = os.getenv("RKLLM_API_BASE", "http://localhost:8000")
ENDPOINT = f"{API_BASE}/v1/chat/completions"


def make_solid_image(color, size=64):
    """Create a small solid-color PNG and return base64 data URI."""
    img = Image.new("RGB", (size, size), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def test_single_image():
    """Baseline: single image should work."""
    print("=== TEST 1: Single image ===")
    red = make_solid_image((255, 0, 0))
    payload = {
        "model": "qwen3-vl-2b",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": red}},
                    {"type": "text", "text": "What color is this image? Reply in one word."},
                ],
            }
        ],
    }
    t0 = time.time()
    r = requests.post(ENDPOINT, json=payload, timeout=120)
    elapsed = time.time() - t0
    print(f"  Status: {r.status_code}  ({elapsed:.1f}s)")
    if r.status_code == 200:
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
        print(f"  Reply: {reply[:200]}")
        return True
    else:
        print(f"  ERROR: {r.text[:300]}")
        return False


def test_two_images():
    """Multi-image: two images in one message."""
    print("\n=== TEST 2: Two images ===")
    red = make_solid_image((255, 0, 0))
    blue = make_solid_image((0, 0, 255))
    payload = {
        "model": "qwen3-vl-2b",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": red}},
                    {"type": "image_url", "image_url": {"url": blue}},
                    {"type": "text", "text": "I sent two images. What color is each one? Reply briefly."},
                ],
            }
        ],
    }
    t0 = time.time()
    r = requests.post(ENDPOINT, json=payload, timeout=120)
    elapsed = time.time() - t0
    print(f"  Status: {r.status_code}  ({elapsed:.1f}s)")
    if r.status_code == 200:
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
        print(f"  Reply: {reply[:300]}")
        return True
    else:
        print(f"  ERROR: {r.text[:300]}")
        return False


def test_two_images_stream():
    """Multi-image + streaming."""
    print("\n=== TEST 3: Two images (streaming) ===")
    red = make_solid_image((255, 0, 0))
    green = make_solid_image((0, 255, 0))
    payload = {
        "model": "qwen3-vl-2b",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": red}},
                    {"type": "image_url", "image_url": {"url": green}},
                    {"type": "text", "text": "Describe the two images briefly."},
                ],
            }
        ],
    }
    t0 = time.time()
    r = requests.post(ENDPOINT, json=payload, timeout=120, stream=True)
    print(f"  Status: {r.status_code}")
    if r.status_code != 200:
        print(f"  ERROR: {r.text[:300]}")
        return False

    full = ""
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content", "")
            full += content
        except Exception:
            pass
    elapsed = time.time() - t0
    print(f"  Reply ({elapsed:.1f}s): {full[:300]}")
    return bool(full.strip())


if __name__ == "__main__":
    results = []
    results.append(("Single image", test_single_image()))
    results.append(("Two images", test_two_images()))
    results.append(("Two images stream", test_two_images_stream()))

    print("\n" + "=" * 50)
    print("RESULTS:")
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    if all(ok for _, ok in results):
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed.")
        sys.exit(1)
