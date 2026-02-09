"""
RKLLM API Server — NPU Model Benchmarks
=========================================
Benchmarks all available NPU models with standardized prompts to measure:
  - Cold load time (model not loaded → first token)
  - Warm prefill + TTFT (model already loaded)
  - Generation speed (tokens/second)
  - Total response time
  - Output quality (response length, coherence check)
  - NPU memory usage (from server logs)

Results are printed as a formatted table suitable for README inclusion.

Usage:
    python tests/benchmark_test.py                  # Benchmark all models
    python tests/benchmark_test.py --models qwen3-1.7b phi-3-mini-4k-instruct
    python tests/benchmark_test.py --skip-vl        # Skip VL model
    python tests/benchmark_test.py --runs 3         # Average over 3 runs
    python tests/benchmark_test.py --remote-log     # Fetch NPU perf from server log via SSH

Requires: network access to the Orange Pi running the API server.
"""
import json, sys, time, urllib.request, urllib.error, subprocess, re, argparse, os

# =============================================================================
# CONFIGURATION
# =============================================================================
API = "http://192.168.2.180:8000"
SSH_HOST = "orangepi"  # SSH alias for fetching server logs
LOG_PATH = "~/rkllm_api/api.log"  # Also tries rkllm_api.log
TIMEOUT = 300  # seconds per request

# Standard benchmark prompts — varying complexity
PROMPTS = {
    "short": {
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "label": "Short Q&A",
        "description": "Simple factual question (~5-15 tokens output)",
    },
    "medium": {
        "messages": [{"role": "user", "content": "Explain how photosynthesis works in plants. Include the light reactions and the Calvin cycle."}],
        "label": "Medium explanation",
        "description": "Technical explanation (~80-200 tokens output)",
    },
    "long": {
        "messages": [{"role": "user", "content": "Write a detailed comparison of Python and JavaScript as programming languages. Cover syntax differences, type systems, common use cases, ecosystem, performance characteristics, and which scenarios each language excels in. Use multiple paragraphs."}],
        "label": "Long generation",
        "description": "Extended multi-paragraph response (~200-500 tokens output)",
    },
    "reasoning": {
        "messages": [{"role": "user", "content": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Think step by step."}],
        "label": "Reasoning",
        "description": "Logic puzzle requiring step-by-step thinking",
    },
}

# VL benchmark (only for models with vision capability)
VL_PROMPT = {
    "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe what you see in this image in detail."},
            {"type": "image_url", "image_url": {"url": None}},  # Filled at runtime
        ]}
    ],
    "label": "VL image description",
    "description": "Vision-language: describe a test image",
}

# Chars per token estimate (for tok/s from client side when server doesn't return it)
CHARS_PER_TOKEN = 3.5


# =============================================================================
# HELPERS
# =============================================================================
def api_request(method, path, body=None, stream=False):
    """Send HTTP request. Returns (status, data, elapsed_s) or streaming iterator."""
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {}
    )
    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
        if stream:
            return resp, start
        body_bytes = resp.read()
        elapsed = time.time() - start
        return resp.status, json.loads(body_bytes), elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start
        try:
            err_body = json.loads(e.read())
        except Exception:
            err_body = {"error": str(e)}
        return e.code, err_body, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return 0, {"error": str(e)}, elapsed


def unload_model():
    """Unload currently loaded model to force cold start."""
    api_request("POST", "/v1/models/unload")
    time.sleep(1)


def get_models():
    """Get list of available model IDs."""
    status, data, _ = api_request("GET", "/v1/models")
    if status == 200 and "data" in data:
        return [m["id"] for m in data["data"]]
    return []


def stream_request(model, messages, max_tokens=512):
    """Send a streaming request, measure TTFT and generation speed.
    
    Returns dict with timing and output info.
    """
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "stream_options": {"include_usage": True},
    }
    url = f"{API}/v1/chat/completions"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"}
    )

    request_start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
    except Exception as e:
        return {"error": str(e), "total_time": time.time() - request_start}

    first_token_time = None
    full_content = ""
    reasoning_content = ""
    token_chunks = 0
    usage_data = {}
    last_chunk_time = request_start

    # Read in small chunks for accurate timing (byte-by-byte is too slow,
    # line-based batches destroy timing — 4KB is a good balance)
    buffer = ""
    done = False
    while not done:
        raw = resp.read(4096)
        if not raw:
            break
        buffer += raw.decode("utf-8", errors="replace")

        while "\n\n" in buffer:
            line, buffer = buffer.split("\n\n", 1)
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                break

            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue

            # Check for usage in the chunk
            if "usage" in obj and obj["usage"]:
                usage_data = obj["usage"]

            choices = obj.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})

            content = delta.get("content", "")
            reasoning = delta.get("reasoning_content", "")

            if (content or reasoning) and first_token_time is None:
                first_token_time = time.time()

            if content:
                full_content += content
                token_chunks += 1
                last_chunk_time = time.time()

            if reasoning:
                reasoning_content += reasoning
                token_chunks += 1
                last_chunk_time = time.time()

    total_time = time.time() - request_start
    ttft = (first_token_time - request_start) if first_token_time else total_time
    generation_time = (last_chunk_time - first_token_time) if first_token_time else 0

    # Token counts from usage or estimate
    completion_tokens = usage_data.get("completion_tokens", 0)
    prompt_tokens = usage_data.get("prompt_tokens", 0)
    if completion_tokens == 0:
        completion_tokens = max(1, int(len(full_content) / CHARS_PER_TOKEN))

    # Generation speed (tok/s) — only counting generation phase after first token
    tok_per_sec = (completion_tokens / generation_time) if generation_time > 0.1 else 0

    return {
        "content": full_content,
        "reasoning": reasoning_content,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_time": total_time,
        "ttft": ttft,
        "generation_time": generation_time,
        "tok_per_sec": tok_per_sec,
        "token_chunks": token_chunks,
    }


def non_stream_request(model, messages, max_tokens=512):
    """Send a non-streaming request for comparison."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
    }
    status, data, elapsed = api_request("POST", "/v1/chat/completions", body)
    if status != 200:
        return {"error": data, "total_time": elapsed}

    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    reasoning = choice.get("message", {}).get("reasoning_content", "")
    usage = data.get("usage", {})

    return {
        "content": content,
        "reasoning": reasoning,
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "total_time": elapsed,
    }


def fetch_server_perf(request_id=None, last_n=1):
    """Fetch NPU perf stats from server log via SSH.
    
    Returns list of dicts with prefill_ms, prefill_tok, generate_ms, generate_tok, mem_mb.
    """
    try:
        # Fetch from the most recent log file
        cmd = f'ssh {SSH_HOST} "grep \'Perf:\' {LOG_PATH} 2>/dev/null | tail -{last_n}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return []

        entries = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            m = re.search(
                r'prefill=(\d+)ms\s*\((\d+)\s*tok\).*?'
                r'generate=(\d+)ms\s*\((\d+)\s*tok\).*?'
                r'mem=(\d+)MB',
                line
            )
            if m:
                entries.append({
                    "prefill_ms": int(m.group(1)),
                    "prefill_tok": int(m.group(2)),
                    "generate_ms": int(m.group(3)),
                    "generate_tok": int(m.group(4)),
                    "mem_mb": int(m.group(5)),
                })
        return entries
    except Exception:
        return []


def fetch_load_time(model_id):
    """Fetch most recent model load time from server log."""
    try:
        cmd = f'ssh {SSH_HOST} "grep \'{model_id} LOADED\' {LOG_PATH} 2>/dev/null | tail -1"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        m = re.search(r'LOADED successfully in (\d+\.\d+)s', result.stdout)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def generate_test_image_base64():
    """Generate a small solid-color test image as base64 PNG.
    
    Uses raw PNG construction — no Pillow dependency needed.
    """
    import struct, zlib, base64
    width, height = 64, 64
    # Create raw pixel data: solid blue (R=50, G=100, B=200)
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00"  # filter byte (none)
        raw_data += b"\x32\x64\xc8" * width  # RGB pixels

    def png_chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)

    png = b"\x89PNG\r\n\x1a\n"
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(raw_data))
    png += png_chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================
def benchmark_model(model_id, prompts, runs=1, fetch_log=False, is_vl=False):
    """Run all benchmark prompts against a model. Returns list of result dicts."""
    results = []

    for prompt_key, prompt_info in prompts.items():
        messages = prompt_info["messages"]
        label = prompt_info["label"]

        run_results = []
        for run_idx in range(runs):
            if run_idx == 0 and not is_vl:
                # First run: cold start (unload first)
                unload_model()
                time.sleep(1)

            print(f"    {label} (run {run_idx + 1}/{runs})...", end=" ", flush=True)
            r = stream_request(model_id, messages, max_tokens=512)

            if "error" in r:
                print(f"ERROR: {r['error']}")
                run_results.append(r)
                continue

            # Fetch NPU perf from server log if enabled
            npu_perf = {}
            load_time = None
            if fetch_log:
                time.sleep(0.5)  # Let log flush
                perfs = fetch_server_perf(last_n=1)
                if perfs:
                    npu_perf = perfs[-1]
                if run_idx == 0:
                    load_time = fetch_load_time(model_id)

            r["npu_perf"] = npu_perf
            r["load_time"] = load_time
            r["is_cold"] = (run_idx == 0 and not is_vl)

            tok_s = npu_perf.get("generate_tok", 0)
            gen_ms = npu_perf.get("generate_ms", 1)
            if gen_ms > 0 and tok_s > 0:
                npu_tok_s = tok_s / gen_ms * 1000
                perf_src = "npu"
            elif r["tok_per_sec"] > 0:
                npu_tok_s = r["tok_per_sec"]
                perf_src = "est"
            else:
                npu_tok_s = r["completion_tokens"] / max(0.1, r["total_time"] - r["ttft"])
                perf_src = "wall"

            content_len = len(r["content"])
            mem_str = f" mem={npu_perf['mem_mb']}MB" if npu_perf.get('mem_mb') else ""
            print(f"{r['completion_tokens']} tok in {r['total_time']:.1f}s "
                  f"({npu_tok_s:.1f} tok/s [{perf_src}]) ttft={r['ttft']:.2f}s"
                  f"{mem_str} [{content_len} chars]")

            run_results.append(r)

        # Aggregate results across runs
        valid_runs = [r for r in run_results if "error" not in r]
        if valid_runs:
            cold_run = next((r for r in valid_runs if r.get("is_cold")), None)
            warm_runs = [r for r in valid_runs if not r.get("is_cold")]

            agg = {
                "model": model_id,
                "prompt": prompt_key,
                "label": label,
                "runs": len(valid_runs),
            }

            # Cold start metrics (first run only)
            if cold_run:
                agg["cold_ttft"] = cold_run["ttft"]
                agg["cold_total"] = cold_run["total_time"]
                agg["load_time"] = cold_run.get("load_time")

            # Use all runs for generation metrics
            all_times = [r["total_time"] for r in valid_runs]
            all_ttft = [r["ttft"] for r in valid_runs]
            all_tokens = [r["completion_tokens"] for r in valid_runs]
            all_content_len = [len(r["content"]) for r in valid_runs]

            # NPU perf (most accurate)
            npu_perfs = [r["npu_perf"] for r in valid_runs if r.get("npu_perf")]
            if npu_perfs:
                gen_toks = [p["generate_tok"] for p in npu_perfs]
                gen_ms = [p["generate_ms"] for p in npu_perfs]
                prefill_ms = [p["prefill_ms"] for p in npu_perfs]
                prefill_toks = [p["prefill_tok"] for p in npu_perfs]
                mem_mb = [p["mem_mb"] for p in npu_perfs]

                agg["npu_tok_s"] = sum(t / m * 1000 for t, m in zip(gen_toks, gen_ms) if m > 0) / len(gen_ms)
                agg["npu_prefill_ms"] = sum(prefill_ms) / len(prefill_ms)
                agg["npu_prefill_tok"] = sum(prefill_toks) / len(prefill_toks)
                agg["npu_gen_ms"] = sum(gen_ms) / len(gen_ms)
                agg["npu_gen_tok"] = sum(gen_toks) / len(gen_toks)
                agg["mem_mb"] = max(mem_mb)
            else:
                # Client-side estimate
                client_tok_s = [r["tok_per_sec"] for r in valid_runs if r["tok_per_sec"] > 0]
                agg["npu_tok_s"] = sum(client_tok_s) / len(client_tok_s) if client_tok_s else 0

            agg["avg_total_time"] = sum(all_times) / len(all_times)
            agg["avg_ttft"] = sum(all_ttft) / len(all_ttft)
            agg["avg_tokens"] = sum(all_tokens) / len(all_tokens)
            agg["avg_content_len"] = sum(all_content_len) / len(all_content_len)

            # Warm TTFT (excluding cold start)
            if warm_runs:
                agg["warm_ttft"] = sum(r["ttft"] for r in warm_runs) / len(warm_runs)
            elif cold_run:
                agg["warm_ttft"] = cold_run["ttft"]

            results.append(agg)

    return results


def benchmark_vl_model(model_id, runs=1, fetch_log=False):
    """Benchmark VL model with a test image."""
    img_b64 = generate_test_image_base64()
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe what you see in this image in detail."},
            {"type": "image_url", "image_url": {"url": img_b64}},
        ]}
    ]
    prompt_info = {"vl_test": {
        "messages": messages, "label": "VL image describe",
        "description": "Vision-language test image"
    }}
    return benchmark_model(model_id, prompt_info, runs=runs, fetch_log=fetch_log, is_vl=True)


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================
def print_summary_table(all_results):
    """Print a comprehensive summary table."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    # Group by model
    models = {}
    for r in all_results:
        model = r["model"]
        if model not in models:
            models[model] = []
        models[model].append(r)

    for model, results in models.items():
        print(f"\n{'─' * 80}")
        print(f"  Model: {model}")
        print(f"{'─' * 80}")

        # Load time (from cold start)
        load_times = [r.get("load_time") for r in results if r.get("load_time")]
        if load_times:
            print(f"  Cold load time: {load_times[0]:.1f}s")

        # Memory
        mem_vals = [r.get("mem_mb", 0) for r in results if r.get("mem_mb")]
        if mem_vals:
            print(f"  NPU memory: {max(mem_vals)} MB")

        print()
        print(f"  {'Prompt':<22} {'Tokens':>7} {'Total':>8} {'TTFT':>8} {'Gen':>8} {'tok/s':>8} {'Prefill':>10}")
        print(f"  {'─' * 22} {'─' * 7} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 10}")

        for r in results:
            label = r["label"][:22]
            tokens = f"{r['avg_tokens']:.0f}"
            total = f"{r['avg_total_time']:.1f}s"
            ttft = f"{r.get('warm_ttft', r['avg_ttft']):.2f}s"
            gen_ms = f"{r.get('npu_gen_ms', 0):.0f}ms" if r.get('npu_gen_ms') else "—"
            tok_s = f"{r.get('npu_tok_s', 0):.1f}"
            prefill = f"{r.get('npu_prefill_ms', 0):.0f}ms ({r.get('npu_prefill_tok', 0):.0f}t)" if r.get('npu_prefill_ms') else "—"

            print(f"  {label:<22} {tokens:>7} {total:>8} {ttft:>8} {gen_ms:>8} {tok_s:>8} {prefill:>10}")


def print_readme_table(all_results):
    """Print results formatted as a Markdown table for README insertion."""
    print("\n\n### Benchmark Results (Orange Pi 5 Plus, RK3588, 3 NPU cores)")
    print()
    print(f"*Benchmarked: {time.strftime('%Y-%m-%d')} | RKLLM Runtime v1.2.3 | RKNPU Driver 0.9.8*")
    print()

    # Model overview table
    print("#### Model Overview")
    print()
    print("| Model | Parameters | Context | Load Time | NPU Memory | Avg tok/s |")
    print("|-------|-----------|---------|-----------|------------|-----------|")

    models = {}
    for r in all_results:
        model = r["model"]
        if model not in models:
            models[model] = []
        models[model].append(r)

    model_meta = {
        "qwen3-1.7b": ("1.7B", "4K"),
        "qwen3-4b-instruct-2507": ("4B", "16K"),
        "gemma-3-4b-it": ("4B", "4K"),
        "phi-3-mini-4k-instruct": ("3.8B", "4K"),
        "deepseekocr": ("3B", "4K"),
    }

    for model, results in models.items():
        params, ctx = model_meta.get(model, ("?", "?"))
        load_times = [r.get("load_time") for r in results if r.get("load_time")]
        load_str = f"{load_times[0]:.1f}s" if load_times else "—"
        mem_vals = [r.get("mem_mb", 0) for r in results if r.get("mem_mb")]
        mem_str = f"{max(mem_vals)} MB" if mem_vals else "—"
        tok_s_vals = [r.get("npu_tok_s", 0) for r in results if r.get("npu_tok_s", 0) > 0]
        avg_tok_s = sum(tok_s_vals) / len(tok_s_vals) if tok_s_vals else 0
        print(f"| **{model}** | {params} | {ctx} | {load_str} | {mem_str} | ~{avg_tok_s:.1f} |")

    # Detailed per-prompt table
    print()
    print("#### Detailed Benchmarks")
    print()
    print("| Model | Prompt | Tokens | TTFT | Generation | tok/s | Prefill |")
    print("|-------|--------|--------|------|------------|-------|---------|")

    for model, results in models.items():
        for r in results:
            ttft = f"{r.get('warm_ttft', r['avg_ttft']):.2f}s"
            gen_ms = f"{r.get('npu_gen_ms', 0):.0f}ms" if r.get('npu_gen_ms') else "—"
            tok_s = f"{r.get('npu_tok_s', 0):.1f}"
            prefill = f"{r.get('npu_prefill_ms', 0):.0f}ms" if r.get('npu_prefill_ms') else "—"
            print(f"| {model} | {r['label']} | {r['avg_tokens']:.0f} | {ttft} | {gen_ms} | {tok_s} | {prefill} |")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="RKLLM NPU Model Benchmarks")
    parser.add_argument("--models", nargs="+", help="Specific model IDs to benchmark")
    parser.add_argument("--skip-vl", action="store_true", help="Skip VL model benchmarks")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per prompt (default: 1)")
    parser.add_argument("--remote-log", action="store_true", default=True,
                        help="Fetch NPU perf stats from server log via SSH (default: on)")
    parser.add_argument("--no-remote-log", action="store_true", help="Disable remote log fetching")
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPTS.keys()),
                        help="Run only specific prompts")
    args = parser.parse_args()

    fetch_log = args.remote_log and not args.no_remote_log

    # Check server connectivity
    print(f"Connecting to {API}...")
    status, data, elapsed = api_request("GET", "/health")
    if status != 200:
        print(f"ERROR: Server not reachable (status={status})")
        sys.exit(1)
    print(f"Server OK ({elapsed:.2f}s) — {data.get('models_available', '?')} models available")
    print()

    # Get models to benchmark
    available = get_models()
    if not available:
        print("ERROR: No models found")
        sys.exit(1)

    # Separate VL models (they have special handling)
    vl_models = [m for m in available if any(vl in m.lower() for vl in
                 ("deepseekocr", "qwen2-vl", "qwen2.5-vl", "qwen3-vl", "internvl", "minicpm"))]
    text_models = [m for m in available if m not in vl_models]

    if args.models:
        text_models = [m for m in text_models if m in args.models]
        vl_models = [m for m in vl_models if m in args.models]

    # Select prompts
    prompts_to_run = {k: v for k, v in PROMPTS.items() if not args.prompts or k in args.prompts}

    print(f"Text models: {text_models}")
    print(f"VL models: {vl_models}")
    print(f"Prompts: {list(prompts_to_run.keys())}")
    print(f"Runs per prompt: {args.runs}")
    print(f"NPU log fetch: {'ON' if fetch_log else 'OFF'}")
    print()

    all_results = []

    # Benchmark each text model
    for model in text_models:
        print(f"\n{'=' * 60}")
        print(f"  BENCHMARKING: {model}")
        print(f"{'=' * 60}")

        results = benchmark_model(model, prompts_to_run, runs=args.runs, fetch_log=fetch_log)
        all_results.extend(results)

    # Benchmark VL models
    if vl_models and not args.skip_vl:
        for model in vl_models:
            print(f"\n{'=' * 60}")
            print(f"  BENCHMARKING (VL): {model}")
            print(f"{'=' * 60}")

            # Text prompts first (VL models can do text too)
            results = benchmark_model(model, prompts_to_run, runs=args.runs, fetch_log=fetch_log)
            all_results.extend(results)

            # VL-specific test
            print(f"\n  --- VL Image Test ---")
            vl_results = benchmark_vl_model(model, runs=args.runs, fetch_log=fetch_log)
            all_results.extend(vl_results)

    # Final unload
    unload_model()

    # Print results
    print_summary_table(all_results)
    print_readme_table(all_results)

    # Save raw results to JSON
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to: {output_path}")


if __name__ == "__main__":
    main()
