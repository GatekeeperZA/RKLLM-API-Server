#!/usr/bin/env python3
"""
RKLLM API Patcher — Timeout & Prefill Fixes (TransferEncodingError RCA)
========================================================================
Usage:
    python3 fix_api_timeouts.py api.py                  # Dry run
    python3 fix_api_timeouts.py api.py --apply          # Patch in-place
    python3 fix_api_timeouts.py api.py -o api_fixed.py  # Write to new file

Root cause: 4B+ models on RK3588 NPU can take 60-90s to prefill long
contexts.  The 30s REQUEST_STALE_TIMEOUT declares the request dead before
the first token arrives, allowing the monitor to unload the model and
abort the generation mid-stream.

Fixes:
    1. [CRITICAL] Raise REQUEST_STALE_TIMEOUT 30 → 180s
    2. [CRITICAL] Raise FIRST_TOKEN_TIMEOUT 120 → 300s
    3. [MODERATE] Raise FALLBACK_SILENCE 12 → 20s
    4. [CRITICAL] Monitor: add GENERATION_COMPLETE guard against unload during inference
    5. [MODERATE] SSE heartbeats during long prefill to keep HTTP connection alive
    6. [MINOR]    Update request activity at generation start (not just on tokens)

NOTE: Apply AFTER fix_api.py (the audit patcher). Order doesn't matter for
      non-overlapping fixes, but this patcher was tested against the
      unpatched source. Both patchers use non-overlapping patterns.
"""

import sys
import os

FIXES = [

    # === FIX 1 — CRITICAL: REQUEST_STALE_TIMEOUT 30 → 180 ================
    # 30s is shorter than NPU prefill time for 4B models with long context.
    # During prefill, no tokens arrive → request looks "idle" → declared
    # stale → monitor can unload model → generation aborted.
    # 180s accommodates worst-case prefill (4B model, ~3000 tokens, NPU).
    # =====================================================================
    (
        'FIX 1 [CRITICAL]: REQUEST_STALE_TIMEOUT 30 \u2192 180',

        'REQUEST_STALE_TIMEOUT = 30',

        'REQUEST_STALE_TIMEOUT = 180',
    ),

    # === FIX 2 — CRITICAL: FIRST_TOKEN_TIMEOUT 120 → 300 =================
    # 120s is tight for 4B+ models with 2500+ token prompts on RK3588.
    # Qwen3-4B at ~2700 tokens takes ~70s, but degraded conditions
    # (thermal throttling, concurrent VL encoder) can push to 150-200s.
    # 300s provides margin without being absurdly long.
    # =====================================================================
    (
        'FIX 2 [CRITICAL]: FIRST_TOKEN_TIMEOUT 120 \u2192 300',

        'FIRST_TOKEN_TIMEOUT = 120     # Max wait for first token (includes prefill)',

        'FIRST_TOKEN_TIMEOUT = 300     # Max wait for first token (includes prefill on NPU)',
    ),

    # === FIX 3 — MODERATE: FALLBACK_SILENCE 12 → 20 ======================
    # NPU token generation can occasionally stall for 10-15s during memory
    # management (GC, page faults on large KV cache).  12s is too tight.
    # =====================================================================
    (
        'FIX 3 [MODERATE]: FALLBACK_SILENCE 12 \u2192 20',

        'FALLBACK_SILENCE = 12         # Max silence between tokens after first token',

        'FALLBACK_SILENCE = 20         # Max silence between tokens after first token',
    ),

    # === FIX 4 — CRITICAL: Monitor GENERATION_COMPLETE guard ==============
    # The monitor already checks `not is_request_active()`, but that relies
    # on REQUEST_STALE_TIMEOUT — the very value that was too low.  Adding
    # a GENERATION_COMPLETE check is defense-in-depth: even if the request
    # tracker declares a request stale, we never unload while the NPU is
    # actively running inference.
    #
    # Applied to BOTH text model and VL model auto-unload blocks.
    # =====================================================================

    # Fix 4a — text model auto-unload
    (
        'FIX 4a [CRITICAL]: Monitor GENERATION_COMPLETE guard (text model)',

        '            # Auto-unload text model after idle timeout\n'
        '            if (IDLE_UNLOAD_TIMEOUT > 0 and is_model_loaded()\n'
        '                    and not is_request_active() and LAST_REQUEST_TIME > 0\n'
        '                    and (time.time() - LAST_REQUEST_TIME) > IDLE_UNLOAD_TIMEOUT):',

        '            # Auto-unload text model after idle timeout\n'
        '            if (IDLE_UNLOAD_TIMEOUT > 0 and is_model_loaded()\n'
        '                    and not is_request_active()\n'
        '                    and GENERATION_COMPLETE.is_set()  # Never unload during active inference\n'
        '                    and LAST_REQUEST_TIME > 0\n'
        '                    and (time.time() - LAST_REQUEST_TIME) > IDLE_UNLOAD_TIMEOUT):',
    ),

    # Fix 4b — VL model auto-unload
    (
        'FIX 4b [CRITICAL]: Monitor GENERATION_COMPLETE guard (VL model)',

        '            # Auto-unload VL model after idle timeout (frees NPU memory)\n'
        '            if (VL_IDLE_UNLOAD_TIMEOUT > 0 and VL_CURRENT_MODEL\n'
        '                    and not is_request_active() and VL_LAST_REQUEST_TIME > 0\n'
        '                    and (time.time() - VL_LAST_REQUEST_TIME) > VL_IDLE_UNLOAD_TIMEOUT):',

        '            # Auto-unload VL model after idle timeout (frees NPU memory)\n'
        '            if (VL_IDLE_UNLOAD_TIMEOUT > 0 and VL_CURRENT_MODEL\n'
        '                    and not is_request_active()\n'
        '                    and GENERATION_COMPLETE.is_set()  # Never unload during active inference\n'
        '                    and VL_LAST_REQUEST_TIME > 0\n'
        '                    and (time.time() - VL_LAST_REQUEST_TIME) > VL_IDLE_UNLOAD_TIMEOUT):',
    ),

    # === FIX 5 — MODERATE: SSE heartbeats during long prefill =============
    # During NPU prefill (10-90s), no data is sent on the SSE stream.
    # Reverse proxies (nginx, Cloudflare) and HTTP clients may close the
    # connection after 30-60s of silence, causing TransferEncodingError.
    #
    # SSE spec allows comment lines (starting with ':') as keep-alive.
    # These are silently ignored by compliant clients (including Open WebUI,
    # browsers, and the OpenAI Python SDK).
    #
    # IMPORTANT: The RCA suggested `data: {}\n\n` — this is WRONG.
    # That's a valid SSE event with empty JSON, which strict OpenAI-format
    # parsers will try to parse as a chat completion chunk and choke on.
    # SSE comments (`:heartbeat\n\n`) are the correct mechanism.
    # =====================================================================
    (
        'FIX 5 [MODERATE]: SSE heartbeats during prefill',

        # The role chunk is sent, then the loop starts reading from the queue.
        # We insert heartbeat logic into the queue-read timeout path.
        # When waiting for the first token, instead of aborting on queue.Empty,
        # we send an SSE comment and continue waiting (up to FIRST_TOKEN_TIMEOUT).

        '        while True:\n'
        '            # Check abort\n'
        '            if ABORT_EVENT.is_set():\n'
        '                logger.info(f"[{request_id}] Abort signal received")\n'
        '                _active_wrapper.abort()\n'
        '                break\n'
        '\n'
        '            # Check overall timeout\n'
        '            if time.time() - generation_start > GENERATION_TIMEOUT:\n'
        '                logger.warning(f"[{request_id}] Generation timeout ({GENERATION_TIMEOUT}s)")\n'
        '                _active_wrapper.abort()\n'
        '                break\n'
        '\n'
        '            # Determine queue read timeout\n'
        '            if not got_first_token:\n'
        '                remaining = GENERATION_TIMEOUT - (time.time() - generation_start)\n'
        '                get_timeout = min(FIRST_TOKEN_TIMEOUT, max(0.1, remaining))\n'
        '            else:\n'
        '                get_timeout = FALLBACK_SILENCE\n'
        '\n'
        '            try:\n'
        '                msg_type, msg_data = _token_queue.get(timeout=get_timeout)\n'
        '            except queue.Empty:\n'
        '                label = "first token" if not got_first_token else "silence"\n'
        '                logger.warning(f"[{request_id}] Timeout waiting for {label}")\n'
        '                _active_wrapper.abort()\n'
        '                break',

        '        _heartbeat_interval = 15   # Send SSE comment every N seconds during prefill\n'
        '        _last_heartbeat = time.time()\n'
        '\n'
        '        while True:\n'
        '            # Check abort\n'
        '            if ABORT_EVENT.is_set():\n'
        '                logger.info(f"[{request_id}] Abort signal received")\n'
        '                _active_wrapper.abort()\n'
        '                break\n'
        '\n'
        '            # Check overall timeout\n'
        '            if time.time() - generation_start > GENERATION_TIMEOUT:\n'
        '                logger.warning(f"[{request_id}] Generation timeout ({GENERATION_TIMEOUT}s)")\n'
        '                _active_wrapper.abort()\n'
        '                break\n'
        '\n'
        '            # Determine queue read timeout\n'
        '            if not got_first_token:\n'
        '                # Short timeout so we can send heartbeats during long prefill\n'
        '                elapsed_prefill = time.time() - generation_start\n'
        '                if elapsed_prefill > FIRST_TOKEN_TIMEOUT:\n'
        '                    logger.warning(f"[{request_id}] Timeout waiting for first token")\n'
        '                    _active_wrapper.abort()\n'
        '                    break\n'
        '                get_timeout = min(_heartbeat_interval, FIRST_TOKEN_TIMEOUT - elapsed_prefill)\n'
        '                get_timeout = max(0.1, get_timeout)\n'
        '            else:\n'
        '                get_timeout = FALLBACK_SILENCE\n'
        '\n'
        '            try:\n'
        '                msg_type, msg_data = _token_queue.get(timeout=get_timeout)\n'
        '            except queue.Empty:\n'
        '                if not got_first_token:\n'
        '                    # Prefill still running — send SSE heartbeat to keep\n'
        '                    # HTTP connection alive through reverse proxies.\n'
        '                    # SSE comments (lines starting with \':\') are silently\n'
        '                    # ignored by all compliant clients.\n'
        '                    now = time.time()\n'
        '                    if now - _last_heartbeat >= _heartbeat_interval:\n'
        '                        elapsed_s = int(now - generation_start)\n'
        '                        yield f": heartbeat prefill {elapsed_s}s\\n\\n"\n'
        '                        _last_heartbeat = now\n'
        '                        update_request_activity()  # Keep request tracker alive too\n'
        '                        logger.debug(f"[{request_id}] SSE heartbeat at {elapsed_s}s (prefill)")\n'
        '                    continue  # Go back to queue.get — prefill still in progress\n'
        '                else:\n'
        '                    logger.warning(f"[{request_id}] Timeout waiting for next token (silence)")\n'
        '                    _active_wrapper.abort()\n'
        '                    break',
    ),

    # === FIX 6 — MINOR: Update request activity at generation start =======
    # The request activity timestamp is set at try_start_request() time,
    # but model loading can take 10-30s.  By the time rkllm_run() starts,
    # the request may already be 30s+ old with the old STALE_TIMEOUT.
    # With the new 180s timeout this is less critical, but still good
    # practice: update activity right before starting the worker thread.
    # =====================================================================
    (
        'FIX 6 [MINOR]: Update activity before worker start (streaming)',

        '    try:\n'
        '        GENERATION_COMPLETE.clear()\n'
        '\n'
        '        # First chunk: role\n'
        '        yield make_sse_chunk(request_id, model_name, created, delta={"role": "assistant"})',

        '    try:\n'
        '        GENERATION_COMPLETE.clear()\n'
        '        update_request_activity()  # Refresh after model load delay\n'
        '\n'
        '        # First chunk: role\n'
        '        yield make_sse_chunk(request_id, model_name, created, delta={"role": "assistant"})',
    ),

]


def apply_fixes(content):
    """Apply all fixes. Returns (fixed_content, results_list)."""
    results = []
    fixed = content

    for label, old, new in FIXES:
        count = fixed.count(old)
        if count == 0:
            results.append((label, "SKIPPED \u2014 pattern not found (already applied?)"))
        elif count > 1:
            results.append((label, f"SKIPPED \u2014 pattern found {count}x (must be unique)"))
        else:
            fixed = fixed.replace(old, new, 1)
            results.append((label, "APPLIED"))

    return fixed, results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    apply_flag = '--apply' in sys.argv
    output_path = None
    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]
        else:
            print("ERROR: -o requires an output path", file=sys.stderr)
            sys.exit(1)

    if not os.path.isfile(input_path):
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    line_count = content.count('\n') + 1
    print(f"Input: {input_path} ({len(content):,} chars, {line_count:,} lines)")
    print(f"Fixes: {len(FIXES)}")
    print()

    fixed, results = apply_fixes(content)

    applied = 0
    skipped = 0
    for label, status in results:
        marker = "\u2713" if status == "APPLIED" else "\u26a0"
        print(f"  {marker} {label}")
        if status != "APPLIED":
            print(f"    \u2192 {status}")
            skipped += 1
        else:
            applied += 1

    print()
    print(f"Summary: {applied} applied, {skipped} skipped")

    if fixed == content:
        print("\nNo changes made.")
        return

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(fixed)
        print(f"\nWritten to: {output_path}")
    elif apply_flag:
        backup = input_path + '.bak'
        if not os.path.exists(backup):
            with open(backup, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"\nBackup: {backup}")
        else:
            print(f"\nBackup already exists: {backup} (not overwritten)")
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write(fixed)
        print(f"Patched: {input_path}")
    else:
        print("\nDRY RUN \u2014 no files modified.")
        print("  --apply     Patch in-place (creates .bak backup)")
        print("  -o <path>   Write patched file to new path")


if __name__ == '__main__':
    main()