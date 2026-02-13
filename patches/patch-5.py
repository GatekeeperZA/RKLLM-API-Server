#!/usr/bin/env python3
"""
RKLLM API Patcher — Advisory Fixes
=====================================================
Usage:
    python3 fix_api_advisories.py api.py                  # Dry run
    python3 fix_api_advisories.py api.py --apply          # Patch in-place
    python3 fix_api_advisories.py api.py -o api_fixed.py  # Write to new file

Fixes:
    1. [ADVISORY] Streaming error handler yields finish_reason="stop" —
       should be "length" since an exception is NOT a clean completion.

    2. [ADVISORY] Double shutdown() on signal during exit — add idempotency
       guard so the second call is a silent no-op (no duplicate log lines).

    3. [ADVISORY] _strip_stale_date_claims false-positives on sentences like
       "Today is the last day to register" — add date-signal validation so
       only matches containing actual date/time tokens are stripped.

    4. [ADVISORY] Non-streaming path has no activity refresh during long NPU
       prefill — mirror the streaming path's heartbeat approach with short
       queue timeouts and periodic update_request_activity() calls.
"""

import sys
import os

FIXES = [

    # === FIX 1 — Streaming error handler finish_reason ==================
    # The except Exception handler yields finish_reason="stop", implying a
    # clean completion.  An unexpected exception is NOT clean — should be
    # "length" (the convention used in the normal-path logic at line 3823).
    # ====================================================================
    (
        'FIX 1 [ADVISORY]: Error handler finish_reason "stop" -> "length"',

        '        logger.error(f"[{request_id}] Stream error: {e}", exc_info=True)\n'
        '        try:\n'
        '            yield make_sse_chunk(request_id, model_name, created, finish_reason="stop")\n'
        '            yield "data: [DONE]\\n\\n"\n'
        '        except Exception:\n'
        '            pass',

        '        logger.error(f"[{request_id}] Stream error: {e}", exc_info=True)\n'
        '        try:\n'
        '            yield make_sse_chunk(request_id, model_name, created, finish_reason="length")\n'
        '            yield "data: [DONE]\\n\\n"\n'
        '        except Exception:\n'
        '            pass',
    ),

    # === FIX 2 — Shutdown idempotency guard =============================
    # Both atexit.register(shutdown) and signal_handler() call shutdown().
    # If a signal arrives during normal exit, shutdown() runs twice,
    # producing duplicate "Shutting down" and "No model to unload" log
    # lines.  Use SHUTDOWN_EVENT (already set at the top of shutdown) as
    # an idempotency guard: if it's already set on entry, skip everything.
    # ====================================================================
    (
        'FIX 2 [ADVISORY]: Shutdown idempotency guard',

        'def shutdown():\n'
        '    """Clean shutdown - destroy text model, VL model, stop monitor."""\n'
        '    logger.info("Shutting down RKLLM API...")\n'
        '    SHUTDOWN_EVENT.set()\n'
        '    ABORT_EVENT.set()',

        'def shutdown():\n'
        '    """Clean shutdown - destroy text model, VL model, stop monitor."""\n'
        '    if SHUTDOWN_EVENT.is_set():\n'
        '        return  # Already shutting down (atexit + signal race)\n'
        '    logger.info("Shutting down RKLLM API...")\n'
        '    SHUTDOWN_EVENT.set()\n'
        '    ABORT_EVENT.set()',
    ),

    # === FIX 3 — Date-signal validation for stale date stripping ========
    # _strip_stale_date_claims uses broad regex patterns that can match
    # non-date sentences like "Today is the last day to register."
    # Fix: after each regex match, verify the matched text contains an
    # actual date/time signal (month name, year, digit:digit time, etc.)
    # before stripping.  Only the replacement loop changes — the patterns
    # themselves are untouched.
    # ====================================================================
    (
        'FIX 3 [ADVISORY]: Date-signal validation for stale date stripping',

        '    result = text\n'
        '    for pat in _STALE_PATTERNS:\n'
        '        matches = list(re.finditer(pat, result, re.IGNORECASE))\n'
        '        if matches:\n'
        '            for m in reversed(matches):\n'
        '                stripped = m.group().strip()\n'
        '                logger.debug(f"Stripped stale date claim: \'{stripped[:80]}\'")\n'
        '            result = re.sub(pat, \'\', result, flags=re.IGNORECASE)',

        '    # Date-signal pattern: a match must contain at least one of these to\n'
        '    # be considered an actual date/time claim (prevents false-positives on\n'
        '    # sentences like "Today is the last day to register").\n'
        '    _DATE_SIGNAL = re.compile(\n'
        '        r\'(?:\'\n'
        '        r\'\\d{4}\'\n'
        '        r\'|\\d{1,2}[/\\-]\\d{1,2}\'\n'
        '        r\'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\'\n'
        '        r\'|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\'\n'
        '        r\'|\\d{1,2}\\s*:\\s*\\d{2}\'\n'
        '        r\'|\\d{1,2}\\s*(?:am|pm)\'\n'
        '        r\')\', re.IGNORECASE)\n'
        '    result = text\n'
        '    for pat in _STALE_PATTERNS:\n'
        '        # Use finditer + reverse replacement so we can validate each match\n'
        '        matches = list(re.finditer(pat, result, re.IGNORECASE))\n'
        '        if matches:\n'
        '            for m in reversed(matches):\n'
        '                matched = m.group()\n'
        '                if _DATE_SIGNAL.search(matched):\n'
        '                    logger.debug(f"Stripped stale date claim: \'{matched.strip()[:80]}\'")\n'
        '                    result = result[:m.start()] + result[m.end():]\n'
        '                else:\n'
        '                    logger.debug(f"Kept non-date sentence: \'{matched.strip()[:80]}\'\")',
    ),

    # === FIX 4 — Non-streaming activity refresh during prefill ==========
    # The streaming path sends heartbeats during long NPU prefill with
    # short queue timeouts + update_request_activity().  The non-streaming
    # path blocks for up to FIRST_TOKEN_TIMEOUT (300s) in a single
    # queue.get() — no activity refresh.  With REQUEST_STALE_TIMEOUT=180s,
    # a >180s prefill would mark the request stale.
    #
    # Fix: mirror the streaming approach — use short queue timeouts during
    # prefill, check FIRST_TOKEN_TIMEOUT manually, and call
    # update_request_activity() on each empty poll.
    # ====================================================================
    (
        'FIX 4 [ADVISORY]: Non-streaming activity refresh during prefill',

        '            if not got_first_token:\n'
        '                remaining = GENERATION_TIMEOUT - (time.time() - generation_start)\n'
        '                get_timeout = min(FIRST_TOKEN_TIMEOUT, max(0.1, remaining))\n'
        '            else:\n'
        '                get_timeout = FALLBACK_SILENCE\n'
        '\n'
        '            try:\n'
        '                msg_type, msg_data = _token_queue.get(timeout=get_timeout)\n'
        '            except queue.Empty:\n'
        '                logger.warning(f"[{request_id}] Silence timeout")\n'
        '                _active_wrapper.abort()\n'
        '                break',

        '            if not got_first_token:\n'
        '                # Short timeout for periodic activity refresh during prefill\n'
        '                # (mirrors streaming path\'s heartbeat approach)\n'
        '                elapsed_prefill = time.time() - generation_start\n'
        '                if elapsed_prefill > FIRST_TOKEN_TIMEOUT:\n'
        '                    logger.warning(f"[{request_id}] Timeout waiting for first token")\n'
        '                    _active_wrapper.abort()\n'
        '                    break\n'
        '                get_timeout = min(15.0, FIRST_TOKEN_TIMEOUT - elapsed_prefill)\n'
        '                get_timeout = max(0.1, get_timeout)\n'
        '            else:\n'
        '                get_timeout = FALLBACK_SILENCE\n'
        '\n'
        '            try:\n'
        '                msg_type, msg_data = _token_queue.get(timeout=get_timeout)\n'
        '            except queue.Empty:\n'
        '                if not got_first_token:\n'
        '                    # Prefill still running — refresh activity tracker\n'
        '                    update_request_activity()\n'
        '                    continue\n'
        '                logger.warning(f"[{request_id}] Silence timeout")\n'
        '                _active_wrapper.abort()\n'
        '                break',
    ),

]


def apply_fixes(content):
    results = []
    fixed = content
    for label, old, new in FIXES:
        count = fixed.count(old)
        if count == 0:
            results.append((label, "SKIPPED — pattern not found (already applied?)"))
        elif count > 1:
            results.append((label, f"SKIPPED — pattern found {count}x (must be unique)"))
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
        print("\nDRY RUN — no files modified.")
        print("  --apply     Patch in-place (creates .bak backup)")
        print("  -o <path>   Write patched file to new path")


if __name__ == '__main__':
    main()