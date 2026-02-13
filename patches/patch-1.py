#!/usr/bin/env python3
"""
RKLLM API Patcher — Applies all audit fixes (Feb 2026 delta audit)
===================================================================
Usage:
    python3 fix_api.py api.py                  # Preview changes (dry run)
    python3 fix_api.py api.py --apply          # Apply changes in-place
    python3 fix_api.py api.py -o api_fixed.py  # Write to new file

Fixes applied:
    1. [BUG]      Streaming finish_reason hardcoded to "stop"
    2. [BUG]      Sliding window trims system prompt first
    3. [ADVISORY] KV tracking polluted by VL requests (streaming + non-streaming)
    4. [ADVISORY] Unused import `timezone`
    5. [ADVISORY] Repetition detection fires on thinking content
    6. [ADVISORY] Non-streaming repetition rejoins list every token
    7. [ADVISORY] Unused import `struct as pystruct` (carried from prior audit)
"""

import sys
import os

# Each fix: (label, old_string, new_string)
# old_string must appear EXACTLY ONCE in the source file.
FIXES = [

    # === FIX 1 — BUG: Streaming finish_reason hardcoded to "stop" ========
    (
        'FIX 1 [BUG]: Streaming finish_reason hardcoded "stop"',

        '        # Final stop chunk\n'
        '        yield make_sse_chunk(request_id, model_name, created, finish_reason="stop")',

        '        # Final chunk — "stop" for clean generation, "length" for timeout/abort\n'
        '        _finish_reason = "stop" if generation_clean else "length"\n'
        '        yield make_sse_chunk(request_id, model_name, created, finish_reason=_finish_reason)',
    ),

    # === FIX 2 — BUG: Sliding window trims system prompt first ============
    (
        'FIX 2 [BUG]: Sliding window system prompt protection',

        '            _keep_front = 1 if (parts and parts[0].startswith("System:")) else 0',

        '            _keep_front = 1 if (parts and not parts[0].startswith(("User:", "Assistant:"))) else 0',
    ),

    # === FIX 3a — ADVISORY: KV tracking VL guard (streaming) ==============
    (
        'FIX 3a [ADVISORY]: KV tracking VL guard (streaming)',

        '            if generation_clean and not is_rag and messages:\n'
        '                _update_kv_tracking(model_name, messages, is_reset=kv_is_reset)\n'
        '        except Exception as _kv_exc:\n'
        '            logger.error(f"[{request_id}] KV tracking update error: {_kv_exc}")\n'
        '\n'
        '    except GeneratorExit:',

        '            if generation_clean and not is_rag and not _is_vl and messages:\n'
        '                _update_kv_tracking(model_name, messages, is_reset=kv_is_reset)\n'
        '        except Exception as _kv_exc:\n'
        '            logger.error(f"[{request_id}] KV tracking update error: {_kv_exc}")\n'
        '\n'
        '    except GeneratorExit:',
    ),

    # === FIX 3b — ADVISORY: KV tracking VL guard (non-streaming) ==========
    (
        'FIX 3b [ADVISORY]: KV tracking VL guard (non-streaming)',

        '            if generation_clean and not is_rag and messages:\n'
        '                _update_kv_tracking(model_name, messages, is_reset=kv_is_reset)\n'
        '        except Exception as _kv_exc:\n'
        '            logger.error(f"[{request_id}] KV tracking update error: {_kv_exc}")\n'
        '\n'
        '        if reasoning_content:',

        '            if generation_clean and not is_rag and not _is_vl and messages:\n'
        '                _update_kv_tracking(model_name, messages, is_reset=kv_is_reset)\n'
        '        except Exception as _kv_exc:\n'
        '            logger.error(f"[{request_id}] KV tracking update error: {_kv_exc}")\n'
        '\n'
        '        if reasoning_content:',
    ),

    # === FIX 4 — ADVISORY: Unused import `timezone` ======================
    (
        'FIX 4 [ADVISORY]: Remove unused import timezone',

        'from datetime import datetime, timezone',

        'from datetime import datetime',
    ),

    # === FIX 5 — ADVISORY: Repetition detection on thinking content =======
    (
        'FIX 5 [ADVISORY]: Repetition detection — visible content only (streaming)',

        '                # Repetition loop detection \u2014 runs every token but the\n'
        '                # _detect_repetition_loop() call is O(n) worst-case: it only\n'
        '                # scans the last REPETITION_WINDOW chars against earlier text.\n'
        '                combined_output = total_reasoning + total_content\n'
        '                if len(combined_output) > REPETITION_WINDOW * 2:\n'
        '                    loop_hits = _detect_repetition_loop(combined_output)',

        '                # Repetition loop detection \u2014 runs every token but the\n'
        '                # _detect_repetition_loop() call is O(n) worst-case: it only\n'
        '                # scans the last REPETITION_WINDOW chars against earlier text.\n'
        '                # Only check user-visible content \u2014 thinking models often\n'
        '                # repeat phrasing inside <think> blocks during exploration.\n'
        '                if len(total_content) > REPETITION_WINDOW * 2:\n'
        '                    loop_hits = _detect_repetition_loop(total_content)',
    ),

    # === FIX 6a — ADVISORY: Non-streaming string accumulator init =========
    (
        'FIX 6a [ADVISORY]: Non-streaming repetition — add string accumulator',

        '    content_parts = []\n'
        '    generation_clean = False',

        '    content_parts = []\n'
        '    combined_output = ""\n'
        '    generation_clean = False',
    ),

    # === FIX 6b — ADVISORY: Non-streaming accumulator usage ===============
    (
        'FIX 6b [ADVISORY]: Non-streaming repetition — use accumulator',

        '            if msg_type == "token":\n'
        '                if not got_first_token:\n'
        '                    got_first_token = True\n'
        '                update_request_activity()\n'
        '                content_parts.append(msg_data)\n'
        '\n'
        '                # Repetition loop detection (non-streaming path)\n'
        '                # Build running output \u2014 O(n) join, but non-streaming\n'
        '                # batches are capped by max_new_tokens so this is bounded.\n'
        '                combined_output = "".join(content_parts)\n'
        '                if len(combined_output) > REPETITION_WINDOW * 2:\n'
        '                    loop_hits = _detect_repetition_loop(combined_output)',

        '            if msg_type == "token":\n'
        '                if not got_first_token:\n'
        '                    got_first_token = True\n'
        '                update_request_activity()\n'
        '                content_parts.append(msg_data)\n'
        '                combined_output += msg_data\n'
        '\n'
        '                # Repetition loop detection (non-streaming path)\n'
        '                if len(combined_output) > REPETITION_WINDOW * 2:\n'
        '                    loop_hits = _detect_repetition_loop(combined_output)',
    ),

    # === FIX 7 — ADVISORY: Unused import struct as pystruct ===============
    (
        'FIX 7 [ADVISORY]: Remove unused import struct as pystruct',

        'import struct as pystruct\n'
        'from collections import OrderedDict',

        'from collections import OrderedDict',
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
        with open(backup, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\nBackup: {backup}")
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write(fixed)
        print(f"Patched: {input_path}")
    else:
        print("\nDRY RUN \u2014 no files modified.")
        print("  --apply     Patch in-place (creates .bak backup)")
        print("  -o <path>   Write patched file to new path")


if __name__ == '__main__':
    main()