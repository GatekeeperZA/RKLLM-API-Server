#!/usr/bin/env python3
"""
RKLLM API Patcher \u2014 Repetition Detection Fixes
=====================================================
Usage:
    python3 fix_api_repetition.py api.py                  # Dry run
    python3 fix_api_repetition.py api.py --apply          # Patch in-place
    python3 fix_api_repetition.py api.py -o api_fixed.py  # Write to new file

Fixes:
    1. [BUG-HIGH] Streaming path: NameError on `combined_output` \u2014 variable
       doesn't exist in _generate_stream.  When a repetition loop triggers,
       Python crashes with NameError BEFORE reaching abort(), causing a
       delayed unclean abort with misleading finish_reason="stop".
       Fix: change to `total_content` (the correct variable).

    2. [ADVISORY] Non-streaming path: repetition detection checks
       combined_output which includes <think> blocks.  Thinking models
       exploring ideas from multiple angles could trigger a false abort.
       Fix: strip completed <think> blocks before checking, matching
       the streaming path's behaviour (which only checks total_content).
"""

import sys
import os

FIXES = [

    # === FIX 1 \u2014 BUG (HIGH): NameError in streaming repetition detection ======
    # _generate_stream has: total_content, total_reasoning
    # It does NOT have: combined_output
    # The logger.warning references combined_output \u2192 NameError at runtime
    # when a repetition loop fires \u2192 abort() never called \u2192 falls through
    # to generic exception handler with wrong finish_reason.
    # =====================================================================
    (
        'FIX 1 [BUG-HIGH]: NameError combined_output \u2192 total_content (streaming)',

        '                if len(total_content) > REPETITION_WINDOW * 2:\n'
        '                    loop_hits = _detect_repetition_loop(total_content)\n'
        '                    if loop_hits >= REPETITION_MAX_HITS:\n'
        '                        logger.warning(f"[{request_id}] Repetition loop detected "\n'
        '                                       f"({loop_hits} repeats of {REPETITION_WINDOW}-char window, "\n'
        '                                       f"{len(combined_output)} chars total) \u2014 aborting")\n'
        '                        _active_wrapper.abort()\n'
        '                        break',

        '                if len(total_content) > REPETITION_WINDOW * 2:\n'
        '                    loop_hits = _detect_repetition_loop(total_content)\n'
        '                    if loop_hits >= REPETITION_MAX_HITS:\n'
        '                        logger.warning(f"[{request_id}] Repetition loop detected "\n'
        '                                       f"({loop_hits} repeats of {REPETITION_WINDOW}-char window, "\n'
        '                                       f"{len(total_content)} chars total) \u2014 aborting")\n'
        '                        _active_wrapper.abort()\n'
        '                        break',
    ),

    # === FIX 2 \u2014 ADVISORY: Non-streaming repetition on thinking content =======
    # combined_output accumulates ALL tokens including <think> blocks.
    # Thinking models repeat phrases during exploration, which is normal
    # reasoning behaviour \u2014 not a generation loop.
    # Fix: strip completed <think>...</think> blocks before checking.
    # Uses _RE_THINK_BLOCK (already defined globally).
    # =====================================================================
    (
        'FIX 2 [ADVISORY]: Non-streaming repetition \u2014 exclude thinking content',

        '                content_parts.append(msg_data)\n'
        '                combined_output += msg_data\n'
        '\n'
        '                # Repetition loop detection (non-streaming path)\n'
        '                if len(combined_output) > REPETITION_WINDOW * 2:\n'
        '                    loop_hits = _detect_repetition_loop(combined_output)\n'
        '                    if loop_hits >= REPETITION_MAX_HITS:\n'
        '                        logger.warning(f"[{request_id}] Repetition loop detected "\n'
        '                                       f"({loop_hits} repeats of {REPETITION_WINDOW}-char window, "\n'
        '                                       f"{len(combined_output)} chars total) \u2014 aborting")\n'
        '                        _active_wrapper.abort()\n'
        '                        break',

        '                content_parts.append(msg_data)\n'
        '                combined_output += msg_data\n'
        '\n'
        '                # Repetition loop detection (non-streaming path)\n'
        '                # Strip completed <think> blocks \u2014 thinking models often\n'
        '                # repeat phrasing during exploration, which is normal.\n'
        '                # Only check user-visible content for loops.\n'
        '                _visible_output = _RE_THINK_BLOCK.sub(\'\', combined_output)\n'
        '                if len(_visible_output) > REPETITION_WINDOW * 2:\n'
        '                    loop_hits = _detect_repetition_loop(_visible_output)\n'
        '                    if loop_hits >= REPETITION_MAX_HITS:\n'
        '                        logger.warning(f"[{request_id}] Repetition loop detected "\n'
        '                                       f"({loop_hits} repeats of {REPETITION_WINDOW}-char window, "\n'
        '                                       f"{len(_visible_output)} chars total) \u2014 aborting")\n'
        '                        _active_wrapper.abort()\n'
        '                        break',
    ),

]


def apply_fixes(content):
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