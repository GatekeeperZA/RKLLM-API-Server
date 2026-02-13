#!/usr/bin/env python3
"""
RKLLM API Patcher — Non-streaming activity refresh
====================================================
Usage:
    python3 fix_api_nonsream_activity.py api.py                  # Dry run
    python3 fix_api_nonsream_activity.py api.py --apply          # Patch in-place
    python3 fix_api_nonsream_activity.py api.py -o api_fixed.py  # Write to new file

Fixes:
    1. Add update_request_activity() before GENERATION_COMPLETE.clear()
       in _generate_complete (non-streaming path), matching the streaming
       path fix already applied.

       Without this, a non-streaming request with 80-180s prefill could
       be declared stale, allowing a second request to claim the slot
       and conflict with the in-flight generation.
"""

import sys
import os

FIXES = [

    (
        'FIX 1: Non-streaming activity refresh before GENERATION_COMPLETE.clear()',

        '    try:\n'
        '        GENERATION_COMPLETE.clear()\n'
        '        while True:\n'
        '            if ABORT_EVENT.is_set():\n'
        '                _active_wrapper.abort()\n'
        '                break',

        '    try:\n'
        '        update_request_activity()  # Refresh after model load delay\n'
        '        GENERATION_COMPLETE.clear()\n'
        '        while True:\n'
        '            if ABORT_EVENT.is_set():\n'
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