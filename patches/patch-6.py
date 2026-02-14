#!/usr/bin/env python3
"""
RKLLM API Patcher — Code Polish (Observations)
================================================
Usage:
    python3 patch-6-polish.py api.py                  # Dry run
    python3 patch-6-polish.py api.py --apply          # Patch in-place
    python3 patch-6-polish.py api.py -o api_fixed.py  # Write to new file

Fixes:
    1. Move _DATE_SIGNAL regex from inside _strip_stale_date_claims() to
       module scope — avoids recompiling on every call.  Placed next to
       the other module-level regex (_RE_THINK_BLOCK).

    2. Add request_ended flag to non-streaming path (_generate_complete)
       to match the streaming path's pattern.  Calls end_request() right
       after the generation loop exits (freeing the request slot sooner)
       and guards the finally block to avoid a redundant idempotent call.
       Mirrors the streaming path exactly.

Not changed (already clean — no action needed):
    3. GENERATION_COMPLETE lifecycle — all .clear()/.set() paired in
       try/finally.  Verified correct.
    4. _worker_thread writes — all under PROCESS_LOCK.  Verified correct.
    5. Unrestricted CORS — intentional for LAN deployment of NPU server.
"""

import sys
import os

FIXES = [

    # === FIX 1 — Move _DATE_SIGNAL to module scope ==========================
    # Part A: Insert _DATE_SIGNAL between _RE_THINK_BLOCK and the Open WebUI
    #         comment.  Anchor includes both sides so it can't match twice.
    # Part B: Remove the inline definition from the function body.
    # =====================================================================

    (
        'FIX 1a: Add _DATE_SIGNAL at module scope (after _RE_THINK_BLOCK)',

        # Anchor: _RE_THINK_BLOCK line through to # Open WebUI comment
        # After patching, _DATE_SIGNAL sits between them so this won't rematch.
        "_RE_THINK_BLOCK = re.compile(r'<think>.*?</think>\\s*', re.DOTALL)\n"
        "\n"
        "# Open WebUI internal task signatures",

        "_RE_THINK_BLOCK = re.compile(r'<think>.*?</think>\\s*', re.DOTALL)\n"
        "\n"
        "# Date-signal pattern: a regex match from _STALE_PATTERNS must contain\n"
        "# at least one of these tokens to be considered an actual date/time claim.\n"
        "# Prevents false-positives on sentences like \"Today is the last day to register\".\n"
        "_DATE_SIGNAL = re.compile(\n"
        "    r'(?:'\n"
        "    r'\\d{4}'\n"
        "    r'|\\d{1,2}[/\\-]\\d{1,2}'\n"
        "    r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*'\n"
        "    r'|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)'\n"
        "    r'|\\d{1,2}\\s*:\\s*\\d{2}'\n"
        "    r'|\\d{1,2}\\s*(?:am|pm)'\n"
        "    r')', re.IGNORECASE)\n"
        "\n"
        "# Open WebUI internal task signatures",
    ),

    (
        'FIX 1b: Remove inline _DATE_SIGNAL definition from function body',

        '    # Date-signal pattern: a match must contain at least one of these to\n'
        '    # be considered an actual date/time claim (prevents false-positives on\n'
        '    # sentences like "Today is the last day to register").\n'
        '    _DATE_SIGNAL = re.compile(\n'
        "        r'(?:'\n"
        "        r'\\d{4}'\n"
        "        r'|\\d{1,2}[/\\-]\\d{1,2}'\n"
        "        r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*'\n"
        "        r'|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)'\n"
        "        r'|\\d{1,2}\\s*:\\s*\\d{2}'\n"
        "        r'|\\d{1,2}\\s*(?:am|pm)'\n"
        "        r')', re.IGNORECASE)\n"
        '    result = text',

        '    result = text',
    ),

    # === FIX 2 — Non-streaming request_ended flag ===========================
    # Part A: Add flag to init vars
    # Part B: End request early after generation loop, anchored to
    #         preceding "break\n\n" so it can't match after patching
    # Part C: Guard finally block
    # =====================================================================

    (
        'FIX 2a: Add request_ended flag to non-streaming init vars',

        '    generation_start = time.time()\n'
        '    got_first_token = False\n'
        '    content_parts = []\n'
        '    combined_output = ""\n'
        '    generation_clean = False\n'
        '    stats_data = {}',

        '    generation_start = time.time()\n'
        '    got_first_token = False\n'
        '    content_parts = []\n'
        '    combined_output = ""\n'
        '    generation_clean = False\n'
        '    request_ended = False\n'
        '    stats_data = {}',
    ),

    (
        'FIX 2b: End request early before bookkeeping (non-streaming)',

        # Anchor includes "break\n\n        # Wait" which won't exist after
        # patching (end_request block sits between break and # Wait).
        '                break\n'
        '\n'
        '        # Wait for worker thread\n'
        '        with PROCESS_LOCK:\n'
        '            if _worker_thread and _worker_thread.is_alive():\n'
        '                _worker_thread.join(timeout=5)\n'
        '            _worker_thread = None\n'
        '\n'
        '        full_content = "".join(content_parts).rstrip()',

        '                break\n'
        '\n'
        '        # End request tracking immediately (frees slot for next request)\n'
        '        end_request(request_id)\n'
        '        request_ended = True\n'
        '\n'
        '        # Wait for worker thread\n'
        '        with PROCESS_LOCK:\n'
        '            if _worker_thread and _worker_thread.is_alive():\n'
        '                _worker_thread.join(timeout=5)\n'
        '            _worker_thread = None\n'
        '\n'
        '        full_content = "".join(content_parts).rstrip()',
    ),

    (
        'FIX 2c: Guard finally end_request with request_ended flag',

        '    finally:\n'
        '        end_request(request_id)\n'
        '        # Ensure worker cleanup',

        '    finally:\n'
        '        if not request_ended:\n'
        '            end_request(request_id)\n'
        '        # Ensure worker cleanup',
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