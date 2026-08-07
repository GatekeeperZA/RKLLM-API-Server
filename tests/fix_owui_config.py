#!/usr/bin/env python3
"""
fix_owui_config.py — Sanitize OpenWebUI config DB entries.

The OpenWebUI config table stores all values as JSON-encoded strings.
If a value is an empty string '' (not valid JSON), Config.get() will crash
with: JSONDecodeError: Expecting value: line 1 column 1 (char 0)

Known cause: direct SQL writes that omit the JSON encoding step.

Usage (run inside the Docker container):
    docker exec open-webui python3 /path/to/fix_owui_config.py

Or copy in first:
    docker cp tests/fix_owui_config.py open-webui:/tmp/
    docker exec open-webui python3 /tmp/fix_owui_config.py
"""

import sqlite3
import json
import time
import sys

DB_PATH = "/app/backend/data/webui.db"

# Keys that should be JSON null when unset (not empty string)
NULL_WHEN_EMPTY = {
    "task.model.external",
    "chat.context_compaction.token_cap",
    "rag.content_extraction.supported_media_map",
}


def audit_and_fix(db_path: str, dry_run: bool = False) -> list[tuple]:
    db = sqlite3.connect(db_path)
    rows = db.execute("SELECT key, value FROM config ORDER BY key").fetchall()

    issues = []
    fixes = []

    for key, value in rows:
        # Native SQLite int/float — fine, SQLAlchemy handles without json.loads
        if not isinstance(value, str):
            continue

        # Empty string — BROKEN: json.loads('') raises JSONDecodeError
        if value == "":
            replacement = "null" if key in NULL_WHEN_EMPTY else None
            issues.append((key, value, replacement))
            if replacement is not None:
                fixes.append((key, replacement))
                print(f"  FIX : {key!r}  '' -> 'null'")
            else:
                print(f"  WARN: {key!r} = '' (empty string) — manual review needed")
            continue

        # Valid JSON string — verify it parses
        try:
            json.loads(value)
        except json.JSONDecodeError as e:
            issues.append((key, value, None))
            print(f"  BAD : {key!r} = {value!r:.60} — {e}")

    if not issues:
        print("  All config entries are valid JSON.")
        return []

    if fixes and not dry_run:
        ts = int(time.time())
        for key, replacement in fixes:
            db.execute(
                "UPDATE config SET value=?, updated_at=? WHERE key=?",
                (replacement, ts, key),
            )
        db.commit()
        print(f"\nApplied {len(fixes)} fix(es).")
    elif fixes and dry_run:
        print(f"\nDry run — {len(fixes)} fix(es) would be applied.")

    db.close()
    return issues


def main():
    dry_run = "--dry-run" in sys.argv
    print(f"Auditing {DB_PATH} {'(dry run)' if dry_run else ''}...")
    print()
    issues = audit_and_fix(DB_PATH, dry_run=dry_run)
    print()
    if issues:
        unfixed = [i for i in issues if i[2] is None]
        if unfixed:
            print(f"WARNING: {len(unfixed)} issue(s) need manual review.")
            sys.exit(1)
    else:
        print("Config DB is healthy.")


if __name__ == "__main__":
    main()
