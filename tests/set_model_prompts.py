#!/usr/bin/env python3
"""
Set model-level system prompt on all Open WebUI models.

This sets the system prompt in each model's params.system field,
which is server-enforced and applies to ALL users automatically.
No user configuration needed — the prompt is injected on every request.

The prompt provides date/time context using Open WebUI template variables.
These are resolved server-side before being sent to the model.

Usage:
    docker exec open-webui python3 /tmp/set_model_prompts.py

To change the prompt later, edit SYSTEM_PROMPT below and re-run.
"""

import sqlite3
import json
import sys

DB_PATH = "/app/backend/data/webui.db"

# The system prompt applied to ALL models.
# Template variables are resolved by Open WebUI before sending to the model:
#   {{CURRENT_DATE}}     → e.g. "February 10, 2026"
#   {{CURRENT_WEEKDAY}}  → e.g. "Tuesday"
#   {{CURRENT_TIME}}     → e.g. "14:30:00"
#   {{USER_NAME}}        → e.g. "Juan-Pierre"
SYSTEM_PROMPT = "Today is {{CURRENT_DATE}} ({{CURRENT_WEEKDAY}}), {{CURRENT_TIME}}. Trust all dates as correct."


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get all models
    cur.execute("SELECT id, name, params FROM model")
    rows = cur.fetchall()

    print(f"Found {len(rows)} models")
    print(f"System prompt: {SYSTEM_PROMPT!r}\n")

    changes = 0

    for row in rows:
        model_id = row["id"]
        model_name = row["name"] or model_id
        params_raw = row["params"]

        # Parse existing params (may be NULL, empty string, or JSON)
        if params_raw:
            try:
                params = json.loads(params_raw)
            except (json.JSONDecodeError, TypeError):
                params = {}
        else:
            params = {}

        current_prompt = params.get("system", None)

        if current_prompt == SYSTEM_PROMPT:
            print(f"  OK   {model_id}: already set")
            continue

        # Set the system prompt
        params["system"] = SYSTEM_PROMPT
        new_params = json.dumps(params)

        cur.execute("UPDATE model SET params = ? WHERE id = ?", (new_params, model_id))

        if current_prompt:
            print(f"  FIX  {model_id}: updated (was: {current_prompt!r:.60})")
        else:
            print(f"  SET  {model_id}: added system prompt")

        changes += 1

    if changes:
        conn.commit()
        print(f"\nCommitted {changes} updates")
    else:
        print("\nNo changes needed")

    # Also clear user-level system prompts to avoid duplication
    # (the model-level prompt is now the single source of truth)
    print("\n--- Checking user-level system prompts ---")
    cur.execute("SELECT id, name, settings FROM user")
    users = cur.fetchall()

    user_changes = 0
    for user in users:
        user_id = user["id"]
        user_name = user["name"]
        settings_raw = user["settings"]

        if not settings_raw:
            print(f"  OK   {user_name}: no settings")
            continue

        try:
            settings = json.loads(settings_raw)
        except (json.JSONDecodeError, TypeError):
            print(f"  OK   {user_name}: no parseable settings")
            continue

        user_prompt = settings.get("system", None)
        if user_prompt:
            print(f"  CLEAR {user_name}: removing user-level prompt (was: {user_prompt!r:.80})")
            del settings["system"]
            cur.execute("UPDATE user SET settings = ? WHERE id = ?",
                       (json.dumps(settings), user_id))
            user_changes += 1
        else:
            print(f"  OK   {user_name}: no user-level prompt")

    if user_changes:
        conn.commit()
        print(f"\nCleared {user_changes} user-level prompts (model-level is now the source of truth)")

    # Verification
    print(f"\n{'='*60}")
    print("VERIFICATION:\n")
    cur.execute("SELECT id, params FROM model")
    for row in cur.fetchall():
        params = json.loads(row["params"]) if row["params"] else {}
        prompt = params.get("system", "NOT SET")
        status = "OK" if prompt == SYSTEM_PROMPT else "MISSING"
        print(f"  [{status}] {row['id']}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
