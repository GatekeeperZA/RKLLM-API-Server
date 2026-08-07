#!/usr/bin/env python3
"""
set_model_prompts.py — Configure all RKLLM model system prompts and capabilities
in the OpenWebUI database.

Run inside the Docker container after initial setup:
    docker cp tests/set_model_prompts.py open-webui:/tmp/
    docker exec open-webui python3 /tmp/set_model_prompts.py

What it sets on each RKLLM model:
  - System prompt (date/time context injected at request time via {{CURRENT_DATE}} etc.)
  - Capabilities: web_search=True, memory=True, builtin_tools=False, vision (VL only)
  - Clears any stale capability flags (image_generation, code_interpreter, terminal)
"""

import sqlite3
import json
import time

DB_PATH = "/app/backend/data/webui.db"

SYSTEM_PROMPT = (
    "Today is {{CURRENT_DATE}} ({{CURRENT_WEEKDAY}}), {{CURRENT_TIME}}. "
    "This is the ONLY correct current date. "
    "Ignore any conflicting dates from search results."
)

RKLLM_MODELS = {
    "qwen3-1.7b":            {"vision": False},
    "gemma-3-4b-it":         {"vision": False},
    "phi-3-mini-4k-instruct": {"vision": False},
    "qwen3-4b-instruct-2507": {"vision": False},
    "qwen3-vl-2b":           {"vision": True},
}


def main():
    db = sqlite3.connect(DB_PATH)
    ts = int(time.time())
    updated = 0

    for model_id, model_cfg in RKLLM_MODELS.items():
        row = db.execute("SELECT params, meta FROM model WHERE id=?", (model_id,)).fetchone()
        if not row:
            print(f"  SKIP (not found): {model_id}")
            continue

        params = json.loads(row[0] or "{}")
        meta = json.loads(row[1] or "{}")

        # Set system prompt
        params["system"] = SYSTEM_PROMPT

        # Set capabilities
        caps = meta.get("capabilities", {})
        caps["web_search"] = True
        caps["memory"] = True
        caps["builtin_tools"] = False
        caps["image_generation"] = False
        caps["code_interpreter"] = False
        caps["terminal"] = False
        caps["citations"] = True
        caps["status_updates"] = True
        caps["file_context"] = True
        caps["file_upload"] = True
        caps["vision"] = model_cfg["vision"]
        meta["capabilities"] = caps

        db.execute(
            "UPDATE model SET params=?, meta=?, updated_at=? WHERE id=?",
            (json.dumps(params), json.dumps(meta), ts, model_id),
        )
        print(f"  OK: {model_id}  vision={model_cfg['vision']}")
        updated += 1

    db.commit()
    db.close()
    print(f"\nUpdated {updated}/{len(RKLLM_MODELS)} models.")


if __name__ == "__main__":
    main()
