#!/usr/bin/env python3
"""
Restore disabled model entries and fix qwen3-vl-2b capabilities.

1. Re-inserts deepseekocr and internlm2-1.8b as is_active=0 (hidden but recoverable)
2. Fixes qwen3-vl-2b capabilities (only vision=true, everything else off)
"""
import sqlite3, json, time

DB_PATH = "/app/backend/data/webui.db"

# System prompt used on all models
SYSTEM_PROMPT = "Today is {{CURRENT_DATE}} ({{CURRENT_WEEKDAY}}), {{CURRENT_TIME}}. This is the ONLY correct current date. Ignore any conflicting dates from search results."

# Admin user id (from existing entries)
ADMIN_USER_ID = "962f2c78-8a36-4102-8d1d-2dca7b35e0c7"

# Models to restore as disabled
RESTORE_MODELS = [
    {
        "id": "deepseekocr",
        "name": "Deepseek Vision & OCR | (DeepSeek OCR) [DISABLED]",
        "vision": True,
    },
    {
        "id": "internlm2-1.8b",
        "name": "InternLM2-1.8b | (Fast General Chat) [DISABLED]",
        "vision": False,
    },
]

def make_meta(vision=False):
    return json.dumps({
        "profile_image_url": "/static/favicon.png",
        "description": None,
        "capabilities": {
            "file_context": True,
            "vision": vision,
            "file_upload": True,
            "web_search": True,
            "image_generation": False,
            "code_interpreter": False,
            "citations": False,
            "status_updates": True,
            "builtin_tools": False,
        },
        "suggestion_prompts": None,
        "tags": [],
    })

def make_params():
    return json.dumps({"system": SYSTEM_PROMPT})

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    now = int(time.time())
    changes = 0

    # --- 1. Restore disabled models ---
    print("=== Restoring disabled models (is_active=0) ===")
    for m in RESTORE_MODELS:
        cur.execute("SELECT id, is_active FROM model WHERE id = ?", (m["id"],))
        existing = cur.fetchone()
        if existing:
            print(f"  EXISTS  {m['id']} (is_active={existing['is_active']})")
            if existing["is_active"] != 0:
                cur.execute("UPDATE model SET is_active = 0 WHERE id = ?", (m["id"],))
                print(f"          -> set is_active=0")
                changes += 1
        else:
            cur.execute(
                "INSERT INTO model (id, user_id, base_model_id, name, meta, params, created_at, updated_at, access_control, is_active) "
                "VALUES (?, ?, NULL, ?, ?, ?, ?, ?, '{}', 0)",
                (m["id"], ADMIN_USER_ID, m["name"], make_meta(m["vision"]), make_params(), now, now),
            )
            print(f"  INSERT  {m['id']} (is_active=0, vision={m['vision']})")
            changes += 1

    # --- 2. Fix qwen3-vl-2b capabilities ---
    print("\n=== Fixing qwen3-vl-2b capabilities ===")
    cur.execute("SELECT id, meta FROM model WHERE id = 'qwen3-vl-2b'")
    row = cur.fetchone()
    if row:
        meta = json.loads(row["meta"])
        caps = meta.get("capabilities", {})
        fixes = []
        correct = {
            "vision": True,
            "image_generation": False,
            "code_interpreter": False,
            "citations": False,
            "builtin_tools": False,
        }
        for k, v in correct.items():
            if caps.get(k) != v:
                fixes.append(f"  {k}: {caps.get(k)} -> {v}")
                caps[k] = v
        if fixes:
            meta["capabilities"] = caps
            cur.execute("UPDATE model SET meta = ? WHERE id = 'qwen3-vl-2b'", (json.dumps(meta),))
            print(f"  FIX  qwen3-vl-2b:")
            for f in fixes:
                print(f"    {f}")
            changes += 1
        else:
            print("  OK   qwen3-vl-2b: capabilities already correct")
    else:
        print("  SKIP  qwen3-vl-2b not found")

    if changes:
        conn.commit()
        print(f"\nCommitted {changes} changes")
    else:
        print("\nNo changes needed")

    # --- Verify ---
    print(f"\n{'='*60}")
    print("VERIFICATION:")
    cur.execute("SELECT id, name, meta, is_active FROM model ORDER BY is_active DESC, id")
    for row in cur.fetchall():
        meta = json.loads(row["meta"]) if row["meta"] else {}
        caps = meta.get("capabilities", {})
        active = "ACTIVE" if row["is_active"] else "HIDDEN"
        vision = "VL" if caps.get("vision") else "  "
        builtin = "TOOLS" if caps.get("builtin_tools") else "     "
        img_gen = "IMG" if caps.get("image_generation") else "   "
        print(f"  [{active:6s}] [{vision}] [{builtin}] [{img_gen}] {row['id']:45s} {row['name']}")

    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
