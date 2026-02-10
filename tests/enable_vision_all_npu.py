#!/usr/bin/env python3
"""
Enable vision=true on all active NPU models so images can be uploaded
on any chat. The API server auto-routes images to the VL model (qwen3-vl-2b)
regardless of which text model is selected.
"""
import sqlite3, json

DB_PATH = "/app/backend/data/webui.db"

# NPU models that should have vision=true (API auto-routes to VL model)
NPU_MODELS = [
    "qwen3-1.7b",
    "qwen3-4b-instruct-2507",
    "gemma-3-4b-it",
    "phi-3-mini-4k-instruct",
    "qwen3-vl-2b",
]

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    changes = 0
    for model_id in NPU_MODELS:
        cur.execute("SELECT id, meta FROM model WHERE id = ?", (model_id,))
        row = cur.fetchone()
        if not row:
            print(f"  SKIP  {model_id}: not found")
            continue

        meta = json.loads(row["meta"]) if row["meta"] else {}
        caps = meta.get("capabilities", {})

        if caps.get("vision") is True:
            print(f"  OK    {model_id}: vision already true")
            continue

        caps["vision"] = True
        meta["capabilities"] = caps
        cur.execute("UPDATE model SET meta = ? WHERE id = ?", (json.dumps(meta), model_id))
        print(f"  SET   {model_id}: vision -> true")
        changes += 1

    if changes:
        conn.commit()
        print(f"\nCommitted {changes} updates")
    else:
        print("\nNo changes needed")

    # Verify
    print(f"\n{'='*60}")
    print("VERIFICATION:")
    cur.execute("SELECT id, meta, is_active FROM model ORDER BY is_active DESC, id")
    for row in cur.fetchall():
        meta = json.loads(row["meta"]) if row["meta"] else {}
        vision = meta.get("capabilities", {}).get("vision", False)
        active = "ACTIVE" if row["is_active"] else "HIDDEN"
        vl = "VL" if vision else "  "
        print(f"  [{active:6s}] [{vl}] {row['id']}")

    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
