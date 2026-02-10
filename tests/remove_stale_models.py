#!/usr/bin/env python3
"""Disable (hide) stale model entries in Open WebUI database.

Sets is_active=0 instead of deleting — models can be re-enabled from
Admin > Workspace > Models in the UI without losing their config.
"""
import sqlite3, json, sys

DB_PATH = "/app/backend/data/webui.db"

# Models to disable (hidden from users but config preserved)
STALE_IDS = ["deepseekocr", "internlm2-1.8b"]

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    for model_id in STALE_IDS:
        cur.execute("SELECT id, is_active FROM model WHERE id = ?", (model_id,))
        row = cur.fetchone()
        if row:
            if row[1] == 0:
                print(f"  SKIP     {model_id}: already hidden")
            else:
                cur.execute("UPDATE model SET is_active = 0 WHERE id = ?", (model_id,))
                print(f"  HIDDEN   {model_id}")
        else:
            print(f"  SKIP     {model_id}: not found")
    
    conn.commit()
    
    # Verify
    cur.execute("SELECT id, name, is_active FROM model ORDER BY is_active DESC, id")
    rows = cur.fetchall()
    print(f"\nAll models ({len(rows)}):")
    for row in rows:
        status = "ACTIVE" if row[2] else "HIDDEN"
        print(f"  [{status}] {row[0]:45s} {row[1]}")
    
    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
