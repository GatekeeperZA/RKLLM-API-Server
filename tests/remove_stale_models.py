#!/usr/bin/env python3
"""Remove stale/disabled model entries from Open WebUI database."""
import sqlite3, json, sys

DB_PATH = "/app/backend/data/webui.db"

# Models to remove (disabled on filesystem)
STALE_IDS = ["deepseekocr", "internlm2-1.8b"]

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    for model_id in STALE_IDS:
        cur.execute("SELECT id FROM model WHERE id = ?", (model_id,))
        if cur.fetchone():
            cur.execute("DELETE FROM model WHERE id = ?", (model_id,))
            print(f"  REMOVED  {model_id}")
        else:
            print(f"  SKIP     {model_id}: not found")
    
    conn.commit()
    
    # Verify
    cur.execute("SELECT id, name FROM model ORDER BY id")
    rows = cur.fetchall()
    print(f"\nRemaining models ({len(rows)}):")
    for row in rows:
        print(f"  {row[0]:45s} {row[1]}")
    
    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
