#!/usr/bin/env python3
"""Quick dump of all Open WebUI model entries."""
import sqlite3, json

DB_PATH = "/app/backend/data/webui.db"
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute("SELECT id, name, meta, params FROM model")
for row in cur.fetchall():
    meta = json.loads(row["meta"]) if row["meta"] else {}
    caps = meta.get("capabilities", {})
    params = json.loads(row["params"]) if row["params"] else {}
    prompt = params.get("system", "NOT SET")[:60]
    vision = caps.get("vision", "?")
    print(f"  {row['id']:45s} name={row['name']!r:40s} vision={str(vision):5s} prompt={prompt!r}")
conn.close()
