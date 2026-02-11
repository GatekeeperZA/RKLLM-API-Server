#!/usr/bin/env python3
"""
Change Open WebUI image compression settings at runtime via:
1. Direct SQLite database update (persists across restarts)
2. Admin API call (applies immediately without restart)

Run inside the open-webui container:
  docker exec open-webui python3 /tmp/owui_set_compression.py [width] [height]
"""
import sqlite3, json, sys, os

DB = "/app/backend/data/webui.db"

# Allow overriding from command line
NEW_WIDTH = int(sys.argv[1]) if len(sys.argv) > 1 else 672
NEW_HEIGHT = int(sys.argv[2]) if len(sys.argv) > 2 else 672

print(f"Setting image compression to {NEW_WIDTH}x{NEW_HEIGHT}")

# ── Step 1: Update the SQLite database directly ──
conn = sqlite3.connect(DB)
c = conn.cursor()
c.execute("SELECT data FROM config WHERE id=1")
row = c.fetchone()
if not row:
    print("ERROR: No config row found")
    sys.exit(1)

data = json.loads(row[0])

# PersistentConfig paths: file.image_compression_width / file.image_compression_height
if "file" not in data:
    data["file"] = {}
old_w = data["file"].get("image_compression_width", "not set (using env var)")
old_h = data["file"].get("image_compression_height", "not set (using env var)")
print(f"DB before: width={old_w}, height={old_h}")

data["file"]["image_compression_width"] = NEW_WIDTH
data["file"]["image_compression_height"] = NEW_HEIGHT

c.execute("UPDATE config SET data=?, updated_at=datetime('now') WHERE id=1", (json.dumps(data),))
conn.commit()
conn.close()
print(f"DB after:  width={NEW_WIDTH}, height={NEW_HEIGHT}")
print("Database updated successfully.")

# ── Step 2: Update the running app via the admin API ──
try:
    import requests
    BASE = "http://localhost:8080"  # Inside the container, OWUI listens on 8080
    EMAIL = os.environ.get("OWUI_EMAIL")
    PASSWORD = os.environ.get("OWUI_PASSWORD")
    if not EMAIL or not PASSWORD:
        print("ERROR: Set OWUI_EMAIL and OWUI_PASSWORD env vars before running.")
        print("  export OWUI_EMAIL='your-email'")
        print("  export OWUI_PASSWORD='your-password'")
        sys.exit(1)

    # Sign in
    r = requests.post(f"{BASE}/api/v1/auths/signin",
                       json={"email": EMAIL, "password": PASSWORD}, timeout=5)
    r.raise_for_status()
    token = r.json()["token"]
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Get current config to verify
    r = requests.get(f"{BASE}/api/v1/retrieval/config", headers=headers, timeout=5)
    if r.status_code == 200:
        cfg = r.json()
        print(f"\nRuntime before: width={cfg.get('FILE_IMAGE_COMPRESSION_WIDTH')}, "
              f"height={cfg.get('FILE_IMAGE_COMPRESSION_HEIGHT')}")

    # Update via the retrieval config endpoint
    r = requests.post(f"{BASE}/api/v1/retrieval/config/update", headers=headers, timeout=10,
                       json={"FILE_IMAGE_COMPRESSION_WIDTH": NEW_WIDTH,
                             "FILE_IMAGE_COMPRESSION_HEIGHT": NEW_HEIGHT})
    if r.status_code == 200:
        result = r.json()
        print(f"Runtime after:  width={result.get('FILE_IMAGE_COMPRESSION_WIDTH')}, "
              f"height={result.get('FILE_IMAGE_COMPRESSION_HEIGHT')}")
        print("Runtime updated successfully — no restart needed!")
    else:
        print(f"API update returned {r.status_code}: {r.text[:200]}")
        print("DB was updated but runtime needs a container restart to apply.")
except ImportError:
    print("requests not available inside container — DB updated, restart container to apply.")
except Exception as e:
    print(f"API update failed: {e}")
    print("DB was updated but runtime needs a container restart to apply.")
