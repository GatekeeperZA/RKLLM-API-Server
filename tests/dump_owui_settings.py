#!/usr/bin/env python3
"""Dump all Open WebUI settings, model configs, and capabilities from the database."""
import sqlite3, json, sys

DB = "/app/backend/data/webui.db"

def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # 1. List all tables
    c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in c.fetchall()]
    print("=" * 60)
    print("TABLES")
    print("=" * 60)
    for t in tables:
        c.execute(f'SELECT COUNT(*) FROM [{t}]')
        cnt = c.fetchone()[0]
        print(f"  {t}: {cnt} rows")

    # 2. Config / settings table
    print("\n" + "=" * 60)
    print("CONFIG TABLE")
    print("=" * 60)
    if 'config' in tables:
        c.execute("SELECT * FROM config")
        for row in c.fetchall():
            d = dict(row)
            for k, v in d.items():
                if isinstance(v, str) and v.startswith('{'):
                    try:
                        d[k] = json.loads(v)
                    except:
                        pass
            print(json.dumps(d, indent=2, default=str))

    # 3. Settings table (if exists)
    for tname in ['setting', 'settings', 'config_metadata']:
        if tname in tables:
            print(f"\n{'=' * 60}")
            print(f"{tname.upper()} TABLE")
            print(f"{'=' * 60}")
            c.execute(f"SELECT * FROM [{tname}]")
            for row in c.fetchall():
                d = dict(row)
                for k, v in d.items():
                    if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                        try:
                            d[k] = json.loads(v)
                        except:
                            pass
                print(json.dumps(d, indent=2, default=str))

    # 4. Model table
    print("\n" + "=" * 60)
    print("MODEL TABLE")
    print("=" * 60)
    if 'model' in tables:
        c.execute("SELECT * FROM model")
        for row in c.fetchall():
            d = dict(row)
            for k, v in d.items():
                if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                    try:
                        d[k] = json.loads(v)
                    except:
                        pass
            print(json.dumps(d, indent=2, default=str))

    # 5. User settings
    print("\n" + "=" * 60)
    print("USER TABLE (settings only)")
    print("=" * 60)
    if 'user' in tables:
        c.execute("SELECT id, name, email, role, settings, info FROM user")
        for row in c.fetchall():
            d = dict(row)
            for k in ['settings', 'info']:
                if d.get(k) and isinstance(d[k], str):
                    try:
                        d[k] = json.loads(d[k])
                    except:
                        pass
            print(json.dumps(d, indent=2, default=str))

    # 6. Function/tool tables
    for tname in ['function', 'tool']:
        if tname in tables:
            print(f"\n{'=' * 60}")
            print(f"{tname.upper()} TABLE")
            print(f"{'=' * 60}")
            c.execute(f"SELECT id, name, type, is_active, is_global FROM [{tname}]")
            for row in c.fetchall():
                print(json.dumps(dict(row), indent=2, default=str))

    # 7. Check for any RAG/document related tables
    rag_tables = [t for t in tables if any(k in t.lower() for k in ['document', 'file', 'knowledge', 'memory', 'rag'])]
    if rag_tables:
        print(f"\n{'=' * 60}")
        print("RAG/DOCUMENT TABLES")
        print(f"{'=' * 60}")
        for t in rag_tables:
            c.execute(f'SELECT COUNT(*) FROM [{t}]')
            cnt = c.fetchone()[0]
            print(f"  {t}: {cnt} rows")
            if cnt > 0 and cnt <= 20:
                c.execute(f"SELECT * FROM [{t}]")
                cols = [desc[0] for desc in c.description]
                print(f"    Columns: {cols}")
                for row in c.fetchall():
                    d = dict(row)
                    # Truncate long values
                    for k, v in d.items():
                        if isinstance(v, str) and len(v) > 200:
                            d[k] = v[:200] + "..."
                    print(f"    {json.dumps(d, default=str)}")

    conn.close()

if __name__ == "__main__":
    main()
