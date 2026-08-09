#!/usr/bin/env python3
"""
db_fix.py — Lock all OpenWebUI interface settings to the database.

Run this after any in-place upgrade (docker compose pull && up -d) to
re-apply correct settings without wiping the volume. Also run it after
a fresh install if you want settings active before a UI visit.

Usage (from repo root on the host):
    docker cp tests/db_fix.py open-webui:/tmp/db_fix.py
    docker exec open-webui python3 /tmp/db_fix.py

Why this exists:
    OpenWebUI's PersistentConfig means env vars only set defaults on a
    fresh volume. Once a key exists in the DB, the DB value wins over
    env vars. This script writes the correct values directly so they
    survive restarts and in-place image upgrades.
"""

import sqlite3
import json
import time

DB = "/app/backend/data/webui.db"

QUERY_PROMPT = """Generate 1-3 web search queries to find information needed to answer the question.
Today: {{CURRENT_DATE}}
Return ONLY JSON, no other text: { "queries": ["query1", "query2"] }
If no search is needed: { "queries": [] }
<chat_history>
{{MESSAGES:END:6}}
</chat_history>"""

COMPACTION_PROMPT = """### Task:
Summarize the conversation history that will be compacted out of the active chat context.

### Instructions:
- Preserve key decisions, user preferences, and constraints.
- Preserve files, artifacts, tool results, and code changes that matter going forward.
- Preserve the current task state, unresolved questions, and next steps.
- Preserve exact technical details: command syntax, file paths, IP addresses, model names, version numbers, error messages, and configuration values.
- Be factual and specific. Do not invent details.
- Keep the summary concise, but complete enough for the assistant to continue without the removed messages.

### Previous Summary:
{{PREVIOUS_SUMMARY}}

### Messages Being Compacted:
{{COMPACTED_MESSAGES}}

### Recent Messages Kept In Context:
{{RECENT_MESSAGES}}"""

SETTINGS = {
    # Task model — fastest local model handles all background tasks
    "task.model.default":                           json.dumps("qwen3-1.7b"),
    # Background generation tasks — OFF (each = 1 extra NPU call → 503 risk)
    "task.tags.enable":                             "false",
    "task.follow_up.enable":                        "false",
    "task.autocomplete.enable":                     "false",
    # Title generation — ON (fires once at natural pause, good UX value)
    "task.title.enable":                            "true",
    # Query generation — ON for both web search and RAG retrieval
    "task.query.search.enable":                     "true",
    "task.query.retrieval.enable":                  "true",
    # Simplified query prompt (default is too long for small models → broken JSON)
    "task.query.prompt_template":                   json.dumps(QUERY_PROMPT),
    # Context compaction — enabled and tuned for phi-3-mini's 4K context window
    "chat.context_compaction.enable":               "true",
    "chat.context_compaction.model":                json.dumps("qwen3-1.7b"),
    "chat.context_compaction.token_threshold":      3000,
    "chat.context_compaction.token_cap":            32000,
    "chat.context_compaction.retention_percentage": 30,
    # Custom compaction prompt — adds technical detail preservation
    "chat.context_compaction.prompt_template":      json.dumps(COMPACTION_PROMPT),
    # Embedding backend — use local NPU service on port 8001 (embed_api.py)
    # Set engine to "openai" so OWUI posts to our OpenAI-compatible /v1/embeddings
    "rag.embedding.engine":                         json.dumps("openai"),
    "rag.embedding.openai.url":                     json.dumps("http://192.168.2.180:8001/v1"),
    "rag.embedding.openai.key":                     json.dumps("na"),
    "rag.embedding.model":                          json.dumps("Qwen3-Embedding-0.6B"),
    "rag.embedding.batch_size":                     8,
}


def main():
    db = sqlite3.connect(DB)
    ts = int(time.time())
    updated = inserted = 0

    for key, val in SETTINGS.items():
        existing = db.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
        if existing:
            db.execute("UPDATE config SET value=?, updated_at=? WHERE key=?", (val, ts, key))
            updated += 1
        else:
            db.execute(
                "INSERT INTO config (key, value, updated_at) VALUES (?,?,?)", (key, val, ts)
            )
            inserted += 1
        short = key.split(".")[-1]
        print(f"  SET  {short} = {str(val)[:60]}")

    db.commit()
    db.close()
    print(f"\nDone — {updated} updated, {inserted} inserted.")


if __name__ == "__main__":
    main()
