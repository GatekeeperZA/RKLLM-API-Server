#!/usr/bin/env python3
"""
Fix Open WebUI model capabilities in the database.

Problem: Most models have incorrect capability flags (vision, image_generation,
code_interpreter, builtin_tools, citations) that don't match actual hardware.

Only deepseekocr has a VL encoder. No models support image generation,
code interpretation, or built-in function calling on our RK3588 setup.
"""

import sqlite3
import json
import sys

DB_PATH = "/app/backend/data/webui.db"

# Define CORRECT capabilities for each model
# Key: model id prefix → correct capability overrides
# Defaults: vision=false, image_generation=false, code_interpreter=false,
#           builtin_tools=false, citations=false
# Only deepseekocr gets vision=true

CORRECT_CAPS = {
    # NPU models (our RKLLM server)
    "qwen3-1.7b":              {"vision": False, "image_generation": False, "code_interpreter": False, "citations": False, "builtin_tools": False},
    "qwen3-4b-instruct-2507":  {"vision": False, "image_generation": False, "code_interpreter": False, "citations": False, "builtin_tools": False},
    "phi-3-mini-4k-instruct":  {"vision": False, "image_generation": False, "code_interpreter": False, "citations": False, "builtin_tools": False},
    "gemma-3-4b-it":           {"vision": False, "image_generation": False, "code_interpreter": False, "citations": False, "builtin_tools": False},
    "deepseekocr":             {"vision": True,  "image_generation": False, "code_interpreter": False, "citations": False, "builtin_tools": False},
    
    # Ollama models
    "deepcoder-1.5b-preview":       {"vision": False, "image_generation": False, "code_interpreter": False, "citations": False, "builtin_tools": False},
    "deepseek-r1-distill-qwen-1.5b":{"vision": False, "image_generation": False, "code_interpreter": False, "citations": False, "builtin_tools": False},
    "deepseek-r1:7b":               {"vision": False, "image_generation": False, "code_interpreter": False, "citations": False, "builtin_tools": False},
    "deepseek-r1:7b-4k":            {"vision": False, "image_generation": False, "code_interpreter": False, "citations": False, "builtin_tools": False},
    "llama-3.2-3b-instruct":        {"vision": False, "image_generation": False, "code_interpreter": False, "citations": False, "builtin_tools": False},
    "nextcoder-7b":                  {"vision": False, "image_generation": False, "code_interpreter": False, "citations": False, "builtin_tools": False},
    "qwen2.5-3b-instruct":          {"vision": False, "image_generation": False, "code_interpreter": False, "citations": False, "builtin_tools": False},
    "qwen2.5-coder-3b-instruct":    {"vision": False, "image_generation": False, "code_interpreter": False, "citations": False, "builtin_tools": False},
    "qwen3:8b-q4_K_M":              {"vision": False, "image_generation": False, "code_interpreter": False, "citations": False, "builtin_tools": False},
}

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get all models
    cur.execute("SELECT id, meta FROM model")
    rows = cur.fetchall()
    
    print(f"Found {len(rows)} models in database\n")
    
    changes = []
    
    for row in rows:
        model_id = row["id"]
        meta_raw = row["meta"]
        
        if not meta_raw:
            print(f"  SKIP {model_id}: no meta")
            continue
            
        meta = json.loads(meta_raw)
        caps = meta.get("capabilities", {})
        
        # Find matching correction
        correct = None
        for key, val in CORRECT_CAPS.items():
            if model_id == key or model_id.startswith(key):
                correct = val
                break
        
        if correct is None:
            print(f"  SKIP {model_id}: no correction defined")
            continue
        
        # Check what needs changing
        needs_update = False
        diffs = []
        
        for cap_name, correct_val in correct.items():
            current_val = caps.get(cap_name, None)
            if current_val != correct_val:
                diffs.append(f"    {cap_name}: {current_val} -> {correct_val}")
                caps[cap_name] = correct_val
                needs_update = True
        
        if needs_update:
            meta["capabilities"] = caps
            new_meta = json.dumps(meta)
            changes.append((model_id, new_meta, diffs))
            print(f"  FIX  {model_id}:")
            for d in diffs:
                print(d)
        else:
            print(f"  OK   {model_id}: all capabilities correct")
    
    if not changes:
        print("\nNo changes needed!")
        return
    
    print(f"\n{'='*60}")
    print(f"Applying {len(changes)} model updates...")
    
    for model_id, new_meta, _ in changes:
        cur.execute("UPDATE model SET meta = ? WHERE id = ?", (new_meta, model_id))
    
    conn.commit()
    print("COMMITTED successfully!")
    
    # Verify
    print(f"\n{'='*60}")
    print("VERIFICATION - reading back all models:\n")
    
    cur.execute("SELECT id, meta FROM model")
    rows = cur.fetchall()
    
    for row in rows:
        model_id = row["id"]
        meta = json.loads(row["meta"]) if row["meta"] else {}
        caps = meta.get("capabilities", {})
        
        vision = caps.get("vision", "?")
        img_gen = caps.get("image_generation", "?")
        code_int = caps.get("code_interpreter", "?")
        citations = caps.get("citations", "?")
        builtin = caps.get("builtin_tools", "?")
        
        status = "VL" if vision else "  "
        print(f"  [{status}] {model_id:40s} vision={str(vision):5s} img_gen={str(img_gen):5s} code_int={str(code_int):5s} citations={str(citations):5s} tools={str(builtin):5s}")
    
    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
