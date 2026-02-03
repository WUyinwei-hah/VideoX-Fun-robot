#!/usr/bin/env python3
"""
Script to add foreground_source_abs field to wan_i2v_staircase_test_prompts.jsonl
"""

import json
import os
import shutil


def main():
    jsonl_path = "/gemini/code/VideoX-Fun/examples/wan2.2_fun/wan_i2v_staircase_test_prompts.jsonl"
    
    # Backup
    backup_path = jsonl_path + ".bak"
    shutil.copy2(jsonl_path, backup_path)
    print(f"Backup created: {backup_path}")
    
    # Read
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    
    # Update each item
    missing = 0
    for item in items:
        source_abs = item.get("source_abs", "")
        if "/source/" in source_abs:
            foreground_abs = source_abs.replace("/source/", "/foreground_segmented/")
        else:
            foreground_abs = ""
        
        if foreground_abs and os.path.exists(foreground_abs):
            item["foreground_source_abs"] = foreground_abs
        else:
            item["foreground_source_abs"] = ""
            missing += 1
    
    # Write back
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Updated {len(items)} items in {jsonl_path}")
    print(f"Missing foreground videos: {missing}")


if __name__ == "__main__":
    main()
