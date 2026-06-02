#!/usr/bin/env python
"""
Convert absolute paths in opt.json files to relative paths for portability.
Relative paths are computed assuming the working directory is the project root.
"""

import json
import os
from pathlib import Path

def convert_abs_to_rel(path_str, opt_json_dir):
    """Convert absolute path to relative path."""
    if not isinstance(path_str, str):
        return path_str
    
    # Common absolute paths to convert (customize these for your environment)
    conversions = {
        "/data1/zhangshihang/Datasets/": "../Datasets/",
        "/data1/zhangshihang/VideoLights/": "../VideoLights/",
        "/data1/zhangshihang/GIRL_DETR/": "./",
    }
    
    for abs_prefix, rel_prefix in conversions.items():
        if path_str.startswith(abs_prefix):
            return path_str.replace(abs_prefix, rel_prefix, 1)
    
    return path_str

def convert_opt_json(opt_json_path):
    """Convert paths in opt.json file to relative paths."""
    print(f"\nProcessing: {opt_json_path}")
    
    with open(opt_json_path, 'r') as f:
        opt = json.load(f)
    
    # Track changes
    changed = False
    
    # Convert v_feat_dirs
    if "v_feat_dirs" in opt and opt["v_feat_dirs"]:
        original = opt["v_feat_dirs"].copy()
        opt["v_feat_dirs"] = [convert_abs_to_rel(p, opt_json_path) for p in opt["v_feat_dirs"]]
        if opt["v_feat_dirs"] != original:
            print(f"  v_feat_dirs changed:")
            for i, (old, new) in enumerate(zip(original, opt["v_feat_dirs"])):
                if old != new:
                    print(f"    [{i}] {old}")
                    print(f"     → {new}")
            changed = True
    
    # Convert t_feat_dirs
    if "t_feat_dirs" in opt and opt["t_feat_dirs"]:
        original = opt["t_feat_dirs"].copy()
        opt["t_feat_dirs"] = [convert_abs_to_rel(p, opt_json_path) for p in opt["t_feat_dirs"]]
        if opt["t_feat_dirs"] != original:
            print(f"  t_feat_dirs changed:")
            for i, (old, new) in enumerate(zip(original, opt["t_feat_dirs"])):
                if old != new:
                    print(f"    [{i}] {old}")
                    print(f"     → {new}")
            changed = True
    
    # Convert a_feat_dirs if exists
    if "a_feat_dirs" in opt and opt["a_feat_dirs"]:
        original = opt["a_feat_dirs"].copy()
        opt["a_feat_dirs"] = [convert_abs_to_rel(p, opt_json_path) for p in opt["a_feat_dirs"]]
        if opt["a_feat_dirs"] != original:
            print(f"  a_feat_dirs changed:")
            for i, (old, new) in enumerate(zip(original, opt["a_feat_dirs"])):
                if old != new:
                    print(f"    [{i}] {old}")
                    print(f"     → {new}")
            changed = True
    
    if changed:
        with open(opt_json_path, 'w') as f:
            json.dump(opt, f, indent=4)
        print(f"  ✓ Saved")
    else:
        print(f"  No changes needed")
    
    return changed

if __name__ == '__main__':
    # Find all opt.json files (can be customized via environment variable)
    results_dir = os.environ.get('RESULTS_DIR', os.path.join(os.path.dirname(__file__), "..", "results"))
    
    opt_json_files = list(Path(results_dir).glob("**/opt.json"))
    print(f"Found {len(opt_json_files)} opt.json files in {results_dir}")
    
    changed_count = 0
    for opt_json_path in sorted(opt_json_files):
        if convert_opt_json(str(opt_json_path)):
            changed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Modified {changed_count} files")
    print(f"{'='*60}")
