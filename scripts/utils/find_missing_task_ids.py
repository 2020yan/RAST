#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断脚本：找出四个文件中缺失的task_id
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def load_json_file(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # 文件路径
    safe_prompts_path = str(PROJECT_ROOT / "data/safe_prompts.json")
    unsafe_prompts_path = str(PROJECT_ROOT / "data/unsafe_prompts.json")
    safe_prompts_replaced_path = str(PROJECT_ROOT / "data/safe_prompts_replaced.json")
    failed_prompts_path = str(PROJECT_ROOT / "data/failed_prompts.json")
    
    # 加载所有数据文件
    print("Loading data files...")
    safe_prompts = load_json_file(safe_prompts_path)
    unsafe_prompts = load_json_file(unsafe_prompts_path)
    safe_prompts_replaced = load_json_file(safe_prompts_replaced_path)
    failed_prompts = load_json_file(failed_prompts_path)
    
    # 创建task_id集合
    safe_ids = set(item['task_id'] for item in safe_prompts)
    unsafe_ids = set(item['task_id'] for item in unsafe_prompts)
    safe_replaced_ids = set(item['task_id'] for item in safe_prompts_replaced)
    failed_ids = set(item['task_id'] for item in failed_prompts)
    
    print(f"File sizes:")
    print(f"  safe_prompts: {len(safe_ids)} task_ids")
    print(f"  unsafe_prompts: {len(unsafe_ids)} task_ids")
    print(f"  safe_prompts_replaced: {len(safe_replaced_ids)} task_ids")
    print(f"  failed_prompts: {len(failed_ids)} task_ids")
    
    # 找出缺失的task_id
    print(f"\nMissing task_ids analysis:")
    
    # 在safe中但不在unsafe中的task_id
    missing_in_unsafe = safe_ids - unsafe_ids
    if missing_in_unsafe:
        print(f"  Task IDs in safe_prompts but NOT in unsafe_prompts: {sorted(missing_in_unsafe)}")
    
    # 在safe_replaced中但不在failed中的task_id  
    missing_in_failed = safe_replaced_ids - failed_ids
    if missing_in_failed:
        print(f"  Task IDs in safe_prompts_replaced but NOT in failed_prompts: {sorted(missing_in_failed)}")
    
    # 在unsafe中但不在failed中的task_id
    missing_between_unsafe_failed = unsafe_ids - failed_ids
    if missing_between_unsafe_failed:
        print(f"  Task IDs in unsafe_prompts but NOT in failed_prompts: {sorted(missing_between_unsafe_failed)}")
    
    # 共同的task_id
    common_ids = safe_ids & unsafe_ids & safe_replaced_ids & failed_ids
    print(f"\nCommon task_ids across all files: {len(common_ids)}")
    print(f"Common task_ids: {sorted(common_ids)[:10]}{'...' if len(common_ids) > 10 else ''}")
    
    # 验证总数
    all_ids = safe_ids | unsafe_ids | safe_replaced_ids | failed_ids
    print(f"\nTotal unique task_ids across all files: {len(all_ids)}")

if __name__ == "__main__":
    main()