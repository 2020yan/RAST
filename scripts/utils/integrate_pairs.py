#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合脚本：将四个JSON文件按照task_id进行配对，构造三组harmful-harmless配对输入

配对方法：
(1) plain pair：未封装的有害指令与未封装的无害指令
(2) failed-template pair：使用RAST搜索得到的越狱未成功模板封装后的有害指令，以及使用相同模板封装后的无害指令
(3) successful-template pair：使用RAST搜索得到的越狱成功模板封装后的有害指令，以及使用相同模板封装后的无害指令
"""

import json
from pathlib import Path

def load_json_file(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, file_path):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def main():
    # 文件路径
    safe_prompts_path = str(PROJECT_ROOT / "data/safe_prompts.json")
    unsafe_prompts_path = str(PROJECT_ROOT / "data/unsafe_prompts.json")
    safe_prompts_replaced_path = str(PROJECT_ROOT / "data/safe_prompts_replaced.json")
    failed_prompts_path = str(PROJECT_ROOT / "data/failed_prompts.json")
    output_path = str(PROJECT_ROOT / "data/integrated_pairs.json")
    
    # 加载所有数据文件
    print("Loading data files...")
    safe_prompts = load_json_file(safe_prompts_path)
    unsafe_prompts = load_json_file(unsafe_prompts_path)
    safe_prompts_replaced = load_json_file(safe_prompts_replaced_path)
    failed_prompts = load_json_file(failed_prompts_path)
    
    # 创建task_id到数据的映射
    safe_map = {item['task_id']: item for item in safe_prompts}
    unsafe_map = {item['task_id']: item for item in unsafe_prompts}
    safe_replaced_map = {item['task_id']: item for item in safe_prompts_replaced}
    failed_map = {item['task_id']: item for item in failed_prompts}
    
    print(f"Loaded data:")
    print(f"  safe_prompts: {len(safe_map)} items")
    print(f"  unsafe_prompts: {len(unsafe_map)} items")
    print(f"  safe_prompts_replaced: {len(safe_replaced_map)} items")
    print(f"  failed_prompts: {len(failed_map)} items")
    
    # 找到所有共同的task_id
    all_task_ids = set(safe_map.keys()) & set(unsafe_map.keys()) & set(safe_replaced_map.keys()) & set(failed_map.keys())
    print(f"Common task_ids: {len(all_task_ids)}")
    
    # 如果没有共同的task_id，尝试找最大的交集
    if len(all_task_ids) == 0:
        print("No common task_ids found. Trying to find maximum intersection...")
        # 找到至少在三个文件中存在的task_id
        task_id_sets = [set(safe_map.keys()), set(unsafe_map.keys()), set(safe_replaced_map.keys()), set(failed_map.keys())]
        all_possible_ids = set()
        for s in task_id_sets:
            all_possible_ids.update(s)
        
        valid_task_ids = []
        for task_id in all_possible_ids:
            count = sum(1 for s in task_id_sets if task_id in s)
            if count >= 3:  # 至少在3个文件中存在
                valid_task_ids.append(task_id)
        
        all_task_ids = set(valid_task_ids)
        print(f"Found {len(all_task_ids)} task_ids present in at least 3 files")
    
    # 构造配对数据
    integrated_data = []
    missing_data_info = []
    
    for task_id in sorted(all_task_ids):
        pair_data = {
            'task_id': task_id,
            'plain_pair': None,
            'failed_template_pair': None,
            'successful_template_pair': None
        }
        
        # (1) plain pair: 未封装的有害指令 vs 未封装的无害指令
        try:
            if task_id in unsafe_map and task_id in safe_map:
                pair_data['plain_pair'] = {
                    'harmful': unsafe_map[task_id]['task'],
                    'harmless': safe_map[task_id]['task']
                }
        except KeyError as e:
            missing_data_info.append(f"Missing data for plain pair, task_id {task_id}: {e}")
        
        # (2) failed-template pair: 越狱未成功模板封装的有害指令 vs 相同模板封装的无害指令
        try:
            if task_id in failed_map and task_id in safe_replaced_map:
                harmful_prompt = failed_map[task_id].get('failed_prompt')
                harmless_prompt = safe_replaced_map[task_id].get('successful_prompt')
                if harmful_prompt is not None and harmless_prompt is not None:
                    pair_data['failed_template_pair'] = {
                        'harmful': harmful_prompt,
                        'harmless': harmless_prompt
                    }
        except KeyError as e:
            missing_data_info.append(f"Missing data for failed-template pair, task_id {task_id}: {e}")
        
        # (3) successful-template pair: 越狱成功模板封装的有害指令 vs 相同模板封装的无害指令
        try:
            if task_id in unsafe_map and task_id in safe_map:
                harmful_prompt = unsafe_map[task_id].get('successful_prompt')
                harmless_prompt = safe_map[task_id].get('successful_prompt')
                if harmful_prompt is not None and harmless_prompt is not None:
                    pair_data['successful_template_pair'] = {
                        'harmful': harmful_prompt,
                        'harmless': harmless_prompt
                    }
        except KeyError as e:
            missing_data_info.append(f"Missing data for successful-template pair, task_id {task_id}: {e}")
        
        # 只有当至少有一个配对存在时才添加
        if (pair_data['plain_pair'] is not None or 
            pair_data['failed_template_pair'] is not None or 
            pair_data['successful_template_pair'] is not None):
            integrated_data.append(pair_data)
    
    print(f"\nIntegration results:")
    print(f"  Total integrated pairs: {len(integrated_data)}")
    
    # 统计每种配对的数量
    plain_count = sum(1 for item in integrated_data if item['plain_pair'] is not None)
    failed_count = sum(1 for item in integrated_data if item['failed_template_pair'] is not None)
    success_count = sum(1 for item in integrated_data if item['successful_template_pair'] is not None)
    
    print(f"  Plain pairs: {plain_count}")
    print(f"  Failed-template pairs: {failed_count}")
    print(f"  Successful-template pairs: {success_count}")
    
    # 打印缺失数据信息
    if missing_data_info:
        print(f"\nMissing data warnings ({len(missing_data_info)}):")
        for i, msg in enumerate(missing_data_info[:10]):  # 只打印前10条
            print(f"  {msg}")
        if len(missing_data_info) > 10:
            print(f"  ... and {len(missing_data_info) - 10} more warnings")
    
    # 保存结果
    save_json_file(integrated_data, output_path)
    print(f"\nIntegrated data saved to {output_path}")

if __name__ == "__main__":
    main()