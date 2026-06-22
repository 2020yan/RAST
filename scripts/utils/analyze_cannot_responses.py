#!/usr/bin/env python3
"""
分析 steering_details.jsonl 文件中不同 setting 下
response_preview 包含 "I cannot" 的样本统计
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_cannot_responses(jsonl_file_path):
    """
    分析 JSONL 文件中每个 setting 下包含 "I cannot" 的 response_preview
    """
    # 存储每个 setting 的统计信息
    stats = defaultdict(lambda: {"total": 0, "cannot_count": 0})
    
    # 读取 JSONL 文件
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                item = json.loads(line.strip())
                setting = item.get("setting", "unknown")
                response_preview = item.get("response_preview", "")
                
                # 统计总数
                stats[setting]["total"] += 1
                
                # 检查是否包含 "I cannot"（不区分大小写）
                if "I cannot" in response_preview or "i cannot" in response_preview.lower():
                    stats[setting]["cannot_count"] += 1
                    
            except json.JSONDecodeError as e:
                print(f"警告: 跳过无效的JSON行: {e}")
                continue
    
    # 打印结果
    print("=" * 60)
    print("不同 setting 下包含 'I cannot' 的 response 统计")
    print("=" * 60)
    print(f"{'Setting':<20} {'包含 I cannot':<15} {'总数':<10} {'百分比(%)':<12}")
    print("-" * 60)
    
    for setting in sorted(stats.keys()):
        total = stats[setting]["total"]
        cannot_count = stats[setting]["cannot_count"]
        percentage = (cannot_count / total * 100) if total > 0 else 0.0
        
        print(f"{setting:<20} {cannot_count:<15} {total:<10} {percentage:<12.2f}")
    
    print("-" * 60)
    print("说明: 统计包含 'I cannot' 或 'i cannot' 的 response")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    jsonl_file = str(PROJECT_ROOT / "SALT/results/rq3_rq4/run4/steering_details.jsonl")
    
    if not Path(jsonl_file).exists():
        print(f"错误: 文件不存在: {jsonl_file}")
        print("请确保路径正确，或者修改脚本中的文件路径")
    else:
        analyze_cannot_responses(jsonl_file)