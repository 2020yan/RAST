#!/usr/bin/env python3
"""
使用sa_successful_prompts.json中的成功策略为adv.jsonl中的查询生成提示词
输出格式与sa_successful_prompts.json保持一致
"""

import json
import random
from pathlib import Path
import sys

# 添加项目路径
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "SALT" / "src"))

from prompt_builder import build_prompt
from strategy_dimensions import DIMENSION_OPTIONS


def load_adv_queries(file_path: str, limit: int = 50) -> list:
    """加载adv.jsonl中的查询和关键词"""
    queries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data = json.loads(line.strip())
            queries.append({
                'query': data['query'],
                'keywords': data['keywords']
            })
    return queries


def load_successful_strategies(file_path: str, limit: int = 50) -> list:
    """加载sa_successful_prompts.json中的成功策略"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    strategies = []
    for i, item in enumerate(data):
        if i >= limit:
            break
        strategies.append(item['successful_strategy'])
    
    return strategies


def extract_sensitive_words(query: str, keywords: list) -> list:
    """从查询和关键词中提取敏感词"""
    # 使用关键词作为敏感词
    sensitive_words = []
    for keyword in keywords:
        # 如果关键词包含多个词，取第一个主要词
        if isinstance(keyword, str):
            words = keyword.split()
            if words:
                sensitive_words.append(words[0])
        else:
            sensitive_words.append(str(keyword))
    
    # 确保至少有一个敏感词
    if not sensitive_words:
        # 从查询中提取主要名词
        query_words = query.replace('?', '').replace('.', '').split()
        sensitive_words = [word for word in query_words if len(word) > 3][:2]
        if not sensitive_words:
            sensitive_words = ['task']
    
    return sensitive_words[:4]  # 最多4个敏感词


def main():
    # 文件路径
    adv_file = str(PROJECT_ROOT / "data/adv.jsonl")
    strategies_file = str(PROJECT_ROOT / "data/sa_successful_prompts.json")
    output_file = str(PROJECT_ROOT / "data/generated_prompts.json")
    
    # 加载数据
    print("Loading queries from adv.jsonl...")
    queries = load_adv_queries(adv_file, 50)
    print(f"Loaded {len(queries)} queries")
    
    print("Loading successful strategies...")
    strategies = load_successful_strategies(strategies_file, 50)
    print(f"Loaded {len(strategies)} strategies")
    
    # 确保数量匹配
    min_count = min(len(queries), len(strategies), 50)
    queries = queries[:min_count]
    strategies = strategies[:min_count]
    
    print(f"Generating prompts for {min_count} query-strategy pairs...")
    
    # 生成结果数组，格式与sa_successful_prompts.json一致
    results = []
    for i in range(min_count):
        query_data = queries[i]
        strategy = strategies[i]
        
        # 提取敏感词
        sensitive_words = extract_sensitive_words(query_data['query'], query_data['keywords'])
        
        # 使用随机种子确保可重现性
        rng = random.Random(i)
        
        # 构建提示词
        prompt = build_prompt(
            base_task=query_data['query'],
            strategy=strategy,
            sensitive_words=sensitive_words,
            rng=rng
        )
        
        result = {
            'task_id': i,
            'task': query_data['query'],
            'successful_prompt': prompt,
            'successful_strategy': strategy
        }
        results.append(result)
        
        print(f"Generated prompt {i+1}/{min_count}")
    
    # 保存结果为JSON数组（与sa_successful_prompts.json格式一致）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")
    print("Sample generated prompt:")
    print("=" * 50)
    print(results[0]['successful_prompt'][:200] + "...")
    print("=" * 50)


if __name__ == "__main__":
    main()