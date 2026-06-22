#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 04_all_formatted_prompts.json 中提取所有latex_format 内容
并保存到 03_v_fmt_positive_training_set.json
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def extract_latex_formats():
    # 定义文件路径
    input_file = str(PROJECT_ROOT / 'data/unsafe_all_formatted_prompts.json')
    output_file = str(PROJECT_ROOT / 'data/04_v_unsafe_fmt_positive_training_set.json')
    
    # 读取输入文件
    print(f"正在读取 {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有latex_format 内容
    latex_list = []
    for item in data:
        if 'formats' in item and 'latex_format' in item['formats']:
            latex_content = item['formats']['latex_format']
            latex_list.append(latex_content)
    
    print(f"成功提取 {len(latex_list)} 条 latex_format 数据")
    
    # 保存到输出文件（格式与 02_v_safe_negative_training_set.json 一致）
    print(f"正在保存到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(latex_list, f, ensure_ascii=False, indent=4)
    
    print("保存完成！")
    print(f"输出文件：{output_file}")
    print(f"数据条数：{len(latex_list)}")

if __name__ == '__main__':
    extract_latex_formats()
