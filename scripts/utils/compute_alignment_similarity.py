#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算三组 harmful--harmless 配对输入在各层上的平均余弦相似度
用于回答 RQ3：结构化对抗模板是否会改变有害输入与无害输入在内部表征空间中的相对接近程度？
"""

import json
import torch
import torch.nn.functional as F
import os
from pathlib import Path
import argparse
import sys

# 添加项目路径
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "embedding-level"))

# 正确导入模块
from core.model_extraction import ModelExtraction
from core.embedding_manager import EmbeddingManager


def compute_cosine_similarity(harmful_embeddings, harmless_embeddings):
    """
    计算两个嵌入向量之间的余弦相似度
    
    Args:
        harmful_embeddings: [num_samples, hidden_dim] 
        harmless_embeddings: [num_samples, hidden_dim]
    
    Returns:
        similarities: [num_samples] 余弦相似度
    """
    # 归一化向量
    harmful_norm = F.normalize(harmful_embeddings, p=2, dim=1)
    harmless_norm = F.normalize(harmless_embeddings, p=2, dim=1)
    
    # 计算余弦相似度
    similarities = torch.sum(harmful_norm * harmless_norm, dim=1)
    return similarities


def extract_representations(model_extractor, harmful_inputs, harmless_inputs):
    """
    提取有害和无害输入在各层的表征
    
    Args:
        model_extractor: ModelExtraction实例
        harmful_inputs: 有害输入列表
        harmless_inputs: 无害输入列表
    
    Returns:
        harmful_embds_manager: 有害输入的EmbeddingManager
        harmless_embds_manager: 无害输入的EmbeddingManager
    """
    print(f"Extracting representations for {len(harmful_inputs)} harmful inputs...")
    harmful_embds_manager = model_extractor.extract_embds(harmful_inputs)
    
    print(f"Extracting representations for {len(harmless_inputs)} harmless inputs...")
    harmless_embds_manager = model_extractor.extract_embds(harmless_inputs)
    
    return harmful_embds_manager, harmless_embds_manager


def compute_layerwise_similarities(harmful_embds_manager, harmless_embds_manager):
    """
    计算各层的平均余弦相似度
    
    Args:
        harmful_embds_manager: 有害输入的EmbeddingManager
        harmless_embds_manager: 无害输入的EmbeddingManager
    
    Returns:
        avg_similarities: [num_layers] 各层的平均余弦相似度
    """
    num_layers = len(harmful_embds_manager.layers)
    num_samples = harmful_embds_manager.layers[0].size(0)
    
    print(f"Computing layerwise similarities for {num_layers} layers and {num_samples} samples...")
    
    avg_similarities = []
    
    for layer_idx in range(num_layers):
        harmful_layer_embds = harmful_embds_manager.layers[layer_idx]  # [num_samples, hidden_dim]
        harmless_layer_embds = harmless_embds_manager.layers[layer_idx]  # [num_samples, hidden_dim]
        
        # 计算该层所有样本的余弦相似度
        layer_similarities = compute_cosine_similarity(harmful_layer_embds, harmless_layer_embds)
        
        # 计算平均相似度
        avg_similarity = torch.mean(layer_similarities).item()
        avg_similarities.append(avg_similarity)
        
        if layer_idx % 5 == 0 or layer_idx == num_layers - 1:
            print(f"  Layer {layer_idx}: avg similarity = {avg_similarity:.4f}")
    
    return avg_similarities


def main():
    parser = argparse.ArgumentParser(description="Compute alignment similarities for RQ3 analysis")
    parser.add_argument("--input-file", type=str, default=str(PROJECT_ROOT / "data/integrated_pairs.json"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "SALT/results/rq3_rq4/run1"))
    parser.add_argument("--model-nickname", type=str, default="llama3-8b")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum number of samples to use (default: 50)")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配对数据
    print(f"Loading integrated pairs from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        integrated_pairs = json.load(f)
    
    print(f"Loaded {len(integrated_pairs)} integrated pairs")
    
    # 限制样本数量到50个（或指定的最大值）
    if len(integrated_pairs) > args.max_samples:
        integrated_pairs = integrated_pairs[:args.max_samples]
        print(f"Using first {args.max_samples} pairs for analysis")
    
    # 初始化模型提取器
    print(f"Initializing model extractor for {args.model_nickname}...")
    model_extractor = ModelExtraction(args.model_nickname)
    
    # 准备三组输入
    plain_harmful = []
    plain_harmless = []
    failed_harmful = []
    failed_harmless = []
    success_harmful = []
    success_harmless = []
    
    valid_pairs_count = 0
    
    for pair_data in integrated_pairs:
        # Plain pair
        if pair_data.get('plain_pair'):
            plain_harmful.append(pair_data['plain_pair']['harmful'])
            plain_harmless.append(pair_data['plain_pair']['harmless'])
            valid_pairs_count += 1
        
        # Failed-template pair
        if pair_data.get('failed_template_pair'):
            failed_harmful.append(pair_data['failed_template_pair']['harmful'])
            failed_harmless.append(pair_data['failed_template_pair']['harmless'])
        
        # Successful-template pair
        if pair_data.get('successful_template_pair'):
            success_harmful.append(pair_data['successful_template_pair']['harmful'])
            success_harmless.append(pair_data['successful_template_pair']['harmless'])
    
    print(f"Prepared inputs:")
    print(f"  Plain pairs: {len(plain_harmful)}")
    print(f"  Failed-template pairs: {len(failed_harmful)}")
    print(f"  Successful-template pairs: {len(success_harmful)}")
    
    # 计算三组配对的层间相似度
    results = {}
    
    # 1. Plain pair
    if plain_harmful:
        print("\n=== Computing similarities for PLAIN PAIR ===")
        harmful_embds, harmless_embds = extract_representations(model_extractor, plain_harmful, plain_harmless)
        plain_similarities = compute_layerwise_similarities(harmful_embds, harmless_embds)
        results['plain_pair'] = plain_similarities
        
        # 保存中间结果
        harmful_embds.save(os.path.join(args.output_dir, "plain_harmful"))
        harmless_embds.save(os.path.join(args.output_dir, "plain_harmless"))
    
    # 2. Failed-template pair
    if failed_harmful:
        print("\n=== Computing similarities for FAILED-TEMPLATE PAIR ===")
        harmful_embds, harmless_embds = extract_representations(model_extractor, failed_harmful, failed_harmless)
        failed_similarities = compute_layerwise_similarities(harmful_embds, harmless_embds)
        results['failed_template_pair'] = failed_similarities
        
        # 保存中间结果
        harmful_embds.save(os.path.join(args.output_dir, "failed_harmful"))
        harmless_embds.save(os.path.join(args.output_dir, "failed_harmless"))
    
    # 3. Successful-template pair
    if success_harmful:
        print("\n=== Computing similarities for SUCCESSFUL-TEMPLATE PAIR ===")
        harmful_embds, harmless_embds = extract_representations(model_extractor, success_harmful, success_harmless)
        success_similarities = compute_layerwise_similarities(harmful_embds, harmless_embds)
        results['successful_template_pair'] = success_similarities
        
        # 保存中间结果
        harmful_embds.save(os.path.join(args.output_dir, "success_harmful"))
        harmless_embds.save(os.path.join(args.output_dir, "success_harmless"))
    
    # 保存最终结果
    output_file = os.path.join(args.output_dir, "alignment_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print("\nSummary:")
    for condition, similarities in results.items():
        print(f"  {condition}:")
        print(f"    First layer similarity: {similarities[0]:.4f}")
        print(f"    Last layer similarity: {similarities[-1]:.4f}")
        print(f"    Max similarity: {max(similarities):.4f} at layer {similarities.index(max(similarities))}")
        print(f"    Min similarity: {min(similarities):.4f} at layer {similarities.index(min(similarities))}")


if __name__ == "__main__":
    main()