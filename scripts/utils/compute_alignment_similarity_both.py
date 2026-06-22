#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算三组 harmful--harmless 配对输入在各层上的平均余弦相似度（归一化和未归一化两个版本）
用于回答 RQ3：结构化对抗模板是否会改变有害输入与无害输入在内部表征空间中的相对接近程度？
"""

import json
import torch
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


def compute_cosine_similarity_normalized(harmful_embeddings, harmless_embeddings):
    """
    计算归一化的余弦相似度（标准余弦相似度）
    
    Args:
        harmful_embeddings: [num_samples, hidden_dim] 
        harmless_embeddings: [num_samples, hidden_dim]
    
    Returns:
        similarities: [num_samples] 归一化的余弦相似度 [-1, 1]
    """
    # 计算L2范数
    harmful_norm = torch.norm(harmful_embeddings, dim=1, keepdim=True)
    harmless_norm = torch.norm(harmless_embeddings, dim=1, keepdim=True)
    
    # 避免除零错误
    harmful_norm = torch.clamp(harmful_norm, min=1e-8)
    harmless_norm = torch.clamp(harmless_norm, min=1e-8)
    
    # 归一化向量
    harmful_normalized = harmful_embeddings / harmful_norm
    harmless_normalized = harmless_embeddings / harmless_norm
    
    # 计算点积（即余弦相似度）
    similarities = torch.sum(harmful_normalized * harmless_normalized, dim=1)
    return similarities


def compute_dot_product_unnormalized(harmful_embeddings, harmless_embeddings):
    """
    计算未归一化的点积相似度（原始内积）
    
    Args:
        harmful_embeddings: [num_samples, hidden_dim] 
        harmless_embeddings: [num_samples, hidden_dim]
    
    Returns:
        similarities: [num_samples] 未归一化的点积值
    """
    # 直接计算点积
    similarities = torch.sum(harmful_embeddings * harmless_embeddings, dim=1)
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


def compute_layerwise_similarities_both(harmful_embds_manager, harmless_embds_manager):
    """
    计算各层的平均相似度（归一化和未归一化两个版本）
    
    Args:
        harmful_embds_manager: 有害输入的EmbeddingManager
        harmless_embds_manager: 无害输入的EmbeddingManager
    
    Returns:
        results: 包含归一化和未归一化结果的字典
    """
    num_layers = len(harmful_embds_manager.layers)
    num_samples = harmful_embds_manager.layers[0].size(0)
    
    print(f"Computing layerwise similarities for {num_layers} layers and {num_samples} samples...")
    
    normalized_similarities = []
    unnormalized_similarities = []
    
    for layer_idx in range(num_layers):
        harmful_layer_embds = harmful_embds_manager.layers[layer_idx]  # [num_samples, hidden_dim]
        harmless_layer_embds = harmless_embds_manager.layers[layer_idx]  # [num_samples, hidden_dim]
        
        # 计算归一化的余弦相似度
        norm_similarities = compute_cosine_similarity_normalized(harmful_layer_embds, harmless_layer_embds)
        avg_norm_similarity = torch.mean(norm_similarities).item()
        normalized_similarities.append(avg_norm_similarity)
        
        # 计算未归一化的点积相似度
        unnorm_similarities = compute_dot_product_unnormalized(harmful_layer_embds, harmless_layer_embds)
        avg_unnorm_similarity = torch.mean(unnorm_similarities).item()
        unnormalized_similarities.append(avg_unnorm_similarity)
        
        if layer_idx % 5 == 0 or layer_idx == num_layers - 1:
            print(f"  Layer {layer_idx}: norm={avg_norm_similarity:.4f}, unnorm={avg_unnorm_similarity:.4f}")
    
    return {
        'normalized': normalized_similarities,
        'unnormalized': unnormalized_similarities
    }


def main():
    parser = argparse.ArgumentParser(description="Compute alignment similarities (both normalized and unnormalized) for RQ3 analysis")
    parser.add_argument("--input-file", type=str, default=str(PROJECT_ROOT / "data/integrated_pairs.json"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "SALT/results/rq3_results_both"))
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
    
    for pair_data in integrated_pairs:
        # Plain pair
        if pair_data.get('plain_pair'):
            plain_harmful.append(pair_data['plain_pair']['harmful'])
            plain_harmless.append(pair_data['plain_pair']['harmless'])
        
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
    
    # 计算三组配对的层间相似度（两种版本）
    results = {}
    
    # 1. Plain pair
    if plain_harmful:
        print("\n=== Computing similarities for PLAIN PAIR ===")
        harmful_embds, harmless_embds = extract_representations(model_extractor, plain_harmful, plain_harmless)
        plain_results = compute_layerwise_similarities_both(harmful_embds, harmless_embds)
        results['plain_pair'] = plain_results
        
        # 保存中间结果 - 修正保存路径
        harmful_embds.message = "plain_harmful"
        harmless_embds.message = "plain_harmless"
        harmful_embds.save(args.output_dir)
        harmless_embds.save(args.output_dir)
    
    # 2. Failed-template pair
    if failed_harmful:
        print("\n=== Computing similarities for FAILED-TEMPLATE PAIR ===")
        harmful_embds, harmless_embds = extract_representations(model_extractor, failed_harmful, failed_harmless)
        failed_results = compute_layerwise_similarities_both(harmful_embds, harmless_embds)
        results['failed_template_pair'] = failed_results
        
        # 保存中间结果 - 修正保存路径
        harmful_embds.message = "failed_harmful"
        harmless_embds.message = "failed_harmless"
        harmful_embds.save(args.output_dir)
        harmless_embds.save(args.output_dir)
    
    # 3. Successful-template pair
    if success_harmful:
        print("\n=== Computing similarities for SUCCESSFUL-TEMPLATE PAIR ===")
        harmful_embds, harmless_embds = extract_representations(model_extractor, success_harmful, success_harmless)
        success_results = compute_layerwise_similarities_both(harmful_embds, harmless_embds)
        results['successful_template_pair'] = success_results
        
        # 保存中间结果 - 修正保存路径
        harmful_embds.message = "success_harmful"
        harmless_embds.message = "success_harmless"
        harmful_embds.save(args.output_dir)
        harmless_embds.save(args.output_dir)
    
    # 保存最终结果
    output_file = os.path.join(args.output_dir, "alignment_data_both.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print("\nSummary (Normalized):")
    for condition, similarities_dict in results.items():
        similarities = similarities_dict['normalized']
        print(f"  {condition}:")
        print(f"    First layer similarity: {similarities[0]:.4f}")
        print(f"    Last layer similarity: {similarities[-1]:.4f}")
        print(f"    Max similarity: {max(similarities):.4f} at layer {similarities.index(max(similarities))}")
        print(f"    Min similarity: {min(similarities):.4f} at layer {similarities.index(min(similarities))}")
    
    print("\nSummary (Unnormalized):")
    for condition, similarities_dict in results.items():
        similarities = similarities_dict['unnormalized']
        print(f"  {condition}:")
        print(f"    First layer similarity: {similarities[0]:.4f}")
        print(f"    Last layer similarity: {similarities[-1]:.4f}")
        print(f"    Max similarity: {max(similarities):.4f} at layer {similarities.index(max(similarities))}")
        print(f"    Min similarity: {min(similarities):.4f} at layer {similarities.index(min(similarities))}")


if __name__ == "__main__":
    main()