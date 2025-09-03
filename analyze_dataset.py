#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 converted_fjs_dataset_new.json 数据集结构
"""

import json
from collections import Counter

def analyze_dataset():
    # 加载数据集
    with open('recommend_model_1.1/labeled_dataset/converted_fjs_dataset_new.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=== 数据集基本信息 ===")
    print(f"总样本数: {len(data)}")
    
    # 分析初始化方法分布
    methods = [sample['initialization_method'] for sample in data.values()]
    method_counts = Counter(methods)
    print(f"初始化方法分布: {dict(method_counts)}")
    
    # 分析原始FJS文件分布
    fjs_paths = [sample['original_fjs_path'] for sample in data.values()]
    fjs_counts = Counter(fjs_paths)
    print(f"唯一FJS文件数: {len(fjs_counts)}")
    print(f"前5个FJS文件: {list(fjs_counts.keys())[:5]}")
    
    print("\n=== 样本结构分析 ===")
    sample = data['sample_0001']
    
    print(f"样本ID: {sample['sample_id']}")
    print(f"原始FJS路径: {sample['original_fjs_path']}")
    print(f"初始化方法: {sample['initialization_method']}")
    
    print("\n--- 特征结构 ---")
    features = sample['features']
    print(f"特征类别: {list(features.keys())}")
    
    print("\n基础特征:")
    for k, v in features['basic_features'].items():
        print(f"  {k}: {v}")
    
    print("\n加工时间特征:")
    for k, v in features['processing_time_features'].items():
        print(f"  {k}: {v}")
    
    print("\n析取图特征:")
    dg = features['disjunctive_graphs_features']
    print(f"  nodes_count: {dg['nodes_count']}")
    print(f"  edges_count: {dg['edges_count']}")
    print(f"  initial_labels数量: {len(dg['initial_labels'])}")
    print(f"  solid_labels数量: {len(dg['solid_labels'])}")
    print(f"  dashed_labels数量: {len(dg['dashed_labels'])}")
    print(f"  solid_frequency数量: {len(dg['solid_frequency'])}")
    print(f"  dashed_frequency数量: {len(dg['dashed_frequency'])}")
    
    print("\nKDE特征:")
    kde = features['kde_features']
    print(f"  x_grid长度: {len(kde['x_grid'])}")
    print(f"  density长度: {len(kde['density'])}")
    print(f"  bandwidth: {kde['bandwidth']}")
    
    print("\n--- 性能数据结构 ---")
    perf = sample['performance_data']
    print(f"元启发式算法: {perf['meta_heuristic']}")
    print(f"执行次数: {perf['execution_times']}")
    print(f"最大迭代次数: {perf['max_iterations']}")
    print(f"初始化方法: {perf['initialization_method']}")
    
    print("\n性能指标:")
    for k, v in perf['performance_metrics'].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    analyze_dataset()





