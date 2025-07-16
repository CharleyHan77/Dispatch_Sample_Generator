#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新数据特征提取模块
专门用于提取new_rdata_la02_j.fjs的特征
"""

import os
import json
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.utils import parser
from feature_extraction.feature_extractor import FeatureExtractor
from base_config import KDE_PARAMS, X_GRID

# 导入析取图特征生成相关函数
import networkx as nx
from typing import Dict, Any
from comparison_disjunctive_graphs.extract_graph_features import (
    create_disjunctive_graph_with_attributes,
    init_node_labels,
    add_edge_attributes,
    wl_step
)

def extract_processing_times(fjs_file):
    """从fjs文件中提取所有加工时间"""
    processing_times = []
    with open(fjs_file, 'r') as f:
        lines = f.readlines()
        # 跳过前两行（作业数和机器数）
        for line in lines[2:]:
            # 分割每行数据
            parts = line.strip().split()
            if len(parts) > 1:  # 确保不是空行
                # 每两个数字为一组（机器编号和加工时间）
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        processing_times.append(float(parts[i + 1]))
    return processing_times

def generate_kde(data, bandwidth, x_grid):
    """生成KDE并评估密度"""
    kde = gaussian_kde(data, bw_method=bandwidth/np.std(data))
    return kde.evaluate(x_grid)

def extract_disjunctive_graph_features(fjs_path: str) -> Dict:
    """提取单个FJS文件的析取图特征"""
    print(f"  提取析取图特征...")
    
    try:
        # 解析FJS文件
        parameters = parser.parse(fjs_path)
        
        # 创建析取图
        graph = create_disjunctive_graph_with_attributes(parameters, os.path.basename(fjs_path))
        
        # 初始化节点标签
        graph = init_node_labels(graph)
        
        # 添加边属性（与compare_graphs_wl.py保持一致）
        graph = add_edge_attributes(graph)
        
        # 获取初始标签
        initial_labels = {node: graph.nodes[node]['label'] for node in graph.nodes()}
        
        # 第一轮WL迭代（实线）
        solid_labels = wl_step(graph, 'solid', initial_labels)
        
        # 第二轮WL迭代（虚线）
        dashed_labels = wl_step(graph, 'dashed', solid_labels)
        
        # 分别统计实线和虚线WL的标签频率，然后加权
        def get_label_frequency(labels):
            """统计标签频率"""
            freq = {}
            for label in labels.values():
                freq[label] = freq.get(label, 0) + 1
            return freq
        
        # 分别获取实线和虚线标签频率
        solid_frequency = get_label_frequency(solid_labels)
        dashed_frequency = get_label_frequency(dashed_labels)
        
        # 构建graph_info，与compare_graphs_wl.py中的graph_info结构完全一致
        graph_info = {
            "nodes_count": len(graph.nodes()),
            "edges_count": len(graph.edges()),
            "initial_labels": initial_labels,
            "solid_labels": solid_labels,
            "dashed_labels": dashed_labels,
            "solid_frequency": solid_frequency,
            "dashed_frequency": dashed_frequency
        }
        
        return graph_info
        
    except Exception as e:
        print(f"    处理析取图特征时发生错误: {str(e)}")
        return {"error": str(e)}

def extract_new_data_features(fjs_file_path: str) -> Dict:
    """
    提取新数据的完整特征
    
    Args:
        fjs_file_path: FJS文件路径
        
    Returns:
        Dict: 包含所有特征的字典
    """
    print(f"开始提取新数据特征: {fjs_file_path}")
    
    # 初始化完整特征字典
    complete_features = {}

    try:
        # 1. 提取基础特征
        print("正在提取基础特征...")
        parameters = parser.parse(fjs_file_path)
        extractor = FeatureExtractor(parameters)
        all_features = extractor.extract_all_features()
        # 只取basic_features部分，避免嵌套
        complete_features["basic_features"] = all_features["basic_features"]
        print("  [OK] 基础特征提取完成")

        # 2. 生成加工时间特征
        print("正在生成加工时间特征...")
        processing_times = extract_processing_times(fjs_file_path)
        processing_time_features = {
            "processing_time_mean": np.mean(processing_times),
            "processing_time_std": np.std(processing_times),
            "processing_time_min": min(processing_times),
            "processing_time_max": max(processing_times),
            "machine_time_variance": np.var(processing_times)
        }
        complete_features["processing_time_features"] = processing_time_features
        print("  [OK] 加工时间特征生成完成")

        # 3. 生成KDE特征
        print("正在生成KDE特征...")
        # 从base_config.py获取参数
        bandwidth = KDE_PARAMS["bandwidth"]
        x_grid = np.array(X_GRID)
        
        # 生成KDE
        density = generate_kde(processing_times, bandwidth, x_grid)
        
        # 构建KDE特征（与原始格式完全一致）
        kde_features = {
            "x_grid": x_grid.tolist(),
            "density": density.tolist(),
            "bandwidth": bandwidth
        }
        complete_features["kde_features"] = kde_features
        print("  [OK] KDE特征生成完成")

        # 4. 生成析取图特征
        print("正在生成析取图特征...")
        disjunctive_graphs_features = extract_disjunctive_graph_features(fjs_file_path)
        complete_features["disjunctive_graphs_features"] = disjunctive_graphs_features
        if "error" not in disjunctive_graphs_features:
            print("  [OK] 析取图特征生成完成")
        else:
            print("  [ERROR] 析取图特征生成失败")

        # 验证特征结构
        print("\n验证特征结构:")
        feature_keys = list(complete_features.keys())
        print(f"  特征字段: {feature_keys}")
        
        expected_features = ['basic_features', 'processing_time_features', 'kde_features', 'disjunctive_graphs_features']
        missing_features = [f for f in expected_features if f not in feature_keys]
        if missing_features:
            print(f"  缺失特征: {missing_features}")
        else:
            print("  [OK] 所有预期特征字段都已生成")

        return complete_features

    except Exception as e:
        print(f"特征提取过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """测试特征提取功能"""
    fjs_file = "new_rdata_la02_j.fjs"
    
    if not os.path.exists(fjs_file):
        print(f"错误: 文件 {fjs_file} 不存在")
        return
    
    # 提取特征
    features = extract_new_data_features(fjs_file)
    
    if features:
        # 保存特征到JSON文件
        output_file = "new_data_features.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({fjs_file: features}, f, indent=2, ensure_ascii=False)
        print(f"\n特征已保存到: {output_file}")
        
        # 打印特征摘要
        print("\n特征摘要:")
        print(f"  基础特征: {len(features['basic_features'])} 个字段")
        print(f"  加工时间特征: {len(features['processing_time_features'])} 个字段")
        print(f"  KDE特征: {len(features['kde_features'])} 个字段")
        print(f"  析取图特征: {len(features['disjunctive_graphs_features'])} 个字段")
    else:
        print("特征提取失败")

if __name__ == "__main__":
    main() 