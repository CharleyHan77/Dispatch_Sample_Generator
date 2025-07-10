#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试析取图相似度计算
"""

import json
import numpy as np

def calculate_disjunctive_graph_similarity(graph_info1, graph_info2):
    """计算两个析取图的相似度，基于图结构特征和WL标签频率"""
    # 获取图的基本结构特征
    nodes1 = graph_info1['nodes_count']
    nodes2 = graph_info2['nodes_count']
    edges1 = graph_info1['edges_count']
    edges2 = graph_info2['edges_count']
    
    print(f"Graph1: nodes={nodes1}, edges={edges1}")
    print(f"Graph2: nodes={nodes2}, edges={edges2}")
    
    # 计算结构相似度（基于节点数和边数的相似性）
    nodes_similarity = 1 - abs(nodes1 - nodes2) / max(nodes1, nodes2)
    edges_similarity = 1 - abs(edges1 - edges2) / max(edges1, edges2)
    structure_similarity = (nodes_similarity + edges_similarity) / 2
    
    print(f"Structure similarity: {structure_similarity:.4f}")
    
    # 获取实线和虚线标签频率
    solid_freq1 = graph_info1['solid_frequency']
    solid_freq2 = graph_info2['solid_frequency']
    dashed_freq1 = graph_info1['dashed_frequency']
    dashed_freq2 = graph_info2['dashed_frequency']
    
    print(f"Graph1 solid_frequency keys: {list(solid_freq1.keys())[:5]}")
    print(f"Graph2 solid_frequency keys: {list(solid_freq2.keys())[:5]}")
    
    # 计算实线标签的Jaccard相似度
    solid_keys1 = set(solid_freq1.keys())
    solid_keys2 = set(solid_freq2.keys())
    if len(solid_keys1.union(solid_keys2)) > 0:
        solid_jaccard = len(solid_keys1.intersection(solid_keys2)) / len(solid_keys1.union(solid_keys2))
    else:
        solid_jaccard = 0.0
    
    # 计算虚线标签的Jaccard相似度
    dashed_keys1 = set(dashed_freq1.keys())
    dashed_keys2 = set(dashed_freq2.keys())
    if len(dashed_keys1.union(dashed_keys2)) > 0:
        dashed_jaccard = len(dashed_keys1.intersection(dashed_keys2)) / len(dashed_keys1.union(dashed_keys2))
    else:
        dashed_jaccard = 0.0
    
    print(f"Solid Jaccard: {solid_jaccard:.4f}, Dashed Jaccard: {dashed_jaccard:.4f}")
    
    # 计算标签分布相似度（使用余弦相似度）
    all_solid_keys = solid_keys1.union(solid_keys2)
    solid_vec1 = np.array([solid_freq1.get(k, 0) for k in all_solid_keys])
    solid_vec2 = np.array([solid_freq2.get(k, 0) for k in all_solid_keys])
    
    solid_norm1 = np.linalg.norm(solid_vec1)
    solid_norm2 = np.linalg.norm(solid_vec2)
    
    if solid_norm1 == 0 or solid_norm2 == 0:
        solid_cosine = 0.0
    else:
        solid_cosine = np.dot(solid_vec1, solid_vec2) / (solid_norm1 * solid_norm2)
    
    all_dashed_keys = dashed_keys1.union(dashed_keys2)
    dashed_vec1 = np.array([dashed_freq1.get(k, 0) for k in all_dashed_keys])
    dashed_vec2 = np.array([dashed_freq2.get(k, 0) for k in all_dashed_keys])
    
    dashed_norm1 = np.linalg.norm(dashed_vec1)
    dashed_norm2 = np.linalg.norm(dashed_vec2)
    
    if dashed_norm1 == 0 or dashed_norm2 == 0:
        dashed_cosine = 0.0
    else:
        dashed_cosine = np.dot(dashed_vec1, dashed_vec2) / (dashed_norm1 * dashed_norm2)
    
    print(f"Solid cosine: {solid_cosine:.4f}, Dashed cosine: {dashed_cosine:.4f}")
    
    # 综合相似度计算
    # 调整权重分配，增加结构相似度的影响
    solid_similarity = 0.3 * solid_jaccard + 0.7 * solid_cosine
    dashed_similarity = 0.3 * dashed_jaccard + 0.7 * dashed_cosine
    
    # 加权组合相似度（实线权重0.6，虚线权重0.4）
    label_similarity = 0.6 * solid_similarity + 0.4 * dashed_similarity
    
    # 最终相似度：增加结构相似度权重，减少标签相似度权重
    weighted_similarity = 0.5 * structure_similarity + 0.5 * label_similarity
    
    print(f"Label similarity: {label_similarity:.4f}")
    print(f"Final weighted similarity: {weighted_similarity:.4f}")
    
    return weighted_similarity

def main():
    # 加载新数据特征
    with open('feature_similarity_weighting/new_data_ptr/new_data_features.json', 'r', encoding='utf-8') as f:
        new_data_features = json.load(f)
    
    # 加载历史数据特征
    with open('output/dataset_features.json', 'r', encoding='utf-8') as f:
        historical_features = json.load(f)
    
    new_data_file = "new_rdata_la02_ptr.fjs"
    
    # 测试与一个历史文件的相似度
    test_hist_file = "Hurink/rdata/la02.fjs"
    
    print("测试析取图相似度计算:")
    print("=" * 50)
    
    try:
        similarity = calculate_disjunctive_graph_similarity(
            new_data_features[new_data_file]["disjunctive_graphs_features"],
            historical_features[test_hist_file]["disjunctive_graphs_features"]
        )
        print(f"\n最终相似度: {similarity}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 