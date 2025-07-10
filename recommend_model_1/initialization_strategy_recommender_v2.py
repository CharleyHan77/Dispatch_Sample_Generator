#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
初始化策略推荐系统 V2
基于标记数据集和calculate_weighted_similarity.py的相似度计算逻辑进行两阶段推荐
"""

import os
import json
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from datetime import datetime
import time

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
import sys
sys.path.append(project_root)

# 导入析取图相关函数
from comparison_disjunctive_graphs.compare_graphs_wl import (
    create_disjunctive_graph_with_attributes,
    init_node_labels,
    add_edge_attributes,
    wl_step
)
from initial_validation.utils import parser


class InitializationStrategyRecommenderV2:
    """初始化策略推荐系统 V2"""
    
    def __init__(self, labeled_dataset_path):
        """
        初始化推荐系统
        
        Args:
            labeled_dataset_path: 标记数据集路径
        """
        self.labeled_dataset_path = labeled_dataset_path
        self.labeled_data = {}
        self.normalized_features = {}
        self.max_basic_distance = 0
        self.max_processing_distance = 0
        
        # 加载标记数据集
        self.load_labeled_dataset()
        
        # 标准化特征
        self.normalize_all_features()
        
        # 计算最大距离（用于归一化）
        self.calculate_max_distances()
    
    def load_labeled_dataset(self):
        """加载标记数据集"""
        print("正在加载标记数据集...")
        try:
            with open(self.labeled_dataset_path, 'r', encoding='utf-8') as f:
                self.labeled_data = json.load(f)
            print(f"成功加载 {len(self.labeled_data)} 条标记数据")
        except Exception as e:
            print(f"加载标记数据集失败: {e}")
            raise
    
    def normalize_features(self, features_dict):
        """
        对特征进行标准化处理
        与calculate_weighted_similarity.py中的normalize_features函数完全一致
        """
        # 提取所有特征值
        basic_values = []
        processing_values = []
        
        for file_features in features_dict.values():
            # 基础特征
            basic_features = file_features['basic_features']
            basic_values.append([
                basic_features['num_jobs'],
                basic_features['num_machines'],
                basic_features['total_operations'],
                basic_features['avg_available_machines'],
                basic_features['std_available_machines']
            ])
            
            # 加工时间特征
            processing_features = file_features['processing_time_features']
            processing_values.append([
                processing_features['processing_time_mean'],
                processing_features['processing_time_std'],
                processing_features['processing_time_min'],
                processing_features['processing_time_max'],
                processing_features['machine_time_variance']
            ])
        
        # 转换为numpy数组
        basic_values = np.array(basic_values)
        processing_values = np.array(processing_values)
        
        # 计算每个特征的均值和标准差
        basic_means = np.mean(basic_values, axis=0)
        basic_stds = np.std(basic_values, axis=0)
        processing_means = np.mean(processing_values, axis=0)
        processing_stds = np.std(processing_values, axis=0)
        
        # 添加小的常数避免0标准差
        epsilon = 1e-10
        basic_stds = np.where(basic_stds == 0, epsilon, basic_stds)
        processing_stds = np.where(processing_stds == 0, epsilon, processing_stds)
        
        # 标准化所有特征
        normalized_features = {}
        for file_name, file_features in features_dict.items():
            normalized_features[file_name] = {
                'basic_features': {},
                'processing_time_features': {}
            }
            
            # 标准化基础特征
            basic_features = file_features['basic_features']
            normalized_features[file_name]['basic_features'] = {
                'num_jobs': (basic_features['num_jobs'] - basic_means[0]) / basic_stds[0],
                'num_machines': (basic_features['num_machines'] - basic_means[1]) / basic_stds[1],
                'total_operations': (basic_features['total_operations'] - basic_means[2]) / basic_stds[2],
                'avg_available_machines': (basic_features['avg_available_machines'] - basic_means[3]) / basic_stds[3],
                'std_available_machines': (basic_features['std_available_machines'] - basic_means[4]) / basic_stds[4]
            }
            
            # 标准化加工时间特征
            processing_features = file_features['processing_time_features']
            normalized_features[file_name]['processing_time_features'] = {
                'processing_time_mean': (processing_features['processing_time_mean'] - processing_means[0]) / processing_stds[0],
                'processing_time_std': (processing_features['processing_time_std'] - processing_means[1]) / processing_stds[1],
                'processing_time_min': (processing_features['processing_time_min'] - processing_means[2]) / processing_stds[2],
                'processing_time_max': (processing_features['processing_time_max'] - processing_means[3]) / processing_stds[3],
                'machine_time_variance': (processing_features['machine_time_variance'] - processing_means[4]) / processing_stds[4]
            }
        
        return normalized_features
    
    def normalize_all_features(self):
        """标准化所有特征"""
        print("正在标准化特征...")
        
        # 提取特征数据
        features_dict = {}
        for fjs_path, data in self.labeled_data.items():
            if 'features' in data:
                features_dict[fjs_path] = data['features']
        
        # 标准化特征
        self.normalized_features = self.normalize_features(features_dict)
        print(f"特征标准化完成，共处理 {len(self.normalized_features)} 个样本")
    
    def calculate_max_distances(self):
        """计算最大距离（用于归一化）"""
        print("正在计算最大距离...")
        
        # 计算所有样本间的最大距离
        for fjs_path1 in self.normalized_features.keys():
            for fjs_path2 in self.normalized_features.keys():
                if fjs_path1 != fjs_path2:
                    # 基础特征距离
                    basic_distance = self.calculate_euclidean_distance(
                        self.normalized_features[fjs_path1]["basic_features"],
                        self.normalized_features[fjs_path2]["basic_features"]
                    )
                    self.max_basic_distance = max(self.max_basic_distance, basic_distance)
                    
                    # 加工时间特征距离
                    processing_distance = self.calculate_euclidean_distance(
                        self.normalized_features[fjs_path1]["processing_time_features"],
                        self.normalized_features[fjs_path2]["processing_time_features"]
                    )
                    self.max_processing_distance = max(self.max_processing_distance, processing_distance)
        
        print(f"最大距离计算完成: 基础特征={self.max_basic_distance:.4f}, 加工时间={self.max_processing_distance:.4f}")
    
    def calculate_euclidean_distance(self, features1, features2):
        """计算两个特征向量之间的欧氏距离"""
        vec1 = np.array(list(features1.values()))
        vec2 = np.array(list(features2.values()))
        return np.sqrt(np.sum((vec1 - vec2) ** 2))
    
    def calculate_js_divergence(self, p, q):
        """计算两个概率分布之间的JS散度"""
        # 确保概率分布和为1
        p = np.array(p)
        q = np.array(q)
        
        # 添加小的常数避免0值
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # 计算平均分布
        m = 0.5 * (p + q)
        
        # 计算JS散度
        js_div = 0.5 * (entropy(p, m) + entropy(q, m))
        
        # 处理可能的无穷大值
        if np.isinf(js_div):
            return 1.0  # 返回最大距离
        
        return js_div
    
    def normalize_distance(self, distance, max_distance):
        """将距离归一化到[0,1]区间，并转换为相似度（1-归一化距离）"""
        if max_distance <= 0:
            return 1.0
        return 1 - (distance / max_distance)
    
    def calculate_disjunctive_graph_similarity(self, graph_info1, graph_info2):
        """
        计算两个析取图的相似度，基于图结构特征和WL标签频率
        与calculate_weighted_similarity.py中的函数完全一致
        """
        # 获取图的基本结构特征
        nodes1 = graph_info1['nodes_count']
        nodes2 = graph_info2['nodes_count']
        edges1 = graph_info1['edges_count']
        edges2 = graph_info2['edges_count']
        
        # 计算结构相似度（基于节点数和边数的相似性）
        nodes_similarity = 1 - abs(nodes1 - nodes2) / max(nodes1, nodes2)
        edges_similarity = 1 - abs(edges1 - edges2) / max(edges1, edges2)
        structure_similarity = (nodes_similarity + edges_similarity) / 2
        
        # 获取实线和虚线标签频率
        solid_freq1 = graph_info1['solid_frequency']
        solid_freq2 = graph_info2['solid_frequency']
        dashed_freq1 = graph_info1['dashed_frequency']
        dashed_freq2 = graph_info2['dashed_frequency']
        
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
        
        # 综合相似度计算
        # 调整权重分配，增加结构相似度的影响
        solid_similarity = 0.3 * solid_jaccard + 0.7 * solid_cosine
        dashed_similarity = 0.3 * dashed_jaccard + 0.7 * dashed_cosine
        
        # 加权组合相似度（实线权重0.6，虚线权重0.4）
        label_similarity = 0.6 * solid_similarity + 0.4 * dashed_similarity
        
        # 最终相似度：增加结构相似度权重，减少标签相似度权重
        weighted_similarity = 0.5 * structure_similarity + 0.5 * label_similarity
        
        return weighted_similarity
    
    def calculate_similarity(self, new_data_features, historical_fjs_path):
        """
        计算新数据与历史数据的综合相似度
        与calculate_weighted_similarity.py中的计算逻辑完全一致
        """
        # 获取历史数据的标准化特征
        hist_normalized = self.normalized_features[historical_fjs_path]
        hist_features = self.labeled_data[historical_fjs_path]['features']
        
        # 1. 计算基础特征相似度
        basic_distance = self.calculate_euclidean_distance(
            new_data_features["basic_features"],
            hist_normalized["basic_features"]
        )
        basic_similarity = self.normalize_distance(basic_distance, self.max_basic_distance)
        
        # 2. 计算加工时间特征相似度
        processing_distance = self.calculate_euclidean_distance(
            new_data_features["processing_time_features"],
            hist_normalized["processing_time_features"]
        )
        processing_similarity = self.normalize_distance(processing_distance, self.max_processing_distance)
        
        # 3. 计算KDE相似度（JS散度）
        kde_similarity = 1 - self.calculate_js_divergence(
            new_data_features["kde_features"]["density"],
            hist_features["kde_features"]["density"]
        )
        
        # 4. 计算析取图相似度
        disjunctive_similarity = self.calculate_disjunctive_graph_similarity(
            new_data_features["disjunctive_graphs_features"],
            hist_features["disjunctive_graphs_features"]
        )
        
        # 5. 计算综合加权相似度（与calculate_weighted_similarity.py中的权重完全一致）
        weighted_similarity = (
            0.3 * basic_similarity +
            0.25 * processing_similarity +
            0.2 * kde_similarity +
            0.25 * disjunctive_similarity
        )
        
        return {
            "basic_similarity": basic_similarity,
            "processing_similarity": processing_similarity,
            "kde_similarity": kde_similarity,
            "disjunctive_similarity": disjunctive_similarity,
            "weighted_similarity": weighted_similarity
        }
    
    def stage_one_similarity_search(self, new_data_features, top_k=5):
        """
        阶段一：多特征相似度检索
        基于四种特征融合加权计算，返回Top K最相似的历史样本
        与calculate_weighted_similarity.py中的逻辑完全一致
        
        Args:
            new_data_features: 新数据的特征
            top_k: 返回的最相似样本数量
            
        Returns:
            list: 候选样本列表，每个元素为 (fjs_path, similarity_score, similarity_details)
        """
        print(f"=== 阶段一：多特征相似度检索 ===")
        print(f"正在计算与 {len(self.labeled_data)} 个历史样本的相似度...")
        
        # 标准化新数据特征
        new_data_normalized = self.normalize_features({"new_data": new_data_features})
        new_data_normalized = new_data_normalized["new_data"]
        
        # 计算所有历史样本的相似度
        similarity_results = {}
        
        for fjs_path in self.labeled_data.keys():
            similarity_details = self.calculate_similarity(new_data_normalized, fjs_path)
            similarity_results[fjs_path] = similarity_details
        
        # 按综合加权相似度排序
        sorted_results = sorted(
            similarity_results.items(),
            key=lambda x: x[1]["weighted_similarity"],
            reverse=True
        )
        
        # 获取Top K结果
        top_k_results = sorted_results[:top_k]
        
        print(f"Top {top_k} 最相似的历史样本:")
        for i, (fjs_path, details) in enumerate(top_k_results, 1):
            print(f"{i}. {fjs_path}")
            print(f"   综合加权相似度: {details['weighted_similarity']:.4f}")
            print(f"   基础特征相似度: {details['basic_similarity']:.4f}")
            print(f"   加工时间特征相似度: {details['processing_similarity']:.4f}")
            print(f"   KDE相似度: {details['kde_similarity']:.4f}")
            print(f"   析取图相似度: {details['disjunctive_similarity']:.4f}")
        
        return [(fjs_path, details["weighted_similarity"], details) for fjs_path, details in top_k_results]
    
    def stage_two_strategy_recommendation(self, candidate_samples, top_k=3):
        """
        阶段二：基于相似度加权的策略推荐
        
        Args:
            candidate_samples: 候选样本列表，每个元素为 (fjs_path, similarity_score, similarity_details)
            top_k: 推荐策略数量
            
        Returns:
            list: 推荐策略列表，每个元素为 (strategy_name, weighted_score)
        """
        print(f"\n=== 阶段二：策略推荐 ===")
        print(f"候选样本数量: {len(candidate_samples)}")
        
        # 收集候选样本的策略性能数据
        strategy_performance = {}
        
        for fjs_path, similarity_score, _ in candidate_samples:
            if fjs_path in self.labeled_data and 'performance_data' in self.labeled_data[fjs_path]:
                performance_data = self.labeled_data[fjs_path]['performance_data']
                
                # 遍历每个初始化策略的性能数据
                for strategy_name, strategy_data in performance_data.items():
                    if strategy_name not in strategy_performance:
                        strategy_performance[strategy_name] = []
                    
                    # 计算策略的综合性能评分
                    # 基于求解精度和收敛效率
                    precision_score = strategy_data.get('avg_precision', 0)
                    convergence_score = strategy_data.get('avg_convergence_rate', 0)
                    
                    # 综合性能评分（精度权重0.7，收敛权重0.3）
                    performance_score = 0.7 * precision_score + 0.3 * convergence_score
                    
                    # 存储策略性能数据
                    strategy_performance[strategy_name].append({
                        'fjs_path': fjs_path,
                        'similarity_score': similarity_score,
                        'performance_score': performance_score,
                        'precision': precision_score,
                        'convergence': convergence_score
                    })
        
        print(f"找到 {len(strategy_performance)} 种初始化策略")
        
        # 计算每种策略的加权平均性能评分
        strategy_scores = {}
        
        for strategy_name, performances in strategy_performance.items():
            if len(performances) > 0:
                # 计算相似度加权的平均性能评分
                total_weighted_score = 0
                total_weight = 0
                
                for perf in performances:
                    weight = perf['similarity_score']
                    total_weighted_score += weight * perf['performance_score']
                    total_weight += weight
                
                if total_weight > 0:
                    weighted_avg_score = total_weighted_score / total_weight
                    strategy_scores[strategy_name] = {
                        'weighted_score': weighted_avg_score,
                        'sample_count': len(performances),
                        'performances': performances
                    }
        
        # 按加权评分排序
        sorted_strategies = sorted(
            strategy_scores.items(),
            key=lambda x: x[1]['weighted_score'],
            reverse=True
        )
        
        # 获取Top K推荐策略
        top_k_strategies = sorted_strategies[:top_k]
        
        print(f"\nTop {top_k} 推荐策略:")
        for i, (strategy_name, score_info) in enumerate(top_k_strategies, 1):
            print(f"{i}. {strategy_name}")
            print(f"   加权性能评分: {score_info['weighted_score']:.4f}")
            print(f"   参考样本数量: {score_info['sample_count']}")
        
        return [(strategy_name, score_info['weighted_score']) for strategy_name, score_info in top_k_strategies]
    
    def recommend(self, new_data_features, top_k_similar=5, top_k_strategies=3):
        """
        执行完整的两阶段推荐流程
        
        Args:
            new_data_features: 新数据的特征
            top_k_similar: 阶段一返回的最相似样本数量
            top_k_strategies: 阶段二推荐的策略数量
            
        Returns:
            dict: 推荐结果
        """
        print("开始执行两阶段推荐流程...")
        start_time = time.time()
        
        # 阶段一：多特征相似度检索
        candidate_samples = self.stage_one_similarity_search(new_data_features, top_k_similar)
        
        # 阶段二：策略推荐
        recommended_strategies = self.stage_two_strategy_recommendation(candidate_samples, top_k_strategies)
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 构建推荐结果
        result = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': total_time,
            'stage_one_results': {
                'candidate_samples': [
                    {
                        'fjs_path': fjs_path,
                        'similarity_score': similarity_score,
                        'similarity_details': similarity_details
                    }
                    for fjs_path, similarity_score, similarity_details in candidate_samples
                ]
            },
            'stage_two_results': {
                'recommended_strategies': [
                    {
                        'strategy_name': strategy_name,
                        'weighted_score': weighted_score
                    }
                    for strategy_name, weighted_score in recommended_strategies
                ]
            }
        }
        
        print(f"\n推荐流程完成，总耗时: {total_time:.2f}秒")
        
        return result
    
    def save_results(self, results, output_file):
        """保存推荐结果到文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"推荐结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存结果失败: {e}")
    
    def visualize_recommendation_results(self, results, output_dir):
        """可视化推荐结果"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. 可视化阶段一结果：相似度对比
            stage_one_data = results['stage_one_results']['candidate_samples']
            
            files = [data['fjs_path'] for data in stage_one_data]
            basic_similarities = [data['similarity_details']['basic_similarity'] for data in stage_one_data]
            processing_similarities = [data['similarity_details']['processing_similarity'] for data in stage_one_data]
            kde_similarities = [data['similarity_details']['kde_similarity'] for data in stage_one_data]
            disjunctive_similarities = [data['similarity_details']['disjunctive_similarity'] for data in stage_one_data]
            weighted_similarities = [data['similarity_details']['weighted_similarity'] for data in stage_one_data]
            
            plt.figure(figsize=(15, 7))
            x = np.arange(len(files))
            width = 0.15
            
            plt.bar(x - width*2, basic_similarities, width, label='基础特征相似度', color='#2ecc71')
            plt.bar(x - width*1, processing_similarities, width, label='加工时间特征相似度', color='#3498db')
            plt.bar(x, kde_similarities, width, label='KDE相似度', color='#9b59b6')
            plt.bar(x + width*1, disjunctive_similarities, width, label='析取图相似度', color='#f39c12')
            plt.bar(x + width*2, weighted_similarities, width, label='综合加权相似度', color='#e74c3c')
            
            plt.xlabel('历史数据文件')
            plt.ylabel('相似度')
            plt.title('候选样本的多特征相似度对比')
            plt.xticks(x, files, rotation=45, ha='right')
            plt.legend(loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            similarity_plot_path = os.path.join(output_dir, 'similarity_comparison.png')
            plt.savefig(similarity_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 可视化阶段二结果：策略推荐评分
            stage_two_data = results['stage_two_results']['recommended_strategies']
            
            strategies = [data['strategy_name'] for data in stage_two_data]
            scores = [data['weighted_score'] for data in stage_two_data]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(strategies, scores, color='#e74c3c', alpha=0.7)
            
            # 添加数值标签
            for bar, score in zip(bars, scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.4f}', ha='center', va='bottom')
            
            plt.xlabel('初始化策略')
            plt.ylabel('加权性能评分')
            plt.title('推荐策略的性能评分')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            strategy_plot_path = os.path.join(output_dir, 'strategy_recommendation.png')
            plt.savefig(strategy_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"可视化结果已保存到: {output_dir}")
            
        except Exception as e:
            print(f"可视化失败: {e}")


def main():
    """主函数：测试推荐系统"""
    # 初始化推荐系统
    labeled_dataset_path = "labeled_dataset/labeled_fjs_dataset.json"
    recommender = InitializationStrategyRecommenderV2(labeled_dataset_path)
    
    # 模拟新数据特征（这里使用第一个样本的特征作为示例）
    first_sample = list(recommender.labeled_data.values())[0]
    new_data_features = first_sample['features']
    
    print(f"使用样本特征作为新数据: {list(recommender.labeled_data.keys())[0]}")
    
    # 执行推荐
    results = recommender.recommend(new_data_features, top_k_similar=5, top_k_strategies=3)
    
    # 保存结果
    output_file = "recommendation_results_v2.json"
    recommender.save_results(results, output_file)
    
    # 可视化结果
    output_dir = "recommendation_visualization_v2"
    recommender.visualize_recommendation_results(results, output_dir)


if __name__ == "__main__":
    main() 