#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于帕累托前沿的初始化策略推荐系统
基于多特征融合相似度计算和标记数据集进行两阶段推荐
第一阶段：多特征相似度搜索
第二阶段：帕累托前沿优化

# 基本使用
python initialization_strategy_recommender.py new_data.fjs

# 自定义参数
python initialization_strategy_recommender.py new_data.fjs --top-k-similar 3 --top-k-strategies 2

# 自定义输出目录
python initialization_strategy_recommender.py new_data.fjs --output-dir my_results
"""

import os
import json
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib
# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.rcParams['font.size'] = 10  # 设置默认字体大小
from datetime import datetime
import time
import logging

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


class InitializationStrategyRecommender:
    """基于帕累托前沿的初始化策略推荐系统"""
    
    def __init__(self, labeled_dataset_path, log_file=None):
        """
        初始化推荐系统
        
        Args:
            labeled_dataset_path: 标记数据集路径
            log_file: 日志文件路径，如果为None则不保存日志
        """
        self.labeled_dataset_path = labeled_dataset_path
        self.labeled_data = {}
        self.normalized_features = {}
        self.max_basic_distance = 0
        self.max_processing_distance = 0
        
        # 设置日志
        if log_file:
            self.setup_logging(log_file)
        else:
            self.logger = None
        
        # 加载标记数据集
        self.load_labeled_dataset()
        
        # 标准化特征
        self.normalize_all_features()
    
    def setup_logging(self, log_file):
        """设置日志记录"""
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化策略推荐系统启动")
    
    def log_info(self, message):
        """记录信息日志"""
        if self.logger:
            self.logger.info(message)
        print(message)
    
    def log_debug(self, message):
        """记录调试日志"""
        if self.logger:
            self.logger.debug(message)
    
    def log_warning(self, message):
        """记录警告日志"""
        if self.logger:
            self.logger.warning(message)
        print(f"警告: {message}")
    
    def log_error(self, message):
        """记录错误日志"""
        if self.logger:
            self.logger.error(message)
        print(f"错误: {message}")
    
    def load_labeled_dataset(self):
        """加载标记数据集"""
        self.log_info("正在加载标记数据集...")
        try:
            with open(self.labeled_dataset_path, 'r', encoding='utf-8') as f:
                self.labeled_data = json.load(f)
            self.log_info(f"成功加载 {len(self.labeled_data)} 条标记数据")
        except Exception as e:
            self.log_error(f"加载标记数据集失败: {e}")
            raise
    
    def normalize_features(self, features_dict):
        """
        对特征进行标准化处理
        与calculate_weighted_similarity.py中的normalize_features函数完全一致
        """
        self.log_debug("开始特征标准化处理...")
        
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
        
        self.log_debug(f"基础特征均值: {basic_means}")
        self.log_debug(f"基础特征标准差: {basic_stds}")
        self.log_debug(f"加工时间特征均值: {processing_means}")
        self.log_debug(f"加工时间特征标准差: {processing_stds}")
        
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
        
        self.log_debug("特征标准化处理完成")
        return normalized_features
    
    def normalize_all_features(self):
        """标准化所有特征"""
        self.log_info("正在标准化特征...")
        
        # 提取特征数据
        features_dict = {}
        for fjs_path, data in self.labeled_data.items():
            if 'features' in data:
                features_dict[fjs_path] = data['features']
        
        # 标准化特征
        self.normalized_features = self.normalize_features(features_dict)
        self.log_info(f"特征标准化完成，共处理 {len(self.normalized_features)} 个样本")
    
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
    
    def calculate_similarity(self, new_data_normalized, historical_fjs_path, max_basic_distance, max_processing_distance, new_data_features=None):
        """
        计算新数据与历史数据的综合相似度
        与calculate_weighted_similarity.py中的计算逻辑完全一致
        """
        # 获取历史数据的标准化特征和原始特征
        hist_normalized = self.normalized_features[historical_fjs_path]
        hist_features = self.labeled_data[historical_fjs_path]['features']
        
        # 1. 计算基础特征相似度
        basic_distance = self.calculate_euclidean_distance(
            new_data_normalized["basic_features"],
            hist_normalized["basic_features"]
        )
        basic_similarity = self.normalize_distance(basic_distance, max_basic_distance)
        
        # 2. 计算加工时间特征相似度
        processing_distance = self.calculate_euclidean_distance(
            new_data_normalized["processing_time_features"],
            hist_normalized["processing_time_features"]
        )
        processing_similarity = self.normalize_distance(processing_distance, max_processing_distance)
        
        # 3. 计算KDE相似度（JS散度）
        # 使用原始特征中的KDE数据
        if new_data_features and "kde_features" in new_data_features and "kde_features" in hist_features:
            kde_similarity = 1 - self.calculate_js_divergence(
                new_data_features["kde_features"]["density"],
                hist_features["kde_features"]["density"]
            )
        else:
            kde_similarity = 0.5  # 默认值
        
        # 4. 计算析取图相似度
        # 使用原始特征中的析取图数据
        if new_data_features and "disjunctive_graphs_features" in new_data_features and "disjunctive_graphs_features" in hist_features:
            disjunctive_similarity = self.calculate_disjunctive_graph_similarity(
                new_data_features["disjunctive_graphs_features"],
                hist_features["disjunctive_graphs_features"]
            )
        else:
            disjunctive_similarity = 0.5  # 默认值
        
        # 5. 计算综合加权相似度（与calculate_weighted_similarity.py中的权重完全一致）
        weighted_similarity = (
            0.3 * basic_similarity +
            0.25 * processing_similarity +
            0.2 * kde_similarity +
            0.25 * disjunctive_similarity
        )
        
        self.log_debug(f"相似度计算 - {historical_fjs_path}: 基础={basic_similarity:.4f}, 加工时间={processing_similarity:.4f}, KDE={kde_similarity:.4f}, 析取图={disjunctive_similarity:.4f}, 加权={weighted_similarity:.4f}")
        
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
        self.log_info(f"=== 阶段一：多特征相似度检索 ===")
        self.log_info(f"正在计算与 {len(self.labeled_data)} 个历史样本的相似度...")
        
        # 合并所有特征用于标准化（包括新数据）
        all_features = {}
        for fjs_path, data in self.labeled_data.items():
            if 'features' in data:
                all_features[fjs_path] = data['features']
        
        # 添加新数据特征
        all_features["new_data"] = new_data_features
        
        # 标准化所有特征（包括新数据）
        normalized_all_features = self.normalize_features(all_features)
        new_data_normalized = normalized_all_features["new_data"]
        
        # 计算最大距离（用于归一化）
        max_basic_distance = 0
        max_processing_distance = 0
        
        # 第一遍：计算最大距离
        for fjs_path in self.normalized_features.keys():
            # 基础特征距离
            basic_distance = self.calculate_euclidean_distance(
                new_data_normalized["basic_features"],
                self.normalized_features[fjs_path]["basic_features"]
            )
            max_basic_distance = max(max_basic_distance, basic_distance)
            
            # 加工时间特征距离
            processing_distance = self.calculate_euclidean_distance(
                new_data_normalized["processing_time_features"],
                self.normalized_features[fjs_path]["processing_time_features"]
            )
            max_processing_distance = max(max_processing_distance, processing_distance)
        
        self.log_info(f"最大距离计算完成: 基础特征={max_basic_distance:.4f}, 加工时间={max_processing_distance:.4f}")
        
        # 计算所有历史样本的相似度
        similarity_results = {}
        
        for fjs_path in self.normalized_features.keys():
            similarity_details = self.calculate_similarity(
                new_data_normalized, fjs_path, max_basic_distance, max_processing_distance, new_data_features
            )
            similarity_results[fjs_path] = similarity_details
        
        # 按综合加权相似度排序
        sorted_results = sorted(
            similarity_results.items(),
            key=lambda x: x[1]["weighted_similarity"],
            reverse=True
        )
        
        # 获取Top K结果
        top_k_results = sorted_results[:top_k]
        
        self.log_info(f"Top {top_k} 最相似的历史样本:")
        for i, (fjs_path, details) in enumerate(top_k_results, 1):
            self.log_info(f"{i}. {fjs_path}")
            self.log_info(f"   综合加权相似度: {details['weighted_similarity']:.4f}")
            self.log_info(f"   基础特征相似度: {details['basic_similarity']:.4f}")
            self.log_info(f"   加工时间特征相似度: {details['processing_similarity']:.4f}")
            self.log_info(f"   KDE相似度: {details['kde_similarity']:.4f}")
            self.log_info(f"   析取图相似度: {details['disjunctive_similarity']:.4f}")
        
        return [(fjs_path, details["weighted_similarity"], details) for fjs_path, details in top_k_results]
    
    def calculate_performance_metrics(self, strategy_data):
        """
        计算四维性能指标（不包括执行时间）
        
        Args:
            strategy_data: 策略性能数据
            
        Returns:
            dict: 四维性能指标
        """
        mean_makespan = strategy_data.get('mean', 0)
        std_makespan = strategy_data.get('std', 0)
        avg_convergence_gen = strategy_data.get('avg_convergence_generation', 0)
        convergence_std = strategy_data.get('convergence_generation_std', 0)
        
        # 1. Makespan精度 (越小越好，转换为0-1评分)
        makespan_score = 1.0 / (1.0 + mean_makespan / 1000.0)
        
        # 2. 求解稳定性 (标准差越小越好，转换为0-1评分)
        stability_score = 1.0 / (1.0 + std_makespan / 10.0)
        
        # 3. 收敛效率 (收敛代数越少越好，转换为0-1评分)
        max_iterations = 100
        convergence_speed_score = 1.0 - (avg_convergence_gen / max_iterations)
        convergence_speed_score = max(0.0, min(1.0, convergence_speed_score))
        
        # 4. 收敛稳定性 (收敛标准差越小越好，转换为0-1评分)
        convergence_stability_score = 1.0 / (1.0 + convergence_std / 10.0)
        
        return {
            'makespan_accuracy': makespan_score,
            'solution_stability': stability_score,
            'convergence_efficiency': convergence_speed_score,
            'convergence_stability': convergence_stability_score
        }
    
    def is_dominated(self, point1, point2):
        """
        判断point1是否被point2支配
        
        Args:
            point1, point2: 四维性能指标向量
            
        Returns:
            bool: point1是否被point2支配
        """
        # 所有目标都是最大化（值越大越好）
        return all(p2 >= p1 for p1, p2 in zip(point1, point2)) and any(p2 > p1 for p1, p2 in zip(point1, point2))
    
    def find_pareto_frontier(self, candidates):
        """
        找到帕累托前沿解
        
        Args:
            candidates: 候选策略列表，每个元素包含策略名和性能指标
            
        Returns:
            list: 帕累托前沿解列表
        """
        pareto_frontier = []
        
        for i, candidate in enumerate(candidates):
            dominated = False
            point1 = list(candidate['performance_metrics'].values())
            
            for j, other in enumerate(candidates):
                if i == j:
                    continue
                    
                point2 = list(other['performance_metrics'].values())
                
                if self.is_dominated(point1, point2):
                    dominated = True
                    break
            
            if not dominated:
                pareto_frontier.append(candidate)
        
        return pareto_frontier
    
    def calculate_weighted_score(self, performance_metrics, weights=None):
        """
        计算加权评分（四维性能指标）
        
        Args:
            performance_metrics: 性能指标字典
            weights: 权重字典，如果为None则使用默认权重
            
        Returns:
            float: 加权评分
        """
        if weights is None:
            weights = {
                'makespan_accuracy': 0.4,
                'solution_stability': 0.2,
                'convergence_efficiency': 0.25,
                'convergence_stability': 0.15
            }
        
        weighted_score = 0.0
        for metric, value in performance_metrics.items():
            weighted_score += value * weights[metric]
        
        return weighted_score
    
    def stage_two_strategy_recommendation(self, candidate_samples, top_k=3):
        """
        阶段二：基于帕累托前沿的策略推荐
        
        Args:
            candidate_samples: 候选样本列表，每个元素为 (fjs_path, similarity_score, similarity_details)
            top_k: 推荐策略数量
            
        Returns:
            tuple: (推荐策略列表，每个元素为 (strategy_name, weighted_score), 所有候选策略, 帕累托前沿解)
        """
        self.log_info(f"\n=== 阶段二：帕累托前沿策略推荐 ===")
        self.log_info(f"候选样本数量: {len(candidate_samples)}")
        
        # 收集候选样本的策略性能数据
        all_strategies = []
        
        for fjs_path, similarity_score, _ in candidate_samples:
            if fjs_path in self.labeled_data and 'performance_data' in self.labeled_data[fjs_path]:
                performance_data = self.labeled_data[fjs_path]['performance_data']
                
                # 检查是否有initialization_methods字段
                if 'initialization_methods' in performance_data:
                    init_methods = performance_data['initialization_methods']
                    
                    # 只考虑heuristic、mixed、random三种初始化方法
                    for strategy_name, strategy_data in init_methods.items():
                        if strategy_name not in ["heuristic", "mixed", "random"]:
                            continue  # 跳过非三种初始化方法
                        
                        # 计算四维性能指标
                        performance_metrics = self.calculate_performance_metrics(strategy_data)
                        
                        all_strategies.append({
                            'strategy_name': strategy_name,
                            'fjs_path': fjs_path,
                            'similarity_score': similarity_score,
                            'performance_metrics': performance_metrics,
                            'raw_metrics': {
                                'mean_makespan': strategy_data.get('mean', 0),
                                'std_makespan': strategy_data.get('std', 0),
                                'avg_convergence_gen': strategy_data.get('avg_convergence_generation', 0),
                                'convergence_std': strategy_data.get('convergence_generation_std', 0)
                            }
                        })
        
        self.log_info(f"收集到 {len(all_strategies)} 个候选策略")
        
        if len(all_strategies) == 0:
            self.log_warning("未找到有效的候选策略")
            return [], [], []
        
        # 为所有候选策略计算加权评分
        for strategy in all_strategies:
            strategy['weighted_score'] = self.calculate_weighted_score(strategy['performance_metrics'])
        
        # 帕累托筛选
        pareto_frontier = self.find_pareto_frontier(all_strategies)
        
        self.log_info(f"帕累托前沿解数量: {len(pareto_frontier)}")
        
        # 最终方案推荐（按加权评分排序）
        pareto_frontier.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # 获取Top K推荐策略
        top_k_strategies = pareto_frontier[:top_k]
        
        self.log_info(f"\nTop {top_k} 帕累托推荐策略:")
        for i, strategy in enumerate(top_k_strategies, 1):
            self.log_info(f"{i}. {strategy['strategy_name']}")
            self.log_info(f"   加权评分: {strategy['weighted_score']:.4f}")
            self.log_info(f"   四维性能指标:")
            for metric, value in strategy['performance_metrics'].items():
                self.log_info(f"     {metric}: {value:.4f}")
        
        return (
            [(strategy['strategy_name'], strategy['weighted_score']) for strategy in top_k_strategies],
            all_strategies,
            pareto_frontier
        )
    
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
        self.log_info("开始执行两阶段帕累托推荐流程...")
        start_time = time.time()
        
        # 阶段一：多特征相似度检索
        candidate_samples = self.stage_one_similarity_search(new_data_features, top_k_similar)
        
        # 阶段二：帕累托前沿策略推荐
        recommended_strategies, all_candidates, pareto_frontier = self.stage_two_strategy_recommendation(candidate_samples, top_k_strategies)
        
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
            },
            'all_candidates': all_candidates,
            'pareto_frontier': pareto_frontier
        }
        
        self.log_info(f"\n推荐流程完成，总耗时: {total_time:.2f}秒")
        
        return result
    
    def save_results(self, results, output_file):
        """保存推荐结果到文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            self.log_info(f"推荐结果已保存到: {output_file}")
        except Exception as e:
            self.log_error(f"保存结果失败: {e}")
    
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
            
            self.log_info(f"可视化结果已保存到: {output_dir}")
            
        except Exception as e:
            self.log_error(f"可视化失败: {e}")


def main():
    """主函数：推荐系统命令行接口"""
    import sys
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='初始化策略推荐系统')
    parser.add_argument('fjs_file', help='输入的FJS文件路径')
    parser.add_argument('--top-k-similar', type=int, default=5, help='阶段一返回的最相似样本数量 (默认: 5)')
    parser.add_argument('--top-k-strategies', type=int, default=3, help='阶段二推荐的策略数量 (默认: 3)')
    parser.add_argument('--output-dir', default='result/recommender_output', help='输出目录 (默认: result/recommender_output)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.fjs_file):
        print(f"错误: 输入文件不存在: {args.fjs_file}")
        return
    
    # 获取文件名（不含扩展名）用于创建子目录
    fjs_basename = os.path.splitext(os.path.basename(args.fjs_file))[0]
    
    # 创建输出目录结构
    base_output_dir = args.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = f"{fjs_basename}_{timestamp}"
    full_output_dir = os.path.join(base_output_dir, output_subdir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # 确保基础输出目录存在
    os.makedirs(base_output_dir, exist_ok=True)
    
    print("=" * 80)
    print("初始化策略推荐系统")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输入文件: {args.fjs_file}")
    print(f"输出目录: {full_output_dir}")
    print(f"Top-K相似样本: {args.top_k_similar}")
    print(f"Top-K推荐策略: {args.top_k_strategies}")
    print("=" * 80)
    
    try:
        # 初始化推荐系统（带日志）
        labeled_dataset_path = "labeled_dataset/labeled_fjs_dataset.json"
        log_file = os.path.join(full_output_dir, f"recommendation_log.log")
        recommender = InitializationStrategyRecommender(labeled_dataset_path, log_file)
        
        # 提取新数据特征
        from extract_new_data_features import extract_new_data_features
        
        recommender.log_info(f"开始提取新数据特征: {args.fjs_file}")
        new_data_features = extract_new_data_features(args.fjs_file)
        
        if new_data_features is None:
            recommender.log_error("新数据特征提取失败")
            return
        
        recommender.log_info(f"新数据特征提取完成")
        
        # 执行推荐
        results = recommender.recommend(new_data_features, 
                                      top_k_similar=args.top_k_similar, 
                                      top_k_strategies=args.top_k_strategies)
        
        # 保存推荐结果
        output_file = os.path.join(full_output_dir, "recommendation_results.json")
        recommender.save_results(results, output_file)
        
        # 可视化结果
        visualization_dir = os.path.join(full_output_dir, "visualization")
        recommender.visualize_recommendation_results(results, visualization_dir)
        
        # 输出推荐结果摘要
        print("\n" + "=" * 80)
        print("推荐结果摘要")
        print("=" * 80)
        
        # 阶段一结果
        print("\n阶段一：最相似的历史数据")
        stage_one_data = results['stage_one_results']['candidate_samples']
        for i, data in enumerate(stage_one_data, 1):
            print(f"{i}. {data['fjs_path']}")
            print(f"   综合相似度: {data['similarity_score']:.4f}")
            print(f"   基础特征相似度: {data['similarity_details']['basic_similarity']:.4f}")
            print(f"   加工时间相似度: {data['similarity_details']['processing_similarity']:.4f}")
            print(f"   KDE相似度: {data['similarity_details']['kde_similarity']:.4f}")
            print(f"   析取图相似度: {data['similarity_details']['disjunctive_similarity']:.4f}")
        
        # 阶段二结果
        print("\n阶段二：推荐策略")
        stage_two_data = results['stage_two_results']['recommended_strategies']
        # 重新计算详细评分
        for i, data in enumerate(stage_two_data, 1):
            print(f"{i}. {data['strategy_name']}")
            print(f"   加权性能评分: {data['weighted_score']:.4f}")
            # 重新查找候选样本的详细评分
            # 由于results中没有detailed_scores，需重新查找
            # 先找到候选样本
            candidate_samples = results['stage_one_results']['candidate_samples']
            # 需要重新加载labeled_dataset
            import json
            with open(labeled_dataset_path, 'r', encoding='utf-8') as f:
                labeled_data = json.load(f)
            # 收集所有候选样本的该策略性能
            makespan_scores = []
            convergence_speed_scores = []
            stability_scores = []
            convergence_stability_scores = []
            for sample in candidate_samples:
                fjs_path = sample['fjs_path']
                if fjs_path in labeled_data and 'performance_data' in labeled_data[fjs_path]:
                    perf = labeled_data[fjs_path]['performance_data']
                    if 'initialization_methods' in perf and data['strategy_name'] in perf['initialization_methods']:
                        strat = perf['initialization_methods'][data['strategy_name']]
                        mean_makespan = strat.get('mean', 0)
                        std_makespan = strat.get('std', 0)
                        avg_convergence_gen = strat.get('avg_convergence_generation', 0)
                        convergence_std = strat.get('convergence_generation_std', 0)
                        makespan_score = 1.0 / (1.0 + mean_makespan / 1000.0)
                        max_iterations = 100
                        convergence_speed_score = 1.0 - (avg_convergence_gen / max_iterations)
                        convergence_speed_score = max(0.0, min(1.0, convergence_speed_score))
                        stability_score = 1.0 / (1.0 + std_makespan / 10.0)
                        convergence_stability_score = 1.0 / (1.0 + convergence_std / 10.0)
                        makespan_scores.append(makespan_score)
                        convergence_speed_scores.append(convergence_speed_score)
                        stability_scores.append(stability_score)
                        convergence_stability_scores.append(convergence_stability_score)
            # 输出均值
            if makespan_scores:
                print(f"   详细评分:")
                print(f"     Makespan评分: {sum(makespan_scores)/len(makespan_scores):.4f}")
                print(f"     收敛速度评分: {sum(convergence_speed_scores)/len(convergence_speed_scores):.4f}")
                print(f"     稳定性评分: {sum(stability_scores)/len(stability_scores):.4f}")
                print(f"     收敛稳定性评分: {sum(convergence_stability_scores)/len(convergence_stability_scores):.4f}")
            else:
                print(f"   详细评分: 无法获取")
        
        print("\n" + "=" * 80)
        print("所有结果已保存")
        print(f"结果目录: {full_output_dir}")
        print(f"日志文件: {log_file}")
        print(f"推荐结果: {output_file}")
        print(f"可视化结果: {visualization_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 