#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一阶段初始化策略推荐系统
基于多特征融合相似度计算和性能目标评分进行一阶段综合推荐

# 基本使用
python initialization_strategy_recommender.py new_data.fjs

# 自定义参数
python initialization_strategy_recommender.py new_data.fjs --top-k-strategies 3 --feature-weight 0.4 --performance-weight 0.6

# 自定义输出目录
python initialization_strategy_recommender.py new_data.fjs --output-dir my_results

# 调整权重示例
python initialization_strategy_recommender.py new_data.fjs --feature-weight 0.3 --performance-weight 0.7
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
    """初始化策略推荐系统"""
    
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
    
    def normalize_single_features(self, single_features):
        """
        标准化单个特征数据（用于新数据）
        基于已有的历史数据统计信息进行标准化
        
        Args:
            single_features: 单个样本的特征字典
            
        Returns:
            dict: 标准化后的特征字典
        """
        # 如果还没有标准化历史特征，先进行标准化
        if not self.normalized_features:
            self.normalize_all_features()
        
        # 计算历史数据的统计信息
        basic_values = []
        processing_values = []
        
        for sample_id, data in self.labeled_data.items():
            if 'features' in data:
                file_features = data['features']
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
        
        basic_values = np.array(basic_values)
        processing_values = np.array(processing_values)
        
        # 计算均值和标准差
        basic_means = np.mean(basic_values, axis=0)
        basic_stds = np.std(basic_values, axis=0)
        processing_means = np.mean(processing_values, axis=0)
        processing_stds = np.std(processing_values, axis=0)
        
        # 避免除以零
        basic_stds[basic_stds == 0] = 1
        processing_stds[processing_stds == 0] = 1
        
        # 标准化新数据特征
        basic_features = single_features['basic_features']
        processing_features = single_features['processing_time_features']
        
        normalized_single = {
            'basic_features': {
                'num_jobs': (basic_features['num_jobs'] - basic_means[0]) / basic_stds[0],
                'num_machines': (basic_features['num_machines'] - basic_means[1]) / basic_stds[1],
                'total_operations': (basic_features['total_operations'] - basic_means[2]) / basic_stds[2],
                'avg_available_machines': (basic_features['avg_available_machines'] - basic_means[3]) / basic_stds[3],
                'std_available_machines': (basic_features['std_available_machines'] - basic_means[4]) / basic_stds[4]
            },
            'processing_time_features': {
                'processing_time_mean': (processing_features['processing_time_mean'] - processing_means[0]) / processing_stds[0],
                'processing_time_std': (processing_features['processing_time_std'] - processing_means[1]) / processing_stds[1],
                'processing_time_min': (processing_features['processing_time_min'] - processing_means[2]) / processing_stds[2],
                'processing_time_max': (processing_features['processing_time_max'] - processing_means[3]) / processing_stds[3],
                'machine_time_variance': (processing_features['machine_time_variance'] - processing_means[4]) / processing_stds[4]
            }
        }
        
        return normalized_single
    
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
    
    def create_fake_performance_data_for_new_data(self):
        """
        为新数据创建假的性能数据，所有评分都设为1（满分）
        这样新数据可以参与性能相似度计算
        
        Returns:
            dict: 假的性能数据结构
        """
        fake_performance_data = {
            "meta_heuristic": "FAKE_FOR_NEW_DATA",
            "execution_times": 20,
            "max_iterations": 100,
            "initialization_method": "unknown",  # 新数据的初始化方法未知
            "performance_metrics": {
                "mean": 1.0,  # 假设所有性能指标评分都为满分1.0
                "std": 1.0,
                "min": 1.0,
                "max": 1.0,
                "avg_convergence_generation": 1.0,
                "convergence_generation_std": 1.0
            }
        }
        return fake_performance_data
    
    def calculate_performance_similarity(self, new_performance_data, hist_performance_data):
        """
        计算新数据与历史数据的性能相似度
        
        Args:
            new_performance_data: 新数据的性能数据（假的）
            hist_performance_data: 历史数据的性能数据
            
        Returns:
            float: 性能相似度评分 [0, 1]
        """
        if not hist_performance_data or 'performance_metrics' not in hist_performance_data:
            return 0.5  # 默认值
        
        new_metrics = new_performance_data['performance_metrics']
        hist_metrics = hist_performance_data['performance_metrics']
        
        # 对每个性能指标计算相似度
        similarity_scores = []
        
        # 1. Makespan相似度（这里实际上是在比较假数据1.0与历史数据的标准化评分）
        hist_makespan = hist_metrics.get('mean', 0)
        hist_makespan_score = 1.0 / (1.0 + hist_makespan / 1000.0)  # 历史数据的makespan评分
        new_makespan_score = 1.0  # 新数据假设为满分
        makespan_similarity = 1.0 - abs(new_makespan_score - hist_makespan_score)
        similarity_scores.append(makespan_similarity)
        
        # 2. 收敛速度相似度
        hist_convergence = hist_metrics.get('avg_convergence_generation', 0)
        max_iterations = hist_performance_data.get('max_iterations', 100)
        hist_convergence_score = 1.0 - (hist_convergence / max_iterations)
        hist_convergence_score = max(0.0, min(1.0, hist_convergence_score))
        new_convergence_score = 1.0  # 新数据假设为满分
        convergence_similarity = 1.0 - abs(new_convergence_score - hist_convergence_score)
        similarity_scores.append(convergence_similarity)
        
        # 3. 稳定性相似度（基于标准差）
        hist_std = hist_metrics.get('std', 0)
        hist_stability_score = 1.0 / (1.0 + hist_std / 10.0)
        new_stability_score = 1.0  # 新数据假设为满分
        stability_similarity = 1.0 - abs(new_stability_score - hist_stability_score)
        similarity_scores.append(stability_similarity)
        
        # 4. 收敛稳定性相似度
        hist_convergence_std = hist_metrics.get('convergence_generation_std', 0)
        hist_convergence_stability_score = 1.0 / (1.0 + hist_convergence_std / 10.0)
        new_convergence_stability_score = 1.0  # 新数据假设为满分
        convergence_stability_similarity = 1.0 - abs(new_convergence_stability_score - hist_convergence_stability_score)
        similarity_scores.append(convergence_stability_similarity)
        
        # 计算加权平均性能相似度
        performance_weights = [0.4, 0.25, 0.2, 0.15]  # 对应makespan, convergence_speed, stability, convergence_stability
        performance_similarity = sum(w * s for w, s in zip(performance_weights, similarity_scores))
        
        return performance_similarity

    def calculate_similarity(self, new_data_normalized, historical_fjs_path, max_basic_distance, max_processing_distance, new_data_features=None, new_performance_data=None):
        """
        计算新数据与历史数据的综合相似度
        现在包含性能相似度计算
        """
        # 获取历史数据的标准化特征和原始特征
        hist_normalized = self.normalized_features[historical_fjs_path]
        hist_features = self.labeled_data[historical_fjs_path]['features']
        hist_performance_data = self.labeled_data[historical_fjs_path].get('performance_data', {})
        
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
        
        # 5. 计算性能相似度（新增）
        if new_performance_data:
            performance_similarity = self.calculate_performance_similarity(new_performance_data, hist_performance_data)
        else:
            performance_similarity = 0.5  # 默认值
        
        # 6. 计算综合加权相似度（包含性能相似度）
        weighted_similarity = (
            0.25 * basic_similarity +     # 减少基础特征权重
            0.20 * processing_similarity + # 减少加工时间权重
            0.15 * kde_similarity +       # 减少KDE权重
            0.20 * disjunctive_similarity + # 减少析取图权重
            0.20 * performance_similarity   # 新增性能相似度权重
        )
        
        self.log_debug(f"相似度计算 - {historical_fjs_path}: 基础={basic_similarity:.6f}, 加工时间={processing_similarity:.6f}, KDE={kde_similarity:.6f}, 析取图={disjunctive_similarity:.6f}, 性能={performance_similarity:.6f}, 加权={weighted_similarity:.6f}")
        
        return {
            "basic_similarity": basic_similarity,
            "processing_similarity": processing_similarity,
            "kde_similarity": kde_similarity,
            "disjunctive_similarity": disjunctive_similarity,
            "performance_similarity": performance_similarity,
            "weighted_similarity": weighted_similarity
        }
    
    def stage_one_similarity_search(self, new_data_features, top_k=5):
        """
        阶段一：多特征相似度检索
        基于五种特征融合加权计算（包含假性能数据），返回Top K最相似的历史样本
        
        Args:
            new_data_features: 新数据的特征
            top_k: 返回的最相似样本数量
            
        Returns:
            list: 候选样本列表，每个元素为 (fjs_path, similarity_score, similarity_details)
        """
        self.log_info(f"=== 阶段一：多特征相似度检索（包含性能相似度） ===")
        self.log_info(f"正在计算与 {len(self.labeled_data)} 个历史样本的相似度...")
        
        # 为新数据创建假的性能数据
        new_performance_data = self.create_fake_performance_data_for_new_data()
        self.log_info(f"已为新数据创建假的性能数据（所有评分设为1.0）")
        
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
        
        self.log_info(f"最大距离计算完成: 基础特征={max_basic_distance:.6f}, 加工时间={max_processing_distance:.6f}")
        
        # 计算所有历史样本的相似度（包含性能相似度）
        similarity_results = {}
        
        for fjs_path in self.normalized_features.keys():
            similarity_details = self.calculate_similarity(
                new_data_normalized, fjs_path, max_basic_distance, max_processing_distance, 
                new_data_features, new_performance_data
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
            self.log_info(f"   综合加权相似度: {details['weighted_similarity']:.6f}")
            self.log_info(f"   基础特征相似度: {details['basic_similarity']:.6f}")
            self.log_info(f"   加工时间特征相似度: {details['processing_similarity']:.6f}")
            self.log_info(f"   KDE相似度: {details['kde_similarity']:.6f}")
            self.log_info(f"   析取图相似度: {details['disjunctive_similarity']:.6f}")
            self.log_info(f"   性能相似度: {details['performance_similarity']:.6f}")
        
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
        self.log_info(f"\n=== 阶段二：策略推荐 ===")
        self.log_info(f"候选样本数量: {len(candidate_samples)}")
        
        # 收集候选样本的策略性能数据
        strategy_performance = {}
        
        for fjs_path, similarity_score, _ in candidate_samples:
            if fjs_path in self.labeled_data and 'performance_data' in self.labeled_data[fjs_path]:
                performance_data = self.labeled_data[fjs_path]['performance_data']
                init_methods = self.labeled_data['initialization_methods']
                    
                for strategy_name, strategy_data in init_methods.items():
                    if strategy_name not in ["heuristic", "mixed", "random"]:
                        continue  # 跳过非三种初始化方法
                    if strategy_name not in strategy_performance:
                        strategy_performance[strategy_name] = []
                    
                    # 计算策略的综合性能评分
                    # 基于求解精度、收敛速度和稳定性
                    mean_makespan = strategy_data.get('mean', 0)
                    std_makespan = strategy_data.get('std', 0)
                    avg_convergence_gen = strategy_data.get('avg_convergence_generation', 0)
                    convergence_std = strategy_data.get('convergence_generation_std', 0)
                    
                    # 多维度性能评分
                    # 1. Makespan评分（越小越好）
                    makespan_score = 1.0 / (1.0 + mean_makespan / 1000.0)
                    
                    # 2. 收敛速度评分（收敛代数越小越好）
                    # 假设最大迭代次数为100，收敛代数越小越好
                    max_iterations = 100
                    convergence_speed_score = 1.0 - (avg_convergence_gen / max_iterations)
                    convergence_speed_score = max(0.0, min(1.0, convergence_speed_score))  # 限制在[0,1]范围
                    
                    # 3. 稳定性评分（标准差越小越好）
                    # 使用makespan的标准差，越小表示结果越稳定
                    stability_score = 1.0 / (1.0 + std_makespan / 10.0)  # 归一化标准差
                    
                    # 4. 收敛稳定性评分（收敛代数的标准差越小越好）
                    convergence_stability_score = 1.0 / (1.0 + convergence_std / 10.0)
                    
                    # 综合性能评分（加权平均）
                    # 可以根据实际需求调整权重
                    weights = {
                        'makespan': 0.4,      # makespan权重最高
                        'convergence_speed': 0.25,  # 收敛速度
                        'stability': 0.2,     # 结果稳定性
                        'convergence_stability': 0.15  # 收敛稳定性
                    }
                    
                    performance_score = (
                        weights['makespan'] * makespan_score +
                        weights['convergence_speed'] * convergence_speed_score +
                        weights['stability'] * stability_score +
                        weights['convergence_stability'] * convergence_stability_score
                    )
                    
                    # 存储策略性能数据
                    strategy_performance[strategy_name].append({
                        'fjs_path': fjs_path,
                        'similarity_score': similarity_score,
                        'performance_score': performance_score,
                        'detailed_scores': {
                            'makespan_score': makespan_score,
                            'convergence_speed_score': convergence_speed_score,
                            'stability_score': stability_score,
                            'convergence_stability_score': convergence_stability_score
                        },
                        'raw_metrics': {
                            'mean_makespan': mean_makespan,
                            'std_makespan': std_makespan,
                            'avg_convergence_gen': avg_convergence_gen,
                            'convergence_std': convergence_std
                        }
                    })
        
        self.log_info(f"找到 {len(strategy_performance)} 种初始化策略")
        
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
                    
                    self.log_debug(f"策略 {strategy_name}: 加权评分={weighted_avg_score:.6f}, 样本数={len(performances)}")
                    
                    # 计算平均详细评分
                    avg_makespan_score = np.mean([p['detailed_scores']['makespan_score'] for p in performances])
                    avg_convergence_speed_score = np.mean([p['detailed_scores']['convergence_speed_score'] for p in performances])
                    avg_stability_score = np.mean([p['detailed_scores']['stability_score'] for p in performances])
                    avg_convergence_stability_score = np.mean([p['detailed_scores']['convergence_stability_score'] for p in performances])
                    
                    self.log_debug(f"  详细评分 - Makespan: {avg_makespan_score:.6f}, 收敛速度: {avg_convergence_speed_score:.6f}, 稳定性: {avg_stability_score:.6f}, 收敛稳定性: {avg_convergence_stability_score:.6f}")
        
        # 按加权评分排序
        sorted_strategies = sorted(
            strategy_scores.items(),
            key=lambda x: x[1]['weighted_score'],
            reverse=True
        )
        
        # 获取Top K推荐策略
        top_k_strategies = sorted_strategies[:top_k]
        
        self.log_info(f"\nTop {top_k} 推荐策略:")
        for i, (strategy_name, score_info) in enumerate(top_k_strategies, 1):
            self.log_info(f"{i}. {strategy_name}")
            self.log_info(f"   加权性能评分: {score_info['weighted_score']:.6f}")
            self.log_info(f"   参考样本数量: {score_info['sample_count']}")
            
            # 计算并显示详细评分
            performances = score_info['performances']
            avg_makespan_score = np.mean([p['detailed_scores']['makespan_score'] for p in performances])
            avg_convergence_speed_score = np.mean([p['detailed_scores']['convergence_speed_score'] for p in performances])
            avg_stability_score = np.mean([p['detailed_scores']['stability_score'] for p in performances])
            avg_convergence_stability_score = np.mean([p['detailed_scores']['convergence_stability_score'] for p in performances])
            
            self.log_info(f"   详细评分:")
            self.log_info(f"     Makespan评分: {avg_makespan_score:.6f}")
            self.log_info(f"     收敛速度评分: {avg_convergence_speed_score:.6f}")
            self.log_info(f"     稳定性评分: {avg_stability_score:.6f}")
            self.log_info(f"     收敛稳定性评分: {avg_convergence_stability_score:.6f}")
        
        return [(strategy_name, score_info['weighted_score']) for strategy_name, score_info in top_k_strategies]
    
    def calculate_detailed_feature_similarity(self, new_data_normalized, historical_fjs_path, max_basic_distance, max_processing_distance, new_data_features=None, new_performance_data=None):
        """
        计算详细的特征相似度，细化到每个具体指标，包含性能相似度
        
        Returns:
            dict: 包含每个指标详细相似度的字典
        """
        # 获取历史数据的标准化特征和原始特征
        hist_normalized = self.normalized_features[historical_fjs_path]
        hist_features = self.labeled_data[historical_fjs_path]['features']
        hist_performance_data = self.labeled_data[historical_fjs_path].get('performance_data', {})
        
        detailed_similarities = {}
        
        # 1. 基础特征的每个指标相似度
        basic_features_new = new_data_normalized["basic_features"]
        basic_features_hist = hist_normalized["basic_features"]
        
        basic_indicators = ['num_jobs', 'num_machines', 'total_operations', 'avg_available_machines', 'std_available_machines']
        basic_similarities = {}
        
        for indicator in basic_indicators:
            distance = abs(basic_features_new[indicator] - basic_features_hist[indicator])
            # 使用高斯相似度函数，距离越小相似度越高
            similarity = np.exp(-distance**2 / 2)
            basic_similarities[indicator] = similarity
        
        detailed_similarities['basic_features'] = basic_similarities
        
        # 2. 加工时间特征的每个指标相似度
        processing_features_new = new_data_normalized["processing_time_features"]
        processing_features_hist = hist_normalized["processing_time_features"]
        
        processing_indicators = ['processing_time_mean', 'processing_time_std', 'processing_time_min', 'processing_time_max', 'machine_time_variance']
        processing_similarities = {}
        
        for indicator in processing_indicators:
            distance = abs(processing_features_new[indicator] - processing_features_hist[indicator])
            similarity = np.exp(-distance**2 / 2)
            processing_similarities[indicator] = similarity
        
        detailed_similarities['processing_time_features'] = processing_similarities
        
        # 3. KDE相似度
        if new_data_features and "kde_features" in new_data_features and "kde_features" in hist_features:
            kde_similarity = 1 - self.calculate_js_divergence(
                new_data_features["kde_features"]["density"],
                hist_features["kde_features"]["density"]
            )
        else:
            kde_similarity = 0.5
        
        detailed_similarities['kde_similarity'] = kde_similarity
        
        # 4. 析取图相似度
        if new_data_features and "disjunctive_graphs_features" in new_data_features and "disjunctive_graphs_features" in hist_features:
            disjunctive_similarity = self.calculate_disjunctive_graph_similarity(
                new_data_features["disjunctive_graphs_features"],
                hist_features["disjunctive_graphs_features"]
            )
        else:
            disjunctive_similarity = 0.5
        
        detailed_similarities['disjunctive_similarity'] = disjunctive_similarity
        
        # 5. 性能相似度（新增）
        if new_performance_data:
            performance_similarity = self.calculate_performance_similarity(new_performance_data, hist_performance_data)
        else:
            performance_similarity = 0.5
        
        detailed_similarities['performance_similarity'] = performance_similarity
        
        # 6. 计算加权综合相似度
        # 基础特征权重分配
        basic_weights = {
            'num_jobs': 0.25,
            'num_machines': 0.25, 
            'total_operations': 0.2,
            'avg_available_machines': 0.15,
            'std_available_machines': 0.15
        }
        
        basic_weighted_similarity = sum(basic_weights[k] * basic_similarities[k] for k in basic_weights)
        
        # 加工时间特征权重分配
        processing_weights = {
            'processing_time_mean': 0.3,
            'processing_time_std': 0.25,
            'processing_time_min': 0.15,
            'processing_time_max': 0.15,
            'machine_time_variance': 0.15
        }
        
        processing_weighted_similarity = sum(processing_weights[k] * processing_similarities[k] for k in processing_weights)
        
        # 五维特征综合相似度（包含性能相似度）
        overall_similarity = (
            0.25 * basic_weighted_similarity +     # 减少基础特征权重
            0.20 * processing_weighted_similarity + # 减少加工时间权重
            0.15 * kde_similarity +               # 减少KDE权重
            0.20 * disjunctive_similarity +       # 减少析取图权重
            0.20 * performance_similarity         # 新增性能相似度权重
        )
        
        detailed_similarities['basic_weighted_similarity'] = basic_weighted_similarity
        detailed_similarities['processing_weighted_similarity'] = processing_weighted_similarity
        detailed_similarities['overall_similarity'] = overall_similarity
        
        return detailed_similarities

    def find_most_similar_historical_samples(self, new_data_features, top_k=5):
        """
        找到与新数据最相似的历史样本（包含性能相似度）
        
        Args:
            new_data_features: 新数据的特征
            top_k: 返回最相似样本的数量
            
        Returns:
            list: 最相似的历史样本列表，每个元素为 (fjs_path, overall_similarity, detailed_similarities)
        """
        self.log_info(f"=== 寻找最相似的历史样本（包含性能相似度） ===")
        self.log_info(f"正在与 {len(self.labeled_data)} 个历史样本进行详细特征比较...")
        
        # 为新数据创建假的性能数据
        new_performance_data = self.create_fake_performance_data_for_new_data()
        self.log_info(f"已为新数据创建假的性能数据（所有评分设为1.0）")
        
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
        
        for fjs_path in self.normalized_features.keys():
            basic_distance = self.calculate_euclidean_distance(
                new_data_normalized["basic_features"],
                self.normalized_features[fjs_path]["basic_features"]
            )
            max_basic_distance = max(max_basic_distance, basic_distance)
            
            processing_distance = self.calculate_euclidean_distance(
                new_data_normalized["processing_time_features"],
                self.normalized_features[fjs_path]["processing_time_features"]
            )
            max_processing_distance = max(max_processing_distance, processing_distance)
        
        # 计算与每个历史样本的详细相似度（包含性能相似度）
        similarity_results = []
        
        for fjs_path in self.normalized_features.keys():
            detailed_similarities = self.calculate_detailed_feature_similarity(
                new_data_normalized, fjs_path, max_basic_distance, max_processing_distance, 
                new_data_features, new_performance_data
            )
            
            similarity_results.append((
                fjs_path, 
                detailed_similarities['overall_similarity'], 
                detailed_similarities
            ))
        
        # 按综合相似度排序
        similarity_results.sort(key=lambda x: x[1], reverse=True)
        
        # 获取Top K最相似样本
        top_similar_samples = similarity_results[:top_k]
        
        self.log_info(f"\nTop {top_k} 最相似的历史样本:")
        for i, (fjs_path, overall_similarity, detailed_similarities) in enumerate(top_similar_samples, 1):
            self.log_info(f"{i}. {fjs_path}")
            self.log_info(f"   综合相似度: {overall_similarity:.6f}")
            self.log_info(f"   基础特征相似度: {detailed_similarities['basic_weighted_similarity']:.6f}")
            self.log_info(f"   加工时间特征相似度: {detailed_similarities['processing_weighted_similarity']:.6f}")
            self.log_info(f"   KDE相似度: {detailed_similarities['kde_similarity']:.6f}")
            self.log_info(f"   析取图相似度: {detailed_similarities['disjunctive_similarity']:.6f}")
            self.log_info(f"   性能相似度: {detailed_similarities['performance_similarity']:.6f}")
            
            # 显示详细指标相似度
            self.log_info(f"   基础特征详细相似度:")
            for indicator, similarity in detailed_similarities['basic_features'].items():
                self.log_info(f"     {indicator}: {similarity:.6f}")
            
            self.log_info(f"   加工时间特征详细相似度:")
            for indicator, similarity in detailed_similarities['processing_time_features'].items():
                self.log_info(f"     {indicator}: {similarity:.6f}")
        
        return top_similar_samples

    def single_stage_recommendation(self, new_data_features, top_k_strategies=3, feature_weight=0.4, performance_weight=0.6):
        """
        基于综合加权评分的初始化策略推荐
        
        核心流程：
        1. 计算新数据与所有历史样本的特征相似度
        2. 对每种初始化方法的所有样本进行加权评分
        3. 综合特征相似度和性能评分，直接推荐最优初始化方法
        
        Args:
            new_data_features: 新数据的特征
            top_k_strategies: 推荐的策略数量
            feature_weight: 特征相似度权重（用于最终评分）
            performance_weight: 性能目标权重（用于最终评分）
            
        Returns:
            list: 推荐策略列表，每个元素为 (strategy_name, final_score, feature_score, performance_score, supporting_samples)
        """
        self.log_info(f"=== 基于综合加权评分的策略推荐 ===")
        
        # 按初始化方法分组历史样本
        method_samples = {
            'heuristic': {},
            'mixed': {},
            'random': {}
        }
        
        for sample_id, sample_data in self.labeled_data.items():
            init_method = sample_data.get('initialization_method', '')
            if init_method in method_samples:
                method_samples[init_method][sample_id] = sample_data
        
        self.log_info(f"历史样本统计:")
        for method, samples in method_samples.items():
            self.log_info(f"  - {method}: {len(samples)} 个样本")
        
        # 标准化新数据特征
        new_data_normalized = self.normalize_single_features(new_data_features)
        
        strategy_evaluations = {}
        
        # 为每种初始化方法计算综合加权评分
        for method_name, method_historical_samples in method_samples.items():
            if not method_historical_samples:
                self.log_warning(f"初始化方法 {method_name} 没有历史样本数据")
                continue
                
            self.log_info(f"\n=== 评估初始化方法: {method_name} ===")
            
            # 存储所有样本的评分信息
            sample_evaluations = []
            similarity_scores = []
            performance_scores = []
            
            # 计算与该方法所有历史样本的综合评分
            for sample_id, sample_data in method_historical_samples.items():
                features = sample_data.get('features', {})
                
                # 为新数据创建假的性能数据
                new_performance_data = self.create_fake_performance_data_for_new_data()
                
                # 计算特征相似度（包含性能相似度）
                detailed_similarities = self.calculate_detailed_feature_similarity(
                    new_data_normalized, 
                    sample_id, 
                    self.max_basic_distance, 
                    self.max_processing_distance, 
                    new_data_features,
                    new_performance_data
                )
                feature_similarity = detailed_similarities.get('overall_similarity', 0)
                
                # 提取性能数据
                performance_data = sample_data.get('performance_data', {})
                if not performance_data or 'performance_metrics' not in performance_data:
                    continue
                
                performance_metrics = performance_data['performance_metrics']
                
                # 计算性能评分
                mean_makespan = performance_metrics.get('mean', 0)
                std_makespan = performance_metrics.get('std', 0)
                avg_convergence_gen = performance_metrics.get('avg_convergence_generation', 0)
                convergence_std = performance_metrics.get('convergence_generation_std', 0)
                
                # 多维度性能评分
                makespan_score = 1.0 / (1.0 + mean_makespan / 1000.0)
                max_iterations = performance_data.get('max_iterations', 100)
                convergence_speed_score = 1.0 - (avg_convergence_gen / max_iterations)
                convergence_speed_score = max(0.0, min(1.0, convergence_speed_score))
                stability_score = 1.0 / (1.0 + std_makespan / 10.0)
                convergence_stability_score = 1.0 / (1.0 + convergence_std / 10.0)
                
                # 综合性能评分
                performance_weights = {
                    'makespan': 0.4,
                    'convergence_speed': 0.25,
                    'stability': 0.2,
                    'convergence_stability': 0.15
                }
                
                performance_score = (
                    performance_weights['makespan'] * makespan_score +
                    performance_weights['convergence_speed'] * convergence_speed_score +
                    performance_weights['stability'] * stability_score +
                    performance_weights['convergence_stability'] * convergence_stability_score
                )
                
                # 计算该样本的综合评分（特征相似度 × 性能评分作为权重）
                sample_weight = feature_similarity * performance_score
                
                sample_info = {
                    'sample_id': sample_id,
                    'original_fjs_path': sample_data.get('original_fjs_path', ''),
                    'feature_similarity': feature_similarity,
                            'performance_score': performance_score,
                    'sample_weight': sample_weight,
                    'performance_metrics': performance_metrics,
                            'detailed_scores': {
                                'makespan_score': makespan_score,
                                'convergence_speed_score': convergence_speed_score,
                                'stability_score': stability_score,
                                'convergence_stability_score': convergence_stability_score
                            },
                            'raw_metrics': {
                                'mean_makespan': mean_makespan,
                                'std_makespan': std_makespan,
                                'avg_convergence_gen': avg_convergence_gen,
                                'convergence_std': convergence_std
                    }
                }
                
                sample_evaluations.append(sample_info)
                similarity_scores.append(feature_similarity)
                performance_scores.append(performance_score)
            
            if not sample_evaluations:
                self.log_warning(f"初始化方法 {method_name} 没有有效的样本数据")
                continue
            
            # 计算加权平均特征相似度和性能评分
            total_weight = sum(sample['sample_weight'] for sample in sample_evaluations)
            if total_weight == 0:
                # 如果所有权重都为0，使用简单平均
                avg_feature_similarity = np.mean(similarity_scores)
                avg_performance_score = np.mean(performance_scores)
            else:
                # 使用加权平均
                avg_feature_similarity = sum(sample['feature_similarity'] * sample['sample_weight'] 
                                            for sample in sample_evaluations) / total_weight
                avg_performance_score = sum(sample['performance_score'] * sample['sample_weight'] 
                                          for sample in sample_evaluations) / total_weight
            
            # 最终综合评分
            final_score = feature_weight * avg_feature_similarity + performance_weight * avg_performance_score
            
            # 选择权重最高的前5个样本作为支持样本
            sample_evaluations.sort(key=lambda x: x['sample_weight'], reverse=True)
            top_supporting_samples = sample_evaluations[:5]
            
            strategy_evaluations[method_name] = {
                'final_score': final_score,
                'feature_similarity_score': avg_feature_similarity,
                'performance_score': avg_performance_score,
                'total_samples_used': len(sample_evaluations),  # 实际参与计算的样本数
                'top_samples_count': len(top_supporting_samples),  # 显示的代表性样本数
                'supporting_samples': [{
                    'sample_id': sample['sample_id'],
                    'original_fjs_path': sample['original_fjs_path'],
                    'similarity': sample['feature_similarity'],
                    'performance_metrics': sample['performance_metrics'],
                    'detailed_scores': sample['detailed_scores'],
                    'raw_metrics': sample['raw_metrics'],
                    'weight': sample['sample_weight']
                } for sample in top_supporting_samples]
            }
            
            self.log_info(f"参与计算的总样本数: {len(sample_evaluations)}")
            self.log_info(f"保留的代表性样本数: {len(top_supporting_samples)}")
            self.log_info(f"加权平均特征相似度: {avg_feature_similarity:.6f}")
            self.log_info(f"加权平均性能评分: {avg_performance_score:.6f}")
            self.log_info(f"最终综合评分: {final_score:.6f}")
            self.log_info(f"前3个最高权重样本:")
            for i, sample in enumerate(top_supporting_samples[:3], 1):
                self.log_info(f"  {i}. {sample['sample_id']} (权重: {sample['sample_weight']:.6f}, 相似度: {sample['feature_similarity']:.6f})")
        
        if not strategy_evaluations:
            self.log_error("未找到有效的策略推荐")
            return []
        
        # 按最终评分排序并返回推荐结果
        sorted_strategies = sorted(strategy_evaluations.items(), key=lambda x: x[1]['final_score'], reverse=True)
        
        self.log_info(f"\n=== 推荐结果排序 ===")
        for i, (strategy_name, evaluation) in enumerate(sorted_strategies[:top_k_strategies], 1):
            self.log_info(f"第{i}名: {strategy_name} (评分: {evaluation['final_score']:.6f})")
            self.log_info(f"   特征相似度评分: {evaluation['feature_similarity_score']:.6f}")
            self.log_info(f"   性能目标评分: {evaluation['performance_score']:.6f}")
            self.log_info(f"   参与计算样本数: {evaluation['total_samples_used']}")
        
        # 构建推荐结果列表
        recommended_strategies = [
            (
                strategy_name,
                evaluation['final_score'],
                evaluation['feature_similarity_score'],
                evaluation['performance_score'],
                evaluation['supporting_samples']
            )
            for strategy_name, evaluation in sorted_strategies[:top_k_strategies]
        ]
        
        return recommended_strategies
    
    def recommend(self, new_data_features, top_k_strategies=3, feature_weight=0.4, performance_weight=0.6):
        """
        执行一阶段推荐流程：融合特征相似度和性能目标评分
        
        Args:
            new_data_features: 新数据的特征
            top_k_strategies: 推荐的策略数量
            feature_weight: 特征相似度权重 (默认: 0.4)
            performance_weight: 性能目标权重 (默认: 0.6)
            
        Returns:
            dict: 推荐结果
        """
        self.log_info("开始执行一阶段推荐流程...")
        self.log_info(f"特征相似度权重: {feature_weight}, 性能目标权重: {performance_weight}")
        start_time = time.time()
        
        # 执行一阶段综合推荐
        recommended_strategies = self.single_stage_recommendation(
            new_data_features, top_k_strategies, feature_weight, performance_weight
        )
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 构建推荐结果
        result = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': total_time,
            'recommendation_parameters': {
                'feature_weight': feature_weight,
                'performance_weight': performance_weight,
                'top_k_strategies': top_k_strategies
            },
            'recommended_strategies': [
                {
                    'strategy_name': strategy_name,
                    'final_score': final_score,
                    'feature_similarity_score': feature_score,
                    'performance_score': performance_score,
                    'supporting_samples': supporting_samples
                }
                for strategy_name, final_score, feature_score, performance_score, supporting_samples in recommended_strategies
            ]
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
        """可视化推荐结果（针对新的加权评分方法）"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. 综合评分对比 - 改进的条形图
            recommended_strategies = results['recommended_strategies']
            
            strategies = [data['strategy_name'] for data in recommended_strategies]
            final_scores = [data['final_score'] for data in recommended_strategies]
            feature_scores = [data['feature_similarity_score'] for data in recommended_strategies]
            performance_scores = [data['performance_score'] for data in recommended_strategies]
            
            # 设置现代化的颜色主题
            colors = {
                'feature': '#2E86AB',      # 深蓝色
                'performance': '#A23B72',  # 深紫红色
                'final': '#F18F01'        # 橙色
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # 左图：分组条形图
            x = np.arange(len(strategies))
            width = 0.25
            
            bars1 = ax1.bar(x - width, feature_scores, width, label='加权平均特征相似度', 
                           color=colors['feature'], alpha=0.8, edgecolor='white', linewidth=1.5)
            bars2 = ax1.bar(x, performance_scores, width, label='加权平均性能评分', 
                           color=colors['performance'], alpha=0.8, edgecolor='white', linewidth=1.5)
            bars3 = ax1.bar(x + width, final_scores, width, label='最终综合评分', 
                           color=colors['final'], alpha=0.8, edgecolor='white', linewidth=1.5)
            
            # 添加数值标签
            for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
                height1, height2, height3 = bar1.get_height(), bar2.get_height(), bar3.get_height()
                ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.005, f'{height1:.6f}', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.005, f'{height2:.6f}', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax1.text(bar3.get_x() + bar3.get_width()/2., height3 + 0.005, f'{height3:.6f}', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax1.set_xlabel('初始化策略', fontsize=12, fontweight='bold')
            ax1.set_ylabel('评分', fontsize=12, fontweight='bold')
            ax1.set_title('综合加权评分对比', fontsize=14, fontweight='bold', pad=20)
            ax1.set_xticks(x)
            ax1.set_xticklabels(strategies, fontsize=11)
            ax1.legend(loc='upper right', frameon=True, shadow=True)
            ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
            ax1.set_ylim(0, max(max(feature_scores), max(performance_scores), max(final_scores)) * 1.15)
            
            # 右图：堆积条形图显示评分构成
            bottom_values = np.zeros(len(strategies))
            
            # 计算各部分的权重贡献
            feature_weight = results['recommendation_parameters']['feature_weight']
            performance_weight = results['recommendation_parameters']['performance_weight']
            
            feature_contributions = [f * feature_weight for f in feature_scores]
            performance_contributions = [p * performance_weight for p in performance_scores]
            
            bars_f = ax2.bar(strategies, feature_contributions, label=f'特征相似度贡献 (权重{feature_weight})', 
                           color=colors['feature'], alpha=0.8, edgecolor='white', linewidth=1.5)
            bars_p = ax2.bar(strategies, performance_contributions, bottom=feature_contributions,
                           label=f'性能评分贡献 (权重{performance_weight})', 
                           color=colors['performance'], alpha=0.8, edgecolor='white', linewidth=1.5)
            
            # 添加总分标签
            for i, (f_contrib, p_contrib) in enumerate(zip(feature_contributions, performance_contributions)):
                total = f_contrib + p_contrib
                ax2.text(i, total + 0.01, f'{total:.6f}', ha='center', va='bottom', 
                        fontsize=11, fontweight='bold')
            
            ax2.set_xlabel('初始化策略', fontsize=12, fontweight='bold')
            ax2.set_ylabel('评分贡献', fontsize=12, fontweight='bold')
            ax2.set_title('评分权重贡献分析', fontsize=14, fontweight='bold', pad=20)
            ax2.legend(loc='upper right', frameon=True, shadow=True)
            ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            plt.tight_layout()
            comprehensive_plot_path = os.path.join(output_dir, 'comprehensive_recommendation.png')
            plt.savefig(comprehensive_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 2. 详细性能评分对比 - 改进的雷达图
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # 为每个策略计算详细性能评分
            detailed_scores = {}
            
            for strategy_data in recommended_strategies:
                strategy_name = strategy_data['strategy_name']
                supporting_samples = strategy_data['supporting_samples']
                
                # 计算加权平均详细评分
                if supporting_samples:
                    total_weight = sum(s.get('weight', 1) for s in supporting_samples)
                    if total_weight > 0:
                        avg_makespan_score = sum(s['detailed_scores']['makespan_score'] * s.get('weight', 1) 
                                               for s in supporting_samples) / total_weight
                        avg_convergence_speed_score = sum(s['detailed_scores']['convergence_speed_score'] * s.get('weight', 1) 
                                                         for s in supporting_samples) / total_weight
                        avg_stability_score = sum(s['detailed_scores']['stability_score'] * s.get('weight', 1) 
                                                for s in supporting_samples) / total_weight
                        avg_convergence_stability_score = sum(s['detailed_scores']['convergence_stability_score'] * s.get('weight', 1) 
                                                             for s in supporting_samples) / total_weight
                    else:
                        avg_makespan_score = np.mean([s['detailed_scores']['makespan_score'] for s in supporting_samples])
                        avg_convergence_speed_score = np.mean([s['detailed_scores']['convergence_speed_score'] for s in supporting_samples])
                        avg_stability_score = np.mean([s['detailed_scores']['stability_score'] for s in supporting_samples])
                        avg_convergence_stability_score = np.mean([s['detailed_scores']['convergence_stability_score'] for s in supporting_samples])
                else:
                    avg_makespan_score = avg_convergence_speed_score = avg_stability_score = avg_convergence_stability_score = 0
                
                detailed_scores[strategy_name] = {
                    'makespan': avg_makespan_score,
                    'convergence_speed': avg_convergence_speed_score,
                    'stability': avg_stability_score,
                    'convergence_stability': avg_convergence_stability_score
                }
            
            # 2.1 雷达图（左）
            ax_radar = plt.subplot(1, 2, 1, projection='polar')
            categories = ['Makespan', '收敛速度', '稳定性', '收敛稳定性']
            strategy_colors = ['#E74C3C', '#3498DB', '#2ECC71']
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            for idx, (strategy_name, scores) in enumerate(detailed_scores.items()):
                values = [scores['makespan'], scores['convergence_speed'], 
                         scores['stability'], scores['convergence_stability']]
                values += values[:1]
                
                ax_radar.plot(angles, values, 'o-', linewidth=3, label=strategy_name, 
                            color=strategy_colors[idx % len(strategy_colors)], markersize=6)
                ax_radar.fill(angles, values, alpha=0.15, color=strategy_colors[idx % len(strategy_colors)])
            
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(categories, fontsize=11)
            ax_radar.set_ylim(0, 1)
            ax_radar.set_title('详细性能评分雷达图', fontsize=14, fontweight='bold', pad=30)
            ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax_radar.grid(True, alpha=0.3)
            
            # 2.2 性能指标对比（右）
            ax_perf = axes[1]
            categories_short = ['Makespan', '收敛速度', '稳定性', '收敛稳定性']
            x_pos = np.arange(len(categories_short))
            
            for idx, (strategy_name, scores) in enumerate(detailed_scores.items()):
                values = [scores['makespan'], scores['convergence_speed'], 
                         scores['stability'], scores['convergence_stability']]
                ax_perf.plot(x_pos, values, 'o-', linewidth=2.5, markersize=8, 
                           label=strategy_name, color=strategy_colors[idx % len(strategy_colors)])
            
            ax_perf.set_xlabel('性能指标', fontsize=12, fontweight='bold')
            ax_perf.set_ylabel('评分', fontsize=12, fontweight='bold')
            ax_perf.set_title('性能指标对比', fontsize=14, fontweight='bold')
            ax_perf.set_xticks(x_pos)
            ax_perf.set_xticklabels(categories_short, rotation=45, ha='right')
            ax_perf.legend()
            ax_perf.grid(True, alpha=0.3)
            ax_perf.set_ylim(0, 1)
            
            plt.tight_layout()
            detailed_plot_path = os.path.join(output_dir, 'detailed_performance_scores.png')
            plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.log_info(f"可视化结果已保存到: {output_dir}")
            self.log_info(f"  - 综合评分对比图: {comprehensive_plot_path}")
            self.log_info(f"  - 详细性能评分分析图: {detailed_plot_path}")
            
        except Exception as e:
            self.log_error(f"可视化失败: {e}")
            import traceback
            self.log_error(f"详细错误信息: {traceback.format_exc()}")


def main():
    """主函数：推荐系统命令行接口"""
    import sys
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='一阶段初始化策略推荐系统')
    parser.add_argument('fjs_file', help='输入的FJS文件路径')
    parser.add_argument('--top-k-strategies', type=int, default=3, help='推荐的策略数量 (默认: 3)')
    parser.add_argument('--feature-weight', type=float, default=0.4, help='特征相似度权重 (默认: 0.4)')
    parser.add_argument('--performance-weight', type=float, default=0.6, help='性能目标权重 (默认: 0.6)')
    parser.add_argument('--output-dir', default=None, help='输出目录 (默认: 自动生成)')
    
    args = parser.parse_args()
    
    # 获取当前脚本的绝对路径
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 转换输入文件为绝对路径
    if not os.path.isabs(args.fjs_file):
        args.fjs_file = os.path.abspath(args.fjs_file)
    
    # 检查输入文件是否存在
    if not os.path.exists(args.fjs_file):
        print(f"错误: 输入文件不存在: {args.fjs_file}")
        return
    
    # 获取文件名（不含扩展名）用于创建子目录
    fjs_basename = os.path.splitext(os.path.basename(args.fjs_file))[0]
    
    # 创建输出目录结构（使用绝对路径）
    if args.output_dir is None:
        base_output_dir = os.path.join(current_script_dir, 'result', 'single_stage_recommender_output')
    else:
        base_output_dir = os.path.abspath(args.output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = f"{fjs_basename}_{timestamp}"
    full_output_dir = os.path.join(base_output_dir, output_subdir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # 确保基础输出目录存在
    os.makedirs(base_output_dir, exist_ok=True)
    
    print("=" * 80)
    print("一阶段初始化策略推荐系统")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输入文件: {args.fjs_file}")
    print(f"输出目录: {full_output_dir}")
    print(f"推荐策略数量: {args.top_k_strategies}")
    print(f"特征相似度权重: {args.feature_weight}")
    print(f"性能目标权重: {args.performance_weight}")
    print("=" * 80)
    
    try:
        # 初始化推荐系统（带日志）- 使用绝对路径
        labeled_dataset_path = os.path.join(current_script_dir, "labeled_dataset", "converted_fjs_dataset_new.json")
        log_file = os.path.join(full_output_dir, "recommendation_log.log")
        recommender = InitializationStrategyRecommender(labeled_dataset_path, log_file)
        
        # 提取新数据特征 - 使用绝对路径导入
        sys.path.append(current_script_dir)
        from extract_new_data_features import extract_new_data_features
        
        recommender.log_info(f"开始提取新数据特征: {args.fjs_file}")
        new_data_features = extract_new_data_features(args.fjs_file)
        
        if new_data_features is None:
            recommender.log_error("新数据特征提取失败")
            return
        
        recommender.log_info(f"新数据特征提取完成")
        
        # 执行一阶段推荐
        results = recommender.recommend(new_data_features, 
                                      top_k_strategies=args.top_k_strategies,
                                      feature_weight=args.feature_weight,
                                      performance_weight=args.performance_weight)
        
        # 保存推荐结果
        output_file = os.path.join(full_output_dir, "recommendation_results.json")
        recommender.save_results(results, output_file)
        
        # 可视化结果
        visualization_dir = os.path.join(full_output_dir, "visualization")
        recommender.visualize_recommendation_results(results, visualization_dir)
        
        # 输出推荐结果摘要
        print("\n" + "=" * 80)
        print("一阶段推荐结果摘要")
        print("=" * 80)
        
        # 推荐参数
        print(f"\n推荐参数:")
        print(f"  特征相似度权重: {results['recommendation_parameters']['feature_weight']}")
        print(f"  性能目标权重: {results['recommendation_parameters']['performance_weight']}")
        print(f"  推荐策略数量: {results['recommendation_parameters']['top_k_strategies']}")
        
        # 推荐策略结果
        print(f"\n推荐策略:")
        recommended_strategies = results['recommended_strategies']
        for i, data in enumerate(recommended_strategies, 1):
            print(f"{i}. {data['strategy_name']}")
            print(f"   综合评分: {data['final_score']:.6f}")
            print(f"   特征相似度评分: {data['feature_similarity_score']:.6f}")
            print(f"   性能目标评分: {data['performance_score']:.6f}")
            print(f"   支持样本数量: {len(data['supporting_samples'])}")
            
            # 计算详细性能评分
            supporting_samples = data['supporting_samples']
            if supporting_samples:
                avg_makespan_score = np.mean([s['detailed_scores']['makespan_score'] for s in supporting_samples])
                avg_convergence_speed_score = np.mean([s['detailed_scores']['convergence_speed_score'] for s in supporting_samples])
                avg_stability_score = np.mean([s['detailed_scores']['stability_score'] for s in supporting_samples])
                avg_convergence_stability_score = np.mean([s['detailed_scores']['convergence_stability_score'] for s in supporting_samples])
                
                print(f"   详细性能评分:")
                print(f"     Makespan评分: {avg_makespan_score:.6f}")
                print(f"     收敛速度评分: {avg_convergence_speed_score:.6f}")
                print(f"     稳定性评分: {avg_stability_score:.6f}")
                print(f"     收敛稳定性评分: {avg_convergence_stability_score:.6f}")
                
                # 显示前3个最相似的支持样本
                sorted_samples = sorted(supporting_samples, key=lambda x: x['similarity'], reverse=True)
                print(f"   主要支持样本 (按特征相似度排序):")
                for j, sample in enumerate(sorted_samples[:3], 1):
                    print(f"     {j}. {sample['sample_id']}")
                    print(f"        特征相似度: {sample['similarity']:.6f}")
                    print(f"        来源数据: {sample['original_fjs_path']}")
            else:
                print(f"   详细评分: 无支持样本")
        
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