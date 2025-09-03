#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
初始化策略推荐系统
基于多特征融合相似度计算和标记数据集进行两阶段推荐
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
    """基于优化KNN算法的初始化策略推荐系统"""
    
    def __init__(self, labeled_dataset_path, log_file=None, detailed_weights=None):
        """
        初始化推荐系统
        
        Args:
            labeled_dataset_path: 标记数据集路径
            log_file: 日志文件路径，如果为None则不保存日志
            detailed_weights: 细化特征权重配置，如果为None则使用默认权重
        """
        self.labeled_dataset_path = labeled_dataset_path
        self.labeled_data = {}
        self.feature_vectors = {}  # 存储合并后的特征向量
        self.feature_scaler = {}   # 特征缩放器
        self.epsilon = 1e-8       # 防止除零的极小值
        
        # 设置细化权重配置
        self.setup_detailed_weights(detailed_weights)
        
        # 设置日志
        if log_file:
            self.setup_logging(log_file)
        else:
            self.logger = None
        
        # 加载新格式的标记数据集
        self.load_new_labeled_dataset()
        
        # 构建特征向量
        self.build_feature_vectors()
    
    def setup_detailed_weights(self, detailed_weights):
        """
        设置细化特征权重配置
        
        Args:
            detailed_weights: 细化权重配置字典
        """
        if detailed_weights:
            self.detailed_weights = detailed_weights
        else:
            # 默认细化权重配置
            self.detailed_weights = {
                'basic_features': {
                    'num_jobs': 0.08,                    # 工件数量
                    'num_machines': 0.08,                # 机器数量  
                    'total_operations': 0.06,            # 总操作数
                    'avg_available_machines': 0.05,      # 平均可用机器数
                    'std_available_machines': 0.03       # 可用机器数标准差
                },
                'processing_time_features': {
                    'processing_time_mean': 0.08,        # 平均加工时间
                    'processing_time_std': 0.06,         # 加工时间标准差
                    'processing_time_min': 0.04,         # 最小加工时间
                    'processing_time_max': 0.04,         # 最大加工时间
                    'machine_time_variance': 0.03        # 机器时间方差
                },
                'kde_similarity_weight': 0.2,
                'disjunctive_similarity_weight': 0.25
            }
    
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
    
    def load_new_labeled_dataset(self):
        """加载新格式的标记数据集"""
        try:
            self.log_info("正在加载新格式标记数据集...")
            with open(self.labeled_dataset_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # 转换新格式数据为内部使用格式
            self.log_info("正在转换数据格式...")
            self.labeled_data = {}
            
            for sample_id, sample_data in raw_data.items():
                # 使用sample_id作为键
                sample_key = sample_id
                
                # 构建兼容的数据结构
                converted_sample = {
                    'sample_id': sample_data['sample_id'],
                    'original_fjs_path': sample_data['original_fjs_path'],
                    'initialization_method': sample_data['initialization_method'],
                    'features': self.convert_features_format(sample_data['features']),
                    'performance_data': self.convert_performance_format(sample_data['performance_data'])
                }
                
                self.labeled_data[sample_key] = converted_sample
            
            self.log_info(f"成功加载 {len(self.labeled_data)} 条标记数据")
            
            # 统计不同初始化方法的数量
            method_counts = {}
            for sample_data in self.labeled_data.values():
                method = sample_data['initialization_method']
                method_counts[method] = method_counts.get(method, 0) + 1
            
            self.log_info("初始化方法分布:")
            for method, count in method_counts.items():
                self.log_info(f"  {method}: {count} 个样本")
                
        except Exception as e:
            error_msg = f"加载新格式标记数据集失败: {e}"
            self.log_error(error_msg)
            raise Exception(error_msg)
    
    def convert_features_format(self, features):
        """转换特征格式以兼容现有代码"""
        converted_features = {
            'basic_features': features['basic_features'],
            'processing_time_features': features['processing_time_features'],
            'kde_features': features['kde_features']
        }
        
        # 转换析取图特征格式
        if 'disjunctive_graphs_features' in features:
            disjunctive_features = features['disjunctive_graphs_features']
            converted_features['disjunctive_graphs_features'] = {
                'num_nodes': disjunctive_features.get('nodes_count', 0),
                'num_edges': disjunctive_features.get('edges_count', 0),
                'num_solid_edges': len(disjunctive_features.get('solid_frequency', {})),
                'num_dashed_edges': len(disjunctive_features.get('dashed_frequency', {})),
                'wl_solid_freq': disjunctive_features.get('solid_frequency', {}),
                'wl_dashed_freq': disjunctive_features.get('dashed_frequency', {})
            }
        
        return converted_features
    
    def convert_performance_format(self, performance_data):
        """转换性能数据格式以兼容现有代码"""
        performance_metrics = performance_data['performance_metrics']
        
        # 构建兼容的性能数据格式
        converted_performance = {
            'meta_heuristic': performance_data.get('meta_heuristic', 'HA(GA+TS)'),
            'execution_times': performance_data.get('execution_times', 20),
            'max_iterations': performance_data.get('max_iterations', 100),
            'mean': performance_metrics['mean'],
            'std': performance_metrics['std'],
            'avg_convergence_generation': performance_metrics['avg_convergence_generation'],
            'convergence_generation_std': performance_metrics['convergence_generation_std']
        }
        
        return converted_performance
    
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
    
    def build_feature_vectors(self):
        """
        构建统一的特征向量，将所有特征类型合并为一个向量
        """
        self.log_info("开始构建特征向量...")
        
        # 收集所有特征数据
        all_features = []
        feature_paths = []
        
        for fjs_path, data in self.labeled_data.items():
            if 'features' in data:
                features = data['features']
                feature_vector = self.extract_feature_vector(features)
                if feature_vector is not None:
                    all_features.append(feature_vector)
                    feature_paths.append(fjs_path)
        
        if not all_features:
            self.log_error("没有找到有效的特征数据")
            return
        
        # 转换为numpy数组并标准化
        all_features = np.array(all_features)
        self.log_info(f"特征向量维度: {all_features.shape}")
        
        # 计算特征缩放参数
        self.feature_scaler = {
            'mean': np.mean(all_features, axis=0),
            'std': np.std(all_features, axis=0) + self.epsilon  # 防止除零
        }
        
        # 标准化特征向量
        normalized_features = (all_features - self.feature_scaler['mean']) / self.feature_scaler['std']
        
        # 存储标准化后的特征向量
        for i, fjs_path in enumerate(feature_paths):
            self.feature_vectors[fjs_path] = normalized_features[i]
        
        self.log_info(f"特征向量构建完成，共处理 {len(self.feature_vectors)} 个样本")
    
    def extract_feature_vector(self, features):
        """
        从特征字典中提取统一的特征向量
        
        Args:
            features: 包含所有特征类型的字典
            
        Returns:
            numpy.array: 合并后的特征向量
        """
        try:
            feature_components = []
            
            # 1. 基础特征
            if 'basic_features' in features:
                basic_features = features['basic_features']
                basic_vector = [
                    basic_features.get('num_jobs', 0),
                    basic_features.get('num_machines', 0),
                    basic_features.get('total_operations', 0),
                    basic_features.get('avg_available_machines', 0),
                    basic_features.get('std_available_machines', 0)
                ]
                feature_components.extend(basic_vector)
            
            # 2. 加工时间特征
            if 'processing_time_features' in features:
                time_features = features['processing_time_features']
                time_vector = [
                    time_features.get('processing_time_mean', 0),
                    time_features.get('processing_time_std', 0),
                    time_features.get('processing_time_min', 0),
                    time_features.get('processing_time_max', 0),
                    time_features.get('machine_time_variance', 0)
                ]
                feature_components.extend(time_vector)
            
            # 3. KDE特征（使用统计特征）
            if 'kde_features' in features and 'density' in features['kde_features']:
                kde_density = features['kde_features']['density']
                if isinstance(kde_density, list) and len(kde_density) > 0:
                    # 提取KDE分布的统计特征
                    kde_stats = [
                        np.mean(kde_density),
                        np.std(kde_density),
                        np.min(kde_density),
                        np.max(kde_density),
                        np.sum(kde_density)  # 总概率密度
                    ]
                    feature_components.extend(kde_stats)
                else:
                    feature_components.extend([0, 0, 0, 0, 0])
            else:
                feature_components.extend([0, 0, 0, 0, 0])
            
            # 4. 析取图特征
            if 'disjunctive_graphs_features' in features:
                graph_features = features['disjunctive_graphs_features']
                # 提取图的基本统计特征
                graph_vector = [
                    graph_features.get('num_nodes', 0),
                    graph_features.get('num_edges', 0),
                    graph_features.get('num_solid_edges', 0),
                    graph_features.get('num_dashed_edges', 0),
                    # 如果有WL标签频率，取前几个主要标签的频率
                ]
                
                # 添加WL标签特征
                if 'wl_solid_freq' in graph_features:
                    solid_freq = list(graph_features['wl_solid_freq'].values())
                    # 取前5个最频繁的标签
                    solid_freq.sort(reverse=True)
                    graph_vector.extend(solid_freq[:5] + [0] * max(0, 5 - len(solid_freq)))
                else:
                    graph_vector.extend([0] * 5)
                
                if 'wl_dashed_freq' in graph_features:
                    dashed_freq = list(graph_features['wl_dashed_freq'].values())
                    dashed_freq.sort(reverse=True)
                    graph_vector.extend(dashed_freq[:5] + [0] * max(0, 5 - len(dashed_freq)))
                else:
                    graph_vector.extend([0] * 5)
                
                feature_components.extend(graph_vector)
            else:
                # 如果没有图特征，用零填充
                feature_components.extend([0] * 14)
            
            return np.array(feature_components, dtype=float)
            
        except Exception as e:
            self.log_error(f"特征向量提取失败: {e}")
            return None
    
    def calculate_feature_distance(self, vector1, vector2):
        """
        计算两个特征向量之间的欧氏距离
        
        Args:
            vector1, vector2: numpy.array, 特征向量
            
        Returns:
            float: 欧氏距离
        """
        return np.sqrt(np.sum((vector1 - vector2) ** 2))
    
    def normalize_new_feature_vector(self, features):
        """
        为新数据构建并标准化特征向量
        
        Args:
            features: 新数据的特征字典
            
        Returns:
            numpy.array: 标准化后的特征向量
        """
        # 提取特征向量
        feature_vector = self.extract_feature_vector(features)
        if feature_vector is None:
            return None
        
        # 使用已训练的缩放器进行标准化
        normalized_vector = (feature_vector - self.feature_scaler['mean']) / self.feature_scaler['std']
        return normalized_vector
    
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
        
        # 5. 计算细化特征的加权相似度
        # 5.1 计算基础特征的细化加权相似度
        basic_features_new = new_data_normalized["basic_features"]
        basic_features_hist = hist_normalized["basic_features"]
        
        basic_detailed_similarity = 0
        for feature_name, weight in self.detailed_weights['basic_features'].items():
            if feature_name in basic_features_new and feature_name in basic_features_hist:
                # 计算单个特征的相似度（使用高斯相似度函数）
                distance = abs(basic_features_new[feature_name] - basic_features_hist[feature_name])
                feature_similarity = np.exp(-distance**2 / 2)
                basic_detailed_similarity += weight * feature_similarity
        
        # 5.2 计算加工时间特征的细化加权相似度
        processing_features_new = new_data_normalized["processing_time_features"]
        processing_features_hist = hist_normalized["processing_time_features"]
        
        processing_detailed_similarity = 0
        for feature_name, weight in self.detailed_weights['processing_time_features'].items():
            if feature_name in processing_features_new and feature_name in processing_features_hist:
                # 计算单个特征的相似度（使用高斯相似度函数）
                distance = abs(processing_features_new[feature_name] - processing_features_hist[feature_name])
                feature_similarity = np.exp(-distance**2 / 2)
                processing_detailed_similarity += weight * feature_similarity
        
        # 5.3 计算最终综合加权相似度（使用细化权重）
        weighted_similarity = (
            basic_detailed_similarity +
            processing_detailed_similarity +
            self.detailed_weights['kde_similarity_weight'] * kde_similarity +
            self.detailed_weights['disjunctive_similarity_weight'] * disjunctive_similarity
        )
        
        self.log_debug(f"细化相似度计算 - {historical_fjs_path}: 基础细化={basic_detailed_similarity:.4f}, 加工时间细化={processing_detailed_similarity:.4f}, KDE={kde_similarity:.4f}, 析取图={disjunctive_similarity:.4f}, 最终加权={weighted_similarity:.4f}")
        
        return {
            "basic_similarity": basic_similarity,
            "processing_similarity": processing_similarity,
            "kde_similarity": kde_similarity,
            "disjunctive_similarity": disjunctive_similarity,
            "basic_detailed_similarity": basic_detailed_similarity,
            "processing_detailed_similarity": processing_detailed_similarity,
            "weighted_similarity": weighted_similarity
        }
    
    def find_k_nearest_neighbors(self, new_data_features, k=5):
        """
        优化的KNN搜索：找到K个最近的邻居
        
        Args:
            new_data_features: 新数据的特征
            k: 邻居数量
            
        Returns:
            list: 邻居列表，每个元素为 (fjs_path, distance, feature_vector)
        """
        self.log_info(f"=== 开始KNN搜索 ===")
        self.log_info(f"正在搜索 {k} 个最近邻居...")
        
        # 为新数据构建并标准化特征向量
        new_vector = self.normalize_new_feature_vector(new_data_features)
        if new_vector is None:
            self.log_error("无法为新数据构建特征向量")
            return []
        
        self.log_info(f"新数据特征向量维度: {new_vector.shape}")
        
        # 计算与所有历史样本的距离
        distances = []
        for fjs_path, hist_vector in self.feature_vectors.items():
            distance = self.calculate_feature_distance(new_vector, hist_vector)
            distances.append((fjs_path, distance, hist_vector))
        
        # 按距离排序，取前K个
        distances.sort(key=lambda x: x[1])
        k_neighbors = distances[:k]
        
        self.log_info(f"找到 {len(k_neighbors)} 个最近邻居:")
        for i, (fjs_path, distance, _) in enumerate(k_neighbors, 1):
            self.log_info(f"{i}. {fjs_path} - 距离: {distance:.4f}")
        
        return k_neighbors
    
    def knn_strategy_recommendation(self, k_neighbors, top_k=3):
        """
        基于KNN邻居的策略推荐，使用改进的加权评分法
        
        Args:
            k_neighbors: KNN邻居列表，每个元素为 (fjs_path, distance, feature_vector)
            top_k: 推荐策略数量
            
        Returns:
            list: 推荐策略列表，每个元素为 (strategy_name, weighted_score)
        """
        self.log_info(f"\n=== KNN策略推荐 ===")
        self.log_info(f"基于 {len(k_neighbors)} 个邻居进行策略推荐")
        
        # 收集邻居的策略性能数据
        strategy_performance = {}
        
        for sample_id, distance, _ in k_neighbors:
            if sample_id in self.labeled_data:
                sample_data = self.labeled_data[sample_id]
                strategy_name = sample_data['initialization_method']
                performance_data = sample_data['performance_data']
                
                # 只考虑heuristic、mixed、random三种初始化方法
                if strategy_name not in ["heuristic", "mixed", "random"]:
                    continue
                    
                if strategy_name not in strategy_performance:
                    strategy_performance[strategy_name] = []
                
                # 从新格式的性能数据中提取指标
                mean_makespan = performance_data.get('mean', 0)
                std_makespan = performance_data.get('std', 0)
                avg_convergence_gen = performance_data.get('avg_convergence_generation', 0)
                convergence_std = performance_data.get('convergence_generation_std', 0)
                
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
                
                # 计算改进的双重加权权重
                # 对距离进行归一化处理，避免权重过小
                # 使用相似度而不是原始距离的倒数
                max_distance = max([d for _, d, _ in neighbors]) if neighbors else distance
                normalized_distance = distance / max_distance if max_distance > 0 else 1.0
                distance_weight = 1.0 - normalized_distance  # 距离越小，权重越大
                
                # 性能权重：使用makespan的倒数作为性能权重
                performance_weight = 1.0 / (mean_makespan + self.epsilon) if mean_makespan > 0 else 1.0
                # 综合权重
                combined_weight = distance_weight * performance_weight
                
                # 存储策略性能数据
                strategy_performance[strategy_name].append({
                    'sample_id': sample_id,
                    'distance': distance,
                    'distance_weight': distance_weight,
                    'performance_weight': performance_weight,
                    'combined_weight': combined_weight,
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
        
        # 计算每种策略的改进加权评分
        strategy_scores = {}
        
        for strategy_name, performances in strategy_performance.items():
            if len(performances) > 0:
                # 使用改进的双重加权机制计算加权评分
                total_weighted_score = 0
                total_weight = 0
                
                for perf in performances:
                    # 使用综合权重（距离权重 * 性能权重）
                    weight = perf['combined_weight']
                    total_weighted_score += weight * perf['performance_score']
                    total_weight += weight
                
                if total_weight > 0:
                    weighted_avg_score = total_weighted_score / total_weight
                    strategy_scores[strategy_name] = {
                        'weighted_score': weighted_avg_score,
                        'sample_count': len(performances),
                        'performances': performances
                    }
                    
                    self.log_debug(f"策略 {strategy_name}: 加权评分={weighted_avg_score:.4f}, 样本数={len(performances)}")
                    
                    # 计算平均详细评分
                    avg_makespan_score = np.mean([p['detailed_scores']['makespan_score'] for p in performances])
                    avg_convergence_speed_score = np.mean([p['detailed_scores']['convergence_speed_score'] for p in performances])
                    avg_stability_score = np.mean([p['detailed_scores']['stability_score'] for p in performances])
                    avg_convergence_stability_score = np.mean([p['detailed_scores']['convergence_stability_score'] for p in performances])
                    
                    self.log_debug(f"  详细评分 - Makespan: {avg_makespan_score:.4f}, 收敛速度: {avg_convergence_speed_score:.4f}, 稳定性: {avg_stability_score:.4f}, 收敛稳定性: {avg_convergence_stability_score:.4f}")
        
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
            self.log_info(f"   加权性能评分: {score_info['weighted_score']:.4f}")
            self.log_info(f"   参考样本数量: {score_info['sample_count']}")
            
            # 计算并显示详细评分
            performances = score_info['performances']
            avg_makespan_score = np.mean([p['detailed_scores']['makespan_score'] for p in performances])
            avg_convergence_speed_score = np.mean([p['detailed_scores']['convergence_speed_score'] for p in performances])
            avg_stability_score = np.mean([p['detailed_scores']['stability_score'] for p in performances])
            avg_convergence_stability_score = np.mean([p['detailed_scores']['convergence_stability_score'] for p in performances])
            
            self.log_info(f"   详细评分:")
            self.log_info(f"     Makespan评分: {avg_makespan_score:.4f}")
            self.log_info(f"     收敛速度评分: {avg_convergence_speed_score:.4f}")
            self.log_info(f"     稳定性评分: {avg_stability_score:.4f}")
            self.log_info(f"     收敛稳定性评分: {avg_convergence_stability_score:.4f}")
        
        return [(strategy_name, score_info['weighted_score']) for strategy_name, score_info in top_k_strategies]
    
    def recommend(self, new_data_features, k_neighbors=5, top_k_strategies=3):
        """
        执行优化的KNN推荐流程
        
        Args:
            new_data_features: 新数据的特征
            k_neighbors: KNN中K的值，即邻居数量
            top_k_strategies: 推荐的策略数量
            
        Returns:
            dict: 推荐结果
        """
        self.log_info("开始执行优化KNN推荐流程...")
        start_time = time.time()
        
        # 步骤1：KNN搜索 - 找到K个最近邻居
        neighbors = self.find_k_nearest_neighbors(new_data_features, k_neighbors)
        
        if not neighbors:
            self.log_error("未找到有效的邻居，推荐失败")
            return None
        
        # 步骤2：基于邻居的策略推荐
        recommended_strategies = self.knn_strategy_recommendation(neighbors, top_k_strategies)
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 构建推荐结果（保存更多数据用于可视化）
        result = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': total_time,
            'knn_search_results': {
                'k_neighbors': k_neighbors,
                'neighbors': [
                    {
                        'sample_id': sample_id,
                        'distance': distance,
                        'rank': i + 1,
                        'initialization_method': self.labeled_data[sample_id]['initialization_method'],
                        'original_fjs_path': self.labeled_data[sample_id]['original_fjs_path']
                    }
                    for i, (sample_id, distance, _) in enumerate(neighbors)
                ]
            },
            'strategy_recommendation_results': {
                'recommended_strategies': [
                    {
                        'strategy_name': strategy_name,
                        'weighted_score': weighted_score,
                        'rank': i + 1
                    }
                    for i, (strategy_name, weighted_score) in enumerate(recommended_strategies)
                ]
            },
            # 保存详细的邻居和策略信息用于可视化（转换numpy数组为列表避免JSON序列化错误）
            'visualization_data': {
                'neighbors_detail': [(sample_id, float(distance), vector.tolist()) 
                                    for sample_id, distance, vector in neighbors],
                'new_data_features': new_data_features,
                'strategy_performance_detail': self._get_strategy_performance_detail(neighbors)
            }
        }
        
        self.log_info(f"\n推荐流程完成，总耗时: {total_time:.2f}秒")
        
        return result
    
    def _get_strategy_performance_detail(self, neighbors):
        """获取邻居的详细策略性能信息，用于可视化"""
        strategy_details = []
        
        # 计算最大距离用于归一化
        max_distance = max([d for _, d, _ in neighbors]) if neighbors else 1.0
        
        for sample_id, distance, _ in neighbors:
            if sample_id in self.labeled_data:
                sample_data = self.labeled_data[sample_id]
                strategy_name = sample_data['initialization_method']
                performance_data = sample_data['performance_data']
                
                mean_makespan = performance_data.get('mean', 0)
                
                # 计算权重（与knn_strategy_recommendation中的计算保持一致）
                normalized_distance = distance / max_distance if max_distance > 0 else 1.0
                distance_weight = 1.0 - normalized_distance  # 距离越小，权重越大
                performance_weight = 1.0 / (mean_makespan + self.epsilon) if mean_makespan > 0 else 1.0
                combined_weight = distance_weight * performance_weight
                
                strategy_details.append({
                    'sample_id': sample_id,
                    'strategy_name': strategy_name,
                    'distance': float(distance),  # 转换为Python float避免JSON序列化问题
                    'distance_weight': float(distance_weight),
                    'performance_weight': float(performance_weight),
                    'combined_weight': float(combined_weight),
                    'mean_makespan': float(mean_makespan),
                    'original_fjs_path': sample_data['original_fjs_path']
                })
        
        return strategy_details
    
    def _make_json_serializable(self, obj):
        """递归地将numpy数组转换为Python列表以支持JSON序列化"""
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def save_results(self, results, output_file):
        """保存推荐结果到文件"""
        try:
            # 转换numpy数组为Python列表以支持JSON序列化
            json_safe_results = self._make_json_serializable(results)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_safe_results, f, indent=4, ensure_ascii=False)
            self.log_info(f"推荐结果已保存到: {output_file}")
        except Exception as e:
            self.log_error(f"保存结果失败: {e}")
    
    def visualize_recommendation_results(self, results, output_dir):
        """可视化推荐结果 - 新版本包含雷达图和散点图"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取可视化数据
            visualization_data = results.get('visualization_data', {})
            neighbors_detail = visualization_data.get('neighbors_detail', [])
            new_data_features = visualization_data.get('new_data_features', {})
            strategy_performance_detail = visualization_data.get('strategy_performance_detail', [])
            
            # 图表一：邻居特征相似性雷达图
            self._create_neighbors_similarity_radar(neighbors_detail, new_data_features, output_dir)
            
            # 图表二：邻居影响力与性能散点图
            self._create_neighbors_influence_scatter(strategy_performance_detail, output_dir)
            
            self.log_info(f"可视化结果已保存到: {output_dir}")
            
        except Exception as e:
            self.log_error(f"可视化失败: {e}")
    
    def _create_neighbors_similarity_radar(self, neighbors_detail, new_data_features, output_dir):
        """创建邻居特征相似性雷达图"""
        try:
            if not neighbors_detail or not new_data_features:
                self.log_warning("缺少邻居数据或新数据特征，跳过雷达图生成")
                return
            
            # 为新数据构建特征向量
            new_vector = self.normalize_new_feature_vector(new_data_features)
            if new_vector is None:
                self.log_warning("无法构建新数据特征向量，跳过雷达图生成")
                return
            
            # 计算特征类型的相似度
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # 定义特征类型和对应的维度范围
            feature_ranges = {
                '基础特征': (0, 5),
                '时间特征': (5, 10), 
                'KDE特征': (10, 15),
                '图特征': (15, 29)
            }
            
            # 计算每类特征的相似度
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
            angles = np.linspace(0, 2 * np.pi, len(feature_ranges), endpoint=False).tolist()
            
            # 为每个邻居计算特征相似度
            neighbor_similarities = []
            neighbor_labels = []
            
            for i, (sample_id, distance, feature_vector) in enumerate(neighbors_detail[:5]):  # 限制显示前5个邻居
                if sample_id in self.labeled_data:
                    similarities = []
                    
                    for feature_name, (start_idx, end_idx) in feature_ranges.items():
                        # 计算该特征类型的相似度（使用余弦相似度）
                        new_sub_vector = new_vector[start_idx:end_idx]
                        neighbor_sub_vector = feature_vector[start_idx:end_idx]
                        
                        # 计算余弦相似度
                        dot_product = np.dot(new_sub_vector, neighbor_sub_vector)
                        norm_new = np.linalg.norm(new_sub_vector)
                        norm_neighbor = np.linalg.norm(neighbor_sub_vector)
                        
                        if norm_new > 0 and norm_neighbor > 0:
                            cosine_sim = dot_product / (norm_new * norm_neighbor)
                            # 转换为0-1范围的相似度
                            similarity = (cosine_sim + 1) / 2
                        else:
                            similarity = 0.5
                        
                        similarities.append(similarity)
                    
                    neighbor_similarities.append(similarities)
                    # 获取原始文件路径的简短版本
                    original_path = self.labeled_data[sample_id]['original_fjs_path']
                    file_name = original_path.split('/')[-1] if '/' in original_path else original_path
                    neighbor_labels.append(f"{sample_id}\n{file_name}\n({self.labeled_data[sample_id]['initialization_method']})")
            
            # 绘制雷达图
            feature_labels = list(feature_ranges.keys())
            angles += angles[:1]  # 完成圆形
            
            for i, (similarities, label) in enumerate(zip(neighbor_similarities, neighbor_labels)):
                similarities += similarities[:1]  # 完成圆形
                ax.plot(angles, similarities, 'o-', linewidth=2, label=label, color=colors[i % len(colors)])
                ax.fill(angles, similarities, alpha=0.25, color=colors[i % len(colors)])
            
            # 设置图表属性
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feature_labels, fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_title('邻居特征相似性雷达图\n（展示新实例与Top-K邻居在不同特征维度的相似性）', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=9)
            ax.grid(True)
            
            # 添加说明文字
            fig.text(0.02, 0.02, 
                    f'说明：图中显示了新实例与前{len(neighbor_similarities)}个最近邻居的特征相似性\n' +
                    '相似度范围：0（完全不同）到 1（完全相同）',
                    fontsize=10, alpha=0.7)
            
            # 保存图表
            radar_plot_path = os.path.join(output_dir, 'neighbors_similarity_radar.png')
            plt.savefig(radar_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_info(f"邻居特征相似性雷达图已保存: {radar_plot_path}")
            
        except Exception as e:
            self.log_error(f"创建雷达图失败: {e}")
    
    def _create_neighbors_influence_scatter(self, strategy_performance_detail, output_dir):
        """创建邻居影响力与性能散点图"""
        try:
            if not strategy_performance_detail:
                self.log_warning("缺少策略性能数据，跳过散点图生成")
                return
            
            # 准备数据
            distance_weights = [detail['distance_weight'] for detail in strategy_performance_detail]
            performance_weights = [detail['performance_weight'] for detail in strategy_performance_detail]
            combined_weights = [detail['combined_weight'] for detail in strategy_performance_detail]
            strategy_names = [detail['strategy_name'] for detail in strategy_performance_detail]
            sample_ids = [detail['sample_id'] for detail in strategy_performance_detail]
            
            # 定义颜色映射
            strategy_colors = {
                'heuristic': '#e74c3c',  # 红色
                'mixed': '#3498db',      # 蓝色
                'random': '#2ecc71'      # 绿色
            }
            
            # 创建散点图
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 为每种策略绘制散点
            for strategy in ['heuristic', 'mixed', 'random']:
                # 筛选该策略的数据
                strategy_indices = [i for i, s in enumerate(strategy_names) if s == strategy]
                
                if strategy_indices:
                    x_values = [distance_weights[i] for i in strategy_indices]
                    y_values = [performance_weights[i] for i in strategy_indices]
                    sizes = [combined_weights[i] * 200 for i in strategy_indices]  # 气泡大小
                    
                    scatter = ax.scatter(x_values, y_values, s=sizes, 
                                       c=strategy_colors[strategy], alpha=0.7, 
                                       label=f'{strategy} ({len(strategy_indices)}个邻居)',
                                       edgecolors='black', linewidth=0.5)
            
            # 设置图表属性
            ax.set_xlabel('距离权重 (1/欧氏距离)\n→ 值越大代表距离越近，影响力越大', fontsize=12)
            ax.set_ylabel('性能权重 (1/makespan)\n→ 值越大代表历史性能越好，影响力越大', fontsize=12)
            ax.set_title(f'Top-{len(strategy_performance_detail)} 邻居影响力与性能散点图\n' +
                        '气泡大小 = 综合影响力 (距离权重 × 性能权重)', 
                        fontsize=14, fontweight='bold')
            
            # 添加网格和图例
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(loc='upper left', fontsize=10)
            
            # 添加象限说明
            ax.axhline(y=np.median(performance_weights), color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=np.median(distance_weights), color='gray', linestyle=':', alpha=0.5)
            
            # 在右上角添加说明文字
            max_x, max_y = ax.get_xlim()[1], ax.get_ylim()[1]
            ax.text(max_x * 0.95, max_y * 0.95, '高质量邻居区域\n(距离近 + 性能好)', 
                   ha='right', va='top', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
            
            # 添加数据点标注（显示sample_id和文件名）
            for i, detail in enumerate(strategy_performance_detail):
                if i < len(strategy_performance_detail):  # 标注所有邻居
                    x = detail['distance_weight']
                    y = detail['performance_weight']
                    sample_id = detail['sample_id']
                    
                    # 获取文件名简短版本
                    original_path = detail['original_fjs_path']
                    file_name = original_path.split('/')[-1] if '/' in original_path else original_path
                    file_name = file_name.replace('.fjs', '')  # 去掉扩展名
                    
                    # 使用偏移避免重叠
                    offset_x = 5 + (i % 3) * 15  # 水平偏移
                    offset_y = 5 + (i // 3) * 10  # 垂直偏移
                    
                    ax.annotate(f'{sample_id}\n{file_name}', (x, y), 
                               xytext=(offset_x, offset_y), 
                               textcoords='offset points', fontsize=8, alpha=0.8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', alpha=0.5))
            
            plt.tight_layout()
            
            # 保存图表
            scatter_plot_path = os.path.join(output_dir, 'neighbors_influence_scatter.png')
            plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_info(f"邻居影响力散点图已保存: {scatter_plot_path}")
            
            # 生成数据分析报告
            self._generate_influence_analysis_report(strategy_performance_detail, output_dir)
            
        except Exception as e:
            self.log_error(f"创建散点图失败: {e}")
    
    def _generate_influence_analysis_report(self, strategy_performance_detail, output_dir):
        """生成影响力分析报告"""
        try:
            report_lines = []
            report_lines.append("# 邻居影响力分析报告\n")
            
            # 按策略分组统计
            strategy_stats = {}
            for detail in strategy_performance_detail:
                strategy = detail['strategy_name']
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = []
                strategy_stats[strategy].append(detail)
            
            report_lines.append("## 策略分布统计\n")
            for strategy, details in strategy_stats.items():
                count = len(details)
                avg_distance_weight = np.mean([d['distance_weight'] for d in details])
                avg_performance_weight = np.mean([d['performance_weight'] for d in details])
                avg_combined_weight = np.mean([d['combined_weight'] for d in details])
                
                report_lines.append(f"### {strategy} 策略")
                report_lines.append(f"- 邻居数量: {count}")
                report_lines.append(f"- 平均距离权重: {avg_distance_weight:.4f}")
                report_lines.append(f"- 平均性能权重: {avg_performance_weight:.4f}")
                report_lines.append(f"- 平均综合影响力: {avg_combined_weight:.4f}")
                report_lines.append("")
            
            # 找出高影响力邻居
            sorted_details = sorted(strategy_performance_detail, 
                                  key=lambda x: x['combined_weight'], reverse=True)
            
            report_lines.append(f"## 所有 {len(strategy_performance_detail)} 个邻居详细信息\n")
            for i, detail in enumerate(sorted_details, 1):
                file_path = detail['original_fjs_path']
                file_name = file_path.split('/')[-1] if '/' in file_path else file_path
                
                report_lines.append(f"### 邻居 {i}: {detail['sample_id']}")
                report_lines.append(f"- **策略**: {detail['strategy_name']}")
                report_lines.append(f"- **文件**: {file_name} ({file_path})")
                report_lines.append(f"- **距离**: {detail['distance']:.4f}")
                report_lines.append(f"- **Makespan**: {detail['mean_makespan']:.2f}")
                report_lines.append(f"- **距离权重**: {detail['distance_weight']:.6f}")
                report_lines.append(f"- **性能权重**: {detail['performance_weight']:.6f}")
                report_lines.append(f"- **综合影响力**: {detail['combined_weight']:.6f}")
                report_lines.append("")
            
            report_lines.append("## Top 5 高影响力邻居汇总\n")
            for i, detail in enumerate(sorted_details[:5], 1):
                file_name = detail['original_fjs_path'].split('/')[-1] if '/' in detail['original_fjs_path'] else detail['original_fjs_path']
                report_lines.append(f"{i}. **{detail['sample_id']}** - {file_name} ({detail['strategy_name']})")
                report_lines.append(f"   - 综合影响力: {detail['combined_weight']:.6f}")
                report_lines.append("")
            
            # 保存报告
            report_path = os.path.join(output_dir, 'influence_analysis_report.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            self.log_info(f"影响力分析报告已保存: {report_path}")
            
        except Exception as e:
            self.log_error(f"生成分析报告失败: {e}")


def main():
    """主函数：推荐系统命令行接口"""
    import sys
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='初始化策略推荐系统 - 支持细化特征权重')
    parser.add_argument('fjs_file', help='输入的FJS文件路径')
    parser.add_argument('--k-neighbors', type=int, default=5, help='KNN算法中K的值，即邻居数量 (默认: 5)')
    parser.add_argument('--top-k-strategies', type=int, default=3, help='推荐的策略数量 (默认: 3)')
    parser.add_argument('--output-dir', default='result/recommender_output', help='输出目录 (默认: result/recommender_output)')
    parser.add_argument('--weights-config', type=str, default=None, help='细化权重配置文件路径 (JSON格式)')
    
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
    print(f"KNN邻居数量: {getattr(args, 'k_neighbors', getattr(args, 'k-neighbors', 5))}")
    print(f"Top-K推荐策略: {args.top_k_strategies}")
    print("=" * 80)
    
    try:
        # 加载权重配置（如果提供）
        detailed_weights = None
        if args.weights_config:
            if os.path.exists(args.weights_config):
                try:
                    import json
                    with open(args.weights_config, 'r', encoding='utf-8') as f:
                        weights_config = json.load(f)
                    detailed_weights = weights_config.get('weights', None)
                    print(f"✅ 已加载细化权重配置: {args.weights_config}")
                except Exception as e:
                    print(f"⚠️ 权重配置文件加载失败: {e}")
                    print("使用默认权重配置")
            else:
                print(f"⚠️ 权重配置文件不存在: {args.weights_config}")
                print("使用默认权重配置")
        
        # 初始化推荐系统（带日志和权重配置）
        labeled_dataset_path = "labeled_dataset/converted_fjs_dataset_new.json"
        log_file = os.path.join(full_output_dir, f"recommendation_log.log")
        recommender = InitializationStrategyRecommender(labeled_dataset_path, log_file, detailed_weights)
        
        # 提取新数据特征
        from extract_new_data_features import extract_new_data_features
        
        recommender.log_info(f"开始提取新数据特征: {args.fjs_file}")
        new_data_features = extract_new_data_features(args.fjs_file)
        
        if new_data_features is None:
            recommender.log_error("新数据特征提取失败")
            return
        
        recommender.log_info(f"新数据特征提取完成")
        
        # 执行推荐
        k_neighbors_value = getattr(args, 'k_neighbors', getattr(args, 'k-neighbors', 5))
        results = recommender.recommend(new_data_features, 
                                      k_neighbors=k_neighbors_value, 
                                      top_k_strategies=args.top_k_strategies)
        
        # 保存推荐结果
        output_file = os.path.join(full_output_dir, "recommendation_results.json")
        recommender.save_results(results, output_file)
        
        # 可视化结果
        visualization_dir = os.path.join(full_output_dir, "visualization")
        recommender.visualize_recommendation_results(results, visualization_dir)
        
        # 推荐完成
        print("\n" + "=" * 80)
        print("KNN推荐系统运行完成")
        print("=" * 80)
        # KNN搜索结果摘要
        knn_data = results['knn_search_results']['neighbors']
        print(f"✅ 找到 {len(knn_data)} 个最近邻居")
        
        # 策略推荐结果摘要
        strategy_data = results['strategy_recommendation_results']['recommended_strategies']
        print(f"✅ 推荐 {len(strategy_data)} 个初始化策略")
        print(f"   最佳策略: {strategy_data[0]['strategy_name']} (评分: {strategy_data[0]['weighted_score']:.4f})")
        
        print("\n详细结果请查看以下文件：")
        
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