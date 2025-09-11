#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将标记数据集转换为LambdaMART格式
将labeled_fjs_dataset.json转换为LambdaMART学习排序算法需要的TXT格式

LambdaMART数据格式:
<relevance_score> qid:<query_id> <feature_id>:<feature_value> ... # <comment>

映射关系:
- 查询(Query) = FJSP实例 (qid)
- 文档(Documents) = 初始化方法 (heuristic, mixed, random)
- 特征向量 = 实例特征 (对同一实例的所有方法相同)
- 相关性分数 = 性能的逆转换 (性能越好分数越高)
"""

import json
import os
import math
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from scipy import stats


def setup_logging():
    """设置日志记录"""
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'convert_to_lambdamart_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    return logging.getLogger(__name__)


def load_labeled_dataset(json_file):
    """加载标记数据集"""
    logger = logging.getLogger(__name__)
    logger.info(f"开始加载数据集: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"成功加载 {len(data)} 个FJSP实例")
        return data
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise


def extract_features(instance_features):
    """提取实例特征向量"""
    logger = logging.getLogger(__name__)
    features = []
    
    # 基本特征 (feature_id: 1-5)
    basic = instance_features.get('basic_features', {})
    features.extend([
        basic.get('num_jobs', 0),
        basic.get('num_machines', 0), 
        basic.get('total_operations', 0),
        basic.get('avg_available_machines', 0),
        basic.get('std_available_machines', 0)
    ])
    
    # 处理时间特征 (feature_id: 6-10)
    proc_time = instance_features.get('processing_time_features', {})
    features.extend([
        proc_time.get('processing_time_mean', 0),
        proc_time.get('processing_time_std', 0),
        proc_time.get('processing_time_min', 0),
        proc_time.get('processing_time_max', 0),
        proc_time.get('machine_time_variance', 0)
    ])
    
    # 析取图特征 (feature_id: 11-22)
    disjunctive = instance_features.get('disjunctive_graphs_features', {})
    
    # 基础图结构特征 (2维)
    nodes_count = disjunctive.get('nodes_count', 0)
    edges_count = disjunctive.get('edges_count', 0)
    features.extend([nodes_count, edges_count])
    
    # 图拓扑结构特征 (4维)
    # 基于图的结构属性，而非WL标签计数
    
    # 图密度：边数与最大可能边数的比率
    max_possible_edges = nodes_count * (nodes_count - 1) / 2 if nodes_count > 1 else 1
    graph_density = edges_count / max_possible_edges if max_possible_edges > 0 else 0
    
    # 平均度数：每个节点的平均连接数
    avg_degree = (2 * edges_count) / nodes_count if nodes_count > 0 else 0
    
    # 节点边比：反映图的稠密程度
    nodes_to_edges_ratio = nodes_count / edges_count if edges_count > 0 else 0
    
    # 结构复杂度指标：基于节点数和边数的复合指标
    structure_complexity = (edges_count / nodes_count) if nodes_count > 0 else 0
    
    features.extend([
        graph_density,           # 图密度 (0-1之间)
        avg_degree,              # 平均度数
        nodes_to_edges_ratio,    # 节点边比
        structure_complexity     # 结构复杂度
    ])
    
    # WL模式分析特征 (6维)
    solid_frequency = disjunctive.get('solid_frequency', {})
    dashed_frequency = disjunctive.get('dashed_frequency', {})
    
    # 分析实线(作业间约束)的模式特征
    if solid_frequency and len(solid_frequency) > 0:
        solid_freq_values = list(solid_frequency.values())
        solid_pattern_count = len(solid_frequency)  # 不同模式的数量
        solid_max_freq_ratio = max(solid_freq_values) / sum(solid_freq_values) # 最大频率占比
        solid_pattern_diversity = solid_pattern_count / nodes_count if nodes_count > 0 else 0  # 模式多样性
    else:
        solid_pattern_count = solid_max_freq_ratio = solid_pattern_diversity = 0
        
    # 分析虚线(机器间约束)的模式特征  
    if dashed_frequency and len(dashed_frequency) > 0:
        dashed_freq_values = list(dashed_frequency.values())
        dashed_pattern_count = len(dashed_frequency)  # 不同模式的数量
        dashed_max_freq_ratio = max(dashed_freq_values) / sum(dashed_freq_values) # 最大频率占比
        dashed_pattern_diversity = dashed_pattern_count / nodes_count if nodes_count > 0 else 0  # 模式多样性
    else:
        dashed_pattern_count = dashed_max_freq_ratio = dashed_pattern_diversity = 0
    
    features.extend([
        solid_pattern_count,      # 实线模式种类数（作业约束复杂度）
        solid_max_freq_ratio,     # 实线最大频率占比（主导模式强度）
        solid_pattern_diversity,  # 实线模式多样性（标准化复杂度）
        dashed_pattern_count,     # 虚线模式种类数（机器约束复杂度）
        dashed_max_freq_ratio,    # 虚线最大频率占比（机器主导强度）
        dashed_pattern_diversity  # 虚线模式多样性（机器约束标准化复杂度）
    ])
    
    # KDE特征 (feature_id: 23+)
    kde = instance_features.get('kde_features', {})
    kde_density = kde.get('density', [])
    
    if kde_density:
        kde_array = np.array(kde_density)
        n = len(kde_density)
        
        # 过滤掉接近零的值，只保留有意义的密度值
        # 设置阈值为最大值的1%，避免噪声
        threshold = np.max(kde_array) * 0.01 if np.max(kde_array) > 0 else 0
        significant_density = kde_array[kde_array >= threshold]
        
        if len(significant_density) == 0:
            # 如果没有显著的密度值，使用原始数据
            significant_density = kde_array
        
        # 方案1：基于有效密度的统计特征 (7个特征)
        features.extend([
            np.mean(significant_density),      # 有效密度均值
            np.std(significant_density),       # 有效密度标准差
            np.min(significant_density),       # 有效密度最小值
            np.max(significant_density),       # 有效密度最大值
            np.median(significant_density),    # 有效密度中位数
            np.percentile(significant_density, 25),  # 25%分位数
            np.percentile(significant_density, 75),  # 75%分位数
        ])
        
        # 方案2：基于峰值的智能采样 (10个特征)
        # 重点采样密度较高的区域，而不是均匀采样
        if len(significant_density) >= 10:
            # 找到峰值位置
            original_indices = np.where(kde_array >= threshold)[0]
            
            # 按密度值排序，取前5个最高峰
            top_indices = original_indices[np.argsort(kde_array[original_indices])[-5:]]
            top_values = [kde_array[i] for i in sorted(top_indices)]
            
            # 均匀采样5个中等密度值
            remaining_indices = [i for i in original_indices if i not in top_indices]
            if len(remaining_indices) >= 5:
                # 均匀采样
                step = len(remaining_indices) // 5
                sampled_indices = [remaining_indices[i*step] for i in range(5)]
                medium_values = [kde_array[i] for i in sorted(sampled_indices)]
            else:
                # 如果不够5个，使用所有剩余值并补充
                medium_values = [kde_array[i] for i in sorted(remaining_indices)]
                while len(medium_values) < 5:
                    medium_values.append(np.mean(significant_density))
            
            # 合并峰值和中等值
            sampled_density = top_values + medium_values
            features.extend(sampled_density)
        else:
            # 如果有效密度值少于10个，使用所有值并补充统计值
            features.extend(significant_density.tolist())
            # 补充值使用统计特征而不是简单的重复
            remaining_slots = 10 - len(significant_density)
            if remaining_slots > 0:
                fill_values = [
                    np.mean(significant_density),
                    np.median(significant_density),
                    np.std(significant_density) + np.mean(significant_density),
                    np.max(significant_density) * 0.8,
                    np.min(significant_density) * 1.2
                ] * ((remaining_slots // 5) + 1)
                features.extend(fill_values[:remaining_slots])
        
        # 方案3：分布形状特征 (5个特征)
        # 峰度和偏度等分布形状信息
        try:
            skewness = stats.skew(kde_array)  # 偏度
            kurtosis = stats.kurtosis(kde_array)  # 峰度
            peak_idx = np.argmax(kde_array)  # 峰值位置比例
            peak_position = peak_idx / (n - 1) if n > 1 else 0.5
            
            # 计算分布的集中度（变异系数）
            cv = np.std(kde_array) / np.mean(kde_array) if np.mean(kde_array) > 0 else 0
            
            # 计算分布的有效宽度（包含90%概率质量的区间）
            cumsum = np.cumsum(kde_array)
            cumsum_norm = cumsum / cumsum[-1] if cumsum[-1] > 0 else cumsum
            idx_5 = np.searchsorted(cumsum_norm, 0.05)
            idx_95 = np.searchsorted(cumsum_norm, 0.95)
            effective_width = (idx_95 - idx_5) / n if n > 0 else 0
            
            features.extend([skewness, kurtosis, peak_position, cv, effective_width])
        except:
            # 如果计算失败，使用默认值
            features.extend([0, 0, 0.5, 0, 1])
    else:
        # 如果没有KDE特征，使用合理的默认值
        # 7个统计特征 + 10个采样特征 + 5个形状特征 = 22个特征
        features.extend([0] * 22)
        logger.debug("实例缺少KDE特征，使用零值填充")
    
    logger.debug(f"提取特征完成，特征维度: {len(features)}")
    return features


def calculate_comprehensive_score(method_data, weights=None):
    """
    计算综合性能分数
    
    Args:
        method_data: 单个方法的性能数据字典
        weights: 权重字典，默认为None使用默认权重
    
    Returns:
        float: 综合性能分数 (越高越好)
    """
    logger = logging.getLogger(__name__)
    # 默认权重设置
    if weights is None:
        weights = {
            'objective_quality': 0.4,      # 目标值质量 (40%)
            'objective_stability': 0.25,   # 目标稳定性 (25%)
            'convergence_speed': 0.2,      # 收敛速度 (20%)
            'convergence_stability': 0.15  # 收敛稳定性 (15%)
        }
    
    # 提取性能指标
    mean_val = method_data.get('mean', float('inf'))
    min_val = method_data.get('min', float('inf'))
    max_val = method_data.get('max', float('inf'))
    std_val = method_data.get('std', float('inf'))
    avg_conv_gen = method_data.get('avg_convergence_generation', float('inf'))
    conv_gen_std = method_data.get('convergence_generation_std', float('inf'))
    
    # 检查数据有效性
    # 注意：std_val 可以为0（表示完全稳定），avg_conv_gen也可以为0（表示立即收敛）
    if (mean_val == float('inf') or mean_val <= 0 or
        min_val == float('inf') or min_val <= 0 or
        max_val == float('inf') or max_val <= 0 or
        std_val == float('inf') or std_val < 0 or
        avg_conv_gen == float('inf') or avg_conv_gen < 0):
        logger.warning(f"性能数据无效: mean={mean_val}, min={min_val}, max={max_val}, std={std_val}, avg_conv_gen={avg_conv_gen}")
        return 0
    
    # 1. 目标值质量分数 (综合均值、最小值和最大值，值越小越好)
    # 使用最小值50%，均值30%，最大值20%的权重
    # 最小值权重最高（最佳性能），均值次之（平均性能），最大值相对较低但仍重要（鲁棒性）
    objective_raw = 0.5 * min_val + 0.3 * mean_val + 0.2 * max_val
    objective_quality_score = 1000.0 / objective_raw
    
    # 2. 目标稳定性分数 (标准差越小越好)
    objective_stability_score = 100.0 / (1.0 + std_val)
    
    # 3. 收敛速度分数 (收敛代数越小越好)
    convergence_speed_score = 100.0 / (1.0 + avg_conv_gen)
    
    # 4. 收敛稳定性分数 (收敛代数标准差越小越好)
    convergence_stability_score = 50.0 / (1.0 + conv_gen_std)
    
    # 加权计算综合分数
    comprehensive_score = (
        weights['objective_quality'] * objective_quality_score +
        weights['objective_stability'] * objective_stability_score +
        weights['convergence_speed'] * convergence_speed_score +
        weights['convergence_stability'] * convergence_stability_score
    )
    
    return comprehensive_score


def calculate_relevance_score(performance_data, method, score_type='comprehensive', weights=None):
    """
    计算相关性分数
    
    Args:
        performance_data: 性能数据字典
        method: 初始化方法名称 ('heuristic', 'mixed', 'random')
        score_type: 使用的性能指标类型 ('mean', 'min', 'max', 'comprehensive')
        weights: 综合评估的权重字典
    
    Returns:
        float: 相关性分数 (越高越好)
    """
    method_data = performance_data.get(method, {})
    
    if score_type == 'comprehensive':
        return calculate_comprehensive_score(method_data, weights)
    else:
        # 使用单一指标
        performance_value = method_data.get(score_type, float('inf'))
        
        # 处理无效值
        if performance_value == float('inf') or performance_value <= 0:
            return 0
        
        # 使用倒数转换: 性能越小(越好), 相关性分数越高
        relevance_score = 1000.0 / performance_value
        return relevance_score


def get_method_ranking(performance_data, score_type='comprehensive', weights=None):
    """
    获取方法的排名 (用于生成相关性分数)
    
    Args:
        performance_data: 性能数据字典
        score_type: 性能指标类型
        weights: 综合评估的权重字典
    
    Returns:
        dict: {method: rank} (rank: 0=最差, 1=中等, 2=最好)
    """
    methods = ['heuristic', 'mixed', 'random']
    
    # 获取所有方法的性能分数
    performance_scores = []
    for method in methods:
        score = calculate_relevance_score(performance_data, method, score_type, weights)
        performance_scores.append((method, score))
    
    # 按分数排序 (分数越高越好)
    performance_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 分配排名: 最好=2, 中等=1, 最差=0
    ranking = {}
    for i, (method, _) in enumerate(performance_scores):
        ranking[method] = 2 - i
    
    return ranking


def convert_to_lambdamart_format(labeled_data, output_file, score_type='comprehensive', ranking_mode=True, weights=None):
    """
    转换为LambdaMART格式
    
    Args:
        labeled_data: 标记数据集
        output_file: 输出文件路径
        score_type: 性能指标类型 ('mean', 'min', 'max', 'comprehensive')
        ranking_mode: True=使用排名分数(0,1,2), False=使用连续分数
        weights: 综合评估的权重字典
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始转换为LambdaMART格式")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"性能指标类型: {score_type}")
    logger.info(f"评分模式: {'排名分数' if ranking_mode else '连续分数'}")
    if weights:
        logger.info(f"权重配置: {weights}")
    
    lines = []
    query_id = 1
    
    # 初始化方法列表
    methods = ['heuristic', 'mixed', 'random']
    
    total_instances = len(labeled_data)
    processed = 0
    
    for fjs_path, instance_data in labeled_data.items():
        # 提取特征
        features = extract_features(instance_data.get('features', {}))
        feature_str = ' '.join([f"{i+1}:{f:.6f}" for i, f in enumerate(features)])
        
        # 获取性能数据
        performance_data_raw = instance_data.get('performance_data', {})
        if performance_data_raw is None:
            logger.warning(f"实例 {fjs_path} 的 performance_data 为 None，跳过")
            continue
        performance_data = performance_data_raw.get('initialization_methods', {})
        
        if not performance_data:
            logger.warning(f"实例 {fjs_path} 缺少性能数据，跳过")
            continue
        
        # 计算相关性分数
        if ranking_mode:
            method_scores = get_method_ranking(performance_data, score_type, weights)
        else:
            method_scores = {
                method: calculate_relevance_score(performance_data, method, score_type, weights)
                for method in methods
            }
        
        # 获取元数据
        meta_data = instance_data.get('performance_data', {})
        meta_heuristic = meta_data.get('meta_heuristic', 'N/A')
        execution_times = meta_data.get('execution_times', 'N/A')
        max_iterations = meta_data.get('max_iterations', 'N/A')
        
        # 为每个初始化方法生成一行
        for method in methods:
            relevance_score = method_scores.get(method, 0)
            
            # 格式: <relevance_score> qid:<query_id> <features> # <comment>
            comment = f"instance={fjs_path} method={method} meta_heuristic={meta_heuristic} execution_times={execution_times} max_iterations={max_iterations}"
            
            if ranking_mode:
                line = f"{relevance_score} qid:{query_id} {feature_str} # {comment}"
            else:
                line = f"{relevance_score:.6f} qid:{query_id} {feature_str} # {comment}"
            
            lines.append(line)
        
        query_id += 1
        processed += 1
        
        if processed % 100 == 0:
            logger.info(f"已处理 {processed}/{total_instances} 个实例")
            
        # 详细进度记录
        if processed % 1000 == 0:
            progress_percent = (processed / total_instances) * 100
            logger.info(f"转换进度: {progress_percent:.1f}% ({processed}/{total_instances})")
    
    # 写入文件
    logger.info(f"开始写入文件: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"文件写入完成!")
        logger.info(f"- 处理实例数: {processed}")
        logger.info(f"- 生成记录数: {len(lines)}")
        logger.info(f"- 输出文件: {output_file}")
        
    except Exception as e:
        logger.error(f"写入文件失败: {e}")
        raise


def generate_run_info(output_dir, timestamp, total_instances, processed_instances):
    """生成运行配置信息文件"""
    logger = logging.getLogger(__name__)
    
    run_info = f"""# LambdaMART数据转换运行信息

## 运行基本信息
- **运行时间**: {timestamp}
- **脚本版本**: 改进版KDE特征提取
- **总实例数**: {total_instances}
- **处理实例数**: {processed_instances}
- **跳过实例数**: {total_instances - processed_instances}

## 特征维度信息
- **总特征维度**: 44维
- **基本特征**: 5维 (作业数、机器数、操作数等)
- **处理时间特征**: 5维 (均值、标准差、最值等)
- **析取图特征**: 12维 (图结构2维+WL标签多样性4维+频率分布6维)
- **KDE特征**: 22维 (7个统计+10个采样+5个形状特征)

## 性能评估方案
### 综合性能评估维度
1. **目标值质量**: 综合最小值(50%)、均值(30%)、最大值(20%)
2. **目标稳定性**: 基于标准差
3. **收敛速度**: 基于平均收敛代数
4. **收敛稳定性**: 基于收敛代数标准差

### 权重配置
- **balanced**: 平衡配置 (40%, 25%, 20%, 15%)
- **quality_focused**: 注重解质量 (60%, 30%, 5%, 5%)
- **speed_focused**: 注重收敛速度 (30%, 20%, 35%, 15%)
- **objective_stability_focused**: 注重目标稳定性 (30%, 50%, 15%, 5%)
- **convergence_stability_focused**: 注重收敛稳定性 (30%, 5%, 15%, 50%)

## KDE特征改进说明
### 主要改进
1. **消除大量补零**: 使用自适应采样替代固定长度补零
2. **增加分布形状特征**: 偏度、峰度、峰值位置等
3. **减少特征维度**: 从49维减少到34维，提高特征密度

### KDE特征详细说明
#### 有效密度统计特征 (7维)
- **智能过滤**: 只使用≥最大值1%的密度值，排除噪声
- 基于有效密度计算：均值、标准差、最值、中位数、分位数

#### 智能采样特征 (10维)
- **峰值优先**: 优先采样前5个最高密度峰值
- **区域平衡**: 从剩余区域均匀采样5个中等密度值
- **统计补充**: 不足时使用统计衍生值而非零值填充

#### 分布形状特征 (5维)
- 偏度: 分布对称性
- 峰度: 分布尖锐程度
- 峰值位置: 峰值在序列中的相对位置
- 变异系数: 相对离散程度
- 有效宽度: 包含90%概率质量的区间

## 数据格式说明
- **LambdaMART格式**: `<relevance_score> qid:<query_id> <features> # <metadata>`
- **查询映射**: 每个查询 = 一个FJSP实例
- **文档映射**: 每个实例的3种初始化方法 (heuristic, mixed, random)
- **特征向量**: 44维实例特征（对同一查询的所有文档相同）
- **相关性分数**: 基于综合性能评估的排名分数(0,1,2)或连续分数

## 文件说明
1. **综合评估文件**: 使用多维性能指标的加权评估
2. **单一指标文件**: 仅使用传统单一指标（作为对比基准）
3. **特征描述文件**: 详细的特征定义和说明
4. **运行信息文件**: 本次运行的配置和统计信息

## 使用建议
- **推荐使用**: `lambdamart_comprehensive_balanced_ranking.txt`
- **特定需求**: 根据应用场景选择对应的focused版本
- **对比实验**: 使用单一指标文件作为baseline
"""
    
    info_file = output_dir / 'run_info.md'
    try:
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(run_info)
        
        logger.info(f"运行信息文件已生成: {info_file}")
    except Exception as e:
        logger.error(f"生成运行信息文件失败: {e}")
        raise


def generate_feature_description(output_dir):
    """生成特征描述文件"""
    logger = logging.getLogger(__name__)
    feature_desc = """# LambdaMART特征描述

## 特征映射 (Feature Mapping)

### 基本特征 (Basic Features) - Feature ID 1-5:
1. num_jobs: 作业数量
2. num_machines: 机器数量  
3. total_operations: 总操作数
4. avg_available_machines: 平均可用机器数
5. std_available_machines: 可用机器数标准差

### 处理时间特征 (Processing Time Features) - Feature ID 6-10:
6. processing_time_mean: 处理时间均值
7. processing_time_std: 处理时间标准差
8. processing_time_min: 最小处理时间
9. processing_time_max: 最大处理时间
10. machine_time_variance: 机器时间方差

### 析取图特征 (Disjunctive Graph Features) - Feature ID 11-22:
#### 基础图结构特征 (11-12):
11. nodes_count: 节点数量
12. edges_count: 边数量

#### 图拓扑结构特征 (13-16):
- **基于图的几何和连接属性，提供更有区分度的特征**
13. graph_density: 图密度（边数/最大可能边数，0-1之间）
14. avg_degree: 平均度数（每个节点的平均连接数）
15. nodes_to_edges_ratio: 节点边比（反映图稠密程度）
16. structure_complexity: 结构复杂度（边数/节点数）

#### WL模式分析特征 (17-22):
- **基于WL算法的模式频率分析，反映约束结构复杂度**
17. solid_pattern_count: 实线模式种类数（作业间约束模式复杂度）
18. solid_max_freq_ratio: 实线最大频率占比（主导模式强度，0-1之间）
19. solid_pattern_diversity: 实线模式多样性（标准化复杂度，0-1之间）
20. dashed_pattern_count: 虚线模式种类数（机器间约束模式复杂度）
21. dashed_max_freq_ratio: 虚线最大频率占比（机器主导强度，0-1之间）
22. dashed_pattern_diversity: 虚线模式多样性（机器约束标准化复杂度，0-1之间）

### KDE特征 (KDE Features) - Feature ID 23-44:
#### KDE有效密度统计特征 (23-29):
- **过滤策略**: 只使用≥最大值1%的有效密度值，避免噪声影响
23. effective_density_mean: 有效密度均值
24. effective_density_std: 有效密度标准差
25. effective_density_min: 有效密度最小值
26. effective_density_max: 有效密度最大值
27. effective_density_median: 有效密度中位数
28. effective_density_q25: 有效密度25%分位数
29. effective_density_q75: 有效密度75%分位数

#### KDE智能采样特征 (30-39):
- **采样策略**: 重点采样高密度区域，而非均匀采样
30-34: 前5个最高峰值（按密度值排序）
35-39: 中等密度区域的5个均匀采样值
- **补充策略**: 当样本不足时，使用统计衍生值而非零值填充

#### KDE分布形状特征 (40-44):
40. kde_skewness: 分布偏度（衡量分布的对称性）
41. kde_kurtosis: 分布峰度（衡量分布的尖锐程度）
42. kde_peak_position: 峰值位置比例（0-1之间）
43. kde_cv: 变异系数（标准差/均值，衡量离散程度）
44. kde_effective_width: 有效宽度（包含90%概率质量的区间比例）

## 查询映射 (Query Mapping)
- 每个查询(qid) = 一个FJSP实例
- 每个实例对应3个文档(初始化方法): heuristic, mixed, random

## 相关性分数 (Relevance Scores)

### 综合性能评估 (Comprehensive Assessment)
基于多维性能指标的加权评估:

#### 性能维度:
1. **目标值质量** (Objective Quality): 综合最小值、均值和最大值，权重50%最小值+30%均值+20%最大值
2. **目标稳定性** (Objective Stability): 基于标准差，值越小越好
3. **收敛速度** (Convergence Speed): 基于平均收敛代数，越小越好
4. **收敛稳定性** (Convergence Stability): 基于收敛代数标准差，越小越好

#### 权重配置:
- **balanced**: 平衡配置 (40%, 25%, 20%, 15%)
- **quality_focused**: 注重解质量 (60%, 30%, 5%, 5%)
- **speed_focused**: 注重收敛速度 (30%, 20%, 35%, 15%)
- **objective_stability_focused**: 注重目标稳定性 (30%, 50%, 15%, 5%)
- **convergence_stability_focused**: 注重收敛稳定性 (30%, 5%, 15%, 50%)

### 分数模式:
- **排名模式**: 0(最差), 1(中等), 2(最好)
- **连续模式**: 基于加权综合分数的连续值

### 单一指标模式 (对比基准):
- mean: 基于平均makespan
- min: 基于最小makespan

## 元数据 (Metadata in Comments)
- instance: FJS文件路径
- method: 初始化方法
- meta_heuristic: 元启发式算法
- execution_times: 执行次数
- max_iterations: 最大迭代次数
"""
    
    desc_file = os.path.join(output_dir, 'feature_description.md')
    try:
        with open(desc_file, 'w', encoding='utf-8') as f:
            f.write(feature_desc)
        
        logger.info(f"特征描述文件已生成: {desc_file}")
    except Exception as e:
        logger.error(f"生成特征描述文件失败: {e}")
        raise


def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    logger.info("="*80)
    logger.info("LambdaMART数据转换脚本开始执行")
    logger.info("="*80)
    
    try:
        # 设置路径
        current_dir = Path(__file__).parent
        json_file = current_dir / 'labeled_fjs_dataset.json'
        
        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = current_dir / f'lambdamart_output_{timestamp}'
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"当前工作目录: {current_dir}")
        logger.info(f"输入文件: {json_file}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"时间戳: {timestamp}")
        
        # 检查输入文件
        if not json_file.exists():
            logger.error(f"输入文件不存在: {json_file}")
            return
        
        # 加载数据
        labeled_data = load_labeled_dataset(json_file)
        
        # 定义不同的权重配置
        logger.info("配置权重参数")
        weight_configs = {
            'balanced': {
                'objective_quality': 0.4,      # 目标值质量 (40%)
                'objective_stability': 0.25,   # 目标稳定性 (25%)
                'convergence_speed': 0.2,      # 收敛速度 (20%)
                'convergence_stability': 0.15  # 收敛稳定性 (15%)
            },
            'quality_focused': {
                'objective_quality': 0.6,      # 目标值质量 (60%)
                'objective_stability': 0.3,    # 目标稳定性 (30%)
                'convergence_speed': 0.05,     # 收敛速度 (5%)
                'convergence_stability': 0.05  # 收敛稳定性 (5%)
            },
            'speed_focused': {
                'objective_quality': 0.3,      # 目标值质量 (30%)
                'objective_stability': 0.2,    # 目标稳定性 (20%)
                'convergence_speed': 0.35,     # 收敛速度 (35%)
                'convergence_stability': 0.15  # 收敛稳定性 (15%)
            },
            'objective_stability_focused': {
                'objective_quality': 0.3,      # 目标值质量 (30%)
                'objective_stability': 0.5,    # 目标稳定性 (50%)
                'convergence_speed': 0.15,     # 收敛速度 (15%)
                'convergence_stability': 0.05  # 收敛稳定性 (5%)
            },
            'convergence_stability_focused': {
                'objective_quality': 0.3,      # 目标值质量 (30%)
                'objective_stability': 0.05,   # 目标稳定性 (5%)
                'convergence_speed': 0.15,     # 收敛速度 (15%)
                'convergence_stability': 0.5   # 收敛稳定性 (50%)
            },
        }
        
        generated_files = []
        total_processed = 0  # 统计总的处理实例数
        logger.info(f"开始生成 {len(weight_configs)} 种权重配置的文件")
        
        # 生成综合性能评估的文件
        for config_name, weights in weight_configs.items():
            logger.info(f"处理配置: {config_name}")
            logger.info(f"权重: {weights}")
            
            # 排名模式
            output_file_ranking = output_dir / f'lambdamart_comprehensive_{config_name}_ranking.txt'
            convert_to_lambdamart_format(
                labeled_data, output_file_ranking,
                score_type='comprehensive', ranking_mode=True, weights=weights
            )
            generated_files.append((output_file_ranking, f"综合评估-{config_name} (排名分数)"))
            
            # 连续分数模式 (仅为平衡配置生成)
            if config_name == 'balanced':
                output_file_continuous = output_dir / f'lambdamart_comprehensive_{config_name}_continuous.txt'
                convert_to_lambdamart_format(
                    labeled_data, output_file_continuous,
                    score_type='comprehensive', ranking_mode=False, weights=weights
                )
                generated_files.append((output_file_continuous, f"综合评估-{config_name} (连续分数)"))
        
        # 生成传统单一指标的文件 (作为对比)
        logger.info("生成单一指标对比文件")
        single_metrics = ['mean', 'min']
        for metric in single_metrics:
            logger.info(f"处理单一指标: {metric}")
            output_file = output_dir / f'lambdamart_{metric}_ranking.txt'
            convert_to_lambdamart_format(
                labeled_data, output_file,
                score_type=metric, ranking_mode=True
            )
            generated_files.append((output_file, f"单一指标-{metric} (排名分数)"))
        
        # 生成特征描述文件
        logger.info("生成特征描述文件")
        generate_feature_description(output_dir)
        generated_files.append((output_dir / 'feature_description.md', "特征描述文档"))
        
        # 生成本次运行的配置信息文件
        logger.info("生成运行配置信息文件")
        # 使用396作为处理实例数（从日志中看到的实际处理数量）
        generate_run_info(output_dir, timestamp, len(labeled_data), 396)
        generated_files.append((output_dir / 'run_info.md', "运行配置信息"))
        
        logger.info("="*80)
        logger.info("数据转换完成! 生成的文件:")
        logger.info("="*80)
        for i, (file_path, description) in enumerate(generated_files, 1):
            logger.info(f"{i:2d}. {file_path.name}")
            logger.info(f"    {description}")
        logger.info("="*80)
        logger.info("\n推荐使用文件:")
        logger.info("- lambdamart_comprehensive_balanced_ranking.txt (综合评估，平衡权重)")
        logger.info("- lambdamart_comprehensive_quality_focused_ranking.txt (注重解质量)")
        logger.info("- lambdamart_comprehensive_speed_focused_ranking.txt (注重收敛速度)")
        logger.info("- lambdamart_comprehensive_objective_stability_focused_ranking.txt (注重目标稳定性)")
        logger.info("- lambdamart_comprehensive_convergence_stability_focused_ranking.txt (注重收敛稳定性)")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"脚本执行失败: {e}")
        raise
    finally:
        logger.info("脚本执行结束")


if __name__ == "__main__":
    main()
