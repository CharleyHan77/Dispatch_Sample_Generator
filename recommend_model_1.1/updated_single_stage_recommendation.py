#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新的单阶段推荐方法，适配新的数据结构
"""

def single_stage_recommendation(self, new_data_features, top_k_strategies=3, feature_weight=0.4, performance_weight=0.6):
    """
    基于新数据结构的初始化策略推荐
    
    核心流程：
    1. 计算新数据与所有历史样本的详细特征相似度（细化到每个指标）
    2. 分别为三种初始化方法（heuristic, mixed, random）找到最相似的历史样本
    3. 基于各自最相似样本的性能数据进行策略评分和推荐
    
    Args:
        new_data_features: 新数据的特征
        top_k_strategies: 推荐的策略数量
        feature_weight: 特征相似度权重（用于最终评分）
        performance_weight: 性能目标权重（用于最终评分）
        
    Returns:
        list: 推荐策略列表，每个元素为 (strategy_name, final_score, feature_score, performance_score, supporting_samples)
    """
    self.log_info(f"=== 基于新样本结构的策略推荐 ===")
    
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
    
    strategy_evaluations = {}
    
    # 为每种初始化方法找到最相似的样本并评分
    for method_name, method_historical_samples in method_samples.items():
        if not method_historical_samples:
            self.log_warning(f"初始化方法 {method_name} 没有历史样本数据")
            continue
            
        self.log_info(f"\n=== 评估初始化方法: {method_name} ===")
        
        # 计算与该方法所有历史样本的相似度
        similarities = []
        for sample_id, sample_data in method_historical_samples.items():
            features = sample_data.get('features', {})
            similarity = self.calculate_detailed_feature_similarity(new_data_features, features)
            similarities.append((sample_id, similarity, sample_data))
        
        # 找到最相似的样本
        if not similarities:
            continue
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_sample_id, best_similarity, best_sample_data = similarities[0]
        
        self.log_info(f"最相似样本: {best_sample_id}")
        self.log_info(f"原始文件: {best_sample_data.get('original_fjs_path', 'N/A')}")
        self.log_info(f"相似度: {best_similarity:.4f}")
        
        # 提取性能数据
        performance_data = best_sample_data.get('performance_data', {})
        if not performance_data or 'performance_metrics' not in performance_data:
            self.log_warning(f"样本 {best_sample_id} 缺少性能数据")
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
        
        # 最终综合评分：特征相似度 + 性能评分
        final_score = feature_weight * best_similarity + performance_weight * performance_score
        
        strategy_evaluations[method_name] = {
            'final_score': final_score,
            'feature_similarity_score': best_similarity,
            'performance_score': performance_score,
            'supporting_samples': [{
                'sample_id': best_sample_id,
                'original_fjs_path': best_sample_data.get('original_fjs_path', ''),
                'similarity': best_similarity,
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
            }]
        }
        
        self.log_info(f"性能指标:")
        self.log_info(f"  - 平均Makespan: {mean_makespan:.2f}")
        self.log_info(f"  - Makespan标准差: {std_makespan:.2f}")
        self.log_info(f"  - 平均收敛代数: {avg_convergence_gen:.2f}")
        self.log_info(f"  - 收敛稳定性: {convergence_std:.2f}")
        self.log_info(f"评分结果:")
        self.log_info(f"  - 特征相似度评分: {best_similarity:.4f}")
        self.log_info(f"  - 性能评分: {performance_score:.4f}")
        self.log_info(f"  - 最终评分: {final_score:.4f}")
    
    if not strategy_evaluations:
        self.log_error("未找到有效的策略推荐")
        return []
    
    # 按最终评分排序并返回推荐结果
    sorted_strategies = sorted(strategy_evaluations.items(), key=lambda x: x[1]['final_score'], reverse=True)
    
    self.log_info(f"\n=== 推荐结果排序 ===")
    for i, (strategy_name, evaluation) in enumerate(sorted_strategies[:top_k_strategies], 1):
        self.log_info(f"第{i}名: {strategy_name} (评分: {evaluation['final_score']:.4f})")
        
        # 显示详细评分信息
        supporting_sample = evaluation['supporting_samples'][0]
        detailed_scores = supporting_sample['detailed_scores']
        self.log_info(f"   特征相似度评分: {evaluation['feature_similarity_score']:.4f}")
        self.log_info(f"   性能目标评分: {evaluation['performance_score']:.4f}")
        self.log_info(f"   详细性能评分:")
        self.log_info(f"     Makespan评分: {detailed_scores['makespan_score']:.4f}")
        self.log_info(f"     收敛速度评分: {detailed_scores['convergence_speed_score']:.4f}")
        self.log_info(f"     稳定性评分: {detailed_scores['stability_score']:.4f}")
        self.log_info(f"     收敛稳定性评分: {detailed_scores['convergence_stability_score']:.4f}")
        self.log_info(f"   支持样本: {supporting_sample['sample_id']} ({supporting_sample['original_fjs_path']})")
    
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






