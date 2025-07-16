#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试初始化策略推荐系统 V2
"""

import os
import json
import sys

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def test_labeled_dataset():
    """测试标记数据集加载"""
    print("=== 测试标记数据集加载 ===")
    
    labeled_dataset_path = "labeled_dataset/labeled_fjs_dataset.json"
    
    if not os.path.exists(labeled_dataset_path):
        print(f"错误: 标记数据集文件不存在: {labeled_dataset_path}")
        return False
    
    try:
        with open(labeled_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"成功加载标记数据集，共 {len(data)} 条记录")
        
        # 检查数据结构
        first_key = list(data.keys())[0]
        first_sample = data[first_key]
        
        print(f"第一个样本: {first_key}")
        print(f"包含字段: {list(first_sample.keys())}")
        
        if 'features' in first_sample:
            features = first_sample['features']
            print(f"特征字段: {list(features.keys())}")
        
        if 'performance_data' in first_sample:
            performance = first_sample['performance_data']
            print(f"性能数据策略数量: {len(performance)}")
            if len(performance) > 0:
                first_strategy = list(performance.keys())[0]
                print(f"第一个策略: {first_strategy}")
                print(f"策略数据字段: {list(performance[first_strategy].keys())}")
        
        return True
        
    except Exception as e:
        print(f"加载标记数据集失败: {e}")
        return False

def test_similarity_calculation():
    """测试相似度计算逻辑"""
    print("\n=== 测试相似度计算逻辑 ===")
    
    # 导入相似度计算函数
    try:
        from feature_similarity_weighting.calculate_weighted_similarity import (
            normalize_features,
            calculate_euclidean_distance,
            calculate_js_divergence,
            normalize_distance
        )
        print("成功导入相似度计算函数")
        return True
    except ImportError as e:
        print(f"导入相似度计算函数失败: {e}")
        return False

def test_disjunctive_graph_similarity():
    """测试析取图相似度计算"""
    print("\n=== 测试析取图相似度计算 ===")
    
    try:
        from feature_similarity_weighting.calculate_weighted_similarity import (
            calculate_disjunctive_graph_similarity
        )
        print("成功导入析取图相似度计算函数")
        return True
    except ImportError as e:
        print(f"导入析取图相似度计算函数失败: {e}")
        return False

def test_recommender_initialization():
    """测试推荐系统初始化"""
    print("\n=== 测试推荐系统初始化 ===")
    
    try:
        from initialization_strategy_recommender import InitializationStrategyRecommender
        
        labeled_dataset_path = "labeled_dataset/labeled_fjs_dataset.json"
        recommender = InitializationStrategyRecommender(labeled_dataset_path)
        
        print("推荐系统初始化成功")
        print(f"标记数据数量: {len(recommender.labeled_data)}")
        print(f"标准化特征数量: {len(recommender.normalized_features)}")
        
        return True, recommender
        
    except Exception as e:
        print(f"推荐系统初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_stage_one_similarity_search(recommender):
    """测试阶段一相似度检索"""
    print("\n=== 测试阶段一相似度检索 ===")
    
    try:
        # 使用第一个样本的特征作为新数据
        first_sample = list(recommender.labeled_data.values())[0]
        new_data_features = first_sample['features']
        
        print(f"使用样本特征作为新数据: {list(recommender.labeled_data.keys())[0]}")
        
        # 执行阶段一检索
        candidate_samples = recommender.stage_one_similarity_search(new_data_features, top_k=5)
        
        print(f"阶段一检索完成，获得 {len(candidate_samples)} 个候选样本")
        
        return True, candidate_samples
        
    except Exception as e:
        print(f"阶段一检索失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_stage_two_strategy_recommendation(recommender, candidate_samples):
    """测试阶段二策略推荐"""
    print("\n=== 测试阶段二策略推荐 ===")
    
    try:
        # 执行阶段二推荐
        recommended_strategies = recommender.stage_two_strategy_recommendation(candidate_samples, top_k=3)
        
        print(f"阶段二推荐完成，获得 {len(recommended_strategies)} 个推荐策略")
        
        return True, recommended_strategies
        
    except Exception as e:
        print(f"阶段二推荐失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_complete_recommendation_flow(recommender):
    """测试完整推荐流程"""
    print("\n=== 测试完整推荐流程 ===")
    
    try:
        # 使用第一个样本的特征作为新数据
        first_sample = list(recommender.labeled_data.values())[0]
        new_data_features = first_sample['features']
        
        print(f"使用样本特征作为新数据: {list(recommender.labeled_data.keys())[0]}")
        
        # 执行完整推荐流程
        results = recommender.recommend(new_data_features, top_k_similar=5, top_k_strategies=3)
        
        print("完整推荐流程执行成功")
        print(f"执行时间: {results['execution_time']:.2f}秒")
        print(f"候选样本数量: {len(results['stage_one_results']['candidate_samples'])}")
        print(f"推荐策略数量: {len(results['stage_two_results']['recommended_strategies'])}")
        
        # 保存结果
        output_file = "test_recommendation_results.json"
        recommender.save_results(results, output_file)
        
        return True, results
        
    except Exception as e:
        print(f"完整推荐流程失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """主测试函数"""
    print("开始测试初始化策略推荐系统 V2")
    print("=" * 50)
    
    # 测试1: 标记数据集加载
    if not test_labeled_dataset():
        print("标记数据集测试失败，退出")
        return
    
    # 测试2: 相似度计算逻辑
    if not test_similarity_calculation():
        print("相似度计算逻辑测试失败，退出")
        return
    
    # 测试3: 析取图相似度计算
    if not test_disjunctive_graph_similarity():
        print("析取图相似度计算测试失败，退出")
        return
    
    # 测试4: 推荐系统初始化
    success, recommender = test_recommender_initialization()
    if not success:
        print("推荐系统初始化测试失败，退出")
        return
    
    # 测试5: 阶段一相似度检索
    success, candidate_samples = test_stage_one_similarity_search(recommender)
    if not success:
        print("阶段一相似度检索测试失败，退出")
        return
    
    # 测试6: 阶段二策略推荐
    success, recommended_strategies = test_stage_two_strategy_recommendation(recommender, candidate_samples)
    if not success:
        print("阶段二策略推荐测试失败，退出")
        return
    
    # 测试7: 完整推荐流程
    success, results = test_complete_recommendation_flow(recommender)
    if not success:
        print("完整推荐流程测试失败，退出")
        return
    
    print("\n" + "=" * 50)
    print("所有测试通过！推荐系统工作正常")
    print("=" * 50)

if __name__ == "__main__":
    main() 