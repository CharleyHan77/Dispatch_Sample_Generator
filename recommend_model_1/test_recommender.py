#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试初始化策略推荐系统
"""

import os
import sys
import json
from pathlib import Path

# 添加父目录到路径
sys.path.append('..')

def test_data_loading():
    """测试数据加载功能"""
    print("=== 测试数据加载 ===")
    
    try:
        from initialization_strategy_recommender import InitializationStrategyRecommender
        
        # 创建推荐器
        recommender = InitializationStrategyRecommender()
        
        # 检查数据加载
        print(f"历史数据集特征数量: {len(recommender.dataset_features)}")
        print(f"新数据特征: {list(recommender.new_data_features.keys())}")
        print(f"初始化策略数据数量: {len(recommender.init_strategy_data)}")
        
        # 显示一些示例数据
        print("\n示例初始化策略数据:")
        for i, (instance, data) in enumerate(list(recommender.init_strategy_data.items())[:3]):
            print(f"{i+1}. {instance}")
            print(f"   数据集: {data['dataset']}")
            print(f"   元启发式: {data['meta_heuristic']}")
            print(f"   策略: {list(data['initialization_methods'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"数据加载测试失败: {e}")
        return False

def test_stage1_candidate_generation():
    """测试阶段一候选集生成"""
    print("\n=== 测试阶段一候选集生成 ===")
    
    try:
        from initialization_strategy_recommender import InitializationStrategyRecommender
        
        recommender = InitializationStrategyRecommender()
        
        # 执行候选集生成
        candidates = recommender.stage1_candidate_generation(top_k=3)
        
        print(f"生成候选集数量: {len(candidates)}")
        for i, (instance, similarity) in enumerate(candidates):
            print(f"{i+1}. {instance}: {similarity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"候选集生成测试失败: {e}")
        return False

def test_stage2_strategy_recommendation():
    """测试阶段二策略推荐"""
    print("\n=== 测试阶段二策略推荐 ===")
    
    try:
        from initialization_strategy_recommender import InitializationStrategyRecommender
        
        recommender = InitializationStrategyRecommender()
        
        # 先生成候选集
        candidates = recommender.stage1_candidate_generation(top_k=3)
        
        # 执行策略推荐
        result = recommender.stage2_strategy_recommendation(candidates)
        
        print(f"推荐结果包含 {len(result['strategy_recommendations'])} 种策略")
        print(f"可用策略: {result['available_strategies']}")
        
        # 显示推荐结果
        print("\n策略推荐结果:")
        for strategy, score_data in result['strategy_recommendations'].items():
            print(f"{strategy}: {score_data['final_score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"策略推荐测试失败: {e}")
        return False

def test_full_recommendation():
    """测试完整推荐流程"""
    print("\n=== 测试完整推荐流程 ===")
    
    try:
        from initialization_strategy_recommender import InitializationStrategyRecommender
        
        recommender = InitializationStrategyRecommender()
        
        # 执行完整推荐
        result = recommender.recommend(top_k=5)
        
        print(f"推荐策略: {result['recommended_strategy']}")
        print(f"推荐评分: {result['recommendation_summary']['top_score']:.4f}")
        print(f"候选样本数: {result['recommendation_summary']['candidate_count']}")
        
        # 保存结果
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, "test_recommendation_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"测试结果已保存到: {json_path}")
        
        return True
        
    except Exception as e:
        print(f"完整推荐流程测试失败: {e}")
        return False

def test_visualization():
    """测试可视化功能"""
    print("\n=== 测试可视化功能 ===")
    
    try:
        from initialization_strategy_recommender import InitializationStrategyRecommender
        
        recommender = InitializationStrategyRecommender()
        
        # 执行推荐
        result = recommender.recommend(top_k=5)
        
        # 生成可视化
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        viz_path = os.path.join(output_dir, "test_visualization.png")
        recommender.visualize_recommendation_result(result, viz_path)
        
        print(f"可视化结果已保存到: {viz_path}")
        
        return True
        
    except Exception as e:
        print(f"可视化测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试初始化策略推荐系统...\n")
    
    tests = [
        ("数据加载", test_data_loading),
        ("阶段一候选集生成", test_stage1_candidate_generation),
        ("阶段二策略推荐", test_stage2_strategy_recommendation),
        ("完整推荐流程", test_full_recommendation),
        ("可视化功能", test_visualization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"正在执行测试: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"测试 {test_name}: {'通过' if success else '失败'}\n")
        except Exception as e:
            print(f"测试 {test_name} 出现异常: {e}\n")
            results.append((test_name, False))
    
    # 输出测试总结
    print("=== 测试总结 ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"总测试数: {total}")
    print(f"通过测试: {passed}")
    print(f"失败测试: {total - passed}")
    print(f"通过率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("所有测试通过！系统运行正常。")
    else:
        print("部分测试失败，请检查相关功能。")
    
    return passed == total

if __name__ == "__main__":
    main() 