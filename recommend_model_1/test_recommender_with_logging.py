#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试带日志记录的初始化策略推荐系统
"""

import os
import sys
from datetime import datetime

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from initialization_strategy_recommender import InitializationStrategyRecommender


def test_recommender_with_logging():
    """测试带日志记录的推荐系统"""
    
    # 创建结果目录
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=== 测试带日志记录的初始化策略推荐系统 ===")
    
    # 初始化推荐系统（带日志）
    labeled_dataset_path = "labeled_dataset/labeled_fjs_dataset.json"
    log_file = os.path.join(result_dir, f"test_recommendation_log_{timestamp}.log")
    
    print(f"日志文件路径: {log_file}")
    print(f"标记数据集路径: {labeled_dataset_path}")
    
    try:
        recommender = InitializationStrategyRecommender(labeled_dataset_path, log_file)
        
        # 模拟新数据特征（使用第一个样本的特征作为示例）
        first_sample = list(recommender.labeled_data.values())[0]
        new_data_features = first_sample['features']
        
        sample_name = list(recommender.labeled_data.keys())[0]
        recommender.log_info(f"使用样本特征作为新数据: {sample_name}")
        
        # 执行推荐
        results = recommender.recommend(new_data_features, top_k_similar=5, top_k_strategies=3)
        
        # 保存推荐结果
        output_file = os.path.join(result_dir, f"test_recommendation_results_{timestamp}.json")
        recommender.save_results(results, output_file)
        
        # 可视化结果
        output_dir = os.path.join(result_dir, f"test_visualization_{timestamp}")
        recommender.visualize_recommendation_results(results, output_dir)
        
        # 输出结果摘要
        print("\n=== 推荐结果摘要 ===")
        print(f"执行时间: {results['execution_time']:.2f}秒")
        print(f"候选样本数量: {len(results['stage_one_results']['candidate_samples'])}")
        print(f"推荐策略数量: {len(results['stage_two_results']['recommended_strategies'])}")
        
        print("\n推荐策略排名:")
        for i, strategy in enumerate(results['stage_two_results']['recommended_strategies'], 1):
            print(f"{i}. {strategy['strategy_name']}: {strategy['weighted_score']:.4f}")
        
        print(f"\n所有结果已保存到 {result_dir} 目录")
        print(f"日志文件: {log_file}")
        print(f"推荐结果: {output_file}")
        print(f"可视化结果: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_recommender_with_logging()
    if success:
        print("\n测试完成！")
    else:
        print("\n测试失败！")
        sys.exit(1) 