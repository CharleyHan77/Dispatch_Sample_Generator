#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行推荐系统对新数据进行推荐
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.append(project_root)

def main():
    """主函数"""
    # 设置输入和输出路径
    new_data_file = "recommend_model_1/result/compare_with_random/new_behnke29_variant.fjs"
    labeled_dataset_path = "recommend_model_1/labeled_dataset/labeled_fjs_dataset.json"
    
    # 创建结果目录
    result_dir = "recommend_model_1/result/recommender_output"
    os.makedirs(result_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("初始化策略推荐系统")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"新数据文件: {new_data_file}")
    print(f"标记数据集: {labeled_dataset_path}")
    print("=" * 80)
    
    # 检查文件是否存在
    if not os.path.exists(new_data_file):
        print(f"错误: 新数据文件不存在: {new_data_file}")
        return
    
    if not os.path.exists(labeled_dataset_path):
        print(f"错误: 标记数据集不存在: {labeled_dataset_path}")
        return
    
    try:
        # 导入推荐系统
        sys.path.append("recommend_model_1")
        from initialization_strategy_recommender import InitializationStrategyRecommender
        
        # 导入特征提取模块
        from extract_new_data_features import extract_new_data_features
        
        # 初始化推荐系统（带日志）
        log_file = os.path.join(result_dir, f"recommendation_log_{timestamp}.log")
        recommender = InitializationStrategyRecommender(labeled_dataset_path, log_file)
        
        print("✓ 推荐系统初始化成功")
        
        # 提取新数据特征
        print(f"\n开始提取新数据特征: {new_data_file}")
        new_data_features = extract_new_data_features(new_data_file)
        
        if new_data_features is None:
            print("错误: 新数据特征提取失败")
            return
        
        print("✓ 新数据特征提取完成")
        
        # 执行推荐
        print("\n开始执行推荐流程...")
        results = recommender.recommend(new_data_features, top_k_similar=5, top_k_strategies=3)
        
        # 保存推荐结果
        output_file = os.path.join(result_dir, f"recommendation_results_{timestamp}.json")
        recommender.save_results(results, output_file)
        
        # 可视化结果
        output_dir = os.path.join(result_dir, f"visualization_{timestamp}")
        recommender.visualize_recommendation_results(results, output_dir)
        
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
        for i, data in enumerate(stage_two_data, 1):
            print(f"{i}. {data['strategy_name']}")
            print(f"   加权性能评分: {data['weighted_score']:.4f}")
        
        print("\n" + "=" * 80)
        print("所有结果已保存")
        print(f"结果目录: {result_dir}")
        print(f"日志文件: {log_file}")
        print(f"推荐结果: {output_file}")
        print(f"可视化结果: {output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 