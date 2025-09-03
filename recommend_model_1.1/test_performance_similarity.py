#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试添加性能相似度功能的脚本
"""

import os
import sys
import json

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from initialization_strategy_recommender_add_new_performance_data import InitializationStrategyRecommender

def test_fake_performance_data():
    """测试假性能数据创建功能"""
    print("=== 测试假性能数据创建功能 ===")
    
    # 创建推荐系统实例
    labeled_dataset_path = os.path.join(current_dir, "labeled_dataset", "converted_fjs_dataset_new.json")
    recommender = InitializationStrategyRecommender(labeled_dataset_path)
    
    # 创建假性能数据
    fake_performance_data = recommender.create_fake_performance_data_for_new_data()
    
    print("假性能数据结构:")
    print(json.dumps(fake_performance_data, indent=2, ensure_ascii=False))
    
    # 验证假性能数据结构
    assert 'performance_metrics' in fake_performance_data
    metrics = fake_performance_data['performance_metrics']
    
    # 检查所有性能指标都是1.0
    expected_metrics = ['mean', 'std', 'min', 'max', 'avg_convergence_generation', 'convergence_generation_std']
    for metric in expected_metrics:
        assert metric in metrics
        assert metrics[metric] == 1.0, f"性能指标 {metric} 应该为1.0，但实际为 {metrics[metric]}"
    
    print("✓ 假性能数据创建功能测试通过")
    print()

def test_performance_similarity_calculation():
    """测试性能相似度计算功能"""
    print("=== 测试性能相似度计算功能 ===")
    
    # 创建推荐系统实例
    labeled_dataset_path = os.path.join(current_dir, "labeled_dataset", "converted_fjs_dataset_new.json")
    recommender = InitializationStrategyRecommender(labeled_dataset_path)
    
    # 创建假性能数据（新数据）
    new_performance_data = recommender.create_fake_performance_data_for_new_data()
    
    # 创建一个模拟的历史性能数据
    hist_performance_data = {
        "meta_heuristic": "HA(GA+TS)",
        "execution_times": 20,
        "max_iterations": 100,
        "initialization_method": "heuristic",
        "performance_metrics": {
            "mean": 1161.05,
            "std": 47.46733087082104,
            "min": 1105,
            "max": 1281,
            "avg_convergence_generation": 16.15,
            "convergence_generation_std": 10.946574806760333
        }
    }
    
    # 计算性能相似度
    performance_similarity = recommender.calculate_performance_similarity(
        new_performance_data, hist_performance_data
    )
    
    print(f"新数据（假性能数据）与历史数据的性能相似度: {performance_similarity:.6f}")
    
    # 验证相似度在合理范围内
    assert 0 <= performance_similarity <= 1, f"性能相似度应该在[0,1]范围内，但实际为 {performance_similarity}"
    
    print("✓ 性能相似度计算功能测试通过")
    print()

def test_comprehensive_similarity_with_performance():
    """测试包含性能相似度的综合相似度计算"""
    print("=== 测试包含性能相似度的综合相似度计算 ===")
    
    # 创建推荐系统实例
    labeled_dataset_path = os.path.join(current_dir, "labeled_dataset", "converted_fjs_dataset_new.json")
    recommender = InitializationStrategyRecommender(labeled_dataset_path)
    
    # 获取第一个历史样本作为测试
    first_sample_id = list(recommender.labeled_data.keys())[0]
    first_sample_data = recommender.labeled_data[first_sample_id]
    
    print(f"使用历史样本 {first_sample_id} 进行测试")
    
    # 模拟新数据特征（使用历史样本的特征作为模拟）
    new_data_features = first_sample_data['features'].copy()
    
    # 创建假性能数据
    new_performance_data = recommender.create_fake_performance_data_for_new_data()
    
    # 标准化新数据特征
    new_data_normalized = recommender.normalize_single_features(new_data_features)
    
    # 计算综合相似度（包含性能相似度）
    similarity_details = recommender.calculate_similarity(
        new_data_normalized, 
        first_sample_id, 
        1.0, 1.0,  # 最大距离（简化测试）
        new_data_features, 
        new_performance_data
    )
    
    print("综合相似度计算结果:")
    for key, value in similarity_details.items():
        print(f"  {key}: {value:.6f}")
    
    # 验证所有相似度指标都在合理范围内
    for key, value in similarity_details.items():
        assert 0 <= value <= 1, f"{key} 应该在[0,1]范围内，但实际为 {value}"
    
    # 验证包含了性能相似度
    assert 'performance_similarity' in similarity_details, "应该包含性能相似度"
    
    print("✓ 包含性能相似度的综合相似度计算测试通过")
    print()

def main():
    """主测试函数"""
    print("开始测试添加性能相似度功能...")
    print()
    
    try:
        # 测试假性能数据创建
        test_fake_performance_data()
        
        # 测试性能相似度计算
        test_performance_similarity_calculation()
        
        # 测试综合相似度计算
        test_comprehensive_similarity_with_performance()
        
        print("=" * 60)
        print("✅ 所有测试通过！性能相似度功能已成功集成")
        print("=" * 60)
        print()
        print("功能总结:")
        print("1. ✓ 为新数据创建假性能数据（所有评分设为1.0）")
        print("2. ✓ 计算新数据与历史数据的性能相似度")
        print("3. ✓ 将性能相似度纳入综合相似度计算")
        print("4. ✓ 权重分配：基础特征(25%) + 加工时间(20%) + KDE(15%) + 析取图(20%) + 性能(20%)")
        print()
        print("现在新数据可以进行完整的相似度对比，包括性能指标的相似度加权！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
