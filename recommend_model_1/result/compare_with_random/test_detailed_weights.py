#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试细化权重配置功能
"""

import sys
import os
import json

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from main_experiment import configure_detailed_feature_weights

def test_weights_configuration():
    """测试权重配置功能"""
    print("=" * 60)
    print("测试细化特征权重配置")
    print("=" * 60)
    
    # 获取权重配置
    detailed_weights = configure_detailed_feature_weights()
    
    print("📊 基础特征权重:")
    total_basic = 0
    for feature, weight in detailed_weights['basic_features'].items():
        print(f"  {feature:25}: {weight:.3f}")
        total_basic += weight
    print(f"  {'基础特征小计':<25}: {total_basic:.3f}")
    
    print("\n⏱️  加工时间特征权重:")
    total_processing = 0
    for feature, weight in detailed_weights['processing_time_features'].items():
        print(f"  {feature:25}: {weight:.3f}")
        total_processing += weight
    print(f"  {'加工时间特征小计':<25}: {total_processing:.3f}")
    
    print("\n🔍 其他特征权重:")
    kde_weight = detailed_weights['kde_similarity_weight']
    disjunctive_weight = detailed_weights['disjunctive_similarity_weight']
    print(f"  {'kde_similarity_weight':<25}: {kde_weight:.3f}")
    print(f"  {'disjunctive_similarity_weight':<25}: {disjunctive_weight:.3f}")
    
    # 计算总权重
    total_weight = total_basic + total_processing + kde_weight + disjunctive_weight
    
    print("\n✅ 权重验证:")
    print(f"  总权重: {total_weight:.3f}")
    print(f"  预期值: 1.000")
    print(f"  差值: {abs(total_weight - 1.0):.6f}")
    
    if abs(total_weight - 1.0) < 0.001:
        print("  ✅ 权重配置正确!")
    else:
        print("  ❌ 权重配置错误!")
    
    print("\n📝 配置详情:")
    print(f"  基础特征指标数量: {len(detailed_weights['basic_features'])}")
    print(f"  加工时间特征指标数量: {len(detailed_weights['processing_time_features'])}")
    print(f"  总特征指标数量: {len(detailed_weights['basic_features']) + len(detailed_weights['processing_time_features']) + 2}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    return detailed_weights

def test_json_serialization():
    """测试JSON序列化功能"""
    print("\n🔧 测试JSON序列化...")
    
    detailed_weights = configure_detailed_feature_weights()
    
    # 模拟保存配置
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    weights_config = {
        'metadata': {
            'version': '1.0',
            'description': '细化特征权重配置',
            'created_at': datetime.datetime.now().isoformat(),
            'timestamp': timestamp
        },
        'weights': detailed_weights
    }
    
    # 转换为JSON字符串测试
    try:
        json_str = json.dumps(weights_config, indent=2, ensure_ascii=False)
        print("  ✅ JSON序列化成功")
        print(f"  配置大小: {len(json_str)} 字符")
        
        # 测试反序列化
        loaded_config = json.loads(json_str)
        print("  ✅ JSON反序列化成功")
        
        # 验证数据完整性
        if loaded_config['weights'] == detailed_weights:
            print("  ✅ 数据完整性验证通过")
        else:
            print("  ❌ 数据完整性验证失败")
            
    except Exception as e:
        print(f"  ❌ JSON序列化失败: {e}")

if __name__ == "__main__":
    # 运行测试
    weights = test_weights_configuration()
    test_json_serialization()
    
    print("\n🎯 使用建议:")
    print("1. 可以通过修改 configure_detailed_feature_weights() 函数调整权重")
    print("2. 权重总和应该接近 1.0")
    print("3. 根据问题特点，可以提高重要特征的权重")
    print("4. 建议进行权重敏感性分析，找到最优配置")
