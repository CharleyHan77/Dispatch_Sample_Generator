#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析和转换labeled_fjs_dataset.json数据结构的脚本
将每个样本按照初始化方法拆分为三个独立样本
"""

import json
import os
from datetime import datetime

def analyze_data_structure(data):
    """分析现有数据结构"""
    print("=" * 50)
    print("数据结构分析")
    print("=" * 50)
    
    total_samples = len(data)
    print(f"总样本数: {total_samples}")
    
    # 分析第一个样本的结构
    first_key = list(data.keys())[0]
    first_sample = data[first_key]
    
    print(f"\n样本示例: {first_key}")
    print(f"- fjs_path: {first_sample.get('fjs_path', 'N/A')}")
    
    # 分析特征结构
    features = first_sample.get('features', {})
    print(f"\n特征类别:")
    for feature_type, feature_data in features.items():
        if isinstance(feature_data, dict):
            print(f"  - {feature_type}: {len(feature_data)} 个指标")
            for key in list(feature_data.keys())[:3]:  # 只显示前3个
                print(f"    * {key}")
            if len(feature_data) > 3:
                print(f"    * ... (共{len(feature_data)}个)")
        else:
            print(f"  - {feature_type}: {type(feature_data).__name__}")
    
    # 分析性能数据结构
    performance_data = first_sample.get('performance_data', {})
    if performance_data:
        print(f"\n性能数据结构:")
        print(f"  - meta_heuristic: {performance_data.get('meta_heuristic', 'N/A')}")
        print(f"  - execution_times: {performance_data.get('execution_times', 'N/A')}")
        print(f"  - max_iterations: {performance_data.get('max_iterations', 'N/A')}")
        
        init_methods = performance_data.get('initialization_methods', {})
        print(f"  - initialization_methods: {len(init_methods)} 种方法")
        for method_name, method_data in init_methods.items():
            print(f"    * {method_name}: {list(method_data.keys())}")
    
    return total_samples

def convert_dataset(input_data):
    """转换数据集格式"""
    print("\n" + "=" * 50)
    print("开始数据转换")
    print("=" * 50)
    
    new_dataset = {}
    sample_counter = 1
    
    for original_key, original_sample in input_data.items():
        features = original_sample.get('features', {})
        performance_data = original_sample.get('performance_data', {})
        
        if not performance_data or 'initialization_methods' not in performance_data:
            print(f"警告: {original_key} 缺少performance_data，跳过")
            continue
        
        init_methods = performance_data['initialization_methods']
        
        # 为每个初始化方法创建一个新样本
        for method_name, method_performance in init_methods.items():
            sample_id = f"sample_{sample_counter:04d}"
            
            # 创建新样本
            new_sample = {
                "sample_id": sample_id,
                "original_fjs_path": original_sample.get('fjs_path', original_key),
                "initialization_method": method_name,  # 明确标记初始化方法
                "features": features.copy(),  # 特征保持相同
                "performance_data": {
                    "meta_heuristic": performance_data.get('meta_heuristic', ''),
                    "execution_times": performance_data.get('execution_times', 0),
                    "max_iterations": performance_data.get('max_iterations', 0),
                    "performance_metrics": method_performance.copy()  # 只包含当前方法的性能指标
                }
            }
            
            new_dataset[sample_id] = new_sample
            sample_counter += 1
            
            if sample_counter % 100 == 1:
                print(f"已处理 {sample_counter-1} 个样本...")
    
    print(f"\n转换完成! 总共生成 {len(new_dataset)} 个样本")
    print(f"原始样本数: {len(input_data)}")
    print(f"新样本数: {len(new_dataset)}")
    print(f"每个原始样本平均拆分为: {len(new_dataset) / len(input_data):.1f} 个样本")
    
    return new_dataset

def validate_conversion(new_data):
    """验证转换结果"""
    print("\n" + "=" * 50)
    print("验证转换结果")
    print("=" * 50)
    
    # 统计初始化方法分布
    method_counts = {}
    for sample_id, sample_data in new_data.items():
        method = sample_data.get('initialization_method', 'unknown')
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print("初始化方法分布:")
    for method, count in method_counts.items():
        print(f"  - {method}: {count} 个样本")
    
    # 检查数据完整性
    print(f"\n数据完整性检查:")
    complete_samples = 0
    for sample_id, sample_data in new_data.items():
        if all(key in sample_data for key in ['sample_id', 'original_fjs_path', 'initialization_method', 'features', 'performance_data']):
            complete_samples += 1
    
    print(f"  - 完整样本数: {complete_samples}/{len(new_data)}")
    print(f"  - 完整率: {complete_samples/len(new_data)*100:.1f}%")
    
    # 显示几个样本示例
    print(f"\n样本示例:")
    for i, (sample_id, sample_data) in enumerate(list(new_data.items())[:6]):  # 显示6个样本，每种方法2个
        print(f"  {sample_id}:")
        print(f"    - 原始文件: {sample_data.get('original_fjs_path', 'N/A')}")
        
        # 显示初始化方法和性能指标
        init_method = sample_data.get('initialization_method', 'N/A')
        print(f"    ✅ 初始化方法: {init_method}")
        
        perf_data = sample_data.get('performance_data', {})
        perf_metrics = perf_data.get('performance_metrics', {})
        if perf_metrics:
            print(f"    - 性能指标: mean={perf_metrics.get('mean', 'N/A')}, std={perf_metrics.get('std', 'N/A')}")
        print()  # 空行分隔

def main():
    """主函数"""
    input_file = "labeled_fjs_dataset.json"
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return
    
    print(f"读取数据文件: {input_file}")
    
    # 读取原始数据
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except Exception as e:
        print(f"错误: 读取文件失败 - {e}")
        return
    
    # 分析数据结构
    total_samples = analyze_data_structure(original_data)
    
    # 转换数据
    converted_data = convert_dataset(original_data)
    
    # 验证转换结果
    validate_conversion(converted_data)
    
    # 生成新文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"converted_fjs_dataset_{timestamp}.json"
    
    # 保存转换结果
    print(f"\n保存转换结果到: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 转换成功! 新数据集已保存为: {output_file}")
        
        # 显示文件大小信息
        input_size = os.path.getsize(input_file) / (1024*1024)
        output_size = os.path.getsize(output_file) / (1024*1024)
        print(f"\n文件大小对比:")
        print(f"  - 原始文件: {input_size:.1f} MB")
        print(f"  - 转换文件: {output_size:.1f} MB")
        print(f"  - 大小变化: {(output_size/input_size-1)*100:+.1f}%")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")

if __name__ == "__main__":
    main()
