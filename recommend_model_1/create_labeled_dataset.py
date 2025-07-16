#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建标记数据集
将FJS文件的特征数据和对应的初始化策略性能数据整合到一个JSON文件中
"""

import os
import json
import re
from pathlib import Path


def normalize_fjs_path(path_str):
    """
    统一FJS文件路径格式
    将各种路径格式统一为 "数据集名/子目录/文件名.fjs" 的格式
    
    Args:
        path_str: 原始路径字符串
        
    Returns:
        str: 标准化后的路径
    """
    # 移除可能的路径前缀和后缀
    path_str = path_str.strip()
    
    # 如果包含完整路径，提取相对部分
    if '/' in path_str or '\\' in path_str:
        # 查找常见的FJS文件路径模式
        patterns = [
            r'([A-Za-z]+/[a-z]+/[^/\\]+\.fjs)',  # 如 "Hurink/edata/abz6.fjs"
            r'([A-Za-z]+/[^/\\]+\.fjs)',         # 如 "Barnes/mt10c1.fjs"
            r'([^/\\]+\.fjs)'                     # 如 "Kacem1.fjs"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, path_str)
            if match:
                return match.group(1)
    
    # 如果没有匹配到模式，返回原字符串
    return path_str


def load_dataset_features(features_file):
    """
    加载数据集特征
    
    Args:
        features_file: 特征文件路径
        
    Returns:
        dict: 特征数据字典
    """
    print(f"正在加载特征数据: {features_file}")
    
    if not os.path.exists(features_file):
        print(f"错误: 特征文件不存在: {features_file}")
        return {}
    
    try:
        with open(features_file, 'r', encoding='utf-8') as f:
            features_data = json.load(f)
        
        print(f"已加载 {len(features_data)} 个数据集的特征")
        return features_data
        
    except Exception as e:
        print(f"错误: 无法加载特征文件: {e}")
        return {}


def load_strategy_performance(strategy_dir):
    """
    加载策略性能数据
    
    Args:
        strategy_dir: 策略数据目录
        
    Returns:
        dict: 策略性能数据字典
    """
    print(f"正在加载策略性能数据: {strategy_dir}")
    
    if not os.path.exists(strategy_dir):
        print(f"错误: 策略数据目录不存在: {strategy_dir}")
        return {}
    
    performance_data = {}
    
    # 遍历目录下的所有JSON文件
    for root, dirs, files in os.walk(strategy_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 尝试从文件名或数据中提取FJS路径
                    fjs_path = None
                    
                    # 方法1: 从文件名提取
                    file_name = os.path.splitext(file)[0]
                    if '_validation_results' in file_name:
                        base_name = file_name.replace('_validation_results', '')
                        # 尝试添加.fjs扩展名
                        fjs_path = f"{base_name}.fjs"
                    else:
                        fjs_path = f"{file_name}.fjs"
                    
                    # 方法2: 从数据内容中查找
                    if isinstance(data, dict):
                        # 查找包含fjs路径的字段
                        for key, value in data.items():
                            if isinstance(value, str) and value.endswith('.fjs'):
                                fjs_path = value
                                break
                    
                    if fjs_path:
                        # 标准化路径格式
                        normalized_path = normalize_fjs_path(fjs_path)
                        performance_data[normalized_path] = data
                        print(f"  加载: {normalized_path}")
                    
                except Exception as e:
                    print(f"警告: 无法加载策略数据文件 {file_path}: {e}")
    
    print(f"已加载 {len(performance_data)} 个策略性能数据")
    return performance_data


def clean_performance_data(performance_data):
    """
    清理性能数据，去除dataset、instance、sub_directory字段
    
    Args:
        performance_data: 原始性能数据
        
    Returns:
        dict: 清理后的性能数据
    """
    if not isinstance(performance_data, dict):
        return performance_data
    
    # 如果包含initialization_methods字段，说明这是完整的性能数据
    if 'initialization_methods' in performance_data:
        cleaned_data = {}
        for key, value in performance_data.items():
            if key not in ['dataset', 'instance', 'sub_directory']:
                cleaned_data[key] = value
        return cleaned_data
    
    # 否则，按原来的逻辑处理
    cleaned_data = {}
    for strategy_name, strategy_data in performance_data.items():
        if isinstance(strategy_data, dict):
            # 复制策略数据，但去除不需要的字段
            cleaned_strategy = {}
            for key, value in strategy_data.items():
                if key not in ['dataset', 'instance', 'sub_directory']:
                    cleaned_strategy[key] = value
            cleaned_data[strategy_name] = cleaned_strategy
        else:
            cleaned_data[strategy_name] = strategy_data
    
    return cleaned_data


def create_labeled_dataset(features_data, performance_data, output_file):
    """
    创建标记数据集
    
    Args:
        features_data: 特征数据字典
        performance_data: 性能数据字典
        output_file: 输出文件路径
    """
    print("正在创建标记数据集...")
    
    labeled_dataset = {}
    
    # 创建文件名到完整路径的映射
    filename_to_path = {}
    for fjs_path in features_data.keys():
        normalized_path = normalize_fjs_path(fjs_path)
        filename = os.path.basename(normalized_path)
        if filename not in filename_to_path:
            filename_to_path[filename] = []
        filename_to_path[filename].append(normalized_path)
    
    # 处理特征数据
    for fjs_path, features in features_data.items():
        # 标准化路径格式
        normalized_path = normalize_fjs_path(fjs_path)
        
        # 创建数据集条目
        dataset_entry = {
            "fjs_path": normalized_path,
            "features": features,
            "performance_data": None
        }
        
        # 查找对应的性能数据
        matched = False
        
        # 方法1: 直接匹配完整路径
        if normalized_path in performance_data:
            # 清理性能数据
            cleaned_performance = clean_performance_data(performance_data[normalized_path])
            dataset_entry["performance_data"] = cleaned_performance
            print(f"✓ 直接匹配: {normalized_path}")
            matched = True
        
        # 方法2: 通过文件名匹配
        if not matched:
            filename = os.path.basename(normalized_path)
            if filename in performance_data:
                # 清理性能数据
                cleaned_performance = clean_performance_data(performance_data[filename])
                dataset_entry["performance_data"] = cleaned_performance
                print(f"✓ 文件名匹配: {normalized_path} -> {filename}")
                matched = True
        
        if not matched:
            print(f"✗ 未找到性能数据: {normalized_path}")
        
        labeled_dataset[normalized_path] = dataset_entry
    
    # 处理只有性能数据但没有特征数据的情况
    for fjs_path, performance in performance_data.items():
        # 检查是否已经处理过
        if fjs_path not in labeled_dataset:
            # 检查是否有对应的特征数据
            filename = os.path.basename(fjs_path)
            found_features = False
            
            if filename in filename_to_path:
                # 如果有多个同名文件，选择第一个
                matched_path = filename_to_path[filename][0]
                if matched_path in labeled_dataset:
                    # 清理性能数据
                    cleaned_performance = clean_performance_data(performance)
                    labeled_dataset[matched_path]["performance_data"] = cleaned_performance
                    print(f"✓ 反向匹配: {fjs_path} -> {matched_path}")
                    found_features = True
            
            if not found_features:
                # 清理性能数据
                cleaned_performance = clean_performance_data(performance)
                dataset_entry = {
                    "fjs_path": fjs_path,
                    "features": None,
                    "performance_data": cleaned_performance
                }
                labeled_dataset[fjs_path] = dataset_entry
                print(f"⚠ 只有性能数据: {fjs_path}")
    
    # 保存到文件
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(labeled_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"标记数据集已保存到: {output_file}")
        print(f"总数据集数量: {len(labeled_dataset)}")
        
        # 统计信息
        with_features = sum(1 for entry in labeled_dataset.values() if entry["features"] is not None)
        with_performance = sum(1 for entry in labeled_dataset.values() if entry["performance_data"] is not None)
        complete = sum(1 for entry in labeled_dataset.values() 
                      if entry["features"] is not None and entry["performance_data"] is not None)
        
        print(f"有特征数据的: {with_features}")
        print(f"有性能数据的: {with_performance}")
        print(f"完整数据: {complete}")
        
    except Exception as e:
        print(f"错误: 无法保存标记数据集: {e}")


def main():
    """主函数"""
    # 配置路径
    features_file = "../output/dataset_features.json"
    strategy_dir = "../output/init_validity_result"
    output_file = "labeled_dataset/labeled_fjs_dataset.json"
    
    print("=== 创建标记数据集 ===")
    
    # 加载特征数据
    features_data = load_dataset_features(features_file)
    if not features_data:
        print("错误: 无法加载特征数据")
        return
    
    # 加载策略性能数据
    performance_data = load_strategy_performance(strategy_dir)
    if not performance_data:
        print("警告: 无法加载策略性能数据")
    
    # 创建标记数据集
    create_labeled_dataset(features_data, performance_data, output_file)
    
    print("=== 完成 ===")


if __name__ == "__main__":
    main() 