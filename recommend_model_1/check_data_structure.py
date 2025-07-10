#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查标记数据集结构
"""

import json

def check_data_structure():
    """检查标记数据集结构"""
    try:
        with open('labeled_dataset/labeled_fjs_dataset.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"数据集大小: {len(data)}")
        
        # 获取第一个样本
        first_key = list(data.keys())[0]
        sample = data[first_key]
        
        print(f"第一个样本: {first_key}")
        print(f"包含字段: {list(sample.keys())}")
        
        if 'features' in sample and sample['features']:
            print(f"特征字段: {list(sample['features'].keys())}")
        
        if 'performance_data' in sample and sample['performance_data']:
            print(f"性能数据字段: {list(sample['performance_data'].keys())}")
            
            if 'initialization_methods' in sample['performance_data']:
                init_methods = sample['performance_data']['initialization_methods']
                print(f"初始化策略: {list(init_methods.keys())}")
                
                # 检查第一个策略的字段
                first_strategy = list(init_methods.keys())[0]
                print(f"第一个策略 '{first_strategy}' 的字段: {list(init_methods[first_strategy].keys())}")
        
        # 检查是否有dataset字段
        if 'performance_data' in sample and sample['performance_data']:
            for key, value in sample['performance_data'].items():
                if isinstance(value, dict) and 'dataset' in value:
                    print(f"警告: 字段 {key} 仍包含 'dataset' 字段")
                if isinstance(value, dict) and 'instance' in value:
                    print(f"警告: 字段 {key} 仍包含 'instance' 字段")
                if isinstance(value, dict) and 'sub_directory' in value:
                    print(f"警告: 字段 {key} 仍包含 'sub_directory' 字段")
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    check_data_structure() 