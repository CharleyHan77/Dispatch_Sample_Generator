#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试路径解析逻辑
"""

import os

def test_path_parsing():
    """测试路径解析逻辑"""
    
    # 模拟数据集根目录
    dataset_root = "dataset"
    
    # 测试用例
    test_cases = [
        "Barnes/mt10c1.fjs",  # 2级路径
        "Hurink/edata/la01.fjs",  # 3级路径
        "Hurink/rdata/abz5.fjs",  # 3级路径
        "Hurink/sdata/car1.fjs",  # 3级路径
        "Hurink/vdata/la01.fjs",  # 3级路径
        "Brandimarte/Mk01.fjs",  # 2级路径
        "Kacem/Kacem1.fjs",  # 2级路径
    ]
    
    print("测试路径解析逻辑:")
    print("=" * 60)
    
    for rel_path in test_cases:
        rel_path = rel_path.replace('\\', '/')  # 统一路径分隔符
        
        # 解析数据集名和实例名，保持子目录结构
        path_parts = rel_path.split('/')
        if len(path_parts) == 1:
            # 直接在dataset根目录下的文件
            dataset_name = "dataset"
            sub_dir = ""
            instance_name = os.path.splitext(path_parts[0])[0]
        elif len(path_parts) == 2:
            # 在子目录中的文件（如 Barnes/mt10c1.fjs）
            dataset_name = path_parts[0]  # 第一级目录名作为数据集名
            sub_dir = ""  # 没有子目录
            instance_name = os.path.splitext(path_parts[1])[0]  # 文件名（不含扩展名）
        elif len(path_parts) == 3:
            # 在子目录中的文件（如 Hurink/edata/la01.fjs）
            dataset_name = path_parts[0]  # 第一级目录名作为数据集名
            sub_dir = path_parts[1]  # 第二级目录名作为子目录
            instance_name = os.path.splitext(path_parts[2])[0]  # 文件名（不含扩展名）
        else:
            # 更深层次的目录结构
            dataset_name = path_parts[0]  # 第一级目录名作为数据集名
            sub_dir = path_parts[1]  # 第二级目录名作为子目录
            instance_name = os.path.splitext(path_parts[-1])[0]  # 最后一级文件名（不含扩展名）
        
        # 组织输出目录和文件名，保持子目录结构
        if sub_dir:
            # 如果有子目录，则在输出目录中创建对应的子目录结构
            out_dir = f"output/init_validity_result/{dataset_name}/{sub_dir}"
            plot_dir = f"output/convergence_curves/{dataset_name}/{sub_dir}"
        else:
            # 没有子目录的情况
            out_dir = f"output/init_validity_result/{dataset_name}"
            plot_dir = f"output/convergence_curves/{dataset_name}"
        
        out_json = f"{out_dir}/{instance_name}_validation_results.json"
        plot_path = f"{plot_dir}/{instance_name}_convergence_curves.png"
        
        print(f"输入路径: {rel_path}")
        print(f"  数据集名: {dataset_name}")
        print(f"  子目录: {sub_dir if sub_dir else '无'}")
        print(f"  实例名: {instance_name}")
        print(f"  输出目录: {out_dir}")
        print(f"  图表目录: {plot_dir}")
        print(f"  结果文件: {out_json}")
        print(f"  图表文件: {plot_path}")
        print("-" * 60)

if __name__ == "__main__":
    test_path_parsing() 