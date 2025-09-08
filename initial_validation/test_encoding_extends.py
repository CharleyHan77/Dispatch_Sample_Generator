#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试encoding_extends.py文件的脚本
"""

import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.utils import parser
from initial_validation.genetic import encoding_extends

def test_encoding_extends_structure():
    """
    测试encoding_extends.py文件的结构
    """
    print("="*60)
    print("测试encoding_extends.py文件结构")
    print("="*60)
    
    # 1. 测试初始化方法信息
    print("\n1. 测试初始化方法信息:")
    info = encoding_extends.get_init_method_info()
    for method, details in info.items():
        print(f"  {method}: {details['name']}")
        print(f"    描述: {details['description']}")
        print(f"    工序策略: {details['os_detail']}")
        print(f"    机器策略: {details['ms_detail']}")
        print()
    
    # 2. 测试方法验证
    print("\n2. 测试方法验证:")
    valid_methods = ["FIFO_SPT", "FIFO_EET", "MOPNR_SPT", "MOPNR_EET",
                    "LWKR_SPT", "LWKR_EET", "MWKR_SPT", "MWKR_EET"]
    invalid_methods = ["random", "heuristic", "mixed", "invalid_method"]
    
    for method in valid_methods:
        is_valid = encoding_extends.validate_init_method(method)
        print(f"  {method}: {'✅' if is_valid else '❌'}")
    
    for method in invalid_methods:
        is_valid = encoding_extends.validate_init_method(method)
        print(f"  {method}: {'✅' if is_valid else '❌'}")
    
    # 3. 测试函数获取
    print("\n3. 测试函数获取:")
    for method in valid_methods:
        try:
            func = encoding_extends.get_init_method_function(method)
            print(f"  {method}: ✅ 函数获取成功")
        except Exception as e:
            print(f"  {method}: ❌ 函数获取失败 - {e}")
    
    # 4. 测试无效方法
    try:
        func = encoding_extends.get_init_method_function("invalid_method")
        print("  invalid_method: ❌ 应该抛出异常但没有")
    except ValueError as e:
        print(f"  invalid_method: ✅ 正确抛出异常 - {e}")


def test_encoding_extends_with_data():
    """
    使用实际数据测试encoding_extends.py
    """
    print("\n" + "="*60)
    print("使用实际数据测试encoding_extends.py")
    print("="*60)
    
    # 使用一个简单的测试文件
    test_file = os.path.join(project_root, "dataset", "Brandimarte", "Mk01.fjs")
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    try:
        # 解析FJS文件
        parameters = parser.parse(test_file)
        print(f"解析成功 - 机器数: {parameters['machinesNb']}, 作业数: {len(parameters['jobs'])}")
        
        # 测试八种初始化方法
        init_methods = [
            "FIFO_SPT", "FIFO_EET", "MOPNR_SPT", "MOPNR_EET",
            "LWKR_SPT", "LWKR_EET", "MWKR_SPT", "MWKR_EET"
        ]
        
        results = {}
        
        for method in init_methods:
            print(f"\n测试 {method} 初始化方法...")
            try:
                # 获取初始化函数
                init_func = encoding_extends.get_init_method_function(method)
                
                # 生成种群
                population = init_func(parameters)
                
                # 验证种群结构
                if len(population) > 0:
                    first_individual = population[0]
                    if isinstance(first_individual, tuple) and len(first_individual) == 2:
                        OS, MS = first_individual
                        if isinstance(OS, list) and isinstance(MS, list):
                            results[method] = {
                                "population_size": len(population),
                                "os_length": len(OS),
                                "ms_length": len(MS),
                                "status": "success"
                            }
                            print(f"  ✅ 成功 - 种群大小: {len(population)}, OS长度: {len(OS)}, MS长度: {len(MS)}")
                        else:
                            results[method] = {"status": "error", "message": "OS或MS不是列表"}
                            print(f"  ❌ 失败 - OS或MS不是列表")
                    else:
                        results[method] = {"status": "error", "message": "个体不是(OS, MS)元组"}
                        print(f"  ❌ 失败 - 个体不是(OS, MS)元组")
                else:
                    results[method] = {"status": "error", "message": "种群为空"}
                    print(f"  ❌ 失败 - 种群为空")
                    
            except Exception as e:
                results[method] = {"status": "error", "message": str(e)}
                print(f"  ❌ 失败 - {e}")
        
        # 输出测试结果总结
        print("\n" + "="*50)
        print("测试结果总结:")
        print("="*50)
        
        success_count = 0
        for method, result in results.items():
            if result["status"] == "success":
                print(f"{method:12s}: ✅ 成功 - 种群{result['population_size']:2d}, OS{result['os_length']:3d}, MS{result['ms_length']:3d}")
                success_count += 1
            else:
                print(f"{method:12s}: ❌ 失败 - {result['message']}")
        
        print(f"\n成功: {success_count}/{len(init_methods)}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")


def test_individual_functions():
    """
    测试各个单独的生成函数
    """
    print("\n" + "="*60)
    print("测试各个单独的生成函数")
    print("="*60)
    
    # 使用一个简单的测试文件
    test_file = os.path.join(project_root, "dataset", "Brandimarte", "Mk01.fjs")
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    try:
        # 解析FJS文件
        parameters = parser.parse(test_file)
        print(f"解析成功 - 机器数: {parameters['machinesNb']}, 作业数: {len(parameters['jobs'])}")
        
        # 测试OS生成函数
        print("\n测试OS生成函数:")
        os_functions = [
            ("generateOS_FIFO", encoding_extends.generateOS_FIFO),
            ("generateOS_MOPNR", encoding_extends.generateOS_MOPNR),
            ("generateOS_LWKR", encoding_extends.generateOS_LWKR),
            ("generateOS_MWKR", encoding_extends.generateOS_MWKR)
        ]
        
        for func_name, func in os_functions:
            try:
                OS = func(parameters)
                print(f"  {func_name:15s}: ✅ 成功 - 长度: {len(OS)}")
            except Exception as e:
                print(f"  {func_name:15s}: ❌ 失败 - {e}")
        
        # 测试MS生成函数
        print("\n测试MS生成函数:")
        ms_functions = [
            ("generateMS_SPT", encoding_extends.generateMS_SPT),
            ("generateMS_EET", encoding_extends.generateMS_EET),
            ("generateMS_LoadBalanced", encoding_extends.generateMS_LoadBalanced),
            ("generateMS_Adaptive", encoding_extends.generateMS_Adaptive)
        ]
        
        for func_name, func in ms_functions:
            try:
                MS = func(parameters)
                print(f"  {func_name:20s}: ✅ 成功 - 长度: {len(MS)}")
            except Exception as e:
                print(f"  {func_name:20s}: ❌ 失败 - {e}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")


if __name__ == "__main__":
    # 运行所有测试
    test_encoding_extends_structure()
    test_encoding_extends_with_data()
    test_individual_functions()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60) 