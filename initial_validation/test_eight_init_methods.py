#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试八种初始化方法的脚本
"""

import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.utils import parser
from initial_validation.ga_fjsp_extends import ga_new

def test_init_methods():
    """
    测试八种初始化方法
    """
    # 使用一个简单的测试文件
    test_file = os.path.join(project_root, "dataset", "Brandimarte", "Mk01.fjs")
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    print("开始测试八种初始化方法...")
    print(f"使用测试文件: {test_file}")
    
    try:
        # 解析FJS文件
        parameters = parser.parse(test_file)
        print(f"解析成功 - 机器数: {parameters['machinesNb']}, 作业数: {len(parameters['jobs'])}")
        
        # 八种初始化方法
        init_methods = [
            "random", "heuristic", "mixed",
            "FIFO_SPT", "FIFO_EET", "MOPNR_SPT", "MOPNR_EET",
            "LWKR_SPT", "LWKR_EET", "MWKR_SPT", "MWKR_EET"
        ]
        
        results = {}
        
        for method in init_methods:
            print(f"\n测试 {method} 初始化方法...")
            try:
                # 运行遗传算法（只运行少量迭代以快速测试）
                best_makespan, convergence_curve = ga_new(parameters, method, return_convergence=True)
                results[method] = {
                    "best_makespan": best_makespan,
                    "convergence_length": len(convergence_curve)
                }
                print(f"  {method}: Makespan = {best_makespan}, 收敛代数 = {len(convergence_curve)}")
            except Exception as e:
                print(f"  {method}: 错误 - {e}")
                results[method] = {"error": str(e)}
        
        # 输出测试结果总结
        print("\n" + "="*50)
        print("测试结果总结:")
        print("="*50)
        
        for method, result in results.items():
            if "error" in result:
                print(f"{method:12s}: ❌ {result['error']}")
            else:
                print(f"{method:12s}: ✅ Makespan={result['best_makespan']:4d}, 收敛={result['convergence_length']:2d}")
        
        print("\n测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")

if __name__ == "__main__":
    test_init_methods() 