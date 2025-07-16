#!/usr/bin/env python3
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, '.')

try:
    from initial_validation.utils import parser
    print("✓ 成功导入parser模块")
    
    fjs_path = 'D:/0-MyCode/Dispatch_Sample_Generator/recommend_model_1/result/new_data_Behnke29.fjs'
    print(f"测试文件路径: {fjs_path}")
    
    # 检查文件是否存在
    if os.path.exists(fjs_path):
        print("✓ 文件存在")
    else:
        print("✗ 文件不存在")
        exit(1)
    
    # 解析文件
    result = parser.parse(fjs_path)
    print("✓ 解析成功")
    print(f"机器数量: {result['machinesNb']}")
    print(f"作业数量: {len(result['jobs'])}")
    
    # 检查第一个作业
    if len(result['jobs']) > 0:
        first_job = result['jobs'][0]
        print(f"第一个作业操作数量: {len(first_job)}")
        if len(first_job) > 0:
            first_operation = first_job[0]
            print(f"第一个操作机器数量: {len(first_operation)}")
    
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc() 