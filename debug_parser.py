#!/usr/bin/env python3
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, '.')

from initial_validation.utils import parser

def debug_parser():
    fjs_file = 'D:/0-MyCode/Dispatch_Sample_Generator/recommend_model_1/result/new_data_Behnke29.fjs'
    
    print("=== 调试FJS文件解析 ===")
    print(f"文件路径: {fjs_file}")
    
    try:
        with open(fjs_file, 'r') as file:
            # 读取第一行
            firstLine = file.readline()
            print(f"第一行: {firstLine.strip()}")
            
            firstLineValues = list(map(int, firstLine.split()[0:2]))
            print(f"第一行解析值: {firstLineValues}")
            
            jobsNb = firstLineValues[0]
            machinesNb = firstLineValues[1]
            print(f"作业数量: {jobsNb}, 机器数量: {machinesNb}")
            
            # 逐个检查作业行
            for i in range(jobsNb):  # 检查所有作业
                currentLine = file.readline()
                if not currentLine:
                    print(f"作业 {i+1}: 文件结束")
                    break
                    
                print(f"\n作业 {i+1} 行: {currentLine.strip()}")
                currentLineValues = list(map(int, currentLine.split()))
                print(f"作业 {i+1} 值数量: {len(currentLineValues)}")
                print(f"作业 {i+1} 前10个值: {currentLineValues[:10]}")
                
                if len(currentLineValues) < 2:
                    print(f"作业 {i+1}: 数据不足")
                    continue
                
                j = 1
                k = currentLineValues[j]  # 操作数量
                print(f"作业 {i+1} 操作数量: {k}")
                j += 1
                
                # 检查是否有足够的数据
                required_values = 1 + k * 2  # 操作数量 + k个(机器,时间)对
                if len(currentLineValues) < required_values:
                    print(f"作业 {i+1}: 数据不足，需要{required_values}个值，实际只有{len(currentLineValues)}个")
                    continue
                
                # 解析操作
                for ik in range(k):
                    if j >= len(currentLineValues):
                        print(f"作业 {i+1} 操作 {ik+1}: 索引越界 j={j}, len={len(currentLineValues)}")
                        break
                    machine = currentLineValues[j]
                    j += 1
                    if j >= len(currentLineValues):
                        print(f"作业 {i+1} 操作 {ik+1}: 索引越界 j={j}, len={len(currentLineValues)}")
                        break
                    processingTime = currentLineValues[j]
                    j += 1
                    print(f"  操作 {ik+1}: 机器 {machine}, 时间 {processingTime}")
                
    except Exception as e:
        print(f"解析错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_parser() 