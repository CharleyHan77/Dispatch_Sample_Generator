#!/usr/bin/env python3
import sys
import os

# 添加fjs_generator到路径
sys.path.insert(0, 'fjs_generator')

from fjs_generator import FJSGenerator

def main():
    # 输入文件路径
    input_file = "dataset/BehnkeGeiger/Behnke29.fjs"
    
    # 输出文件路径
    output_file = "recommend_model_1/result/compare_with_random/new_behnke29_variant.fjs"
    
    print(f"基于 {input_file} 生成新的FJS文件...")
    
    # 创建生成器实例
    generator = FJSGenerator(input_file)
    
    print(f"原始文件信息:")
    print(f"  作业数量: {generator.jobs}")
    print(f"  机器数量: {generator.machines}")
    print(f"  柔性度: {generator.flexibility}")
    
    # 生成新的实例，保持原有参数但调整加工时间
    generator.generate_new_instance(
        output_file=output_file,
        num_jobs=generator.jobs,  # 保持原有作业数
        num_machines=generator.machines,  # 保持原有机器数
        num_operations=None,  # 随机生成工序数
        flexibility=generator.flexibility,  # 保持原有柔性度
        time_range=(10, 100),  # 加工时间范围
        time_variation=0.3  # 允许30%的加工时间变化
    )
    
    print(f"新文件已生成: {output_file}")
    print("生成完成！")

if __name__ == "__main__":
    main() 