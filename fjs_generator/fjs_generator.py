import os
import random
import numpy as np
from typing import List, Dict, Tuple

class FJSGenerator:
    def __init__(self, input_file: str = None):
        """
        初始化FJS生成器
        :param input_file: 可选的输入FJS文件路径，用于基于现有文件生成新文件
        """
        self.jobs = 0
        self.machines = 0
        self.flexibility = 0
        self.processing_times = []
        
        if input_file:
            self._load_input_file(input_file)
    
    def _load_input_file(self, input_file: str):
        """加载输入FJS文件"""
        with open(input_file, 'r') as f:
            # 读取第一行
            first_line = f.readline().strip().split()
            self.jobs = int(first_line[0])
            self.machines = int(first_line[1])
            self.flexibility = int(first_line[2]) if len(first_line) > 2 else 2
            
            # 读取每个工件的数据
            self.processing_times = []
            for _ in range(self.jobs):
                line = f.readline().strip().split()
                operations = int(line[0])
                operation_times = []
                idx = 1
                
                for _ in range(operations):
                    k = int(line[idx])  # 可选机器数
                    idx += 1
                    times = []
                    for _ in range(k):
                        machine = int(line[idx])
                        time = int(line[idx + 1])
                        times.append((machine, time))
                        idx += 2
                    operation_times.append(times)
                
                self.processing_times.append(operation_times)
    
    def generate_new_instance(self, 
                        output_file: str,
                        num_jobs: int = None,
                        num_machines: int = None,
                        num_operations: int = None,
                        flexibility: int = None,
                        time_range: Tuple[int, int] = (10, 100),
                        time_variation: float = 0.2) -> None:
        """
        生成新的FJS实例
        :param output_file: 输出文件路径
        :param num_jobs: 工件数量，None表示保持原样
        :param num_machines: 机器数量，None表示保持原样
        :param num_operations: 每个工件的工序数，None表示随机生成3-7个工序
        :param flexibility: 每个工序的可选机器数，None表示保持原样
        :param time_range: 加工时间范围（最小值，最大值）
        :param time_variation: 加工时间变化范围（相对于原始时间的百分比）
        """
        # 设置参数
        num_jobs = num_jobs if num_jobs is not None else self.jobs
        num_machines = num_machines if num_machines is not None else self.machines
        flexibility = flexibility if flexibility is not None else self.flexibility
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 生成新的实例
        with open(output_file, 'w') as f:
            # 写入第一行
            f.write(f"{num_jobs} {num_machines} {flexibility}\n")
            
            # 生成每个工件的数据
            for job_idx in range(num_jobs):
                # 确定工序数
                if num_operations is not None:
                    operations = num_operations
                else:
                    # 随机生成工序数，范围在3-7之间
                    operations = random.randint(3, 7)
                
                # 写入工序数
                f.write(f"{operations}")
                
                # 生成每个工序的数据
                for op_idx in range(operations):
                    # 确定可选机器数
                    k = min(flexibility, num_machines)
                    f.write(f" {k}")
                    
                    # 生成机器和加工时间
                    machines = random.sample(range(1, num_machines + 1), k)
                    for machine in machines:
                        time = random.randint(time_range[0], time_range[1])
                        f.write(f" {machine} {time}")
                
                f.write("\n")
    
    def adjust_flexibility(self, 
                          input_file: str,
                          output_file: str,
                          new_flexibility: int,
                          time_variation: float = 0.2) -> None:
        """
        调整现有FJS文件的机器柔性
        :param input_file: 输入文件路径
        :param output_file: 输出文件路径
        :param new_flexibility: 新的机器柔性
        :param time_variation: 加工时间变化范围
        """
        self._load_input_file(input_file)
        self.generate_new_instance(
            output_file=output_file,
            flexibility=new_flexibility,
            time_variation=time_variation
        )

def main():
    # 创建生成器实例
    generator = FJSGenerator("dataset/Hurink/rdata/la02.fjs")
    
    # 生成新的实例，随机调整加工时间变化范围
    generator.generate_new_instance(
        output_file="feature_similarity_weighting/new_data_ptr/new_rdata_la02_ptr.fjs",
        num_jobs=10,  # 保持原有工件数
        num_machines=5,  # 保持原有机器数
        num_operations=5,  # 保持原有工序数
        flexibility=2,  # 保持原有机器柔性
        time_range=(10, 100),  # 保持合理的加工时间范围
        time_variation=0.5  # 允许50%的加工时间变化
    )

if __name__ == "__main__":
    main()
