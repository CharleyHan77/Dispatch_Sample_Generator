import random
import os
from typing import List, Dict, Any, Optional

class FJSPDataGenerator:
    def __init__(self, 
                 num_jobs: Optional[int] = None,
                 num_machines: Optional[int] = None,
                 avg_processing_time: Optional[int] = None,
                 min_operations_per_job: int = 3,
                 max_operations_per_job: int = 7,
                 min_machines_per_operation: int = 1,
                 max_machines_per_operation: int = 3,
                 processing_time_variation: float = 0.3):
        """
        初始化FJSP数据生成器
        
        :param num_jobs: 作业数量，如果为None则随机生成(5-20)
        :param num_machines: 机器数量，如果为None则随机生成(5-20)
        :param avg_processing_time: 平均加工时间，如果为None则随机生成(10-50)
        :param min_operations_per_job: 每个作业的最小工序数
        :param max_operations_per_job: 每个作业的最大工序数
        :param min_machines_per_operation: 每个工序可选的最小机器数
        :param max_machines_per_operation: 每个工序可选的最大机器数
        :param processing_time_variation: 加工时间的变异系数
        """
        self.num_jobs = num_jobs if num_jobs is not None else random.randint(5, 20)
        self.num_machines = num_machines if num_machines is not None else random.randint(5, 20)
        self.avg_processing_time = avg_processing_time if avg_processing_time is not None else random.randint(10, 50)
        self.min_operations_per_job = min_operations_per_job
        self.max_operations_per_job = max_operations_per_job
        self.min_machines_per_operation = min_machines_per_operation
        self.max_machines_per_operation = max_machines_per_operation
        self.processing_time_variation = processing_time_variation
        
    def generate_processing_time(self) -> int:
        """生成加工时间，基于平均加工时间和变异系数"""
        variation = int(self.avg_processing_time * self.processing_time_variation)
        return max(1, self.avg_processing_time + random.randint(-variation, variation))
    
    def generate_operation(self) -> List[Dict[str, int]]:
        """生成一个工序的可选机器和加工时间"""
        # 随机选择可用的机器数量
        num_machines = random.randint(self.min_machines_per_operation, 
                                    min(self.max_machines_per_operation, self.num_machines))
        
        # 随机选择机器
        available_machines = random.sample(range(1, self.num_machines + 1), num_machines)
        
        # 为每个机器生成加工时间
        operation = []
        for machine in available_machines:
            processing_time = self.generate_processing_time()
            operation.append({
                'machine': machine,
                'processingTime': processing_time
            })
        
        return operation
    
    def generate_job(self) -> List[List[Dict[str, int]]]:
        """生成一个作业的所有工序"""
        # 随机选择工序数量
        num_operations = random.randint(self.min_operations_per_job, self.max_operations_per_job)
        
        # 生成每个工序
        job = []
        for _ in range(num_operations):
            operation = self.generate_operation()
            job.append(operation)
        
        return job
    
    def generate_data(self) -> Dict[str, Any]:
        """生成完整的FJSP数据"""
        jobs = []
        for _ in range(self.num_jobs):
            job = self.generate_job()
            jobs.append(job)
        
        return {
            'machinesNb': self.num_machines,
            'jobs': jobs
        }
    
    def save_to_file(self, file_path: str):
        """将生成的数据保存到文件"""
        data = self.generate_data()
        
        with open(file_path, 'w') as f:
            # 写入第一行：作业数和机器数
            f.write(f"{self.num_jobs} {self.num_machines}\n")
            
            # 写入每个作业的数据
            for job in data['jobs']:
                # 写入作业的工序数
                f.write(f"{len(job)} ")
                
                # 写入每个工序的数据
                for operation in job:
                    # 写入工序的可选机器数
                    f.write(f"{len(operation)} ")
                    
                    # 写入每个可选机器的数据
                    for option in operation:
                        f.write(f"{option['machine']} {option['processingTime']} ")
                
                f.write("\n")

def main():
    # 创建输出目录
    os.makedirs("new_data", exist_ok=True)
    
    # 创建数据生成器实例
    generator = FJSPDataGenerator(
        num_jobs=10,  # 可以修改这些参数
        num_machines=6,
        avg_processing_time=30
    )
    
    # 生成并保存数据
    output_file = os.path.join("new_data", "new_fjsp_data.fjs")
    generator.save_to_file(output_file)
    print(f"数据已生成并保存到: {output_file}")

if __name__ == "__main__":
    main() 