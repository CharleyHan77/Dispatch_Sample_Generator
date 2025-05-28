import numpy as np
from typing import Dict, Any
import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.utils import parser

class FeatureExtractor:
    def __init__(self, parameters: Dict[str, Any]):
        """
        初始化特征提取器
        :param parameters: 从parser.parse()得到的解析结果
        """
        self.parameters = parameters
        self.jobs = parameters['jobs']
        self.machines = parameters['machinesNb']
        
    def extract_basic_features(self) -> Dict[str, Any]:
        """提取基础特征"""
        # 计算工序总数
        total_operations = sum(len(job) for job in self.jobs)
        
        # 计算每个工序的可选机器数
        available_machines = [len(operation) for job in self.jobs for operation in job]
        avg_available_machines = np.mean(available_machines)
        std_available_machines = np.std(available_machines)
        
        return {
            "num_jobs": len(self.jobs),
            "num_machines": self.machines,
            "total_operations": total_operations,
            "avg_available_machines": float(avg_available_machines),
            "std_available_machines": float(std_available_machines)
        }
    
    def extract_processing_time_features(self) -> Dict[str, Any]:
        """提取加工时间特征"""
        # 收集所有加工时间
        all_processing_times = []
        for job in self.jobs:
            for operation in job:
                for machine_info in operation:
                    all_processing_times.append(machine_info['processingTime'])
        
        # 计算加工时间统计特征
        processing_times = np.array(all_processing_times, dtype=float)
        
        # 计算不同机器间的加工时间差异
        machine_times = {}
        for job in self.jobs:
            for operation in job:
                for machine_info in operation:
                    machine = machine_info['machine']
                    time = machine_info['processingTime']
                    if machine not in machine_times:
                        machine_times[machine] = []
                    machine_times[machine].append(float(time))
        
        # 计算每台机器的平均加工时间
        machine_avg_times = {machine: np.mean(times) for machine, times in machine_times.items()}
        machine_time_variance = np.var(list(machine_avg_times.values()))
        
        return {
            "processing_time_mean": float(np.mean(processing_times)),
            "processing_time_std": float(np.std(processing_times)),
            "processing_time_min": float(np.min(processing_times)),
            "processing_time_max": float(np.max(processing_times)),
            "machine_time_variance": float(machine_time_variance)
        }
    
    def extract_all_features(self) -> Dict[str, Any]:
        """提取所有特征"""
        return {
            "basic_features": self.extract_basic_features(),
            "processing_time_features": self.extract_processing_time_features()
        }

def main():
    try:
        # 获取当前文件所在目录的上级目录（项目根目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        file_path = os.path.join(project_root, "dataset", "Brandimarte", "Text", "Mk01.fjs")
        
        print(f"正在读取文件: {file_path}")
        parameters = parser.parse(file_path)
        print("\n解析得到的参数：")
        print(parameters)
        
        extractor = FeatureExtractor(parameters)
        features = extractor.extract_all_features()
        
        # 打印特征（使用json格式）
        import json
        print("\n提取的特征：")
        print(json.dumps(features, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 