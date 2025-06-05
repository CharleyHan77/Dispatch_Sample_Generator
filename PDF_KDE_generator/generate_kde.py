import os
import json
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import glob

def extract_processing_times(fjs_file):
    """从fjs文件中提取所有加工时间"""
    processing_times = []
    with open(fjs_file, 'r') as f:
        lines = f.readlines()
        # 跳过前两行（作业数和机器数）
        for line in lines[2:]:
            # 分割每行数据
            parts = line.strip().split()
            if len(parts) > 1:  # 确保不是空行
                # 每两个数字为一组（机器编号和加工时间）
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        processing_times.append(float(parts[i + 1]))
    return processing_times

def calculate_bandwidth(data_list):
    """使用改进的Silverman规则计算统一带宽"""
    pooled_data = np.concatenate(data_list)
    n = len(pooled_data)
    sigma = np.std(pooled_data)
    iqr = stats.iqr(pooled_data)
    bandwidth = 0.9 * min(sigma, iqr/1.34) * (n ** (-1/5))
    
    # 打印中间值
    print(f"总数据量: {n}")
    print(f"标准差: {sigma:.4f}")
    print(f"IQR: {iqr:.4f}")
    print(f"计算得到的带宽: {bandwidth:.4f}")
    
    return n, sigma, iqr, bandwidth

def update_base_config(total_count, std_dev, iqr, bandwidth, x_grid):
    """更新base_config.py文件中的KDE参数"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "base_config.py")
    
    config_content = f'''"""
基础配置文件
1.包含KDE计算相关的参数，通过PDF_KDE_generator/generate_kde.py执行得到
（记录历史数据 概率密度函数的KPDF的全局统一带宽，用于新数据构造PDF）
"""

# 1.KDE计算参数
KDE_PARAMS = {{
    "total_data_count": {total_count},  # 总数据量
    "std_deviation": {std_dev:.4f},    # 标准差
    "iqr": {iqr:.4f},             # IQR
    "bandwidth": {bandwidth:.4f}         # 计算得到的带宽
}}

# 2.评估网格参数
X_GRID = {x_grid.tolist()}  # 用于KDE评估的统一网格点
'''
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"\n已更新配置文件: {config_path}")

def generate_kde(data, bandwidth, x_grid):
    """生成KDE并评估密度"""
    kde = gaussian_kde(data, bw_method=bandwidth/np.std(data))
    return kde.evaluate(x_grid)

def main():
    # 创建输出目录
    output_dir = "output/PDF_KDE_generator"
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有fjs文件
    fjs_files = []
    for root, dirs, files in os.walk("dataset"):
        for file in files:
            if file.endswith(".fjs"):
                fjs_files.append(os.path.join(root, file))
    
    # 提取所有加工时间
    all_processing_times = []
    file_data_map = {}
    
    for fjs_file in fjs_files:
        processing_times = extract_processing_times(fjs_file)
        all_processing_times.append(processing_times)
        file_data_map[fjs_file] = processing_times
    
    # 计算统一带宽
    n, sigma, iqr, bandwidth = calculate_bandwidth(all_processing_times)
    
    # 创建统一的评估网格
    x_grid = np.linspace(
        min([min(data) for data in all_processing_times]),
        max([max(data) for data in all_processing_times]),
        1000
    )
    
    # 更新配置文件
    update_base_config(n, sigma, iqr, bandwidth, x_grid)
    
    # 生成每个文件的KDE并保存结果
    results = {}
    for fjs_file, processing_times in file_data_map.items():
        dataset_name = os.path.basename(os.path.dirname(fjs_file))
        file_name = os.path.basename(fjs_file)
        
        # 生成KDE
        density = generate_kde(processing_times, bandwidth, x_grid)
        
        # 处理文件路径
        if dataset_name in ['edata', 'sdata', 'rdata', 'vdata']:
            key = f"Hurink/{dataset_name}/{file_name}"
        else:
            key = f"{dataset_name}/{file_name}"
        
        # 保存结果
        results[key] = {
            "x_grid": x_grid.tolist(),
            "density": density.tolist(),
            "bandwidth": bandwidth
        }
    
    # 保存到JSON文件
    output_file = os.path.join(output_dir, "kde_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 