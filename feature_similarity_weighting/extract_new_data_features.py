import os
import json
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.utils import parser
from feature_extraction.feature_extractor import FeatureExtractor
from base_config import KDE_PARAMS, X_GRID

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

def generate_kde(data, bandwidth, x_grid):
    """生成KDE并评估密度"""
    kde = gaussian_kde(data, bw_method=bandwidth/np.std(data))
    return kde.evaluate(x_grid)

def plot_pdf(x_grid, density, title, output_path):
    """绘制概率密度图"""
    # 设置中文字体
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 设置背景样式
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # 绘制主曲线
    ax.plot(x_grid, density, 'b-', linewidth=1.5, alpha=0.8, label='概率密度')
    
    # 填充曲线下方区域
    ax.fill_between(x_grid, density, alpha=0.2, color='blue')
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('加工时间（单位s）', fontsize=12, labelpad=10)
    ax.set_ylabel('概率密度（Density）', fontsize=12, labelpad=10)
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # 设置y轴从0开始
    ax.set_ylim(bottom=0)
    
    # 优化刻度
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 添加图例
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    try:
        # 获取当前目录下的所有new_data_*目录
        data_dirs = [d for d in os.listdir(current_dir) if d.startswith('new_data_') and os.path.isdir(os.path.join(current_dir, d))]
        
        for data_dir in data_dirs:
            print(f"\n处理目录: {data_dir}")
            # 获取目录下的fjs文件
            fjs_files = [f for f in os.listdir(os.path.join(current_dir, data_dir)) if f.endswith('.fjs')]
            
            for fjs_file in fjs_files:
                print(f"\n处理文件: {fjs_file}")
                # 设置输入和输出路径
                input_file = os.path.join(current_dir, data_dir, fjs_file)
                output_dir = os.path.join(current_dir, data_dir)
                os.makedirs(output_dir, exist_ok=True)

                # 1. 提取基础特征
                print("正在提取基础特征...")
                parameters = parser.parse(input_file)
                extractor = FeatureExtractor(parameters)
                features = extractor.extract_all_features()
                
                # 保存基础特征
                features_output = os.path.join(output_dir, "new_data_features.json")
                with open(features_output, 'w', encoding='utf-8') as f:
                    json.dump({fjs_file: features}, f, indent=2, ensure_ascii=False)
                print(f"基础特征已保存到: {features_output}")

                # 2. 生成KDE特征
                print("\n正在生成KDE特征...")
                processing_times = extract_processing_times(input_file)
                
                # 从base_config.py获取参数
                bandwidth = KDE_PARAMS["bandwidth"]
                x_grid = np.array(X_GRID)
                
                # 生成KDE
                density = generate_kde(processing_times, bandwidth, x_grid)
                
                # 保存KDE结果
                kde_results = {
                    fjs_file: {
                        "x_grid": x_grid.tolist(),
                        "density": density.tolist(),
                        "bandwidth": bandwidth
                    }
                }
                
                kde_output = os.path.join(output_dir, "new_data_kde.json")
                with open(kde_output, 'w') as f:
                    json.dump(kde_results, f, indent=4)
                print(f"KDE特征已保存到: {kde_output}")

                # 3. 生成概率密度图
                print("\n正在生成概率密度图...")
                # 设置图表标题
                title = f"{fjs_file} - 加工时间概率密度分布"
                
                # 设置输出路径
                pdf_output = os.path.join(output_dir, "new_data_pdf.png")
                
                # 绘制并保存图表
                plot_pdf(x_grid, density, title, pdf_output)
                print(f"概率密度图已保存到: {pdf_output}")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 