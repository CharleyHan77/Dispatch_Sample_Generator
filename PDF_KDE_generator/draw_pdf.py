import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def load_kde_results(json_file):
    """加载KDE结果"""
    with open(json_file, 'r') as f:
        return json.load(f)

def create_output_dirs(base_dir, dataset_names):
    """创建输出目录结构"""
    for dataset in dataset_names:
        os.makedirs(os.path.join(base_dir, dataset), exist_ok=True)

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
    # 输入和输出路径
    input_file = "output/PDF_KDE_generator/kde_results.json"
    output_base_dir = "output/PDF_KDE_generator/PDF"
    
    # 加载KDE结果
    kde_results = load_kde_results(input_file)
    
    # 获取所有数据集名称
    dataset_names = set()
    for key in kde_results.keys():
        dataset_name = key.split('/')[0]
        dataset_names.add(dataset_name)
    
    # 创建输出目录
    create_output_dirs(output_base_dir, dataset_names)
    
    # 为每个文件绘制PDF图
    for key, data in kde_results.items():
        dataset_name, file_name = key.split('/')
        file_name = file_name.replace('.fjs', '.png')
        
        # 准备数据
        x_grid = np.array(data['x_grid'])
        density = np.array(data['density'])
        
        # 设置输出路径
        output_path = os.path.join(output_base_dir, dataset_name, file_name)
        
        # 设置图表标题
        title = f"{dataset_name}/{file_name.replace('.png', '')} - 加工时间概率密度分布"
        
        # 绘制并保存图表
        plot_pdf(x_grid, density, title, output_path)
        print(f"已生成: {output_path}")

if __name__ == "__main__":
    main()
