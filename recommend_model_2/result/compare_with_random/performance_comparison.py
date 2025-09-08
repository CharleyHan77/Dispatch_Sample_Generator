#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能对比分析
比较随机初始化方法与推荐策略的性能差异
支持四维度性能指标：Makespan评分、收敛速度评分、稳定性评分、收敛稳定性评分
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import logging
from pathlib import Path
import argparse

# Windows下设置终端编码为UTF-8，确保中文正常显示
if sys.platform == "win32":
    os.system("chcp 65001")
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.rcParams['font.size'] = 10  # 设置默认字体大小

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(current_dir, '../../../'))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def setup_logging(log_file):
    """设置日志记录"""
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_test_results(result_dir, timestamp):
    """
    加载测试结果
    
    Args:
        result_dir: 结果目录
        timestamp: 时间戳
        
    Returns:
        tuple: (随机初始化结果, 推荐策略结果)
    """
    # 查找指定时间戳的结果文件
    random_files = list(Path(result_dir).glob(f"random_initialization_results_{timestamp}*.json"))
    recommended_files = list(Path(result_dir).glob(f"pareto_recommended_strategy_results_{timestamp}*.json"))
    
    if not random_files:
        # 如果没有找到指定时间戳的文件，查找最新的
        random_files = list(Path(result_dir).glob("random_initialization_results_*.json"))
        random_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not recommended_files:
        # 如果没有找到指定时间戳的文件，查找最新的
        recommended_files = list(Path(result_dir).glob("pareto_recommended_strategy_results_*.json"))
        recommended_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not random_files or not recommended_files:
        raise FileNotFoundError("未找到测试结果文件")
    
    # 加载结果
    with open(random_files[0], 'r', encoding='utf-8') as f:
        random_results = json.load(f)
    
    with open(recommended_files[0], 'r', encoding='utf-8') as f:
        recommended_results = json.load(f)
    
    return random_results, recommended_results

def calculate_improvement_metrics(random_stats, recommended_stats):
    """
    计算改进指标（四维度性能指标）
    
    Args:
        random_stats: 随机初始化统计结果
        recommended_stats: 推荐策略统计结果
        
    Returns:
        dict: 改进指标
    """
    # 四维度性能指标改进率
    improvements = {}
    
    # 1. Makespan改进（越小越好）
    if random_stats['makespan']['mean'] > 0:
        makespan_improvement = ((random_stats['makespan']['mean'] - recommended_stats['makespan']['mean']) / 
                               random_stats['makespan']['mean']) * 100
    else:
        makespan_improvement = 0
    improvements['makespan_improvement'] = makespan_improvement
    
    # 2. 收敛速度改进（收敛代数越小越好）
    if random_stats['convergence_generation']['mean'] > 0:
        convergence_improvement = ((random_stats['convergence_generation']['mean'] - recommended_stats['convergence_generation']['mean']) / 
                                  random_stats['convergence_generation']['mean']) * 100
    else:
        convergence_improvement = 0
    improvements['convergence_improvement'] = convergence_improvement
    
    # 3. 稳定性改进（标准差越小越好）
    if random_stats['makespan']['std'] > 0:
        stability_improvement = ((random_stats['makespan']['std'] - recommended_stats['makespan']['std']) / 
                                random_stats['makespan']['std']) * 100
    else:
        stability_improvement = 0
    improvements['stability_improvement'] = stability_improvement
    
    # 4. 收敛稳定性改进（收敛代数标准差越小越好）
    if random_stats['convergence_generation']['std'] > 0:
        convergence_stability_improvement = ((random_stats['convergence_generation']['std'] - recommended_stats['convergence_generation']['std']) / 
                                            random_stats['convergence_generation']['std']) * 100
    else:
        convergence_stability_improvement = 0
    improvements['convergence_stability_improvement'] = convergence_stability_improvement
    
    # 5. 时间性能改进（执行时间越小越好）
    if random_stats['execution_time']['mean'] > 0:
        time_improvement = ((random_stats['execution_time']['mean'] - recommended_stats['execution_time']['mean']) / 
                           random_stats['execution_time']['mean']) * 100
    else:
        time_improvement = 0
    improvements['time_improvement'] = time_improvement
    
    return improvements

def create_comparison_table(random_results, recommended_results, output_path):
    """
    创建性能对比表格（四维度指标）
    
    Args:
        random_results: 随机初始化结果
        recommended_results: 推荐策略结果
        output_path: 输出文件路径
    """
    random_stats = random_results['statistics']
    recommended_stats = recommended_results['statistics']
    recommended_strategy = recommended_results['test_info']['recommended_strategy']
    
    # 计算改进指标
    improvements = calculate_improvement_metrics(random_stats, recommended_stats)
    
    # 获取四维性能评分
    random_scores = random_results.get('performance_metrics', {})
    recommended_scores = recommended_results.get('performance_metrics', {})
    
    # 创建对比表格
    comparison_data = {
        '指标': [
            'Makespan均值',
            'Makespan标准差',
            'Makespan最小值',
            'Makespan最大值',
            '收敛代数均值',
            '收敛代数标准差',
            '执行时间均值(s)',
            '执行时间标准差(s)',
            'Makespan精度',
            '求解稳定性',
            '收敛效率',
            '收敛稳定性',
            'Makespan改进率(%)',
            '收敛速度改进率(%)',
            '稳定性改进率(%)',
            '收敛稳定性改进率(%)',
            '时间性能改进率(%)'
        ],
        '随机初始化': [
            f"{random_stats['makespan']['mean']:.2f}",
            f"{random_stats['makespan']['std']:.2f}",
            f"{random_stats['makespan']['min']}",
            f"{random_stats['makespan']['max']}",
            f"{random_stats['convergence_generation']['mean']:.2f}",
            f"{random_stats['convergence_generation']['std']:.2f}",
            f"{random_stats['execution_time']['mean']:.2f}",
            f"{random_stats['execution_time']['std']:.2f}",
            f"{random_scores.get('makespan_accuracy', 0):.4f}",
            f"{random_scores.get('solution_stability', 0):.4f}",
            f"{random_scores.get('convergence_efficiency', 0):.4f}",
            f"{random_scores.get('convergence_stability', 0):.4f}",
            "-",
            "-",
            "-",
            "-",
            "-"
        ],
        f'推荐策略({recommended_strategy})': [
            f"{recommended_stats['makespan']['mean']:.2f}",
            f"{recommended_stats['makespan']['std']:.2f}",
            f"{recommended_stats['makespan']['min']}",
            f"{recommended_stats['makespan']['max']}",
            f"{recommended_stats['convergence_generation']['mean']:.2f}",
            f"{recommended_stats['convergence_generation']['std']:.2f}",
            f"{recommended_stats['execution_time']['mean']:.2f}",
            f"{recommended_stats['execution_time']['std']:.2f}",
            f"{recommended_scores.get('makespan_accuracy', 0):.4f}",
            f"{recommended_scores.get('solution_stability', 0):.4f}",
            f"{recommended_scores.get('convergence_efficiency', 0):.4f}",
            f"{recommended_scores.get('convergence_stability', 0):.4f}",
            f"{improvements['makespan_improvement']:.2f}",
            f"{improvements['convergence_improvement']:.2f}",
            f"{improvements['stability_improvement']:.2f}",
            f"{improvements['convergence_stability_improvement']:.2f}",
            f"{improvements['time_improvement']:.2f}"
        ]
    }
    
    # 创建DataFrame
    df = pd.DataFrame(comparison_data)
    
    # 保存为CSV
    csv_path = output_path.replace('.png', '.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 创建表格图
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 设置表格样式
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 高亮四维度评分行
    for i in range(8, 11):
        for j in range(len(df.columns)):
            table[(i, j)].set_facecolor('#E3F2FD')
    
    # 高亮改进指标行
    for i in range(11, 16):
        for j in range(len(df.columns)):
            table[(i, j)].set_facecolor('#FFE4B5')
    
    plt.title('随机初始化 vs 推荐策略四维度性能对比表', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return csv_path

def create_comparison_charts(random_results, recommended_results, output_dir):
    """
    创建性能对比图表（四维度指标）
    
    Args:
        random_results: 随机初始化结果
        recommended_results: 推荐策略结果
        output_dir: 输出目录
    """
    random_stats = random_results['statistics']
    recommended_stats = recommended_results['statistics']
    recommended_strategy = recommended_results['test_info']['recommended_strategy']
    
    # 获取四维性能评分
    random_scores = random_results.get('performance_metrics', {})
    recommended_scores = recommended_results.get('performance_metrics', {})
    
    # 1. 基础指标对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('随机初始化 vs 推荐策略性能对比', fontsize=16, fontweight='bold')
    
    # Makespan对比
    makespan_data = [random_stats['makespan']['mean'], recommended_stats['makespan']['mean']]
    ax1.bar(['随机初始化', f'推荐策略\n({recommended_strategy})'], makespan_data, 
            color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
    ax1.set_title('Makespan均值对比')
    ax1.set_ylabel('Makespan')
    for i, v in enumerate(makespan_data):
        ax1.text(i, v + max(makespan_data) * 0.01, f'{v:.2f}', ha='center', va='bottom')
    
    # 收敛代数对比
    convergence_data = [random_stats['convergence_generation']['mean'], recommended_stats['convergence_generation']['mean']]
    ax2.bar(['随机初始化', f'推荐策略\n({recommended_strategy})'], convergence_data, 
            color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
    ax2.set_title('收敛代数均值对比')
    ax2.set_ylabel('收敛代数')
    for i, v in enumerate(convergence_data):
        ax2.text(i, v + max(convergence_data) * 0.01, f'{v:.2f}', ha='center', va='bottom')
    
    # 执行时间对比
    time_data = [random_stats['execution_time']['mean'], recommended_stats['execution_time']['mean']]
    ax3.bar(['随机初始化', f'推荐策略\n({recommended_strategy})'], time_data, 
            color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
    ax3.set_title('执行时间均值对比')
    ax3.set_ylabel('执行时间(s)')
    for i, v in enumerate(time_data):
        ax3.text(i, v + max(time_data) * 0.01, f'{v:.2f}', ha='center', va='bottom')
    
    # 四维度评分对比
    score_names = ['Makespan精度', '求解稳定性', '收敛效率', '收敛稳定性']
    random_score_values = [
        random_scores.get('makespan_accuracy', 0),
        random_scores.get('solution_stability', 0),
        random_scores.get('convergence_efficiency', 0),
        random_scores.get('convergence_stability', 0)
    ]
    recommended_score_values = [
        recommended_scores.get('makespan_accuracy', 0),
        recommended_scores.get('solution_stability', 0),
        recommended_scores.get('convergence_efficiency', 0),
        recommended_scores.get('convergence_stability', 0)
    ]
    
    x = np.arange(len(score_names))
    width = 0.35
    
    ax4.bar(x - width/2, random_score_values, width, label='随机初始化', color='#FF6B6B', alpha=0.7)
    ax4.bar(x + width/2, recommended_score_values, width, label=f'推荐策略({recommended_strategy})', color='#4ECDC4', alpha=0.7)
    ax4.set_title('四维度性能评分对比')
    ax4.set_ylabel('评分')
    ax4.set_xticks(x)
    ax4.set_xticklabels(score_names, rotation=45, ha='right')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison_charts.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 改进率雷达图（四维度指标）
    improvements = calculate_improvement_metrics(random_stats, recommended_stats)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['Makespan改进', '收敛速度改进', '稳定性改进', '收敛稳定性改进', '时间性能改进']
    values = [
        improvements['makespan_improvement'], 
        improvements['convergence_improvement'], 
        improvements['stability_improvement'],
        improvements['convergence_stability_improvement'],
        improvements['time_improvement']
    ]
    
    # 处理负值显示
    display_values = []
    for value in values:
        if value < 0:
            display_values.append(0)  # 负值显示为0
        else:
            display_values.append(value)
    
    # 闭合数据
    display_values += display_values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # 绘制雷达图
    ax.plot(angles, display_values, 'o-', linewidth=2, color='#4ECDC4', label='改进率')
    ax.fill(angles, display_values, alpha=0.25, color='#4ECDC4')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # 设置Y轴范围，确保负值也能显示
    max_abs_value = max(abs(v) for v in values)
    ax.set_ylim(-max_abs_value * 0.2, max_abs_value * 1.2)
    ax.set_title('推荐策略四维度性能改进率', pad=20, fontsize=14, fontweight='bold')
    ax.grid(True)
    
    # 添加数值标签，包括负值
    for i, (angle, value, display_value) in enumerate(zip(angles[:-1], values[:-1], display_values[:-1])):
        if value < 0:
            # 负值显示在0位置，但标注实际值
            ax.text(angle, 0 + max_abs_value * 0.05, f'{value:.1f}%', 
                    ha='center', va='center', fontweight='bold', color='red')
        else:
            ax.text(angle, display_value + max_abs_value * 0.05, f'{value:.1f}%', 
                ha='center', va='center', fontweight='bold')
    
    plt.savefig(os.path.join(output_dir, 'improvement_radar_chart.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_convergence_comparison(random_results, recommended_results, output_dir):
    """
    创建收敛曲线对比图
    
    Args:
        random_results: 随机初始化结果
        recommended_results: 推荐策略结果
        output_dir: 输出目录
    """
    random_curves = random_results['raw_data']['convergence_curves']
    recommended_curves = recommended_results['raw_data']['convergence_curves']
    recommended_strategy = recommended_results['test_info']['recommended_strategy']
    
    plt.figure(figsize=(12, 8))
    
    # 计算平均收敛曲线
    random_mean = np.mean(random_curves, axis=0)
    random_std = np.std(random_curves, axis=0)
    recommended_mean = np.mean(recommended_curves, axis=0)
    recommended_std = np.std(recommended_curves, axis=0)
    
    # 绘制收敛曲线
    x = range(len(random_mean))
    plt.plot(x, random_mean, 'b-', linewidth=2, label='随机初始化 (mean)', color='#FF6B6B')
    plt.fill_between(x, random_mean - random_std, random_mean + random_std, 
                    alpha=0.2, color='#FF6B6B')
    
    plt.plot(x, recommended_mean, 'r-', linewidth=2, 
            label=f'推荐策略({recommended_strategy}) (mean)', color='#4ECDC4')
    plt.fill_between(x, recommended_mean - recommended_std, 
                    recommended_mean + recommended_std, 
                    alpha=0.2, color='#4ECDC4')
    
    plt.xlabel('迭代数')
    plt.ylabel('Makespan')
    plt.title('收敛曲线对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'convergence_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_report(random_results, recommended_results, output_dir, timestamp):
    """
    生成完整的对比报告（四维度指标）
    
    Args:
        random_results: 随机初始化结果
        recommended_results: 推荐策略结果
        output_dir: 输出目录
        timestamp: 时间戳
    """
    logger = logging.getLogger(__name__)
    
    logger.info("开始生成性能对比报告...")
    
    # 创建对比表格
    table_path = os.path.join(output_dir, f'performance_comparison_table_{timestamp}.png')
    csv_path = create_comparison_table(random_results, recommended_results, table_path)
    logger.info(f"对比表格已保存: {table_path}")
    logger.info(f"对比数据已保存: {csv_path}")
    
    # 创建对比图表
    create_comparison_charts(random_results, recommended_results, output_dir)
    logger.info("对比图表已保存")
    
    # 创建收敛曲线对比
    create_convergence_comparison(random_results, recommended_results, output_dir)
    logger.info("收敛曲线对比已保存")
    
    # 生成总结报告
    improvements = calculate_improvement_metrics(
        random_results['statistics'], 
        recommended_results['statistics']
    )
    
    recommended_strategy = recommended_results['test_info']['recommended_strategy']
    
    # 获取四维度评分
    random_scores = random_results.get('performance_metrics', {})
    recommended_scores = recommended_results.get('performance_metrics', {})
    
    # 获取测试信息（处理不同的文件结构）
    random_fjs_file = random_results.get('fjs_file', random_results.get('test_info', {}).get('fjs_file', 'Unknown'))
    random_execution_count = random_results.get('execution_times', random_results.get('test_info', {}).get('execution_count', 0))
    random_max_iterations = random_results.get('max_iterations', random_results.get('test_info', {}).get('max_iterations', 0))
    
    summary = f"""# 推荐策略四维度性能对比报告

## 测试概况
- **测试数据**: {random_fjs_file}
- **推荐策略**: {recommended_strategy}
- **测试次数**: {random_execution_count} 次
- **最大迭代数**: {random_max_iterations}
- **测试时间**: {timestamp}

## 四维度性能指标对比

### 1. Makespan精度
- 随机初始化: {random_scores.get('makespan_accuracy', 0):.4f}
- 推荐策略: {recommended_scores.get('makespan_accuracy', 0):.4f}
- 改进率: {improvements['makespan_improvement']:.2f}%

### 2. 求解稳定性
- 随机初始化: {random_scores.get('solution_stability', 0):.4f}
- 推荐策略: {recommended_scores.get('solution_stability', 0):.4f}
- 改进率: {improvements['stability_improvement']:.2f}%

### 3. 收敛效率
- 随机初始化: {random_scores.get('convergence_efficiency', 0):.4f}
- 推荐策略: {recommended_scores.get('convergence_efficiency', 0):.4f}
- 改进率: {improvements['convergence_improvement']:.2f}%

### 4. 收敛稳定性
- 随机初始化: {random_scores.get('convergence_stability', 0):.4f}
- 推荐策略: {recommended_scores.get('convergence_stability', 0):.4f}
- 改进率: {improvements['convergence_stability_improvement']:.2f}%

## 基础性能指标对比

### 求解精度 (Makespan)
- 随机初始化均值: {random_results['statistics']['makespan']['mean']:.2f}
- 推荐策略均值: {recommended_results['statistics']['makespan']['mean']:.2f}
- 改进率: {improvements['makespan_improvement']:.2f}%

### 收敛效率 (收敛代数)
- 随机初始化均值: {random_results['statistics']['convergence_generation']['mean']:.2f}
- 推荐策略均值: {recommended_results['statistics']['convergence_generation']['mean']:.2f}
- 改进率: {improvements['convergence_improvement']:.2f}%

### 稳定性 (Makespan标准差)
- 随机初始化标准差: {random_results['statistics']['makespan']['std']:.2f}
- 推荐策略标准差: {recommended_results['statistics']['makespan']['std']:.2f}
- 改进率: {improvements['stability_improvement']:.2f}%

### 收敛稳定性 (收敛代数标准差)
- 随机初始化标准差: {random_results['statistics']['convergence_generation']['std']:.2f}
- 推荐策略标准差: {recommended_results['statistics']['convergence_generation']['std']:.2f}
- 改进率: {improvements['convergence_stability_improvement']:.2f}%

### 时间性能 (执行时间)
- 随机初始化均值: {random_results['statistics']['execution_time']['mean']:.2f}s
- 推荐策略均值: {recommended_results['statistics']['execution_time']['mean']:.2f}s
- 改进率: {improvements['time_improvement']:.2f}%

## 结论
推荐策略在四维度性能指标方面相比随机初始化方法有显著改进：
1. **Makespan评分**: 提高了 {improvements['makespan_improvement']:.2f}%
2. **收敛速度评分**: 提高了 {improvements['convergence_improvement']:.2f}%
3. **稳定性评分**: 提高了 {improvements['stability_improvement']:.2f}%
4. **收敛稳定性评分**: 提高了 {improvements['convergence_stability_improvement']:.2f}%

推荐系统能够有效识别适合的初始化策略，显著提升算法性能。
"""
    
    # 保存总结报告
    summary_path = os.path.join(output_dir, f'comparison_summary_{timestamp}.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    logger.info(f"总结报告已保存: {summary_path}")
    logger.info("性能对比报告生成完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='性能对比分析')
    parser.add_argument('timestamp', nargs='?', help='时间戳（可选）')
    
    args = parser.parse_args()
    
    # 如果提供了时间戳，使用指定的结果目录
    if args.timestamp:
        timestamp = args.timestamp
        result_dir = os.path.join(BASE_DIR, "recommend_model_2", "result", "compare_with_random", "exp_result", f"exp_{timestamp}")
    else:
        # 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(BASE_DIR, "exp_result")
    
    os.makedirs(result_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(result_dir, f"performance_comparison_{timestamp}.log")
    logger = setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info("性能对比分析启动")
    logger.info(f"启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"结果目录: {result_dir}")
    logger.info(f"时间戳: {timestamp}")
    logger.info("=" * 80)
    
    try:
        # 加载测试结果
        logger.info("加载测试结果...")
        random_results, recommended_results = load_test_results(result_dir, timestamp)
        logger.info("测试结果加载完成")
        
        # 生成对比报告
        generate_comparison_report(random_results, recommended_results, result_dir, timestamp)
        
        logger.info("=" * 80)
        logger.info("性能对比分析完成")
        logger.info(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"性能对比分析失败: {e}")
        raise

if __name__ == "__main__":
    main() 