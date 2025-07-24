#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于帕累托前沿的推荐策略性能测试
使用帕累托优化方法进行多目标策略推荐和性能测试
五维性能目标：makespan精度、求解稳定性、收敛效率、收敛稳定性、执行时间
"""

import sys, os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.insert(0, BASE_DIR)

# 调试信息
print(f"当前目录: {current_dir}")
print(f"项目根目录: {BASE_DIR}")
print(f"sys.path[0]: {sys.path[0]}")

import json
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler

# Windows下设置终端编码为UTF-8，确保中文正常显示
if sys.platform == "win32":
    os.system("chcp 65001")
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10

# 导入GA相关模块
from initial_validation.utils import parser
from initial_validation.ga_fjsp import ga_new

# 导入推荐系统
sys.path.append(os.path.join(BASE_DIR, 'recommend_model_2'))
from initialization_strategy_recommender import InitializationStrategyRecommender



def setup_logging(log_file):
    """设置日志记录"""
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def find_convergence_point(curve):
    """使用Savitzky-Golay滤波和峰值检测找到收敛点"""
    from scipy.signal import savgol_filter
    
    window_length = min(21, len(curve))
    if window_length % 2 == 0:
        window_length += 1
    smoothed_curve = savgol_filter(curve, window_length, 3)
    
    diff = np.diff(smoothed_curve)
    threshold = np.std(diff) * 0.1
    convergence_idx = np.where(np.abs(diff) < threshold)[0][0]
    
    return convergence_idx

def plot_convergence_curves(convergence_data, save_path=None):
    """绘制收敛曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 6))
    
    for init_method, curves in convergence_data.items():
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        
        plt.plot(mean_curve, label=f'{init_method} (mean)')
        plt.fill_between(range(len(mean_curve)), 
                        mean_curve - std_curve, 
                        mean_curve + std_curve, 
                        alpha=0.2)
        
        convergence_idx = find_convergence_point(mean_curve)
        convergence_value = mean_curve[convergence_idx]
        plt.plot(convergence_idx, convergence_value, 'ro')
        plt.annotate(f'({convergence_idx}, {convergence_value:.2f})',
                    xy=(convergence_idx, convergence_value),
                    xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('迭代数')
    plt.ylabel('Makespan')
    plt.title('帕累托推荐策略的收敛曲线')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pareto_frontier(all_candidates, pareto_frontier, save_path=None):
    """绘制帕累托前沿分析 - 改进版"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    
    # 创建多个子图来展示不同角度的帕累托前沿
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 散点图：展示所有候选策略和帕累托前沿
    ax1 = plt.subplot(2, 3, 1)
    metrics = ['makespan_accuracy', 'solution_stability', 'convergence_efficiency', 'convergence_stability']
    
    # 提取所有候选策略的四维性能指标
    all_points = np.array([[s['performance_metrics'][metric] for metric in metrics] for s in all_candidates])
    pareto_points = np.array([[s['performance_metrics'][metric] for metric in metrics] for s in pareto_frontier])
    
    # 绘制所有候选策略（灰色点）
    ax1.scatter(all_points[:, 0], all_points[:, 1], c='lightgray', alpha=0.6, s=30, label='所有候选策略')
    
    # 绘制帕累托前沿策略（红色点）
    ax1.scatter(pareto_points[:, 0], pareto_points[:, 1], c='red', s=80, marker='o', label='帕累托前沿策略')
    
    # 标注帕累托前沿策略的名称
    for i, strategy in enumerate(pareto_frontier):
        ax1.annotate(strategy['strategy_name'], 
                    (pareto_points[i, 0], pareto_points[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Makespan精度')
    ax1.set_ylabel('求解稳定性')
    ax1.set_title('帕累托前沿分析 (Makespan精度 vs 求解稳定性)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 散点图：收敛效率 vs 收敛稳定性
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(all_points[:, 2], all_points[:, 3], c='lightgray', alpha=0.6, s=30, label='所有候选策略')
    ax2.scatter(pareto_points[:, 2], pareto_points[:, 3], c='red', s=80, marker='o', label='帕累托前沿策略')
    
    for i, strategy in enumerate(pareto_frontier):
        ax2.annotate(strategy['strategy_name'], 
                    (pareto_points[i, 2], pareto_points[i, 3]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('收敛效率')
    ax2.set_ylabel('收敛稳定性')
    ax2.set_title('帕累托前沿分析 (收敛效率 vs 收敛稳定性)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 3D散点图：三维性能指标
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    ax3.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], 
               c='lightgray', alpha=0.6, s=30, label='所有候选策略')
    ax3.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], 
               c='red', s=80, marker='o', label='帕累托前沿策略')
    
    ax3.set_xlabel('Makespan精度')
    ax3.set_ylabel('求解稳定性')
    ax3.set_zlabel('收敛效率')
    ax3.set_title('帕累托前沿3D分析')
    ax3.legend()
    
    # 4. 雷达图：帕累托前沿策略的详细性能对比
    ax4 = plt.subplot(2, 3, 4, projection='polar')
    
    # 准备雷达图数据
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 为每个帕累托前沿策略绘制雷达图
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, strategy in enumerate(pareto_frontier):
        values = [strategy['performance_metrics'][metric] for metric in metrics]
        values += values[:1]  # 闭合图形
        
        ax4.plot(angles, values, 'o-', linewidth=2, label=strategy['strategy_name'], color=colors[i % len(colors)])
        ax4.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(['Makespan精度', '求解稳定性', '收敛效率', '收敛稳定性'])
    ax4.set_ylim(0, 1)
    ax4.set_title('帕累托前沿策略性能雷达图')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 5. 加权评分对比
    ax5 = plt.subplot(2, 3, 5)
    pareto_strategies = [s['strategy_name'] for s in pareto_frontier]
    pareto_scores = [s['weighted_score'] for s in pareto_frontier]
    
    bars = ax5.bar(pareto_strategies, pareto_scores, color='red', alpha=0.7)
    ax5.set_xlabel('策略名称')
    ax5.set_ylabel('加权评分')
    ax5.set_title('帕累托前沿策略加权评分')
    ax5.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, score in zip(bars, pareto_scores):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.4f}', ha='center', va='bottom')
    
    # 6. 支配关系可视化
    ax6 = plt.subplot(2, 3, 6)
    
    # 计算每个策略被支配的次数
    domination_counts = {}
    for strategy in all_candidates:
        strategy_name = strategy['strategy_name']
        if strategy_name not in domination_counts:
            domination_counts[strategy_name] = 0
    
    # 统计支配关系
    for i, strategy1 in enumerate(all_candidates):
        for j, strategy2 in enumerate(all_candidates):
            if i != j:
                point1 = [strategy1['performance_metrics'][metric] for metric in metrics]
                point2 = [strategy2['performance_metrics'][metric] for metric in metrics]
                
                # 检查是否被支配
                if all(p2 >= p1 for p1, p2 in zip(point1, point2)) and any(p2 > p1 for p1, p2 in zip(point1, point2)):
                    domination_counts[strategy1['strategy_name']] += 1
    
    # 绘制支配关系图
    pareto_names = [s['strategy_name'] for s in pareto_frontier]
    pareto_dominations = [domination_counts[name] for name in pareto_names]
    
    bars = ax6.bar(pareto_names, pareto_dominations, color='green', alpha=0.7)
    ax6.set_xlabel('策略名称')
    ax6.set_ylabel('被支配次数')
    ax6.set_title('帕累托前沿策略支配关系')
    ax6.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, count in zip(bars, pareto_dominations):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"帕累托前沿分析图已保存到: {save_path}")
    
    plt.close()

def plot_pareto_frontier_simple(all_candidates, pareto_frontier, save_path=None):
    """绘制简化的帕累托前沿分析图"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('帕累托前沿分析（四维性能指标分布）', fontsize=16)
    
    metrics = ['makespan_accuracy', 'solution_stability', 'convergence_efficiency', 'convergence_stability']
    metric_names = ['Makespan精度', '求解稳定性', '收敛效率', '收敛稳定性']
    
    # 绘制每个维度的分布
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        row = i // 2
        col = i % 2
        
        pareto_values = [s['performance_metrics'][metric] for s in pareto_frontier]
        all_values = [s['performance_metrics'][metric] for s in all_candidates]
        
        axes[row, col].hist(all_values, bins=20, alpha=0.5, label='所有候选', color='lightblue')
        axes[row, col].hist(pareto_values, bins=20, alpha=0.7, label='帕累托前沿', color='red')
        axes[row, col].set_xlabel(name)
        axes[row, col].set_ylabel('频次')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # 修改保存路径，添加simple后缀
        base_path = save_path.replace('.png', '_simple.png')
        plt.savefig(base_path, dpi=300, bbox_inches='tight')
        print(f"简化帕累托前沿分析图已保存到: {base_path}")
    
    plt.close()

def save_results(results_data, output_path):
    """保存结果到JSON文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    converted_data = convert_numpy_types(results_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {output_path}")

def get_pareto_recommended_strategy(fjs_file_path, logger):
    """
    获取基于帕累托前沿的推荐策略
    
    Args:
        fjs_file_path: FJS文件路径
        logger: 日志记录器
        
    Returns:
        tuple: (推荐策略名称, 推荐结果)
    """
    logger.info("开始获取帕累托推荐策略...")
    
    try:
        # 初始化推荐系统
        labeled_dataset_path = os.path.join(BASE_DIR, "recommend_model_1", "labeled_dataset", "labeled_fjs_dataset.json")
        
        # 初始化帕累托推荐系统
        pareto_recommender = InitializationStrategyRecommender(labeled_dataset_path)
        
        # 提取新数据特征
        sys.path.append(os.path.join(BASE_DIR, 'recommend_model_1'))
        from extract_new_data_features import extract_new_data_features
        
        logger.info(f"提取新数据特征: {fjs_file_path}")
        new_data_features = extract_new_data_features(fjs_file_path)
        
        if new_data_features is None:
            logger.error("新数据特征提取失败")
            return None, None
        
        logger.info("新数据特征提取完成")
        
        # 执行帕累托推荐
        logger.info("执行帕累托前沿推荐...")
        pareto_results = pareto_recommender.recommend(new_data_features, top_k_similar=5, top_k_strategies=3)
        
        # 输出推荐结果
        logger.info("帕累托推荐结果:")
        logger.info(f"  候选样本数量: {len(pareto_results['stage_one_results']['candidate_samples'])}")
        
        for i, strategy_data in enumerate(pareto_results['stage_two_results']['recommended_strategies'], 1):
            strategy_name = strategy_data['strategy_name']
            weighted_score = strategy_data['weighted_score']
            logger.info(f"  {i}. {strategy_name}")
            logger.info(f"     加权评分: {weighted_score:.4f}")
        
        # 返回最佳推荐策略
        best_strategy = pareto_results['stage_two_results']['recommended_strategies'][0]['strategy_name']
        logger.info(f"选择帕累托最佳推荐策略: {best_strategy}")
        
        return best_strategy, pareto_results
        
    except Exception as e:
        logger.error(f"获取帕累托推荐策略失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """主函数"""
    # 获取命令行参数
    if len(sys.argv) != 3:
        print("使用方法: python pareto_recommended_strategy_test.py <fjs_file_path> <timestamp>")
        print("示例: python pareto_recommended_strategy_test.py new_Behnke3.fjs 20250717_062546")
        return
    
    fjs_file_path = sys.argv[1]
    timestamp = sys.argv[2]
    
    # 检查FJS文件是否存在
    if not os.path.exists(fjs_file_path):
        print(f"错误: FJS文件不存在: {fjs_file_path}")
        return
    
    # 创建输出目录（使用传递的timestamp）
    exp_result_dir = os.path.join("exp_result", f"exp_{timestamp}")
    os.makedirs(exp_result_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(exp_result_dir, f"pareto_recommended_strategy_test_{timestamp}.log")
    logger = setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info("基于帕累托前沿的推荐策略性能测试")
    logger.info(f"启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"FJS文件: {os.path.abspath(fjs_file_path)}")
    logger.info(f"输出目录: {os.path.abspath(exp_result_dir)}")
    logger.info("=" * 80)
    
    # 记录测试参数
    logger.info("测试参数配置:")
    logger.info(f"  FJS文件: {os.path.abspath(fjs_file_path)}")
    logger.info(f"  执行次数: 20")
    logger.info(f"  最大迭代数: 100")
    logger.info(f"  性能指标: 四维帕累托优化（Makespan精度、求解稳定性、收敛效率、收敛稳定性）")
    logger.info("-" * 80)
    
    # 获取帕累托推荐策略
    recommended_strategy, pareto_results = get_pareto_recommended_strategy(fjs_file_path, logger)
    
    if recommended_strategy is None:
        logger.error("获取推荐策略失败，测试终止")
        return
    
    # 解析FJS文件
    logger.info("解析FJS文件...")
    try:
        fjs_data = parser.parse(fjs_file_path)
        logger.info(f"解析成功 - 机器数: {fjs_data['machinesNb']}, 作业数: {len(fjs_data['jobs'])}")
    except Exception as e:
        logger.error(f"解析FJS文件失败: {e}")
        return
    
    # 执行推荐策略测试
    logger.info(f"执行20次推荐策略测试 (策略: {recommended_strategy})...")
    
    results = []
    convergence_data = {recommended_strategy: []}
    
    for i in range(20):
        logger.info(f"  第 {i+1:2d} 次测试...")
        
        start_time = time.time()
        
        # 执行GA算法
        best_makespan, convergence_curve = ga_new(fjs_data, recommended_strategy, return_convergence=True)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 计算收敛点
        convergence_idx = find_convergence_point(convergence_curve)
        
        # 记录结果
        result = {
            'test_id': i + 1,
            'makespan': best_makespan,
            'convergence_generation': convergence_idx,
            'execution_time': execution_time
        }
        results.append(result)
        
        # 记录收敛曲线
        convergence_data[recommended_strategy].append(convergence_curve)
        
        logger.info(f"    第 {i+1:2d}次测试 - Makespan: {best_makespan:3d}, 收敛代数: {convergence_idx:2d}, 耗时: {execution_time:5.2f}s")
    
    # 计算统计信息
    makespans = [r['makespan'] for r in results]
    convergence_generations = [r['convergence_generation'] for r in results]
    execution_times = [r['execution_time'] for r in results]
    
    logger.info("推荐策略测试完成")
    logger.info(f"  Makespan - 均值: {np.mean(makespans):.2f}, 标准差: {np.std(makespans):.2f}, 最小值: {min(makespans)}, 最大值: {max(makespans)}")
    logger.info(f"  执行时间 - 均值: {np.mean(execution_times):.2f}s, 标准差: {np.std(execution_times):.2f}")
    logger.info(f"  收敛代数 - 均值: {np.mean(convergence_generations):.2f}, 标准差: {np.std(convergence_generations):.2f}")
    
    # 计算四维性能评分
    mean_makespan = np.mean(makespans)
    std_makespan = np.std(makespans)
    mean_convergence_gen = np.mean(convergence_generations)
    std_convergence_gen = np.std(convergence_generations)
    
    performance_metrics = {
        'makespan_accuracy': 1.0 / (1.0 + mean_makespan / 1000.0),
        'solution_stability': 1.0 / (1.0 + std_makespan / 10.0),
        'convergence_efficiency': 1.0 - (mean_convergence_gen / 100.0),
        'convergence_stability': 1.0 / (1.0 + std_convergence_gen / 10.0)
    }
    
    logger.info("四维性能评分:")
    for metric, score in performance_metrics.items():
        logger.info(f"  {metric}: {score:.4f}")
    
    # 保存结果
    output_file = os.path.join(exp_result_dir, f"pareto_recommended_strategy_results_{timestamp}.json")
    results_data = {
        'test_info': {
            'fjs_file': fjs_file_path,
            'recommended_strategy': recommended_strategy,
            'execution_count': 20,
            'max_iterations': 100,
            'timestamp': timestamp
        },
        'statistics': {
            'makespan': {
                'mean': float(np.mean(makespans)),
                'std': float(np.std(makespans)),
                'min': int(min(makespans)),
                'max': int(max(makespans))
            },
            'execution_time': {
                'mean': float(np.mean(execution_times)),
                'std': float(np.std(execution_times))
            },
            'convergence_generation': {
                'mean': float(np.mean(convergence_generations)),
                'std': float(np.std(convergence_generations))
            }
        },
        'performance_metrics': performance_metrics,
        'detailed_results': results,
        'pareto_analysis': {
            'stage_one_results': pareto_results['stage_one_results'],
            'stage_two_results': pareto_results['stage_two_results']
        },
        'raw_data': {
            'convergence_curves': convergence_data[recommended_strategy]
        }
    }
    
    save_results(results_data, output_file)
    
    # 绘制收敛曲线
    if convergence_data[recommended_strategy]:
        convergence_plot_path = os.path.join(exp_result_dir, f"pareto_recommended_convergence_curves_{timestamp}.png")
        plot_convergence_curves(convergence_data, convergence_plot_path)
        logger.info(f"收敛曲线图已保存: {convergence_plot_path}")
    
    # 绘制帕累托前沿分析图（改进版）
    pareto_plot_path = os.path.join(exp_result_dir, f"pareto_frontier_analysis_{timestamp}.png")
    plot_pareto_frontier(pareto_results['all_candidates'], pareto_results['pareto_frontier'], pareto_plot_path)
    logger.info(f"改进版帕累托前沿分析图已保存: {pareto_plot_path}")
    
    # 绘制帕累托前沿分析图（简化版）
    plot_pareto_frontier_simple(pareto_results['all_candidates'], pareto_results['pareto_frontier'], pareto_plot_path)
    logger.info(f"简化版帕累托前沿分析图已保存: {pareto_plot_path.replace('.png', '_simple.png')}")
    
    # 记录完成信息
    logger.info("=" * 80)
    logger.info("基于帕累托前沿的推荐策略测试完成")
    logger.info(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"结果文件: {output_file}")
    logger.info(f"收敛曲线图: {convergence_plot_path}")
    logger.info(f"帕累托前沿分析图: {pareto_plot_path}")
    logger.info("四维性能评分:")
    for metric, score in performance_metrics.items():
        logger.info(f"  {metric}: {score:.4f}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main() 