#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推荐策略性能测试
使用推荐系统推荐的初始化方法进行性能测试
"""

import sys, os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(current_dir, '../..'))
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

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.rcParams['font.size'] = 10  # 设置默认字体大小

# 导入GA相关模块
from initial_validation.utils import parser
from initial_validation.ga_fjsp import ga_new

# 导入推荐系统
recommender_dir = os.path.join(BASE_DIR, 'recommend_model_1.1')
sys.path.append(recommender_dir)
from initialization_strategy_recommender import InitializationStrategyRecommender

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

def find_convergence_point(curve):
    """
    使用Savitzky-Golay滤波和峰值检测找到收敛点
    :param curve: 收敛曲线数据
    :return: 收敛点的索引
    """
    from scipy.signal import savgol_filter
    
    # 使用Savitzky-Golay滤波平滑曲线
    window_length = min(21, len(curve))
    if window_length % 2 == 0:
        window_length += 1
    smoothed_curve = savgol_filter(curve, window_length, 3)
    
    # 计算一阶差分
    diff = np.diff(smoothed_curve)
    
    # 找到差分值开始稳定的点（即收敛点）
    threshold = np.std(diff) * 0.1
    convergence_idx = np.where(np.abs(diff) < threshold)[0][0]
    
    return convergence_idx

def plot_convergence_curves(convergence_data, save_path=None):
    """绘制收敛曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 6))
    
    for init_method, curves in convergence_data.items():
        # 计算平均收敛曲线
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        
        # 绘制平均曲线
        plt.plot(mean_curve, label=f'{init_method} (mean)')
        
        # 绘制标准差范围
        plt.fill_between(range(len(mean_curve)), 
                        mean_curve - std_curve, 
                        mean_curve + std_curve, 
                        alpha=0.2)
        
        # 找到并标记收敛点
        convergence_idx = find_convergence_point(mean_curve)
        convergence_value = mean_curve[convergence_idx]
        plt.plot(convergence_idx, convergence_value, 'ro')
        plt.annotate(f'({convergence_idx}, {convergence_value:.2f})',
                    xy=(convergence_idx, convergence_value),
                    xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('迭代数')
    plt.ylabel('Makespan')
    plt.title('推荐策略的收敛曲线')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_results(results_data, output_path):
    """保存结果到JSON文件"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换numpy类型为Python原生类型
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
    
    # 转换数据类型
    converted_data = convert_numpy_types(results_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {output_path}")

def get_recommended_strategy(fjs_file_path, logger):
    """
    获取推荐系统推荐的初始化策略
    
    Args:
        fjs_file_path: FJS文件路径
        logger: 日志记录器
        
    Returns:
        str: 推荐的初始化策略名称
    """
    logger.info("开始获取推荐策略...")
    
    try:
        # 初始化推荐系统
        labeled_dataset_path = os.path.join(BASE_DIR, "recommend_model_1.1", "labeled_dataset", "converted_fjs_dataset_new.json")
        recommender = InitializationStrategyRecommender(labeled_dataset_path)
        
        # 提取新数据特征
        sys.path.append(os.path.join(BASE_DIR, 'recommend_model_1.1'))
        from extract_new_data_features import extract_new_data_features
        
        logger.info(f"提取新数据特征: {fjs_file_path}")
        new_data_features = extract_new_data_features(fjs_file_path)
        
        if new_data_features is None:
            logger.error("新数据特征提取失败")
            return None
        
        logger.info("新数据特征提取完成")
        
        # 执行推荐
        logger.info("执行推荐流程...")
        results = recommender.recommend(new_data_features, top_k_strategies=3, feature_weight=0.4, performance_weight=0.6)
        
        # 获取推荐策略
        recommended_strategies = results['recommended_strategies']
        
        logger.info("推荐结果:")
        for i, strategy in enumerate(recommended_strategies, 1):
            logger.info(f"  {i}. {strategy['strategy_name']} - 评分: {strategy['final_score']:.4f}")
        
        # 返回最佳推荐策略
        best_strategy = recommended_strategies[0]['strategy_name']
        logger.info(f"选择最佳推荐策略: {best_strategy}")
        
        return best_strategy, results
        
    except Exception as e:
        logger.error(f"获取推荐策略失败: {e}")
        return None, None

def test_recommended_strategy(fjs_file_path, runs=20, max_iterations=100, logger=None):
    """
    测试推荐策略的性能
    
    Args:
        fjs_file_path: FJS文件路径
        runs: 执行次数
        max_iterations: 最大迭代数
        logger: 日志记录器
        
    Returns:
        dict: 测试结果
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"开始测试推荐策略: {fjs_file_path}")
    
    # 获取推荐策略
    result = get_recommended_strategy(fjs_file_path, logger)
    
    if result is None or len(result) != 2:
        logger.error("无法获取推荐策略，测试终止")
        return None
    
    recommended_strategy, recommendation_results = result
    
    try:
        # 解析fjs文件
        logger.info(f"解析fjs文件...")
        parameters = parser.parse(fjs_file_path)
        logger.info(f"解析成功 - 机器数: {parameters['machinesNb']}, 作业数: {len(parameters['jobs'])}")
        
    except Exception as e:
        logger.error(f"解析失败: {e}")
        return None
    
    # 初始化结果存储
    results = []
    convergence_data = []
    execution_times = []
    convergence_times = []
    
    logger.info(f"执行 {runs} 次推荐策略测试 (策略: {recommended_strategy})...")
    
    for i in range(runs):
        logger.info(f"  第 {i+1:2d} 次运行...")
        run_start_time = time.time()
        
        # 执行GA算法
        best_makespan, convergence_curve = ga_new(parameters, recommended_strategy, return_convergence=True)
        
        run_end_time = time.time()
        run_duration = run_end_time - run_start_time
        
        # 计算收敛点
        convergence_idx = find_convergence_point(convergence_curve)
        
        # 记录结果
        results.append(best_makespan)
        convergence_data.append(convergence_curve)
        execution_times.append(run_duration)
        convergence_times.append(convergence_idx)
        
        logger.info(f"    第{i+1:2d}次运行 - Makespan: {best_makespan:4d}, 收敛代数: {convergence_idx:2d}, 耗时: {run_duration:5.2f}s")
    
    # 计算统计结果
    mean_makespan = np.mean(results)
    std_makespan = np.std(results)
    min_makespan = np.min(results)
    max_makespan = np.max(results)
    
    mean_execution_time = np.mean(execution_times)
    std_execution_time = np.std(execution_times)
    
    mean_convergence_gen = np.mean(convergence_times)
    std_convergence_gen = np.std(convergence_times)
    
    # 构建结果字典
    test_results = {
        "test_type": "recommended_strategy",
        "fjs_file": fjs_file_path,
        "meta_heuristic": "GA",
        "execution_times": runs,
        "max_iterations": max_iterations,
        "timestamp": datetime.datetime.now().isoformat(),
        "recommendation_info": {
            "recommended_strategy": recommended_strategy,
            "recommendation_results": recommendation_results
        },
        "statistics": {
            "makespan": {
                "mean": mean_makespan,
                "std": std_makespan,
                "min": min_makespan,
                "max": max_makespan
            },
            "execution_time": {
                "mean": mean_execution_time,
                "std": std_execution_time,
                "min": np.min(execution_times),
                "max": np.max(execution_times)
            },
            "convergence_generation": {
                "mean": mean_convergence_gen,
                "std": std_convergence_gen,
                "min": np.min(convergence_times),
                "max": np.max(convergence_times)
            }
        },
        "raw_data": {
            "makespans": results,
            "execution_times": execution_times,
            "convergence_times": convergence_times,
            "convergence_curves": convergence_data
        }
    }
    
    logger.info(f"推荐策略测试完成")
    logger.info(f"  Makespan - 均值: {mean_makespan:.2f}, 标准差: {std_makespan:.2f}, 最小值: {min_makespan}, 最大值: {max_makespan}")
    logger.info(f"  执行时间 - 均值: {mean_execution_time:.2f}s, 标准差: {std_execution_time:.2f}s")
    logger.info(f"  收敛代数 - 均值: {mean_convergence_gen:.2f}, 标准差: {std_convergence_gen:.2f}")
    
    return test_results

def main():
    import sys
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='推荐策略方法性能测试')
    parser.add_argument('fjs_file', help='FJS文件路径')
    parser.add_argument('timestamp', nargs='?', help='时间戳（可选）')
    
    args = parser.parse_args()
    fjs_file_path = args.fjs_file
    
    # 如果提供了时间戳，使用指定的结果目录
    if args.timestamp:
        timestamp = args.timestamp
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result_dir = os.path.join(script_dir, "exp_result", f"exp_{timestamp}")
    else:
        # 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result_dir = os.path.join(script_dir, "exp_result")
    
    os.makedirs(result_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(result_dir, f"recommended_strategy_test_{timestamp}.log")
    logger = setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info("推荐策略方法性能测试启动")
    logger.info(f"启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"FJS文件: {fjs_file_path}")
    logger.info(f"结果目录: {result_dir}")
    logger.info("=" * 80)
    
    # 测试参数
    runs = 20
    max_iterations = 100
    
    logger.info("测试参数配置:")
    logger.info(f"  FJS文件: {fjs_file_path}")
    logger.info(f"  执行次数: {runs}")
    logger.info(f"  最大迭代数: {max_iterations}")
    logger.info("-" * 80)
    
    # 执行测试
    start_time = time.time()
    results = test_recommended_strategy(fjs_file_path, runs, max_iterations, logger)
    end_time = time.time()
    
    if results is None:
        logger.error("测试失败")
        return
    
    # 计算四维度性能评分
    mean_makespan = results["statistics"]["makespan"]["mean"]
    std_makespan = results["statistics"]["makespan"]["std"]
    mean_convergence_gen = results["statistics"]["convergence_generation"]["mean"]
    std_convergence_gen = results["statistics"]["convergence_generation"]["std"]
    
    # 四维度性能评分计算（与推荐系统保持一致）
    # 1. Makespan评分（越小越好）
    makespan_score = 1.0 / (1.0 + mean_makespan / 1000.0)
    
    # 2. 收敛速度评分（收敛代数越小越好）
    convergence_speed_score = 1.0 - (mean_convergence_gen / max_iterations)
    convergence_speed_score = max(0.0, min(1.0, convergence_speed_score))
    
    # 3. 稳定性评分（标准差越小越好）
    stability_score = 1.0 / (1.0 + std_makespan / 10.0)
    
    # 4. 收敛稳定性评分（收敛代数标准差越小越好）
    convergence_stability_score = 1.0 / (1.0 + std_convergence_gen / 10.0)
    
    # 添加四维度评分到结果中
    results["four_dimension_scores"] = {
        "makespan_score": makespan_score,
        "convergence_speed_score": convergence_speed_score,
        "stability_score": stability_score,
        "convergence_stability_score": convergence_stability_score
    }
    
    # 保存结果
    output_file = os.path.join(result_dir, f"recommended_strategy_results_{timestamp}.json")
    save_results(results, output_file)
    
    # 绘制收敛曲线
    convergence_plot_path = os.path.join(result_dir, f"recommended_convergence_curves_{timestamp}.png")
    plot_convergence_curves({"recommended": results["raw_data"]["convergence_curves"]}, convergence_plot_path)
    
    # 记录最终统计信息
    total_duration = end_time - start_time
    logger.info("=" * 80)
    logger.info("推荐策略测试完成")
    logger.info(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总耗时: {total_duration:.2f}秒")
    logger.info(f"结果文件: {output_file}")
    logger.info(f"收敛曲线图: {convergence_plot_path}")
    
    # 输出四维度评分
    logger.info("四维度性能评分:")
    logger.info(f"  Makespan评分: {makespan_score:.4f}")
    logger.info(f"  收敛速度评分: {convergence_speed_score:.4f}")
    logger.info(f"  稳定性评分: {stability_score:.4f}")
    logger.info(f"  收敛稳定性评分: {convergence_stability_score:.4f}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main() 