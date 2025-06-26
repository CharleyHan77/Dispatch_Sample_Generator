import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import os
import sys
import json
import logging
import datetime
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import glob

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.utils import parser, gantt
from initial_validation.ga_fjsp import ga_new

def setup_logging(log_dir):
    """
    设置日志记录
    :param log_dir: 日志目录
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"initialization_validation_{timestamp}.log")
    
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

def plot_convergence_curves(convergence_data: dict, save_path: str = None):
    """
    绘制收敛曲线
    :param convergence_data: 收敛数据字典
    :param save_path: 保存路径
    """
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
        # 添加坐标值标注
        plt.annotate(f'({convergence_idx}, {convergence_value:.2f})',
                    xy=(convergence_idx, convergence_value),
                    xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('迭代数')
    plt.ylabel('Makespan')
    plt.title('三种初始化方式的平均收敛曲线')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_validation_results(results_data: dict, output_path: str):
    """
    保存验证结果到JSON文件
    :param results_data: 结果数据字典
    :param output_path: 输出文件路径
    """
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
    
    print(f"验证结果已保存到: {output_path}")

def get_all_fjs_files(dataset_root):
    # 递归查找所有.fjs文件
    return glob.glob(os.path.join(dataset_root, '**', '*.fjs'), recursive=True)

if __name__ == '__main__':
    # 设置日志记录
    log_dir = os.path.join(project_root, "logs")
    logger = setup_logging(log_dir)
    
    # 记录程序启动信息
    start_time = datetime.datetime.now()
    logger.info("=" * 80)
    logger.info("初始化验证程序启动")
    logger.info(f"启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # 配置参数
    dataset_root = os.path.join(project_root, "dataset")
    output_root = os.path.join(project_root, "output", "init_validity_result")
    plot_root = os.path.join(project_root, "output", "convergence_curves")
    runs = 20
    max_iterations = 100
    meta_heuristic = "HA(GA+TS)"
    
    # 记录执行参数
    logger.info("执行参数配置:")
    logger.info(f"  数据集根目录: {dataset_root}")
    logger.info(f"  输出根目录: {output_root}")
    logger.info(f"  图表根目录: {plot_root}")
    logger.info(f"  执行次数: {runs}")
    logger.info(f"  最大迭代数: {max_iterations}")
    logger.info(f"  元启发算法: {meta_heuristic}")
    logger.info("-" * 80)
    
    # 获取所有fjs文件
    all_fjs_files = get_all_fjs_files(dataset_root)
    logger.info(f"共检测到 {len(all_fjs_files)} 个fjs实例文件")
    
    # 记录文件列表
    logger.info("检测到的文件列表:")
    for i, fjs_path in enumerate(all_fjs_files, 1):
        rel_path = os.path.relpath(fjs_path, dataset_root)
        logger.info(f"  {i:3d}. {rel_path}")
    logger.info("-" * 80)
    
    # 统计信息
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    
    for file_idx, fjs_path in enumerate(all_fjs_files, 1):
        # 获取相对路径，用于提取数据集名和实例名
        rel_path = os.path.relpath(fjs_path, dataset_root)
        rel_path = rel_path.replace('\\', '/')  # 统一路径分隔符
        
        # 解析数据集名和实例名
        path_parts = rel_path.split('/')
        if len(path_parts) == 1:
            # 直接在dataset根目录下的文件
            dataset_name = "dataset"
            instance_name = os.path.splitext(path_parts[0])[0]
        else:
            # 在子目录中的文件
            dataset_name = path_parts[0]  # 第一级目录名作为数据集名
            instance_name = os.path.splitext(path_parts[-1])[0]  # 最后一级文件名（不含扩展名）
        
        # 组织输出目录和文件名
        out_dir = os.path.join(output_root, dataset_name)
        out_json = os.path.join(out_dir, f"{instance_name}_validation_results.json")
        plot_dir = os.path.join(plot_root, dataset_name)
        plot_path = os.path.join(plot_dir, f"{instance_name}_convergence_curves.png")
        
        # 记录当前处理的文件信息
        logger.info(f"\n[{file_idx}/{len(all_fjs_files)}] 处理文件: {rel_path}")
        logger.info(f"  数据集: {dataset_name}")
        logger.info(f"  实例名: {instance_name}")
        logger.info(f"  输出JSON: {out_json}")
        logger.info(f"  输出图表: {plot_path}")
        
        # 跳过已存在的结果
        if os.path.exists(out_json):
            logger.info(f"  状态: 已存在，跳过")
            total_skipped += 1
            continue
            
        file_start_time = datetime.datetime.now()
        logger.info(f"  开始时间: {file_start_time.strftime('%H:%M:%S')}")
        
        try:
            # 解析fjs文件
            logger.info(f"  解析fjs文件...")
            parameters = parser.parse(fjs_path)
            logger.info(f"  解析成功 - 机器数: {parameters['machinesNb']}, 作业数: {len(parameters['jobs'])}")
            
        except Exception as e:
            logger.error(f"  解析失败: {e}")
            total_failed += 1
            continue
            
        # 初始化结果存储
        results = {"heuristic": [], "mixed": [], "random": []}
        convergence_data = {"heuristic": [], "mixed": [], "random": []}
        convergence_times = {"heuristic": [], "mixed": [], "random": []}
        
        # 执行三种初始化方法
        for init_method in results.keys():
            logger.info(f"  执行 {init_method} 初始化方法...")
            method_start_time = datetime.datetime.now()
            
            for i in range(runs):
                run_start_time = datetime.datetime.now()
                best_makespan, convergence_curve = ga_new(parameters, init_method, return_convergence=True)
                run_end_time = datetime.datetime.now()
                run_duration = (run_end_time - run_start_time).total_seconds()
                
                results[init_method].append(best_makespan)
                convergence_data[init_method].append(convergence_curve)
                convergence_idx = find_convergence_point(convergence_curve)
                convergence_times[init_method].append(convergence_idx)
                
                logger.info(f"    第{i+1:2d}次运行 - Makespan: {best_makespan:4d}, 收敛代数: {convergence_idx:2d}, 耗时: {run_duration:5.2f}s")
            
            method_end_time = datetime.datetime.now()
            method_duration = (method_end_time - method_start_time).total_seconds()
            avg_makespan = np.mean(results[init_method])
            logger.info(f"  {init_method} 完成 - 平均Makespan: {avg_makespan:.2f}, 总耗时: {method_duration:.2f}s")
        
        # 构建验证结果
        validation_results = {
            "dataset": dataset_name,
            "instance": os.path.basename(fjs_path),
            "meta_heuristic": meta_heuristic,
            "execution_times": runs,
            "max_iterations": max_iterations,
            "initialization_methods": {}
        }
        
        # 计算统计结果
        for init_method, metrics in results.items():
            mean_makespan = np.mean(metrics)
            std_makespan = np.std(metrics)
            min_makespan = np.min(metrics)
            max_makespan = np.max(metrics)
            avg_convergence_gen = np.mean(convergence_times[init_method])
            convergence_gen_std = np.std(convergence_times[init_method])
            
            validation_results["initialization_methods"][init_method] = {
                "mean": mean_makespan,
                "std": std_makespan,
                "min": min_makespan,
                "max": max_makespan,
                "avg_convergence_generation": avg_convergence_gen,
                "convergence_generation_std": convergence_gen_std
            }
            
            logger.info(f"  {init_method} 统计结果:")
            logger.info(f"    Makespan - 均值: {mean_makespan:.2f}, 标准差: {std_makespan:.2f}, 最小值: {min_makespan}, 最大值: {max_makespan}")
            logger.info(f"    收敛代数 - 均值: {avg_convergence_gen:.2f}, 标准差: {convergence_gen_std:.2f}")
        
        # 保存结果
        save_validation_results(validation_results, out_json)
        logger.info(f"  结果已保存到: {out_json}")
        
        # 保存收敛曲线图
        os.makedirs(plot_dir, exist_ok=True)
        plot_convergence_curves(convergence_data, save_path=plot_path)
        logger.info(f"  收敛曲线已保存到: {plot_path}")
        
        file_end_time = datetime.datetime.now()
        file_duration = (file_end_time - file_start_time).total_seconds()
        logger.info(f"  完成时间: {file_end_time.strftime('%H:%M:%S')}, 总耗时: {file_duration:.2f}s")
        
        total_processed += 1
    
    # 记录最终统计信息
    end_time = datetime.datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    logger.info("=" * 80)
    logger.info("程序执行完成")
    logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总耗时: {total_duration:.2f}秒 ({total_duration/3600:.2f}小时)")
    logger.info("执行统计:")
    logger.info(f"  总文件数: {len(all_fjs_files)}")
    logger.info(f"  成功处理: {total_processed}")
    logger.info(f"  跳过文件: {total_skipped}")
    logger.info(f"  处理失败: {total_failed}")
    logger.info("=" * 80)
